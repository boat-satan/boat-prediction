#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
決まり手(4分類: NIGE/SASHI/MAKURI/MAKURI_SASHI) 学習・推論 一体スクリプト
- 物理制約を学習前整形と推論時の両方で適用
  * lane==1 : 決まり手は NIGE のみ（学習では NIGE 以外をNIGEに矯正）
  * lane!=1 : NIGE は成立しない（学習では NIGE 行を除去）

入出力想定:
  入力: data/staging/YYYY/MMDD/decision4_train.csv （日付範囲で集約）
        必須列: hd,jcd,rno,lane,decision4
        主要特徴: 下記 FEATURES に存在する列は自動で使う（足りない分は無視）
  出力:
    - 予測CSV: data/proba/decision4/{YYYY}/{MMDD}/proba_{test_start}_{test_end}.csv
        列: hd,jcd,rno,lane,racer_id,racer_name,proba_nige,proba_sashi,proba_makuri,proba_makuri_sashi
    - モデル:  data/models/decision4/model.txt
    - メタ:    data/models/decision4/meta.json
    - 特徴量重要度: data/models/decision4/feature_importance.csv
"""

from __future__ import annotations
import argparse, json, sys, glob
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Sequence

import polars as pl
import pandas as pd
import lightgbm as lgb

# ----------------------------
# 設定
# ----------------------------
LABELS = ["NIGE", "SASHI", "MAKURI", "MAKURI_SASHI"]
LABEL2ID = {k:i for i,k in enumerate(LABELS)}

ID_COLS = ["hd","jcd","rno","lane","racer_id","racer_name"]

# 使えるものだけ自動で使う。存在しない列は無視。
FEATURES_PREF = [
    "tenji_sec","tenji_rank","st_rank",
    "course_avg_st","course_first_rate","course_3rd_rate",
    "motor_rate2","motor_rate3","boat_rate2","boat_rate3",
    "wind_speed_m","wave_height_cm",
    "kimarite_makuri","kimarite_sashi","kimarite_makuri_sashi","kimarite_nuki",
    "power_lane","power_inner","power_outer","outer_over_inner",
    "dash_attack_flag","is_strong_wind","is_crosswind",
    "lane",  # 衝突防止のため後で lane_feat に退避
]

# ----------------------------
# ユーティリティ
# ----------------------------
def log(msg: str): print(msg, flush=True)
def err(msg: str): print(msg, file=sys.stderr, flush=True)

def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _ymd_dirs(root: Path, d1: str, d2: str) -> list[Path]:
    def ord3(y,m,d): return y*372 + m*31 + d
    s = (int(d1[:4]), int(d1[4:6]), int(d1[6:8]))
    e = (int(d2[:4]), int(d2[4:6]), int(d2[6:8]))
    s_ord, e_ord = ord3(*s), ord3(*e)
    out = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            m, d = int(md.name[:2]), int(md.name[2:])
            o = ord3(y,m,d)
            if s_ord <= o <= e_ord:
                out.append(md)
    return out

def find_csvs(staging_root: Path, start: str, end: str, fname="decision4_train.csv") -> list[Path]:
    files: list[Path] = []
    for dd in _ymd_dirs(staging_root, start, end):
        p = dd / fname
        if p.exists():
            files.append(p)
    return files

# ----------------------------
# データ読み込み・整形
# ----------------------------
def load_pool(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in paths]
    return pl.concat(lfs).collect(streaming=True)

def apply_physical_constraints_labels(df: pl.DataFrame) -> pl.DataFrame:
    """
    学習ラベルに物理制約を適用:
      - lane==1 & decision4!=NIGE -> NIGE に矯正
      - lane!=1 & decision4==NIGE -> 行を除去
      - その他 -> そのまま
    """
    # 型
    cast_map = {"lane": pl.Int64, "decision4": pl.Utf8}
    for c,dt in cast_map.items():
        if c in df.columns:
            df = df.with_columns(pl.col(c).cast(dt, strict=False))
    # lane==1 で NIGE 以外 → NIGE
    df = df.with_columns(
        pl.when((pl.col("lane")==1) & (pl.col("decision4")!="NIGE"))
          .then(pl.lit("NIGE"))
          .otherwise(pl.col("decision4")).alias("decision4")
    )
    # lane!=1 で NIGE は除去
    df = df.filter(~((pl.col("lane")!=1) & (pl.col("decision4")=="NIGE")))
    # 4ラベル以外は除外
    df = df.filter(pl.col("decision4").is_in(LABELS))
    return df

def select_features(df: pl.DataFrame):
    cols = set(df.columns)
    if "decision4" not in cols:
        raise RuntimeError("入力に 'decision4' 列がありません。")

    feats: list[str] = []
    for c in FEATURES_PREF:
        if c in cols:
            feats.append(c)
    # lane の衝突回避
    out_cols: list[pl.Expr] = []
    for c in ID_COLS:
        if c in cols:
            out_cols.append(pl.col(c).alias(c))
    for c in feats:
        if c == "lane":
            out_cols.append(pl.col("lane").alias("lane_feat"))
        else:
            out_cols.append(pl.col(c).alias(c))
    # ラベルID
    out_cols.append(pl.col("decision4").map_elements(lambda x: LABEL2ID.get(x, None)).alias("label_id"))
    out = df.select(out_cols)
    out = out.filter(pl.col("label_id").is_not_null())
    # 特徴名決定
    feature_cols = [("lane_feat" if f=="lane" else f) for f in feats]
    return out, feature_cols

# ----------------------------
# 学習・推論
# ----------------------------
def fit_lgbm(train_df: pl.DataFrame, feature_cols: Sequence[str]) -> lgb.Booster:
    pdf = train_df.select(feature_cols + ["label_id"]).to_pandas(use_pyarrow_extension_array=False)
    X = pdf[feature_cols]
    y = pdf["label_id"].astype(int).values
    dtrain = lgb.Dataset(X, label=y, feature_name=list(feature_cols), free_raw_data=True)
    params = dict(
        objective="multiclass",
        num_class=4,
        metric="multi_logloss",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=1.0,
        num_threads=0,
        seed=20240301,
        verbose=-1,
        force_col_wise=True,
    )
    booster = lgb.train(
        params, dtrain,
        num_boost_round=800,
        valid_sets=[dtrain], valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    return booster

def enforce_physical_constraints_on_probs(pdf_pred: pd.DataFrame) -> pd.DataFrame:
    """
    推論後確率に物理制約を適用:
      - lane==1: NIGE=1, others=0 （元のNIGEが0/NaNでも1に寄せる）
      - lane!=1: NIGE=0、SASHI/MAKURI/MS を再正規化（和0のときは等分）
    """
    Pn, Ps, Pm, Pms = "proba_nige","proba_sashi","proba_makuri","proba_makuri_sashi"
    lane = pdf_pred["lane"].astype(int).values

    # lane==1
    mask1 = (lane == 1)
    pdf_pred.loc[mask1, [Ps,Pm,Pms]] = 0.0
    pdf_pred.loc[mask1, [Pn]] = 1.0

    # lane!=1
    maskx = ~mask1
    pdf_pred.loc[maskx, [Pn]] = 0.0
    s = (pdf_pred.loc[maskx, [Ps,Pm,Pms]].fillna(0.0)).sum(axis=1)
    # 和>0 → 正規化、和==0 → 等分
    nz = s > 0
    pdf_pred.loc[maskx & nz, [Ps,Pm,Pms]] = pdf_pred.loc[maskx & nz, [Ps,Pm,Pms]].div(s[nz], axis=0)
    pdf_pred.loc[maskx & ~nz, [Ps,Pm,Pms]] = 1.0/3

    return pdf_pred

def predict_blocks(booster: lgb.Booster, test_df: pl.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    keep = [c for c in ID_COLS if c in test_df.columns]
    X = test_df.select(feature_cols).to_pandas(use_pyarrow_extension_array=False)
    ids = test_df.select(keep + (["lane_feat"] if "lane_feat" in test_df.columns else [])).to_pandas(use_pyarrow_extension_array=False)

    proba = booster.predict(X, num_iteration=booster.best_iteration)  # shape (n,4)
    out = pd.DataFrame({
        "proba_nige":   proba[:, LABEL2ID["NIGE"]],
        "proba_sashi":  proba[:, LABEL2ID["SASHI"]],
        "proba_makuri": proba[:, LABEL2ID["MAKURI"]],
        "proba_makuri_sashi": proba[:, LABEL2ID["MAKURI_SASHI"]],
    })
    pdf = pd.concat([ids.reset_index(drop=True), out], axis=1)
    # lane_feat を lane へ（存在すれば）
    if "lane" not in pdf.columns and "lane_feat" in pdf.columns:
        pdf["lane"] = pdf["lane_feat"].astype(int)
    # 物理制約適用
    pdf = enforce_physical_constraints_on_probs(pdf)
    # 整列
    order = [c for c in ["hd","jcd","rno","lane","racer_id","racer_name",
                         "proba_nige","proba_sashi","proba_makuri","proba_makuri_sashi"] if c in pdf.columns]
    return pdf[order]

# ----------------------------
# メイン
# ----------------------------
@dataclass
class Cfg:
    staging_root: str = "data/staging"
    model_root:   str = "data/models/decision4"
    out_root:     str = "data/proba/decision4"
    train_start:  str = "20240101"
    train_end:    str = "20240131"
    test_start:   str = "20240201"
    test_end:     str = "20240229"
    model_out:    str | None = None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--model_root",   default="data/models/decision4")
    ap.add_argument("--out_root",     default="data/proba/decision4")

    ap.add_argument("--train_start",  default="20240101")
    ap.add_argument("--train_end",    default="20240131")
    ap.add_argument("--test_start",   default="20240201")
    ap.add_argument("--test_end",     default="20240229")

    ap.add_argument("--model_out",    default=None, help="モデル保存先を単一ファイルで指定（省略時は model_root/model.txt）")
    args = ap.parse_args()

    cfg = Cfg(
        staging_root=args.staging_root,
        model_root=args.model_root,
        out_root=args.out_root,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
        model_out=args.model_out,
    )

    # 学習ファイル収集
    train_files = find_csvs(Path(cfg.staging_root), cfg.train_start, cfg.train_end, fname="decision4_train.csv")
    test_files  = find_csvs(Path(cfg.staging_root), cfg.test_start,  cfg.test_end,  fname="decision4_train.csv")  # 推論も同じスキーマ

    if not train_files:
        err(f"[FATAL] 学習CSVが見つかりません: {cfg.staging_root} [{cfg.train_start}..{cfg.train_end}]")
        sys.exit(1)

    log(f"[INFO] train files: {len(train_files)}")
    if test_files:
        log(f"[INFO] test files : {len(test_files)}")

    df_train_raw = load_pool(train_files)
    if df_train_raw.height == 0:
        err("[FATAL] 学習データが空です。")
        sys.exit(1)

    # 物理制約をラベルに適用
    df_train = apply_physical_constraints_labels(df_train_raw)

    # 特徴選択
    df_train2, feat_cols = select_features(df_train)
    if df_train2.height == 0:
        err("[FATAL] 整形後の学習データが空です。")
        sys.exit(1)

    log(f"[INFO] train rows: {df_train2.height}, features: {len(feat_cols)} -> {feat_cols}")

    # 学習
    booster = fit_lgbm(df_train2, feat_cols)

    # 保存
    model_path = ensure_parent(Path(cfg.model_out) if cfg.model_out else Path(cfg.model_root) / "model.txt")
    meta_path  = ensure_parent(model_path.with_suffix(".json"))
    fi_path    = ensure_parent(Path(cfg.model_root) / "feature_importance.csv")

    booster.save_model(str(model_path))
    log(f"[WRITE] model -> {model_path}")

    meta = {
        "train_range": [cfg.train_start, cfg.train_end],
        "rows": int(df_train2.height),
        "features": feat_cols,
        "labels": LABELS,
        "params": {
            "objective": "multiclass", "num_class": 4, "metric": "multi_logloss",
            "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 50,
            "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1,
            "lambda_l2": 1.0, "num_boost_round": 800, "seed": 20240301
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[WRITE] meta -> {meta_path}")

    # FI
    import numpy as np
    fi = pd.DataFrame({
        "feature": feat_cols,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(fi_path, index=False)
    log(f"[WRITE] feature importance -> {fi_path}")

    # 推論（存在する場合）
    if test_files:
        df_test_raw = load_pool(test_files)
        df_test = df_test_raw  # 推論ではラベル不要
        # 特徴の列名調整（lane -> lane_feat）
        keep_ids = [c for c in ID_COLS if c in df_test.columns]
        # lane_feat への退避
        if "lane" in df_test.columns and "lane_feat" not in df_test.columns:
            df_test = df_test.with_columns(pl.col("lane").alias("lane_feat"))
        # 必要列のみ
        need = list(set(keep_ids + feat_cols))
        df_test2 = df_test.select([c for c in need if c in df_test.columns])

        pdf = predict_blocks(booster, df_test2, feat_cols)

        y, md = cfg.test_start[:4], cfg.test_start[4:8]
        out_csv = ensure_parent(Path(cfg.out_root) / y / md / f"proba_{cfg.test_start}_{cfg.test_end}.csv")
        pdf.to_csv(out_csv, index=False)
        log(f"[WRITE] {out_csv} rows={len(pdf)}")

if __name__ == "__main__":
    main()
