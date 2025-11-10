#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
決まり手(4分類: NIGE/SASHI/MAKURI/MAKURI_SASHI) 学習・推論 一体スクリプト
- 学習前整形と推論後に物理制約を適用
  * lane==1 : 決まり手は NIGE のみ（学習では NIGE 以外を NIGE に矯正）
  * lane!=1 : NIGE は成立しない（学習では NIGE を除去）
- Polars で dtype が混在しても落ちないように安全キャスト

入出力:
  入力: data/staging/YYYY/MMDD/decision4_train.csv
  出力:
    data/proba/decision4/{YYYY}/{MMDD}/proba_{test_start}_{test_end}.csv
    data/models/decision4/model.txt / meta.json / feature_importance.csv
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

FEATURES_PREF = [
    "tenji_sec","tenji_rank","st_rank",
    "course_avg_st","course_first_rate","course_3rd_rate",
    "motor_rate2","motor_rate3","boat_rate2","boat_rate3",
    "wind_speed_m","wave_height_cm",
    "kimarite_makuri","kimarite_sashi","kimarite_makuri_sashi","kimarite_nuki",
    "power_lane","power_inner","power_outer","outer_over_inner",
    "dash_attack_flag","is_strong_wind","is_crosswind",
    "lane",  # -> lane_feat に退避
]

def log(msg: str): print(msg, flush=True)
def err(msg: str): print(msg, file=sys.stderr, flush=True)
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# ----------------------------
# ファイル探索
# ----------------------------
def _ymd_dirs(root: Path, d1: str, d2: str) -> list[Path]:
    def ord3(y,m,d): return y*372 + m*31 + d
    s = (int(d1[:4]), int(d1[4:6]), int(d1[6:8]))
    e = (int(d2[:4]), int(d2[4:6]), int(d2[6:8]))
    s_ord, e_ord = ord3(*s), ord3(*e)
    out = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            y = int(ydir.name); m = int(md.name[:2]); d = int(md.name[2:])
            if s_ord <= ord3(y,m,d) <= e_ord:
                out.append(md)
    return out

def find_csvs(staging_root: Path, start: str, end: str, fname="decision4_train.csv") -> list[Path]:
    files: list[Path] = []
    for dd in _ymd_dirs(staging_root, start, end):
        p = dd / fname
        if p.exists():
            files.append(p)
    return files

def load_pool(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in paths]
    return pl.concat(lfs).collect(streaming=True)

# ----------------------------
# 前処理（物理制約を学習ラベルへ）
# ----------------------------
def apply_physical_constraints_labels(df: pl.DataFrame) -> pl.DataFrame:
    # 型の安定化
    casts = {}
    if "lane" in df.columns: casts["lane"] = pl.Int64
    if "decision4" in df.columns: casts["decision4"] = pl.Utf8
    if casts:
        df = df.with_columns([pl.col(k).cast(v, strict=False) for k,v in casts.items()])

    # lane==1 で NIGE 以外 → NIGE
    if {"lane","decision4"}.issubset(df.columns):
        df = df.with_columns(
            pl.when((pl.col("lane")==1) & (pl.col("decision4")!="NIGE"))
              .then(pl.lit("NIGE"))
              .otherwise(pl.col("decision4"))
              .alias("decision4")
        )
        # lane!=1 で NIGE は除去
        df = df.filter(~((pl.col("lane")!=1) & (pl.col("decision4")=="NIGE")))

    # 4ラベル以外は除去（念のため）
    df = df.filter(pl.col("decision4").cast(pl.Utf8, strict=False).is_in(LABELS))
    return df

# ----------------------------
# 特徴選択（衝突回避 & ラベルID安全変換）
# ----------------------------
def select_features(df: pl.DataFrame):
    cols = set(df.columns)
    if "decision4" not in cols:
        raise RuntimeError("入力に 'decision4' 列がありません。")

    feats: list[str] = [c for c in FEATURES_PREF if c in cols]

    exprs: list[pl.Expr] = []
    # ID列
    for c in ID_COLS:
        if c in cols:
            exprs.append(pl.col(c).alias(c))
    # 特徴列（lane -> lane_feat）
    for c in feats:
        if c == "lane":
            exprs.append(pl.col("lane").cast(pl.Int64, strict=False).alias("lane_feat"))
        else:
            exprs.append(pl.col(c).alias(c))

    # ラベルID: when/then で安全に数値化（map_elements は使わない）
    label_expr = (
        pl.when(pl.col("decision4").cast(pl.Utf8, strict=False) == "NIGE").then(pl.lit(LABEL2ID["NIGE"]))
         .when(pl.col("decision4").cast(pl.Utf8, strict=False) == "SASHI").then(pl.lit(LABEL2ID["SASHI"]))
         .when(pl.col("decision4").cast(pl.Utf8, strict=False) == "MAKURI").then(pl.lit(LABEL2ID["MAKURI"]))
         .when(pl.col("decision4").cast(pl.Utf8, strict=False) == "MAKURI_SASHI").then(pl.lit(LABEL2ID["MAKURI_SASHI"]))
         .otherwise(None)
         .alias("label_id")
    )
    exprs.append(label_expr)

    out = df.select(exprs).filter(pl.col("label_id").is_not_null())

    feature_cols = [("lane_feat" if f=="lane" else f) for f in feats]
    return out, feature_cols

# ----------------------------
# 学習・推論
# ----------------------------
def fit_lgbm(train_df: pl.DataFrame, feature_cols: Sequence[str]) -> lgb.Booster:
    pdf = train_df.select(list(feature_cols) + ["label_id"]).to_pandas(use_pyarrow_extension_array=False)
    X = pdf[list(feature_cols)]
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
    Pn, Ps, Pm, Pms = "proba_nige","proba_sashi","proba_makuri","proba_makuri_sashi"
    lane = pdf_pred["lane"].astype(int).values

    mask1 = (lane == 1)
    pdf_pred.loc[mask1, [Ps,Pm,Pms]] = 0.0
    pdf_pred.loc[mask1, [Pn]] = 1.0

    maskx = ~mask1
    pdf_pred.loc[maskx, [Pn]] = 0.0
    s = (pdf_pred.loc[maskx, [Ps,Pm,Pms]].fillna(0.0)).sum(axis=1)
    nz = s > 0
    pdf_pred.loc[maskx & nz, [Ps,Pm,Pms]] = pdf_pred.loc[maskx & nz, [Ps,Pm,Pms]].div(s[nz], axis=0)
    pdf_pred.loc[maskx & ~nz, [Ps,Pm,Pms]] = 1.0/3
    return pdf_pred

def predict_blocks(booster: lgb.Booster, test_df: pl.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    keep_ids = [c for c in ID_COLS if c in test_df.columns]
    # lane → lane_feat が学習側で使われている場合に備えて合わせる
    if "lane" in test_df.columns and "lane_feat" not in test_df.columns:
        test_df = test_df.with_columns(pl.col("lane").alias("lane_feat"))

    X = test_df.select(feature_cols).to_pandas(use_pyarrow_extension_array=False)
    ids = test_df.select(keep_ids + (["lane_feat"] if "lane_feat" in test_df.columns else [])).to_pandas(use_pyarrow_extension_array=False)

    proba = booster.predict(X, num_iteration=booster.best_iteration)
    out = pd.DataFrame({
        "proba_nige":   proba[:, LABEL2ID["NIGE"]],
        "proba_sashi":  proba[:, LABEL2ID["SASHI"]],
        "proba_makuri": proba[:, LABEL2ID["MAKURI"]],
        "proba_makuri_sashi": proba[:, LABEL2ID["MAKURI_SASHI"]],
    })
    pdf = pd.concat([ids.reset_index(drop=True), out], axis=1)

    # lane_feat → lane
    if "lane" not in pdf.columns and "lane_feat" in pdf.columns:
        pdf["lane"] = pdf["lane_feat"].astype(int)

    pdf = enforce_physical_constraints_on_probs(pdf)

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
    ap.add_argument("--model_out",    default=None)
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

    train_files = find_csvs(Path(cfg.staging_root), cfg.train_start, cfg.train_end, "decision4_train.csv")
    test_files  = find_csvs(Path(cfg.staging_root), cfg.test_start,  cfg.test_end,  "decision4_train.csv")

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

    # 物理制約（学習ラベル）
    df_train = apply_physical_constraints_labels(df_train_raw)

    # 特徴/ラベルID抽出（安全）
    df_train2, feat_cols = select_features(df_train)
    if df_train2.height == 0:
        err("[FATAL] 整形後の学習データが空です。")
        sys.exit(1)

    log(f"[INFO] train rows: {df_train2.height}, features: {len(feat_cols)} -> {feat_cols}")

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
    fi = pd.DataFrame({
        "feature": feat_cols,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(fi_path, index=False)
    log(f"[WRITE] feature importance -> {fi_path}")

    # 推論
    if test_files:
        df_test_raw = load_pool(test_files)
        df_test = df_test_raw
        # lane → lane_feat を準備（学習との整合）
        if "lane" in df_test.columns and "lane_feat" not in df_test.columns:
            df_test = df_test.with_columns(pl.col("lane").alias("lane_feat"))
        need = list(set([c for c in ID_COLS if c in df_test.columns] + feat_cols))
        df_test2 = df_test.select([c for c in need if c in df_test.columns])

        pdf = predict_blocks(booster, df_test2, feat_cols)

        y, md = cfg.test_start[:4], cfg.test_start[4:8]
        out_csv = ensure_parent(Path(cfg.out_root) / y / md / f"proba_{cfg.test_start}_{cfg.test_end}.csv")
        pdf.to_csv(out_csv, index=False)
        log(f"[WRITE] {out_csv} rows={len(pdf)}")

if __name__ == "__main__":
    main()
