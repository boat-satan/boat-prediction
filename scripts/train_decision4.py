#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
決まり手4分類(逃げ/差し/まくり/まくり差し) 学習・推論 全差し替え版（物理制約適用）

- 入力: data/staging/YYYY/MMDD/decision4_train.csv
  例の列（最低限）:
    hd,jcd,rno,lane,racer_id,racer_name,decision4,
    course_first_rate,course_3rd_rate,course_avg_st,
    tenji_st,tenji_sec,tenji_rank,st_rank,
    motor_rate2,motor_rate3,boat_rate2,boat_rate3,
    wind_speed_m,wave_height_cm,
    kimarite_makuri,kimarite_sashi,kimarite_makuri_sashi,kimarite_nuki,
    power_lane,power_inner,power_outer,outer_over_inner,
    dash_attack_flag,is_strong_wind,is_crosswind

- 出力:
  モデル: data/models/decision4/model.txt (+ .meta.json / .feature_importance.csv)
  予測:   data/proba/decision4/YYYY/MMDD/decision4_proba_{test_start}_{test_end}.csv
    列: hd,jcd,rno,lane,racer_id,racer_name,
        proba_nige,proba_sashi,proba_makuri,proba_makuri_sashi
"""

from __future__ import annotations
import argparse, json, sys, glob
from pathlib import Path
from typing import List, Sequence

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb

# -----------------------------
# 定数
# -----------------------------
VALID_LABELS = ["NIGE", "SASHI", "MAKURI", "MAKURI_SASHI"]
LABEL_TO_IDX = {k:i for i,k in enumerate(VALID_LABELS)}
ID_COLS = ["hd","jcd","rno","lane","racer_id","racer_name"]
TARGET = "decision4"

# 特徴候補（存在しない列は自動スキップ）
PREF_FEATS = [
    # 展示/スタート系
    "tenji_sec","tenji_rank","st_rank",
    "course_avg_st","course_first_rate","course_3rd_rate",
    # モーター/ボート
    "motor_rate2","motor_rate3","boat_rate2","boat_rate3",
    # 風・波
    "wind_speed_m","wave_height_cm",
    # 過去決まり手傾向
    "kimarite_makuri","kimarite_sashi","kimarite_makuri_sashi","kimarite_nuki",
    # パワー系
    "power_lane","power_inner","power_outer","outer_over_inner",
    # フラグ
    "dash_attack_flag","is_strong_wind","is_crosswind",
    # lane を特徴としても使う（重複回避のため lane_feat 名で持つ）
    "lane",
]

# -----------------------------
# ユーティリティ
# -----------------------------
def log(msg: str): print(msg, flush=True)
def err(msg: str): print(msg, file=sys.stderr, flush=True)

def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _ymd_paths(root: Path, yyyymmdd_from: str, yyyymmdd_to: str) -> list[Path]:
    def to_ord(y, m, d): return y * 372 + m * 31 + d
    s_y, s_m, s_d = int(yyyymmdd_from[:4]), int(yyyymmdd_from[4:6]), int(yyyymmdd_from[6:8])
    e_y, e_m, e_d = int(yyyymmdd_to[:4]),   int(yyyymmdd_to[4:6]),   int(yyyymmdd_to[6:8])
    s_ord, e_ord = to_ord(s_y, s_m, s_d), to_ord(e_y, e_m, e_d)

    picked: list[Path] = []
    for ydir in sorted(root.glob("[0-9]" * 4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]" * 4)):
            if not md.is_dir(): continue
            m, d = int(md.name[:2]), int(md.name[2:])
            od = to_ord(y, m, d)
            if s_ord <= od <= e_ord:
                picked.append(md)
    return picked

def find_decision_csvs(staging_root: str, start: str, end: str) -> list[Path]:
    root = Path(staging_root)
    day_dirs = _ymd_paths(root, start, end)
    files: list[Path] = []
    for dd in day_dirs:
        p = dd / "decision4_train.csv"
        if p.exists():
            files.append(p)
        else:
            # フォールバック: *decision* を含むCSV
            alt = [Path(x) for x in glob.glob(str(dd / "*decision*.csv"))]
            for a in alt:
                if a.exists() and a.suffix.lower() == ".csv":
                    files.append(a)
                    break
    return files

def load_lazy(paths: list[Path]) -> pl.LazyFrame:
    if not paths:
        return pl.LazyFrame()
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in paths]
    return pl.concat(lfs)

# -----------------------------
# データ整形（学習用）
# -----------------------------
def build_train_table(lf: pl.LazyFrame) -> tuple[pl.DataFrame, list[str]]:
    if lf is None or (isinstance(lf, pl.LazyFrame) and len(lf.collect_schema().names()) == 0):
        return pl.DataFrame(), []

    need = list(set(ID_COLS + [TARGET] + PREF_FEATS))
    exprs: list[pl.Expr] = []
    for c in need:
        if c in ("hd","jcd","racer_id","racer_name"):
            exprs.append(pl.col(c).cast(pl.Utf8, strict=False).alias(c))
        else:
            exprs.append(pl.when(pl.col(c).is_not_null()).then(pl.col(c)).otherwise(None).alias(c))
    lf2 = lf.select(exprs)

    # lane を数値化、特徴では lane_feat に（IDには lane を残す）
    lf2 = lf2.with_columns([
        pl.col("lane").cast(pl.Int64, strict=False).alias("lane"),
        pl.col("lane").cast(pl.Float64, strict=False).alias("lane_feat"),
    ])

    # ラベルを学習前に物理補正：
    # - lane==1 は必ず NIGE
    # - lane!=1 の NIGE サンプルは除外（=学習しない）
    lf2 = lf2.with_columns([
        pl.when(pl.col("lane") == 1).then(pl.lit("NIGE")).otherwise(pl.col(TARGET)).alias(TARGET)
    ])
    lf2 = lf2.filter(pl.col(TARGET).is_in(VALID_LABELS))
    lf2 = lf2.filter(~((pl.col("lane") != 1) & (pl.col(TARGET) == "NIGE")))

    # 特徴列（lane→lane_feat）
    feat_cols: list[str] = []
    feat_exprs: list[pl.Expr] = []
    for c in PREF_FEATS:
        if c == "lane":
            feat_exprs.append(pl.col("lane_feat"))
            feat_cols.append("lane_feat")
        else:
            feat_exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
            feat_cols.append(c)

    df = lf2.select([pl.col(k) for k in ID_COLS] + [pl.col(TARGET)] + feat_exprs).collect(streaming=True)
    return df, feat_cols

def to_lgb_dataset(df: pl.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    pdf = df.select(ID_COLS + [TARGET] + feat_cols).to_pandas(use_pyarrow_extension_array=False)
    y = pdf[TARGET].map(LABEL_TO_IDX).astype(int)
    X = pdf[feat_cols]
    return X, y

# -----------------------------
# 学習
# -----------------------------
def train_lgbm_multiclass(X: pd.DataFrame, y: pd.Series) -> lgb.Booster:
    params = dict(
        objective="multiclass",
        num_class=len(VALID_LABELS),
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
    dtrain = lgb.Dataset(X, label=y, feature_name=list(X.columns), free_raw_data=True)
    booster = lgb.train(
        params, dtrain, num_boost_round=800,
        valid_sets=[dtrain], valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    return booster

# -----------------------------
# 推論用テーブル & 物理制約
# -----------------------------
def build_test_table(lf: pl.LazyFrame, feat_cols: list[str]) -> pl.DataFrame:
    if lf is None or (isinstance(lf, pl.LazyFrame) and len(lf.collect_schema().names()) == 0):
        return pl.DataFrame()

    exprs: list[pl.Expr] = [pl.col(c).alias(c) for c in ID_COLS if c in lf.collect_schema().names()]
    # lane_feat を作る
    exprs.append(pl.col("lane").cast(pl.Int64, strict=False).alias("lane"))
    exprs.append(pl.col("lane").cast(pl.Float64, strict=False).alias("lane_feat"))

    # 学習時に使った特徴だけ揃える
    for c in feat_cols:
        if c == "lane_feat":
            continue
        exprs.append(pl.when(pl.col(c).is_not_null()).then(pl.col(c)).otherwise(None).cast(pl.Float64, strict=False).alias(c))

    return lf.select(exprs).collect(streaming=True)

def enforce_physical_constraints_on_probs(pdf: pd.DataFrame) -> pd.DataFrame:
    """
    物理制約:
      - lane==1: NIGE=1, 他=0
      - lane!=1: NIGE=0, SASHI/MAKURI/MS を和1に再正規化（和0は1/3ずつ）
    """
    Pn, Ps, Pm, Pms = "proba_nige","proba_sashi","proba_makuri","proba_makuri_sashi"
    lane = pdf["lane"].astype(int).values

    mask1 = (lane == 1)
    maskx = ~mask1

    # lane==1
    pdf.loc[mask1, [Pn]] = 1.0
    pdf.loc[mask1, [Ps, Pm, Pms]] = 0.0

    # lane!=1
    pdf.loc[maskx, [Pn]] = 0.0
    sub = pdf.loc[maskx, [Ps, Pm, Pms]].fillna(0.0)
    s = sub.sum(axis=1)

    nz = s > 0
    eq = ~nz
    sub.loc[nz, [Ps, Pm, Pms]] = sub.loc[nz, [Ps, Pm, Pms]].div(s[nz], axis=0)
    sub.loc[eq, [Ps, Pm, Pms]] = 1.0 / 3
    pdf.loc[maskx, [Ps, Pm, Pms]] = sub[[Ps, Pm, Pms]].values

    return pdf

def predict_probs(booster: lgb.Booster, df: pl.DataFrame, feat_cols: Sequence[str]) -> pd.DataFrame:
    keep = ID_COLS + ["lane"] + list(feat_cols)
    df2 = df.select([c for c in keep if c in df.columns])
    pdf = df2.to_pandas(use_pyarrow_extension_array=False)

    X = pdf[list(feat_cols)]
    proba = booster.predict(X, num_iteration=booster.best_iteration)  # (N,4)
    col_order = ["proba_nige","proba_sashi","proba_makuri","proba_makuri_sashi"]
    proba_df = pd.DataFrame(proba, columns=col_order)

    out = pd.concat([pdf[ID_COLS + ["lane"]].copy(), proba_df], axis=1)
    out = enforce_physical_constraints_on_probs(out)
    return out

# -----------------------------
# メイン
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--out_root",     default="data/proba/decision4")
    ap.add_argument("--model_root",   default="data/models/decision4")
    ap.add_argument("--train_start",  required=True)
    ap.add_argument("--train_end",    required=True)
    ap.add_argument("--test_start",   default=None)
    ap.add_argument("--test_end",     default=None)
    ap.add_argument("--model_out",    default=None)
    args = ap.parse_args()

    # 収集
    train_files = find_decision_csvs(args.staging_root, args.train_start, args.train_end)
    if not train_files:
        err(f"[FATAL] train CSV が見つかりません: {args.staging_root} [{args.train_start}..{args.train_end}]")
        sys.exit(1)
    log(f"[INFO] train files: {len(train_files)}")

    lf_train = load_lazy(train_files)
    df_train, feat_cols = build_train_table(lf_train)
    if df_train.height == 0:
        err("[FATAL] train table empty after physical filtering.")
        sys.exit(1)

    X_train, y_train = to_lgb_dataset(df_train, feat_cols)
    log(f"[INFO] train rows: {len(X_train)}, features: {len(feat_cols)} -> {feat_cols}")
    booster = train_lgbm_multiclass(X_train, y_train)

    # 保存
    model_path = Path(args.model_out) if args.model_out else Path(args.model_root) / "model.txt"
    meta_path  = model_path.with_suffix(".meta.json")
    fi_path    = model_path.with_suffix(".feature_importance.csv")
    ensure_parent(model_path); ensure_parent(meta_path); ensure_parent(fi_path)

    booster.save_model(str(model_path)); log(f"[SAVE] model -> {model_path}")
    meta = {
        "train_range": [args.train_start, args.train_end],
        "rows": int(len(X_train)),
        "features": feat_cols,
        "labels": VALID_LABELS,
        "params": {
            "objective": "multiclass",
            "num_class": 4,
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "num_boost_round": 800,
            "seed": 20240301,
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[SAVE] meta -> {meta_path}")

    fi = pd.DataFrame({
        "feature": feat_cols,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(fi_path, index=False)
    log(f"[SAVE] feature importance -> {fi_path}")

    # 推論
    if args.test_start and args.test_end:
        test_files = find_decision_csvs(args.staging_root, args.test_start, args.test_end)
        if not test_files:
            log(f"[WARN] test CSV が見つかりません: {args.staging_root} [{args.test_start}..{args.test_end}]")
            return
        log(f"[INFO] test files: {len(test_files)}")

        lf_test = load_lazy(test_files)
        df_test = build_test_table(lf_test, feat_cols)
        if df_test.height == 0:
            log("[WARN] テストデータが空。予測スキップ。")
            return

        pred = predict_probs(booster, df_test, feat_cols)
        y = args.test_start[:4]; md = args.test_start[4:8]
        out_csv = ensure_parent(Path(args.out_root) / y / md / f"decision4_proba_{args.test_start}_{args.test_end}.csv")
        pred.to_csv(out_csv, index=False)
        log(f"[WRITE] {out_csv} rows={len(pred)}")

if __name__ == "__main__":
    main()
