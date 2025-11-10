#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
決まり手4分類（NIGE/SASHI/MAKURI/MAKURI_SASHI） 学習・推論・物理制約適用 一体スクリプト

入力:
  data/staging/YYYY/MMDD/decision4_train.csv
    必須列:
      hd,jcd,rno,lane,racer_id,racer_name,decision4,
      tenji_sec,tenji_rank,st_rank,
      course_avg_st,course_first_rate,course_3rd_rate,
      motor_rate2,motor_rate3,boat_rate2,boat_rate3,
      wind_speed_m,wave_height_cm,
      kimarite_makuri,kimarite_sashi,kimarite_makuri_sashi,kimarite_nuki,
      power_lane,power_inner,power_outer,outer_over_inner,
      dash_attack_flag,is_strong_wind,is_crosswind

出力:
  - 予測CSV: data/proba/decision4/YYYY/MMDD/proba_{test_start}_{test_end}.csv
  - モデル  : data/models/decision4/model.txt
  - メタ    : data/models/decision4/meta.json
  - FI      : data/models/decision4/feature_importance.csv
"""

from __future__ import annotations
import argparse, json, sys, glob
from pathlib import Path
from typing import List, Sequence

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb

LABELS = ["NIGE", "SASHI", "MAKURI", "MAKURI_SASHI"]
LABEL_TO_ID = {k:i for i,k in enumerate(LABELS)}

ID_COLS = ["hd","jcd","rno","lane","racer_id","racer_name"]
# lane は特徴にも使うが、重複名を避けるため lane_feat として入れる
FEAT_PREF = [
    "tenji_sec","tenji_rank","st_rank",
    "course_avg_st","course_first_rate","course_3rd_rate",
    "motor_rate2","motor_rate3","boat_rate2","boat_rate3",
    "wind_speed_m","wave_height_cm",
    "kimarite_makuri","kimarite_sashi","kimarite_makuri_sashi","kimarite_nuki",
    "power_lane","power_inner","power_outer","outer_over_inner",
    "dash_attack_flag","is_strong_wind","is_crosswind",
]
TARGET = "decision4"

# ---------- utils ----------
def log(x: str): print(x, flush=True)
def err(x: str): print(x, file=sys.stderr, flush=True)
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _ymd_dirs(root: Path, d1: str, d2: str) -> list[Path]:
    def ord3(y,m,d): return y*372 + m*31 + d
    sy,sm,sd = int(d1[:4]), int(d1[4:6]), int(d1[6:8])
    ey,em,ed = int(d2[:4]), int(d2[4:6]), int(d2[6:8])
    s_ord, e_ord = ord3(sy,sm,sd), ord3(ey,em,ed)
    out = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            m,d = int(md.name[:2]), int(md.name[2:])
            if s_ord <= ord3(y,m,d) <= e_ord:
                out.append(md)
    return out

def find_csvs(staging_root: str, start: str, end: str, basename: str) -> list[Path]:
    root = Path(staging_root)
    paths = []
    for dd in _ymd_dirs(root, start, end):
        p = dd / basename
        if p.exists():
            paths.append(p)
    return paths

# ---------- load ----------
SCHEMA = {
    "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64,
    "racer_id": pl.Utf8, "racer_name": pl.Utf8, "decision4": pl.Utf8,
    "tenji_sec": pl.Float64, "tenji_rank": pl.Int64, "st_rank": pl.Int64,
    "course_avg_st": pl.Float64, "course_first_rate": pl.Float64, "course_3rd_rate": pl.Float64,
    "motor_rate2": pl.Float64, "motor_rate3": pl.Float64, "boat_rate2": pl.Float64, "boat_rate3": pl.Float64,
    "wind_speed_m": pl.Float64, "wave_height_cm": pl.Float64,
    "kimarite_makuri": pl.Float64, "kimarite_sashi": pl.Float64,
    "kimarite_makuri_sashi": pl.Float64, "kimarite_nuki": pl.Float64,
    "power_lane": pl.Float64, "power_inner": pl.Float64, "power_outer": pl.Float64, "outer_over_inner": pl.Float64,
    "dash_attack_flag": pl.Int64, "is_strong_wind": pl.Int64, "is_crosswind": pl.Int64,
}

def load_lf(files: list[Path]) -> pl.LazyFrame:
    if not files:
        return pl.LazyFrame()
    patterns = [str(p) for p in files]
    # 複数ファイルを scan_csv でまとめて読む
    lf_list = [pl.scan_csv(p, schema_overrides=SCHEMA, ignore_errors=True) for p in patterns]
    return pl.concat(lf_list)

# ---------- table build ----------
def build_table(lf: pl.LazyFrame, drop_invalid: bool = True) -> tuple[pl.DataFrame, list[str]]:
    if lf == pl.LazyFrame():
        return pl.DataFrame(), []

    # ID と特徴を安全に select（無い列は null）
    def safe(name: str): 
        return pl.when(pl.col(name).is_not_null()).then(pl.col(name)).otherwise(None).alias(name)

    select_cols = [safe(c) for c in ID_COLS + [TARGET] + FEAT_PREF + ["lane"]]
    lf = lf.select(select_cols)

    # lane_feat 作成（数値化）
    lf = lf.with_columns([
        pl.col("lane").cast(pl.Int64, strict=False).alias("lane"),
        pl.col("lane").cast(pl.Float64, strict=False).alias("lane_feat"),
    ])

    # 1コース以外のラベル "NIGE" は学習時に除外（物理制約）
    # かつ 1コースのラベルは強制的に "NIGE" に丸める（データに揺れがあっても物理整合）
    lf = lf.with_columns([
        pl.when(pl.col("lane") == 1).then(pl.lit("NIGE")).otherwise(pl.col("decision4")).alias("decision4")
    ])

    if drop_invalid:
        lf = lf.filter(
            (pl.col("decision4").is_in(LABELS)) &
            (pl.col("lane").is_not_null())
        )

    # pandas へ渡す前に収集
    df = lf.collect(streaming=True)

    # 2～6 コースで decision4=="NIGE" になってしまっている行を落とす（最終防波堤）
    mask_physical = ~((df["lane"] != 1) & (df["decision4"] == "NIGE"))
    df = df.filter(mask_physical)

    # 特徴列を確定
    feat_cols = [c for c in FEAT_PREF if c in df.columns] + ["lane_feat"]
    return df, feat_cols

# ---------- train ----------
def fit_multiclass(train_df: pl.DataFrame, feat_cols: Sequence[str]) -> tuple[lgb.Booster, list[str]]:
    # LightGBM は pandas を期待
    pdf = train_df.select(ID_COLS + [TARGET] + list(feat_cols)).to_pandas(use_pyarrow_extension_array=False)

    # y をラベルIDへ
    y = pdf[TARGET].map(LABEL_TO_ID).astype(int).values
    X = pdf[list(feat_cols)]

    params = dict(
        objective="multiclass",
        num_class=len(LABELS),
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
    ds = lgb.Dataset(X, label=y, feature_name=list(feat_cols), free_raw_data=True)
    booster = lgb.train(params, ds, num_boost_round=800,
                        valid_sets=[ds], valid_names=["train"],
                        callbacks=[lgb.log_evaluation(period=200)])
    return booster, list(feat_cols)

# ---------- postprocess ----------
def enforce_physical_constraints_on_probs(pdf_pred: pd.DataFrame) -> pd.DataFrame:
    """
    lane==1: NIGE=1, others=0
    lane!=1: NIGE=0、SASHI/MAKURI/MS を和1に正規化（和0は1/3ずつ）
    """
    Pn, Ps, Pm, Pms = "proba_nige","proba_sashi","proba_makuri","proba_makuri_sashi"
    lane = pdf_pred["lane"].astype(int).values

    mask1 = (lane == 1)
    pdf_pred.loc[mask1, [Ps, Pm, Pms]] = 0.0
    pdf_pred.loc[mask1, [Pn]] = 1.0

    maskx = ~mask1
    pdf_pred.loc[maskx, [Pn]] = 0.0

    sub = pdf_pred.loc[maskx, [Ps, Pm, Pms]].fillna(0.0)
    s = sub.sum(axis=1)

    nz_mask = s > 0
    eq_rows = ~nz_mask

    sub.loc[nz_mask, [Ps, Pm, Pms]] = sub.loc[nz_mask, [Ps, Pm, Pms]].div(s[nz_mask], axis=0)
    sub.loc[eq_rows, [Ps, Pm, Pms]] = 1.0/3
    pdf_pred.loc[maskx, [Ps, Pm, Pms]] = sub[[Ps, Pm, Pms]].values

    return pdf_pred

# ---------- predict ----------
def predict_probs(booster: lgb.Booster, df: pl.DataFrame, feat_cols: Sequence[str]) -> pd.DataFrame:
    keep = ID_COLS + ["lane"] + list(feat_cols)
    df2 = df.select([c for c in keep if c in df.columns])
    pdf = df2.to_pandas(use_pyarrow_extension_array=False)

    X = pdf[list(feat_cols)]
    proba = booster.predict(X, num_iteration=booster.best_iteration)  # shape (N, 4)
    # ラベル順は LABELS と一致させる
    colmap = {
        "NIGE": "proba_nige",
        "SASHI": "proba_sashi",
        "MAKURI": "proba_makuri",
        "MAKURI_SASHI": "proba_makuri_sashi",
    }
    proba_df = pd.DataFrame(proba, columns=[colmap[k] for k in LABELS])

    out = pdf[ID_COLS + ["lane"]].copy()
    out = pd.concat([out, proba_df], axis=1)

    # 物理制約を適用
    out = enforce_physical_constraints_on_probs(out)
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--out_root",     default="data/proba/decision4")
    ap.add_argument("--model_root",   default="data/models/decision4")
    ap.add_argument("--train_start",  required=True)
    ap.add_argument("--train_end",    required=True)
    ap.add_argument("--test_start",   required=True)
    ap.add_argument("--test_end",     required=True)
    args = ap.parse_args()

    train_files = find_csvs(args.staging_root, args.train_start, args.train_end, "decision4_train.csv")
    test_files  = find_csvs(args.staging_root, args.test_start,  args.test_end,  "decision4_train.csv")

    if not train_files:
        err(f"[WARN] train files not found: {args.staging_root} {args.train_start}..{args.train_end}")
        sys.exit(1)

    log(f"[INFO] train files: {len(train_files)}")
    if test_files: log(f"[INFO] test  files: {len(test_files)}")

    lf_tr = load_lf(train_files)
    df_train, feat_cols = build_table(lf_tr, drop_invalid=True)

    if df_train.height == 0:
        err("[FATAL] train table empty after filtering.")
        sys.exit(1)

    log(f"[INFO] train rows: {df_train.height}, features: {len(feat_cols)} -> {feat_cols}")

    booster, feats = fit_multiclass(df_train, feat_cols)

    # 保存
    model_path = ensure_parent(Path(args.model_root) / "model.txt")
    meta_path  = ensure_parent(Path(args.model_root) / "meta.json")
    fi_path    = ensure_parent(Path(args.model_root) / "feature_importance.csv")

    booster.save_model(str(model_path))
    log(f"[WRITE] {model_path}")

    meta = {
        "train_range": [args.train_start, args.train_end],
        "rows": int(df_train.height),
        "features": feats,
        "labels": LABELS,
        "params": {
            "objective": "multiclass",
            "num_class": len(LABELS),
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
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[WRITE] {meta_path}")

    # FI
    fi = pd.DataFrame({
        "feature": feats,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(fi_path, index=False)
    log(f"[WRITE] {fi_path}")

    # 推論
    if test_files:
        lf_te = load_lf(test_files)
        df_test, _ = build_table(lf_te, drop_invalid=False)
        if df_test.height == 0:
            log("[WARN] test table empty -> skip predict")
            return

        pdf_pred = predict_probs(booster, df_test, feat_cols)
        y, md = args.test_start[:4], args.test_start[4:8]
        out_csv = ensure_parent(Path(args.out_root) / y / md / f"proba_{args.test_start}_{args.test_end}.csv")
        pdf_pred.to_csv(out_csv, index=False)
        log(f"[WRITE] {out_csv} rows={len(pdf_pred)}")

if __name__ == "__main__":
    main()
