#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
決まり手4分類(逃げ/差し/まくり/まくり差し) 学習・推論 全差し替え版（1=逃げ限定 / 2-6=逃げ禁止）

- 入力: data/staging/YYYY/MMDD/decision4_train.csv
- 出力:
  モデル: data/models/decision4/model.txt (+ meta.json / feature_importance.csv)
  予測:   data/proba/decision4/YYYY/MMDD/decision4_proba_{test_start}_{test_end}.csv
    (列: hd,jcd,rno,lane,racer_id,racer_name,proba_nige,proba_sashi,proba_makuri,proba_makuri_sashi)
"""

from __future__ import annotations
import argparse, json, sys, glob
from pathlib import Path
from typing import List, Sequence

import polars as pl
import pandas as pd
import lightgbm as lgb

# -----------------------------
# 定数
# -----------------------------
VALID_LABELS = ["NIGE", "SASHI", "MAKURI", "MAKURI_SASHI"]
LABEL_TO_IDX = {k:i for i,k in enumerate(VALID_LABELS)}
IDX_TO_LABEL = {i:k for k,i in LABEL_TO_IDX.items()}

ID_COLS = ["hd","jcd","rno","lane","racer_id","racer_name"]
TARGET = "decision4"

PREF_FEATS = [
    "tenji_sec","tenji_rank","st_rank",
    "course_avg_st","course_first_rate","course_3rd_rate",
    "motor_rate2","motor_rate3","boat_rate2","boat_rate3",
    "wind_speed_m","wave_height_cm",
    "kimarite_makuri","kimarite_sashi","kimarite_makuri_sashi","kimarite_nuki",
    "power_lane","power_inner","power_outer","outer_over_inner",
    "dash_attack_flag","is_strong_wind","is_crosswind",
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
    # 列ズレ保険（最小変更）
    return pl.concat(lfs, how="diagonal_relaxed")

# -----------------------------
# データ整形
# -----------------------------
def build_train_table(lf: pl.LazyFrame) -> tuple[pl.DataFrame, list[str]]:
    """
    - lane=1 は強制的に 'NIGE'
    - lane!=1 で 'NIGE' は除外
    - IDはそのまま、特徴は lane -> lane_feat
    """
    if lf is None or (isinstance(lf, pl.LazyFrame) and len(lf.collect_schema().names()) == 0):
        return pl.DataFrame(), []

    all_need = list(set(ID_COLS + [TARGET] + PREF_FEATS))
    exprs: list[pl.Expr] = []
    for c in all_need:
        if c in ID_COLS:
            exprs.append(pl.col(c).alias(c).cast(pl.Utf8) if c in ("hd","jcd","racer_id","racer_name") else pl.col(c).alias(c))
        else:
            exprs.append(pl.when(pl.col(c).is_not_null()).then(pl.col(c)).otherwise(None).alias(c))

    lf2 = lf.select(exprs)

    # --- ここから制約適用 ---
    # 1コースは決まり手を強制的に NIGE
    lf2 = lf2.with_columns(
        pl.when(pl.col("lane") == 1).then(pl.lit("NIGE")).otherwise(pl.col(TARGET)).alias(TARGET)
    )
    # 2-6コースの NIGE は除外
    lf2 = lf2.filter(~((pl.col("lane") != 1) & (pl.col(TARGET) == "NIGE")))
    # ラベルは4クラス限定
    lf2 = lf2.filter(pl.col(TARGET).is_in(VALID_LABELS))
    # --- 制約ここまで ---

    # 特徴: lane は lane_feat 名に
    feat_exprs: list[pl.Expr] = []
    use_feats: list[str] = []
    for c in PREF_FEATS:
        if c == "lane":
            feat_exprs.append(pl.col("lane").cast(pl.Int64, strict=False).alias("lane_feat"))
            use_feats.append("lane_feat")
        else:
            feat_exprs.append(pl.col(c).cast(pl.Float64, strict=False).alias(c))
            use_feats.append(c)

    out = lf2.select([pl.col(k) for k in ID_COLS] + [pl.col(TARGET)] + feat_exprs).collect()
    return out, use_feats

def to_lgb_dataset(df: pl.DataFrame, feat_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    pdf = df.select(ID_COLS + [TARGET] + feat_cols).to_pandas(use_pyarrow_extension_array=False)
    y = pdf[TARGET].map(LABEL_TO_IDX).astype(int)
    X = pdf[feat_cols]
    return X, y

# -----------------------------
# 学習
# -----------------------------
def train_lgbm_multiclass(X: pd.DataFrame, y: pd.Series, num_class: int = 4) -> lgb.Booster:
    params = dict(
        objective="multiclass",
        num_class=num_class,
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
        params,
        dtrain,
        num_boost_round=800,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    return booster

# -----------------------------
# 推論
# -----------------------------
def build_test_table(lf: pl.LazyFrame, feat_cols: list[str]) -> pl.DataFrame:
    if lf is None or (isinstance(lf, pl.LazyFrame) and len(lf.collect_schema().names()) == 0):
        return pl.DataFrame()

    exprs: list[pl.Expr] = [pl.col(c).alias(c) for c in ID_COLS if c in lf.collect_schema().names()]
    for c in feat_cols:
        if c == "lane_feat":
            exprs.append(pl.col("lane").cast(pl.Int64, strict=False).alias("lane_feat"))
        else:
            exprs.append(
                pl.when(pl.col(c).is_not_null()).then(pl.col(c)).otherwise(None)
                  .cast(pl.Float64, strict=False).alias(c)
            )
    return lf.select(exprs).collect()

def _predict(booster: lgb.Booster, test_df: pl.DataFrame, feat_cols: list[str]) -> pl.DataFrame:
    if test_df.height == 0:
        return pl.DataFrame()

    out = test_df.select([c for c in ID_COLS if c in test_df.columns])

    # 特徴→pandas
    X = test_df.select(feat_cols).to_pandas(use_pyarrow_extension_array=False)
    proba = booster.predict(X)  # (n,4)

    # --- 予測時の制約クランプ（再正規化なし） ---
    if "lane" in out.columns:
        lane_np = out.get_column("lane").cast(pl.Int64, strict=False).to_numpy()
        m_l1 = (lane_np == 1)
        m_not1 = (lane_np != 1)
        # lane=1: 非逃げを0
        proba[m_l1, LABEL_TO_IDX["SASHI"]] = 0.0
        proba[m_l1, LABEL_TO_IDX["MAKURI"]] = 0.0
        proba[m_l1, LABEL_TO_IDX["MAKURI_SASHI"]] = 0.0
        # lane!=1: 逃げを0
        proba[m_not1, LABEL_TO_IDX["NIGE"]] = 0.0
    # -------------------------------------------

    out = out.with_columns([
        pl.lit(proba[:, LABEL_TO_IDX["NIGE"]]).alias("proba_nige"),
        pl.lit(proba[:, LABEL_TO_IDX["SASHI"]]).alias("proba_sashi"),
        pl.lit(proba[:, LABEL_TO_IDX["MAKURI"]]).alias("proba_makuri"),
        pl.lit(proba[:, LABEL_TO_IDX["MAKURI_SASHI"]]).alias("proba_makuri_sashi"),
    ])
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
    ap.add_argument("--model_out", default=None, help="直指定する場合のみ使用。未指定なら model_root/model.txt")
    args = ap.parse_args()

    # 入力探索
    train_files = find_decision_csvs(args.staging_root, args.train_start, args.train_end)
    if not train_files:
        err(f"[FATAL] train CSV が見つかりません: {args.staging_root} [{args.train_start}..{args.train_end}]")
        sys.exit(1)
    log(f"[INFO] train files: {len(train_files)}")

    lf_train = load_lazy(train_files)
    df_train, feat_cols = build_train_table(lf_train)
    if df_train.height == 0:
        err("[FATAL] 学習対象データが0件（decision4が NIGE/SASHI/MAKURI/MAKURI_SASHI に該当しない可能性）。")
        sys.exit(1)

    # 学習
    X_train, y_train = to_lgb_dataset(df_train, feat_cols)
    log(f"[INFO] train rows: {len(X_train)}, features: {len(feat_cols)} -> {feat_cols}")
    booster = train_lgbm_multiclass(X_train, y_train, num_class=4)

    # 保存
    model_path = Path(args.model_out) if args.model_out else Path(args.model_root) / "model.txt"
    meta_path  = model_path.with_suffix(".meta.json")
    fi_path    = model_path.with_suffix(".feature_importance.csv")
    ensure_parent(model_path); ensure_parent(meta_path); ensure_parent(fi_path)

    booster.save_model(str(model_path))
    log(f"[SAVE] model -> {model_path}")

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

    # FI
    import numpy as np
    fi = pd.DataFrame({
        "feature": feat_cols,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(fi_path, index=False)
    log(f"[SAVE] feature importance -> {fi_path}")

    # 推論（任意）
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

        pred = _predict(booster, df_test, feat_cols)

        y = args.test_start[:4]; md = args.test_start[4:8]
        out_csv = ensure_parent(Path(args.out_root) / y / md / f"decision4_proba_{args.test_start}_{args.test_end}.csv")
        pred.write_csv(str(out_csv))
        log(f"[WRITE] {out_csv} rows={pred.height}")

if __name__ == "__main__":
    main()
