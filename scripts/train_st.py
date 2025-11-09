#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新規ST学習スクリプト（既存st_model非依存）
- 入力: data/staging/YYYY/MMDD/st_train.csv （複数日対応）
- 目的変数: st_sec（実走ST）
- 学習対象: st_observed=1 かつ st_penalized=0 かつ st_secが非NULL
- 特徴量: lane, tenji_st_sec, tenji_rank, st_rank, course_avg_st, course_first_rate, course_3rd_rate
- 出力:
  - data/models/st_new.txt            (LightGBMモデル)
  - data/models/st_new.meta.json      (メタ情報)
  - data/models/st_new.fi.csv         (特徴量重要度)
  - data/history/st_history.parquet   (学習に使った行の記録)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple

import polars as pl
import numpy as np
import lightgbm as lgb

FEATURES = [
    "lane",
    "tenji_st_sec",
    "tenji_rank",
    "st_rank",
    "course_avg_st",
    "course_first_rate",
    "course_3rd_rate",
]

NEEDED_FOR_FILTER = [
    "hd", "jcd", "rno", "lane", "regno",
    "st_sec", "st_observed", "st_penalized",
]

def find_days(staging_root: Path, start: str|None, end: str|None) -> List[Tuple[str, Path]]:
    out = []
    for ydir in sorted(staging_root.glob("[0-9][0-9][0-9][0-9]")):
        for mddir in sorted(ydir.glob("[0-9][0-9][0-9][0-9]")):
            hd = f"{ydir.name}{mddir.name}"
            if start and hd < start: 
                continue
            if end and hd > end: 
                continue
            csv = mddir / "st_train.csv"
            if csv.exists():
                out.append((hd, csv))
    return out

def _ensure_cols(df: pl.DataFrame, cols: List[str]) -> pl.DataFrame:
    for c in cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    return df

def load_pool(staging_root: Path, start: str|None, end: str|None) -> pl.DataFrame:
    days = find_days(staging_root, start, end)
    if not days:
        raise SystemExit(f"[FATAL] no st_train.csv under {staging_root} for range {start}..{end}")

    parts: List[pl.DataFrame] = []
    for hd, csv in days:
        df = pl.read_csv(csv, infer_schema_length=0)
        df = _ensure_cols(df, NEEDED_FOR_FILTER + FEATURES)

        # 型の明示
        df = df.with_columns([
            pl.col("hd").cast(pl.Utf8),
            pl.col("jcd").cast(pl.Utf8),
            pl.col("rno").cast(pl.Int64),
            pl.col("lane").cast(pl.Int64),
            pl.col("regno").cast(pl.Utf8),
            pl.col("st_sec").cast(pl.Float64),
            pl.col("st_observed").fill_null(0).cast(pl.Int64),
            pl.col("st_penalized").fill_null(0).cast(pl.Int64),
        ])

        # フィーチャ列の型
        df = df.with_columns([
            pl.col("tenji_st_sec").cast(pl.Float64),
            pl.col("tenji_rank").cast(pl.Int64),
            pl.col("st_rank").cast(pl.Int64),
            pl.col("course_avg_st").cast(pl.Float64),
            pl.col("course_first_rate").cast(pl.Float64),
            pl.col("course_3rd_rate").cast(pl.Float64),
        ])

        # フィルタ: 実観測 & 非ペナルティ & 目的変数あり
        df = df.filter(
            (pl.col("st_observed") == 1) &
            (pl.col("st_penalized") == 0) &
            (pl.col("st_sec").is_not_null())
        )

        if df.height > 0:
            parts.append(df)

    if not parts:
        raise SystemExit("[FATAL] no usable rows after filtering (all penalized/missing st_sec)")

    return pl.concat(parts, how="vertical", rechunk=True)

def build_xy(df: pl.DataFrame):
    # 欠損の単純補完
    df = df.with_columns([
        pl.col("tenji_st_sec").fill_null(0.15),
        pl.col("tenji_rank").fill_null(3),
        pl.col("st_rank").fill_null(3),
        pl.col("course_avg_st").fill_null(0.15),
        pl.col("course_first_rate").fill_null(0.0),
        pl.col("course_3rd_rate").fill_null(0.0),
    ])

    X = np.column_stack([df[col].to_numpy() for col in FEATURES])
    y = df["st_sec"].to_numpy()
    return X, y, df

def train_lgbm(X: np.ndarray, y: np.ndarray, num_threads: int = 0):
    ds = lgb.Dataset(X, label=y, feature_name=FEATURES, free_raw_data=True)
    params = dict(
        objective="regression",
        metric="l2",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=1.0,
        num_threads=num_threads,
        force_col_wise=True,
        verbose=-1,
        seed=20241109,
    )
    booster = lgb.train(
        params,
        ds,
        num_boost_round=800,
        valid_sets=[ds],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    return booster

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--start", default=None, help="YYYYMMDD（含む）")
    ap.add_argument("--end",   default=None, help="YYYYMMDD（含む）")
    ap.add_argument("--model_out", default="data/models/st_new.txt")
    ap.add_argument("--meta_out",  default="data/models/st_new.meta.json")
    ap.add_argument("--history_out", default="data/history/st_history.parquet")
    ap.add_argument("--fi_out", default="data/models/st_new.fi.csv")
    args = ap.parse_args()

    staging_root = Path(args.staging_root)
    staging_root.mkdir(parents=True, exist_ok=True)

    pool = load_pool(staging_root, args.start, args.end)
    # 保存（学習に使った履歴）
    Path(args.history_out).parent.mkdir(parents=True, exist_ok=True)
    pool.select(["hd","jcd","rno","lane","regno","st_sec"] + FEATURES)\
        .write_parquet(args.history_out)
    print(f"[WRITE] {args.history_out}  rows={pool.height}")

    X, y, df_used = build_xy(pool)

    booster = train_lgbm(X, y)

    # 保存
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.model_out)
    print(f"[SAVE] model -> {args.model_out}")

    # 重要度
    fi_gain = booster.feature_importance(importance_type="gain")
    fi_split = booster.feature_importance(importance_type="split")
    fi_df = pl.DataFrame({
        "feature": FEATURES,
        "gain": fi_gain,
        "split": fi_split,
    })
    fi_path = Path(args.fi_out)
    fi_path.parent.mkdir(parents=True, exist_ok=True)
    fi_df.write_csv(fi_path)
    print(f"[SAVE] feature importance -> {fi_path}")

    # メタ
    meta = {
        "features": FEATURES,
        "rows": int(pool.height),
        "range": {
            "start": args.start,
            "end": args.end,
        },
        "model_path": str(Path(args.model_out)),
        "created_with": "st_train_new.py",
    }
    with open(args.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] meta -> {args.meta_out}")

if __name__ == "__main__":
    main()
