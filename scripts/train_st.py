#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import List, Sequence

import polars as pl
import pandas as pd
import lightgbm as lgb


# -----------------------------
# ユーティリティ
# -----------------------------
def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _ymd_ord(s: str) -> int:
    return int(s[:4]) * 372 + int(s[4:6]) * 31 + int(s[6:8])

def _within(hd: str, d1: str, d2: str) -> bool:
    return _ymd_ord(d1) <= _ymd_ord(hd) <= _ymd_ord(d2)


# -----------------------------
# 入力読み込み（Lazy）
# -----------------------------
def read_staging_days(staging_root: Path, d1: str, d2: str) -> pl.LazyFrame:
    """
    staging_root/YYYY/MMDD/st_train.csv を d1..d2 で集約 (Lazy)
    """
    if len(d1) != 8 or len(d2) != 8:
        raise ValueError("train_start/train_end は YYYYMMDD 形式で指定してください。")

    pattern = str(staging_root / "*" / "*" / "st_train.csv")
    lf = (
        pl.scan_csv(
            pattern,
            schema_overrides={
                "hd": pl.Utf8,
                "jcd": pl.Utf8,
                "rno": pl.Int64,
                "lane": pl.Int64,
                "regno": pl.Utf8,
                # 目的変数とST関連
                "st_sec": pl.Float64,
                "st_is_f": pl.Int64,
                "st_is_late": pl.Int64,
                "st_penalized": pl.Int64,
                "st_observed": pl.Int64,
                # 展示・コース統計
                "tenji_st_sec": pl.Float64,
                "tenji_is_f": pl.Int64,
                "tenji_f_over_sec": pl.Float64,
                "tenji_rank": pl.Int64,
                "st_rank": pl.Int64,
                "course_avg_st": pl.Float64,
                "course_first_rate": pl.Float64,
                "course_3rd_rate": pl.Float64,
            },
            ignore_errors=True,
        )
        .with_columns(pl.col("hd").cast(pl.Utf8))
        .filter((pl.col("hd") >= d1) & (pl.col("hd") <= d2))
    )
    return lf


# -----------------------------
# 特徴量定義
# -----------------------------
ID_COLS = ["hd", "jcd", "rno", "lane", "regno"]
TARGET = "st_sec"

# staging 作成時点で揃っている典型的な説明変数たち
RAW_FEATURES: List[str] = [
    "tenji_st_sec",
    "tenji_is_f",
    "tenji_f_over_sec",
    "tenji_rank",
    "st_rank",
    "course_avg_st",
    "course_first_rate",
    "course_3rd_rate",
    # 学習上の安定化用フラグ（情報になる可能性があるのでそのまま使えるようにする）
    "st_observed",
    "st_is_f",
    "st_is_late",
    "st_penalized",
    # 位置（弱いがリークではない）
    "lane",
]


# -----------------------------
# 学習テーブル構築
# -----------------------------
def build_training_table(
    lf: pl.LazyFrame,
    drop_penalized: bool = True,
    drop_missing_target: bool = True,
) -> tuple[pl.DataFrame, List[str]]:
    """
    学習テーブルを構築。LightGBM が欠損を扱えるので数値の欠損はあえて埋めない。
    フラグは 0/1 化しておく。
    """
    # 必要列を安全に選択（存在しなければ None）
    def safe_col(name: str, dtype: pl.DataType | None = None) -> pl.Expr:
        expr = pl.when(pl.col(name).is_not_null()).then(pl.col(name)).otherwise(None)
        if dtype is not None:
            expr = expr.cast(dtype, strict=False)
        return expr.alias(name)

    need_cols = list(set(ID_COLS + [TARGET] + RAW_FEATURES))
    sel_exprs: List[pl.Expr] = [safe_col(c) for c in need_cols]
    lf = lf.select(sel_exprs)

    # フラグ系は 0/1 に正規化
    flag_cols = ["st_observed", "st_is_f", "st_is_late", "st_penalized", "tenji_is_f", "tenji_rank", "st_rank"]
    lf = lf.with_columns([pl.col(c).cast(pl.Int64).fill_null(0) for c in flag_cols])

    # 数値系の型を固定（欠損は保持）
    num_cols = [
        "st_sec",
        "tenji_st_sec",
        "tenji_f_over_sec",
        "course_avg_st",
        "course_first_rate",
        "course_3rd_rate",
    ]
    lf = lf.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num_cols])

    # 裁決あり（F/遅れ等）を除外したい場合
    if drop_penalized:
        lf = lf.filter(pl.col("st_penalized") == 0)

    # 目的変数の欠損除外
    if drop_missing_target:
        lf = lf.filter(pl.col(TARGET).is_not_null())

    # 出力用：ID・TARGET・特徴量
    out_exprs: List[pl.Expr] = [pl.col(c) for c in ID_COLS + [TARGET]]
    feature_cols: List[str] = []
    for c in RAW_FEATURES:
        # ID と被る列名は _feat に退避
        out_name = c if c not in ID_COLS else f"{c}_feat"
        out_exprs.append(pl.col(c).alias(out_name))
        feature_cols.append(out_name)

    df = lf.select(out_exprs).collect()
    return df, feature_cols


# -----------------------------
# LightGBM 学習
# -----------------------------
def train_lgbm(df: pl.DataFrame, feature_cols: Sequence[str]) -> tuple[lgb.Booster, List[str]]:
    pdf = df.to_pandas(use_pyarrow_extension_array=False)
    y = pdf[TARGET].astype(float).values
    X = pdf[list(feature_cols)]

    dset = lgb.Dataset(X, label=y, feature_name=list(feature_cols), free_raw_data=True)
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
        num_threads=0,
        seed=20240301,
        verbose=-1,
        force_col_wise=True,
    )
    booster = lgb.train(
        params,
        dset,
        num_boost_round=800,
        valid_sets=[dset],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    return booster, list(feature_cols)


# -----------------------------
# メイン
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--train_start", required=True, help="YYYYMMDD")
    ap.add_argument("--train_end",   required=True, help="YYYYMMDD")
    ap.add_argument("--model_out", default="data/models/st_lgbm.txt")
    ap.add_argument("--meta_out",  default="data/models/st_lgbm.meta.json")
    ap.add_argument("--fi_out",    default="data/models/st_lgbm.feature_importance.csv")
    ap.add_argument("--keep_penalized", action="store_true", help="裁決あり(F/遅れなど)も学習に含める")
    args = ap.parse_args()

    staging_root = Path(args.staging_root)

    # 収集
    lf_pool = read_staging_days(staging_root, args.train_start, args.train_end)

    # 学習テーブル
    df_train, feature_cols = build_training_table(
        lf_pool,
        drop_penalized=(not args.keep_penalized),
        drop_missing_target=True,
    )

    if df_train.height == 0:
        print("[FATAL] 学習データが0件です。staging に対象期間の st_train.csv があるか確認してください。", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] train rows: {df_train.height}, features: {len(feature_cols)} -> {feature_cols}")

    # 学習
    booster, feats = train_lgbm(df_train, feature_cols)

    # 保存
    model_out = Path(args.model_out)
    meta_out  = Path(args.meta_out)
    fi_out    = Path(args.fi_out)
    ensure_parent(model_out); ensure_parent(meta_out); ensure_parent(fi_out)

    booster.save_model(str(model_out))
    print(f"[SAVE] model -> {model_out}")

    meta = {
        "train_range": [args.train_start, args.train_end],
        "rows": int(df_train.height),
        "features": feats,
        "target": TARGET,
        "params": {
            "objective": "regression",
            "metric": "l2",
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
    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] meta -> {meta_out}")

    # Feature Importance
    fi = pd.DataFrame({
        "feature": feats,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(fi_out, index=False)
    print(f"[SAVE] feature importance -> {fi_out}")


if __name__ == "__main__":
    main()
