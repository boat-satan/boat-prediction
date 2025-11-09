#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
from typing import List, Tuple, Sequence, Union

import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb

# -----------------------------
# ユーティリティ
# -----------------------------
def zfill2(x: Union[int, str]) -> str:
    return str(x).zfill(2)

def read_staging_days(staging_root: Path, d1: str, d2: str) -> pl.LazyFrame:
    """
    staging_root/YYYY/MMDD/st_train.csv を d1..d2 で読み集約 (Lazy)
    """
    if len(d1) != 8 or len(d2) != 8:
        raise ValueError("train_start/train_end は YYYYMMDD 形式で指定してください。")
    y1, m1, d_1 = d1[:4], d1[4:6], d1[6:8]
    y2, m2, d_2 = d2[:4], d2[4:6], d2[6:8]

    # 年跨ぎでもシンプルにワイルドカードで読む（下位で日付フィルタ）
    pattern = str(staging_root / "*" / "*" / "st_train.csv")
    lf = pl.scan_csv(
        pattern,
        dtypes={
            "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64, "regno": pl.Utf8,
            "st_sec": pl.Float64, "st_is_f": pl.Int64, "st_is_late": pl.Int64, "st_penalized": pl.Int64, "st_observed": pl.Int64,
            "tenji_st_sec": pl.Float64, "tenji_is_f": pl.Int64, "tenji_f_over_sec": pl.Float64,
            "tenji_rank": pl.Int64, "st_rank": pl.Int64,
            "course_avg_st": pl.Float64, "course_first_rate": pl.Float64, "course_3rd_rate": pl.Float64,
        },
        ignore_errors=True,
    )
    # hd で日付範囲を絞る
    lf = (
        lf
        .with_columns(pl.col("hd").cast(pl.Utf8))
        .filter((pl.col("hd") >= d1) & (pl.col("hd") <= d2))
    )
    return lf

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 特徴量定義
# -----------------------------
# ベースの必須列（重複禁止）
BASE = ["hd", "jcd", "rno", "lane", "regno", "st_sec"]

# 学習に使う文字列列（Polars -> pandas）
FEATURES_STR: List[str] = [
    # 展示・経歴系
    "tenji_st_sec",
    "tenji_is_f",
    "tenji_f_over_sec",
    "tenji_rank",
    "st_rank",             # 公式掲示のST順序(展示)
    "course_avg_st",
    "course_first_rate",
    "course_3rd_rate",
    # 数値として入れるID的列
    "lane",
    # 将来: コース毎決まり手比率などを拡張するならここに追加
]

# 予測・学習対象
TARGET = "st_sec"

# -----------------------------
# データ整形
# -----------------------------
def build_training_table(
    lf: pl.LazyFrame,
    drop_penalized: bool = True,
    drop_missing_target: bool = True,
) -> Tuple[pl.DataFrame, List[str]]:
    """
    学習テーブルを作る。
    - 既に st_train.csv で F/遅れのフラグ化済み。
    - デフォルトではペナルティ付き（F/遅れ等）は学習から除外。
    """
    # まず列があるか念のため保障（存在しない場合は作る）
    def have(c: str) -> pl.Expr:
        return pl.when(pl.col(c).is_not_null()).then(pl.col(c)).otherwise(None).alias(c)

    cols_need = list(set(BASE + FEATURES_STR + ["st_is_f", "st_is_late", "st_penalized", "st_observed"]))
    sel_exprs: List[pl.Expr] = []
    for c in cols_need:
        sel_exprs.append(have(c))

    lf = lf.select(sel_exprs)

    # クリーニング
    lf = lf.with_columns([
        pl.col("st_observed").cast(pl.Int64).fill_null(0),
        pl.col("st_penalized").cast(pl.Int64).fill_null(0),
    ])

    if drop_penalized:
        lf = lf.filter(pl.col("st_penalized") == 0)

    if drop_missing_target:
        lf = lf.filter(pl.col(TARGET).is_not_null())

    # NaN埋め（LightGBMは欠損OKだが、rank/フラグは0埋め）
    lf = lf.with_columns([
        pl.col("tenji_rank").fill_null(0),
        pl.col("st_rank").fill_null(0),
        pl.col("tenji_is_f").fill_null(0),
        pl.col("tenji_f_over_sec").fill_null(0.0),
        pl.col("tenji_st_sec").fill_null(None),  # ここは欠損も学習上の情報にする
        pl.col("course_avg_st").fill_null(None),
        pl.col("course_first_rate").fill_null(None),
        pl.col("course_3rd_rate").fill_null(None),
    ])

    # BASE + FEATURES の重複除去
    features_clean = []
    reserved = set(BASE)
    for c in FEATURES_STR:
        if c in reserved:
            # 衝突回避：lane 等が重複する場合は _feat に退避
            features_clean.append(pl.col(c).alias(f"{c}_feat"))
        else:
            features_clean.append(pl.col(c))

    # select の最終組み立て
    out_exprs: List[pl.Expr] = [pl.col(c).alias(c) for c in BASE] + features_clean
    df = lf.select(out_exprs).collect()

    # 実際に残った特徴列名を抽出
    feature_cols: List[str] = []
    for e in features_clean:
        try:
            nm = e.meta.output_name()
        except Exception:
            # 念のため
            nm = None
        if nm is None:
            # 列名が取れなかった場合はスキップ
            continue
        feature_cols.append(nm)

    return df, feature_cols

# -----------------------------
# 学習
# -----------------------------
def train_lgbm(df: pl.DataFrame, feature_cols: Sequence[str]) -> Tuple[lgb.Booster, List[str]]:
    """
    Polars -> pandas に変換して LightGBM 回帰で学習
    """
    # pandas へ（pyarrow がない環境でも動くように safe に）
    pdf = df.to_pandas(use_pyarrow_extension_array=False)

    # 目的変数
    y = pdf[TARGET].astype(float).values

    # 特徴行列
    X = pdf[list(feature_cols)].copy()

    # LightGBM の NaN はそのままでOK
    ds = lgb.Dataset(X, label=y, feature_name=list(feature_cols), free_raw_data=True)

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
        ds,
        num_boost_round=800,
        valid_sets=[ds],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    return booster, list(feature_cols)

# -----------------------------
# メイン
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging", help="st_train.csv の親 (data/staging/YYYY/MMDD/)")
    ap.add_argument("--train_start", required=True, help="YYYYMMDD")
    ap.add_argument("--train_end",   required=True, help="YYYYMMDD")
    ap.add_argument("--model_out",   default="data/models/st_lgbm.txt")
    ap.add_argument("--meta_out",    default="data/models/st_lgbm.meta.json")
    ap.add_argument("--fi_out",      default="data/models/st_lgbm.feature_importance.csv")
    ap.add_argument("--keep_penalized", action="store_true", help="F/遅れ等のペナルティ付サンプルも学習に含める")
    args = ap.parse_args()

    staging_root = Path(args.staging_root)

    lf_pool = read_staging_days(staging_root, args.train_start, args.train_end)

    df_train, feature_cols = build_training_table(
        lf_pool,
        drop_penalized=(not args.keep_penalized),
        drop_missing_target=True,
    )

    if df_train.height == 0:
        print("[FATAL] 学習データが0件です。staging に対象期間の st_train.csv があるか確認してください。", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] train rows: {df_train.height}, features: {len(feature_cols)} -> {feature_cols}")

    booster, feats = train_lgbm(df_train, feature_cols)

    # 保存
    model_out = Path(args.model_out)
    meta_out  = Path(args.meta_out)
    fi_out    = Path(args.fi_out)
    ensure_dir(model_out)
    ensure_dir(meta_out)
    ensure_dir(fi_out)

    booster.save_model(str(model_out))
    print(f"[SAVE] model -> {model_out}")

    # メタ
    meta = {
        "train_range": [args.train_start, args.train_end],
        "rows": df_train.height,
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

    # FI
    fi = pd.DataFrame({
        "feature": feats,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(fi_out, index=False)
    print(f"[SAVE] feature importance -> {fi_out}")

if __name__ == "__main__":
    main()
