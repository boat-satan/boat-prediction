#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
place23 用 学習スクリプト（スキーマ不一致に強い安定版）

- 入力: data/integrated/YYYY/MMDD/integrated_train.csv
- 期間: --train_start YYYYMMDD .. --train_end YYYYMMDD
- 出力: model 等（従来の振る舞いを維持。学習ターゲット列は従来通りを優先）
- 変更点:
  1) 日跨ぎ結合を `pl.concat(..., how="diagonal")` で実施（列ユニオン）
  2) 学習で使う想定列を補完（欠けていれば null 追加）
  3) 必須列が null の行をドロップ
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import List

import polars as pl
import pandas as pd
import lightgbm as lgb


# ===== ログユーティリティ =====
def log(s: str): print(s, flush=True)
def err(s: str): print(s, file=sys.stderr, flush=True)


# ===== パス探索 =====
def _ymd_paths(root: Path, y1: str, y2: str) -> list[Path]:
    def to_ord(yyyymmdd: str) -> int:
        return int(yyyymmdd[:4]) * 372 + int(yyyymmdd[4:6]) * 31 + int(yyyymmdd[6:8])
    s_ord, e_ord = to_ord(y1), to_ord(y2)

    out: list[Path] = []
    for ydir in sorted(root.glob("[0-9]" * 4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]" * 4)):
            if not md.is_dir(): continue
            m, d = int(md.name[:2]), int(md.name[2:])
            od = y * 372 + m * 31 + d
            if s_ord <= od <= e_ord:
                p = md / "integrated_train.csv"
                if p.exists():
                    out.append(p)
    return out


# ===== データ読み込み（安定版） =====
# ここは “最小変更” でエラーを潰すのが目的
# - concat を diagonal に
# - 欠け列の補完
# - 必須列の欠損ドロップ
FEATURE_CANDIDATES: List[str] = [
    "st_pred_sec", "st_rel_sec", "st_rank_in_race",
    "dash_advantage", "wall_weak_flag",
    "proba_nige", "proba_sashi", "proba_makuri", "proba_makuri_sashi",
    "proba_win",
    "wind_speed_m", "wave_height_cm", "is_strong_wind", "is_crosswind",
    "dash_attack_flag",
    "power_lane", "power_inner", "power_outer", "outer_over_inner",
    "course_avg_st", "course_first_rate", "course_3rd_rate",
    "tenji_st", "tenji_sec", "tenji_rank", "st_rank",
    "motor_rate2", "motor_rate3", "boat_rate2", "boat_rate3",
    "kimarite_makuri", "kimarite_sashi", "kimarite_makuri_sashi", "kimarite_nuki",
]

ID_COLS = ["hd","jcd","rno","lane","regno"]

# 必須列（これが無い/欠損は学習に使えない想定）
REQUIRED_MIN = ID_COLS + [
    # 学習に最低限使いたいカラム（従来の処理を尊重しつつ汎化）
    "proba_win", "proba_nige", "proba_sashi", "proba_makuri", "proba_makuri_sashi",
    "st_pred_sec"
]

# ラベル候補（従来のスクリプトがどれを使っていても動きやすく）
LABEL_CANDIDATES = [
    "label_place23",      # 2着or3着フラグ（0/1）
    "label_top3",         # 上位3着フラグ（0/1）
    "is_place23",         # 同上の別名
    "is_top3",            # 同上の別名
]

def load_train(files: list[Path]) -> pl.DataFrame:
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in files]
    # ★ここが肝：列ユニオンで結合（足りない列は null 埋め）
    df = pl.concat(lfs, how="diagonal").collect()

    # 型の素直化（IDは数値→Utf8でもOKだが、下流互換で数値寄せ）
    cast_map = {
        "hd": pl.Int64, "jcd": pl.Int64, "rno": pl.Int64, "lane": pl.Int64, "regno": pl.Int64
    }
    for k, dt in cast_map.items():
        if k in df.columns:
            df = df.with_columns(pl.col(k).cast(dt, strict=False))

    # 欠けている列を補完（null で作る）
    for c in set(FEATURE_CANDIDATES + REQUIRED_MIN + LABEL_CANDIDATES):
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))

    # 必須列の欠損除去
    df = df.drop_nulls([c for c in REQUIRED_MIN if c in df.columns])

    return df


# ===== 学習（従来の流儀を尊重：バイナリ place23 例） =====
def pick_label_name(df: pl.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            # 値が0/1 として使えるか軽く確認
            try:
                s = df.select(pl.col(c)).to_series()
                if s.drop_nulls().unique().to_list() and set(s.drop_nulls().unique()) <= {0,1}:
                    return c
            except Exception:
                pass
    err("[FATAL] ラベル列が見つかりません（候補: %s）" % LABEL_CANDIDATES)
    sys.exit(1)

def select_features(df: pl.DataFrame) -> list[str]:
    feats = []
    for c in FEATURE_CANDIDATES:
        if c in df.columns:
            feats.append(c)
    # あまりに列が少ない場合の保険
    if len(feats) == 0:
        err("[FATAL] 学習に使える特徴量がありません。integrated の生成を確認してください。")
        sys.exit(1)
    return feats

def train_model(df: pl.DataFrame, label_col: str, feat_cols: list[str]) -> lgb.Booster:
    pdf = df.select(feat_cols + [label_col]).to_pandas(use_pyarrow_extension_array=False)
    y = pdf[label_col].astype(int).values
    X = pdf[feat_cols]

    dtrain = lgb.Dataset(X, label=y, feature_name=list(X.columns), free_raw_data=True)
    params = dict(
        objective="binary",
        metric="auc",
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
        dtrain,
        num_boost_round=800,
        valid_sets=[dtrain],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    return booster


# ===== メイン =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated_root", default="data/integrated")
    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end",   required=True)
    ap.add_argument("--model_root",  default="data/models/place23")
    args = ap.parse_args()

    root = Path(args.integrated_root)
    files = _ymd_paths(root, args.train_start, args.train_end)
    if not files:
        err(f"[FATAL] integrated_train.csv が見つかりません: {args.integrated_root} {args.train_start}..{args.train_end}")
        sys.exit(1)

    log(f"[INFO] train files: {len(files)}")

    # ★ 強化した読み込み
    df = load_train(files)
    if df.height == 0:
        err("[FATAL] 出力対象データがありません（必須列の欠損で全落ちした可能性）。")
        sys.exit(1)

    log(f"[INFO] merged rows: {df.height}")
    label_col = pick_label_name(df)
    feat_cols  = select_features(df)
    log(f"[INFO] label: {label_col}")
    log(f"[INFO] features: {len(feat_cols)} -> {feat_cols[:12]}{'...' if len(feat_cols)>12 else ''}")

    booster = train_model(df, label_col, feat_cols)

    # 保存
    model_dir = Path(args.model_root)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.txt"
    meta_path  = model_dir / "meta.json"
    fi_path    = model_dir / "feature_importance.csv"

    booster.save_model(str(model_path))

    import json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "train_range": [args.train_start, args.train_end],
            "rows": int(df.height),
            "label": label_col,
            "features": feat_cols,
            "params": {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.05,
                "num_leaves": 63,
                "min_data_in_leaf": 50,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.9,
                "bagging_freq": 1,
                "lambda_l2": 1.0,
                "num_boost_round": 800,
                "seed": 20240301,
            }
        }, f, ensure_ascii=False, indent=2)

    # FI
    fi = pd.DataFrame({
        "feature": feat_cols,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(fi_path, index=False)

    log(f"[SAVE] model -> {model_path}")
    log(f"[SAVE] meta  -> {meta_path}")
    log(f"[SAVE] FI    -> {fi_path}")


if __name__ == "__main__":
    main()
