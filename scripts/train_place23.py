#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
place2/3（2着以内=place2 or 3着以内=place3）系 学習スクリプト（列スキーマ自動整形・全差し替え）

- 入力: data/integrated/YYYY/MMDD/integrated_train.csv
  ※日によって列が違っても OK（足りない列は自動で Null 追加 & 型揃え）

- ラベル列の候補（いずれか必須）:
    - 'label_place23' / 'label_top3' / 'is_place23' / 'is_top3' （1/0 または True/False）
  ※見つからない場合は FATAL で終了

- 出力:
  モデル: data/models/place23/model.txt (+ meta.json / feature_importance.csv)
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Dict, Iterable

import polars as pl
import pandas as pd
import lightgbm as lgb

# =========================
# 設定
# =========================
LABEL_CANDIDATES: List[str] = ["label_place23", "label_top3", "is_place23", "is_top3"]

# 「この列はこの型で扱いたい」という希望（存在しない列は自動で追加）
PREFERRED_DTYPES: Dict[str, pl.DataType] = {
    # IDs / ints
    "hd": pl.Int64, "jcd": pl.Int64, "rno": pl.Int64, "lane": pl.Int64, "regno": pl.Int64,
    "racer_id": pl.Int64, "st_rank_in_race": pl.Int64, "st_rank": pl.Int64,
    "tenji_rank": pl.Int64, "is_crosswind": pl.Int64, "is_strong_wind": pl.Int64,
    "dash_attack_flag": pl.Int64, "wall_weak_flag": pl.Int64,
    # strings
    "decision4": pl.Utf8, "racer_name": pl.Utf8,
    # floats（代表的なもの）
    "st_pred_sec": pl.Float64, "st_rel_sec": pl.Float64, "dash_advantage": pl.Float64,
    "proba_nige": pl.Float64, "proba_sashi": pl.Float64, "proba_makuri": pl.Float64, "proba_makuri_sashi": pl.Float64,
    "proba_win": pl.Float64,
    "wind_speed_m": pl.Float64, "wave_height_cm": pl.Float64,
    "kimarite_sashi": pl.Float64, "kimarite_makuri": pl.Float64, "kimarite_makuri_sashi": pl.Float64, "kimarite_nuki": pl.Float64,
    "power_lane": pl.Float64, "power_inner": pl.Float64, "power_outer": pl.Float64, "outer_over_inner": pl.Float64,
    "tenji_st": pl.Float64, "tenji_sec": pl.Float64,
    "boat_rate2": pl.Float64, "boat_rate3": pl.Float64, "motor_rate2": pl.Float64, "motor_rate3": pl.Float64,
    "course_avg_st": pl.Float64, "course_first_rate": pl.Float64, "course_3rd_rate": pl.Float64,
}

# ============== ユーティリティ ==============
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

def find_integrated_csvs(integrated_root: str, start: str, end: str) -> list[Path]:
    root = Path(integrated_root)
    day_dirs = _ymd_paths(root, start, end)
    files: list[Path] = []
    for dd in day_dirs:
        p = dd / "integrated_train.csv"
        if p.exists():
            files.append(p)
    return files

# ============== スキーマ整形（核） ==============
def read_header_cols(p: Path) -> List[str]:
    # 最速のヘッダー取得（データは読まない）
    try:
        df0 = pl.read_csv(str(p), n_rows=0)
        return df0.columns
    except Exception:
        # 万一でも空配列返し
        return []

def columns_union(files: Iterable[Path]) -> List[str]:
    seen = []
    seen_set = set()
    for fp in files:
        cols = read_header_cols(fp)
        for c in cols:
            if c not in seen_set:
                seen.append(c); seen_set.add(c)
    # 望む列順：既知の優先列 → 見つかったその他
    preferred_order = list(PREFERRED_DTYPES.keys())
    rest = [c for c in seen if c not in preferred_order]
    return preferred_order + rest

def dtype_for(col: str) -> pl.DataType:
    # 既知で指定、なければ Float64 を既定（stringは上で指定してる）
    return PREFERRED_DTYPES.get(col, pl.Float64)

def scan_and_align(fp: Path, all_cols: List[str]) -> pl.LazyFrame:
    lf = pl.scan_csv(str(fp), ignore_errors=True)
    # 現在持っている列名
    schema_names = set(lf.collect_schema().names())
    exprs: List[pl.Expr] = []
    for c in all_cols:
        if c in schema_names:
            exprs.append(pl.col(c).cast(dtype_for(c), strict=False).alias(c))
        else:
            # 欠損列を Null で追加し、型を合わせる
            exprs.append(pl.lit(None, dtype=dtype_for(c)).alias(c))
    return lf.select(exprs)

def load_train(files: List[Path]) -> pl.DataFrame:
    if not files:
        return pl.DataFrame()
    # 列の完全和集合を作る
    all_cols = columns_union(files)
    lfs = [scan_and_align(fp, all_cols) for fp in files]
    log(f"[INFO] train files: {len(files)}")
    df = pl.concat(lfs).collect()
    log(f"[INFO] merged rows: {df.height}")
    return df

# ============== ラベル抽出 ==============
def pick_label_column(df: pl.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return ""

# ============== 特徴量選択 ==============
ID_COLS = ["hd", "jcd", "rno", "lane", "regno"]
EXCLUDE_COLS = set(ID_COLS + ["racer_id", "racer_name", "decision4"])  # ID/説明系は除外

def select_features(df: pl.DataFrame, label_col: str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    cols = df.columns
    feat_cols: List[str] = []
    for c in cols:
        if c in EXCLUDE_COLS or c == label_col:
            continue
        # 数値っぽい列を採用
        dt = df.schema.get(c)
        if isinstance(dt, pl.datatypes.DataTypeClass):
            dt = str(dt)
        if "Int" in str(dt) or "Float" in str(dt):
            feat_cols.append(c)

    # pandas へ
    pdf = df.select(ID_COLS + feat_cols + [label_col]).to_pandas(use_pyarrow_extension_array=False)
    y = pdf[label_col].astype(int)  # 1/0 を期待
    X = pdf[feat_cols]
    return X, y, feat_cols

# ============== 学習 ==============
def train_lgbm_classifier(X: pd.DataFrame, y: pd.Series, feat_cols: list[str]) -> lgb.Booster:
    dtrain = lgb.Dataset(X, label=y, feature_name=list(feat_cols), free_raw_data=True)
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

# ============== メイン ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated_root", default="data/integrated")
    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--model_root", default="data/models/place23")
    args = ap.parse_args()

    files = find_integrated_csvs(args.integrated_root, args.train_start, args.train_end)
    if not files:
        err(f"[FATAL] integrated_train.csv が見つかりません: {args.integrated_root} {args.train_start}..{args.train_end}")
        sys.exit(1)

    df = load_train(files)
    if df.height == 0:
        err("[FATAL] 結合後の学習データが0件です。")
        sys.exit(1)

    label_col = pick_label_column(df)
    if not label_col:
        err(f"[FATAL] ラベル列が見つかりません（候補: {LABEL_CANDIDATES}）")
        sys.exit(1)

    X, y, feat_cols = select_features(df, label_col)
    log(f"[INFO] features: {len(feat_cols)} -> {feat_cols[:12]}{'...' if len(feat_cols)>12 else ''}")
    log(f"[INFO] train rows: {len(X)} label={label_col}")

    booster = train_lgbm_classifier(X, y, feat_cols)

    # 保存
    model_path = ensure_parent(Path(args.model_root) / "model.txt")
    meta_path  = ensure_parent(Path(args.model_root) / "meta.json")
    fi_path    = ensure_parent(Path(args.model_root) / "feature_importance.csv")

    booster.save_model(str(model_path))
    log(f"[SAVE] model -> {model_path}")

    meta = {
        "train_range": [args.train_start, args.train_end],
        "rows": int(len(X)),
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
        },
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[SAVE] meta -> {meta_path}")

    fi = pd.DataFrame({
        "feature": feat_cols,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(fi_path, index=False)
    log(f"[SAVE] feature importance -> {fi_path}")

if __name__ == "__main__":
    main()
