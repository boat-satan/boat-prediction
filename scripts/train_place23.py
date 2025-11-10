#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
place2/3（Top3）学習スクリプト（integrated × results 結合でラベル生成）

入力:
  - data/integrated/YYYY/MMDD/integrated_train.csv
  - data/results/YYYY/MMDD/{jcd}/{rno}R.json  ※想定パス

出力:
  - data/models/place23/model.txt
  - data/models/place23/meta.json
  - data/models/place23/feature_importance.csv
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import List, Dict, Iterable, Optional, Tuple

import polars as pl
import pandas as pd
import lightgbm as lgb

# --------------------------------------------------
# 設定
# --------------------------------------------------
ID_COLS = ["hd", "jcd", "rno", "lane", "regno"]

LABEL_CANDIDATES = ["label_place23", "label_top3", "is_place23", "is_top3"]

PREFERRED_DTYPES: Dict[str, pl.DataType] = {
    "hd": pl.Int64, "jcd": pl.Int64, "rno": pl.Int64, "lane": pl.Int64, "regno": pl.Int64,
    "racer_id": pl.Int64, "st_rank_in_race": pl.Int64, "st_rank": pl.Int64,
    "tenji_rank": pl.Int64, "is_crosswind": pl.Int64, "is_strong_wind": pl.Int64,
    "dash_attack_flag": pl.Int64, "wall_weak_flag": pl.Int64,
    "decision4": pl.Utf8, "racer_name": pl.Utf8,
}

EXCLUDE_COLS = set(ID_COLS + ["racer_id", "racer_name", "decision4"])  # 説明系/IDは特徴から除外

# --------------------------------------------------
# ユーティリティ
# --------------------------------------------------
def log(m: str): print(m, flush=True)
def err(m: str): print(m, file=sys.stderr, flush=True)
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True); return p

def _ord_ymd(s: str) -> int:
    return int(s[:4]) * 372 + int(s[4:6]) * 31 + int(s[6:8])

def _ymd_paths(root: Path, ymd1: str, ymd2: str) -> List[Path]:
    s, e = _ord_ymd(ymd1), _ord_ymd(ymd2)
    out: List[Path] = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            y, m, d = int(ydir.name), int(md.name[:2]), int(md.name[2:])
            o = y*372 + m*31 + d
            if s <= o <= e:
                out.append(md)
    return out

# --------------------------------------------------
# integrated 読み込み（列そろえ）
# --------------------------------------------------
def _read_header_cols(p: Path) -> List[str]:
    try:
        return pl.read_csv(str(p), n_rows=0).columns
    except Exception:
        return []

def _columns_union(files: Iterable[Path]) -> List[str]:
    seen: List[str] = []
    S = set()
    for fp in files:
        for c in _read_header_cols(fp):
            if c not in S:
                seen.append(c); S.add(c)
    pref = list(PREFERRED_DTYPES.keys())
    rest = [c for c in seen if c not in pref]
    return pref + rest

def _dtype_for(c: str) -> pl.DataType:
    return PREFERRED_DTYPES.get(c, pl.Float64)

def _scan_align(fp: Path, all_cols: List[str]) -> pl.LazyFrame:
    lf = pl.scan_csv(str(fp), ignore_errors=True)
    names = set(lf.collect_schema().names())
    exprs: List[pl.Expr] = []
    for c in all_cols:
        if c in names:
            exprs.append(pl.col(c).cast(_dtype_for(c), strict=False).alias(c))
        else:
            exprs.append(pl.lit(None, dtype=_dtype_for(c)).alias(c))
    return lf.select(exprs)

def load_integrated_integral(integrated_root: str, start: str, end: str) -> pl.DataFrame:
    root = Path(integrated_root)
    day_dirs = _ymd_paths(root, start, end)
    files = [dd / "integrated_train.csv" for dd in day_dirs if (dd / "integrated_train.csv").exists()]
    if not files:
        return pl.DataFrame()
    all_cols = _columns_union(files)
    lfs = [_scan_align(fp, all_cols) for fp in files]
    log(f"[INFO] train files: {len(files)}")
    df = pl.concat(lfs).collect()
    log(f"[INFO] merged rows: {df.height}")
    return df

# --------------------------------------------------
# results ⇒ Top3 ラベル作成
# 期待パス: data/results/YYYY/MMDD/{jcd}/{rno}R.json
# JSONの構造が多少違っても、lane と 順位(1..6) を頑張って抽出
# --------------------------------------------------
def _iter_result_json_paths(results_root: str, start: str, end: str) -> Iterable[Tuple[Path, int, int, int]]:
    root = Path(results_root)
    for dd in _ymd_paths(root, start, end):
        # /YYYY/MMDD 下の jcd ディレクトリを全探索
        for jdir in sorted(dd.glob("[0-9][0-9]")):
            jcd = int(jdir.name)
            for jp in sorted(jdir.glob("*R.json")):
                # ファイル名: 10R.json → rno
                name = jp.stem  # "10R"
                if name.endswith("R") and name[:-1].isdigit():
                    rno = int(name[:-1])
                    # hd はディレクトリから
                    hd = int(f"{dd.parent.name}{dd.name}")
                    yield jp, hd, jcd, rno

def _to_int_or_none(x) -> Optional[int]:
    if x is None: return None
    try:
        return int(x)
    except Exception:
        # "1位", "F", "妨", "欠" などに対応（数字以外は None）
        s = str(x)
        digits = "".join(ch for ch in s if ch.isdigit())
        return int(digits) if digits else None

def _extract_entries(js) -> List[dict]:
    """
    JSON から (lane, rank) 相当を持つ要素配列を返す。
    想定キー例:
      - top-level: results / res / entries
      - per entry: lane / course / pit / lane_no, rank / finish / arrival / order / goal_rank
    """
    # どのキーに配列があるか当てに行く
    for key in ("results", "res", "entries", "result", "data"):
        arr = js.get(key)
        if isinstance(arr, list):
            return arr
    # 直接 list な場合
    if isinstance(js, list):
        return js
    # ダメなら空
    return []

def _lane_and_rank(rec: dict) -> Tuple[Optional[int], Optional[int]]:
    lane = None
    for k in ("lane", "course", "pit", "lane_no", "entry", "waku"):
        if k in rec:
            lane = _to_int_or_none(rec[k]); break
    rank = None
    for k in ("rank", "finish", "arrival", "order", "goal_rank", "result_rank"):
        if k in rec:
            rank = _to_int_or_none(rec[k]); break
    return lane, rank

def build_top3_labels_from_results(results_root: str, start: str, end: str) -> pl.DataFrame:
    rows: List[dict] = []
    for jp, hd, jcd, rno in _iter_result_json_paths(results_root, start, end):
        try:
            with jp.open("r", encoding="utf-8") as f:
                js = json.load(f)
        except Exception:
            continue
        for rec in _extract_entries(js):
            lane, rank = _lane_and_rank(rec)
            if lane is None or rank is None:
                continue
            if not (1 <= lane <= 6):
                continue
            rows.append({
                "hd": hd, "jcd": jcd, "rno": rno, "lane": lane,
                "__finish_rank__": rank,
                "__is_top3__": 1 if (1 <= rank <= 3) else 0,
            })
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows).with_columns([
        pl.col("hd").cast(pl.Int64),
        pl.col("jcd").cast(pl.Int64),
        pl.col("rno").cast(pl.Int64),
        pl.col("lane").cast(pl.Int64),
        pl.col("__finish_rank__").cast(pl.Int64),
        pl.col("__is_top3__").cast(pl.Int64),
    ])
    log(f"[INFO] results join rows: {df.height}")
    return df

# --------------------------------------------------
# 特徴量選択
# --------------------------------------------------
def select_features(df: pl.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feat_cols: List[str] = []
    for c, dt in df.schema.items():
        if c == label_col or c in EXCLUDE_COLS: continue
        sdt = str(dt)
        if "Int" in sdt or "Float" in sdt:
            feat_cols.append(c)
    pdf = df.select(ID_COLS + feat_cols + [label_col]).to_pandas(use_pyarrow_extension_array=False)
    y = pdf[label_col].astype(int)
    X = pdf[feat_cols]
    return X, y, feat_cols

# --------------------------------------------------
# 学習
# --------------------------------------------------
def train_lgbm_classifier(X: pd.DataFrame, y: pd.Series, feat_cols: List[str]) -> lgb.Booster:
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

# --------------------------------------------------
# メイン
# --------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated_root", default="data/integrated")
    ap.add_argument("--results_root", default="data/results")
    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--model_root", default="data/models/place23")
    args = ap.parse_args()

    # integrated 読み込み
    df_int = load_integrated_integral(args.integrated_root, args.train_start, args.train_end)
    if df_int.height == 0:
        err(f"[FATAL] integrated_train.csv が見つかりません: {args.integrated_root} {args.train_start}..{args.train_end}")
        sys.exit(1)

    # ラベル列の既存チェック
    label_col = next((c for c in LABEL_CANDIDATES if c in df_int.columns), None)

    if label_col:
        log(f"[INFO] label column detected in integrated -> {label_col}")
        df = df_int
    else:
        # results から Top3 ラベル生成して結合
        df_res = build_top3_labels_from_results(args.results_root, args.train_start, args.train_end)
        if df_res.height == 0:
            err("[FATAL] results から順位が取得できませんでした。results のパス/形式を確認してください。")
            sys.exit(1)
        df = df_int.join(df_res, on=["hd","jcd","rno","lane"], how="left")
        if "__is_top3__" not in df.columns:
            err("[FATAL] results 結合に失敗しました（キー不一致の可能性）。")
            sys.exit(1)
        # 結合できなかった行は除外（学習に不要）
        before = df.height
        df = df.filter(pl.col("__is_top3__").is_not_null())
        after = df.height
        log(f"[INFO] dropped rows without result labels: {before-after}")
        if after == 0:
            err("[FATAL] 結合後に学習可能な行が0件です。")
            sys.exit(1)
        label_col = "__is_top3__"

    # 特徴量抽出
    X, y, feat_cols = select_features(df, label_col)
    log(f"[INFO] features: {len(feat_cols)} -> {feat_cols[:12]}{'...' if len(feat_cols)>12 else ''}")
    log(f"[INFO] train rows: {len(X)} label={label_col}")

    # 学習
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
        "label_source": "integrated" if label_col in LABEL_CANDIDATES else "results_top3",
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
