#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三連単確率推論スクリプト（place2/3モデル使用）

入力:
  data/integrated/YYYY/MMDD/integrated_pro.csv（or integrated_train.csv）
  + place2/place3 モデル
出力:
  data/proba/trifecta/YYYY/MMDD/trifecta_proba_YYYYMMDD_YYYYMMDD.csv
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import glob
import polars as pl
import lightgbm as lgb


# ======================================================
# Utility
# ======================================================
def log(msg: str):
    print(msg, flush=True)


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ======================================================
# ファイル探索
# ======================================================
def _ymd_paths(root: Path, yyyymmdd_from: str, yyyymmdd_to: str) -> list[Path]:
    def ordkey(s: str) -> int:
        return int(s[:4]) * 372 + int(s[4:6]) * 31 + int(s[6:8])
    s, e = yyyymmdd_from, yyyymmdd_to
    picked = []
    for ydir in sorted(root.glob("[0-9]" * 4)):
        if not ydir.is_dir():
            continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]" * 4)):
            if not md.is_dir():
                continue
            d = f"{y}{md.name}"
            if ordkey(s) <= ordkey(d) <= ordkey(e):
                picked.append(md)
    return picked


def find_integrated_csvs(integrated_root: str, start: str, end: str) -> list[Path]:
    """
    優先: integrated_pro.csv
    無ければ: integrated_train.csv
    さらに無ければ: *integrated*.csv の先頭ヒットを採用
    """
    root = Path(integrated_root)
    day_dirs = _ymd_paths(root, start, end)
    files: list[Path] = []
    for dd in day_dirs:
        cand = dd / "integrated_pro.csv"
        if cand.exists():
            files.append(cand)
            continue
        cand2 = dd / "integrated_train.csv"
        if cand2.exists():
            files.append(cand2)
            continue
        globs = sorted(Path(p) for p in glob.glob(str(dd / "*integrated*.csv")))
        if globs:
            files.append(globs[0])
    return files


def load_range(integrated_root: str, start: str, end: str) -> pl.DataFrame:
    paths = find_integrated_csvs(integrated_root, start, end)
    if not paths:
        raise SystemExit(f"[FATAL] integrated_* CSV が見つかりません: {integrated_root} {start}..{end}")
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in paths]
    df = pl.concat(lfs).collect()
    log(f"[INFO] integrated files: {len(paths)}  rows={df.height}")
    return df


# ======================================================
# 推論ロジック
# ======================================================
def infer_trifecta(df: pl.DataFrame, model_root: Path, use_proba_win: bool = False) -> pl.DataFrame:
    log("[INFO] Loading LightGBM models (place2 / place3)")
    b2_path = model_root / "place2.txt"
    b3_path = model_root / "place3.txt"

    if not b2_path.exists() or not b3_path.exists():
        raise SystemExit(f"[FATAL] モデルが不足しています: {b2_path}, {b3_path}")

    b2 = lgb.Booster(model_file=str(b2_path))
    b3 = lgb.Booster(model_file=str(b3_path))

    # 特徴量選択（モデルに存在する feature_name と共通部分を使う）
    f2 = [f for f in df.columns if f in b2.feature_name()]
    f3 = [f for f in df.columns if f in b3.feature_name()]

    X2 = df.select(f2).to_pandas()
    X3 = df.select(f3).to_pandas()

    log(f"[INFO] Predicting with place2 ({len(f2)} feats) and place3 ({len(f3)} feats)")
    df2 = df.with_columns(pl.Series("proba_place2", b2.predict(X2)))
    df3 = df2.with_columns(pl.Series("proba_place3", b3.predict(X3)))

    # 三連単用正規化（簡易版：proba_win * place2 * place3）
    if "proba_win" in df3.columns and use_proba_win:
        df3 = df3.with_columns(
            (pl.col("proba_win") * pl.col("proba_place2") * pl.col("proba_place3"))
            .alias("proba_trifecta_raw")
        )
    else:
        df3 = df3.with_columns(
            (pl.col("proba_place2") * pl.col("proba_place3"))
            .alias("proba_trifecta_raw")
        )

    return df3


# ======================================================
# メイン
# ======================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", required=True, help="data/integrated")
    ap.add_argument("--pred_start", required=True)
    ap.add_argument("--pred_end", required=True)
    ap.add_argument("--model_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--use_proba_win", action="store_true", help="proba_win を掛ける")
    args = ap.parse_args()

    df = load_range(args.staging_root, args.pred_start, args.pred_end)
    model_root = Path(args.model_root)

    out_df = infer_trifecta(df, model_root, use_proba_win=args.use_proba_win)

    # 出力
    y = args.pred_start[:4]
    md = args.pred_start[4:8]
    out_csv = ensure_parent(Path(args.out_root) / y / md / f"trifecta_proba_{args.pred_start}_{args.pred_end}.csv")
    out_df.write_csv(str(out_csv))
    log(f"[WRITE] {out_csv} rows={out_df.height}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        sys.exit(1)
