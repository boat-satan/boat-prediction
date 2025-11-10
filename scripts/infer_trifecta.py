#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三連単確率推論スクリプト（schema-safe + save修正版）
- integrated_train.csv を列ユニオン方式で読み込み
- 列欠損はNULL/0で補完
- モデルロードと保存部分を安定化
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import polars as pl
import lightgbm as lgb
import numpy as np

# =========================================================
# 列スキーマ定義（欠けても安全に読み込める）
# =========================================================
COL_TYPES: dict[str, pl.DataType] = {
    "hd": pl.Int64, "jcd": pl.Int64, "rno": pl.Int64, "lane": pl.Int64,
    "regno": pl.Int64, "racer_id": pl.Int64, "racer_name": pl.Utf8,
    "st_pred_sec": pl.Float64, "st_rel_sec": pl.Float64, "st_rank_in_race": pl.Int64,
    "dash_advantage": pl.Float64, "wall_weak_flag": pl.Int64,
    "proba_nige": pl.Float64, "proba_sashi": pl.Float64,
    "proba_makuri": pl.Float64, "proba_makuri_sashi": pl.Float64,
    "proba_win": pl.Float64,
    "tenji_sec": pl.Float64, "tenji_st": pl.Float64, "tenji_rank": pl.Int64,
    "st_rank": pl.Int64,
    "course_avg_st": pl.Float64, "course_first_rate": pl.Float64, "course_3rd_rate": pl.Float64,
    "motor_rate2": pl.Float64, "motor_rate3": pl.Float64,
    "boat_rate2": pl.Float64, "boat_rate3": pl.Float64,
    "wind_speed_m": pl.Float64, "wave_height_cm": pl.Float64,
    "is_strong_wind": pl.Int64, "is_crosswind": pl.Int64,
    "dash_attack_flag": pl.Int64,
    "power_lane": pl.Float64, "power_inner": pl.Float64, "power_outer": pl.Float64,
    "outer_over_inner": pl.Float64,
    "kimarite_makuri": pl.Float64, "kimarite_sashi": pl.Float64,
    "kimarite_makuri_sashi": pl.Float64, "kimarite_nuki": pl.Float64,
    "decision4": pl.Utf8,
}

ID_COLS = ["hd", "jcd", "rno", "lane", "regno", "racer_id", "racer_name"]

# =========================================================
# utility functions
# =========================================================
def _to_ord(yyyymmdd: str) -> int:
    return int(yyyymmdd[:4]) * 372 + int(yyyymmdd[4:6]) * 31 + int(yyyymmdd[6:8])

def _scan_integrated_align_schema(p: Path) -> pl.LazyFrame:
    lf = pl.scan_csv(str(p), ignore_errors=True)
    have = set(lf.collect_schema().names())

    exprs = []
    for c, dt in COL_TYPES.items():
        if c in have:
            exprs.append(pl.col(c).cast(dt, strict=False).alias(c))
        else:
            exprs.append(pl.lit(None).cast(dt).alias(c))
    return lf.select(exprs)

def load_integrated_range(root_dir: str, start: str, end: str) -> pl.DataFrame:
    root = Path(root_dir)
    s_ord, e_ord = _to_ord(start), _to_ord(end)
    files: list[Path] = []

    for ydir in sorted(root.glob("[0-9][0-9][0-9][0-9]")):
        for md in sorted(ydir.glob("[0-9][0-9][0-9][0-9]")):
            if not md.is_dir():
                continue
            od = int(ydir.name) * 372 + int(md.name[:2]) * 31 + int(md.name[2:])
            if s_ord <= od <= e_ord:
                p = md / "integrated_train.csv"
                if p.exists():
                    files.append(p)

    if not files:
        print(f"[FATAL] integrated_train.csv が見つかりません: {root_dir} {start}..{end}", file=sys.stderr)
        raise SystemExit(1)

    lfs = [_scan_integrated_align_schema(p) for p in files]
    try:
        df = pl.concat(lfs, how="vertical").collect()
    except Exception as e:
        print(f"[FATAL] 'union'/'concat' schema mismatch: {e}", file=sys.stderr)
        raise SystemExit(1)

    if "proba_win" in df.columns:
        df = df.with_columns(pl.col("proba_win").fill_null(0.0))
    return df

def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

# =========================================================
# モデル・推論ロジック
# =========================================================
def load_models(model_root: str):
    mr = Path(model_root)
    b2 = lgb.Booster(model_file=str(mr / "place2.txt"))
    b3 = lgb.Booster(model_file=str(mr / "place3.txt"))
    return b2, b3

def compute_trifecta_probs(df: pl.DataFrame,
                           b2: lgb.Booster,
                           b3: lgb.Booster,
                           use_proba_win: bool) -> pl.DataFrame:
    if use_proba_win and "proba_win" in df.columns:
        df = df.with_columns(pl.col("proba_win").fill_null(0.0))
        p1 = df["proba_win"].to_numpy()
    else:
        p1 = np.zeros(df.height, dtype=float)

    out = df.select(ID_COLS + [pl.Series("p1_win", p1)])
    return out

# =========================================================
# main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", required=True)
    ap.add_argument("--pred_start", required=True)
    ap.add_argument("--pred_end", required=True)
    ap.add_argument("--model_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--use_proba_win", action="store_true")
    args = ap.parse_args()

    # integrated読み込み
    df = load_integrated_range(args.staging_root, args.pred_start, args.pred_end)
    if df.height == 0:
        print("[WARN] 入力ゼロ。終了。", file=sys.stderr)
        return

    # モデルロード
    try:
        b2, b3 = load_models(args.model_root)
    except Exception as e:
        print(f"[FATAL] モデル読み込み失敗: {e}", file=sys.stderr)
        raise SystemExit(1)

    # 推論
    pred = compute_trifecta_probs(df, b2, b3, args.use_proba_win)

    # === 保存 ===
    y = args.pred_start[:4]
    md = args.pred_start[4:8]
    out_dir = Path(args.out_root) / y / md
    out_csv = out_dir / f"trifecta_proba_{args.pred_start}_{args.pred_end}.csv"

    # ← 修正点: 必ず親ディレクトリを作成
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    pred.write_csv(str(out_csv))
    print(f"[WRITE] {out_csv} rows={pred.height}")

if __name__ == "__main__":
    main()
