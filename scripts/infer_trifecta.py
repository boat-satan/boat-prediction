#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys, json, glob, itertools
from pathlib import Path
from typing import List, Dict, Any

import polars as pl
import pandas as pd
import lightgbm as lgb
import numpy as np

RACE_KEY = ["hd","jcd","rno"]
ID_COLS  = ["hd","jcd","rno","lane","regno"]

def log(*a): print(*a, flush=True)
def err(*a): print(*a, file=sys.stderr, flush=True)
def ensure_parent(p: Path): p.parent.mkdir(parents=True, exist_ok=True)

def _ymd_dirs(root: Path, start: str, end: str) -> list[Path]:
    def to_ord(s): return int(s[:4])*372 + int(s[4:6])*31 + int(s[6:8])
    s, e = to_ord(start), to_ord(end)
    out = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        for md in sorted(ydir.glob("[0-9]"*4)):
            oo = int(ydir.name)*372 + int(md.name[:2])*31 + int(md.name[2:])
            if s <= oo <= e: out.append(md)
    return out

def find_csvs(staging_root: str, start: str, end: str, name="integrated_pro.csv") -> list[Path]:
    files = []
    for dd in _ymd_dirs(Path(staging_root), start, end):
        p = dd / name
        if p.exists(): files.append(p)
        else:
            for a in glob.glob(str(dd/"*integrated*pro*.csv")):
                files.append(Path(a)); break
    return files

def load_pro(paths: list[Path]) -> pl.DataFrame:
    if not paths: return pl.DataFrame()
    df = pl.concat([pl.scan_csv(str(p), ignore_errors=True) for p in paths]).collect()
    # 型
    for c in ("hd","jcd"):
        if c in df.columns: df = df.with_columns(pl.col(c).cast(pl.Utf8))
    for c in ("rno","lane"):
        if c in df.columns: df = df.with_columns(pl.col(c).cast(pl.Int64))
    return df

def pick_numeric_features(df: pl.DataFrame, extra_keep: list[str]) -> list[str]:
    bad = set(ID_COLS + ["rank","first_lane","second_lane"])  # rankは予測期には無い想定
    for s in ["racer_name","decision4"]:
        if s in df.columns: bad.add(s)
    feats = [c for c,dt in df.schema.items() if (c not in bad) and ("Utf8" not in str(dt))]
    for k in extra_keep:
        if k not in feats: feats.append(k)
    return feats

def to_pandas(df: pl.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.select(cols).to_pandas(use_pyarrow_extension_array=False)

def normalize_probs(vals: np.ndarray, mask: np.ndarray|None=None) -> np.ndarray:
    arr = vals.copy().astype(float)
    if mask is not None: arr = arr[mask]
    s = arr.sum()
    if s <= 0 or not np.isfinite(s): 
        # すべて同率で
        if mask is None: return np.ones_like(vals)/len(vals)
        else:
            out = np.zeros_like(vals); out[mask] = 1.0/arr.size; return out
    if mask is None:
        return arr / s
    else:
        out = np.zeros_like(vals, dtype=float)
        out[mask] = arr / s
        return out

def predict_place2(booster: lgb.Booster, race_df: pl.DataFrame, feats: list[str], first_lane: int) -> np.ndarray:
    df = race_df.with_columns(pl.lit(first_lane).cast(pl.Int64).alias("first_lane"))
    X = to_pandas(df, feats)
    p = booster.predict(X)  # prob of class 1
    return p

def predict_place3(booster: lgb.Booster, race_df: pl.DataFrame, feats: list[str], first_lane: int, second_lane: int) -> np.ndarray:
    df = (race_df
          .with_columns(pl.lit(first_lane).cast(pl.Int64).alias("first_lane"))
          .with_columns(pl.lit(second_lane).cast(pl.Int64).alias("second_lane")))
    X = to_pandas(df, feats)
    p = booster.predict(X)
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--pred_start", required=True)
    ap.add_argument("--pred_end", required=True)
    ap.add_argument("--model_root", default="data/models/trifecta")
    ap.add_argument("--out_root", default="data/proba/3t")
    ap.add_argument("--use_proba_win", action="store_true", help="integrated_pro.csvのproba_winをP1として使用（なければ均等）")
    args = ap.parse_args()

    # モデル読み込み
    b2 = lgb.Booster(model_file=str(Path(args.model_root)/"place2.txt"))
    b3 = lgb.Booster(model_file=str(Path(args.model_root)/"place3.txt"))

    # 推論入力
    files = find_csvs(args.staging_root, args.pred_start, args.pred_end, "integrated_pro.csv")
    if not files:
        err(f"[FATAL] integrated_pro.csv が見つかりません: {args.staging_root} {args.pred_start}..{args.pred_end}")
        sys.exit(1)
    log(f"[INFO] infer files: {len(files)}")
    df = load_pro(files)

    # レース毎に処理
    out_rows: List[Dict[str,Any]] = []
    # 1着のprior特徴（proba_win等）
    has_p1 = args.use_proba_win and ("proba_win" in df.columns)

    # place2/3の特徴選定（first_lane/second_laneはスクリプトで付与）
    feats2 = pick_numeric_features(df, extra_keep=["first_lane"])
    feats3 = pick_numeric_features(df, extra_keep=["first_lane","second_lane"])

    for (hd,jcd,rno), g in df.groupby(RACE_KEY, maintain_order=True):
        race = g.sort("lane")
        lanes = race["lane"].to_list()
        regno = race["regno"].to_list()

        # P1
        if has_p1:
            p1_raw = race["proba_win"].to_numpy()
            # 念のため正規化
            p1 = normalize_probs(p1_raw)
        else:
            p1 = np.ones(len(lanes))/len(lanes)

        # P2,P3による連鎖
        # 全120通り
        for i, a in enumerate(lanes):
            # b候補のマスク
            mask_b = np.array([x!=a for x in lanes])
            p2_raw = predict_place2(b2, race, feats2, first_lane=a)
            p2 = normalize_probs(p2_raw, mask_b)  # b≠aに正規化

            for j, b in enumerate(lanes):
                if b==a: continue
                # c候補
                mask_c = np.array([(x!=a) and (x!=b) for x in lanes])
                p3_raw = predict_place3(b3, race, feats3, first_lane=a, second_lane=b)
                p3 = normalize_probs(p3_raw, mask_c)  # c≠a,bに正規化

                # cを走査
                for k, c in enumerate(lanes):
                    if (c==a) or (c==b): continue
                    prob = float(p1[i] * p2[j] * p3[k])
                    out_rows.append({
                        "hd": hd, "jcd": jcd, "rno": rno,
                        "a_lane": a, "b_lane": b, "c_lane": c,
                        "a_regno": regno[lanes.index(a)],
                        "b_regno": regno[lanes.index(b)],
                        "c_regno": regno[lanes.index(c)],
                        "proba_3t": prob
                    })

    # 出力
    y, md = args.pred_start[:4], args.pred_start[4:8]
    outdir = Path(args.out_root)/y/md
    ensure_parent(outdir/"_.keep")
    out_csv = outdir/f"trifecta_proba_{args.pred_start}_{args.pred_end}.csv"
    pl.DataFrame(out_rows).write_csv(str(out_csv))
    log(f"[WRITE] {out_csv} rows={len(out_rows)}")

if __name__ == "__main__":
    main()
