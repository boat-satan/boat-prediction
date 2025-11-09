#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単勝モデル出力の後処理：
 - レース内正規化（Σp=1）
 - 温度スケーリング（softmax形式, 任意T指定）

入力:
  data/proba/single/YYYY/MMDD/proba_*.csv

出力:
  data/proba/single/YYYY/MMDD/proba_*.csv （同名上書き or _norm.csv）

例:
  python scripts/postprocess_single_proba.py \
      --proba_root data/proba/single \
      --start 20240301 \
      --end 20240331 \
      --temperature 1.0 \
      --inplace true
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import glob


def softmax(x, T=1.0):
    x = np.array(x, dtype=np.float64)
    x = x / T
    x -= np.max(x)
    expx = np.exp(x)
    return expx / expx.sum() if expx.sum() > 0 else np.full_like(expx, 1/len(expx))


def process_one_csv(path: Path, temperature: float, inplace: bool):
    df = pd.read_csv(path)
    if "proba_win" not in df.columns:
        print(f"[WARN] skip {path.name}: no proba_win column")
        return

    # 正規化列を計算
    df["proba_norm"] = np.nan

    grouped = df.groupby(["hd", "jcd", "rno"], group_keys=False)
    out_frames = []
    for (hd, jcd, rno), sub in grouped:
        ps = sub["proba_win"].to_numpy(dtype=float)
        if temperature != 1.0:
            ps_soft = softmax(np.log(ps + 1e-12), T=temperature)
        else:
            ps_soft = ps / ps.sum() if ps.sum() > 0 else np.full_like(ps, 1/len(ps))
        sub = sub.copy()
        sub["proba_norm"] = ps_soft
        out_frames.append(sub)

    df2 = pd.concat(out_frames, ignore_index=True)
    out_path = path if inplace else path.with_name(path.stem + "_norm.csv")
    df2.to_csv(out_path, index=False)
    print(f"[WRITE] {out_path} ({len(df2)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba_root", default="data/proba/single")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--inplace", type=lambda x: x.lower()=="true", default=True)
    args = ap.parse_args()

    y_s, y_e = int(args.start[:4]), int(args.end[:4])
    files = []
    for y in range(y_s, y_e + 1):
        ydir = Path(args.proba_root) / str(y)
        if not ydir.exists(): continue
        for md in sorted(ydir.glob("*")):
            if not md.is_dir(): continue
            for f in glob.glob(str(md / "proba_*.csv")):
                files.append(Path(f))

    if not files:
        print(f"[WARN] no proba csv found in {args.proba_root} [{args.start}..{args.end}]")
        return

    for f in files:
        process_one_csv(f, args.temperature, args.inplace)


if __name__ == "__main__":
    main()
