#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
integrated_pro.csv → staging CSV 生成スクリプト

入力:
  data/shards/YYYY/MMDD/integrated_pro.csv

出力:
  data/staging/YYYY/MMDD/lanes.csv       … 6艇×レース行（学習素材）
  data/staging/YYYY/MMDD/race_meta.csv   … レース単位メタ（決まり手・天候・風など）
  data/staging/YYYY/MMDD/result_3t.csv   … 3連単正解（label_3t, win_lane）

使い方:
  python scripts/build_staging_from_integrated.py \
      --in_root data/shards \
      --out_root data/staging
"""

from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl

def parse_win_lane(label: str | None):
    if not label:
        return None
    try:
        return int(str(label).split("-")[0])
    except Exception:
        return None

def build_staging(in_csv: Path, out_dir: Path):
    print(f"[LOAD] {in_csv}")
    df = pl.read_csv(in_csv, infer_schema_length=2000)
    out_dir.mkdir(parents=True, exist_ok=True)

    # lanes.csv = そのまま（最低限の整形）
    lanes = df.clone()
    lanes.write_csv(out_dir / "lanes.csv")
    print(f"[WRITE] {out_dir/'lanes.csv'} ({lanes.height} rows)")

    # race_meta.csv = レース単位の決まり手・環境情報
    meta_cols = [
        "hd", "jcd", "rno", "title", "decision",
        "weather_sky", "wind_dir", "wind_speed_m", "wave_height_cm", "label_3t",
    ]
    race_meta = (
        df.select([c for c in meta_cols if c in df.columns])
          .unique(subset=["hd", "jcd", "rno"], keep="first")
          .sort(["hd", "jcd", "rno"])
    )
    race_meta.write_csv(out_dir / "race_meta.csv")
    print(f"[WRITE] {out_dir/'race_meta.csv'} ({race_meta.height} rows)")

    # result_3t.csv = label_3t + 1着コース(win_lane)
    res_cols = ["hd", "jcd", "rno", "label_3t"]
    result = (
        df.select([c for c in res_cols if c in df.columns])
          .unique(subset=["hd", "jcd", "rno"], keep="first")
          .with_columns([
              pl.col("label_3t").map_elements(parse_win_lane, return_dtype=pl.Int64).alias("win_lane")
          ])
          .sort(["hd", "jcd", "rno"])
    )
    result.write_csv(out_dir / "result_3t.csv")
    print(f"[WRITE] {out_dir/'result_3t.csv'} ({result.height} rows)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", default="data/shards")
    ap.add_argument("--out_root", default="data/staging")
    args = ap.parse_args()

    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    csvs = sorted(in_root.glob("**/integrated_pro.csv"))
    if not csvs:
        print("[WARN] No integrated_pro.csv found under", in_root)
        return

    for c in csvs:
        # 例: data/shards/2024/0101/integrated_pro.csv
        parts = list(c.parts)
        try:
            year = [p for p in parts if p.isdigit() and len(p) == 4][0]
            md = [p for p in parts if len(p) == 4 and p.isdigit()][-1]
        except Exception:
            year, md = "0000", "0000"
        out_dir = out_root / year / md
        build_staging(c, out_dir)

if __name__ == "__main__":
    main()
