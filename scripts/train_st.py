#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
from pathlib import Path
import polars as pl
import pandas as pd

# 既存のSTモデルを利用
from src.boatpred.st_model import STConfig, train as st_train

def find_staging_days(staging_root: Path, start: str|None, end: str|None):
    """data/staging/YYYY/MMDD/st_train.csv を期間で抽出"""
    for ydir in sorted(staging_root.glob("[0-9][0-9][0-9][0-9]")):
        for mddir in sorted(ydir.glob("[0-9][0-9][0-9][0-9]")):
            hd = f"{ydir.name}{mddir.name}"
            if start and hd < start: 
                continue
            if end and hd > end: 
                continue
            csv = mddir / "st_train.csv"
            if csv.exists():
                yield hd, csv

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--start", default=None, help="YYYYMMDD（含む）")
    ap.add_argument("--end",   default=None, help="YYYYMMDD（含む）")
    ap.add_argument("--model_out", default="data/models/st_lgbm.txt")
    ap.add_argument("--meta_out",  default="data/models/st_lgbm.meta.json")
    ap.add_argument("--history_out", default="data/history/st_history.parquet")  # 再利用用
    args = ap.parse_args()

    staging_root = Path(args.staging_root)
    days = list(find_staging_days(staging_root, args.start, args.end))
    if not days:
        raise SystemExit(f"[FATAL] no st_train.csv under {staging_root} for range {args.start}..{args.end}")
