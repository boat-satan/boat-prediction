#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, sys, json, glob
from pathlib import Path
from typing import List, Sequence

import polars as pl
import pandas as pd
import lightgbm as lgb

ID_COLS = ["hd","jcd","rno","lane","regno"]
RACE_KEY = ["hd","jcd","rno"]
TARGET2 = "is_second"
TARGET3 = "is_third"

NUM_FEAT_BLACKLIST = set(ID_COLS + ["rank", TARGET2, TARGET3, "first_lane", "second_lane"])
STR_FEAT_BLACKLIST = set(["racer_name","decision4"])  # あれば除外

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

def find_csvs(staging_root: str, start: str, end: str, name="integrated_train.csv") -> list[Path]:
    files = []
    for dd in _ymd_dirs(Path(staging_root), start, end):
        p = dd / name
        if p.exists(): files.append(p)
        else:
            # フォールバック
            for a in glob.glob(str(dd/"*integrated*train*.csv")):
                files.append(Path(a)); break
    return files

def load_train(paths: list[Path]) -> pl.DataFrame:
    if not paths: return pl.DataFrame()
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in paths]
    df = pl.concat(lfs).collect()
    # rank 必須
    if "rank" not in df.columns:
        raise RuntimeError("integrated_train.csv に rank(1..6) 列が必要です。作成側で付与してください。")
    # 型と欠損軽整形
    for c in ("hd","jcd"):
        if c in df.columns: df = df.with_columns(pl.col(c).cast(pl.Utf8))
    for c in ("rno","lane"):
        if c in df.columns: df = df.with_columns(pl.col(c).cast(pl.Int64))
    df = df.filter(pl.col("rank").is_in([1,2,3,4,5,6]))
    return df

def build_place2_table(df: pl.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    # 各レース winner を特定し、winner 以外を候補、2着=1 ラベル
    win = df.filter(pl.col("rank")==1).select(RACE_KEY+["lane"]).rename({"lane":"first_lane"})
    df2 = df.join(win, on=RACE_KEY, how="inner")
    df2 = df2.filter(pl.col("lane") != pl.col("first_lane"))
    df2 = df2.with_columns((pl.col("rank")==2).cast(pl.Int64).alias(TARGET2))

    # 特徴抽出（数値列を自動選別＋ first_lane を特徴に含める）
    num_cols = [c for c,dt in df2.schema.items()
                if (c not in NUM_FEAT_BLACKLIST) and (c not in STR_FEAT_BLACKLIST) and ("Utf8" not in str(dt))]
    if "first_lane" not in num_cols: num_cols.append("first_lane")
    X = df2.select(num_cols).to_pandas(use_pyarrow_extension_array=False)
    y = df2.select(TARGET2).to_pandas()[TARGET2]
    return X, y, num_cols

def build_place3_table(df: pl.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    # winner/second を特定し、2者以外を候補、3着=1 ラベル
    win = df.filter(pl.col("rank")==1).select(RACE_KEY+["lane"]).rename({"lane":"first_lane"})
    sec = df.filter(pl.col("rank")==2).select(RACE_KEY+["lane"]).rename({"lane":"second_lane"})
    df3 = df.join(win, on=RACE_KEY, how="inner").join(sec, on=RACE_KEY, how="inner")
    df3 = df3.filter((pl.col("lane") != pl.col("first_lane")) & (pl.col("lane") != pl.col("second_lane")))
    df3 = df3.with_columns((pl.col("rank")==3).cast(pl.Int64).alias(TARGET3))

    num_cols = [c for c,dt in df3.schema.items()
                if (c not in NUM_FEAT_BLACKLIST) and (c not in STR_FEAT_BLACKLIST) and ("Utf8" not in str(dt))]
    for need in ("first_lane","second_lane"):
        if need not in num_cols: num_cols.append(need)
    X = df3.select(num_cols).to_pandas(use_pyarrow_extension_array=False)
    y = df3.select(TARGET3).to_pandas()[TARGET3]
    return X, y, num_cols

def train_lgbm_bin(X: pd.DataFrame, y: pd.Series, seed=20240301) -> lgb.Booster:
    params = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, num_leaves=63, max_depth=-1, min_data_in_leaf=50,
        feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0,
        num_threads=0, seed=seed, verbose=-1, force_col_wise=True
    )
    dtrain = lgb.Dataset(X, label=y, feature_name=list(X.columns), free_raw_data=True)
    booster = lgb.train(params, dtrain, num_boost_round=600,
                        valid_sets=[dtrain], valid_names=["train"],
                        callbacks=[lgb.log_evaluation(period=200)])
    return booster

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end", required=True)
    ap.add_argument("--model_root", default="data/models/trifecta")
    args = ap.parse_args()

    files = find_csvs(args.staging_root, args.train_start, args.train_end, "integrated_train.csv")
    if not files: 
        err(f"[FATAL] integrated_train.csv が見つかりません: {args.staging_root} {args.train_start}..{args.train_end}")
        sys.exit(1)
    log(f"[INFO] train files: {len(files)}")
    df = load_train(files)

    # place2
    X2, y2, feats2 = build_place2_table(df)
    log(f"[INFO] place2 rows={len(X2)}, feats={len(feats2)}")
    b2 = train_lgbm_bin(X2, y2)
    # place3
    X3, y3, feats3 = build_place3_table(df)
    log(f"[INFO] place3 rows={len(X3)}, feats={len(feats3)}")
    b3 = train_lgbm_bin(X3, y3)

    # 保存
    root = Path(args.model_root); root.mkdir(parents=True, exist_ok=True)
    b2.save_model(str(root/"place2.txt")); b3.save_model(str(root/"place3.txt"))
    meta = {
        "train_range":[args.train_start,args.train_end],
        "place2":{"rows":int(len(X2)),"features":feats2},
        "place3":{"rows":int(len(X3)),"features":feats3},
        "params":{"lr":0.05,"num_leaves":63,"num_boost_round":600,"seed":20240301}
    }
    with open(root/"meta.json","w",encoding="utf-8") as f: json.dump(meta,f,ensure_ascii=False,indent=2)
    log(f"[SAVE] models -> {root}")

if __name__ == "__main__":
    main()
