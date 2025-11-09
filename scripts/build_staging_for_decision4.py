#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from dataclasses import dataclass, asdict

import polars as pl
import lightgbm as lgb

# ====== 定数 ======
CLS_ORDER = ["NIGE", "SASHI", "MAKURI", "MAKURI_SASHI"]
ID_COLS = ["hd","jcd","rno","lane","racer_id","racer_name"]

PREF_FEATS = [
    "course_first_rate","course_3rd_rate","course_avg_st",
    "tenji_st","tenji_sec","tenji_rank","st_rank",
    "motor_rate2","motor_rate3","boat_rate2","boat_rate3",
    "wind_speed_m","wave_height_cm",
    "kimarite_makuri","kimarite_sashi","kimarite_makuri_sashi","kimarite_nuki",
    "power_lane","power_inner","power_outer","outer_over_inner",
    "dash_attack_flag","is_strong_wind","is_crosswind",
    # lane はIDと衝突するので後で lane_feat として扱う
    "lane",
]

@dataclass
class Cfg:
    staging_root: str = "data/staging"
    out_root: str     = "data/proba/decision4"
    model_root: str   = "data/models/decision4"
    train_start: str  = "20240101"
    train_end: str    = "20240131"
    test_start: str   = "20240201"
    test_end: str     = "20240229"
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_boost_round: int = 800
    num_threads: int = 0
    seed: int = 20240301

def log(x): print(x, flush=True)
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True); return p

# ====== 入力収集 ======
def _ymd_dirs(root: Path, s: str, e: str) -> list[Path]:
    def ordv(y,m,d): return y*372+m*31+d
    sy,sm,sd = int(s[:4]), int(s[4:6]), int(s[6:8])
    ey,em,ed = int(e[:4]), int(e[4:6]), int(e[6:8])
    so, eo = ordv(sy,sm,sd), ordv(ey,em,ed)
    out = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            m, d = int(md.name[:2]), int(md.name[2:])
            if so <= ordv(y,m,d) <= eo:
                out.append(md)
    return out

def _collect(staging_root: str, start: str, end: str, fname="decision4_train.csv") -> pl.DataFrame:
    root = Path(staging_root)
    lfs = []
    for dd in _ymd_dirs(root, start, end):
        p = dd / fname
        if p.exists():
            lfs.append(pl.scan_csv(str(p), ignore_errors=True))
    if not lfs:
        return pl.DataFrame()
    return pl.concat(lfs).collect(streaming=True)

# ====== 特徴選択 ======
def _select_features(df: pl.DataFrame):
    cols = set(df.columns)
    if "decision4" not in cols:
        raise RuntimeError("入力に decision4 列がありません（NIGE/SASHI/MAKURI/MAKURI_SASHI）。")

    # ラベル -> 0..3
    df = df.with_columns(
        pl.col("decision4").cast(pl.Utf8, strict=False)
    ).with_columns(
        pl.when(pl.col("decision4") == "NIGE").then(0)
         .when(pl.col("decision4") == "SASHI").then(1)
         .when(pl.col("decision4") == "MAKURI").then(2)
         .when(pl.col("decision4") == "MAKURI_SASHI").then(3)
         .otherwise(None)
         .alias("y")
    ).drop_nulls(subset=["y"])

    # 数値のみ拾う
    cand_feats = []
    for c in PREF_FEATS:
        if c in cols:
            dt = df.schema[c]
            if "Int" in str(dt) or "Float" in str(dt):
                cand_feats.append(c)

    # lane はIDと重複するため、特徴としては lane_feat にリネームして選択
    feature_exprs = []
    feature_names = []
    for c in cand_feats:
        if c == "lane":
            feature_exprs.append(pl.col("lane").alias("lane_feat"))
            feature_names.append("lane_feat")
        else:
            feature_exprs.append(pl.col(c).alias(c))
            feature_names.append(c)

    # ID 列（存在するものだけ）
    id_exprs = [pl.col(c).alias(c) for c in ID_COLS if c in cols]

    # 重複回避した select
    df2 = df.select(id_exprs + [pl.col("y")] + feature_exprs)
    return df2, feature_names

# ====== 学習/推論 ======
def _fit(train_df: pl.DataFrame, feats: list[str], cfg: Cfg) -> lgb.Booster:
    X = train_df.select(feats).to_pandas(use_pyarrow_extension_array=False)
    y = train_df["y"].to_pandas(use_pyarrow_extension_array=False).astype(int).values
    ds = lgb.Dataset(X, label=y, feature_name=feats, free_raw_data=True)
    params = dict(
        objective="multiclass", num_class=4, metric="multi_logloss",
        learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth, min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.feature_fraction, bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq, lambda_l2=cfg.lambda_l2,
        num_threads=cfg.num_threads, seed=cfg.seed, verbose=-1,
        force_col_wise=True,
    )
    booster = lgb.train(params, ds, num_boost_round=cfg.num_boost_round,
                        valid_sets=[ds], valid_names=["train"],
                        callbacks=[lgb.log_evaluation(period=200)])
    return booster

def _predict(booster: lgb.Booster, test_df: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    id_df = test_df.select([c for c in ID_COLS if c in test_df.columns])
    X = test_df.select(feats).to_pandas(use_pyarrow_extension_array=False)
    proba = booster.predict(X)  # (n,4)
    pdf = id_df.to_pandas(use_pyarrow_extension_array=False)
    pdf["proba_nige"]          = proba[:,0]
    pdf["proba_sashi"]         = proba[:,1]
    pdf["proba_makuri"]        = proba[:,2]
    pdf["proba_makuri_sashi"]  = proba[:,3]
    return pl.from_pandas(pdf)

# ====== main ======
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--out_root",     default="data/proba/decision4")
    ap.add_argument("--model_root",   default="data/models/decision4")
    ap.add_argument("--train_start",  required=True)
    ap.add_argument("--train_end",    required=True)
    ap.add_argument("--test_start",   required=True)
    ap.add_argument("--test_end",     required=True)
    args = ap.parse_args()

    cfg = Cfg(
        staging_root=args.staging_root, out_root=args.out_root, model_root=args.model_root,
        train_start=args.train_start, train_end=args.train_end,
        test_start=args.test_start,   test_end=args.test_end,
    )

    train_df_raw = _collect(cfg.staging_root, cfg.train_start, cfg.train_end, "decision4_train.csv")
    if train_df_raw.height == 0:
        print(f"[FATAL] 学習データが見つかりません: {cfg.staging_root} [{cfg.train_start}..{cfg.train_end}]", file=sys.stderr)
        sys.exit(1)
    train_df, feats = _select_features(train_df_raw)
    log(f"[INFO] train rows={train_df.height}, feats={len(feats)} -> {feats[:10]}{'...' if len(feats)>10 else ''}")

    booster = _fit(train_df, feats, cfg)

    model_path = ensure_parent(Path(cfg.model_root) / "model.txt")
    booster.save_model(str(model_path))
    meta_path  = ensure_parent(Path(cfg.model_root) / "meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "classes": CLS_ORDER, "features": feats,
                   "train_rows": int(train_df.height)}, f, ensure_ascii=False, indent=2)
    log(f"[WRITE] {model_path}")
    log(f"[WRITE] {meta_path}")

    test_df_raw = _collect(cfg.staging_root, cfg.test_start, cfg.test_end, "decision4_train.csv")
    if test_df_raw.height == 0:
        log("[WARN] テスト期間に decision4_train.csv が見つかりません。推論スキップ。")
        return

    # test も lane を lane_feat に揃えた形で選択
    cols = set(test_df_raw.columns)
    keep_exprs = [pl.col(c).alias(c) for c in ID_COLS if c in cols]
    feat_exprs = []
    for c in feats:
        if c == "lane_feat":
            feat_exprs.append(pl.col("lane").alias("lane_feat"))
        else:
            feat_exprs.append(pl.col(c).alias(c))
    test_df = test_df_raw.select(keep_exprs + feat_exprs)

    pred_df = _predict(booster, test_df, feats)

    for (hd,), sub in pred_df.group_by("hd", maintain_order=True):
        y, md = str(hd)[:4], str(hd)[4:8]
        out_dir = Path(cfg.out_root) / y / md
        out_path = ensure_parent(out_dir / "decision4_proba.csv")
        sub.sort(by=["jcd","rno","lane"]).write_csv(out_path)
        log(f"[WRITE] {out_path} rows={sub.height}")

if __name__ == "__main__":
    main()
