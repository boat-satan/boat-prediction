#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単勝モデル 学習・推論（is_win を results から自動補完対応版）

入力:
  data/staging/YYYY/MMDD/single_train.csv（推奨）
  ※ is_win が無い場合は public/results から 1着 lane を読み取り補完

出力:
  - 予測CSV: {out_root}/{YYYY}/{MMDD}/proba_{test_start}_{test_end}.csv
  - モデル:   {model_root}/model.txt  (or --model_out)
  - メタJSON: {model_root}/meta.json
"""

from __future__ import annotations
import argparse, json, sys, glob
from dataclasses import dataclass, asdict
from pathlib import Path

import polars as pl
import lightgbm as lgb

try:
    import pandas as pd
    _HAS_PYARROW = True
except Exception:
    import pandas as pd
    _HAS_PYARROW = False


# -------------------- utils --------------------
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def log(msg: str): print(msg, flush=True)
def err(msg: str): print(msg, file=sys.stderr, flush=True)


# -------------------- config --------------------
@dataclass
class TrainConfig:
    staging_root: str = "data/staging"
    results_root: str = "public/results"   # ← 追加: is_win 補完に使用
    out_root: str = "data/proba/single"
    model_root: str = "data/models/single"
    model_out: str | None = None
    train_start: str = "20240101"
    train_end: str   = "20240131"
    test_start: str  = "20240201"
    test_end: str    = "20240229"
    # LightGBM
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_boost_round: int = 500
    num_threads: int = 0
    seed: int = 20240301


# -------------------- date helpers --------------------
def _ymd_paths(root: Path, ymd_from: str, ymd_to: str) -> list[Path]:
    def ord3(y,m,d): return y*372 + m*31 + d
    sy,sm,sd = int(ymd_from[:4]), int(ymd_from[4:6]), int(ymd_from[6:8])
    ey,em,ed = int(ymd_to[:4]),   int(ymd_to[4:6]),   int(ymd_to[6:8])
    so, eo = ord3(sy,sm,sd), ord3(ey,em,ed)
    out = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            m, d = int(md.name[:2]), int(md.name[2:])
            if so <= ord3(y,m,d) <= eo:
                out.append(md)
    return out


# -------------------- input discovery --------------------
def find_staging_csvs(staging_root: str, start: str, end: str) -> list[Path]:
    root = Path(staging_root)
    day_dirs = _ymd_paths(root, start, end)
    files: list[Path] = []
    for dd in day_dirs:
        cand = [
            dd / "single_train.csv",
            *map(Path, glob.glob(str(dd / "*single*.csv"))),
            *map(Path, glob.glob(str(dd / "*_train.csv"))),
        ]
        for c in cand:
            if c.exists() and c.suffix.lower() == ".csv":
                files.append(c)
                break
    return files


# -------------------- is_win 補完 --------------------
def _zero2(s: str|int) -> str:
    try: return f"{int(s):02d}"
    except Exception: return str(s).zfill(2)

def _read_winners_from_results(results_root: str, start: str, end: str) -> pl.DataFrame:
    """results JSON から (hd,jcd,rno,lane,is_win=1) を作る"""
    root = Path(results_root)
    day_dirs = _ymd_paths(root, start, end)
    rows = []
    for md in day_dirs:
        # public/results/YYYY/MMDD/**/{rno}R.json
        for p in md.glob("*/*/*R.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    js = json.load(f)
                meta = js.get("meta", {})
                hd  = str(meta.get("date"))
                jcd = _zero2(meta.get("jcd"))
                rno = int(meta.get("rno"))
                # winner lane: rank "1"
                win_lane = None
                for r in js.get("results", []):
                    if str(r.get("rank")) == "1":
                        win_lane = int(r.get("lane"))
                        break
                if win_lane is None:
                    continue
                rows.append({"hd": hd, "jcd": jcd, "rno": rno, "lane": win_lane, "is_win": 1})
            except Exception:
                continue
    if not rows:
        return pl.DataFrame({"hd": [], "jcd": [], "rno": [], "lane": [], "is_win": []}).with_columns([
            pl.col("rno").cast(pl.Int64, strict=False),
            pl.col("lane").cast(pl.Int64, strict=False),
            pl.col("is_win").cast(pl.Int64, strict=False),
        ])
    df = pl.DataFrame(rows).with_columns([
        pl.col("hd").cast(pl.Utf8),
        pl.col("jcd").cast(pl.Utf8),
        pl.col("rno").cast(pl.Int64),
        pl.col("lane").cast(pl.Int64),
        pl.col("is_win").cast(pl.Int64),
    ])
    return df

def attach_is_win(df: pl.DataFrame, results_root: str, start: str, end: str) -> pl.DataFrame:
    """df に is_win が無ければ results から補完"""
    if "is_win" in df.columns:
        return df
    if df.height == 0:
        return df
    # キー整形（桁揃え）
    df2 = df.with_columns([
        pl.col("hd").cast(pl.Utf8),
        pl.col("jcd").cast(pl.Utf8).str.zfill(2),
        pl.col("rno").cast(pl.Int64),
        pl.col("lane").cast(pl.Int64),
    ])
    winners = _read_winners_from_results(results_root, start, end)
    if winners.height == 0:
        log("[WARN] results から winner を取得できませんでした。is_win 補完はスキップします。")
        return df2.with_columns(pl.lit(0).alias("is_win"))
    out = df2.join(winners, on=["hd","jcd","rno","lane"], how="left")
    out = out.with_columns(pl.col("is_win").fill_null(0).cast(pl.Int64))
    return out


# -------------------- features --------------------
BASE_ID_COLS = ["hd","jcd","rno","lane","regno"]
PREF_FEATURES = [
    "course_first_rate", "course_3rd_rate", "course_avg_st",
    "tenji_st_sec", "tenji_rank", "st_rank",
    "wind_speed_m", "wave_height_cm",
]

def load_dataset(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in paths]
    return pl.concat(lfs).collect(streaming=True)

def select_features(df: pl.DataFrame):
    cols = set(df.columns)
    if "is_win" not in cols:
        raise RuntimeError("is_win が見つかりません（補完に失敗）。")
    use_feats: list[str] = [c for c in PREF_FEATURES if c in cols]
    # 数値カラムを追加（ID/ラベル以外）
    for c, dt in df.schema.items():
        if c in BASE_ID_COLS or c == "is_win":
            continue
        sdt = str(dt)
        if ("Int" in sdt) or ("Float" in sdt):
            if c not in use_feats:
                use_feats.append(c)
    if len(use_feats) > 64:
        use_feats = use_feats[:64]

    dfx = df.select(BASE_ID_COLS + ["is_win"] + use_feats).fill_null(0)
    pdf = dfx.to_pandas(use_pyarrow_extension_array=False) if _HAS_PYARROW else dfx.to_pandas()
    return pdf, use_feats


# -------------------- train/predict --------------------
def fit_lgb(train_pd, feat_cols, cfg: TrainConfig) -> lgb.Booster:
    X = train_pd[feat_cols]
    y = train_pd["is_win"].astype(int).values
    dset = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=True)
    params = dict(
        objective="binary", metric="auc",
        learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth, min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.feature_fraction, bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq, lambda_l2=cfg.lambda_l2,
        num_threads=cfg.num_threads, seed=cfg.seed, verbose=-1,
        force_col_wise=True,
    )
    booster = lgb.train(params, dset, num_boost_round=cfg.num_boost_round,
                        valid_sets=[dset], valid_names=["train"],
                        callbacks=[lgb.log_evaluation(period=200)])
    return booster

def predict_df(booster: lgb.Booster, test_pd, feat_cols):
    X = test_pd[feat_cols]
    proba = booster.predict(X, num_iteration=booster.best_iteration)
    out = test_pd[BASE_ID_COLS].copy()
    out["proba_win"] = proba
    return out


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    # 現行
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--results_root", default="public/results")  # ← 追加
    ap.add_argument("--out_root",     default="data/proba/single")
    ap.add_argument("--model_root",   default="data/models/single")
    ap.add_argument("--train_start",  default="20240101")
    ap.add_argument("--train_end",    default="20240131")
    ap.add_argument("--test_start",   default="20240201")
    ap.add_argument("--test_end",     default="20240229")
    # 互換
    ap.add_argument("--results_root_unused", default=None, help="互換引数: 受理して無視（非推奨）")
    ap.add_argument("--pred_start",   default=None, help="互換: -> test_start")
    ap.add_argument("--pred_end",     default=None, help="互換: -> test_end")
    ap.add_argument("--model_out",    default=None, help="単一ファイルで保存したい場合")
    ap.add_argument("--proba_out_root", default=None, help="互換: -> out_root")

    args = ap.parse_args()

    test_start = args.pred_start if args.pred_start else args.test_start
    test_end   = args.pred_end   if args.pred_end   else args.test_end
    out_root   = args.proba_out_root if args.proba_out_root else args.out_root
    model_out  = args.model_out

    cfg = TrainConfig(
        staging_root=args.staging_root,
        results_root=args.results_root,
        out_root=out_root,
        model_root=args.model_root,
        model_out=model_out,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=test_start,
        test_end=test_end,
    )

    # 収集
    train_files = find_staging_csvs(cfg.staging_root, cfg.train_start, cfg.train_end)
    test_files  = find_staging_csvs(cfg.staging_root, cfg.test_start,  cfg.test_end)
    if not train_files:
        err(f"[FATAL] train CSV が見つかりません: {cfg.staging_root} [{cfg.train_start}..{cfg.train_end}]")
        sys.exit(1)
    log(f"[INFO] train files: {len(train_files)}")
    if test_files:
        log(f"[INFO] test  files: {len(test_files)}")

    # 学習テーブル読み込み＋is_win 補完
    df_train = load_dataset(train_files)
    if df_train.height == 0:
        err("[FATAL] 学習データが空です。"); sys.exit(1)
    if "is_win" not in df_train.columns:
        df_train = attach_is_win(df_train, cfg.results_root, cfg.train_start, cfg.train_end)

    train_pd, feat_cols = select_features(df_train)
    log(f"[INFO] features used: {len(feat_cols)} -> {feat_cols[:10]}{'...' if len(feat_cols)>10 else ''}")
    log(f"[INFO] train rows: {len(train_pd)}")

    booster = fit_lgb(train_pd, feat_cols, cfg)

    # モデル保存
    if cfg.model_out:
        model_path = ensure_parent(Path(cfg.model_out))
        meta_path  = ensure_parent(Path(cfg.model_out).with_suffix(".json"))
    else:
        model_path = ensure_parent(Path(cfg.model_root) / "model.txt")
        meta_path  = ensure_parent(Path(cfg.model_root) / "meta.json")
    booster.save_model(str(model_path))
    meta = {"config": asdict(cfg), "feature_names": feat_cols, "train_rows": int(len(train_pd))}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[WRITE] {model_path}")
    log(f"[WRITE] {meta_path}")

    # 予測
    if test_files:
        df_test = load_dataset(test_files)
        if df_test.height == 0:
            log("[WARN] テストデータが空です。予測スキップ"); return
        if "is_win" not in df_test.columns:
            df_test = attach_is_win(df_test, cfg.results_root, cfg.test_start, cfg.test_end)

        keep = BASE_ID_COLS + feat_cols + (["is_win"] if "is_win" in df_test.columns else [])
        df_test2 = df_test.select([c for c in keep if c in df_test.columns]).fill_null(0)
        test_pd = df_test2.to_pandas(use_pyarrow_extension_array=False) if _HAS_PYARROW else df_test2.to_pandas()

        pred = predict_df(booster, test_pd, feat_cols)
        y, md = cfg.test_start[:4], cfg.test_start[4:8]
        out_csv = ensure_parent(Path(cfg.out_root) / y / md / f"proba_{cfg.test_start}_{cfg.test_end}.csv")
        pred.to_csv(out_csv, index=False)
        log(f"[WRITE] {out_csv} rows={len(pred)}")


if __name__ == "__main__":
    main()
