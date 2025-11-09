#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単勝モデルの学習・推論スクリプト（互換引数対応・全差し替え）

新規引数（現行）:
  --staging_root, --out_root, --model_root, --train_start, --train_end, --test_start, --test_end

互換引数（旧ワークフローからの受け取り用・あってもなくてもOK）:
  --results_root         -> 受理して無視（単勝学習では不要）
  --pred_start           -> test_start にマップ
  --pred_end             -> test_end にマップ
  --model_out            -> モデル保存先を単一ファイルで指定（指定があれば model_root より優先）
  --proba_out_root       -> out_root にマップ

出力:
  - 予測CSV: {out_root}/{YYYY}/{MMDD}/proba_{test_start}_{test_end}.csv
  - モデル:   {model_root}/model.txt  (もしくは --model_out で明示されたパス)
  - メタJSON: {model_root}/meta.json

入力:
  data/staging/YYYY/MMDD/single_train.csv（推奨）
  なければ *single*.csv や *_train.csv をフォールバック探索
  必須列: is_win（1=1着, 0=その他）
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import sys
import glob

import polars as pl

try:
    import pandas as pd
    _HAS_PYARROW = True
except Exception:
    import pandas as pd
    _HAS_PYARROW = False

import lightgbm as lgb


# -------------------- utils --------------------
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def log(msg: str):
    print(msg, flush=True)

def err(msg: str):
    print(msg, file=sys.stderr, flush=True)


# -------------------- config --------------------
@dataclass
class TrainConfig:
    staging_root: str = "data/staging"
    out_root: str = "data/proba/single"
    model_root: str = "data/models/single"
    model_out: str | None = None          # 明示ファイル指定があれば優先
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


# -------------------- input discovery --------------------
def _ymd_paths(root: Path, yyyymmdd_from: str, yyyymmdd_to: str) -> list[Path]:
    def to_ord(y, m, d): return y * 372 + m * 31 + d
    s_y, s_m, s_d = int(yyyymmdd_from[:4]), int(yyyymmdd_from[4:6]), int(yyyymmdd_from[6:8])
    e_y, e_m, e_d = int(yyyymmdd_to[:4]),   int(yyyymmdd_to[4:6]),   int(yyyymmdd_to[6:8])
    s_ord, e_ord = to_ord(s_y, s_m, s_d), to_ord(e_y, e_m, e_d)

    picked = []
    for ydir in sorted(root.glob("[0-9]" * 4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]" * 4)):
            if not md.is_dir(): continue
            m, d = int(md.name[:2]), int(md.name[2:])
            od = to_ord(y, m, d)
            if s_ord <= od <= e_ord:
                picked.append(md)
    return picked


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
        use = None
        for c in cand:
            if c.exists() and c.suffix.lower() == ".csv":
                use = c
                break
        if use:
            files.append(use)
    return files


# -------------------- feature/label --------------------
BASE_ID_COLS = ["hd","jcd","rno","lane","regno"]
PREF_FEATURES = [
    "course_first_rate", "course_3rd_rate", "course_avg_st",
    "tenji_st_sec", "tenji_rank", "st_rank",
    "wind_speed_m", "wave_height_cm",
]

def load_dataset(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    lfs = []
    for p in paths:
        lfs.append(pl.scan_csv(str(p), ignore_errors=True))
    return pl.concat(lfs).collect(streaming=True)

def select_features(df: pl.DataFrame):
    cols = set(df.columns)
    if "is_win" not in cols:
        raise RuntimeError(
            "入力CSVに 'is_win' 列がありません。単勝ラベル(1=1着,0=それ以外)を追加してください。"
        )
    use_feats: list[str] = [c for c in PREF_FEATURES if c in cols]
    # 数値列を自動追加（ID/ラベル以外）
    for c, dt in df.schema.items():
        if c in BASE_ID_COLS or c == "is_win":
            continue
        if "Int" in str(dt) or "Float" in str(dt):
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

    # 現行引数
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--out_root",     default="data/proba/single")
    ap.add_argument("--model_root",   default="data/models/single")
    ap.add_argument("--train_start",  default="20240101")
    ap.add_argument("--train_end",    default="20240131")
    ap.add_argument("--test_start",   default="20240201")
    ap.add_argument("--test_end",     default="20240229")

    # 互換引数（旧ワークフロー対応）
    ap.add_argument("--results_root", default=None, help="互換引数: 受理して無視")
    ap.add_argument("--pred_start",   default=None, help="互換引数: -> test_start にマップ")
    ap.add_argument("--pred_end",     default=None, help="互換引数: -> test_end にマップ")
    ap.add_argument("--model_out",    default=None, help="互換引数: モデル保存ファイルを直指定")
    ap.add_argument("--proba_out_root", default=None, help="互換引数: -> out_root にマップ")

    args = ap.parse_args()

    # 互換引数のマッピング
    test_start = args.pred_start if args.pred_start else args.test_start
    test_end   = args.pred_end   if args.pred_end else args.test_end
    out_root   = args.proba_out_root if args.proba_out_root else args.out_root
    model_out  = args.model_out  # None なら model_root/model.txt に保存

    cfg = TrainConfig(
        staging_root=args.staging_root,
        out_root=out_root,
        model_root=args.model_root,
        model_out=model_out,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=test_start,
        test_end=test_end,
    )

    # 入力収集
    train_files = find_staging_csvs(cfg.staging_root, cfg.train_start, cfg.train_end)
    test_files  = find_staging_csvs(cfg.staging_root, cfg.test_start,  cfg.test_end)

    if not train_files:
        err(f"[FATAL] train CSV が見つかりません: {cfg.staging_root} [{cfg.train_start}..{cfg.train_end}]")
        sys.exit(1)
    if not test_files:
        log(f"[WARN] test CSV が見つかりません: {cfg.staging_root} [{cfg.test_start}..{cfg.test_end}] -> 予測CSVは出力されません")

    log(f"[INFO] train files: {len(train_files)}")
    if test_files:
        log(f"[INFO] test  files: {len(test_files)}")

    df_train = load_dataset(train_files)
    if df_train.height == 0:
        err("[FATAL] 学習データが空です。")
        sys.exit(1)

    train_pd, feat_cols = select_features(df_train)
    log(f"[INFO] features used: {len(feat_cols)} -> {feat_cols[:10]}{'...' if len(feat_cols)>10 else ''}")
    log(f"[INFO] train rows: {len(train_pd)}")

    booster = fit_lgb(train_pd, feat_cols, cfg)

    # モデル保存
    if cfg.model_out:
        model_path = ensure_parent(Path(cfg.model_out))
        meta_path = ensure_parent(Path(cfg.model_out).with_suffix(".json"))
    else:
        model_path = ensure_parent(Path(cfg.model_root) / "model.txt")
        meta_path  = ensure_parent(Path(cfg.model_root) / "meta.json")

    booster.save_model(str(model_path))
    meta = {"config": asdict(cfg), "feature_names": feat_cols, "train_rows": int(len(train_pd))}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[WRITE] {model_path}")
    log(f"[WRITE] {meta_path}")

    # 予測（テスト存在時）
    if test_files:
        df_test = load_dataset(test_files)
        if df_test.height == 0:
            log("[WARN] テストデータが空です。予測はスキップします。")
            return
        keep_cols = BASE_ID_COLS + feat_cols + (["is_win"] if "is_win" in df_test.columns else [])
        df_test2 = df_test.select([c for c in keep_cols if c in df_test.columns]).fill_null(0)
        test_pd = df_test2.to_pandas(use_pyarrow_extension_array=False) if _HAS_PYARROW else df_test2.to_pandas()

        pred = predict_df(booster, test_pd, feat_cols)
        y = cfg.test_start[:4]; md = cfg.test_start[4:8]
        out_dir = Path(cfg.out_root) / y / md
        out_csv = ensure_parent(out_dir / f"proba_{cfg.test_start}_{cfg.test_end}.csv")
        pred.to_csv(out_csv, index=False)
        log(f"[WRITE] {out_csv} rows={len(pred)}")


if __name__ == "__main__":
    main()
