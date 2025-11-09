#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単勝モデルの学習・推論スクリプト（全差し替え版）

主な変更点:
- 出力前に親ディレクトリを必ず作成
- to_pandas()でpyarrow未導入による落ちを回避（pyarrow前提 / フォールバックあり）
- ログとモデル保存先も mkdir(parents=True, exist_ok=True)
- Staging の CSV を日付範囲で収集して学習（single_train.csv 優先 / 無ければ *_train.csvを探索）

想定入力:
  data/staging/YYYY/MMDD/single_train.csv
    必須列（最低限）:
      - hd,jcd,rno,lane,regno
      - is_win（単勝ラベル: 1=1着, 0=その他）
    あれば使う代表的特徴:
      - course_first_rate, course_3rd_rate, course_avg_st
      - tenji_st_sec, tenji_rank, st_rank
      - wind_speed_m, wave_height_cm など（存在すれば自動で使用）

出力:
  - 予測CSV: data/proba/single/{YYYY}/{MMDD}/proba_{start}_{end}.csv
  - モデル:   data/models/single/model.txt
  - メタJSON: data/models/single/meta.json
"""

from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import sys
import glob
import math

import polars as pl

# pandas/pyarrow は to_pandas()/LightGBM用
try:
    import pandas as pd
    _HAS_PYARROW = True
except Exception:
    import pandas as pd  # fallback (古いpandasでもOKにする)
    _HAS_PYARROW = False

import lightgbm as lgb


# -------------------- ユーティリティ --------------------
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def log(msg: str):
    print(msg, file=sys.stdout, flush=True)

def err(msg: str):
    print(msg, file=sys.stderr, flush=True)


# -------------------- 設定 --------------------
@dataclass
class TrainConfig:
    staging_root: str = "data/staging"
    out_root: str = "data/proba/single"
    model_root: str = "data/models/single"
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


# -------------------- 入力探索 --------------------
def _ymd_paths(root: Path, yyyymmdd_from: str, yyyymmdd_to: str) -> list[Path]:
    """ staging_root/YYYY/MMDD を日付範囲で探索 """
    y0, m0, d0 = int(yyyymmdd_from[:4]), int(yyyymmdd_from[4:6]), int(yyyymmdd_from[6:8])
    y1, m1, d1 = int(yyyymmdd_to[:4]),   int(yyyymmdd_to[4:6]),   int(yyyymmdd_to[6:8])

    def to_ord(y, m, d):
        # 簡易通し日（厳密な暦は不要）
        return y * 372 + m * 31 + d

    start_ord = to_ord(y0, m0, d0)
    end_ord   = to_ord(y1, m1, d1)

    picked = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            m, d = int(md.name[:2]), int(md.name[2:])
            od = to_ord(y, m, d)
            if start_ord <= od <= end_ord:
                picked.append(md)
    return picked


def find_staging_csvs(staging_root: str, start: str, end: str) -> list[Path]:
    """
    優先: single_train.csv
    次善: *single*.csv
    さらに: *_train.csv（保険）
    """
    root = Path(staging_root)
    day_dirs = _ymd_paths(root, start, end)

    files: list[Path] = []
    for dd in day_dirs:
        cand = [
            dd / "single_train.csv",
            *map(Path, glob.glob(str(dd / "*single*.csv"))),
            *map(Path, glob.glob(str(dd / "*_train.csv"))),
        ]
        used = None
        for c in cand:
            if c.exists() and c.suffix.lower() == ".csv":
                used = c
                break
        if used:
            files.append(used)
    return files


# -------------------- 特徴・ラベル抽出 --------------------
BASE_ID_COLS = ["hd","jcd","rno","lane","regno"]
PREF_FEATURES = [
    "course_first_rate", "course_3rd_rate", "course_avg_st",
    "tenji_st_sec", "tenji_rank", "st_rank",
    "wind_speed_m", "wave_height_cm",
]

def load_dataset(paths: list[Path]) -> pl.DataFrame:
    if not paths:
        return pl.DataFrame()
    # schema_overrides は最新Polarsでは dtypes の代替
    lf_list = []
    for p in paths:
        lf = pl.scan_csv(
            str(p),
            ignore_errors=True,
        )
        lf_list.append(lf)
    return pl.concat(lf_list).collect(streaming=True)


def select_features(df: pl.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    cols = set(df.columns)
    if "is_win" not in cols:
        raise RuntimeError(
            "入力CSVに 'is_win' 列が見つかりません。単勝の教師ラベル(1=1着,0=それ以外)を含めてください。\n"
            "例) single_train.csv に is_win 列を追加してから再実行してください。"
        )

    # 使える特徴 = 事前に推した列 + 追加で取り得る数値列（ID/ラベル除外）
    use_feats: list[str] = []
    for c in PREF_FEATURES:
        if c in cols:
            use_feats.append(c)
    # 数値列を自動追加（IDとラベル以外）
    for c in df.columns:
        if c in BASE_ID_COLS or c == "is_win":
            continue
        dt = df.schema.get(c)
        if dt is None:
            continue
        # 数値っぽいものだけ
        if "Int" in str(dt) or "Float" in str(dt):
            if c not in use_feats:
                use_feats.append(c)

    # 取りすぎると欠損が増えるので過剰に広げない（上限）
    if len(use_feats) > 64:
        use_feats = use_feats[:64]

    # 欠損はとりあえず0埋め（LightGBMは欠損扱いもできるが簡易に）
    dfx = df.select(BASE_ID_COLS + ["is_win"] + use_feats).fill_null(0)

    # pandasへ
    if _HAS_PYARROW:
        pdf = dfx.to_pandas(use_pyarrow_extension_array=False)
    else:
        pdf = dfx.to_pandas()

    # 出力
    feat_cols = use_feats[:]  # 学習に使う列名
    return pdf, feat_cols


# -------------------- 学習・推論 --------------------
def fit_lgb(train_pd: pd.DataFrame, feat_cols: list[str], cfg: TrainConfig) -> lgb.Booster:
    X = train_pd[feat_cols]
    y = train_pd["is_win"].astype(int).values

    ds = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=True)
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.feature_fraction,
        bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq,
        lambda_l2=cfg.lambda_l2,
        num_threads=cfg.num_threads,
        seed=cfg.seed,
        verbose=-1,
        force_col_wise=True,
    )
    booster = lgb.train(
        params,
        ds,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[ds],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    return booster


def predict_df(booster: lgb.Booster, test_pd: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    X = test_pd[feat_cols]
    proba = booster.predict(X, num_iteration=booster.best_iteration)
    out = test_pd[BASE_ID_COLS].copy()
    out["proba_win"] = proba
    return out


# -------------------- メイン --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--out_root",     default="data/proba/single")
    ap.add_argument("--model_root",   default="data/models/single")
    ap.add_argument("--train_start",  default="20240101")
    ap.add_argument("--train_end",    default="20240131")
    ap.add_argument("--test_start",   default="20240201")
    ap.add_argument("--test_end",     default="20240229")
    args = ap.parse_args()

    cfg = TrainConfig(
        staging_root=args.staging_root,
        out_root=args.out_root,
        model_root=args.model_root,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    # 1) 入力収集
    train_files = find_staging_csvs(cfg.staging_root, cfg.train_start, cfg.train_end)
    test_files  = find_staging_csvs(cfg.staging_root, cfg.test_start,  cfg.test_end)

    if not train_files:
        err(f"[FATAL] train CSV が見つかりません: {cfg.staging_root} [{cfg.train_start}..{cfg.train_end}]")
        sys.exit(1)
    if not test_files:
        err(f"[WARN] test CSV が見つかりません: {cfg.staging_root} [{cfg.test_start}..{cfg.test_end}] -> 予測CSVは出力されません")

    log(f"[INFO] train files: {len(train_files)}")
    if test_files:
        log(f"[INFO] test  files: {len(test_files)}")

    df_train = load_dataset(train_files)
    if df_train.height == 0:
        err("[FATAL] 学習データが空です。")
        sys.exit(1)

    # 2) 特徴選択
    train_pd, feat_cols = select_features(df_train)
    log(f"[INFO] features used: {len(feat_cols)} -> {feat_cols[:10]}{'...' if len(feat_cols)>10 else ''}")
    log(f"[INFO] train rows: {len(train_pd)}")

    # 3) 学習
    booster = fit_lgb(train_pd, feat_cols, cfg)

    # 4) モデル保存
    model_dir = ensure_parent(Path(cfg.model_root) / "model.txt")
    booster.save_model(str(model_dir))
    meta_path = ensure_parent(Path(cfg.model_root) / "meta.json")
    meta = {
        "config": asdict(cfg),
        "feature_names": feat_cols,
        "train_rows": int(len(train_pd)),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[WRITE] {model_dir}")
    log(f"[WRITE] {meta_path}")

    # 5) 予測（テスト存在時）
    if test_files:
        df_test = load_dataset(test_files)
        if df_test.height == 0:
            err("[WARN] テストデータが空です。予測はスキップします。")
            return
        # テストにも 'is_win' があっても学習に使わない。あるときは評価側で使用。
        # 欠損0埋めは学習側と合わせる
        keep_cols = BASE_ID_COLS + feat_cols + (["is_win"] if "is_win" in df_test.columns else [])
        df_test2 = df_test.select([c for c in keep_cols if c in df_test.columns]).fill_null(0)

        if _HAS_PYARROW:
            test_pd = df_test2.to_pandas(use_pyarrow_extension_array=False)
        else:
            test_pd = df_test2.to_pandas()

        pred = predict_df(booster, test_pd, feat_cols)

        # 出力先を確定（test_startベースで /YYYY/MMDD/ を切る）
        y = cfg.test_start[:4]
        md = cfg.test_start[4:8]
        out_dir = Path(cfg.out_root) / y / md
        out_csv = ensure_parent(out_dir / f"proba_{cfg.test_start}_{cfg.test_end}.csv")
        pred.to_csv(out_csv, index=False)
        log(f"[WRITE] {out_csv} rows={len(pred)}")


if __name__ == "__main__":
    main()
