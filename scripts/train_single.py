#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単勝モデル 学習・推論（is_winをresultsから自動補完対応）

入力:
  data/staging/YYYY/MMDD/single_train.csv（推奨）
  is_winが無ければ public/results から勝ち艇を自動補完

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


# ==================== 基本ユーティリティ ====================
def log(msg: str): print(msg, flush=True)
def err(msg: str): print(msg, file=sys.stderr, flush=True)
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ==================== 設定 ====================
@dataclass
class TrainConfig:
    staging_root: str = "data/staging"
    results_root: str = "public/results"
    out_root: str = "data/proba/single"
    model_root: str = "data/models/single"
    model_out: str | None = None
    train_start: str = "20240101"
    train_end: str   = "20240131"
    test_start: str  = "20240201"
    test_end: str    = "20240229"
    # LightGBM設定
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


# ==================== 日付フォルダ列挙 ====================
def _ymd_paths(root: Path, ymd_from: str, ymd_to: str) -> list[Path]:
    def to_ord(y,m,d): return y*372 + m*31 + d
    fy,fm,fd = int(ymd_from[:4]), int(ymd_from[4:6]), int(ymd_from[6:8])
    ty,tm,td = int(ymd_to[:4]),   int(ymd_to[4:6]),   int(ymd_to[6:8])
    f_ord, t_ord = to_ord(fy,fm,fd), to_ord(ty,tm,td)
    out = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            m,d = int(md.name[:2]), int(md.name[2:])
            if f_ord <= to_ord(y,m,d) <= t_ord:
                out.append(md)
    return out


# ==================== stagingファイル探索 ====================
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


# ==================== results→勝ち艇抽出 ====================
def _rank_is_1(x: str) -> bool:
    s = str(x).strip()
    return s in ("1", "１", "1着", "１着")

def _iter_result_jsons(root: Path, ymd_from: str, ymd_to: str):
    def to_ord(y,m,d): return y*372 + m*31 + d
    fy,fm,fd = int(ymd_from[:4]), int(ymd_from[4:6]), int(ymd_from[6:8])
    ty,tm,td = int(ymd_to[:4]),   int(ymd_to[4:6]),   int(ymd_to[6:8])
    f_ord, t_ord = to_ord(fy,fm,fd), to_ord(ty,tm,td)
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            m,d = int(md.name[:2]), int(md.name[2:])
            if not (f_ord <= to_ord(y,m,d) <= t_ord):
                continue
            for j in md.rglob("*.json"):
                yield j

def build_winners_map(results_root: str, start: str, end: str) -> dict[tuple[str,str,int], int]:
    """(hd,jcd,rno)→win_lane マップ"""
    root = Path(results_root)
    winners: dict[tuple[str,str,int], int] = {}
    for p in _iter_result_jsons(root, start, end):
        try:
            with p.open("r", encoding="utf-8") as f:
                js = json.load(f)
            meta = js.get("meta", {})
            hd  = str(meta.get("date") or "")
            jcd = str(meta.get("jcd") or "").zfill(2)
            rno = int(meta.get("rno") or 0)
            if len(hd) != 8 or rno <= 0:
                continue
            win_lane = None
            for r in js.get("results", []):
                if _rank_is_1(r.get("rank")):
                    try:
                        win_lane = int(r.get("lane"))
                    except Exception:
                        win_lane = None
                    break
            if win_lane is not None:
                winners[(hd, jcd, rno)] = win_lane
        except Exception:
            continue
    return winners

def attach_is_win(df: pl.DataFrame, winners_map: dict[tuple[str,str,int], int]) -> pl.DataFrame:
    if "is_win" in df.columns:
        return df
    if not winners_map:
        return df.with_columns(pl.lit(0).alias("is_win"))
    keys, vals = zip(*winners_map.items())
    wdf = pl.DataFrame({
        "hd": [k[0] for k in keys],
        "jcd": [k[1] for k in keys],
        "rno": [k[2] for k in keys],
        "win_lane": vals
    })
    df2 = df.with_columns([
        pl.col("hd").cast(pl.Utf8),
        pl.col("jcd").cast(pl.Utf8).str.zfill(2),
        pl.col("rno").cast(pl.Int64),
        pl.col("lane").cast(pl.Int64),
    ])
    out = df2.join(wdf, on=["hd","jcd","rno"], how="left")
    out = out.with_columns([
        pl.when(pl.col("lane") == pl.col("win_lane"))
          .then(pl.lit(1)).otherwise(pl.lit(0)).alias("is_win")
    ]).drop("win_lane")
    return out


# ==================== 特徴量選定 ====================
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
        raise RuntimeError("is_win が見つかりません。")
    use_feats = [c for c in PREF_FEATURES if c in cols]
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


# ==================== 学習/予測 ====================
def fit_lgb(train_pd, feat_cols, cfg: TrainConfig) -> lgb.Booster:
    X = train_pd[feat_cols]
    y = train_pd["is_win"].astype(int).values
    dset = lgb.Dataset(X, label=y, feature_name=feat_cols)
    params = dict(
        objective="binary", metric="auc",
        learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth, min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.feature_fraction, bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq, lambda_l2=cfg.lambda_l2,
        num_threads=cfg.num_threads, seed=cfg.seed, verbose=-1,
        force_col_wise=True,
    )
    booster = lgb.train(params, dset, num_boost_round=cfg.num_boost_round)
    return booster

def predict_df(booster: lgb.Booster, test_pd, feat_cols):
    X = test_pd[feat_cols]
    proba = booster.predict(X, num_iteration=booster.best_iteration)
    out = test_pd[BASE_ID_COLS].copy()
    out["proba_win"] = proba
    return out


# ==================== メイン ====================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--out_root",     default="data/proba/single")
    ap.add_argument("--model_root",   default="data/models/single")
    ap.add_argument("--train_start",  default="20240101")
    ap.add_argument("--train_end",    default="20240131")
    ap.add_argument("--test_start",   default="20240201")
    ap.add_argument("--test_end",     default="20240229")
    ap.add_argument("--model_out",    default=None)
    ap.add_argument("--proba_out_root", default=None)
    args = ap.parse_args()

    out_root = args.proba_out_root or args.out_root
    cfg = TrainConfig(
        staging_root=args.staging_root,
        results_root=args.results_root,
        out_root=out_root,
        model_root=args.model_root,
        model_out=args.model_out,
        train_start=args.train_start,
        train_end=args.train_end,
        test_start=args.test_start,
        test_end=args.test_end,
    )

    # ===== データロード =====
    train_files = find_staging_csvs(cfg.staging_root, cfg.train_start, cfg.train_end)
    if not train_files:
        err(f"[FATAL] train CSV not found: {cfg.staging_root}")
        sys.exit(1)
    log(f"[INFO] train files: {len(train_files)}")
    test_files = find_staging_csvs(cfg.staging_root, cfg.test_start, cfg.test_end)
    log(f"[INFO] test files: {len(test_files)}")

    df_train = load_dataset(train_files)
    if df_train.height == 0:
        err("[FATAL] 学習データ空。")
        sys.exit(1)

    # is_win補完
    winners_train = build_winners_map(cfg.results_root, cfg.train_start, cfg.train_end)
    df_train = attach_is_win(df_train, winners_train)
    log(f"[INFO] winners(train): {len(winners_train)} races")

    train_pd, feat_cols = select_features(df_train)
    booster = fit_lgb(train_pd, feat_cols, cfg)

    # モデル保存
    model_path = Path(cfg.model_out) if cfg.model_out else Path(cfg.model_root) / "model.txt"
    ensure_parent(model_path)
    booster.save_model(str(model_path))
    meta_path = model_path.with_suffix(".json")
    meta = {"config": asdict(cfg), "feature_names": feat_cols, "train_rows": len(train_pd)}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[WRITE] {model_path}")
    log(f"[WRITE] {meta_path}")

    # ===== テスト予測 =====
    if test_files:
        df_test = load_dataset(test_files)
        winners_test = build_winners_map(cfg.results_root, cfg.test_start, cfg.test_end)
        df_test = attach_is_win(df_test, winners_test)
        test_pd, _ = select_features(df_test)
        pred = predict_df(booster, test_pd, feat_cols)
        y, md = cfg.test_start[:4], cfg.test_start[4:8]
        out_csv = ensure_parent(Path(cfg.out_root) / y / md / f"proba_{cfg.test_start}_{cfg.test_end}.csv")
        pred.to_csv(out_csv, index=False)
        log(f"[WRITE] {out_csv} rows={len(pred)}")


if __name__ == "__main__":
    main()
