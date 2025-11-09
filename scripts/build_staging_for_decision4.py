#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import Counter
import polars as pl
import lightgbm as lgb

# =========================
# 基本設定
# =========================
CLS_ORDER = ["NIGE", "SASHI", "MAKURI", "MAKURI_SASHI"]
ID_COLS = ["hd","jcd","rno","lane_id","racer_id","racer_name"]

PREF_FEATS = [
    "course_first_rate","course_3rd_rate","course_avg_st",
    "tenji_st","tenji_sec","tenji_rank","st_rank",
    "motor_rate2","motor_rate3","boat_rate2","boat_rate3",
    "wind_speed_m","wave_height_cm",
    "kimarite_makuri","kimarite_sashi","kimarite_makuri_sashi","kimarite_nuki",
    "power_lane","power_inner","power_outer","outer_over_inner",
    "dash_attack_flag","is_strong_wind","is_crosswind",
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


# =========================
# 共通ユーティリティ
# =========================
def log(x): print(x, flush=True)
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True); return p

def _dedup_columns(cols: list[str]) -> list[str]:
    """重複列名に __dupN を付けてユニーク化"""
    cnt = Counter()
    new_cols = []
    for c in cols:
        cnt[c] += 1
        if cnt[c] == 1:
            new_cols.append(c)
        else:
            new_cols.append(f"{c}__dup{cnt[c]}")
    return new_cols

def _ymd_dirs(root: Path, s: str, e: str) -> list[Path]:
    """日付範囲で staging/YYYY/MMDD を返す"""
    def ordv(y,m,d): return y*372+m*31+d
    sy,sm,sd = int(s[:4]), int(s[4:6]), int(s[6:8])
    ey,em,ed = int(e[:4]), int(e[4:6]), int(e[6:8])
    so, eo = ordv(sy,sm,sd), ordv(ey,em,ed)
    out = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            m, d = int(md.name[:2]), int(md.name[2:])
            if so <= ordv(int(ydir.name), m, d) <= eo:
                out.append(md)
    return out


# =========================
# CSV読み込み & 正規化
# =========================
def _read_clean_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(str(path), ignore_errors=True)

    # 1. 重複ヘッダ改名
    if len(set(df.columns)) != len(df.columns):
        df = df.rename(dict(zip(df.columns, _dedup_columns(df.columns))))
    cols = set(df.columns)

    # 2. regno → racer_id
    if "racer_id" not in cols and "regno" in cols:
        df = df.rename({"regno": "racer_id"})
        cols.add("racer_id")

    # 3. lane 系クリーニング（lane_id 一意に）
    lane_like = [c for c in df.columns if c.startswith("lane")]
    if lane_like:
        # lane_id 優先。その他 lane 系は削除。
        keep = "lane_id" if "lane_id" in lane_like else None
        drop = [c for c in lane_like if c != keep]
        if drop:
            df = df.drop(drop)
        if keep is None and "lane" in cols:
            df = df.rename({"lane": "lane_id"})
        cols = set(df.columns)
    # 念のため lane 系重複が残ってたら再削除
    lane_like = [c for c in df.columns if c.startswith("lane") and c != "lane_id"]
    if lane_like:
        df = df.drop(lane_like)

    # 4. hd, jcd は文字列化
    for k in ("hd","jcd"):
        if k in df.columns:
            df = df.with_columns(pl.col(k).cast(pl.Utf8, strict=False))
    return df


def _collect(staging_root: str, start: str, end: str, fname="decision4_train.csv") -> pl.DataFrame:
    """全日分を eagar で読み込み後に結合（安全版）"""
    root = Path(staging_root)
    dfs = []
    for dd in _ymd_dirs(root, start, end):
        p = dd / fname
        if not p.exists(): continue
        try:
            dfs.append(_read_clean_csv(p))
        except Exception as e:
            log(f"[WARN] {p}: {e}")
    if not dfs:
        return pl.DataFrame()
    return pl.concat(dfs, how="diagonal_relaxed")


# =========================
# 特徴量構築
# =========================
def _select_features(df: pl.DataFrame):
    # 重複ヘッダ除去
    if len(set(df.columns)) != len(df.columns):
        df = df.rename(dict(zip(df.columns, _dedup_columns(df.columns))))

    # lane 系を再クリーニング
    bad_lanes = [c for c in df.columns if c.startswith("lane") and c != "lane_id"]
    if bad_lanes:
        df = df.drop(bad_lanes)

    # decision4 の確認
    if "decision4" not in df.columns:
        raise RuntimeError("入力に decision4 列がありません。")

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

    feat_exprs, feat_names = [], []

    # lane_feat
    if "lane_id" in df.columns:
        feat_exprs.append(pl.col("lane_id").cast(pl.Float64).alias("lane_feat"))
    else:
        feat_exprs.append(pl.lit(0.0).alias("lane_feat"))
    feat_names.append("lane_feat")

    for c in PREF_FEATS:
        if c in df.columns:
            dt = df.schema[c]
            if "Int" in str(dt) or "Float" in str(dt):
                feat_exprs.append(pl.col(c).alias(c))
                feat_names.append(c)

    id_exprs = [pl.col(c) for c in ID_COLS if c in df.columns]

    out = df.select(id_exprs + [pl.col("y")] + feat_exprs)
    if len(set(out.columns)) != len(out.columns):
        out = out.rename(dict(zip(out.columns, _dedup_columns(out.columns))))
    return out, feat_names


# =========================
# 学習 / 推論
# =========================
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
    keep_ids = [c for c in ID_COLS if c in test_df.columns]
    out = test_df.select(keep_ids)

    if "lane" in out.columns:
        out = out.drop("lane")
    if "lane_id" in out.columns:
        out = out.with_columns(pl.col("lane_id").alias("lane")).drop("lane_id")

    X = test_df.select(feats).to_pandas(use_pyarrow_extension_array=False)
    proba = booster.predict(X)
    out = out.with_columns([
        pl.lit(proba[:,0]).alias("proba_nige"),
        pl.lit(proba[:,1]).alias("proba_sashi"),
        pl.lit(proba[:,2]).alias("proba_makuri"),
        pl.lit(proba[:,3]).alias("proba_makuri_sashi"),
    ])
    return out


# =========================
# main
# =========================
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
        test_start=args.test_start, test_end=args.test_end,
    )

    train_df_raw = _collect(cfg.staging_root, cfg.train_start, cfg.train_end)
    if train_df_raw.height == 0:
        print(f"[FATAL] 学習データなし: {cfg.train_start}..{cfg.train_end}", file=sys.stderr)
        sys.exit(1)
    train_df, feats = _select_features(train_df_raw)
    log(f"[INFO] train rows={train_df.height}, feats={len(feats)}")

    booster = _fit(train_df, feats, cfg)

    model_path = ensure_parent(Path(cfg.model_root) / "model.txt")
    booster.save_model(str(model_path))
    meta_path = ensure_parent(Path(cfg.model_root) / "meta.json")
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "classes": CLS_ORDER, "features": feats,
                   "train_rows": int(train_df.height)}, f, ensure_ascii=False, indent=2)
    log(f"[WRITE] {model_path}")
    log(f"[WRITE] {meta_path}")

    test_df_raw = _collect(cfg.staging_root, cfg.test_start, cfg.test_end)
    if test_df_raw.height == 0:
        log("[WARN] テスト期間のデータなし。")
        return
    test_df, _ = _select_features(test_df_raw)
    pred_df = _predict(booster, test_df, feats)

    for (hd,), sub in pred_df.group_by("hd", maintain_order=True):
        y, md = str(hd)[:4], str(hd)[4:8]
        out_dir = Path(cfg.out_root) / y / md
        out_path = ensure_parent(out_dir / "decision4_proba.csv")
        sub.sort(by=["jcd","rno","lane"]).write_csv(out_path)
        log(f"[WRITE] {out_path} rows={sub.height}")

if __name__ == "__main__":
    main()
