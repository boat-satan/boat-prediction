#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
place2/3（2着・3着）向け学習スクリプト（ラベル自動付与対応・全差し替え）

入力: data/integrated/YYYY/MMDD/integrated_train.csv
      （ラベル列が無ければ public/results の公式結果JSONから付与）
出力: data/models/place23/model.txt, meta.json, feature_importance.csv
"""

from __future__ import annotations
import argparse, json, sys, glob, os
from pathlib import Path
from typing import List, Tuple

import polars as pl
import pandas as pd
import lightgbm as lgb

ID_COLS = ["hd","jcd","rno","lane","regno"]
# 学習に使う候補（存在しなければ自動スキップ）
PREF_FEATS = [
    "st_pred_sec","st_rel_sec","st_rank_in_race","dash_advantage","wall_weak_flag",
    "proba_nige","proba_sashi","proba_makuri","proba_makuri_sashi","proba_win",
    "wind_speed_m","wave_height_cm","is_strong_wind","is_crosswind",
    "dash_attack_flag","course_avg_st","course_first_rate","course_3rd_rate",
    "tenji_st","tenji_sec","tenji_rank","st_rank",
    "motor_rate2","motor_rate3","boat_rate2","boat_rate3",
    "power_lane","power_inner","power_outer","outer_over_inner",
    "kimarite_makuri","kimarite_sashi","kimarite_makuri_sashi","kimarite_nuki",
]
LABEL_CANDIDATES = ["label_place23", "label_top3", "is_place23", "is_top3"]

def log(m: str): print(m, flush=True)
def err(m: str): print(m, file=sys.stderr, flush=True)
def ensure_parent(p: Path): p.parent.mkdir(parents=True, exist_ok=True); return p

def _ymd_paths(root: Path, d1: str, d2: str) -> list[Path]:
    def ord8(s: str) -> int: return int(s[:4])*372 + int(s[4:6])*31 + int(s[6:8])
    s_ord, e_ord = ord8(d1), ord8(d2)
    out = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            m, d = int(md.name[:2]), int(md.name[2:])
            od = y*372 + m*31 + d
            if s_ord <= od <= e_ord:
                out.append(md)
    return out

def find_integrated_csvs(integrated_root: str, d1: str, d2: str) -> list[Path]:
    files = []
    for day in _ymd_paths(Path(integrated_root), d1, d2):
        p = day / "integrated_train.csv"
        if p.exists(): files.append(p)
    return files

def load_train(files: list[Path]) -> pl.DataFrame:
    if not files: return pl.DataFrame()
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in files]
    df = pl.concat(lfs).collect()
    log(f"[INFO] merged rows: {df.height}")
    return df

# ---------- 公式結果JSONからラベル作成 ----------
def _iter_results_jsons(results_root: Path, d1: str, d2: str) -> list[Path]:
    # 日付範囲を走査して日付配下の *.json を拾う（ディープに再帰）
    paths = []
    for day in _ymd_paths(results_root, d1, d2):
        # 例: public/results/YYYY/MMDD/**.json
        paths.extend(day.rglob("*.json"))
    return paths

def build_labels_from_results(results_root: str, d1: str, d2: str) -> pl.DataFrame:
    root = Path(results_root)
    rows = []
    for p in _iter_results_jsons(root, d1, d2):
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("meta", {})
            hd = int(meta.get("date") or meta.get("hd"))
            jcd = int(meta.get("jcd"))
            rno = int(meta.get("rno"))
            for item in data.get("results", []):
                lane = int(item.get("lane"))
                rank_raw = item.get("rank")
                # "1" or "F"/"L"/"妨"などあり得るので数字のみ
                try:
                    rank = int(rank_raw)
                except:
                    continue
                rows.append((hd,jcd,rno,lane,rank))
        except Exception:
            continue
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, schema=["hd","jcd","rno","lane","rank"])

def attach_labels(df: pl.DataFrame, results_root: str, d1: str, d2: str) -> pl.DataFrame:
    # すでにラベルがあれば何もしない
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return df

    log("[INFO] label columns not found; attaching from results json...")
    lab = build_labels_from_results(results_root, d1, d2)
    if lab.height == 0:
        err("[FATAL] 公式結果JSONからラベルを作れませんでした。results_root/配下を確認してください。")
        sys.exit(1)

    # join
    df2 = df.join(lab, on=["hd","jcd","rno","lane"], how="left")
    if "rank" not in df2.columns:
        err("[FATAL] rank付与に失敗しました（キー不一致）。hd/jcd/rno/lane の型と値を確認してください。")
        sys.exit(1)

    df2 = df2.with_columns([
        (pl.col("rank").is_in([2,3])).cast(pl.Int8).alias("is_place23"),
        (pl.col("rank").is_in([1,2,3])).cast(pl.Int8).alias("is_top3"),
    ])
    return df2

def pick_features(df: pl.DataFrame) -> Tuple[pl.DataFrame, list[str], str]:
    # 使える特徴のみ
    feats = [c for c in PREF_FEATS if c in df.columns]
    if not feats:
        err("[FATAL] 有効な特徴量が見つかりません。integrated_train.csv の列を確認してください。")
        sys.exit(1)

    # ラベル選定（優先順位）
    label = None
    for c in ["is_place23","is_top3","label_place23","label_top3"]:
        if c in df.columns:
            label = c
            break
    if label is None:
        err(f"[FATAL] ラベル列が見つかりません（候補: {LABEL_CANDIDATES}）")
        sys.exit(1)

    # ID + label + features
    keep = [c for c in ID_COLS if c in df.columns] + [label] + feats
    out = df.select(keep)
    return out, feats, label

def train_lgb_binary(df: pl.DataFrame, feat_cols: list[str], label: str) -> Tuple[lgb.Booster, list[str], int]:
    pdf = df.to_pandas(use_pyarrow_extension_array=False)
    y = pdf[label].astype(int).values
    X = pdf[feat_cols]

    ds = lgb.Dataset(X, label=y, feature_name=list(feat_cols), free_raw_data=True)
    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=1.0,
        num_threads=0,
        seed=20240301,
        verbose=-1,
        force_col_wise=True,
    )
    booster = lgb.train(
        params, ds, num_boost_round=600,
        valid_sets=[ds], valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return booster, feat_cols, pos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated_root", default="data/integrated")
    ap.add_argument("--results_root",    default="public/results")
    ap.add_argument("--train_start",     required=True)
    ap.add_argument("--train_end",       required=True)
    ap.add_argument("--model_root",      default="data/models/place23")
    args = ap.parse_args()

    files = find_integrated_csvs(args.integrated_root, args.train_start, args.train_end)
    if not files:
        err(f"[FATAL] integrated_train.csv が見つかりません: {args.integrated_root} {args.train_start}..{args.train_end}")
        sys.exit(1)
    log(f"[INFO] train files: {len(files)}")

    df = load_train(files)

    # ラベル無ければ公式結果から付与
    df = attach_labels(df, args.results_root, args.train_start, args.train_end)

    df2, feat_cols, label = pick_features(df)
    log(f"[INFO] target: {label}, features: {len(feat_cols)}")

    booster, feats, pos = train_lgb_binary(df2, feat_cols, label)
    log(f"[INFO] positives: {pos}, negatives: {int(df2.height - pos)}")

    # 保存
    model_root = Path(args.model_root)
    model_path = ensure_parent(model_root / "model.txt")
    meta_path  = ensure_parent(model_root / "meta.json")
    fi_path    = ensure_parent(model_root / "feature_importance.csv")

    booster.save_model(str(model_path))
    log(f"[SAVE] model -> {model_path}")

    meta = {
        "train_range": [args.train_start, args.train_end],
        "rows": int(df2.height),
        "label": label,
        "features": feats,
        "params": {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": 1.0,
            "num_boost_round": 600,
            "seed": 20240301,
        },
        "pos_samples": pos,
        "neg_samples": int(df2.height - pos),
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[SAVE] meta -> {meta_path}")

    fi = pd.DataFrame({
        "feature": feats,
        "gain": booster.feature_importance(importance_type="gain"),
        "split": booster.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    fi.to_csv(fi_path, index=False)
    log(f"[SAVE] feature importance -> {fi_path}")

if __name__ == "__main__":
    main()
