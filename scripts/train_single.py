#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単勝モデル 学習・推論 (フル互換 & 全差し替え版)

- 互換引数を広く受け付ける（--start/--end, --pred_start/--pred_end, --proba_root など）
- is_win を results JSON から自動付与（既存に is_win があれば尊重）
- staging CSV を日付範囲で自動探索
- LightGBM で学習し、テスト範囲があれば確率CSV出力

入出力:
  入力: data/staging/YYYY/MMDD/*single*.csv（推奨: single_train.csv）
  ラベル: public/results/YYYY/MMDD/{jcd}/{rno}R.json から 1着 lane を取得
  出力:
    - モデル: data/models/single/model.txt（または --model_out）
    - メタ:   data/models/single/meta.json
    - 予測:   data/proba/single/YYYY/MMDD/proba_{test_start}_{test_end}.csv
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import glob
from typing import Dict, Tuple, List, Optional

import polars as pl

try:
    import pandas as pd
    _HAS_PYARROW = True
except Exception:
    import pandas as pd
    _HAS_PYARROW = False

import lightgbm as lgb


# -------------------- utils --------------------
def log(msg: str): print(msg, flush=True)
def err(msg: str): print(msg, file=sys.stderr, flush=True)
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def to_ord(y: int, m: int, d: int) -> int:
    # 月31日固定の簡易ordinal（範囲フィルタ用。順序性だけ満たせばOK）
    return y * 372 + m * 31 + d

def ymd_to_ord(yyyymmdd: str) -> int:
    y, m, d = int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])
    return to_ord(y, m, d)

def parse_year_md_from_dir(day_dir: Path) -> Tuple[int, int, int]:
    # day_dir = .../YYYY/MMDD
    y = int(day_dir.parent.name)
    md = day_dir.name
    m, d = int(md[:2]), int(md[2:])
    return y, m, d


# -------------------- config --------------------
@dataclass
class TrainConfig:
    staging_root: str = "data/staging"
    results_root: Optional[str] = "public/results"
    out_root: str = "data/proba/single"
    model_root: str = "data/models/single"
    model_out: Optional[str] = None
    train_start: str = "20240101"
    train_end: str   = "20240131"
    test_start: str  = "20240201"
    test_end: str    = "20240229"
    # モデル設定
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_boost_round: int = 400
    num_threads: int = 0
    seed: int = 20240301
    # オプション
    normalize: bool = True
    temperature: float = 1.0


# -------------------- file discovery --------------------
def _list_day_dirs(root: Path) -> List[Path]:
    # root/YYYY/MMDD
    out = []
    for ydir in sorted(root.glob("[0-9]" * 4)):
        if not ydir.is_dir(): continue
        for md in sorted(ydir.glob("[0-9]" * 4)):
            if md.is_dir():
                out.append(md)
    return out

def pick_day_dirs_by_range(root: Path, start: str, end: str) -> List[Path]:
    if not root.exists(): return []
    s_ord, e_ord = ymd_to_ord(start), ymd_to_ord(end)
    picked = []
    for dd in _list_day_dirs(root):
        y, m, d = parse_year_md_from_dir(dd)
        if s_ord <= to_ord(y, m, d) <= e_ord:
            picked.append(dd)
    return picked

def find_staging_csvs(staging_root: str, start: str, end: str) -> List[Path]:
    root = Path(staging_root)
    day_dirs = pick_day_dirs_by_range(root, start, end)
    files: List[Path] = []
    for dd in day_dirs:
        # 優先順: single_train.csv -> *single*.csv -> *_train.csv
        cand = [
            dd / "single_train.csv",
            *map(Path, glob.glob(str(dd / "*single*.csv"))),
            *map(Path, glob.glob(str(dd / "*_train.csv"))),
        ]
        use = None
        for c in cand:
            if c.exists() and c.suffix.lower() == ".csv":
                use = c; break
        if use: files.append(use)
    return files


# -------------------- label (is_win) from results --------------------
def _winner_lane_from_results(result_json_path: Path) -> Optional[int]:
    try:
        with result_json_path.open("r", encoding="utf-8") as f:
            js = json.load(f)
        for rec in js.get("results", []):
            if str(rec.get("rank")) == "1":
                lane = rec.get("lane")
                try:
                    return int(lane)
                except Exception:
                    return None
    except Exception:
        return None
    return None

def build_winner_index(results_root: str, candidate_days: List[Path]) -> Dict[Tuple[str,str,int], int]:
    """
    key: (hd, jcd, rno) -> winner_lane
    results_root: public/results
    candidate_days: data/staging/YYYY/MMDD ディレクトリのリスト（日付抽出に使う）
    """
    index: Dict[Tuple[str,str,int], int] = {}
    rroot = Path(results_root) if results_root else None
    if not rroot or not rroot.exists():
        return index

    for dd in candidate_days:
        year = dd.parent.name
        md = dd.name
        res_day_dir = rroot / year / md
        if not res_day_dir.exists():  # ない日はスキップ
            continue
        # 形: public/results/YYYY/MMDD/{jcd}/{rno}R.json
        for jcd_dir in res_day_dir.glob("*"):
            if not jcd_dir.is_dir(): continue
            jcd = jcd_dir.name
            for rfile in jcd_dir.glob("*R.json"):
                name = rfile.stem  # "10R"
                try:
                    rno = int(name[:-1])
                except Exception:
                    continue
                lane = _winner_lane_from_results(rfile)
                if lane is not None:
                    index[(year + md, jcd, rno)] = lane  # hd="YYYYMMDD"
    return index


# -------------------- dataset load & feature select --------------------
BASE_ID_COLS = ["hd","jcd","rno","lane","regno"]
# 最初はコアだけ（増やすのはいつでもできる）
PREF_FEATURES = [
    "course_first_rate", "course_3rd_rate", "course_avg_st",
    "tenji_st_sec", "tenji_rank", "st_rank",
    "wind_speed_m", "wave_height_cm",
]

def load_dataset(paths: List[Path]) -> pl.DataFrame:
    if not paths: return pl.DataFrame()
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in paths]
    return pl.concat(lfs).collect(streaming=True)

def add_is_win_from_results(df: pl.DataFrame, results_root: Optional[str], day_dirs: List[Path]) -> pl.DataFrame:
    """
    df に is_win 列がなければ results から補完して追加する。
    既に is_win があればそのまま返す。
    """
    if "is_win" in df.columns:
        return df

    if not results_root:
        log("[WARN] results_root が未指定のため is_win を付与できません（既存列も無し）。")
        return df

    index = build_winner_index(results_root, day_dirs)
    if not index:
        log("[WARN] winner index が空です。is_win は付与されません。")
        return df

    # hd は "YYYYMMDD" を想定
    def winner_lane_expr() -> pl.Expr:
        # join 的なことを udf なしでやるのは難しいので map_rows 相当で処理
        return (pl.struct(["hd","jcd","rno","lane"])
                  .map_elements(lambda s: 1 if index.get((str(s["hd"]), str(s["jcd"]), int(s["rno"]))) == int(s["lane"]) else 0)
                  .alias("is_win"))

    try:
        df2 = df.with_columns([winner_lane_expr()])
        return df2
    except Exception as e:
        err(f"[WARN] is_win 付与に失敗: {e}")
        return df

def select_features(df: pl.DataFrame):
    cols = set(df.columns)
    if "is_win" not in cols:
        raise RuntimeError("入力に 'is_win' 列がありません。（results からの自動付与も失敗）")

    use_feats: List[str] = [c for c in PREF_FEATURES if c in cols]
    # 数値列を追加（ID/ラベル以外）
    for c, dt in df.schema.items():
        if c in BASE_ID_COLS or c == "is_win":
            continue
        dt_str = str(dt)
        if ("Int" in dt_str) or ("Float" in dt_str):
            if c not in use_feats:
                use_feats.append(c)
    # 冗長なら上限（適宜調整）
    if len(use_feats) > 64:
        use_feats = use_feats[:64]

    dfx = df.select(BASE_ID_COLS + ["is_win"] + use_feats).fill_null(0)
    pdf = dfx.to_pandas()  # pyarrow 拡張は使わない（環境依存を減らす）
    return pdf, use_feats


# -------------------- model --------------------
def fit_lgb(train_pd, feat_cols, cfg: TrainConfig) -> lgb.Booster:
    X = train_pd[feat_cols]
    y = train_pd["is_win"].astype(int).values
    params = dict(
        objective="binary", metric="auc",
        learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth, min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.feature_fraction, bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq, lambda_l2=cfg.lambda_l2,
        num_threads=cfg.num_threads, seed=cfg.seed, verbose=-1,
        force_col_wise=True,
    )
    dset = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=True)
    booster = lgb.train(params, dset, num_boost_round=cfg.num_boost_round,
                        valid_sets=[dset], valid_names=["train"],
                        callbacks=[lgb.log_evaluation(period=200)])
    return booster

def predict_df(booster: lgb.Booster, test_pd, feat_cols, temperature: float = 1.0):
    X = test_pd[feat_cols]
    p = booster.predict(X, num_iteration=booster.best_iteration)
    # 温度スケーリング（>0、1.0で等価）
    if temperature and temperature > 0 and temperature != 1.0:
        import numpy as np
        # ロジット変換
        eps = 1e-12
        p_ = np.clip(p, eps, 1 - eps)
        logit = np.log(p_ / (1 - p_))
        logit /= float(temperature)
        p = 1 / (1 + np.exp(-logit))
    out = test_pd[BASE_ID_COLS].copy()
    out["proba_win"] = p
    return out


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()

    # 現行引数
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--out_root",     default="data/proba/single")
    ap.add_argument("--model_root",   default="data/models/single")
    ap.add_argument("--train_start",  default="20240101")
    ap.add_argument("--train_end",    default="20240131")
    ap.add_argument("--test_start",   default="20240201")
    ap.add_argument("--test_end",     default="20240229")
    ap.add_argument("--normalize",    type=lambda x: str(x).lower()=="true", default=True)
    ap.add_argument("--temperature",  type=float, default=1.0)

    # 互換（旧ワークフロー）
    ap.add_argument("--pred_start",     default=None)  # -> test_start
    ap.add_argument("--pred_end",       default=None)  # -> test_end
    ap.add_argument("--model_out",      default=None)  # モデル保存ファイル明示
    ap.add_argument("--proba_out_root", default=None)  # -> out_root

    # さらに古い書式（今回エラーを出していた系）
    ap.add_argument("--start",        default=None)   # -> train_start
    ap.add_argument("--end",          default=None)   # -> train_end
    ap.add_argument("--proba_root",   default=None)   # -> out_root
    ap.add_argument("--inplace",      default=None)   # 無視

    args = ap.parse_args()

    # 互換マッピング
    test_start = args.pred_start if args.pred_start else args.test_start
    test_end   = args.pred_end   if args.pred_end   else args.test_end

    out_root = args.out_root
    if args.proba_out_root: out_root = args.proba_out_root
    elif args.proba_root:   out_root = args.proba_root

    train_start = args.train_start if args.train_start else (args.start or "20240101")
    train_end   = args.train_end   if args.train_end   else (args.end   or "20240131")

    cfg = TrainConfig(
        staging_root=args.staging_root,
        results_root=args.results_root,
        out_root=out_root,
        model_root=args.model_root,
        model_out=args.model_out,
        train_start=train_start, train_end=train_end,
        test_start=test_start,   test_end=test_end,
        normalize=args.normalize, temperature=args.temperature,
    )

    # 入力収集
    train_files = find_staging_csvs(cfg.staging_root, cfg.train_start, cfg.train_end)
    test_files  = find_staging_csvs(cfg.staging_root, cfg.test_start,  cfg.test_end)

    if not train_files:
        err(f"[FATAL] train CSV が見つかりません: {cfg.staging_root} [{cfg.train_start}..{cfg.train_end}]")
        sys.exit(1)

    # day_dirs（is_win付与に使う）
    train_day_dirs = sorted({p.parent for p in train_files})
    test_day_dirs  = sorted({p.parent for p in test_files}) if test_files else []

    log(f"[INFO] train files: {len(train_files)}")
    if test_files:
        log(f"[INFO] test  files: {len(test_files)}")

    df_train = load_dataset(train_files)
    if df_train.height == 0:
        err("[FATAL] 学習データが空です。")
        sys.exit(1)

    # is_win 自動付与（存在すれば上書きしない）
    df_train = add_is_win_from_results(df_train, cfg.results_root, train_day_dirs)
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

    # 予測（テスト範囲があれば）
    if test_files:
        df_test = load_dataset(test_files)
        if df_test.height == 0:
            log("[WARN] テストデータが空です。予測はスキップします。")
            return
        # is_win は不要だが、残っていても問題なし
        keep_cols = BASE_ID_COLS + feat_cols + (["is_win"] if "is_win" in df_test.columns else [])
        df_test2 = df_test.select([c for c in keep_cols if c in df_test.columns]).fill_null(0)
        test_pd = df_test2.to_pandas()

        pred = predict_df(booster, test_pd, feat_cols, temperature=cfg.temperature)
        # 出力
        y = cfg.test_start[:4]; md = cfg.test_start[4:8]
        out_dir = Path(cfg.out_root) / y / md
        out_csv = ensure_parent(out_dir / f"proba_{cfg.test_start}_{cfg.test_end}.csv")
        pred.to_csv(out_csv, index=False)
        log(f"[WRITE] {out_csv} rows={len(pred)}")


if __name__ == "__main__":
    main()
