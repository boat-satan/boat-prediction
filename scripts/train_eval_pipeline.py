# scripts/train_eval_pipeline.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_pipeline.py — 統合パイプライン（ST学習を実行 + 既存評価を一発実行）

機能:
- YAMLを読んで実行。
- (1) ST学習ブロック: data/history/ や data/shards/** を横断探索して
      必須列 [hd, regno, lane, st] を含む履歴を読み込み、LightGBM回帰で学習。
      出力: data/models/st_lgbm.txt, data/models/st_lgbm.meta.json,
            data/st_out/st_feature_importance.csv
  ※見つからなければスキップし、data/st_out/ST_SKIPPED.txt を出すだけ。

- (2) 予測+評価ブロック:
      inputs.pred_csv があればそれを読み込み（hd,jcd,rno,combo,proba[,is_hit]）
      無ければ data/shards/**/train_120_pro.(parquet|csv|csv.gz) を使って
      簡易LightGBMで学習→テスト期間に対して120通りの proba を出力し、評価。
      評価は src/boatpred/eval.py の evaluate_topN を使用。

使い方:
  python scripts/train_eval_pipeline.py --config data/pipeline.config.yaml
"""

from __future__ import annotations
import argparse, json, sys, glob, os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

# 評価ロジック
from src.boatpred.eval import EvalConfig, evaluate_topN
# ST学習ブロック
from src.boatpred.st_model import STConfig as STCfg, train as st_train

# ----------------- ユーティリティ -----------------
def read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception:
        print("[FATAL] PyYAML が未導入です: pip install pyyaml", file=sys.stderr)
        raise
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _safen(x) -> str:
    try:
        return str(float(x)).replace(".", "_")
    except Exception:
        return "na"

# ----------------- ST: 履歴ローダ -----------------
REQUIRED_ST_COLS = {"hd", "regno", "lane", "st"}

def _has_cols(df: pd.DataFrame, cols=REQUIRED_ST_COLS) -> bool:
    return cols.issubset(set(df.columns))

def _read_parquet_safe(p: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(p)
    except Exception:
        return None

def _read_csv_safe(p: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(p)
    except Exception:
        return None

def load_st_history_candidates() -> pd.DataFrame | None:
    """
    data/history/ と data/shards/** を探索し、hd/regno/lane/st を含む
    Parquet/CSV を返す（最初に見つかったもの）。
    """
    cand_patterns = [
        "data/history/*.parquet",
        "data/history/*.csv",
        "data/shards/**/*.parquet",
        "data/shards/**/*.csv",
    ]
    for pat in cand_patterns:
        for p in glob.glob(pat, recursive=True):
            df = None
            if p.endswith(".parquet"):
                df = _read_parquet_safe(p)
            elif p.endswith(".csv") or p.endswith(".csv.gz"):
                df = _read_csv_safe(p)
            if df is not None and _has_cols(df):
                return df
    return None

# ----------------- 120通り用: 学習/予測（簡易） -----------------
def split_period_lazy(scan: pl.LazyFrame, start: str, end: str) -> pl.LazyFrame:
    return (
        scan.with_columns(pl.col("hd").cast(pl.Utf8))
            .filter(pl.col("hd").is_between(pl.lit(start), pl.lit(end), closed="both"))
    )

def build_feature_df_120(df120: pl.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """
    既存の 120通りシャード (train_120_pro.*) を pandas へ落として
    LGBM用の (X, y, keys, feat_cols) を返す。
    """
    pdf = df120.to_pandas(use_pyarrow_extension_array=False)
    keys = ["hd","jcd","rno","combo"]
    y = pdf["is_hit"].astype(int)
    drop_cols = set(keys + ["is_hit"])
    feat_cols = [c for c in pdf.columns if c not in drop_cols]
    # 文字列はカテゴリに
    for c in feat_cols:
        if pdf[c].dtype == "object":
            pdf[c] = pdf[c].astype("category")
    X = pdf[feat_cols]
    return X, y, pdf[keys], feat_cols

def train_and_predict_120(train_start: str, train_end: str, test_start: str, test_end: str, model_out: str
) -> Tuple[pd.DataFrame, lgb.Booster]:
    """
    data/shards/**/train_120_pro.(parquet|csv|csv.gz) を読み、学習→テスト予測。
    返り値: pred_df(hd,jcd,rno,combo,proba,is_hit), booster
    """
    shards_root = Path("data/shards")
    pq_paths    = sorted(shards_root.rglob("train_120_pro.parquet"))
    csv_paths   = sorted(shards_root.rglob("train_120_pro.csv"))
    csv_gz_paths= sorted(shards_root.rglob("train_120_pro.csv.gz"))
    if pq_paths:
        lf = pl.scan_parquet([str(p) for p in pq_paths])
    elif csv_paths or csv_gz_paths:
        lf = pl.scan_csv([*(str(p) for p in csv_paths), *(str(p) for p in csv_gz_paths)], infer_schema_length=10000)
    else:
        raise FileNotFoundError("No shards under data/shards/**/train_120_pro.(parquet|csv|csv.gz)")

    tr_df = split_period_lazy(lf, train_start, train_end).collect(engine="streaming")
    te_df = split_period_lazy(lf, test_start,  test_end ).collect(engine="streaming")
    if tr_df.is_empty(): raise RuntimeError("学習期間データが空")
    if te_df.is_empty(): raise RuntimeError("テスト期間データが空")

    Xtr, ytr, keytr, feat_cols = build_feature_df_120(tr_df)
    train_set = lgb.Dataset(Xtr, label=ytr, categorical_feature=[c for c in feat_cols if str(Xtr[c].dtype) == "category"])
    params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63, max_depth=-1,
                  min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, num_threads=0, seed=20240301, force_col_wise=True)
    booster = lgb.train(params, train_set, num_boost_round=1000, valid_sets=[train_set], valid_names=["train"],
                        callbacks=[lgb.log_evaluation(period=200)])
    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(model_out)

    Xte, yte, keyte, _ = build_feature_df_120(te_df)
    proba = booster.predict(Xte, num_iteration=booster.best_iteration)
    pred_df = keyte.copy()
    pred_df["proba"]  = proba
    pred_df["is_hit"] = yte.values
    return pred_df, booster

# ----------------- メイン -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="パイプライン設定 YAML")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = read_yaml(cfg_path)

    # パス
    paths        = cfg.get("paths", {})
    data_dir     = Path(paths.get("data_dir", "data"))
    out_dir      = Path(paths.get("out_dir", str(data_dir / "eval")))
    odds_root    = Path(paths.get("odds_root", "public/odds/v1"))
    results_root = Path(paths.get("results_dir", "public/results"))
    ensure_dir(out_dir)

    # 期間
    train_cfg   = cfg.get("train", {})
    test_cfg    = cfg.get("test", {})
    train_start = str(train_cfg.get("start", ""))
    train_end   = str(train_cfg.get("end",   ""))
    test_start  = str(test_cfg.get("start",  ""))
    test_end    = str(test_cfg.get("end",    ""))

    # 評価設定
    eval_opts   = cfg.get("eval", {})
    topk        = int(eval_opts.get("topk", cfg.get("topk", 18)))
    unit        = int(eval_opts.get("unit_stake", cfg.get("unit_stake", 100)))
    rkey        = eval_opts.get("rank_key", cfg.get("rank_key", "proba"))
    ev_min      = float(eval_opts.get("ev_min", cfg.get("ev_min", 0.0)))
    ev_drop_miss= bool(eval_opts.get("ev_drop_if_missing_odds", cfg.get("ev_drop_if_missing_odds", False)))
    load_odds_always = bool(eval_opts.get("load_odds_always", cfg.get("load_odds_always", False)))

    filters     = cfg.get("filters", {})
    p1win_max   = float(filters.get("p1win_max", 1.0))
    s18_min     = float(filters.get("s18_min", 0.0))
    oh_max      = float(filters.get("one_head_ratio_top18_max", 1.0))
    non1_min    = int(filters.get("min_non1_candidates_top18", 0))
    daily_cap   = int(filters.get("daily_cap", 0))

    synth_odds_min    = float(eval_opts.get("synth_odds_min",  cfg.get("synth_odds_min",  0.0)))
    min_odds_min      = float(eval_opts.get("min_odds_min",    cfg.get("min_odds_min",    0.0)))
    ev_basket_min     = float(eval_opts.get("ev_basket_min",   cfg.get("ev_basket_min",   0.0)))
    odds_coverage_min = int(eval_opts.get("odds_coverage_min", cfg.get("odds_coverage_min",0)))

    model_out = str(cfg.get("model_out", "data/model_lgbm.txt"))

    # ----------------- (1) ST 学習ブロック -----------------
    st_out_dir = Path("data/st_out"); ensure_dir(st_out_dir)
    st_models_dir = Path("data/models"); ensure_dir(st_models_dir)

    try:
        hist_df = load_st_history_candidates()
        if hist_df is None:
            (st_out_dir / "ST_SKIPPED.txt").write_text(
                "hd,regno,lane,st を含む履歴ファイルが見つからないため ST 学習をスキップしました。\n"
                "data/history/*.parquet や data/shards/** を配置してください。\n",
                encoding="utf-8"
            )
            print("[ST] history not found -> SKIP")
        else:
            # ref_end は通常テスト期間の end を使う
            ref_end = test_end or train_end
            cfg_st = STCfg()
            booster, feat_cols = st_train(hist_df, ref_end=ref_end, cfg=cfg_st)
            # 特徴重要度
            try:
                imp = pd.DataFrame({
                    "feature": feat_cols,
                    "importance": booster.feature_importance(importance_type="gain")
                }).sort_values("importance", ascending=False)
                imp.to_csv(st_out_dir / "st_feature_importance.csv", index=False, encoding="utf-8")
                print("[ST] feature importance -> data/st_out/st_feature_importance.csv")
            except Exception as e:
                print(f"[ST] importance export failed: {e}", file=sys.stderr)
    except Exception as e:
        # 失敗しても、以降の評価には進む
        (st_out_dir / "ST_FAILED.txt").write_text(
            f"ST学習で例外が発生: {e}\n(評価は続行済み)", encoding="utf-8"
        )
        print(f"[ST] ERROR but continue eval: {e}", file=sys.stderr)

    # ----------------- (2) 予測ソースの決定 -----------------
    inputs = cfg.get("inputs", {})
    pred_csv = inputs.get("pred_csv") if isinstance(inputs, dict) else None

    if pred_csv:
        pred_csv_path = Path(pred_csv)
        if not pred_csv_path.exists():
            raise FileNotFoundError(f"pred_csv not found: {pred_csv_path}")
        pred_df = pd.read_csv(pred_csv_path)
    else:
        # 既存120通りシャードを使って学習→予測
        pred_df, _ = train_and_predict_120(train_start, train_end, test_start, test_end, model_out)

    # ----------------- 評価 -----------------
    ec = EvalConfig(
        topk=topk,
        unit_stake=unit,
        rank_key=rkey,
        ev_min=ev_min,
        ev_drop_if_missing_odds=ev_drop_miss,
        load_odds_always=load_odds_always,
        synth_odds_min=synth_odds_min,
        min_odds_min=min_odds_min,
        ev_basket_min=ev_basket_min,
        odds_coverage_min=odds_coverage_min,
        p1win_max=p1win_max,
        s18_min=s18_min,
        one_head_ratio_top18_max=oh_max,
        min_non1_candidates_top18=non1_min,
        daily_cap=daily_cap,
    )

    picks_df, summary = evaluate_topN(
        pred_df=pred_df,
        odds_root=odds_root,
        results_root=results_root,
        cfg=ec,
    )

    # 出力
    try:
        hds = sorted(map(str, pred_df["hd"].astype(str).unique()))
        span = f"{hds[0]}_{hds[-1]}_" if hds else ""
    except Exception:
        span = ""

    mode = f"{ec.rank_key}_evmin{_safen(ec.ev_min)}_p1{_safen(ec.p1win_max)}_s18{_safen(ec.s18_min)}_oh{_safen(ec.one_head_ratio_top18_max)}"
    if ec.daily_cap > 0: mode += f"_cap{ec.daily_cap}"
    if (ec.rank_key == "ev") or ec.load_odds_always or ec.synth_odds_min>0 or ec.min_odds_min>0 or ec.ev_basket_min>0:
        mode += "_odds"
    if ec.synth_odds_min>0 or ec.min_odds_min>0 or ec.ev_basket_min>0 or ec.odds_coverage_min>0:
        mode += f"_sb{_safen(ec.synth_odds_min)}_mo{_safen(ec.min_odds_min)}_eb{_safen(ec.ev_basket_min)}_cov{int(ec.odds_coverage_min)}"

    picks_path   = out_dir / f"picks_{span}k{ec.topk}_{mode}.csv"
    summary_path = out_dir / f"summary_{span}k{ec.topk}_{mode}.json"

    if not picks_df.empty:
        picks_df.to_csv(picks_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")


if __name__ == "__main__":
    main()
