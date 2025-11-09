#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_pipeline.py — 統合: 学習→予測→評価（pred_csv 未指定なら自動学習）

必要パッケージ: polars, pandas, lightgbm, pyyaml, numpy
"""

from __future__ import annotations
import argparse, json, sys, os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

# ============== 共通ユーティリティ ==============
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception:
        print("[FATAL] PyYAML が未導入です: pip install pyyaml", file=sys.stderr)
        raise
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def scan_shards() -> pl.LazyFrame:
    r = Path("data/shards")
    pq = sorted(r.rglob("train_120_pro.parquet"))
    csv = sorted(r.rglob("train_120_pro.csv"))
    csv_gz = sorted(r.rglob("train_120_pro.csv.gz"))
    if pq:
        return pl.scan_parquet([str(p) for p in pq])
    if csv or csv_gz:
        return pl.scan_csv([*(str(p) for p in csv), *(str(p) for p in csv_gz)], infer_schema_length=10000)
    raise FileNotFoundError("No shards under data/shards/**/train_120_pro.(parquet|csv|csv.gz)")

def split_period_lazy(scan: pl.LazyFrame, start: str, end: str) -> pl.LazyFrame:
    return (scan.with_columns(pl.col("hd").cast(pl.Utf8))
                .filter(pl.col("hd").is_between(pl.lit(start), pl.lit(end), closed="both")))

def build_feature_df(df120: pl.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, list]:
    pdf = df120.to_pandas(use_pyarrow_extension_array=False)
    keys = ["hd","jcd","rno","combo"]
    y = pdf["is_hit"].astype(int)
    drop_cols = set(keys + ["is_hit"])
    feat_cols = [c for c in pdf.columns if c not in drop_cols]
    for c in feat_cols:
        if pdf[c].dtype == "object":
            pdf[c] = pdf[c].astype("category")
    X = pdf[feat_cols]
    return X, y, pdf[keys], feat_cols

# ============== オッズ/結果のIO（評価用） ==============
def load_results_amount(results_root: Path, hd, jcd, rno) -> int | None:
    y  = str(hd)[:4]; md = str(hd)[4:8]
    try:    jcd_str = f"{int(jcd):02d}"
    except: jcd_str = str(jcd)
    try:    rno_int = int(rno)
    except: rno_int = int(str(rno).replace("R",""))
    p = results_root / y / md / jcd_str / f"{rno_int}R.json"
    if not p.exists(): return None
    js = json.loads(p.read_text(encoding="utf-8"))
    tri = (js.get("payouts") or {}).get("trifecta")
    if not tri: return None
    try:
        return int(str(tri.get("amount")).replace(",", ""))
    except Exception:
        return None

def load_odds_map(odds_root: Path, hd, jcd, rno) -> dict[str, float] | None:
    y  = str(hd)[:4]; md = str(hd)[4:8]
    try:    jcd_str = f"{int(jcd):02d}"
    except: jcd_str = str(jcd)
    try:    rno_int = int(rno)
    except: rno_int = int(str(rno).replace("R",""))
    p = odds_root / y / md / jcd_str / f"{rno_int}R.json"
    if not p.exists(): return None
    js = json.loads(p.read_text(encoding="utf-8"))

    # {"trifecta":[{"combo":"1-2-3","odds":...},...]} / {"trifecta":{"1-2-3":...}} / いずれにも対応
    if isinstance(js.get("trifecta"), list):
        m = {}
        for row in js["trifecta"]:
            if not isinstance(row, dict): continue
            combo = row.get("combo") or row.get("combination") or row.get("key")
            odd   = row.get("odds")  or row.get("odd")        or row.get("value")
            if combo is None or odd in (None, ""): continue
            try: m[str(combo)] = float(str(odd).replace(",", ""))
            except: continue
        return m if m else None
    if isinstance(js.get("trifecta"), dict):
        m = {}
        for k, v in js["trifecta"].items():
            if v in (None, ""): continue
            try: m[str(k)] = float(str(v).replace(",", ""))
            except: continue
        return m if m else None
    # fallbacks
    cand = js.get("odds") or js.get("list")
    if isinstance(cand, list):
        m = {}
        for row in cand:
            if not isinstance(row, dict): continue
            combo = row.get("combo") or row.get("combination") or row.get("key")
            odd   = row.get("odds")  or row.get("odd")        or row.get("value")
            if combo is None or odd in (None, ""): continue
            try: m[str(combo)] = float(str(odd).replace(",", ""))
            except: continue
        return m if m else None
    flat = {k: js[k] for k in js.keys() if "-" in k}
    if flat:
        m = {}
        for k, v in flat.items():
            try: m[k] = float(str(v).replace(",", ""))
            except: continue
        return m if m else None
    return None

# ============== 評価ロジック import（失敗時はフォールバック） ==============
def try_import_evaluator():
    try:
        # ルートを PYTHONPATH に追加（actionsでも動くように保険）
        repo_root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(repo_root))
        from src.boatpred.eval import EvalConfig, evaluate_topN
        return EvalConfig, evaluate_topN
    except Exception as e:
        print(f"[WARN] import src.boatpred.eval 失敗: {e}\n→ 簡易フォールバック評価で実行します。", file=sys.stderr)
        return None, None

# 簡易フォールバック評価（主要フラグのみ対応）
def fallback_evaluate_topN(pred_df: pd.DataFrame, odds_root: Path, results_root: Path, cfg) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # p1win/S18/一頭比 判定用に proba ソートTop18
    def top18_metrics(g: pd.DataFrame):
        g18 = g.sort_values("proba", ascending=False).head(18)
        s18 = g18["proba"].sum()
        is_one = g18["combo"].astype(str).str.startswith("1-")
        one_prob = g18.loc[is_one, "proba"].sum()
        return pd.Series({
            "S18": float(s18),
            "one_head_ratio_top18": float(one_prob / s18) if s18 > 0 else 0.0,
            "non1_candidates_top18": int((~is_one).sum())
        })

    m = pred_df.groupby(["hd","jcd","rno"]).apply(top18_metrics).reset_index()
    e = m[
        (m["S18"] >= cfg["s18_min"]) &
        (m["one_head_ratio_top18"] <= cfg["one_head_ratio_top18_max"]) &
        (m["non1_candidates_top18"] >= cfg["min_non1_candidates_top18"])
    ][["hd","jcd","rno"]]
    if e.empty:
        # 全採用（フィルタOFF同等）に倒す
        e = pred_df[["hd","jcd","rno"]].drop_duplicates()

    df = pred_df.merge(e, on=["hd","jcd","rno"], how="inner").copy()

    # オッズ/EV必要？
    need_odds = (cfg["rank_key"] == "ev") or cfg["ev_min"] > 0 or cfg["ev_drop_if_missing_odds"] or cfg["load_odds_always"] \
                or cfg["synth_odds_min"] > 0 or cfg["min_odds_min"] > 0 or cfg["ev_basket_min"] > 0 or cfg["odds_coverage_min"] > 0

    if need_odds:
        df["odds"] = np.nan
        df["combo"] = df["combo"].astype(str)
        for (hd, jcd, rno), idxs in df.groupby(["hd","jcd","rno"]).groups.items():
            om = load_odds_map(odds_root, hd, jcd, rno)
            if not om: continue
            sub = df.loc[idxs]
            mapped = pd.to_numeric(sub["combo"].map(om), errors="coerce")
            df.loc[idxs, "odds"] = mapped.to_numpy(dtype=float)
        df["ev"] = df["proba"] * df["odds"]

    # TopN抽出
    picks = []
    for (hd, jcd, rno), g in df.groupby(["hd","jcd","rno"], as_index=False):
        gg = g.copy()
        if cfg["rank_key"] == "ev":
            if cfg["ev_drop_if_missing_odds"]:
                gg = gg[gg["ev"].notna()]
            gg = gg.sort_values("ev", ascending=False)
            if cfg["ev_min"] > 0:
                gg = gg[gg["ev"].fillna(-np.inf) >= cfg["ev_min"]]
        else:
            gg = gg.sort_values("proba", ascending=False)
        gg = gg.head(cfg["topk"])
        if len(gg) == 0: continue
        gg["rank"] = range(1, len(gg)+1)
        picks.append(gg)
    if not picks:
        raise RuntimeError("選定結果が0件（フォールバック評価）")
    picks_df = pd.concat(picks, ignore_index=True)

    # ROI（実配当）
    race_hit = picks_df.groupby(["hd","jcd","rno"])["is_hit"].max().reset_index(name="race_hit")
    hit_rate = float(race_hit["race_hit"].mean())
    unique_races = race_hit[["hd","jcd","rno"]].to_records(index=False)
    returns = 0
    total_bet = 0
    for hd, jcd, rno in unique_races:
        sub = picks_df[(picks_df["hd"]==hd)&(picks_df["jcd"]==jcd)&(picks_df["rno"]==rno)]
        total_bet += len(sub) * cfg["unit_stake"]
        if sub["is_hit"].max() == 1:
            amt = load_results_amount(results_root, str(hd), jcd, rno)
            if amt is not None:
                returns += amt
    summary = {
        "num_test_races": int(len(unique_races)),
        "topk": int(cfg["topk"]),
        "unit_stake": int(cfg["unit_stake"]),
        "rank_key": cfg["rank_key"],
        "hit_rate": float(hit_rate),
        "returns": int(returns),
        "total_bet": int(total_bet),
        "roi": float((returns / total_bet) if total_bet > 0 else 0.0),
    }
    return picks_df, summary

# ============== 学習＆予測 ==============
def train_and_predict(train_start: str, train_end: str, test_start: str, test_end: str, model_out: Path) -> Tuple[pd.DataFrame, lgb.Booster]:
    lf = scan_shards()
    tr_df = split_period_lazy(lf, train_start, train_end).collect(engine="streaming")
    te_df = split_period_lazy(lf, test_start,  test_end ).collect(engine="streaming")
    if tr_df.is_empty(): raise RuntimeError("学習期間データが空")
    if te_df.is_empty(): raise RuntimeError("テスト期間データが空")

    Xtr, ytr, keytr, feat_cols = build_feature_df(tr_df)
    train_set = lgb.Dataset(
        Xtr, label=ytr,
        categorical_feature=[c for c in feat_cols if str(Xtr[c].dtype) == "category"]
    )
    params = dict(
        objective="binary", metric="auc", learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
        lambda_l2=1.0, verbose=-1, num_threads=0, seed=20240301, force_col_wise=True
    )
    booster = lgb.train(params, train_set, num_boost_round=1000,
                        valid_sets=[train_set], valid_names=["train"],
                        callbacks=[lgb.log_evaluation(period=200)])
    ensure_dir(model_out.parent)
    booster.save_model(str(model_out))

    Xte, yte, keyte, _ = build_feature_df(te_df)
    proba = booster.predict(Xte, num_iteration=booster.best_iteration)

    pred_df = keyte.copy()
    pred_df["proba"] = proba
    pred_df["is_hit"] = yte.values  # 評価用（命中フラグ）
    return pred_df, booster

# ============== メイン ==============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="パイプライン設定 YAML")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = read_yaml(cfg_path)

    # パス
    paths = cfg.get("paths", {})
    data_dir      = Path(paths.get("data_dir", "data"))
    out_dir       = Path(paths.get("out_dir", str(data_dir / "eval")))
    odds_root     = Path(paths.get("odds_root", "public/odds/v1"))
    results_root  = Path(paths.get("results_dir", "public/results"))
    ensure_dir(out_dir)
    compose_dir   = Path(paths.get("compose_dir", str(data_dir / "compose")))
    ensure_dir(compose_dir)

    # 期間
    train_cfg = cfg.get("train", {})
    test_cfg  = cfg.get("test", {})
    train_start = str(train_cfg.get("start", ""))
    train_end   = str(train_cfg.get("end", ""))
    test_start  = str(test_cfg.get("start", ""))
    test_end    = str(test_cfg.get("end", ""))

    # 評価オプション
    eval_opts = cfg.get("eval", {})
    topk   = int(eval_opts.get("topk", 18))
    unit   = int(eval_opts.get("unit_stake", 100))
    rkey   = eval_opts.get("rank_key", "proba")
    ev_min = float(eval_opts.get("ev_min", 0.0))
    ev_drop_miss = bool(eval_opts.get("ev_drop_if_missing_odds", False))
    load_odds_always = bool(eval_opts.get("load_odds_always", False))

    filters = cfg.get("filters", {})
    p1win_max = float(filters.get("p1win_max", 1.0))
    s18_min   = float(filters.get("s18_min", 0.0))
    oh_max    = float(filters.get("one_head_ratio_top18_max", 1.0))
    non1_min  = int(filters.get("min_non1_candidates_top18", 0))
    daily_cap = int(filters.get("daily_cap", 0))

    synth_odds_min    = float(eval_opts.get("synth_odds_min", 0.0))
    min_odds_min      = float(eval_opts.get("min_odds_min",   0.0))
    ev_basket_min     = float(eval_opts.get("ev_basket_min",  0.0))
    odds_coverage_min = int(eval_opts.get("odds_coverage_min",0))

    # 入力 pred_csv（未指定なら学習→予測）
    inputs = cfg.get("inputs", {})
    pred_csv = inputs.get("pred_csv", "")
    model_out = Path(cfg.get("model_out", str(data_dir / "model_lgbm.txt")))

    if pred_csv:
        pred_csv_path = Path(pred_csv)
        if not pred_csv_path.exists():
            raise FileNotFoundError(f"pred_csv not found: {pred_csv_path}")
        pred_df = pd.read_csv(pred_csv_path)
    else:
        # 学習→予測
        pred_df, _ = train_and_predict(train_start, train_end, test_start, test_end, model_out)
        # 出力predを保存
        span = f"{test_start}_{test_end}" if (test_start and test_end) else "test"
        pred_csv_path = compose_dir / f"preds_{span}_k120.csv"
        pred_df[["hd","jcd","rno","combo","proba","is_hit"]].to_csv(pred_csv_path, index=False, encoding="utf-8")
        print(f"[WRITE] {pred_csv_path}")

    # 評価
    EvalConfig, evaluate_topN = try_import_evaluator()
    if EvalConfig is not None:
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
        picks_df, summary = evaluate_topN(pred_df=pred_df, odds_root=odds_root, results_root=results_root, cfg=ec)
    else:
        cfg_fb = dict(
            topk=topk, unit_stake=unit, rank_key=rkey, ev_min=ev_min, ev_drop_if_missing_odds=ev_drop_miss,
            load_odds_always=load_odds_always, synth_odds_min=synth_odds_min, min_odds_min=min_odds_min,
            ev_basket_min=ev_basket_min, odds_coverage_min=odds_coverage_min,
            p1win_max=p1win_max, s18_min=s18_min, one_head_ratio_top18_max=oh_max,
            min_non1_candidates_top18=non1_min, daily_cap=daily_cap
        )
        picks_df, summary = fallback_evaluate_topN(pred_df, odds_root, results_root, cfg_fb)

    # 出力（期間をファイル名に）
    try:
        hds = sorted(map(str, pred_df["hd"].astype(str).unique()))
        span = f"{hds[0]}_{hds[-1]}_" if hds else ""
    except Exception:
        span = ""

    def _safen(x) -> str:
        try: return str(float(x)).replace(".", "_")
        except: return "na"

    mode = f"{rkey}_evmin{_safen(ev_min)}_p1{_safen(p1win_max)}_s18{_safen(s18_min)}_oh{_safen(oh_max)}"
    if daily_cap > 0: mode += f"_cap{daily_cap}"
    if (rkey == "ev") or load_odds_always or synth_odds_min>0 or min_odds_min>0 or ev_basket_min>0 or odds_coverage_min>0:
        mode += "_odds"
    if synth_odds_min>0 or min_odds_min>0 or ev_basket_min>0 or odds_coverage_min>0:
        mode += f"_sb{_safen(synth_odds_min)}_mo{_safen(min_odds_min)}_eb{_safen(ev_basket_min)}_cov{int(odds_coverage_min)}"

    picks_path = out_dir / f"picks_{span}k{topk}_{mode}.csv"
    summary_path = out_dir / f"summary_{span}k{topk}_{mode}.json"

    if not picks_df.empty:
        picks_df.to_csv(picks_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # 追加で学習/テスト件数も出力（pred_dfにis_hitがあれば）
    if "is_hit" in pred_df.columns:
        summary["num_test_rows"] = int(len(pred_df))
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")

if __name__ == "__main__":
    main()
