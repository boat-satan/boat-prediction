#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb

try:
    import yaml  # pip install pyyaml
except Exception:
    yaml = None

# ---------------- utils ----------------
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def split_period_lazy(scan: pl.LazyFrame, start: str, end: str) -> pl.LazyFrame:
    return (
        scan.with_columns(pl.col("hd").cast(pl.Utf8))
            .filter(pl.col("hd").is_between(pl.lit(start), pl.lit(end), closed="both"))
    )

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

    if isinstance(js.get("trifecta"), dict):
        return {str(k): float(v) for k, v in js["trifecta"].items() if v not in (None, "")}

    cand = js.get("odds") or js.get("trifecta_list") or js.get("3t") or js.get("list")
    if isinstance(cand, list):
        m = {}
        for row in cand:
            if not isinstance(row, dict): continue
            combo = row.get("combo") or row.get("combination") or row.get("key")
            odd   = row.get("odd")   or row.get("odds")        or row.get("value")
            if combo is None or odd in (None, ""): continue
            try:
                m[str(combo)] = float(str(odd).replace(",", ""))
            except Exception:
                continue
        return m if m else None

    flat = {k: js[k] for k in js.keys() if "-" in k}
    if flat:
        try:
            return {k: float(str(v).replace(",", "")) for k, v in flat.items()}
        except Exception:
            return None

    return None

def build_feature_df(df120: pl.DataFrame):
    pdf = df120.to_pandas()
    keys = ["hd","jcd","rno","combo"]
    y = pdf["is_hit"].astype(int)
    drop_cols = set(keys + ["is_hit"])
    feat_cols = [c for c in pdf.columns if c not in drop_cols]
    for c in feat_cols:
        if pdf[c].dtype == "object":
            pdf[c] = pdf[c].astype("category")
    X = pdf[feat_cols]
    return X, y, pdf[keys], feat_cols

def add_cli_and_yaml_args():
    ap = argparse.ArgumentParser()
    # 期間
    ap.add_argument("--train_start")
    ap.add_argument("--train_end")
    ap.add_argument("--test_start")
    ap.add_argument("--test_end")

    # TopN & 購入
    ap.add_argument("--topk", type=int, default=18, choices=[4,6,8,12,16,18,24,36])
    ap.add_argument("--unit_stake", type=int, default=100)

    # データルート
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_dir", default="public/results")
    ap.add_argument("--odds_root", default="public/odds/v1")

    # 並べ替え & EV
    ap.add_argument("--rank_key", default="proba", choices=["proba","ev"])
    ap.add_argument("--ev_min", type=float, default=0.0)
    ap.add_argument("--ev_drop_if_missing_odds", action="store_true")

    # ★参戦条件：1号艇単勝率（モデル確率）しきい値 & 1頭除外
    ap.add_argument("--p1win_max", type=float, default=0.50,
                    help="参戦するレースの1号艇単勝率の上限（p1win<=この値）")
    ap.add_argument("--exclude_one_head", action="store_true",
                    help="TopN選定時に 1頭('1-')の買い目を除外する")

    # 学習モデル
    ap.add_argument("--model_out", default="data/model_lgbm.txt")

    # YAML設定
    ap.add_argument("--config", help="yaml設定ファイル（指定時はYAMLが優先）")
    return ap

def merge_with_yaml(args: argparse.Namespace) -> argparse.Namespace:
    if not args.config:
        return args
    if yaml is None:
        print("[WARN] PyYAML未導入のため --config は無視。pip install pyyaml", file=sys.stderr)
        return args
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[WARN] configが見つかりません: {cfg_path}", file=sys.stderr)
        return args
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    def set_default(k, v):
        if hasattr(args, k) and v is not None:
            setattr(args, k, v)

    # 期間
    train = cfg.get("train", {})
    test  = cfg.get("test", {})
    set_default("train_start", str(train.get("start", args.train_start)))
    set_default("train_end",   str(train.get("end",   args.train_end)))
    set_default("test_start",  str(test.get("start",  args.test_start)))
    set_default("test_end",    str(test.get("end",    args.test_end)))

    # パス
    paths = cfg.get("paths", {})
    set_default("data_dir",    paths.get("data_dir",    args.data_dir))
    set_default("results_dir", paths.get("results_dir", args.results_dir))
    set_default("odds_root",   paths.get("odds_root",   args.odds_root))

    # TopN/単価
    set_default("topk",        int(cfg.get("topk", args.topk)))
    set_default("unit_stake",  int(cfg.get("unit_stake", args.unit_stake)))

    # 並べ順/EV
    set_default("rank_key", cfg.get("rank_key", args.rank_key))
    ev = cfg.get("ev", {})
    set_default("ev_min", float(ev.get("min", args.ev_min)))
    set_default("ev_drop_if_missing_odds", bool(ev.get("drop_if_missing_odds", args.ev_drop_if_missing_odds)))

    # 参戦条件
    filt = cfg.get("filters", {})
    set_default("p1win_max", float(filt.get("p1win_max", args.p1win_max)))
    set_default("exclude_one_head", bool(filt.get("exclude_one_head", args.exclude_one_head)))

    # モデル
    set_default("model_out", cfg.get("model_out", args.model_out))
    return args

# ---------------- main ----------------
def main():
    ap = add_cli_and_yaml_args()
    args = ap.parse_args()
    args = merge_with_yaml(args)

    data_dir = Path(args.data_dir)
    eval_dir = data_dir / "eval"
    ensure_dir(eval_dir)

    # ---- シャード検出 ----
    shards_root = Path("data/shards")
    pq_paths = sorted(shards_root.rglob("train_120_pro.parquet"))
    csv_paths = sorted(shards_root.rglob("train_120_pro.csv"))
    csv_gz_paths = sorted(shards_root.rglob("train_120_pro.csv.gz"))

    if pq_paths:
        lf = pl.scan_parquet([str(p) for p in pq_paths])
    elif csv_paths or csv_gz_paths:
        lf = pl.scan_csv([*(str(p) for p in csv_paths), *(str(p) for p in csv_gz_paths)], infer_schema_length=10000)
    else:
        raise FileNotFoundError("No shards under data/shards/**/train_120_pro.(parquet|csv|csv.gz)")

    # ---- 期間分割 ----
    if not (args.train_start and args.train_end and args.test_start and args.test_end):
        raise RuntimeError("train/test は --train_start/end と --test_start/end で指定するか、YAMLで指定してください。")
    tr_df = split_period_lazy(lf, args.train_start, args.train_end).collect(engine="streaming")
    te_df = split_period_lazy(lf, args.test_start,  args.test_end ).collect(engine="streaming")
    if tr_df.is_empty(): raise RuntimeError("学習期間データが空")
    if te_df.is_empty(): raise RuntimeError("テスト期間データが空")

    # ---- 学習 ----
    Xtr, ytr, keytr, feat_cols = build_feature_df(tr_df)
    train_set = lgb.Dataset(Xtr, label=ytr, categorical_feature=[c for c in feat_cols if str(Xtr[c].dtype) == "category"])
    params = dict(objective="binary", metric="auc", learning_rate=0.05, num_leaves=63, max_depth=-1,
                  min_data_in_leaf=50, feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                  lambda_l2=1.0, verbose=-1, num_threads=0, seed=20240301, force_col_wise=True)
    booster = lgb.train(params, train_set, num_boost_round=1000, valid_sets=[train_set], valid_names=["train"],
                        callbacks=[lgb.log_evaluation(period=200)])
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.model_out)

    # ---- 予測（テスト）----
    Xte, yte, keyte, _ = build_feature_df(te_df)
    proba = booster.predict(Xte, num_iteration=booster.best_iteration)
    pred_df = keyte.copy()
    pred_df["proba"] = proba
    pred_df["is_hit"] = yte.values

    # ---- 1号艇単勝率 p1win（モデル確率から）----
    p1_df = (
        pred_df[pred_df["combo"].str.startswith("1-")]
        .groupby(["hd","jcd","rno"], as_index=False)["proba"].sum()
        .rename(columns={"proba":"p1win"})
    )
    pred_df = pred_df.merge(p1_df, on=["hd","jcd","rno"], how="left")
    pred_df["p1win"] = pred_df["p1win"].fillna(0.0)

    # ---- オッズ→EV 付与（必要時のみ）----
    use_ev = (args.rank_key == "ev")
    if use_ev or args.ev_min > 0 or args.ev_drop_if_missing_odds:
        pred_df["odds"] = pd.NA
        odds_root = Path(args.odds_root)
        for (hd, jcd, rno), idxs in pred_df.groupby(["hd","jcd","rno"]).groups.items():
            omap = load_odds_map(odds_root, hd, jcd, rno)
            if not omap: continue
            sub = pred_df.loc[idxs]
            pred_df.loc[idxs, "odds"] = sub["combo"].map(omap).astype("float64")
        pred_df["ev"] = pred_df["proba"] * pred_df["odds"]

    # ---- 参戦対象（p1win <= 閾値）でレース抽出 ----
    eligible = pred_df[pred_df["p1win"] <= args.p1win_max]
    if eligible.empty:
        raise RuntimeError(f"p1win<= {args.p1win_max} のレースが0。閾値を緩めてください。")
    eligible_keys = eligible.groupby(["hd","jcd","rno"]).size().reset_index()[["hd","jcd","rno"]]
    pred_df = pred_df.merge(eligible_keys, on=["hd","jcd","rno"], how="inner")

    # ---- TopN選定（1頭除外+EV/確率順）----
    picks = []
    for (hd, jcd, rno), g in pred_df.groupby(["hd","jcd","rno"], as_index=False):
        gg = g.copy()
        if args.exclude_one_head:
            gg = gg[~gg["combo"].str.startswith("1-")].copy()

        if use_ev:
            if args.ev_min > 0:
                gg = gg[gg["ev"].fillna(-np.inf) >= args.ev_min]
            if args.ev_drop_if_missing_odds:
                gg = gg[gg["ev"].notna()]
            gg = gg.sort_values("ev", ascending=False)
        else:
            gg = gg.sort_values("proba", ascending=False)

        if len(gg) > args.topk:
            gg = gg.head(args.topk)
        if len(gg) == 0:
            continue

        gg = gg.copy()
        gg["rank"] = range(1, len(gg)+1)
        picks.append(gg)

    if len(picks) == 0:
        raise RuntimeError("選定結果がゼロ件。p1win/1頭除外/EV条件が厳しすぎる可能性。")
    picks_df = pd.concat(picks, ignore_index=True)

    # ---- ヒット率（レース単位）----
    race_hit = picks_df.groupby(["hd","jcd","rno"])["is_hit"].max().reset_index(name="race_hit")
    hit_rate = race_hit["race_hit"].mean()

    # ---- ROI（実配当）----
    results_root = Path(args.results_dir)
    unique_races = race_hit[["hd","jcd","rno"]].to_records(index=False)
    returns = 0
    total_bet = 0
    for hd, jcd, rno in unique_races:
        sub = picks_df[(picks_df["hd"]==hd)&(picks_df["jcd"]==jcd)&(picks_df["rno"]==rno)]
        total_bet += len(sub) * args.unit_stake
        if sub["is_hit"].max() == 1:
            amt = load_results_amount(results_root, str(hd), jcd, rno)
            if amt is not None:
                returns += amt
    roi = (returns / total_bet) if total_bet > 0 else 0.0

    # ---- 出力 ----
    mode_tag = f"{args.rank_key}_evmin{args.ev_min}_p1max{args.p1win_max}".replace(".","_")
    if args.exclude_one_head: mode_tag += "_no1head"
    picks_path = (eval_dir / f"picks_{args.test_start}_{args.test_end}_k{args.topk}_{mode_tag}.csv")
    picks_df.to_csv(picks_path, index=False, encoding="utf-8")

    summary = {
        "train_range":[args.train_start,args.train_end],
        "test_range":[args.test_start,args.test_end],
        "num_train_rows": int(tr_df.height),
        "num_test_rows": int(te_df.height),
        "num_test_races": int(len(unique_races)),
        "topk": args.topk,
        "unit_stake": int(args.unit_stake),
        "rank_key": args.rank_key,
        "ev_min": float(args.ev_min),
        "ev_drop_if_missing_odds": bool(args.ev_drop_if_missing_odds),
        "p1win_max": float(args.p1win_max),
        "exclude_one_head": bool(args.exclude_one_head),
        "hit_rate": float(hit_rate),
        "returns": int(returns),
        "total_bet": int(total_bet),
        "roi": float(roi),
        "model_path": args.model_out,
        "picks_path": str(picks_path),
    }
    summary_path = (eval_dir / f"summary_{args.test_start}_{args.test_end}_k{args.topk}_{mode_tag}.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")

if __name__ == "__main__":
    main()
