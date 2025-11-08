#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import polars as pl
import pandas as pd
import lightgbm as lgb

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# 文字列日付の範囲フィルタ（Literalを渡す）
def split_period_lazy(scan: pl.LazyFrame, start: str, end: str) -> pl.LazyFrame:
    return (
        scan
        .with_columns(pl.col("hd").cast(pl.Utf8))
        .filter(pl.col("hd").is_between(pl.lit(start), pl.lit(end), closed="both"))
    )

def load_results_amount(results_root: Path, hd: str, jcd: str, rno: int) -> int | None:
    y, md = hd[:4], hd[4:8]
    p = results_root / y / md / jcd / f"{rno}R.json"
    if not p.exists(): return None
    js = json.loads(p.read_text(encoding="utf-8"))
    tri = (js.get("payouts") or {}).get("trifecta")
    if not tri: return None
    try:
        return int(str(tri.get("amount")).replace(",", ""))
    except Exception:
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", required=True)
    ap.add_argument("--train_end",   required=True)
    ap.add_argument("--test_start",  required=True)
    ap.add_argument("--test_end",    required=True)
    ap.add_argument("--topk", type=int, default=18, choices=[12,16,18,24,36])
    ap.add_argument("--unit_stake", type=int, default=100)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_dir", default="public/results")
    ap.add_argument("--model_out", default="data/model_lgbm.txt")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    eval_dir = data_dir / "eval"
    ensure_dir(eval_dir)

    # ---- シャード検出（Parquet優先、無ければCSV/CSV.GZをLazyで）----
    shards_root = Path("data/shards")
    pq_paths = sorted(shards_root.rglob("train_120_pro.parquet"))
    csv_paths = sorted(shards_root.rglob("train_120_pro.csv"))
    csv_gz_paths = sorted(shards_root.rglob("train_120_pro.csv.gz"))

    if pq_paths:
        lf = pl.scan_parquet([str(p) for p in pq_paths])
    elif csv_paths or csv_gz_paths:
        lf = pl.scan_csv(
            [*(str(p) for p in csv_paths), *(str(p) for p in csv_gz_paths)],
            infer_schema_length=10000,
        )
    else:
        raise FileNotFoundError("No shards under data/shards/**/train_120_pro.(parquet|csv|csv.gz)")

    # ---- 期間分割（Lazyのまま）----
    tr_lf = split_period_lazy(lf, args.train_start, args.train_end)
    te_lf = split_period_lazy(lf, args.test_start, args.test_end)

    # ---- collect（推奨のengine指定でストリーミング実行）----
    tr_df = tr_lf.collect(engine="streaming")
    te_df = te_lf.collect(engine="streaming")

    if tr_df.is_empty():
        raise RuntimeError("学習期間に該当するデータが空です。")
    if te_df.is_empty():
        raise RuntimeError("テスト期間に該当するデータが空です。")

    # ---- 学習 ----
    Xtr, ytr, keytr, feat_cols = build_feature_df(tr_df)
    train_set = lgb.Dataset(
        Xtr, label=ytr,
        categorical_feature=[c for c in feat_cols if str(Xtr[c].dtype) == "category"]
    )
    params = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_data_in_leaf=50, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0,
        verbose=-1, num_threads=0, seed=20240301, force_col_wise=True,
    )
    booster = lgb.train(
        params, train_set, num_boost_round=1000,
        valid_sets=[train_set], valid_names=["train"], verbose_eval=200
    )
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.model_out)

    # ---- 予測（テスト）----
    Xte, yte, keyte, _ = build_feature_df(te_df)
    proba = booster.predict(Xte, num_iteration=booster.best_iteration)
    pred_df = keyte.copy(); pred_df["proba"] = proba; pred_df["is_hit"] = yte.values

    # TopK選定
    picks = []
    for (hd, jcd, rno), g in pred_df.groupby(["hd","jcd","rno"], as_index=False):
        gg = g.sort_values("proba", ascending=False).head(args.topk).copy()
        gg["rank"] = range(1, len(gg)+1)
        picks.append(gg)
    picks_df = pd.concat(picks, ignore_index=True)

    # Hit率（レース単位）
    race_hit = picks_df.groupby(["hd","jcd","rno"])["is_hit"].max().reset_index(name="race_hit")
    hit_rate = race_hit["race_hit"].mean()

    # ROI（的中レースの3連単払い戻し合計 / 総投資）
    results_root = Path(args.results_dir)
    unique_races = race_hit[["hd","jcd","rno"]].to_records(index=False)
    returns = 0
    total_bet = len(unique_races) * args.topk * args.unit_stake
    for hd, jcd, rno in unique_races:
        if race_hit[(race_hit["hd"]==hd)&(race_hit["jcd"]==jcd)&(race_hit["rno"]==rno)]["race_hit"].iloc[0] == 1:
            amt = load_results_amount(results_root, hd, jcd, int(rno))
            if amt is not None:
                returns += amt
    roi = returns / total_bet if total_bet > 0 else 0.0

    # 出力
    picks_path = eval_dir / f"picks_{args.test_start}_{args.test_end}_k{args.topk}.csv"
    picks_df.to_csv(picks_path, index=False, encoding="utf-8")
    summary = {
        "train_range":[args.train_start,args.train_end],
        "test_range":[args.test_start,args.test_end],
        "num_train_rows": int(tr_df.height),
        "num_test_rows": int(te_df.height),
        "num_test_races": int(len(unique_races)),
        "topk": args.topk,
        "unit_stake": args.unit_stake,
        "hit_rate": float(hit_rate),
        "returns": int(returns),
        "total_bet": int(total_bet),
        "roi": float(roi),
        "model_path": args.model_out,
        "features_used": [c for c in tr_df.columns if c not in ("hd","jcd","rno","combo","is_hit")],
    }
    summary_path = eval_dir / f"summary_{args.test_start}_{args.test_end}_k{args.topk}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")

if __name__ == "__main__":
    main()
