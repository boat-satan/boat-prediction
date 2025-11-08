#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
import polars as pl
import pandas as pd
import lightgbm as lgb

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

# 期間フィルタ（文字列で between）
def split_period_lazy(scan: pl.LazyFrame, start: str, end: str) -> pl.LazyFrame:
    return (
        scan.with_columns(pl.col("hd").cast(pl.Utf8))
            .filter(pl.col("hd").is_between(pl.lit(start), pl.lit(end), closed="both"))
    )

# 実配当の取得（評価用）
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

# オッズ読取（3連単）。柔軟パーサ：{combo:odd} or [{"combo":..,"odds":..}] 等
def load_odds_map(odds_root: Path, hd, jcd, rno) -> dict[str, float] | None:
    y  = str(hd)[:4]; md = str(hd)[4:8]
    try:    jcd_str = f"{int(jcd):02d}"
    except: jcd_str = str(jcd)
    try:    rno_int = int(rno)
    except: rno_int = int(str(rno).replace("R",""))
    # 例: public/odds/v1/2024/0201/06/1R.json
    p = odds_root / y / md / jcd_str / f"{rno_int}R.json"
    if not p.exists(): return None
    js = json.loads(p.read_text(encoding="utf-8"))

    # 取りうる形を吸収
    # 1) {"trifecta": {"1-2-3": 15.1, ...}}
    if isinstance(js.get("trifecta"), dict):
        return {str(k): float(v) for k, v in js["trifecta"].items() if v not in (None, "")}

    # 2) {"odds": [{"combo":"1-2-3","odd":15.1}, ...]} など
    cand = js.get("odds") or js.get("trifecta_list") or js.get("3t") or js.get("list")
    if isinstance(cand, list):
        m = {}
        for row in cand:
            if not isinstance(row, dict): continue
            combo = row.get("combo") or row.get("combination") or row.get("key")
            odd   = row.get("odd") or row.get("odds") or row.get("value")
            if combo is None or odd in (None, ""): continue
            try:
                m[str(combo)] = float(str(odd).replace(",", ""))
            except Exception:
                continue
        return m if m else None

    # 3) {"1-2-3": 15.1, ...}
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

    # ▼ EV 関連オプション
    ap.add_argument("--odds_root", default="public/odds/v1",
                    help="3連単オッズのルート（例: public/odds/v1）")
    ap.add_argument("--rank_key", default="ev", choices=["proba","ev"],
                    help="TopK選定の並び順（確率 or EV）")
    ap.add_argument("--ev_min", type=float, default=0.0,
                    help="EVフィルタ（この値以上のみ採用。例: 1.0）")
    ap.add_argument("--ev_drop_if_missing_odds", action="store_true",
                    help="オッズ不明の組をEV不採用としてドロップする（rank_key=ev時推奨）")

    ap.add_argument("--model_out", default="data/model_lgbm.txt")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    eval_dir = data_dir / "eval"
    ensure_dir(eval_dir)

    # ---- シャード検出（Parquet優先、無ければCSV/CSV.GZ）----
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

    # ---- 期間分割（Lazy）→ collect（ストリーミング）----
    tr_lf = split_period_lazy(lf, args.train_start, args.train_end)
    te_lf = split_period_lazy(lf, args.test_start, args.test_end)
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
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)]
    )
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(args.model_out)

    # ---- 予測（テスト）----
    Xte, yte, keyte, _ = build_feature_df(te_df)
    proba = booster.predict(Xte, num_iteration=booster.best_iteration)

    pred_df = keyte.copy()
    pred_df["proba"] = proba
    pred_df["is_hit"] = yte.values

    # ---- オッズ→EV 付与（EV = proba * odds）----
    odds_root = Path(args.odds_root)
    # レース単位で odds を読み、pred_df に結合
    pred_df["odds"] = pd.NA
    for (hd, jcd, rno), idxs in pred_df.groupby(["hd","jcd","rno"]).groups.items():
        omap = load_odds_map(odds_root, hd, jcd, rno)
        if not omap:
            continue
        # そのレースの該当行にオッズ代入
        sub = pred_df.loc[idxs]
        # comboは "1-2-3" の形。omapキーと一致を試みる
        pred_df.loc[idxs, "odds"] = sub["combo"].map(omap).astype("float64")

    # EV計算（odds欠損はNaNのまま）
    pred_df["ev"] = pred_df["proba"] * pred_df["odds"]

    # ---- TopK選定（rank_key と EVフィルタ適用）----
    use_ev = (args.rank_key == "ev")
    picks = []
    for (hd, jcd, rno), g in pred_df.groupby(["hd","jcd","rno"], as_index=False):
        gg = g.copy()
        if use_ev:
            # EVフィルタ
            if args.ev_min > 0:
                gg = gg[gg["ev"] >= args.ev_min]
            if args.ev_drop_if_missing_odds:
                gg = gg[gg["ev"].notna()]
            # EV降順
            gg = gg.sort_values("ev", ascending=False)
        else:
            gg = gg.sort_values("proba", ascending=False)

        # TopK
        if len(gg) > args.topk:
            gg = gg.head(args.topk)
        gg = gg.copy()
        gg["rank"] = range(1, len(gg)+1)
        picks.append(gg)

    if len(picks) == 0:
        raise RuntimeError("選定結果がゼロ件です。オッズ欠損+厳しすぎるEVフィルタの可能性。")

    picks_df = pd.concat(picks, ignore_index=True)

    # ---- ヒット率（レース単位）----
    race_hit = picks_df.groupby(["hd","jcd","rno"])["is_hit"].max().reset_index(name="race_hit")
    hit_rate = race_hit["race_hit"].mean()

    # ---- ROI（実配当で評価）----
    results_root = Path(args.results_dir)
    unique_races = race_hit[["hd","jcd","rno"]].to_records(index=False)
    returns = 0
    total_bet = 0

    # レースごとの実投資点数 = 選定後の件数（TopK固定でなくEVフィルタ後の件数）
    # 1点あたり unit_stake 円
    for hd, jcd, rno in unique_races:
        sub = picks_df[(picks_df["hd"]==hd)&(picks_df["jcd"]==jcd)&(picks_df["rno"]==rno)]
        total_bet += len(sub) * args.unit_stake
        if sub["is_hit"].max() == 1:
            amt = load_results_amount(results_root, str(hd), jcd, rno)
            if amt is not None:
                returns += amt

    roi = (returns / total_bet) if total_bet > 0 else 0.0

    # ---- 出力 ----
    mode_tag = f"{args.rank_key}_evmin{args.ev_min}".replace(".","_")
    picks_path = (eval_dir / f"picks_{args.test_start}_{args.test_end}_k{args.topk}_{mode_tag}.csv")
    picks_df.to_csv(picks_path, index=False, encoding="utf-8")

    summary = {
        "train_range":[args.train_start,args.train_end],
        "test_range":[args.test_start,args.test_end],
        "num_train_rows": int(tr_df.height),
        "num_test_rows": int(te_df.height),
        "num_test_races": int(len(unique_races)),
        "topk": args.topk,
        "unit_stake": args.unit_stake,
        "rank_key": args.rank_key,
        "ev_min": args.ev_min,
        "ev_drop_if_missing_odds": bool(args.ev_drop_if_missing_odds),
        "hit_rate": float(hit_rate),
        "returns": int(returns),
        "total_bet": int(total_bet),
        "roi": float(roi),
        "model_path": args.model_out,
    }
    summary_path = (eval_dir / f"summary_{args.test_start}_{args.test_end}_k{args.topk}_{mode_tag}.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")

if __name__ == "__main__":
    main()
