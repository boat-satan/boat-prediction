#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_pro.py

学習:
  - data/train_120_pro.parquet を読み込み
  - 学習期間(例: 20240101-20240229)の120行テーブルから特徴→LightGBMで is_hit を2値学習
予測:
  - テスト期間(例: 20240301-20240331)に対して全120通りの的中確率を推論
選定:
  - レース毎に確率降順で TopK (= 12 or 18) を採用
評価:
  - ヒット率 = TopK内に的中が含まれたレース割合
  - ROI      = Σ(的中したレースの3連単払い戻し) / (総投資額)
    * 払い戻しは results/YYYY/MMDD/JCD/{R}R.json の trifecta.amount を使用
    * 1点あたり unit_stake 円の均等買い（デフォ 100円）
出力:
  - data/eval/summary_{test_start}_{test_end}_k{K}.json
  - data/eval/picks_{test_start}_{test_end}_k{K}.csv
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
import polars as pl
import pandas as pd
import lightgbm as lgb

# -----------------------
# Helpers
# -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def in_range(hd: str, start: str, end: str) -> bool:
    return start <= hd <= end

def load_results_amount(results_root: Path, hd: str, jcd: str, rno: int) -> int | None:
    """そのレースの的中3連単払い戻し額（円）。存在しない場合は None"""
    y, md = hd[:4], hd[4:8]
    p = results_root / y / md / jcd / f"{rno}R.json"
    if not p.exists():
        return None
    js = json.loads(p.read_text(encoding="utf-8"))
    tri = (js.get("payouts") or {}).get("trifecta")
    if not tri: return None
    amt = tri.get("amount")
    try:
        return int(amt)
    except Exception:
        # "15,090" のようなケースもケア
        try:
            return int(str(amt).replace(",", ""))
        except Exception:
            return None

def build_feature_df(df120: pl.DataFrame) -> pd.DataFrame:
    """Polars→Pandas。特徴行列X, 目的変数y, レースキー, 文字列カテゴリ整形。"""
    pdf = df120.to_pandas()

    # キー/ラベル
    keys = ["hd", "jcd", "rno", "combo"]
    y = pdf["is_hit"].astype(int)

    # 使わない列
    drop_cols = set(keys + ["is_hit"])
    # 残りを特徴に
    feat_cols = [c for c in pdf.columns if c not in drop_cols]

    # objectはカテゴリへ（LightGBMのcategorical_feature対応）
    for c in feat_cols:
        if pdf[c].dtype == "object":
            pdf[c] = pdf[c].astype("category")

    X = pdf[feat_cols]
    return X, y, pdf[keys], feat_cols

# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", required=True)  # YYYYMMDD
    ap.add_argument("--train_end",   required=True)
    ap.add_argument("--test_start",  required=True)
    ap.add_argument("--test_end",    required=True)
    ap.add_argument("--topk", type=int, default=12, choices=[12,16,18,24,36])
    ap.add_argument("--unit_stake", type=int, default=100)   # 1点あたりの金額（円）
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_dir", default="public/results")
    ap.add_argument("--model_out", default="data/model_lgbm.txt")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    results_root = Path(args.results_dir)
    eval_dir = Path(args.data_dir) / "eval"
    ensure_dir(eval_dir)

    # 120行テーブルを読み込み
    p_train120 = data_dir / "train_120_pro.parquet"
    if not p_train120.exists():
        raise FileNotFoundError(f"{p_train120} not found. 先に integrate_pro を実行してください。")
    df120 = pl.read_parquet(p_train120)

    # 期間で分割
    is_train = df120["hd"].cast(pl.Utf8).map_elements(
        lambda s: args.train_start <= s <= args.train_end
    )
    is_test  = df120["hd"].cast(pl.Utf8).map_elements(
        lambda s: args.test_start <= s <= args.test_end
    )
    tr_df = df120.filter(is_train)
    te_df = df120.filter(is_test)

    if tr_df.is_empty():
        raise RuntimeError("学習期間に該当するデータが空です。")
    if te_df.is_empty():
        raise RuntimeError("テスト期間に該当するデータが空です。")

    # ---- 学習 ----
    Xtr, ytr, keytr, feat_cols = build_feature_df(tr_df)
    train_set = lgb.Dataset(Xtr, label=ytr, categorical_feature=[c for c in feat_cols if str(Xtr[c].dtype)=="category"])

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
        lambda_l1=0.0,
        lambda_l2=1.0,
        verbose=-1,
        num_threads=0,
        seed=20240301,
        force_col_wise=True,
    )
    booster = lgb.train(params, train_set, num_boost_round=1000, valid_sets=[train_set], valid_names=["train"], verbose_eval=200)
    booster.save_model(args.model_out)

    # ---- 予測（テスト期間）----
    Xte, yte, keyte, _ = build_feature_df(te_df)
    proba = booster.predict(Xte, num_iteration=booster.best_iteration)

    # DataFrame化
    pred_df = keyte.copy()
    pred_df["proba"] = proba
    pred_df["is_hit"] = yte.values

    # レースごとにTopK選定
    topk = args.topk
    picks = []
    grouped = pred_df.groupby(["hd","jcd","rno"], as_index=False)
    for (hd, jcd, rno), g in grouped:
        g_sorted = g.sort_values("proba", ascending=False)
        sel = g_sorted.head(topk).copy()
        sel["rank"] = range(1, len(sel)+1)
        picks.append(sel)
    picks_df = pd.concat(picks, ignore_index=True)

    # ---- 評価（Hit率 & ROI）----
    # Hit率（レース単位で、TopK内に is_hit==1 が1つでもあれば的中）
    race_hit = picks_df.groupby(["hd","jcd","rno"])["is_hit"].max().reset_index(name="race_hit")
    hit_rate = race_hit["race_hit"].mean()

    # ROI計算:
    # 1レースあたり TopK * unit_stake 円を投資。
    # 的中レースは、そのレースの3連単払い戻し額を獲得（unit_stake=100円を前提に払戻額はそのまま加算）。
    # ※ 払戻は1点的中時の全額（100円基準）。TopK内に一つでも当たりがあればその額を計上。
    #    （複数当たりはありえないので race_hit∈{0,1}）
    unique_races = race_hit[["hd","jcd","rno"]].to_records(index=False)
    returns = 0
    total_bet = len(unique_races) * topk * args.unit_stake

    for hd, jcd, rno in unique_races:
        # 当たりが含まれていれば、そのレースの払い戻し額を加算
        if race_hit[(race_hit["hd"]==hd)&(race_hit["jcd"]==jcd)&(race_hit["rno"]==rno)]["race_hit"].iloc[0] == 1:
            amt = load_results_amount(results_root, hd, jcd, int(rno))
            if amt is not None:
                returns += amt

    roi = returns / total_bet if total_bet > 0 else 0.0

    # ---- 保存 ----
    summary = {
        "train_range": [args.train_start, args.train_end],
        "test_range":  [args.test_start, args.test_end],
        "num_train_rows": int(tr_df.height),
        "num_test_rows": int(te_df.height),
        "num_test_races": int(len(unique_races)),
        "topk": topk,
        "unit_stake": args.unit_stake,
        "hit_rate": float(hit_rate),     # レース的中率（TopK内）
        "returns": int(returns),         # 総回収（円）
        "total_bet": int(total_bet),     # 総投資（円）
        "roi": float(roi),               # 回収率（総回収/総投資）
        "model_path": args.model_out,
        "features_used": feat_cols,
    }

    # per-pick詳細（CSV）
    # 各行: hd,jcd,rno,combo,proba,rank,is_hit
    picks_path = eval_dir / f"picks_{args.test_start}_{args.test_end}_k{topk}.csv"
    picks_df.to_csv(picks_path, index=False, encoding="utf-8")

    # サマリ（JSON）
    summary_path = eval_dir / f"summary_{args.test_start}_{args.test_end}_k{topk}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")

if __name__ == "__main__":
    main()
