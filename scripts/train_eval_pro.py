#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_pro.py  (全差し替え版 / CSV・Parquet自動対応・一体型)

入出力想定:
  - 入力テーブル: data/train_120_pro.csv(.gz) もしくは data/train_120_pro.parquet
      * 120展開(レース×120組み合わせ)行。is_hit{0/1}、キー列 hd,jcd,rno,combo を含む。
  - 払戻参照: public/results/YYYY/MMDD/JCD/{R}R.json の payouts.trifecta.amount （100円基準）
  - 出力:
      * data/model_lgbm.txt
      * data/eval/picks_{test_start}_{test_end}_k{K}.csv
      * data/eval/summary_{test_start}_{test_end}_k{K}.json

実行例:
  python scripts/train_eval_pro.py \
    --train_start 20240101 --train_end 20240131 \
    --test_start  20240201 --test_end  20240210 \
    --topk 18 --unit_stake 100 \
    --data_dir data \
    --results_dir public/results \
    --csv_source data/train_120_pro.csv
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple

import polars as pl
import pandas as pd
import lightgbm as lgb


# -----------------------
# Utils
# -----------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def load_results_amount(results_root: Path, hd: str, jcd: str, rno: int) -> int | None:
    """そのレースの3連単払い戻し（円, 100円基準）。無ければ None。"""
    p = results_root / hd[:4] / hd[4:8] / jcd / f"{int(rno)}R.json"
    if not p.exists():
        return None
    js = json.loads(p.read_text(encoding="utf-8"))
    tri = (js.get("payouts") or {}).get("trifecta")
    if not tri:
        return None
    amt = tri.get("amount")
    try:
        return int(amt)
    except Exception:
        try:
            return int(str(amt).replace(",", ""))
        except Exception:
            return None

def load_120_table(parquet_path: Path, csv_path: Path | None = None) -> pl.DataFrame:
    """
    120行テーブルを CSV/CSV.GZ/Parquet の順で自動検出して読み込む。
    明示 --csv_source があれば最優先。
    """
    # 明示CSV
    if csv_path is not None:
        if csv_path.exists():
            return pl.read_csv(csv_path)
        # .gz も併せてチェック
        gz = Path(str(csv_path) + ".gz") if not str(csv_path).endswith(".gz") else csv_path
        if gz.exists():
            return pl.read_csv(gz)

    # 自動検出（同ディレクトリ）
    base_dir = parquet_path.parent
    for name in ("train_120_pro.csv", "train_120_pro.csv.gz"):
        p = base_dir / name
        if p.exists():
            return pl.read_csv(p)

    # Parquet
    if parquet_path.exists():
        return pl.read_parquet(parquet_path)

    raise FileNotFoundError(
        "train_120 が見つかりません。以下のいずれかを配置してください:\n"
        f" - {base_dir/'train_120_pro.csv'}\n"
        f" - {base_dir/'train_120_pro.csv.gz'}\n"
        f" - {parquet_path}"
    )

def split_period(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    return df.filter(pl.col("hd").cast(pl.Utf8).is_between(start, end, closed="both"))

def build_feature_df(df120: pl.DataFrame, feat_ref: List[str] | None = None
                     ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, List[str]]:
    """
    Polars→Pandasへ変換し、特徴・ラベル・キーを返す。
    feat_ref を与えた場合は、その列集合に必ず合わせる（欠損列はNaNで補完）。
    """
    pdf = df120.to_pandas()

    keys = ["hd", "jcd", "rno", "combo"]
    if not set(keys).issubset(pdf.columns):
        missing = [c for c in keys if c not in pdf.columns]
        raise ValueError(f"必要キー列が不足: {missing}")

    if "is_hit" not in pdf.columns:
        raise ValueError("is_hit 列が見つかりません（学習・評価に必須）")

    y = pdf["is_hit"].astype(int)
    drop_cols = set(keys + ["is_hit"])
    # 特徴列候補
    if feat_ref is None:
        feat_cols = [c for c in pdf.columns if c not in drop_cols]
    else:
        # 参照の列順に合わせる。無い列はNaNで補完
        for c in feat_ref:
            if c not in pdf.columns:
                pdf[c] = pd.NA
        feat_cols = feat_ref

    X = pdf[feat_cols].copy()
    # object→category（LGBMのcategorical_featureで扱いやすく）
    for c in X.columns:
        if X[c].dtype == "object":
            X[c] = X[c].astype("category")

    return X, y, pdf[keys].copy(), feat_cols


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_start", required=True)  # YYYYMMDD
    ap.add_argument("--train_end",   required=True)
    ap.add_argument("--test_start",  required=True)
    ap.add_argument("--test_end",    required=True)

    ap.add_argument("--topk", type=int, default=18, choices=[12,16,18,24,36])
    ap.add_argument("--unit_stake", type=int, default=100)

    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_dir", default="public/results")
    ap.add_argument("--model_out", default="data/model_lgbm.txt")

    # 入力テーブルの場所（自動検出付き）
    ap.add_argument("--parquet", default="data/train_120_pro.parquet")
    ap.add_argument("--csv_source", default=None, help="CSVまたはCSV.GZのパス（例: data/train_120_pro.csv[.gz]）")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    results_root = Path(args.results_dir)
    eval_dir = data_dir / "eval"
    ensure_dir(eval_dir)

    # 120行テーブル読み込み（CSV優先）
    parquet_path = Path(args.parquet)
    csv_path = Path(args.csv_source) if args.csv_source else None
    df120 = load_120_table(parquet_path, csv_path)

    # 収録日の範囲を表示
    dates = df120.select(pl.col("hd").cast(pl.Utf8)).unique().sort("hd")
    if dates.height:
        print("[INFO] train_120 含有期間:",
              dates["hd"].min(), "〜", dates["hd"].max(),
              f"(ユニーク日数={dates.height})")

    # 期間で分割
    tr_df = split_period(df120, args.train_start, args.train_end)
    te_df = split_period(df120, args.test_start, args.test_end)
    print(f"[INFO] 学習行数={tr_df.height}, 検証行数={te_df.height}")

    if tr_df.is_empty():
        raise RuntimeError("学習期間に該当するデータが空です。指定範囲の hd が train_120 に含まれていません。")
    if te_df.is_empty():
        raise RuntimeError("テスト期間に該当するデータが空です。指定範囲の hd が train_120 に含まれていません。")

    # ---- 学習 ----
    Xtr, ytr, keytr, feat_cols = build_feature_df(tr_df)
    train_set = lgb.Dataset(Xtr, label=ytr,
                            categorical_feature=[c for c in feat_cols if str(Xtr[c].dtype) == "category"])

    params = dict(
        objective="binary", metric="auc",
        learning_rate=0.05, num_leaves=63, max_depth=-1,
        min_data_in_leaf=50, feature_fraction=0.9,
        bagging_fraction=0.9, bagging_freq=1,
        lambda_l1=0.0, lambda_l2=1.0,
        verbose=-1, num_threads=0,
        seed=20240301, force_col_wise=True,
    )
    booster = lgb.train(params, train_set, num_boost_round=1000,
                        valid_sets=[train_set], valid_names=["train"], verbose_eval=200)
    # モデル保存
    ensure_dir(Path(args.model_out).parent)
    booster.save_model(args.model_out)

    # ---- 予測（テスト期間）----
    # 学習時の特徴セットに列合わせ（不足はNaN）
    Xte, yte, keyte, _ = build_feature_df(te_df, feat_ref=feat_cols)
    proba = booster.predict(Xte, num_iteration=booster.best_iteration)

    # DataFrame化
    pred_df = keyte.copy()
    pred_df["proba"] = proba
    pred_df["is_hit"] = yte.values

    # レースごとにTopK選定
    topk = args.topk
    picks = []
    for (hd, jcd, rno), g in pred_df.groupby(["hd", "jcd", "rno"], as_index=False):
        sel = g.sort_values("proba", ascending=False).head(topk).copy()
        sel["rank"] = range(1, len(sel) + 1)
        picks.append(sel)
    picks_df = pd.concat(picks, ignore_index=True) if len(picks) else pd.DataFrame(columns=["hd","jcd","rno","combo","proba","is_hit","rank"])

    # ---- 評価（Hit率 & ROI）----
    # Hit率（レース単位：TopK内に is_hit==1 があれば1）
    if not picks_df.empty:
        race_hit = picks_df.groupby(["hd","jcd","rno"])["is_hit"].max().reset_index(name="race_hit")
    else:
        race_hit = pd.DataFrame(columns=["hd","jcd","rno","race_hit"])
    hit_rate = float(race_hit["race_hit"].mean()) if not race_hit.empty else 0.0

    # ROI:
    # 1レース: TopK * unit_stake を投資。的中レースは results の trifecta.amount を加算（100円基準）
    unique_races = race_hit[["hd","jcd","rno"]].to_records(index=False) if not race_hit.empty else []
    total_bet = len(unique_races) * topk * args.unit_stake
    returns = 0
    for hd, jcd, rno in unique_races:
        if race_hit[(race_hit["hd"]==hd)&(race_hit["jcd"]==jcd)&(race_hit["rno"]==rno)]["race_hit"].iloc[0] == 1:
            amt = load_results_amount(results_root, hd, jcd, int(rno))
            if amt is not None:
                returns += amt
    roi = (returns / total_bet) if total_bet > 0 else 0.0

    # ---- 保存 ----
    ensure_dir(eval_dir)
    picks_path = eval_dir / f"picks_{args.test_start}_{args.test_end}_k{topk}.csv"
    picks_df.to_csv(picks_path, index=False, encoding="utf-8")

    summary = {
        "train_range": [args.train_start, args.train_end],
        "test_range":  [args.test_start, args.test_end],
        "num_train_rows": int(tr_df.height),
        "num_test_rows": int(te_df.height),
        "num_test_races": int(len(unique_races)),
        "topk": topk,
        "unit_stake": args.unit_stake,
        "hit_rate": hit_rate,          # レース的中率（TopK内に当たりが含まれる割合）
        "returns": int(returns),       # 総回収（円, 100円基準）
        "total_bet": int(total_bet),   # 総投資（円）
        "roi": roi,                    # 回収率（総回収/総投資）
        "model_path": args.model_out,
        "features_used": feat_cols,
    }
    summary_path = eval_dir / f"summary_{args.test_start}_{args.test_end}_k{topk}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")
    print(f"[WRITE] {args.model_out}")

if __name__ == "__main__":
    main()
