#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_pro.py  —  ML予測 + 参戦ルール + TopN選定 + 実配当ROI評価

更新点:
- EV/オッズ列の初期値を np.nan に統一し、map→to_numeric(coerce) で欠損安全に。
- --load_odds_always を追加（YAML でも指定可）。
- 合成オッズ/バスケット系フィルタに対応:
    * --synth_odds_min        : TopN 合成オッズの下限 (大きいほど「荒れ」)
    * --min_odds_min          : TopN 内の最小オッズの下限
    * --ev_basket_min         : TopN の平均EVの下限 (EV=proba*odds)
    * --odds_coverage_min     : TopN でオッズが取得できている件数の下限
  ※ これらはオッズが無いと判定できないため、必要なときは --load_odds_always を true に。

使い方例:
  python ./scripts/train_eval_pro.py --config data/train_eval_pro.config.yaml
"""
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

    # 1) {"trifecta": {"1-2-3": 15.1, ...}}
    if isinstance(js.get("trifecta"), dict):
        out = {}
        for k, v in js["trifecta"].items():
            if v in (None, ""): continue
            try:
                out[str(k)] = float(v)
            except Exception:
                try:
                    out[str(k)] = float(str(v).replace(",", ""))
                except Exception:
                    continue
        return out if out else None

    # 2) {"odds":[{"combo":"1-2-3","odd":15.1}, ...]} 等
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

    # 3) {"1-2-3": 15.1, ...}
    flat = {k: js[k] for k in js.keys() if "-" in k}
    if flat:
        out = {}
        for k, v in flat.items():
            try:
                out[k] = float(str(v).replace(",", ""))
            except Exception:
                continue
        return out if out else None

    return None

def build_feature_df(df120: pl.DataFrame):
    # pyarrow不要モードでpandasへ
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


# ---------------- CLI & YAML ----------------
def add_cli_and_yaml_args():
    ap = argparse.ArgumentParser()
    # 期間
    ap.add_argument("--train_start")
    ap.add_argument("--train_end")
    ap.add_argument("--test_start")
    ap.add_argument("--test_end")

    # TopN & 購入
    ap.add_argument("--topk", type=int, default=18, choices=[4,6,8,12,16,18,24,30,36,60,120])
    ap.add_argument("--unit_stake", type=int, default=100)

    # データルート
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--results_dir", default="public/results")
    ap.add_argument("--odds_root", default="public/odds/v1")

    # 並べ替え & EV
    ap.add_argument("--rank_key", default="proba", choices=["proba","ev"])
    ap.add_argument("--ev_min", type=float, default=0.0)
    ap.add_argument("--ev_drop_if_missing_odds", action="store_true")

    # オッズ常時読込
    ap.add_argument("--load_odds_always", action="store_true",
                    help="EVを使わなくても全レースでオッズを読み込む（合成オッズ/バスケットEV等の計算用）")

    # レース単体の絶対ルール
    ap.add_argument("--p1win_max", type=float, default=0.40)
    ap.add_argument("--s18_min", type=float, default=0.26)
    ap.add_argument("--one_head_ratio_top18_max", type=float, default=0.55)
    ap.add_argument("--min_non1_candidates_top18", type=int, default=12)

    # 買い方
    ap.add_argument("--exclude_one_head", action="store_true",
                    help="TopN選定前に '1-' を除外（その後TopNを補充）")
    ap.add_argument("--drop_one_after_rank", action="store_true",
                    help="TopN確定後に '1-' を削除し、補充しない（点数圧縮）")

    # 日内上限
    ap.add_argument("--daily_cap", type=int, default=0)

    # バスケット系フィルタ（TopNで評価）
    ap.add_argument("--synth_odds_min", type=float, default=0.0,
                    help="TopN合成オッズ下限（オッズ要; 0で無効）")
    ap.add_argument("--min_odds_min", type=float, default=0.0,
                    help="TopN内の最小オッズ下限（オッズ要; 0で無効）")
    ap.add_argument("--ev_basket_min", type=float, default=0.0,
                    help="TopN平均EV下限（オッズ要; 0で無効）")
    ap.add_argument("--odds_coverage_min", type=int, default=0,
                    help="TopN内のオッズ取得件数下限（0で無効）")

    # モデル出力
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

    # オッズ常時読込
    set_default("load_odds_always", bool(cfg.get("load_odds_always", args.load_odds_always)))

    # 絶対ルール
    filt = cfg.get("filters", {})
    set_default("p1win_max", float(filt.get("p1win_max", args.p1win_max)))
    set_default("s18_min", float(filt.get("s18_min", args.s18_min)))
    set_default("one_head_ratio_top18_max", float(filt.get("one_head_ratio_top18_max", args.one_head_ratio_top18_max)))
    set_default("min_non1_candidates_top18", int(filt.get("min_non1_candidates_top18", args.min_non1_candidates_top18)))
    set_default("exclude_one_head", bool(filt.get("exclude_one_head", getattr(args, "exclude_one_head", False))))
    set_default("drop_one_after_rank", bool(filt.get("drop_one_after_rank", getattr(args, "drop_one_after_rank", False))))
    set_default("daily_cap", int(filt.get("daily_cap", args.daily_cap)))

    # バスケット系（ルート直下に置く想定）
    set_default("synth_odds_min", float(cfg.get("synth_odds_min", args.synth_odds_min)))
    set_default("min_odds_min", float(cfg.get("min_odds_min", args.min_odds_min)))
    set_default("ev_basket_min", float(cfg.get("ev_basket_min", args.ev_basket_min)))
    set_default("odds_coverage_min", int(cfg.get("odds_coverage_min", args.odds_coverage_min)))

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

    # ---- p1win（1号艇単勝率）----
    p1_df = (
        pred_df[pred_df["combo"].astype(str).str.startswith("1-")]
        .groupby(["hd","jcd","rno"], as_index=False)["proba"].sum()
        .rename(columns={"proba":"p1win"})
    )
    pred_df = pred_df.merge(p1_df, on=["hd","jcd","rno"], how="left")
    pred_df["p1win"] = pred_df["p1win"].fillna(0.0)

    # ---- EV/オッズ付与（条件 or 常時）----
    use_ev = (args.rank_key == "ev")
    need_odds = (use_ev or args.ev_min > 0 or args.ev_drop_if_missing_odds or args.load_odds_always
                 or args.synth_odds_min > 0 or args.min_odds_min > 0 or args.ev_basket_min > 0 or args.odds_coverage_min > 0)

    if need_odds:
        pred_df["odds"] = np.nan
        pred_df["combo"] = pred_df["combo"].astype(str)
        odds_root = Path(args.odds_root)
        for (hd, jcd, rno), idxs in pred_df.groupby(["hd","jcd","rno"]).groups.items():
            omap = load_odds_map(odds_root, hd, jcd, rno)
            if not omap:
                continue
            sub = pred_df.loc[idxs]
            mapped = pd.to_numeric(sub["combo"].map(omap), errors="coerce")
            pred_df.loc[idxs, "odds"] = mapped.to_numpy(dtype=float)
        pred_df["ev"] = pred_df["proba"] * pred_df["odds"]

    # ---- レース単体メトリクス（Top18ベース：S18 / 一頭比 / 非1頭本数）----
    def top18_metrics(g: pd.DataFrame):
        g_sorted = g.sort_values("proba", ascending=False).head(18)
        s18 = g_sorted["proba"].sum()
        if s18 <= 0:
            return pd.Series({"S18": 0.0, "one_head_ratio_top18": 0.0, "non1_candidates_top18": 0})
        is_one = g_sorted["combo"].astype(str).str.startswith("1-")
        one_prob = g_sorted.loc[is_one, "proba"].sum()
        non1_candidates = int((~is_one).sum())
        return pd.Series({
            "S18": float(s18),
            "one_head_ratio_top18": float(one_prob / s18) if s18 > 0 else 0.0,
            "non1_candidates_top18": non1_candidates
        })

    metrics = pred_df.groupby(["hd","jcd","rno"]).apply(top18_metrics).reset_index()
    pred_df = pred_df.merge(metrics, on=["hd","jcd","rno"], how="left")

    # ---- 絶対ルールで参戦候補抽出（レース単体で完結）----
    cond = (
        (pred_df["p1win"] <= args.p1win_max) &
        (pred_df["S18"] >= args.s18_min) &
        (pred_df["one_head_ratio_top18"] <= args.one_head_ratio_top18_max) &
        (pred_df["non1_candidates_top18"] >= args.min_non1_candidates_top18)
    )
    eligible = pred_df.loc[cond, ["hd","jcd","rno","p1win","S18","one_head_ratio_top18"]].drop_duplicates()
    if eligible.empty:
        raise RuntimeError("参戦候補が0件。閾値（p1win_max / s18_min / one_head_ratio / 非1頭本数）を緩めてください。")

    # ---- 日内上限（local scoreで間引き）----
    if args.daily_cap and args.daily_cap > 0:
        e = eligible.copy()
        e["score"] = (0.45 - e["p1win"]) + 0.5*(1.0 - e["one_head_ratio_top18"]) + 0.3*(e["S18"] - 0.26)
        keep = []
        for _, day_df in e.groupby("hd", as_index=False):
            keep.append(day_df.sort_values("score", ascending=False).head(args.daily_cap))
        eligible = pd.concat(keep, ignore_index=True)

    # ---- バスケット系フィルタ（TopNで判定）----
    # 並べ替えキー（proba/ev）に基づき TopN を仮選定し、その TopN で合成オッズ等を計算してレース採否を決定
    if need_odds:  # オッズ必須系は、オッズが無い場合はスキップ扱い
        basket_rows = []
        # まず対象レースのみに絞る
        base_df = pred_df.merge(eligible[["hd","jcd","rno"]], on=["hd","jcd","rno"], how="inner")
        for (hd, jcd, rno), g in base_df.groupby(["hd","jcd","rno"], as_index=False):
            gg = g.copy()

            # exclude_one_head はバスケット評価には適用しない（TopNのスコア判定用）
            if use_ev:
                if args.ev_drop_if_missing_odds:
                    gg = gg[gg["ev"].notna()].copy()
                gg = gg.sort_values("ev", ascending=False)
                if args.ev_min > 0:
                    gg = gg[gg["ev"].fillna(-np.inf) >= args.ev_min]
            else:
                gg = gg.sort_values("proba", ascending=False)

            if len(gg) == 0:
                continue

            topN = gg.head(args.topk).copy()

            # 取得済みオッズの件数
            cov = int(topN["odds"].notna().sum()) if "odds" in topN.columns else 0

            # 合成オッズ（調和平均の逆数: 1 / sum(1/odds)）
            valid_odds = topN["odds"].dropna()
            if cov > 0:
                inv_sum = (1.0 / valid_odds).sum()
                synth_odds = float(1.0 / inv_sum) if inv_sum > 0 else np.nan
                min_odds = float(valid_odds.min())
            else:
                synth_odds = np.nan
                min_odds = np.nan

            # EVバスケット平均
            if "ev" in topN.columns:
                ev_basket = float(topN["ev"].mean(skipna=True))
            else:
                ev_basket = np.nan

            basket_rows.append({
                "hd": hd, "jcd": jcd, "rno": rno,
                "cov": cov,
                "synth_odds": synth_odds,
                "min_odds": min_odds,
                "ev_basket": ev_basket,
            })

        if basket_rows:
            basket_df = pd.DataFrame(basket_rows)
            # 閾値を適用（指定が 0/NaN の場合は無効）
            cond_b = pd.Series([True] * len(basket_df))
            if args.odds_coverage_min > 0:
                cond_b &= (basket_df["cov"] >= int(args.odds_coverage_min))
            if args.synth_odds_min > 0:
                cond_b &= (basket_df["synth_odds"] >= float(args.synth_odds_min))
            if args.min_odds_min > 0:
                cond_b &= (basket_df["min_odds"] >= float(args.min_odds_min))
            if args.ev_basket_min > 0:
                cond_b &= (basket_df["ev_basket"] >= float(args.ev_basket_min))

            keep_basket = basket_df.loc[cond_b, ["hd","jcd","rno"]]
            if keep_basket.empty:
                raise RuntimeError("バスケット系フィルタ適用後に参戦候補が0件。合成オッズ/EV/coverageの閾値を緩めてください。")
            # eligible を置き換え
            eligible = eligible.merge(keep_basket, on=["hd","jcd","rno"], how="inner")

    # ---- 参戦レースに限定 ----
    pred_df = pred_df.merge(eligible[["hd","jcd","rno"]], on=["hd","jcd","rno"], how="inner")

    # ---- TopN選定（買い方ロジック）----
    picks = []
    for (hd, jcd, rno), g in pred_df.groupby(["hd","jcd","rno"], as_index=False):
        gg = g.copy()

        # 事前除外（補充あり）
        if args.exclude_one_head and not args.drop_one_after_rank:
            gg = gg[~gg["combo"].astype(str).str.startswith("1-")].copy()

        # 並べ替え
        if use_ev:
            if args.ev_drop_if_missing_odds:
                gg = gg[gg["ev"].notna()].copy()
            gg = gg.sort_values("ev", ascending=False)
            if args.ev_min > 0:
                gg = gg[gg["ev"].fillna(-np.inf) >= args.ev_min]
        else:
            gg = gg.sort_values("proba", ascending=False)

        # TopK 確保
        if len(gg) > args.topk:
            gg = gg.head(args.topk)

        # TopK後に1頭削除（補充なし）
        if args.drop_one_after_rank:
            gg = gg[~gg["combo"].astype(str).str.startswith("1-")].copy()

        if len(gg) == 0:
            continue

        gg = gg.copy()
        gg["rank"] = range(1, len(gg)+1)
        picks.append(gg)

    if len(picks) == 0:
        raise RuntimeError("選定結果がゼロ件。閾値/買い方条件が厳しすぎます。")

    picks_df = pd.concat(picks, ignore_index=True)

    # ---- ヒット率（レース単位）----
    race_hit = picks_df.groupby(["hd","jcd","rno"])["is_hit"].max().reset_index(name="race_hit")
    hit_rate = float(race_hit["race_hit"].mean())

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

    # ---- 追加メトリクス ----
    num_test_races = int(len(unique_races))
    avg_picks_per_race = float(len(picks_df) / num_test_races) if num_test_races > 0 else 0.0
    avg_p1win = float(eligible["p1win"].mean()) if not eligible.empty else 0.0

    # ---- 出力 ----
    mode_tag = f"{args.rank_key}_evmin{args.ev_min}_p1max{args.p1win_max}_s18{args.s18_min}_oh{args.one_head_ratio_top18_max}".replace(".","_")
    if args.exclude_one_head: mode_tag += "_pre_no1"
    if args.drop_one_after_rank: mode_tag += "_post_no1"
    if args.daily_cap and args.daily_cap > 0: mode_tag += f"_cap{args.daily_cap}"
    if need_odds: mode_tag += "_odds"
    if args.synth_odds_min > 0 or args.min_odds_min > 0 or args.ev_basket_min > 0 or args.odds_coverage_min > 0:
        mode_tag += f"_sb{safen(args.synth_odds_min)}_mo{safen(args.min_odds_min)}_eb{safen(args.ev_basket_min)}_cov{int(args.odds_coverage_min)}"

    picks_path = (eval_dir / f"picks_{args.test_start}_{args.test_end}_k{args.topk}_{mode_tag}.csv")
    picks_df.to_csv(picks_path, index=False, encoding="utf-8")

    summary = {
        "train_range":[args.train_start,args.train_end],
        "test_range":[args.test_start,args.test_end],
        "num_train_rows": int(tr_df.height),
        "num_test_rows": int(te_df.height),
        "num_test_races": num_test_races,
        "topk": args.topk,
        "unit_stake": int(args.unit_stake),
        "rank_key": args.rank_key,
        "ev_min": float(args.ev_min),
        "ev_drop_if_missing_odds": bool(args.ev_drop_if_missing_odds),
        "load_odds_always": bool(args.load_odds_always),
        "p1win_max": float(args.p1win_max),
        "s18_min": float(args.s18_min),
        "one_head_ratio_top18_max": float(args.one_head_ratio_top18_max),
        "min_non1_candidates_top18": int(args.min_non1_candidates_top18),
        "exclude_one_head": bool(args.exclude_one_head),
        "drop_one_after_rank": bool(args.drop_one_after_rank),
        "daily_cap": int(args.daily_cap),
        "synth_odds_min": float(args.synth_odds_min),
        "min_odds_min": float(args.min_odds_min),
        "ev_basket_min": float(args.ev_basket_min),
        "odds_coverage_min": int(args.odds_coverage_min),
        "hit_rate": float(hit_rate),
        "returns": int(returns),
        "total_bet": int(total_bet),
        "roi": float(roi),
        "avg_picks_per_race": float(avg_picks_per_race),
        "avg_p1win": float(avg_p1win),
        "model_path": args.model_out,
        "picks_path": str(picks_path),
    }
    summary_path = (eval_dir / f"summary_{args.test_start}_{args.test_end}_k{args.topk}_{mode_tag}.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")


def safen(x):
    try:
        f = float(x)
        return str(f).replace(".", "_")
    except Exception:
        return "na"


if __name__ == "__main__":
    main()
