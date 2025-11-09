# src/boatpred/eval.py
from __future__ import annotations
import argparse, json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


# =========================
# オッズ/結果 ローダ
# =========================

def _norm_jcd(jcd) -> str:
    try:
        return f"{int(str(jcd)) :02d}"
    except Exception:
        s = str(jcd).strip()
        return s if len(s) == 2 else s.zfill(2)

def _norm_rno(rno) -> int:
    try:
        return int(rno)
    except Exception:
        return int(str(rno).replace("R", "").strip())

def _odds_path(odds_root: Path, hd, jcd, rno) -> Path:
    y, md = str(hd)[:4], str(hd)[4:8]
    return odds_root / y / md / _norm_jcd(jcd) / f"{_norm_rno(rno)}R.json"

def _results_path(results_root: Path, hd, jcd, rno) -> Path:
    y, md = str(hd)[:4], str(hd)[4:8]
    return results_root / y / md / _norm_jcd(jcd) / f"{_norm_rno(rno)}R.json"

def load_odds_map(odds_root: Path, hd, jcd, rno) -> Optional[Dict[str, float]]:
    """
    オッズJSONの多形を吸収して { '1-2-3': 15.1, ... } を返す。
    対応例:
      - {"trifecta": {"1-2-3": 15.1, ...}}
      - {"trifecta": [{"combo":"1-2-3","odds":15.1}, ...]}
      - {"odds":[{"combination":"1-2-3","value":15.1}, ...]}
      - フラット: {"1-2-3": 15.1, ...}
    """
    p = _odds_path(odds_root, hd, jcd, rno)
    if not p.exists():
        return None
    js = json.loads(p.read_text(encoding="utf-8"))

    # 1) dict 形式
    tri = js.get("trifecta")
    if isinstance(tri, dict):
        out = {}
        for k, v in tri.items():
            if v in (None, ""): 
                continue
            try:
                out[str(k)] = float(v)
            except Exception:
                try:
                    out[str(k)] = float(str(v).replace(",", ""))
                except Exception:
                    continue
        return out if out else None

    # 1b) list 形式（例: {"trifecta":[{combo,odds}, ...] }）
    if isinstance(tri, list):
        m = {}
        for row in tri:
            if not isinstance(row, dict):
                continue
            combo = row.get("combo") or row.get("combination") or row.get("key")
            odd   = row.get("odds") or row.get("odd") or row.get("value")
            if combo is None or odd in (None, ""):
                continue
            try:
                m[str(combo)] = float(str(odd).replace(",", ""))
            except Exception:
                continue
        return m if m else None

    # 2) 別キーのリスト
    cand = js.get("odds") or js.get("trifecta_list") or js.get("3t") or js.get("list")
    if isinstance(cand, list):
        m = {}
        for row in cand:
            if not isinstance(row, dict):
                continue
            combo = row.get("combo") or row.get("combination") or row.get("key")
            odd   = row.get("odds") or row.get("odd") or row.get("value")
            if combo is None or odd in (None, ""):
                continue
            try:
                m[str(combo)] = float(str(odd).replace(",", ""))
            except Exception:
                continue
        return m if m else None

    # 3) フラット
    flat = {k: js[k] for k in js.keys() if isinstance(k, str) and "-" in k}
    if flat:
        out = {}
        for k, v in flat.items():
            try:
                out[k] = float(str(v).replace(",", ""))
            except Exception:
                continue
        return out if out else None

    return None

def load_result_trifecta(results_root: Path, hd, jcd, rno) -> Tuple[Optional[str], Optional[int]]:
    """
    結果JSONから 勝ち目(例 '1-2-3') と 三連単配当金額(int円) を返す。
    対応例:
      js['payouts']['trifecta'] に { 'combination' or 'combo' or 'key', 'amount' } がある想定。
    """
    p = _results_path(results_root, hd, jcd, rno)
    if not p.exists():
        return None, None
    js = json.loads(p.read_text(encoding="utf-8"))
    tri = (js.get("payouts") or {}).get("trifecta")
    if not tri:
        return None, None

    # 勝ち目
    comb = tri.get("combination") or tri.get("combo") or tri.get("key")
    if comb is not None:
        comb = str(comb)

    # 金額
    amt = tri.get("amount")
    if amt in (None, ""):
        return comb, None
    try:
        amount = int(str(amt).replace(",", ""))
    except Exception:
        amount = None
    return comb, amount


# =========================
# 評価ロジック
# =========================

@dataclass
class EvalConfig:
    topk: int = 18
    unit_stake: int = 100
    rank_key: str = "proba"   # "proba" | "ev"

    # EV/オッズの扱い
    ev_min: float = 0.0
    ev_drop_if_missing_odds: bool = False
    load_odds_always: bool = True

    # バスケット（TopN集計でのフィルタ）
    synth_odds_min: float = 0.0      # 調和平均の逆数（=合成オッズ）の下限
    min_odds_min: float = 0.0        # TopN内の最小オッズ下限
    ev_basket_min: float = 0.0       # TopN平均EVの下限（EV=proba*odds）
    odds_coverage_min: int = 0       # TopNでoddsが取れている件数下限

    # 絶対ルール（レース単独の形状でフィルタ）
    p1win_max: float = 1.00
    s18_min: float = 0.0
    one_head_ratio_top18_max: float = 1.0
    min_non1_candidates_top18: int = 0

    # 日内上限
    daily_cap: int = 0

def _top18_metrics(g: pd.DataFrame) -> pd.Series:
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

def _basket_stats(topN: pd.DataFrame) -> Dict[str, float]:
    # coverage
    cov = int(topN["odds"].notna().sum()) if "odds" in topN.columns else 0
    if cov > 0:
        v = topN["odds"].dropna().astype(float)
        inv_sum = (1.0 / v).sum()
        synth = float(1.0 / inv_sum) if inv_sum > 0 else np.nan
        min_od = float(v.min())
    else:
        synth = np.nan
        min_od = np.nan
    ev_b = float(topN["ev"].mean(skipna=True)) if "ev" in topN.columns else np.nan
    return {"cov": cov, "synth_odds": synth, "min_odds": min_od, "ev_basket": ev_b}

def evaluate_topN(
    pred_df: pd.DataFrame,                 # columns: hd,jcd,rno,combo,proba  (+任意: is_hit)
    odds_root: Path,
    results_root: Path,
    cfg: EvalConfig,
) -> Tuple[pd.DataFrame, Dict]:
    """
    - pred_df を（必要なら）オッズ付与→EV算出
    - 形状フィルタ（S18等）→バスケットフィルタ（TopNの合成オッズ等）
    - TopN抽出→結果JSONから勝ち目・配当読込→ROI/命中率算出
    戻り値: (picks_df, summary_dict)
    """
    df = pred_df.copy()
    # 基本キーは文字列で統一
    for c in ["hd","jcd","rno"]:
        df[c] = df[c].astype(str)
    df["combo"] = df["combo"].astype(str)

    use_ev = (cfg.rank_key == "ev")
    need_odds = (use_ev or cfg.ev_min > 0 or cfg.ev_drop_if_missing_odds or cfg.load_odds_always
                 or cfg.synth_odds_min > 0 or cfg.min_odds_min > 0 or cfg.ev_basket_min > 0 or cfg.odds_coverage_min > 0)

    # ---- p1win（1頭の総和）----
    p1_df = (
        df[df["combo"].str.startswith("1-")]
        .groupby(["hd","jcd","rno"], as_index=False)["proba"].sum()
        .rename(columns={"proba": "p1win"})
    )
    df = df.merge(p1_df, on=["hd","jcd","rno"], how="left")
    df["p1win"] = df["p1win"].fillna(0.0)

    # ---- odds/EV 付与 ----
    if need_odds:
        df["odds"] = np.nan
        # レース単位でJSONを読みにいく
        for (hd, jcd, rno), idxs in df.groupby(["hd","jcd","rno"]).groups.items():
            omap = load_odds_map(odds_root, hd, jcd, rno)
            if not omap:
                continue
            sub = df.loc[idxs]
            mapped = pd.to_numeric(sub["combo"].map(omap), errors="coerce")
            df.loc[idxs, "odds"] = mapped.to_numpy(dtype=float)
        df["ev"] = df["proba"] * df["odds"]

    # ---- レース形状（Top18ベース）の絶対ルール ----
    metrics = df.groupby(["hd","jcd","rno"]).apply(_top18_metrics).reset_index()
    df = df.merge(metrics, on=["hd","jcd","rno"], how="left")

    cond_shape = (
        (df["p1win"] <= cfg.p1win_max) &
        (df["S18"] >= cfg.s18_min) &
        (df["one_head_ratio_top18"] <= cfg.one_head_ratio_top18_max) &
        (df["non1_candidates_top18"] >= cfg.min_non1_candidates_top18)
    )
    eligible = df.loc[cond_shape, ["hd","jcd","rno","p1win","S18","one_head_ratio_top18"]].drop_duplicates()
    if eligible.empty:
        # 形状で0なら以降は空
        return pd.DataFrame(columns=df.columns.tolist()+["rank"]), {
            "num_test_races": 0, "hit_rate": 0.0, "returns": 0, "total_bet": 0, "roi": 0.0,
            "avg_picks_per_race": 0.0, "avg_p1win": 0.0,
        }

    # ---- バスケット（TopNで判定）----
    if need_odds:
        base_df = df.merge(eligible[["hd","jcd","rno"]], on=["hd","jcd","rno"], how="inner")
        rows = []
        for (hd, jcd, rno), g in base_df.groupby(["hd","jcd","rno"], as_index=False):
            gg = g.copy()
            if use_ev:
                if cfg.ev_drop_if_missing_odds:
                    gg = gg[gg["ev"].notna()].copy()
                gg = gg.sort_values("ev", ascending=False)
                if cfg.ev_min > 0:
                    gg = gg[gg["ev"].fillna(-np.inf) >= cfg.ev_min]
            else:
                gg = gg.sort_values("proba", ascending=False)

            if gg.empty:
                continue
            topN = gg.head(cfg.topk)
            stat = _basket_stats(topN)
            rows.append({"hd":hd,"jcd":jcd,"rno":rno, **stat})

        if rows:
            basket_df = pd.DataFrame(rows)
            cond_b = pd.Series([True]*len(basket_df))
            if cfg.odds_coverage_min > 0:
                cond_b &= (basket_df["cov"] >= int(cfg.odds_coverage_min))
            if cfg.synth_odds_min > 0:
                cond_b &= (basket_df["synth_odds"] >= float(cfg.synth_odds_min))
            if cfg.min_odds_min > 0:
                cond_b &= (basket_df["min_odds"] >= float(cfg.min_odds_min))
            if cfg.ev_basket_min > 0:
                cond_b &= (basket_df["ev_basket"] >= float(cfg.ev_basket_min))

            keep = basket_df.loc[cond_b, ["hd","jcd","rno"]]
            eligible = eligible.merge(keep, on=["hd","jcd","rno"], how="inner")
            if eligible.empty:
                return pd.DataFrame(columns=df.columns.tolist()+["rank"]), {
                    "num_test_races": 0, "hit_rate": 0.0, "returns": 0, "total_bet": 0, "roi": 0.0,
                    "avg_picks_per_race": 0.0, "avg_p1win": 0.0,
                }

    # ---- 日内上限（local scoreで間引き）----
    if cfg.daily_cap and cfg.daily_cap > 0:
        e = eligible.copy()
        e["score"] = (0.45 - e["p1win"]) + 0.5*(1.0 - e["one_head_ratio_top18"]) + 0.3*(e["S18"] - 0.26)
        keep_rows = []
        for _, day_df in e.groupby("hd", as_index=False):
            keep_rows.append(day_df.sort_values("score", ascending=False).head(cfg.daily_cap))
        eligible = pd.concat(keep_rows, ignore_index=True)

    # ---- TopN抽出 ----
    df = df.merge(eligible[["hd","jcd","rno"]], on=["hd","jcd","rno"], how="inner")
    picks = []
    for (hd, jcd, rno), g in df.groupby(["hd","jcd","rno"], as_index=False):
        gg = g.copy()
        if use_ev:
            if cfg.ev_drop_if_missing_odds:
                gg = gg[gg["ev"].notna()].copy()
            gg = gg.sort_values("ev", ascending=False)
            if cfg.ev_min > 0:
                gg = gg[gg["ev"].fillna(-np.inf) >= cfg.ev_min]
        else:
            gg = gg.sort_values("proba", ascending=False)
        if len(gg) > cfg.topk:
            gg = gg.head(cfg.topk)
        if gg.empty:
            continue
        gg = gg.copy()
        gg["rank"] = range(1, len(gg)+1)
        picks.append(gg)

    if not picks:
        return pd.DataFrame(columns=df.columns.tolist()+["rank"]), {
            "num_test_races": 0, "hit_rate": 0.0, "returns": 0, "total_bet": 0, "roi": 0.0,
            "avg_picks_per_race": 0.0, "avg_p1win": 0.0,
        }

    picks_df = pd.concat(picks, ignore_index=True)

    # ---- 勝ち目＆配当を結果から判定 ----
    unique_races = picks_df[["hd","jcd","rno"]].drop_duplicates().to_records(index=False)
    win_map: Dict[Tuple[str,str,str], Tuple[Optional[str], Optional[int]]] = {}
    for hd, jcd, rno in unique_races:
        win_map[(hd,jcd,rno)] = load_result_trifecta(results_root, hd, jcd, rno)

    # is_hit を生成（pred_dfにない場合）
    if "is_hit" not in picks_df.columns:
        hits = []
        for _, row in picks_df.iterrows():
            key = (row["hd"], row["jcd"], row["rno"])
            win_combo, _ = win_map.get(key, (None, None))
            hits.append(1 if (win_combo is not None and str(row["combo"]) == str(win_combo)) else 0)
        picks_df["is_hit"] = hits

    # ROI 計算
    returns = 0
    total_bet = int(len(picks_df) * cfg.unit_stake)
    for key, (_, amount) in win_map.items():
        if amount is None:
            continue
        sub = picks_df[(picks_df["hd"]==key[0]) & (picks_df["jcd"]==key[1]) & (picks_df["rno"]==key[2])]
        if sub["is_hit"].max() == 1:
            returns += int(amount)

    # 命中率（レース単位）
    race_hit = picks_df.groupby(["hd","jcd","rno"])["is_hit"].max().reset_index(name="race_hit")
    hit_rate = float(race_hit["race_hit"].mean()) if not race_hit.empty else 0.0
    num_test_races = int(len(race_hit))
    avg_picks_per_race = float(len(picks_df) / num_test_races) if num_test_races > 0 else 0.0
    avg_p1win = float(eligible["p1win"].mean()) if not eligible.empty else 0.0
    roi = (returns / total_bet) if total_bet > 0 else 0.0

    summary = {
        "num_test_races": num_test_races,
        "topk": int(cfg.topk),
        "unit_stake": int(cfg.unit_stake),
        "rank_key": cfg.rank_key,
        "ev_min": float(cfg.ev_min),
        "ev_drop_if_missing_odds": bool(cfg.ev_drop_if_missing_odds),
        "load_odds_always": bool(cfg.load_odds_always),
        "synth_odds_min": float(cfg.synth_odds_min),
        "min_odds_min": float(cfg.min_odds_min),
        "ev_basket_min": float(cfg.ev_basket_min),
        "odds_coverage_min": int(cfg.odds_coverage_min),
        "p1win_max": float(cfg.p1win_max),
        "s18_min": float(cfg.s18_min),
        "one_head_ratio_top18_max": float(cfg.one_head_ratio_top18_max),
        "min_non1_candidates_top18": int(cfg.min_non1_candidates_top18),
        "daily_cap": int(cfg.daily_cap),
        "hit_rate": float(hit_rate),
        "returns": int(returns),
        "total_bet": int(total_bet),
        "roi": float(roi),
        "avg_picks_per_race": float(avg_picks_per_race),
        "avg_p1win": float(avg_p1win),
    }
    return picks_df, summary


# =========================
# 簡易CLI
# =========================

def _safen(x) -> str:
    try:
        f = float(x)
        return str(f).replace(".", "_")
    except Exception:
        return "na"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="compose_120 の出力CSV (hd,jcd,rno,combo,proba)")
    ap.add_argument("--odds_root", default="public/odds/v1")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--out_dir", default="data/eval")

    # EvalConfig
    ap.add_argument("--topk", type=int, default=18)
    ap.add_argument("--unit_stake", type=int, default=100)
    ap.add_argument("--rank_key", choices=["proba","ev"], default="proba")
    ap.add_argument("--ev_min", type=float, default=0.0)
    ap.add_argument("--ev_drop_if_missing_odds", action="store_true")
    ap.add_argument("--load_odds_always", action="store_true")

    ap.add_argument("--synth_odds_min", type=float, default=0.0)
    ap.add_argument("--min_odds_min", type=float, default=0.0)
    ap.add_argument("--ev_basket_min", type=float, default=0.0)
    ap.add_argument("--odds_coverage_min", type=int, default=0)

    ap.add_argument("--p1win_max", type=float, default=1.0)
    ap.add_argument("--s18_min", type=float, default=0.0)
    ap.add_argument("--one_head_ratio_top18_max", type=float, default=1.0)
    ap.add_argument("--min_non1_candidates_top18", type=int, default=0)

    ap.add_argument("--daily_cap", type=int, default=0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 入力
    pred_df = pd.read_csv(args.pred_csv)
    cfg = EvalConfig(
        topk=args.topk,
        unit_stake=args.unit_stake,
        rank_key=args.rank_key,
        ev_min=args.ev_min,
        ev_drop_if_missing_odds=bool(args.ev_drop_if_missing_odds),
        load_odds_always=bool(args.load_odds_always),
        synth_odds_min=args.synth_odds_min,
        min_odds_min=args.min_odds_min,
        ev_basket_min=args.ev_basket_min,
        odds_coverage_min=args.odds_coverage_min,
        p1win_max=args.p1win_max,
        s18_min=args.s18_min,
        one_head_ratio_top18_max=args.one_head_ratio_top18_max,
        min_non1_candidates_top18=args.min_non1_candidates_top18,
        daily_cap=args.daily_cap,
    )

    picks_df, summary = evaluate_topN(
        pred_df=pred_df,
        odds_root=Path(args.odds_root),
        results_root=Path(args.results_root),
        cfg=cfg,
    )

    # 出力ファイル名
    mode = f"{cfg.rank_key}_evmin{_safen(cfg.ev_min)}_p1{_safen(cfg.p1win_max)}_s18{_safen(cfg.s18_min)}_oh{_safen(cfg.one_head_ratio_top18_max)}"
    if cfg.daily_cap > 0: mode += f"_cap{cfg.daily_cap}"
    if (cfg.rank_key=="ev") or cfg.load_odds_always or cfg.synth_odds_min>0 or cfg.min_odds_min>0 or cfg.ev_basket_min>0:
        mode += "_odds"
    if cfg.synth_odds_min>0 or cfg.min_odds_min>0 or cfg.ev_basket_min>0 or cfg.odds_coverage_min>0:
        mode += f"_sb{_safen(cfg.synth_odds_min)}_mo{_safen(cfg.min_odds_min)}_eb{_safen(cfg.ev_basket_min)}_cov{int(cfg.odds_coverage_min)}"

    # 期間を推定（任意）
    span = ""
    if {"hd"}.issubset(pred_df.columns):
        hds = sorted(map(str, pred_df["hd"].astype(str).unique()))
        if hds:
            span = f"{hds[0]}_{hds[-1]}_"
    picks_path = out_dir / f"picks_{span}k{cfg.topk}_{mode}.csv"
    summary_path = out_dir / f"summary_{span}k{cfg.topk}_{mode}.json"

    if not picks_df.empty:
        picks_df.to_csv(picks_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")


if __name__ == "__main__":
    main()
