#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三連単 予測CSV を、public/results の実着順/配当で評価するスクリプト（堅牢版）

- 入力:
    --proba_csv:  infer_trifecta.py の出力 (例: data/proba/trifecta/2024/0301/trifecta_proba_20240301_20240331.csv)
    --results_root: public/results ルート（省略可・デフォルト public/results）
    --odds_root:    public/odds    ルート（未使用だが将来拡張のため残置）

- 出力:
    data/eval/trifecta/{YYYY}/{MMDD}/eval_{start}_{end}.json
"""

from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import polars as pl


# -----------------------------
# Utils
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def parse_range_from_filename(p: Path) -> tuple[str, str]:
    m = re.search(r'(\d{8})_(\d{8})\.csv$', p.name)
    if not m:
        return ("00000000", "99999999")
    return m.group(1), m.group(2)

def in_range(date8: str, start: str, end: str) -> bool:
    return start <= date8 <= end

def _safe_int(s, default=None) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return default

def _norm_two(s: str) -> str:
    return str(s).zfill(2)


# -----------------------------
# Truth loader
# -----------------------------
def collect_truth(results_root: Path, start: str, end: str) -> Dict[Tuple[str, str, int], Dict]:
    """
    public/results/** から該当期間の正解情報を取得
    key: (hd, jcd, rno)  ※ hdは8桁文字列、jcdはゼロ埋め2、rnoはint
    value: {"combo": "1-5-4", "payout": 990}
    """
    truth: Dict[Tuple[str, str, int], Dict] = {}
    if not results_root.exists():
        log(f"[WARN] results_root not found: {results_root}")
        return truth

    for jf in results_root.rglob("*.json"):
        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            meta = data.get("meta", {})
            hd = str(meta.get("date") or "")
            jcd = _norm_two(meta.get("jcd") or "")
            rno = _safe_int(meta.get("rno"), None)

            if not hd or jcd == "00" or rno is None:
                continue
            if not in_range(hd, start, end):
                continue

            combo = None
            payout = None
            payouts = data.get("payouts", {})
            trifecta = payouts.get("trifecta")
            if isinstance(trifecta, dict):
                combo = trifecta.get("combo") or trifecta.get("comb")
                payout = trifecta.get("amount")

            if not combo:
                ranks = data.get("results", [])
                top = [r for r in ranks if r.get("rank") in ("1", "2", "3")]
                if len(top) >= 3:
                    byrank = sorted(top, key=lambda x: int(x["rank"]))
                    lanes = [str(b.get("lane")) for b in byrank[:3]]
                    if all(lanes):
                        combo = "-".join(lanes)

            if combo:
                truth[(hd, jcd, int(rno))] = {"combo": combo, "payout": payout}
        except Exception:
            continue

    return truth


# -----------------------------
# Column resolver (robust)
# -----------------------------
def _resolve_columns(df: pl.DataFrame) -> tuple[str, str, str, str, str]:
    """
    予測CSVの列名ゆらぎを吸収して (hd, jcd, rno, combo, proba) 名を返す
    - combo は 'combo' / 'comb' のどちらでもOK
    - proba は以下の候補から自動検出
        正確一致: proba, prob, probability, p, score, likelihood, weight
        部分一致: 'proba' を含む列 or 'prob' を含む列（ただし 'proba_nige' 等の明らかな別物を除外）
    """
    original = df.columns
    lower_map = {c.lower(): c for c in original}

    # 必須: hd, jcd, rno
    for need in ("hd", "jcd", "rno"):
        if need not in lower_map:
            raise ValueError(f"[FATAL] required column '{need}' not found in CSV. found={original}")

    # combo
    combo_col = lower_map.get("combo") or lower_map.get("comb")
    if not combo_col:
        raise ValueError(f"[FATAL] required column 'combo' (or 'comb') not found. found={original}")

    # proba
    # 1) exact candidates
    exact_candidates = ["proba", "prob", "probability", "p", "score", "likelihood", "weight"]
    for k in exact_candidates:
        if k in lower_map:
            proba_col = lower_map[k]
            break
    else:
        # 2) fuzzy: contains 'proba' or 'prob'
        fuzzy = [c for c in original if ("proba" in c.lower() or "prob" in c.lower())]
        # 除外: 明らかな別物（例: proba_nige 等の複数決まり手）を雑に除外 -> 'trifecta' を含む or 'combo' と対になる単一列っぽいのを優先
        if len(fuzzy) == 1:
            proba_col = fuzzy[0]
        else:
            # 'trifecta' を含む or 単に一番短い列名を選ぶ
            tri_like = [c for c in fuzzy if "trifecta" in c.lower()]
            if len(tri_like) >= 1:
                proba_col = sorted(tri_like, key=len)[0]
            elif len(fuzzy) >= 1:
                proba_col = sorted(fuzzy, key=len)[0]
            else:
                raise ValueError(f"[FATAL] probability column not found. found={original}")

    return lower_map["hd"], lower_map["jcd"], lower_map["rno"], combo_col, proba_col


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_trifecta(
    proba_csv: Path,
    truth: Dict[Tuple[str, str, int], Dict],
    top_ks: List[int] = [1, 6, 12, 18],
    unit_stake: int = 100,
) -> Dict:
    if not proba_csv.exists():
        raise FileNotFoundError(f"proba_csv not found: {proba_csv}")

    df = pl.read_csv(proba_csv, infer_schema_length=10000)

    # 列解決
    hd_col, jcd_col, rno_col, combo_col, proba_col = _resolve_columns(df)

    # 型整形＆必要列抽出
    df = (
        df
        .with_columns([
            pl.col(hd_col).cast(pl.Utf8).alias("hd"),
            pl.col(jcd_col).cast(pl.Utf8).alias("jcd"),
            pl.col(rno_col).cast(pl.Int64).alias("rno"),
            pl.col(combo_col).cast(pl.Utf8).alias("combo"),
            pl.col(proba_col).cast(pl.Float64).alias("proba"),
        ])
        .select(["hd", "jcd", "rno", "combo", "proba"])
    )

    grouped = df.sort(["hd", "jcd", "rno", "proba"],
                      descending=[False, False, False, True]
                     ).group_by(["hd", "jcd", "rno"])

    stats = {
        "topk": {},
        "races_total": 0,
        "races_scored": 0,
        "unit_stake": unit_stake,
        "columns_used": {
            "hd": hd_col, "jcd": jcd_col, "rno": rno_col,
            "combo": combo_col, "proba": proba_col
        }
    }
    for k in top_ks:
        stats["topk"][str(k)] = {"hits": 0, "races": 0, "invest": 0, "return": 0, "hit_rate": 0.0, "roi": 0.0}

    races = grouped.agg(
        pl.col("combo").alias("combos"),
        pl.col("proba").alias("probas"),
    ).iter_rows(named=True)

    for row in races:
        hd, jcd, rno = row["hd"], _norm_two(row["jcd"]), int(row["rno"])
        combos: List[str] = list(row["combos"])

        key = (hd, jcd, rno)
        if key not in truth:
            continue

        stats["races_total"] += 1
        gt_combo = truth[key]["combo"]
        payout = truth[key].get("payout")

        for k in top_ks:
            topN = combos[:k]
            s = stats["topk"][str(k)]
            s["races"] += 1
            s["invest"] += unit_stake * len(topN)

            if gt_combo in topN:
                s["hits"] += 1
                if payout is not None:
                    s["return"] += int(payout) * (unit_stake // 100)

        stats["races_scored"] += 1

    for k in top_ks:
        s = stats["topk"][str(k)]
        if s["races"] > 0:
            s["hit_rate"] = s["hits"] / s["races"]
        if s["invest"] > 0:
            s["roi"] = s["return"] / s["invest"]

    return stats


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba_csv", required=True, help="三連単予測CSV（infer_trifecta.py の出力）")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--odds_root", default="public/odds")  # いまは未使用
    ap.add_argument("--report_out_root", default="data/eval/trifecta")
    ap.add_argument("--topks", default="1,6,12,18", help="評価するTop-N（カンマ区切り）")
    ap.add_argument("--unit_stake", type=int, default=100)
    args = ap.parse_args()

    proba_csv = Path(args.proba_csv)
    start, end = parse_range_from_filename(proba_csv)
    log(f"[INFO] Using proba_csv: {proba_csv}")
    log(f"[INFO] Date range inferred: {start}..{end}")

    truth = collect_truth(Path(args.results_root), start, end)
    if not truth:
        log(f"[WARN] truth not found in {args.results_root} for {start}..{end}")

    topks = [int(x) for x in str(args.topks).split(",") if x.strip()]
    stats = evaluate_trifecta(
        proba_csv=proba_csv,
        truth=truth,
        top_ks=topks,
        unit_stake=args.unit_stake,
    )

    y, md = (start[:4] if len(start) >= 8 else "0000"), (start[4:8] if len(start) >= 8 else "0000")
    out_path = ensure_parent(Path(args.report_out_root) / y / md / f"eval_{start}_{end}.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log(f"[WRITE] {out_path}")

if __name__ == "__main__":
    main()
