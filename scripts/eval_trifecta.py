#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三連単 予測CSV を、public/results の実着順/配当で評価するスクリプト（安定版）

- 入力:
    --proba_csv:  infer_trifecta.py の出力 (例: data/proba/trifecta/2024/0301/trifecta_proba_20240301_20240331.csv)
    --results_root: public/results ルート（省略可・デフォルト public/results）
    --odds_root:    public/odds    ルート（未使用だが将来拡張のため残置）

- 出力:
    data/eval/trifecta/{YYYY}/{MMDD}/eval_{start}_{end}.json にメトリクスを保存
      例のメトリクス: hit率 (top1/6/12/18), ROI (100円/点), 的中数, 投資/回収/レース数 等

- 前提:
    results JSON 例: public/results/2024/0319/01/12R.json
      - meta.date = "YYYYMMDD"
      - meta.jcd  = "01" (ゼロ埋め2)
      - meta.rno  = 12
      - payouts.trifecta.combo = "1-5-4"  があればそれを使用。無ければ results[].rank == "1","2","3" で生成
      - payouts.trifecta.amount = 990  (100円基準払戻)
"""

from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import polars as pl

# -----------------------------
# ユーティリティ
# -----------------------------
def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def parse_range_from_filename(p: Path) -> Tuple[str, str]:
    """
    probaファイル名の末尾から YYYYMMDD_YYYYMMDD を抽出
    例: trifecta_proba_20240301_20240331.csv
    """
    m = re.search(r'(\d{8})_(\d{8})\.csv$', p.name)
    if not m:
        # フォールバック（エラーにはしない）
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
# 正解(的中)読み込み
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

    # 再帰で *.json を舐める
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

            # trifecta combo/payout を優先
            combo = None
            payout = None
            payouts = data.get("payouts", {})
            trifecta = payouts.get("trifecta")
            if isinstance(trifecta, dict):
                combo = trifecta.get("combo") or trifecta.get("comb")
                payout = trifecta.get("amount")

            # ない場合は results[].rank から生成
            if not combo:
                ranks = data.get("results", [])
                top = [r for r in ranks if r.get("rank") in ("1", "2", "3")]
                if len(top) >= 3:
                    # rank=1,2,3 の lane を順に
                    byrank = sorted(top, key=lambda x: int(x["rank"]))
                    lanes = [str(b.get("lane")) for b in byrank[:3]]
                    if all(lanes):
                        combo = "-".join(lanes)
                # payout は不明のまま

            if not combo:
                # 正解として扱えないのでスキップ
                continue

            truth[(hd, jcd, int(rno))] = {
                "combo": combo,
                "payout": payout,  # None の場合あり
            }
        except Exception:
            # 1ファイル壊れていても全体は続行
            continue

    return truth

# -----------------------------
# 評価ロジック
# -----------------------------
def evaluate_trifecta(
    proba_csv: Path,
    truth: Dict[Tuple[str, str, int], Dict],
    top_ks: List[int] = [1, 6, 12, 18],
    unit_stake: int = 100,
) -> Dict:
    """
    予測CSV（1行が ある三連単組の確率）を読み込み、
    レース毎に proba 降順で Top-N を投票したと見なして評価。
    期待列:
        hd (str/int), jcd (str/int), rno (int), combo (str), proba (float)
    ※ combo列名は "combo" or "comb" を検出して自動利用
    """
    if not proba_csv.exists():
        raise FileNotFoundError(f"proba_csv not found: {proba_csv}")

    # CSVロード（大きくてもPolarsでOK）
    df = pl.read_csv(proba_csv, infer_schema_length=10000)

    # 列名の正規化
    cols = {c.lower(): c for c in df.columns}
    # 必須: hd, jcd, rno, combo/comb, proba
    need_num = ["hd", "jcd", "rno", "proba"]
    for k in need_num:
        if k not in cols:
            raise ValueError(f"[FATAL] required column '{k}' not found in {proba_csv.name}")
    combo_col = cols.get("combo") or cols.get("comb")
    if not combo_col:
        raise ValueError(f"[FATAL] required column 'combo' (or 'comb') not found in {proba_csv.name}")

    # 型整形
    df = df.with_columns([
        pl.col(cols["hd"]).cast(pl.Utf8).alias("hd"),
        pl.col(cols["jcd"]).cast(pl.Utf8).alias("jcd"),
        pl.col(cols["rno"]).cast(pl.Int64).alias("rno"),
        pl.col(combo_col).cast(pl.Utf8).alias("combo"),
        pl.col(cols["proba"]).cast(pl.Float64).alias("proba"),
    ]).select(["hd", "jcd", "rno", "combo", "proba"])

    # レース単位にランキング
    grouped = df.sort(["hd", "jcd", "rno", "proba"], descending=[False, False, False, True]).group_by(["hd", "jcd", "rno"])

    # 集計器の入れ物
    stats = {
        "topk": {},
        "races_total": 0,
        "races_scored": 0,
        "unit_stake": unit_stake,
    }
    for k in top_ks:
        stats["topk"][str(k)] = {
            "hits": 0,
            "races": 0,
            "invest": 0,   # 円
            "return": 0,   # 円
            "hit_rate": 0.0,
            "roi": 0.0,
        }

    # レースごとに評価
    races = grouped.agg(
        pl.col("combo").alias("combos"),
        pl.col("proba").alias("probas"),
    ).iter_rows(named=True)

    for row in races:
        hd, jcd, rno = row["hd"], _norm_two(row["jcd"]), int(row["rno"])
        combos: List[str] = list(row["combos"])
        # probas = row["probas"]  # 今は未使用

        # 正解がなければスキップ
        key = (hd, jcd, rno)
        if key not in truth:
            continue

        stats["races_total"] += 1

        gt_combo = truth[key]["combo"]
        payout = truth[key].get("payout")  # None 可

        # Top-N それぞれで評価
        for k in top_ks:
            topN = combos[:k]
            s = stats["topk"][str(k)]
            s["races"] += 1

            # 投資：100円/点 × N点
            invest = unit_stake * len(topN)
            s["invest"] += invest

            if gt_combo in topN:
                s["hits"] += 1
                # 払戻設定：payout が None なら 0（ROIは的中率評価中心）
                if payout is not None:
                    s["return"] += int(payout) * (unit_stake // 100)

        stats["races_scored"] += 1

    # 尤度計算
    for k in top_ks:
        s = stats["topk"][str(k)]
        if s["races"] > 0:
            s["hit_rate"] = s["hits"] / s["races"]
        if s["invest"] > 0:
            s["roi"] = s["return"] / s["invest"]

    return stats

# -----------------------------
# メイン
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

    # 正解集計
    truth = collect_truth(Path(args.results_root), start, end)
    if not truth:
        log(f"[WARN] truth not found in {args.results_root} for {start}..{end}")

    # 評価
    topks = [int(x) for x in str(args.topks).split(",") if x.strip()]
    stats = evaluate_trifecta(
        proba_csv=proba_csv,
        truth=truth,
        top_ks=topks,
        unit_stake=args.unit_stake,
    )

    # 保存
    y, md = (start[:4] if len(start) >= 8 else "0000"), (start[4:8] if len(start) >= 8 else "0000")
    out_path = ensure_parent(Path(args.report_out_root) / y / md / f"eval_{start}_{end}.json")
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    log(f"[WRITE] {out_path}")

if __name__ == "__main__":
    main()
