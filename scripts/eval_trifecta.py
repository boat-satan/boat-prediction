#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三連単の評価（確定版）
- 入力CSV: columns = hd,jcd,rno,combo,proba
- 真値    : public/results/**/<...>.json を総なめして各レースの1-2-3着laneで truth combo を作成
- 出力    : JSONレポート（全体件数, 的中率, TopN 的中, 平均logloss 他）
"""

from __future__ import annotations
import argparse, json, sys, glob
from pathlib import Path
from typing import Dict, Tuple, List
import polars as pl
import numpy as np

def log(m: str): print(m, flush=True)
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

# --------- 真値収集 ---------
def _collect_truth(results_root: Path) -> Dict[Tuple[str,str,int], str]:
    """
    results_root 配下の *.json を全探索し、(hd,jcd,rno) -> "a-b-c" を返す
    """
    res: Dict[Tuple[str,str,int], str] = {}
    # 例: public/results/2024/0301/01/12R.json 以外の階層にも対応
    for fp in glob.glob(str(results_root / "**" / "*.json"), recursive=True):
        try:
            data = json.loads(Path(fp).read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = data.get("meta", {})
        hd = str(meta.get("date") or meta.get("hd") or "")
        jcd = str(meta.get("jcd") or "")
        rno = meta.get("rno")
        if not hd or not jcd or rno is None:
            # 別フォーマット（キーが外にある）を救済
            hd = str(data.get("date") or data.get("hd") or hd)
            jcd = str(data.get("jcd") or jcd)
            rno = data.get("rno") if rno is None else rno
        try:
            rno = int(rno)
        except Exception:
            continue

        results = data.get("results", [])
        lanes = []
        for row in results:
            rk = str(row.get("rank", ""))
            if rk in ("1","2","3"):
                lanes.append( (int(rk), str(row.get("lane"))) )
        if len(lanes) < 3:
            continue
        lanes.sort(key=lambda x: x[0])  # rank順
        combo = "-".join([lanes[0][1], lanes[1][1], lanes[2][1]])
        res[(hd, jcd, rno)] = combo
    return res

# --------- 評価本体 ---------
def evaluate(proba_csv: Path, results_root: Path, report_out: Path|None=None) -> Path:
    df = pl.read_csv(proba_csv)

    need = ["hd","jcd","rno","combo","proba"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"[FATAL] required column '{c}' not found in {proba_csv.name}. found={list(df.columns)}")

    # 型を揃える
    df = (
        df
        .with_columns([
            pl.col("hd").cast(pl.Utf8),
            pl.col("jcd").cast(pl.Utf8),
            pl.col("rno").cast(pl.Int64),
            pl.col("combo").cast(pl.Utf8),
            pl.col("proba").cast(pl.Float64).clip(0.0, 1.0)
        ])
    )

    truth = _collect_truth(results_root)
    if not truth:
        raise ValueError("[FATAL] results から真値が収集できませんでした。public/results のパスや構成を確認してください。")

    # レースごとに topN, 的中, logloss を計算
    stats_rows = []
    grouped = df.group_by(["hd","jcd","rno"], maintain_order=True)
    total = 0
    hit = 0
    top3 = 0
    top6 = 0
    loglosses = []

    for (hd, jcd, rno), g in grouped:
        key = (hd, jcd, int(rno))
        if key not in truth:  # 真値がないレースはスキップ
            continue
        total += 1
        truth_combo = truth[key]
        g = g.sort("proba", descending=True)
        combos = g["combo"].to_list()
        probas = g["proba"].to_numpy()

        # 的中
        is_hit = int(truth_combo == combos[0])
        hit += is_hit

        # TopN
        def in_top(n: int) -> int:
            sl = combos[:min(n, len(combos))]
            return int(truth_combo in sl)

        top3 += in_top(3)
        top6 += in_top(6)

        # logloss（負の対数尤度、真値コンボ確率）
        try:
            idx = combos.index(truth_combo)
            p_true = float(probas[idx])
        except ValueError:
            p_true = 1e-12
        p_true = max(min(p_true, 1-1e-12), 1e-12)
        loglosses.append(-np.log(p_true))

        stats_rows.append({
            "hd": hd, "jcd": jcd, "rno": int(rno),
            "truth": truth_combo,
            "pred_top1": combos[0],
            "pred_top1_proba": float(probas[0]),
            "is_hit": is_hit,
            "in_top3": in_top(3),
            "in_top6": in_top(6),
            "logloss": float(-np.log(p_true)),
        })

    if total == 0:
        raise ValueError("[FATAL] 評価対象レースが0件でした（真値突合に失敗）。")

    acc = hit / total
    acc_top3 = top3 / total
    acc_top6 = top6 / total
    mean_logloss = float(np.mean(loglosses)) if loglosses else None

    report = {
        "proba_csv": str(proba_csv),
        "results_root": str(results_root),
        "total_races": total,
        "hit_rate_top1": acc,
        "hit_rate_top3": acc_top3,
        "hit_rate_top6": acc_top6,
        "mean_logloss": mean_logloss,
    }

    # 詳細CSVも一緒に
    details = pl.DataFrame(stats_rows)

    if report_out is None:
        # data/eval/trifecta/YYYY/MM/eval_{start}_{end}.json を推定
        # proba_csv から範囲抽出（ファイル名規約: ..._YYYYMMDD_YYYYMMDD.csv）
        name = proba_csv.name
        parts = name.replace(".csv","").split("_")
        start, end = parts[-2], parts[-1]
        out_dir = Path("data/eval/trifecta") / start[:4] / start[4:8]
        ensure_parent(out_dir / "x")
        report_out = out_dir / f"eval_{start}_{end}.json"
        details_out = out_dir / f"details_{start}_{end}.csv"
    else:
        ensure_parent(report_out)
        details_out = report_out.with_suffix(".details.csv")

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    details.write_csv(str(details_out))

    log(f"[WRITE] report -> {report_out}")
    log(f"[WRITE] details -> {details_out}")
    return report_out
        

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba_csv", required=True)
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--report_out", default=None)
    args = ap.parse_args()

    report_path = evaluate(Path(args.proba_csv), Path(args.results_root),
                           Path(args.report_out) if args.report_out else None)

if __name__ == "__main__":
    main()
