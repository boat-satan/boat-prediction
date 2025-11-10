#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三連単 確率CSVの検証（TopN的中率/ROI/ログロス）
- 入力その1: 直接CSV指定 (--proba_csv)
  例: data/proba/trifecta/2024/0301/trifecta_proba_20240301_20240331.csv
  必須列: hd,jcd,rno,combo,proba

- 入力その2: 範囲＋ルート指定 (--proba_root, --pred_start, --pred_end)
  例: proba_root=data/proba/trifecta, start=20240301, end=20240331
      → data/proba/trifecta/2024/0301/trifecta_proba_20240301_20240331.csv

- 正解: public/results/YYYY/PPDD/PP/RR.json （既存構成）
  trifectaオッズは payouts.trifecta.amount を使用（なければ金額は None）

- 出力: data/eval/trifecta/YYYY/MMDD/trifecta_eval_{start}_{end}.json
"""

from __future__ import annotations
import argparse, json, math, sys, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import polars as pl

def log(m: str): print(m, flush=True)
def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

# ---------- helpers ----------
def _infer_range_from_filename(csv_path: Path) -> Tuple[str,str]:
    # ...trifecta_proba_YYYYMMDD_YYYYMMDD.csv
    m = re.search(r"trifecta_proba_(\d{8})_(\d{8})\.csv$", csv_path.name)
    if not m:
        raise ValueError(f"[FATAL] cannot infer date range from filename: {csv_path.name}")
    return m.group(1), m.group(2)

def _dlist(start: str, end: str) -> List[str]:
    sy, sm, sd = int(start[:4]), int(start[4:6]), int(start[6:8])
    ey, em, ed = int(end[:4]), int(end[4:6]), int(end[6:8])
    s_ord = sy*372 + sm*31 + sd
    e_ord = ey*372 + em*31 + ed
    out = []
    y = sy
    m = sm
    d = sd
    for ordv in range(s_ord, e_ord+1):
        out.append(f"{y:04d}{m:02d}{d:02d}")
        # 擬似進行
        d += 1
        if d > 31:
            d = 1; m += 1
        if m > 12:
            m = 1; y += 1
    return out

def _read_proba_csv(proba_csv: Path) -> pl.DataFrame:
    if not proba_csv.exists():
        raise SystemExit(f"[FATAL] proba_csv not found: {proba_csv}")
    df = pl.read_csv(str(proba_csv), infer_schema_length=0)
    need = ["hd","jcd","rno","combo","proba"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"[FATAL] required column '{c}' not found in {proba_csv.name}. found={list(df.columns)}")
    # 型整備
    df = df.with_columns([
        pl.col("hd").cast(pl.Utf8),
        pl.col("jcd").cast(pl.Utf8),
        pl.col("rno").cast(pl.Int64),
        pl.col("combo").cast(pl.Utf8),
        pl.col("proba").cast(pl.Float64),
    ])
    return df

def _truth_combo_amount(results_root: Path, hd: str, jcd: str, rno: int) -> Tuple[Optional[str], Optional[float]]:
    # public/results/YYYY/PPDD/PP/RR.json
    y = hd[:4]; md = hd[4:8]; pp = jcd
    path = results_root / y / md / pp / f"{rno}R.json"
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None, None

    # 正解 combo
    trifecta = (data.get("payouts") or {}).get("trifecta")
    combo = None
    amount = None
    if isinstance(trifecta, dict):
        combo = trifecta.get("combo")
        amt = trifecta.get("amount")
        if isinstance(amt, (int, float)):
            amount = float(amt)

    # combo が無い場合は results[].rank から組み立て（1-2-3着の lane ）
    if not combo:
        results = data.get("results") or []
        # rank "1","2","3" の lane を拾う
        top = {}
        for row in results:
            r = str(row.get("rank","")).strip()
            if r in ("1","2","3"):
                ln = str(row.get("lane","")).strip()
                if ln:
                    top[int(r)] = ln
        if len(top) == 3:
            combo = f"{top[1]}-{top[2]}-{top[3]}"

    return combo, amount

def _neg_logloss(p: float) -> float:
    p = min(max(p, 1e-15), 1.0)
    return -math.log(p)

def evaluate_trifecta(proba_csv: Path, results_root: Path, report_out_root: Path) -> Path:
    df = _read_proba_csv(proba_csv)
    start, end = _infer_range_from_filename(proba_csv)
    log(f"[INFO] Date range inferred: {start}..{end}")

    # レースごとに結合しやすい形へ
    # truth 辞書: (hd,jcd,rno) -> (combo, amount)
    truth: Dict[Tuple[str,str,int], Tuple[Optional[str], Optional[float]]] = {}
    for d in _dlist(start, end):
        # jcd/rno は proba 側から辿るため、ここでは日付だけ進める
        pass

    # proba 側のユニークキーで走査しつつ truth を引く
    keys = df.select(["hd","jcd","rno"]).unique().iter_rows()
    for hd, jcd, rno in keys:
        c, amt = _truth_combo_amount(results_root, hd, jcd, int(rno))
        truth[(hd, jcd, int(rno))] = (c, amt)

    # 評価
    TOPNS = [1, 3, 6, 12, 18, 36, 120]
    cnt = {n: 0 for n in TOPNS}
    tot = 0
    logloss_sum = 0.0
    roi_bet_sum = {n: 0.0 for n in TOPNS}
    roi_ret_sum = {n: 0.0 for n in TOPNS}

    for (hd, jcd, rno), g in df.group_by(["hd","jcd","rno"], maintain_order=True):
        g = g.sort("proba", descending=True)
        t_combo, t_amount = truth.get((hd, jcd, int(rno)), (None, None))
        if t_combo is None:
            # 正解不明のレースは評価から除外
            continue
        tot += 1

        # 正解確率
        p_true_row = g.filter(pl.col("combo") == t_combo)
        if p_true_row.height == 0:
            p_true = 1e-15
        else:
            p_true = float(p_true_row["proba"][0])
        logloss_sum += _neg_logloss(p_true)

        # TopN 的中/ROI（均等BET=1）
        for N in TOPNS:
            topN = g.head(N)
            roi_bet_sum[N] += float(topN.height)  # N 口
            if t_combo in set(topN["combo"].to_list()):
                cnt[N] += 1
                if t_amount is not None:
                    # 払戻は「100円あたり金額」を想定 → 1口=100でベットしている前提なら amount をそのまま加算
                    roi_ret_sum[N] += float(t_amount)

    if tot == 0:
        raise SystemExit("[FATAL] no races to evaluate (truth not found).")

    # まとめ
    report = {
        "range": [start, end],
        "races_evaluated": tot,
        "logloss": logloss_sum / tot,
        "topN": [
            {
                "N": N,
                "hit": cnt[N],
                "rate": cnt[N] / tot,
                "bets": roi_bet_sum[N],
                "returns": roi_ret_sum[N],
                "roi": (roi_ret_sum[N] / roi_bet_sum[N]) if roi_bet_sum[N] > 0 else None
            }
            for N in TOPNS
        ]
    }

    # 保存
    y, md = start[:4], start[4:8]
    out = report_out_root / y / md / f"trifecta_eval_{start}_{end}.json"
    ensure_parent(out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"[WRITE] {out}")
    return out

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # 新: 直接CSV指定
    ap.add_argument("--proba_csv", default=None)
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--report_out_root", default="data/eval/trifecta")
    # 旧インターフェイス互換
    ap.add_argument("--proba_root", default="data/proba/trifecta")
    ap.add_argument("--pred_start", default=None)
    ap.add_argument("--pred_end", default=None)
    args = ap.parse_args()

    # 入力解決
    if args.proba_csv:
        proba_csv = Path(args.proba_csv)
    else:
        if not (args.pred_start and args.pred_end):
            ap.error("either --proba_csv or (--proba_root & --pred_start & --pred_end) is required")
        y, md = args.pred_start[:4], args.pred_start[4:8]
        proba_csv = Path(args.proba_root) / y / md / f"trifecta_proba_{args.pred_start}_{args.pred_end}.csv"

    log(f"[INFO] Using proba_csv: {proba_csv}")
    evaluate_trifecta(proba_csv, Path(args.results_root), Path(args.report_out_root))

if __name__ == "__main__":
    main()
