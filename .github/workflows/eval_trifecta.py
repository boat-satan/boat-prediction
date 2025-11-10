#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, sys, glob
from pathlib import Path
from typing import Dict, Any, Tuple, List

import polars as pl
import numpy as np

RACE_KEY = ["hd","jcd","rno"]

def log(*a): print(*a, flush=True)
def err(*a): print(*a, file=sys.stderr, flush=True)
def ensure_parent(p: Path): p.parent.mkdir(parents=True, exist_ok=True)

def load_proba(csv_path: Path) -> pl.DataFrame:
    df = pl.read_csv(str(csv_path))
    # 型
    for c in ("hd","jcd"):
        if c in df.columns: df = df.with_columns(pl.col(c).cast(pl.Utf8))
    for c in ("rno","a_lane","b_lane","c_lane"):
        if c in df.columns: df = df.with_columns(pl.col(c).cast(pl.Int64))
    return df

def _collect_results_json(results_root: Path, start: str, end: str) -> Dict[Tuple[str,str,int], Tuple[int,int,int]]:
    """
    public/results/**/**/*.json を走査し、(hd,jcd,rno) -> (a_lane,b_lane,c_lane) を返す。
    JSON 仕様:
      {
        "results":[ {"rank":"1","lane":"1",...}, {"rank":"2","lane":"3",...}, {"rank":"3","lane":"4",...}, ... ]
      }
    """
    out: Dict[Tuple[str,str,int], Tuple[int,int,int]] = {}
    globs = [
        str(results_root/"**"/"*.json"),
        str(results_root/"**"|"*/*.json")  # best-effort
    ]
    files = []
    for g in globs:
        files.extend(glob.glob(g, recursive=True))
    for fp in files:
        try:
            import json as _json
            with open(fp,"r",encoding="utf-8") as f:
                data = _json.load(f)
            # hd/jcd/rno をファイル名や中身から推測
            # まず中身に meta があれば優先
            if "meta" in data and all(k in data["meta"] for k in ("date","jcd","rno")):
                hd = str(data["meta"]["date"])
                jcd = str(data["meta"]["jcd"]).zfill(2)
                rno = int(data["meta"]["rno"])
            else:
                # ファイルパスから抽出（YYYYMMDD を含むとして雑に）
                import re
                m = re.search(r"(\d{8}).*[/\\](\d{2}).*[/\\](\d{1,2})", fp)
                if not m: 
                    continue
                hd, jcd, rno = m.group(1), m.group(2), int(m.group(3))
            if not (start <= hd <= end): 
                continue
            lanes = {}
            for r in data.get("results", []):
                rk = int(str(r.get("rank","0")).replace("着","")) if r.get("rank") else None
                ln = int(r.get("lane")) if r.get("lane") else None
                if rk in (1,2,3) and ln:
                    lanes[rk] = ln
            if len(lanes)==3:
                out[(hd,jcd,rno)] = (lanes[1],lanes[2],lanes[3])
        except Exception:
            continue
    return out

def topk_hit_rate(sorted_df: pl.DataFrame, truth_map: Dict[Tuple[str,str,int], Tuple[int,int,int]], K: int) -> Tuple[int,int]:
    hits, total = 0, 0
    for (hd,jcd,rno), g in sorted_df.groupby(RACE_KEY, maintain_order=True):
        key = (hd,jcd,int(rno))
        if key not in truth_map: 
            continue
        a,b,c = truth_map[key]
        total += 1
        topk = g.head(K)
        ok = ((topk["a_lane"]==a) & (topk["b_lane"]==b) & (topk["c_lane"]==c)).any()
        if bool(ok): hits += 1
    return hits, total

def roi_from_odds(sorted_df: pl.DataFrame, truth_map: Dict[Tuple[str,str,int], Tuple[int,int,int]], odds_root: Path, K: int=12) -> Tuple[float,int,int]:
    """
    optional: public/odds/v1/YYYY/MMDD/jcd/{rno}R.json 形式などをできる範囲で拾う（存在すれば）
    期待: json["trifecta"]["odds"]["a-b-c"] = 金額（100円あたりの払戻）
    """
    import json as _json, glob as _glob, re
    bet = 0; ret = 0; races = 0
    # ざっくり，各レースのTopKを均等100円
    for (hd,jcd,rno), g in sorted_df.groupby(RACE_KEY, maintain_order=True):
        key = (hd,jcd,int(rno))
        if key not in truth_map: 
            continue
        a,b,c = truth_map[key]
        # oddsファイル探索
        y,md = hd[:4], hd[4:]
        patterns = [
            str(odds_root/y/md/jcd/f"{rno}R.json"),
            str(odds_root/y/md/jcd/f"{rno}.json"),
            str(odds_root/"**"/y/md/jcd/f"{rno}R.json"),
        ]
        target = None
        for p in patterns:
            got = _glob.glob(p, recursive=True)
            if got:
                target = got[0]; break
        if not target: 
            continue
        try:
            with open(target,"r",encoding="utf-8") as f:
                js = _json.load(f)
        except Exception:
            continue
        # 本命TopK
        topk = g.head(K).to_dict(as_series=False)
        cnt = len(topk["a_lane"])
        if cnt==0: 
            continue
        bet += 100*cnt
        races += 1
        # 的中がTopKにあるか
        for i in range(cnt):
            aa,bb,cc = int(topk["a_lane"][i]), int(topk["b_lane"][i]), int(topk["c_lane"][i])
            key_combo = f"{aa}-{bb}-{cc}"
            # JSON 仕様に合わせて適宜調整（例: js["trifecta"]["odds"][key]）
            try:
                odds = None
                if "trifecta" in js and "odds" in js["trifecta"]:
                    odds = js["trifecta"]["odds"].get(key_combo)
                if odds:
                    if (aa,bb,cc)==(a,b,c):
                        ret += int(odds)
            except Exception:
                pass
    roi = (ret / bet) if bet>0 else 0.0
    return roi, ret, bet

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba_csv", required=True, help="infer_trifecta.py の出力CSV")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--odds_root", default="public/odds/v1", help="任意。存在すればROI算出")
    args = ap.parse_args()

    proba = load_proba(Path(args.proba_csv))
    # 時間範囲をCSV名から推測
    import re, os
    m = re.search(r"trifecta_proba_(\d{8})_(\d{8})\.csv$", os.path.basename(args.proba_csv))
    start, end = (m.group(1), m.group(2)) if m else ("00000000","99999999")

    truth = _collect_results_json(Path(args.results_root), start, end)
    if not truth:
        err("[WARN] 結果JSONが見つからないため、精度算出はスキップします。")
        sys.exit(0)

    # レース内降順
    sorted_df = proba.sort(by=RACE_KEY+["proba_3t"], descending=[False,False,False,True])

    res: Dict[str,Any] = {}
    for k in (1,3,6,12,18):
        hit, tot = topk_hit_rate(sorted_df, truth, k)
        res[f"Top{k}_hit_rate"] = {"hit": hit, "total": tot, "rate": (hit/tot if tot>0 else None)}

    # ROI (あれば)
    roi, ret, bet = roi_from_odds(sorted_df, truth, Path(args.odds_root), K=12)
    if bet>0:
        res["ROI@Top12_even100"] = {"roi": roi, "return": ret, "bet": bet}

    # 出力
    y,md = start[:4], start[4:]
    outdir = Path("data/eval")/f"{start}_{end}"
    ensure_parent(outdir/"_.keep")
    with open(outdir/"summary.json","w",encoding="utf-8") as f:
        json.dump(res,f,ensure_ascii=False,indent=2)
    log(json.dumps(res, ensure_ascii=False, indent=2))
    log(f"[WRITE] {outdir/'summary.json'}")

if __name__ == "__main__":
    main()
