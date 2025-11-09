#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単勝モデルのランキング精度をまとめて評価するスクリプト
- Top1正解率
- Top2/Top3カバー率
- AUC（全サンプル）
- レーン別 AUC / Top1率

入力:
  --proba_csv: 単一CSV へのパス もしくは CSV を含むディレクトリ
  --results_root: public/results のルート（JSONに1着laneが入っている想定）

出力:
  すべて標準出力（ファイル出力なし）
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Iterable, List

import pandas as pd

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False


def log(msg: str):
    print(msg, flush=True)


def load_proba_tables(proba_csv_or_dir: str) -> pd.DataFrame:
    p = Path(proba_csv_or_dir)
    files: List[Path] = []
    if p.is_file() and p.suffix.lower() == ".csv":
        files = [p]
    elif p.is_dir():
        files = sorted(p.rglob("proba_*.csv"))
    else:
        raise FileNotFoundError(f"proba_csv が見つかりません: {proba_csv_or_dir}")

    if not files:
        raise FileNotFoundError(f"CSVが1件も見つかりません: {proba_csv_or_dir}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # 型/列チェック
            need = {"hd","jcd","rno","lane","regno","proba_win"}
            if not need.issubset(set(df.columns)):
                log(f"[WARN] 列不足のためスキップ: {f}")
                continue
            dfs.append(df)
        except Exception as e:
            log(f"[WARN] 読み込み失敗: {f} ({e})")
    if not dfs:
        raise RuntimeError("有効なproba CSVが読み込めませんでした。")

    out = pd.concat(dfs, axis=0, ignore_index=True)
    # 正規化: 型の揺れを抑える
    out["hd"]  = out["hd"].astype(str)
    out["jcd"] = out["jcd"].astype(str).str.zfill(2)
    out["rno"] = out["rno"].astype(int)
    out["lane"]= out["lane"].astype(int)
    out["proba_win"] = pd.to_numeric(out["proba_win"], errors="coerce").fillna(0.0)
    return out


def build_winner_index(results_root: str) -> Dict[Tuple[str,str,int], int]:
    """(hd, jcd, rno) -> winner lane"""
    idx: Dict[Tuple[str,str,int], int] = {}
    root = Path(results_root)
    for jf in root.rglob("*R.json"):
        try:
            with jf.open("r", encoding="utf-8") as f:
                js = json.load(f)
            meta = js.get("meta", {})
            hd  = str(meta.get("date"))
            jcd = str(meta.get("jcd")).zfill(2)
            rno = int(meta.get("rno"))
            win_lane = None
            for rec in js.get("results", []):
                if str(rec.get("rank")) == "1":
                    win_lane = int(rec.get("lane"))
                    break
            if win_lane is not None:
                idx[(hd, jcd, rno)] = win_lane
        except Exception:
            # スキップ
            continue
    return idx


def compute_topk_cover(df: pd.DataFrame, winners: Dict[Tuple[str,str,int], int], k: int) -> Tuple[int,int,float]:
    total, hit = 0, 0
    for (hd, jcd, rno), grp in df.groupby(["hd","jcd","rno"], sort=False):
        key = (hd, jcd, int(rno))
        win_lane = winners.get(key)
        if win_lane is None:
            continue
        total += 1
        topk = grp.sort_values("proba_win", ascending=False).head(k)["lane"].astype(int).tolist()
        if int(win_lane) in topk:
            hit += 1
    acc = hit / total if total else 0.0
    return hit, total, acc


def compute_auc_overall(df: pd.DataFrame, winners: Dict[Tuple[str,str,int], int]) -> Tuple[float,int]:
    """全サンプル（各レーンを1サンプル）に対してAUC"""
    if not _HAS_SKLEARN:
        return float("nan"), 0
    # ラベル化
    df2 = df.copy()
    df2["key"] = list(zip(df2["hd"], df2["jcd"], df2["rno"]))
    df2["is_win"] = df2.apply(lambda r: 1 if winners.get((r["hd"], r["jcd"], int(r["rno"]))) == int(r["lane"]) else 0, axis=1)
    # 学習可能条件: 正例/負例が両方必要
    pos = df2["is_win"].sum()
    neg = len(df2) - pos
    if pos == 0 or neg == 0:
        return float("nan"), len(df2)
    auc = roc_auc_score(df2["is_win"].values, df2["proba_win"].values)
    return float(auc), len(df2)


def compute_lanewise(df: pd.DataFrame, winners: Dict[Tuple[str,str,int], int]) -> pd.DataFrame:
    """レーン別 AUC / Top1率"""
    rows = []
    for lane in range(1, 7):
        sub = df[df["lane"] == lane].copy()
        if sub.empty:
            rows.append(dict(lane=lane, auc="n/a", top1_rate="n/a", samples=0, races=0))
            continue
        # AUC（このレーンが勝者=1, それ以外=0）
        if _HAS_SKLEARN:
            sub["is_win"] = sub.apply(lambda r: 1 if winners.get((r["hd"], r["jcd"], int(r["rno"]))) == lane else 0, axis=1)
            pos = sub["is_win"].sum()
            neg = len(sub) - pos
            if pos > 0 and neg > 0:
                auc_val = roc_auc_score(sub["is_win"].values, sub["proba_win"].values)
            else:
                auc_val = "n/a"
        else:
            auc_val = "n/a"

        # Top1率（そのレースで lane がTop1かつ実際勝っている割合 ではなく、
        # そのレーンが出走したレースのうち「レース全体でTop1に選ばれ」かつ勝った割合）
        # → 分母は「laneが出たレース数」、分子は「そのレースのTop1 lane==lane かつ laneが実勝」
        #   偏りチェック用の厳しめ指標
        races = 0
        ok = 0
        for (hd, jcd, rno), grp in df.groupby(["hd","jcd","rno"], sort=False):
            # このレースに lane が居なければスキップ
            if lane not in grp["lane"].values:
                continue
            races += 1
            win_lane = winners.get((hd, jcd, int(rno)))
            top1_lane = int(grp.sort_values("proba_win", ascending=False).iloc[0]["lane"])
            if top1_lane == lane and win_lane == lane:
                ok += 1
        top1_rate = (ok / races) if races else "n/a"

        rows.append(dict(
            lane=lane,
            auc=auc_val if isinstance(auc_val, str) else round(float(auc_val), 4),
            top1_rate=top1_rate if isinstance(top1_rate, str) else round(float(top1_rate), 4),
            samples=len(sub),
            races=races,
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba_csv", required=True, help="単一CSV もしくは CSV を含むディレクトリ")
    ap.add_argument("--results_root", default="public/results")
    args = ap.parse_args()

    df = load_proba_tables(args.proba_csv)
    log(f"[INFO] predictions loaded: rows={len(df)}, races={df.groupby(['hd','jcd','rno']).ngroups}")

    winners = build_winner_index(args.results_root)
    log(f"[INFO] winners indexed: {len(winners)}")

    # Top1/Top2/Top3 coverage
    for k in (1, 2, 3):
        hit, total, acc = compute_topk_cover(df, winners, k=k)
        log(f"[RESULT] Top{k} カバー率: {acc:.4f}  ({hit}/{total})")

    # AUC overall
    auc, n = compute_auc_overall(df, winners)
    if auc == auc:  # not NaN
        log(f"[RESULT] 全サンプルAUC: {auc:.4f}  (samples={n})")
    else:
        log(f"[RESULT] 全サンプルAUC: n/a  (正例/負例が片側のみ)")

    # lane-wise
    lane_df = compute_lanewise(df, winners)
    log("\n[LANE-WISE]")
    if lane_df.empty:
        log("no data")
    else:
        # きれいに表示
        disp = lane_df[["lane","auc","top1_rate","samples","races"]].copy()
        log(disp.to_string(index=False))


if __name__ == "__main__":
    main()
