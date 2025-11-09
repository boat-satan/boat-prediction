#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/boatpred/io.py

入出力ユーティリティ:
- shards(学習用120通り特徴)の読み込み (Parquet/CSV/CSV.GZ)
- 期間フィルタで train/test DataFrame を抽出
- オッズJSON/結果JSONの読み込み（フォーマット揺れ対応）
- DataFrameへのオッズ一括マッピング
- 評価出力( picks/summary )の保存

依存: polars, pandas
"""
from __future__ import annotations
from pathlib import Path
import json
from typing import Iterable, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import polars as pl


# ---------- 基本 ----------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------- shards 読み込み ----------
def scan_shards(root: Path | str = "data/shards",
                fname_basenames: Iterable[str] = ("train_120_pro.parquet",
                                                  "train_120_pro.csv",
                                                  "train_120_pro.csv.gz"),
                infer_schema_length: int = 10000) -> pl.LazyFrame:
    """
    data/shards/**/{fname_basenames} を探索し、最初に見つかった拡張子群で LazyFrame を返す
    """
    root = Path(root)
    pq = sorted(root.rglob(fname_basenames[0]))
    csv = sorted(root.rglob(fname_basenames[1]))
    csv_gz = sorted(root.rglob(fname_basenames[2]))

    if pq:
        return pl.scan_parquet([str(p) for p in pq])
    if csv or csv_gz:
        paths = [*(str(p) for p in csv), *(str(p) for p in csv_gz)]
        return pl.scan_csv(paths, infer_schema_length=infer_schema_length)

    raise FileNotFoundError(
        f"No shards under {root}/**/({', '.join(fname_basenames)})"
    )


def split_period(lf: pl.LazyFrame, start: str, end: str) -> pl.DataFrame:
    """
    hd: 'YYYYMMDD' の列に対して期間フィルタ (both inclusive) を適用してcollect
    """
    return (
        lf.with_columns(pl.col("hd").cast(pl.Utf8))
          .filter(pl.col("hd").is_between(pl.lit(start), pl.lit(end), closed="both"))
          .collect(engine="streaming")
    )


# ---------- 結果(配当) 読み込み ----------
def load_result_amount(results_root: Path | str, hd, jcd, rno) -> Optional[int]:
    """
    public/results/YYYY/MMDD/{jcd}/{rno}R.json から三連単配当amountを返す
    """
    results_root = Path(results_root)
    y = str(hd)[:4]
    md = str(hd)[4:8]
    try:
        jcd_str = f"{int(jcd):02d}"
    except Exception:
        jcd_str = str(jcd)
    try:
        rno_int = int(rno)
    except Exception:
        rno_int = int(str(rno).replace("R", ""))

    p = results_root / y / md / jcd_str / f"{rno_int}R.json"
    if not p.exists():
        return None

    js = json.loads(p.read_text(encoding="utf-8"))
    tri = (js.get("payouts") or {}).get("trifecta")
    if not tri:
        return None
    try:
        return int(str(tri.get("amount")).replace(",", ""))
    except Exception:
        return None


# ---------- オッズ 読み込み ----------
def _odds_map_from_json(js: dict) -> Optional[Dict[str, float]]:
    """
    いろいろなフォーマットを許容して combo -> odds(float) の辞書を返す
    サポート:
      1) {"trifecta": {"1-2-3": 15.1, ...}}  (dict形式)
      2) {"trifecta": [{"combo":"1-2-3","odds":15.1}, ...]}  (list形式) ← ユーザーの例
      3) {"odds":[{"combo":"1-2-3","odd":15.1}, ...]} などの list
      4) {"1-2-3": 15.1, ...}  (フラット)
    """
    # 1) trifecta が dict
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
        if out:
            return out

    # 2) trifecta が list
    if isinstance(tri, list):
        out = {}
        for row in tri:
            if not isinstance(row, dict):
                continue
            combo = row.get("combo") or row.get("combination") or row.get("key")
            odd = row.get("odds") or row.get("odd") or row.get("value")
            if combo is None or odd in (None, ""):
                continue
            try:
                out[str(combo)] = float(str(odd).replace(",", ""))
            except Exception:
                continue
        if out:
            return out

    # 3) list in "odds"/"trifecta_list"/"3t"/"list"
    cand = js.get("odds") or js.get("trifecta_list") or js.get("3t") or js.get("list")
    if isinstance(cand, list):
        out = {}
        for row in cand:
            if not isinstance(row, dict):
                continue
            combo = row.get("combo") or row.get("combination") or row.get("key")
            odd = row.get("odds") or row.get("odd") or row.get("value")
            if combo is None or odd in (None, ""):
                continue
            try:
                out[str(combo)] = float(str(odd).replace(",", ""))
            except Exception:
                continue
        if out:
            return out

    # 4) フラット
    flat = {k: js[k] for k in js.keys() if isinstance(k, str) and "-" in k}
    if flat:
        out = {}
        for k, v in flat.items():
            try:
                out[k] = float(str(v).replace(",", ""))
            except Exception:
                continue
        if out:
            return out

    return None


def load_odds_map(odds_root: Path | str, hd, jcd, rno) -> Optional[Dict[str, float]]:
    """
    public/odds/v1/YYYY/MMDD/{jcd}/{rno}R.json から combo->odds を返す
    """
    odds_root = Path(odds_root)
    y = str(hd)[:4]
    md = str(hd)[4:8]
    try:
        jcd_str = f"{int(jcd):02d}"
    except Exception:
        jcd_str = str(jcd)
    try:
        rno_int = int(rno)
    except Exception:
        rno_int = int(str(rno).replace("R", ""))

    p = odds_root / y / md / jcd_str / f"{rno_int}R.json"
    if not p.exists():
        return None

    js = json.loads(p.read_text(encoding="utf-8"))
    return _odds_map_from_json(js)


# ---------- DataFrame へのオッズ一括付与 ----------
def map_odds_for_frame(df: pd.DataFrame,
                       odds_root: Path | str,
                       hd_col: str = "hd",
                       jcd_col: str = "jcd",
                       rno_col: str = "rno",
                       combo_col: str = "combo",
                       out_col: str = "odds") -> pd.DataFrame:
    """
    与えられた df (hd,jcd,rno,combo を含む) に、対応する 'odds' 列を追加して返す。
    取得不能は np.nan。型は float。
    """
    df = df.copy()
    if out_col not in df.columns:
        df[out_col] = np.nan

    # 文字列化（マッピングの鍵）
    df[combo_col] = df[combo_col].astype(str)

    # グループ毎に読み込み・マップ
    for (hd, jcd, rno), idx in df.groupby([hd_col, jcd_col, rno_col]).groups.items():
        omap = load_odds_map(odds_root, hd, jcd, rno)
        if not omap:
            continue
        sub = df.loc[idx]
        mapped = pd.to_numeric(sub[combo_col].map(omap), errors="coerce")
        df.loc[idx, out_col] = mapped.to_numpy(dtype=float)

    return df


# ---------- 評価出力 ----------
def save_picks_and_summary(picks_df: pd.DataFrame,
                           summary: dict,
                           eval_dir: Path | str,
                           picks_filename: str,
                           summary_filename: str) -> Tuple[Path, Path]:
    """
    picksとsummaryを所定の場所に保存し、保存先Pathを返す
    """
    eval_dir = Path(eval_dir)
    ensure_dir(eval_dir)

    picks_path = eval_dir / picks_filename
    picks_df.to_csv(picks_path, index=False, encoding="utf-8")

    summary_path = eval_dir / summary_filename
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return picks_path, summary_path
