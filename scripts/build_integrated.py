#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
統合テーブル生成 (ステージング + 予測出力を自動検出してjoin)

入力:
  - staging ベース: data/staging/YYYY/MMDD/
      優先順: decision4_train.csv > single_train.csv > st_train.csv
  - 予測:
      ST:         data/proba/st/YYYY/MMDD/st_pred_*.csv
                   期待列: hd,jcd,rno,lane,regno,(pred_st_sec|st_pred_sec)
      決まり手:   data/proba/decision4/YYYY/MMDD/decision4_proba_*.csv
                   期待列: hd,jcd,rno,lane,(racer_id|regno),proba_nige,proba_sashi,proba_makuri,proba_makuri_sashi
      単勝:       data/proba/single/YYYY/MMDD/proba_*.csv
                   期待列: hd,jcd,rno,lane,regno,proba_win

出力:
  data/integrated/YYYY/MMDD/integrated_train.csv

主キー:
  hd(YYYYMMDD), jcd(ゼロ埋め2桁推奨), rno, lane, regno

派生特徴:
  - st_pred_sec            : ST予測（あれば）
  - st_rel_sec             : st_pred_sec - レース内最小
  - st_rank_in_race        : レース内で速い方から順位(1=最速)
  - dash_advantage         : mean(内1-3のst_pred_sec) - mean(外4-6のst_pred_sec)
  - wall_weak_flag         : 1コースのst_pred_secが2コースより0.03以上遅い or (st_rank>3) → 1
"""

from __future__ import annotations
import argparse, glob, sys
from pathlib import Path
from typing import List, Optional

import polars as pl

ID_COLS = ["hd","jcd","rno","lane","regno"]

def log(x: str): print(x, flush=True)
def err(x: str): print(x, file=sys.stderr, flush=True)

def _ymd_dirs(root: Path, d1: str, d2: str) -> list[Path]:
    def to_ord(s: str) -> int:
        return int(s[:4]) * 372 + int(s[4:6]) * 31 + int(s[6:8])
    s_ord, e_ord = to_ord(d1), to_ord(d2)
    out = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        if not ydir.is_dir(): continue
        y = ydir.name
        for md in sorted(ydir.glob("[0-9]"*4)):
            if not md.is_dir(): continue
            hd = y + md.name
            if s_ord <= to_ord(hd) <= e_ord:
                out.append(md)
    return out

def _scan_first_existing(dd: Path, names: list[str]) -> Optional[pl.LazyFrame]:
    """day dirの中で最初に存在したCSVをscan"""
    for nm in names:
        p = dd / nm
        if p.exists():
            return pl.scan_csv(str(p), ignore_errors=True)
    return None

def _normalize_ids(lf: pl.LazyFrame) -> pl.LazyFrame:
    """キー列の型と名前を正規化。racer_id->regno 等も吸収。"""
    schema = lf.collect_schema()
    names = set(schema.names())

    exprs: List[pl.Expr] = []

    # hd/jcdは文字列、rno/laneはInt、regnoは文字列
    if "hd" in names:
        exprs.append(pl.col("hd").cast(pl.Utf8).alias("hd"))
    if "jcd" in names:
        # jcdをゼロ埋め2桁に寄せる（数字でも文字でもOK）
        exprs.append(pl.col("jcd").cast(pl.Utf8).str.zfill(2).alias("jcd"))
    if "rno" in names:
        exprs.append(pl.col("rno").cast(pl.Int64).alias("rno"))
    if "lane" in names:
        exprs.append(pl.col("lane").cast(pl.Int64).alias("lane"))

    # regno/racer_id のどちらかを regno に
    if "regno" in names:
        exprs.append(pl.col("regno").cast(pl.Utf8).alias("regno"))
    elif "racer_id" in names:
        exprs.append(pl.col("racer_id").cast(pl.Utf8).alias("regno"))

    # 残りの列はそのまま
    for c in names - set([e.meta.output_name() for e in exprs if hasattr(e.meta, "output_name")]):
        exprs.append(pl.col(c))

    return lf.select(exprs)

def _pick_base_dayframe(dd: Path) -> Optional[pl.LazyFrame]:
    """ステージングのベース行を決める。優先: decision4_train > single_train > st_train"""
    lf = _scan_first_existing(dd, ["decision4_train.csv", "single_train.csv", "st_train.csv"])
    if lf is None:
        return None
    return _normalize_ids(lf)

def _find_pred_files(pattern_dir: Path, pattern_glob: str) -> list[Path]:
    return [Path(p) for p in glob.glob(str(pattern_dir / pattern_glob))]

def _load_st_pred(day: Path) -> Optional[pl.LazyFrame]:
    # 期待: columns include pred_st_sec or st_pred_sec
    files = _find_pred_files(day, "st_pred_*.csv")
    if not files:
        return None
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in files]
    lf = _normalize_ids(pl.concat(lfs, how="vertical_relaxed"))
    # 列名吸収
    cols = set(lf.collect_schema().names())
    if "pred_st_sec" in cols:
        target = "pred_st_sec"
    elif "st_pred_sec" in cols:
        target = "st_pred_sec"
    else:
        # 名前が想定外なら諦める
        return None
    return lf.select([*(pl.col(c) for c in ID_COLS if c in cols), pl.col(target).alias("st_pred_sec")])

def _load_decision4(day: Path) -> Optional[pl.LazyFrame]:
    files = _find_pred_files(day, "decision4_proba_*.csv")
    if not files:
        return None
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in files]
    lf = _normalize_ids(pl.concat(lfs, how="vertical_relaxed"))
    cols = set(lf.collect_schema().names())
    need = {"proba_nige","proba_sashi","proba_makuri","proba_makuri_sashi"}
    if not need.issubset(cols):
        return None
    keep = [c for c in ID_COLS if c in cols] + sorted(list(need))
    return lf.select([pl.col(c) for c in keep])

def _load_win(day: Path) -> Optional[pl.LazyFrame]:
    files = _find_pred_files(day, "proba_*.csv")
    if not files:
        return None
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in files]
    lf = _normalize_ids(pl.concat(lfs, how="vertical_relaxed"))
    cols = set(lf.collect_schema().names())
    if "proba_win" not in cols:
        return None
    keep = [c for c in ID_COLS if c in cols] + ["proba_win"]
    return lf.select([pl.col(c) for c in keep])

def _add_racewise_features(df: pl.DataFrame) -> pl.DataFrame:
    if "st_pred_sec" not in df.columns:
        # STがない場合は派生を作らずそのまま
        return df

    gb = ["hd","jcd","rno"]
    # レース内最小、順位、内外平均差
    df = df.with_columns([
        pl.col("st_pred_sec").min().over(gb).alias("st_min_in_race"),
        pl.col("st_pred_sec").rank(method="dense", descending=False).over(gb).alias("st_rank_in_race"),
        # 内外平均
        pl.when(pl.col("lane") <= 3).then(pl.col("st_pred_sec")).otherwise(None).mean().over(gb).alias("st_mean_inner"),
        pl.when(pl.col("lane") >= 4).then(pl.col("st_pred_sec")).otherwise(None).mean().over(gb).alias("st_mean_outer"),
    ])
    df = df.with_columns([
        (pl.col("st_pred_sec") - pl.col("st_min_in_race")).alias("st_rel_sec"),
        (pl.col("st_mean_inner") - pl.col("st_mean_outer")).alias("dash_advantage"),
    ])
    # 壁弱判定（簡易）
    # 条件: lane==1 かつ (st_pred_sec - 最小st) > 0.03, または st_rank_in_race > 3
    df = df.with_columns([
        pl.when(
            ((pl.col("lane")==1) & ((pl.col("st_pred_sec") - pl.col("st_min_in_race")) > 0.03))
            | ((pl.col("lane")==1) & (pl.col("st_rank_in_race") > 3))
        ).then(1).otherwise(0).alias("wall_weak_flag")
    ])
    return df.drop(["st_min_in_race","st_mean_inner","st_mean_outer"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--proba_st_root", default="data/proba/st")
    ap.add_argument("--proba_decision_root", default="data/proba/decision4")
    ap.add_argument("--proba_single_root", default="data/proba/single")
    ap.add_argument("--out_root", default="data/integrated")

    ap.add_argument("--start", required=True)  # YYYYMMDD
    ap.add_argument("--end",   required=True)  # YYYYMMDD
    args = ap.parse_args()

    day_dirs = _ymd_dirs(Path(args.staging_root), args.start, args.end)
    if not day_dirs:
        err(f"[FATAL] 対象日無し: {args.start}..{args.end}")
        sys.exit(1)

    total_rows = 0
    for dd in day_dirs:
        hd = dd.parent.name + dd.name  # YYYY + MMDD
        y, md = dd.parent.name, dd.name

        # ベース
        lf_base = _pick_base_dayframe(dd)
        if lf_base is None:
            log(f"[SKIP] base not found: {dd}")
            continue

        base_cols = set(lf_base.collect_schema().names())
        # ID欠けがあれば作れない
        if not {"hd","jcd","rno","lane"}.issubset(base_cols):
            log(f"[SKIP] base missing id cols at {dd}")
            continue

        # regno (or racer_id) が無ければ落ちるので補う努力はせずスキップ
        if not (("regno" in base_cols) or ("racer_id" in base_cols)):
            log(f"[SKIP] base has no regno/racer_id: {dd}")
            continue

        df_base = _normalize_ids(lf_base).collect()
        # ここから join
        # ST
        st_day = Path(args.proba_st_root) / y / md
        dec_day = Path(args.proba_decision_root) / y / md
        win_day = Path(args.proba_single_root) / y / md

        joins: List[pl.DataFrame] = []

        lf_st = _load_st_pred(st_day)
        if lf_st is not None:
            joins.append(lf_st.collect())

        lf_dec = _load_decision4(dec_day)
        if lf_dec is not None:
            joins.append(lf_dec.collect())

        lf_win = _load_win(win_day)
        if lf_win is not None:
            joins.append(lf_win.collect())

        df = df_base
        for jdf in joins:
            # IDキーで左結合
            use_keys = [k for k in ID_COLS if k in df.columns and k in jdf.columns]
            if not use_keys:
                continue
            df = df.join(jdf, on=use_keys, how="left")

        # 派生特徴
        df = _add_racewise_features(df)

        out_dir = Path(args.out_root) / y / md
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "integrated_train.csv"

        # 最後にID優先の列順へ（存在すれば）
        prefer = [c for c in ID_COLS if c in df.columns] + \
                 [c for c in ["st_pred_sec","st_rel_sec","st_rank_in_race","dash_advantage","wall_weak_flag"] if c in df.columns] + \
                 [c for c in ["proba_nige","proba_sashi","proba_makuri","proba_makuri_sashi","proba_win"] if c in df.columns]
        others = [c for c in df.columns if c not in prefer]
        df.select(prefer + others).write_csv(out_path)
        log(f"[WRITE] {out_path} rows={df.height}")
        total_rows += df.height

    log(f"[DONE] days={len(day_dirs)} rows={total_rows}")

if __name__ == "__main__":
    main()
