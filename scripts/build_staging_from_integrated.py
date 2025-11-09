#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_staging_from_integrated.py
- data/shards/YYYY/MMDD/integrated_pro.csv を読み、
  ST学習用のステージングを data/staging/YYYY/MMDD/ に出力する。
- F(フライング)や異常STをランク対象から除外（null化）し、null を巨大数に置換して rank することで
  実質的に「nulls_last」相当の並びを再現。
- リザルト(public/results/YYYY/MMDD/JCD/{rno}R.json)から実STを取得しターゲット列に付与。

出力（1日ごと）:
- st_train.csv[.gz] : hd,jcd,rno,lane,regno,st (+ tenji_st, tenji_rank, st_rank 等)
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import polars as pl


# ------------------------
# ユーティリティ
# ------------------------
def _z2(jcd: str | int) -> str:
    s = str(jcd)
    return s.zfill(2) if len(s) < 2 else s

def _read_json(p: Path) -> Optional[dict]:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _collect_results_st(results_root: Path, keys: List[Tuple[str, str, int]]) -> Dict[Tuple[str, str, int, int], float]:
    """
    リザルトから ST 値を集める。
    return: {(hd,jcd,rno,lane) -> st_float}
    """
    out: Dict[Tuple[str, str, int, int], float] = {}
    for hd, jcd, rno in keys:
        year, md = hd[:4], hd[4:8]
        jcd2 = _z2(jcd)
        rp = results_root / year / md / jcd2 / f"{rno}R.json"
        js = _read_json(rp)
        if not js:
            continue
        start = js.get("start") or []
        for ent in start:
            try:
                lane = int(ent.get("lane"))
                stv = float(ent.get("st"))
            except Exception:
                continue
            out[(hd, jcd2, rno, lane)] = stv
    return out

def _write_csv(df: pl.DataFrame, path: Path, compress: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        p = Path(str(path) + ".gz") if not str(path).endswith(".gz") else path
        df.write_csv(p)
        print(f"[WRITE] {p} (rows={df.height})")
    else:
        df.write_csv(path)
        print(f"[WRITE] {path} (rows={df.height})")


# ------------------------
# ステージング作成
# ------------------------
def process_one_day(integrated_path: Path,
                    results_root: Path,
                    out_day_root: Path,
                    out_format: str,
                    csv_compress: bool):
    """
    1日分の integrated_pro.csv を読み、st_train.* を出力
    """
    print(f"[INFO] reading {integrated_path}")
    df6 = pl.read_csv(integrated_path)

    # ---- 型の正規化 ----
    df6 = df6.with_columns([
        pl.col("hd").cast(pl.Utf8, strict=False),
        pl.col("jcd").cast(pl.Utf8, strict=False),
        pl.col("rno").cast(pl.Int64, strict=False),
        pl.col("lane").cast(pl.Int64, strict=False),
        pl.col("racer_id").cast(pl.Utf8, strict=False),
        pl.col("tenji_st").cast(pl.Float64, strict=False),
        pl.col("tenji_sec").cast(pl.Float64, strict=False),
        pl.col("isF_tenji").cast(pl.Int64, strict=False),
    ])

    # regno 列
    df6 = df6.with_columns([
        pl.col("racer_id").str.strip_chars().cast(pl.Int64, strict=False).alias("regno")
    ])

    # ---- F/異常ST の除外（ランク対象から）----
    df6 = df6.with_columns([
        pl.when(pl.col("isF_tenji") == 1).then(1).otherwise(0).alias("_is_f")
    ])
    # 0.03 未満はFや異常値扱いとしてランクから除外
    df6 = df6.with_columns([
        pl.when(pl.col("_is_f") == 1).then(None)
         .otherwise(
            pl.when(pl.col("tenji_st") < 0.03).then(None).otherwise(pl.col("tenji_st"))
         ).alias("_st_sanit")
    ])

    # ---- ランク付け（nulls_last 相当を擬似的に再現）----
    # null は巨大数へ置換して rank → 実質的に「nullが最後」
    BIG = 1e9
    df6 = df6.with_columns([
        pl.col("tenji_sec").fill_null(BIG).alias("_tenji_sec_rank_src"),
        pl.col("_st_sanit").fill_null(BIG).alias("_st_rank_src"),
    ])
    df6 = df6.with_columns([
        pl.col("_tenji_sec_rank_src").rank(method="dense", descending=False).alias("tenji_rank"),
        pl.col("_st_rank_src").rank(method="dense", descending=False).alias("st_rank"),
    ])

    # ---- リザルトから実STを付与 ----
    keys_df = df6.select(["hd", "jcd", "rno"]).unique()
    keys: List[Tuple[str, str, int]] = [(r[0], r[1], int(r[2])) for r in keys_df.iter_rows()]
    st_map = _collect_results_st(results_root, keys)

    df6 = df6.with_columns([
        pl.col("jcd").map_elements(lambda x: _z2(x), return_dtype=pl.Utf8).alias("jcd2")
    ])

    if st_map:
        st_rows = [{"hd": k[0], "jcd2": k[1], "rno": k[2], "lane": k[3], "st": v} for k, v in st_map.items()]
        st_df = pl.DataFrame(st_rows, schema={"hd": pl.Utf8, "jcd2": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64, "st": pl.Float64})
    else:
        st_df = pl.DataFrame({"hd": [], "jcd2": [], "rno": [], "lane": [], "st": []},
                             schema={"hd": pl.Utf8, "jcd2": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64, "st": pl.Float64})

    joined = df6.join(st_df, on=["hd", "jcd2", "rno", "lane"], how="left")

    # ---- ST学習用の最小カラムに落とす（+参考特徴）----
    st_train = joined.select([
        "hd",
        pl.col("jcd2").alias("jcd"),
        "rno",
        "lane",
        "regno",
        pl.col("st").alias("st"),
        "tenji_st",
        "tenji_rank",
        "st_rank",
        "course_avg_st",
        "course_first_rate",
        "course_3rd_rate",
    ])

    # ---- 出力（YYYY/MMDD）----
    out_day_root.mkdir(parents=True, exist_ok=True)
    if out_format in ("csv", "both"):
        _write_csv(st_train, out_day_root / "st_train.csv", csv_compress)
    if out_format in ("parquet", "both"):
        p = out_day_root / "st_train.parquet"
        st_train.write_parquet(p)
        print(f"[WRITE] {p} (rows={st_train.height})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_root",   default="data/shards")
    ap.add_argument("--results_root",  default="public/results")
    ap.add_argument("--out_root",      default="data/staging")
    ap.add_argument("--out_format",    default="csv", choices=["csv","parquet","both"])
    ap.add_argument("--csv_compress",  action="store_true")
    args = ap.parse_args()

    shards_root  = Path(args.shards_root)
    results_root = Path(args.results_root)
    out_root     = Path(args.out_root)

    if not shards_root.exists():
        raise FileNotFoundError(f"[FATAL] shards_root not found: {shards_root}")

    # data/shards/YYYY/MMDD/integrated_pro.csv
    targets: List[Path] = sorted(shards_root.glob("*/????/integrated_pro.csv"))
    print(f"[INFO] found {len(targets)} integrated_pro.csv files")

    for ipath in targets:
        md = ipath.parent.name   # MMDD
        year = ipath.parent.parent.name  # YYYY
        out_day_root = out_root / year / md
        process_one_day(
            integrated_path=ipath,
            results_root=results_root,
            out_day_root=out_day_root,
            out_format=args.out_format,
            csv_compress=args.csv_compress,
        )

    print("[DONE] staging build finished.")

if __name__ == "__main__":
    main()
