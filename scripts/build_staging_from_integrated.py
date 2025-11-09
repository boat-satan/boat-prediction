#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_staging_from_integrated.py

目的:
- data/shards/YYYY/MMDD/integrated_pro.csv を読み取り、
  ST 学習用のステージング (st_train.csv) を YYYY/MMDD 配下に生成する。
- 展示STのF/欠測は「小数点以下が3桁以上(例: 0.0001, 0.0013)」のみを欠測扱いとする。
- tenji_rank / st_rank は同一レース内での昇順ランキング。欠測は最下位へ。

入力:
- data/shards/YYYY/MMDD/integrated_pro.csv
- public/results/YYYY/MMDD/**/R.json (各レースの本番STを取得するため)

出力:
- data/staging/YYYY/MMDD/st_train.csv (or .csv.gz)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import re
from typing import Dict, Tuple, List

import polars as pl

# ----------------------------
# 解析・型ユーティリティ
# ----------------------------

RE_TENJI_3_MORE_DEC = re.compile(r"^\d+\.\d{3,}$")

def _ensure_cols(df: pl.DataFrame, cols: List[Tuple[str, pl.DataType]]) -> pl.DataFrame:
    """存在しない列を作成し、指定型にキャストする"""
    for name, dtype in cols:
        if name not in df.columns:
            df = df.with_columns(pl.lit(None).alias(name))
        df = df.with_columns(pl.col(name).cast(dtype, strict=False))
    return df

def _sanitize_tenji_st_expr() -> pl.Expr:
    """
    tenji_stのF/欠測判定: 「小数点以下3桁以上」だけを欠測(None)にする。
    例) 0.0001 / 0.0013 → None, 0.01 / 0.05 / 0.10 → 有効
    """
    s = pl.col("tenji_st").cast(pl.Utf8)
    is_3more = s.str.contains(r"^\d+\.\d{3,}$")
    return (
        pl.when(pl.col("tenji_st").is_null() | is_3more)
        .then(pl.lit(None, dtype=pl.Float64))
        .otherwise(pl.col("tenji_st").cast(pl.Float64, strict=False))
    )

def _rank_dense_asc(expr: pl.Expr) -> pl.Expr:
    """
    欠測を最下位にしたいので、まず欠測を大きな数で埋め、その後 rank(dense)。
    """
    return expr.fill_null(1e12).rank(method="dense", descending=False)

# ----------------------------
# 結果JSONから 本番ST を抽出
# ----------------------------

def _read_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def _collect_st_for_day(results_day_root: Path) -> pl.DataFrame:
    """
    public/results/YYYY/MMDD 配下を走査し、(hd,jcd,rno,lane,st) を返す。
    """
    rows = []
    if not results_day_root.exists():
        return pl.DataFrame({"hd": [], "jcd": [], "rno": [], "lane": [], "st": []})

    for j in results_day_root.glob("**/*.json"):
        try:
            js = _read_json(j)
            meta = js.get("meta", {})
            hd = str(meta.get("date") or "")
            jcd = str(meta.get("jcd") or meta.get("pid") or "")  # 互換
            rno = int(meta.get("rno") or str(j.stem).replace("R", "") or 0)
            start = js.get("start") or []
            for ent in start:
                lane = int(ent.get("lane"))
                st = ent.get("st")
                # st は float想定。文字なら float 化、失敗なら None
                try:
                    st = float(st)
                except Exception:
                    st = None
                rows.append({"hd": hd, "jcd": jcd, "rno": rno, "lane": lane, "st": st})
        except Exception:
            # 壊れたJSONはスキップ
            continue

    if not rows:
        return pl.DataFrame({"hd": [], "jcd": [], "rno": [], "lane": [], "st": []})

    df = pl.DataFrame(rows)
    # 型を固定
    df = df.with_columns([
        pl.col("hd").cast(pl.Utf8, strict=False),
        pl.col("jcd").cast(pl.Utf8, strict=False),
        pl.col("rno").cast(pl.Int64, strict=False),
        pl.col("lane").cast(pl.Int64, strict=False),
        pl.col("st").cast(pl.Float64, strict=False),
    ])
    return df

# ----------------------------
# メイン処理 (1日単位)
# ----------------------------

def process_one_day(
    shards_day_dir: Path,
    results_day_dir: Path,
    out_day_dir: Path,
    out_format: str = "csv",
    csv_compress: bool = False,
):
    """
    1日の integrated_pro.csv を読み、結果STをjoinして st_train.* を出力
    """
    in_csv = shards_day_dir / "integrated_pro.csv"
    if not in_csv.exists():
        print(f"[WARN] not found: {in_csv}")
        return

    print(f"[INFO] reading {in_csv}")
    df6 = pl.read_csv(in_csv, infer_schema_length=10000)

    # 最低限の型・列整形
    need_cols = [
        ("hd", pl.Utf8), ("jcd", pl.Utf8), ("rno", pl.Int64), ("lane", pl.Int64),
        ("racer_id", pl.Utf8),
        ("tenji_sec", pl.Float64), ("tenji_st", pl.Utf8),  # tenji_st は一旦文字扱い → 後で整形
        ("course_avg_st", pl.Float64),
        ("course_first_rate", pl.Float64),
        ("course_3rd_rate", pl.Float64),
    ]
    df6 = _ensure_cols(df6, need_cols)

    # regno を数値に（学習側で使いやすい名前に統一）
    df6 = df6.with_columns([
        pl.col("racer_id").alias("regno")
    ]).with_columns([
        pl.col("regno").cast(pl.Int64, strict=False)
    ])

    # 展示STのF/欠測を除去（= None）→ 0.0001等は欠測扱い
    df6 = df6.with_columns([
        _sanitize_tenji_st_expr().alias("tenji_st_filt")
    ])

    # 同レース内ランク (欠測は最下位に落とす)
    df6 = df6.with_columns([
        _rank_dense_asc(pl.col("tenji_sec")).over(["hd", "jcd", "rno"]).alias("tenji_rank"),
        _rank_dense_asc(pl.col("tenji_st_filt")).over(["hd", "jcd", "rno"]).alias("st_rank"),
    ])

    # 本番STの取得 (public/results)
    st_day = _collect_st_for_day(results_day_dir)
    # join キーの型を合わせる
    df6 = df6.with_columns([
        pl.col("hd").cast(pl.Utf8, strict=False),
        pl.col("jcd").cast(pl.Utf8, strict=False),
        pl.col("rno").cast(pl.Int64, strict=False),
        pl.col("lane").cast(pl.Int64, strict=False),
    ])
    st_day = st_day.with_columns([
        pl.col("hd").cast(pl.Utf8, strict=False),
        pl.col("jcd").cast(pl.Utf8, strict=False),
        pl.col("rno").cast(pl.Int64, strict=False),
        pl.col("lane").cast(pl.Int64, strict=False),
    ])

    df_join = df6.join(
        st_day,
        on=["hd", "jcd", "rno", "lane"],
        how="left",
        coalesce=True,
    ).rename({"st": "st_race"})

    # 出力ディレクトリ (YYYY/MMDD)
    out_day_dir.mkdir(parents=True, exist_ok=True)

    # --- ST学習用 ---
    st_train_cols = [
        "hd", "jcd", "rno", "lane", "regno",
        "st_race",                   # 目的変数
        "tenji_st_filt", "tenji_rank", "st_rank",
        "course_avg_st", "course_first_rate", "course_3rd_rate",
    ]
    st_train = df_join.select([c for c in st_train_cols if c in df_join.columns]) \
                      .rename({"st_race": "st", "tenji_st_filt": "tenji_st"})

    # 保存
    if out_format in ("csv", "both"):
        out_csv = out_day_dir / "st_train.csv"
        if csv_compress:
            out_csv = Path(str(out_csv) + ".gz")
            st_train.write_csv(out_csv)
        else:
            st_train.write_csv(out_csv)
        print(f"[WRITE] {out_csv} (rows={st_train.height})")

    if out_format in ("parquet", "both"):
        out_parquet = out_day_dir / "st_train.parquet"
        st_train.write_parquet(out_parquet)
        print(f"[WRITE] {out_parquet} (rows={st_train.height})")

# ----------------------------
# 全体制御
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_root", default="data/shards", help="入力: integrated_pro.csv があるルート")
    ap.add_argument("--results_root", default="public/results", help="入力: 本番結果(JSON)ルート")
    ap.add_argument("--out_root", default="data/staging", help="出力: ステージングルート")
    ap.add_argument("--out_format", default="csv", choices=["csv", "parquet", "both"])
    ap.add_argument("--csv_compress", action="store_true")
    args = ap.parse_args()

    shards_root = Path(args.shards_root)
    results_root = Path(args.results_root)
    out_root = Path(args.out_root)

    # data/shards/**/integrated_pro.csv を列挙
    files = sorted(shards_root.glob("**/integrated_pro.csv"))
    print(f"[INFO] found {len(files)} integrated_pro.csv files")

    for f in files:
        # f: data/shards/YYYY/MMDD/integrated_pro.csv を想定
        try:
            parts = f.parts
            # .../shards/YYYY/MMDD/integrated_pro.csv
            year = parts[-3]
            md   = parts[-2]
            # 形式チェック (YYYY, MMDD)
            if not (len(year) == 4 and year.isdigit() and len(md) == 4 and md.isdigit()):
                print(f"[WARN] skip (unexpected path shape): {f}")
                continue

            shards_day_dir  = shards_root / year / md
            results_day_dir = results_root / year / md
            out_day_dir     = out_root / year / md  # ← YYYY/MMDD で保存

            process_one_day(
                shards_day_dir=shards_day_dir,
                results_day_dir=results_day_dir,
                out_day_dir=out_day_dir,
                out_format=args.out_format,
                csv_compress=args.csv_compress,
            )
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

if __name__ == "__main__":
    main()
