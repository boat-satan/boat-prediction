#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_staging_from_integrated.py
- data/shards/YYYY/MMDD/integrated_pro.csv を読んで ST 学習用のステージングを作る
- public/results/YYYY/MMDD/{jcd}/{rno}R.json から公式ST/決まり手も吸い上げ
- 出力: data/staging/YYYY/MMDD/st_train.csv  （CSV or gzip, 1行1艇）
    列:
      hd,jcd,rno,lane,regno,
      st_sec,st_is_f,st_is_late,st_penalized,st_observed,
      tenji_st_sec,tenji_is_f,tenji_f_over_sec,
      tenji_rank,st_rank,
      course_avg_st,course_first_rate,course_3rd_rate
- 仕様:
  * 展示STが 0.000x の “ダミーF表現”（例 F.04→0.0004）を再変換してフラグ化。
  * 展示ランクは F を最下位、欠損も最下位で dense-rank（1が最良）。
  * 出力ディレクトリは必ず data/staging/YYYY/MMDD/
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import polars as pl

# ---------------------
# ユーティリティ
# ---------------------
def z2(x: str | int) -> str:
    s = str(x)
    return s if len(s) >= 2 else s.zfill(2)

def read_integrated_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        infer_schema_length=2000,
        null_values=["", "null", "None"],
    )
    # 必須列が無ければエラー
    need = [
        "hd","jcd","rno","lane","racer_id",
        "tenji_st","st_rank","course_avg_st","course_first_rate","course_3rd_rate"
    ]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"[FATAL] {path}: missing column '{c}'")
    # 型と整形
    df = df.with_columns([
        pl.col("hd").cast(pl.Utf8, strict=False),
        pl.col("jcd").cast(pl.Utf8, strict=False),
        pl.col("rno").cast(pl.Int64, strict=False),
        pl.col("lane").cast(pl.Int64, strict=False),
        pl.col("racer_id").alias("regno").cast(pl.Int64, strict=False),

        pl.col("tenji_st").cast(pl.Float64, strict=False).alias("tenji_st_raw"),
        pl.col("st_rank").cast(pl.Int64, strict=False),
        pl.col("course_avg_st").cast(pl.Float64, strict=False),
        pl.col("course_first_rate").cast(pl.Float64, strict=False),
        pl.col("course_3rd_rate").cast(pl.Float64, strict=False),
    ])
    return df

def parse_official_st_text(st: Optional[str]) -> Tuple[Optional[float], int, int, int, int]:
    """
    公式結果JSONの start[*].st 文字列を解釈:
      "0.05" -> (0.05,0,0,0,1)
      "F.04" -> (None,1,0,1,1)  # over秒はここでは保持しない（学習には不要）
      "L.02" -> (None,0,1,1,1)
      None/"" -> (None,0,0,0,0)
    """
    if not st:
        return (None, 0, 0, 0, 0)
    s = st.strip().upper()
    if s.startswith("F."):
        return (None, 1, 0, 1, 1)
    if s.startswith("L."):
        return (None, 0, 1, 1, 1)
    # 通常 0.XX
    try:
        return (float(s), 0, 0, 0, 1)
    except Exception:
        return (None, 0, 0, 0, 0)

def decode_tenji_st(val: Optional[float]) -> Tuple[Optional[float], int, Optional[float]]:
    """
    integrated_pro の展示STは、通常は 0.05 のような秒。
    F の場合は 0.0004 (F.04) のようにエンコードされている前提。
    ここで F を検出してフラグ化し、over秒(0.04)を算出。
    戻り値: (tenji_st_sec, tenji_is_f, tenji_f_over_sec)
      - F でない通常値→ (そのまま, 0, None)
      - F ダミー(0.000x) → (None, 1, x/100 のover秒 = 0.000x*100)
      - 欠損 → (None, 0, None)
    """
    if val is None:
        return (None, 0, None)
    if val < 0.001:  # 0.0009以下は F のダミー
        over = round(val * 100.0, 3)  # 0.0004 -> 0.04
        return (None, 1, over)
    return (float(val), 0, None)

def tenji_dense_rank(group: pl.DataFrame) -> pl.Series:
    """
    展示ランク: 値の小さい方が上位。
    ただし F(tenji_is_f=1) と 欠損 は最下位扱いにするため、ソートキーを工夫。
      key = 
        - 通常: tenji_st_sec
        - F or 欠損: 大きな値 (1e9)
    """
    key = []
    for is_f, sec in zip(group["tenji_is_f"], group["tenji_st_sec"]):
        if is_f == 1 or sec is None:
            key.append(1e9)
        else:
            key.append(float(sec))
    # 昇順で順位（1..n）、同値は同順位(dense-rank)
    order = pl.Series(key).rank(method="dense", descending=False)
    return order

def load_official_results(results_root: Path, hd: str, jcd: str, rno: int) -> Dict[int, Tuple[Optional[float], int, int, int, int]]:
    """
    公式結果JSONから lane -> (st_sec,is_f,is_late,penalized,observed) を返す
    なければ空dict
    """
    year = hd[:4]
    md = hd[4:8]
    j = z2(jcd)
    p = results_root / year / md / j / f"{rno}R.json"
    if not p.exists():
        return {}
    try:
        js = json.loads(p.read_text(encoding="utf-8"))
        arr = js.get("start") or []
        out: Dict[int, Tuple[Optional[float], int, int, int, int]] = {}
        for it in arr:
            lane = int(it.get("lane"))
            sttxt = it.get("st")
            out[lane] = parse_official_st_text(sttxt)
        return out
    except Exception:
        return {}

# ---------------------
# メイン処理
# ---------------------
def process_one_day(integrated_path: Path, results_root: Path, out_root: Path, out_format: str = "csv", csv_compress: bool = False) -> None:
    # 入力読み込み
    df = read_integrated_csv(integrated_path)

    # パスから日付フォルダを復元
    # .../data/shards/YYYY/MMDD/integrated_pro.csv
    year = integrated_path.parent.parent.name    # YYYY
    md   = integrated_path.parent.name           # MMDD
    day_out = out_root / year / md
    day_out.mkdir(parents=True, exist_ok=True)

    # 展示STの再解釈（Fダミー→フラグ化）
    df = df.with_columns([
        pl.col("tenji_st_raw").map_elements(lambda v: decode_tenji_st(v)[0]).alias("tenji_st_sec"),
        pl.col("tenji_st_raw").map_elements(lambda v: decode_tenji_st(v)[1]).alias("tenji_is_f"),
        pl.col("tenji_st_raw").map_elements(lambda v: decode_tenji_st(v)[2]).alias("tenji_f_over_sec"),
    ])

    # 公式STの取り込み（レース毎）
    # 先にキーだけ抽出
    key_cols = ["hd", "jcd", "rno", "lane", "regno",
                "tenji_st_sec", "tenji_is_f", "tenji_f_over_sec",
                "st_rank", "course_avg_st", "course_first_rate", "course_3rd_rate"]
    use = df.select([c for c in key_cols if c in df.columns]).with_columns([
        pl.col("hd").cast(pl.Utf8),
        pl.col("jcd").cast(pl.Utf8),
        pl.col("rno").cast(pl.Int64),
        pl.col("lane").cast(pl.Int64),
        pl.col("regno").cast(pl.Int64),
    ])

    # 公式STを列として埋める
    # collect rows, enrich per race
    rows = []
    for (hd, jcd, rno), g in use.group_by(["hd", "jcd", "rno"], maintain_order=True):
        st_map = load_official_results(results_root, hd, jcd, int(rno))
        # 展示ランクをこのグループ内で再算定（F/欠損を最下位）
        g = g.with_columns([
            tenji_dense_rank(g).alias("tenji_rank")
        ])
        for rec in g.iter_rows(named=True):
            lane = int(rec["lane"])
            st_tuple = st_map.get(lane, (None, 0, 0, 0, 0))
            st_sec, st_is_f, st_is_late, st_pen, st_obs = st_tuple
            rows.append({
                "hd": hd,
                "jcd": jcd,
                "rno": int(rno),
                "lane": lane,
                "regno": int(rec["regno"]) if rec["regno"] is not None else None,

                "st_sec": st_sec,
                "st_is_f": st_is_f,
                "st_is_late": st_is_late,
                "st_penalized": st_pen,
                "st_observed": st_obs,

                "tenji_st_sec": rec["tenji_st_sec"],
                "tenji_is_f": rec["tenji_is_f"],
                "tenji_f_over_sec": rec["tenji_f_over_sec"],
                "tenji_rank": int(rec["tenji_rank"]),

                "st_rank": int(rec["st_rank"]) if rec["st_rank"] is not None else None,
                "course_avg_st": rec["course_avg_st"],
                "course_first_rate": rec["course_first_rate"],
                "course_3rd_rate": rec["course_3rd_rate"],
            })

    out_df = pl.DataFrame(rows, schema={
        "hd": pl.Utf8,
        "jcd": pl.Utf8,
        "rno": pl.Int64,
        "lane": pl.Int64,
        "regno": pl.Int64,

        "st_sec": pl.Float64,
        "st_is_f": pl.Int64,
        "st_is_late": pl.Int64,
        "st_penalized": pl.Int64,
        "st_observed": pl.Int64,

        "tenji_st_sec": pl.Float64,
        "tenji_is_f": pl.Int64,
        "tenji_f_over_sec": pl.Float64,
        "tenji_rank": pl.Int64,

        "st_rank": pl.Int64,
        "course_avg_st": pl.Float64,
        "course_first_rate": pl.Float64,
        "course_3rd_rate": pl.Float64,
    })

    # 出力
    if out_format in ("csv", "both"):
        p = day_out / "st_train.csv"
        if csv_compress:
            p = Path(str(p) + ".gz")
            out_df.write_csv(p, include_header=True)
        else:
            out_df.write_csv(p, include_header=True)
        print(f"[WRITE] {p} (rows={out_df.height})")

    if out_format in ("parquet", "both"):
        p = day_out / "st_train.parquet"
        out_df.write_parquet(p)
        print(f"[WRITE] {p} (rows={out_df.height})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_root", default="data/shards")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--out_root", default="data/staging")
    ap.add_argument("--out_format", default="csv", choices=["csv","parquet","both"])
    ap.add_argument("--csv_compress", action="store_true")
    args = ap.parse_args()

    shards_root = Path(args.shards_root)
    results_root = Path(args.results_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(shards_root.glob("**/integrated_pro.csv"))
    print(f"[INFO] found {len(files)} integrated_pro.csv files")
    if not files:
        print(f"[WARN] no integrated_pro.csv under {shards_root}")
        return

    for f in files:
        print(f"[INFO] reading {f}")
        try:
            process_one_day(
                integrated_path=f,
                results_root=results_root,
                out_root=out_root,
                out_format=args.out_format,
                csv_compress=args.csv_compress,
            )
        except Exception as e:
            print(f"[ERROR] {f}: {e}")

if __name__ == "__main__":
    main()
