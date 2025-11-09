#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import re
from pathlib import Path
import sys

import polars as pl

# 4クラスだけ残す
DEC4 = {
    "逃げ": "NIGE",
    "差し": "SASHI",
    "まくり": "MAKURI",
    "まくり差し": "MAKURI_SASHI",
}

INTEG_BASENAME = "integrated_pro.csv"

def log(msg: str): print(msg, flush=True)
def warn(msg: str): print(f"[WARN] {msg}", flush=True)
def fatal(msg: str):
    print(f"[FATAL] {msg}", file=sys.stderr, flush=True)
    sys.exit(1)

def find_integrated_csvs(root: Path) -> list[Path]:
    """
    root が日付フォルダでも、年フォルダでも、ルートでもOK。
    とにかく再帰で integrated_pro.csv を集める。
    """
    if root.is_file() and root.name == INTEG_BASENAME:
        return [root]
    if root.is_dir():
        return list(root.rglob(INTEG_BASENAME))
    return []

def y_md_from_path(p: Path) -> tuple[str, str]:
    """
    パス末尾の “…/YYYY/MMDD/integrated_pro.csv” から YYYY, MMDD を安全抽出。
    失敗したら空文字を返す。
    """
    s = str(p).replace("\\", "/")
    m = re.search(r"/(\d{4})/(\d{4})/integrated_pro\.csv$", s)
    if not m:
        return "", ""
    return m.group(1), m.group(2)

def read_integrated(path: Path) -> pl.DataFrame:
    """
    integrated_pro.csv を読み込む（型は必要最小限だけ上書き）
    """
    return pl.read_csv(
        path,
        ignore_errors=True,
        try_parse_dates=False,
        new_columns=None,
    )

def filter_to_day(df: pl.DataFrame, y: str, md: str) -> pl.DataFrame:
    """
    ファイル1本に複数日が混ざっていても hd で日付絞り込み。
    hd は数値/文字の両対応でフィルタ。
    """
    ymd = y + md  # YYYYMMDD
    if "hd" not in df.columns:
        return df  # 念の為（無ければスルー）
    # 文字・数値両対応
    return df.with_columns(pl.col("hd").cast(pl.Utf8, strict=False)) \
             .filter(pl.col("hd") == ymd)

def normalize_decision4(df: pl.DataFrame) -> pl.DataFrame:
    """
    決まり手4クラスに正規化。対象外は除外。
    """
    if "decision" not in df.columns:
        return df.head(0)

    # decision を文字列へ
    df2 = df.with_columns(pl.col("decision").cast(pl.Utf8, strict=False))
    # マッピング
    dec_map = pl.Series("decision", list(DEC4.keys()))
    cls_map = pl.Series("decision4", list(DEC4.values()))
    map_df = pl.DataFrame({"decision": dec_map, "decision4": cls_map})

    df3 = df2.join(map_df, on="decision", how="left")
    df3 = df3.filter(pl.col("decision4").is_not_null())

    return df3

def select_features_for_decision4(df: pl.DataFrame) -> pl.DataFrame:
    """
    各レーン行を保持したまま、学習に使いそうな列を抽出。
    ラベルは race-level（同一レースで同一）だが lane ごとに特徴が異なる想定。
    """
    # 必須ID
    base_cols = ["hd", "jcd", "rno", "lane", "racer_id", "racer_name"]
    # integrated_proに入っている代表的な特徴
    cand_feats = [
        "course_first_rate", "course_3rd_rate", "course_avg_st",
        "tenji_st", "tenji_sec", "tenji_rank", "st_rank",
        "motor_rate2", "motor_rate3", "boat_rate2", "boat_rate3",
        "wind_speed_m", "wave_height_cm",
        "kimarite_makuri", "kimarite_sashi", "kimarite_makuri_sashi", "kimarite_nuki",
        "power_lane", "power_inner", "power_outer", "outer_over_inner",
        "dash_attack_flag", "is_strong_wind", "is_crosswind",
    ]

    keep = [c for c in base_cols if c in df.columns] + ["decision4"]
    for c in cand_feats:
        if c in df.columns:
            keep.append(c)

    # tenji_st(=展示ST秒) が “F0.01→0.0001” などの前処理を既にしている前提の環境もあるので、
    # ここでは文字列→数値の安全変換（失敗はnull）
    out = df.select([pl.col(c) for c in keep]).with_columns([
        pl.col("hd").cast(pl.Utf8, strict=False),
        pl.col("jcd").cast(pl.Utf8, strict=False),
        pl.col("rno").cast(pl.Int64, strict=False),
        pl.col("lane").cast(pl.Int64, strict=False),
        pl.col("tenji_st").cast(pl.Float64, strict=False) if "tenji_st" in keep else pl.lit(None).alias("_dummy0"),
        pl.col("tenji_sec").cast(pl.Float64, strict=False) if "tenji_sec" in keep else pl.lit(None).alias("_dummy1"),
        pl.col("tenji_rank").cast(pl.Int64, strict=False) if "tenji_rank" in keep else pl.lit(None).alias("_dummy2"),
        pl.col("st_rank").cast(pl.Int64, strict=False) if "st_rank" in keep else pl.lit(None).alias("_dummy3"),
    ]).drop(["_dummy0","_dummy1","_dummy2","_dummy3"], strict=False)

    return out

def write_day(out_root: Path, y: str, md: str, df: pl.DataFrame) -> Path:
    out_dir = out_root / y / md
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "decision4_train.csv"
    df.write_csv(out_path)
    return out_path

def process_one_csv(in_csv: Path, out_root: Path) -> int:
    y, md = y_md_from_path(in_csv)
    if not y or not md:
        warn(f"{in_csv}: year/mmdd の抽出に失敗（スキップ）")
        return 0

    try:
        raw = read_integrated(in_csv)
    except Exception as e:
        warn(f"{in_csv}: read_csv failed: {e}")
        return 0

    # 日付で絞る（保険）
    day = filter_to_day(raw, y, md)
    if day.height == 0:
        # ファイル内に他日付しかない/欠損ならスキップ
        warn(f"{in_csv}: day filter no rows for {y}{md}")
        return 0

    # 4クラスだけ残す
    day4 = normalize_decision4(day)
    if day4.height == 0:
        # その日の全Rが対象外決まり手（抜き/恵まれ 等）ならスキップ
        return 0

    # 特徴列ピックアップ
    out_df = select_features_for_decision4(day4)
    if out_df.height == 0:
        return 0

    out_path = write_day(out_root, y, md, out_df)
    log(f"[WRITE] {out_path} rows={out_df.height}")
    return out_df.height

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated_root", required=True, help="data/shards もしくは日付フォルダ/年フォルダ")
    ap.add_argument("--results_root", default=None, help="未使用（将来拡張）")
    ap.add_argument("--out_root", required=True, help="出力先 data/staging")
    args = ap.parse_args()

    integ_root = Path(args.integrated_root)
    out_root = Path(args.out_root)

    files = find_integrated_csvs(integ_root)
    if not files:
        fatal(f"integrated_pro.csv が見つかりません: {integ_root}")

    log(f"[INFO] found {len(files)} integrated_pro.csv files")

    total = 0
    for p in sorted(files):
        try:
            total += process_one_csv(p, out_root)
        except Exception as e:
            warn(f"{p}: {e}")

    if total == 0:
        warn("出力対象データがありません。")
    else:
        log(f"[DONE] total rows = {total}")

if __name__ == "__main__":
    main()
