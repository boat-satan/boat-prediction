#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_staging_from_integrated.py
- data/shards/**/integrated_pro.csv を読み、学習用のステージングCSVを生成
- 出力:
    data/staging/YYYY/MMDD/st_staging.csv[.gz]     … ST学習 (hd,jcd,rno,lane,regno,st)
    data/staging/YYYY/MMDD/fin_staging.csv[.gz]    … 決まり手学習 (hd,jcd,rno,finisher_label)
    data/staging/YYYY/MMDD/win_staging.csv[.gz]    … 単勝学習 (hd,jcd,rno,win_lane,win_regno)
    data/staging/YYYY/MMDD/runners.csv[.gz]        … 特徴量の土台 (hd,jcd,rno,lane,regno,各種feature)

使い方例:
  python scripts/build_staging_from_integrated.py \
    --shards_root data/shards \
    --results_root public/results \
    --out_root data/staging \
    --out_format csv --csv_compress
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any

import polars as pl


def jload(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _year_md_from_hd(hd: str) -> Tuple[str, str]:
    return hd[:4], hd[4:8]


def _read_integrated_csv(p: Path) -> pl.DataFrame:
    # NOTE: integrated_pro.csv はヘッダありCSV
    return pl.read_csv(p, infer_schema_length=10000)


def _collect_unique_races(df6: pl.DataFrame) -> List[Tuple[str, str, int]]:
    # pandas では .groupby だが、polars は .group_by
    # ここではユニーク抽出で回すのがシンプル
    keys = (
        df6.select(["hd", "jcd", "rno"])
           .unique(maintain_order=True)
           .iter_rows()
    )
    races: List[Tuple[str, str, int]] = []
    for hd, jcd, rno in keys:
        races.append((str(hd), str(jcd), int(rno)))
    return races


def _subset_race(df6: pl.DataFrame, hd: str, jcd: str, rno: int) -> pl.DataFrame:
    return df6.filter(
        (pl.col("hd") == hd) & (pl.col("jcd") == jcd) & (pl.col("rno") == rno)
    )


def _lane_to_regno_map(df6_race: pl.DataFrame) -> Dict[int, str]:
    mp: Dict[int, str] = {}
    for lane, rid in df6_race.select(["lane", "racer_id"]).iter_rows():
        # racer_id は int/str 混在の可能性があるので文字列化
        if lane is not None and rid is not None:
            mp[int(lane)] = str(rid)
    return mp


def _load_result_json(results_root: Path, hd: str, jcd: str, rno: int) -> Dict[str, Any] | None:
    year, md = _year_md_from_hd(hd)
    res_path = results_root / year / md / jcd / f"{rno}R.json"
    if res_path.exists():
        try:
            return jload(res_path)
        except Exception:
            return None
    return None


def _append_rows(
    st_rows: List[Dict[str, Any]],
    fin_rows: List[Dict[str, Any]],
    win_rows: List[Dict[str, Any]],
    runners_rows: List[Dict[str, Any]],
    df6_race: pl.DataFrame,
    result_js: Dict[str, Any] | None,
) -> None:
    # runners: 出走行そのまま＋最低限の列名を正規化
    for row in df6_race.iter_rows(named=True):
        runners_rows.append({
            "hd": row.get("hd"),
            "jcd": row.get("jcd"),
            "rno": row.get("rno"),
            "lane": row.get("lane"),
            "regno": str(row.get("racer_id")) if row.get("racer_id") is not None else None,
            # よく使う特徴をいくつか拾っておく（必要に応じて追加）
            "course_first_rate": row.get("course_first_rate"),
            "course_3rd_rate": row.get("course_3rd_rate"),
            "course_avg_st": row.get("course_avg_st"),
            "kimarite_makuri": row.get("kimarite_makuri"),
            "kimarite_sashi": row.get("kimarite_sashi"),
            "kimarite_makuri_sashi": row.get("kimarite_makuri_sashi"),
            "kimarite_nuki": row.get("kimarite_nuki"),
            "tenji_st": row.get("tenji_st"),
            "tenji_rank": row.get("tenji_rank"),
            "st_rank": row.get("st_rank"),
            "national_win": row.get("national_win"),
            "motor_rate2": row.get("motor_rate2"),
            "boat_rate2": row.get("boat_rate2"),
            "wind_speed_m": row.get("wind_speed_m"),
            "wave_height_cm": row.get("wave_height_cm"),
        })

    hd = str(df6_race["hd"][0])
    jcd = str(df6_race["jcd"][0])
    rno = int(df6_race["rno"][0])

    # 決まり手（レース単位のラベル）
    finisher_label = None
    if result_js and isinstance(result_js.get("meta"), dict):
        finisher_label = result_js["meta"].get("decision")

    if finisher_label:
        fin_rows.append({
            "hd": hd, "jcd": jcd, "rno": rno,
            "finisher_label": finisher_label
        })

    # ST（スタート）: result_json.start に lane/st が入っている
    lane2reg = _lane_to_regno_map(df6_race)
    if result_js and isinstance(result_js.get("start"), list):
        for item in result_js["start"]:
            try:
                lane = int(item.get("lane"))
            except Exception:
                continue
            st_val = item.get("st")
            st_rows.append({
                "hd": hd, "jcd": jcd, "rno": rno,
                "lane": lane,
                "regno": lane2reg.get(lane),
                "st": float(st_val) if st_val is not None else None
            })

    # 単勝（勝者）: results[].rank == "1"
    if result_js and isinstance(result_js.get("results"), list):
        win_lane = None
        win_regno = None
        for item in result_js["results"]:
            if str(item.get("rank")) == "1":
                try:
                    win_lane = int(item.get("lane"))
                except Exception:
                    win_lane = None
                win_regno = str(item.get("racer_id")) if item.get("racer_id") is not None else None
                break
        if win_lane is not None or win_regno is not None:
            win_rows.append({
                "hd": hd, "jcd": jcd, "rno": rno,
                "win_lane": win_lane,
                "win_regno": win_regno
            })


def _write_csv(df: pl.DataFrame, out_path: Path, compress: bool) -> None:
    _ensure_dir(out_path.parent)
    if compress:
        # Polars はパス拡張子で圧縮判断しないので自前で .gz にして書く
        gz_path = Path(str(out_path) + ".gz")
        df.write_csv(gz_path)
        print(f"[WRITE] {gz_path} (rows={df.height})")
    else:
        df.write_csv(out_path)
        print(f"[WRITE] {out_path} (rows={df.height})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_root", required=True, help="入力: data/shards の親ディレクトリ")
    ap.add_argument("--results_root", required=True, help="公表リザルトJSONのルート (public/results)")
    ap.add_argument("--out_root", required=True, help="出力: data/staging の親ディレクトリ")
    ap.add_argument("--out_format", default="csv", choices=["csv", "parquet", "both"])
    ap.add_argument("--csv_compress", action="store_true")
    args = ap.parse_args()

    shards_root = Path(args.shards_root)
    results_root = Path(args.results_root)
    out_root = Path(args.out_root)

    # shards 下の integrated_pro.csv を全部探す（day 単位で保存されている想定）
    integrated_files = sorted(shards_root.glob("**/integrated_pro.csv"))
    print(f"[INFO] found {len(integrated_files)} integrated_pro.csv files")

    for integ_csv in integrated_files:
        try:
            df6 = _read_integrated_csv(integ_csv)
        except Exception as e:
            print(f"[WARN] skip {integ_csv}: {e}")
            continue

        if df6.is_empty():
            print(f"[WARN] empty: {integ_csv}")
            continue

        # 当日の年/月日を推定（先頭行の hd を使う）
        hd0 = str(df6["hd"][0])
        year, md = _year_md_from_hd(hd0)

        # 出力先（当日ディレクトリ）
        day_out = out_root / year / md
        _ensure_dir(day_out)

        st_rows: List[Dict[str, Any]] = []
        fin_rows: List[Dict[str, Any]] = []
        win_rows: List[Dict[str, Any]] = []
        runners_rows: List[Dict[str, Any]] = []

        races = _collect_unique_races(df6)
        for (hd, jcd, rno) in races:
            df6_race = _subset_race(df6, hd, jcd, rno)
            result_js = _load_result_json(results_root, hd, jcd, rno)
            _append_rows(st_rows, fin_rows, win_rows, runners_rows, df6_race, result_js)

        # DataFrame 化して書き出し
        if st_rows:
            st_df = pl.DataFrame(st_rows)
            _write_csv(st_df, day_out / "st_staging.csv", args.csv_compress)

        if fin_rows:
            fin_df = pl.DataFrame(fin_rows).unique(maintain_order=True)
            _write_csv(fin_df, day_out / "fin_staging.csv", args.csv_compress)

        if win_rows:
            win_df = pl.DataFrame(win_rows)
            _write_csv(win_df, day_out / "win_staging.csv", args.csv_compress)

        if runners_rows:
            runners_df = pl.DataFrame(runners_rows)
            _write_csv(runners_df, day_out / "runners.csv", args.csv_compress)

    print("[DONE] build staging complete")


if __name__ == "__main__":
    main()
