# scripts/build_staging_from_integrated.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, gzip
from pathlib import Path
from typing import Dict, Any, List, Tuple

import polars as pl

# =========================
# Utils
# =========================
def _z2(x: str | int) -> str:
    try:
        return f"{int(str(x).strip()):02d}"
    except Exception:
        s = str(x).strip()
        return s.zfill(2)

def _read_json(p: Path) -> Dict[str, Any] | None:
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARN] JSON load failed: {p} ({e})")
        return None

def _write_csv(df: pl.DataFrame, path: Path, compress: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        if not str(path).endswith(".gz"):
            path = Path(str(path) + ".gz")
        with gzip.open(path, "wt", encoding="utf-8", newline="") as fo:
            df.write_csv(fo)
    else:
        df.write_csv(path)
    print(f"[WRITE] {path} (rows={df.height})")

def _write_parquet(df: pl.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"[WRITE] {path} (rows={df.height})")

# =========================
# Load integrated_pro.csv
# =========================
INTEGRATED_DTYPES: Dict[str, pl.DataType] = {
    "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64,
    "racer_id": pl.Utf8, "racer_name": pl.Utf8,
    "weather_sky": pl.Utf8, "wind_dir": pl.Utf8,
    "wind_speed_m": pl.Float64, "wave_height_cm": pl.Float64,
    "title": pl.Utf8, "decision": pl.Utf8,
    "national_win": pl.Float64, "national_2r": pl.Float64, "national_3r": pl.Float64,
    "local_win": pl.Float64, "local_2r": pl.Float64, "local_3r": pl.Float64,
    "motor_id": pl.Float64, "motor_rate2": pl.Float64, "motor_rate3": pl.Float64,
    "boat_id": pl.Float64, "boat_rate2": pl.Float64, "boat_rate3": pl.Float64,
    "weightKg": pl.Float64, "tiltDeg": pl.Float64,
    "tenji_sec": pl.Float64, "tenji_st": pl.Float64, "isF_tenji": pl.Int64,
    "course_first_rate": pl.Float64, "course_3rd_rate": pl.Float64, "course_avg_st": pl.Float64,
    "kimarite_makuri": pl.Float64, "kimarite_sashi": pl.Float64,
    "kimarite_makuri_sashi": pl.Float64, "kimarite_nuki": pl.Float64,
    "tenji_rank": pl.Int64, "st_rank": pl.Int64,
    "power_lane": pl.Float64, "power_inner": pl.Float64, "power_outer": pl.Float64,
    "outer_over_inner": pl.Float64,
    "st_diff_4_vs_3": pl.Float64, "st_diff_5_vs_4": pl.Float64, "st_diff_6_vs_5": pl.Float64,
    "dash_attack_flag": pl.Int64, "is_strong_wind": pl.Int64, "is_crosswind": pl.Int64,
    "label_3t": pl.Utf8,
}

def load_integrated_all(shards_root: Path) -> pl.DataFrame:
    files = sorted(shards_root.glob("**/integrated_pro.csv"))
    print(f"[INFO] found {len(files)} integrated_pro.csv files")
    if not files:
        return pl.DataFrame(schema={"hd": pl.Utf8})

    dfs = []
    for p in files:
        df = pl.read_csv(p, dtypes=INTEGRATED_DTYPES, ignore_errors=True)
        # 最低限の型とキー整形
        df = df.with_columns([
            pl.col("hd").cast(pl.Utf8),
            pl.col("jcd").cast(pl.Utf8).str.zfill(2),
            pl.col("rno").cast(pl.Int64),
            pl.col("lane").cast(pl.Int64),
            pl.col("racer_id").cast(pl.Utf8),
        ]).filter(pl.col("hd").is_not_null())

        # 欠落カラムを埋める（将来互換）
        for c, dt in INTEGRATED_DTYPES.items():
            if c not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=dt).alias(c))

        dfs.append(df)

    return pl.concat(dfs, how="vertical", rechunk=True)

# =========================
# Build staging per day
# =========================
def build_staging(df_all: pl.DataFrame, results_root: Path, out_root: Path,
                  out_format: str = "csv", csv_compress: bool = False):
    if df_all.is_empty():
        print("[WARN] no rows in integrated dataframe")
        return

    # hd(YYYYMMDD) から year, md を作って日毎に処理
    df_all = df_all.with_columns([
        pl.col("hd").cast(pl.Utf8),
        pl.col("jcd").cast(pl.Utf8).str.zfill(2),
        pl.col("rno").cast(pl.Int64),
        pl.col("lane").cast(pl.Int64),
        pl.col("racer_id").cast(pl.Utf8),
        pl.col("hd").str.slice(0, 4).alias("_year"),
        pl.col("hd").str.slice(4, 4).alias("_md"),
    ])

    # /YYYY/MMDD/ で書くために日毎のパーティション
    day_parts = df_all.partition_by(["_year", "_md"], maintain_order=True)
    for day_df in day_parts:
        year = day_df[0, "_year"]
        md   = day_df[0, "_md"]
        out_dir = out_root / year / md
        out_dir.mkdir(parents=True, exist_ok=True)

        # レース毎パーティション
        race_parts = day_df.partition_by(["hd","jcd","rno"], maintain_order=True)

        # それぞれで展示STランク（F最下位）を再計算
        racecard_list: List[pl.DataFrame] = []
        st_rows: List[dict] = []
        win_rows: List[dict] = []

        for g in race_parts:
            if g.is_empty():
                continue

            # F判定（tenji_st < 0.01 をF扱い）
            g = g.with_columns([
                (pl.col("tenji_st").is_not_null() & (pl.col("tenji_st") < 0.01)).cast(pl.Int8).alias("_is_f"),
                # ランク用の値: F→999.0、未定義→null、それ以外→そのまま
                pl.when(pl.col("_is_f") == 1).then(pl.lit(999.0))
                  .when(pl.col("tenji_st").is_null()).then(pl.lit(None, dtype=pl.Float64))
                  .otherwise(pl.col("tenji_st"))
                  .alias("_st_for_rank"),
            ])

            # ここで within-race のランク再計算
            g = g.with_columns([
                pl.col("tenji_sec").rank(method="dense", descending=False).alias("tenji_rank_re"),
                pl.col("_st_for_rank").rank(method="dense", descending=False).alias("st_rank_re"),
            ])

            # racecard（再計算した rank を使う）
            racecard_cols = [
                "hd","jcd","rno","lane","racer_id","racer_name","grade","age",
                "weather_sky","wind_dir","wind_speed_m","wave_height_cm","title","decision",
                "national_win","national_2r","national_3r",
                "local_win","local_2r","local_3r",
                "motor_id","motor_rate2","motor_rate3",
                "boat_id","boat_rate2","boat_rate3",
                "weightKg","tiltDeg","tenji_sec","tenji_st","isF_tenji",
                "course_first_rate","course_3rd_rate","course_avg_st",
                "kimarite_makuri","kimarite_sashi","kimarite_makuri_sashi","kimarite_nuki",
                # 置き換えたランク
                "tenji_rank_re","st_rank_re",
                "power_lane","power_inner","power_outer","outer_over_inner",
                "st_diff_4_vs_3","st_diff_5_vs_4","st_diff_6_vs_5",
                "dash_attack_flag","is_strong_wind","is_crosswind",
                "label_3t",
            ]
            rc = g.select([c for c in racecard_cols if c in g.columns]) \
                 .rename({"tenji_rank_re":"tenji_rank","st_rank_re":"st_rank"})
            racecard_list.append(rc)

            # results から実ST・決まり手
            hd = g[0, "hd"]; jcd = g[0, "jcd"]; rno = int(g[0, "rno"])
            res_path = results_root / hd[:4] / hd[4:8] / _z2(jcd) / f"{rno}R.json"
            res = _read_json(res_path)

            lane_to_st = {}
            decision = None
            if res:
                for it in (res.get("start") or []):
                    try:
                        lane_to_st[int(it.get("lane"))] = float(it.get("st"))
                    except Exception:
                        pass
                meta = res.get("meta") or {}
                decision = meta.get("decision")

            # races_st行
            for row in g.iter_rows(named=True):
                st_rows.append({
                    "hd": hd, "jcd": _z2(jcd), "rno": rno,
                    "lane": int(row.get("lane")),
                    "racer_id": row.get("racer_id"),
                    "st": lane_to_st.get(int(row.get("lane"))),
                    "tenji_st": row.get("tenji_st"),
                    "course_avg_st": row.get("course_avg_st"),
                    "st_rank": row.get("st_rank"),      # 再計算済み
                    "power_lane": row.get("power_lane"),
                })

            # winners行（1-2-3 の 先頭コース）
            label = g[0, "label_3t"]
            win_lane, win_regno = None, None
            if isinstance(label, str) and "-" in label:
                try:
                    win_lane = int(label.split("-")[0])
                    sub = g.filter(pl.col("lane") == win_lane)
                    if not sub.is_empty():
                        win_regno = sub[0, "racer_id"]
                except Exception:
                    pass
            win_rows.append({
                "hd": hd, "jcd": _z2(jcd), "rno": rno,
                "win_lane": win_lane, "win_racer_id": win_regno,
                "decision": (decision if decision else g[0, "decision"]),
            })

        # 日のDF
        racecard_df = pl.concat(racecard_list, how="vertical", rechunk=True) if racecard_list else pl.DataFrame()
        races_st_df = pl.DataFrame(st_rows, schema={
            "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64,
            "racer_id": pl.Utf8, "st": pl.Float64,
            "tenji_st": pl.Float64, "course_avg_st": pl.Float64,
            "st_rank": pl.Int64, "power_lane": pl.Float64
        }) if st_rows else pl.DataFrame(schema={
            "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64,
            "racer_id": pl.Utf8, "st": pl.Float64,
            "tenji_st": pl.Float64, "course_avg_st": pl.Float64,
            "st_rank": pl.Int64, "power_lane": pl.Float64
        })
        winners_df = pl.DataFrame(win_rows, schema={
            "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64,
            "win_lane": pl.Int64, "win_racer_id": pl.Utf8, "decision": pl.Utf8
        }) if win_rows else pl.DataFrame(schema={
            "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64,
            "win_lane": pl.Int64, "win_racer_id": pl.Utf8, "decision": pl.Utf8
        })

        # 書き出し（/YYYY/MMDD/）
        if out_format in ("csv","both"):
            _write_csv(racecard_df, out_dir / "racecard.csv", csv_compress)
            _write_csv(races_st_df, out_dir / "races_st.csv", csv_compress)
            _write_csv(winners_df,  out_dir / "winners.csv",  csv_compress)
        if out_format in ("parquet","both"):
            _write_parquet(racecard_df, out_dir / "racecard.parquet")
            _write_parquet(races_st_df, out_dir / "races_st.parquet")
            _write_parquet(winners_df,  out_dir / "winners.parquet")

# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_root",  default="data/shards")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--out_root",     default="data/staging")
    ap.add_argument("--out_format",   default="csv", choices=["csv","parquet","both"])
    ap.add_argument("--csv_compress", action="store_true")
    args = ap.parse_args()

    shards_root  = Path(args.shards_root)
    results_root = Path(args.results_root)
    out_root     = Path(args.out_root)

    df_all = load_integrated_all(shards_root)
    if df_all.is_empty():
        print(f"[WARN] no integrated_pro.csv under {shards_root}")
        return

    build_staging(
        df_all=df_all,
        results_root=results_root,
        out_root=out_root,
        out_format=args.out_format,
        csv_compress=args.csv_compress,
    )

if __name__ == "__main__":
    main()
