# scripts/build_staging_from_integrated.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, gzip
from pathlib import Path
from typing import Dict, Any, List, Tuple

import polars as pl

# =========================================================
# Helpers
# =========================================================
def _z2(x: str | int) -> str:
    s = str(x).strip()
    # "06" も "6" も 2桁 0埋めへ
    try:
        return f"{int(s):02d}"
    except Exception:
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
        print(f"[WRITE] {path} (rows={df.height})")
    else:
        df.write_csv(path)
        print(f"[WRITE] {path} (rows={df.height})")

def _write_parquet(df: pl.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)
    print(f"[WRITE] {path} (rows={df.height})")

# =========================================================
# Load & normalize integrated_pro.csv
# =========================================================
INTEGRATED_DTYPES: Dict[str, pl.DataType] = {
    # keys
    "hd": pl.Utf8,
    "jcd": pl.Utf8,
    "rno": pl.Int64,
    "lane": pl.Int64,
    "racer_id": pl.Utf8,
    "racer_name": pl.Utf8,
    # weather/meta
    "weather_sky": pl.Utf8,
    "wind_dir": pl.Utf8,           # ← 混在回避（数値/文字列）
    "wind_speed_m": pl.Float64,
    "wave_height_cm": pl.Float64,
    "title": pl.Utf8,
    "decision": pl.Utf8,
    # stats
    "national_win": pl.Float64,
    "national_2r": pl.Float64,
    "national_3r": pl.Float64,
    "local_win": pl.Float64,
    "local_2r": pl.Float64,
    "local_3r": pl.Float64,
    "motor_id": pl.Float64,
    "motor_rate2": pl.Float64,
    "motor_rate3": pl.Float64,
    "boat_id": pl.Float64,
    "boat_rate2": pl.Float64,
    "boat_rate3": pl.Float64,
    "weightKg": pl.Float64,
    "tiltDeg": pl.Float64,
    "tenji_sec": pl.Float64,
    "tenji_st": pl.Float64,
    "isF_tenji": pl.Int64,
    # per-lane yearly course stats (if present)
    "course_first_rate": pl.Float64,
    "course_3rd_rate": pl.Float64,
    "course_avg_st": pl.Float64,
    "kimarite_makuri": pl.Float64,
    "kimarite_sashi": pl.Float64,
    "kimarite_makuri_sashi": pl.Float64,
    "kimarite_nuki": pl.Float64,
    # derived
    "tenji_rank": pl.Int64,
    "st_rank": pl.Int64,
    "power_lane": pl.Float64,
    "power_inner": pl.Float64,
    "power_outer": pl.Float64,
    "outer_over_inner": pl.Float64,
    "st_diff_4_vs_3": pl.Float64,
    "st_diff_5_vs_4": pl.Float64,
    "st_diff_6_vs_5": pl.Float64,
    "dash_attack_flag": pl.Int64,
    "is_strong_wind": pl.Int64,
    "is_crosswind": pl.Int64,
    "label_3t": pl.Utf8,
}

INTEGRATED_MIN_COLS = ["hd","jcd","rno","lane","racer_id","label_3t","decision","tenji_st","course_avg_st"]

def load_integrated_all(shards_root: Path) -> List[Tuple[str, str, str, pl.DataFrame]]:
    """
    shards_root/**/integrated_pro.csv をすべて読み込み、列型を正規化して返す。
    戻り値: [(hd/YYYYMMDD, year, md, df6_day), ...]
    """
    files = sorted(shards_root.glob("**/integrated_pro.csv"))
    print(f"[INFO] found {len(files)} integrated_pro.csv files")
    out: List[Tuple[str,str,str,pl.DataFrame]] = []

    for p in files:
        # パスから日付階層を推定: .../YYYY/MMDD/integrated_pro.csv
        parts = p.parts
        try:
            year = [x for x in parts if x.isdigit() and len(x) == 4][-1]
            md = [x for x in parts if x.isdigit() and len(x) == 4 and x != year][-1]
        except Exception:
            # フォルダ構成が違う場合はスキップ
            print(f"[WARN] skip (cannot detect year/md): {p}")
            continue

        df = pl.read_csv(
            p,
            dtypes=INTEGRATED_DTYPES,
            ignore_errors=True,
        )

        # 必須キーの型合わせ（最終防御）
        df = df.with_columns([
            pl.col("hd").cast(pl.Utf8),
            pl.col("jcd").cast(pl.Utf8).str.zfill(2),   # "6" → "06"
            pl.col("rno").cast(pl.Int64),
            pl.col("lane").cast(pl.Int64),
            pl.col("racer_id").cast(pl.Utf8),
        ])

        # 欠落カラムを作っておく（将来の一貫性）
        for c, dt in INTEGRATED_DTYPES.items():
            if c not in df.columns:
                df = df.with_columns(pl.lit(None, dtype=dt).alias(c))

        # 最低限の検査
        missing = [c for c in ["hd","jcd","rno","lane"] if c not in df.columns]
        if missing:
            print(f"[WARN] skip (missing keys {missing}): {p}")
            continue

        # hdが空の行は落とす
        df = df.filter(pl.col("hd").is_not_null())

        # このファイルの hd は複数日が混ざらない想定だが、混ざっていても問題なし
        out.append((None, year, md, df))  # hd は行に乗っているので None 可

    return out

# =========================================================
# Build daily staging
# =========================================================
def build_day_staging(df6_day: pl.DataFrame, results_root: Path, out_dir: Path,
                      out_format: str = "csv", csv_compress: bool = False):
    """
    1日の integrated_pro（複数場/複数R含む）から
      - races_st
      - winners
      - racecard
    を構築し out_dir に出力する。
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # まずキー列の正規化
    df6 = df6_day.with_columns([
        pl.col("hd").cast(pl.Utf8),
        pl.col("jcd").cast(pl.Utf8).str.zfill(2),
        pl.col("rno").cast(pl.Int64),
        pl.col("lane").cast(pl.Int64),
        pl.col("racer_id").cast(pl.Utf8),
    ])

    # --- パーティション分割（hd,jcd,rno 毎）
    groups: List[pl.DataFrame] = df6.partition_by(["hd","jcd","rno"], maintain_order=True)

    st_rows: List[Dict[str, Any]] = []
    win_rows: List[Dict[str, Any]] = []
    # racecard は integrated を正規化コピー：そのまま全行書き出す
    racecard_df = df6.select([
        "hd","jcd","rno","lane","racer_id","racer_name","grade","age",
        "weather_sky","wind_dir","wind_speed_m","wave_height_cm","title","decision",
        "national_win","national_2r","national_3r",
        "local_win","local_2r","local_3r",
        "motor_id","motor_rate2","motor_rate3",
        "boat_id","boat_rate2","boat_rate3",
        "weightKg","tiltDeg","tenji_sec","tenji_st","isF_tenji",
        "course_first_rate","course_3rd_rate","course_avg_st",
        "kimarite_makuri","kimarite_sashi","kimarite_makuri_sashi","kimarite_nuki",
        "tenji_rank","st_rank",
        "power_lane","power_inner","power_outer","outer_over_inner",
        "st_diff_4_vs_3","st_diff_5_vs_4","st_diff_6_vs_5",
        "dash_attack_flag","is_strong_wind","is_crosswind",
        "label_3t",
    ])

    for g in groups:
        if g.is_empty():
            continue
        # 代表値
        hd = g[0, "hd"]
        jcd = _z2(g[0, "jcd"])
        rno = int(g[0, "rno"])

        # results の path
        year = hd[:4]
        md = hd[4:8]
        res_path = results_root / year / md / jcd / f"{rno}R.json"
        res = _read_json(res_path)

        lane_to_st: Dict[int, float] = {}
        decision = None
        if res:
            # 実 ST
            start = res.get("start") or []
            for item in start:
                try:
                    lane = int(item.get("lane"))
                    st = float(item.get("st"))
                    lane_to_st[lane] = st
                except Exception:
                    continue
            # 決まり手
            meta = res.get("meta") or {}
            decision = meta.get("decision")

        # races_st 行を追加（全艇分）
        for row in g.iter_rows(named=True):
            lane = int(row.get("lane"))
            st_val = lane_to_st.get(lane, None)
            st_rows.append({
                "hd": hd,
                "jcd": jcd,
                "rno": rno,
                "lane": lane,
                "racer_id": row.get("racer_id"),
                "st": st_val,                     # 正解ST（無ければnull）
                # 相関特徴（学習に便利なものをいくつか）
                "tenji_st": row.get("tenji_st"),
                "course_avg_st": row.get("course_avg_st"),
                "st_rank": row.get("st_rank"),
                "power_lane": row.get("power_lane"),
            })

        # winners 行を追加（1レース1行）
        # label_3t が "a-b-c" なので win_lane = a
        label = g[0, "label_3t"]
        win_lane = None
        win_regno = None
        if isinstance(label, str) and "-" in label:
            try:
                win_lane = int(label.split("-")[0])
                # lane=win_lane の racer_id を引く
                sub = g.filter(pl.col("lane") == win_lane)
                if not sub.is_empty():
                    win_regno = sub[0, "racer_id"]
            except Exception:
                pass

        # decision は integrated の列か results の meta.decision を採用（results優先）
        dec_src = decision if decision else (g[0, "decision"] if "decision" in g.columns else None)

        win_rows.append({
            "hd": hd,
            "jcd": jcd,
            "rno": rno,
            "win_lane": win_lane,
            "win_racer_id": win_regno,
            "decision": dec_src,
        })

    # DataFrames
    races_st_df = pl.DataFrame(st_rows, schema={
        "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64,
        "racer_id": pl.Utf8, "st": pl.Float64,
        "tenji_st": pl.Float64, "course_avg_st": pl.Float64,
        "st_rank": pl.Int64, "power_lane": pl.Float64,
    }) if st_rows else pl.DataFrame(
        schema={"hd": pl.Utf8,"jcd": pl.Utf8,"rno": pl.Int64,"lane": pl.Int64,
                "racer_id": pl.Utf8,"st": pl.Float64,
                "tenji_st": pl.Float64,"course_avg_st": pl.Float64,
                "st_rank": pl.Int64,"power_lane": pl.Float64}
    )

    winners_df = pl.DataFrame(win_rows, schema={
        "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64,
        "win_lane": pl.Int64, "win_racer_id": pl.Utf8, "decision": pl.Utf8
    }) if win_rows else pl.DataFrame(
        schema={"hd": pl.Utf8,"jcd": pl.Utf8,"rno": pl.Int64,
                "win_lane": pl.Int64,"win_racer_id": pl.Utf8,"decision": pl.Utf8}
    )

    # ---- write ----
    if out_format in ("csv","both"):
        _write_csv(races_st_df, out_dir / "races_st.csv", csv_compress)
        _write_csv(winners_df,   out_dir / "winners.csv",  csv_compress)
        _write_csv(racecard_df,  out_dir / "racecard.csv", csv_compress)
    if out_format in ("parquet","both"):
        _write_parquet(races_st_df, out_dir / "races_st.parquet")
        _write_parquet(winners_df,  out_dir / "winners.parquet")
        _write_parquet(racecard_df, out_dir / "racecard.parquet")

# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_root",  default="data/shards", help="integrated_pro.csv の親ディレクトリ")
    ap.add_argument("--results_root", default="public/results", help="boatrace結果JSONルート")
    ap.add_argument("--out_root",     default="data/staging", help="出力ルート")
    ap.add_argument("--out_format",   default="csv", choices=["csv","parquet","both"])
    ap.add_argument("--csv_compress", action="store_true")
    args = ap.parse_args()

    shards_root  = Path(args.shards_root)
    results_root = Path(args.results_root)
    out_root     = Path(args.out_root)

    days = load_integrated_all(shards_root)
    if not days:
        print(f"[WARN] no integrated_pro.csv under {shards_root}")
        return

    # hd は行側に入っているので、ここではフォルダ階層（year/md）単位で処理
    seen_dirs = set()
    for _, year, md, df in days:
        day_out = out_root / year / md
        key = str(day_out)
        if key in seen_dirs:
            # 同日が複数ファイルに分かれている場合は結合して1回で吐く
            continue
        # 同日の integrated を全部束ねる
        same_day_parts = [d for _, y, m, d in days if y == year and m == md]
        df_day = pl.concat(same_day_parts, how="vertical", rechunk=True) if len(same_day_parts) > 1 else df

        print(f"[INFO] build staging for {year}/{md} (rows={df_day.height})")
        build_day_staging(
            df6_day=df_day,
            results_root=results_root,
            out_dir=day_out,
            out_format=args.out_format,
            csv_compress=args.csv_compress,
        )
        seen_dirs.add(key)

if __name__ == "__main__":
    main()
