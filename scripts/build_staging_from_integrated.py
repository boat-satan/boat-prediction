#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_staging.py — ST/決まり手/三連単 分解モデルのステージング生成

入力（既存資産を利用）:
  - data/shards/**/integrated_pro.csv
      必須列:
        hd,jcd,rno,lane,racer_id,grade,tenji_st,st_rank,tenji_rank,
        course_first_rate,course_3rd_rate,course_avg_st,
        motor_rate2,boat_rate2,weather_sky,wind_dir,wind_speed_m,wave_height_cm,
        power_lane,power_inner,power_outer,outer_over_inner,
        st_diff_4_vs_3,st_diff_5_vs_4,st_diff_6_vs_5,
        dash_attack_flag,is_strong_wind,is_crosswind,label_3t
  - public/results/YYYY/MMDD/JCD/RR.json
      参照（ターゲットSTと着順/決まり手）:
        ["start"] の lane→st（本番ST）, ["meta"].decision, ["results"] 着順 等

出力（CSV/Parquet選択可）:
  - data/staging/YYYY/MMDD/features_lane.(csv|parquet)
  - data/staging/YYYY/MMDD/targets_st.(csv|parquet)
  - data/staging/YYYY/MMDD/expanded_120.(csv|parquet)

使い方例:
  python scripts/build_staging.py \
    --shards_root data/shards \
    --results_root public/results \
    --out_root data/staging \
    --out_format csv

メモ:
  - 既存 integrated_pro に course_avg_st が入っているため、
    過年度の集計や別ファイルは不要。
  - 本番ST（ターゲット）は results JSON の "start" から注入。
"""

from __future__ import annotations
import argparse, json, sys, gzip
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import polars as pl


# ---------- I/O ヘルパ ----------
def read_csv_safe(path: Path) -> Optional[pl.DataFrame]:
    if not path.exists():
        return None
    try:
        return pl.read_csv(path)
    except Exception as e:
        print(f"[WARN] failed to read CSV: {path} ({e})")
        return None

def write_tabular(df: pl.DataFrame, out: Path, fmt: str):
    out.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.write_csv(out)
    elif fmt == "parquet":
        df.write_parquet(out)
    elif fmt == "both":
        df.write_csv(out.with_suffix(".csv"))
        df.write_parquet(out.with_suffix(".parquet"))
    else:
        raise ValueError(f"unknown out_format: {fmt}")
    print(f"[WRITE] {out} (rows={df.height})")

def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] failed to load JSON: {path} ({e})")
        return None


# ---------- リザルトJSON → 本番STマップ ----------
def extract_race_targets(result_js: dict) -> Tuple[Dict[int, float], str, Optional[str]]:
    """
    returns:
      st_map: { lane(int) -> st(float) }  # 本番ST
      decision: 決まり手（例: '逃げ','まくり','まくり差し','差し','抜き',...）
      trifecta: 'a-b-c' or None
    """
    st_map: Dict[int, float] = {}
    decision = (result_js.get("meta") or {}).get("decision", "")
    trifecta = (result_js.get("payouts", {}) or {}).get("trifecta", {}).get("combo")

    for ent in (result_js.get("start") or []):
        lane = ent.get("lane")
        st   = ent.get("st")
        try:
            lane_i = int(lane)
            st_f   = float(st)
            st_map[lane_i] = st_f
        except Exception:
            continue
    return st_map, decision, trifecta


# ---------- 120通り展開 ----------
def expand_120_from_df6(df6: pl.DataFrame) -> pl.DataFrame:
    """
    df6: 同一レース(hd,jcd,rno) 6行（lane=1..6）想定
    必要列: hd,jcd,rno,lane,tenji_st,st_rank,tenji_rank,motor_rate2,grade,label_3t, ...
    """
    if df6.is_empty():
        return pl.DataFrame()
    # 辞書: lane -> レコード
    recs = {int(r["lane"]): r for r in df6.iter_rows(named=True)}

    # 120通り生成
    combos = []
    for a in (1,2,3,4,5,6):
        for b in (1,2,3,4,5,6):
            if b == a: continue
            for c in (1,2,3,4,5,6):
                if c == a or c == b: continue
                combos.append((a,b,c))

    rows = []
    hd = df6["hd"][0]; jcd = df6["jcd"][0]; rno = df6["rno"][0]
    label = df6["label_3t"][0] if "label_3t" in df6.columns else None

    def _get(rec, k): 
        try: return rec.get(k)
        except: return None

    for a,b,c in combos:
        r1 = recs.get(a, {})
        r2 = recs.get(b, {})
        r3 = recs.get(c, {})
        combo = f"{a}-{b}-{c}"

        rows.append({
            "hd": hd, "jcd": jcd, "rno": rno, "combo": combo,
            "p1_lane": a, "p2_lane": b, "p3_lane": c,
            "p1_tenji_st": _get(r1, "tenji_st"), "p2_tenji_st": _get(r2, "tenji_st"), "p3_tenji_st": _get(r3, "tenji_st"),
            "p1_st_rank":  _get(r1, "st_rank"),  "p2_st_rank":  _get(r2, "st_rank"),  "p3_st_rank":  _get(r3, "st_rank"),
            "p1_motor_rate2": _get(r1, "motor_rate2"), "p2_motor_rate2": _get(r2, "motor_rate2"), "p3_motor_rate2": _get(r3, "motor_rate2"),
            "p1_grade": _get(r1, "grade"), "p2_grade": _get(r2, "grade"), "p3_grade": _get(r3, "grade"),
            "wind_speed_m": _get(r1, "wind_speed_m"),
            "wave_height_cm": _get(r1, "wave_height_cm"),
            "is_strong_wind": _get(r1, "is_strong_wind"),
            "is_crosswind": _get(r1, "is_crosswind"),
            "label_3t": label,
            "is_hit": 1 if (label == combo) else 0,
        })
    return pl.DataFrame(rows)


# ---------- メイン処理 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_root",  default="data/shards")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--out_root",     default="data/staging")
    ap.add_argument("--out_format",   default="csv", choices=["csv","parquet","both"])
    args = ap.parse_args()

    shards_root  = Path(args.shards_root)
    results_root = Path(args.results_root)
    out_root     = Path(args.out_root)
    fmt          = args.out_format

    # data/shards/**/integrated_pro.csv を総なめ
    targets = sorted(shards_root.glob("**/integrated_pro.csv"))
    if not targets:
        print(f"[FATAL] not found: {shards_root}/**/integrated_pro.csv", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] found {len(targets)} integrated_pro.csv files")

    for csv_path in targets:
        # パスから YYYY/MMDD を抽出（shards/YYYY/MMDD/.../integrated_pro.csv 想定）
        try:
            # .../data/shards/2024/0101/integrated_pro.csv
            year   = csv_path.parent.parent.name
            mmdd   = csv_path.parent.name
            hd_hint = f"{year}{mmdd}"
        except Exception:
            year, mmdd, hd_hint = "unknown", "unknown", None

        df6 = read_csv_safe(csv_path)
        if df6 is None or df6.is_empty():
            print(f"[WARN] empty or unreadable: {csv_path}")
            continue

        # 型の整備（最低限）
        cast_map = {
            "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64,
            "tenji_st": pl.Float64, "st_rank": pl.Int64, "tenji_rank": pl.Int64,
            "course_first_rate": pl.Float64, "course_3rd_rate": pl.Float64, "course_avg_st": pl.Float64,
            "motor_rate2": pl.Float64, "boat_rate2": pl.Float64,
            "wind_speed_m": pl.Float64, "wave_height_cm": pl.Float64,
            "dash_attack_flag": pl.Int64, "is_strong_wind": pl.Int64, "is_crosswind": pl.Int64,
        }
        for c, t in cast_map.items():
            if c in df6.columns:
                df6 = df6.with_columns(pl.col(c).cast(t, strict=False))

        # （レース単位に処理）→ 本番ST（ターゲット）注入
        out_feat_parts: List[pl.DataFrame] = []
        out_tgt_parts:  List[pl.DataFrame] = []
        out_120_parts:  List[pl.DataFrame] = []

        for (hd, jcd, rno), grp in df6.groupby(["hd","jcd","rno"]):
            # リザルトJSON
            year = str(hd)[:4]; mmdd = str(hd)[4:8]
            rjson = results_root / year / mmdd / str(jcd).zfill(2) / f"{int(rno)}R.json"
            rjs = load_json(rjson) or {}

            st_map, decision, trifecta = extract_race_targets(rjs)

            # features_lane
            feat_cols = [
                "hd","jcd","rno","lane",
                "racer_id","grade",
                "tenji_st","st_rank","tenji_rank",
                "course_first_rate","course_3rd_rate","course_avg_st",
                "motor_rate2","boat_rate2",
                "power_lane","power_inner","power_outer","outer_over_inner",
                "st_diff_4_vs_3","st_diff_5_vs_4","st_diff_6_vs_5",
                "dash_attack_flag","is_strong_wind","is_crosswind",
                "weather_sky","wind_dir","wind_speed_m","wave_height_cm",
                "label_3t"
            ]
            feat_cols = [c for c in feat_cols if c in grp.columns]
            feat_df = grp.select(feat_cols)

            # targets_st
            # laneごとに results.start の本番STを付与
            if st_map:
                add = []
                for rec in grp.iter_rows(named=True):
                    lane = int(rec["lane"])
                    target_st = st_map.get(lane)
                    add.append({"hd":hd,"jcd":jcd,"rno":rno,"lane":lane,"target_st":target_st})
                tgt_df = pl.DataFrame(add)
            else:
                # リザルトが無い/読めない場合はスキップ（空で落とす）
                tgt_df = pl.DataFrame({"hd":[hd],"jcd":[jcd],"rno":[rno],"lane":[None],"target_st":[None]}).head(0)

            # 120通り（将来の三連単用に用意）
            exp120 = expand_120_from_df6(grp)

            out_feat_parts.append(feat_df)
            out_tgt_parts.append(tgt_df)
            if not exp120.is_empty():
                out_120_parts.append(exp120)

        # 結合
        features_lane = pl.concat([p for p in out_feat_parts if not p.is_empty()], how="vertical") if out_feat_parts else pl.DataFrame()
        targets_st    = pl.concat([p for p in out_tgt_parts  if not p.is_empty()], how="vertical") if out_tgt_parts  else pl.DataFrame()
        expanded_120  = pl.concat([p for p in out_120_parts  if not p.is_empty()], how="vertical") if out_120_parts  else pl.DataFrame()

        # 出力
        day_out = out_root / str(hd_hint)[:4] / str(hd_hint)[4:8]
        if not features_lane.is_empty():
            write_tabular(features_lane, day_out / f"features_lane.{ 'parquet' if fmt=='parquet' else 'csv' }", fmt)
        else:
            print(f"[WARN] no features to write for {csv_path}")

        if not targets_st.is_empty():
            write_tabular(targets_st, day_out / f"targets_st.{ 'parquet' if fmt=='parquet' else 'csv' }", fmt)
        else:
            print(f"[WARN] no targets to write for {csv_path}")

        if not expanded_120.is_empty():
            write_tabular(expanded_120, day_out / f"expanded_120.{ 'parquet' if fmt=='parquet' else 'csv' }", fmt)
        else:
            print(f"[INFO] no expanded_120 for {csv_path} (OK)")

    print("[DONE] staging build completed.")


if __name__ == "__main__":
    main()
