#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boat-ml Pro統合パイプライン（出走表/展示/結果/選手年次を統合 → 6行 → 120行展開）
- 選手個人データは「対象レースの前年」を参照（例: 2024レース → 2023フォルダ）
- 出力:
    data/integrated_pro.parquet   # レース×艇（6行/レース、特徴入り）
    data/train_120_pro.parquet    # 三連単120行（is_hitラベル付き）
"""
from __future__ import annotations
import argparse, itertools, json, re
from pathlib import Path
import polars as pl

RE_DATE = re.compile(r"^(\d{4})(\d{2})(\d{2})$")

def jload(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def fnum(x, default=None):
    try:
        if x is None: return default
        if isinstance(x, (int, float)): return float(x)
        s = str(x).replace("％","").replace("%","").replace(",","").strip()
        if s == "": return default
        return float(s)
    except Exception:
        return default

def parse_triplet(s: str|None):
    # "6.46 50.94 65.09" → (6.46, 50.94, 65.09)
    if not s: return (None, None, None)
    parts = [p for p in str(s).split() if p]
    vals = [fnum(p) for p in parts[:3]]
    while len(vals) < 3: vals.append(None)
    return tuple(vals)

def rank_dense_compat(col: pl.Expr, asc=True):
    """
    Polarsのrank互換：nullを末尾扱いにするため、フィラーを入れてからrank("dense")。
    - 昇順: nullは非常に大きい値で埋めて末尾へ
    - 降順: nullは非常に小さい値で埋めて末尾へ
    """
    filler = 1e18 if asc else -1e18
    return col.fill_null(filler).rank(method="dense", descending=not asc)

def integrate_one(program_js: dict, exhibition_js: dict|None,
                  result_js: dict|None, racer_root: Path) -> pl.DataFrame:
    # keys
    hd = program_js.get("date")
    jcd = program_js.get("pid")
    rno = int(str(program_js.get("race","1R")).replace("R",""))
    # meta（result優先）
    meta = (result_js or {}).get("meta", {})
    weather_sky   = meta.get("weather_sky")
    wind_dir      = meta.get("wind_dir")
    wind_speed_m  = meta.get("wind_speed_m")
    wave_height_cm= meta.get("wave_height_cm")
    title         = meta.get("title")
    decision      = meta.get("decision")  # 学習には使わない

    # 出走（6艇）
    rows = []
    for e in program_js.get("entries", []):
        lane      = int(e.get("lane"))
        racer_id  = str(e.get("number")) if e.get("number") is not None else None
        name      = e.get("name")
        grade     = e.get("grade")
        age       = e.get("age")

        nat_win, nat_2r, nat_3r = parse_triplet((e.get("stats") or {}).get("national"))
        loc_win, loc_2r, loc_3r = parse_triplet((e.get("stats") or {}).get("local"))
        m_id, m2, m3            = parse_triplet((e.get("stats") or {}).get("motor"))
        b_id, b2, b3            = parse_triplet((e.get("stats") or {}).get("boat"))

        # 展示（lane一致）
        ex = None
        if exhibition_js:
            for exi in exhibition_js.get("entries", []):
                if int(exi.get("lane")) == lane:
                    ex = exi; break
        wkg = tdeg = tenji_sec = tenji_st = isF = None
        if ex:
            nz = ex.get("normalized", {})
            wkg       = nz.get("weightKg")
            tdeg      = nz.get("tiltDeg")
            tenji_sec = nz.get("tenjiSec")
            tenji_st  = nz.get("stSec")
            isF       = nz.get("isF")

        # 「前年」の選手年次を読む（2024レース→2023）
        m = RE_DATE.match(hd or "")
        cur_year = int(m.group(1)) if m else 2024
        prev_year = cur_year - 1
        c1st=c3rd=cavg=kim_m=kim_s=kim_ms=kim_n=None
        if racer_root and racer_id:
            rp = racer_root / str(prev_year) / f"{racer_id}.json"
            if rp.exists():
                js = jload(rp)
                lane_key = str(lane)
                cs = (js.get("course_stats") or {}).get(lane_key) or {}
                km = (js.get("course_kimarite") or {}).get(lane_key) or {}
                c1st   = fnum(cs.get("1着率"))
                c3rd   = fnum(cs.get("3連対率"))
                cavg   = fnum(cs.get("平均ST"))
                kim_m  = fnum(km.get("まくり"))
                kim_s  = fnum(km.get("差し"))
                kim_ms = fnum(km.get("まくり差し"))
                kim_n  = fnum(km.get("抜き"))

        rows.append(dict(
            hd=hd, jcd=jcd, rno=rno, lane=lane,
            racer_id=racer_id, racer_name=name, grade=grade, age=age,
            weather_sky=weather_sky, wind_dir=wind_dir,
            wind_speed_m=wind_speed_m, wave_height_cm=wave_height_cm,
            title=title, decision=decision,  # 保持のみ（学習投入禁止）
            national_win=nat_win, national_2r=nat_2r, national_3r=nat_3r,
            local_win=loc_win, local_2r=loc_2r, local_3r=loc_3r,
            motor_id=m_id, motor_rate2=m2, motor_rate3=m3,
            boat_id=b_id,  boat_rate2=b2,  boat_rate3=b3,
            weightKg=wkg, tiltDeg=tdeg, tenji_sec=tenji_sec,
            tenji_st=tenji_st, isF_tenji=isF,
            course_first_rate=c1st, course_3rd_rate=c3rd, course_avg_st=cavg,
            kimarite_makuri=kim_m, kimarite_sashi=kim_s,
            kimarite_makuri_sashi=kim_ms, kimarite_nuki=kim_n,
        ))

    df = pl.DataFrame(rows)

    if not df.is_empty():
        df = df.with_columns([
            rank_dense_compat(pl.col("tenji_sec"), asc=True).alias("tenji_rank"),
            rank_dense_compat(pl.col("tenji_st"),  asc=True).alias("st_rank"),
        ])
        df = df.with_columns([
            (pl.col("national_win").fill_null(0)*0.6 +
             pl.col("motor_rate2").fill_null(0)*0.3 +
             (5 - pl.col("st_rank").fill_null(3))*0.1).alias("power_lane")
        ])
        inner = df.filter(pl.col("lane")<=3)["power_lane"].mean()
        outer = df.filter(pl.col("lane")>=4)["power_lane"].mean()
        df = df.with_columns([
            pl.lit(float(inner) if inner is not None else None).alias("power_inner"),
            pl.lit(float(outer) if outer is not None else None).alias("power_outer"),
            (pl.lit(float(outer) if outer is not None else 0) -
             pl.lit(float(inner) if inner is not None else 0)).alias("outer_over_inner"),
        ])

        # ST相対差（4→3, 5→4, 6→5）
        def st_of(l):
            try:
                return float(df.filter(pl.col("lane")==l)["tenji_st"][0])
            except Exception:
                return None
        st3, st4, st5, st6 = st_of(3), st_of(4), st_of(5), st_of(6)
        df = df.with_columns([
            pl.lit((st4-st3) if (st4 is not None and st3 is not None) else None).alias("st_diff_4_vs_3"),
            pl.lit((st5-st4) if (st5 is not None and st4 is not None) else None).alias("st_diff_5_vs_4"),
            pl.lit((st6-st5) if (st6 is not None and st5 is not None) else None).alias("st_diff_6_vs_5"),
        ])
        df = df.with_columns([
            ((pl.col("lane")==4) & (pl.col("st_diff_4_vs_3")<=-0.02) &
             (pl.col("tenji_st")<=0.10)).cast(pl.Int8).alias("dash_attack_flag")
        ])
        df = df.with_columns([
            (pl.col("wind_speed_m")>=6).cast(pl.Int8).alias("is_strong_wind"),
            (pl.col("wind_dir").is_in([2,3,6,7])).cast(pl.Int8).alias("is_crosswind"),
        ])

    # ラベル（的中三連単）
    label_3t = None
    if result_js and result_js.get("payouts", {}).get("trifecta"):
        label_3t = result_js["payouts"]["trifecta"].get("combo")
    df = df.with_columns([
        pl.lit(hd).alias("hd"), pl.lit(jcd).alias("jcd"),
        pl.lit(rno).alias("rno"), pl.lit(label_3t).alias("label_3t"),
    ])
    return df

def expand_120(df6: pl.DataFrame) -> pl.DataFrame:
    if df6.is_empty(): return pl.DataFrame()
    hd, jcd, rno = df6["hd"][0], df6["jcd"][0], df6["rno"][0]
    label = df6["label_3t"][0]
    recs = {int(r["lane"]): r for r in df6.iter_rows(named=True)}
    combos = [(a,b,c) for a,b,c in itertools.permutations([1,2,3,4,5,6], 3)]
    rows=[]
    for a,b,c in combos:
        r1,r2,r3 = recs.get(a,{}), recs.get(b,{}), recs.get(c,{})
        combo = f"{a}-{b}-{c}"
        is_hit = 1 if (label==combo) else 0
        rows.append(dict(
            hd=hd, jcd=jcd, rno=rno, combo=combo, is_hit=is_hit,
            # 1着
            p1_lane=a, p1_grade=r1.get("grade"),
            p1_national_win=r1.get("national_win"),
            p1_motor_rate2=r1.get("motor_rate2"),
            p1_tenji_st=r1.get("tenji_st"),
            p1_tenji_rank=r1.get("tenji_rank"),
            p1_st_rank=r1.get("st_rank"),
            p1_course_first_rate=r1.get("course_first_rate"),
            # 2着
            p2_lane=b, p2_grade=r2.get("grade"),
            p2_national_win=r2.get("national_win"),
            p2_motor_rate2=r2.get("motor_rate2"),
            p2_tenji_st=r2.get("tenji_st"),
            p2_tenji_rank=r2.get("tenji_rank"),
            p2_st_rank=r2.get("st_rank"),
            # 3着
            p3_lane=c, p3_grade=r3.get("grade"),
            p3_motor_rate2=r3.get("motor_rate2"),
            p3_tenji_st=r3.get("tenji_st"),
            p3_tenji_rank=r3.get("tenji_rank"),
            p3_st_rank=r3.get("st_rank"),
            # 相互作用
            diff_p1p2_st=_safediff(r1.get("tenji_st"), r2.get("tenji_st")),
            sum_p1p2_motor=_safesum(r1.get("motor_rate2"), r2.get("motor_rate2")),
            outer_push=1 if (a in (4,5,6) and r1.get("dash_attack_flag")==1) else 0,
            in_cover=1 if (b==1 and (recs.get(2,{}).get("tenji_st") is not None)) else 0,
            # 環境（共通）
            wind_speed_m=r1.get("wind_speed_m"),
            wave_height_cm=r1.get("wave_height_cm"),
            is_strong_wind=r1.get("is_strong_wind"),
            is_crosswind=r1.get("is_crosswind"),
        ))
    return pl.DataFrame(rows)

def _safediff(a,b):
    try:
        if a is None or b is None: return None
        return float(a)-float(b)
    except Exception:
        return None

def _safesum(a,b):
    try:
        return (0.0 if a is None else float(a)) + (0.0 if b is None else float(b))
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program_dir",   default="public/programs/v1")
    ap.add_argument("--exhibition_dir",default="public/exhibition/v1")
    ap.add_argument("--results_dir",   default="public/results")
    ap.add_argument("--racer_dir",     default="public/racers-annual")
    ap.add_argument("--out_dir",       default="data")
    args = ap.parse_args()

    prog_root = Path(args.program_dir)
    exh_root  = Path(args.exhibition_dir)
    res_root  = Path(args.results_dir)
    racer_root= Path(args.racer_dir)
    out_dir   = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    prog_files = sorted(prog_root.glob("**/*.json"))
    integrated_parts, expanded_parts = [], []

    for p in prog_files:
        try:
            pjs = jload(p)
            hd = pjs.get("date"); jcd = pjs.get("pid")
            rno = int(str(pjs.get("race","1R")).replace("R",""))
            # exhibition: public/exhibition/v1/YYYY/MMDD/PP/1R.json
            exh_path = exh_root / hd[:4] / hd[4:8] / jcd / f"{rno}R.json"
            ejs = jload(exh_path) if exh_path.exists() else None
            # results: public/results/YYYY/MMDD/PP/1R.json
            res_path = res_root / hd[:4] / hd[4:8] / jcd / f"{rno}R.json"
            rjs = jload(res_path) if res_path.exists() else None

            df6 = integrate_one(pjs, ejs, rjs, racer_root)
            if not df6.is_empty():
                integrated_parts.append(df6)
                df120 = expand_120(df6)
                if not df120.is_empty():
                    expanded_parts.append(df120)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")

    if integrated_parts:
        pl.concat(integrated_parts).write_parquet(out_dir/"integrated_pro.parquet")
        print(f"[OK] integrated -> {out_dir/'integrated_pro.parquet'}")
    else:
        print("[WARN] no integrated rows.")

    if expanded_parts:
        pl.concat(expanded_parts).write_parquet(out_dir/"train_120_pro.parquet")
        print(f"[OK] 120-expanded -> {out_dir/'train_120_pro.parquet'}")
    else:
        print("[WARN] no 120-expanded rows.")

if __name__ == "__main__":
    main()
