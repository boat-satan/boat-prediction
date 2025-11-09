#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_staging_from_integrated.py
- 入力: data/shards/YYYY/MMDD/integrated_pro.csv
- 公式結果: public/results/YYYY/MMDD/JC/NR.json
- 出力: data/staging/YYYY/MMDD/st_train.csv

出力列:
  hd,jcd,rno,lane,regno,
  st_sec,st_is_f,st_is_late,st_penalized,st_observed,
  tenji_st_sec,tenji_is_f,tenji_f_over_sec,
  tenji_rank,st_rank,
  course_avg_st,course_first_rate,course_3rd_rate
"""
from __future__ import annotations
from pathlib import Path
import argparse, csv, json
from typing import Dict, Tuple, List, Any, Optional

import polars as pl


# ---------- helpers ----------
def z2(x) -> str:
    try:
        return f"{int(str(x).strip()):02d}"
    except Exception:
        s = str(x).strip()
        return s if len(s) >= 2 else s.zfill(2)


def parse_official_st_text(sttxt) -> Tuple[Optional[float], int, int, int, int]:
    """
    公式 ST テキスト正規化:
      '0.13' → (0.13,0,0,0,1)
      'F.04' → (0.04,1,0,1,1)
      'L.02' → (0.02,0,1,1,1)
      None/空 → (None,0,0,0,0)
    """
    s = "" if sttxt is None else str(sttxt).strip()
    if s == "":
        return (None, 0, 0, 0, 0)

    up = s.upper()
    is_f = 1 if up.startswith("F") else 0
    is_l = 1 if up.startswith("L") else 0
    penal = 1 if (is_f or is_l) else 0

    # 'F.04' → '.04'
    num = up.replace("F", "").replace("L", "").strip()
    if num.startswith("."):
        num = "0" + num

    try:
        v = float(num)
    except Exception:
        return (None, 0, 0, 0, 0)

    return (v, is_f, is_l, penal, 1)


def load_official_results(results_root: Path, hd: str, jcd: str, rno: int
                          ) -> Dict[int, Tuple[Optional[float], int, int, int, int]]:
    """
    lane -> (st_sec, st_is_f, st_is_late, st_penalized, st_observed)
    優先度: results[].st → start[].st（未観測のみ補完）
    """
    year, md = hd[:4], hd[4:8]
    jj = z2(jcd)
    p = results_root / year / md / jj / f"{rno}R.json"
    if not p.exists():
        return {}

    try:
        js = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

    out: Dict[int, Tuple[Optional[float], int, int, int, int]] = {}

    for it in (js.get("results") or []):
        try:
            lane = int(str(it.get("lane")).strip())
        except Exception:
            continue
        out[lane] = parse_official_st_text(it.get("st"))

    for it in (js.get("start") or []):
        try:
            lane = int(str(it.get("lane")).strip())
        except Exception:
            continue
        if lane in out and out[lane][4] == 1:  # already observed from results
            continue
        out[lane] = parse_official_st_text(it.get("st"))

    return out


def normalize_exhibition_st(val) -> Tuple[Optional[float], int, Optional[float]]:
    """
    展示STの元データは F.04 → 0.0004 のように格納されている想定。
    - 0 < val < 0.01 → F扱い: tenji_is_f=1, tenji_f_over_sec=val*100, tenji_st_sec=None
    - val >= 0.01   → 通常: tenji_is_f=0, tenji_f_over_sec=None, tenji_st_sec=val
    - None/その他   → 欠損
    """
    if val is None:
        return (None, 0, None)
    try:
        x = float(val)
    except Exception:
        return (None, 0, None)

    if x > 0.0 and x < 0.01:
        return (None, 1, round(x * 100.0, 3))  # 0.0004 → 0.04
    elif x >= 0.01:
        return (x, 0, None)
    else:
        return (None, 0, None)


def dense_rank_asc(values: List[Optional[float]]) -> List[int]:
    """
    None は最後に回す昇順 dense rank。最小=1。
    """
    # sentinel: None -> +inf
    keyed = [(float("inf") if v is None else float(v), i) for i, v in enumerate(values)]
    keyed_sorted = sorted(keyed, key=lambda t: t[0])

    ranks = [0] * len(values)
    cur_rank = 0
    last_val = None
    for (val, idx) in keyed_sorted:
        if last_val is None or val != last_val:
            cur_rank += 1
            last_val = val
        ranks[idx] = cur_rank
    return ranks


# ---------- core processing ----------
def process_one_day(integrated_csv: Path,
                    results_root: Path,
                    out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 読み込み（eagerでOK）
    df = pl.read_csv(integrated_csv, infer_schema_length=2000)

    # 必要列チェック & 補完
    req_cols = [
        "hd","jcd","rno","lane","racer_id",
        "tenji_st","st_rank",  # st_rank は展示の相対順位（元データ）。再計算は tenji_rank で行う。
        "course_avg_st","course_first_rate","course_3rd_rate",
    ]
    for c in req_cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))

    # 型整備
    df = df.with_columns([
        pl.col("hd").cast(pl.Utf8, strict=False),
        pl.col("jcd").cast(pl.Utf8, strict=False),
        pl.col("rno").cast(pl.Int64, strict=False),
        pl.col("lane").cast(pl.Int64, strict=False),
        pl.col("racer_id").cast(pl.Utf8, strict=False),
        pl.col("tenji_st").cast(pl.Float64, strict=False),
        pl.col("st_rank").cast(pl.Int64, strict=False),
        pl.col("course_avg_st").cast(pl.Float64, strict=False),
        pl.col("course_first_rate").cast(pl.Float64, strict=False),
        pl.col("course_3rd_rate").cast(pl.Float64, strict=False),
    ])

    # jcd をゼロパディング
    df = df.with_columns(pl.col("jcd").map_elements(z2).alias("jcd"))

    # 組ごとに処理
    groups = df.group_by(["hd","jcd","rno"], maintain_order=True).agg(pl.all())

    # 収集行
    out_rows: List[Dict[str, Any]] = []

    for rec in groups.iter_rows(named=True):
        hd = rec["hd"]; jcd = rec["jcd"]; rno = int(rec["rno"])
        sub = pl.DataFrame({k: rec[k] for k in df.columns})

        # 公式ST読み込み（results優先→start補完）
        st_map = load_official_results(results_root, str(hd), str(jcd), rno)

        # 展示STの正規化（F.04→0.0004 をここで戻す）
        tenji_vals: List[Optional[float]] = []
        for lane_val, tenji_st_val in zip(sub["lane"].to_list(), sub["tenji_st"].to_list()):
            tenji_st_sec, tenji_is_f, tenji_f_over_sec = normalize_exhibition_st(tenji_st_val)
            tenji_vals.append(tenji_st_sec)
            # 仮で格納（後で row 作る時にまた計算して入れる）
            # ここでは rank 用だけ先に欲しい
            pass

        # tenji_rank（Noneは最後）
        tenji_rank_list = dense_rank_asc(tenji_vals)

        # st_rank（公式STで再ランキング。欠損は最後）
        # まず lane順に st を並べる
        lanes: List[int] = [int(x) for x in sub["lane"].to_list()]
        st_vals: List[Optional[float]] = []
        st_meta: Dict[int, Tuple[Optional[float], int, int, int, int]] = {}
        for lane in lanes:
            tpl = st_map.get(lane, (None,0,0,0,0))
            st_meta[lane] = tpl
            st_vals.append(tpl[0])
        st_rank_list = dense_rank_asc(st_vals)

        # レコード化
        for i in range(len(sub)):
            lane = int(sub["lane"][i])
            regno_raw = sub["racer_id"][i]
            try:
                regno = int(str(regno_raw)) if regno_raw is not None else None
            except Exception:
                regno = None

            tenji_st_val = sub["tenji_st"][i]
            tenji_st_sec, tenji_is_f, tenji_f_over_sec = normalize_exhibition_st(tenji_st_val)

            st_sec, st_is_f, st_is_late, st_penalized, st_observed = st_meta.get(lane, (None,0,0,0,0))

            row = {
                "hd": str(hd),
                "jcd": str(jcd),
                "rno": rno,
                "lane": lane,
                "regno": regno,
                "st_sec": st_sec,
                "st_is_f": st_is_f,
                "st_is_late": st_is_late,
                "st_penalized": st_penalized,
                "st_observed": st_observed,
                "tenji_st_sec": tenji_st_sec,
                "tenji_is_f": tenji_is_f,
                "tenji_f_over_sec": tenji_f_over_sec,
                "tenji_rank": int(tenji_rank_list[i]),
                "st_rank": int(st_rank_list[i]),
                "course_avg_st": sub["course_avg_st"][i],
                "course_first_rate": sub["course_first_rate"][i],
                "course_3rd_rate": sub["course_3rd_rate"][i],
            }
            out_rows.append(row)

    if not out_rows:
        print(f"[WARN] no rows for {integrated_csv}")
        return

    # 出力（YYYY/MMDD で保存）
    # integrated_pro.csv の親ディレクトリから日付を特定
    # shards_root/YYYY/MMDD/integrated_pro.csv
    parts = integrated_csv.parts
    # .../shards/YYYY/MMDD/integrated_pro.csv
    try:
        md = parts[-2]  # MMDD
        yyyy = parts[-3]  # YYYY
    except Exception:
        # フォールバック: レコード先頭から抽出
        hd0 = out_rows[0]["hd"]
        yyyy, md = hd0[:4], hd0[4:8]

    day_out_dir = out_dir / str(yyyy) / str(md)
    day_out_dir.mkdir(parents=True, exist_ok=True)

    out_path = day_out_dir / "st_train.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=[
            "hd","jcd","rno","lane","regno",
            "st_sec","st_is_f","st_is_late","st_penalized","st_observed",
            "tenji_st_sec","tenji_is_f","tenji_f_over_sec",
            "tenji_rank","st_rank",
            "course_avg_st","course_first_rate","course_3rd_rate",
        ])
        w.writeheader()
        w.writerows(out_rows)

    print(f"[WRITE] {out_path} (rows={len(out_rows)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_root", default="data/shards")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--out_root", default="data/staging")
    ap.add_argument("--out_format", default="csv", choices=["csv"])  # 現状csvのみ
    args = ap.parse_args()

    shards_root = Path(args.shards_root)
    results_root = Path(args.results_root)
    out_root = Path(args.out_root)

    files = sorted(shards_root.glob("**/integrated_pro.csv"))
    print(f"[INFO] found {len(files)} integrated_pro.csv files")

    for p in files:
        print(f"[INFO] reading {p}")
        try:
            process_one_day(p, results_root, out_root)
        except Exception as e:
            print(f"[ERROR] {p}: {e}")


if __name__ == "__main__":
    main()
