#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable

import polars as pl

# ====== 正規表現 ======
RE_DAYDIR = re.compile(r"^\d{4}/\d{4}$")          # e.g. 2024/0101
RE_TENJI_F = re.compile(r"^0\.\d{3,}$")           # "0.0004" 等 → 展示Fの符号
RE_ST_F    = re.compile(r"^[Ff][\.\-]?\s*(\d{1,2})$")  # "F.04" / "f04" → 本番F

# ====== ユーティリティ ======
def _safe_float(x, default=None) -> Optional[float]:
    try:
        if x is None or x == "":
            return default
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        # "F.04" のような表示はここでは扱わない（本番ST解析側で扱う）
        s = s.replace("％", "").replace("%", "").replace(",", "")
        return float(s)
    except Exception:
        return default

def _rank_dense(expr: pl.Expr, ascending=True) -> pl.Expr:
    # Polars 0.20 以降: nulls_lastは引数無し。Noneは自動で末尾になる
    return expr.rank(method="dense", descending=not ascending)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _iter_integrated_files(shards_root: Path) -> Iterable[Tuple[str, Path]]:
    """data/shards/YYYY/MMDD/integrated_pro.csv を見つけて (YYYY/MMDD, path) を返す"""
    for p in sorted(shards_root.rglob("integrated_pro.csv")):
        # 期待: .../data/shards/YYYY/MMDD/integrated_pro.csv
        try:
            md = p.parent.name      # "MMDD"
            year = p.parent.parent.name  # "YYYY"
            key = f"{year}/{md}"
            if RE_DAYDIR.match(key):
                yield key, p
        except Exception:
            continue

def _read_integrated_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    # 型を強制（混在エラー対策）
    casts = {
        "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64,
        "racer_id": pl.Utf8, "regno": pl.Utf8 if "regno" in df.columns else pl.Utf8,
        "tenji_st": pl.Float64,  # 0.0004等が入っている可能性
    }
    use_cols = [c for c in [
        "hd","jcd","rno","lane","racer_id","regno","tenji_st",
        "course_avg_st","course_first_rate","course_3rd_rate",
        "label_3t",
    ] if c in df.columns]
    df = df.select(use_cols).with_columns([pl.col(k).cast(v, strict=False) for k, v in casts.items() if k in use_cols])
    # regno 補完（racer_idが標準名）
    if "regno" not in use_cols and "racer_id" in df.columns:
        df = df.with_columns(pl.col("racer_id").alias("regno"))
    return df

def _load_result_start_map(results_root: Path, day_key: str) -> Dict[Tuple[str,int], Dict[int, Dict[str, Optional[float]]]]:
    """
    リザルトから本番STを読み、(jcd, rno) -> { lane -> {st_sec, is_f, is_late, penalized, observed} } のマップを返す。
    - ST が "F.04" 型なら is_f=1, st_sec=None とし、超過は保持しない（ここは生STの正規化だけ）。
    - ST が数値なら st_sec に入れ、is_f=0。
    """
    year, md = day_key.split("/")
    day_root = results_root / year / md
    out: Dict[Tuple[str,int], Dict[int, Dict[str, Optional[float]]]] = {}
    if not day_root.exists():
        return out

    for jcd_dir in sorted(day_root.iterdir()):
        if not jcd_dir.is_dir():
            continue
        jcd = jcd_dir.name
        for f in sorted(jcd_dir.glob("*R.json")):
            m = re.search(r"(\d+)R\.json$", f.name)
            if not m:
                continue
            rno = int(m.group(1))
            try:
                js = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue
            # start 配列を優先（フォーマットが安定）
            start = js.get("start") or []
            lane_map: Dict[int, Dict[str, Optional[float]]] = {}
            for s in start:
                try:
                    lane = int(s.get("lane"))
                except Exception:
                    continue
                st_raw = s.get("st")
                st_info = _parse_race_st(st_raw)
                lane_map[lane] = st_info
            # fall back: results[].st から補完
            if not lane_map:
                for r in js.get("results", []):
                    try:
                        lane = int(r.get("lane"))
                    except Exception:
                        continue
                    st_raw = r.get("st")
                    st_info = _parse_race_st(st_raw)
                    if lane not in lane_map:
                        lane_map[lane] = st_info
            if lane_map:
                out[(jcd, rno)] = lane_map
    return out

def _parse_race_st(st_raw) -> Dict[str, Optional[float]]:
    """
    本番ST表記を正規化。
    - 数値: 0.05 → st_sec=0.05, is_f=0, is_late=0, penalized=0, observed=1
    - "F.04" / "f04": is_f=1, st_sec=None, observed=1
    - "L.07" / "l07": is_late=1, st_sec=None, observed=1
    - それ以外/欠損: observed=0
    """
    s = str(st_raw).strip() if st_raw is not None else ""
    if s == "":
        return {"st_sec": None, "is_f": 0, "is_late": 0, "penalized": 0, "observed": 0}
    # F/L表記
    m = RE_ST_F.match(s.replace(" ", "").replace("\u3000", ""))
    if m:
        # 明示的なF.xx表記 → F扱い
        return {"st_sec": None, "is_f": 1, "is_late": 0, "penalized": 1, "observed": 1}
    if s[0] in ("L","l"):
        return {"st_sec": None, "is_f": 0, "is_late": 1, "penalized": 1, "observed": 1}
    # それ以外は数値解釈トライ
    val = _safe_float(s, default=None)
    if val is not None:
        return {"st_sec": float(val), "is_f": 0, "is_late": 0, "penalized": 0, "observed": 1}
    # とれなかった
    return {"st_sec": None, "is_f": 0, "is_late": 0, "penalized": 0, "observed": 0}

def _normalize_tenji_st(df: pl.DataFrame) -> pl.DataFrame:
    """
    展示STの特殊値（0.0004 等）を：
      - tenji_is_f=1
      - tenji_f_over_sec=0.04 のように復元（末尾2桁 / 100）
      - tenji_st_sec=None（ランクから除外）
    通常の展示値（0.05 等）は tenji_st_sec にそのまま入れ、tenji_is_f=0。
    """
    s = pl.col("tenji_st").cast(pl.Utf8)

    is_f = s.str.contains(RE_TENJI_F.pattern)

    frac = (
        s.str.replace(r"^0\.", "")
         .str.slice(-2)
         .cast(pl.Int64, strict=False)
         .cast(pl.Float64, strict=False) / 100.0
    )

    tenji_f_over_sec = pl.when(is_f).then(frac).otherwise(None).alias("tenji_f_over_sec")
    tenji_is_f = is_f.cast(pl.Int8).alias("tenji_is_f")
    tenji_st_sec = (
        pl.when(is_f | pl.col("tenji_st").is_null())
          .then(None)
          .otherwise(pl.col("tenji_st").cast(pl.Float64, strict=False))
          .alias("tenji_st_sec")
    )

    df = df.with_columns([tenji_st_sec, tenji_is_f, tenji_f_over_sec])
    # 展示ランク：同一レース内で None は自動で後方扱い
    df = df.with_columns([
        _rank_dense(pl.col("tenji_st_sec")).over(["hd","jcd","rno"]).alias("tenji_rank")
    ])
    return df

def _attach_race_st(df: pl.DataFrame, start_map: Dict[Tuple[str,int], Dict[int, Dict[str, Optional[float]]]]) -> pl.DataFrame:
    # (jcd,rno,lane) に join するためキー列を作る
    df = df.with_columns([
        pl.col("jcd").cast(pl.Utf8).alias("_j"),
        pl.col("rno").cast(pl.Int64).alias("_r"),
        pl.col("lane").cast(pl.Int64).alias("_l"),
    ])
    # マップを展開して DataFrame 化
    rows = []
    for (jcd, rno), lanes in start_map.items():
        for lane, info in lanes.items():
            rows.append({
                "_j": jcd, "_r": int(rno), "_l": int(lane),
                "st_sec": info.get("st_sec"),
                "st_is_f": int(info.get("is_f") or 0),
                "st_is_late": int(info.get("is_late") or 0),
                "st_penalized": int(info.get("penalized") or 0),
                "st_observed": int(info.get("observed") or 0),
            })
    if not rows:
        # 何もなければ空列を付ける
        return df.with_columns([
            pl.lit(None, dtype=pl.Float64).alias("st_sec"),
            pl.lit(0, dtype=pl.Int8).alias("st_is_f"),
            pl.lit(0, dtype=pl.Int8).alias("st_is_late"),
            pl.lit(0, dtype=pl.Int8).alias("st_penalized"),
            pl.lit(0, dtype=pl.Int8).alias("st_observed"),
        ])

    st_df = pl.DataFrame(rows)
    df = df.join(st_df, on=["_j","_r","_l"], how="left")
    df = df.drop(["_j","_r","_l"])
    # STランク（観測があるものだけでランク）
    df = df.with_columns([
        pl.when(pl.col("st_observed")==1).then(pl.col("st_sec")).otherwise(None).alias("_st_rank_base")
    ])
    df = df.with_columns([
        _rank_dense(pl.col("_st_rank_base")).over(["hd","jcd","rno"]).alias("st_rank")
    ]).drop(["_st_rank_base"])
    return df

def process_one_day(day_key: str, integrated_path: Path, results_root: Path, out_root: Path, out_format: str):
    print(f"[INFO] processing {day_key} from {integrated_path}")
    df6 = _read_integrated_csv(integrated_path)

    # 展示STの正規化 & tenji_* 列生成
    df6 = _normalize_tenji_st(df6)

    # 本番STのマッピング
    start_map = _load_result_start_map(results_root, day_key)
    df6 = _attach_race_st(df6, start_map)

    # 出力列を確定
    cols = [c for c in [
        "hd","jcd","rno","lane",
        # regno 優先、なければ racer_id から作成済み
        "regno",
        # 本番 ST
        "st_sec","st_is_f","st_is_late","st_penalized","st_observed",
        # 展示 ST（正規化）
        "tenji_st_sec","tenji_is_f","tenji_f_over_sec","tenji_rank",
        # 本番 ST ランク
        "st_rank",
        # コース別指標
        "course_avg_st","course_first_rate","course_3rd_rate",
    ] if c in df6.columns]

    out_df = df6.select(cols)

    # 出力先: data/staging/YYYY/MMDD/st_train.csv
    year, md = day_key.split("/")
    out_dir = out_root / year / md
    _ensure_dir(out_dir)
    out_path = out_dir / "st_train.csv"

    if out_format in ("csv","both"):
        out_df.write_csv(out_path)
        print(f"[WRITE] {out_path} (rows={out_df.height})")
    if out_format in ("parquet","both"):
        out_df.write_parquet(out_dir / "st_train.parquet")
        print(f"[WRITE] {out_dir/'st_train.parquet'} (rows={out_df.height})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards_root", default="data/shards")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--out_root", default="data/staging")
    ap.add_argument("--out_format", default="csv", choices=["csv","parquet","both"])
    args = ap.parse_args()

    shards_root = Path(args.shards_root)
    results_root = Path(args.results_root)
    out_root = Path(args.out_root)

    files = list(_iter_integrated_files(shards_root))
    if not files:
        print(f"[WARN] no integrated_pro.csv under {shards_root}")
        return

    print(f"[INFO] found {len(files)} integrated_pro.csv files")
    for day_key, ipath in files:
        print(f"[INFO] reading {ipath}")
        process_one_day(day_key, ipath, results_root, out_root, args.out_format)

if __name__ == "__main__":
    main()
