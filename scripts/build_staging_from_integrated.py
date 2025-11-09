# 1) integrated 読み込み後に jcd を 2桁へ
def _read_integrated_csv(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    use_cols = [c for c in [
        "hd","jcd","rno","lane","racer_id","regno","tenji_st",
        "course_avg_st","course_first_rate","course_3rd_rate","label_3t",
    ] if c in df.columns]

    df = df.select(use_cols).with_columns([
        pl.col("hd").cast(pl.Utf8),
        pl.col("jcd").cast(pl.Utf8).str.strip_chars().alias("jcd"),
        pl.col("rno").cast(pl.Int64, strict=False),
        pl.col("lane").cast(pl.Int64, strict=False),
    ])

    # regno 補完
    if "regno" not in df.columns and "racer_id" in df.columns:
        df = df.with_columns(pl.col("racer_id").alias("regno"))

    # ★ jcd を 2桁ゼロ埋め
    df = df.with_columns(pl.col("jcd").str.replace_all(r"^0+", "").alias("jcd"))
    df = df.with_columns(pl.col("jcd").fill_null("").alias("jcd"))
    df = df.with_columns(pl.when(pl.col("jcd") == "")
                         .then(pl.lit(None))
                         .otherwise(pl.col("jcd")).alias("jcd"))
    df = df.with_columns(pl.col("jcd").cast(pl.Int64, strict=False))
    df = df.with_columns(pl.col("jcd").cast(pl.Utf8).str.zfill(2).alias("jcd"))

    return df

# 2) results 側の jcd 取得でも 2桁ゼロ埋めを保証
def _load_result_start_map(results_root: Path, day_key: str) -> Dict[Tuple[str,int], Dict[int, Dict[str, Optional[float]]]]:
    year, md = day_key.split("/")
    day_root = results_root / year / md
    out = {}
    if not day_root.exists():
        return out

    for jcd_dir in sorted(day_root.iterdir()):
        if not jcd_dir.is_dir():
            continue
        # ★ ディレクトリ名を 2桁ゼロ埋め化（"6" → "06"）
        jcd_name = jcd_dir.name.strip()
        try:
            jcd = f"{int(jcd_name):02d}"
        except Exception:
            # もし "06" 形式ならそのまま
            jcd = jcd_name.zfill(2)

        for f in sorted(jcd_dir.glob("*R.json")):
            m = re.search(r"(\d+)R\.json$", f.name)
            if not m:
                continue
            rno = int(m.group(1))
            try:
                js = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue

            lane_map = {}
            for s in (js.get("start") or []):
                try:
                    lane = int(s.get("lane"))
                except Exception:
                    continue
                lane_map[lane] = _parse_race_st(s.get("st"))

            if not lane_map:
                for r in js.get("results", []):
                    try:
                        lane = int(r.get("lane"))
                    except Exception:
                        continue
                    if lane not in lane_map:
                        lane_map[lane] = _parse_race_st(r.get("st"))

            if lane_map:
                out[(jcd, rno)] = lane_map
    return out

# 3) join キーを明示的に 2桁 jcd / int rno,int lane で合わせる
def _attach_race_st(df: pl.DataFrame, start_map):
    df = df.with_columns([
        pl.col("jcd").cast(pl.Utf8).str.zfill(2).alias("_j"),
        pl.col("rno").cast(pl.Int64).alias("_r"),
        pl.col("lane").cast(pl.Int64).alias("_l"),
    ])
    rows = []
    for (jcd, rno), lanes in start_map.items():
        for lane, info in lanes.items():
            rows.append({
                "_j": str(jcd).zfill(2), "_r": int(rno), "_l": int(lane),
                "st_sec": info.get("st_sec"),
                "st_is_f": int(info.get("is_f") or 0),
                "st_is_late": int(info.get("is_late") or 0),
                "st_penalized": int(info.get("penalized") or 0),
                "st_observed": int(info.get("observed") or 0),
            })
    st_df = pl.DataFrame(rows) if rows else pl.DataFrame({
        "_j": [], "_r": [], "_l": [],
        "st_sec": [], "st_is_f": [], "st_is_late": [], "st_penalized": [], "st_observed": [],
    })
    df = df.join(st_df, on=["_j","_r","_l"], how="left").drop(["_j","_r","_l"])

    df = df.with_columns([
        pl.when(pl.col("st_observed")==1).then(pl.col("st_sec")).otherwise(None).alias("_st_rank_base")
    ]).with_columns([
        _rank_dense(pl.col("_st_rank_base")).over(["hd","jcd","rno"]).alias("st_rank")
    ]).drop(["_st_rank_base"])
    return df
