# src/boatpred/io.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import polars as pl


# ========= Path helpers =========
def project_root(from_file: Optional[Path] = None) -> Path:
    """
    推奨: srcレイアウト。通常は本ファイルの2階層上がリポジトリROOTになる想定:
      src/boatpred/io.py -> ROOT
    """
    if from_file is None:
        here = Path(__file__).resolve()
    else:
        here = from_file.resolve()
    return here.parents[2]


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ========= Shards (train_120_pro.*) =========
def find_shards(shards_root: Union[str, Path]) -> List[Path]:
    root = Path(shards_root)
    pq = sorted(root.rglob("train_120_pro.parquet"))
    csv = sorted(root.rglob("train_120_pro.csv"))
    csv_gz = sorted(root.rglob("train_120_pro.csv.gz"))
    return [*pq, *csv, *csv_gz]


def scan_shards(shard_paths: List[Path]) -> pl.LazyFrame:
    if not shard_paths:
        raise FileNotFoundError("No shards under data/shards/**/train_120_pro.(parquet|csv|csv.gz)")
    # 優先順: parquet > csv > csv.gz
    pq = [str(p) for p in shard_paths if p.suffix == ".parquet"]
    if pq:
        return pl.scan_parquet(pq)
    csv = [str(p) for p in shard_paths if p.suffix == ".csv"]
    gz  = [str(p) for p in shard_paths if p.suffixes[-2:] == [".csv", ".gz"]]
    if csv or gz:
        return pl.scan_csv([*csv, *gz], infer_schema_length=10000)
    # 保険
    return pl.scan_parquet([str(shard_paths[0])])


def split_period_lazy(scan: pl.LazyFrame, start: str, end: str) -> pl.LazyFrame:
    # hd は "YYYYMMDD" の文字列として扱う
    return (
        scan.with_columns(pl.col("hd").cast(pl.Utf8))
            .filter(pl.col("hd").is_between(pl.lit(start), pl.lit(end), closed="both"))
    )


# ========= Results (official) =========
def _norm_jcd(jcd: Union[str, int]) -> str:
    try:
        return f"{int(jcd):02d}"
    except Exception:
        return str(jcd)


def _norm_rno(rno: Union[str, int]) -> int:
    try:
        return int(rno)
    except Exception:
        return int(str(rno).replace("R", "").replace("r", ""))


def load_result_json(results_root: Union[str, Path], hd, jcd, rno) -> Optional[dict]:
    y  = str(hd)[:4]
    md = str(hd)[4:8]
    j  = _norm_jcd(jcd)
    r  = _norm_rno(rno)
    p = Path(results_root) / y / md / j / f"{r}R.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_trifecta_amount(results_root: Union[str, Path], hd, jcd, rno) -> Optional[int]:
    """
    3連単 配当金額（円）。見つからなければ None。
    期待キー: obj['payouts']['trifecta']['amount']
    """
    js = load_result_json(results_root, hd, jcd, rno)
    if not js:
        return None
    tri = (js.get("payouts") or {}).get("trifecta")
    if not tri:
        return None
    try:
        return int(str(tri.get("amount")).replace(",", ""))
    except Exception:
        return None


# ========= Odds (3連単) =========
def load_odds_json(odds_root: Union[str, Path], hd, jcd, rno) -> Optional[dict]:
    y  = str(hd)[:4]
    md = str(hd)[4:8]
    j  = _norm_jcd(jcd)
    r  = _norm_rno(rno)
    p = Path(odds_root) / y / md / j / f"{r}R.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def odds_map_from_json(js: dict) -> Optional[Dict[str, float]]:
    """
    入力JSONの多様なスキーマを吸収して { "1-2-3": 6.7, ... } を返す。
    サポート:
      1) {"trifecta": { "1-2-3": 6.7, ... }}
      2) {"trifecta": [ {"combo":"1-2-3","odds":6.7}, ... ]}  ※公開例
      3) {"odds":[{"combo":...,"odd"/"odds"/"value":...}, ...]}
      4) フラット {"1-2-3": "6.7", ...}
    """
    if not isinstance(js, dict):
        return None

    # 1) dict under 'trifecta'
    tri = js.get("trifecta")
    if isinstance(tri, dict):
        out = {}
        for k, v in tri.items():
            if v in (None, ""):
                continue
            try:
                out[str(k)] = float(v)
            except Exception:
                try:
                    out[str(k)] = float(str(v).replace(",", ""))
                except Exception:
                    continue
        if out:
            return out

    # 2) list under 'trifecta'
    if isinstance(tri, list):
        m = {}
        for row in tri:
            if not isinstance(row, dict):
                continue
            combo = row.get("combo") or row.get("combination") or row.get("key")
            odd   = row.get("odds")  or row.get("odd")        or row.get("value")
            if combo is None or odd in (None, ""):
                continue
            try:
                m[str(combo)] = float(str(odd).replace(",", ""))
            except Exception:
                continue
        if m:
            return m

    # 3) 'odds' list
    cand = js.get("odds") or js.get("trifecta_list") or js.get("3t") or js.get("list")
    if isinstance(cand, list):
        m = {}
        for row in cand:
            if not isinstance(row, dict):
                continue
            combo = row.get("combo") or row.get("combination") or row.get("key")
            odd   = row.get("odds")  or row.get("odd")        or row.get("value")
            if combo is None or odd in (None, ""):
                continue
            try:
                m[str(combo)] = float(str(odd).replace(",", ""))
            except Exception:
                continue
        if m:
            return m

    # 4) flat
    flat_keys = [k for k in js.keys() if isinstance(k, str) and "-" in k]
    if flat_keys:
        out = {}
        for k in flat_keys:
            v = js.get(k)
            try:
                out[str(k)] = float(str(v).replace(",", ""))
            except Exception:
                continue
        if out:
            return out

    return None


def load_odds_map(odds_root: Union[str, Path], hd, jcd, rno) -> Optional[Dict[str, float]]:
    js = load_odds_json(odds_root, hd, jcd, rno)
    if not js:
        return None
    return odds_map_from_json(js)


# ========= Writers =========
def write_picks_csv(df: pd.DataFrame, out_path: Union[str, Path]) -> Path:
    p = Path(out_path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False, encoding="utf-8")
    return p


def write_json(obj: dict, out_path: Union[str, Path]) -> Path:
    p = Path(out_path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ========= Convenience (train/evalで共通利用) =========
def collect_period_df(
    shards_root: Union[str, Path],
    start: str,
    end: str,
    engine: str = "streaming",
) -> pl.DataFrame:
    """
    shards_root から train_120_pro.* を見つけ、期間で絞って Collect。
    """
    lf = scan_shards(find_shards(shards_root))
    return split_period_lazy(lf, start, end).collect(engine=engine)


def add_odds_and_ev(
    pred_df: pd.DataFrame,
    odds_root: Union[str, Path],
    rank_key: str = "proba",
    ev_min: float = 0.0,
    ev_drop_if_missing_odds: bool = False,
    force_load: bool = False,
    also_require: Tuple[bool, bool, bool, bool] = (False, False, False, False),
) -> pd.DataFrame:
    """
    予測DFに 'odds' と 'ev=proba*odds' を付与。
    force_load=True か、rank_key=='ev' or ev_min>0 or ev_drop_if_missing_odds がTrueのときロード。
    also_require は追加条件（合成オッズ/平均EV/coverage等の後段が必要な場合）をORで与える。
    """
    need_extra = any(also_require)
    need = (rank_key == "ev") or (ev_min > 0) or ev_drop_if_missing_odds or force_load or need_extra
    if not need:
        return pred_df

    df = pred_df.copy()
    if "combo" not in df.columns:
        raise ValueError("pred_df requires 'combo' column for odds mapping.")
    df["combo"] = df["combo"].astype(str)
    df["odds"] = np.nan

    # まとめてループ（レース単位）
    for (hd, jcd, rno), idxs in df.groupby(["hd", "jcd", "rno"]).groups.items():
        om = load_odds_map(odds_root, hd, jcd, rno)
        if not om:
            continue
        sub = df.loc[idxs]
        mapped = pd.to_numeric(sub["combo"].map(om), errors="coerce")
        df.loc[idxs, "odds"] = mapped.to_numpy(dtype=float)

    df["ev"] = df["proba"] * df["odds"]
    return df
