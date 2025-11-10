#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三連単 確率推論（堅牢版）
- 入力: data/integrated/YYYY/MMDD/integrated_train.csv （日付範囲で集約）
- 期待列:
    必須: hd, jcd, rno, lane
    勝率: proba_win or p1_win のどちらか
    任意: st_pred_sec, st_rel_sec, tenji_sec, st_rank など（あれば2着/3着の重み付けに利用）
- 追加利用（任意）:
    --model_root に place2.txt / place3.txt があれば読んで lane別 P2/P3 スコアを補助に使う
- 出力: data/proba/trifecta/YYYY/MMDD/trifecta_proba_{pred_start}_{pred_end}.csv
    列: hd,jcd,rno,combo,proba
"""

from __future__ import annotations
import argparse, sys, glob
from pathlib import Path
from typing import List, Dict, Tuple

import polars as pl
import numpy as np

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False


def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _ymd_ord(s: str) -> int:
    return int(s[:4]) * 372 + int(s[4:6]) * 31 + int(s[6:8])

def _ymd_paths(root: Path, d1: str, d2: str) -> List[Path]:
    s_ord, e_ord = _ymd_ord(d1), _ymd_ord(d2)
    picked: List[Path] = []
    for ydir in sorted(root.glob("[0-9]" * 4)):
        if not ydir.is_dir(): continue
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]" * 4)):
            if not md.is_dir(): continue
            m, d = int(md.name[:2]), int(md.name[2:])
            od = y * 372 + m * 31 + d
            if s_ord <= od <= e_ord:
                picked.append(md)
    return picked

# ----------------------------------------------------
# 入力読み込み（スキーマ差異を吸収）
# ----------------------------------------------------
NEED_BASE = ["hd","jcd","rno","lane","regno","racer_id","racer_name"]
OPTIONALS = [
    "proba_win","p1_win","st_pred_sec","st_rel_sec","st_rank",
    "tenji_sec","tenji_st","course_avg_st"
]

def load_integrated(integrated_root: str, d1: str, d2: str) -> pl.DataFrame:
    root = Path(integrated_root)
    day_dirs = _ymd_paths(root, d1, d2)
    files: List[str] = []
    for dd in day_dirs:
        p = dd / "integrated_train.csv"
        if p.exists():
            files.append(str(p))
    if not files:
        print(f"[FATAL] integrated_train.csv が見つかりません: {integrated_root} {d1}..{d2}", file=sys.stderr)
        sys.exit(1)

    # スキーマ合わせ：存在しない列はnullで埋める
    lfs = []
    for f in files:
        lf = pl.scan_csv(f, ignore_errors=True)
        cols = lf.collect_schema().names()
        select_exprs: List[pl.Expr] = []
        for c in NEED_BASE + OPTIONALS:
            if c in cols:
                select_exprs.append(pl.col(c))
            else:
                select_exprs.append(pl.lit(None).alias(c))
        lfs.append(lf.select(select_exprs))
    df = pl.concat(lfs).collect()

    # 型整形
    df = (
        df
        .with_columns([
            pl.col("hd").cast(pl.Utf8),
            pl.col("jcd").cast(pl.Utf8),
            pl.col("rno").cast(pl.Int64),
            pl.col("lane").cast(pl.Int64),
        ])
    )

    # 勝率列の決定（proba_win or p1_win）
    if "proba_win" in df.columns and df["proba_win"].dtype != pl.Null:
        win_col = "proba_win"
    elif "p1_win" in df.columns and df["p1_win"].dtype != pl.Null:
        win_col = "p1_win"
    else:
        print(f"[FATAL] 勝率列(proba_win/p1_win)が見つかりませんでした。列={df.columns}", file=sys.stderr)
        sys.exit(1)

    # 欠損/負値ガード
    df = df.with_columns([
        pl.col(win_col).cast(pl.Float64).fill_null(0.0).clip(0.0, 1.0).alias(win_col),
        pl.col("st_pred_sec").cast(pl.Float64, strict=False),
        pl.col("st_rel_sec").cast(pl.Float64, strict=False),
        pl.col("st_rank").cast(pl.Int64, strict=False),
        pl.col("tenji_sec").cast(pl.Float64, strict=False),
        pl.col("course_avg_st").cast(pl.Float64, strict=False),
    ])

    return df, win_col

# ----------------------------------------------------
# 2着/3着スコア（モデルがあれば利用、無ければヒューリスティック）
# ----------------------------------------------------
def load_place_models(model_root: str):
    b2 = b3 = None
    if not HAS_LGB:
        return None, None
    mr = Path(model_root)
    f2, f3 = mr / "place2.txt", mr / "place3.txt"
    try:
        if f2.exists():
            b2 = lgb.Booster(model_file=str(f2))
        if f3.exists():
            b3 = lgb.Booster(model_file=str(f3))
    except Exception:
        b2 = b3 = None
    return b2, b3

PLACE_FEATS = [
    # ここは integrated にだいたい存在する前提の軽い特徴群
    "st_pred_sec","st_rel_sec","st_rank","tenji_sec","course_avg_st","lane"
]

def predict_place_scores(pdf: pl.DataFrame, booster, default: float=1.0) -> np.ndarray:
    if booster is None:
        return np.full((len(pdf),), default, dtype=float)
    # 欠損は0埋め、型合わせ
    use = []
    for c in PLACE_FEATS:
        if c in pdf.columns:
            use.append(c)
        else:
            pdf = pdf.with_columns(pl.lit(None).alias(c))
            use.append(c)
    X = pdf.select(use).fill_null(0).to_pandas(use_pyarrow_extension_array=False)
    try:
        p = booster.predict(X)
        # 回帰 or バイナリ確率を想定（vectorの場合は1列目を使う）
        if p.ndim == 2 and p.shape[1] > 1:
            p = p[:, 1]
        return np.asarray(p, dtype=float)
    except Exception:
        return np.full((len(pdf),), default, dtype=float)

# ----------------------------------------------------
# 三連単の確率生成
# ----------------------------------------------------
def enumerate_trifecta(df: pl.DataFrame, win_col: str, b2, b3) -> pl.DataFrame:
    """
    レース単位で全120通りを生成。
    近似: Plackett–Luce 風
      P(i-j-k) = P1[i] * softmax_j_not_i(S2[j]) * softmax_k_not_(i,j)(S3[k])
    S2, S3 はモデル出力があればそれ、無ければヒューリスティック。
    """
    out_rows: List[dict] = []
    for (hd, jcd, rno), g in df.group_by(["hd","jcd","rno"], maintain_order=True):
        g = g.sort("lane")
        lanes = g["lane"].to_list()
        if len(lanes) < 3:
            continue

        p1 = g[win_col].to_numpy()
        # 正規化
        s = p1.sum()
        if s <= 0:
            p1 = np.ones_like(p1) / len(p1)
        else:
            p1 = p1 / s

        # 2着・3着スコア
        s2 = predict_place_scores(g, b2, default=1.0)
        s3 = predict_place_scores(g, b3, default=1.0)

        # ヒューリスティックのスパイス: スタート優位は2着寄与↑
        if "st_pred_sec" in g.columns:
            st = g["st_pred_sec"].fill_null(g["st_pred_sec"].mean()).to_numpy()
            # 速い(小さい)ほど重め：1 / (eps + st)
            s2 = s2 * (1.0 / np.clip(st, 0.02, None))

        # 全順列（6艇→最大120通り）
        L = len(lanes)
        for a in range(L):
            pa = p1[a]
            if pa <= 0:
                continue
            den2 = s2.sum() - s2[a]
            if den2 <= 0:
                # 残り均等
                w2 = np.ones(L); w2[a] = 0.0; w2 = w2 / w2.sum()
            else:
                w2 = s2.copy(); w2[a] = 0.0; w2 = w2 / den2

            for b in range(L):
                if b == a: continue
                den3 = s3.sum() - s3[a] - s3[b]
                if den3 <= 0:
                    w3 = np.ones(L); w3[a] = 0.0; w3[b] = 0.0; w3 = w3 / w3.sum()
                else:
                    w3 = s3.copy(); w3[a] = 0.0; w3[b] = 0.0; w3 = w3 / den3

                for c in range(L):
                    if c == a or c == b: continue
                    prob = float(pa * w2[b] * w3[c])
                    if prob <= 0:
                        continue
                    combo = f"{lanes[a]}-{lanes[b]}-{lanes[c]}"
                    out_rows.append({
                        "hd": hd, "jcd": jcd, "rno": int(rno),
                        "combo": combo, "proba": prob,
                    })

    if not out_rows:
        return pl.DataFrame(schema={"hd":pl.Utf8,"jcd":pl.Utf8,"rno":pl.Int64,"combo":pl.Utf8,"proba":pl.Float64})
    out = pl.DataFrame(out_rows)
    # レース内で正規化（数値の安定化）
    out = (
        out
        .with_columns(pl.col("proba").cast(pl.Float64))
        .group_by(["hd","jcd","rno"])
        .agg([
            pl.col("combo"),
            (pl.col("proba") / pl.col("proba").sum()).alias("proba")
        ])
        .explode(["combo","proba"])
    )
    return out

# ----------------------------------------------------
# main
# ----------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/integrated")   # 互換のためこの名前のまま
    ap.add_argument("--pred_start", required=True)
    ap.add_argument("--pred_end", required=True)
    ap.add_argument("--model_root", default="data/models/place23")
    ap.add_argument("--out_root", default="data/proba/trifecta")
    ap.add_argument("--use_proba_win", action="store_true")  # 将来の互換用（今は常に使う）
    args = ap.parse_args()

    df, win_col = load_integrated(args.staging_root, args.pred_start, args.pred_end)
    log(f"[INFO] loaded rows: {df.height}, win_col={win_col}")

    b2, b3 = load_place_models(args.model_root)
    if b2 is None or b3 is None:
        log("[WARN] place2/place3 モデルが見つからないか読めません。ヒューリスティックで近似します。")

    pred = enumerate_trifecta(df, win_col, b2, b3)

    y, md = args.pred_start[:4], args.pred_start[4:8]
    out_csv = ensure_parent(Path(args.out_root) / y / md / f"trifecta_proba_{args.pred_start}_{args.pred_end}.csv")
    pred.write_csv(str(out_csv))
    log(f"[WRITE] {out_csv} rows={pred.height}")

if __name__ == "__main__":
    main()
