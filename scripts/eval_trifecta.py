#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三連単 確率推論（コンボ行で出力）
入力: data/integrated/YYYY/MMDD/integrated_pro.csv （pred_start..pred_end の日付範囲）
必要列（存在する分だけ使う/不足は無視）:
  hd,jcd,rno,lane,regno,racer_id,racer_name,
  p1_win (または proba_win),
  st_pred_sec, st_rank_in_race, dash_advantage, wall_weak_flag, ...
  ほか place2/3 モデルが使う特徴（存在すれば）

出力: data/proba/trifecta/YYYY/MMDD/trifecta_proba_{pred_start}_{pred_end}.csv
  列: hd,jcd,rno,combo,proba
"""

from __future__ import annotations
import argparse, sys, itertools
from pathlib import Path
import polars as pl
import numpy as np
import lightgbm as lgb

def log(m: str): print(m, flush=True)
def ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

# --------------- I/O ---------------
def _date_dirs(root: Path, s: str, e: str) -> list[Path]:
    s_ord = int(s[:4]) * 372 + int(s[4:6]) * 31 + int(s[6:8])
    e_ord = int(e[:4]) * 372 + int(e[4:6]) * 31 + int(e[6:8])
    picked = []
    for ydir in sorted(root.glob("[0-9]"*4)):
        y = int(ydir.name)
        for md in sorted(ydir.glob("[0-9]"*4)):
            m, d = int(md.name[:2]), int(md.name[2:])
            ordv = y*372 + m*31 + d
            if s_ord <= ordv <= e_ord:
                picked.append(md)
    return picked

def _read_integrated(root: Path, s: str, e: str) -> pl.DataFrame:
    dirs = _date_dirs(root, s, e)
    files = [d/"integrated_pro.csv" for d in dirs if (d/"integrated_pro.csv").exists()]
    if not files:
        raise SystemExit(f"[FATAL] integrated_pro.csv が見つかりません: {root} {s}..{e}")
    # スキーマ差異吸収: まず全列読み込み、足りない列は追加してから縦結合
    frames = []
    all_cols = set()
    scans = []
    for f in files:
        lf = pl.scan_csv(str(f), ignore_errors=True)
        scans.append(lf)
        all_cols |= set(lf.collect_schema().names())
    # 代表順序
    all_cols = list(all_cols)
    for lf in scans:
        have = set(lf.collect_schema().names())
        add_exprs = [pl.lit(None).alias(c) for c in all_cols if c not in have]
        frames.append(lf.select([pl.col(c) for c in have] + add_exprs).select([pl.col(c) for c in all_cols]))
    return pl.concat(frames).collect()

# --------------- features ---------------
ID = ["hd","jcd","rno","lane","regno","racer_id","racer_name"]
# place2/3 の入力に使う“無難な”特徴（あれば使う）
FEATS = [
    "st_pred_sec","st_rel_sec","st_rank_in_race","dash_advantage","wall_weak_flag",
    "proba_nige","proba_sashi","proba_makuri","proba_makuri_sashi",
    "wind_speed_m","wave_height_cm","is_strong_wind","is_crosswind",
    "power_lane","power_inner","power_outer","outer_over_inner",
    "course_avg_st","course_first_rate","course_3rd_rate",
    "motor_rate2","motor_rate3","boat_rate2","boat_rate3",
    "tenji_st","tenji_sec","tenji_rank","st_rank","dash_attack_flag",
]

def _safe_cols(df: pl.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]

def _p1_col(df: pl.DataFrame, use_flag: bool) -> str:
    # 明示フラグ or 自動
    for cand in (["p1_win","proba_win"] if use_flag else ["p1_win","proba_win"]):
        if cand in df.columns: return cand
    raise SystemExit("[FATAL] p1（勝率）列が見つかりません（p1_win / proba_win など）")

# --------------- modeling ---------------
def _load_booster(path: Path) -> lgb.Booster|None:
    if not path.exists(): return None
    return lgb.Booster(model_file=str(path))

def _predict_rows(bst: lgb.Booster|None, pdf) -> np.ndarray:
    if bst is None or pdf.shape[0] == 0:
        return np.zeros((pdf.shape[0],), dtype=float)
    return bst.predict(pdf)

# --------------- trifecta assembly ---------------
def infer_trifecta(df: pl.DataFrame, model_root: Path, use_p1_flag: bool) -> pl.DataFrame:
    # 整形
    id_cols = _safe_cols(df, ID)
    feat_cols = _safe_cols(df, FEATS)
    p1col = _p1_col(df, use_p1_flag)

    # 型
    df = df.with_columns([
        pl.col("hd").cast(pl.Utf8) if "hd" in df.columns else pl.lit(""),
        pl.col("jcd").cast(pl.Utf8) if "jcd" in df.columns else pl.lit(""),
        pl.col("rno").cast(pl.Int64) if "rno" in df.columns else pl.lit(0),
    ])

    # LightGBM モデル読み込み
    b2 = _load_booster(model_root/"place2.txt")
    b3 = _load_booster(model_root/"place3.txt")
    if b2 is None or b3 is None:
        raise SystemExit(f"[FATAL] place2/place3 モデルが見つかりません: {model_root}")

    # レース単位に分割
    out_rows = []
    for (hd, jcd, rno), g in df.group_by(["hd","jcd","rno"], maintain_order=True):
        g = g.sort("lane")  # lane=1..6 想定
        lanes = g["lane"].to_list()
        if len(lanes) < 3:
            continue

        # P1: そのまま抽出
        p1 = g[p1col].to_numpy().astype(float)
        p1 = np.clip(p1, 1e-12, 1.0)

        # P2(j|i), P3(k|i,j) を LGBMで推定
        # 特徴は “選ばれうる艇” の行を入れて bst.predict
        X = g.select(feat_cols).to_pandas(use_pyarrow_extension_array=False)

        # 全コンボ 6P3
        best = []
        for i in range(len(lanes)):
            # 2着候補
            idx_others = [t for t in range(len(lanes)) if t != i]
            # P2: モデルは「2着確率」を各艇の row に対して出す前提（ない場合は一様）
            p2_all = _predict_rows(b2, X.iloc[idx_others])  # shape (5,)
            if np.all(p2_all == 0):
                p2_all = np.ones_like(p2_all) / len(p2_all)
            p2_all = p2_all / (p2_all.sum() + 1e-12)

            for j_rel, j in enumerate(idx_others):
                # 3着候補
                idx_others2 = [t for t in idx_others if t != j]
                p3_all = _predict_rows(b3, X.iloc[idx_others2])
                if np.all(p3_all == 0):
                    p3_all = np.ones_like(p3_all) / len(p3_all)
                p3_all = p3_all / (p3_all.sum() + 1e-12)

                for k_rel, k in enumerate(idx_others2):
                    prob = float(p1[i] * p2_all[j_rel] * p3_all[k_rel])
                    combo = f"{lanes[i]}-{lanes[j]}-{lanes[k]}"
                    best.append((combo, prob))

        # 正規化 & 上位のみ（全件でもOK。CSVサイズ抑制したいなら上位N）
        if not best:
            continue
        combos, probs = zip(*best)
        probs = np.array(probs, dtype=float)
        probs /= probs.sum() + 1e-12

        # そのまま全部吐く（必要なら[:100]などに制限可能）
        for c, p in zip(combos, probs):
            out_rows.append({"hd":hd, "jcd":jcd, "rno":int(rno), "combo":c, "proba":float(p)})

    return pl.DataFrame(out_rows)

# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/integrated")
    ap.add_argument("--pred_start", required=True)
    ap.add_argument("--pred_end", required=True)
    ap.add_argument("--model_root", default="data/models/place23")
    ap.add_argument("--out_root", default="data/proba/trifecta")
    ap.add_argument("--use_proba_win", action="store_true", help="p1=proba_win/p1_win を使用")
    args = ap.parse_args()

    root = Path(args.staging_root)
    df = _read_integrated(root, args.pred_start, args.pred_end)
    pred = infer_trifecta(df, Path(args.model_root), args.use_proba_win)

    # 保存
    y, md = args.pred_start[:4], args.pred_start[4:8]
    out_dir = Path(args.out_root)/y/md
    ensure_parent(out_dir/"x")
    out_csv = out_dir/f"trifecta_proba_{args.pred_start}_{args.pred_end}.csv"
    pred.write_csv(str(out_csv))
    log(f"[WRITE] {out_csv} rows={pred.height}")

if __name__ == "__main__":
    main()
