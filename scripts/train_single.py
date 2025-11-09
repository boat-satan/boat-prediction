#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import polars as pl
import pandas as pd
import numpy as np
import lightgbm as lgb

# ---- utils ----
def _z2(n: int) -> str: return str(n).zfill(2)
def _ymd_parts(hd: str) -> Tuple[str,str]:
    y, m, d = hd[:4], hd[4:6], hd[6:8]
    return y, m + d

def _scan_staging(staging_root: Path, start: str, end: str) -> List[Path]:
    # data/staging/YYYY/MMDD/st_train.csv を全走査して期間フィルタ
    files = sorted(staging_root.glob("**/st_train.csv"))
    if not files: return []
    # ざっくり期間での事前フィルタ（hd列でもう一段絞るためここは緩く）
    return files

def _read_staging_csv(p: Path) -> pl.DataFrame:
    # st_train.csv の想定列:
    # hd,jcd,rno,lane,regno,st_sec,st_is_f,st_is_late,st_penalized,st_observed,
    # tenji_st_sec,tenji_is_f,tenji_f_over_sec,tenji_rank,st_rank,
    # course_avg_st,course_first_rate,course_3rd_rate
    dtypes = {
        "hd": pl.Utf8, "jcd": pl.Utf8, "rno": pl.Int64, "lane": pl.Int64, "regno": pl.Int64,
        "st_sec": pl.Float64, "st_is_f": pl.Int64, "st_is_late": pl.Int64, "st_penalized": pl.Int64, "st_observed": pl.Int64,
        "tenji_st_sec": pl.Float64, "tenji_is_f": pl.Int64, "tenji_f_over_sec": pl.Float64, "tenji_rank": pl.Int64,
        "st_rank": pl.Int64, "course_avg_st": pl.Float64, "course_first_rate": pl.Float64, "course_3rd_rate": pl.Float64,
    }
    return pl.read_csv(p, dtypes=dtypes, null_values=["", "null", "None"])

def _winner_regno_from_result(results_root: Path, hd: str, jcd: str, rno: int) -> Optional[int]:
    y, md = _ymd_parts(hd)
    rp = results_root / y / md / jcd / f"{rno}R.json"
    if not rp.exists(): return None
    try:
        js = json.loads(rp.read_text(encoding="utf-8"))
        arr = js.get("results") or []
        for item in arr:
            # 1着の regno を返す（なければ lane でも良いが regno 優先）
            if str(item.get("rank")) == "1":
                reg = item.get("racer_id")
                if reg is not None:
                    return int(reg)
                # フォールバック: lane は別カラムと一致させにくいので極力使用しない
        return None
    except Exception:
        return None

def _make_label_win(df: pl.DataFrame, results_root: Path) -> pl.DataFrame:
    # レース単位で winner_regno を付与して is_win を作成
    # df は一日に複数レースが混在している可能性がある
    def _label_one(group: pl.DataFrame) -> pl.DataFrame:
        hd = group["hd"][0]
        jcd = str(group["jcd"][0])
        rno = int(group["rno"][0])
        wreg = _winner_regno_from_result(results_root, hd, jcd, rno)
        if wreg is None:
            return group.with_columns(pl.lit(None, dtype=pl.Int64).alias("is_win"))
        else:
            return group.with_columns((pl.col("regno") == wreg).cast(pl.Int64).alias("is_win"))

    out_parts: List[pl.DataFrame] = []
    for (hd, jcd, rno), sub in df.group_by(["hd","jcd","rno"], maintain_order=True):
        out_parts.append(_label_one(sub))
    return pl.concat(out_parts) if out_parts else df

def _select_feature_table(df: pl.DataFrame, drop_leak: bool=True) -> Tuple[pd.DataFrame, List[str]]:
    # 漏洩防止のため、"st_sec"（実 ST）や "st_rank"（実績からの生成）を基本除外
    # 使う特徴：展示系 + コース別過去統計
    feat_cols = [
        "tenji_st_sec","tenji_is_f","tenji_f_over_sec","tenji_rank",
        "course_avg_st","course_first_rate","course_3rd_rate"
    ]
    base_cols = ["hd","jcd","rno","lane","regno"]
    use_cols = base_cols + feat_cols + ["is_win"]
    dfx = df.select([c for c in use_cols if c in df.columns]).to_pandas()
    # 欠損の穴埋め（0/median）
    if "tenji_is_f" in dfx.columns: dfx["tenji_is_f"] = dfx["tenji_is_f"].fillna(0)
    if "tenji_rank" in dfx.columns: dfx["tenji_rank"] = dfx["tenji_rank"].fillna(dfx["tenji_rank"].median())
    for c in ["tenji_st_sec","tenji_f_over_sec","course_avg_st","course_first_rate","course_3rd_rate"]:
        if c in dfx.columns:
            dfx[c] = dfx[c].astype(float)
            dfx[c] = dfx[c].fillna(dfx[c].median())
    return dfx, feat_cols

def _filter_by_date(df: pl.DataFrame, start: str, end: str) -> pl.DataFrame:
    return df.filter((pl.col("hd") >= start) & (pl.col("hd") <= end))

def _ensure_dirs(p: Path): p.parent.mkdir(parents=True, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--train_start",  required=True)
    ap.add_argument("--train_end",    required=True)
    ap.add_argument("--pred_start",   required=True)
    ap.add_argument("--pred_end",     required=True)
    ap.add_argument("--model_out",    default="data/models/single_lgbm.txt")
    ap.add_argument("--proba_out_root", default="data/proba/single")
    ap.add_argument("--num_boost_round", type=int, default=800)
    args = ap.parse_args()

    staging_root = Path(args.staging_root)
    results_root = Path(args.results_root)
    files = _scan_staging(staging_root, args.train_start, args.pred_end)
    if not files:
        print("[FATAL] no staging files under", staging_root)
        raise SystemExit(1)

    # すべてロードしてまとめる（期間フィルタは hd で）
    frames = []
    for f in files:
        try:
            df = _read_staging_csv(f)
            frames.append(df)
        except Exception as e:
            print(f"[WARN] read failed {f}: {e}")
    if not frames:
        print("[FATAL] cannot read any staging files.")
        raise SystemExit(1)

    pool = pl.concat(frames, how="vertical", rechunk=True)
    # 学習・推論それぞれに必要な期間だけ抽出
    df_train = _filter_by_date(pool, args.train_start, args.train_end)
    df_pred  = _filter_by_date(pool, args.pred_start, args.pred_end)

    if df_train.is_empty():
        print("[FATAL] train slice empty.")
        raise SystemExit(1)
    if df_pred.is_empty():
        print("[WARN] pred slice empty. Proceeding anyway.")

    # 学習ラベル付与
    df_train = _make_label_win(df_train, results_root)
    if "is_win" not in df_train.columns:
        print("[FATAL] label (is_win) missing.")
        raise SystemExit(1)
    df_train = df_train.filter(pl.col("is_win").is_not_null())
    if df_train.is_empty():
        print("[FATAL] train slice has no labels.")
        raise SystemExit(1)

    # pandas へ
    train_pd, feat_cols = _select_feature_table(df_train)
    X = train_pd[feat_cols].values
    y = train_pd["is_win"].astype(int).values

    # LightGBM binary
    dtrain = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=True)
    params = dict(
        objective="binary",
        metric="binary_logloss",
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_data_in_leaf=50,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=1,
        lambda_l2=1.0,
        seed=20240301,
        num_threads=0,
        verbose=-1,
        force_col_wise=True,
    )
    booster = lgb.train(
        params, dtrain, num_boost_round=args.num_boost_round,
        valid_sets=[dtrain], valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)]
    )
    _ensure_dirs(Path(args.model_out))
    booster.save_model(args.model_out)
    print(f"[WRITE] model -> {args.model_out}")

    # 予測
    if not df_pred.is_empty():
        pred_pd, _ = _select_feature_table(df_pred.with_columns(pl.lit(None).alias("is_win")))
        Xp = pred_pd[feat_cols].values
        pwin = booster.predict(Xp, num_iteration=booster.best_iteration)
        pred_pd["p_win"] = pwin

        # 出力： race毎にまとめて書く
        out_root = Path(args.proba_out_root)
        rows = 0
        for (hd, jcd, rno), g in pred_pd.groupby(["hd","jcd","rno"]):
            y, md = _ymd_parts(str(hd))
            out_dir = out_root / y / md
            _ensure_dirs(out_dir)
            out_path = out_dir / f"proba_single_{hd}_{jcd}_{int(rno)}R.csv"
            g[["hd","jcd","rno","lane","regno","p_win"]].to_csv(out_path, index=False)
            rows += len(g)
        print(f"[WRITE] proba files -> {out_root} (rows={rows})")

if __name__ == "__main__":
    main()
