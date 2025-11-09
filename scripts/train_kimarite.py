#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_kimarite.py

決まり手(マルチクラス: nige/sashi/makuri/makurizashi/sonota)の確率分布を
レース単位で学習・推論する単体CLI。LightGBMマルチクラスを使用。

入力は「レース単位」または「艇(レーン)単位」のどちらでも可。
- レーン単位入力の場合は (hd,jcd,rno) で集約してレース特徴量を作る。
- 任意で ST 予測CSV (--st_pred_csv) を読込み、(hd,jcd,rno,lane) ごとの st_pred を
  ピボット(各レーンの st_pred)＋集約統計(平均/最小/最大/標準偏差/分散/レンジ)として追加。

必須列:
  - hd  : YYYYMMDD (str/int 可)
  - jcd : 場コード
  - rno : レース番号 (int/str 可)
学習時に必要（教師ラベル）:
  - kimarite : {"nige","sashi","makuri","makurizashi","sonota"} のいずれか

任意列:
  - lane : 1..6 があれば「レーン単位入力」とみなし、数値列を (hd,jcd,rno) に対して
           mean/min/max/std/sum を自動集約してレース特徴量を生成。
  - その他の数値列・カテゴリ列は自動で前処理（カテゴリは category 化）

出力:
  - モデル:   --model_out  (デフォルト: data/models/kimarite_lgbm.txt)
  - 予測CSV:  --pred_out   (デフォルト: data/eval/kimarite_pred.csv)
              列: hd,jcd,rno,[p_nige,p_sashi,p_makuri,p_makurizashi,p_sonota],(kimarite_true任意)

使用例:
  1) 単一ソースを日付で分割（学習＋検証＋予測保存）
    python scripts/train_kimarite.py \
      --input "data/race_rows/*.parquet" \
      --train_start 20240101 --train_end 20240131 \
      --test_start  20240201 --test_end  20240229 \
      --model_out data/models/kimarite_lgbm.txt \
      --pred_out  data/eval/kimarite_pred_202402.csv

  2) train/test を別ファイルで指定 + ST予測を追加（任意）
    python scripts/train_kimarite.py \
      --train_input "data/train_race_rows.parquet" \
      --test_input  "data/test_race_rows.parquet" \
      --st_pred_csv data/eval/st_pred_202402.csv \
      --model_out data/models/kimarite_lgbm.txt \
      --pred_out  data/eval/kimarite_pred_test.csv
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb


# ----------------- IO -----------------
def _read_any(path: str) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".parquet") or p.endswith(".pq"):
        return pd.read_parquet(p)
    elif p.endswith(".csv") or p.endswith(".csv.gz"):
        return pd.read_csv(p)
    else:
        raise ValueError(f"Unsupported file type: {p}")

def _read_glob(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    dfs = [_read_any(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


# ----------------- 前処理 -----------------
KIMARITE2ID: Dict[str, int] = {
    "nige": 0,
    "sashi": 1,
    "makuri": 2,
    "makurizashi": 3,
    "sonota": 4,
}
ID2KIMARITE = {v: k for k, v in KIMARITE2ID.items()}

def _normalize_id_cols(df: pd.DataFrame, hd_col: str) -> pd.DataFrame:
    dd = df.copy()
    dd[hd_col] = dd[hd_col].astype(str)
    # jcd/rno は文字/数値どちらでも良いが、join整合のため str に寄せる
    if "jcd" in dd.columns:
        dd["jcd"] = dd["jcd"].astype(str)
    if "rno" in dd.columns:
        # "1R" のような表記にも耐える：数字抽出
        dd["rno"] = dd["rno"].apply(lambda x: int("".join(ch for ch in str(x) if ch.isdigit())) if pd.notna(x) else x)
    return dd

def _filter_by_date(df: pd.DataFrame, hd_col: str, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    if start is None and end is None:
        return df
    s = str(start) if start is not None else None
    e = str(end) if end is not None else None
    dd = df.copy()
    dd[hd_col] = dd[hd_col].astype(str)
    if s is not None:
        dd = dd[dd[hd_col] >= s]
    if e is not None:
        dd = dd[dd[hd_col] <= e]
    return dd

def _aggregate_to_race(df: pd.DataFrame, id_cols: List[str]) -> pd.DataFrame:
    """
    lane 列があれば「レーン単位入力」とみなし、数値列を (hd,jcd,rno) で集約してレース特徴量へ。
    lane が無い場合はそのまま（レース単位入力とみなす）。
    """
    dd = df.copy()
    for c in id_cols:
        if c not in dd.columns:
            raise KeyError(f"ID column missing: {c}")
    if "lane" not in dd.columns:
        # 既にレース単位の行とみなす
        return dd

    # 数値列のみ対象（ID系/目的列/明らかに不要な文字列列は除外）
    drop_like = set(id_cols + ["kimarite"])
    num_cols = [c for c in dd.columns if c not in drop_like and pd.api.types.is_numeric_dtype(dd[c])]
    if not num_cols:
        # 数値列が無いなら ID のみ返す
        return dd[id_cols].drop_duplicates().reset_index(drop=True)

    aggs = {}
    for c in num_cols:
        aggs[c] = ["mean", "min", "max", "std", "sum"]
    grp = dd.groupby(id_cols, dropna=False).agg(aggs)
    # 列フラット化
    grp.columns = [f"{c}_{stat}" for c, stat in grp.columns]
    grp = grp.reset_index()
    return grp

def _merge_st_features(base: pd.DataFrame, st_csv: Optional[str], id_cols: List[str]) -> pd.DataFrame:
    """
    ST 予測CSV（列: hd,jcd,rno,lane,st_pred[,st_true]）を race 特徴に展開して付与。
    - 各レーンの st_pred をピボット: st_pred_lane1..6
    - さらに集約統計: st_mean, st_min, st_max, st_std, st_var, st_range
    """
    if not st_csv:
        return base

    st = _read_any(st_csv)
    need = id_cols + ["lane", "st_pred"]
    for c in need:
        if c not in st.columns:
            raise KeyError(f"ST CSV missing column: {c}")
    st = _normalize_id_cols(st, hd_col=id_cols[0])

    # ピボット
    st_piv = st.pivot_table(index=id_cols, columns="lane", values="st_pred", aggfunc="mean")
    st_piv = st_piv.rename(columns=lambda ln: f"st_pred_lane{int(ln)}" if pd.notna(ln) else "st_pred_laneX")
    st_piv = st_piv.reset_index()

    # 集約統計
    lane_cols = [c for c in st_piv.columns if c.startswith("st_pred_lane")]
    tmp = st_piv[lane_cols].copy()
    st_stats = pd.DataFrame({
        "st_mean": tmp.mean(axis=1),
        "st_min":  tmp.min(axis=1),
        "st_max":  tmp.max(axis=1),
        "st_std":  tmp.std(axis=1).fillna(0.0),
        "st_var":  tmp.var(axis=1).fillna(0.0),
        "st_range": (tmp.max(axis=1) - tmp.min(axis=1)),
    })
    st_feat = pd.concat([st_piv[id_cols], st_piv[lane_cols], st_stats], axis=1)

    # 結合
    out = base.merge(st_feat, on=id_cols, how="left")
    return out

def _prepare_Xy(df: pd.DataFrame, id_cols: List[str], target_col: Optional[str]) -> Tuple[pd.DataFrame, Optional[np.ndarray], pd.DataFrame, List[str]]:
    """
    特徴行列X, 目的y, メタ(meta=ID列), 使用特徴列名を返却。
    文字列列は category 化して LightGBM categorical_feature として扱えるようにする。
    """
    meta = df[id_cols].copy()
    feat_df = df.drop(columns=[c for c in id_cols if c in df.columns], errors="ignore")

    y = None
    if target_col and target_col in feat_df.columns:
        y_raw = feat_df[target_col].astype(str)
        if not set(y_raw.unique()).issubset(set(KIMARITE2ID.keys())):
            miss = sorted(set(y_raw.unique()) - set(KIMARITE2ID.keys()))
            raise ValueError(f"Unknown label(s) in '{target_col}': {miss}")
        y = y_raw.map(KIMARITE2ID).to_numpy(dtype=int)
        feat_df = feat_df.drop(columns=[target_col])

    # dtype 整形
    for c in feat_df.columns:
        if pd.api.types.is_object_dtype(feat_df[c]):
            feat_df[c] = feat_df[c].astype("category")

    feat_cols = list(feat_df.columns)
    return feat_df, y, meta, feat_cols


# ----------------- CLI -----------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # 入力（どちらか）
    ap.add_argument("--input", help="学習/検証/推論を単一ソースから日付で分割するglob")
    ap.add_argument("--train_input", help="学習用入力（glob可）")
    ap.add_argument("--test_input", help="テスト/推論用入力（glob可）")

    # 日付分割
    ap.add_argument("--hd_col", default="hd")
    ap.add_argument("--train_start")
    ap.add_argument("--train_end")
    ap.add_argument("--test_start")
    ap.add_argument("--test_end")

    # ID/列指定
    ap.add_argument("--id_cols", default="hd,jcd,rno", help="ID列（カンマ区切り）。lane は任意")
    ap.add_argument("--target_col", default="kimarite", help="決まり手ラベル列名")
    ap.add_argument("--drop_cols", default="", help="特徴から除外する列（カンマ区切り）")

    # 任意: ST予測CSV（hd,jcd,rno,lane,st_pred）
    ap.add_argument("--st_pred_csv", default="", help="train_st.py の出力など（任意）")

    # 出力
    ap.add_argument("--model_out", default="data/models/kimarite_lgbm.txt")
    ap.add_argument("--pred_out", default="data/eval/kimarite_pred.csv")

    # LGBMハイパラ
    ap.add_argument("--num_boost_round", type=int, default=2000)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--feature_fraction", type=float, default=0.9)
    ap.add_argument("--bagging_fraction", type=float, default=0.9)
    ap.add_argument("--bagging_freq", type=int, default=1)
    ap.add_argument("--lambda_l2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20240301)

    return ap


def _load_dataframe_from_args(args: argparse.Namespace) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if args.input:
        df = _read_glob(args.input)
        df = _normalize_id_cols(df, hd_col=args.hd_col)
        # レーン単位→レース単位へ
        id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
        df_race = _aggregate_to_race(df, id_cols=id_cols + ([] if "lane" not in df.columns else []))
        # 日付で分割
        tr = _filter_by_date(df_race, args.hd_col, args.train_start, args.train_end) if (args.train_start or args.train_end) else None
        te = _filter_by_date(df_race, args.hd_col, args.test_start, args.test_end) if (args.test_start or args.test_end) else None
        if tr is None and te is None:
            tr = df_race
        return tr, te

    # train/test を別ファイルで指定
    tr = _read_glob(args.train_input) if args.train_input else None
    te = _read_glob(args.test_input)  if args.test_input  else None

    if tr is not None:
        tr = _normalize_id_cols(tr, hd_col=args.hd_col)
        tr = _aggregate_to_race(tr, id_cols=[c.strip() for c in args.id_cols.split(",") if c.strip()] + ([] if "lane" not in tr.columns else []))
        if (args.train_start or args.train_end):
            tr = _filter_by_date(tr, args.hd_col, args.train_start, args.train_end)

    if te is not None:
        te = _normalize_id_cols(te, hd_col=args.hd_col)
        te = _aggregate_to_race(te, id_cols=[c.strip() for c in args.id_cols.split(",") if c.strip()] + ([] if "lane" not in te.columns else []))
        if (args.test_start or args.test_end):
            te = _filter_by_date(te, args.hd_col, args.test_start, args.test_end)

    if tr is None and te is None:
        raise RuntimeError("No input given. Use --input or --train_input/--test_input.")
    return tr, te


def main():
    ap = build_argparser()
    args = ap.parse_args()

    id_cols   = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    train_df, test_df = _load_dataframe_from_args(args)

    # ST 予測の付与（任意）
    if args.st_pred_csv:
        if train_df is not None:
            train_df = _merge_st_features(train_df, args.st_pred_csv, id_cols)
        if test_df is not None:
            test_df  = _merge_st_features(test_df, args.st_pred_csv, id_cols)

    # ドロップ指定列
    if drop_cols:
        if train_df is not None:
            train_df = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns], errors="ignore")
        if test_df is not None:
            test_df  = test_df.drop(columns=[c for c in drop_cols if c in test_df.columns], errors="ignore")

    # ---- 学習 ----
    model = None
    cat_cols: List[str] = []

    if train_df is not None:
        Xtr, ytr, meta_tr, feat_cols = _prepare_Xy(train_df, id_cols=id_cols, target_col=args.target_col)

        # 検証 split: hd 単位で 8:2
        hd_vals = train_df[id_cols[0]].astype(str).unique()
        rng = np.random.default_rng(args.seed)
        rng.shuffle(hd_vals)
        cut = int(len(hd_vals) * 0.8)
        hd_tr = set(hd_vals[:cut])
        tr_mask = train_df[id_cols[0]].astype(str).isin(hd_tr)

        X_train = Xtr[tr_mask]
        y_train = ytr[tr_mask]
        X_valid = Xtr[~tr_mask]
        y_valid = ytr[~tr_mask]

        cat_cols = [c for c in feat_cols if str(Xtr[c].dtype).startswith("category")]

        lgb_train = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
        lgb_valid = lgb.Dataset(X_valid, label=y_valid, categorical_feature=cat_cols, reference=lgb_train, free_raw_data=False)

        params = dict(
            objective="multiclass",
            num_class=len(KIMARITE2ID),
            metric="multi_logloss",
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            max_depth=-1,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
            lambda_l2=args.lambda_l2,
            num_threads=0,
            seed=args.seed,
            verbose=-1,
            force_col_wise=True,
        )

        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=args.num_boost_round,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=True),
                       lgb.log_evaluation(period=200)],
        )

        _ensure_dir(Path(args.model_out))
        model.save_model(args.model_out)
        print(f"[WRITE] model -> {args.model_out}")

        # 検証ログロス
        yhat_valid = model.predict(X_valid, num_iteration=model.best_iteration)
        # 出力は (n, num_class)。ここではログロスのみ出力（詳細評価は別途でも）
        print(f"[INFO] valid n={len(y_valid)}  (best_iteration={model.best_iteration})")

    else:
        # 推論のみ
        if not Path(args.model_out).exists():
            raise FileNotFoundError(f"model file not found: {args.model_out}")
        model = lgb.Booster(model_file=args.model_out)

    # ---- 推論 ----
    pred_base = test_df if test_df is not None else train_df
    if pred_base is not None:
        # target 無しでも OK
        has_target = args.target_col in pred_base.columns
        Xte, yte_dummy, meta_te, feat_cols_te = _prepare_Xy(
            pred_base.drop(columns=[args.target_col], errors="ignore"),
            id_cols=id_cols, target_col=None
        )
        yproba = model.predict(Xte, num_iteration=getattr(model, "best_iteration", None))

        out = meta_te.copy()
        proba_cols = ["p_nige","p_sashi","p_makuri","p_makurizashi","p_sonota"]
        proba_df = pd.DataFrame(yproba, columns=proba_cols)
        out = pd.concat([out, proba_df], axis=1)

        if has_target:
            out["kimarite_true"] = pred_base[args.target_col].astype(str).map(lambda x: x if x in KIMARITE2ID else np.nan)

        _ensure_dir(Path(args.pred_out))
        out.to_csv(args.pred_out, index=False, encoding="utf-8")
        print(f"[WRITE] preds -> {args.pred_out} (n={len(out)})")

if __name__ == "__main__":
    main()
