#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/train_st.py

ST(スタートタイミング)回帰モデルの学習/推論 CLI。
前段のライブラリ: src/boatpred/st_model.py (STRegressor, build_Xy)

前提データ: 1行=1艇レベルのテーブル（例: entries/exhibition結合済み）
必須列:
  - hd:   YYYYMMDD (str/int可)
  - jcd:  場コード
  - rno:  レース番号
  - lane: 1..6
  - target_st (デフォルト "st"): 実績ST(秒) … train では必要、test では不要でも可

特徴量は上記以外の全列（除外指定も可）を利用。object列は自動でcategory化。

使い方例:
  学習+評価+予測CSV保存:
    python scripts/train_st.py \
      --input "data/entries/*.parquet" \
      --target_col st \
      --train_start 20240101 --train_end 20240131 \
      --test_start  20240201 --test_end  20240229 \
      --model_out data/models/st_lgbm.txt \
      --pred_out  data/eval/st_pred_202402.csv

  学習データ/テストデータを別ファイルで与える例:
    python scripts/train_st.py \
      --train_input "data/train_entries.parquet" \
      --test_input  "data/test_entries.parquet" \
      --target_col st \
      --model_out data/models/st_lgbm.txt \
      --pred_out  data/eval/st_pred_test.csv
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ライブラリ本体
from src.boatpred.st_model import STRegressor, build_Xy, rmse


# ----------------- IO ユーティリティ -----------------
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


# ----------------- CLI -----------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    # 入力指定（どちらかを利用）
    ap.add_argument("--input", help="学習/評価/推論を単一ソースから日付で分割する場合のglob (例: data/entries/*.parquet)")
    ap.add_argument("--train_input", help="学習用入力（glob可）")
    ap.add_argument("--test_input", help="テスト/推論用入力（glob可）")

    # 日付分割
    ap.add_argument("--hd_col", default="hd", help="日付列名 (default: hd)")
    ap.add_argument("--train_start")
    ap.add_argument("--train_end")
    ap.add_argument("--test_start")
    ap.add_argument("--test_end")

    # 列設定
    ap.add_argument("--target_col", default="st", help="ターゲットST列名 (default: st)")
    ap.add_argument("--id_cols", default="hd,jcd,rno,lane", help="ID列（カンマ区切り）")
    ap.add_argument("--drop_cols", default="", help="学習から除外する列（カンマ区切り）")

    # 出力
    ap.add_argument("--model_out", default="data/models/st_lgbm.txt")
    ap.add_argument("--pred_out", default="data/eval/st_pred.csv")

    # 学習パラメタ（必要に応じて微調整）
    ap.add_argument("--num_boost_round", type=int, default=2000)
    ap.add_argument("--early_stopping_rounds", type=int, default=200)

    return ap


def _load_dataframe_from_args(args: argparse.Namespace) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    args から train_df / test_df を構築する。
    - --input 指定: 1つのソースを日付で分割
    - --train_input/--test_input 指定: 別々のファイル
    """
    if args.input:
        df = _read_glob(args.input)
        train_df = _filter_by_date(df, args.hd_col, args.train_start, args.train_end) if args.train_start or args.train_end else None
        test_df  = _filter_by_date(df, args.hd_col, args.test_start, args.test_end) if args.test_start or args.test_end else None
        if train_df is None and test_df is None:
            # どちらも未指定なら全体をtrainに
            train_df = df
        return train_df, test_df

    # train/test 別指定
    train_df = _read_glob(args.train_input) if args.train_input else None
    test_df  = _read_glob(args.test_input) if args.test_input else None

    # 日付でさらに絞りたい場合
    if train_df is not None and (args.train_start or args.train_end):
        train_df = _filter_by_date(train_df, args.hd_col, args.train_start, args.train_end)
    if test_df is not None and (args.test_start or args.test_end):
        test_df = _filter_by_date(test_df, args.hd_col, args.test_start, args.test_end)

    if train_df is None and test_df is None:
        raise RuntimeError("No input given. Use --input or --train_input/--test_input.")
    return train_df, test_df


def main():
    ap = build_argparser()
    args = ap.parse_args()

    id_cols  = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    drop_cols= [c.strip() for c in args.drop_cols.split(",") if c.strip()]

    train_df, test_df = _load_dataframe_from_args(args)

    # ---- 学習 ----
    model = STRegressor(num_boost_round=args.num_boost_round,
                        early_stopping_rounds=args.early_stopping_rounds)

    if train_df is not None:
        if args.target_col not in train_df.columns:
            raise KeyError(f"target_col '{args.target_col}' not found in train data.")

        # 簡易バリデーション: 同日/同場などで8:2スプリット (hdをキーに分割)
        # hd基準でシャッフル
        hd_vals = train_df[args.hd_col].astype(str).unique()
        rng = np.random.default_rng(20240301)
        rng.shuffle(hd_vals)
        cut = int(len(hd_vals) * 0.8)
        hd_tr = set(hd_vals[:cut])
        tr = train_df[train_df[args.hd_col].astype(str).isin(hd_tr)].copy()
        va = train_df[~train_df[args.hd_col].astype(str).isin(hd_tr)].copy()

        Xtr, ytr, meta_tr, feat_cols = build_Xy(tr, target_col=args.target_col, id_cols=id_cols, drop_cols=drop_cols)
        Xva, yva, meta_va, _         = build_Xy(va, target_col=args.target_col, id_cols=id_cols, drop_cols=drop_cols)

        # 文字列/カテゴリ列の自動抽出
        cat_cols = [c for c in feat_cols if (str(Xtr[c].dtype).startswith("category"))]

        model.fit(Xtr, ytr, categorical=cat_cols, valid=(Xva, yva))

        # 検証RMSE
        yhat_va = model.predict(Xva)
        print(f"[VAL] RMSE={rmse(yva, yhat_va):.6f}  (n={len(yva)})")

        # モデル保存
        _ensure_dir(Path(args.model_out))
        model.save(args.model_out)
        print(f"[WRITE] model -> {args.model_out}")

    else:
        # 推論のみでモデル必要
        if not Path(args.model_out).exists():
            raise FileNotFoundError(f"model file not found: {args.model_out}")
        model = STRegressor.load(args.model_out)

    # ---- 推論（test_df があればそれで出す。無ければ学習データ全体で出す）----
    pred_base = test_df if test_df is not None else train_df
    if pred_base is not None:
        # 目的列が無くても OK
        has_target = args.target_col in pred_base.columns
        drop_cols_pred = list(drop_cols)
        if has_target:
            drop_cols_pred.append(args.target_col)

        Xte, yte, meta_te, feat_cols_te = build_Xy(pred_base,
                                                   target_col=(args.target_col if has_target else args.hd_col),  # ダミー
                                                   id_cols=id_cols,
                                                   drop_cols=drop_cols_pred)
        # has_target=False の場合、上のダミー指定で y は hd になるため破棄
        if not has_target:
            yte = np.zeros(len(meta_te), dtype=float)

        yhat = model.predict(Xte)

        out = meta_te.copy()
        out["st_pred"] = yhat
        if has_target:
            out["st_true"] = pd.to_numeric(pred_base[args.target_col], errors="coerce").to_numpy(dtype=float)

        _ensure_dir(Path(args.pred_out))
        out.to_csv(args.pred_out, index=False, encoding="utf-8")
        print(f"[WRITE] preds -> {args.pred_out}  (n={len(out)})")

        if has_target:
            print(f"[TEST] RMSE={rmse(out['st_true'].to_numpy(dtype=float), out['st_pred'].to_numpy(dtype=float)):.6f}")

if __name__ == "__main__":
    main()
