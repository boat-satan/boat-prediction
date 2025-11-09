#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/boatpred/st_model.py

ST(スタートタイミング)回帰モデルのコア:
- DataFrame から特徴量/目的変数を抽出
- LightGBM で学習 (回帰)
- 予測
- モデルの保存/読み込み

前提: 1行=1艇（レーン）レベルの行構造
必須列:
  - hd:        'YYYYMMDD' (str/int 可)
  - jcd:       場コード
  - rno:       レース番号
  - lane:      1..6
  - target_st: 実績ST (秒; 例: 0.12). 列名は引数 target_col で指定
その他は特徴量として利用

使い方（ライブラリとして）:
    from boatpred.st_model import STRegressor, build_Xy

    # 1) 特徴/目的の抽出
    X, y, meta, feat_cols = build_Xy(df, target_col="st",
                                     id_cols=("hd","jcd","rno","lane"),
                                     drop_cols=("st","some_label_only_cols"))

    # 2) 学習
    model = STRegressor().fit(X, y, categorical=cat_cols)

    # 3) 推論
    yhat = model.predict(X_new)

    # 4) 保存/読込
    model.save("data/models/st_lgbm.txt")
    model2 = STRegressor.load("data/models/st_lgbm.txt")

依存: lightgbm, pandas, numpy
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb


# ---------------------- 前処理ユーティリティ ----------------------
def _as_category_inplace(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns and (df[c].dtype == "object" or str(df[c].dtype).startswith("category")):
            df[c] = df[c].astype("category")


def build_Xy(df_entries: pd.DataFrame,
             target_col: str = "st",
             id_cols: Iterable[str] = ("hd", "jcd", "rno", "lane"),
             drop_cols: Iterable[str] = ()) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, List[str]]:
    """
    1行=1艇の DataFrame から (X, y, meta, feat_cols) を作る。
    - meta: id列のみ（hd,jcd,rno,lane）を返す（予測後のマージ用）
    - feat_cols: 学習に使用した特徴量リスト
    """
    df = df_entries.copy()

    # id/meta
    id_cols = list(id_cols)
    for c in id_cols:
        if c not in df.columns:
            raise KeyError(f"required id column not found: {c}")

    if target_col not in df.columns:
        raise KeyError(f"target column not found: {target_col}")

    meta = df[id_cols].copy()
    y = pd.to_numeric(df[target_col], errors="coerce").to_numpy(dtype=float)

    # 使用不可列を落として特徴量抽出
    banned = set(id_cols) | {target_col} | set(drop_cols)
    feat_cols = [c for c in df.columns if c not in banned]

    # 欠損や文字列の最低限対応
    X = df[feat_cols].copy()
    # object→category（カテゴリ扱い）
    obj_cols = [c for c in feat_cols if X[c].dtype == "object"]
    _as_category_inplace(X, obj_cols)

    # すべて数値/カテゴリへ（LightGBMはカテゴリを直接受けられる）
    return X, y, meta, feat_cols


# ---------------------- モデル本体 ----------------------
@dataclass
class STRegressor:
    """
    LightGBM回帰の薄いラッパ。
    """
    params: Optional[dict] = None
    num_boost_round: int = 2000
    early_stopping_rounds: Optional[int] = 200
    model: Optional[lgb.Booster] = None
    feature_names_: Optional[List[str]] = None
    categorical_: Optional[List[str]] = None

    def default_params(self) -> dict:
        return dict(
            objective="regression",
            metric="rmse",
            learning_rate=0.03,
            num_leaves=127,
            max_depth=-1,
            min_data_in_leaf=80,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            lambda_l2=1.0,
            verbose=-1,
            num_threads=0,
            seed=20240301,
            force_col_wise=True,
        )

    def fit(self,
            X: pd.DataFrame,
            y: np.ndarray,
            categorical: Optional[Iterable[str]] = None,
            valid: Optional[Tuple[pd.DataFrame, np.ndarray]] = None) -> "STRegressor":
        """
        学習。validが無ければ学習データをそのまま検証に用いる（過学習検知用ログ程度）。
        """
        self.feature_names_ = list(X.columns)
        self.categorical_ = [c for c in (categorical or []) if c in X.columns]

        params = (self.params or self.default_params()).copy()

        train_set = lgb.Dataset(
            X,
            label=y,
            categorical_feature=self.categorical_,
            feature_name=self.feature_names_,
            free_raw_data=False,
        )

        valid_sets = [train_set]
        valid_names = ["train"]

        if valid is not None:
            Xv, yv = valid
            valid_set = lgb.Dataset(
                Xv,
                label=yv,
                categorical_feature=[c for c in self.categorical_ if c in Xv.columns],
                feature_name=list(Xv.columns),
                free_raw_data=False,
            )
            valid_sets = [train_set, valid_set]
            valid_names = ["train", "valid"]

        self.model = lgb.train(
            params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=[lgb.log_evaluation(period=200)]
                     + ([lgb.early_stopping(self.early_stopping_rounds)] if self.early_stopping_rounds else [])
        )
        return self

    def predict(self, X: pd.DataFrame, num_iteration: Optional[int] = None) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        return self.model.predict(X, num_iteration=num_iteration or self.model.best_iteration)

    # --------- 保存/読込 ----------
    def save(self, path: str) -> None:
        if self.model is None:
            raise RuntimeError("Model is not fitted.")
        self.model.save_model(path)

    @classmethod
    def load(cls, path: str) -> "STRegressor":
        booster = lgb.Booster(model_file=path)
        reg = cls()
        reg.model = booster
        # feature名は格納されているが、外部から参照したい場合は Booster.feature_name() を使う
        reg.feature_names_ = booster.feature_name()
        return reg


# ---------------------- 簡易評価 ----------------------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def group_rmse(meta: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
               group_cols: Iterable[str] = ("hd", "jcd")) -> pd.DataFrame:
    """
    日別/場別などで RMSE を集計する補助。
    """
    df = meta.copy()
    df["y_true"] = y_true
    df["y_pred"] = y_pred
    out = []
    for keys, g in df.groupby(list(group_cols), as_index=False):
        out.append({**({c: g.iloc[0][c] for c in group_cols}), "rmse": rmse(g["y_true"], g["y_pred"])})
    return pd.DataFrame(out)
