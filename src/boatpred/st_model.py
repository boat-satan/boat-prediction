# src/boatpred/st_model.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from datetime import datetime
import json
import numpy as np
import pandas as pd
import lightgbm as lgb


# ========= 時間減衰ユーティリティ =========
def _to_date(s: str) -> datetime:
    return datetime.strptime(str(s), "%Y%m%d")


def time_decay_weights(hd: pd.Series, ref_end: str, half_life_days: float) -> np.ndarray:
    """指数減衰 w = 2^(-days/half_life). hdはYYYYMMDD文字列または数値。"""
    ref = _to_date(ref_end)
    days = (ref - hd.astype(str).map(_to_date)).dt.days.astype(float)
    w = np.power(2.0, -days / max(half_life_days, 1e-6))
    w[w < 1e-12] = 0.0
    return w.values


# ========= 設定 =========
@dataclass
class STConfig:
    # 目標値(ST)のクリップ範囲（単位: 秒）
    st_clip_min: float = -0.20
    st_clip_max: float = 0.40

    # 時間減衰
    half_life_racer: float = 240.0     # 選手個人の時系列
    half_life_course: float = 240.0    # コース平均

    # 直近窓
    roll_short: int = 5
    roll_long: int = 20

    # LGBM回帰のハイパラ（軽めのデフォルト）
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_boost_round: int = 800
    seed: int = 20240301
    num_threads: int = 0

    # メタ保存
    model_out: str = "data/models/st_lgbm.txt"
    meta_out: str = "data/models/st_lgbm.meta.json"


# ========= 特徴量生成（学習/推論共通で使う） =========
def _clip_st(x: pd.Series, mn: float, mx: float) -> pd.Series:
    return x.clip(lower=mn, upper=mx)


def _agg_mean_std(x: pd.Series) -> Tuple[float, float]:
    return float(x.mean()), float(x.std(ddof=0) if len(x) > 1 else 0.0)


def _build_historical_tables(
    hist_df: pd.DataFrame,
    ref_end: str,
    cfg: STConfig,
    hd_col: str = "hd",
    reg_col: str = "regno",
    lane_col: str = "lane",
    st_col: str = "st",
) -> Dict[str, pd.DataFrame]:
    """
    過去実績から派生テーブルを作成:
      - 選手×コース（加重平均/分散、直近rolling）
      - 選手全体
      - コース平均
    """
    df = hist_df[[hd_col, reg_col, lane_col, st_col]].dropna().copy()
    df[lane_col] = df[lane_col].astype(int)
    df[st_col] = _clip_st(df[st_col].astype(float), cfg.st_clip_min, cfg.st_clip_max)

    # 重み
    w_r = time_decay_weights(df[hd_col], ref_end, cfg.half_life_racer)
    w_c = time_decay_weights(df[hd_col], ref_end, cfg.half_life_course)
    df["_wr"] = w_r
    df["_wc"] = w_c

    # ---- 選手×コース：時間減衰付き 平均/分散 と 直近ローリング ----
    # 時系列ソート
    df = df.sort_values([reg_col, lane_col, hd_col]).reset_index(drop=True)

    # 加重平均・分散
    def _w_stats(group: pd.DataFrame) -> pd.Series:
        w = group["_wr"].values
        x = group[st_col].values
        if w.sum() <= 0:
            mu = float(np.nan)
            var = float(np.nan)
        else:
            mu = float(np.average(x, weights=w))
            var = float(np.average((x - mu) ** 2, weights=w))
        # 直近ローリング（重み無しの素の移動平均）
        x_s = pd.Series(x)
        r_short = float(x_s.tail(cfg.roll_short).mean()) if len(x_s) >= 1 else np.nan
        r_long = float(x_s.tail(cfg.roll_long).mean()) if len(x_s) >= 1 else np.nan
        return pd.Series({
            "st_rc_mu": mu,
            "st_rc_var": var,
            "st_rc_roll_s": r_short,
            "st_rc_roll_l": r_long,
            "n_rc": float((group["_wr"] > 0).sum())
        })

    rc_tbl = (
        df.groupby([reg_col, lane_col], as_index=False)
          .apply(_w_stats)
          .reset_index(drop=True)
    )

    # ---- 選手全体：時間減衰付き ----
    def _r_stats(group: pd.DataFrame) -> pd.Series:
        w = group["_wr"].values
        x = group[st_col].values
        if w.sum() <= 0:
            mu = float(np.nan)
            var = float(np.nan)
        else:
            mu = float(np.average(x, weights=w))
            var = float(np.average((x - mu) ** 2, weights=w))
        x_s = pd.Series(x)
        r_short = float(x_s.tail(cfg.roll_short).mean()) if len(x_s) >= 1 else np.nan
        r_long = float(x_s.tail(cfg.roll_long).mean()) if len(x_s) >= 1 else np.nan
        return pd.Series({
            "st_r_mu": mu,
            "st_r_var": var,
            "st_r_roll_s": r_short,
            "st_r_roll_l": r_long,
            "n_r": float((group["_wr"] > 0).sum())
        })

    r_tbl = (
        df.groupby(reg_col, as_index=False)
          .apply(_r_stats)
          .reset_index(drop=True)
    )

    # ---- コース平均：時間減衰付き ----
    def _c_stats(group: pd.DataFrame) -> pd.Series:
        w = group["_wc"].values
        x = group[st_col].values
        if w.sum() <= 0:
            mu = float(np.nan)
            var = float(np.nan)
        else:
            mu = float(np.average(x, weights=w))
            var = float(np.average((x - mu) ** 2, weights=w))
        return pd.Series({
            "st_c_mu": mu,
            "st_c_var": var,
            "n_c": float((group["_wc"] > 0).sum())
        })

    c_tbl = (
        df.groupby(lane_col, as_index=False)
          .apply(_c_stats)
          .reset_index(drop=True)
    )

    return {"rc": rc_tbl, "r": r_tbl, "c": c_tbl}


def build_features_for_training(
    hist_df: pd.DataFrame,
    ref_end: str,
    cfg: STConfig,
    hd_col: str = "hd",
    reg_col: str = "regno",
    lane_col: str = "lane",
    st_col: str = "st",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    学習用に特徴を構築。ターゲット列 'y' を含むDataFrameを返す。
    """
    tables = _build_historical_tables(hist_df, ref_end, cfg, hd_col, reg_col, lane_col, st_col)

    df = hist_df[[hd_col, reg_col, lane_col, st_col]].dropna().copy()
    df[lane_col] = df[lane_col].astype(int)
    df[st_col] = _clip_st(df[st_col].astype(float), cfg.st_clip_min, cfg.st_clip_max)

    # マージ
    df = df.merge(tables["rc"], on=[reg_col, lane_col], how="left")
    df = df.merge(tables["r"],  on=[reg_col],           how="left")
    df = df.merge(tables["c"],  on=[lane_col],          how="left")

    # 目的変数
    df["y"] = df[st_col].astype(float)

    # 欠損補完（極端な欠損はコース平均との差/分散で埋める）
    for col in ["st_rc_mu", "st_rc_var", "st_rc_roll_s", "st_rc_roll_l",
                "st_r_mu", "st_r_var", "st_r_roll_s", "st_r_roll_l",
                "st_c_mu", "st_c_var",
                "n_rc", "n_r", "n_c"]:
        if col in df.columns:
            if col.startswith("st_"):
                df[col] = df[col].fillna(df["st_c_mu"] if "mu" in col else 0.0)
            else:
                df[col] = df[col].fillna(0.0)

    # 特徴リスト
    feat_cols = [
        "st_rc_mu","st_rc_var","st_rc_roll_s","st_rc_roll_l",
        "st_r_mu","st_r_var","st_r_roll_s","st_r_roll_l",
        "st_c_mu","st_c_var",
        "n_rc","n_r","n_c",
        # 補助的に lane を数値特徴として入れる
        lane_col
    ]
    return df[[hd_col, reg_col, lane_col, "y"] + feat_cols].copy(), feat_cols


# ========= 学習 / 保存 / ロード =========
def train(
    hist_df: pd.DataFrame,
    ref_end: str,
    cfg: STConfig = STConfig(),
    hd_col: str = "hd",
    reg_col: str = "regno",
    lane_col: str = "lane",
    st_col: str = "st",
) -> Tuple[lgb.Booster, List[str]]:
    """
    過去実績から特徴を作り LightGBM で ST回帰モデルを学習。
    """
    train_df, feat_cols = build_features_for_training(hist_df, ref_end, cfg, hd_col, reg_col, lane_col, st_col)

    X = train_df[feat_cols]
    y = train_df["y"].values

    ds = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=True)
    params = dict(
        objective="regression",
        metric="l2",
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.feature_fraction,
        bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq,
        lambda_l2=cfg.lambda_l2,
        num_threads=cfg.num_threads,
        seed=cfg.seed,
        verbose=-1,
        force_col_wise=True,
    )
    booster = lgb.train(
        params,
        ds,
        num_boost_round=cfg.num_boost_round,
        valid_sets=[ds],
        valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)],
    )
    # 保存
    PathLike(cfg.model_out).parent.mkdir(parents=True, exist_ok=True)  # type: ignore
    booster.save_model(cfg.model_out)
    meta = {"config": asdict(cfg), "feature_names": feat_cols, "ref_end": ref_end}
    with open(cfg.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return booster, feat_cols


def load(model_path: str) -> lgb.Booster:
    return lgb.Booster(model_file=model_path)


# ========= 推論（出走表に対して予測） =========
def build_features_for_racecard(
    racecard_df: pd.DataFrame,
    history_df: pd.DataFrame,
    ref_end: str,
    cfg: STConfig = STConfig(),
    hd_col: str = "hd",
    reg_col: str = "regno",
    lane_col: str = "lane",
    st_col: str = "st",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    当日の出走表（racecard_df: hd, jcd, rno, lane, regno）に
    過去実績（history_df: hd, lane, regno, st）から派生特徴を付与して返す。
    """
    tables = _build_historical_tables(history_df, ref_end, cfg, hd_col, reg_col, lane_col, st_col)

    use = racecard_df.copy()
    use[lane_col] = use[lane_col].astype(int)
    use[reg_col] = use[reg_col].astype(str)

    use = use.merge(tables["rc"], on=[reg_col, lane_col], how="left")
    use = use.merge(tables["r"],  on=[reg_col],           how="left")
    use = use.merge(tables["c"],  on=[lane_col],          how="left")

    for col in ["st_rc_mu","st_rc_var","st_rc_roll_s","st_rc_roll_l",
                "st_r_mu","st_r_var","st_r_roll_s","st_r_roll_l",
                "st_c_mu","st_c_var","n_rc","n_r","n_c"]:
        if col in use.columns:
            if col.startswith("st_"):
                use[col] = use[col].fillna(use["st_c_mu"] if "mu" in col else 0.0)
            else:
                use[col] = use[col].fillna(0.0)

    feat_cols = [
        "st_rc_mu","st_rc_var","st_rc_roll_s","st_rc_roll_l",
        "st_r_mu","st_r_var","st_r_roll_s","st_r_roll_l",
        "st_c_mu","st_c_var",
        "n_rc","n_r","n_c",
        lane_col
    ]
    return use, feat_cols


def predict(
    booster: lgb.Booster,
    racecard_df: pd.DataFrame,
    history_df: pd.DataFrame,
    ref_end: str,
    cfg: STConfig = STConfig(),
    hd_col: str = "hd",
    jcd_col: str = "jcd",
    rno_col: str = "rno",
    reg_col: str = "regno",
    lane_col: str = "lane",
    st_col: str = "st",
) -> pd.DataFrame:
    """
    racecard_df（hd,jcd,rno,lane,regno ...）に対して ST を予測して返す。
    出力: hd,jcd,rno,lane,regno,st_pred
    """
    feat_df, feat_cols = build_features_for_racecard(
        racecard_df, history_df, ref_end, cfg, hd_col, reg_col, lane_col, st_col
    )
    X = feat_df[feat_cols]
    st_pred = booster.predict(X, num_iteration=booster.best_iteration)
    out = racecard_df[[hd_col, jcd_col, rno_col, lane_col, reg_col]].copy()
    out["st_pred"] = st_pred
    # 物理的にあり得ない極端値を丸める（学習クリップと同じレンジに）
    out["st_pred"] = out["st_pred"].clip(cfg.st_clip_min, cfg.st_clip_max)
    return out


# ========= ちょい便利 =========
class PathLike(str):
    """str を pathlib.Path ライクに parent.mkdir したいだけの小道具"""
    @property
    def parent(self):
        import os, pathlib
        return pathlib.Path(os.path.dirname(self))


# ========= 使い方メモ =========
# 1) 学習
#   hist = pd.read_parquet("data/history/results.parquet")  # 必須列: hd, regno, lane, st
#   booster, feats = train(hist, ref_end="20240229", cfg=STConfig())
#
# 2) 推論
#   race = pd.read_parquet("data/racecards/20240301.parquet")  # 必須列: hd,jcd,rno,regno,lane
#   pred = predict(booster, race, hist, ref_end="20240229", cfg=STConfig())
#   # pred: hd,jcd,rno,lane,regno,st_pred
