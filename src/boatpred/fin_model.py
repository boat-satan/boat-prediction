# src/boatpred/fin_model.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json
import numpy as np
import pandas as pd
import lightgbm as lgb


# ===================== 共通ユーティリティ =====================
def _to_date(s: str) -> datetime:
    return datetime.strptime(str(s), "%Y%m%d")


def time_decay_weights(hd: pd.Series, ref_end: str, half_life_days: float) -> np.ndarray:
    """指数減衰 w = 2^(-days/half_life). hdはYYYYMMDDの文字/数。"""
    ref = _to_date(ref_end)
    days = (ref - hd.astype(str).map(_to_date)).dt.days.astype(float)
    w = np.power(2.0, -days / max(half_life_days, 1e-6))
    w[w < 1e-12] = 0.0
    return w.values


# 決まり手の正規化（学習ラベル統一）
FIN_MAP = {
    "逃げ": "nige",
    "差し": "sashi",
    "まくり": "makuri",
    "まくり差し": "makuri_sashi",
    "抜き": "sonota",
    "恵まれ": "sonota",
    "その他": "sonota",
}
FIN_CLASSES = ["nige", "sashi", "makuri", "makuri_sashi", "sonota"]
FIN_TO_ID = {c: i for i, c in enumerate(FIN_CLASSES)}


def normalize_fin(x: str) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s in FIN_MAP:
        return FIN_MAP[s]
    # 既に英語想定で来た場合
    if s in FIN_CLASSES:
        return s
    return "sonota"


# ===================== 設定 =====================
@dataclass
class FinConfig:
    # 時間減衰
    half_life_racer: float = 240.0
    half_life_course: float = 240.0

    # ローリング窓（素の移動平均）
    roll_short: int = 10
    roll_long: int = 40

    # 事前分布の強さ（Beta/Dirichlet）
    #  単勝のBeta(α0, β0)は、コース基準の事前（例: 1コース強めなど）を想定
    beta_alpha0: float = 2.0
    beta_beta0: float = 8.0
    #  決まり手のDirichlet(αベクトル)。nigeをやや強め、その他最小限。
    dirichlet_alpha_base: Dict[str, float] = None  # Noneならデフォルトを適用

    # LightGBMハイパラ（win=Binary, fin=Multiclass）
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

    # 出力先
    win_model_out: str = "data/models/win_lgbm.txt"
    fin_model_out: str = "data/models/fin_lgbm.txt"
    meta_out: str = "data/models/fin_meta.json"

    def __post_init__(self):
        if self.dirichlet_alpha_base is None:
            self.dirichlet_alpha_base = {
                "nige": 2.0,
                "sashi": 1.0,
                "makuri": 1.2,
                "makuri_sashi": 1.2,
                "sonota": 0.6,
            }


# ===================== 集計テーブル作成 =====================
def _build_aggregates(
    hist_df: pd.DataFrame,
    ref_end: str,
    cfg: FinConfig,
    hd_col="hd", reg_col="regno", lane_col="lane",
    win_col="is_win", fin_col="fin"
) -> Dict[str, pd.DataFrame]:
    """
    過去実績から以下を作る:
      - 選手×コース: win率, fin分布（時間減衰つき）＋直近窓
      - 選手全体:    win率, fin分布（時間減衰つき）
      - コース全体:  win率, fin分布（時間減衰つき）
    """
    df = hist_df[[hd_col, reg_col, lane_col, win_col, fin_col]].copy()
    df[lane_col] = df[lane_col].astype(int)
    df[win_col] = df[win_col].astype(int)
    df[fin_col] = df[fin_col].map(normalize_fin)

    # 時間減衰（選手用・コース用は同じハーフライフでもOK）
    w_r = time_decay_weights(df[hd_col], ref_end, cfg.half_life_racer)
    df["_w"] = w_r

    # --- 決まり手 one-hot ---
    for c in FIN_CLASSES:
        df[f"fin_{c}"] = (df[fin_col] == c).astype(int) * df[win_col]  # 勝者の決まり手のみ1

    # ========== 選手×コース ==========
    df = df.sort_values([reg_col, lane_col, hd_col]).reset_index(drop=True)

    def _rc_stats(g: pd.DataFrame) -> pd.Series:
        w = g["_w"].values
        win = g[win_col].values
        # 加重和
        w_sum = w.sum()
        win_wsum = float(np.dot(win, w))
        # Beta 平滑のための事前（コース基準は別で掛けるのでRCは素の確率）
        p_win = (win_wsum + 0.0) / (w_sum + 1e-9)

        # fin 分布（勝ったレコードのみ one-hot を重み付き和）
        fin_counts = {c: float(np.dot(g[f"fin_{c}"].values, w)) for c in FIN_CLASSES}
        fin_total = sum(fin_counts.values())
        if fin_total > 0:
            fin_dist = {c: fin_counts[c] / fin_total for c in FIN_CLASSES}
        else:
            fin_dist = {c: np.nan for c in FIN_CLASSES}

        # 直近ローリング（素の移動平均; win率のみ）
        win_s = pd.Series(win.astype(float))
        r_short = float(win_s.tail(cfg.roll_short).mean()) if len(win_s) > 0 else np.nan
        r_long  = float(win_s.tail(cfg.roll_long).mean())  if len(win_s) > 0 else np.nan

        out = {
            "rc_win_p": float(p_win),
            "rc_win_roll_s": r_short,
            "rc_win_roll_l": r_long,
            "rc_w": float((w > 0).sum()),
        }
        for c in FIN_CLASSES:
            out[f"rc_fin_{c}"] = fin_dist[c]
            out[f"rc_finw_{c}"] = fin_counts[c]
        out["rc_finw_total"] = fin_total
        return pd.Series(out)

    rc_tbl = (
        df.groupby([reg_col, lane_col], as_index=False)
          .apply(_rc_stats)
          .reset_index(drop=True)
    )

    # ========== 選手全体 ==========
    def _r_stats(g: pd.DataFrame) -> pd.Series:
        w = g["_w"].values
        win = g[win_col].values
        w_sum = w.sum()
        win_wsum = float(np.dot(win, w))
        p_win = (win_wsum + 0.0) / (w_sum + 1e-9)

        fin_counts = {c: float(np.dot(g[f"fin_{c}"].values, w)) for c in FIN_CLASSES}
        fin_total = sum(fin_counts.values())
        if fin_total > 0:
            fin_dist = {c: fin_counts[c] / fin_total for c in FIN_CLASSES}
        else:
            fin_dist = {c: np.nan for c in FIN_CLASSES}

        out = {"r_win_p": float(p_win), "r_w": float((w > 0).sum())}
        for c in FIN_CLASSES:
            out[f"r_fin_{c}"] = fin_dist[c]
            out[f"r_finw_{c}"] = fin_counts[c]
        out["r_finw_total"] = fin_total
        return pd.Series(out)

    r_tbl = (
        df.groupby(reg_col, as_index=False)
          .apply(_r_stats)
          .reset_index(drop=True)
    )

    # ========== コース全体 ==========
    def _c_stats(g: pd.DataFrame) -> pd.Series:
        w = g["_w"].values
        win = g[win_col].values
        w_sum = w.sum()
        win_wsum = float(np.dot(win, w))
        p_win = (win_wsum + 0.0) / (w_sum + 1e-9)

        fin_counts = {c: float(np.dot(g[f"fin_{c}"].values, w)) for c in FIN_CLASSES}
        fin_total = sum(fin_counts.values())
        if fin_total > 0:
            fin_dist = {c: fin_counts[c] / fin_total for c in FIN_CLASSES}
        else:
            fin_dist = {c: np.nan for c in FIN_CLASSES}

        out = {"c_win_p": float(p_win), "c_w": float((w > 0).sum())}
        for c in FIN_CLASSES:
            out[f"c_fin_{c}"] = fin_dist[c]
            out[f"c_finw_{c}"] = fin_counts[c]
        out["c_finw_total"] = fin_total
        return pd.Series(out)

    c_tbl = (
        df.groupby(lane_col, as_index=False)
          .apply(_c_stats)
          .reset_index(drop=True)
    )

    return {"rc": rc_tbl, "r": r_tbl, "c": c_tbl}


# ===================== 特徴量作成 =====================
def build_features_for_training(
    hist_df: pd.DataFrame,
    ref_end: str,
    cfg: FinConfig,
    hd_col="hd", reg_col="regno", lane_col="lane",
    win_col="is_win", fin_col="fin",
    use_st: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    学習用の特徴を作成し、(win学習データ, fin学習データ, win_feat_cols, fin_feat_cols) を返す。
    - hist_df に st_pred があれば特徴に含める（use_st=True時）
    - win学習: 全行に対して is_win を目的変数
    - fin学習: 勝者のみ抽出し fin(正規化5クラス) を目的変数
    """
    aggs = _build_aggregates(hist_df, ref_end, cfg, hd_col, reg_col, lane_col, win_col, fin_col)

    df = hist_df.copy()
    df[lane_col] = df[lane_col].astype(int)
    df[win_col] = df[win_col].astype(int)
    df["fin_norm"] = df[fin_col].map(normalize_fin)

    # マージ
    df = df.merge(aggs["rc"], on=[reg_col, lane_col], how="left")
    df = df.merge(aggs["r"],  on=[reg_col],           how="left")
    df = df.merge(aggs["c"],  on=[lane_col],          how="left")

    # 欠損埋め（win率はコース平均、fin分布はコース分布、重みは0）
    for col in ["rc_win_p","rc_win_roll_s","rc_win_roll_l","r_win_p","c_win_p","rc_w","r_w","c_w"]:
        if col in df.columns:
            if col.endswith("_p"):
                df[col] = df[col].fillna(df["c_win_p"])
            else:
                df[col] = df[col].fillna(0.0)
    for src in ["rc","r","c"]:
        for c in FIN_CLASSES:
            prob_col = f"{src}_fin_{c}"
            count_col = f"{src}_finw_{c}"
            if prob_col in df.columns:
                df[prob_col] = df[prob_col].fillna(1.0/len(FIN_CLASSES))  # 極端な欠損は一様
            if count_col in df.columns:
                df[count_col] = df[count_col].fillna(0.0)

    # ST予測があれば特徴に
    st_cols = []
    if use_st and "st_pred" in df.columns:
        st_cols = ["st_pred"]

    # --- 単勝モデル用特徴 ---
    win_feat_cols = [
        "rc_win_p","rc_win_roll_s","rc_win_roll_l",
        "r_win_p","c_win_p",
        "rc_w","r_w","c_w",
        lane_col
    ] + st_cols

    win_train = df[[hd_col, reg_col, lane_col, win_col] + win_feat_cols].copy()
    win_train.rename(columns={win_col: "y"}, inplace=True)

    # --- 決まり手モデル用特徴（勝者のみ）---
    fin_df = df[df[win_col] == 1].copy()
    fin_df = fin_df[~fin_df["fin_norm"].isna()].copy()
    fin_target = fin_df["fin_norm"].map(FIN_TO_ID).astype(int)

    fin_feat_cols = []
    for src in ["rc","r","c"]:
        for c in FIN_CLASSES:
            fin_feat_cols.append(f"{src}_fin_{c}")
    # 勝率系も入れる（逃げやすさ等のproxy）
    fin_feat_cols += ["rc_win_p", "r_win_p", "c_win_p", lane_col] + st_cols

    fin_train = fin_df[[hd_col, reg_col, lane_col, "fin_norm"] + fin_feat_cols].copy()
    fin_train["y"] = fin_target.values

    return win_train, fin_train, win_feat_cols, fin_feat_cols


# ===================== 学習・保存・ロード =====================
def train(
    hist_df: pd.DataFrame,
    ref_end: str,
    cfg: FinConfig = FinConfig(),
    hd_col="hd", reg_col="regno", lane_col="lane",
    win_col="is_win", fin_col="fin",
    use_st: bool = True
) -> Tuple[lgb.Booster, lgb.Booster, List[str], List[str]]:
    """
    単勝（Binary）と決まり手（Multiclass）を学習。
    """
    win_train, fin_train, win_feats, fin_feats = build_features_for_training(
        hist_df, ref_end, cfg, hd_col, reg_col, lane_col, win_col, fin_col, use_st=use_st
    )

    # ---- 単勝 Binary ----
    Xw = win_train[win_feats]
    yw = win_train["y"].astype(int).values
    ds_w = lgb.Dataset(Xw, label=yw, feature_name=win_feats, free_raw_data=True)

    params_w = dict(
        objective="binary",
        metric="auc",
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
    booster_win = lgb.train(
        params_w, ds_w, num_boost_round=cfg.num_boost_round,
        valid_sets=[ds_w], valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)]
    )

    # ---- 決まり手 Multiclass（勝者のみ）----
    Xf = fin_train[fin_feats]
    yf = fin_train["y"].astype(int).values
    ds_f = lgb.Dataset(Xf, label=yf, feature_name=fin_feats, free_raw_data=True)

    params_f = dict(
        objective="multiclass",
        num_class=len(FIN_CLASSES),
        metric="multi_logloss",
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
    booster_fin = lgb.train(
        params_f, ds_f, num_boost_round=cfg.num_boost_round,
        valid_sets=[ds_f], valid_names=["train"],
        callbacks=[lgb.log_evaluation(period=200)]
    )

    # 保存
    PathLike(cfg.win_model_out).parent.mkdir(parents=True, exist_ok=True)
    booster_win.save_model(cfg.win_model_out)
    booster_fin.save_model(cfg.fin_model_out)
    meta = {
        "config": asdict(cfg),
        "win_features": win_feats,
        "fin_features": fin_feats,
        "ref_end": ref_end,
        "fin_classes": FIN_CLASSES,
    }
    with open(cfg.meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return booster_win, booster_fin, win_feats, fin_feats


def load(win_model_path: str, fin_model_path: str) -> Tuple[lgb.Booster, lgb.Booster]:
    return lgb.Booster(model_file=win_model_path), lgb.Booster(model_file=fin_model_path)


# ===================== 推論（出走表→単勝/決まり手） =====================
def build_features_for_racecard(
    racecard_df: pd.DataFrame,
    history_df: pd.DataFrame,
    ref_end: str,
    cfg: FinConfig = FinConfig(),
    hd_col="hd", reg_col="regno", lane_col="lane",
    win_col="is_win", fin_col="fin",
    use_st: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    出走表に対して、学習時と同じ派生特徴を付与して（win用, fin用）を返す。
    racecard_df 必須: hd, jcd, rno, regno, lane (+ 任意 st_pred)
    """
    aggs = _build_aggregates(history_df, ref_end, cfg, hd_col, reg_col, lane_col, win_col, fin_col)

    use = racecard_df.copy()
    use[lane_col] = use[lane_col].astype(int)
    use[reg_col] = use[reg_col].astype(str)

    use = use.merge(aggs["rc"], on=[reg_col, lane_col], how="left")
    use = use.merge(aggs["r"],  on=[reg_col],           how="left")
    use = use.merge(aggs["c"],  on=[lane_col],          how="left")

    # 欠損埋め
    for col in ["rc_win_p","rc_win_roll_s","rc_win_roll_l","r_win_p","c_win_p","rc_w","r_w","c_w"]:
        if col in use.columns:
            if col.endswith("_p"):
                use[col] = use[col].fillna(use["c_win_p"])
            else:
                use[col] = use[col].fillna(0.0)
    for src in ["rc","r","c"]:
        for c in FIN_CLASSES:
            prob_col = f"{src}_fin_{c}"
            count_col = f"{src}_finw_{c}"
            if prob_col in use.columns:
                use[prob_col] = use[prob_col].fillna(1.0/len(FIN_CLASSES))
            if count_col in use.columns:
                use[count_col] = use[count_col].fillna(0.0)

    st_cols = ["st_pred"] if (use_st and "st_pred" in use.columns) else []

    win_feat_cols = [
        "rc_win_p","rc_win_roll_s","rc_win_roll_l",
        "r_win_p","c_win_p","rc_w","r_w","c_w",
        lane_col
    ] + st_cols

    fin_feat_cols = []
    for src in ["rc","r","c"]:
        for c in FIN_CLASSES:
            fin_feat_cols.append(f"{src}_fin_{c}")
    fin_feat_cols += ["rc_win_p","r_win_p","c_win_p", lane_col] + st_cols

    return use.copy(), use.copy(), win_feat_cols, fin_feat_cols


def predict(
    booster_win: lgb.Booster,
    booster_fin: lgb.Booster,
    racecard_df: pd.DataFrame,
    history_df: pd.DataFrame,
    ref_end: str,
    cfg: FinConfig = FinConfig(),
    hd_col="hd", jcd_col="jcd", rno_col="rno",
    reg_col="regno", lane_col="lane",
    win_col="is_win", fin_col="fin",
    use_st: bool = True,
    apply_bayes_smoothing: bool = True
) -> pd.DataFrame:
    """
    出走表に対して:
      - p_win（単勝確率）
      - p_fin[5クラス]（決まり手の事後確率; 既定では p(fin|win) を事前と混合）
      - p_fin_marginal = p_win * p(fin|win)
    を返す。
    """
    use_w, use_f, win_feats, fin_feats = build_features_for_racecard(
        racecard_df, history_df, ref_end, cfg, hd_col, reg_col, lane_col, win_col, fin_col, use_st=use_st
    )

    # ---- 学習済みメタの読み込み（特徴整合性チェック用途） ----
    # 省略可: 本番では meta を読み、feature差異をlog警告など

    # ---- 予測 ----
    pw = booster_win.predict(use_w[win_feats], num_iteration=booster_win.best_iteration)
    # LightGBMのbinaryは 1 の確率を返す
    if pw.ndim > 1:
        pw = pw[:, 0]
    use_w = use_w[[hd_col, jcd_col, rno_col, lane_col, reg_col]].copy()
    use_w["p_win_raw"] = pw

    pf = booster_fin.predict(use_f[fin_feats], num_iteration=booster_fin.best_iteration)
    # pf shape: [N, 5]
    pf = np.asarray(pf, dtype=float)

    # ---- 事前分布とベイズ平滑（オプション）----
    if apply_bayes_smoothing:
        # 単勝: Beta 平滑。ここではコース事前(c_win_p)を Beta(α0,β0) に焼き直す感覚で混合。
        #   近似として、p_win_post = (α0 * c_win_p + 1 * p_win_raw) / (α0 + 1) などでもよいが
        #   ここでは「データ重み」を rc_w + r_w + c_w から導く簡易混合にする。
        alpha0 = cfg.beta_alpha0
        beta0 = cfg.beta_beta0
        # 事前平均
        prior_p = use_w.merge(
            use_f[[hd_col, lane_col, "c_win_p"]].drop_duplicates(),
            on=[hd_col, lane_col], how="left"
        )["c_win_p"].fillna(1/6).values
        # 有効サンプル重み（かなり簡略化）
        eff = racecard_df.merge(
            use_f[[hd_col, reg_col, lane_col, "rc_w","r_w","c_w"]].drop_duplicates(),
            on=[hd_col, reg_col, lane_col], how="left"
        )[["rc_w","r_w","c_w"]].fillna(0.0).sum(axis=1).clip(lower=0.0, upper=200.0).values
        # 平滑: 事前Betaの相当サンプル数 = alpha0+beta0
        post = ((alpha0 + beta0) * prior_p + eff * pw) / ((alpha0 + beta0) + eff + 1e-9)
        p_win = np.clip(post, 1e-6, 1-1e-6)
    else:
        p_win = np.clip(pw, 1e-6, 1-1e-6)

    # 決まり手: Dirichlet 平滑（rc/r/cのカウントを事前に）
    if apply_bayes_smoothing:
        # 事前カウント = rc_finw + r_finw + c_finw + alpha_base
        finw_cols = []
        for src in ["rc","r","c"]:
            for c in FIN_CLASSES:
                finw_cols.append(f"{src}_finw_{c}")
        finw = use_f[finw_cols].fillna(0.0).to_numpy(dtype=float)
        # alpha_base を足す
        alpha_vec = np.array([cfg.dirichlet_alpha_base[c] for c in FIN_CLASSES], dtype=float)
        # finw shape: [N, 3*5] → rc(5),r(5),c(5)を足し合わせ
        rc = finw[:, 0:5]
        r  = finw[:, 5:10]
        c  = finw[:, 10:15]
        prior_counts = rc + r + c + alpha_vec[None, :]
        prior_sum = prior_counts.sum(axis=1, keepdims=True)
        prior_probs = prior_counts / np.clip(prior_sum, 1e-9, None)

        # 予測pf（モデル出力）との混合（重みは有効件数由来）
        eff = racecard_df.merge(
            use_f[[hd_col, reg_col, lane_col, "rc_w","r_w","c_w"]].drop_duplicates(),
            on=[hd_col, reg_col, lane_col], how="left"
        )[["rc_w","r_w","c_w"]].fillna(0.0).sum(axis=1).clip(lower=0.0, upper=200.0).values
        eff = eff[:, None]
        # 事前強度 k を適度に（例: 50）固定。eff が大きいほどモデル出力に寄せる
        k = 50.0
        weight_model = eff / (eff + k)
        weight_prior = 1.0 - weight_model
        p_fin_given_win = np.clip(weight_model * pf + weight_prior * prior_probs, 1e-7, 1.0)
        p_fin_given_win = p_fin_given_win / p_fin_given_win.sum(axis=1, keepdims=True)
    else:
        p_fin_given_win = pf
        p_fin_given_win = p_fin_given_win / p_fin_given_win.sum(axis=1, keepdims=True)

    # ---- マージナル（winを掛ける）----
    p_fin_marginal = p_fin_given_win * p_win[:, None]

    out = racecard_df[[hd_col, jcd_col, rno_col, lane_col, reg_col]].copy()
    out["p_win"] = p_win
    for i, c in enumerate(FIN_CLASSES):
        out[f"p_{c}_cond"] = p_fin_given_win[:, i]
        out[f"p_{c}_marg"] = p_fin_marginal[:, i]
    return out


# ===================== 便利クラス =====================
class PathLike(str):
    @property
    def parent(self):
        import os, pathlib
        return pathlib.Path(os.path.dirname(self))


# ===================== 使い方メモ =====================
# hist（学習用の逐次行）に必要な最小列:
#   hd(YYYYMMDD), jcd, rno, lane(1..6), regno(str), is_win(0/1), fin(勝者の決まり手; 非勝者はNaでもOK), [任意] st_pred
#
# 1) 学習:
#   cfg = FinConfig()
#   booster_win, booster_fin, win_feats, fin_feats = train(hist_df, ref_end="20240229", cfg=cfg)
#
# 2) 予測:
#   rc = racecard_df  # 必須: hd,jcd,rno,lane,regno,[任意]st_pred
#   out = predict(booster_win, booster_fin, rc, hist_df, ref_end="20240229", cfg=cfg, use_st=True)
#   # out には p_win, p_(各決まり手)_cond（条件付き）, p_(各決まり手)_marg（周辺＝p_win掛け）が入る。
