# src/boatpred/priors.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ========== 基本ユーティリティ ==========
def _to_date(s: str) -> datetime:
    # s: "YYYYMMDD"
    return datetime.strptime(str(s), "%Y%m%d")


def time_decay_weights(hd: Sequence[str], ref_end: str, half_life_days: float = 180.0) -> np.ndarray:
    """
    hd: 各行の日付(YYYYMMDD)
    ref_end: 参照終端日(YYYYMMDD) 例: 学習終了日
    half_life_days: 半減期。大きいほど過去も効く
    返り値: 0..1 の重み（指数減衰）
    """
    ref = _to_date(ref_end)
    days = np.array([(ref - _to_date(d)).days for d in hd], dtype=float)
    # w = 2^(-days/half_life)
    w = np.power(2.0, -days / max(1e-6, half_life_days))
    # 数値安定化（極小値は0に）
    w[w < 1e-12] = 0.0
    return w


def safe_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    s = x.sum(axis=axis, keepdims=True)
    s[s == 0.0] = 1.0
    return x / s


# ========== ベイズ平滑 ==========
def beta_smooth(success: float, fail: float, alpha0: float = 2.0, beta0: float = 2.0) -> float:
    """
    成功/失敗カウントに Beta(alpha0, beta0) 事前を加えて平滑化した確率を返す
    """
    return (success + alpha0) / (success + fail + alpha0 + beta0)


def dirichlet_smooth(counts: np.ndarray, alpha0: np.ndarray) -> np.ndarray:
    """
    多項カウント + Dirichlet(alpha0) → 平滑確率ベクトル
    """
    post = counts + alpha0
    return post / post.sum()


# ========== カラム仕様 ==========
# 必須: hd(YYYYMMDD), lane(1..6), is_win(単勝=1位フラグ 0/1), fin_type(決まり手カテゴリ文字列)
# 追加: regno(選手登録番号) — 選手×コースの階層化で使用

@dataclass
class PriorsConfig:
    # 時間減衰
    half_life_win: float = 180.0
    half_life_fin: float = 180.0

    # Beta/Dirichlet 事前
    beta_alpha0_win: float = 2.0
    beta_beta0_win: float = 2.0
    # 決まり手の事前ハイパー（カテゴリ数Kと同次元）。未指定なら一様(1.0)
    dirichlet_alpha0_fin: Optional[Dict[str, float]] = None

    # 階層化（選手×コース ↔ コース平均）の擬似サンプル強度
    tau_win: float = 50.0        # 単勝率: 選手×コースが小標本のときコース平均に寄せる強さ
    tau_fin: float = 100.0       # 決まり手: 同上


# ========== 集計ヘルパ ==========
def _prepare_fin_categories(df: pd.DataFrame, fin_col: str, preset_alpha: Optional[Dict[str, float]] = None) -> Tuple[List[str], np.ndarray]:
    """
    決まり手カテゴリの順序を確定。preset_alpha があればカテゴリ順にalphaベクトルを並べる。
    """
    cats = list(pd.Index(df[fin_col].dropna().astype(str).unique()).sort_values())
    if not cats:
        # 空ならダミー1カテゴリ
        cats = ["その他"]
    if preset_alpha:
        alpha = np.array([float(preset_alpha.get(c, 1.0)) for c in cats], dtype=float)
    else:
        alpha = np.ones(len(cats), dtype=float)
    return cats, alpha


def _weighted_counts(series: pd.Series, weights: np.ndarray, categories: List[str]) -> np.ndarray:
    m = dict(zip(categories, range(len(categories))))
    acc = np.zeros(len(categories), dtype=float)
    for v, w in zip(series.astype(str).values, weights):
        idx = m.get(v)
        if idx is not None:
            acc[idx] += float(w)
    return acc


# ========== コース平均の事前 ==========
def course_priors(
    df: pd.DataFrame,
    ref_end: str,
    cfg: PriorsConfig = PriorsConfig(),
    hd_col: str = "hd",
    lane_col: str = "lane",
    win_col: str = "is_win",
    fin_col: str = "fin_type",
) -> pd.DataFrame:
    """
    コース(lane)ごとの単勝率 & 決まり手分布の事前を作る。
    返り値: laneごとに
      - win_p: Beta平滑単勝率
      - fin_{cat}: Dirichlet平滑決まり手確率（カテゴリごと）
      - n_eff_win: 単勝の有効標本量（重み合計）
      - n_eff_fin: 決まり手の有効標本量（重み合計）
    """
    use = df[[hd_col, lane_col, win_col, fin_col]].dropna(subset=[hd_col, lane_col])
    use = use.copy()
    use[lane_col] = use[lane_col].astype(int)

    # 時間減衰重み
    w_win = time_decay_weights(use[hd_col].astype(str).values, ref_end, cfg.half_life_win)
    w_fin = time_decay_weights(use[hd_col].astype(str).values, ref_end, cfg.half_life_fin)

    # 決まり手カテゴリとDirichlet事前
    cats, alpha0_vec = _prepare_fin_categories(use, fin_col, cfg.dirichlet_alpha0_fin)

    rows = []
    for lane, g in use.groupby(lane_col):
        # 単勝（Beta）
        gw = w_win[g.index]
        win_s = float((g[win_col].astype(float).values * gw).sum())
        win_f = float((1.0 - g[win_col].astype(float).values) * gw).sum()
        win_p = beta_smooth(win_s, win_f, cfg.beta_alpha0_win, cfg.beta_beta0_win)

        # 決まり手（Dirichlet）
        gwf = w_fin[g.index]
        counts = _weighted_counts(g[fin_col], gwf, cats)
        fin_p = dirichlet_smooth(counts, alpha0_vec)

        row = {"lane": int(lane), "win_p": float(win_p), "n_eff_win": float(gw.sum()), "n_eff_fin": float(gwf.sum())}
        for c, p in zip(cats, fin_p):
            row[f"fin_{c}"] = float(p)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("lane").reset_index(drop=True)
    out.attrs["fin_categories"] = cats  # 後段で再利用できるようメタに保存
    return out


# ========== 選手×コースの階層化事前 ==========
def racer_course_priors(
    df: pd.DataFrame,
    ref_end: str,
    cfg: PriorsConfig = PriorsConfig(),
    hd_col: str = "hd",
    reg_col: str = "regno",
    lane_col: str = "lane",
    win_col: str = "is_win",
    fin_col: str = "fin_type",
) -> pd.DataFrame:
    """
    選手×コースの事前（単勝率・決まり手分布）を、コース平均へ階層スムージングして推定。
    返り値: columns
      regno, lane, win_p, n_eff_win, n_eff_fin, fin_{cat}...
    """
    req = [hd_col, reg_col, lane_col, win_col, fin_col]
    use = df[req].dropna(subset=[hd_col, reg_col, lane_col]).copy()
    use[lane_col] = use[lane_col].astype(int)
    use[reg_col] = use[reg_col].astype(str)

    # コース平均
    course_base = course_priors(use, ref_end, cfg, hd_col, lane_col, win_col, fin_col)
    cats: List[str] = course_base.attrs.get("fin_categories", [])
    if not cats:
        cats = ["その他"]

    # 索引
    base_map = {
        int(row.lane): {
            "win_p": float(row.win_p),
            "fin": np.array([float(row[f"fin_{c}"]) for c in cats], dtype=float),
        }
        for _, row in course_base.iterrows()
    }

    # 時間減衰
    w_win = time_decay_weights(use[hd_col].astype(str).values, ref_end, cfg.half_life_win)
    w_fin = time_decay_weights(use[hd_col].astype(str).values, ref_end, cfg.half_life_fin)

    rows = []
    for (regno, lane), g in use.groupby([reg_col, lane_col]):
        # 単勝（観測）
        gw = w_win[g.index]
        win_s = float((g[win_col].astype(float).values * gw).sum())
        win_f = float((1.0 - g[win_col].astype(float).values) * gw).sum())

        # コース平均への階層化: 観測カウント + tau * base
        base = base_map.get(int(lane), {"win_p": 1/6, "fin": np.ones(len(cats))/len(cats)})
        # Beta擬似観測
        s_eff = win_s + cfg.tau_win * base["win_p"]
        f_eff = win_f + cfg.tau_win * (1.0 - base["win_p"])
        win_p = beta_smooth(s_eff, f_eff, cfg.beta_alpha0_win, cfg.beta_beta0_win)

        # 決まり手（観測）
        gwf = w_fin[g.index]
        counts = _weighted_counts(g["fin_type"], gwf, cats)
        # Dirichlet擬似観測
        fin_counts_eff = counts + cfg.tau_fin * base["fin"]
        fin_p = dirichlet_smooth(fin_counts_eff, np.zeros_like(fin_counts_eff))  # alpha0=0で既に擬似観測が事前相当

        row = {"regno": str(regno), "lane": int(lane), "win_p": float(win_p),
               "n_eff_win": float(gw.sum()), "n_eff_fin": float(gwf.sum())}
        for c, p in zip(cats, fin_p):
            row[f"fin_{c}"] = float(p)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["regno", "lane"]).reset_index(drop=True)
    out.attrs["fin_categories"] = cats
    return out


# ========== 使い方メモ ==========
# df は行＝艇（出走×120組の元データではなく、1走行あたり1行の「結果行」が扱いやすい）。
# 必須カラム:
#   - hd: "YYYYMMDD"
#   - lane: 1..6
#   - regno: 選手番号（racer id）
#   - is_win: 1位=1, それ以外=0
#   - fin_type: "逃げ","差し","まくり","まくり差し","その他" 等
#
# 例:
#   cfg = PriorsConfig(
#       half_life_win=240, half_life_fin=240,
#       beta_alpha0_win=2.0, beta_beta0_win=2.0,
#       dirichlet_alpha0_fin={"逃げ":1.5,"差し":1.2,"まくり":1.2,"まくり差し":1.2,"その他":0.8},
#       tau_win=80, tau_fin=150
#   )
#   course_df = course_priors(df, ref_end="20241231", cfg=cfg)
#   racer_course_df = racer_course_priors(df, ref_end="20241231", cfg=cfg)
