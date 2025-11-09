# src/boatpred/placement_model.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd
from datetime import datetime

# fin_model と同じクラス順序を採用
FIN_CLASSES = ["nige", "sashi", "makuri", "makuri_sashi", "sonota"]
FIN_TO_ID = {c: i for i, c in enumerate(FIN_CLASSES)}


# ========= ユーティリティ =========
def _to_date(s: str) -> datetime:
    return datetime.strptime(str(s), "%Y%m%d")


def time_decay_weights(hd: pd.Series, ref_end: str, half_life_days: float) -> np.ndarray:
    ref = _to_date(ref_end)
    days = (ref - hd.astype(str).map(_to_date)).dt.days.astype(float)
    w = np.power(2.0, -days / max(half_life_days, 1e-6))
    w[w < 1e-12] = 0.0
    return w.values


def normalize_fin_label(x) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    mp = {
        "逃げ": "nige",
        "差し": "sashi",
        "まくり": "makuri",
        "まくり差し": "makuri_sashi",
        "抜き": "sonota",
        "恵まれ": "sonota",
        "その他": "sonota",
    }
    if s in mp:
        return mp[s]
    if s in FIN_CLASSES:
        return s
    return "sonota"


def _safe_norm(x: np.ndarray, axis: int) -> np.ndarray:
    s = x.sum(axis=axis, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return x / s


# ========= 設定 =========
@dataclass
class PlacementConfig:
    # 参照終了日（学習・推論の境界）
    ref_end: str = "20240229"

    # 減衰
    half_life: float = 240.0

    # Dirichlet 平滑強度（2着/3着）
    alpha_p2: float = 0.5
    alpha_p3: float = 0.25

    # 保存先
    out_dir: str = "data/models"
    p2_file: str = "place_p2.npy"         # shape (5, 6, 6)
    p3_file: str = "place_p3.npy"         # shape (5, 6, 6, 6)
    meta_file: str = "place_meta.json"


# ========= 学習（頻度ベースの事前） =========
def fit_priors_from_history(
    history_df: pd.DataFrame,
    cfg: PlacementConfig = PlacementConfig(),
    hd_col="hd", jcd_col="jcd", rno_col="rno",
    lane_col="lane", fin_col="fin", rank_col="rank",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    過去結果（1着/2着/3着を同一レース内で識別できるテーブル）から
      p2[fin, L1, L2] = P(2着=L2 | 決まり手=fin, 1着=L1)
      p3[fin, L1, L2, L3] = P(3着=L3 | fin, L1, L2)
    を時間減衰付き頻度＋Dirichlet平滑で推定する。

    必須カラム:
      - hd(YYYYMMDD), jcd, rno, lane(1..6), rank(1..6), fin(勝者行に付く決まり手)
    """
    df = history_df[[hd_col, jcd_col, rno_col, lane_col, rank_col, fin_col]].copy()
    df[hd_col] = df[hd_col].astype(str)
    df[lane_col] = df[lane_col].astype(int)
    df[rank_col] = df[rank_col].astype(int)

    # レース単位へ組み直し（1〜3着の艇番と勝者の決まり手）
    def _reduce_one_race(g: pd.DataFrame) -> pd.Series:
        # 1,2,3着 lane
        g = g.sort_values(rank_col)
        lanes = dict(zip(g[rank_col].tolist(), g[lane_col].tolist()))
        l1 = lanes.get(1, None)
        l2 = lanes.get(2, None)
        l3 = lanes.get(3, None)
        # 決まり手（勝者行から）
        fin_vals = g.loc[g[rank_col] == 1, fin_col].dropna().astype(str).tolist()
        fin_norm = normalize_fin_label(fin_vals[0]) if fin_vals else "sonota"
        return pd.Series({"L1": l1, "L2": l2, "L3": l3, "fin": fin_norm})

    races = (
        df.groupby([hd_col, jcd_col, rno_col], as_index=False)
          .apply(_reduce_one_race)
          .reset_index(drop=True)
    )
    # 欠損・不備除去
    races = races.dropna(subset=["L1", "L2", "L3", "fin"]).copy()
    races["L1"] = races["L1"].astype(int)
    races["L2"] = races["L2"].astype(int)
    races["L3"] = races["L3"].astype(int)
    races["fin_id"] = races["fin"].map(FIN_TO_ID).fillna(FIN_TO_ID["sonota"]).astype(int)

    # 減衰重み
    w = time_decay_weights(df.drop_duplicates([hd_col, jcd_col, rno_col])[hd_col], cfg.ref_end, cfg.half_life)
    # races と一致させる（groupbyから復元済みのため、順序を合わせる）
    races["_w"] = w[: len(races)]

    # カウントテンソル
    p2_counts = np.zeros((len(FIN_CLASSES), 6, 6), dtype=float)
    p3_counts = np.zeros((len(FIN_CLASSES), 6, 6, 6), dtype=float)

    for _, row in races.iterrows():
        f = int(row["fin_id"])
        L1 = int(row["L1"]) - 1
        L2 = int(row["L2"]) - 1
        L3 = int(row["L3"]) - 1
        ww = float(row["_w"])
        if 0 <= L1 < 6 and 0 <= L2 < 6 and 0 <= L3 < 6:
            p2_counts[f, L1, L2] += ww
            p3_counts[f, L1, L2, L3] += ww

    # Dirichlet 平滑
    p2_counts += cfg.alpha_p2
    p3_counts += cfg.alpha_p3

    # 物理制約: L2 != L1, L3 != L1, L3 != L2 を保証
    for f in range(len(FIN_CLASSES)):
        for L1 in range(6):
            # p2: L2 != L1
            p2_counts[f, L1, L1] = 0.0
            # p3: L3 != L1, L3 != L2 は正規化時に反映
            for L2 in range(6):
                p3_counts[f, L1, L2, L1] = 0.0
                p3_counts[f, L1, L2, L2] = 0.0

    # 正規化
    p2 = _safe_norm(p2_counts, axis=2)  # (..., L2)
    # p3 は L3 次元で正規化（L3禁止セルは0にしてある）
    denom = p3_counts.sum(axis=3, keepdims=True)
    denom = np.where(denom <= 0.0, 1.0, denom)
    p3 = p3_counts / denom

    return p2, p3


# ========= 保存/読み込み =========
def save_priors(p2: np.ndarray, p3: np.ndarray, cfg: PlacementConfig) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / cfg.p2_file, p2)
    np.save(out_dir / cfg.p3_file, p3)
    meta = {
        "config": asdict(cfg),
        "fin_classes": FIN_CLASSES,
        "shapes": {"p2": list(p2.shape), "p3": list(p3.shape)},
    }
    (out_dir / cfg.meta_file).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def load_priors(cfg: PlacementConfig) -> Tuple[np.ndarray, np.ndarray, Dict]:
    out_dir = Path(cfg.out_dir)
    p2 = np.load(out_dir / cfg.p2_file)
    p3 = np.load(out_dir / cfg.p3_file)
    meta = json.loads((out_dir / cfg.meta_file).read_text(encoding="utf-8"))
    return p2, p3, meta


# ========= 推論（条件付き分布の取り出し） =========
def predict_conditional_tables(
    p2: np.ndarray,
    p3: np.ndarray,
    racecard_df: pd.DataFrame,
    fin_probs_df: pd.DataFrame,
    hd_col="hd", jcd_col="jcd", rno_col="rno",
    lane_col="lane", reg_col="regno",
) -> Dict[str, pd.DataFrame]:
    """
    出走表と fin_model の出力を使って、レース単位で
      - P(2着=ℓ2 | fin, 1着=ℓ1) のテーブル
      - P(3着=ℓ3 | fin, 1着=ℓ1, 2着=ℓ2) のテーブル
    を返す（頻度事前からのlookup）。

    引数:
      - p2: shape (5,6,6)
      - p3: shape (5,6,6,6)
      - racecard_df: 必須列 = hd,jcd,rno,lane,regno
      - fin_probs_df: fin_model.predict の出力（hd,jcd,rno,lane,regno, p_win, p_*_cond 等）
        ※ ここでは p_* を使わず、テーブル出力のみ行う（compose_120で合成時に掛ける）

    戻り値:
      {
        "p2": DataFrame [hd,jcd,rno, fin, L1, L2, p2],
        "p3": DataFrame [hd,jcd,rno, fin, L1, L2, L3, p3]
      }
    """
    use = racecard_df[[hd_col, jcd_col, rno_col]].drop_duplicates().copy()

    rows_p2 = []
    rows_p3 = []
    for _, r in use.iterrows():
        hd = str(r[hd_col])
        jcd = r[jcd_col]
        rno = r[rno_col]
        # 6艇想定: L1/L2/L3 は1..6で走者が埋まる前提（テーブル自体はレーン抽象）
        for f_name, f_id in FIN_TO_ID.items():
            for L1 in range(6):
                # P(l2 | f, L1)
                vec2 = p2[f_id, L1, :]  # (6,)
                for L2 in range(6):
                    rows_p2.append([hd, jcd, rno, f_name, L1+1, L2+1, float(vec2[L2])])
                    # P(l3 | f, L1, L2)
                    vec3 = p3[f_id, L1, L2, :]  # (6,)
                    for L3 in range(6):
                        rows_p3.append([hd, jcd, rno, f_name, L1+1, L2+1, L3+1, float(vec3[L3])])

    df_p2 = pd.DataFrame(rows_p2, columns=[hd_col, jcd_col, rno_col, "fin", "L1", "L2", "p2"])
    df_p3 = pd.DataFrame(rows_p3, columns=[hd_col, jcd_col, rno_col, "fin", "L1", "L2", "L3", "p3"])
    return {"p2": df_p2, "p3": df_p3}


# ========= 入口（学習→保存） =========
def train_and_save_priors(
    history_df: pd.DataFrame,
    cfg: PlacementConfig = PlacementConfig(),
    hd_col="hd", jcd_col="jcd", rno_col="rno",
    lane_col="lane", fin_col="fin", rank_col="rank",
) -> Tuple[np.ndarray, np.ndarray]:
    p2, p3 = fit_priors_from_history(
        history_df, cfg, hd_col, jcd_col, rno_col, lane_col, fin_col, rank_col
    )
    save_priors(p2, p3, cfg)
    return p2, p3
