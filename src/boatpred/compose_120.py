# src/boatpred/compose_120.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd

FIN_CLASSES = ["nige", "sashi", "makuri", "makuri_sashi", "sonota"]
FIN_TO_ID = {c: i for i, c in enumerate(FIN_CLASSES)}

@dataclass
class ComposeConfig:
    # 温度/指数補正（必要に応じてリスク調整）
    win_temp: float = 1.0          # 単勝 p_win への温度（<1で尖らせる、>1で平らに）
    fin_temp: float = 1.0          # p(fin|lane) への温度
    p2_temp: float = 1.0           # P(2着|fin,L1) への温度
    p3_temp: float = 1.0           # P(3着|fin,L1,L2) への温度

    # “不正な” 配列時の数値安定化
    eps: float = 1e-12

    # 出走艇の最大スロット（基本6艇）
    max_lanes: int = 6

def _pow_norm(p: np.ndarray, t: float, eps: float) -> np.ndarray:
    """温度 t でべき乗→正規化（t=1は恒等）。"""
    if t == 1.0:
        # 0負の値は来ない前提だが念のためクリップ
        z = np.clip(p, 0.0, None)
        s = z.sum()
        return z / s if s > 0 else np.full_like(p, 1.0 / len(p))
    z = np.power(np.clip(p, 0.0, None) + eps, 1.0 / max(t, 1e-6))
    s = z.sum()
    return z / s if s > 0 else np.full_like(p, 1.0 / len(p))

def _ensure_lanes_mask(race_lanes: List[int], max_lanes: int) -> np.ndarray:
    """使用するレーン(1..6)のマスクを返す（欠場対応）。"""
    mask = np.zeros(max_lanes, dtype=bool)
    for L in race_lanes:
        if 1 <= L <= max_lanes:
            mask[L-1] = True
    return mask

def compose_one_race(
    race_key: Tuple[str, str, str],              # (hd, jcd, rno)
    lanes_present: List[int],                    # そのレースに出走するレーン(1..6)
    fin_rows: pd.DataFrame,                      # fin_model 出力の該当レース行（laneごと1行）
    p2: np.ndarray,                              # shape (5,6,6)
    p3: np.ndarray,                              # shape (5,6,6,6)
    cfg: ComposeConfig = ComposeConfig(),
) -> pd.DataFrame:
    """
    1レース分の p(F, L1, L2, L3) を合成し、三連単120通り（実際は出走レーンの順列）に落とし込む。
    必須: fin_rows は列 ['hd','jcd','rno','lane','p_win','p_nige','p_sashi','p_makuri','p_makuri_sashi','p_sonota']
    """

    hd, jcd, rno = race_key

    # ---- 参加レーンのマスク（欠場対応）----
    lane_mask = _ensure_lanes_mask(lanes_present, cfg.max_lanes)

    # ---- 取り出し：単勝 p_win と p(fin|lane) ----
    # 並びは lane=1..6 に揃える
    row_map = {int(r["lane"]): r for _, r in fin_rows.iterrows()}

    p_win = np.zeros(cfg.max_lanes, dtype=float)
    p_fin_cond = np.zeros((cfg.max_lanes, len(FIN_CLASSES)), dtype=float)

    for L in range(1, cfg.max_lanes+1):
        if not lane_mask[L-1]:
            continue
        r = row_map.get(L, None)
        if r is None:
            # 情報がないレーンは均等（最小限の崩れ防止）
            p_win[L-1] = 0.0
            p_fin_cond[L-1, :] = 0.0
            continue
        p_win[L-1] = float(r.get("p_win", 0.0))
        p_fin_cond[L-1, 0] = float(r.get("p_nige", 0.0))
        p_fin_cond[L-1, 1] = float(r.get("p_sashi", 0.0))
        p_fin_cond[L-1, 2] = float(r.get("p_makuri", 0.0))
        p_fin_cond[L-1, 3] = float(r.get("p_makuri_sashi", 0.0))
        p_fin_cond[L-1, 4] = float(r.get("p_sonota", 0.0))

    # マスク外はゼロ、温度補正＆正規化
    p_win = np.where(lane_mask, p_win, 0.0)
    p_win = _pow_norm(p_win, cfg.win_temp, cfg.eps)

    for L in range(cfg.max_lanes):
        if lane_mask[L]:
            p_fin_cond[L, :] = _pow_norm(p_fin_cond[L, :], cfg.fin_temp, cfg.eps)
        else:
            p_fin_cond[L, :] = 0.0

    # ---- 事前（placement priors）に温度補正 ----
    # p2[fin, L1, L2], p3[fin, L1, L2, L3] を lane_mask でゼロ化→正規化し直す
    p2_tmp = p2.copy()
    p3_tmp = p3.copy()

    # L1/L2/L3 が欠場ならその質量を0にして再正規化
    for f in range(len(FIN_CLASSES)):
        for L1 in range(cfg.max_lanes):
            # L1 が不在なら全ゼロ（後で p(F,L1,...) もゼロになる）
            if not lane_mask[L1]:
                p2_tmp[f, L1, :] = 0.0
                p3_tmp[f, L1, :, :] = 0.0
                continue
            # p2: L2 の欠場をゼロ → 正規化
            vec2 = p2_tmp[f, L1, :].copy()
            vec2 = np.where(lane_mask, vec2, 0.0)
            vec2 = _pow_norm(vec2, cfg.p2_temp, cfg.eps)
            p2_tmp[f, L1, :] = vec2

            # p3: L2→L3 の順にマスク＆正規化
            for L2 in range(cfg.max_lanes):
                if not lane_mask[L2] or L2 == L1:
                    p3_tmp[f, L1, L2, :] = 0.0
                    continue
                vec3 = p3_tmp[f, L1, L2, :].copy()
                ok3 = lane_mask.copy()
                ok3[L1] = False
                ok3[L2] = False
                vec3 = np.where(ok3, vec3, 0.0)
                vec3 = _pow_norm(vec3, cfg.p3_temp, cfg.eps)
                p3_tmp[f, L1, L2, :] = vec3

    # ---- 合成：p(F, L1, L2, L3) = p_win[L1] * p_fin|L1[F] * p2[F,L1,L2] * p3[F,L1,L2,L3] ----
    rows = []
    for L1 in range(cfg.max_lanes):
        if not lane_mask[L1] or p_win[L1] <= 0:
            continue
        for f_name, f_id in FIN_TO_ID.items():
            pF = p_fin_cond[L1, f_id]
            if pF <= 0:
                continue
            base = p_win[L1] * pF
            vec2 = p2_tmp[f_id, L1, :]
            if vec2.sum() <= 0:
                continue
            for L2 in range(cfg.max_lanes):
                if L2 == L1 or not lane_mask[L2]:
                    continue
                p2c = vec2[L2]
                if p2c <= 0:
                    continue
                vec3 = p3_tmp[f_id, L1, L2, :]
                if vec3.sum() <= 0:
                    continue
                for L3 in range(cfg.max_lanes):
                    if (L3 == L1) or (L3 == L2) or (not lane_mask[L3]):
                        continue
                    p3c = vec3[L3]
                    p = base * p2c * p3c
                    if p <= 0:
                        continue
                    combo = f"{L1+1}-{L2+1}-{L3+1}"
                    rows.append([hd, jcd, rno, combo, f_name, L1+1, L2+1, L3+1, p])

    df = pd.DataFrame(rows, columns=[
        "hd", "jcd", "rno", "combo", "fin", "L1", "L2", "L3", "proba_raw"
    ])

    # ---- 三連単集合の正規化（合計=1.0）----
    if not df.empty:
        gsum = df.groupby(["hd", "jcd", "rno"])["proba_raw"].transform("sum")
        df["proba"] = np.where(gsum > 0, df["proba_raw"] / gsum, 0.0)
    else:
        # 全滅時はダミー（均等）を返す：実運用ではこのケースは上流で除外してOK
        combos = []
        Ls = [i+1 for i, m in enumerate(lane_mask) if m]
        for L1 in Ls:
            for L2 in Ls:
                if L2 == L1: continue
                for L3 in Ls:
                    if L3 == L1 or L3 == L2: continue
                    combos.append(f"{L1}-{L2}-{L3}")
        if not combos:
            return pd.DataFrame(columns=["hd","jcd","rno","combo","fin","L1","L2","L3","proba_raw","proba"])
        p = 1.0 / len(combos)
        df = pd.DataFrame([[hd,jcd,rno,c,"sonota",*map(int,c.split("-")),p,p] for c in combos],
                          columns=["hd","jcd","rno","combo","fin","L1","L2","L3","proba_raw","proba"])

    return df[["hd","jcd","rno","combo","proba","fin","L1","L2","L3","proba_raw"]]


def compose_all(
    racecard_df: pd.DataFrame,     # 必須列: hd,jcd,rno,lane
    fin_df: pd.DataFrame,          # 必須列: hd,jcd,rno,lane,p_win,p_nige,p_sashi,p_makuri,p_makuri_sashi,p_sonota
    p2: np.ndarray,                # placement priors P(2着|fin,L1)
    p3: np.ndarray,                # placement priors P(3着|fin,L1,L2)
    cfg: ComposeConfig = ComposeConfig(),
) -> pd.DataFrame:
    """全レースを一括合成して (hd,jcd,rno,combo,proba,...) を返す。"""
    # racecard のレーン集合を作る
    lane_lists = (
        racecard_df.groupby(["hd","jcd","rno"])["lane"]
        .apply(lambda s: sorted(list(map(int, set(s))))).reset_index(name="lanes")
    )
    fin_df = fin_df.copy()
    key_cols = ["hd","jcd","rno","lane"]
    # 型・文字列整備
    for c in ["hd","jcd","rno"]:
        racecard_df[c] = racecard_df[c].astype(str)
        lane_lists[c]  = lane_lists[c].astype(str)
        fin_df[c]      = fin_df[c].astype(str)
    fin_df["lane"] = fin_df["lane"].astype(int)

    out_list = []
    for _, row in lane_lists.iterrows():
        key = (row["hd"], row["jcd"], row["rno"])
        lanes_present = list(row["lanes"])
        sub_fin = fin_df[(fin_df["hd"]==key[0])&(fin_df["jcd"]==key[1])&(fin_df["rno"]==key[2])]
        if sub_fin.empty:
            # fin情報が無ければスキップ（上流で補完推奨）
            continue
        df_one = compose_one_race(key, lanes_present, sub_fin, p2, p3, cfg)
        out_list.append(df_one)

    if not out_list:
        return pd.DataFrame(columns=["hd","jcd","rno","combo","proba","fin","L1","L2","L3","proba_raw"])
    out = pd.concat(out_list, ignore_index=True)
    return out
