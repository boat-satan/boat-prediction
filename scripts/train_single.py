#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単勝モデル 学習・推論＋（ログ専用の）検証モード 〈全置き換え版〉

- 互換引数を広く受け付ける（--start/--end, --pred_start/--pred_end, --proba_root など）
- is_win を results JSON から自動付与（既存に is_win があれば尊重）
- staging CSV を日付範囲で自動探索
- LightGBM で学習し、テスト範囲があれば確率CSV出力
- --eval_mode=true なら「検証指標を GitHub Actions ログにだけ表示」（ファイルには書かない）

入出力:
  入力: data/staging/YYYY/MMDD/*single*.csv（推奨: single_train.csv）
  ラベル: public/results/YYYY/MMDD/{jcd}/{rno}R.json → 1着 lane
  出力:
    - モデル: data/models/single/model.txt（または --model_out）
    - メタ:   data/models/single/meta.json
    - 予測:   data/proba/single/YYYY/MMDD/proba_{test_start}_{test_end}.csv
"""

from __future__ import annotations
import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import glob
from typing import Dict, Tuple, List, Optional

import polars as pl

try:
    import pandas as pd
except Exception:
    # フォールバック（環境依存回避）
    import pandas as pd

import lightgbm as lgb


# -------------------- utils --------------------
def log(msg: str): print(msg, flush=True)
def err(msg: str): print(msg, file=sys.stderr, flush=True)
def ensure_parent(p: Path) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def to_ord(y: int, m: int, d: int) -> int:
    # 月31日固定の簡易ordinal（範囲フィルタ用。順序性だけ満たせばOK）
    return y * 372 + m * 31 + d

def ymd_to_ord(yyyymmdd: str) -> int:
    y, m, d = int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8])
    return to_ord(y, m, d)

def parse_year_md_from_dir(day_dir: Path) -> Tuple[int, int, int]:
    # day_dir = .../YYYY/MMDD
    y = int(day_dir.parent.name)
    md = day_dir.name
    m, d = int(md[:2]), int(md[2:])
    return y, m, d


# -------------------- config --------------------
@dataclass
class TrainConfig:
    staging_root: str = "data/staging"
    results_root: Optional[str] = "public/results"
    out_root: str = "data/proba/single"
    model_root: str = "data/models/single"
    model_out: Optional[str] = None
    train_start: str = "20240101"
    train_end: str   = "20240131"
    test_start: str  = "20240201"
    test_end: str    = "20240229"
    # モデル設定
    learning_rate: float = 0.05
    num_leaves: int = 63
    max_depth: int = -1
    min_data_in_leaf: int = 50
    feature_fraction: float = 0.9
    bagging_fraction: float = 0.9
    bagging_freq: int = 1
    lambda_l2: float = 1.0
    num_boost_round: int = 400
    num_threads: int = 0
    seed: int = 20240301
    # オプション
    normalize: bool = True
    temperature: float = 1.0
    # 検証モード（ログ専用出力）
    eval_mode: bool = False
    eval_topk: str = "1,3,6,12"  # ログ表示するTopK


# -------------------- file discovery --------------------
def _list_day_dirs(root: Path) -> List[Path]:
    # root/YYYY/MMDD
    out = []
    for ydir in sorted(root.glob("[0-9]" * 4)):
        if not ydir.is_dir(): continue
        for md in sorted(ydir.glob("[0-9]" * 4)):
            if md.is_dir():
                out.append(md)
    return out

def pick_day_dirs_by_range(root: Path, start: str, end: str) -> List[Path]:
    if not root.exists(): return []
    s_ord, e_ord = ymd_to_ord(start), ymd_to_ord(end)
    picked = []
    for dd in _list_day_dirs(root):
        y, m, d = parse_year_md_from_dir(dd)
        if s_ord <= to_ord(y, m, d) <= e_ord:
            picked.append(dd)
    return picked

def find_staging_csvs(staging_root: str, start: str, end: str) -> List[Path]:
    root = Path(staging_root)
    day_dirs = pick_day_dirs_by_range(root, start, end)
    files: List[Path] = []
    for dd in day_dirs:
        # 優先順: single_train.csv -> *single*.csv -> *_train.csv
        cand = [
            dd / "single_train.csv",
            *map(Path, glob.glob(str(dd / "*single*.csv"))),
            *map(Path, glob.glob(str(dd / "*_train.csv"))),
        ]
        use = None
        for c in cand:
            if c.exists() and c.suffix.lower() == ".csv":
                use = c; break
        if use: files.append(use)
    return files


# -------------------- label (is_win) from results --------------------
def _winner_lane_from_results(result_json_path: Path) -> Optional[int]:
    try:
        with result_json_path.open("r", encoding="utf-8") as f:
            js = json.load(f)
        for rec in js.get("results", []):
            if str(rec.get("rank")) == "1":
                lane = rec.get("lane")
                try:
                    return int(lane)
                except Exception:
                    return None
    except Exception:
        return None
    return None

def build_winner_index(results_root: str, candidate_days: List[Path]) -> Dict[Tuple[str,str,int], int]:
    """
    key: (hd, jcd, rno) -> winner_lane
    results_root: public/results
    candidate_days: data/staging/YYYY/MMDD ディレクトリ（日付抽出）
    """
    index: Dict[Tuple[str,str,int], int] = {}
    rroot = Path(results_root) if results_root else None
    if not rroot or not rroot.exists():
        return index

    for dd in candidate_days:
        year = dd.parent.name
        md = dd.name
        res_day_dir = rroot / year / md
        if not res_day_dir.exists():
            continue
        # 形: public/results/YYYY/MMDD/{jcd}/{rno}R.json
        for jcd_dir in res_day_dir.glob("*"):
            if not jcd_dir.is_dir(): continue
            jcd = jcd_dir.name
            for rfile in jcd_dir.glob("*R.json"):
                name = rfile.stem  # "10R"
                try:
                    rno = int(name[:-1])
                except Exception:
                    continue
                lane = _winner_lane_from_results(rfile)
                if lane is not None:
                    index[(year + md, jcd, rno)] = lane  # hd="YYYYMMDD"
    return index


# -------------------- dataset load & feature select --------------------
BASE_ID_COLS = ["hd","jcd","rno","lane","regno"]
# 最初はコアだけ（増やすのはいつでもできる）
PREF_FEATURES = [
    "course_first_rate", "course_3rd_rate", "course_avg_st",
    "tenji_st_sec", "tenji_rank", "st_rank",
    "wind_speed_m", "wave_height_cm",
]

def load_dataset(paths: List[Path]) -> pl.DataFrame:
    if not paths: return pl.DataFrame()
    lfs = [pl.scan_csv(str(p), ignore_errors=True) for p in paths]
    return pl.concat(lfs).collect(streaming=True)

def add_is_win_from_results(df: pl.DataFrame, results_root: Optional[str], day_dirs: List[Path]) -> pl.DataFrame:
    """
    df に is_win 列がなければ results から補完して追加。
    既に is_win があれば尊重（上書きしない）。
    """
    if "is_win" in df.columns:
        return df
    if not results_root:
        log("[WARN] results_root 未指定: is_win 付与スキップ")
        return df

    index = build_winner_index(results_root, day_dirs)
    if not index:
        log("[WARN] winner index 空: is_win 付与スキップ")
        return df

    def winner_lane_expr() -> pl.Expr:
        return (pl.struct(["hd","jcd","rno","lane"])
                  .map_elements(lambda s: 1 if index.get((str(s["hd"]), str(s["jcd"]), int(s["rno"]))) == int(s["lane"]) else 0)
                  .alias("is_win"))
    try:
        return df.with_columns([winner_lane_expr()])
    except Exception as e:
        err(f"[WARN] is_win 付与失敗: {e}")
        return df

def select_features(df: pl.DataFrame):
    cols = set(df.columns)
    if "is_win" not in cols:
        raise RuntimeError("入力に 'is_win' 列がありません。（results からの自動付与も失敗）")

    use_feats: List[str] = [c for c in PREF_FEATURES if c in cols]
    # 数値列を追加（ID/ラベル以外）
    for c, dt in df.schema.items():
        if c in BASE_ID_COLS or c == "is_win":
            continue
        dt_str = str(dt)
        if ("Int" in dt_str) or ("Float" in dt_str):
            if c not in use_feats:
                use_feats.append(c)
    if len(use_feats) > 64:
        use_feats = use_feats[:64]

    dfx = df.select(BASE_ID_COLS + ["is_win"] + use_feats).fill_null(0)
    pdf = dfx.to_pandas()
    return pdf, use_feats


# -------------------- model --------------------
def fit_lgb(train_pd, feat_cols, cfg: TrainConfig) -> lgb.Booster:
    X = train_pd[feat_cols]
    y = train_pd["is_win"].astype(int).values
    params = dict(
        objective="binary", metric="auc",
        learning_rate=cfg.learning_rate, num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth, min_data_in_leaf=cfg.min_data_in_leaf,
        feature_fraction=cfg.feature_fraction, bagging_fraction=cfg.bagging_fraction,
        bagging_freq=cfg.bagging_freq, lambda_l2=cfg.lambda_l2,
        num_threads=cfg.num_threads, seed=cfg.seed, verbose=-1,
        force_col_wise=True,
    )
    dset = lgb.Dataset(X, label=y, feature_name=feat_cols, free_raw_data=True)
    booster = lgb.train(params, dset, num_boost_round=cfg.num_boost_round,
                        valid_sets=[dset], valid_names=["train"],
                        callbacks=[lgb.log_evaluation(period=200)])
    return booster

def predict_df(booster: lgb.Booster, test_pd, feat_cols, temperature: float = 1.0):
    X = test_pd[feat_cols]
    p = booster.predict(X, num_iteration=booster.best_iteration)
    if temperature and temperature > 0 and temperature != 1.0:
        import numpy as np
        eps = 1e-12
        p_ = np.clip(p, eps, 1 - eps)
        logit = np.log(p_ / (1 - p_))
        logit /= float(temperature)
        p = 1 / (1 + np.exp(-logit))
    out = test_pd[BASE_ID_COLS].copy()
    out["proba_win"] = p
    return out


# -------------------- eval (log-only) --------------------
def safe_auc(y_true, y_score) -> Optional[float]:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None

def safe_logloss(y_true, y_prob) -> Optional[float]:
    try:
        from sklearn.metrics import log_loss
        return float(log_loss(y_true, y_prob, labels=[0,1]))
    except Exception:
        return None

def compute_topk_hits(pred_df: pd.DataFrame, test_df: pd.DataFrame, topk_list: List[int]) -> List[Tuple[int,float]]:
    """
    レース単位（hd,jcd,rno）で proba_win 上位kの中に is_win=1 が含まれる率（TopKヒット率）
    """
    # 真値（各レースの勝ちlane）
    truth_by_race = (test_df.loc[test_df["is_win"]==1, ["hd","jcd","rno","lane"]]
                           .rename(columns={"lane":"truth_lane"}))
    df = pred_df.merge(truth_by_race, on=["hd","jcd","rno"], how="inner")
    if df.empty:
        return [(k, float("nan")) for k in topk_list]

    # レースごとに proba 降順インデックス
    hits = []
    for k in topk_list:
        # 各レース上位kの lane セット
        topk = (df.sort_values(["hd","jcd","rno","proba_win"], ascending=[True,True,True,False])
                  .groupby(["hd","jcd","rno"])
                  .head(k))
        # truth 含有チェック
        m = topk.assign(in_topk=(topk["lane"]==topk["truth_lane"]).astype(int)) \
                .groupby(["hd","jcd","rno"])["in_topk"].max()
        hit_rate = float(m.mean()) if len(m)>0 else float("nan")
        hits.append((k, hit_rate))
    return hits


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()

    # 現行引数
    ap.add_argument("--staging_root", default="data/staging")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--out_root",     default="data/proba/single")
    ap.add_argument("--model_root",   default="data/models/single")
    ap.add_argument("--train_start",  default="20240101")
    ap.add_argument("--train_end",    default="20240131")
    ap.add_argument("--test_start",   default="20240201")
    ap.add_argument("--test_end",     default="20240229")
    ap.add_argument("--normalize",    type=lambda x: str(x).lower()=="true", default=True)
    ap.add_argument("--temperature",  type=float, default=1.0)

    # 互換（旧ワークフロー）
    ap.add_argument("--pred_start",     default=None)  # -> test_start
    ap.add_argument("--pred_end",       default=None)  # -> test_end
    ap.add_argument("--model_out",      default=None)  # モデル保存ファイル明示
    ap.add_argument("--proba_out_root", default=None)  # -> out_root

    # さらに古い書式
    ap.add_argument("--start",        default=None)   # -> train_start
    ap.add_argument("--end",          default=None)   # -> train_end
    ap.add_argument("--proba_root",   default=None)   # -> out_root
    ap.add_argument("--inplace",      default=None)   # 無視

    # 検証モード（ログだけ）
    ap.add_argument("--eval_mode",     type=lambda x: str(x).lower()=="true", default=False,
                    help="検証指標をGitHub Actionsログにのみ表示（ファイル出力なし）")
    ap.add_argument("--eval_topk",     default="1,3,6,12",
                    help="ログ出力するTopK（カンマ区切り）例: '1,3,6,12,18,30'")

    args = ap.parse_args()

    # 互換マッピング
    test_start = args.pred_start if args.pred_start else args.test_start
    test_end   = args.pred_end   if args.pred_end   else args.test_end

    out_root = args.out_root
    if args.proba_out_root: out_root = args.proba_out_root
    elif args.proba_root:   out_root = args.proba_root

    train_start = args.train_start if args.train_start else (args.start or "20240101")
    train_end   = args.train_end   if args.train_end   else (args.end   or "20240131")

    cfg = TrainConfig(
        staging_root=args.staging_root,
        results_root=args.results_root,
        out_root=out_root,
        model_root=args.model_root,
        model_out=args.model_out,
        train_start=train_start, train_end=train_end,
        test_start=test_start,   test_end=test_end,
        normalize=args.normalize, temperature=args.temperature,
        eval_mode=args.eval_mode, eval_topk=args.eval_topk
    )

    # 入力収集
    train_files = find_staging_csvs(cfg.staging_root, cfg.train_start, cfg.train_end)
    test_files  = find_staging_csvs(cfg.staging_root, cfg.test_start,  cfg.test_end)

    if not train_files:
        err(f"[FATAL] train CSV が見つかりません: {cfg.staging_root} [{cfg.train_start}..{cfg.train_end}]")
        sys.exit(1)

    # day_dirs（is_win付与に使う）
    train_day_dirs = sorted({p.parent for p in train_files})
    test_day_dirs  = sorted({p.parent for p in test_files}) if test_files else []

    log(f"[INFO] train files: {len(train_files)}")
    if test_files:
        log(f"[INFO] test  files: {len(test_files)}")

    # ------- 学習 -------
    df_train = load_dataset(train_files)
    if df_train.height == 0:
        err("[FATAL] 学習データが空です。")
        sys.exit(1)

    # is_win 自動付与（存在すれば上書きしない）
    df_train = add_is_win_from_results(df_train, cfg.results_root, train_day_dirs)
    train_pd, feat_cols = select_features(df_train)
    log(f"[INFO] features used: {len(feat_cols)} -> {feat_cols[:10]}{'...' if len(feat_cols)>10 else ''}")
    log(f"[INFO] train rows: {len(train_pd)}")

    booster = fit_lgb(train_pd, feat_cols, cfg)

    # モデル保存
    if cfg.model_out:
        model_path = ensure_parent(Path(cfg.model_out))
        meta_path  = ensure_parent(Path(cfg.model_out).with_suffix(".json"))
    else:
        model_path = ensure_parent(Path(cfg.model_root) / "model.txt")
        meta_path  = ensure_parent(Path(cfg.model_root) / "meta.json")

    booster.save_model(str(model_path))
    meta = {"config": asdict(cfg), "feature_names": feat_cols, "train_rows": int(len(train_pd))}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log(f"[WRITE] {model_path}")
    log(f"[WRITE] {meta_path}")

    # ------- 予測（テスト範囲があれば） -------
    if not test_files:
        log("[INFO] テストデータなし。予測・評価はスキップ。")
        return

    df_test_raw = load_dataset(test_files)
    if df_test_raw.height == 0:
        log("[WARN] テストデータが空です。予測・評価はスキップ。")
        return

    # is_win 付与（評価用）。存在していれば尊重。
    df_test = add_is_win_from_results(df_test_raw, cfg.results_root, test_day_dirs)

    # 予測用に列整形
    keep_cols = list({*BASE_ID_COLS, *feat_cols, *(['is_win'] if 'is_win' in df_test.columns else [])})
    keep_cols = [c for c in keep_cols if c in df_test.columns]
    df_test2 = df_test.select(keep_cols).fill_null(0)
    test_pd = df_test2.to_pandas()

    # 推論
    pred = predict_df(booster, test_pd, feat_cols, temperature=cfg.temperature)

    # 予測CSV出力（従来どおり）
    y = cfg.test_start[:4]; md = cfg.test_start[4:8]
    out_dir = Path(cfg.out_root) / y / md
    out_csv = ensure_parent(out_dir / f"proba_{cfg.test_start}_{cfg.test_end}.csv")
    pred.to_csv(out_csv, index=False)
    log(f"[WRITE] {out_csv} rows={len(pred)}")

    # ------- 検証モード（ログ出力のみ） -------
    if cfg.eval_mode:
        if "is_win" not in test_pd.columns:
            log("[WARN] eval_mode: is_win が無いので評価をスキップ（resultsが未取得の可能性）")
            return

        # 対象レース（返還/不成立で is_win が全0 になる等の異常は除外）
        # 正常レース判定：各レースで is_win==1 がちょうど1件
        g = test_pd.groupby(["hd","jcd","rno"])["is_win"].sum()
        valid_keys = g.index[g.values == 1]
        if len(valid_keys) == 0:
            log("[WARN] eval_mode: 正常レースが見つからないため評価スキップ")
            return

        # valid レースのみで評価
        valid_df = test_pd.merge(
            pd.DataFrame(valid_keys.tolist(), columns=["hd","jcd","rno"]),
            on=["hd","jcd","rno"], how="inner"
        )
        valid_pred = pred.merge(
            pd.DataFrame(valid_keys.tolist(), columns=["hd","jcd","rno"]),
            on=["hd","jcd","rno"], how="inner"
        )

        y_true = valid_df["is_win"].astype(int).values
        y_prob = valid_pred["proba_win"].values

        auc = safe_auc(y_true, y_prob)
        logloss = safe_logloss(y_true, y_prob)

        # TopK
        topk_list = []
        for t in str(cfg.eval_topk).split(","):
            t = t.strip()
            if t.isdigit():
                topk_list.append(int(t))
        if not topk_list:
            topk_list = [1,3,6,12]

        topk_hits = compute_topk_hits(valid_pred.merge(valid_df[["hd","jcd","rno","is_win","lane"]],
                                                       on=["hd","jcd","rno","lane"], how="left"),
                                      valid_df, topk_list)

        # ログ出力（ファイル保存はしない）
        log("\n===== Eval (単勝モデル / log-only) =====")
        log(f"test_range        : {cfg.test_start} .. {cfg.test_end}")
        log(f"valid_races       : {len(valid_keys)} / total_rows={len(valid_df)}")
        if auc is not None:     log(f"AUC               : {auc:.6f}")
        if logloss is not None: log(f"LogLoss           : {logloss:.6f}")

        # Top1は別表示（= rank1 的中率）
        top1 = next((h for k,h in topk_hits if k==1), None)
        if top1 is not None:
            log(f"Top1正解率        : {top1*100:.2f}%")

        # 追加TopK
        for k, hit in topk_hits:
            if k == 1: continue
            log(f"Top{k}ヒット率     : {hit*100:.2f}%")
        log("=========================================\n")


if __name__ == "__main__":
    main()
