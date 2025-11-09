# scripts/train_eval_pipeline.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_pipeline.py — 統合パイプライン v1
- STブロック: data/shards/**/st_rows.(csv|parquet) を読んで学習・推論し、data/st_out に保存
- 評価ブロック: cfg.inputs.pred_csv があれば従来の TopN/ROI 評価を実行（任意）

使い方:
  python scripts/train_eval_pipeline.py --config data/pipeline.config.yaml
"""

from __future__ import annotations
import argparse, json, sys, glob
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# === 自作モジュール ===
from src.boatpred.st_model import STConfig, train as st_train, predict as st_predict
from src.boatpred.eval import EvalConfig, evaluate_topN  # 評価は任意

# ---------- 小物 ----------
def read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception:
        print("[FATAL] PyYAML 未導入: pip install pyyaml", file=sys.stderr); raise
    if not path.exists(): raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def scan_st_rows(shards_root: Path) -> List[Path]:
    pats = [
        str(shards_root / "**" / "st_rows.parquet"),
        str(shards_root / "**" / "st_rows.csv"),
        str(shards_root / "**" / "st_rows.csv.gz"),
    ]
    out: List[Path] = []
    for pat in pats:
        out.extend(Path(p) for p in glob.glob(pat, recursive=True))
    return sorted(out)

def read_concat(paths: List[Path]) -> pd.DataFrame:
    if not paths: return pd.DataFrame()
    # 優先: parquet → csv.gz → csv
    prq = [p for p in paths if p.suffix == ".parquet"]
    if prq:
        return pd.concat([pd.read_parquet(p) for p in prq], ignore_index=True)
    gz = [p for p in paths if p.suffixes[-2:] == [".csv", ".gz"]]
    if gz:
        return pd.concat([pd.read_csv(p) for p in gz], ignore_index=True)
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)

def subset_by_period(df: pd.DataFrame, start: str, end: str, hd_col="hd") -> pd.DataFrame:
    s = str(start); e = str(end)
    d = df.copy(); d[hd_col] = d[hd_col].astype(str)
    return d[(d[hd_col] >= s) & (d[hd_col] <= e)].copy()

def build_racecard_from_rows(df: pd.DataFrame) -> pd.DataFrame:
    """st_rows から当日の出走表ビューを作る（hd,jcd,rno,lane,regnoの重複なし集合）"""
    use_cols = ["hd","jcd","rno","lane","regno"]
    miss = [c for c in use_cols if c not in df.columns]
    if miss: raise RuntimeError(f"[st] st_rows に必要列がありません: {miss}")
    rc = df[use_cols].drop_duplicates().copy()
    rc["lane"] = rc["lane"].astype(int)
    rc["regno"] = rc["regno"].astype(str)
    return rc

# ---------- ST ブロック ----------
def run_st(cfg: Dict[str, Any]) -> Path | None:
    st_cfg = cfg.get("st", {}) or {}
    if not st_cfg.get("enabled", False):
        print("[st] skipped (st.enabled=false)")
        return None

    shards_root = Path(st_cfg.get("shards_root", "data/shards"))
    out_dir = Path(st_cfg.get("out_dir", "data/st_out"))
    ensure_dir(out_dir)

    train_start = str(cfg.get("train", {}).get("start", ""))
    train_end   = str(cfg.get("train", {}).get("end", ""))
    test_start  = str(cfg.get("test",  {}).get("start", ""))
    test_end    = str(cfg.get("test",  {}).get("end", ""))

    paths = scan_st_rows(shards_root)
    if not paths:
        print(f"[st] no st_rows under {shards_root}/**", file=sys.stderr)
        return None

    all_df = read_concat(paths)
    # 期待列: hd,jcd,rno,lane,regno,st（学習は st を使う）
    need_cols = {"hd","jcd","rno","lane","regno","st"}
    if not need_cols.issubset(all_df.columns):
        raise RuntimeError(f"[st] st_rows 必須列が不足: {sorted(need_cols - set(all_df.columns))}")

    tr_df = subset_by_period(all_df, train_start, train_end)
    te_df = subset_by_period(all_df, test_start,  test_end)
    if tr_df.empty: raise RuntimeError("[st] 学習期間データが空")
    if te_df.empty: print("[st] WARN: テスト期間データが空。racecardは作れません。", file=sys.stderr); return None

    # racecard（予測対象）は st_rows から派生（hd,jcd,rno,lane,regno）
    racecard = build_racecard_from_rows(te_df)

    # ST 学習
    print(f"[st] train rows: {len(tr_df):,}   racecard rows: {len(racecard):,}")
    stconf = STConfig(
        model_out=st_cfg.get("model_out", "data/models/st_lgbm.txt"),
        meta_out =st_cfg.get("meta_out",  "data/models/st_lgbm.meta.json"),
    )
    booster, _ = st_train(tr_df, ref_end=train_end, cfg=stconf)

    # 予測
    pred = st_predict(
        booster=booster,
        racecard_df=racecard,
        history_df=tr_df,       # 直近までの履歴で特徴構築
        ref_end=train_end,
        cfg=stconf,
    )
    # 出力
    out_path = out_dir / f"st_pred_{test_start}_{test_end}.csv"
    pred.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[st] WRITE {out_path}")
    return out_path

# ---------- 評価（従来のTopN/ROI） ----------
def maybe_run_eval(cfg: Dict[str, Any]) -> None:
    inputs = cfg.get("inputs", {}) or {}
    pred_csv = inputs.get("pred_csv")
    if not pred_csv:
        print("[eval] skipped (inputs.pred_csv 未指定)")
        return

    paths = cfg.get("paths", {}) or {}
    data_dir   = Path(paths.get("data_dir", "data"))
    out_dir    = Path(paths.get("out_dir", str(data_dir / "eval")))
    odds_root  = Path(paths.get("odds_root", "public/odds/v1"))
    results_rt = Path(paths.get("results_dir", "public/results"))
    ensure_dir(out_dir)

    pred_df = pd.read_csv(Path(pred_csv))

    eval_opts = cfg.get("eval", {}) or {}
    ec = EvalConfig(
        topk=int(eval_opts.get("topk", cfg.get("topk", 18))),
        unit_stake=int(eval_opts.get("unit_stake", cfg.get("unit_stake", 100))),
        rank_key=eval_opts.get("rank_key", cfg.get("rank_key", "proba")),
        ev_min=float(eval_opts.get("ev_min", cfg.get("ev_min", 0.0))),
        ev_drop_if_missing_odds=bool(eval_opts.get("ev_drop_if_missing_odds", cfg.get("ev_drop_if_missing_odds", False))),
        load_odds_always=bool(cfg.get("load_odds_always", eval_opts.get("load_odds_always", False))),
        synth_odds_min=float(cfg.get("synth_odds_min", eval_opts.get("synth_odds_min", 0.0))),
        min_odds_min=float(cfg.get("min_odds_min", eval_opts.get("min_odds_min", 0.0))),
        ev_basket_min=float(cfg.get("ev_basket_min", eval_opts.get("ev_basket_min", 0.0))),
        odds_coverage_min=int(cfg.get("odds_coverage_min", eval_opts.get("odds_coverage_min", 0))),
        p1win_max=float(cfg.get("filters", {}).get("p1win_max", 1.0)),
        s18_min=float(cfg.get("filters", {}).get("s18_min", 0.0)),
        one_head_ratio_top18_max=float(cfg.get("filters", {}).get("one_head_ratio_top18_max", 1.0)),
        min_non1_candidates_top18=int(cfg.get("filters", {}).get("min_non1_candidates_top18", 0)),
        daily_cap=int(cfg.get("filters", {}).get("daily_cap", 0)),
    )

    picks_df, summary = evaluate_topN(pred_df, odds_root, results_rt, ec)

    # span
    try:
        hds = sorted(map(str, pred_df["hd"].astype(str).unique()))
        span = f"{hds[0]}_{hds[-1]}_" if hds else ""
    except Exception:
        span = ""

    def safen(x):
        try: return str(float(x)).replace(".","_")
        except: return "na"

    mode = f"{ec.rank_key}_evmin{safen(ec.ev_min)}_p1{safen(ec.p1win_max)}_s18{safen(ec.s18_min)}_oh{safen(ec.one_head_ratio_top18_max)}"
    if ec.daily_cap>0: mode += f"_cap{ec.daily_cap}"
    if (ec.rank_key=="ev") or ec.load_odds_always or ec.synth_odds_min>0 or ec.min_odds_min>0 or ec.ev_basket_min>0:
        mode += "_odds"
    if ec.synth_odds_min>0 or ec.min_odds_min>0 or ec.ev_basket_min>0 or ec.odds_coverage_min>0:
        mode += f"_sb{safen(ec.synth_odds_min)}_mo{safen(ec.min_odds_min)}_eb{safen(ec.ev_basket_min)}_cov{int(ec.odds_coverage_min)}"

    picks_path = out_dir / f"picks_{span}k{ec.topk}_{mode}.csv"
    summary_path = out_dir / f"summary_{span}k{ec.topk}_{mode}.json"

    if not picks_df.empty: picks_df.to_csv(picks_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = read_yaml(Path(args.config))

    # ST を実行
    st_out = run_st(cfg)
    if st_out is None:
        print("[st] note: STはスキップ or 失敗（ログ参照）")

    # 評価は任意実行（inputs.pred_csv がある場合のみ）
    maybe_run_eval(cfg)

if __name__ == "__main__":
    main()
