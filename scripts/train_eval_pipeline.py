# scripts/train_eval_pipeline.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_pipeline.py  —  統合パイプライン（v0: 評価フェーズ中心）

現状:
- YAML を読み込み、compose_120 の出力CSVを評価 (src/boatpred/eval.py) に渡す。
- 合成オッズ/EV/TopN/ROI/命中率を出力。

将来拡張:
- st_model / fin_model / placement_model / compose_120 を順番に起動して pred_csv を生成し、
  そのまま評価へ渡す（ブロックは TODO として雛形を残してあります）。

使い方例:
  python scripts/train_eval_pipeline.py --config data/pipeline.config.yaml
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

# 評価ロジック（既に作成済み）
from src.boatpred.eval import EvalConfig, evaluate_topN


def read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # pip install pyyaml
    except Exception:
        print("[FATAL] PyYAML が未導入です: pip install pyyaml", file=sys.stderr)
        raise
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="パイプライン設定 YAML")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = read_yaml(cfg_path)

    # ---------- パス系 ----------
    paths = cfg.get("paths", {})
    data_dir      = Path(paths.get("data_dir", "data"))
    out_dir       = Path(paths.get("out_dir", str(data_dir / "eval")))
    odds_root     = Path(paths.get("odds_root", "public/odds/v1"))
    results_root  = Path(paths.get("results_dir", "public/results"))
    ensure_dir(out_dir)

    # ---------- 期間 ----------
    train_cfg = cfg.get("train", {})
    test_cfg  = cfg.get("test", {})
    train_start = str(train_cfg.get("start", ""))
    train_end   = str(train_cfg.get("end", ""))
    test_start  = str(test_cfg.get("start", ""))
    test_end    = str(test_cfg.get("end", ""))

    # ---------- 評価設定 ----------
    eval_opts = cfg.get("eval", {})
    topk   = int(eval_opts.get("topk", cfg.get("topk", 18)))
    unit   = int(eval_opts.get("unit_stake", cfg.get("unit_stake", 100)))
    rkey   = eval_opts.get("rank_key", cfg.get("rank_key", "proba"))
    ev_min = float(eval_opts.get("ev_min", cfg.get("ev_min", 0.0)))
    ev_drop_miss = bool(eval_opts.get("ev_drop_if_missing_odds", cfg.get("ev_drop_if_missing_odds", False)))
    load_odds_always = bool(cfg.get("load_odds_always", eval_opts.get("load_odds_always", True)))

    # 形状系フィルタ（Top18）
    filters = cfg.get("filters", {})
    p1win_max = float(filters.get("p1win_max", 1.0))
    s18_min   = float(filters.get("s18_min", 0.0))
    oh_max    = float(filters.get("one_head_ratio_top18_max", 1.0))
    non1_min  = int(filters.get("min_non1_candidates_top18", 0))
    daily_cap = int(filters.get("daily_cap", 0))

    # バスケット（TopN）
    synth_odds_min   = float(cfg.get("synth_odds_min", eval_opts.get("synth_odds_min", 0.0)))
    min_odds_min     = float(cfg.get("min_odds_min",   eval_opts.get("min_odds_min",   0.0)))
    ev_basket_min    = float(cfg.get("ev_basket_min",  eval_opts.get("ev_basket_min",  0.0)))
    odds_coverage_min= int(cfg.get("odds_coverage_min",eval_opts.get("odds_coverage_min",0)))

    # ---------- 入力（v0は compose_120 の出力CSVを受け取る） ----------
    # cfg["inputs"]["pred_csv"] にパスを置く想定。無ければエラー。
    inputs = cfg.get("inputs", {})
    pred_csv = inputs.get("pred_csv")
    if not pred_csv:
        raise RuntimeError("inputs.pred_csv が未指定です。compose_120 の出力CSV (hd,jcd,rno,combo,proba) を指定してください。")
    pred_csv_path = Path(pred_csv)
    if not pred_csv_path.exists():
        raise FileNotFoundError(f"pred_csv not found: {pred_csv_path}")

    # ---------- 将来の学習＆合成（雛形） ----------
    # TODO: ST・決まり手・単勝・2着・3着の学習＆推論を順に行い、pred_csv を生成。
    # 例:
    #   from src.boatpred.st_model import train_st, predict_st
    #   from src.boatpred.fin_model import train_finisher, predict_finisher
    #   from src.boatpred.placement_model import train_place, predict_place
    #   from src.boatpred.compose_120 import compose
    # 現状は pred_csv をそのまま評価へ渡す。

    # ---------- 評価 ----------
    pred_df = pd.read_csv(pred_csv_path)

    ec = EvalConfig(
        topk=topk,
        unit_stake=unit,
        rank_key=rkey,
        ev_min=ev_min,
        ev_drop_if_missing_odds=ev_drop_miss,
        load_odds_always=load_odds_always,
        synth_odds_min=synth_odds_min,
        min_odds_min=min_odds_min,
        ev_basket_min=ev_basket_min,
        odds_coverage_min=odds_coverage_min,
        p1win_max=p1win_max,
        s18_min=s18_min,
        one_head_ratio_top18_max=oh_max,
        min_non1_candidates_top18=non1_min,
        daily_cap=daily_cap,
    )

    picks_df, summary = evaluate_topN(
        pred_df=pred_df,
        odds_root=odds_root,
        results_root=results_root,
        cfg=ec,
    )

    # ---------- 出力 ----------
    # ファイル名に期間ヒントを入れる（任意）
    try:
        hds = sorted(map(str, pred_df["hd"].astype(str).unique()))
        span = f"{hds[0]}_{hds[-1]}_" if hds else ""
    except Exception:
        span = ""

    def _safen(x) -> str:
        try:
            f = float(x)
            return str(f).replace(".", "_")
        except Exception:
            return "na"

    mode = f"{ec.rank_key}_evmin{_safen(ec.ev_min)}_p1{_safen(ec.p1win_max)}_s18{_safen(ec.s18_min)}_oh{_safen(ec.one_head_ratio_top18_max)}"
    if ec.daily_cap > 0: mode += f"_cap{ec.daily_cap}"
    if (ec.rank_key == "ev") or ec.load_odds_always or ec.synth_odds_min>0 or ec.min_odds_min>0 or ec.ev_basket_min>0:
        mode += "_odds"
    if ec.synth_odds_min>0 or ec.min_odds_min>0 or ec.ev_basket_min>0 or ec.odds_coverage_min>0:
        mode += f"_sb{_safen(ec.synth_odds_min)}_mo{_safen(ec.min_odds_min)}_eb{_safen(ec.ev_basket_min)}_cov{int(ec.odds_coverage_min)}"

    picks_path = out_dir / f"picks_{span}k{ec.topk}_{mode}.csv"
    summary_path = out_dir / f"summary_{span}k{ec.topk}_{mode}.json"

    if not picks_df.empty:
        picks_df.to_csv(picks_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")


if __name__ == "__main__":
    main()
