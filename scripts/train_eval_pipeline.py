# scripts/train_eval_pipeline.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_pipeline.py — 統合パイプライン（v0: 評価フェーズ中心）

現状:
- YAML を読み込み、compose_120 等で作られた pred_csv( hd,jcd,rno,combo,proba ) を
  評価モジュール (src/boatpred/eval.py) に渡して、TopN/合成オッズ/EV/ROI/命中率を算出。

将来拡張（雛形のみ）:
- st_model / fin_model / placement_model / compose_120 を順次起動して pred_csv を生成してから評価へ。

使い方:
  python scripts/train_eval_pipeline.py --config data/pipeline.config.yaml

必須:
- 環境変数 PYTHONPATH=src を通す（CI なら `echo "PYTHONPATH=src" >> $GITHUB_ENV`）
- `src/__init__.py`, `src/boatpred/__init__.py` は空でOK
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd

# ★ ここを 'boatpred' 直下 import に統一（PYTHONPATH=src 前提）
from boatpred.eval import EvalConfig, evaluate_topN


# ---------------- I/O utils ----------------
def read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # pip install pyyaml
    except Exception:
        print("[FATAL] PyYAML が未導入です: pip install pyyaml", file=sys.stderr)
        raise
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safen_num(x) -> str:
    try:
        f = float(x)
        return str(f).replace(".", "_")
    except Exception:
        return "na"


def _span_from_pred(pred_df: pd.DataFrame) -> str:
    """hd の最小最大からファイル名用スパン文字列を作る（失敗時は空）"""
    try:
        hds = sorted(map(str, pred_df["hd"].astype(str).unique()))
        return f"{hds[0]}_{hds[-1]}_" if hds else ""
    except Exception:
        return ""


def _parse_paths(cfg: Dict[str, Any]) -> Tuple[Path, Path, Path, Path]:
    paths = cfg.get("paths", {})
    data_dir = Path(paths.get("data_dir", "data"))
    out_dir = Path(paths.get("out_dir", str(data_dir / "eval")))
    odds_root = Path(paths.get("odds_root", "public/odds/v1"))
    results_root = Path(paths.get("results_dir", "public/results"))
    return data_dir, out_dir, odds_root, results_root


def _parse_period(cfg: Dict[str, Any]) -> Tuple[str, str, str, str]:
    train_cfg = cfg.get("train", {})
    test_cfg = cfg.get("test", {})
    return (
        str(train_cfg.get("start", "")),
        str(train_cfg.get("end", "")),
        str(test_cfg.get("start", "")),
        str(test_cfg.get("end", "")),
    )


def _parse_eval_config(cfg: Dict[str, Any]) -> EvalConfig:
    # 基本オプション（eval節優先、fallbackとして root にも許容）
    ev = cfg.get("eval", {})
    topk = int(ev.get("topk", cfg.get("topk", 18)))
    unit = int(ev.get("unit_stake", cfg.get("unit_stake", 100)))
    rkey = ev.get("rank_key", cfg.get("rank_key", "proba"))
    ev_min = float(ev.get("ev_min", cfg.get("ev_min", 0.0)))
    ev_drop_miss = bool(ev.get("ev_drop_if_missing_odds", cfg.get("ev_drop_if_missing_odds", False)))
    load_odds_always = bool(cfg.get("load_odds_always", ev.get("load_odds_always", True)))

    # Top18 形状フィルタ（filters 節）
    filters = cfg.get("filters", {})
    p1win_max = float(filters.get("p1win_max", 1.0))
    s18_min = float(filters.get("s18_min", 0.0))
    oh_max = float(filters.get("one_head_ratio_top18_max", 1.0))
    non1_min = int(filters.get("min_non1_candidates_top18", 0))
    daily_cap = int(filters.get("daily_cap", 0))

    # TopN バスケット系（eval節優先、fallback root）
    synth_odds_min = float(cfg.get("synth_odds_min", ev.get("synth_odds_min", 0.0)))
    min_odds_min = float(cfg.get("min_odds_min", ev.get("min_odds_min", 0.0)))
    ev_basket_min = float(cfg.get("ev_basket_min", ev.get("ev_basket_min", 0.0)))
    odds_coverage_min = int(cfg.get("odds_coverage_min", ev.get("odds_coverage_min", 0)))

    return EvalConfig(
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


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="パイプライン設定 YAML のパス")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = read_yaml(cfg_path)

    # パス類
    data_dir, out_dir, odds_root, results_root = _parse_paths(cfg)
    ensure_dir(out_dir)

    # 期間（現時点では出力名ヒントに使うだけ）
    train_start, train_end, test_start, test_end = _parse_period(cfg)
    _ = (train_start, train_end, test_start, test_end)  # 未来拡張で使用

    # 入力 CSV（compose_120 相当の出力）
    inputs = cfg.get("inputs", {})
    pred_csv = inputs.get("pred_csv")
    if not pred_csv:
        raise RuntimeError(
            "inputs.pred_csv が未指定です。compose_120 の出力 CSV（列: hd,jcd,rno,combo,proba）を指定してください。"
        )
    pred_csv_path = Path(pred_csv)
    if not pred_csv_path.exists():
        raise FileNotFoundError(f"pred_csv not found: {pred_csv_path}")

    # 評価設定
    ec = _parse_eval_config(cfg)

    # 将来: ST/決まり手/単勝/2着/3着 → compose → pred_csv 生成（今は外部生成を読み込む）
    pred_df = pd.read_csv(pred_csv_path)

    # 列バリデーション（最低限）
    needed = {"hd", "jcd", "rno", "combo", "proba"}
    missing = [c for c in needed if c not in pred_df.columns]
    if missing:
        raise ValueError(f"pred_csv に必要な列が不足しています: missing={missing}, path={pred_csv_path}")

    # 実行
    picks_df, summary = evaluate_topN(
        pred_df=pred_df,
        odds_root=odds_root,
        results_root=results_root,
        cfg=ec,
    )

    # 出力ファイル名のモード文字列
    span = _span_from_pred(pred_df)
    mode = (
        f"{ec.rank_key}"
        f"_evmin{_safen_num(ec.ev_min)}"
        f"_p1{_safen_num(ec.p1win_max)}"
        f"_s18{_safen_num(ec.s18_min)}"
        f"_oh{_safen_num(ec.one_head_ratio_top18_max)}"
    )
    if ec.daily_cap > 0:
        mode += f"_cap{ec.daily_cap}"
    if (ec.rank_key == "ev") or ec.load_odds_always or ec.synth_odds_min > 0 or ec.min_odds_min > 0 or ec.ev_basket_min > 0:
        mode += "_odds"
    if ec.synth_odds_min > 0 or ec.min_odds_min > 0 or ec.ev_basket_min > 0 or ec.odds_coverage_min > 0:
        mode += f"_sb{_safen_num(ec.synth_odds_min)}_mo{_safen_num(ec.min_odds_min)}_eb{_safen_num(ec.ev_basket_min)}_cov{int(ec.odds_coverage_min)}"

    picks_path = out_dir / f"picks_{span}k{ec.topk}_{mode}.csv"
    summary_path = out_dir / f"summary_{span}k{ec.topk}_{mode}.json"

    # 保存
    if not picks_df.empty:
        picks_df.to_csv(picks_path, index=False, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # ログ
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[WRITE] {picks_path}")
    print(f"[WRITE] {summary_path}")


if __name__ == "__main__":
    main()
