#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_eval_pipeline.py — 統合パイプライン
- src/boatpred の各モジュール（io / priors / st_model / fin_model / placement_model / compose_120 / eval）
  を “存在していれば” 自動実行。無ければスキップして次へ進む柔軟実装。
- 最終的に (hd,jcd,rno,combo,proba) のCSVを得られたら evaluate_topN で評価。

使い方:
  python scripts/train_eval_pipeline.py --config data/pipeline.config.yaml
"""
from __future__ import annotations
import argparse, json, sys, importlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

# --- src を import 可能に ---
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

# 評価は固定エントリ
from src.boatpred.eval import EvalConfig, evaluate_topN

try:
    import yaml
except Exception as e:
    yaml = None


# ----------------- util -----------------
def read_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML 未導入です。pip install pyyaml")
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def has_attr(mod, names):
    for n in names:
        if hasattr(mod, n):
            return n
    return None

def _slice_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = str(start); e = str(end)
    m = (df["hd"].astype(str) >= s) & (df["hd"].astype(str) <= e)
    return df.loc[m].copy()

def _save(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"[WRITE] {path}")
    return path


# ----------------- main blocks -----------------
def run_st(cfg: Dict[str, Any], train_span: Tuple[str,str], test_span: Tuple[str,str]) -> Optional[Path]:
    """ST学習/推論（src.boatpred.st_model）"""
    st_cfg = cfg.get("st", {})
    if not st_cfg or not bool(st_cfg.get("enabled", False)):
        print("[st] disabled — skip")
        return None

    print("[st] start")
    mod = importlib.import_module("src.boatpred.st_model")

    # 設定
    shards_root = Path(st_cfg.get("shards_root", "data/shards"))
    model_out   = Path(st_cfg.get("model_out", "data/models/st_lgbm.txt"))
    out_dir     = Path(st_cfg.get("out_dir", "data/st_out"))
    ensure_dir(model_out.parent); ensure_dir(out_dir)

    # 入力 st_rows.* を io から読む or 直接読む
    try:
        io_mod = importlib.import_module("src.boatpred.io")
        fn = has_attr(io_mod, ["load_st_rows"])
        if fn:
            all_rows = getattr(io_mod, fn)(shards_root)
        else:
            # フォールバック: 直接走査
            import glob
            files = []
            files += glob.glob(str(shards_root / "**" / "st_rows.parquet"), recursive=True)
            files += glob.glob(str(shards_root / "**" / "st_rows.csv"), recursive=True)
            files += glob.glob(str(shards_root / "**" / "st_rows.csv.gz"), recursive=True)
            if not files:
                raise FileNotFoundError("st_rows.* が見つかりません")
            dfs = []
            for p in files:
                pth = Path(p)
                if pth.suffix == ".parquet":
                    dfs.append(pd.read_parquet(pth))
                else:
                    dfs.append(pd.read_csv(pth))
            all_rows = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        raise RuntimeError(f"[st] st_rows 読み込み失敗: {e}")

    # 型整理
    for c in ("hd","jcd","rno","lane","regno"):
        if c in all_rows.columns:
            all_rows[c] = all_rows[c].astype(str)
    if "lane" in all_rows.columns:
        try: all_rows["lane"] = all_rows["lane"].astype(int)
        except: pass

    train_start, train_end = train_span
    test_start,  test_end  = test_span
    tr = _slice_by_date(all_rows, train_start, train_end)
    te = _slice_by_date(all_rows, test_start, test_end)

    # entrypoints の自動検出
    STConfig = getattr(mod, "STConfig", None)
    fn_train = has_attr(mod, ["train"])
    fn_pred  = has_attr(mod, ["predict"])

    if not (STConfig and fn_train and fn_pred):
        raise RuntimeError("[st] st_model のエントリ(train/predict/STConfig)が見つかりません")

    stc = STConfig(model_out=str(model_out), meta_out=str(model_out.with_suffix(".meta.json")))
    print(f"[st] train rows={len(tr)} / test rows (racecard base)={len(te)}")

    booster, feats = getattr(mod, fn_train)(
        hist_df=tr,
        ref_end=train_end,
        cfg=stc,
        hd_col="hd", reg_col="regno", lane_col="lane", st_col="st"
    )

    # レースカード生成（te からユニーク抽出）
    need = ["hd","jcd","rno","lane","regno"]
    for col in need:
        if col not in te.columns:
            raise KeyError(f"[st] test rows に列がありません: {col}")
    racecard = (
        te[need]
        .dropna()
        .drop_duplicates()
        .sort_values(["hd","jcd","rno","lane"])
        .reset_index(drop=True)
    )

    pred = getattr(mod, fn_pred)(
        booster=booster,
        racecard_df=racecard,
        history_df=tr,
        ref_end=train_end,
        cfg=stc,
        hd_col="hd", jcd_col="jcd", rno_col="rno", reg_col="regno", lane_col="lane", st_col="st"
    )

    return _save(pred, out_dir / f"st_pred_{test_start}_{test_end}.csv")


def run_fin(cfg: Dict[str, Any], train_span: Tuple[str,str], test_span: Tuple[str,str], st_pred_csv: Optional[Path]) -> Optional[Path]:
    """決まり手/単勝ブロック（src.boatpred.fin_model）— あれば実行"""
    try:
        mod = importlib.import_module("src.boatpred.fin_model")
    except Exception:
        print("[fin] module not found — skip")
        return None

    fin_cfg = cfg.get("fin", {"enabled": True})
    if not bool(fin_cfg.get("enabled", True)):
        print("[fin] disabled — skip")
        return None

    print("[fin] start")
    out_dir = Path(fin_cfg.get("out_dir", "data/fin_out")); ensure_dir(out_dir)
    model_out = Path(fin_cfg.get("model_out", "data/models/fin_model.txt"))

    # 入力データ（io から取れるなら使う）
    io_mod = None
    try:
        io_mod = importlib.import_module("src.boatpred.io")
    except Exception:
        pass

    loader = has_attr(io_mod, ["load_fin_rows"]) if io_mod else None
    if loader:
        all_rows = getattr(io_mod, loader)()
    else:
        # ない場合は st_pred が最低限のキーを持つと仮定し、そこから補完
        if not st_pred_csv or not st_pred_csv.exists():
            print("[fin] no input rows — skip")
            return None
        all_rows = pd.read_csv(st_pred_csv)

    train_start, train_end = train_span
    test_start,  test_end  = test_span
    tr = _slice_by_date(all_rows, train_start, train_end)
    te = _slice_by_date(all_rows, test_start, test_end)

    fn_train = has_attr(mod, ["train", "train_finisher"])
    fn_pred  = has_attr(mod, ["predict", "predict_finisher"])
    if not (fn_train and fn_pred):
        print("[fin] entrypoints not found — skip")
        return None

    booster = getattr(mod, fn_train)(tr, train_end, model_out=str(model_out))
    pred = getattr(mod, fn_pred)(booster, te, ref_end=train_end)

    return _save(pred, out_dir / f"fin_pred_{test_start}_{test_end}.csv")


def run_place(cfg: Dict[str, Any], train_span: Tuple[str,str], test_span: Tuple[str,str], fin_pred_csv: Optional[Path]) -> Optional[Path]:
    """2着/3着条件付き分布ブロック（src.boatpred.placement_model）— あれば実行"""
    try:
        mod = importlib.import_module("src.boatpred.placement_model")
    except Exception:
        print("[place] module not found — skip")
        return None

    pl_cfg = cfg.get("placement", {"enabled": True})
    if not bool(pl_cfg.get("enabled", True)):
        print("[place] disabled — skip")
        return None

    print("[place] start")
    out_dir = Path(pl_cfg.get("out_dir", "data/place_out")); ensure_dir(out_dir)
    model_out = Path(pl_cfg.get("model_out", "data/models/place_model.txt"))

    if not fin_pred_csv or not fin_pred_csv.exists():
        print("[place] fin_pred not found — skip")
        return None

    fin_pred = pd.read_csv(fin_pred_csv)

    fn_train = has_attr(mod, ["train", "train_place"])
    fn_pred  = has_attr(mod, ["predict", "predict_place"])
    if not (fn_train and fn_pred):
        print("[place] entrypoints not found — skip")
        return None

    train_start, train_end = train_span
    test_start,  test_end  = test_span

    booster = getattr(mod, fn_train)(fin_pred, ref_end=train_end, model_out=str(model_out))
    pred = getattr(mod, fn_pred)(booster, fin_pred, ref_end=train_end)

    return _save(pred, out_dir / f"place_pred_{test_start}_{test_end}.csv")


def run_compose(cfg: Dict[str, Any],
                st_pred_csv: Optional[Path],
                fin_pred_csv: Optional[Path],
                place_pred_csv: Optional[Path]) -> Optional[Path]:
    """120通り合成（src.boatpred.compose_120）— あれば実行。出力は (hd,jcd,rno,combo,proba)"""
    try:
        mod = importlib.import_module("src.boatpred.compose_120")
    except Exception:
        print("[compose] module not found — skip")
        return None

    comp_cfg = cfg.get("compose", {"enabled": True})
    if not bool(comp_cfg.get("enabled", True)):
        print("[compose] disabled — skip")
        return None

    print("[compose] start")
    out_dir = Path(comp_cfg.get("out_dir", "data/compose")); ensure_dir(out_dir)

    fn = has_attr(mod, ["compose", "compose_from_layers"])
    if not fn:
        print("[compose] compose() not found — skip")
        return None

    kwargs = {}
    if st_pred_csv and st_pred_csv.exists():    kwargs["st_pred_csv"] = str(st_pred_csv)
    if fin_pred_csv and fin_pred_csv.exists():  kwargs["fin_pred_csv"] = str(fin_pred_csv)
    if place_pred_csv and place_pred_csv.exists(): kwargs["place_pred_csv"] = str(place_pred_csv)

    pred_df = getattr(mod, fn)(**kwargs)  # 期待: DataFrame を返す or CSVパス返す
    if isinstance(pred_df, (str, Path)):
        path = Path(pred_df)
        if path.exists():
            print(f"[compose] use existing file: {path}")
            return path
        else:
            raise RuntimeError("[compose] returned path not found")
    elif isinstance(pred_df, pd.DataFrame):
        out_csv = out_dir / "preds_composed.csv"
        return _save(pred_df, out_csv)
    else:
        print("[compose] unexpected return — skip")
        return None


def maybe_evaluate(cfg: Dict[str, Any], pred_csv: Optional[Path]) -> Optional[Path]:
    """(hd,jcd,rno,combo,proba) を受けて evaluate_topN を実行"""
    if not pred_csv or not pred_csv.exists():
        # inputs.pred_csv の明示指定があればそれを使う
        inputs = cfg.get("inputs", {})
        if inputs.get("pred_csv"):
            alt = Path(inputs["pred_csv"])
            if alt.exists():
                pred_csv = alt
            else:
                raise FileNotFoundError(f"[eval] pred_csv not found: {alt}")
        else:
            print("[eval] pred_csv 未指定 — skip")
            return None

    paths = cfg.get("paths", {})
    odds_root    = Path(paths.get("odds_root", "public/odds/v1"))
    results_root = Path(paths.get("results_dir", "public/results"))
    out_dir      = Path(paths.get("out_dir", "data/eval")); ensure_dir(out_dir)

    eval_opts = cfg.get("eval", {})
    filters   = cfg.get("filters", {})

    ec = EvalConfig(
        topk=int(eval_opts.get("topk", 18)),
        unit_stake=int(eval_opts.get("unit_stake", 100)),
        rank_key=str(eval_opts.get("rank_key", "proba")),
        ev_min=float(eval_opts.get("ev_min", 0.0)),
        ev_drop_if_missing_odds=bool(eval_opts.get("ev_drop_if_missing_odds", False)),
        load_odds_always=bool(eval_opts.get("load_odds_always", False)),
        synth_odds_min=float(eval_opts.get("synth_odds_min", 0.0)),
        min_odds_min=float(eval_opts.get("min_odds_min", 0.0)),
        ev_basket_min=float(eval_opts.get("ev_basket_min", 0.0)),
        odds_coverage_min=int(eval_opts.get("odds_coverage_min", 0)),
        p1win_max=float(filters.get("p1win_max", 1.0)),
        s18_min=float(filters.get("s18_min", 0.0)),
        one_head_ratio_top18_max=float(filters.get("one_head_ratio_top18_max", 1.0)),
        min_non1_candidates_top18=int(filters.get("min_non1_candidates_top18", 0)),
        daily_cap=int(filters.get("daily_cap", 0)),
    )

    pred_df = pd.read_csv(pred_csv)
    picks_df, summary = evaluate_topN(pred_df=pred_df, odds_root=odds_root, results_root=results_root, cfg=ec)

    # span
    try:
        hds = sorted(map(str, pred_df["hd"].astype(str).unique()))
        span = f"{hds[0]}_{hds[-1]}_" if hds else ""
    except Exception:
        span = ""

    def _safen(x):
        try: return str(float(x)).replace(".","_")
        except: return "na"

    mode = f"{ec.rank_key}_evmin{_safen(ec.ev_min)}_p1{_safen(ec.p1win_max)}_s18{_safen(ec.s18_min)}_oh{_safen(ec.one_head_ratio_top18_max)}"
    if ec.daily_cap > 0: mode += f"_cap{ec.daily_cap}"
    if (ec.rank_key == "ev") or ec.load_odds_always or ec.synth_odds_min>0 or ec.min_odds_min>0 or ec.ev_basket_min>0:
        mode += "_odds"
    if ec.synth_odds_min>0 or ec.min_odds_min>0 or ec.ev_basket_min>0 or ec.odds_coverage_min>0:
        mode += f"_sb{_safen(ec.synth_odds_min)}_mo{_safen(ec.min_odds_min)}_eb{_safen(ec.ev_basket_min)}_cov{int(ec.odds_coverage_min)}"

    picks_path = out_dir / f"picks_{span}k{ec.topk}_{mode}.csv"
    sum_path   = out_dir / f"summary_{span}k{ec.topk}_{mode}.json"
    if not picks_df.empty:
        picks_df.to_csv(picks_path, index=False, encoding="utf-8")
    sum_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[eval] WRITE {picks_path}")
    print(f"[eval] WRITE {sum_path}")
    return sum_path


# ----------------- entry -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = read_yaml(cfg_path)

    # 期間
    train = cfg.get("train", {}); test = cfg.get("test", {})
    train_span = (str(train.get("start","")).strip(), str(train.get("end","")).strip())
    test_span  = (str(test.get("start","")).strip(),  str(test.get("end","")).strip())
    if not all(train_span) or not all(test_span):
        raise RuntimeError("train/test の start/end を YAML に指定してください。")

    # 1) ST
    st_pred = run_st(cfg, train_span, test_span)

    # 2) FIN（決まり手/単勝）
    fin_pred = run_fin(cfg, train_span, test_span, st_pred)

    # 3) PLACEMENT（2・3着条件付き）
    place_pred = run_place(cfg, train_span, test_span, fin_pred)

    # 4) COMPOSE 120
    composed = run_compose(cfg, st_pred, fin_pred, place_pred)

    # 5) EVAL（compose が出ていればそれを、無ければ inputs.pred_csv を使う）
    maybe_evaluate(cfg, composed)


if __name__ == "__main__":
    main()
