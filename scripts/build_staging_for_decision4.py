#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
決まり手4分類（逃げ・差し・まくり・まくり差し）学習用ステージング生成スクリプト

出力: data/staging/YYYY/MMDD/decision4_train.csv

各レースについて:
  - 1着艇の決まり手(decision)を全艇に紐付け
  - 1着艇の決まり手が ['逃げ','差し','まくり','まくり差し'] の場合のみ採用
  - 他の決まり手（抜き・恵まれ・不明）は除外
  - 1着艇 → 対応する決まり手クラス
  - それ以外の艇 → "lose" クラス

使用データ:
  - integrated_pro.csv（staging元）
  - results/*.json（決まり手情報を取得）
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import polars as pl


# -----------------------------
# 定義
# -----------------------------
VALID_DECISIONS = ["逃げ", "差し", "まくり", "まくり差し"]
LABELS = ["lose"] + VALID_DECISIONS  # 5クラス

# -----------------------------
# ユーティリティ
# -----------------------------
def log(msg: str):
    print(msg, flush=True)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# -----------------------------
# JSONロード
# -----------------------------
def load_result_json(result_path: Path) -> dict | None:
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# -----------------------------
# ステージング構築
# -----------------------------
def build_decision_staging(
    integrated_dir: Path,
    results_dir: Path,
    out_root: Path,
):
    out_rows = []

    for csv_path in sorted(integrated_dir.rglob("integrated_pro.csv")):
        try:
            parts = csv_path.parts
            if len(parts) < 4:
                continue
            year = parts[-3]
            md = parts[-2]
            date = f"{year}{md}"

            df = pl.read_csv(csv_path, ignore_errors=True)
            if "rno" not in df.columns or "lane" not in df.columns:
                continue

            for (jcd, rno), grp in df.group_by(["jcd", "rno"], maintain_order=True):
                jcd_str = str(jcd)
                rno_int = int(rno)
                result_path = results_dir / year / md / f"{jcd_str}" / f"{rno_int}R.json"
                if not result_path.exists():
                    continue

                result = load_result_json(result_path)
                if not result:
                    continue
                meta = result.get("meta", {})
                dec = meta.get("decision", "")
                if dec not in VALID_DECISIONS:
                    # 無効な決まり手ならスキップ
                    continue

                rows = []
                for lane, sub in grp.iter_rows(named=True):
                    lane_num = sub["lane"]
                    regno = sub.get("racer_id") or sub.get("regno")
                    rank = None
                    # 結果データから1着判定
                    for rr in result.get("results", []):
                        if str(rr.get("lane")) == str(lane_num):
                            rank = rr.get("rank")
                            break
                    label = dec if str(rank) == "1" else "lose"
                    rows.append({
                        "hd": date,
                        "jcd": jcd_str,
                        "rno": rno_int,
                        "lane": lane_num,
                        "regno": regno,
                        "decision_label": label,
                        "decision_code": LABELS.index(label),
                        # 特徴量（抜粋: 軽量ver）
                        "tenji_st_sec": sub.get("tenji_st_sec"),
                        "tenji_rank": sub.get("tenji_rank"),
                        "st_rank": sub.get("st_rank"),
                        "course_avg_st": sub.get("course_avg_st"),
                        "course_first_rate": sub.get("course_first_rate"),
                        "course_3rd_rate": sub.get("course_3rd_rate"),
                        "motor_rate2": sub.get("motor_rate2"),
                        "boat_rate2": sub.get("boat_rate2"),
                        "wind_speed_m": sub.get("wind_speed_m"),
                        "wave_height_cm": sub.get("wave_height_cm"),
                    })
                out_rows.extend(rows)
        except Exception as e:
            print(f"[WARN] {csv_path}: {e}", file=sys.stderr)
            continue

    if not out_rows:
        print("[WARN] 出力対象データがありません。")
        return

    df_out = pl.DataFrame(out_rows)
    for y, grp_y in df_out.group_by("hd"):
        year = str(y)[:4]
        md = str(y)[4:]
        out_dir = out_root / year / md
        ensure_dir(out_dir)
        out_path = out_dir / "decision4_train.csv"
        grp_y.write_csv(out_path)
        log(f"[WRITE] {out_path} rows={grp_y.height}")


# -----------------------------
# main
# -----------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--integrated_root", default="data/shards")
    ap.add_argument("--results_root", default="public/results")
    ap.add_argument("--out_root", default="data/staging")
    args = ap.parse_args()

    integrated_root = Path(args.integrated_root)
    results_root = Path(args.results_root)
    out_root = Path(args.out_root)

    build_decision_staging(integrated_root, results_root, out_root)


if __name__ == "__main__":
    main()
