#!/usr/bin/env python3
"""Card-boost drift report.

Joins leaderboard actual multipliers from data/top_performers.csv to
saved predictions in data/predictions/{date}.csv and reports:
  - overall MAE / bias / underprediction rate
  - segmented boost-band metrics
  - likely-source layer attribution (daily ingestion vs model path)

Run:
  python scripts/boost_drift_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import unicodedata
from collections import defaultdict
from datetime import date
from statistics import mean


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREDICTIONS_DIR = os.path.join(REPO_ROOT, "data", "predictions")
BOOSTS_DIR = os.path.join(REPO_ROOT, "data", "boosts")
TOP_PERFORMERS_CSV = os.path.join(REPO_ROOT, "data", "top_performers.csv")
ACTUALS_DIR = os.path.join(REPO_ROOT, "data", "actuals")
MODEL_CONFIG_JSON = os.path.join(REPO_ROOT, "data", "model-config.json")
BOOST_MODEL_PKL = os.path.join(REPO_ROOT, "boost_model.pkl")


def _normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name or "")
    return nfkd.encode("ASCII", "ignore").decode("ASCII").strip().lower()


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _date_in_range(date_str: str, start: str, end: str) -> bool:
    try:
        d = date.fromisoformat(date_str)
        return date.fromisoformat(start) <= d <= date.fromisoformat(end)
    except ValueError:
        return False


def _load_model_config() -> dict:
    if not os.path.exists(MODEL_CONFIG_JSON):
        return {}
    with open(MODEL_CONFIG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_top_performers(path: str, start: str, end: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        d = (r.get("date") or "").strip()
        if not _date_in_range(d, start, end):
            continue
        if (r.get("source") or "").strip() != "highest_value":
            continue
        if not (r.get("player_name") or "").strip():
            continue
        out.append(r)
    return out


def _load_actuals_highest_value(start: str, end: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.isdir(ACTUALS_DIR):
        return rows
    for fname in sorted(os.listdir(ACTUALS_DIR)):
        if not fname.endswith(".csv"):
            continue
        d = fname[:-4]
        if not _date_in_range(d, start, end):
            continue
        path = os.path.join(ACTUALS_DIR, fname)
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("source") or "").strip() != "highest_value":
                    continue
                if not (row.get("player_name") or "").strip():
                    continue
                rows.append(
                    {
                        "date": d,
                        "player_name": row.get("player_name", ""),
                        "actual_card_boost": row.get("actual_card_boost", ""),
                        "source": "highest_value_actuals_fallback",
                    }
                )
    return rows


def _prediction_lookup_by_date(date_str: str) -> dict[str, dict]:
    path = os.path.join(PREDICTIONS_DIR, f"{date_str}.csv")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    best = {}
    for row in rows:
        key = _normalize_name(row.get("player_name", ""))
        if not key:
            continue
        pred_rs = _safe_float(row.get("predicted_rs"))
        if pred_rs <= 0:
            continue
        # Card-boost quality should use slate-wide lineups only.
        # Per-game `the_lineup` intentionally zeros est_card_boost.
        scope = (row.get("scope") or "").strip().lower()
        lineup_type = (row.get("lineup_type") or "").strip().lower()
        if scope != "slate" and lineup_type == "the_lineup":
            continue
        if scope != "slate":
            continue
        current = best.get(key)
        if current is None:
            best[key] = row
            continue
        current_boost = _safe_float(current.get("est_card_boost"))
        row_boost = _safe_float(row.get("est_card_boost"))
        # Prefer records that carry non-zero est_card_boost, then highest RS.
        if (row_boost > 0 and current_boost <= 0) or pred_rs > _safe_float(current.get("predicted_rs")):
            best[key] = row
    return best


def _daily_boost_map(date_str: str) -> dict[str, float]:
    path = os.path.join(BOOSTS_DIR, f"{date_str}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    players = data.get("players", []) if isinstance(data, dict) else []
    out: dict[str, float] = {}
    for p in players:
        name = _normalize_name(p.get("player_name") or p.get("name") or "")
        if not name:
            continue
        boost = _safe_float(p.get("boost"))
        if boost > 0:
            out[name] = boost
    return out


def _band(actual_boost: float) -> str:
    if actual_boost < 1.0:
        return "<1.0"
    if actual_boost < 1.5:
        return "1.0-1.5"
    if actual_boost < 2.0:
        return "1.5-2.0"
    return "2.0+"


def _safe_mean(vals: list[float]) -> float:
    return mean(vals) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Card-boost drift report")
    parser.add_argument("--start-date", default="2026-01-17")
    parser.add_argument("--end-date", default="2026-03-19")
    parser.add_argument("--top-performers-csv", default=TOP_PERFORMERS_CSV)
    parser.add_argument("--disable-actuals-fallback", action="store_true")
    args = parser.parse_args()

    cfg = _load_model_config()
    overrides = cfg.get("card_boost", {}).get("player_overrides", {}) if isinstance(cfg, dict) else {}
    overrides_norm = {_normalize_name(k) for k in overrides.keys()}
    boost_model_available = os.path.exists(BOOST_MODEL_PKL)

    top_rows = _load_top_performers(args.top_performers_csv, args.start_date, args.end_date)
    if not args.disable_actuals_fallback:
        fallback_rows = _load_actuals_highest_value(args.start_date, args.end_date)
        existing = {(r.get("date"), _normalize_name(r.get("player_name", ""))) for r in top_rows}
        for row in fallback_rows:
            key = (row.get("date"), _normalize_name(row.get("player_name", "")))
            if key not in existing:
                top_rows.append(row)
    pred_cache: dict[str, dict[str, dict]] = {}
    daily_cache: dict[str, dict[str, float]] = {}

    comparisons = []
    missing_preds = 0

    for row in top_rows:
        d = row["date"]
        n = _normalize_name(row.get("player_name", ""))
        if d not in pred_cache:
            pred_cache[d] = _prediction_lookup_by_date(d)
        if d not in daily_cache:
            daily_cache[d] = _daily_boost_map(d)

        pred_row = pred_cache[d].get(n)
        if not pred_row:
            missing_preds += 1
            continue

        pred_boost = _safe_float(pred_row.get("est_card_boost"))
        actual_boost = _safe_float(row.get("actual_card_boost"))
        if pred_boost <= 0 and actual_boost <= 0:
            continue

        if n in daily_cache[d]:
            layer = "layer0_daily_ingestion"
        elif boost_model_available:
            layer = "layer1_ml_model"
        elif n in overrides_norm:
            layer = "layer2_config_override"
        else:
            layer = "layer3_sigmoid_or_loglinear"

        err = actual_boost - pred_boost
        comparisons.append(
            {
                "date": d,
                "player": row.get("player_name", ""),
                "pred_boost": pred_boost,
                "actual_boost": actual_boost,
                "error": err,
                "abs_error": abs(err),
                "band": _band(actual_boost),
                "layer": layer,
                "pred_min": _safe_float(pred_row.get("pred_min")),
                "predicted_rs": _safe_float(pred_row.get("predicted_rs")),
            }
        )

    print("=" * 78)
    print(f"CARD BOOST DRIFT REPORT  ({args.start_date} -> {args.end_date})")
    print("=" * 78)
    print(f"Top performer rows considered: {len(top_rows)}")
    print(f"Missing prediction joins:       {missing_preds}")
    print(f"Comparisons made:              {len(comparisons)}")

    if not comparisons:
        print("\nNo joined boost comparisons available.")
        return

    abs_errors = [r["abs_error"] for r in comparisons]
    signed_errors = [r["error"] for r in comparisons]
    underpred = sum(1 for r in comparisons if r["error"] > 0)
    overpred = sum(1 for r in comparisons if r["error"] < 0)
    print("\nOverall")
    print("-" * 78)
    print(f"MAE:                 {_safe_mean(abs_errors):.3f}")
    print(f"Bias (actual-pred):  {_safe_mean(signed_errors):+.3f}")
    print(f"Underprediction:     {underpred}/{len(comparisons)} ({100*underpred/max(1,len(comparisons)):.1f}%)")
    print(f"Overprediction:      {overpred}/{len(comparisons)} ({100*overpred/max(1,len(comparisons)):.1f}%)")

    by_band: dict[str, list[dict]] = defaultdict(list)
    by_layer: dict[str, list[dict]] = defaultdict(list)
    for r in comparisons:
        by_band[r["band"]].append(r)
        by_layer[r["layer"]].append(r)

    print("\nBy actual boost band")
    print("-" * 78)
    print(f"{'Band':<10} {'N':>5} {'MAE':>8} {'Bias':>10} {'Under%':>8}")
    for band in ("<1.0", "1.0-1.5", "1.5-2.0", "2.0+"):
        rows = by_band.get(band, [])
        if not rows:
            print(f"{band:<10} {0:>5} {'-':>8} {'-':>10} {'-':>8}")
            continue
        mae = _safe_mean([r["abs_error"] for r in rows])
        bias = _safe_mean([r["error"] for r in rows])
        under_pct = 100 * sum(1 for r in rows if r["error"] > 0) / len(rows)
        print(f"{band:<10} {len(rows):>5} {mae:>8.3f} {bias:>+10.3f} {under_pct:>7.1f}%")

    print("\nBy inferred boost layer")
    print("-" * 78)
    print(f"{'Layer':<28} {'N':>5} {'MAE':>8} {'Bias':>10} {'Under%':>8}")
    for layer, rows in sorted(by_layer.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        mae = _safe_mean([r["abs_error"] for r in rows])
        bias = _safe_mean([r["error"] for r in rows])
        under_pct = 100 * sum(1 for r in rows if r["error"] > 0) / len(rows)
        print(f"{layer:<28} {len(rows):>5} {mae:>8.3f} {bias:>+10.3f} {under_pct:>7.1f}%")

    top_pos = sorted(comparisons, key=lambda r: r["error"], reverse=True)[:12]
    top_neg = sorted(comparisons, key=lambda r: r["error"])[:12]

    print("\nWorst underpredictions (actual > predicted)")
    print("-" * 78)
    print(f"{'Player':<24} {'Date':<12} {'Pred':>6} {'Actual':>7} {'Error':>7} {'Layer':<24}")
    for r in top_pos:
        print(f"{r['player'][:24]:<24} {r['date']:<12} {r['pred_boost']:>6.2f} {r['actual_boost']:>7.2f} {r['error']:>+7.2f} {r['layer']:<24}")

    print("\nWorst overpredictions (actual < predicted)")
    print("-" * 78)
    print(f"{'Player':<24} {'Date':<12} {'Pred':>6} {'Actual':>7} {'Error':>7} {'Layer':<24}")
    for r in top_neg:
        print(f"{r['player'][:24]:<24} {r['date']:<12} {r['pred_boost']:>6.2f} {r['actual_boost']:>7.2f} {r['error']:>+7.2f} {r['layer']:<24}")


if __name__ == "__main__":
    main()
