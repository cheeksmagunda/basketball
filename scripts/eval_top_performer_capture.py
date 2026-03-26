#!/usr/bin/env python3
"""
Evaluate daily Top Performer capture from historical data.

Ranking target (no slot assignment):
    total_value = actual_rs * (1 + actual_card_boost)

For each overlapping date between:
  - data/top_performers.csv (historical outcomes)
  - data/predictions/{date}.csv (predicted player set, scope=slate)

The script reports per-day and aggregate capture for Top-K leaderboards.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
TOP_PERFORMERS = ROOT / "data" / "top_performers.csv"
PREDICTIONS_DIR = ROOT / "data" / "predictions"


def _safe_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _parse_topks(raw: str) -> list[int]:
    out = []
    for part in (raw or "").split(","):
        s = part.strip()
        if not s:
            continue
        k = int(s)
        if k <= 0:
            continue
        out.append(k)
    return sorted(set(out))


def _load_labels() -> dict[str, list[tuple[str, float]]]:
    """Return {date: [(player_name, total_value), ...]} sorted descending by value."""
    by_date: dict[str, list[tuple[str, float]]] = defaultdict(list)
    if not TOP_PERFORMERS.exists():
        return by_date

    with TOP_PERFORMERS.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            date = (row.get("date") or "").strip()
            player = (row.get("player_name") or "").strip()
            actual_rs = _safe_float(row.get("actual_rs") or "")
            actual_boost = _safe_float(row.get("actual_card_boost") or "")
            if not date or not player or actual_rs <= 0:
                continue
            total_value = actual_rs * (1.0 + actual_boost)
            by_date[date].append((player, total_value))

    for d in by_date:
        by_date[d].sort(key=lambda x: x[1], reverse=True)
    return by_date


def _load_predictions() -> dict[str, set[str]]:
    """Return {date: {predicted player names}} from scope=slate rows."""
    by_date: dict[str, set[str]] = defaultdict(set)
    if not PREDICTIONS_DIR.exists():
        return by_date

    for csv_path in sorted(PREDICTIONS_DIR.glob("*.csv")):
        date = csv_path.stem
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if (row.get("scope") or "").strip().lower() != "slate":
                    continue
                player = (row.get("player_name") or "").strip()
                if player:
                    by_date[date].add(player)
    return by_date


def _format_pct(hit: int, total: int) -> str:
    if total <= 0:
        return "0.0%"
    return f"{(100.0 * hit / total):.1f}%"


def _evaluate(topks: Iterable[int]) -> int:
    labels = _load_labels()
    preds = _load_predictions()
    dates = sorted(set(labels).intersection(preds))
    if not dates:
        print("No overlapping dates between top_performers and predictions.")
        return 1

    agg_hits = {k: 0 for k in topks}
    agg_totals = {k: 0 for k in topks}
    per_day_rows = []

    for d in dates:
        ranked = labels[d]
        pred_set = preds[d]
        day = {"date": d, "pred_count": len(pred_set)}
        for k in topks:
            top_names = [name for name, _ in ranked[:k]]
            hits = sum(1 for n in top_names if n in pred_set)
            total = len(top_names)
            day[f"h{k}"] = hits
            day[f"t{k}"] = total
            agg_hits[k] += hits
            agg_totals[k] += total
        per_day_rows.append(day)

    print(f"Overlapping dates: {len(dates)}")
    print("Per-day capture (Top performers by total_value = actual_rs * (1 + actual_card_boost))")
    for row in per_day_rows:
        parts = [f"{row['date']} preds={row['pred_count']:2d}"]
        for k in topks:
            h = row[f"h{k}"]
            t = row[f"t{k}"]
            parts.append(f"Top{k} {h}/{t} ({_format_pct(h, t)})")
        print(" | ".join(parts))

    print("\nAggregate:")
    for k in topks:
        h = agg_hits[k]
        t = agg_totals[k]
        print(f"Top{k}: {h}/{t} = {_format_pct(h, t)}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Top Performer capture by day.")
    parser.add_argument(
        "--topks",
        default="5,10,20",
        help="Comma-separated K values for Top-K capture (default: 5,10,20)",
    )
    args = parser.parse_args()
    topks = _parse_topks(args.topks)
    if not topks:
        print("No valid --topks values provided.")
        return 2
    return _evaluate(topks)


if __name__ == "__main__":
    raise SystemExit(main())
