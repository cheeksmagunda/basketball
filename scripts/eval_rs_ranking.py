#!/usr/bin/env python3
"""
Evaluate slate RS ranking vs logged actuals (offline KPIs).

Reads data/predictions/{date}.csv (all scopes/lineups; max predicted_rs per player) and data/actuals/{date}.csv.
Reports mean top-5 RS recall, mean NDCG@5 (RS as relevance), RS mass capture ratio,
and a hindsight winner value ratio (std library only).

Usage:
  python3 scripts/eval_rs_ranking.py
  python3 scripts/eval_rs_ranking.py --from 2026-03-05 --to 2026-03-19
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _normalize_name(name: str) -> str:
    n = str(name).lower().strip()
    n = re.sub(r"['.\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _ndcg_at_k(order: list[str], relevance: dict[str, float], k: int = 5) -> float:
    dcg = 0.0
    for i, key in enumerate(order[:k]):
        rel = max(0.0, relevance.get(key, 0.0))
        dcg += rel / math.log2(i + 2)
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(max(0.0, rel) / math.log2(i + 2) for i, rel in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


def _load_prediction_pool(path: str) -> dict[str, dict]:
    """norm -> {player_name, predicted_rs, est_card_boost} — all scopes/lineups, max RS per player."""
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            nm = _normalize_name(row.get("player_name", ""))
            if not nm:
                continue
            try:
                prs = float(row.get("predicted_rs") or 0)
            except (TypeError, ValueError):
                continue
            try:
                boost = float(row.get("est_card_boost") or 0)
            except (TypeError, ValueError):
                boost = 0.0
            if nm not in out or prs > out[nm]["predicted_rs"]:
                out[nm] = {
                    "player_name": row.get("player_name", ""),
                    "predicted_rs": prs,
                    "est_card_boost": boost,
                }
    return out


def _load_actuals(path: str) -> dict[str, dict]:
    """norm -> {actual_rs, actual_card_boost}"""
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "player_name" not in r.fieldnames or "actual_rs" not in r.fieldnames:
            return {}
        for row in r:
            try:
                ars = float(row.get("actual_rs") or 0)
            except (TypeError, ValueError):
                continue
            nm = _normalize_name(row.get("player_name", ""))
            if not nm:
                continue
            acb = 0.0
            if "actual_card_boost" in row and row.get("actual_card_boost") not in (None, ""):
                try:
                    acb = float(row["actual_card_boost"])
                except (TypeError, ValueError):
                    acb = 0.0
            if nm not in out or ars > out[nm]["actual_rs"]:
                out[nm] = {"actual_rs": ars, "actual_card_boost": acb}
    return out


def _eval_date(pred_path: str, act_path: str) -> dict | None:
    preds = _load_prediction_pool(pred_path)
    if not preds or not os.path.isfile(act_path):
        return None
    act = _load_actuals(act_path)
    if not act:
        return None

    merged: list[tuple[str, float, float, float]] = []
    for norm, p in preds.items():
        if norm not in act:
            continue
        merged.append(
            (
                norm,
                p["predicted_rs"],
                act[norm]["actual_rs"],
                act[norm]["actual_card_boost"],
            )
        )
    if len(merged) < 8:
        return None

    merged.sort(key=lambda x: -x[1])
    pred_order = [m[0] for m in merged]
    rel = {m[0]: m[2] for m in merged}

    by_actual = sorted(merged, key=lambda x: -x[2])
    act_top5 = {x[0] for x in by_actual[:5]}
    pred_top5 = set(pred_order[:5])
    recall = len(pred_top5 & act_top5) / 5.0
    ndcg = _ndcg_at_k(pred_order, rel, k=5)

    oracle_rs_mass = sum(x[2] for x in by_actual[:5])
    row = {m[0]: m for m in merged}
    pred_rs_mass = sum(row[n][2] for n in pred_top5 if n in row)
    rs_capture = (pred_rs_mass / oracle_rs_mass) if oracle_rs_mass > 0 else 0.0

    slots = [2.0, 1.8, 1.6, 1.4, 1.2]

    def _lineup_value(norms: list[str]) -> float:
        total = 0.0
        for i, nm in enumerate(norms[:5]):
            if nm not in row:
                continue
            _, _, ars, acb = row[nm]
            total += ars * (slots[i] + acb)
        return total

    pred_val = _lineup_value(pred_order)
    oracle_val = _lineup_value([x[0] for x in by_actual])
    value_ratio = (pred_val / oracle_val) if oracle_val > 0 else 0.0

    return {
        "n_players": len(merged),
        "top5_recall": recall,
        "ndcg5": ndcg,
        "rs_capture_ratio": rs_capture,
        "winner_value_ratio": value_ratio,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="RS ranking KPIs vs actuals")
    ap.add_argument("--from", dest="date_from", default=None, help="YYYY-MM-DD inclusive")
    ap.add_argument("--to", dest="date_to", default=None, help="YYYY-MM-DD inclusive")
    args = ap.parse_args()

    pred_dir = os.path.join(ROOT, "data", "predictions")
    act_dir = os.path.join(ROOT, "data", "actuals")
    pred_files = sorted(glob.glob(os.path.join(pred_dir, "*.csv")))
    recalls, ndcgs, caps, vals = [], [], [], []

    for pf in pred_files:
        base = os.path.basename(pf).replace(".csv", "")
        if args.date_from and base < args.date_from:
            continue
        if args.date_to and base > args.date_to:
            continue
        af = os.path.join(act_dir, f"{base}.csv")
        stats = _eval_date(pf, af)
        if not stats:
            continue
        recalls.append(stats["top5_recall"])
        ndcgs.append(stats["ndcg5"])
        caps.append(stats["rs_capture_ratio"])
        vals.append(stats["winner_value_ratio"])
        print(
            f"{base}  n={stats['n_players']:<3}  top5_recall={stats['top5_recall']:.2f}  "
            f"ndcg@5={stats['ndcg5']:.3f}  rs_capture={stats['rs_capture_ratio']:.3f}  "
            f"value_ratio={stats['winner_value_ratio']:.3f}"
        )

    if not recalls:
        print("No overlapping prediction/actual dates found.", file=sys.stderr)
        return 1

    print("---")
    print(
        f"MEAN  top5_recall={sum(recalls)/len(recalls):.3f}  ndcg@5={sum(ndcgs)/len(ndcgs):.3f}  "
        f"rs_capture={sum(caps)/len(caps):.3f}  value_ratio={sum(vals)/len(vals):.3f}  (n_dates={len(recalls)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
