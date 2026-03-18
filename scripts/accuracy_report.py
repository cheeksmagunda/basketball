#!/usr/bin/env python3
"""Accuracy report: RS projection performance for Mar 5–17, 2026.

Reads predictions, actuals (highest_value), and audit JSONs to produce
a comprehensive accuracy analysis. Read-only — no files modified.

Run from repo root:  python scripts/accuracy_report.py
"""

import csv
import json
import os
import unicodedata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTUALS_DIR = os.path.join(REPO_ROOT, "data", "actuals")
AUDIT_DIR = os.path.join(REPO_ROOT, "data", "audit")
PREDICTIONS_DIR = os.path.join(REPO_ROOT, "data", "predictions")

DATES = [f"2026-03-{d:02d}" for d in range(5, 18)]


def _normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return nfkd.encode("ASCII", "ignore").decode("ASCII").strip().lower()


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def load_predictions(date_str: str) -> list[dict]:
    path = os.path.join(PREDICTIONS_DIR, f"{date_str}.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_actuals(date_str: str) -> list[dict]:
    path = os.path.join(ACTUALS_DIR, f"{date_str}.csv")
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("source") == "highest_value"]


def load_audit(date_str: str) -> dict | None:
    path = os.path.join(AUDIT_DIR, f"{date_str}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("  RS PROJECTION ACCURACY REPORT  —  Mar 5–17, 2026")
    print("=" * 70)

    # Collect all data
    all_errors = []          # (abs_error, date)
    all_misses_detail = []   # full miss records with date
    all_missed_value = []    # players in actuals but not predictions
    all_boost_comparisons = []  # (pred_boost, actual_boost, player, date)
    total_actuals = 0
    total_matched = 0
    over_count = 0
    under_count = 0
    per_date = []

    for date_str in DATES:
        preds = load_predictions(date_str)
        actuals = load_actuals(date_str)
        audit = load_audit(date_str)

        if not actuals:
            continue

        # Build prediction lookup (normalized name → best pred row)
        # Use highest predicted_rs if player appears multiple times (multi-scope)
        pred_map = {}
        for row in preds:
            key = _normalize_name(row.get("player_name", ""))
            pred_rs = _safe_float(row.get("predicted_rs"))
            if key and pred_rs > 0:
                if key not in pred_map or pred_rs > _safe_float(pred_map[key].get("predicted_rs")):
                    pred_map[key] = row

        matched = 0
        date_errors = []
        date_over = 0
        date_under = 0
        biggest_miss_name = ""
        biggest_miss_err = 0

        for act in actuals:
            total_actuals += 1
            aname = _normalize_name(act.get("player_name", ""))
            actual_rs = _safe_float(act.get("actual_rs"))

            if aname not in pred_map:
                # Missed value — leaderboard player we didn't predict
                all_missed_value.append({
                    "player": act.get("player_name", ""),
                    "date": date_str,
                    "actual_rs": actual_rs,
                    "card_boost": act.get("actual_card_boost", ""),
                    "drafts": act.get("drafts", ""),
                    "total_value": act.get("total_value", ""),
                })
                continue

            if actual_rs <= 0:
                continue

            pred_row = pred_map[aname]
            pred_rs = _safe_float(pred_row.get("predicted_rs"))
            err = actual_rs - pred_rs

            matched += 1
            total_matched += 1
            date_errors.append(abs(err))
            all_errors.append(abs(err))

            if err < 0:
                date_over += 1
                over_count += 1
            elif err > 0:
                date_under += 1
                under_count += 1

            all_misses_detail.append({
                "player": pred_row.get("player_name", ""),
                "date": date_str,
                "predicted_rs": round(pred_rs, 2),
                "actual_rs": round(actual_rs, 2),
                "error": round(err, 2),
                "drafts": act.get("drafts", ""),
            })

            if abs(err) > abs(biggest_miss_err):
                biggest_miss_err = err
                biggest_miss_name = pred_row.get("player_name", "")

            # Card boost comparison
            pred_boost = _safe_float(pred_row.get("est_card_boost"))
            actual_boost = _safe_float(act.get("actual_card_boost"))
            if pred_boost > 0 or actual_boost > 0:
                all_boost_comparisons.append({
                    "player": pred_row.get("player_name", ""),
                    "date": date_str,
                    "pred_boost": round(pred_boost, 2),
                    "actual_boost": round(actual_boost, 2),
                    "boost_error": round(actual_boost - pred_boost, 2),
                })

        mae = round(sum(date_errors) / len(date_errors), 2) if date_errors else None
        per_date.append({
            "date": date_str,
            "actuals": len(actuals),
            "matched": matched,
            "missed": len(actuals) - matched,
            "mae": mae,
            "over": date_over,
            "under": date_under,
            "biggest_miss": f"{biggest_miss_name} ({biggest_miss_err:+.1f})" if biggest_miss_name else "—",
        })

    # ── Section 1: Aggregate MAE ──
    agg_mae = round(sum(all_errors) / len(all_errors), 3) if all_errors else 0
    print(f"\n{'─' * 70}")
    print(f"  1. AGGREGATE MAE (weighted by player count)")
    print(f"{'─' * 70}")
    print(f"  Overall MAE:        {agg_mae}")
    print(f"  Total comparisons:  {total_matched} players across {len(per_date)} dates")
    print(f"  Total on leaderboard: {total_actuals} player-days")
    print(f"  Coverage:           {total_matched}/{total_actuals} ({100*total_matched/max(total_actuals,1):.0f}%)")

    # ── Section 2: Per-date breakdown ──
    print(f"\n{'─' * 70}")
    print(f"  2. PER-DATE BREAKDOWN")
    print(f"{'─' * 70}")
    print(f"  {'Date':<12} {'Act':>4} {'Match':>5} {'Miss':>5} {'MAE':>6} {'Over':>5} {'Under':>5}  Biggest Miss")
    print(f"  {'─'*12} {'─'*4} {'─'*5} {'─'*5} {'─'*6} {'─'*5} {'─'*5}  {'─'*25}")
    for d in per_date:
        mae_str = f"{d['mae']:.2f}" if d['mae'] is not None else "—"
        print(f"  {d['date']:<12} {d['actuals']:>4} {d['matched']:>5} {d['missed']:>5} {mae_str:>6} {d['over']:>5} {d['under']:>5}  {d['biggest_miss']}")

    # ── Section 3: Over/under bias ──
    print(f"\n{'─' * 70}")
    print(f"  3. PROJECTION BIAS")
    print(f"{'─' * 70}")
    total_dir = over_count + under_count
    print(f"  Over-projected:  {over_count} ({100*over_count/max(total_dir,1):.0f}%)")
    print(f"  Under-projected: {under_count} ({100*under_count/max(total_dir,1):.0f}%)")
    bias = "OVER-PROJECTS" if over_count > under_count * 1.2 else \
           "UNDER-PROJECTS" if under_count > over_count * 1.2 else "BALANCED"
    print(f"  Verdict:         Model {bias} Real Scores")

    # ── Section 4: Top 15 biggest misses ──
    print(f"\n{'─' * 70}")
    print(f"  4. TOP 15 BIGGEST MISSES (across all dates)")
    print(f"{'─' * 70}")
    all_misses_detail.sort(key=lambda x: abs(x["error"]), reverse=True)
    print(f"  {'Player':<25} {'Date':<12} {'Pred':>5} {'Actual':>7} {'Error':>6} {'Drafts':>7}")
    print(f"  {'─'*25} {'─'*12} {'─'*5} {'─'*7} {'─'*6} {'─'*7}")
    for m in all_misses_detail[:15]:
        print(f"  {m['player']:<25} {m['date']:<12} {m['predicted_rs']:>5.1f} {m['actual_rs']:>7.1f} {m['error']:>+6.1f} {m['drafts']:>7}")

    # ── Section 5: Missed value players ──
    print(f"\n{'─' * 70}")
    print(f"  5. MISSED VALUE PLAYERS (on leaderboard but NOT in predictions)")
    print(f"{'─' * 70}")
    # Sort by total_value descending
    all_missed_value.sort(key=lambda x: _safe_float(x.get("total_value", 0)), reverse=True)
    print(f"  {len(all_missed_value)} players across {len(per_date)} dates were on the Highest Value")
    print(f"  leaderboard but absent from our predictions entirely.\n")
    print(f"  {'Player':<25} {'Date':<12} {'RS':>5} {'Boost':>6} {'Value':>7} {'Drafts':>7}")
    print(f"  {'─'*25} {'─'*12} {'─'*5} {'─'*6} {'─'*7} {'─'*7}")
    for m in all_missed_value[:25]:
        boost_str = f"+{m['card_boost']}x" if m['card_boost'] else "—"
        print(f"  {m['player']:<25} {m['date']:<12} {m['actual_rs']:>5.1f} {boost_str:>6} {m.get('total_value',''):>7} {m['drafts']:>7}")

    # ── Section 6: Card boost accuracy ──
    print(f"\n{'─' * 70}")
    print(f"  6. CARD BOOST ACCURACY (predicted vs actual)")
    print(f"{'─' * 70}")
    if all_boost_comparisons:
        boost_errors = [abs(b["boost_error"]) for b in all_boost_comparisons]
        boost_mae = round(sum(boost_errors) / len(boost_errors), 3)
        print(f"  Boost MAE:     {boost_mae} (across {len(boost_errors)} comparisons)")
        # Show worst boost misses
        all_boost_comparisons.sort(key=lambda x: abs(x["boost_error"]), reverse=True)
        print(f"\n  {'Player':<25} {'Date':<12} {'Pred':>6} {'Actual':>7} {'Error':>6}")
        print(f"  {'─'*25} {'─'*12} {'─'*6} {'─'*7} {'─'*6}")
        for b in all_boost_comparisons[:10]:
            print(f"  {b['player']:<25} {b['date']:<12} {b['pred_boost']:>+5.1f}x {b['actual_boost']:>+6.1f}x {b['boost_error']:>+6.1f}")
    else:
        print("  No card boost comparisons available.")

    # ── Section 7: Coverage ──
    print(f"\n{'─' * 70}")
    print(f"  7. COVERAGE RATE")
    print(f"{'─' * 70}")
    print(f"  {'Date':<12} {'Leaderboard':>12} {'In Preds':>10} {'Coverage':>10}")
    print(f"  {'─'*12} {'─'*12} {'─'*10} {'─'*10}")
    for d in per_date:
        cov = f"{100*d['matched']/max(d['actuals'],1):.0f}%"
        print(f"  {d['date']:<12} {d['actuals']:>12} {d['matched']:>10} {cov:>10}")
    overall_cov = f"{100*total_matched/max(total_actuals,1):.0f}%"
    print(f"  {'OVERALL':<12} {total_actuals:>12} {total_matched:>10} {overall_cov:>10}")

    # ── Recurring missed players ──
    print(f"\n{'─' * 70}")
    print(f"  8. RECURRING MISSED PLAYERS (appeared 3+ times on leaderboard, never predicted)")
    print(f"{'─' * 70}")
    from collections import Counter
    missed_names = Counter(_normalize_name(m["player"]) for m in all_missed_value)
    recurring = [(name, count) for name, count in missed_names.items() if count >= 3]
    recurring.sort(key=lambda x: x[1], reverse=True)
    if recurring:
        # Get original name and latest stats
        name_lookup = {_normalize_name(m["player"]): m["player"] for m in all_missed_value}
        for norm_name, count in recurring:
            orig = name_lookup.get(norm_name, norm_name)
            entries = [m for m in all_missed_value if _normalize_name(m["player"]) == norm_name]
            avg_rs = round(sum(_safe_float(e["actual_rs"]) for e in entries) / len(entries), 1)
            avg_val = round(sum(_safe_float(e.get("total_value", 0)) for e in entries) / len(entries), 1)
            dates_str = ", ".join(e["date"][-5:] for e in entries)
            print(f"  {orig:<25} {count}x  avg RS={avg_rs}  avg Value={avg_val}  ({dates_str})")
    else:
        print("  None found.")

    print(f"\n{'=' * 70}")
    print(f"  REPORT COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
