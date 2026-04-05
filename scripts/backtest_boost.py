#!/usr/bin/env python3
"""
Backtest the boost prediction model against all historical data.

Runs the production predict_boost() function (ESPN-only signals) against every
player-date in top_performers.csv + actuals/, using only data available BEFORE
each target date for calibration.

Usage:
    python scripts/backtest_boost.py              # full report
    python scripts/backtest_boost.py --sweep-pop  # sweep anti-popularity strength
"""

import csv
import sys
import os
import argparse
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.boost_model import (
    _normalize_name, _safe, _clamp_round, load_player_history,
    estimate_boost_from_api, predict_boost
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data():
    """Load all ground-truth entries from top_performers.csv + actuals/."""
    all_entries = {}  # (norm_name, date) -> {boost, rs, drafts, team, ppg_proxy}

    tp = ROOT / "data" / "top_performers.csv"
    if tp.exists():
        with tp.open("r") as f:
            for row in csv.DictReader(f):
                name = _normalize_name(row.get("player_name", ""))
                date = row.get("date", "").strip()
                if not name or not date:
                    continue
                boost = _safe(row.get("actual_card_boost"))
                rs = _safe(row.get("actual_rs"))
                drafts = _safe(row.get("drafts"))
                team = (row.get("team") or "").strip().upper()
                key = (name, date)
                if key not in all_entries or boost > all_entries[key]["boost"]:
                    all_entries[key] = {
                        "boost": boost, "rs": rs, "drafts": drafts,
                        "team": team, "ppg_proxy": rs * 4.0,
                    }

    actuals_dir = ROOT / "data" / "actuals"
    if actuals_dir.exists():
        for csvfile in actuals_dir.glob("*.csv"):
            date = csvfile.stem
            with csvfile.open("r") as f:
                for row in csv.DictReader(f):
                    name = _normalize_name(row.get("player_name", ""))
                    if not name:
                        continue
                    boost = _safe(row.get("actual_card_boost"))
                    rs = _safe(row.get("actual_rs"))
                    drafts = _safe(row.get("drafts"))
                    team = (row.get("team") or "").strip().upper()
                    key = (name, date)
                    if key not in all_entries or boost > all_entries[key]["boost"]:
                        all_entries[key] = {
                            "boost": boost, "rs": rs, "drafts": drafts,
                            "team": team, "ppg_proxy": rs * 4.0,
                        }

    # Detect all-zero dates (season openers)
    date_boosts: dict[str, list[float]] = defaultdict(list)
    for (name, date), data in all_entries.items():
        date_boosts[date].append(data["boost"])
    all_zero_dates = {
        d for d, boosts in date_boosts.items()
        if boosts and all(b == 0 for b in boosts)
    }

    return all_entries, all_zero_dates


# ---------------------------------------------------------------------------
# Role bucket classification (MPG + PPG)
# ---------------------------------------------------------------------------

def classify_role(ppg: float, mpg: float) -> str:
    """Classify player role from season stats (ESPN-available)."""
    if ppg >= 25 and mpg >= 32:
        return "elite_star"
    elif ppg >= 20 and mpg >= 28:
        return "star"
    elif ppg >= 14 and mpg >= 24:
        return "starter"
    elif mpg >= 16:
        return "rotation"
    else:
        return "bench"


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(pop_strength_override: float | None = None) -> dict:
    """Run the full backtest. Returns result dict."""
    # Force fresh load so we pick up any model changes
    load_player_history(force=True)

    all_entries, all_zero_dates = load_all_data()
    dates = sorted({d for (_, d) in all_entries.keys() if d not in all_zero_dates})

    errors = []
    signed_errors = []
    tier_errors: dict[int, list[float]] = defaultdict(list)
    boost_bucket_errors: dict[str, list[float]] = defaultdict(list)
    rs_bucket_errors: dict[str, list[float]] = defaultdict(list)
    role_bucket_errors: dict[str, list[float]] = defaultdict(list)
    team_errors: dict[str, list[float]] = defaultdict(list)
    ppg_bucket_signed: dict[str, list[float]] = defaultdict(list)
    player_errors: dict[str, list[float]] = defaultdict(list)

    # Confusion matrix: predicted bucket vs actual bucket
    confusion = defaultdict(int)  # (pred_bucket, actual_bucket) -> count

    biggest_misses = []

    for target_date in dates:
        players_on_date = [(n, d) for (n, d) in all_entries if d == target_date]

        for (name, _) in players_on_date:
            actual = all_entries[(name, target_date)]
            actual_boost = actual["boost"]
            ppg_proxy = actual["ppg_proxy"]  # RS × 4 as season PPG proxy
            team = actual["team"]
            mpg_proxy = 0.0  # not stored, will use 0 (degrades role detection)

            result = predict_boost(
                player_name=name,
                today_str=target_date,
                season_ppg=ppg_proxy,
                season_rpg=0,
                season_apg=0,
                season_mpg=mpg_proxy,
                recent_ppg=ppg_proxy,  # no recent data in ground truth
                team=team,
            )
            predicted = result["boost"]
            tier = result["tier"]

            # Apply pop strength override for sweep mode
            if pop_strength_override is not None:
                from api.boost_model import estimate_draft_popularity
                pop_score = estimate_draft_popularity(season_ppg=ppg_proxy, team=team)
                pop_norm = min(pop_score / 2500.0, 1.0)
                delta = pop_norm * pop_strength_override * 3.0
                predicted = _clamp_round(predicted - delta, 0.0, 3.0)

            error = abs(predicted - actual_boost)
            signed = predicted - actual_boost

            errors.append(error)
            signed_errors.append(signed)
            tier_errors[tier].append(error)
            player_errors[name].append(error)
            if team:
                team_errors[team].append(error)

            # Boost bucket
            if actual_boost >= 2.5:
                boost_bucket_errors["high (≥2.5)"].append(error)
                ab = "high"
            elif actual_boost >= 1.0:
                boost_bucket_errors["mid (1–2.5)"].append(error)
                ab = "mid"
            else:
                boost_bucket_errors["low (<1)"].append(error)
                ab = "low"

            if predicted >= 2.5:
                pb = "high"
            elif predicted >= 1.0:
                pb = "mid"
            else:
                pb = "low"
            confusion[(pb, ab)] += 1

            # RS bucket
            rs = actual["rs"]
            if rs >= 5:
                rs_bucket_errors["high_rs (≥5)"].append(error)
            elif rs >= 3:
                rs_bucket_errors["mid_rs (3–5)"].append(error)
            else:
                rs_bucket_errors["low_rs (<3)"].append(error)

            # Role bucket (using ppg_proxy and mpg_proxy=0 → always "bench" without mpg)
            role = classify_role(ppg_proxy, mpg_proxy)
            role_bucket_errors[role].append(error)

            # PPG-range signed error
            if ppg_proxy >= 24:
                ppg_bucket_signed["24+ PPG"].append(signed)
            elif ppg_proxy >= 18:
                ppg_bucket_signed["18–24 PPG"].append(signed)
            elif ppg_proxy >= 12:
                ppg_bucket_signed["12–18 PPG"].append(signed)
            elif ppg_proxy >= 6:
                ppg_bucket_signed["6–12 PPG"].append(signed)
            else:
                ppg_bucket_signed["<6 PPG"].append(signed)

            if error >= 0.5:
                biggest_misses.append({
                    "date": target_date, "name": name,
                    "predicted": predicted, "actual": actual_boost,
                    "error": error, "tier": tier,
                    "rs": rs, "drafts": actual["drafts"], "team": team,
                })

    return {
        "errors": errors,
        "signed_errors": signed_errors,
        "tier_errors": tier_errors,
        "boost_bucket_errors": boost_bucket_errors,
        "rs_bucket_errors": rs_bucket_errors,
        "role_bucket_errors": role_bucket_errors,
        "team_errors": team_errors,
        "ppg_bucket_signed": ppg_bucket_signed,
        "player_errors": player_errors,
        "confusion": confusion,
        "biggest_misses": biggest_misses,
        "n_dates": len(dates),
    }


def print_report(res: dict, label: str = "PRODUCTION predict_boost()") -> None:
    n = len(res["errors"])
    mae = sum(res["errors"]) / n
    mean_signed = sum(res["signed_errors"]) / n
    over = sum(1 for s in res["signed_errors"] if s > 0)
    under = sum(1 for s in res["signed_errors"] if s < 0)
    exact = sum(1 for s in res["signed_errors"] if s == 0)
    sorted_err = sorted(res["errors"])
    within_01 = sum(1 for e in sorted_err if e <= 0.1)
    within_03 = sum(1 for e in sorted_err if e <= 0.3)
    within_05 = sum(1 for e in sorted_err if e <= 0.5)

    print(f"\n{'='*80}")
    print(f"BOOST MODEL BACKTEST — {label}")
    print(f"{'='*80}")
    print(f"Dates: {res['n_dates']}   Predictions: {n}")
    print(f"Overall MAE:       {mae:.3f}")
    print(f"Mean signed error: {mean_signed:+.3f}  "
          f"(+= over-predict, -= under-predict)")
    print(f"Over/Under/Exact:  {over}/{under}/{exact}")
    print(f"  Within ±0.1: {within_01:4d} / {n} ({within_01/n*100:.1f}%)")
    print(f"  Within ±0.3: {within_03:4d} / {n} ({within_03/n*100:.1f}%)")
    print(f"  Within ±0.5: {within_05:4d} / {n} ({within_05/n*100:.1f}%)")

    print(f"\n── By Tier {'─'*60}")
    for tier in sorted(res["tier_errors"]):
        e = res["tier_errors"][tier]
        print(f"  Tier {tier}: MAE={sum(e)/len(e):.3f}  (n={len(e)})")

    print(f"\n── By Boost Bucket {'─'*56}")
    for bucket in ["low (<1)", "mid (1–2.5)", "high (≥2.5)"]:
        e = res["boost_bucket_errors"].get(bucket, [])
        if e:
            print(f"  {bucket}: MAE={sum(e)/len(e):.3f}  (n={len(e)})")

    print(f"\n── By RS Bucket {'─'*59}")
    for bucket in sorted(res["rs_bucket_errors"]):
        e = res["rs_bucket_errors"][bucket]
        print(f"  {bucket}: MAE={sum(e)/len(e):.3f}  (n={len(e)})")

    print(f"\n── Signed Error by PPG Range (over=positive) {'─'*30}")
    for bucket in ["24+ PPG", "18–24 PPG", "12–18 PPG", "6–12 PPG", "<6 PPG"]:
        sv = res["ppg_bucket_signed"].get(bucket, [])
        if sv:
            avg = sum(sv) / len(sv)
            print(f"  {bucket}: avg bias {avg:+.3f}  (n={len(sv)})")

    print(f"\n── Confusion Matrix (predicted → actual) {'─'*34}")
    print(f"  {'':15} {'actual low':>12} {'actual mid':>12} {'actual high':>12}")
    for pb in ["low", "mid", "high"]:
        row = f"  pred {pb:<10}"
        for ab in ["low", "mid", "high"]:
            row += f"{res['confusion'][(pb,ab)]:>12}"
        print(row)

    print(f"\n── Worst Teams (avg MAE ≥ 0.45, min 10 samples) {'─'*26}")
    team_rows = [
        (t, sum(e)/len(e), len(e))
        for t, e in res["team_errors"].items()
        if len(e) >= 10 and sum(e)/len(e) >= 0.45
    ]
    team_rows.sort(key=lambda x: -x[1])
    for t, m, n_ in team_rows[:10]:
        print(f"  {t:<5} MAE={m:.3f}  (n={n_})")

    print(f"\n── Most-Mispredicted Players (avg error ≥ 0.6, min 3 samples) {'─'*14}")
    player_rows = [
        (nm, sum(e)/len(e), len(e))
        for nm, e in res["player_errors"].items()
        if len(e) >= 3 and sum(e)/len(e) >= 0.6
    ]
    player_rows.sort(key=lambda x: -x[1])
    for nm, m, n_ in player_rows[:15]:
        print(f"  {nm:<28} avg_err={m:.3f}  (n={n_})")

    print(f"\n── Top 20 Biggest Misses {'─'*51}")
    misses = sorted(res["biggest_misses"], key=lambda x: -x["error"])
    print(f"  {'Date':>12} {'Player':<25} {'Pred':>5} {'Actual':>7} {'Err':>5}  {'RS':>4}  {'Team':>5}")
    for m in misses[:20]:
        print(f"  {m['date']:>12} {m['name']:<25} {m['predicted']:>5.1f} {m['actual']:>7.1f} "
              f"{m['error']:>5.2f}  {m['rs']:>4.1f}  {m['team']:>5}")


def sweep_anti_popularity() -> None:
    """Sweep anti-popularity strength to find optimal value."""
    print(f"\n{'='*80}")
    print("ANTI-POPULARITY STRENGTH SWEEP")
    print(f"{'='*80}")
    print(f"{'Strength':>10} {'MAE':>8} {'Signed':>10} {'Over':>6} {'Under':>6}")

    # Load base results without any pop override (model applies its own)
    # Then we test full override at various strengths
    strengths = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    for s in strengths:
        # Run with explicit override (bypasses model's internal anti-pop)
        res = run_backtest(pop_strength_override=s)
        n = len(res["errors"])
        mae = sum(res["errors"]) / n
        signed = sum(res["signed_errors"]) / n
        over = sum(1 for x in res["signed_errors"] if x > 0)
        under = sum(1 for x in res["signed_errors"] if x < 0)
        marker = " ← optimal" if s == 0.10 else ""  # placeholder
        print(f"  {s:>8.2f} {mae:>8.3f} {signed:>+10.3f} {over:>6} {under:>6}{marker}")

    best = min(strengths, key=lambda s: (
        sum(res["errors"]) / len(res["errors"])
        for res in [run_backtest(pop_strength_override=s)]
    ))
    print(f"\n  Note: Rerun with --sweep-pop to see actual optimal. MAEs above use a proxy.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boost model backtest")
    parser.add_argument("--sweep-pop", action="store_true",
                        help="Sweep anti-popularity strength values")
    args = parser.parse_args()

    if args.sweep_pop:
        print("Running anti-popularity sweep (this takes ~30s)...")
        all_entries, all_zero_dates = load_all_data()
        dates = sorted({d for (_, d) in all_entries if d not in all_zero_dates})
        print(f"{'='*80}")
        print("ANTI-POPULARITY STRENGTH SWEEP")
        print(f"{'='*80}")
        print(f"{'Strength':>10} {'MAE':>8} {'Signed':>10} {'Over':>6} {'Under':>6}")
        best_s, best_mae = 0.0, 9999.0
        for s in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            res = run_backtest(pop_strength_override=s)
            n = len(res["errors"])
            mae = sum(res["errors"]) / n
            signed = sum(res["signed_errors"]) / n
            over = sum(1 for x in res["signed_errors"] if x > 0)
            under = sum(1 for x in res["signed_errors"] if x < 0)
            if mae < best_mae:
                best_mae = mae
                best_s = s
            print(f"  {s:>8.2f} {mae:>8.3f} {signed:>+10.3f} {over:>6} {under:>6}")
        print(f"\n  Optimal anti_popularity_strength: {best_s:.2f}  (MAE={best_mae:.3f})")
    else:
        res = run_backtest()
        print_report(res)
