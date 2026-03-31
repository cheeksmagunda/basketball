#!/usr/bin/env python3
"""
Backtest the 3-tier cascade boost model against all historical data.
For each date, predict boost for every player using only data available BEFORE that date.
"""

import csv
import sys
import os
from collections import defaultdict
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.boost_model import (
    _normalize_name, _safe, get_tier_baseline, estimate_boost_from_api,
    _clamp_round, _TIER1_MAX_GAP_DAYS
)

def load_all_data():
    """Load all data from top_performers.csv + actuals, organized by date and player."""
    all_entries = {}  # (norm_name, date) -> {boost, rs, drafts}

    # Load top_performers.csv
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
                key = (name, date)
                if key not in all_entries or boost > all_entries[key]["boost"]:
                    all_entries[key] = {"boost": boost, "rs": rs, "drafts": drafts}

    # Load actuals
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
                    key = (name, date)
                    if key not in all_entries or boost > all_entries[key]["boost"]:
                        all_entries[key] = {"boost": boost, "rs": rs, "drafts": drafts}

    # Organize by player
    player_history = defaultdict(list)
    for (name, date), data in all_entries.items():
        player_history[name].append({"date": date, **data})

    # Sort by date
    for name in player_history:
        player_history[name].sort(key=lambda e: e["date"])

    # Detect all-zero dates
    date_boosts = defaultdict(list)
    for (name, date), data in all_entries.items():
        date_boosts[date].append(data["boost"])
    all_zero_dates = {d for d, boosts in date_boosts.items() if boosts and all(b == 0 for b in boosts)}

    return player_history, all_entries, all_zero_dates


def simulate_tier1(prev_entries, gap_days):
    """Simulate Tier 1 prediction using entries available before target date."""
    prev = prev_entries[-1]
    base = prev["boost"]
    prev_rs = prev["rs"]
    prev_drafts = prev["drafts"]

    all_boosts = [e["boost"] for e in prev_entries]
    all_rs = [e["rs"] for e in prev_entries if e["rs"] > 0]
    hist_boost_mean = sum(all_boosts) / len(all_boosts) if all_boosts else base
    hist_rs_mean = sum(all_rs) / len(all_rs) if all_rs else prev_rs
    tier_baseline = get_tier_baseline(hist_rs_mean)

    # Factor 1: RS performance decay
    if prev_rs >= 7.0:
        base -= 0.1
    elif prev_rs >= 5.0:
        base -= 0.05
    elif prev_rs < 2.0 and prev_rs > 0:
        base += 0.1

    # Factor 2: Draft popularity
    if prev_drafts > 1000:
        base -= 0.05
    elif prev_drafts < 10 and prev_drafts > 0:
        base += 0.05

    # Factor 3: Mean reversion
    reversion = (tier_baseline - base) * 0.05
    base += reversion

    # Factor 4: Trend continuation
    if len(prev_entries) >= 2:
        prev2 = prev_entries[-2]
        if prev2["boost"] > 0:
            trend = prev["boost"] - prev2["boost"]
            trend_adj = trend * 0.15
            base += trend_adj

    # Factor 5: Gap days
    if gap_days > 5:
        gap_weight = min(gap_days / 30, 0.3)
        base = base * (1 - gap_weight) + tier_baseline * gap_weight

    # Factor 6: Minutes change signal
    if len(prev_entries) >= 3:
        recent_drafts_avg = sum(e["drafts"] for e in prev_entries[-3:]) / 3
        if recent_drafts_avg > 500 and prev_drafts > recent_drafts_avg * 1.3:
            base -= 0.05

    # Boundary persistence
    if prev["boost"] >= 3.0 and base >= 2.7:
        base = max(base, 2.9)
    if prev["boost"] <= 0.1 and hist_rs_mean >= 5.5:
        base = min(base, 0.2)

    return _clamp_round(base, 0.0, 3.0)


def backtest():
    print("=" * 80)
    print("BOOST MODEL BACKTEST — All Historical Dates")
    print("=" * 80)

    player_history, all_entries, all_zero_dates = load_all_data()

    dates = sorted(set(d for (_, d) in all_entries.keys()))
    dates = [d for d in dates if d not in all_zero_dates]

    print(f"Total dates: {len(dates)}")
    print(f"All-zero dates skipped: {len(all_zero_dates)}")

    # Track errors by tier
    tier1_errors = []
    tier2_errors = []
    tier3_errors = []
    all_errors = []

    # Track error patterns
    error_by_rs_bucket = defaultdict(list)
    error_by_boost_bucket = defaultdict(list)
    error_by_drafts_bucket = defaultdict(list)

    # Track biggest misses
    biggest_misses = []

    # For each date, predict each player's boost using only prior data
    for target_date in dates:
        players_on_date = [(name, date) for (name, date) in all_entries.keys() if date == target_date]

        for (name, _) in players_on_date:
            actual = all_entries[(name, target_date)]
            actual_boost = actual["boost"]

            # Get history BEFORE this date
            history = player_history.get(name, [])
            prior = [e for e in history if e["date"] < target_date and e["date"] not in all_zero_dates]

            if not prior:
                # Tier 3: cold start — approximate season PPG from actual RS
                # In production, ESPN provides real season_ppg. Here we use a proxy.
                # RS correlates ~0.5 with PPG; rough mapping from historical data.
                _approx_ppg = actual["rs"] * 4.0  # RS 5.0 → ~20 PPG proxy
                predicted = estimate_boost_from_api(season_ppg=_approx_ppg)
                predicted = _clamp_round(predicted, 0.0, 3.0)
                tier = 3
                tier3_errors.append(abs(predicted - actual_boost))
            else:
                from datetime import datetime
                last_date = datetime.strptime(prior[-1]["date"], "%Y-%m-%d").date()
                curr_date = datetime.strptime(target_date, "%Y-%m-%d").date()
                gap_days = (curr_date - last_date).days

                if gap_days <= _TIER1_MAX_GAP_DAYS:
                    predicted = simulate_tier1(prior, gap_days)
                    tier = 1
                    tier1_errors.append(abs(predicted - actual_boost))
                else:
                    # Tier 2: stale
                    all_boosts = [e["boost"] for e in prior]
                    hist_boost_mean = sum(all_boosts) / len(all_boosts) if all_boosts else 1.5
                    staleness = min((gap_days - _TIER1_MAX_GAP_DAYS) / 30, 1.0)
                    # Without API data, use hist_boost_mean as both signals
                    predicted_raw = hist_boost_mean
                    predicted = _clamp_round(predicted_raw, 0.0, 3.0)
                    tier = 2
                    tier2_errors.append(abs(predicted - actual_boost))

            error = abs(predicted - actual_boost)
            all_errors.append(error)

            # Track by categories
            rs = actual["rs"]
            if rs >= 5:
                error_by_rs_bucket["high_rs (≥5)"].append(error)
            elif rs >= 3:
                error_by_rs_bucket["mid_rs (3-5)"].append(error)
            else:
                error_by_rs_bucket["low_rs (<3)"].append(error)

            if actual_boost >= 2.5:
                error_by_boost_bucket["high_boost (≥2.5)"].append(error)
            elif actual_boost >= 1.0:
                error_by_boost_bucket["mid_boost (1-2.5)"].append(error)
            else:
                error_by_boost_bucket["low_boost (<1)"].append(error)

            drafts = actual["drafts"]
            if drafts >= 1000:
                error_by_drafts_bucket["popular (≥1000)"].append(error)
            elif drafts >= 100:
                error_by_drafts_bucket["moderate (100-1000)"].append(error)
            else:
                error_by_drafts_bucket["low_pop (<100)"].append(error)

            if error >= 0.5:
                biggest_misses.append({
                    "date": target_date,
                    "name": name,
                    "predicted": predicted,
                    "actual": actual_boost,
                    "error": error,
                    "tier": tier,
                    "rs": rs,
                    "drafts": drafts,
                })

    # Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    n = len(all_errors)
    print(f"\nTotal predictions: {n}")
    print(f"Overall MAE: {sum(all_errors)/n:.3f}")

    all_errors.sort()
    print(f"  p50: {all_errors[int(n*0.5)]:.2f}")
    print(f"  p75: {all_errors[int(n*0.75)]:.2f}")
    print(f"  p90: {all_errors[int(n*0.9)]:.2f}")
    print(f"  p95: {all_errors[int(n*0.95)]:.2f}")

    within_01 = sum(1 for e in all_errors if e <= 0.1)
    within_03 = sum(1 for e in all_errors if e <= 0.3)
    within_05 = sum(1 for e in all_errors if e <= 0.5)
    print(f"\n  Within 0.1: {within_01}/{n} ({within_01/n*100:.1f}%)")
    print(f"  Within 0.3: {within_03}/{n} ({within_03/n*100:.1f}%)")
    print(f"  Within 0.5: {within_05}/{n} ({within_05/n*100:.1f}%)")

    print(f"\nBy Tier:")
    for label, errors in [("Tier 1 (returning)", tier1_errors),
                          ("Tier 2 (stale)", tier2_errors),
                          ("Tier 3 (cold start)", tier3_errors)]:
        if errors:
            print(f"  {label}: MAE={sum(errors)/len(errors):.3f} (n={len(errors)})")

    print(f"\nBy RS Bucket:")
    for bucket in sorted(error_by_rs_bucket.keys()):
        errors = error_by_rs_bucket[bucket]
        print(f"  {bucket}: MAE={sum(errors)/len(errors):.3f} (n={len(errors)})")

    print(f"\nBy Boost Bucket:")
    for bucket in sorted(error_by_boost_bucket.keys()):
        errors = error_by_boost_bucket[bucket]
        print(f"  {bucket}: MAE={sum(errors)/len(errors):.3f} (n={len(errors)})")

    print(f"\nBy Draft Popularity:")
    for bucket in sorted(error_by_drafts_bucket.keys()):
        errors = error_by_drafts_bucket[bucket]
        print(f"  {bucket}: MAE={sum(errors)/len(errors):.3f} (n={len(errors)})")

    # Biggest misses
    biggest_misses.sort(key=lambda x: x["error"], reverse=True)
    print(f"\n{'='*80}")
    print("TOP 30 BIGGEST MISSES")
    print(f"{'='*80}")
    print(f"{'Date':>12} {'Player':<25} {'Pred':>5} {'Actual':>7} {'Error':>6} {'Tier':>5} {'RS':>5} {'Drafts':>7}")
    for m in biggest_misses[:30]:
        print(f"  {m['date']:>10} {m['name']:<25} {m['predicted']:>5.1f} {m['actual']:>7.1f} {m['error']:>6.2f} {m['tier']:>5} {m['rs']:>5.1f} {m['drafts']:>7.0f}")

    # Directional analysis: over-predict vs under-predict
    print(f"\n{'='*80}")
    print("DIRECTIONAL BIAS")
    print(f"{'='*80}")

    # Need signed errors
    signed_errors = []
    for target_date in dates:
        players_on_date = [(name, date) for (name, date) in all_entries.keys() if date == target_date]
        for (name, _) in players_on_date:
            actual = all_entries[(name, target_date)]
            history = player_history.get(name, [])
            prior = [e for e in history if e["date"] < target_date and e["date"] not in all_zero_dates]

            if not prior:
                predicted = 3.0
            else:
                from datetime import datetime
                last_date = datetime.strptime(prior[-1]["date"], "%Y-%m-%d").date()
                curr_date = datetime.strptime(target_date, "%Y-%m-%d").date()
                gap_days = (curr_date - last_date).days

                if gap_days <= _TIER1_MAX_GAP_DAYS:
                    predicted = simulate_tier1(prior, gap_days)
                else:
                    all_boosts = [e["boost"] for e in prior]
                    hist_boost_mean = sum(all_boosts) / len(all_boosts) if all_boosts else 1.5
                    predicted = _clamp_round(hist_boost_mean, 0.0, 3.0)

            signed_error = predicted - actual["boost"]
            signed_errors.append((signed_error, actual["boost"], name, target_date))

    over_pred = sum(1 for e, _, _, _ in signed_errors if e > 0)
    under_pred = sum(1 for e, _, _, _ in signed_errors if e < 0)
    exact = sum(1 for e, _, _, _ in signed_errors if e == 0)
    avg_signed = sum(e for e, _, _, _ in signed_errors) / len(signed_errors)

    print(f"  Over-predict:  {over_pred}/{len(signed_errors)} ({over_pred/len(signed_errors)*100:.1f}%)")
    print(f"  Under-predict: {under_pred}/{len(signed_errors)} ({under_pred/len(signed_errors)*100:.1f}%)")
    print(f"  Exact:         {exact}/{len(signed_errors)} ({exact/len(signed_errors)*100:.1f}%)")
    print(f"  Mean signed error: {avg_signed:+.3f}")

    # Bias by actual boost level
    print(f"\n  Directional Bias by Actual Boost Level:")
    for lo, hi, label in [(0, 0.5, "0-0.5"), (0.5, 1.0, "0.5-1.0"), (1.0, 2.0, "1.0-2.0"),
                          (2.0, 3.0, "2.0-3.0"), (3.0, 3.5, "3.0+")]:
        bucket = [(e, b) for e, b, _, _ in signed_errors if lo <= b < hi]
        if bucket:
            avg_e = sum(e for e, _ in bucket) / len(bucket)
            print(f"    Actual {label}: avg prediction bias {avg_e:+.3f} (n={len(bucket)})")

    # Analyze snap-to-3.0 behavior
    print(f"\n{'='*80}")
    print("SNAP-TO-3.0 ANALYSIS")
    print(f"{'='*80}")
    snap_errors = [(e, b, n, d) for e, b, n, d in signed_errors if b >= 2.5]
    if snap_errors:
        pred_30_when_high = sum(1 for e, b, _, _ in snap_errors if e >= -0.1)
        print(f"  High-boost actuals (≥2.5): {len(snap_errors)}")
        print(f"  Predicted ~3.0: {pred_30_when_high} ({pred_30_when_high/len(snap_errors)*100:.1f}%)")

        # How many actual 3.0s did we predict as 3.0?
        actual_30 = [(e, b, n, d) for e, b, n, d in signed_errors if b == 3.0]
        if actual_30:
            pred_30_correct = sum(1 for e, _, _, _ in actual_30 if abs(e) <= 0.1)
            print(f"  Actual 3.0 correctly predicted: {pred_30_correct}/{len(actual_30)} ({pred_30_correct/len(actual_30)*100:.1f}%)")

if __name__ == "__main__":
    backtest()
