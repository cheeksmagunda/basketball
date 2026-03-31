#!/usr/bin/env python3
"""
Comprehensive historical data analysis for draft optimization.
Analyzes every date in top_performers.csv + actuals to find winning patterns.
"""

import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def load_top_performers():
    """Load the mega file with all historical data."""
    rows = []
    path = DATA / "top_performers.csv"
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["actual_rs"] = float(row.get("actual_rs", 0) or 0)
                row["actual_card_boost"] = float(row.get("actual_card_boost", 0) or 0)
                row["drafts"] = int(float(row.get("drafts", 0) or 0))
                row["total_value"] = float(row.get("total_value", 0) or 0)
                rows.append(row)
            except (ValueError, TypeError):
                continue
    return rows

def load_actuals():
    """Load all per-date actuals files."""
    actuals_dir = DATA / "actuals"
    all_rows = []
    if not actuals_dir.exists():
        return all_rows
    for f in sorted(actuals_dir.glob("*.csv")):
        date = f.stem
        with open(f, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    row["date"] = date
                    row["actual_rs"] = float(row.get("actual_rs", 0) or 0)
                    row["actual_card_boost"] = float(row.get("actual_card_boost", 0) or 0)
                    row["drafts"] = int(float(row.get("drafts", 0) or 0))
                    row["total_value"] = float(row.get("total_value", 0) or 0)
                    all_rows.append(row)
                except (ValueError, TypeError):
                    continue
    return all_rows

def load_predictions():
    """Load all prediction CSVs."""
    pred_dir = DATA / "predictions"
    all_rows = []
    if not pred_dir.exists():
        return all_rows
    for f in sorted(pred_dir.glob("*.csv")):
        date = f.stem
        with open(f, "r") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                row["date"] = date
                all_rows.append(row)
    return all_rows

def analyze():
    print("=" * 80)
    print("COMPREHENSIVE HISTORICAL DATA ANALYSIS")
    print("=" * 80)

    # Load data
    tp_rows = load_top_performers()
    actuals_rows = load_actuals()

    print(f"\nTop Performers rows: {len(tp_rows)}")
    print(f"Actuals rows: {len(actuals_rows)}")

    # Organize by date
    by_date = defaultdict(list)
    seen = set()

    # Prefer top_performers (primary)
    for row in tp_rows:
        key = (row["date"], row["player_name"])
        if key not in seen:
            seen.add(key)
            by_date[row["date"]].append(row)

    # Fill from actuals
    for row in actuals_rows:
        key = (row["date"], row["player_name"])
        if key not in seen:
            seen.add(key)
            by_date[row["date"]].append(row)

    dates = sorted(by_date.keys())
    print(f"Unique dates: {len(dates)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # =========================================================================
    # ANALYSIS 1: What RS ranges do top performers have?
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 1: RS Distribution of Top Performers")
    print("=" * 80)

    all_rs = [r["actual_rs"] for r in tp_rows if r["actual_rs"] > 0]
    all_rs.sort()

    if all_rs:
        n = len(all_rs)
        print(f"  Count: {n}")
        print(f"  Min:   {min(all_rs):.1f}")
        print(f"  p10:   {all_rs[int(n*0.1)]:.1f}")
        print(f"  p25:   {all_rs[int(n*0.25)]:.1f}")
        print(f"  p50:   {all_rs[int(n*0.5)]:.1f}")
        print(f"  p75:   {all_rs[int(n*0.75)]:.1f}")
        print(f"  p90:   {all_rs[int(n*0.9)]:.1f}")
        print(f"  Max:   {max(all_rs):.1f}")
        print(f"  Mean:  {sum(all_rs)/n:.2f}")

    # RS buckets
    print("\n  RS Bucket Distribution:")
    buckets = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 15)]
    for lo, hi in buckets:
        count = sum(1 for r in all_rs if lo <= r < hi)
        pct = count / len(all_rs) * 100
        print(f"    RS {lo:.0f}-{hi:.0f}: {count} ({pct:.1f}%)")

    # =========================================================================
    # ANALYSIS 2: Card Boost Distribution
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Card Boost Distribution of Top Performers")
    print("=" * 80)

    all_boost = [r["actual_card_boost"] for r in tp_rows]
    all_boost.sort()

    if all_boost:
        n = len(all_boost)
        print(f"  Count: {n}")
        print(f"  Min:   {min(all_boost):.1f}")
        print(f"  p25:   {all_boost[int(n*0.25)]:.1f}")
        print(f"  p50:   {all_boost[int(n*0.5)]:.1f}")
        print(f"  p75:   {all_boost[int(n*0.75)]:.1f}")
        print(f"  Max:   {max(all_boost):.1f}")
        print(f"  Mean:  {sum(all_boost)/n:.2f}")

    boost_buckets = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5)]
    print("\n  Boost Bucket Distribution:")
    for lo, hi in boost_buckets:
        count = sum(1 for b in all_boost if lo <= b < hi)
        pct = count / len(all_boost) * 100
        print(f"    Boost {lo:.1f}-{hi:.1f}: {count} ({pct:.1f}%)")

    # =========================================================================
    # ANALYSIS 3: Total Value Distribution
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Total Value Distribution")
    print("=" * 80)

    all_tv = [r["total_value"] for r in tp_rows if r["total_value"] > 0]
    all_tv.sort()

    if all_tv:
        n = len(all_tv)
        print(f"  Count: {n}")
        print(f"  Min:   {min(all_tv):.1f}")
        print(f"  p25:   {all_tv[int(n*0.25)]:.1f}")
        print(f"  p50:   {all_tv[int(n*0.5)]:.1f}")
        print(f"  p75:   {all_tv[int(n*0.75)]:.1f}")
        print(f"  p90:   {all_tv[int(n*0.9)]:.1f}")
        print(f"  Max:   {max(all_tv):.1f}")
        print(f"  Mean:  {sum(all_tv)/n:.2f}")

    # =========================================================================
    # ANALYSIS 4: Drafts vs Boost vs Value (Anti-popularity)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 4: Draft Popularity vs Boost & Value")
    print("=" * 80)

    draft_buckets = [
        ("< 50 drafts (hidden gems)", 0, 50),
        ("50-200 drafts (low pop)", 50, 200),
        ("200-500 drafts (moderate)", 200, 500),
        ("500-1000 drafts (popular)", 500, 1000),
        ("1000-3000 drafts (very popular)", 1000, 3000),
        ("3000+ drafts (superstars)", 3000, 999999),
    ]

    for label, lo, hi in draft_buckets:
        bucket = [r for r in tp_rows if lo <= r["drafts"] < hi and r["actual_rs"] > 0]
        if not bucket:
            continue
        avg_rs = sum(r["actual_rs"] for r in bucket) / len(bucket)
        avg_boost = sum(r["actual_card_boost"] for r in bucket) / len(bucket)
        avg_value = sum(r["total_value"] for r in bucket) / len(bucket)
        # EV = RS * (2.0 + boost)
        avg_ev = sum(r["actual_rs"] * (2.0 + r["actual_card_boost"]) for r in bucket) / len(bucket)
        print(f"\n  {label} (n={len(bucket)}):")
        print(f"    Avg RS:    {avg_rs:.2f}")
        print(f"    Avg Boost: {avg_boost:.2f}")
        print(f"    Avg Value: {avg_value:.2f}")
        print(f"    Avg EV:    {avg_ev:.2f}")

    # =========================================================================
    # ANALYSIS 5: Winning Lineup Profile (Top 5 by Total Value per date)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 5: Winning Lineup Profile (Top 5 by Total Value per date)")
    print("=" * 80)

    winning_rs = []
    winning_boost = []
    winning_value = []
    winning_drafts_list = []
    dates_with_enough = 0

    # Also track: could our model have won?
    model_would_pick = []  # Using EV formula

    for date in dates:
        players = by_date[date]
        # Filter to players with real data
        valid = [p for p in players if p["actual_rs"] > 0 and p["total_value"] > 0]
        if len(valid) < 5:
            continue
        dates_with_enough += 1

        # Sort by total_value (what actually wins)
        valid.sort(key=lambda x: x["total_value"], reverse=True)
        top5 = valid[:5]

        for p in top5:
            winning_rs.append(p["actual_rs"])
            winning_boost.append(p["actual_card_boost"])
            winning_value.append(p["total_value"])
            winning_drafts_list.append(p["drafts"])

    print(f"\n  Dates with 5+ players: {dates_with_enough}")
    print(f"  Top-5 player records: {len(winning_rs)}")

    if winning_rs:
        n = len(winning_rs)
        print(f"\n  Winning Players RS:")
        winning_rs.sort()
        print(f"    Min:  {min(winning_rs):.1f}")
        print(f"    p25:  {winning_rs[int(n*0.25)]:.1f}")
        print(f"    p50:  {winning_rs[int(n*0.5)]:.1f}")
        print(f"    p75:  {winning_rs[int(n*0.75)]:.1f}")
        print(f"    Max:  {max(winning_rs):.1f}")
        print(f"    Mean: {sum(winning_rs)/n:.2f}")

        print(f"\n  Winning Players Boost:")
        winning_boost.sort()
        print(f"    Min:  {min(winning_boost):.1f}")
        print(f"    p25:  {winning_boost[int(n*0.25)]:.1f}")
        print(f"    p50:  {winning_boost[int(n*0.5)]:.1f}")
        print(f"    p75:  {winning_boost[int(n*0.75)]:.1f}")
        print(f"    Max:  {max(winning_boost):.1f}")
        print(f"    Mean: {sum(winning_boost)/n:.2f}")

        print(f"\n  Winning Players Drafts:")
        winning_drafts_list.sort()
        print(f"    Min:  {min(winning_drafts_list)}")
        print(f"    p25:  {winning_drafts_list[int(n*0.25)]}")
        print(f"    p50:  {winning_drafts_list[int(n*0.5)]}")
        print(f"    p75:  {winning_drafts_list[int(n*0.75)]}")
        print(f"    Max:  {max(winning_drafts_list)}")

    # =========================================================================
    # ANALYSIS 6: EV Formula Validation
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 6: EV = RS * (2.0 + Boost) vs Actual Total Value")
    print("=" * 80)

    # Check how well the EV formula predicts total value
    ev_errors = []
    for r in tp_rows:
        if r["actual_rs"] > 0 and r["total_value"] > 0:
            predicted_ev = r["actual_rs"] * (2.0 + r["actual_card_boost"])
            actual_tv = r["total_value"]
            # Normalize: what slot multiplier was achieved?
            # total_value = RS * (slot_mult + boost)
            # If we assume slot_mult ~ 1.6 (average), then predicted ~ RS * (1.6 + boost)
            ev_errors.append((predicted_ev, actual_tv, r["actual_rs"], r["actual_card_boost"], r["player_name"], r["date"]))

    if ev_errors:
        # The EV with 2.0x (best slot) vs actual
        abs_errors = [abs(e[0] - e[1]) for e in ev_errors]
        # But total_value depends on slot assignment...
        # Let's compute what the IMPLIED slot multiplier was
        implied_slots = []
        for pred_ev, actual_tv, rs, boost, name, date in ev_errors:
            if rs > 0:
                # total_value = RS * (slot + boost)
                implied_slot = (actual_tv / rs) - boost
                if 0.5 <= implied_slot <= 2.5:
                    implied_slots.append(implied_slot)

        if implied_slots:
            implied_slots.sort()
            n = len(implied_slots)
            print(f"\n  Implied Slot Multipliers (n={n}):")
            print(f"    p25: {implied_slots[int(n*0.25)]:.2f}")
            print(f"    p50: {implied_slots[int(n*0.5)]:.2f}")
            print(f"    p75: {implied_slots[int(n*0.75)]:.2f}")
            print(f"    Mean: {sum(implied_slots)/n:.2f}")

    # =========================================================================
    # ANALYSIS 7: What makes a TOP VALUE player? RS vs Boost contribution
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 7: RS vs Boost Contribution to Top Value")
    print("=" * 80)

    # For each date, look at the #1 value player
    high_rs_wins = 0
    high_boost_wins = 0
    balanced_wins = 0
    total_dates = 0

    top1_profiles = []

    for date in dates:
        players = by_date[date]
        valid = [p for p in players if p["actual_rs"] > 0 and p["total_value"] > 0]
        if len(valid) < 5:
            continue
        total_dates += 1

        valid.sort(key=lambda x: x["total_value"], reverse=True)
        top1 = valid[0]

        rs = top1["actual_rs"]
        boost = top1["actual_card_boost"]

        top1_profiles.append({
            "date": date,
            "name": top1["player_name"],
            "rs": rs,
            "boost": boost,
            "value": top1["total_value"],
            "drafts": top1["drafts"]
        })

        # Classify
        if rs >= 5.0 and boost <= 1.0:
            high_rs_wins += 1
        elif rs <= 3.5 and boost >= 2.0:
            high_boost_wins += 1
        else:
            balanced_wins += 1

    print(f"\n  Dates analyzed: {total_dates}")
    print(f"  #1 Value Player Profile:")
    print(f"    High RS (≥5, boost≤1):      {high_rs_wins} ({high_rs_wins/total_dates*100:.1f}%)")
    print(f"    High Boost (RS≤3.5, boost≥2): {high_boost_wins} ({high_boost_wins/total_dates*100:.1f}%)")
    print(f"    Balanced (mix):              {balanced_wins} ({balanced_wins/total_dates*100:.1f}%)")

    if top1_profiles:
        avg_rs_top1 = sum(p["rs"] for p in top1_profiles) / len(top1_profiles)
        avg_boost_top1 = sum(p["boost"] for p in top1_profiles) / len(top1_profiles)
        avg_value_top1 = sum(p["value"] for p in top1_profiles) / len(top1_profiles)
        avg_drafts_top1 = sum(p["drafts"] for p in top1_profiles) / len(top1_profiles)
        print(f"\n  #1 Value Player Averages:")
        print(f"    Avg RS:     {avg_rs_top1:.2f}")
        print(f"    Avg Boost:  {avg_boost_top1:.2f}")
        print(f"    Avg Value:  {avg_value_top1:.2f}")
        print(f"    Avg Drafts: {avg_drafts_top1:.0f}")

    # =========================================================================
    # ANALYSIS 8: Optimal EV Selection Backtest
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 8: Optimal Draft Strategy Backtest (Hindsight)")
    print("=" * 80)

    # If we knew the actual RS and boost, what would EV-based selection give us?
    slot_mults = [2.0, 1.8, 1.6, 1.4, 1.2]

    hindsight_scores = []
    random_scores = []
    top_rs_scores = []
    top_boost_scores = []
    top_ev_scores = []

    for date in dates:
        players = by_date[date]
        valid = [p for p in players if p["actual_rs"] > 0 and p["total_value"] > 0]
        if len(valid) < 5:
            continue

        # Compute actual EV for each player
        for p in valid:
            p["ev"] = p["actual_rs"] * (2.0 + p["actual_card_boost"])

        # Strategy 1: Pick top 5 by EV, assign slots by RS
        by_ev = sorted(valid, key=lambda x: x["ev"], reverse=True)[:5]
        by_ev.sort(key=lambda x: x["actual_rs"], reverse=True)
        score_ev = sum(p["actual_rs"] * (slot_mults[i] + p["actual_card_boost"]) for i, p in enumerate(by_ev))
        top_ev_scores.append(score_ev)

        # Strategy 2: Pick top 5 by RS only, assign slots by RS
        by_rs = sorted(valid, key=lambda x: x["actual_rs"], reverse=True)[:5]
        by_rs.sort(key=lambda x: x["actual_rs"], reverse=True)
        score_rs = sum(p["actual_rs"] * (slot_mults[i] + p["actual_card_boost"]) for i, p in enumerate(by_rs))
        top_rs_scores.append(score_rs)

        # Strategy 3: Pick top 5 by boost only, assign slots by RS
        by_boost = sorted(valid, key=lambda x: x["actual_card_boost"], reverse=True)[:5]
        by_boost.sort(key=lambda x: x["actual_rs"], reverse=True)
        score_boost = sum(p["actual_rs"] * (slot_mults[i] + p["actual_card_boost"]) for i, p in enumerate(by_boost))
        top_boost_scores.append(score_boost)

        # Hindsight optimal: try to find the best 5
        by_tv = sorted(valid, key=lambda x: x["total_value"], reverse=True)[:5]
        hindsight_scores.append(sum(p["total_value"] for p in by_tv))

    if top_ev_scores:
        n = len(top_ev_scores)
        print(f"\n  Dates analyzed: {n}")
        print(f"\n  Strategy Comparison (avg total lineup score):")
        print(f"    Top-5 by EV (RS*(2+boost)):  {sum(top_ev_scores)/n:.1f}")
        print(f"    Top-5 by RS only:            {sum(top_rs_scores)/n:.1f}")
        print(f"    Top-5 by Boost only:         {sum(top_boost_scores)/n:.1f}")
        print(f"    Hindsight (actual top value): {sum(hindsight_scores)/n:.1f}")

        # How often does each strategy find the actual #1 value player?
        ev_got_top1 = 0
        rs_got_top1 = 0
        boost_got_top1 = 0

        for date in dates:
            players = by_date[date]
            valid = [p for p in players if p["actual_rs"] > 0 and p["total_value"] > 0]
            if len(valid) < 5:
                continue

            for p in valid:
                p["ev"] = p["actual_rs"] * (2.0 + p["actual_card_boost"])

            actual_top1 = max(valid, key=lambda x: x["total_value"])["player_name"]

            ev_top5_names = {p["player_name"] for p in sorted(valid, key=lambda x: x["ev"], reverse=True)[:5]}
            rs_top5_names = {p["player_name"] for p in sorted(valid, key=lambda x: x["actual_rs"], reverse=True)[:5]}
            boost_top5_names = {p["player_name"] for p in sorted(valid, key=lambda x: x["actual_card_boost"], reverse=True)[:5]}

            if actual_top1 in ev_top5_names:
                ev_got_top1 += 1
            if actual_top1 in rs_top5_names:
                rs_got_top1 += 1
            if actual_top1 in boost_top5_names:
                boost_got_top1 += 1

        print(f"\n  Hit rate (top-5 includes actual #1 value):")
        print(f"    EV strategy:    {ev_got_top1}/{n} ({ev_got_top1/n*100:.1f}%)")
        print(f"    RS-only:        {rs_got_top1}/{n} ({rs_got_top1/n*100:.1f}%)")
        print(f"    Boost-only:     {boost_got_top1}/{n} ({boost_got_top1/n*100:.1f}%)")

    # =========================================================================
    # ANALYSIS 9: Boost Predictability (Day-over-Day)
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 9: Boost Predictability (Day-over-Day Changes)")
    print("=" * 80)

    # Track each player's boost across dates
    player_boost_history = defaultdict(list)
    for row in tp_rows:
        if row["actual_card_boost"] >= 0:
            player_boost_history[row["player_name"]].append(
                (row["date"], row["actual_card_boost"], row["actual_rs"], row["drafts"])
            )

    # Sort each player's history by date
    for name in player_boost_history:
        player_boost_history[name].sort(key=lambda x: x[0])

    # Compute day-over-day changes
    boost_changes = []
    boost_change_by_rs = defaultdict(list)
    boost_change_by_drafts = defaultdict(list)

    for name, history in player_boost_history.items():
        for i in range(1, len(history)):
            prev_date, prev_boost, prev_rs, prev_drafts = history[i-1]
            curr_date, curr_boost, curr_rs, curr_drafts = history[i]

            change = curr_boost - prev_boost
            boost_changes.append(change)

            # Categorize by RS performance
            if prev_rs >= 5.0:
                boost_change_by_rs["high_rs (≥5)"].append(change)
            elif prev_rs >= 3.0:
                boost_change_by_rs["mid_rs (3-5)"].append(change)
            else:
                boost_change_by_rs["low_rs (<3)"].append(change)

            # Categorize by draft popularity
            if prev_drafts >= 1000:
                boost_change_by_drafts["very_popular (≥1000)"].append(change)
            elif prev_drafts >= 200:
                boost_change_by_drafts["popular (200-1000)"].append(change)
            elif prev_drafts >= 50:
                boost_change_by_drafts["moderate (50-200)"].append(change)
            else:
                boost_change_by_drafts["low_pop (<50)"].append(change)

    if boost_changes:
        boost_changes.sort()
        n = len(boost_changes)
        print(f"\n  Day-over-Day Boost Changes (n={n}):")
        print(f"    Mean change:  {sum(boost_changes)/n:+.3f}")
        print(f"    p10:          {boost_changes[int(n*0.1)]:+.2f}")
        print(f"    p25:          {boost_changes[int(n*0.25)]:+.2f}")
        print(f"    p50 (median): {boost_changes[int(n*0.5)]:+.2f}")
        print(f"    p75:          {boost_changes[int(n*0.75)]:+.2f}")
        print(f"    p90:          {boost_changes[int(n*0.9)]:+.2f}")

        # What % are exactly 0?
        exact_zero = sum(1 for c in boost_changes if c == 0.0)
        within_01 = sum(1 for c in boost_changes if abs(c) <= 0.1)
        within_03 = sum(1 for c in boost_changes if abs(c) <= 0.3)
        within_05 = sum(1 for c in boost_changes if abs(c) <= 0.5)
        print(f"\n    Exactly 0:    {exact_zero} ({exact_zero/n*100:.1f}%)")
        print(f"    Within ±0.1:  {within_01} ({within_01/n*100:.1f}%)")
        print(f"    Within ±0.3:  {within_03} ({within_03/n*100:.1f}%)")
        print(f"    Within ±0.5:  {within_05} ({within_05/n*100:.1f}%)")

        print(f"\n  Boost Change by Previous RS Performance:")
        for category in sorted(boost_change_by_rs.keys()):
            changes = boost_change_by_rs[category]
            if changes:
                avg = sum(changes) / len(changes)
                print(f"    {category}: avg change {avg:+.3f} (n={len(changes)})")

        print(f"\n  Boost Change by Previous Draft Popularity:")
        for category in sorted(boost_change_by_drafts.keys()):
            changes = boost_change_by_drafts[category]
            if changes:
                avg = sum(changes) / len(changes)
                print(f"    {category}: avg change {avg:+.3f} (n={len(changes)})")

    # =========================================================================
    # ANALYSIS 10: The "Sweet Spot" — What RS+Boost combo wins most?
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 10: The 'Sweet Spot' — RS+Boost Combos That Win")
    print("=" * 80)

    # Create a 2D grid of RS x Boost and count appearances in top-5
    grid = defaultdict(lambda: {"count": 0, "total": 0, "avg_value": 0})

    for date in dates:
        players = by_date[date]
        valid = [p for p in players if p["actual_rs"] > 0 and p["total_value"] > 0]
        if len(valid) < 5:
            continue
        valid.sort(key=lambda x: x["total_value"], reverse=True)
        top5 = valid[:5]

        for p in top5:
            rs_bucket = int(p["actual_rs"])  # Round down
            if rs_bucket > 8:
                rs_bucket = 8
            boost_bucket = round(p["actual_card_boost"] * 2) / 2  # Round to 0.5
            if boost_bucket > 3.0:
                boost_bucket = 3.0
            key = (rs_bucket, boost_bucket)
            grid[key]["count"] += 1
            grid[key]["total"] += p["total_value"]

    for key in grid:
        g = grid[key]
        g["avg_value"] = g["total"] / g["count"] if g["count"] > 0 else 0

    # Sort by count
    sorted_grid = sorted(grid.items(), key=lambda x: x[1]["count"], reverse=True)
    print(f"\n  Top 20 RS×Boost combos in winning lineups:")
    print(f"  {'RS':>4} {'Boost':>6} {'Count':>6} {'Avg Value':>10}")
    for (rs, boost), data in sorted_grid[:20]:
        print(f"  {rs:>4} {boost:>6.1f} {data['count']:>6} {data['avg_value']:>10.1f}")

    # =========================================================================
    # ANALYSIS 11: RS Floor Analysis - What % of winners have RS < current floors?
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 11: RS Floor Sensitivity (Winners Below Thresholds)")
    print("=" * 80)

    total_winners = len(winning_rs) if winning_rs else 0
    if total_winners > 0:
        for threshold in [1.5, 2.0, 2.5, 3.0, 3.5]:
            below = sum(1 for r in winning_rs if r < threshold)
            pct = below / total_winners * 100
            print(f"  Winners with RS < {threshold}: {below}/{total_winners} ({pct:.1f}%)")

    # =========================================================================
    # ANALYSIS 12: Per-Team Analysis
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 12: Per-Team Performance in Top Performers")
    print("=" * 80)

    team_stats = defaultdict(lambda: {"count": 0, "rs_sum": 0, "boost_sum": 0, "value_sum": 0})
    for r in tp_rows:
        team = r.get("team", "UNK")
        if team and r["actual_rs"] > 0:
            team_stats[team]["count"] += 1
            team_stats[team]["rs_sum"] += r["actual_rs"]
            team_stats[team]["boost_sum"] += r["actual_card_boost"]
            team_stats[team]["value_sum"] += r["total_value"]

    print(f"\n  {'Team':>5} {'Count':>6} {'Avg RS':>7} {'Avg Boost':>10} {'Avg Value':>10}")
    for team in sorted(team_stats.keys(), key=lambda t: team_stats[t]["count"], reverse=True):
        s = team_stats[team]
        if s["count"] >= 5:
            print(f"  {team:>5} {s['count']:>6} {s['rs_sum']/s['count']:>7.2f} {s['boost_sum']/s['count']:>10.2f} {s['value_sum']/s['count']:>10.2f}")

    # =========================================================================
    # ANALYSIS 13: Season evolution - has the game changed over time?
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 13: Temporal Evolution (Monthly Averages)")
    print("=" * 80)

    monthly = defaultdict(lambda: {"rs": [], "boost": [], "value": [], "drafts": []})
    for r in tp_rows:
        if r["actual_rs"] > 0:
            month = r["date"][:7]  # YYYY-MM
            monthly[month]["rs"].append(r["actual_rs"])
            monthly[month]["boost"].append(r["actual_card_boost"])
            monthly[month]["value"].append(r["total_value"])
            monthly[month]["drafts"].append(r["drafts"])

    print(f"\n  {'Month':>8} {'N':>5} {'Avg RS':>7} {'Avg Boost':>10} {'Avg Value':>10} {'Avg Drafts':>11}")
    for month in sorted(monthly.keys()):
        m = monthly[month]
        n = len(m["rs"])
        print(f"  {month:>8} {n:>5} {sum(m['rs'])/n:>7.2f} {sum(m['boost'])/n:>10.2f} {sum(m['value'])/n:>10.2f} {sum(m['drafts'])/n:>11.0f}")

    # =========================================================================
    # ANALYSIS 14: Simulate our model's draft strategy
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 14: Model Strategy Simulation")
    print("=" * 80)
    print("  (What score would we get if we PERFECTLY predicted RS and boost,")
    print("   then used EV = RS*(2.0+boost) to select top 5?)")

    model_scores = []
    actual_best_scores = []
    overlap_counts = []

    for date in dates:
        players = by_date[date]
        valid = [p for p in players if p["actual_rs"] > 0 and p["total_value"] > 0]
        if len(valid) < 5:
            continue

        # Our strategy: pick top 5 by EV, slot by RS
        for p in valid:
            p["ev"] = p["actual_rs"] * (2.0 + p["actual_card_boost"])

        our_pick = sorted(valid, key=lambda x: x["ev"], reverse=True)[:5]
        our_pick.sort(key=lambda x: x["actual_rs"], reverse=True)
        our_score = sum(p["actual_rs"] * (slot_mults[i] + p["actual_card_boost"]) for i, p in enumerate(our_pick))
        model_scores.append(our_score)

        # Actual best: what WOULD have been the best 5-player score?
        # This is a harder optimization... approximate with top 5 by total_value
        actual_best = sorted(valid, key=lambda x: x["total_value"], reverse=True)[:5]
        actual_best_score = sum(p["total_value"] for p in actual_best)
        actual_best_scores.append(actual_best_score)

        # Overlap
        our_names = {p["player_name"] for p in our_pick}
        best_names = {p["player_name"] for p in actual_best}
        overlap = len(our_names & best_names)
        overlap_counts.append(overlap)

    if model_scores:
        n = len(model_scores)
        print(f"\n  Dates: {n}")
        print(f"  Our EV strategy avg score:  {sum(model_scores)/n:.1f}")
        print(f"  Actual best-5 avg score:    {sum(actual_best_scores)/n:.1f}")
        print(f"  Capture rate:               {sum(model_scores)/sum(actual_best_scores)*100:.1f}%")
        print(f"\n  Player overlap with actual best 5:")
        for k in range(6):
            count = sum(1 for o in overlap_counts if o == k)
            print(f"    {k}/5 overlap: {count} dates ({count/n*100:.1f}%)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    analyze()
