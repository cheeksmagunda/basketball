#!/usr/bin/env python3
"""
validate_archetypes.py — Archetype Validation & Middle-Class Trap Detector

Loads a known "stinky" draft day (2026-03-24.csv predictions) alongside the
most recent actuals data, runs the redesigned MILP solver against it, and
prints a comparison showing how the new barbell logic bypasses the
"middle-class trap" that plagued earlier optimizer versions.

The middle-class trap: lineups full of RS ~3.2 / Boost ~1.8x players that
*look* reasonable but mathematically cannot compete with the Elite Hybrid
archetype (RS ≥ 4.0, Boost ≥ 1.5) that wins 48.2% of slates.

Usage:
    python scripts/validate_archetypes.py
    python scripts/validate_archetypes.py --date 2026-03-22
"""

import csv
import os
import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from api.asset_optimizer import optimize_lineup
from api.shared import SLOT_MULTIPLIERS, SLOT_LABELS

# ─────────────────────────────────────────────────────────────────────────────
# ARCHETYPE DEFINITIONS (from 85-entry leaderboard analysis, Mar 5–22)
# ─────────────────────────────────────────────────────────────────────────────
ARCHETYPES = {
    "Elite Hybrid":   {"rs_min": 4.0, "boost_min": 1.5, "win_pct": 48.2},
    "Star Anchor":    {"rs_min": 5.5, "boost_min": 0.0, "boost_max": 1.5, "win_pct": 23.5},
    "Boost Leverage": {"rs_min": 2.5, "rs_max": 4.0, "boost_min": 2.5, "win_pct": 20.0},
    "Pure Boost":     {"rs_max": 2.5, "boost_min": 3.0, "win_pct": 0.0},
    "Middle Class":   {"rs_min": 2.5, "rs_max": 4.0, "boost_min": 1.0, "boost_max": 2.5, "win_pct": 8.3},
}


def classify_archetype(rs, boost):
    """Classify a player into one of the data-driven archetypes."""
    if rs >= 5.5 and boost < 1.5:
        return "Star Anchor"
    if rs >= 4.0 and boost >= 1.5:
        return "Elite Hybrid"
    if 2.5 <= rs < 4.0 and boost >= 2.5:
        return "Boost Leverage"
    if rs < 2.5 and boost >= 3.0:
        return "Pure Boost"
    if 2.5 <= rs < 4.0 and 1.0 <= boost < 2.5:
        return "Middle Class"
    return "Other"


def compute_total_value(rs, boost, slot_mult):
    """Real Sports additive formula: Value = RS × (SlotMult + CardBoost)."""
    return rs * (slot_mult + boost)


def load_predictions(date_str):
    """Load slate-wide predictions from CSV."""
    pred_path = PROJECT_ROOT / "data" / "predictions" / f"{date_str}.csv"
    if not pred_path.exists():
        print(f"  ERROR: No predictions file at {pred_path}")
        return [], []

    chalk_players = []
    moonshot_players = []

    with open(pred_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("scope") != "slate":
                continue
            player = {
                "name": row["player_name"],
                "id": row.get("player_id", ""),
                "team": row.get("team", ""),
                "pos": row.get("pos", ""),
                "rating": float(row.get("predicted_rs", 0)),
                "est_mult": float(row.get("est_card_boost", 0)),
                "pred_min": float(row.get("pred_min", 0)),
                "pts": float(row.get("pts", 0)),
                "slot": row.get("slot", ""),
                "player_variance": 0.15,  # default
            }
            lt = row.get("lineup_type", "")
            if lt == "chalk":
                chalk_players.append(player)
            elif lt == "upside":
                moonshot_players.append(player)

    return chalk_players, moonshot_players


def load_actuals(date_str):
    """Load actuals CSV, returns dict of player_name -> {actual_rs, actual_card_boost, total_value}."""
    actuals_path = PROJECT_ROOT / "data" / "actuals" / f"{date_str}.csv"
    if not actuals_path.exists():
        return {}

    result = {}
    with open(actuals_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("player_name", "")
            rs = float(row.get("actual_rs", 0) or 0)
            boost = float(row.get("actual_card_boost", 0) or 0)
            tv = float(row.get("total_value", 0) or 0)
            result[name] = {"actual_rs": rs, "actual_card_boost": boost, "total_value": tv}
    return result


def build_candidate_pool(date_str):
    """Build a full candidate pool from predictions CSV — all slate players regardless of lineup type."""
    pred_path = PROJECT_ROOT / "data" / "predictions" / f"{date_str}.csv"
    if not pred_path.exists():
        return []

    seen = set()
    pool = []
    with open(pred_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("scope") != "slate":
                continue
            name = row["player_name"]
            if name in seen:
                continue
            seen.add(name)
            pool.append({
                "name": name,
                "id": row.get("player_id", ""),
                "team": row.get("team", ""),
                "pos": row.get("pos", ""),
                "rating": float(row.get("predicted_rs", 0)),
                "est_mult": float(row.get("est_card_boost", 0)),
                "pred_min": float(row.get("pred_min", 0)),
                "pts": float(row.get("pts", 0)),
                "player_variance": 0.15,
                "adj_ceiling": float(row.get("predicted_rs", 0)) * 1.1,
            })
    return pool


def print_lineup(label, players, actuals=None, slot_mults=None):
    """Pretty-print a lineup with archetype classification."""
    if slot_mults is None:
        slot_mults = SLOT_MULTIPLIERS

    print(f"\n  {'─' * 70}")
    print(f"  {label}")
    print(f"  {'─' * 70}")
    print(f"  {'Slot':<6} {'Player':<25} {'RS':>5} {'Boost':>6} {'TV':>7} {'Archetype':<16}")
    print(f"  {'─' * 70}")

    total_tv = 0
    archetypes_found = {}

    for i, p in enumerate(players):
        rs = p.get("rating", 0)
        boost = p.get("est_mult", 0)
        slot_label = p.get("slot", SLOT_LABELS[i] if i < len(SLOT_LABELS) else "1.0x")

        # Parse slot multiplier from label
        try:
            slot_mult = float(slot_label.replace("x", ""))
        except (ValueError, AttributeError):
            slot_mult = slot_mults[i] if i < len(slot_mults) else 1.0

        tv = compute_total_value(rs, boost, slot_mult)
        total_tv += tv
        archetype = classify_archetype(rs, boost)
        archetypes_found[archetype] = archetypes_found.get(archetype, 0) + 1

        # Show actual RS if available
        actual_str = ""
        if actuals and p.get("name") in actuals:
            actual = actuals[p["name"]]
            actual_str = f"  (actual RS: {actual['actual_rs']:.1f})"

        print(f"  {slot_label:<6} {p.get('name', '?'):<25} {rs:>5.1f} {boost:>5.1f}x {tv:>7.1f} {archetype:<16}{actual_str}")

    print(f"  {'─' * 70}")
    print(f"  {'TOTAL VALUE':>44} {total_tv:>7.1f}")
    print(f"  Archetype mix: {archetypes_found}")

    # Middle-class trap detection
    mc_count = archetypes_found.get("Middle Class", 0)
    if mc_count >= 3:
        print(f"  ** MIDDLE-CLASS TRAP DETECTED: {mc_count}/5 players in dead zone **")
    elif mc_count == 0:
        print(f"  ** CLEAN: No middle-class trap players **")

    return total_tv, archetypes_found


def run_new_solver(pool, mode="chalk"):
    """Run the redesigned MILP solver on a candidate pool."""
    if mode == "chalk":
        return optimize_lineup(
            pool, n=5,
            sort_key="chalk_ev",
            rating_key="rating",
            card_boost_key="est_mult",
            objective_mode="chalk",
        )
    else:
        # Moonshot: boost leverage with β=0.8 (Elite Hybrid preference)
        return optimize_lineup(
            pool, n=5,
            sort_key="moonshot_ev",
            rating_key="rating",
            card_boost_key="est_mult",
            objective_mode="moonshot",
            variance_uplift=0.35,
            boost_leverage_extra_power=0.8,
            two_phase=True,
            raw_rating_key="rating",
        )


def analyze_actuals_archetypes():
    """Analyze archetype distribution across all actuals files."""
    actuals_dir = PROJECT_ROOT / "data" / "actuals"
    if not actuals_dir.exists():
        print("No actuals directory found")
        return

    all_winners = []
    dates_analyzed = 0

    for csv_file in sorted(actuals_dir.glob("*.csv")):
        actuals = {}
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("player_name", "")
                rs = float(row.get("actual_rs", 0) or 0)
                boost = float(row.get("actual_card_boost", 0) or 0)
                tv = float(row.get("total_value", 0) or 0)
                if rs > 0:
                    actuals[name] = {"rs": rs, "boost": boost, "tv": tv}

        if len(actuals) < 5:
            continue

        dates_analyzed += 1
        # Sort by total_value or by RS×(1.6+boost) as proxy
        ranked = sorted(
            actuals.items(),
            key=lambda x: x[1]["tv"] if x[1]["tv"] > 0 else x[1]["rs"] * (1.6 + x[1]["boost"]),
            reverse=True,
        )

        for name, data in ranked[:5]:
            archetype = classify_archetype(data["rs"], data["boost"])
            all_winners.append({
                "date": csv_file.stem,
                "name": name,
                "rs": data["rs"],
                "boost": data["boost"],
                "tv": data["tv"],
                "archetype": archetype,
            })

    print(f"\n{'=' * 72}")
    print(f"  ARCHETYPE DISTRIBUTION — Top 5 per slate across {dates_analyzed} dates")
    print(f"  ({len(all_winners)} total top-5 entries)")
    print(f"{'=' * 72}")

    arch_counts = {}
    for w in all_winners:
        arch_counts[w["archetype"]] = arch_counts.get(w["archetype"], 0) + 1

    total = len(all_winners) or 1
    for arch in ["Elite Hybrid", "Star Anchor", "Boost Leverage", "Middle Class", "Pure Boost", "Other"]:
        count = arch_counts.get(arch, 0)
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        expected_pct = ARCHETYPES.get(arch, {}).get("win_pct", 0)
        print(f"  {arch:<16} {count:>3} ({pct:>5.1f}%) {bar:<30} expected: {expected_pct}%")

    # Show some examples
    print(f"\n  Top 10 highest-value actuals:")
    by_value = sorted(all_winners, key=lambda x: x["tv"] if x["tv"] > 0 else x["rs"] * (1.6 + x["boost"]), reverse=True)
    for w in by_value[:10]:
        print(f"    {w['date']} {w['name']:<25} RS={w['rs']:.1f} Boost={w['boost']:.1f}x TV={w['tv']:.1f} → {w['archetype']}")

    return arch_counts


def main():
    parser = argparse.ArgumentParser(description="Validate archetype optimizer against historical data")
    parser.add_argument("--date", default="2026-03-24", help="Date to analyze (YYYY-MM-DD)")
    args = parser.parse_args()
    date_str = args.date

    print("=" * 72)
    print(f"  ARCHETYPE VALIDATION — {date_str}")
    print(f"  Barbell Strategy vs Middle-Class Trap")
    print("=" * 72)

    # 1. Show the old lineup (what was actually drafted)
    chalk_old, moonshot_old = load_predictions(date_str)

    # Try to find actuals for this date or closest prior
    actuals = load_actuals(date_str)
    if not actuals:
        # Try recent dates
        for d in range(24, 4, -1):
            alt_date = f"2026-03-{d:02d}"
            actuals = load_actuals(alt_date)
            if actuals:
                print(f"  (Using actuals from {alt_date})")
                break

    if chalk_old:
        print("\n  ORIGINAL STARTING 5 (pre-redesign predictions)")
        old_chalk_tv, old_chalk_arch = print_lineup("Starting 5 (OLD)", chalk_old, actuals)
    else:
        old_chalk_tv = 0
        print("  No chalk predictions found for this date")

    if moonshot_old:
        print("\n  ORIGINAL MOONSHOT (pre-redesign predictions)")
        old_moon_tv, old_moon_arch = print_lineup("Moonshot (OLD)", moonshot_old, actuals)
    else:
        old_moon_tv = 0
        print("  No moonshot predictions found for this date")

    # 2. Build candidate pool and run the new solver
    pool = build_candidate_pool(date_str)
    if not pool:
        print("\n  ERROR: No candidate pool available. Cannot run solver.")
        return

    print(f"\n  Candidate pool: {len(pool)} players")
    print(f"  Pool RS range: {min(p['rating'] for p in pool):.1f} – {max(p['rating'] for p in pool):.1f}")
    print(f"  Pool Boost range: {min(p['est_mult'] for p in pool):.1f}x – {max(p['est_mult'] for p in pool):.1f}x")

    # Run new solver
    new_chalk = run_new_solver(pool, mode="chalk")
    new_moonshot = run_new_solver(pool, mode="moonshot")

    if new_chalk:
        print("\n  NEW STARTING 5 (barbell optimizer)")
        new_chalk_tv, new_chalk_arch = print_lineup("Starting 5 (NEW)", new_chalk, actuals)
    else:
        new_chalk_tv = 0

    if new_moonshot:
        print("\n  NEW MOONSHOT (boost-leverage optimizer, β=0.8)")
        new_moon_tv, new_moon_arch = print_lineup("Moonshot (NEW)", new_moonshot, actuals)
    else:
        new_moon_tv = 0

    # 3. Comparison summary
    print(f"\n{'=' * 72}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 72}")

    if old_chalk_tv > 0 and new_chalk_tv > 0:
        chalk_delta = new_chalk_tv - old_chalk_tv
        print(f"  Starting 5:  OLD={old_chalk_tv:.1f}  NEW={new_chalk_tv:.1f}  delta={chalk_delta:+.1f}")

    if old_moon_tv > 0 and new_moon_tv > 0:
        moon_delta = new_moon_tv - old_moon_tv
        print(f"  Moonshot:    OLD={old_moon_tv:.1f}  NEW={new_moon_tv:.1f}  delta={moon_delta:+.1f}")

    # 4. Full archetype analysis across all dates
    print("\n")
    analyze_actuals_archetypes()

    # 5. Explain the barbell logic
    print(f"\n{'=' * 72}")
    print(f"  BARBELL STRATEGY EXPLAINED")
    print(f"{'=' * 72}")
    print("""
  The additive formula Value = RS × (SlotMult + CardBoost) means:

  WINNING ARCHETYPES:
    Elite Hybrid  (48.2%): RS 4.0+ AND Boost 1.5+ → both terms large
    Star Anchor   (23.5%): RS 5.5+ even with 0.0x  → RS dominates
    Boost Leverage (20%):  RS 3.0+ with Boost 2.5+ → boost amplifies decent RS

  LOSING ARCHETYPE:
    Pure Boost     (0.0%): RS < 2.5 with ANY boost → weak base kills value
    Middle Class   (8.3%): RS 3.0 / Boost 1.5x     → neither term is large enough

  The old optimizer filled lineups with Middle Class players (RS ~3.2, Boost ~1.8x).
  The new barbell optimizer uses:
    - CHALK: Pure TV maximization. Stars (RS 5.5+) compete freely even with 0.0x boost.
    - MOONSHOT: Boost^0.8 leverage. RS 4.0 / Boost 3.0x player gets 3.0^0.8 = 2.41x
      rating multiplier vs RS 3.2 / Boost 1.5x getting 1.5^0.8 = 1.38x. The nonlinear
      boost exponent strongly selects for Elite Hybrids over Middle Class.
""")


if __name__ == "__main__":
    main()
