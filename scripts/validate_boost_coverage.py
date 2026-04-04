#!/usr/bin/env python3
"""
Validate whether the HYBRID draft archetype (1 star anchor + 4 high-boost
players per lineup, 2 lineups = 10 picks) can capture 10 of the top 15
highest-value performers on each historical date.

Uses actual outcomes from data/top_performers.csv — tests whether the
*archetype* is correct, not the projection model.

Usage:
    python scripts/validate_boost_coverage.py
    python scripts/validate_boost_coverage.py --verbose
    python scripts/validate_boost_coverage.py --boost-min 1.5 --star-rs 3.5
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP_PERFORMERS = ROOT / "data" / "top_performers.csv"


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def load_top_performers() -> dict[str, list[dict]]:
    """Load top_performers.csv → {date: [rows sorted by total_value desc]}."""
    by_date: dict[str, list[dict]] = defaultdict(list)
    with open(TOP_PERFORMERS, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            date = row.get("date", "").strip()
            if not date:
                continue
            by_date[date].append({
                "player_name": row.get("player_name", "").strip(),
                "team": (row.get("team") or "").strip().upper(),
                "actual_rs": _safe_float(row.get("actual_rs")),
                "actual_card_boost": _safe_float(row.get("actual_card_boost")),
                "total_value": _safe_float(row.get("total_value")),
                "drafts": _safe_float(row.get("drafts")),
            })
    # Sort each date by total_value descending
    for date in by_date:
        by_date[date].sort(key=lambda x: -x["total_value"])
    return dict(by_date)


def simulate_date(
    performers: list[dict],
    top_k: int = 15,
    star_rs_min: float = 4.0,
    boost_min: float = 2.0,
    starter_rs_min: float = 3.0,
    max_per_team: int = 1,
) -> dict:
    """Simulate 2 HYBRID lineups (1 star + 1 starter + 3 boost each) from top-K.

    Per lineup:
      Phase A: 1 star anchor (highest RS, RS >= star_rs_min)
      Phase B: 1 high-production starter (high RS, boost < boost_min)
      Phase C: 3 high-boost players (boost >= boost_min)

    Returns dict with coverage count, diagnostics, and selected players.
    """
    top = performers[:top_k]
    top_names = {p["player_name"] for p in top}

    # Classify
    n_stars = sum(1 for p in top if p["actual_rs"] >= star_rs_min)
    n_boost = sum(1 for p in top if p["actual_card_boost"] >= boost_min)
    n_starters = sum(1 for p in top if p["actual_rs"] >= starter_rs_min and p["actual_card_boost"] < boost_min)
    n_both = sum(1 for p in top if p["actual_rs"] >= star_rs_min and p["actual_card_boost"] >= boost_min)
    n_neither = sum(1 for p in top if p["actual_rs"] < starter_rs_min and p["actual_card_boost"] < boost_min)

    available = list(top)
    all_selected = []
    lineups = []

    for lineup_num in (1, 2):
        lineup_teams: set[str] = set()
        lineup_players: list[dict] = []
        reason = None

        # Phase A: pick 1 star anchor (highest RS where RS >= star_rs_min)
        star_candidates = [
            p for p in available
            if p["actual_rs"] >= star_rs_min and p["team"] not in lineup_teams
        ]
        star_candidates.sort(key=lambda p: -p["actual_rs"])

        star = star_candidates[0] if star_candidates else None
        if star is None:
            reason = "no_star"
            lineups.append({"num": lineup_num, "players": [], "reason": reason})
            continue

        available.remove(star)
        lineup_teams.add(star["team"])
        lineup_players.append(star)

        # Phase B: pick 1 high-production starter (high RS, boost < boost_min)
        # These are productive starters who don't get high boosts (popular players)
        starter_candidates = [
            p for p in available
            if p["actual_rs"] >= starter_rs_min
            and p["actual_card_boost"] < boost_min
            and p["team"] not in lineup_teams
        ]
        starter_candidates.sort(key=lambda p: -p["actual_rs"])

        starter = starter_candidates[0] if starter_candidates else None
        if starter:
            available.remove(starter)
            lineup_teams.add(starter["team"])
            lineup_players.append(starter)

        # Phase C: pick 3 high-boost players (boost >= boost_min, max_per_team)
        boost_needed = 5 - len(lineup_players)  # 3 if starter found, 4 if not
        boost_candidates = [
            p for p in available
            if p["actual_card_boost"] >= boost_min and p["team"] not in lineup_teams
        ]
        boost_candidates.sort(key=lambda p: -p["total_value"])

        boost_picks = []
        for c in boost_candidates:
            if c["team"] not in lineup_teams:
                boost_picks.append(c)
                lineup_teams.add(c["team"])
                if len(boost_picks) == boost_needed:
                    break

        for bp in boost_picks:
            available.remove(bp)

        lineup_players.extend(boost_picks)
        if len(lineup_players) < 5:
            reason = f"only_{len(lineup_players)}_players"

        lineups.append({"num": lineup_num, "players": lineup_players, "reason": reason})
        all_selected.extend(lineup_players)

    selected_names = {p["player_name"] for p in all_selected}
    coverage = len(selected_names & top_names)

    # Diagnose failure mode
    failure_mode = None
    if coverage < min(10, len(top)):
        if n_stars < 2:
            failure_mode = "insufficient_stars"
        elif n_boost < 6:
            failure_mode = "insufficient_boost"
        elif n_starters < 2 and n_boost < 8:
            failure_mode = "insufficient_starters"
        elif n_neither > 5:
            failure_mode = "neither_eligible"
        else:
            failure_mode = "team_conflicts"

    return {
        "top_k_actual": len(top),
        "n_stars": n_stars,
        "n_boost": n_boost,
        "n_starters": n_starters,
        "n_both": n_both,
        "n_neither": n_neither,
        "selected": len(all_selected),
        "coverage": coverage,
        "failure_mode": failure_mode,
        "lineups": lineups,
        "top": top,
        "all_selected": all_selected,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate HYBRID draft archetype coverage")
    parser.add_argument("--target", type=int, default=10, help="Coverage target (default: 10)")
    parser.add_argument("--top-k", type=int, default=15, help="Top K performers per date (default: 15)")
    parser.add_argument("--star-rs", type=float, default=4.0, help="Star RS minimum (default: 4.0)")
    parser.add_argument("--boost-min", type=float, default=2.0, help="Boost floor (default: 2.0)")
    parser.add_argument("--starter-rs", type=float, default=3.0, help="Starter RS minimum (default: 3.0)")
    parser.add_argument("--verbose", action="store_true", help="Show per-player detail on miss dates")
    args = parser.parse_args()

    data = load_top_performers()
    dates = sorted(data.keys())

    print(f"\n{'='*100}")
    print(f"  HYBRID DRAFT COVERAGE (1 star + 1 starter + 3 boost per lineup)")
    print(f"  star_rs >= {args.star_rs}  |  starter_rs >= {args.starter_rs}  |  boost >= {args.boost_min}  |  target: {args.target}/{args.top_k}")
    print(f"{'='*100}")
    print(f"{'DATE':<12} {'TOP-K':>5} {'STARS':>5} {'START':>5} {'BOOST':>7} {'NEITHER':>7} {'SEL':>4} {'COV':>7} {'STATUS':>6}  REASON")
    print(f"{'-'*100}")

    hits = 0
    misses = 0
    total_coverage = 0
    total_possible = 0
    failure_modes: dict[str, int] = defaultdict(int)
    miss_dates = []

    for date in dates:
        performers = data[date]
        result = simulate_date(
            performers,
            top_k=args.top_k,
            star_rs_min=args.star_rs,
            boost_min=args.boost_min,
            starter_rs_min=args.starter_rs,
        )

        top_k_actual = result["top_k_actual"]
        target = min(args.target, top_k_actual)
        is_hit = result["coverage"] >= target
        status = "HIT" if is_hit else "MISS"

        if is_hit:
            hits += 1
        else:
            misses += 1
            if result["failure_mode"]:
                failure_modes[result["failure_mode"]] += 1
            miss_dates.append((date, result))

        total_coverage += result["coverage"]
        total_possible += top_k_actual

        reason_str = result["failure_mode"] or ""
        print(
            f"{date:<12} {top_k_actual:>5} {result['n_stars']:>5} "
            f"{result['n_starters']:>5} {result['n_boost']:>7} {result['n_neither']:>7} "
            f"{result['selected']:>4} {result['coverage']:>3}/{top_k_actual:<3} "
            f"{'✓' if is_hit else '✗':>4}   {reason_str}"
        )

    # Verbose: show per-player detail on miss dates
    if args.verbose and miss_dates:
        print(f"\n{'='*90}")
        print("  MISS DATE DETAILS")
        print(f"{'='*90}")
        for date, result in miss_dates:
            selected_names = {p["player_name"] for p in result["all_selected"]}
            print(f"\n  {date}  (coverage: {result['coverage']}/{result['top_k_actual']}, "
                  f"reason: {result['failure_mode']})")
            print(f"  {'RANK':<5} {'PLAYER':<25} {'TEAM':<5} {'RS':>5} {'BOOST':>5} {'VALUE':>7} {'TYPE':<12} {'SEL':<4}")
            for i, p in enumerate(result["top"], 1):
                ptype = []
                if p["actual_rs"] >= args.star_rs:
                    ptype.append("STAR")
                if p["actual_card_boost"] >= args.boost_min:
                    ptype.append("BOOST")
                elif p["actual_rs"] >= args.starter_rs:
                    ptype.append("START")
                if not ptype:
                    ptype.append("--")
                sel = "✓" if p["player_name"] in selected_names else ""
                print(
                    f"  {i:<5} {p['player_name']:<25} {p['team']:<5} "
                    f"{p['actual_rs']:>5.1f} {p['actual_card_boost']:>5.1f} "
                    f"{p['total_value']:>7.1f} {'+'.join(ptype):<12} {sel}"
                )
            # Show lineups
            for lu in result["lineups"]:
                players = lu["players"]
                if not players:
                    print(f"    Lineup {lu['num']}: EMPTY ({lu['reason']})")
                    continue
                names = [p["player_name"] for p in players]
                print(f"    Lineup {lu['num']}: {', '.join(names)}"
                      + (f"  ({lu['reason']})" if lu["reason"] else ""))

    # Summary
    total_dates = hits + misses
    avg_coverage = total_coverage / total_dates if total_dates else 0
    avg_possible = total_possible / total_dates if total_dates else 0

    print(f"\n{'='*90}")
    print("  SUMMARY")
    print(f"{'='*90}")
    print(f"  Dates analyzed:    {total_dates}")
    print(f"  Coverage target:   {args.target}/{args.top_k}")
    print(f"  Hit rate:          {hits}/{total_dates} ({100*hits/total_dates:.1f}%)" if total_dates else "")
    print(f"  Miss rate:         {misses}/{total_dates} ({100*misses/total_dates:.1f}%)" if total_dates else "")
    print(f"  Avg coverage:      {avg_coverage:.1f}/{avg_possible:.1f}")
    print(f"  Avg coverage %:    {100*total_coverage/total_possible:.1f}%" if total_possible else "")
    if failure_modes:
        print(f"\n  Failure modes:")
        for mode, count in sorted(failure_modes.items(), key=lambda x: -x[1]):
            print(f"    {mode:<25} {count:>4} dates")
    print()


if __name__ == "__main__":
    main()
