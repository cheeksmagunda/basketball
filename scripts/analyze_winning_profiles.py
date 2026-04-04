#!/usr/bin/env python3
"""
Deep analysis of what actually wins across the full season.

For every date in top_performers.csv, analyzes the top 15 highest-value
performers to answer:
  1. What RS + boost profiles do winners actually have?
  2. What's the optimal star anchor count and RS threshold?
  3. What boost range captures the most value?
  4. What archetype (star/boost/hybrid) dominates?
  5. Given max_per_team=1, what's the best achievable coverage?

Usage:
    python scripts/analyze_winning_profiles.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOP_PERFORMERS = ROOT / "data" / "top_performers.csv"


def _sf(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def load():
    by_date = defaultdict(list)
    with open(TOP_PERFORMERS, newline="") as f:
        for row in csv.DictReader(f):
            d = row.get("date", "").strip()
            if not d:
                continue
            by_date[d].append({
                "name": row.get("player_name", "").strip(),
                "team": (row.get("team") or "").strip().upper(),
                "rs": _sf(row.get("actual_rs")),
                "boost": _sf(row.get("actual_card_boost")),
                "value": _sf(row.get("total_value")),
                "drafts": _sf(row.get("drafts")),
            })
    for d in by_date:
        by_date[d].sort(key=lambda x: -x["value"])
    return dict(by_date)


def classify(p):
    """Classify a player into archetype based on RS and boost."""
    rs, boost = p["rs"], p["boost"]
    if rs >= 4.0 and boost >= 2.0:
        return "star+boost"  # rare unicorn
    elif rs >= 4.0:
        return "star"        # high RS, low boost
    elif boost >= 2.0:
        return "boost"       # high boost, modest RS
    elif rs >= 3.0:
        return "starter"     # decent RS, low boost
    else:
        return "sleeper"     # low RS, low boost (lucky pop-off)


def greedy_select(top, n_stars, remaining_by, max_per_team=1):
    """Select n_stars by RS + fill to 5 by `remaining_by` metric, max_per_team."""
    avail = list(top)
    lineup = []
    teams = set()

    # Star picks
    star_pool = sorted(avail, key=lambda x: -x["rs"])
    for p in star_pool:
        if p["team"] in teams:
            continue
        lineup.append(p)
        teams.add(p["team"])
        if len(lineup) >= n_stars:
            break

    used = {p["name"] for p in lineup}

    # Fill remaining by chosen metric
    if remaining_by == "ev":
        rest = sorted([p for p in avail if p["name"] not in used],
                      key=lambda x: -x["value"])
    elif remaining_by == "boost":
        rest = sorted([p for p in avail if p["name"] not in used],
                      key=lambda x: (-x["boost"], -x["value"]))
    else:
        rest = sorted([p for p in avail if p["name"] not in used],
                      key=lambda x: -x["value"])

    for p in rest:
        if p["team"] in teams:
            continue
        lineup.append(p)
        teams.add(p["team"])
        if len(lineup) >= 5:
            break

    return lineup


def run_strategy(data, top_k, n_stars_l1, fill_l1, n_stars_l2, fill_l2, boost_floor=None):
    """Run a strategy across all dates. Returns (avg_coverage, hit_rate, details)."""
    hits = 0
    total = 0
    coverage_sum = 0

    for date, performers in sorted(data.items()):
        top = performers[:top_k]
        if len(top) < 5:
            continue
        total += 1
        top_names = {p["name"] for p in top}

        # Apply boost floor filter to candidate pool if specified
        if boost_floor is not None:
            filtered = [p for p in top if p["boost"] >= boost_floor or p["rs"] >= 4.0]
            pool1 = filtered if len(filtered) >= 5 else top
        else:
            pool1 = top

        l1 = greedy_select(pool1, n_stars_l1, fill_l1)
        l1_names = {p["name"] for p in l1}

        # Lineup 2 from remaining
        pool2 = [p for p in top if p["name"] not in l1_names]
        if boost_floor is not None:
            filtered2 = [p for p in pool2 if p["boost"] >= boost_floor or p["rs"] >= 4.0]
            pool2 = filtered2 if len(filtered2) >= 5 else pool2

        l2 = greedy_select(pool2, n_stars_l2, fill_l2)
        l2_names = {p["name"] for p in l2}

        selected = l1_names | l2_names
        cov = len(selected & top_names)
        coverage_sum += cov
        target = min(10, len(top))
        if cov >= target:
            hits += 1

    return {
        "avg_cov": coverage_sum / total if total else 0,
        "hit_rate": hits / total if total else 0,
        "hits": hits,
        "total": total,
    }


def main():
    data = load()
    dates = sorted(data.keys())
    print(f"Loaded {len(dates)} dates, {sum(len(v) for v in data.values())} total performers\n")

    # ═══════════════════════════════════════════════════════════════════
    # PART 1: Profile analysis — what do winners actually look like?
    # ═══════════════════════════════════════════════════════════════════
    print("=" * 80)
    print("  PART 1: WINNER PROFILES (top 15 per date)")
    print("=" * 80)

    all_top = []
    for d in dates:
        for i, p in enumerate(data[d][:15]):
            p["rank"] = i + 1
            p["date"] = d
            all_top.append(p)

    # Archetype distribution
    archetypes = defaultdict(list)
    for p in all_top:
        archetypes[classify(p)].append(p)

    print(f"\n  {'ARCHETYPE':<15} {'COUNT':>6} {'%':>6} {'AVG_RS':>7} {'AVG_BOOST':>10} {'AVG_VALUE':>10}")
    print(f"  {'-'*60}")
    for arch in ["star+boost", "star", "boost", "starter", "sleeper"]:
        players = archetypes[arch]
        if not players:
            continue
        avg_rs = sum(p["rs"] for p in players) / len(players)
        avg_boost = sum(p["boost"] for p in players) / len(players)
        avg_val = sum(p["value"] for p in players) / len(players)
        pct = 100 * len(players) / len(all_top)
        print(f"  {arch:<15} {len(players):>6} {pct:>5.1f}% {avg_rs:>7.2f} {avg_boost:>10.2f} {avg_val:>10.1f}")

    # Boost distribution in top 15
    print(f"\n  BOOST DISTRIBUTION (top 15 per date):")
    boost_buckets = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 3.5)]
    for lo, hi in boost_buckets:
        count = sum(1 for p in all_top if lo <= p["boost"] < hi)
        avg_val = 0
        vals = [p["value"] for p in all_top if lo <= p["boost"] < hi]
        avg_val = sum(vals) / len(vals) if vals else 0
        avg_rs = 0
        rss = [p["rs"] for p in all_top if lo <= p["boost"] < hi]
        avg_rs = sum(rss) / len(rss) if rss else 0
        bar = "█" * int(count / 10)
        print(f"    boost [{lo:.1f}-{hi:.1f}): {count:>5}  avg_val={avg_val:>6.1f}  avg_rs={avg_rs:>5.2f}  {bar}")

    # RS distribution in top 15
    print(f"\n  RS DISTRIBUTION (top 15 per date):")
    rs_buckets = [(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 8), (8, 20)]
    for lo, hi in rs_buckets:
        count = sum(1 for p in all_top if lo <= p["rs"] < hi)
        vals = [p["value"] for p in all_top if lo <= p["rs"] < hi]
        avg_val = sum(vals) / len(vals) if vals else 0
        boosts = [p["boost"] for p in all_top if lo <= p["rs"] < hi]
        avg_boost = sum(boosts) / len(boosts) if boosts else 0
        bar = "█" * int(count / 10)
        print(f"    RS [{lo}-{hi}): {count:>5}  avg_val={avg_val:>6.1f}  avg_boost={avg_boost:>5.2f}  {bar}")

    # Per-date: how many of each type in top 5 vs top 10 vs top 15?
    print(f"\n  ARCHETYPE PRESENCE IN TOP TIERS (avg per date):")
    for tier_name, tier_k in [("Top 5", 5), ("Top 10", 10), ("Top 15", 15)]:
        tier_counts = defaultdict(list)
        for d in dates:
            top = data[d][:tier_k]
            counts = defaultdict(int)
            for p in top:
                counts[classify(p)] += 1
            for arch in ["star+boost", "star", "boost", "starter", "sleeper"]:
                tier_counts[arch].append(counts.get(arch, 0))
        parts = []
        for arch in ["star+boost", "star", "boost", "starter", "sleeper"]:
            avg = sum(tier_counts[arch]) / len(tier_counts[arch]) if tier_counts[arch] else 0
            parts.append(f"{arch}={avg:.1f}")
        print(f"    {tier_name}: {', '.join(parts)}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 2: Strategy sweep — find optimal archetype
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  PART 2: STRATEGY SWEEP (max_per_team=1, top_k=15)")
    print(f"{'='*80}")

    strategies = [
        # (label, n_stars_l1, fill_l1, n_stars_l2, fill_l2, boost_floor)
        ("Pure EV (0 star + 5 ev)", 0, "ev", 0, "ev", None),
        ("1 star + 4 ev (no floor)", 1, "ev", 1, "ev", None),
        ("2 star + 3 ev (no floor)", 2, "ev", 2, "ev", None),
        ("1 star + 4 ev (boost>=1.0)", 1, "ev", 1, "ev", 1.0),
        ("1 star + 4 ev (boost>=1.5)", 1, "ev", 1, "ev", 1.5),
        ("1 star + 4 ev (boost>=2.0)", 1, "ev", 1, "ev", 2.0),
        ("1 star + 4 boost-sort", 1, "boost", 1, "boost", None),
        ("1 star + 4 boost-sort (>=1.5)", 1, "boost", 1, "boost", 1.5),
        ("1 star + 4 boost-sort (>=2.0)", 1, "boost", 1, "boost", 2.0),
        ("2 star + 3 boost-sort", 2, "boost", 2, "boost", None),
        ("Pure boost-sort (no star)", 0, "boost", 0, "boost", None),
    ]

    print(f"\n  {'STRATEGY':<38} {'HIT%':>6} {'AVG_COV':>8} {'HITS':>5}/{'':<5}")
    print(f"  {'-'*70}")

    for label, ns1, f1, ns2, f2, bf in strategies:
        r = run_strategy(data, 15, ns1, f1, ns2, f2, boost_floor=bf)
        print(f"  {label:<38} {100*r['hit_rate']:>5.1f}% {r['avg_cov']:>8.1f} {r['hits']:>5}/{r['total']}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 3: Boost floor sweep — find the sweet spot
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  PART 3: BOOST FLOOR SWEEP (1 star + 4 ev, max_per_team=1)")
    print(f"{'='*80}\n")

    for bf in [None, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
        label = f"floor={bf}" if bf is not None else "no floor"
        r = run_strategy(data, 15, 1, "ev", 1, "ev", boost_floor=bf)
        bar = "█" * int(r["hit_rate"] * 40)
        print(f"  {label:<12} hit={100*r['hit_rate']:>5.1f}%  avg_cov={r['avg_cov']:.1f}  {bar}")

    # ═══════════════════════════════════════════════════════════════════
    # PART 4: Value by slot position — where does boost matter?
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*80}")
    print("  PART 4: VALUE BY RANK POSITION")
    print(f"{'='*80}\n")

    for rank in range(1, 16):
        players = [p for p in all_top if p["rank"] == rank]
        if not players:
            continue
        avg_rs = sum(p["rs"] for p in players) / len(players)
        avg_boost = sum(p["boost"] for p in players) / len(players)
        avg_val = sum(p["value"] for p in players) / len(players)
        pct_boost2 = 100 * sum(1 for p in players if p["boost"] >= 2.0) / len(players)
        pct_star = 100 * sum(1 for p in players if p["rs"] >= 4.0) / len(players)
        print(f"  Rank {rank:>2}: avg_rs={avg_rs:.2f}  avg_boost={avg_boost:.2f}  "
              f"avg_val={avg_val:.1f}  boost≥2: {pct_boost2:.0f}%  rs≥4: {pct_star:.0f}%")

    print()


if __name__ == "__main__":
    main()
