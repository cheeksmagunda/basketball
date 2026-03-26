#!/usr/bin/env python3
"""
CORRECTED Winning Drafts Analysis

The displayed "Xx" on each player card = slot_mult + card_boost (total multiplier).
Slot multipliers: 2.0, 1.8, 1.6, 1.4, 1.2 for positions 1-5.
So actual card_boost = displayed_value - slot_mult.

This changes everything about what the data actually shows.
"""

import statistics
from collections import defaultdict, Counter

SLOT_MULTS = [2.0, 1.8, 1.6, 1.4, 1.2]

# All screenshot data: display_mult is what's shown on screen (slot + boost)
# RS is the Real Score shown
SCREENSHOT_DATA = [
    # ═══ Mar 23 ═══
    {"date": "2026-03-23", "rank": 1, "user": "qwizzywizzy", "score": 80.71, "players": [
        {"name": "D. Jenkins", "display": 4.6, "rs": 5.1},
        {"name": "G. Santos", "display": 4.3, "rs": 3.1},
        {"name": "A. Bailey", "display": 3.5, "rs": 5.1},
        {"name": "W. Clayton Jr.", "display": 4.4, "rs": 2.1},
        {"name": "B. Lopez", "display": 4.2, "rs": 4.1},
    ]},
    {"date": "2026-03-23", "rank": 2, "user": "yogurtslapss", "score": 79.56, "players": [
        {"name": "G. Payton II", "display": 5.0, "rs": 3.7},
        {"name": "S. Mamukelashvili", "display": 3.7, "rs": 4.2},
        {"name": "D. Gafford", "display": 3.6, "rs": 4.5},
        {"name": "D. Jenkins", "display": 4.0, "rs": 5.1},
        {"name": "J. McCain", "display": 4.2, "rs": 2.2},
    ]},
    {"date": "2026-03-23", "rank": 3, "user": "yvngandy", "score": 78.34, "players": [
        {"name": "D. Jenkins", "display": 4.6, "rs": 5.1},
        {"name": "G. Payton II", "display": 4.8, "rs": 3.7},
        {"name": "V. Edgecombe", "display": 2.6, "rs": 4.3},
        {"name": "A. Bailey", "display": 3.3, "rs": 5.1},
        {"name": "J. McCain", "display": 4.2, "rs": 2.2},
    ]},
    {"date": "2026-03-23", "rank": 4, "user": "burneracc0", "score": 78.25, "players": [
        {"name": "C. Sexton", "display": 3.8, "rs": 4.3},
        {"name": "A. Bailey", "display": 3.7, "rs": 5.1},
        {"name": "D. Gafford", "display": 3.6, "rs": 4.5},
        {"name": "Z. Williams", "display": 4.2, "rs": 1.9},
        {"name": "D. Jenkins", "display": 3.8, "rs": 5.1},
    ]},

    # ═══ Mar 22 ═══
    {"date": "2026-03-22", "rank": 1, "user": "axnz1", "score": 66.68, "players": [
        {"name": "M. Monk", "display": 4.0, "rs": 4.9},
        {"name": "B. Hyland", "display": 4.8, "rs": 3.2},
        {"name": "A. Dosunmu", "display": 3.0, "rs": 3.6},
        {"name": "J. Green", "display": 2.7, "rs": 3.6},
        {"name": "M. Raynaud", "display": 3.1, "rs": 3.6},
    ]},
    {"date": "2026-03-22", "rank": 2, "user": "limca", "score": 61.83, "players": [
        {"name": "M. Monk", "display": 4.0, "rs": 4.9},
        {"name": "B. Hyland", "display": 4.8, "rs": 3.2},
        {"name": "A. Dosunmu", "display": 3.0, "rs": 3.6},
        {"name": "D. Avdija", "display": 1.8, "rs": 5.0},
        {"name": "T. Vukcevic", "display": 4.2, "rs": 1.7},
    ]},
    {"date": "2026-03-22", "rank": 3, "user": "felix", "score": 61.81, "players": [
        {"name": "N. Jokic", "display": 2.0, "rs": 6.2},
        {"name": "C. Johnson", "display": 3.7, "rs": 3.8},
        {"name": "B. Hyland", "display": 4.6, "rs": 3.2},
        {"name": "Z. Williams", "display": 4.3, "rs": 2.9},
        {"name": "J. Watkins", "display": 4.2, "rs": 2.0},
    ]},
    {"date": "2026-03-22", "rank": 4, "user": "blueberry5", "score": 60.45, "players": [
        {"name": "J. Green", "display": 3.3, "rs": 3.6},
        {"name": "P. Achiuwa", "display": 3.8, "rs": 2.7},
        {"name": "M. Raynaud", "display": 3.5, "rs": 3.6},
        {"name": "Z. Williams", "display": 4.3, "rs": 2.9},
        {"name": "B. Hyland", "display": 4.2, "rs": 3.2},
    ]},

    # ═══ Mar 19 ═══
    {"date": "2026-03-19", "rank": 1, "user": "elaynablack", "score": 94.98, "players": [
        {"name": "L. Doncic", "display": 2.0, "rs": 9.1},
        {"name": "O. Ighodaro", "display": 4.6, "rs": 3.7},
        {"name": "J. Edwards", "display": 4.6, "rs": 5.3},
        {"name": "C. Williams", "display": 4.4, "rs": 4.0},
        {"name": "E. Harkless", "display": 4.2, "rs": 4.3},
    ]},
    {"date": "2026-03-19", "rank": 2, "user": "taad", "score": 94.54, "players": [
        {"name": "A. Bailey", "display": 4.0, "rs": 5.4},
        {"name": "C. Williams", "display": 4.8, "rs": 4.0},
        {"name": "J. Edwards", "display": 4.6, "rs": 5.3},
        {"name": "O. Ighodaro", "display": 4.2, "rs": 3.7},
        {"name": "M. Raynaud", "display": 3.2, "rs": 4.4},
    ]},
    {"date": "2026-03-19", "rank": 3, "user": "currylenda", "score": 92.66, "players": [
        {"name": "A. Bailey", "display": 4.0, "rs": 5.4},
        {"name": "C. Williams", "display": 4.8, "rs": 4.0},
        {"name": "E. Harkless", "display": 4.6, "rs": 4.3},
        {"name": "J. Edwards", "display": 4.4, "rs": 5.3},
        {"name": "D. Plowden", "display": 4.2, "rs": 2.2},
    ]},
    {"date": "2026-03-19", "rank": 4, "user": "bunabh", "score": 91.87, "players": [
        {"name": "L. Doncic", "display": 2.0, "rs": 9.1},
        {"name": "V. Edgecombe", "display": 2.8, "rs": 7.8},
        {"name": "Q. Grimes", "display": 3.2, "rs": 4.5},
        {"name": "J. Edwards", "display": 4.4, "rs": 5.3},
        {"name": "M. Raynaud", "display": 3.2, "rs": 4.4},
    ]},

    # ═══ Mar 16 ═══
    {"date": "2026-03-16", "rank": 1, "user": "roccozikarskyfan", "score": 68.34, "players": [
        {"name": "N. Alexander-Walker", "display": 2.9, "rs": 6.8},
        {"name": "K. Porzingis", "display": 2.8, "rs": 5.0},
        {"name": "G. Santos", "display": 4.1, "rs": 3.4},
        {"name": "G. Payton II", "display": 4.4, "rs": 2.9},
        {"name": "P. Spencer", "display": 4.2, "rs": 1.9},
    ]},
    {"date": "2026-03-16", "rank": 2, "user": "jasolace", "score": 67.06, "players": [
        {"name": "G. Santos", "display": 4.5, "rs": 3.4},
        {"name": "G. Payton II", "display": 4.8, "rs": 2.9},
        {"name": "N. Alexander-Walker", "display": 2.5, "rs": 6.8},
        {"name": "M. Buzelis", "display": 2.7, "rs": 5.0},
        {"name": "S. Castle", "display": 2.0, "rs": 3.7},
    ]},
    {"date": "2026-03-16", "rank": 3, "user": "kev33", "score": 67.03, "players": [
        {"name": "M. Buzelis", "display": 3.3, "rs": 5.0},
        {"name": "G. Payton II", "display": 4.8, "rs": 2.9},
        {"name": "G. Santos", "display": 4.1, "rs": 3.4},
        {"name": "T. Camara", "display": 3.0, "rs": 3.6},
        {"name": "N. Marshall", "display": 2.5, "rs": 4.8},
    ]},
    {"date": "2026-03-16", "rank": 4, "user": "1ggy", "score": 66.81, "players": [
        {"name": "N. Alexander-Walker", "display": 2.9, "rs": 6.8},
        {"name": "G. Santos", "display": 4.3, "rs": 3.4},
        {"name": "G. Payton II", "display": 4.6, "rs": 2.9},
        {"name": "O. Prosper", "display": 4.4, "rs": 1.7},
        {"name": "L. Miller", "display": 4.2, "rs": 2.7},
    ]},

    # ═══ Mar 15 ═══
    {"date": "2026-03-15", "rank": 1, "user": "cflinger", "score": 83.42, "players": [
        {"name": "B. Sensabaugh", "display": 4.1, "rs": 2.1},
        {"name": "C. Williams", "display": 4.8, "rs": 6.0},
        {"name": "G. Payton II", "display": 4.6, "rs": 4.3},
        {"name": "J. Edwards", "display": 4.4, "rs": 2.8},
        {"name": "Q. Post", "display": 4.2, "rs": 3.3},
    ]},
    {"date": "2026-03-15", "rank": 2, "user": "egreenwaldjr", "score": 83.09, "players": [
        {"name": "P. Achiuwa", "display": 4.1, "rs": 3.0},
        {"name": "C. Williams", "display": 4.8, "rs": 6.0},
        {"name": "D. DeRozan", "display": 2.5, "rs": 6.8},
        {"name": "O. Tshiebwe", "display": 4.4, "rs": 2.5},
        {"name": "K. Hayes", "display": 4.2, "rs": 3.3},
    ]},
    {"date": "2026-03-15", "rank": 3, "user": "silvehr", "score": 80.84, "players": [
        {"name": "I. Collier", "display": 3.4, "rs": 2.7},
        {"name": "G. Santos", "display": 4.3, "rs": 3.2},
        {"name": "J. Walker", "display": 3.9, "rs": 3.5},
        {"name": "C. Williams", "display": 4.4, "rs": 6.0},
        {"name": "G. Payton II", "display": 4.2, "rs": 4.3},
    ]},
    {"date": "2026-03-15", "rank": 4, "user": "unamazingausten", "score": 79.86, "players": [
        {"name": "C. Williams", "display": 5.0, "rs": 6.0},
        {"name": "B. Podziemski", "display": 3.2, "rs": 3.7},
        {"name": "B. Sensabaugh", "display": 3.7, "rs": 2.1},
        {"name": "G. Santos", "display": 3.9, "rs": 3.2},
        {"name": "G. Payton II", "display": 4.2, "rs": 4.3},
    ]},
]


def run():
    # Compute actual card_boost = display - slot_mult
    all_players = []
    all_drafts = []

    for draft in SCREENSHOT_DATA:
        draft_players = []
        for i, p in enumerate(draft["players"]):
            slot_mult = SLOT_MULTS[i] if i < 5 else 1.2
            card_boost = round(p["display"] - slot_mult, 2)
            player = {
                "name": p["name"],
                "rs": p["rs"],
                "display_mult": p["display"],
                "slot_mult": slot_mult,
                "card_boost": card_boost,
                "value": round(p["rs"] * p["display"], 2),  # RS × (slot + boost)
                "slot_index": i,
                "date": draft["date"],
                "draft_rank": draft["rank"],
                "draft_score": draft["score"],
            }
            all_players.append(player)
            draft_players.append(player)
        all_drafts.append({
            "date": draft["date"],
            "rank": draft["rank"],
            "score": draft["score"],
            "players": draft_players,
        })

    print(f"Drafts: {len(all_drafts)} across {len(set(d['date'] for d in all_drafts))} dates")
    print(f"Player-slots: {len(all_players)}")

    # ═══ CORRECTED BOOST ANALYSIS ═══
    boosts = [p["card_boost"] for p in all_players]
    rs_vals = [p["rs"] for p in all_players]
    values = [p["value"] for p in all_players]

    print(f"\n{'CORRECTED CARD BOOST (display - slot_mult)':=^70}")
    print(f"  Mean:   {statistics.mean(boosts):.2f}")
    print(f"  Median: {statistics.median(boosts):.1f}")
    print(f"  Min:    {min(boosts):.1f}")
    print(f"  Max:    {max(boosts):.1f}")
    for t in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        c = sum(1 for b in boosts if b >= t)
        print(f"    Boost >= {t:.1f}: {c:>3d} / {len(boosts)} ({c/len(boosts)*100:.0f}%)")

    print(f"\n{'REAL SCORE':=^70}")
    print(f"  Mean:   {statistics.mean(rs_vals):.2f}")
    print(f"  Median: {statistics.median(rs_vals):.1f}")
    for t in [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        c = sum(1 for r in rs_vals if r >= t)
        print(f"    RS >= {t:.1f}: {c:>3d} / {len(rs_vals)} ({c/len(rs_vals)*100:.0f}%)")

    print(f"\n{'PER-PLAYER VALUE (RS × total_mult)':=^70}")
    print(f"  Mean:   {statistics.mean(values):.1f}")
    print(f"  Median: {statistics.median(values):.1f}")

    # ═══ BY SLOT ═══
    print(f"\n{'STATS BY SLOT POSITION':=^70}")
    for slot_idx in range(5):
        sp = [p for p in all_players if p["slot_index"] == slot_idx]
        if not sp:
            continue
        avg_boost = statistics.mean([p["card_boost"] for p in sp])
        avg_rs = statistics.mean([p["rs"] for p in sp])
        avg_val = statistics.mean([p["value"] for p in sp])
        avg_display = statistics.mean([p["display_mult"] for p in sp])
        print(f"  Slot {slot_idx+1} ({SLOT_MULTS[slot_idx]}x): "
              f"boost={avg_boost:.2f}  RS={avg_rs:.2f}  "
              f"display={avg_display:.2f}  value={avg_val:.1f}")

    # ═══ DRAFT-LEVEL ANALYSIS ═══
    print(f"\n{'DRAFT-LEVEL COMPOSITION':=^70}")
    scores = [d["score"] for d in all_drafts]
    print(f"  Score: mean={statistics.mean(scores):.1f}  median={statistics.median(scores):.1f}")
    print(f"         min={min(scores):.1f}  max={max(scores):.1f}")

    for d in all_drafts:
        d["total_boost"] = sum(p["card_boost"] for p in d["players"])
        d["total_rs"] = sum(p["rs"] for p in d["players"])
        d["avg_boost"] = statistics.mean([p["card_boost"] for p in d["players"]])
        d["avg_rs"] = statistics.mean([p["rs"] for p in d["players"]])
        d["max_rs"] = max(p["rs"] for p in d["players"])
        d["min_rs"] = min(p["rs"] for p in d["players"])
        d["high_boost_3plus"] = sum(1 for p in d["players"] if p["card_boost"] >= 3.0)
        d["high_boost_2plus"] = sum(1 for p in d["players"] if p["card_boost"] >= 2.0)
        d["has_star"] = any(p["rs"] >= 6.0 for p in d["players"])

    print(f"\n  Per-draft averages:")
    print(f"    Avg total boost (sum of 5):  {statistics.mean([d['total_boost'] for d in all_drafts]):.1f}")
    print(f"    Avg total RS (sum of 5):     {statistics.mean([d['total_rs'] for d in all_drafts]):.1f}")
    print(f"    Avg per-player boost:        {statistics.mean([d['avg_boost'] for d in all_drafts]):.2f}")
    print(f"    Avg per-player RS:           {statistics.mean([d['avg_rs'] for d in all_drafts]):.2f}")
    print(f"    Avg max RS in draft:         {statistics.mean([d['max_rs'] for d in all_drafts]):.2f}")
    print(f"    Avg min RS in draft:         {statistics.mean([d['min_rs'] for d in all_drafts]):.2f}")
    print(f"    Drafts with a star (RS>=6):  {sum(1 for d in all_drafts if d['has_star'])} / {len(all_drafts)} ({sum(1 for d in all_drafts if d['has_star'])/len(all_drafts)*100:.0f}%)")
    print(f"    Avg players boost>=2.0:      {statistics.mean([d['high_boost_2plus'] for d in all_drafts]):.1f}")
    print(f"    Avg players boost>=3.0:      {statistics.mean([d['high_boost_3plus'] for d in all_drafts]):.1f}")

    # ═══ CORRELATION ═══
    print(f"\n{'WHAT DRIVES WINNING SCORE?':=^70}")
    if len(all_drafts) > 3:
        # Total boost vs score
        bs = [(d["total_boost"], d["score"]) for d in all_drafts]
        n = len(bs)
        mb = statistics.mean([b for b, _ in bs])
        ms = statistics.mean([s for _, s in bs])
        cov = sum((b - mb) * (s - ms) for b, s in bs) / n
        r_b = cov / (statistics.stdev([b for b, _ in bs]) * statistics.stdev([s for _, s in bs]))
        print(f"  Total Boost ↔ Score: r = {r_b:.3f}")

        # Total RS vs score
        rs_s = [(d["total_rs"], d["score"]) for d in all_drafts]
        mr = statistics.mean([r for r, _ in rs_s])
        cov_r = sum((r - mr) * (s - ms) for r, s in rs_s) / n
        r_rs = cov_r / (statistics.stdev([r for r, _ in rs_s]) * statistics.stdev([s for _, s in rs_s]))
        print(f"  Total RS ↔ Score:    r = {r_rs:.3f}")

        # Max RS vs score
        mx = [(d["max_rs"], d["score"]) for d in all_drafts]
        mmx = statistics.mean([m for m, _ in mx])
        cov_mx = sum((m - mmx) * (s - ms) for m, s in mx) / n
        r_mx = cov_mx / (statistics.stdev([m for m, _ in mx]) * statistics.stdev([s for _, s in mx]))
        print(f"  Max RS ↔ Score:      r = {r_mx:.3f}")

    # ═══ PLAYER FREQUENCY ═══
    print(f"\n{'MOST COMMON PLAYERS (across dates, not weighting repeats)':=^70}")
    # Count unique dates per player
    player_dates = defaultdict(set)
    player_data = defaultdict(list)
    for p in all_players:
        player_dates[p["name"]].add(p["date"])
        player_data[p["name"]].append(p)

    by_date_count = [(name, len(dates)) for name, dates in player_dates.items()]
    by_date_count.sort(key=lambda x: (-x[1], x[0]))

    for name, n_dates in by_date_count[:20]:
        entries = player_data[name]
        avg_boost = statistics.mean([e["card_boost"] for e in entries])
        avg_rs = statistics.mean([e["rs"] for e in entries])
        avg_val = statistics.mean([e["value"] for e in entries])
        print(f"    {name:25s}  dates={n_dates}  appearances={len(entries)}  "
              f"boost={avg_boost:.1f}  RS={avg_rs:.1f}  value={avg_val:.1f}")

    # ═══ ARCHETYPE ANALYSIS ═══
    print(f"\n{'PLAYER ARCHETYPES IN WINNING DRAFTS':=^70}")
    archetypes = defaultdict(list)
    for p in all_players:
        rs = p["rs"]
        boost = p["card_boost"]
        if rs >= 6.0:
            arch = "Star (RS>=6)"
        elif rs >= 4.0 and boost >= 2.0:
            arch = "Elite Hybrid (RS 4-6, Boost>=2)"
        elif rs >= 4.0:
            arch = "Solid Producer (RS 4-6, Boost<2)"
        elif rs >= 2.5 and boost >= 2.0:
            arch = "Boost Player (RS 2.5-4, Boost>=2)"
        elif rs >= 2.5:
            arch = "Role Player (RS 2.5-4, Boost<2)"
        else:
            arch = "Deep Bench (RS<2.5)"
        archetypes[arch].append(p)

    for arch in ["Star (RS>=6)", "Elite Hybrid (RS 4-6, Boost>=2)", "Solid Producer (RS 4-6, Boost<2)",
                 "Boost Player (RS 2.5-4, Boost>=2)", "Role Player (RS 2.5-4, Boost<2)", "Deep Bench (RS<2.5)"]:
        rows = archetypes.get(arch, [])
        if not rows:
            continue
        pct = len(rows) / len(all_players) * 100
        avg_boost = statistics.mean([r["card_boost"] for r in rows])
        avg_rs = statistics.mean([r["rs"] for r in rows])
        avg_val = statistics.mean([r["value"] for r in rows])
        print(f"  {arch:40s}: {len(rows):>3d} ({pct:4.1f}%)  boost={avg_boost:.1f}  RS={avg_rs:.1f}  value={avg_val:.1f}")

    # ═══ DRAFT PATTERNS ═══
    print(f"\n{'WINNING DRAFT PATTERNS':=^70}")
    # How many drafts have 0, 1, 2+ stars?
    star_counts = [sum(1 for p in d["players"] if p["rs"] >= 6.0) for d in all_drafts]
    for n_stars in [0, 1, 2]:
        c = sum(1 for s in star_counts if s >= n_stars)
        print(f"  Drafts with {n_stars}+ stars (RS>=6): {c} / {len(all_drafts)} ({c/len(all_drafts)*100:.0f}%)")

    # Verify scores by computing RS × display_mult
    print(f"\n{'SCORE VERIFICATION (RS × display_mult summed)':=^70}")
    for d in all_drafts[:5]:
        computed = sum(p["value"] for p in d["players"])
        print(f"  {d['date']} rank {d['rank']}: displayed={d['score']:.2f}  computed={computed:.2f}  diff={d['score']-computed:+.2f}")


if __name__ == "__main__":
    run()
