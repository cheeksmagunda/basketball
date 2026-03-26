#!/usr/bin/env python3
"""
Winning Drafts Analysis — What do the users who WIN drafts actually do?

Analyzes actual winning draft lineups (top 1-4 finishers per date) to understand:
  - What total scores win? What's the target?
  - What boost levels do winners draft?
  - What RS levels do their picks achieve?
  - How many "stars" vs "role players" do winners have?
  - What's the slot assignment strategy?

Data sources:
  1. Screenshots manually parsed below (Mar 15, 16, 19, 22, 23)
  2. data/winning_drafts/*.csv (archived)
  3. data/top_performers.csv (for cross-reference)
"""

import csv
import statistics
from collections import defaultdict, Counter
from pathlib import Path


# ── Manually parsed from screenshots ──────────────────────────────────────────

SCREENSHOT_DATA = [
    # Mar 23
    {"date": "2026-03-23", "rank": 1, "user": "qwizzywizzy", "score": 80.71, "players": [
        {"name": "D. Jenkins", "boost": 4.6, "rs": 5.1},
        {"name": "G. Santos", "boost": 4.3, "rs": 3.1},
        {"name": "A. Bailey", "boost": 3.5, "rs": 5.1},
        {"name": "W. Clayton Jr.", "boost": 4.4, "rs": 2.1},
        {"name": "B. Lopez", "boost": 4.2, "rs": 4.1},
    ]},
    {"date": "2026-03-23", "rank": 2, "user": "yogurtslapss", "score": 79.56, "players": [
        {"name": "G. Payton II", "boost": 5.0, "rs": 3.7},
        {"name": "S. Mamukelashvili", "boost": 3.7, "rs": 4.2},
        {"name": "D. Gafford", "boost": 3.6, "rs": 4.5},
        {"name": "D. Jenkins", "boost": 4.0, "rs": 5.1},
        {"name": "J. McCain", "boost": 4.2, "rs": 2.2},
    ]},
    {"date": "2026-03-23", "rank": 3, "user": "yvngandy", "score": 78.34, "players": [
        {"name": "D. Jenkins", "boost": 4.6, "rs": 5.1},
        {"name": "G. Payton II", "boost": 4.8, "rs": 3.7},
        {"name": "V. Edgecombe", "boost": 2.6, "rs": 4.3},
        {"name": "A. Bailey", "boost": 3.3, "rs": 5.1},
        {"name": "J. McCain", "boost": 4.2, "rs": 2.2},
    ]},
    {"date": "2026-03-23", "rank": 4, "user": "burneracc0", "score": 78.25, "players": [
        {"name": "C. Sexton", "boost": 3.8, "rs": 4.3},
        {"name": "A. Bailey", "boost": 3.7, "rs": 5.1},
        {"name": "D. Gafford", "boost": 3.6, "rs": 4.5},
        {"name": "Z. Williams", "boost": 4.2, "rs": 1.9},
        {"name": "D. Jenkins", "boost": 3.8, "rs": 5.1},
    ]},

    # Mar 22
    {"date": "2026-03-22", "rank": 1, "user": "axnz1", "score": 66.68, "players": [
        {"name": "M. Monk", "boost": 4.0, "rs": 4.9},
        {"name": "B. Hyland", "boost": 4.8, "rs": 3.2},
        {"name": "A. Dosunmu", "boost": 3.0, "rs": 3.6},
        {"name": "J. Green", "boost": 2.7, "rs": 3.6},
        {"name": "M. Raynaud", "boost": 3.1, "rs": 3.6},
    ]},
    {"date": "2026-03-22", "rank": 2, "user": "limca", "score": 61.83, "players": [
        {"name": "M. Monk", "boost": 4.0, "rs": 4.9},
        {"name": "B. Hyland", "boost": 4.8, "rs": 3.2},
        {"name": "A. Dosunmu", "boost": 3.0, "rs": 3.6},
        {"name": "D. Avdija", "boost": 1.8, "rs": 5.0},
        {"name": "T. Vukcevic", "boost": 4.2, "rs": 1.7},
    ]},
    {"date": "2026-03-22", "rank": 3, "user": "felix", "score": 61.81, "players": [
        {"name": "N. Jokic", "boost": 2.0, "rs": 6.2},
        {"name": "C. Johnson", "boost": 3.7, "rs": 3.8},
        {"name": "B. Hyland", "boost": 4.6, "rs": 3.2},
        {"name": "Z. Williams", "boost": 4.3, "rs": 2.9},
        {"name": "J. Watkins", "boost": 4.2, "rs": 2.0},
    ]},
    {"date": "2026-03-22", "rank": 4, "user": "blueberry5", "score": 60.45, "players": [
        {"name": "J. Green", "boost": 3.3, "rs": 3.6},
        {"name": "P. Achiuwa", "boost": 3.8, "rs": 2.7},
        {"name": "M. Raynaud", "boost": 3.5, "rs": 3.6},
        {"name": "Z. Williams", "boost": 4.3, "rs": 2.9},
        {"name": "B. Hyland", "boost": 4.2, "rs": 3.2},
    ]},

    # Mar 19
    {"date": "2026-03-19", "rank": 1, "user": "elaynablack", "score": 94.98, "players": [
        {"name": "L. Doncic", "boost": 2.0, "rs": 9.1},
        {"name": "O. Ighodaro", "boost": 4.6, "rs": 3.7},
        {"name": "J. Edwards", "boost": 4.6, "rs": 5.3},
        {"name": "C. Williams", "boost": 4.4, "rs": 4.0},
        {"name": "E. Harkless", "boost": 4.2, "rs": 4.3},
    ]},
    {"date": "2026-03-19", "rank": 2, "user": "taad", "score": 94.54, "players": [
        {"name": "A. Bailey", "boost": 4.0, "rs": 5.4},
        {"name": "C. Williams", "boost": 4.8, "rs": 4.0},
        {"name": "J. Edwards", "boost": 4.6, "rs": 5.3},
        {"name": "O. Ighodaro", "boost": 4.2, "rs": 3.7},
        {"name": "M. Raynaud", "boost": 3.2, "rs": 4.4},
    ]},
    {"date": "2026-03-19", "rank": 3, "user": "currylenda", "score": 92.66, "players": [
        {"name": "A. Bailey", "boost": 4.0, "rs": 5.4},
        {"name": "C. Williams", "boost": 4.8, "rs": 4.0},
        {"name": "E. Harkless", "boost": 4.6, "rs": 4.3},
        {"name": "J. Edwards", "boost": 4.4, "rs": 5.3},
        {"name": "D. Plowden", "boost": 4.2, "rs": 2.2},
    ]},
    {"date": "2026-03-19", "rank": 4, "user": "bunabh", "score": 91.87, "players": [
        {"name": "L. Doncic", "boost": 2.0, "rs": 9.1},
        {"name": "V. Edgecombe", "boost": 2.8, "rs": 7.8},
        {"name": "Q. Grimes", "boost": 3.2, "rs": 4.5},
        {"name": "J. Edwards", "boost": 4.4, "rs": 5.3},
        {"name": "M. Raynaud", "boost": 3.2, "rs": 4.4},
    ]},

    # Mar 16
    {"date": "2026-03-16", "rank": 1, "user": "roccozikarskyfan", "score": 68.34, "players": [
        {"name": "N. Alexander-Walker", "boost": 2.9, "rs": 6.8},
        {"name": "K. Porzingis", "boost": 2.8, "rs": 5.0},
        {"name": "G. Santos", "boost": 4.1, "rs": 3.4},
        {"name": "G. Payton II", "boost": 4.4, "rs": 2.9},
        {"name": "P. Spencer", "boost": 4.2, "rs": 1.9},
    ]},
    {"date": "2026-03-16", "rank": 2, "user": "jasolace", "score": 67.06, "players": [
        {"name": "G. Santos", "boost": 4.5, "rs": 3.4},
        {"name": "G. Payton II", "boost": 4.8, "rs": 2.9},
        {"name": "N. Alexander-Walker", "boost": 2.5, "rs": 6.8},
        {"name": "M. Buzelis", "boost": 2.7, "rs": 5.0},
        {"name": "S. Castle", "boost": 2.0, "rs": 3.7},
    ]},
    {"date": "2026-03-16", "rank": 3, "user": "kev33", "score": 67.03, "players": [
        {"name": "M. Buzelis", "boost": 3.3, "rs": 5.0},
        {"name": "G. Payton II", "boost": 4.8, "rs": 2.9},
        {"name": "G. Santos", "boost": 4.1, "rs": 3.4},
        {"name": "T. Camara", "boost": 3.0, "rs": 3.6},
        {"name": "N. Marshall", "boost": 2.5, "rs": 4.8},
    ]},
    {"date": "2026-03-16", "rank": 4, "user": "1ggy", "score": 66.81, "players": [
        {"name": "N. Alexander-Walker", "boost": 2.9, "rs": 6.8},
        {"name": "G. Santos", "boost": 4.3, "rs": 3.4},
        {"name": "G. Payton II", "boost": 4.6, "rs": 2.9},
        {"name": "O. Prosper", "boost": 4.4, "rs": 1.7},
        {"name": "L. Miller", "boost": 4.2, "rs": 2.7},
    ]},

    # Mar 15
    {"date": "2026-03-15", "rank": 1, "user": "cflinger", "score": 83.42, "players": [
        {"name": "B. Sensabaugh", "boost": 4.1, "rs": 2.1},
        {"name": "C. Williams", "boost": 4.8, "rs": 6.0},
        {"name": "G. Payton II", "boost": 4.6, "rs": 4.3},
        {"name": "J. Edwards", "boost": 4.4, "rs": 2.8},
        {"name": "Q. Post", "boost": 4.2, "rs": 3.3},
    ]},
    {"date": "2026-03-15", "rank": 2, "user": "egreenwaldjr", "score": 83.09, "players": [
        {"name": "P. Achiuwa", "boost": 4.1, "rs": 3.0},
        {"name": "C. Williams", "boost": 4.8, "rs": 6.0},
        {"name": "D. DeRozan", "boost": 2.5, "rs": 6.8},
        {"name": "O. Tshiebwe", "boost": 4.4, "rs": 2.5},
        {"name": "K. Hayes", "boost": 4.2, "rs": 3.3},
    ]},
    {"date": "2026-03-15", "rank": 3, "user": "silvehr", "score": 80.84, "players": [
        {"name": "I. Collier", "boost": 3.4, "rs": 2.7},
        {"name": "G. Santos", "boost": 4.3, "rs": 3.2},
        {"name": "J. Walker", "boost": 3.9, "rs": 3.5},
        {"name": "C. Williams", "boost": 4.4, "rs": 6.0},
        {"name": "G. Payton II", "boost": 4.2, "rs": 4.3},
    ]},
    {"date": "2026-03-15", "rank": 4, "user": "unamazingausten", "score": 79.86, "players": [
        {"name": "C. Williams", "boost": 5.0, "rs": 6.0},
        {"name": "B. Podziemski", "boost": 3.2, "rs": 3.7},
        {"name": "B. Sensabaugh", "boost": 3.7, "rs": 2.1},
        {"name": "G. Santos", "boost": 3.9, "rs": 3.2},
        {"name": "G. Payton II", "boost": 4.2, "rs": 4.3},
    ]},
]


def load_archived_winning_drafts():
    """Load winning drafts from data/winning_drafts/*.csv."""
    drafts = []
    for f in sorted(Path("data/winning_drafts").glob("*.csv")):
        date_str = f.stem
        rows = []
        with open(f, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)
        if not rows:
            continue

        # Group by drafter
        by_drafter = defaultdict(list)
        for row in rows:
            key = row.get("drafter_label") or row.get("winner_rank") or "unknown"
            by_drafter[key].append(row)

        for drafter, players in by_drafter.items():
            score = 0
            try:
                score = float(players[0].get("total_score") or 0)
            except (ValueError, TypeError):
                pass
            rank = 0
            try:
                rank = int(players[0].get("winner_rank") or 0)
            except (ValueError, TypeError):
                pass

            draft = {
                "date": date_str,
                "rank": rank,
                "user": drafter,
                "score": score,
                "players": [],
                "source": "archive",
            }
            for p in players:
                boost = 0
                try:
                    boost = float(p.get("card_boost") or p.get("slot_mult") or 0)
                except (ValueError, TypeError):
                    pass
                rs = 0
                try:
                    rs = float(p.get("actual_rs") or 0)
                except (ValueError, TypeError):
                    pass
                draft["players"].append({
                    "name": p.get("player_name", ""),
                    "boost": boost,
                    "rs": rs,
                })
            if draft["players"]:
                drafts.append(draft)
    return drafts


def run():
    # Combine all data
    all_drafts = []
    screenshot_dates = set()
    for d in SCREENSHOT_DATA:
        d["source"] = "screenshot"
        all_drafts.append(d)
        screenshot_dates.add(d["date"])

    archived = load_archived_winning_drafts()
    for d in archived:
        if d["date"] not in screenshot_dates:  # don't double-count
            all_drafts.append(d)

    print(f"Total winning drafts: {len(all_drafts)}")
    print(f"Dates covered: {len(set(d['date'] for d in all_drafts))}")
    print(f"From screenshots: {sum(1 for d in all_drafts if d.get('source') == 'screenshot')}")
    print(f"From archives: {sum(1 for d in all_drafts if d.get('source') == 'archive')}")

    # Filter to drafts with actual data (boost > 0 or rs > 0)
    valid = [d for d in all_drafts if any(p["boost"] > 0 or p["rs"] > 0 for p in d["players"])]
    print(f"Drafts with usable data: {len(valid)}")

    # ── Total Scores ──
    scores = [d["score"] for d in valid if d["score"] > 0]
    if scores:
        print(f"\n{'WINNING DRAFT SCORES':=^70}")
        print(f"  Mean:   {statistics.mean(scores):.1f}")
        print(f"  Median: {statistics.median(scores):.1f}")
        print(f"  Min:    {min(scores):.1f}")
        print(f"  Max:    {max(scores):.1f}")
        print(f"  Std:    {statistics.stdev(scores):.1f}" if len(scores) > 1 else "")
        # By rank
        for rank in [1, 2, 3, 4]:
            rank_scores = [d["score"] for d in valid if d["rank"] == rank and d["score"] > 0]
            if rank_scores:
                print(f"  Rank {rank}: mean={statistics.mean(rank_scores):.1f}  median={statistics.median(rank_scores):.1f}  n={len(rank_scores)}")

    # ── Player-level analysis ──
    all_players = []
    for d in valid:
        for i, p in enumerate(d["players"]):
            if p["boost"] > 0 or p["rs"] > 0:
                all_players.append({
                    **p,
                    "date": d["date"],
                    "draft_rank": d["rank"],
                    "draft_score": d["score"],
                    "slot_index": i,  # 0 = highest slot (2.0x)
                })

    print(f"\n{'PLAYER-LEVEL STATS ({len(all_players)} player-slots)':=^70}")

    boosts = [p["boost"] for p in all_players if p["boost"] > 0]
    rs_vals = [p["rs"] for p in all_players if p["rs"] > 0]

    if boosts:
        print(f"\n  CARD BOOST:")
        print(f"    Mean:   {statistics.mean(boosts):.2f}")
        print(f"    Median: {statistics.median(boosts):.1f}")
        print(f"    Min:    {min(boosts):.1f}")
        print(f"    Max:    {max(boosts):.1f}")
        for t in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
            c = sum(1 for b in boosts if b >= t)
            print(f"    Boost >= {t:.1f}: {c:>4d} / {len(boosts)} ({c/len(boosts)*100:.0f}%)")

    if rs_vals:
        print(f"\n  REAL SCORE:")
        print(f"    Mean:   {statistics.mean(rs_vals):.2f}")
        print(f"    Median: {statistics.median(rs_vals):.1f}")
        for t in [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]:
            c = sum(1 for r in rs_vals if r >= t)
            print(f"    RS >= {t:.1f}: {c:>4d} / {len(rs_vals)} ({c/len(rs_vals)*100:.0f}%)")

    # ── Value per player: RS × (slot + boost) ──
    print(f"\n  PER-PLAYER VALUE (RS × (slot + boost)):")
    slot_mults = [2.0, 1.8, 1.6, 1.4, 1.2]
    values = []
    for p in all_players:
        if p["rs"] > 0 and p["boost"] > 0:
            slot = slot_mults[p["slot_index"]] if p["slot_index"] < 5 else 1.2
            val = p["rs"] * (slot + p["boost"])
            values.append({"val": val, **p})
    if values:
        vals = [v["val"] for v in values]
        print(f"    Mean:   {statistics.mean(vals):.1f}")
        print(f"    Median: {statistics.median(vals):.1f}")

    # ── Boost distribution by slot ──
    print(f"\n  BOOST BY SLOT POSITION:")
    for slot_idx in range(5):
        slot_boosts = [p["boost"] for p in all_players if p["slot_index"] == slot_idx and p["boost"] > 0]
        slot_rs = [p["rs"] for p in all_players if p["slot_index"] == slot_idx and p["rs"] > 0]
        if slot_boosts:
            print(f"    Slot {slot_idx+1} ({slot_mults[slot_idx]}x): boost={statistics.mean(slot_boosts):.2f}  RS={statistics.mean(slot_rs):.2f}  n={len(slot_boosts)}")

    # ── How many high-boost players per winning draft? ──
    print(f"\n  HIGH-BOOST PLAYERS PER WINNING DRAFT:")
    for threshold in [2.0, 2.5, 3.0, 3.5, 4.0]:
        counts = []
        for d in valid:
            c = sum(1 for p in d["players"] if p["boost"] >= threshold)
            counts.append(c)
        if counts:
            print(f"    Boost >= {threshold:.1f}: mean={statistics.mean(counts):.1f}  median={statistics.median(counts):.0f}  min={min(counts)}  max={max(counts)}")

    # ── How many low-RS players do winners tolerate? ──
    print(f"\n  LOW-RS TOLERANCE IN WINNING DRAFTS:")
    for threshold in [2.0, 2.5, 3.0]:
        counts = []
        for d in valid:
            rs_players = [p for p in d["players"] if p["rs"] > 0]
            c = sum(1 for p in rs_players if p["rs"] < threshold)
            counts.append(c)
        if counts:
            print(f"    Players with RS < {threshold:.1f}: mean={statistics.mean(counts):.1f}  max={max(counts)}")

    # ── Most drafted players across winning lineups ──
    print(f"\n  MOST COMMON PLAYERS IN WINNING DRAFTS:")
    name_counter = Counter()
    name_data = defaultdict(list)
    for p in all_players:
        # Normalize short names
        name_counter[p["name"]] += 1
        name_data[p["name"]].append(p)

    for name, count in name_counter.most_common(25):
        entries = name_data[name]
        avg_boost = statistics.mean([e["boost"] for e in entries if e["boost"] > 0]) if any(e["boost"] > 0 for e in entries) else 0
        avg_rs = statistics.mean([e["rs"] for e in entries if e["rs"] > 0]) if any(e["rs"] > 0 for e in entries) else 0
        dates = len(set(e["date"] for e in entries))
        print(f"    {name:25s}  appearances={count:>2d}  dates={dates:>2d}  avg_boost={avg_boost:.1f}  avg_rs={avg_rs:.1f}")

    # ── THE KEY INSIGHT: What does a winning 5-player draft look like? ──
    print(f"\n{'THE WINNING DRAFT BLUEPRINT':=^70}")

    # Average draft composition
    draft_profiles = []
    for d in valid:
        if len(d["players"]) < 5:
            continue
        players_sorted = sorted(d["players"], key=lambda x: x["boost"], reverse=True)
        boosts_sorted = [p["boost"] for p in players_sorted]
        rs_sorted = [p["rs"] for p in players_sorted]
        avg_boost = statistics.mean([p["boost"] for p in d["players"] if p["boost"] > 0]) if any(p["boost"] > 0 for p in d["players"]) else 0
        avg_rs = statistics.mean([p["rs"] for p in d["players"] if p["rs"] > 0]) if any(p["rs"] > 0 for p in d["players"]) else 0
        total_boost = sum(p["boost"] for p in d["players"])
        total_rs = sum(p["rs"] for p in d["players"])
        draft_profiles.append({
            "score": d["score"],
            "avg_boost": avg_boost,
            "avg_rs": avg_rs,
            "total_boost": total_boost,
            "total_rs": total_rs,
            "min_boost": min(b for b in boosts_sorted if b > 0) if any(b > 0 for b in boosts_sorted) else 0,
            "max_boost": max(boosts_sorted),
            "min_rs": min(r for r in rs_sorted if r > 0) if any(r > 0 for r in rs_sorted) else 0,
            "max_rs": max(rs_sorted),
            "high_boost_count": sum(1 for p in d["players"] if p["boost"] >= 3.5),
        })

    if draft_profiles:
        print(f"\n  Across {len(draft_profiles)} winning drafts:")
        print(f"  Avg total boost (sum of 5): {statistics.mean([d['total_boost'] for d in draft_profiles]):.1f}")
        print(f"  Avg total RS (sum of 5):    {statistics.mean([d['total_rs'] for d in draft_profiles]):.1f}")
        print(f"  Avg per-player boost:       {statistics.mean([d['avg_boost'] for d in draft_profiles]):.2f}")
        print(f"  Avg per-player RS:           {statistics.mean([d['avg_rs'] for d in draft_profiles]):.2f}")
        print(f"  Avg min boost in draft:      {statistics.mean([d['min_boost'] for d in draft_profiles]):.2f}")
        print(f"  Avg max RS in draft:         {statistics.mean([d['max_rs'] for d in draft_profiles]):.2f}")
        print(f"  Avg players with boost>=3.5: {statistics.mean([d['high_boost_count'] for d in draft_profiles]):.1f}")

        # Correlation: does higher total boost → higher score?
        if len(draft_profiles) > 3:
            boost_scores = [(d["total_boost"], d["score"]) for d in draft_profiles if d["score"] > 0]
            rs_scores = [(d["total_rs"], d["score"]) for d in draft_profiles if d["score"] > 0]
            if boost_scores:
                n = len(boost_scores)
                mb = statistics.mean([b for b, _ in boost_scores])
                ms = statistics.mean([s for _, s in boost_scores])
                cov = sum((b - mb) * (s - ms) for b, s in boost_scores) / n
                std_b = statistics.stdev([b for b, _ in boost_scores])
                std_s = statistics.stdev([s for _, s in boost_scores])
                r_boost = cov / (std_b * std_s) if std_b > 0 and std_s > 0 else 0
                print(f"\n  Total Boost ↔ Draft Score correlation: r = {r_boost:.3f}")
            if rs_scores:
                n = len(rs_scores)
                mr = statistics.mean([r for r, _ in rs_scores])
                ms = statistics.mean([s for _, s in rs_scores])
                cov = sum((r - mr) * (s - ms) for r, s in rs_scores) / n
                std_r = statistics.stdev([r for r, _ in rs_scores])
                std_s = statistics.stdev([s for _, s in rs_scores])
                r_rs = cov / (std_r * std_s) if std_r > 0 and std_s > 0 else 0
                print(f"  Total RS ↔ Draft Score correlation:    r = {r_rs:.3f}")

    # ── Summary recommendation ──
    print(f"\n{'ACTIONABLE FINDINGS':=^70}")
    if draft_profiles and boosts:
        p25_boost = sorted(boosts)[int(len(boosts) * 0.25)]
        p50_boost = statistics.median(boosts)
        p25_rs = sorted(rs_vals)[int(len(rs_vals) * 0.25)] if rs_vals else 0
        p50_rs = statistics.median(rs_vals) if rs_vals else 0
        print(f"  Target draft score: {statistics.median([d['score'] for d in draft_profiles if d['score'] > 0]):.0f}+")
        print(f"  Per-player boost: median={p50_boost:.1f}, P25={p25_boost:.1f}")
        print(f"  Per-player RS: median={p50_rs:.1f}, P25={p25_rs:.1f}")
        print(f"  High-boost (>=3.5) players per draft: {statistics.mean([d['high_boost_count'] for d in draft_profiles]):.1f} out of 5")
        print(f"  Winners tolerate RS as low as {statistics.mean([d['min_rs'] for d in draft_profiles]):.1f} if boost is high enough")


if __name__ == "__main__":
    run()
