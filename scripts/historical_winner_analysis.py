#!/usr/bin/env python3
"""
Historical Winner Analysis — What do Real Sports daily winners actually look like?

Reads top_performers.csv (880 rows, 63 dates), fetches ESPN box scores for each
date, joins with predictions where available, and outputs a comprehensive analysis
of what winning players' actual stat lines look like.

Key questions answered:
  - How many actual points do daily winners score?
  - What's the RS-to-points relationship?
  - What % of winners scored 10+, 15+, 20+ pts?
  - Do we systematically miss a certain player archetype?
  - Where does our model fail — RS, boost, or player selection?

Usage:
  python scripts/historical_winner_analysis.py [--fetch] [--output FILE]

  --fetch   Actually call ESPN API to backfill box scores (slow, ~63 API calls)
            Without this flag, uses cached data only.
  --output  Path for enriched CSV (default: data/analysis/winner_profiles.csv)
"""

import csv
import json
import os
import sys
import time
import requests
import statistics
from collections import defaultdict
from datetime import date as date_cls
from pathlib import Path

# ── ESPN API ──────────────────────────────────────────────────────────────────

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CACHE_DIR = Path("data/analysis")
BOX_CACHE_DIR = CACHE_DIR / "box_cache"


def _espn_get(url):
    try:
        r = requests.get(url, timeout=15)
        if not r.ok:
            return {}
        return r.json()
    except (requests.RequestException, ValueError):
        return {}


def fetch_games_for_date(d: date_cls):
    """Fetch game IDs for a specific date from ESPN scoreboard."""
    date_str = d.strftime("%Y%m%d")
    data = _espn_get(f"{ESPN}/scoreboard?dates={date_str}")
    games = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        home = away = None
        for cd in comp.get("competitors", []):
            t = {"abbr": cd.get("team", {}).get("abbreviation", "")}
            if cd["homeAway"] == "home":
                home = t
            else:
                away = t
        games.append({
            "gameId": ev["id"],
            "label": f"{away['abbr'] if away else '?'} @ {home['abbr'] if home else '?'}",
        })
    return games


def fetch_box_scores_for_date(d: date_cls, use_cache=True):
    """Fetch all player box scores for a given date.
    Returns dict: player_name -> {pts, reb, ast, stl, blk, min, team}
    """
    date_str = d.isoformat()
    cache_file = BOX_CACHE_DIR / f"{date_str}.json"

    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    games = fetch_games_for_date(d)
    if not games:
        return {}

    player_stats = {}
    want_labels = {"MIN", "PTS", "REB", "AST", "STL", "BLK"}

    for game in games:
        game_id = game.get("gameId")
        if not game_id:
            continue
        data = _espn_get(f"{ESPN}/summary?event={game_id}")
        if not data:
            continue

        for team_block in data.get("boxscore", {}).get("players", []):
            team_abbr = team_block.get("team", {}).get("abbreviation", "")
            stats_sections = team_block.get("statistics", [])
            if not stats_sections:
                continue
            labels = stats_sections[0].get("labels", [])
            idx_map = {l: i for i, l in enumerate(labels) if l in want_labels}

            for ath in stats_sections[0].get("athletes", []):
                name = ath.get("athlete", {}).get("displayName", "")
                if not name:
                    continue
                vals = ath.get("stats", [])
                pdata = {"team": team_abbr}
                for lbl, key in [("PTS", "pts"), ("REB", "reb"), ("AST", "ast"),
                                 ("STL", "stl"), ("BLK", "blk"), ("MIN", "min")]:
                    if lbl in idx_map and idx_map[lbl] < len(vals):
                        try:
                            raw = vals[idx_map[lbl]]
                            pdata[key] = float(raw.split(":")[0]) if ":" in str(raw) else float(raw)
                        except (ValueError, TypeError):
                            pdata[key] = 0.0
                if pdata.get("pts") is not None:
                    player_stats[name] = pdata

    # Cache the result
    BOX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(player_stats, f, indent=2)

    return player_stats


# ── Data Loaders ──────────────────────────────────────────────────────────────

def load_top_performers():
    """Load top_performers.csv into a list of dicts."""
    rows = []
    with open("data/top_performers.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_predictions_for_date(date_str):
    """Load predictions CSV for a date, return dict: player_name -> prediction row."""
    path = Path(f"data/predictions/{date_str}.csv")
    if not path.exists():
        return {}
    preds = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Only take slate-level chalk/upside predictions (not per-game)
            if row.get("scope") == "slate":
                name = row.get("player_name", "")
                if name:
                    # Keep the first occurrence (highest slot)
                    key = f"{name}_{row.get('lineup_type', '')}"
                    preds[key] = row
                    # Also store by name for simple lookup
                    if name not in preds:
                        preds[name] = row
    return preds


def load_most_popular_for_date(date_str):
    """Load most_popular CSV for a date, return dict: player_name -> {draft_count, rank}."""
    path = Path(f"data/most_popular/{date_str}.csv")
    if not path.exists():
        return {}
    pop = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("player", row.get("player_name", ""))
            if name:
                pop[name] = {
                    "draft_count": int(row.get("draft_count", 0) or 0),
                    "rank": int(row.get("rank", 0) or 0),
                }
    return pop


# ── Name matching ─────────────────────────────────────────────────────────────

def _normalize_name(name):
    """Normalize player name for fuzzy matching."""
    return name.strip().lower().replace(".", "").replace("'", "").replace("-", " ")


def find_box_score(player_name, box_scores):
    """Find a player's box score, handling name variations."""
    if player_name in box_scores:
        return box_scores[player_name]

    norm = _normalize_name(player_name)
    for bname, bdata in box_scores.items():
        if _normalize_name(bname) == norm:
            return bdata

    # Try last name match as fallback
    parts = player_name.split()
    if len(parts) >= 2:
        last = parts[-1].lower()
        matches = [(k, v) for k, v in box_scores.items() if k.split()[-1].lower() == last]
        if len(matches) == 1:
            return matches[0][1]

    return None


# ── Main Analysis ─────────────────────────────────────────────────────────────

def run_analysis(do_fetch=False):
    """Run the full historical winner analysis."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BOX_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    top_performers = load_top_performers()
    print(f"Loaded {len(top_performers)} top performer rows")

    # Group by date
    by_date = defaultdict(list)
    for row in top_performers:
        by_date[row["date"]].append(row)

    dates = sorted(by_date.keys())
    print(f"Spanning {len(dates)} unique dates: {dates[0]} to {dates[-1]}")

    # ── Phase 1: Fetch box scores ──
    enriched_rows = []
    dates_fetched = 0
    dates_cached = 0
    dates_failed = 0
    unmatched_players = []

    for date_str in dates:
        d = date_cls.fromisoformat(date_str)
        cache_file = BOX_CACHE_DIR / f"{date_str}.json"

        if cache_file.exists():
            with open(cache_file) as f:
                box_scores = json.load(f)
            dates_cached += 1
        elif do_fetch:
            print(f"  Fetching ESPN box scores for {date_str}...", end=" ", flush=True)
            box_scores = fetch_box_scores_for_date(d, use_cache=False)
            if box_scores:
                dates_fetched += 1
                print(f"OK ({len(box_scores)} players)")
            else:
                dates_failed += 1
                print("FAILED")
            time.sleep(1.5)  # Rate limit
        else:
            box_scores = {}

        # Load predictions if available
        predictions = load_predictions_for_date(date_str)
        popularity = load_most_popular_for_date(date_str)

        # Enrich each top performer
        for row in by_date[date_str]:
            player_name = row["player_name"]
            box = find_box_score(player_name, box_scores) if box_scores else None
            pred = predictions.get(player_name, {})
            pop = popularity.get(player_name, {})

            enriched = {
                "date": date_str,
                "player_name": player_name,
                "team": row.get("team", ""),
                "actual_rs": float(row.get("actual_rs", 0) or 0),
                "actual_card_boost": float(row.get("actual_card_boost", 0) or 0),
                "drafts": int(row.get("drafts", 0) or 0),
                "total_value": float(row.get("total_value", 0) or 0),
                # ESPN box score stats
                "actual_pts": box.get("pts", "") if box else "",
                "actual_reb": box.get("reb", "") if box else "",
                "actual_ast": box.get("ast", "") if box else "",
                "actual_stl": box.get("stl", "") if box else "",
                "actual_blk": box.get("blk", "") if box else "",
                "actual_min": box.get("min", "") if box else "",
                # Our predictions (if available)
                "predicted_rs": pred.get("predicted_rs", ""),
                "predicted_pts": pred.get("pts", ""),
                "predicted_min": pred.get("pred_min", ""),
                "est_card_boost": pred.get("est_card_boost", ""),
                "lineup_type": pred.get("lineup_type", ""),
                "slot": pred.get("slot", ""),
                # Popularity
                "draft_count": pop.get("draft_count", ""),
                "popularity_rank": pop.get("rank", ""),
            }
            enriched_rows.append(enriched)

            if box_scores and box is None:
                unmatched_players.append(f"{date_str}: {player_name}")

    print(f"\nFetch summary: {dates_cached} cached, {dates_fetched} freshly fetched, {dates_failed} failed")
    if unmatched_players:
        print(f"Unmatched players ({len(unmatched_players)}):")
        for p in unmatched_players[:20]:
            print(f"  {p}")
        if len(unmatched_players) > 20:
            print(f"  ... and {len(unmatched_players) - 20} more")

    # ── Phase 2: Write enriched CSV ──
    output_path = CACHE_DIR / "winner_profiles.csv"
    fieldnames = [
        "date", "player_name", "team", "actual_rs", "actual_card_boost",
        "drafts", "total_value",
        "actual_pts", "actual_reb", "actual_ast", "actual_stl", "actual_blk", "actual_min",
        "predicted_rs", "predicted_pts", "predicted_min", "est_card_boost",
        "lineup_type", "slot",
        "draft_count", "popularity_rank",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)
    print(f"\nWrote {len(enriched_rows)} enriched rows to {output_path}")

    # ── Phase 3: Statistical analysis ──
    print("\n" + "=" * 80)
    print("HISTORICAL WINNER ANALYSIS — What do Real Sports daily winners look like?")
    print("=" * 80)

    # Filter to rows with actual box scores
    with_stats = [r for r in enriched_rows if r["actual_pts"] != ""]
    if not with_stats:
        print("\nNo box score data available. Run with --fetch to download ESPN data.")
        return enriched_rows

    pts_list = [float(r["actual_pts"]) for r in with_stats]
    rs_list = [float(r["actual_rs"]) for r in with_stats]
    boost_list = [float(r["actual_card_boost"]) for r in with_stats]
    min_list = [float(r["actual_min"]) for r in with_stats if r["actual_min"]]
    tv_list = [float(r["total_value"]) for r in with_stats]

    print(f"\n{'SCORING PROFILE':=^60}")
    print(f"  Rows with box scores: {len(with_stats)} / {len(enriched_rows)}")
    print(f"\n  ACTUAL POINTS SCORED:")
    print(f"    Mean:   {statistics.mean(pts_list):.1f}")
    print(f"    Median: {statistics.median(pts_list):.1f}")
    print(f"    Std:    {statistics.stdev(pts_list):.1f}" if len(pts_list) > 1 else "")
    print(f"    Min:    {min(pts_list):.0f}")
    print(f"    Max:    {max(pts_list):.0f}")

    # Scoring brackets
    brackets = [(5, "5+"), (8, "8+"), (10, "10+"), (12, "12+"), (15, "15+"), (20, "20+"), (25, "25+")]
    print(f"\n  SCORING DISTRIBUTION:")
    for threshold, label in brackets:
        count = sum(1 for p in pts_list if p >= threshold)
        pct = count / len(pts_list) * 100
        print(f"    {label:>4s} pts: {count:>4d} / {len(pts_list)} ({pct:.1f}%)")

    print(f"\n  ACTUAL MINUTES:")
    if min_list:
        print(f"    Mean:   {statistics.mean(min_list):.1f}")
        print(f"    Median: {statistics.median(min_list):.1f}")
        min_brackets = [(10, "10+"), (15, "15+"), (20, "20+"), (25, "25+"), (30, "30+")]
        for threshold, label in min_brackets:
            count = sum(1 for m in min_list if m >= threshold)
            pct = count / len(min_list) * 100
            print(f"    {label:>4s} min: {count:>4d} / {len(min_list)} ({pct:.1f}%)")

    print(f"\n{'REAL SCORE vs POINTS':=^60}")
    print(f"  Actual RS:    mean={statistics.mean(rs_list):.2f}, median={statistics.median(rs_list):.1f}")
    print(f"  Card Boost:   mean={statistics.mean(boost_list):.2f}, median={statistics.median(boost_list):.1f}")
    print(f"  Total Value:  mean={statistics.mean(tv_list):.1f}, median={statistics.median(tv_list):.1f}")

    # RS-to-PTS correlation
    if len(with_stats) > 2:
        # Simple correlation
        n = len(pts_list)
        mean_pts = statistics.mean(pts_list)
        mean_rs = statistics.mean(rs_list)
        cov = sum((p - mean_pts) * (r - mean_rs) for p, r in zip(pts_list, rs_list)) / n
        std_pts = statistics.stdev(pts_list)
        std_rs = statistics.stdev(rs_list)
        corr = cov / (std_pts * std_rs) if std_pts > 0 and std_rs > 0 else 0
        print(f"\n  PTS ↔ RS correlation:   r = {corr:.3f}")

        # PTS-to-Boost correlation
        mean_boost = statistics.mean(boost_list)
        cov_b = sum((p - mean_pts) * (b - mean_boost) for p, b in zip(pts_list, boost_list)) / n
        std_boost = statistics.stdev(boost_list)
        corr_b = cov_b / (std_pts * std_boost) if std_pts > 0 and std_boost > 0 else 0
        print(f"  PTS ↔ Boost correlation: r = {corr_b:.3f}")

        # RS-to-Boost correlation
        cov_rb = sum((r - mean_rs) * (b - mean_boost) for r, b in zip(rs_list, boost_list)) / n
        corr_rb = cov_rb / (std_rs * std_boost) if std_rs > 0 and std_boost > 0 else 0
        print(f"  RS  ↔ Boost correlation: r = {corr_rb:.3f}")

    print(f"\n{'WINNER ARCHETYPE BREAKDOWN':=^60}")

    # Classify each winner into archetypes
    archetypes = defaultdict(list)
    for r in with_stats:
        pts = float(r["actual_pts"])
        rs = float(r["actual_rs"])
        boost = float(r["actual_card_boost"])
        tv = float(r["total_value"])

        if rs >= 5.0 and boost >= 2.0:
            arch = "Elite Hybrid (RS≥5 + Boost≥2)"
        elif rs >= 5.0 and boost < 2.0:
            arch = "Star Performer (RS≥5, low boost)"
        elif rs >= 3.0 and boost >= 2.5:
            arch = "Boost Leverage (RS 3-5, Boost≥2.5)"
        elif rs < 3.0 and boost >= 2.5:
            arch = "Pure Boost (RS<3, Boost≥2.5)"
        else:
            arch = "Mid-Tier (RS 3-5, Boost<2.5)"
        archetypes[arch].append(r)

    for arch in ["Elite Hybrid (RS≥5 + Boost≥2)", "Star Performer (RS≥5, low boost)",
                 "Boost Leverage (RS 3-5, Boost≥2.5)", "Mid-Tier (RS 3-5, Boost<2.5)",
                 "Pure Boost (RS<3, Boost≥2.5)"]:
        rows = archetypes.get(arch, [])
        if not rows:
            print(f"\n  {arch}: 0 players")
            continue
        pct = len(rows) / len(with_stats) * 100
        a_pts = [float(r["actual_pts"]) for r in rows]
        a_rs = [float(r["actual_rs"]) for r in rows]
        a_boost = [float(r["actual_card_boost"]) for r in rows]
        a_tv = [float(r["total_value"]) for r in rows]
        a_min = [float(r["actual_min"]) for r in rows if r["actual_min"]]
        print(f"\n  {arch}: {len(rows)} players ({pct:.1f}%)")
        print(f"    Avg PTS:   {statistics.mean(a_pts):.1f}  (median {statistics.median(a_pts):.0f})")
        print(f"    Avg RS:    {statistics.mean(a_rs):.2f}")
        print(f"    Avg Boost: {statistics.mean(a_boost):.2f}")
        print(f"    Avg TV:    {statistics.mean(a_tv):.1f}")
        if a_min:
            print(f"    Avg MIN:   {statistics.mean(a_min):.1f}")

    # ── Per-date analysis: top 5 winners ──
    print(f"\n{'TOP 5 WINNERS PER DATE (by total_value)':=^60}")

    dates_with_stats = sorted(set(r["date"] for r in with_stats))
    daily_top5_pts = []
    daily_min_pts = []

    for date_str in dates_with_stats:
        day_rows = sorted(
            [r for r in with_stats if r["date"] == date_str],
            key=lambda x: float(x["total_value"]),
            reverse=True
        )[:5]
        if len(day_rows) < 5:
            continue

        day_pts = [float(r["actual_pts"]) for r in day_rows]
        daily_top5_pts.extend(day_pts)
        daily_min_pts.append(min(day_pts))

    if daily_top5_pts:
        print(f"\n  Across top-5 winners per date ({len(daily_top5_pts)} player-slots):")
        print(f"    Mean PTS:     {statistics.mean(daily_top5_pts):.1f}")
        print(f"    Median PTS:   {statistics.median(daily_top5_pts):.0f}")
        print(f"    Min PTS in winning top-5: {min(daily_top5_pts):.0f}")
        print(f"\n  MINIMUM points scored by any player in a winning top-5:")
        print(f"    Mean of daily minimums:   {statistics.mean(daily_min_pts):.1f}")
        print(f"    Median of daily minimums: {statistics.median(daily_min_pts):.0f}")
        for threshold in [3, 5, 8, 10, 12]:
            count = sum(1 for m in daily_min_pts if m >= threshold)
            pct = count / len(daily_min_pts) * 100
            print(f"    Min pts >= {threshold:>2d}: {count:>3d} / {len(daily_min_pts)} dates ({pct:.1f}%)")

    # ── Prediction accuracy for dates we have predictions ──
    with_preds = [r for r in with_stats if r["predicted_rs"]]
    if with_preds:
        print(f"\n{'PREDICTION vs ACTUALS (dates with predictions)':=^60}")
        print(f"  Matched rows: {len(with_preds)}")

        rs_errors = []
        boost_errors = []
        pts_errors = []
        in_our_picks = 0

        for r in with_preds:
            actual_rs = float(r["actual_rs"])
            pred_rs = float(r["predicted_rs"])
            rs_errors.append(actual_rs - pred_rs)

            if r["est_card_boost"]:
                actual_boost = float(r["actual_card_boost"])
                pred_boost = float(r["est_card_boost"])
                boost_errors.append(actual_boost - pred_boost)

            if r["predicted_pts"]:
                actual_pts = float(r["actual_pts"])
                pred_pts = float(r["predicted_pts"])
                pts_errors.append(actual_pts - pred_pts)

            if r["slot"]:
                in_our_picks += 1

        print(f"  Winners we actually drafted: {in_our_picks} / {len(with_preds)} ({in_our_picks/len(with_preds)*100:.0f}%)")

        if rs_errors:
            print(f"\n  RS Error (actual - predicted):")
            print(f"    Mean:  {statistics.mean(rs_errors):+.2f} ({'under' if statistics.mean(rs_errors) > 0 else 'over'}-projecting)")
            print(f"    MAE:   {statistics.mean([abs(e) for e in rs_errors]):.2f}")

        if boost_errors:
            print(f"\n  Boost Error (actual - predicted):")
            print(f"    Mean:  {statistics.mean(boost_errors):+.2f}")
            print(f"    MAE:   {statistics.mean([abs(e) for e in boost_errors]):.2f}")

        if pts_errors:
            print(f"\n  Points Error (actual - predicted):")
            print(f"    Mean:  {statistics.mean(pts_errors):+.2f}")
            print(f"    MAE:   {statistics.mean([abs(e) for e in pts_errors]):.2f}")

    # ── Key finding: the minimum viable scoring profile ──
    print(f"\n{'KEY FINDINGS — MINIMUM VIABLE WINNER':=^60}")

    # What's the 10th percentile of points among winners?
    pts_sorted = sorted(pts_list)
    p10 = pts_sorted[int(len(pts_sorted) * 0.10)]
    p25 = pts_sorted[int(len(pts_sorted) * 0.25)]
    p50 = pts_sorted[int(len(pts_sorted) * 0.50)]
    print(f"  Points percentiles among ALL top performers:")
    print(f"    P10: {p10:.0f}  P25: {p25:.0f}  P50: {p50:.0f}")

    rs_sorted = sorted(rs_list)
    p10_rs = rs_sorted[int(len(rs_sorted) * 0.10)]
    p25_rs = rs_sorted[int(len(rs_sorted) * 0.25)]
    print(f"  RS percentiles among ALL top performers:")
    print(f"    P10: {p10_rs:.1f}  P25: {p25_rs:.1f}  P50: {statistics.median(rs_list):.1f}")

    boost_sorted = sorted(boost_list)
    p10_b = boost_sorted[int(len(boost_sorted) * 0.10)]
    p25_b = boost_sorted[int(len(boost_sorted) * 0.25)]
    print(f"  Boost percentiles among ALL top performers:")
    print(f"    P10: {p10_b:.1f}  P25: {p25_b:.1f}  P50: {statistics.median(boost_list):.1f}")

    if min_list:
        min_sorted = sorted(min_list)
        p10_m = min_sorted[int(len(min_sorted) * 0.10)]
        p25_m = min_sorted[int(len(min_sorted) * 0.25)]
        print(f"  Minutes percentiles among ALL top performers:")
        print(f"    P10: {p10_m:.0f}  P25: {p25_m:.0f}  P50: {statistics.median(min_list):.0f}")

    print(f"\n  RECOMMENDATION:")
    print(f"    Based on {len(with_stats)} historical winners:")
    print(f"    - Minimum pts floor for moonshot should be ~{p10:.0f} (P10)")
    print(f"    - Production anchor threshold should be ~{p25:.0f} pts (P25)")
    print(f"    - 50% of winners scored >= {p50:.0f} pts")
    if min_list:
        print(f"    - 90% of winners played >= {p10_m:.0f} minutes")

    return enriched_rows


if __name__ == "__main__":
    do_fetch = "--fetch" in sys.argv
    run_analysis(do_fetch=do_fetch)
