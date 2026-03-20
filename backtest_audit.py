"""
V50 Backtest Audit — Mar 5–19, 2026
Runs v50 model on each date, compares Starting 5 + Moonshot against actual leaderboard.
Identifies hit rate, missed winners, false positives, and strategic patterns.
"""
import csv
import sys
import json
import datetime
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
import api.index as idx

DATES = [
    "2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08", "2026-03-09",
    "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13", "2026-03-14",
    "2026-03-15", "2026-03-16", "2026-03-17", "2026-03-19",
]
SLOT_ORDER = {"2.0x": 1, "1.8x": 2, "1.6x": 3, "1.4x": 4, "1.2x": 5}


def load_projections(date_str):
    path = Path(f"data/predictions/{date_str}.csv")
    if not path.exists():
        return []
    seen, players = set(), []
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("player_name", "")
            if not name or name in seen:
                continue
            seen.add(name)
            pts = float(row.get("pts", 0) or 0)
            reb = float(row.get("reb", 0) or 0)
            ast = float(row.get("ast", 0) or 0)
            stl = float(row.get("stl", 0) or 0)
            blk = float(row.get("blk", 0) or 0)
            pred_min = float(row.get("pred_min", 0) or 0)
            boost = float(row.get("est_card_boost", 0) or 0)
            dfs = idx._dfs_score(pts, reb, ast, stl, blk, 0)
            cd = idx._cfg("real_score.compression_divisor", 4.5)
            cp = idx._cfg("real_score.compression_power", 0.78)
            rc = idx._cfg("real_score.rs_cap", 20.0)
            rs = min((dfs / cd) ** cp, rc)
            players.append({
                "name": name, "id": row.get("player_id", name),
                "team": row.get("team", ""), "pos": row.get("pos", ""),
                "rating": round(rs, 3), "est_mult": boost,
                "predMin": pred_min, "pts": pts, "reb": reb, "ast": ast,
                "stl": stl, "blk": blk,
                "season_pts": pts, "season_min": pred_min * 0.9,
                "recent_min": pred_min * 0.95,
                "player_variance": 0.3, "injury_status": "", "opp": "",
                "_cascade_bonus": 0,
                "chalk_ev": round(rs * (1.6 + boost), 2),
            })
    return players


def load_actuals(date_str):
    """Returns list of {name, rs, boost, value, rank} sorted by value desc."""
    path = Path(f"data/actuals/{date_str}.csv")
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("player_name", "")
            rs   = float(row.get("actual_rs", 0) or 0)
            boost = float(row.get("actual_card_boost", 0) or 0)
            val  = float(row.get("total_value", 0) or 0)
            if name and val > 0:
                rows.append({"name": name, "rs": rs, "boost": boost, "value": val})
    rows.sort(key=lambda x: x["value"], reverse=True)
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    return rows


# ── Run backtest ─────────────────────────────────────────────────────────────
config = idx._load_config()
print(f"\n{'='*76}")
print(f"  V50 AUDIT — Mar 5–19, 2026  (v{config.get('version','?')})")
print(f"  Scoring-dominant weights: pts={idx._cfg('real_score.dfs_weights',{}).get('pts')} "
      f"stl={idx._cfg('real_score.dfs_weights',{}).get('stl')} "
      f"blk={idx._cfg('real_score.dfs_weights',{}).get('blk')}")
print(f"{'='*76}\n")

# Tracking vars
total_picks = 0
total_hits = 0
top3_hits = 0
false_positives = []      # picks never in top-10 actual
missed_winners = []       # top-3 actual never picked
by_profile = defaultdict(lambda: {"picks": 0, "hits": 0})
star_picks = []
all_daily = []

for date_str in DATES:
    target = datetime.date.fromisoformat(date_str)
    orig_et = idx._et_date
    idx._et_date = lambda d=target: d

    projs = load_projections(date_str)
    actuals = load_actuals(date_str)

    if not projs:
        idx._et_date = orig_et
        continue

    try:
        chalk, upside, _pool = idx._build_lineups(projs)
    except Exception as e:
        print(f"  {date_str}: BUILD ERROR — {e}")
        idx._et_date = orig_et
        continue
    idx._et_date = orig_et

    actual_by_name = {a["name"]: a for a in actuals}
    actual_names_top10 = {a["name"] for a in actuals[:10]}
    actual_names_top5  = {a["name"] for a in actuals[:5]}
    actual_names_top3  = {a["name"] for a in actuals[:3]}

    s5_names = {p["name"] for p in chalk}
    ms_names = {p["name"] for p in upside}
    all_picks = list(chalk) + [p for p in upside if p["name"] not in s5_names]

    hits_s5 = s5_names & actual_names_top10
    hits_ms = ms_names & actual_names_top10
    hits_top5 = (s5_names | ms_names) & actual_names_top5
    hits_top3 = (s5_names | ms_names) & actual_names_top3

    date_hits = len(hits_s5) + len(hits_ms - s5_names)
    date_picks = len(set(p["name"] for p in all_picks))
    total_picks += date_picks
    total_hits += date_hits
    if hits_top3:
        top3_hits += 1

    # Stars picked
    for p in all_picks:
        if p.get("_is_star_anchor"):
            star_picks.append({"date": date_str, "name": p["name"],
                                "rs": p.get("rating",0), "boost": p.get("est_mult",0),
                                "in_s5": p["name"] in s5_names,
                                "actual": actual_by_name.get(p["name"])})

    # Profile tagging for picked players
    for p in all_picks:
        name = p["name"]
        boost = p.get("est_mult", 0)
        pts = p.get("pts", 0)
        # Profile buckets
        if pts >= 20 or p.get("_is_star_anchor"):
            profile = "star(20+ppg)"
        elif boost >= 2.5:
            profile = "max_boost(2.5+)"
        elif boost >= 1.5 and pts >= 12:
            profile = "scorer+boost(15+,1.5+)"
        elif boost >= 1.5:
            profile = "high_boost_role(1.5+)"
        else:
            profile = "mid_scorer(1-1.5x)"
        hit = name in actual_names_top10
        by_profile[profile]["picks"] += 1
        by_profile[profile]["hits"] += (1 if hit else 0)

    # False positives: picked but not in top 10 actual (use S5 only for FP tracking)
    for p in chalk:
        if p["name"] not in actual_names_top10 and actuals:
            false_positives.append({
                "date": date_str, "name": p["name"],
                "rs": p.get("rating",0), "boost": p.get("est_mult",0), "pts": p.get("pts",0)
            })

    # Missed winners: top-3 actual not in any of our picks
    for a in actuals[:5]:
        if a["name"] not in (s5_names | ms_names):
            missed_winners.append({
                "date": date_str, "name": a["name"], "rank": a["rank"],
                "rs": a["rs"], "boost": a["boost"], "value": a["value"]
            })

    all_daily.append({
        "date": date_str, "hits": date_hits, "picks": date_picks,
        "hits_s5": len(hits_s5), "hits_ms": len(hits_ms),
        "hits_top5": len(hits_top5), "hits_top3": len(hits_top3),
        "chalk": chalk, "upside": upside, "actuals": actuals,
        "s5_names": s5_names, "ms_names": ms_names,
    })

    # Per-day output
    emoji = "🟢" if date_hits >= 3 else ("🟡" if date_hits >= 1 else "🔴")
    print(f"{emoji} {date_str}  hits={date_hits}/{date_picks}  top5={len(hits_top5)}  top3={len(hits_top3)}")

    # Starting 5 vs actuals
    print(f"   S5: ", end="")
    for p in sorted(chalk, key=lambda x: SLOT_ORDER.get(x.get("slot",""), 9)):
        name = p["name"]
        actual = actual_by_name.get(name)
        star = "★" if p.get("_is_star_anchor") else ""
        if actual:
            rank_str = f"#{actual['rank']}"
            marker = "✓" if actual["rank"] <= 10 else "✗"
        else:
            rank_str = "—"
            marker = "✗"
        print(f"{name}({rank_str}{marker}{star})", end="  ")
    print()

    # Moonshot vs actuals (unique picks only)
    ms_unique = [p for p in upside if p["name"] not in s5_names]
    if ms_unique:
        print(f"   MS: ", end="")
        for p in ms_unique:
            name = p["name"]
            actual = actual_by_name.get(name)
            star = "★" if p.get("_is_star_anchor") else ""
            if actual:
                rank_str = f"#{actual['rank']}"
                marker = "✓" if actual["rank"] <= 10 else "✗"
            else:
                rank_str = "—"
                marker = "✗"
            print(f"{name}({rank_str}{marker}{star})", end="  ")
        print()

    # Top 5 actual for context
    if actuals:
        print(f"   Actual top 5: ", end="")
        for a in actuals[:5]:
            in_picks = "★" if a["name"] in (s5_names | ms_names) else ""
            print(f"{a['name']}(RS {a['rs']:.1f} +{a['boost']:.1f}x ={a['value']:.0f}){in_picks}", end="  ")
        print()
    print()

# ── Summary ─────────────────────────────────────────────────────────────────
hit_rate = total_hits / total_picks * 100 if total_picks else 0
dates_with_top3 = sum(1 for d in all_daily if d["hits_top3"] > 0)

print(f"{'='*76}")
print(f"  OVERALL: {total_hits}/{total_picks} picks hit top-10 ({hit_rate:.0f}%)")
print(f"  Dates with ≥1 top-3 actual hit: {dates_with_top3}/{len(all_daily)}")
print()

print("  HIT RATE BY PLAYER PROFILE:")
for profile, data in sorted(by_profile.items(), key=lambda x: -x[1]["hits"]):
    n, h = data["picks"], data["hits"]
    pct = h/n*100 if n else 0
    print(f"    {profile:<32s}  {h}/{n} ({pct:.0f}%)")
print()

print("  STAR ANCHOR PICKS:")
if star_picks:
    for s in star_picks:
        a = s.get("actual")
        where = "S5" if s["in_s5"] else "MS"
        if a:
            print(f"    {s['date']}  {s['name']:<22s} RS:{s['rs']:.1f} +{s['boost']:.1f}x  [{where}]"
                  f"  → Actual #{a['rank']} RS:{a['rs']:.1f} val:{a['value']:.0f}")
        else:
            print(f"    {s['date']}  {s['name']:<22s} RS:{s['rs']:.1f} +{s['boost']:.1f}x  [{where}]"
                  f"  → Not in actuals")
else:
    print("    (none)")
print()

print("  TOP MISSED WINNERS (top-5 actual never in our picks):")
missed_counts = defaultdict(int)
for m in missed_winners:
    missed_counts[m["name"]] += 1
# Show top repeated misses first
sorted_misses = sorted(
    [(n, c, [m for m in missed_winners if m["name"] == n][0]) for n, c in missed_counts.items()],
    key=lambda x: -x[1]
)[:20]
for name, count, example in sorted_misses:
    print(f"    {name:<26s} missed {count}x  "
          f"(e.g. {example['date']} RS:{example['rs']:.1f} +{example['boost']:.1f}x val:{example['value']:.0f})")
print()

print("  MOST COMMON FALSE POSITIVES (S5 picks never in top-10 actual):")
fp_counts = defaultdict(int)
for fp in false_positives:
    fp_counts[fp["name"]] += 1
sorted_fps = sorted(fp_counts.items(), key=lambda x: -x[1])[:15]
for name, count in sorted_fps:
    examples = [fp for fp in false_positives if fp["name"] == name]
    ex = examples[0]
    print(f"    {name:<26s} false positive {count}x  "
          f"(RS:{ex['rs']:.1f} +{ex['boost']:.1f}x {ex['pts']:.0f}pts)")
print()

print("  BOOST RANGE OF ACTUAL TOP-3 WINNERS:")
boost_ranges = defaultdict(int)
for d in all_daily:
    for a in d["actuals"][:3]:
        b = a["boost"]
        if b >= 2.5:    boost_ranges["2.5+ (max)"] += 1
        elif b >= 1.5:  boost_ranges["1.5–2.4 (high)"] += 1
        elif b >= 0.8:  boost_ranges["0.8–1.4 (mid)"] += 1
        else:           boost_ranges["0–0.7 (star)"] += 1
total_w = sum(boost_ranges.values())
for k, v in sorted(boost_ranges.items()):
    print(f"    {k:<22s}  {v}/{total_w} ({v/total_w*100:.0f}%)")
print()

print("  RS RANGE OF ACTUAL TOP-3 WINNERS:")
rs_ranges = defaultdict(int)
for d in all_daily:
    for a in d["actuals"][:3]:
        r = a["rs"]
        if r >= 7:     rs_ranges["7.0+ (elite)"] += 1
        elif r >= 5:   rs_ranges["5.0–6.9 (good)"] += 1
        elif r >= 3.5: rs_ranges["3.5–4.9 (role)"] += 1
        else:          rs_ranges["<3.5 (lottery)"] += 1
total_r = sum(rs_ranges.values())
for k, v in sorted(rs_ranges.items()):
    print(f"    {k:<22s}  {v}/{total_r} ({v/total_r*100:.0f}%)")
print()

print(f"{'='*76}\n")
