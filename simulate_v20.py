"""
Simulate v23 model config predictions for past dates.
Uses player stats recorded in prediction CSVs and re-applies current model formulas.
No PuLP needed — greedy optimizer with constraint enforcement.
"""
import csv
import math
import sys
from collections import defaultdict

# ─────────────────────────────────────────────
# V23 CONFIG — Lookup + Sigmoid Tier Boost Model
# ─────────────────────────────────────────────
# Player overrides from real ownership/actuals data (boost is stable per-player ±0.1x)
PLAYER_OVERRIDES = {
    "Aaron Nesmith": 1.9, "Ace Bailey": 2.1, "Al Horford": 2.0,
    "Amen Thompson": 0.6, "Andre Drummond": 2.3, "Anthony Edwards": 0.2,
    "Bam Adebayo": 0.6, "Brook Lopez": 3.0, "Bryce McGowens": 3.0,
    "Cade Cunningham": 0.2, "Cameron Payne": 3.0, "Clint Capela": 3.0,
    "Cody Williams": 3.0, "Collin Sexton": 2.0, "Cooper Flagg": 0.7,
    "De'Aaron Fox": 0.6, "De'Anthony Melton": 1.9, "Derik Queen": 1.4,
    "Derrick White": 0.7, "Devin Carter": 3.0, "Donovan Clingan": 1.0,
    "Grant Williams": 2.8, "Gui Santos": 2.6, "Isaiah Collier": 1.5,
    "Jalen Johnson": 0.3, "Jarace Walker": 2.3, "Jaylen Brown": 0.3,
    "Jordan Miller": 2.5, "Julian Reese": 3.0, "Julius Randle": 0.6,
    "Kam Jones": 3.0, "Kel'el Ware": 1.4, "Kevin Durant": 0.5,
    "Klay Thompson": 2.6, "Kon Knueppel": 0.8, "Kyle Filipowski": 2.1,
    "Kyle Kuzma": 1.7, "LaMelo Ball": 0.6, "Leonard Miller": 3.0,
    "Luka Dončić": 0.0, "Luke Kennard": 3.0, "Matas Buzelis": 1.4,
    "Maxime Raynaud": 2.1, "Myles Turner": 1.6, "Noah Clowney": 2.0,
    "OG Anunoby": 1.1, "Ousmane Dieng": 3.0, "Pat Spencer": 3.0,
    "Precious Achiuwa": 2.2, "Reed Sheppard": 1.5, "Robert Williams III": 2.3,
    "Ron Harper Jr.": 3.0, "Royce O'Neale": 2.0, "Rudy Gobert": 1.1,
    "Russell Westbrook": 1.1, "Scottie Barnes": 0.5, "Simone Fontecchio": 3.0,
    "Tristan da Silva": 2.8, "Tyler Herro": 0.8, "Victor Wembanyama": 0.3,
}

# Sigmoid tier estimation params
SIG_CEILING = 3.0
SIG_RANGE = 2.8
SIG_MIDPOINT = 12.0
SIG_SCALE = 4.0
BIG_MARKET_DISCOUNT = 0.15
BIG_MARKET_TEAMS = {"LAL","GS","GSW","BOS","NY","NYK","PHI","MIA","DEN","LAC","CHI"}
BOOST_CEILING = 3.0
BOOST_FLOOR = 0.2

CHALK_SEASON_MIN_FLOOR = 25.0
CHALK_RECENT_MIN_FLOOR = 15.0
CHALK_BOOST_CAP = 2.5
CHALK_MAX_STARS = 1
CHALK_STAR_BOOST_THRESHOLD = 0.8
LEVERAGE_TOP_SLOTS = 1          # v21: 2→1, allows star anchor in top slot
LEVERAGE_BOOST_THRESHOLD = 1.5

MOONSHOT_MIN_SEASON = 20.0
MOONSHOT_MIN_RECENT = 20.0
MOONSHOT_MIN_BOOST = 1.5
MOONSHOT_MIN_RATING = 2.0
MOONSHOT_MAX_PER_TEAM = 2       # v22: reverted to 2, user preference
MOONSHOT_WILDCARD_BOOST = 2.0   # v22: 2.5→2.0, boost estimation runs 30-50% below actuals
MOONSHOT_WILDCARD_MIN = 8.0     # v22: 15→8, matches effective_gate floor
MOONSHOT_DEV_PTS_FLOOR = 8.0    # v21: suppress dev bonus for aging bench fodder
BOOST_LEVERAGE_POWER = 1.6
BIG_POS_EFFICIENCY = 0.85
VARIANCE_PENALTY = 0.3
MAX_CENTERS = 2

SLOT_MULTS = [2.0, 1.8, 1.6, 1.4, 1.2]
AVG_SLOT = 1.6

# Approximate win% for dev team bonus (current mid-season standings, used as proxy)
WIN_PCT = {
    "SA": 0.34, "MEM": 0.35, "WAS": 0.10, "NO": 0.20, "TOR": 0.28, "CHI": 0.32,
    "CHA": 0.36, "UTAH": 0.28, "BKN": 0.24, "DET": 0.38, "PHX": 0.37, "ORL": 0.49,
    "POR": 0.30, "SAC": 0.25, "ATL": 0.42, "DAL": 0.48, "HOU": 0.50, "IND": 0.53,
    "LAC": 0.36, "MIL": 0.54, "MIA": 0.43, "GS": 0.44, "GSW": 0.44,
    "LAL": 0.54, "DEN": 0.55, "MIN": 0.57, "OKC": 0.65, "CLE": 0.63,
    "BOS": 0.70, "NY": 0.52, "NYK": 0.52, "PHI": 0.35,
}


def card_boost(pts, pred_min, team, player_name=None):
    """v23: Lookup + sigmoid tier boost model."""
    # Layer 1: Player lookup
    if player_name and player_name in PLAYER_OVERRIDES:
        return PLAYER_OVERRIDES[player_name]
    # Layer 2: Sigmoid tier estimation from PPG
    sigmoid_val = 1.0 / (1.0 + math.exp(-(pts - SIG_MIDPOINT) / SIG_SCALE))
    boost = SIG_CEILING - SIG_RANGE * sigmoid_val
    if team in BIG_MARKET_TEAMS:
        boost -= BIG_MARKET_DISCOUNT
    return round(max(BOOST_FLOOR, min(BOOST_CEILING, boost)), 1)


def dev_team_bonus(team, pts=0):
    """Dev bonus only applies when pts >= dev_team_pts_floor (v21)."""
    if pts < MOONSHOT_DEV_PTS_FLOOR:
        return 1.0
    wp = WIN_PCT.get(team, 0.5)
    return max(1.0, 1.0 + max(0, 0.5 - wp))


def pos_group(pos):
    """Match actual api/index.py POS_GROUPS: PF→F, not C."""
    p = pos.upper()
    if p == "C":
        return "C"
    if p in ("PF", "SF", "F"):
        return "F"
    return "G"


def is_center(pos):
    return pos.upper() == "C"  # PF is F group per actual code


def chalk_ev(rating, boost, reliability=1.0):
    capped = min(boost, CHALK_BOOST_CAP)
    return rating * (AVG_SLOT + capped) * reliability


def moonshot_ev(rating, boost, pos, team, pts=0, stl=0, blk=0, pred_min=28):
    if boost < MOONSHOT_MIN_BOOST:
        return 0.0
    # consistency base (variance proxy from pts variability — approximate)
    variance = min(0.5, max(0.0, (rating - 2.0) / 8.0))  # rough proxy
    consistency = rating * max(0.75, 1.0 - variance * VARIANCE_PENALTY)

    team_bonus = dev_team_bonus(team, pts=pts)  # v21: pts-gated dev bonus
    pos_eff = BIG_POS_EFFICIENCY if is_center(pos) else 1.0

    # defensive bonus: estimate from stl+blk relative to minutes
    stl_blk_per36 = (stl + blk) / max(pred_min, 1) * 36
    def_bonus = min(0.20, max(0.0, (stl_blk_per36 - 1.5) * 0.10))

    boost_leverage = boost ** BOOST_LEVERAGE_POWER
    adj_ceiling = consistency * team_bonus * pos_eff * (1 + def_bonus) * boost_leverage
    return adj_ceiling * (AVG_SLOT + boost)


def load_players(csvpath):
    """Load all unique players from a date's prediction CSV."""
    players = {}  # player_id → dict
    with open(csvpath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["player_id"]
            if pid in players:
                # keep the slate-scope version if available (has card boost)
                if row["scope"] == "slate":
                    players[pid] = row
            else:
                players[pid] = row
    return list(players.values())


def pick_starting5(pool):
    """
    Pick slate Starting 5 under v20 chalk rules.
    Constraints: ≤2 per team, ≤1 per (team, pos_group), ≤1 star (boost < 0.8),
    force ≥2 contrarians (boost ≥ 1.5) in first 2 slots.
    """
    # Compute card boost + chalk_ev for all eligible players
    candidates = []
    for p in pool:
        pts = float(p["pts"])
        pred_min = float(p["pred_min"])
        rating = float(p["predicted_rs"])
        team = p["team"]
        pos = p["pos"]
        if rating < 2.0:
            continue
        # Minutes gate: use pred_min as proxy for season_min (conservative)
        if pred_min < CHALK_SEASON_MIN_FLOOR:
            continue
        boost = card_boost(pts, pred_min, team, player_name=p.get("player_name"))
        ev = chalk_ev(rating, boost)
        candidates.append({
            "name": p["player_name"],
            "team": team,
            "pos": pos,
            "rating": rating,
            "boost": boost,
            "ev": ev,
            "pred_min": pred_min,
            "pts": pts,
            "reb": p.get("reb", "?"),
            "ast": p.get("ast", "?"),
        })

    # Sort by ev descending
    candidates.sort(key=lambda x: x["ev"], reverse=True)

    # Greedy pick with constraints
    lineup = []
    team_count = defaultdict(int)
    team_pos_used = set()
    star_count = 0

    # Two passes: first fill contrarian slots (boost >= 1.5), then fill remaining
    contrarians = [c for c in candidates if c["boost"] >= LEVERAGE_BOOST_THRESHOLD]
    others = [c for c in candidates if c["boost"] < LEVERAGE_BOOST_THRESHOLD]

    def can_add(p, lineup, team_count, team_pos_used, star_count):
        team = p["team"]
        pg = pos_group(p["pos"])
        key = (team, pg)
        is_star = p["boost"] < CHALK_STAR_BOOST_THRESHOLD
        if team_count[team] >= 2:
            return False
        if key in team_pos_used:
            return False
        if is_star and star_count >= CHALK_MAX_STARS:
            return False
        return True

    # Force first 2 contrarians in top 2 slots
    slots_filled = []
    for p in contrarians:
        if len(slots_filled) >= LEVERAGE_TOP_SLOTS:
            break
        pg = pos_group(p["pos"])
        key = (p["team"], pg)
        is_star = p["boost"] < CHALK_STAR_BOOST_THRESHOLD
        if can_add(p, lineup, team_count, team_pos_used, star_count):
            slots_filled.append(p)
            team_count[p["team"]] += 1
            team_pos_used.add(key)
            if is_star:
                star_count += 1

    # Fill remaining 3 slots from all candidates by ev
    remaining_slots = 5 - len(slots_filled)
    used_ids = {p["name"] for p in slots_filled}
    rest_candidates = [c for c in candidates if c["name"] not in used_ids]
    rest_candidates.sort(key=lambda x: x["ev"], reverse=True)

    for p in rest_candidates:
        if len(lineup) + len(slots_filled) >= 5:
            break
        pg = pos_group(p["pos"])
        key = (p["team"], pg)
        is_star = p["boost"] < CHALK_STAR_BOOST_THRESHOLD
        if can_add(p, lineup, team_count, team_pos_used, star_count):
            lineup.append(p)
            team_count[p["team"]] += 1
            team_pos_used.add(key)
            if is_star:
                star_count += 1

    # Combine: top contrarians in first slots, rest by ev
    full_lineup = slots_filled + lineup
    # Sort full lineup by ev for slot assignment
    full_lineup.sort(key=lambda x: x["ev"], reverse=True)
    return full_lineup[:5]


def pick_moonshot(pool):
    """
    Pick slate Moonshot under v21 rules.
    Constraints: ≤3 per team (v21 stacking), ≤1 per (team, pos_group), ≤2 centers.
    Gates: A (regular: season>=20 & recent>=20), B (cascade: pred>=28 & cascade>=10),
           C (wildcard: boost>=2.5 & pred_min>=15) — v21 addition.
    Dev team bonus suppressed for pts < 8 (v21).
    """
    candidates = []
    for p in pool:
        pts = float(p["pts"])
        pred_min = float(p["pred_min"])
        rating = float(p["predicted_rs"])
        team = p["team"]
        pos = p["pos"]
        stl = float(p.get("stl", 0))
        blk = float(p.get("blk", 0))

        boost = card_boost(pts, pred_min, team, player_name=p.get("player_name"))
        if boost < MOONSHOT_MIN_BOOST:
            continue

        # Gate A: proven regular
        gate_a = (pred_min >= MOONSHOT_MIN_SEASON)
        # Gate B: spot-starter via cascade (approximate — no cascade data in CSV)
        gate_b = False
        # Gate C (v21): ultra-high boost wildcard — bypasses minutes floor
        gate_c = (boost >= MOONSHOT_WILDCARD_BOOST and pred_min >= MOONSHOT_WILDCARD_MIN)

        if not (gate_a or gate_b or gate_c):
            continue

        # Rating floor — wildcards bypass it (their upside is pure boost leverage)
        if not gate_c and rating < MOONSHOT_MIN_RATING:
            continue

        ev = moonshot_ev(rating, boost, pos, team, pts=pts, stl=stl, blk=blk, pred_min=pred_min)
        candidates.append({
            "name": p["player_name"],
            "team": team,
            "pos": pos,
            "rating": rating,
            "boost": boost,
            "ev": ev,
            "pred_min": pred_min,
            "pts": pts,
            "reb": p.get("reb", "?"),
            "ast": p.get("ast", "?"),
            "wildcard": gate_c and not gate_a,
        })

    candidates.sort(key=lambda x: x["ev"], reverse=True)

    lineup = []
    team_count = defaultdict(int)
    team_pos_used = set()
    center_count = 0

    for p in candidates:
        if len(lineup) >= 5:
            break
        team = p["team"]
        pg = pos_group(p["pos"])
        key = (team, pg)
        is_c = is_center(p["pos"])

        if team_count[team] >= MOONSHOT_MAX_PER_TEAM:  # v21: was 2, now 3
            continue
        if key in team_pos_used:
            continue
        if is_c and center_count >= MAX_CENTERS:
            continue

        lineup.append(p)
        team_count[team] += 1
        team_pos_used.add(key)
        if is_c:
            center_count += 1

    return lineup


def print_lineup(label, lineup):
    slots = SLOT_MULTS[:len(lineup)]
    print(f"\n  {label}")
    print(f"  {'Slot':<6} {'Player':<28} {'Team':<5} {'Pos':<4} {'RS':>5} {'Boost':>6} {'Mins':>5}  {'Pts/Reb/Ast':<14}")
    print("  " + "─" * 85)
    for i, (p, slot) in enumerate(zip(lineup, slots)):
        flag = " ★WC" if p.get("wildcard") else ""
        print(f"  {slot:.1f}x   {p['name']:<28} {p['team']:<5} {p['pos']:<4} "
              f"{p['rating']:>5.1f} +{p['boost']:>4.2f}x {p['pred_min']:>5.1f}  "
              f"{p['pts']:.0f}/{p.get('reb','?')}/{p.get('ast','?'):<6}"
              f"  [ev={p['ev']:.1f}]{flag}")


def simulate_date(date_str):
    path = f"/home/user/basketball/data/predictions/{date_str}.csv"
    print(f"\n{'═'*87}")
    print(f"  {date_str}  (v23 model simulation)")
    print(f"{'═'*87}")

    # Load full player pool - include reb/ast in enrichment
    enriched = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = row["player_id"]
            if pid not in enriched or row["scope"] == "slate":
                enriched[pid] = row
    pool = list(enriched.values())

    # Augment reb/ast for moonshot label display
    for p in pool:
        p.setdefault("reb", "?")
        p.setdefault("ast", "?")
        p.setdefault("stl", "0")
        p.setdefault("blk", "0")

    starting5 = pick_starting5(pool)
    moonshot = pick_moonshot(pool)

    print_lineup("STARTING 5  (chalk · v23)", starting5)
    print_lineup("MOONSHOT    (upside · v23)  ★WC = wildcard gate", moonshot)

    # Quick summary
    s5_boost_total = sum(min(p["boost"], CHALK_BOOST_CAP) for p in starting5)
    ms_boost_total = sum(p["boost"] for p in moonshot)
    wc_count = sum(1 for p in moonshot if p.get("wildcard"))
    print(f"\n  S5 avg boost: {s5_boost_total/len(starting5):.2f}x  |  "
          f"Moonshot avg boost: {ms_boost_total/max(len(moonshot),1):.2f}x  |  "
          f"Wildcards in moonshot: {wc_count}")
    print(f"  Player pool size fed into optimizer: {len(pool)}")


if __name__ == "__main__":
    dates = ["2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08", "2026-03-09",
             "2026-03-11", "2026-03-12", "2026-03-13", "2026-03-14"]
    print("\n  NOTE: Re-simulating with v23 model config (lookup + sigmoid tier boost)")
    print("  applied to player stats recorded in prediction CSVs for those dates.")
    print("  v23: player_overrides lookup from real ownership data + sigmoid PPG fallback.")
    print("  Pool = all unique players across all lineup types/scopes.")
    for d in dates:
        simulate_date(d)
    print("\n")
