import json
import hashlib
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

import requests
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

CACHE_DIR = Path("/tmp/nba_real_cache_v6")
CACHE_DIR.mkdir(exist_ok=True)

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"

# Draft slot multipliers for the 5-pick lineup
SLOT_VALUES = ["2.0x", "1.8x", "1.6x", "1.4x", "1.2x"]

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cp(k):
    return CACHE_DIR / f"{hashlib.md5(f'{date.today().isoformat()}:{k}'.encode()).hexdigest()}.json"


def _cg(k):
    p = _cp(k)
    return json.loads(p.read_text()) if p.exists() else None


def _cs(k, v):
    _cp(k).write_text(json.dumps(v))


def _safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default


def _espn_get(url, timeout=15):
    """Fetch any ESPN URL with retry."""
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 2:
                raise
    return {}


# ---------------------------------------------------------------------------
# Data fetching (ESPN — works from Vercel cloud, unlike stats.nba.com)
# ---------------------------------------------------------------------------

def fetch_games():
    """Today's NBA games from ESPN scoreboard."""
    c = _cg("games")
    if c is not None:
        return c

    data = _espn_get(f"{ESPN}/scoreboard")
    games = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        home = away = None
        for cd in comp.get("competitors", []):
            t = {
                "id": cd["team"]["id"],
                "name": cd["team"]["displayName"],
                "abbr": cd["team"].get("abbreviation", ""),
            }
            if cd["homeAway"] == "home":
                home = t
            else:
                away = t
        if not home or not away:
            continue

        odds_list = comp.get("odds", [])
        odds = odds_list[0] if odds_list else {}

        games.append({
            "gameId": ev["id"],
            "label": f"{away['abbr']} @ {home['abbr']}",
            "home": home,
            "away": away,
            "spread": _safe_float(odds.get("spread"), None),
            "total": _safe_float(odds.get("overUnder"), None),
        })

    _cs("games", games)
    return games


def fetch_roster(team_id):
    """Team roster from ESPN."""
    c = _cg(f"ros_{team_id}")
    if c is not None:
        return c

    data = _espn_get(f"{ESPN}/teams/{team_id}/roster")
    players = []
    for a in data.get("athletes", []):
        players.append({
            "id": a["id"],
            "name": a.get("fullName", a.get("displayName", "Unknown")),
            "pos": a.get("position", {}).get("abbreviation", ""),
            "age": a.get("age", 25),
        })

    _cs(f"ros_{team_id}", players)
    return players


# ---------------------------------------------------------------------------
# Player stats + game log parsing (ESPN overview endpoint)
# ---------------------------------------------------------------------------

NAME_MAP = {
    "gamesplayed": "gp", "avgminutes": "min", "fieldgoalpct": "fgp",
    "avgrebounds": "reb", "avgassists": "ast", "avgblocks": "blk",
    "avgsteals": "stl", "avgturnovers": "tov", "avgpoints": "pts",
}

GAMELOG_LABEL_MAP = {
    "min": "min", "minutes": "min",
    "pts": "pts", "points": "pts",
    "reb": "reb", "rebounds": "reb", "totalrebounds": "reb",
    "ast": "ast", "assists": "ast",
    "stl": "stl", "steals": "stl",
    "blk": "blk", "blocks": "blk",
    "to": "tov", "turnovers": "tov",
}


def _fetch_athlete(pid):
    """Fetch season stats + recent game log for one player."""
    c = _cg(f"ath3_{pid}")
    if c is not None:
        return c

    url = f"https://site.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{pid}/overview"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            data = r.json()
            stats = _parse_season_stats(data)
            if stats and stats["min"] > 0:
                recent = _parse_game_log(data.get("gameLog", {}))
                if recent:
                    stats["recent"] = recent
                _cs(f"ath3_{pid}", stats)
                return stats
    except Exception:
        pass

    _cs(f"ath3_{pid}", None)
    return None


def _parse_season_stats(data):
    """Parse season averages from ESPN overview endpoint."""
    result = {"min": 0, "pts": 0, "reb": 0, "ast": 0, "stl": 0, "blk": 0,
              "tov": 0, "fg3m": 0, "fgp": 0.44, "gp": 0}

    stats_block = data.get("statistics", {})
    names = stats_block.get("names", [])
    splits = stats_block.get("splits", [])

    if not names or not splits:
        return None

    values = splits[0].get("stats", [])
    if len(names) != len(values):
        return None

    for name, val in zip(names, values):
        mapped = NAME_MAP.get(name.lower())
        if not mapped:
            continue
        v = _safe_float(val)
        if mapped == "fgp" and v > 1:
            v /= 100
        if mapped == "gp":
            v = int(v)
        result[mapped] = v

    return result if result["min"] > 0 else None


def _parse_game_log(game_log):
    """Extract last 5 game averages from ESPN gameLog.

    Handles multiple ESPN formats:
    - Format A: statistics[].labels/names + events[].stats
    - Format B: seasonTypes[].categories[].events{}
    - Format C: categories[].events[]
    - Format D: top-level labels[] + entries[]
    """
    if not game_log:
        return None

    try:
        for stat_block in game_log.get("statistics", []):
            raw_labels = stat_block.get("labels", stat_block.get("names", []))
            labels = [l.lower() for l in raw_labels]
            events = stat_block.get("events", [])
            if isinstance(events, dict):
                events = list(events.values())
            if labels and events:
                result = _avg_entries(labels, events[-5:])
                if result:
                    return result

        season_types = game_log.get("seasonTypes", [])
        if season_types:
            for st in season_types:
                for cat in st.get("categories", []):
                    labels = [l.lower() for l in cat.get("labels", [])]
                    events = cat.get("events", {})
                    if isinstance(events, dict):
                        entries = list(events.values())
                    elif isinstance(events, list):
                        entries = events
                    else:
                        continue
                    if labels and entries:
                        result = _avg_entries(labels, entries[-5:])
                        if result:
                            return result

        for cat in game_log.get("categories", []):
            labels = [l.lower() for l in cat.get("labels", [])]
            events = cat.get("events", cat.get("entries", []))
            if isinstance(events, dict):
                events = list(events.values())
            if labels and events:
                result = _avg_entries(labels, events[-5:])
                if result:
                    return result

        labels = [l.lower() for l in game_log.get("labels", [])]
        entries = game_log.get("entries", game_log.get("events", []))
        if isinstance(entries, dict):
            entries = list(entries.values())
        if labels and entries:
            result = _avg_entries(labels, entries[-5:])
            if result:
                return result

    except Exception:
        pass

    return None


def _avg_entries(labels, entries):
    """Average numeric stats from game log entries."""
    sums = {"min": 0, "pts": 0, "reb": 0, "ast": 0, "stl": 0, "blk": 0, "tov": 0}
    count = 0

    for entry in entries:
        stats = entry if isinstance(entry, list) else entry.get("stats", [])
        if not stats or len(stats) != len(labels):
            continue

        found_any = False
        for lbl, val in zip(labels, stats):
            mapped = GAMELOG_LABEL_MAP.get(lbl)
            if mapped:
                v = _safe_float(val)
                if v >= 0:
                    sums[mapped] += v
                    found_any = True
        if found_any:
            count += 1

    if count == 0:
        return None

    return {k: round(v / count, 1) for k, v in sums.items()}


# ---------------------------------------------------------------------------
# Team defensive stats — hardcoded fallback + ESPN Core API
# ---------------------------------------------------------------------------
# DEF_RATING = points allowed per 100 possessions. Higher = worse defense.
# Updated ~March 2026. Sources: StatMuse, FOX Sports, NBA.com

LEAGUE_AVG_DEF_RATING = 112.0

_DEFENSE_BY_ID = {
    "25": (107.0, 107.6),   # OKC Thunder
    "8":  (109.0, 109.7),   # Detroit Pistons
    "14": (112.0, 112.8),   # Miami Heat
    "24": (112.5, 112.9),   # San Antonio Spurs
    "16": (112.0, 112.6),   # Minnesota Timberwolves
    "10": (111.0, 112.8),   # Houston Rockets
    "19": (113.0, 113.5),   # Orlando Magic
    "29": (113.5, 113.8),   # Memphis Grizzlies
    "5":  (113.5, 114.1),   # Cleveland Cavaliers
    "18": (114.0, 114.7),   # New York Knicks
    "2":  (110.0, 115.2),   # Boston Celtics
    "15": (115.0, 115.5),   # Milwaukee Bucks
    "12": (115.0, 115.8),   # LA Clippers
    "11": (116.0, 116.0),   # Indiana Pacers
    "6":  (116.0, 116.2),   # Dallas Mavericks
    "21": (116.5, 116.5),   # Phoenix Suns
    "9":  (117.0, 116.8),   # Golden State Warriors
    "1":  (117.5, 117.0),   # Atlanta Hawks
    "23": (117.5, 117.2),   # Sacramento Kings
    "4":  (118.0, 117.5),   # Chicago Bulls
    "13": (118.5, 118.0),   # LA Lakers
    "7":  (119.0, 118.0),   # Denver Nuggets
    "20": (119.0, 118.5),   # Philadelphia 76ers
    "22": (119.5, 118.8),   # Portland Trail Blazers
    "28": (120.0, 119.0),   # Toronto Raptors
    "17": (120.5, 119.5),   # Brooklyn Nets
    "30": (121.0, 120.0),   # Charlotte Hornets
    "3":  (121.5, 120.3),   # New Orleans Pelicans
    "27": (124.0, 121.2),   # Washington Wizards
    "26": (125.9, 122.2),   # Utah Jazz
}


def fetch_team_defense(team_id):
    """Fetch team defensive stats. ESPN Core API with hardcoded fallback."""
    c = _cg(f"def_{team_id}")
    if c is not None:
        return c

    fallback = _DEFENSE_BY_ID.get(str(team_id), (112.0, 112.0))
    defaults = {"opp_ppg": fallback[0], "def_rating": fallback[1], "opp_fgp": 0.465}

    try:
        data = _espn_get(
            f"{ESPN_CORE}/seasons/2026/types/2/teams/{team_id}/statistics",
            timeout=10,
        )
        for cat in data.get("splits", {}).get("categories", data.get("categories", [])):
            cat_name = cat.get("name", "").lower()
            if "defen" not in cat_name and "opponent" not in cat_name:
                continue
            for stat in cat.get("stats", []):
                sname = stat.get("name", "").lower()
                val = _safe_float(stat.get("value"), None)
                if val is None:
                    continue
                if "opponentpoints" in sname or "opppoints" in sname:
                    defaults["opp_ppg"] = val
                elif "defensiverating" in sname or "defrating" in sname:
                    defaults["def_rating"] = val
                elif ("opponent" in sname and "fieldgoal" in sname
                      and "pct" in sname):
                    defaults["opp_fgp"] = val if val < 1 else val / 100
    except Exception:
        pass

    _cs(f"def_{team_id}", defaults)
    return defaults


# ---------------------------------------------------------------------------
# Fetch all player + team data for a game (parallel)
# ---------------------------------------------------------------------------

def fetch_game_players(home_id, away_id):
    """Fetch rosters, player stats, and team defense for both teams."""
    home_roster = fetch_roster(home_id)
    away_roster = fetch_roster(away_id)

    players = [(p, "home") for p in home_roster] + [(p, "away") for p in away_roster]
    pids = list({p["id"] for p, _ in players})

    stats = {}
    home_def = {"opp_ppg": 112.0, "def_rating": 112.0, "opp_fgp": 0.465}
    away_def = {"opp_ppg": 112.0, "def_rating": 112.0, "opp_fgp": 0.465}

    with ThreadPoolExecutor(max_workers=12) as ex:
        player_futs = {ex.submit(_fetch_athlete, pid): pid for pid in pids}
        home_def_fut = ex.submit(fetch_team_defense, home_id)
        away_def_fut = ex.submit(fetch_team_defense, away_id)

        for f in as_completed(player_futs):
            pid = player_futs[f]
            try:
                r = f.result()
                if r:
                    stats[pid] = r
            except Exception:
                pass

        try:
            home_def = home_def_fut.result()
        except Exception:
            pass
        try:
            away_def = away_def_fut.result()
        except Exception:
            pass

    return players, stats, home_def, away_def


# ---------------------------------------------------------------------------
# OUTPERFORMANCE PROJECTION MODEL
# ---------------------------------------------------------------------------
# Ranks by "who will EXCEED their baseline tonight" — not raw volume.
# Surfaces: hot players facing weak defenses, role players with expanded
# roles due to injuries, stars with juicy matchups.
# ---------------------------------------------------------------------------

def recent_form_multiplier(stats):
    """How hot/cold is this player vs season baseline? (0.80 to 1.50)

    Wider range than typical DFS models because we specifically want to
    surface hot-streak players (e.g. Sheppard 28pts vs 12 PPG season).
    """
    recent = stats.get("recent")
    if not recent:
        return 1.0

    season_composite = (
        max(stats.get("pts", 1), 1) * 1.0
        + max(stats.get("reb", 0.5), 0.5) * 0.6
        + max(stats.get("ast", 0.5), 0.5) * 0.8
        + max(stats.get("stl", 0.1), 0.1) * 1.5
        + max(stats.get("blk", 0.1), 0.1) * 1.5
    )
    recent_composite = (
        recent.get("pts", stats.get("pts", 0)) * 1.0
        + recent.get("reb", stats.get("reb", 0)) * 0.6
        + recent.get("ast", stats.get("ast", 0)) * 0.8
        + recent.get("stl", stats.get("stl", 0)) * 1.5
        + recent.get("blk", stats.get("blk", 0)) * 1.5
    )

    if season_composite <= 0:
        return 1.0

    return max(0.80, min(recent_composite / season_composite, 1.50))


def matchup_multiplier(opp_defense, pos):
    """Opponent defense weakness — position-aware. (0.90 to 1.25)

    Wider range + higher sensitivity so that truly bad defenses (WAS, UTA)
    give a meaningfully larger edge than average matchups.
    """
    opp_ppg = opp_defense.get("opp_ppg", LEAGUE_AVG_DEF_RATING)
    def_rating = opp_defense.get("def_rating", LEAGUE_AVG_DEF_RATING)

    ppg_diff = opp_ppg - LEAGUE_AVG_DEF_RATING
    rating_diff = def_rating - LEAGUE_AVG_DEF_RATING
    matchup_signal = max(ppg_diff / 12.0, rating_diff / 12.0)

    # Position adjustment: bigs feast on bad interior D, guards on bad perimeter D
    if pos in ("PG", "SG", "G"):
        matchup_signal *= 1.05
    elif pos in ("C", "PF"):
        matchup_signal *= 1.10

    mult = 1.0 + matchup_signal * 0.15
    return max(0.90, min(mult, 1.25))


def usage_trend_multiplier(stats):
    """Detect expanding role from recent minutes increase. (0.95 to 1.35)

    Wider cap to better capture major role expansions when a star teammate
    goes down (e.g. Moody with Curry out, Sensabaugh with 3 starters out).
    """
    recent = stats.get("recent")
    if not recent:
        return 1.0

    season_min = max(stats.get("min", 1), 1)
    recent_min = recent.get("min", season_min)
    min_ratio = recent_min / season_min

    if min_ratio > 1.05:
        boost = (min_ratio - 1.0) * 1.5
        return min(1.0 + boost, 1.35)
    elif min_ratio < 0.90:
        return max(0.95, min_ratio)
    return 1.0


def game_closeness_factor(spread):
    """Close games = more clutch opportunities, starters play full minutes."""
    if spread is None:
        return 1.0
    expected_margin = abs(spread) * 1.3
    return 1.0 + 0.25 * (1.0 - min(expected_margin / 30.0, 1.0))


def pace_factor(total):
    """Scoring environment. Inverted-U around O/U 222."""
    if total is None:
        return 1.0
    deviation = abs(total - 222) / 30.0
    return 1.0 + 0.06 * max(1.0 - deviation, 0)


def base_production_score(stats, pos):
    """Base per-game production. Two-way stats weighted heavily."""
    pts = stats.get("pts", 0)
    reb = stats.get("reb", 0)
    ast = stats.get("ast", 0)
    stl = stats.get("stl", 0)
    blk = stats.get("blk", 0)
    tov = stats.get("tov", 0)
    fgp = stats.get("fgp", 0.44)

    score = (pts * 1.0 + reb * 1.0 + ast * 1.5
             + stl * 3.5 + blk * 3.0 - tov * 1.2)
    score += (fgp - 0.44) * 10.0

    if pos in ("PG", "SG", "G"):
        if stl + blk >= 2.0:
            score *= 1.08
    elif pos in ("C", "PF"):
        if ast >= 5.0:
            score *= 1.06

    return score


def project_player(name, pos, age, side, stats, spread, total, game_label="",
                   team_abbrev="", opp_defense=None):
    """Project OUTPERFORMANCE: who will exceed their baseline tonight?"""
    avg_min = stats["min"]
    if avg_min < 15:
        return None

    if opp_defense is None:
        opp_defense = {"opp_ppg": 112.0, "def_rating": 112.0, "opp_fgp": 0.465}

    base = base_production_score(stats, pos)
    form = recent_form_multiplier(stats)
    matchup = matchup_multiplier(opp_defense, pos)
    usage = usage_trend_multiplier(stats)
    closeness = game_closeness_factor(spread)
    pace = pace_factor(total)
    home_adj = 1.015 if side == "home" else 0.985

    # Minutes projection
    pred_min = avg_min
    recent = stats.get("recent")
    if recent and recent.get("min", 0) > avg_min:
        pred_min = recent["min"] * 0.7 + avg_min * 0.3

    if spread is not None:
        abs_spread = abs(spread)
        if abs_spread <= 4:
            pred_min *= 1.0 + (4 - abs_spread) * 0.006
        elif abs_spread > 8:
            pred_min *= max(1.0 - (abs_spread - 8) * 0.015, 0.82)

    if age and age > 35:
        pred_min *= 0.94
    elif age and age > 32:
        pred_min *= 0.97

    # Composite outperformance score
    outperformance = base * form * matchup * usage * closeness * pace * home_adj
    min_scale = pred_min / 30.0
    final_rating = outperformance * min_scale

    # Boost: product of all situational multipliers (>1.0 = above-average situation)
    boost = form * matchup * usage * closeness * pace * home_adj
    # Also account for minutes expansion
    if avg_min > 0:
        boost *= (pred_min / avg_min)

    return {
        "name": name,
        "pos": pos,
        "team": team_abbrev,
        "game": game_label,
        "rating": round(final_rating, 1),
        "predMin": round(pred_min, 1),
        "boost": round(boost, 2),
        "form": round(form, 2),
        "matchup": round(matchup, 3),
        "usage": round(usage, 3),
        "closeness": round(closeness, 3),
        "side": side,
        # Internal sort keys
        "_form": form,
        "_matchup": matchup,
        "_usage": usage,
        "_base": base,
    }


# ---------------------------------------------------------------------------
# Lineup building — 3 modes with draft slot assignment
# ---------------------------------------------------------------------------

def build_lineups(projections):
    """Build 3 meaningfully different lineup variants with draft slots.

    The key insight: we rank by OUTPERFORMANCE POTENTIAL, not raw production.
    This is what surfaces hot role players facing bad defenses over safe stars.

    Chalk:   rating^0.8 × boost^1.5  — best outperformers with a production floor.
    Diff:    matchup^2.5 × form^2.5 × rating^0.4  — matchup + hot-streak gems.
    Contra:  usage^3 × form^2 × rating^0.3  — injury-driven breakout plays.
    """
    if len(projections) < 5:
        return [], [], []

    by_rating = sorted(projections, key=lambda x: x["rating"], reverse=True)
    pool = by_rating[:25]

    # === CHALK: Outperformance-weighted ===
    # rating^0.8 dampens raw production dominance
    # boost^1.5 amplifies situational edge (form × matchup × usage × ...)
    # A hot player (boost=2.0) vs bad defense beats a safe star (boost=1.05)
    def _chalk_key(p):
        return (max(p["rating"], 0.1) ** 0.8) * (max(p.get("boost", 1.0), 0.5) ** 1.5)

    chalk_sorted = sorted(pool, key=_chalk_key, reverse=True)
    chalk = [dict(p) for p in chalk_sorted[:5]]
    chalk_names = {p["name"] for p in chalk}
    _assign_slots(chalk)

    # === DIFF: Matchup + form gems ===
    # Hunts players facing terrible defenses AND on hot streaks
    # rating^0.4 keeps a production floor but lets matchup/form dominate
    def _diff_key(p):
        return (p.get("_matchup", 1.0) ** 2.5
                * p.get("_form", 1.0) ** 2.5
                * max(p["rating"], 0.1) ** 0.4)

    diff_sorted = sorted(pool, key=_diff_key, reverse=True)
    diff = _select_with_diversity(diff_sorted, chalk_names, min_unique=2)
    _assign_slots(diff)

    # === CONTRARIAN: Usage/breakout plays ===
    # Hunts injury-driven role expansions (Moody w/ Curry out, etc.)
    # usage^3 massively rewards expanding minutes
    def _contra_key(p):
        return (p.get("_usage", 1.0) ** 3.0
                * p.get("_form", 1.0) ** 2.0
                * max(p["rating"], 0.1) ** 0.3)

    contra_sorted = sorted(pool, key=_contra_key, reverse=True)
    contra = _select_with_diversity(contra_sorted, chalk_names, min_unique=2)
    _assign_slots(contra)

    return chalk, diff, contra


def _select_with_diversity(sorted_pool, chalk_names, min_unique=2):
    """Pick top 5 from sorted_pool, guaranteeing min_unique non-chalk players.

    Maintains the sorted_pool order for slot assignment — the mode-specific
    ranking determines who gets the 2.0x slot, not raw production.
    """
    unique = []
    shared = []
    for p in sorted_pool:
        if p["name"] not in chalk_names:
            unique.append(dict(p))
        else:
            shared.append(dict(p))
        if len(unique) >= min_unique and len(unique) + len(shared) >= 5:
            break

    # Take required unique picks first, then fill with best remaining
    lineup = unique[:min_unique]
    remaining = shared + unique[min_unique:]
    for p in remaining:
        if len(lineup) >= 5:
            break
        if p["name"] not in {x["name"] for x in lineup}:
            lineup.append(p)

    # Maintain the mode-specific sort order for slot assignment
    pool_order = {p["name"]: i for i, p in enumerate(sorted_pool)}
    lineup.sort(key=lambda x: pool_order.get(x["name"], 999))
    return lineup[:5]


def _assign_slots(lineup):
    """Assign draft slot values (2.0x, 1.8x, ...) to a 5-player lineup."""
    for i, p in enumerate(lineup):
        p["slot"] = SLOT_VALUES[i] if i < len(SLOT_VALUES) else "1.0x"


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.get("/api/games")
async def get_games():
    try:
        return JSONResponse(content=fetch_games())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/picks")
async def get_picks(gameId: str = Query(...)):
    """Per-game picks: top 5 for a specific game, 3 lineup variants."""
    try:
        games = fetch_games()
        game = next((g for g in games if g["gameId"] == gameId), None)
        if not game:
            return JSONResponse(content={"error": "Game not found."}, status_code=404)

        spread = game.get("spread")
        total = game.get("total")

        players, stats, home_def, away_def = fetch_game_players(
            game["home"]["id"], game["away"]["id"]
        )

        projections = []
        for pinfo, side in players:
            s = stats.get(pinfo["id"])
            if not s:
                continue
            opp_def = away_def if side == "home" else home_def
            team_abbrev = game["home"]["abbr"] if side == "home" else game["away"]["abbr"]
            p = project_player(
                pinfo["name"], pinfo["pos"], pinfo.get("age", 25),
                side, s, spread, total, game["label"],
                team_abbrev=team_abbrev,
                opp_defense=opp_def,
            )
            if p:
                projections.append(p)

        if len(projections) < 5:
            return JSONResponse(content={
                "error": f"Not enough eligible players ({len(projections)} found, need 5). "
                         f"Stats fetched for {len(stats)}/{len(players)} players."
            })

        chalk, diff, contra = build_lineups(projections)

        return JSONResponse(content={
            "game": {
                "label": game["label"],
                "home": game["home"]["name"],
                "away": game["away"]["name"],
                "spread": spread,
                "total": total,
            },
            "lineups": {
                "chalk": chalk,
                "differentiated": diff,
                "contrarian": contra,
            },
        })
    except Exception as e:
        return JSONResponse(content={"error": f"Failed: {str(e)}"}, status_code=500)


@app.get("/api/slate")
async def get_slate():
    """Full-slate top 5 picks across ALL games today."""
    try:
        games = fetch_games()
        if not games:
            return JSONResponse(content={"error": "No games on today's slate."})

        all_projections = []
        game_summaries = []

        def process_game(game):
            spread = game.get("spread")
            total = game.get("total")
            players, stats, home_def, away_def = fetch_game_players(
                game["home"]["id"], game["away"]["id"]
            )
            projs = []
            for pinfo, side in players:
                s = stats.get(pinfo["id"])
                if not s:
                    continue
                opp_def = away_def if side == "home" else home_def
                team_abbrev = game["home"]["abbr"] if side == "home" else game["away"]["abbr"]
                p = project_player(
                    pinfo["name"], pinfo["pos"], pinfo.get("age", 25),
                    side, s, spread, total, game["label"],
                    team_abbrev=team_abbrev,
                    opp_defense=opp_def,
                )
                if p:
                    projs.append(p)
            return game, projs

        with ThreadPoolExecutor(max_workers=5) as ex:
            futs = {ex.submit(process_game, g): g for g in games}
            for f in as_completed(futs):
                try:
                    game, projs = f.result()
                    all_projections.extend(projs)
                    game_summaries.append({
                        "label": game["label"],
                        "spread": game.get("spread"),
                        "total": game.get("total"),
                        "playerCount": len(projs),
                    })
                except Exception:
                    pass

        if len(all_projections) < 5:
            return JSONResponse(content={
                "error": f"Not enough players across slate ({len(all_projections)} found)."
            })

        chalk, diff, contra = build_lineups(all_projections)

        return JSONResponse(content={
            "games": game_summaries,
            "totalPlayers": len(all_projections),
            "lineups": {
                "chalk": chalk,
                "differentiated": diff,
                "contrarian": contra,
            },
        })
    except Exception as e:
        return JSONResponse(content={"error": f"Slate failed: {str(e)}"}, status_code=500)
