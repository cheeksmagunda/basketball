import json
import math
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

CACHE_DIR = Path("/tmp/nba_real_cache_v4")
CACHE_DIR.mkdir(exist_ok=True)

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"

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
# Data fetching
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
            "startTime": ev.get("date", ""),
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
# Player stats + game log parsing
# ---------------------------------------------------------------------------
# ESPN overview returns:
#   statistics.names = ["gamesPlayed", "avgMinutes", ...]
#   statistics.splits[0].stats = ["50", "34.2", ...]
#   gameLog = { ... recent game data ... }
NAME_MAP = {
    "gamesplayed": "gp", "avgminutes": "min", "fieldgoalpct": "fgp",
    "avgrebounds": "reb", "avgassists": "ast", "avgblocks": "blk",
    "avgsteals": "stl", "avgturnovers": "tov", "avgpoints": "pts",
}

# Labels that appear in game log entries
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
                # Also extract recent game log
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
    """Parse season averages from overview endpoint."""
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
    """Extract last 5 game averages from the overview gameLog.

    Handles multiple ESPN formats:
    - Format A: statistics[].labels/names + statistics[].events[].stats
      (current ESPN format as of March 2026)
    - Format B: seasonTypes[].categories[].events{} with labels
    - Format C: categories[].events[] or entries[]
    - Format D: labels[] + entries[] at top level
    """
    if not game_log:
        return None

    try:
        # --- Format A: statistics[0].labels + events[].stats ---
        # This is the actual format ESPN returns as of March 2026:
        #   gameLog.statistics[0].labels = ["MIN", "FG%", "REB", "AST", ...]
        #   gameLog.statistics[0].names  = ["minutes", "fieldGoalPct", ...]
        #   gameLog.statistics[0].events = [{"eventId": "...", "stats": ["18", ...]}, ...]
        for stat_block in game_log.get("statistics", []):
            # Prefer 'labels' (short: "MIN", "PTS") but fall back to 'names'
            raw_labels = stat_block.get("labels", stat_block.get("names", []))
            labels = [l.lower() for l in raw_labels]
            events = stat_block.get("events", [])
            if isinstance(events, dict):
                events = list(events.values())
            if labels and events:
                result = _avg_entries(labels, events[-5:])
                if result:
                    return result

        # --- Format B: seasonTypes[0].categories[0].events ---
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

        # --- Format C: categories[0].events ---
        for cat in game_log.get("categories", []):
            labels = [l.lower() for l in cat.get("labels", [])]
            events = cat.get("events", cat.get("entries", []))
            if isinstance(events, dict):
                events = list(events.values())
            if labels and events:
                result = _avg_entries(labels, events[-5:])
                if result:
                    return result

        # --- Format D: top-level labels + entries ---
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
        if not stats:
            continue
        if len(stats) != len(labels):
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
# Team defensive stats
# ---------------------------------------------------------------------------
# Fallback defensive ratings by ESPN team ID.
# Used when ESPN Core API doesn't return data. Updated ~March 2026.
# Sources: StatMuse, FOX Sports, NBA.com (DRtg = pts allowed per 100 poss)
_DEFENSE_BY_ID = {
    # id: (opp_ppg, def_rating)
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
    """Fetch team defensive stats (how bad their defense is).

    Tries ESPN Core API first, falls back to hardcoded table.
    Returns opponent PPG, defensive rating, and opponent FG%.
    """
    c = _cg(f"def_{team_id}")
    if c is not None:
        return c

    # Start with fallback data
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
        pass  # fallback data already set

    _cs(f"def_{team_id}", defaults)
    return defaults


# ---------------------------------------------------------------------------
# Fetch all player + team data for a game
# ---------------------------------------------------------------------------
def fetch_game_players(home_id, away_id):
    """Fetch rosters, player stats, and team defense for both teams."""
    home_roster = fetch_roster(home_id)
    away_roster = fetch_roster(away_id)

    players = [(p, "home") for p in home_roster] + [(p, "away") for p in away_roster]
    pids = list({p["id"] for p, _ in players})

    # Fetch player stats + team defense in parallel
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
# OUTPERFORMANCE projection model
# ---------------------------------------------------------------------------
# This model predicts WHO WILL EXCEED THEIR BASELINE tonight.
# It does NOT rank by raw production (that always picks biggest stars).
#
# Key signals:
#   1. Recent form: Is the player on a hot streak? (last 5 games vs season)
#   2. Matchup quality: Is the opponent defense weak? (defense-vs-position)
#   3. Usage trend: Are recent minutes UP? (proxy for teammate injuries)
#   4. Game context: Closeness, underdog status, pace
#
# Final ranking = outperformance score (expected tonight / baseline - 1.0)
# This naturally surfaces:
#   - Hot players facing weak defenses (Sheppard vs WSH)
#   - Role players with expanded roles (Moody with Curry out)
#   - Stars with juicy matchups (Jokic vs depleted Utah)
# ---------------------------------------------------------------------------

# League averages for normalization
LEAGUE_AVG_PPG_ALLOWED = 112.0
LEAGUE_AVG_DEF_RATING = 112.0


def recent_form_multiplier(stats):
    """How much is this player trending UP from their season baseline?

    Compares last 5 game averages to season averages.
    A player averaging 25 PPG recently vs 18 PPG season = massive boost.

    Returns multiplier: 0.85 (cold) to 1.40 (scorching).
    """
    recent = stats.get("recent")
    if not recent:
        return 1.0  # no game log data, neutral

    season_pts = max(stats.get("pts", 1), 1)
    season_reb = max(stats.get("reb", 1), 0.5)
    season_ast = max(stats.get("ast", 1), 0.5)
    season_stl = max(stats.get("stl", 0.3), 0.1)
    season_blk = max(stats.get("blk", 0.3), 0.1)

    recent_pts = recent.get("pts", season_pts)
    recent_reb = recent.get("reb", season_reb)
    recent_ast = recent.get("ast", season_ast)
    recent_stl = recent.get("stl", season_stl)
    recent_blk = recent.get("blk", season_blk)

    # Weighted composite: points matter most, but two-way stats count
    season_composite = (season_pts * 1.0 + season_reb * 0.6
                        + season_ast * 0.8 + season_stl * 1.5
                        + season_blk * 1.5)
    recent_composite = (recent_pts * 1.0 + recent_reb * 0.6
                        + recent_ast * 0.8 + recent_stl * 1.5
                        + recent_blk * 1.5)

    if season_composite <= 0:
        return 1.0

    raw_ratio = recent_composite / season_composite

    # Clamp to 0.85 - 1.40 range
    return max(0.85, min(raw_ratio, 1.40))


def matchup_multiplier(opp_defense, pos):
    """How much does the opponent's weak defense help this player?

    Bad defensive teams allow more points = higher production for everyone.
    Position-aware: guards benefit more from perimeter defense weakness,
    bigs benefit more from interior defense weakness.

    Returns multiplier: 0.92 (elite defense) to 1.20 (worst defense).
    """
    opp_ppg = opp_defense.get("opp_ppg", LEAGUE_AVG_PPG_ALLOWED)
    def_rating = opp_defense.get("def_rating", LEAGUE_AVG_DEF_RATING)

    # How much worse than average is this defense?
    # opp_ppg 120 vs avg 112 = 8 points above average = terrible defense
    ppg_diff = opp_ppg - LEAGUE_AVG_PPG_ALLOWED  # positive = bad defense
    rating_diff = def_rating - LEAGUE_AVG_DEF_RATING

    # Use the worse signal (more favorable for the player)
    matchup_signal = max(ppg_diff / 12.0, rating_diff / 12.0)

    # Position adjustment: guards benefit slightly more from bad perimeter D
    if pos in ("PG", "SG", "G"):
        matchup_signal *= 1.05
    elif pos in ("C", "PF"):
        matchup_signal *= 1.08  # bigs feast on bad interior D

    # Convert to multiplier: -1 to +1 signal → 0.92 to 1.20
    mult = 1.0 + matchup_signal * 0.10
    return max(0.92, min(mult, 1.20))


def usage_trend_multiplier(stats):
    """Detect expanding role from recent minutes increase.

    If recent minutes >> season minutes, teammates are probably injured
    and this player is getting more opportunity.
    This is the signal that catches Moody (Curry out) and Sensabaugh
    (Markkanen/Kessler/Nurkic out).

    Returns multiplier: 0.95 to 1.25.
    """
    recent = stats.get("recent")
    if not recent:
        return 1.0

    season_min = max(stats.get("min", 1), 1)
    recent_min = recent.get("min", season_min)

    # How much have minutes increased?
    min_ratio = recent_min / season_min

    if min_ratio > 1.05:
        # Minutes are UP: teammate injuries, expanded role
        # More minutes = more production opportunity
        boost = (min_ratio - 1.0) * 1.5  # 10% more minutes → 15% boost
        return min(1.0 + boost, 1.25)
    elif min_ratio < 0.90:
        # Minutes are DOWN: maybe coming back from injury, load managed
        return max(0.95, min_ratio)
    else:
        return 1.0


def game_closeness_factor(spread):
    """Close games = higher-quality production, more clutch opportunities."""
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
    """Base per-game production score from season averages.

    Weights emphasize two-way players (steals + blocks from guards
    are rare and signal high-quality production like Derrick White).
    """
    pts = stats.get("pts", 0)
    reb = stats.get("reb", 0)
    ast = stats.get("ast", 0)
    stl = stats.get("stl", 0)
    blk = stats.get("blk", 0)
    tov = stats.get("tov", 0)
    fgp = stats.get("fgp", 0.44)

    # Two-way stats weighted heavily (steals + blocks from guards = rare/valuable)
    score = (pts * 1.0 + reb * 1.0 + ast * 1.5
             + stl * 3.5 + blk * 3.0 - tov * 1.2)

    # Efficiency bonus
    score += (fgp - 0.44) * 10.0

    # Per-position: guards who stuff the stat sheet get extra credit
    if pos in ("PG", "SG", "G"):
        two_way = stl + blk
        if two_way >= 2.0:  # guards with 2+ combined stl+blk = elite
            score *= 1.08
    elif pos in ("C", "PF"):
        # Bigs with assists = rare playmaking (Jokic-type)
        if ast >= 5.0:
            score *= 1.06

    return score


def project_player(name, pos, age, side, stats, spread, total, game_label="",
                   opp_defense=None):
    """Project OUTPERFORMANCE score: who will exceed their baseline tonight?

    The model finds players whose situation tonight is much better than
    their season average situation — hot streaks, weak opponents, expanded
    roles due to injuries.

    Pipeline:
        1. Base production score (season averages, two-way weighted)
        2. Recent form multiplier (are they hot right now?)
        3. Matchup multiplier (is the opponent defense bad?)
        4. Usage trend (are minutes expanding — injury proxy?)
        5. Game context (closeness, pace)
        6. Final outperformance score
    """
    avg_min = stats["min"]
    if avg_min < 15:
        return None

    if opp_defense is None:
        opp_defense = {"opp_ppg": 112.0, "def_rating": 112.0, "opp_fgp": 0.465}

    # --- Stage 1: Base production ---
    base = base_production_score(stats, pos)

    # --- Stage 2: Recent form (THE key signal) ---
    form = recent_form_multiplier(stats)

    # --- Stage 3: Matchup quality ---
    matchup = matchup_multiplier(opp_defense, pos)

    # --- Stage 4: Usage trend (injury proxy) ---
    usage = usage_trend_multiplier(stats)

    # --- Stage 5: Game context ---
    closeness = game_closeness_factor(spread)
    pace = pace_factor(total)
    home_adj = 1.015 if side == "home" else 0.985

    # --- Stage 6: Minutes projection ---
    pred_min = avg_min
    recent = stats.get("recent")
    if recent and recent.get("min", 0) > avg_min:
        # Use recent minutes if higher (captures injury-driven expansion)
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

    # --- Final: Outperformance score ---
    # Base × Form × Matchup × Usage × Context adjustments
    # This ranks by "how much better will tonight be vs baseline?"
    outperformance = (base * form * matchup * usage
                      * closeness * pace * home_adj)

    # Scale by minutes (more minutes = more opportunity)
    min_scale = pred_min / 30.0  # normalize around 30 min
    final_rating = outperformance * min_scale

    # --- Tier ---
    tier = "star" if pred_min >= 33 else ("starter" if pred_min >= 24 else "role")

    # --- Variance ---
    base_var = 0.12 if tier == "star" else (0.18 if tier == "starter" else 0.28)
    # Hot streaks reduce variance (they're more predictable)
    if form > 1.10:
        base_var *= 0.85
    # Close games reduce star variance
    if spread is not None and abs(spread) <= 4:
        base_var *= 0.90

    return {
        "name": name, "pos": pos, "tier": tier,
        "rating": round(final_rating, 1),
        "predMin": round(pred_min, 1),
        "rpm": round(form, 2),           # repurpose: show form multiplier
        "vegasAdj": round(matchup, 3),    # repurpose: show matchup quality
        "blowoutAdj": round(usage, 3),    # repurpose: show usage trend
        "homeAdj": round(closeness, 3),   # repurpose: show closeness
        "stdDev": round(final_rating * base_var, 1),
        "side": side,
        "game": game_label,
        # Internal
        "_form": form,
        "_matchup": matchup,
        "_usage": usage,
        "_closeness": closeness,
    }


def pairwise_conf(a, b):
    """P(a ranks above b)."""
    diff = a["rating"] - b["rating"]
    comb = math.sqrt(a["stdDev"] ** 2 + b["stdDev"] ** 2)
    if comb == 0:
        return 0.5
    return round(0.5 * (1 + math.erf(diff / (comb * math.sqrt(2)))), 3)


def build_lineups(projections):
    """Build 3 meaningfully different lineup variants.

    Chalk:   Top 5 by overall outperformance score (balanced).
    Diff:    Best matchup + form plays — guarantees at least 2 players
             NOT in chalk by heavily weighting matchup quality and
             recent hot streaks over raw production.
    Contra:  Breakout/usage plays — guarantees at least 2 players
             NOT in chalk by hunting injury-driven role expansions
             and undervalued players with high recent minutes.
    """
    if len(projections) < 5:
        return [], [], []

    ranked = sorted(projections, key=lambda x: x["rating"], reverse=True)
    pool = ranked[:20]  # wider pool for finding differentiated picks
    chalk_names = {p["name"] for p in ranked[:5]}

    # --- Chalk: top 5 by outperformance ---
    chalk = [dict(p) for p in ranked[:5]]
    _annotate(chalk)

    # --- Differentiated: matchup + form hunting ---
    # Sort by matchup×form signal rather than raw rating.
    # This surfaces players facing terrible defenses who are also hot,
    # even if their season baseline is lower.
    diff_sorted = sorted(pool, key=lambda x: (
        x.get("_matchup", 1.0) ** 2.0
        * x.get("_form", 1.0) ** 2.0
        * (x["rating"] ** 0.5)
    ), reverse=True)
    # Guarantee at least 2 non-chalk players
    diff = _build_with_min_unique(diff_sorted, chalk_names, min_unique=2)
    _annotate(diff)

    # --- Contrarian: usage/breakout hunting ---
    # Sort by usage trend signal — finds players whose minutes are
    # expanding (teammates injured) and who are capitalizing.
    contra_sorted = sorted(pool, key=lambda x: (
        x.get("_usage", 1.0) ** 3.0
        * x.get("_form", 1.0) ** 1.5
        * (x["rating"] ** 0.4)
    ), reverse=True)
    # Guarantee at least 2 non-chalk players
    contra = _build_with_min_unique(contra_sorted, chalk_names, min_unique=2)
    _annotate(contra)

    return chalk, diff, contra


def _build_with_min_unique(sorted_pool, chalk_names, min_unique=2):
    """Pick top 5, guaranteeing at least min_unique players NOT in chalk."""
    unique = []
    shared = []
    for p in sorted_pool:
        if p["name"] in chalk_names:
            shared.append(dict(p))
        else:
            unique.append(dict(p))
        if len(unique) >= min_unique and len(unique) + len(shared) >= 5:
            break

    # Fill lineup: take required unique picks, then fill with best remaining
    lineup = unique[:min_unique]
    remaining = shared + unique[min_unique:]
    remaining.sort(key=lambda x: x["rating"], reverse=True)
    for p in remaining:
        if len(lineup) >= 5:
            break
        if p["name"] not in {x["name"] for x in lineup}:
            lineup.append(p)

    # Sort final lineup by rating for ranking
    lineup.sort(key=lambda x: x["rating"], reverse=True)
    return lineup[:5]


def _annotate(lineup):
    for i, p in enumerate(lineup):
        p["rank"] = i + 1
        if i < len(lineup) - 1:
            c = pairwise_conf(p, lineup[i + 1])
            p["confidence"] = c
            p["confLabel"] = "high" if c >= 0.70 else ("medium" if c >= 0.55 else "low")
        else:
            p["confidence"] = None
            p["confLabel"] = "-"


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
            # Opponent defense: home player faces away team's D, vice versa
            opp_def = away_def if side == "home" else home_def
            p = project_player(
                pinfo["name"], pinfo["pos"], pinfo.get("age", 25),
                side, s, spread, total, game["label"],
                opp_defense=opp_def,
            )
            if p:
                projections.append(p)

        if len(projections) < 5:
            return JSONResponse(content={
                "error": f"Not enough eligible players ({len(projections)} found, need 5). Stats fetched for {len(stats)}/{len(players)} players."
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
                p = project_player(
                    pinfo["name"], pinfo["pos"], pinfo.get("age", 25),
                    side, s, spread, total, game["label"],
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


@app.get("/api/debug")
async def debug():
    """Debug: show raw data for first player to verify parsing."""
    try:
        games = fetch_games()
        if not games:
            return JSONResponse(content={"error": "No games"})
        g = games[0]
        roster = fetch_roster(g["home"]["id"])
        # Pick a starter (higher index often = more established player)
        p = roster[min(3, len(roster) - 1)] if len(roster) > 3 else roster[0]
        pid = p["id"]

        url = f"https://site.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{pid}/overview"
        r = requests.get(url, timeout=12)
        data = r.json() if r.status_code == 200 else {}

        # Extract what we'd parse
        stats = _parse_season_stats(data)
        game_log_raw = data.get("gameLog", {})
        game_log_keys = list(game_log_raw.keys()) if isinstance(game_log_raw, dict) else []
        recent = _parse_game_log(game_log_raw)

        # Team defense
        team_def = fetch_team_defense(g["home"]["id"])

        return JSONResponse(content={
            "player": p,
            "season_stats": stats,
            "recent_5_game_avg": recent,
            "game_log_top_keys": game_log_keys,
            "game_log_sample": str(json.dumps(game_log_raw))[:2000],
            "team_defense": team_def,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
