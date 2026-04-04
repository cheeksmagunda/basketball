"""MLB Data Fetchers — ESPN MLB API + Park Factors + Weather.

# grep: MLB DATA FETCHERS

Provides MLB-specific data for the Filter-Not-Forecast draft strategy:
  - Game schedule with Vegas lines (spread, total, moneyline)
  - Starting pitcher identification and stats
  - Batting lineup positions
  - Static park factors
  - Player season stats (batting and pitching)

All fetchers follow the same caching pattern as the NBA pipeline:
  Thread-safe, /tmp or Redis cache, graceful fallback on API errors.
"""

from __future__ import annotations

import json
import os
import re
import time
import threading
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# ESPN MLB API
# ─────────────────────────────────────────────────────────────────────────────

ESPN_MLB_BASE = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb"

# ── Cache config ─────────────────────────────────────────────────────────────
CACHE_DIR = Path(os.getenv("CACHE_DIR", "/tmp/mlb_cache_v1"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_TTL_GAMES = 300        # 5 min
_TTL_ROSTER = 1800      # 30 min
_TTL_PLAYER = 1800      # 30 min
_TTL_PITCHER = 1800     # 30 min

_GAMES_CACHE: dict = {}
_GAMES_CACHE_TS: float = 0
_GAMES_LOCK = threading.Lock()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / f"{key}.json"


def _cache_get(key: str, ttl: int = 300) -> Any:
    p = _cache_path(key)
    if p.exists():
        try:
            age = time.time() - p.stat().st_mtime
            if age < ttl:
                return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _cache_set(key: str, data: Any) -> None:
    try:
        _cache_path(key).write_text(json.dumps(data, default=str))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# TEAM ABBREVIATION CANONICALIZATION
# Strategy doc Issue 2: KC/KCR, CWS/CHW must be unified
# ─────────────────────────────────────────────────────────────────────────────

_TEAM_CANON = {
    "KCR": "KC", "KAN": "KC",
    "CHW": "CWS",
    "SFG": "SF", "SFO": "SF",
    "SDP": "SD",
    "TBR": "TB",
    "WSH": "WSN",
    "ANA": "LAA",
}

def canonicalize_team(abbr: str) -> str:
    """Normalize team abbreviation to single canonical form."""
    up = (abbr or "").upper().strip()
    return _TEAM_CANON.get(up, up)


# ─────────────────────────────────────────────────────────────────────────────
# PARK FACTORS — Static run environment multipliers
# Source: Fangraphs park factors (2024-2025 averages)
# 100 = neutral. >100 = hitter-friendly. <100 = pitcher-friendly.
# ─────────────────────────────────────────────────────────────────────────────

PARK_FACTORS = {
    "COL": 114,   # Coors Field — extreme hitter
    "BOS": 107,   # Fenway Park
    "CIN": 106,   # Great American Ball Park
    "TEX": 105,   # Globe Life Field
    "PHI": 104,   # Citizens Bank Park
    "CHC": 103,   # Wrigley Field
    "NYY": 103,   # Yankee Stadium (HR-friendly)
    "ATL": 102,   # Truist Park
    "MIN": 102,   # Target Field
    "MIL": 101,   # American Family Field
    "ARI": 101,   # Chase Field
    "TOR": 101,   # Rogers Centre
    "LAA": 100,   # Angel Stadium
    "DET": 100,   # Comerica Park
    "HOU": 100,   # Minute Maid Park
    "CLE": 100,   # Progressive Field
    "BAL": 100,   # Camden Yards
    "STL": 100,   # Busch Stadium
    "CWS": 99,    # Guaranteed Rate Field
    "KC":  99,    # Kauffman Stadium
    "PIT": 98,    # PNC Park
    "WSN": 98,    # Nationals Park
    "TB":  97,    # Tropicana Field
    "SEA": 96,    # T-Mobile Park
    "LAD": 96,    # Dodger Stadium
    "NYM": 95,    # Citi Field
    "MIA": 94,    # LoanDepot Park
    "OAK": 94,    # Oakland Coliseum
    "SD":  93,    # Petco Park
    "SF":  92,    # Oracle Park — extreme pitcher
}

def get_park_factor(team_abbr: str) -> int:
    """Get park factor for a team's home stadium. 100 = neutral."""
    return PARK_FACTORS.get(canonicalize_team(team_abbr), 100)


def is_pitcher_park(team_abbr: str) -> bool:
    return get_park_factor(team_abbr) <= 96


def is_hitter_park(team_abbr: str) -> bool:
    return get_park_factor(team_abbr) >= 104


# ─────────────────────────────────────────────────────────────────────────────
# HANDEDNESS / PLATOON DATA
# Static: pitcher handedness must be fetched from ESPN or injected
# ─────────────────────────────────────────────────────────────────────────────

def has_platoon_advantage(batter_bats: str, pitcher_throws: str) -> bool:
    """Check if batter has platoon advantage (opposite hand).
    LHH vs RHP or RHH vs LHP = True. Switch hitters always have advantage."""
    if not batter_bats or not pitcher_throws:
        return False
    b = batter_bats.upper().strip()
    p = pitcher_throws.upper().strip()
    if b in ("S", "B", "SWITCH"):
        return True
    return b != p


# ─────────────────────────────────────────────────────────────────────────────
# ESPN MLB GAME SCHEDULE
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mlb_games(date: str | None = None) -> list[dict]:
    """Fetch MLB schedule from ESPN scoreboard API.

    Returns list of game dicts with: gameId, home, away, startTime,
    home_abbr, away_abbr, home_moneyline, away_moneyline, total,
    home_probable_pitcher, away_probable_pitcher, status, etc.

    date: 'YYYY-MM-DD' string or None for today (ET).
    """
    global _GAMES_CACHE, _GAMES_CACHE_TS

    if date is None:
        try:
            from api.shared import et_date
            date = str(et_date())
        except ImportError:
            from datetime import datetime, timezone, timedelta
            date = (datetime.now(timezone.utc) - timedelta(hours=5)).strftime("%Y-%m-%d")

    date_compact = date.replace("-", "")

    with _GAMES_LOCK:
        if _GAMES_CACHE.get("date") == date and time.time() - _GAMES_CACHE_TS < _TTL_GAMES:
            return _GAMES_CACHE.get("games", [])

    # Check file cache
    cached = _cache_get(f"mlb_games_{date_compact}", _TTL_GAMES)
    if cached:
        with _GAMES_LOCK:
            _GAMES_CACHE = {"date": date, "games": cached}
            _GAMES_CACHE_TS = time.time()
        return cached

    url = f"{ESPN_MLB_BASE}/scoreboard?dates={date_compact}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[mlb_data] ESPN scoreboard fetch failed: {e}")
        return []

    games = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        if len(competitors) < 2:
            continue

        home = away = None
        for c in competitors:
            team_data = c.get("team", {})
            entry = {
                "id": team_data.get("id", ""),
                "abbr": canonicalize_team(team_data.get("abbreviation", "")),
                "name": team_data.get("displayName", ""),
                "short": team_data.get("shortDisplayName", ""),
                "score": c.get("score", "0"),
                "moneyline": 0,
                "probable_pitcher": _extract_probable_pitcher(c),
            }
            # Extract moneyline from odds if available
            for odds_item in comp.get("odds", []):
                for aw in odds_item.get("awayTeamOdds", {}).get("moneyLine", [None]):
                    pass  # structure varies
                break

            if c.get("homeAway") == "home":
                home = entry
            else:
                away = entry

        if not home or not away:
            continue

        # Extract odds (spread/total/moneyline)
        spread = 0
        total = 0
        home_ml = 0
        away_ml = 0
        for odds_block in comp.get("odds", []):
            spread = _safe_float(odds_block.get("spread", 0))
            total = _safe_float(odds_block.get("overUnder", 0))
            home_odds = odds_block.get("homeTeamOdds", {})
            away_odds = odds_block.get("awayTeamOdds", {})
            home_ml = _safe_float(home_odds.get("moneyLine", 0))
            away_ml = _safe_float(away_odds.get("moneyLine", 0))
            break

        # Determine status
        status_obj = comp.get("status", {}).get("type", {})
        status = status_obj.get("name", "STATUS_SCHEDULED")

        game = {
            "gameId": ev.get("id", ""),
            "home": home["abbr"],
            "away": away["abbr"],
            "home_name": home["name"],
            "away_name": away["name"],
            "home_id": home["id"],
            "away_id": away["id"],
            "startTime": ev.get("date", ""),
            "spread": spread,
            "total": total,
            "home_moneyline": home_ml,
            "away_moneyline": away_ml,
            "home_probable_pitcher": home["probable_pitcher"],
            "away_probable_pitcher": away["probable_pitcher"],
            "park_factor": get_park_factor(home["abbr"]),
            "status": status,
            "home_score": home["score"],
            "away_score": away["score"],
        }
        games.append(game)

    _cache_set(f"mlb_games_{date_compact}", games)
    with _GAMES_LOCK:
        _GAMES_CACHE = {"date": date, "games": games}
        _GAMES_CACHE_TS = time.time()

    print(f"[mlb_data] Fetched {len(games)} MLB games for {date}")
    return games


def _extract_probable_pitcher(competitor: dict) -> dict | None:
    """Extract probable pitcher info from ESPN competitor data."""
    # ESPN embeds probable pitchers in the competitor or in a probables array
    probables = competitor.get("probables", [])
    if probables:
        p = probables[0]
        athlete = p.get("athlete", {})
        return {
            "id": athlete.get("id", ""),
            "name": athlete.get("displayName", athlete.get("fullName", "")),
            "hand": athlete.get("hand", {}).get("abbreviation", "R"),
            "era": _safe_float(p.get("statistics", [{}])[0].get("displayValue") if p.get("statistics") else None),
        }
    return None


def _safe_float(v, default=0.0) -> float:
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER STATS FETCHERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_mlb_roster(team_id: str, team_abbr: str = "") -> list[dict]:
    """Fetch team roster from ESPN MLB."""
    cache_key = f"mlb_roster_{team_id}"
    cached = _cache_get(cache_key, _TTL_ROSTER)
    if cached:
        return cached

    url = f"{ESPN_MLB_BASE}/teams/{team_id}/roster"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[mlb_data] Roster fetch failed for team {team_id}: {e}")
        return []

    players = []
    for group in data.get("athletes", []):
        position_group = group.get("position", "")
        for athlete in group.get("items", []):
            player = {
                "id": athlete.get("id", ""),
                "name": athlete.get("displayName", athlete.get("fullName", "")),
                "pos": athlete.get("position", {}).get("abbreviation", ""),
                "bats": athlete.get("bats", {}).get("abbreviation", ""),
                "throws": athlete.get("throws", {}).get("abbreviation", ""),
                "team": canonicalize_team(team_abbr),
                "team_id": team_id,
                "jersey": athlete.get("jersey", ""),
                "position_group": position_group,
                "is_pitcher": athlete.get("position", {}).get("abbreviation", "") in ("SP", "RP", "P", "CL"),
                "injury_status": "",
                "is_out": False,
            }
            # Check injuries
            for inj in athlete.get("injuries", []):
                status = inj.get("status", "")
                if status.upper() in ("OUT", "IL", "10-DAY IL", "60-DAY IL", "15-DAY IL"):
                    player["is_out"] = True
                    player["injury_status"] = "OUT"
                elif status.upper() in ("DAY-TO-DAY", "DTD"):
                    player["injury_status"] = "DTD"
            players.append(player)

    _cache_set(cache_key, players)
    return players


def fetch_mlb_player_stats(player_id: str) -> dict:
    """Fetch season stats for an MLB player from ESPN."""
    cache_key = f"mlb_player_{player_id}"
    cached = _cache_get(cache_key, _TTL_PLAYER)
    if cached:
        return cached

    url = f"{ESPN_MLB_BASE}/athletes/{player_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[mlb_data] Player stats fetch failed for {player_id}: {e}")
        return {}

    athlete = data.get("athlete", data)
    stats = {}

    # Extract season stats from statistics array
    for stat_group in athlete.get("statistics", []):
        splits = stat_group.get("splits", {})
        categories = splits.get("categories", [])
        for cat in categories:
            for stat in cat.get("stats", []):
                name = stat.get("name", "")
                value = stat.get("value", stat.get("displayValue", ""))
                stats[name] = _safe_float(value, value)

    result = {
        "id": player_id,
        "name": athlete.get("displayName", ""),
        "pos": athlete.get("position", {}).get("abbreviation", ""),
        "bats": athlete.get("bats", {}).get("abbreviation", "R"),
        "throws": athlete.get("throws", {}).get("abbreviation", "R"),
        # Hitter stats
        "avg": _safe_float(stats.get("avg", stats.get("battingAverage", 0))),
        "hr": _safe_float(stats.get("homeRuns", stats.get("HR", 0))),
        "rbi": _safe_float(stats.get("RBIs", stats.get("RBI", 0))),
        "runs": _safe_float(stats.get("runs", stats.get("R", 0))),
        "sb": _safe_float(stats.get("stolenBases", stats.get("SB", 0))),
        "hits": _safe_float(stats.get("hits", stats.get("H", 0))),
        "ab": _safe_float(stats.get("atBats", stats.get("AB", 0))),
        "obp": _safe_float(stats.get("OBP", stats.get("onBasePercentage", 0))),
        "slg": _safe_float(stats.get("SLG", stats.get("sluggingPercentage", 0))),
        "ops": _safe_float(stats.get("OPS", 0)),
        "bb": _safe_float(stats.get("walks", stats.get("BB", 0))),
        "so": _safe_float(stats.get("strikeOuts", stats.get("SO", 0))),
        "games": _safe_float(stats.get("gamesPlayed", stats.get("GP", 0))),
        # Pitcher stats
        "era": _safe_float(stats.get("ERA", stats.get("earnedRunAverage", 0))),
        "whip": _safe_float(stats.get("WHIP", 0)),
        "k9": _safe_float(stats.get("strikeoutsPerNineInnings", stats.get("SO9", 0))),
        "ip": _safe_float(stats.get("inningsPitched", stats.get("IP", 0))),
        "wins": _safe_float(stats.get("wins", stats.get("W", 0))),
        "losses": _safe_float(stats.get("losses", stats.get("L", 0))),
        "k_rate": _safe_float(stats.get("strikeoutRate", 0)),
        "bb9": _safe_float(stats.get("walksPerNineInnings", stats.get("BB9", 0))),
        "hr9": _safe_float(stats.get("homeRunsPerNineInnings", stats.get("HR9", 0))),
        "raw_stats": stats,
    }

    _cache_set(cache_key, result)
    return result


def fetch_mlb_team_stats(team_id: str) -> dict:
    """Fetch team-level batting/pitching stats from ESPN."""
    cache_key = f"mlb_team_{team_id}"
    cached = _cache_get(cache_key, _TTL_ROSTER)
    if cached:
        return cached

    url = f"{ESPN_MLB_BASE}/teams/{team_id}/statistics"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[mlb_data] Team stats fetch failed for {team_id}: {e}")
        return {}

    stats = {}
    for group in data.get("results", data.get("statistics", [])):
        for cat in group.get("splits", {}).get("categories", []):
            for stat in cat.get("stats", []):
                name = stat.get("name", "")
                value = stat.get("value", stat.get("displayValue", ""))
                stats[name] = _safe_float(value, value)

    result = {
        "team_id": team_id,
        "batting_avg": _safe_float(stats.get("avg", stats.get("battingAverage", 0))),
        "runs_per_game": _safe_float(stats.get("runsPerGame", stats.get("R/G", 0))),
        "team_ops": _safe_float(stats.get("OPS", 0)),
        "team_k_rate": _safe_float(stats.get("strikeoutRate", 0)),
        "team_woba": _safe_float(stats.get("wOBA", 0)),
        "team_era": _safe_float(stats.get("ERA", stats.get("earnedRunAverage", 0))),
        "team_whip": _safe_float(stats.get("WHIP", 0)),
        "raw_stats": stats,
    }

    _cache_set(cache_key, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# BATTING LINEUP FETCHER
# ─────────────────────────────────────────────────────────────────────────────

def fetch_batting_lineup(game_id: str) -> dict:
    """Fetch batting order for a specific game from ESPN.
    Returns {home: [{name, pos, order, id}], away: [...]}.
    Available closer to game time."""
    cache_key = f"mlb_lineup_{game_id}"
    cached = _cache_get(cache_key, _TTL_PITCHER)
    if cached:
        return cached

    # ESPN boxscore endpoint has lineup data
    url = f"{ESPN_MLB_BASE}/summary?event={game_id}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[mlb_data] Lineup fetch failed for game {game_id}: {e}")
        return {"home": [], "away": []}

    result = {"home": [], "away": []}
    for roster_entry in data.get("rosters", []):
        side = "home" if roster_entry.get("homeAway") == "home" else "away"
        for player in roster_entry.get("roster", []):
            athlete = player.get("athlete", {})
            order = player.get("battingOrder", player.get("order", 0))
            result[side].append({
                "id": athlete.get("id", ""),
                "name": athlete.get("displayName", ""),
                "pos": athlete.get("position", {}).get("abbreviation", ""),
                "order": _safe_float(order),
                "bats": athlete.get("bats", {}).get("abbreviation", ""),
            })

    _cache_set(cache_key, result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PARALLEL FETCHER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_game_data(games: list[dict], max_workers: int = 8) -> dict:
    """Parallel-fetch rosters, stats, and lineups for all games.
    Returns a dict keyed by gameId with roster/lineup/stats data."""
    result = {}

    def _fetch_game(game):
        gid = game["gameId"]
        home_id = game.get("home_id", "")
        away_id = game.get("away_id", "")
        home_abbr = game.get("home", "")
        away_abbr = game.get("away", "")

        home_roster = fetch_mlb_roster(home_id, home_abbr) if home_id else []
        away_roster = fetch_mlb_roster(away_id, away_abbr) if away_id else []
        lineup = fetch_batting_lineup(gid)

        return gid, {
            "home_roster": home_roster,
            "away_roster": away_roster,
            "lineup": lineup,
            "game": game,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_fetch_game, g): g for g in games}
        for fut in as_completed(futures):
            try:
                gid, data = fut.result()
                result[gid] = data
            except Exception as e:
                print(f"[mlb_data] Game data fetch error: {e}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER (simplified — real integration needs a weather API)
# ─────────────────────────────────────────────────────────────────────────────

# Dome stadiums where weather is irrelevant
_DOME_STADIUMS = {"HOU", "TB", "MIA", "TOR", "MIL", "ARI", "SEA", "TEX"}

def is_dome(team_abbr: str) -> bool:
    """Check if team plays in a dome/retractable roof stadium."""
    return canonicalize_team(team_abbr) in _DOME_STADIUMS


def get_weather_factor(home_team: str, temperature: float = 72, wind_mph: float = 5, wind_out: bool = False) -> float:
    """Compute weather-based run environment multiplier.
    Returns a factor: >1.0 = hitter-friendly, <1.0 = pitcher-friendly.

    Dome stadiums always return 1.0 (neutral).
    High temp (>85°F) + wind blowing out = hitter boost.
    Cold (<50°F) = pitcher boost.
    """
    if is_dome(home_team):
        return 1.0

    factor = 1.0

    # Temperature effect
    if temperature >= 85:
        factor += 0.03
    elif temperature >= 75:
        factor += 0.01
    elif temperature <= 50:
        factor -= 0.03
    elif temperature <= 60:
        factor -= 0.01

    # Wind effect
    if wind_out and wind_mph >= 15:
        factor += 0.05
    elif wind_out and wind_mph >= 10:
        factor += 0.02
    elif not wind_out and wind_mph >= 15:
        factor -= 0.02

    return round(factor, 3)
