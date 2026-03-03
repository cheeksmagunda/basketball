import os
import time
import json
import hashlib
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
CACHE_DIR = Path("/tmp/nba_dfs_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Browser-like headers for stats.nba.com requests
NBA_HEADERS = {
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Connection": "keep-alive",
}

# Scoring weights (fantasy points)
FP_WEIGHTS = {
    "PTS": 1.0,
    "REB": 1.25,
    "AST": 1.5,
    "STL": 2.0,
    "BLK": 2.0,
    "TOV": -0.5,
    "FG3M": 0.5,
}

TRICODE_TO_NAME = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
    "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

# Draft slot multipliers for the 5-pick lineup
SLOT_VALUES = ["2.0x", "1.8x", "1.6x", "1.4x", "1.2x"]


# ---------------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str) -> Path:
    today = date.today().isoformat()
    safe = hashlib.md5(f"{today}:{key}".encode()).hexdigest()
    return CACHE_DIR / f"{safe}.json"


def _read_cache(key: str):
    p = _cache_path(key)
    if p.exists():
        return json.loads(p.read_text())
    return None


def _write_cache(key: str, data):
    p = _cache_path(key)
    p.write_text(json.dumps(data))


# ---------------------------------------------------------------------------
# NBA data fetching — uses cdn.nba.com (no IP blocking) where possible,
# falls back to stats.nba.com with browser headers + retries.
# ---------------------------------------------------------------------------

def _nba_stats_request(url: str, params: dict, retries: int = 3) -> dict:
    """Make a request to stats.nba.com with browser headers and retries."""
    for attempt in range(retries):
        try:
            resp = requests.get(
                url, params=params, headers=NBA_HEADERS, timeout=60
            )
            resp.raise_for_status()
            return resp.json()
        except Exception:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    return {}


def fetch_todays_games():
    """Fetch today's NBA games from cdn.nba.com (no IP restrictions)."""
    cached = _read_cache("todays_games")
    if cached is not None:
        return cached

    url = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    games = []
    for game in data.get("scoreboard", {}).get("games", []):
        games.append({
            "gameId": game["gameId"],
            "homeTeam": {
                "teamId": game["homeTeam"]["teamId"],
                "teamTricode": game["homeTeam"]["teamTricode"],
            },
            "awayTeam": {
                "teamId": game["awayTeam"]["teamId"],
                "teamTricode": game["awayTeam"]["teamTricode"],
            },
        })
    _write_cache("todays_games", games)
    return games


def fetch_team_roster(team_id: int):
    """Fetch roster via stats.nba.com with browser headers."""
    cache_key = f"roster_{team_id}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    time.sleep(1)
    url = "https://stats.nba.com/stats/commonteamroster"
    params = {"TeamID": team_id, "Season": "2025-26"}
    data = _nba_stats_request(url, params)

    players = []
    result_sets = data.get("resultSets", [])
    if result_sets:
        headers = result_sets[0].get("headers", [])
        rows = result_sets[0].get("rowSet", [])
        pid_idx = headers.index("PLAYER_ID") if "PLAYER_ID" in headers else None
        name_idx = headers.index("PLAYER") if "PLAYER" in headers else None
        if pid_idx is not None and name_idx is not None:
            for row in rows:
                players.append({
                    "playerId": row[pid_idx],
                    "playerName": row[name_idx],
                })

    _write_cache(cache_key, players)
    return players


def fetch_player_gamelog(player_id: int, season: str = "2025-26"):
    """Fetch player game log via stats.nba.com with browser headers."""
    cache_key = f"gamelog_{player_id}_{season}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    time.sleep(1)
    url = "https://stats.nba.com/stats/playergamelog"
    params = {
        "PlayerID": player_id,
        "Season": season,
        "SeasonType": "Regular Season",
    }
    data = _nba_stats_request(url, params)

    games = []
    result_sets = data.get("resultSets", [])
    if result_sets:
        headers = result_sets[0].get("headers", [])
        rows = result_sets[0].get("rowSet", [])
        col = {h: i for i, h in enumerate(headers)}
        for row in rows:
            min_val = row[col["MIN"]] if "MIN" in col else 0
            try:
                mins = float(min_val) if min_val else 0.0
            except (ValueError, TypeError):
                mins = 0.0
            games.append({
                "MIN": mins,
                "PTS": row[col.get("PTS", 0)] or 0,
                "REB": row[col.get("REB", 0)] or 0,
                "AST": row[col.get("AST", 0)] or 0,
                "STL": row[col.get("STL", 0)] or 0,
                "BLK": row[col.get("BLK", 0)] or 0,
                "TOV": row[col.get("TOV", 0)] or 0,
                "FG3M": row[col.get("FG3M", 0)] or 0,
            })

    _write_cache(cache_key, games)
    return games


def fetch_all_player_stats():
    """Fetch all player season stats in one request instead of per-player."""
    cache_key = "all_player_stats"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    url = "https://stats.nba.com/stats/leaguedashplayerstats"
    params = {
        "Conference": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "GameScope": "",
        "GameSegment": "",
        "Height": "",
        "ISTRound": "",
        "LastNGames": 0,
        "LeagueID": "00",
        "Location": "",
        "MeasureType": "Base",
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "",
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": "2025-26",
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": 0,
        "TwoWay": 0,
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }
    data = _nba_stats_request(url, params)

    stats = {}
    result_sets = data.get("resultSets", [])
    if result_sets:
        headers = result_sets[0].get("headers", [])
        rows = result_sets[0].get("rowSet", [])
        col = {h: i for i, h in enumerate(headers)}
        for row in rows:
            pid = row[col.get("PLAYER_ID", 0)]
            stats[pid] = {
                "name": row[col.get("PLAYER_NAME", 1)],
                "team_id": row[col.get("TEAM_ID", 2)],
                "team_abbrev": row[col.get("TEAM_ABBREVIATION", 3)] if "TEAM_ABBREVIATION" in col else "",
                "gp": row[col.get("GP", 0)] or 0,
                "min": row[col.get("MIN", 0)] or 0,
                "pts": row[col.get("PTS", 0)] or 0,
                "reb": row[col.get("REB", 0)] or 0,
                "ast": row[col.get("AST", 0)] or 0,
                "stl": row[col.get("STL", 0)] or 0,
                "blk": row[col.get("BLK", 0)] or 0,
                "tov": row[col.get("TOV", 0)] or 0,
                "fg3m": row[col.get("FG3M", 0)] or 0,
            }

    _write_cache(cache_key, stats)
    return stats


def fetch_last5_stats():
    """Fetch last-5-game stats for all players (full stat lines)."""
    cache_key = "all_player_stats_last5_full"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    url = "https://stats.nba.com/stats/leaguedashplayerstats"
    params = {
        "Conference": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "GameScope": "",
        "GameSegment": "",
        "Height": "",
        "ISTRound": "",
        "LastNGames": 5,
        "LeagueID": "00",
        "Location": "",
        "MeasureType": "Base",
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "",
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": "2025-26",
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": 0,
        "TwoWay": 0,
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }
    data = _nba_stats_request(url, params)

    stats = {}
    result_sets = data.get("resultSets", [])
    if result_sets:
        headers = result_sets[0].get("headers", [])
        rows = result_sets[0].get("rowSet", [])
        col = {h: i for i, h in enumerate(headers)}
        for row in rows:
            pid = row[col.get("PLAYER_ID", 0)]
            stats[pid] = {
                "min": row[col.get("MIN", 0)] or 0,
                "pts": row[col.get("PTS", 0)] or 0,
                "reb": row[col.get("REB", 0)] or 0,
                "ast": row[col.get("AST", 0)] or 0,
                "stl": row[col.get("STL", 0)] or 0,
                "blk": row[col.get("BLK", 0)] or 0,
                "tov": row[col.get("TOV", 0)] or 0,
                "fg3m": row[col.get("FG3M", 0)] or 0,
            }

    _write_cache(cache_key, stats)
    return stats


def fetch_vegas_lines():
    """Fetch NBA game totals from The Odds API."""
    cached = _read_cache("vegas_lines")
    if cached is not None:
        return cached

    if not ODDS_API_KEY:
        return {}

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "totals",
        "oddsFormat": "american",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}

    lines = {}
    for game in data:
        home = game.get("home_team", "")
        away = game.get("away_team", "")
        total = None
        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market["key"] == "totals":
                    for outcome in market.get("outcomes", []):
                        if outcome["name"] == "Over":
                            total = outcome.get("point")
                            break
            if total is not None:
                break
        if total is not None:
            lines[home] = total
            lines[away] = total

    _write_cache("vegas_lines", lines)
    return lines


def fetch_team_defense_ratings():
    """Fetch team defensive ratings (DEF_RATING = points allowed per 100 poss).

    Higher DEF_RATING means worse defense, which is better for opposing players.
    """
    cache_key = "team_def_ratings"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    url = "https://stats.nba.com/stats/leaguedashteamstats"
    params = {
        "Conference": "",
        "DateFrom": "",
        "DateTo": "",
        "Division": "",
        "GameScope": "",
        "GameSegment": "",
        "Height": "",
        "ISTRound": "",
        "LastNGames": 0,
        "LeagueID": "00",
        "Location": "",
        "MeasureType": "Advanced",
        "Month": 0,
        "OpponentTeamID": 0,
        "Outcome": "",
        "PORound": 0,
        "PaceAdjust": "N",
        "PerMode": "PerGame",
        "Period": 0,
        "PlayerExperience": "",
        "PlayerPosition": "",
        "PlusMinus": "N",
        "Rank": "N",
        "Season": "2025-26",
        "SeasonSegment": "",
        "SeasonType": "Regular Season",
        "ShotClockRange": "",
        "StarterBench": "",
        "TeamID": 0,
        "TwoWay": 0,
        "VsConference": "",
        "VsDivision": "",
        "Weight": "",
    }
    data = _nba_stats_request(url, params)

    ratings = {}
    result_sets = data.get("resultSets", [])
    if result_sets:
        headers = result_sets[0].get("headers", [])
        rows = result_sets[0].get("rowSet", [])
        col = {h: i for i, h in enumerate(headers)}
        for row in rows:
            tid = row[col.get("TEAM_ID", 0)]
            def_rating = row[col.get("DEF_RATING", 0)] if "DEF_RATING" in col else None
            pace = row[col.get("PACE", 0)] if "PACE" in col else None
            ratings[tid] = {
                "def_rating": float(def_rating) if def_rating else 112.0,
                "pace": float(pace) if pace else 100.0,
            }

    _write_cache(cache_key, ratings)
    return ratings


# ---------------------------------------------------------------------------
# Projection model — value-based with matchup, form, and opportunity factors
# ---------------------------------------------------------------------------

def _calc_fp(stats: dict) -> float:
    """Calculate fantasy points from a stat line dict (lowercase keys)."""
    return (
        (stats.get("pts", 0) or 0) * FP_WEIGHTS["PTS"]
        + (stats.get("reb", 0) or 0) * FP_WEIGHTS["REB"]
        + (stats.get("ast", 0) or 0) * FP_WEIGHTS["AST"]
        + (stats.get("stl", 0) or 0) * FP_WEIGHTS["STL"]
        + (stats.get("blk", 0) or 0) * FP_WEIGHTS["BLK"]
        + (stats.get("tov", 0) or 0) * FP_WEIGHTS["TOV"]
        + (stats.get("fg3m", 0) or 0) * FP_WEIGHTS["FG3M"]
    )


def project_player_value(
    name: str,
    team_abbrev: str,
    season: dict,
    last5: dict,
    opp_def_rating: float,
    league_avg_def: float,
    vegas_total: Optional[float],
    opp_team: str,
    game_label: str,
) -> Optional[dict]:
    """Project player value for tonight using DFS-style analysis.

    Factors in: matchup quality (defense vs position), recent form,
    minutes opportunity, and Vegas game environment.  Returns a projection
    dict with outperformance boost score for draft slot ranking.
    """
    avg_min = season.get("min", 0) or 0
    if avg_min < 15:
        return None

    # --- Base fantasy points from season averages ---
    season_fp = _calc_fp(season)
    if season_fp <= 0:
        return None

    # Fantasy points per minute (efficiency metric)
    fppm = season_fp / avg_min

    # --- Last 5 games fantasy points (recent form) ---
    last5_fp = _calc_fp(last5) if last5 else season_fp
    last5_min = (last5.get("min", 0) or 0) if last5 else avg_min
    if last5_min <= 0:
        last5_min = avg_min

    # 1. FORM FACTOR — how hot/cold is the player vs their season baseline?
    #    Capped at +/-40% adjustment per DFS best practices (avoid recency bias)
    form_ratio = last5_fp / season_fp if season_fp > 0 else 1.0
    form_factor = max(0.6, min(1.4, form_ratio))

    # 2. MINUTES OPPORTUNITY — trending minutes indicate role changes / injuries
    #    Players seeing more minutes recently likely have expanded opportunity
    min_ratio = last5_min / avg_min if avg_min > 0 else 1.0
    min_factor = max(0.8, min(1.3, min_ratio))

    # 3. MATCHUP FACTOR — opponent defensive weakness (Defense vs Position proxy)
    #    Higher DEF_RATING = worse defense = more fantasy points for opposing players
    #    Typical range: 105 (elite D) to 120 (terrible D), league avg ~112
    if league_avg_def > 0 and opp_def_rating > 0:
        def_ratio = opp_def_rating / league_avg_def
        matchup_factor = max(0.85, min(1.20, def_ratio))
    else:
        matchup_factor = 1.0

    # 4. VEGAS PACE FACTOR — game total implies scoring environment
    #    High totals (230+) = run-and-gun, Low totals (<210) = grind
    if vegas_total and vegas_total > 0:
        vegas_factor = vegas_total / 220.0
    else:
        vegas_factor = 1.0

    # 5. PREDICTED MINUTES — weighted toward recent (40% recency per DFS models)
    pred_minutes = (avg_min * 0.6) + (last5_min * 0.4)

    # 6. COMPOSITE PROJECTION
    projected_fp = pred_minutes * fppm * form_factor * matchup_factor * vegas_factor

    # 7. BOOST SCORE — outperformance potential relative to season baseline
    #    This is the key metric for value-based draft slot assignment
    #    boost > 1.0 means projecting ABOVE their season average tonight
    boost = projected_fp / season_fp if season_fp > 0 else 1.0

    return {
        "name": name,
        "team": team_abbrev,
        "opponent": opp_team,
        "game": game_label,
        "projected_fp": round(projected_fp, 1),
        "season_fp": round(season_fp, 1),
        "boost": round(boost, 2),
        "form_ratio": round(form_ratio, 2),
        "matchup_factor": round(matchup_factor, 2),
        "min_factor": round(min_factor, 2),
        "vegas_factor": round(vegas_factor, 2),
    }


def rank_players_for_mode(projections: list, mode: str) -> list:
    """Rank and select top 5 players based on the mode-specific scoring.

    Chalk:       Raw projection power — safest, highest-floor picks.
    Diff:        Balanced projection * boost — smart value plays.
    Contrarian:  Boost-heavy — finds hidden gems outperforming expectations.
    """
    for p in projections:
        fp = p["projected_fp"]
        boost = max(p["boost"], 0.1)

        if mode == "chalk":
            # Favor high raw projection with slight matchup tilt
            # Stars with good matchups dominate here
            p["mode_score"] = fp * (boost ** 0.3)
        elif mode == "diff":
            # Balanced — projection weighted by outperformance potential
            # Smart mid-tier picks with great matchups rise
            p["mode_score"] = fp * boost
        else:  # contrarian
            # Value-first — find players who will exceed expectations the most
            # Role players on hot streaks facing bad defenses shine here
            p["mode_score"] = fp * (boost ** 1.5)

    ranked = sorted(projections, key=lambda x: x["mode_score"], reverse=True)
    return ranked[:5]


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@app.get("/api/picks")
async def get_picks(mode: str = Query(default="chalk")):
    try:
        if mode not in ("chalk", "diff", "contrarian"):
            mode = "chalk"

        # 1. Get today's games from CDN (reliable from cloud)
        games = fetch_todays_games()
        if not games:
            return JSONResponse(
                content={"error": "No NBA games scheduled today."},
                status_code=200,
            )

        # 2. Build game context — map every team to its opponent and game label
        team_ids_today = set()
        team_vegas: dict = {}
        team_opponent: dict = {}       # team_id -> opponent tricode
        team_game_label: dict = {}     # team_id -> "HOU @ WAS"
        tri_to_id: dict = {}           # tricode -> team_id
        id_to_tri: dict = {}           # team_id -> tricode

        vegas_lines = fetch_vegas_lines()

        for game in games:
            home_id = game["homeTeam"]["teamId"]
            away_id = game["awayTeam"]["teamId"]
            home_tri = game["homeTeam"]["teamTricode"]
            away_tri = game["awayTeam"]["teamTricode"]

            team_ids_today.add(home_id)
            team_ids_today.add(away_id)

            tri_to_id[home_tri] = home_id
            tri_to_id[away_tri] = away_id
            id_to_tri[home_id] = home_tri
            id_to_tri[away_id] = away_tri

            # Opponent mapping
            team_opponent[home_id] = away_tri
            team_opponent[away_id] = home_tri

            # Game label
            label = f"{away_tri} @ {home_tri}"
            team_game_label[home_id] = label
            team_game_label[away_id] = label

            # Vegas lines by team_id
            for side_key, tid in [("homeTeam", home_id), ("awayTeam", away_id)]:
                tri = game[side_key]["teamTricode"]
                full_name = TRICODE_TO_NAME.get(tri, "")
                if full_name in vegas_lines:
                    team_vegas[tid] = vegas_lines[full_name]

        # 3. Fetch all data sources
        all_stats = fetch_all_player_stats()
        if not all_stats:
            return JSONResponse(
                content={"error": "Could not fetch player stats from NBA.com. Please try again later."},
                status_code=200,
            )

        last5 = fetch_last5_stats()
        defense_ratings = fetch_team_defense_ratings()

        # League average DEF_RATING for normalization
        if defense_ratings:
            all_defs = [
                t["def_rating"]
                for t in defense_ratings.values()
                if isinstance(t, dict)
            ]
            league_avg_def = sum(all_defs) / len(all_defs) if all_defs else 112.0
        else:
            league_avg_def = 112.0

        # 4. Project all eligible players on today's teams
        projections = []
        for pid_str, pdata in all_stats.items():
            pid = int(pid_str) if isinstance(pid_str, str) else pid_str
            tid = pdata["team_id"]

            if tid not in team_ids_today:
                continue

            # Opponent info
            opp_tri = team_opponent.get(tid, "")
            opp_id = tri_to_id.get(opp_tri)

            # Opponent's defensive rating (higher = worse defense = better for player)
            opp_def = 112.0
            if opp_id is not None and defense_ratings:
                opp_entry = defense_ratings.get(opp_id) or defense_ratings.get(str(opp_id))
                if opp_entry and isinstance(opp_entry, dict):
                    opp_def = opp_entry.get("def_rating", 112.0)

            # Player's last 5 game stats (full stat line)
            p_last5 = last5.get(str(pid)) or last5.get(pid) or {}

            # Player's team tricode
            player_tri = pdata.get("team_abbrev") or id_to_tri.get(tid, "")

            proj = project_player_value(
                name=pdata["name"],
                team_abbrev=player_tri,
                season=pdata,
                last5=p_last5,
                opp_def_rating=opp_def,
                league_avg_def=league_avg_def,
                vegas_total=team_vegas.get(tid),
                opp_team=opp_tri,
                game_label=team_game_label.get(tid, ""),
            )
            if proj is not None:
                projections.append(proj)

        # 5. Rank for the selected mode and pick top 5
        top5 = rank_players_for_mode(projections, mode)

        # 6. Assign draft slots and build response
        result = []
        for i, p in enumerate(top5):
            slot = SLOT_VALUES[i] if i < len(SLOT_VALUES) else "1.0x"
            result.append({
                "name": p["name"],
                "team": p["team"],
                "opponent": p["opponent"],
                "game": p["game"],
                "slot": slot,
                "projected_fp": p["projected_fp"],
                "boost": p["boost"],
                "matchup_factor": p["matchup_factor"],
                "form_ratio": p["form_ratio"],
            })

        return JSONResponse(content=result)

    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to generate picks: {str(e)}"},
            status_code=500,
        )
