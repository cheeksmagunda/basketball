import os
import time
import json
import hashlib
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
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
    """Fetch all player season stats in one request instead of per-player.
    This is much faster and avoids dozens of individual API calls."""
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
    """Fetch last-5-game stats for all players in one request."""
    cache_key = "all_player_stats_last5"
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


# ---------------------------------------------------------------------------
# Projection model
# ---------------------------------------------------------------------------

def calc_fpts(stats: dict) -> float:
    """Calculate fantasy points from per-game averages."""
    return sum(stats.get(k, 0) * v for k, v in FP_WEIGHTS.items())


def project_player_from_avgs(
    name: str, season: dict, last5_min: float, vegas_total: Optional[float]
) -> Optional[dict]:
    """Project fantasy points using season averages + last-5 minutes."""
    avg_min = season.get("min", 0)
    if avg_min < 15:
        return None

    # Fantasy points per minute from season averages
    fpts_per_game = (
        season["pts"] * FP_WEIGHTS["PTS"]
        + season["reb"] * FP_WEIGHTS["REB"]
        + season["ast"] * FP_WEIGHTS["AST"]
        + season["stl"] * FP_WEIGHTS["STL"]
        + season["blk"] * FP_WEIGHTS["BLK"]
        + season["tov"] * FP_WEIGHTS["TOV"]
        + season["fg3m"] * FP_WEIGHTS["FG3M"]
    )
    fppm = fpts_per_game / avg_min if avg_min > 0 else 0

    # Stage 1: Predicted minutes
    pred_minutes = (avg_min * 0.75) + (last5_min * 0.25)

    # Vegas adjustment
    vegas_adj = 1.0
    if vegas_total is not None:
        vegas_adj = vegas_total / 220.0

    projected_fp = pred_minutes * fppm * vegas_adj

    return {
        "name": name,
        "projected_fp": round(projected_fp, 2),
        "pred_minutes": round(pred_minutes, 1),
        "fppm": round(fppm, 3),
        "vegas_adj": round(vegas_adj, 3),
    }


# ---------------------------------------------------------------------------
# API endpoint
# ---------------------------------------------------------------------------

@app.get("/api/picks")
async def get_picks():
    try:
        # 1. Get today's games from CDN (reliable from cloud)
        games = fetch_todays_games()
        if not games:
            return JSONResponse(
                content={"error": "No NBA games scheduled today."},
                status_code=200,
            )

        # 2. Collect team IDs playing today
        team_ids_today = set()
        team_vegas = {}
        vegas_lines = fetch_vegas_lines()
        for game in games:
            for side in ("homeTeam", "awayTeam"):
                tid = game[side]["teamId"]
                tri = game[side]["teamTricode"]
                team_ids_today.add(tid)
                full_name = TRICODE_TO_NAME.get(tri, "")
                if full_name in vegas_lines:
                    team_vegas[tid] = vegas_lines[full_name]

        # 3. Fetch all player season stats in ONE request (instead of per-player)
        all_stats = fetch_all_player_stats()
        if not all_stats:
            return JSONResponse(
                content={"error": "Could not fetch player stats from NBA.com. Please try again later."},
                status_code=200,
            )

        # 4. Fetch last-5 stats in one request
        last5 = fetch_last5_stats()

        # 5. Project players on today's teams
        projections = []
        for pid_str, pdata in all_stats.items():
            pid = int(pid_str) if isinstance(pid_str, str) else pid_str
            if pdata["team_id"] not in team_ids_today:
                continue
            last5_min = last5.get(str(pid), last5.get(pid, {})).get("min", pdata["min"])
            vegas_total = team_vegas.get(pdata["team_id"])
            proj = project_player_from_avgs(pdata["name"], pdata, last5_min, vegas_total)
            if proj is not None:
                projections.append(proj)

        projections.sort(key=lambda x: x["projected_fp"], reverse=True)
        top5 = [{"name": p["name"]} for p in projections[:5]]
        return JSONResponse(content=top5)

    except Exception as e:
        return JSONResponse(
            content={"error": f"Failed to generate picks: {str(e)}"},
            status_code=500,
        )
