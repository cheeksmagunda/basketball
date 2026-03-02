import os
import time
import json
import hashlib
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

import requests
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
CACHE_DIR = Path("/tmp/nba_dfs_cache")
CACHE_DIR.mkdir(exist_ok=True)

# DraftKings scoring weights
DK_WEIGHTS = {
    "PTS": 1.0,
    "REB": 1.25,
    "AST": 1.5,
    "STL": 2.0,
    "BLK": 2.0,
    "TOV": -0.5,
    "FG3M": 0.5,
}

# Mapping from nba_api team tricodes to Odds API full names
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


def fetch_todays_games():
    """Fetch today's NBA games using nba_api live scoreboard."""
    cached = _read_cache("todays_games")
    if cached is not None:
        return cached

    from nba_api.live.nba.endpoints import scoreboard

    sb = scoreboard.ScoreBoard()
    games_data = sb.get_dict()
    games = []
    for game in games_data.get("scoreboard", {}).get("games", []):
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
    """Fetch roster for a team."""
    cache_key = f"roster_{team_id}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    from nba_api.stats.endpoints import commonteamroster

    time.sleep(1.5)
    roster = commonteamroster.CommonTeamRoster(team_id=team_id)
    players = []
    for row in roster.get_normalized_dict()["CommonTeamRoster"]:
        players.append(
            {"playerId": row["PLAYER_ID"], "playerName": row["PLAYER"]}
        )
    _write_cache(cache_key, players)
    return players


def fetch_player_gamelog(player_id: int, season: str = "2025-26"):
    """Fetch season game log for a player."""
    cache_key = f"gamelog_{player_id}_{season}"
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    from nba_api.stats.endpoints import playergamelog

    time.sleep(1.5)
    log = playergamelog.PlayerGameLog(
        player_id=player_id, season=season, season_type_all_star="Regular Season"
    )
    rows = log.get_normalized_dict().get("PlayerGameLog", [])
    games = []
    for r in rows:
        min_str = r.get("MIN", "0")
        try:
            mins = float(min_str) if min_str else 0.0
        except (ValueError, TypeError):
            mins = 0.0
        games.append({
            "MIN": mins,
            "PTS": r.get("PTS", 0) or 0,
            "REB": r.get("REB", 0) or 0,
            "AST": r.get("AST", 0) or 0,
            "STL": r.get("STL", 0) or 0,
            "BLK": r.get("BLK", 0) or 0,
            "TOV": r.get("TOV", 0) or 0,
            "FG3M": r.get("FG3M", 0) or 0,
        })
    _write_cache(cache_key, games)
    return games


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


def calc_dk_fpts(stats: dict) -> float:
    """Calculate DraftKings fantasy points from a stat line."""
    return sum(stats.get(k, 0) * v for k, v in DK_WEIGHTS.items())


def project_player(player_name: str, games: list, vegas_total: Optional[float]) -> Optional[dict]:
    """Project fantasy points for a player using the two-stage model."""
    if not games:
        return None

    season_mins = [g["MIN"] for g in games]
    avg_min_season = sum(season_mins) / len(season_mins) if season_mins else 0

    if avg_min_season < 15:
        return None

    season_fpts = [calc_dk_fpts(g) for g in games]
    season_fpts_per_min = []
    for g, fpts in zip(games, season_fpts):
        if g["MIN"] > 0:
            season_fpts_per_min.append(fpts / g["MIN"])
    avg_fppm = (
        sum(season_fpts_per_min) / len(season_fpts_per_min)
        if season_fpts_per_min
        else 0
    )

    last5 = games[:5]
    last5_mins = [g["MIN"] for g in last5]
    avg_min_last5 = sum(last5_mins) / len(last5_mins) if last5_mins else avg_min_season

    pred_minutes = (avg_min_season * 0.75) + (avg_min_last5 * 0.25)

    vegas_adj = 1.0
    if vegas_total is not None:
        vegas_adj = vegas_total / 220.0

    projected_fp = pred_minutes * avg_fppm * vegas_adj

    return {
        "name": player_name,
        "projected_fp": round(projected_fp, 2),
        "pred_minutes": round(pred_minutes, 1),
        "fppm": round(avg_fppm, 3),
        "vegas_adj": round(vegas_adj, 3),
    }


@app.get("/api/picks")
async def get_picks():
    try:
        games = fetch_todays_games()
        if not games:
            return JSONResponse(
                content={"error": "No NBA games scheduled today."},
                status_code=200,
            )

        vegas_lines = fetch_vegas_lines()

        teams = []
        for game in games:
            teams.append(
                (game["homeTeam"]["teamId"], game["homeTeam"]["teamTricode"])
            )
            teams.append(
                (game["awayTeam"]["teamId"], game["awayTeam"]["teamTricode"])
            )

        projections = []
        for team_id, tricode in teams:
            team_full_name = TRICODE_TO_NAME.get(tricode, "")
            game_total = vegas_lines.get(team_full_name)

            roster = fetch_team_roster(team_id)
            for player in roster:
                gamelog = fetch_player_gamelog(player["playerId"])
                proj = project_player(player["playerName"], gamelog, game_total)
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
