import os
import json
import math
import hashlib
import traceback
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

import requests
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

CACHE_DIR = Path("/tmp/nba_real_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Real App rating weights (NOT DraftKings/FanDuel scoring)
REAL_W = {"pts": 1.0, "reb": 1.2, "ast": 1.5, "stl": 2.0, "blk": 2.0, "tov": -0.5, "fg3m": 0.5}

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def _cp(k):
    return CACHE_DIR / f"{hashlib.md5(f'{date.today().isoformat()}:{k}'.encode()).hexdigest()}.json"

def _cg(k):
    p = _cp(k)
    return json.loads(p.read_text()) if p.exists() else None

def _cs(k, v):
    _cp(k).write_text(json.dumps(v))

# ---------------------------------------------------------------------------
# ESPN helpers
# ---------------------------------------------------------------------------
def _espn(path, timeout=15):
    """Fetch from ESPN API with retry."""
    url = f"{ESPN}{path}"
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 2:
                raise
    return {}


def _safe_float(v, default=0.0):
    try:
        return float(v) if v is not None else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_games():
    """Today's NBA games from ESPN scoreboard (includes odds)."""
    c = _cg("games")
    if c is not None:
        return c

    data = _espn("/scoreboard")
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

    data = _espn(f"/teams/{team_id}/roster")
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


def _fetch_athlete(pid):
    """Fetch season stats for one player from ESPN."""
    c = _cg(f"ath_{pid}")
    if c is not None:
        return c

    urls = [
        f"https://site.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{pid}",
        f"https://site.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{pid}/overview",
        f"{ESPN}/athletes/{pid}/statistics",
        f"{ESPN}/athletes/{pid}",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=12)
            if r.status_code != 200:
                continue
            stats = _parse_stats(r.json())
            if stats and stats["min"] > 0:
                _cs(f"ath_{pid}", stats)
                return stats
        except Exception:
            continue

    # Cache misses too so we don't retry every time
    _cs(f"ath_{pid}", None)
    return None


def _listify(x):
    if isinstance(x, list):
        return x
    if isinstance(x, dict):
        return [x]
    return []


STAT_MAP = {
    "avgminutes": "min", "minutes": "min", "min": "min", "minutespergame": "min",
    "avgpoints": "pts", "points": "pts", "pts": "pts", "pointspergame": "pts",
    "avgtotalrebounds": "reb", "totalrebounds": "reb", "rebounds": "reb", "reb": "reb",
    "avgrebounds": "reb",
    "avgassists": "ast", "assists": "ast", "ast": "ast",
    "avgsteals": "stl", "steals": "stl", "stl": "stl",
    "avgblocks": "blk", "blocks": "blk", "blk": "blk",
    "avgturnovers": "tov", "turnovers": "tov", "tov": "tov", "to": "tov",
    "avgthreepointersmade": "fg3m", "threepointersmade": "fg3m",
    "threepointfieldgoalsmade": "fg3m", "fg3m": "fg3m", "3pm": "fg3m",
    "fieldgoalpct": "fgp", "fieldgoalpercentage": "fgp", "fg%": "fgp", "fgpct": "fgp",
    "gamesplayed": "gp", "gp": "gp",
}


def _parse_stats(data):
    """Parse ESPN athlete response for season averages. Handles multiple formats."""
    result = {"min": 0, "pts": 0, "reb": 0, "ast": 0, "stl": 0, "blk": 0,
              "tov": 0, "fg3m": 0, "fgp": 0.44, "gp": 0}

    athlete = data.get("athlete", data)

    # --- Format A: labels[] + values/displayValues[] ---
    for block in _listify(athlete.get("statsSummary", athlete.get("statistics", []))):
        labels = block.get("labels", block.get("names", []))
        values = block.get("displayValues", block.get("values", []))
        if labels and values and len(labels) == len(values):
            for lbl, val in zip(labels, values):
                _assign_stat(result, lbl, val)
            if result["min"] > 0:
                return result

    # --- Format B: splits.categories[].stats[] ---
    stat_items = []
    sources = [
        athlete.get("statistics", []),
        data.get("statistics", []),
        data.get("stats", []),
    ]
    for src in sources:
        for block in _listify(src):
            for cat in _listify(block.get("splits", {}).get("categories", [])):
                stat_items.extend(_listify(cat.get("stats", [])))
            for cat in _listify(block.get("categories", [])):
                stat_items.extend(_listify(cat.get("stats", [])))
            stat_items.extend(_listify(block.get("stats", [])))

    for s in stat_items:
        if not isinstance(s, dict):
            continue
        name = s.get("name", s.get("abbreviation", ""))
        val = s.get("perGame", s.get("averageValue", s.get("avg", s.get("value", 0))))
        _assign_stat(result, name, val)

    return result if result["min"] > 0 else None


def _assign_stat(result, label, val):
    key = str(label).lower().replace(" ", "").replace("pergame", "").replace("/game", "")
    mapped = STAT_MAP.get(key)
    if not mapped:
        for prefix, field in STAT_MAP.items():
            if key.startswith(prefix):
                mapped = field
                break
    if mapped:
        v = _safe_float(val)
        if mapped == "fgp" and v > 1:
            v /= 100
        if mapped == "gp":
            v = int(v)
        result[mapped] = v


def fetch_game_players(home_id, away_id):
    """Fetch rosters + stats for both teams. Uses parallel HTTP."""
    home_roster = fetch_roster(home_id)
    away_roster = fetch_roster(away_id)

    players = [(p, "home") for p in home_roster] + [(p, "away") for p in away_roster]
    pids = list({p["id"] for p, _ in players})

    stats = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futs = {ex.submit(_fetch_athlete, pid): pid for pid in pids}
        for f in as_completed(futs):
            pid = futs[f]
            try:
                r = f.result()
                if r:
                    stats[pid] = r
            except Exception:
                pass

    return players, stats


# ---------------------------------------------------------------------------
# Projection & ranking model
# ---------------------------------------------------------------------------
def real_rating(s):
    """Per-game Real App rating from season averages."""
    base = sum(s.get(k, 0) * w for k, w in REAL_W.items())
    eff = (s.get("fgp", 0.44) - 0.44) * 5.0
    return base + eff


def project_player(name, pos, age, side, stats, spread, total, game_label=""):
    avg_min = stats["min"]
    if avg_min < 15:
        return None

    pred_min = avg_min

    # Blowout risk: |spread| > 7 reduces star minutes 1.5%/pt above 7
    blowout = 1.0
    if spread is not None and abs(spread) > 7:
        blowout = max(1.0 - (abs(spread) - 7) * 0.015, 0.85)

    # Age fatigue
    age_adj = 0.94 if age and age > 35 else (0.97 if age and age > 32 else 1.0)

    # Home/away
    home_adj = 1.015 if side == "home" else 0.985

    pred_min *= blowout * age_adj

    # Per-minute rating
    rpg = real_rating(stats)
    rpm = rpg / avg_min if avg_min > 0 else 0

    # Vegas pace/total adjustment
    vegas = (total / 220.0) if total else 1.0

    proj_rating = pred_min * rpm * vegas * home_adj

    tier = "star" if pred_min >= 34 else ("starter" if pred_min >= 25 else "role")
    var = 0.15 if tier == "star" else (0.20 if tier == "starter" else 0.30)

    return {
        "name": name, "pos": pos, "tier": tier,
        "rating": round(proj_rating, 1),
        "predMin": round(pred_min, 1),
        "rpm": round(rpm, 2),
        "vegasAdj": round(vegas, 3),
        "blowoutAdj": round(blowout, 3),
        "homeAdj": round(home_adj, 3),
        "stdDev": round(proj_rating * var, 1),
        "side": side,
        "game": game_label,
    }


def pairwise_conf(a, b):
    """P(a ranks above b)."""
    diff = a["rating"] - b["rating"]
    comb = math.sqrt(a["stdDev"] ** 2 + b["stdDev"] ** 2)
    if comb == 0:
        return 0.5
    return round(0.5 * (1 + math.erf(diff / (comb * math.sqrt(2)))), 3)


def build_lineups(projections):
    """Build 3 lineup variants from projections."""
    ranked = sorted(projections, key=lambda x: x["rating"], reverse=True)[:8]
    if len(ranked) < 5:
        return [], [], []

    chalk = [dict(p) for p in ranked[:5]]
    _annotate(chalk)

    diff = [dict(p) for p in ranked[:5]]
    min_conf_idx = min(range(4), key=lambda i: chalk[i].get("confidence", 1))
    if chalk[min_conf_idx].get("confidence", 1) < 0.65:
        diff[min_conf_idx], diff[min_conf_idx + 1] = diff[min_conf_idx + 1], diff[min_conf_idx]
    _annotate(diff)

    contra = [dict(p) for p in ranked[:5]]
    if len(ranked) > 5:
        contra[4] = dict(ranked[5])
    _annotate(contra)

    return chalk, diff, contra


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

        players, stats = fetch_game_players(game["home"]["id"], game["away"]["id"])

        projections = []
        for pinfo, side in players:
            s = stats.get(pinfo["id"])
            if not s:
                continue
            p = project_player(
                pinfo["name"], pinfo["pos"], pinfo.get("age", 25),
                side, s, spread, total, game["label"]
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

        # Fetch all games in parallel
        def process_game(game):
            spread = game.get("spread")
            total = game.get("total")
            players, stats = fetch_game_players(game["home"]["id"], game["away"]["id"])
            projs = []
            for pinfo, side in players:
                s = stats.get(pinfo["id"])
                if not s:
                    continue
                p = project_player(
                    pinfo["name"], pinfo["pos"], pinfo.get("age", 25),
                    side, s, spread, total, game["label"]
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
