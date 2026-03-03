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

CACHE_DIR = Path("/tmp/nba_real_cache_v2")
CACHE_DIR.mkdir(exist_ok=True)

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
    """Fetch season stats for one player from ESPN overview endpoint."""
    c = _cg(f"ath_{pid}")
    if c is not None:
        return c

    # The /overview endpoint is the only one that reliably works
    url = f"https://site.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{pid}/overview"
    try:
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            stats = _parse_overview(r.json())
            if stats and stats["min"] > 0:
                _cs(f"ath_{pid}", stats)
                return stats
    except Exception:
        pass

    _cs(f"ath_{pid}", None)
    return None


# ESPN overview endpoint returns:
#   statistics.names = ["gamesPlayed", "avgMinutes", "fieldGoalPct", ..., "avgPoints"]
#   statistics.splits[0].stats = ["50", "34.2", "48.1", ..., "25.7"]  (Regular Season)
NAME_MAP = {
    "gamesplayed": "gp",
    "avgminutes": "min",
    "fieldgoalpct": "fgp",
    "threepointpct": None,      # not used directly
    "freethrowpct": None,
    "avgrebounds": "reb",
    "avgassists": "ast",
    "avgblocks": "blk",
    "avgsteals": "stl",
    "avgfouls": None,
    "avgturnovers": "tov",
    "avgpoints": "pts",
}


def _parse_overview(data):
    """Parse the ESPN /overview endpoint response for season averages."""
    result = {"min": 0, "pts": 0, "reb": 0, "ast": 0, "stl": 0, "blk": 0,
              "tov": 0, "fg3m": 0, "fgp": 0.44, "gp": 0}

    stats_block = data.get("statistics", {})
    names = stats_block.get("names", [])
    splits = stats_block.get("splits", [])

    # Use first split (Regular Season) if available
    if not names or not splits:
        return None

    values = splits[0].get("stats", [])
    if len(names) != len(values):
        return None

    for name, val in zip(names, values):
        key = name.lower()
        mapped = NAME_MAP.get(key)
        if not mapped:
            continue
        v = _safe_float(val)
        if mapped == "fgp" and v > 1:
            v /= 100
        if mapped == "gp":
            v = int(v)
        result[mapped] = v

    return result if result["min"] > 0 else None


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
# Real Rating projection model
# ---------------------------------------------------------------------------
# This model approximates Real Ratings, NOT DFS fantasy points.
# Key insight: Real Rating rewards IMPORTANT stats in CLOSE games,
# not raw volume in blowouts. A 20-point game in a 2-point thriller
# outranks a 35-point game in a 20-point blowout.
#
# Pipeline:
#   Stage 1: Predict minutes (closeness-aware)
#   Stage 2: Compute importance-weighted production rate per minute
#   Stage 3: Game closeness factor (the core differentiator vs DFS)
#   Stage 4: Clutch usage proxy + underdog boost + pace
#   Final:   Minutes × Rate × Closeness × Clutch × Underdog × Pace
# ---------------------------------------------------------------------------

# Importance-weighted stat values (NOT DFS scoring)
# Assists/steals/blocks weighted higher because they represent high-leverage
# plays that swing close games. Raw points weighted lower than DFS.
REAL_W = {
    "pts": 0.85,   # points: important but volume ≠ importance
    "reb": 1.1,    # rebounds: possession-winning, matters late
    "ast": 1.8,    # assists: playmaking drives winning basketball
    "stl": 3.0,    # steals: game-changing defensive plays
    "blk": 2.5,    # blocks: momentum-shifting, highlight plays
    "tov": -1.5,   # turnovers: devastating in close games
    "fg3m": 0.4,   # 3PM: slight bonus for momentum/crowd factor
}


def game_closeness_factor(spread):
    """Core Real Rating differentiator: close games produce higher ratings.

    The audit guide formula:
        Game_Closeness_Factor = 1.0 + 0.5 × (1 - |Final_Margin| / 30)
    We use spread as proxy for expected final margin (~1.3× spread).

    Returns:
        ~1.22  for pick'em (spread 0)
        ~1.15  for spread ±3
        ~1.05  for spread ±7
        ~0.93  for spread ±11
        ~0.82  for spread ±15+
    """
    if spread is None:
        return 1.0
    expected_margin = abs(spread) * 1.3
    return 1.0 + 0.45 * (1.0 - min(expected_margin / 30.0, 1.0))


def clutch_usage_proxy(stats, pos):
    """Estimate clutch involvement from position and production profile.

    Without actual clutch data, we approximate who gets the ball in crunch
    time (last 5 min, margin ≤ 5):
    - High PPG + AST guards = primary late-game ball handlers
    - High STL players = active defenders creating key turnovers
    - Guards/wings get position bonus (they run late-game sets)
    - Big men get clutch bonus only if they're primary scorers

    Returns multiplier 1.0 to ~1.18.
    """
    pts = stats.get("pts", 0)
    ast = stats.get("ast", 0)
    stl = stats.get("stl", 0)
    min_ = max(stats.get("min", 1), 1)

    # Per-minute ball dominance = proxy for late-game usage
    usage = (pts + 1.5 * ast + stl) / min_

    # Position-based clutch tendency
    if pos in ("PG", "SG", "G"):
        pos_mult = 1.06        # guards handle ball late
    elif pos in ("SF", "F"):
        pos_mult = 1.03        # wings are secondary options
    elif pos in ("C", "PF"):
        pos_mult = 0.97 if pts < 18 else 1.02  # bigs only if they're stars
    else:
        pos_mult = 1.0

    # Scale usage into 1.0 to ~1.18 range
    clutch = 1.0 + min(usage * 0.10, 0.12) * pos_mult
    return clutch


def underdog_boost(spread, side):
    """Slight underdogs generate more dramatic Real Rating swings.

    Being down forces higher-leverage plays. Comebacks are rewarded heavily
    in Real Rating. Sweet spot: team spread +1 to +6.

    ESPN spread convention: positive = home underdog.
    """
    if spread is None:
        return 1.0

    # Compute this player's team spread
    # spread > 0 = home is underdog, so away is favorite
    team_spread = spread if side == "home" else -spread

    if 1.0 <= team_spread <= 6.0:
        # Sweet spot underdog: peak boost at +3.5
        return 1.0 + (3.5 - abs(team_spread - 3.5)) * 0.014  # max ~1.049
    elif team_spread > 6.0:
        # Heavy underdog: blowout risk outweighs comeback potential
        return max(0.96 - (team_spread - 6) * 0.004, 0.92)
    elif team_spread < -5.0:
        # Heavy favorite: less drama, slight penalty
        return max(1.0 - (abs(team_spread) - 5) * 0.006, 0.94)
    else:
        return 1.0


def pace_factor(total):
    """Scoring environment adjustment.

    Optimal for Real Rating: O/U 215-230 (enough scoring, still competitive).
    Very high totals (240+) suggest pace-inflated, less meaningful stats.
    Very low totals (<205) mean fewer opportunities overall.
    Inverted-U centered around 222.
    """
    if total is None:
        return 1.0
    deviation = abs(total - 222) / 30.0
    return 1.0 + 0.06 * max(1.0 - deviation, 0)  # max ~1.06 at sweet spot


def real_rating_per_min(stats, pos):
    """Stage 2: Importance-weighted production rate per minute.

    Unlike DFS (all stats equal value), Real Rating emphasizes:
    - Assists/steals/blocks (high-leverage plays) weighted highest
    - Points weighted lower per unit (volume ≠ importance)
    - Turnovers penalized heavily (game-changing in crunch time)
    - FG% bonus: efficient scoring = fewer wasted possessions
    """
    min_ = stats.get("min", 1)
    if min_ <= 0:
        return 0

    base = sum(stats.get(k, 0) * w for k, w in REAL_W.items())

    # Efficiency: high FG% means fewer wasted possessions
    fgp = stats.get("fgp", 0.44)
    eff_bonus = (fgp - 0.44) * 8.0

    return (base + eff_bonus) / min_


def project_player(name, pos, age, side, stats, spread, total, game_label=""):
    """Full Real Rating projection.

    Pipeline:
        Stage 1: Minutes (closeness-aware — close games = more starter minutes)
        Stage 2: Rate per minute (importance-weighted, not DFS-weighted)
        Stage 3: Game closeness boost (THE differentiator)
        Stage 4: Clutch × Underdog × Pace × Home
        Final:   Minutes × Rate × Closeness × Clutch × Underdog × Pace × Home

    Key interaction: Minutes × Closeness — minutes matter MORE in close games.
    """
    avg_min = stats["min"]
    if avg_min < 15:
        return None

    # --- Stage 1: Minutes prediction (closeness-aware) ---
    pred_min = avg_min

    if spread is not None:
        abs_spread = abs(spread)
        if abs_spread <= 4:
            # Close game: starters play full 4th quarter (+1-3% minutes)
            pred_min *= 1.0 + (4 - abs_spread) * 0.007
        elif abs_spread > 7:
            # Blowout risk: stars sit late in 4th
            pred_min *= max(1.0 - (abs_spread - 7) * 0.020, 0.80)

    # Age fatigue
    if age and age > 35:
        pred_min *= 0.94
    elif age and age > 32:
        pred_min *= 0.97

    # --- Stage 2: Importance-weighted rate per minute ---
    rate = real_rating_per_min(stats, pos)

    # --- Stage 3: Game closeness (the core differentiator) ---
    closeness = game_closeness_factor(spread)

    # --- Stage 4: Contextual multipliers ---
    clutch = clutch_usage_proxy(stats, pos)
    dog = underdog_boost(spread, side)
    pace = pace_factor(total)
    home_adj = 1.015 if side == "home" else 0.985

    # --- Final: Minutes × Rate × Closeness × Clutch × Underdog × Pace × Home ---
    proj_rating = pred_min * rate * closeness * clutch * dog * pace * home_adj

    # --- Tier classification ---
    tier = "star" if pred_min >= 33 else ("starter" if pred_min >= 24 else "role")

    # --- Variance (ranking confidence) ---
    # Stars are more predictable; role players are volatile.
    # Close games reduce variance for stars (reliable clutch usage).
    base_var = 0.12 if tier == "star" else (0.18 if tier == "starter" else 0.28)
    if spread is not None and abs(spread) <= 4:
        base_var *= 0.85  # close games = more predictable star usage

    return {
        "name": name, "pos": pos, "tier": tier,
        "rating": round(proj_rating, 1),
        "predMin": round(pred_min, 1),
        "rpm": round(rate, 2),
        "vegasAdj": round(pace, 3),
        "blowoutAdj": round(closeness, 3),
        "homeAdj": round(home_adj, 3),
        "stdDev": round(proj_rating * base_var, 1),
        "side": side,
        "game": game_label,
        # Internal fields for lineup construction (not rendered by frontend)
        "_closeness": closeness,
        "_clutch": clutch,
        "_underdog": dog,
    }


def pairwise_conf(a, b):
    """P(a ranks above b) using normal CDF approximation."""
    diff = a["rating"] - b["rating"]
    comb = math.sqrt(a["stdDev"] ** 2 + b["stdDev"] ** 2)
    if comb == 0:
        return 0.5
    return round(0.5 * (1 + math.erf(diff / (comb * math.sqrt(2)))), 3)


def build_lineups(projections):
    """Build 3 lineup variants optimized for Real Rating ranking accuracy.

    Chalk:   Top 5 by projected Real Rating.
    Diff:    Favor close-game players when ratings are tight.
             Re-sorts top pool with extra closeness weight.
    Contra:  Maximize game-importance exposure. Heavy closeness +
             underdog weight to find "importance over volume" plays.
    """
    if len(projections) < 5:
        return [], [], []

    ranked = sorted(projections, key=lambda x: x["rating"], reverse=True)
    pool = ranked[:12]  # wider pool for diff/contra construction

    # --- Chalk: straightforward top 5 ---
    chalk = [dict(p) for p in ranked[:5]]
    _annotate(chalk)

    # --- Differentiated: re-rank with extra closeness weight ---
    # When two players are similar in rating, prefer the one in a closer game.
    # closeness^0.4 gives moderate extra weight to close-game players.
    diff_sorted = sorted(pool, key=lambda x: (
        x["rating"] * (x.get("_closeness", 1.0) ** 0.4)
    ), reverse=True)
    diff = [dict(p) for p in diff_sorted[:5]]
    _annotate(diff)

    # --- Contrarian: maximize game importance exposure ---
    # Heavy closeness + underdog weighting. Finds players whose importance
    # context is undervalued by volume-based models.
    contra_sorted = sorted(pool, key=lambda x: (
        x["rating"]
        * (x.get("_closeness", 1.0) ** 1.0)
        * (x.get("_underdog", 1.0) ** 1.5)
    ), reverse=True)
    contra = [dict(p) for p in contra_sorted[:5]]
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
