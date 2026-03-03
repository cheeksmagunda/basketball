import json
import hashlib
import pickle
import time
import numpy as np
import requests
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

DATA_DIR = Path("/tmp/nba_data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "lineup_history.json"
CACHE_DIR = Path("/tmp/nba_cache_v19")
CACHE_DIR.mkdir(exist_ok=True)

AI_MODEL = None
for _p in [
    Path(__file__).parent.parent / "lgbm_model.pkl",
    Path(__file__).parent / "lgbm_model.pkl",
    Path("lgbm_model.pkl"),
]:
    if _p.exists():
        try:
            with open(_p, "rb") as f:
                AI_MODEL = pickle.load(f)
            break
        except: pass

ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SLOT_VALUES = ["2.0x", "1.8x", "1.6x", "1.4x", "1.2x"]
MIN_GATE = 15  # Minimum projected minutes to qualify
CACHE_TTL = 1800  # 30 minutes in seconds
ET = ZoneInfo("America/New_York")

def _today():
    """NBA date in Eastern Time (YYYYMMDD). ESPN and NBA operate on ET."""
    return datetime.now(ET).strftime("%Y%m%d")

def _cp(k): return CACHE_DIR / f"{hashlib.md5(f'{_today()}:{k}'.encode()).hexdigest()}.json"
def _cg(k):
    p = _cp(k)
    if not p.exists(): return None
    if time.time() - p.stat().st_mtime > CACHE_TTL: return None  # expired
    return json.loads(p.read_text())
def _cs(k, v): _cp(k).write_text(json.dumps(v))
def _safe_float(v, default=0.0):
    try: return float(v) if v is not None else default
    except: return default

def _espn_get(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except: return {}

def fetch_games():
    c = _cg("games")
    if c: return c
    data = _espn_get(f"{ESPN}/scoreboard?dates={_today()}")
    games = []
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        home = away = None
        for cd in comp.get("competitors", []):
            t = {"id": cd["team"]["id"], "name": cd["team"]["displayName"],
                 "abbr": cd["team"].get("abbreviation", "")}
            if cd["homeAway"] == "home": home = t
            else: away = t
        if not home or not away: continue
        odds = comp.get("odds", [{}])[0]
        games.append({
            "gameId": ev["id"], "label": f"{away['abbr']} @ {home['abbr']}",
            "home": home, "away": away,
            "spread": _safe_float(odds.get("spread"), None),
            "total":  _safe_float(odds.get("overUnder"), None),
        })
    _cs("games", games)
    return games

def fetch_roster(team_id, team_abbr):
    c = _cg(f"roster_{team_id}")
    if c: return c
    data = _espn_get(f"{ESPN}/teams/{team_id}/roster")
    players = []
    for a in data.get("athletes", []):
        inj = a.get("injuries", [])
        is_out = inj[0].get("status", "").lower() in ["out", "injured"] if inj else False
        players.append({
            "id": a["id"], "name": a["fullName"],
            "pos": a.get("position", {}).get("abbreviation", "G"),
            "is_out": is_out, "team_abbr": team_abbr,
        })
    _cs(f"roster_{team_id}", players)
    return players

def _parse_split(names, split):
    s = {"min": 0.0, "pts": 0.0, "reb": 0.0, "ast": 0.0, "stl": 0.0, "blk": 0.0, "tov": 0.0}
    for name, val in zip(names, split.get("stats", [])):
        k = name.lower()
        if   "min" in k:                   s["min"] = _safe_float(val)
        elif "pts" in k or "point" in k:   s["pts"] = _safe_float(val)
        elif "reb" in k or "rebound" in k: s["reb"] = _safe_float(val)
        elif "ast" in k or "assist" in k:  s["ast"] = _safe_float(val)
        elif "stl" in k or "steal" in k:   s["stl"] = _safe_float(val)
        elif "blk" in k or "block" in k:   s["blk"] = _safe_float(val)
        elif "tov" in k or "turnover" in k:s["tov"] = _safe_float(val)
    return s

def _fetch_athlete(pid):
    c = _cg(f"ath3_{pid}")
    if c: return c
    url = (f"https://site.api.espn.com/apis/common/v3/sports/basketball"
           f"/nba/athletes/{pid}/overview")
    data = _espn_get(url)
    if not data: return None
    try:
        stat_obj = data.get("statistics", {})
        names    = stat_obj.get("names", [])
        splits   = stat_obj.get("splits", [])
        if not names or not splits: return None
        season = _parse_split(names, splits[0])
        if season["min"] <= 0: return None
        recent = None
        for split in splits[1:]:
            label = (str(split.get("displayName","")) + str(split.get("type",""))).lower()
            if any(kw in label for kw in ["last","recent","l5","l10","l3"]):
                c2 = _parse_split(names, split)
                if c2["min"] >= 10:
                    recent = c2
                    break
        if recent is None and len(splits) > 1:
            c2 = _parse_split(names, splits[1])
            if 10 <= c2["min"] <= 48 and c2["pts"] > 0:
                recent = c2
        if recent:
            blended = {k: round(season[k] * 0.6 + recent[k] * 0.4, 2) for k in season}
            blended["season_min"] = season["min"]
            blended["recent_pts"] = recent["pts"]
            blended["season_pts"] = season["pts"]
            blended["recent_min"] = recent["min"]
            blended["recent_stl"] = recent["stl"]
            blended["recent_blk"] = recent["blk"]
        else:
            blended = dict(season)
            blended["season_min"] = season["min"]
            blended["recent_min"] = season["min"]
            blended["recent_pts"] = season["pts"]
            blended["season_pts"] = season["pts"]
            blended["recent_stl"] = season["stl"]
            blended["recent_blk"] = season["blk"]
    except Exception as e:
        print(f"Stat parse error pid={pid}: {e}")
        return None
    _cs(f"ath3_{pid}", blended)
    return blended

# ─────────────────────────────────────────────────────────────────────────────
# INJURY CASCADE ENGINE
#
# When a player is OUT, their avg minutes get redistributed to remaining
# teammates at the same position (or adjacent positions).
# This is what found González (18→26 min) and Cooper (13→17 min) on March 2.
#
# Position adjacency: G↔G, F↔F, C↔F (centers share with forwards)
# ─────────────────────────────────────────────────────────────────────────────

POS_GROUPS = {
    "PG": "G", "SG": "G", "G": "G",
    "SF": "F", "PF": "F", "F": "F",
    "C": "C",
}

def _pos_group(pos):
    return POS_GROUPS.get(pos, "G")

def _cascade_minutes(roster, stats_map):
    """Redistribute minutes from OUT players to eligible teammates."""
    cascade_flags = {}

    # Group players by team
    teams = {}
    for p in roster:
        team = p.get("team_abbr", "")
        if team not in teams:
            teams[team] = []
        teams[team].append(p)

    for team, team_players in teams.items():
        # Find OUT players with known minutes
        out_players = []
        active_players = []
        for p in team_players:
            pid = p["id"]
            s = stats_map.get(pid)
            if p.get("is_out") and s and s.get("min", 0) > 0:
                out_players.append((p, s))
            elif not p.get("is_out") and s and s.get("min", 0) > 0:
                active_players.append((p, s))

        if not out_players or not active_players:
            continue

        # Calculate total minutes freed per position group
        freed_by_group = {}
        for op, os in out_players:
            pg = _pos_group(op["pos"])
            freed_by_group[pg] = freed_by_group.get(pg, 0) + os.get("min", 0)
            # Centers also share with forwards
            if pg == "C":
                freed_by_group["F"] = freed_by_group.get("F", 0) + os.get("min", 0) * 0.3

        # Distribute freed minutes to active players in same position group
        for group, freed_min in freed_by_group.items():
            # Find eligible recipients: same group, sorted by current minutes (lowest first = biggest benefit)
            recipients = []
            for ap, astat in active_players:
                apg = _pos_group(ap["pos"])
                # G receives G minutes, F receives F minutes, C receives both C and F
                if apg == group or (apg == "C" and group == "F") or (apg == "F" and group == "C"):
                    recipients.append((ap, astat))

            if not recipients:
                continue

            # Sort by minutes ascending — bench players get proportionally more
            recipients.sort(key=lambda x: x[1].get("min", 0))

            # Weight distribution: lower-minute players get more of the freed minutes
            total_weight = sum(1.0 / max(r[1].get("min", 1), 1) for r in recipients)
            for rp, rs in recipients:
                weight = (1.0 / max(rs.get("min", 1), 1)) / total_weight
                bonus = freed_min * weight * 0.7  # 70% of freed minutes get redistributed
                pid = rp["id"]
                if pid not in cascade_flags:
                    cascade_flags[pid] = 0.0
                cascade_flags[pid] += bonus

    # Cap: no player gains more than 10 minutes from cascade
    for pid in cascade_flags:
        cascade_flags[pid] = min(cascade_flags[pid], 10.0)

    return cascade_flags


# ─────────────────────────────────────────────────────────────────────────────
# THE CORE MODEL
#
# DFS Scoring Formula: PTS + REB + AST×1.5 + STL×3.5 + BLK×3.0 - TOV×1.2
#
# Real Sports Value = actual_score × slot_multiplier_received
# The slot multiplier is determined by ownership — high-owned players (stars)
# always land in low-multiplier slots.
#
# We estimate ownership via projected minutes (including cascade):
#   Stars (33+ min)     → everyone drafts them → low slot mult ~0.9x → AVOID
#   Starters (28-33)    → popular → slot mult ~1.5x
#   Role players(22-28) → moderate ownership → slot mult ~2.2x
#   Bench (15-22)       → low ownership, high mult ~2.8x ← SWEET SPOT
#   Deep bench (<15)    → below minutes gate, filtered out
#
# CHALK = best EV at moderate risk (role players + starters in form)
# UPSIDE = best EV at high risk (deep bench + bench with hot streaks)
# ─────────────────────────────────────────────────────────────────────────────

def _lerp_mult(minutes, points):
    """Linearly interpolate ownership mult from control points — no cliffs."""
    if minutes <= points[0][0]: return points[0][1]
    if minutes >= points[-1][0]: return points[-1][1]
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        if x0 <= minutes <= x1:
            t = (minutes - x0) / (x1 - x0)
            return round(y0 + t * (y1 - y0), 3)
    return points[-1][1]

# Smooth ownership curves: (minutes, multiplier) control points
# Chalk: peaks at role players (22 min), gently slopes down for stars
CHALK_CURVE = [(15, 2.0), (22, 2.2), (28, 1.5), (33, 1.1), (38, 0.8)]
# Upside: aggressively rewards low-minute players, steeper star penalty
UPSIDE_CURVE = [(15, 3.0), (22, 2.8), (28, 1.8), (33, 0.9), (38, 0.5)]

def _ownership_mult_chalk(avg_min):
    """Smooth inverse-ownership mult for Starting 5 (safe) mode."""
    return _lerp_mult(avg_min, CHALK_CURVE)

def _ownership_mult_upside(avg_min):
    """Smooth aggressive inverse-ownership mult for Rotation (upside) mode."""
    return _lerp_mult(avg_min, UPSIDE_CURVE)

def _dfs_score(pts, reb, ast, stl, blk, tov):
    """Full DFS scoring formula — matches the leaderboard exactly."""
    return pts + reb + (ast * 1.5) + (stl * 3.5) + (blk * 3.0) - (tov * 1.2)

def project_player(pinfo, stats, spread, total, side, team_abbr="",
                   cascade_bonus=0.0):
    if pinfo.get("is_out"): return None
    avg_min = stats.get("min", 0)
    if avg_min <= 0: return None

    # Apply cascade minute boost
    proj_min = avg_min + cascade_bonus
    is_cascade = cascade_bonus >= 2.0  # Flag if cascade added 2+ minutes

    # Minutes gate: must project to at least 15 minutes
    if proj_min < MIN_GATE: return None

    pts = stats["pts"]
    reb = stats["reb"]
    ast = stats["ast"]
    stl = stats.get("stl", 0)
    blk = stats.get("blk", 0)
    tov = stats.get("tov", 0)
    if pts + reb + ast <= 0: return None

    # Full DFS scoring formula (not just pts+reb+ast)
    heuristic = _dfs_score(pts, reb, ast, stl, blk, tov)

    # Declining usage penalty: if recent minutes dropped >15% vs season,
    # scale down production (fewer minutes = fewer counting stats)
    recent_min = stats.get("recent_min", avg_min)
    season_min = stats.get("season_min", avg_min)
    if recent_min < season_min * 0.85 and season_min > 0:
        decline_factor = recent_min / season_min
        heuristic *= decline_factor

    # Scale heuristic by minute boost from cascade
    if cascade_bonus > 0 and avg_min > 0:
        min_scale = min(proj_min / avg_min, 1.4)  # cap at 40% boost
        heuristic *= min_scale

    # AI blend
    base = heuristic
    if AI_MODEL is not None:
        try:
            usage = min(max(pts / max(avg_min, 1) * 0.8, 0.9), 1.5)
            features = np.array([[avg_min, stats.get("season_pts", pts), usage, 112.0]])
            ai_pred = AI_MODEL.predict(features)[0]
            ai_norm = ai_pred * (heuristic / max(ai_pred, 1))
            base = (ai_norm * 0.7) + (heuristic * 0.3)
        except: pass

    # Contextual multipliers (strengthened pace adjustment)
    pace_adj   = 1.0 + (0.06 * ((total or 222) - 222) / 20)   # doubled from 0.03
    spread_adj = 1.0 + (0.015 * (15 - abs(spread or 0)) / 15)
    home_adj   = 1.02 if side == "home" else 1.0

    # Raw projected score (what they'll actually score in Real Sports)
    raw_score = (base * pace_adj * spread_adj * home_adj) / 5.0

    # Use the LOWER of blended and recent minutes for ownership tiers
    # If recent minutes dropped (trade, role change), ownership reflects current role
    recent_min = stats.get("recent_min", avg_min)
    ownership_min = min(avg_min, recent_min)
    om_chalk  = _ownership_mult_chalk(ownership_min)
    om_upside = _ownership_mult_upside(ownership_min)

    # EV scores (recent form is already captured in the 60/40 blended stats)
    chalk_ev  = round(raw_score * om_chalk, 2)
    upside_ev = round(raw_score * om_upside, 2)

    # Expected draft points (EDP) = raw_score / 5 * est_mult
    expected_dp = round(raw_score * om_chalk, 1)

    return {
        "id":      pinfo["id"],
        "name":    pinfo["name"],
        "pos":     pinfo["pos"],
        "team":    team_abbr,
        "rating":  round(raw_score, 1),
        "chalk_ev":chalk_ev,
        "upside_ev":upside_ev,
        "expected_dp": expected_dp,
        "predMin": round(proj_min, 1),
        "pts":     round(pts, 1),
        "reb":     round(reb, 1),
        "ast":     round(ast, 1),
        "stl":     round(stl, 1),
        "blk":     round(blk, 1),
        "est_mult": om_chalk,
        "om":      om_chalk,
        "slot":    "1.0x",
        "_base":   base,
        "is_cascade_pick": is_cascade,
    }

def _run_game(game):
    home_r = fetch_roster(game["home"]["id"], game["home"]["abbr"])
    away_r = fetch_roster(game["away"]["id"], game["away"]["abbr"])

    all_roster = home_r + away_r
    players_in = (
        [(p, game["home"]["abbr"], "home") for p in home_r] +
        [(p, game["away"]["abbr"], "away") for p in away_r]
    )

    # Fetch all athlete stats first
    stats_map = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_athlete, p["id"]): p for p, _, _ in players_in}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                stats = fut.result()
                if stats:
                    stats_map[p["id"]] = stats
            except Exception as e:
                print(f"fetch err {p['name']}: {e}")

    # Run cascade engine to redistribute minutes from OUT players
    cascade_flags = _cascade_minutes(all_roster, stats_map)

    # Project all players with cascade-adjusted minutes
    out = []
    for p, ab, sd in players_in:
        stats = stats_map.get(p["id"])
        if not stats:
            continue
        cascade_bonus = cascade_flags.get(p["id"], 0.0)
        proj = project_player(p, stats, game["spread"], game["total"], sd, ab,
                              cascade_bonus=cascade_bonus)
        if proj:
            out.append(proj)
    return out

def _classify_tiers(projections):
    """Classify all projections into TOP / ACCEPTABLE / AVOID tiers."""
    top = []
    acceptable = []
    avoid = []

    for p in projections:
        proj_min = p.get("predMin", 0)
        edp = p.get("expected_dp", 0)
        is_cascade = p.get("is_cascade_pick", False)

        if proj_min >= 33:
            # Stars — always AVOID regardless of raw score
            avoid.append(p)
        elif edp >= 15 or (is_cascade and edp >= 12):
            # High EDP or cascade-boosted players
            top.append(p)
        elif edp >= 8:
            acceptable.append(p)
        else:
            avoid.append(p)

    # Sort each tier by expected_dp descending
    top.sort(key=lambda x: x.get("expected_dp", 0), reverse=True)
    acceptable.sort(key=lambda x: x.get("expected_dp", 0), reverse=True)
    avoid.sort(key=lambda x: x.get("expected_dp", 0), reverse=True)

    return {"top_picks": top[:8], "acceptable_fills": acceptable[:8], "avoid": avoid[:5]}

def _build_lineups(projections):
    # CHALK: production floor (rating >= 4.0) + sorted by chalk_ev
    chalk_eligible = [p for p in projections if p["rating"] >= 4.0]
    if len(chalk_eligible) < 5:
        # Fallback: fill remaining from all projections
        chalk_eligible = sorted(projections, key=lambda x: x["chalk_ev"], reverse=True)
    chalk = sorted(chalk_eligible, key=lambda x: x["chalk_ev"], reverse=True)[:5]
    for i, p in enumerate(chalk): p["slot"] = SLOT_VALUES[i]

    # UPSIDE: sorted by upside_ev (aggressively rewards bench + hot streaks)
    upside_sorted = sorted(projections, key=lambda x: x["upside_ev"], reverse=True)
    chalk_names = {p["name"] for p in chalk}

    upside = []
    for p in upside_sorted:
        if len(upside) >= 5: break
        upside.append(dict(p))

    # Force at least 2 different names vs chalk
    if sum(1 for p in upside if p["name"] not in chalk_names) < 2:
        upside = [dict(p) for p in upside_sorted[:5]]

    for i, p in enumerate(upside): p["slot"] = SLOT_VALUES[i]

    # Build tier classification
    tiers = _classify_tiers(projections)

    return chalk, upside, tiers

def _save_history(game_label, players):
    hist = []
    if HISTORY_FILE.exists():
        try: hist = json.loads(HISTORY_FILE.read_text())
        except: pass
    hist.append({
        "game": game_label,
        "timestamp": datetime.now().isoformat(),
        "players": [{"name": p["name"], "rating": p["rating"],
                     "team": p["team"], "pos": p["pos"], "actual_score": None}
                    for p in players]
    })
    HISTORY_FILE.write_text(json.dumps(hist[-50:], indent=2))

@app.get("/api/games")
async def get_games():
    return fetch_games()

@app.get("/api/slate")
async def get_slate():
    games = fetch_games()
    if not games:
        return {"games": [], "lineups": {"chalk": [], "upside": []}, "tiers": None}
    cached = _cg("slate_v4")
    if cached: return cached
    all_proj = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed({pool.submit(_run_game, g): g for g in games}):
            try: all_proj.extend(fut.result())
            except Exception as e: print(f"slate err: {e}")
    chalk, upside, tiers = _build_lineups(all_proj)
    result = {"games": games, "lineups": {"chalk": chalk, "upside": upside}, "tiers": tiers}
    _cs("slate_v4", result)
    return result

@app.get("/api/picks")
async def get_picks(gameId: str = Query(...)):
    game = next((g for g in fetch_games() if g["gameId"] == gameId), None)
    if not game:
        return JSONResponse({"error": "Game not found"}, status_code=404)
    projections = _run_game(game)
    if not projections:
        return JSONResponse({"error": "No projections available."}, status_code=503)
    chalk, upside, tiers = _build_lineups(projections)
    _save_history(game["label"], chalk)
    return {"game": game, "lineups": {"chalk": chalk, "upside": upside}, "tiers": tiers}

@app.get("/api/history")
async def get_history():
    if not HISTORY_FILE.exists(): return []
    try: return json.loads(HISTORY_FILE.read_text())
    except: return []

@app.get("/api/evaluate")
async def evaluate():
    return {"status": "success"}

@app.get("/api/refresh")
async def refresh():
    cleared = 0
    try:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink(); cleared += 1
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return {"status": "ok", "cleared": cleared, "ts": datetime.now().isoformat()}
