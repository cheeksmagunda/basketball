import json
import hashlib
import pickle
import numpy as np
import requests
from datetime import date, datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

DATA_DIR = Path("/tmp/nba_data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "lineup_history.json"
CACHE_DIR = Path("/tmp/nba_cache_v18")
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

def _cp(k): return CACHE_DIR / f"{hashlib.md5(f'{date.today().isoformat()}:{k}'.encode()).hexdigest()}.json"
def _cg(k): return json.loads(_cp(k).read_text()) if _cp(k).exists() else None
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
    data = _espn_get(f"{ESPN}/scoreboard")
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
        else:
            blended = dict(season)
            blended["season_min"] = season["min"]
            blended["recent_pts"] = season["pts"]
            blended["season_pts"] = season["pts"]
    except Exception as e:
        print(f"Stat parse error pid={pid}: {e}")
        return None
    _cs(f"ath3_{pid}", blended)
    return blended

# ─────────────────────────────────────────────────────────────────────────────
# THE CORE MODEL
#
# Real Sports Value = actual_score × slot_multiplier_received
# The slot multiplier is determined by ownership — high-owned players (stars)
# always land in low-multiplier slots. Bench players get 3x+ if smart players
# identify them early.
#
# We estimate ownership via minutes:
#   Stars (35+ min)     → everyone drafts them → low slot mult ~1.2x → low EV
#   Starters (28-35)    → popular → slot mult ~1.5x
#   Role players(22-28) → moderate ownership → slot mult ~2.0x
#   Bench (15-22)       → low ownership, high mult ~2.8x ← SWEET SPOT
#   Deep bench (<15)    → very low owned, mult ~3.5x IF they play
#
# CHALK = best EV at moderate risk (role players + starters in form)
#   Sort key: projected_score × moderate_ownership_mult × hot_streak
#   This surfaces Reed Sheppard (22 min, hot) over Jokic (35 min, cold)
#
# UPSIDE = best EV at high risk (deep bench + bench with hot streaks)
#   Sort key: projected_score × aggressive_ownership_mult × hot_streak^2
#   This surfaces Hugo González, Sharife Cooper, Filipowski
# ─────────────────────────────────────────────────────────────────────────────

def _ownership_mult_chalk(avg_min):
    """Moderate inverse-ownership mult. Penalizes stars, rewards role players."""
    if avg_min < 15:  return 1.8   # deep bench — too risky for chalk
    if avg_min < 22:  return 2.8   # bench sweet spot (Sheppard, González tier)
    if avg_min < 28:  return 2.2   # role players
    if avg_min < 33:  return 1.5   # starters
    return 0.9                      # stars — everyone drafts them, low mult

def _ownership_mult_upside(avg_min):
    """Aggressive inverse-ownership mult. Maximizes deep bench upside."""
    if avg_min < 15:  return 3.5   # deep bench lottery tickets
    if avg_min < 22:  return 3.0   # bench players
    if avg_min < 28:  return 2.0   # role players
    if avg_min < 33:  return 1.2   # starters
    return 0.6                      # stars — skip them for upside

def project_player(pinfo, stats, spread, total, side, team_abbr=""):
    if pinfo.get("is_out"): return None
    avg_min = stats.get("min", 0)
    if avg_min <= 0: return None

    pts = stats["pts"]
    reb = stats["reb"]
    ast = stats["ast"]
    if pts + reb + ast <= 0: return None

    # AI blend
    heuristic = pts + reb + ast
    base = heuristic
    if AI_MODEL is not None:
        try:
            usage = min(max(pts / max(avg_min, 1) * 0.8, 0.9), 1.5)
            features = np.array([[avg_min, stats.get("season_pts", pts), usage, 112.0]])
            ai_pred = AI_MODEL.predict(features)[0]
            ai_norm = ai_pred * (heuristic / max(ai_pred, 1))
            base = (ai_norm * 0.7) + (heuristic * 0.3)
        except: pass

    # Contextual multipliers
    pace_adj   = 1.0 + (0.03 * ((total or 222) - 222) / 20)
    spread_adj = 1.0 + (0.015 * (15 - abs(spread or 0)) / 15)
    home_adj   = 1.02 if side == "home" else 1.0

    # Raw projected ▼ score (what they'll actually score in Real Sports)
    raw_score = (base * pace_adj * spread_adj * home_adj) / 5.0

    # Hot streak: recent form vs season avg
    season_pts = stats.get("season_pts", pts)
    recent_pts = stats.get("recent_pts", pts)
    hot = round((recent_pts / season_pts) if season_pts > 0 else 1.0, 2)
    hot = min(hot, 2.5)  # cap at 2.5x to prevent outliers

    om_chalk  = _ownership_mult_chalk(avg_min)
    om_upside = _ownership_mult_upside(avg_min)

    # EV scores
    chalk_ev  = round(raw_score * om_chalk  * max(hot, 1.0), 2)
    upside_ev = round(raw_score * om_upside * max(hot, 1.0) * max(hot, 1.0), 2)  # hot^2 for upside

    return {
        "id":      pinfo["id"],
        "name":    pinfo["name"],
        "pos":     pinfo["pos"],
        "team":    team_abbr,
        "rating":  round(raw_score, 1),   # actual projected ▼ score shown on card
        "chalk_ev":chalk_ev,
        "upside_ev":upside_ev,
        "predMin": round(avg_min, 1),
        "pts":     round(pts, 1),
        "reb":     round(reb, 1),
        "ast":     round(ast, 1),
        "hot":     hot,
        "om":      om_chalk,
        "slot":    "1.0x",
        "_base":   base,
    }

def _run_game(game):
    home_r = fetch_roster(game["home"]["id"], game["home"]["abbr"])
    away_r = fetch_roster(game["away"]["id"], game["away"]["abbr"])
    players_in = (
        [(p, game["home"]["abbr"], "home") for p in home_r] +
        [(p, game["away"]["abbr"], "away") for p in away_r]
    )
    out = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_fetch_athlete, p["id"]): (p, ab, sd)
                for p, ab, sd in players_in}
        for fut in as_completed(futs):
            p, ab, sd = futs[fut]
            try:
                stats = fut.result()
                if stats:
                    proj = project_player(p, stats, game["spread"], game["total"], sd, ab)
                    if proj: out.append(proj)
            except Exception as e:
                print(f"proj err {p['name']}: {e}")
    return out

def _build_lineups(projections):
    # CHALK: sorted by chalk_ev (value-weighted, penalizes stars)
    chalk = sorted(projections, key=lambda x: x["chalk_ev"], reverse=True)[:5]
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
    return chalk, upside

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
        return {"games": [], "lineups": {"chalk": [], "upside": []}}
    cached = _cg("slate_v3")
    if cached: return cached
    all_proj = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed({pool.submit(_run_game, g): g for g in games}):
            try: all_proj.extend(fut.result())
            except Exception as e: print(f"slate err: {e}")
    chalk, upside = _build_lineups(all_proj)
    result = {"games": games, "lineups": {"chalk": chalk, "upside": upside}}
    _cs("slate_v3", result)
    return result

@app.get("/api/picks")
async def get_picks(gameId: str = Query(...)):
    game = next((g for g in fetch_games() if g["gameId"] == gameId), None)
    if not game:
        return JSONResponse({"error": "Game not found"}, status_code=404)
    projections = _run_game(game)
    if not projections:
        return JSONResponse({"error": "No projections available."}, status_code=503)
    chalk, upside = _build_lineups(projections)
    _save_history(game["label"], chalk)
    return {"game": game, "lineups": {"chalk": chalk, "upside": upside}}

@app.get("/api/history")
async def get_history():
    if not HISTORY_FILE.exists(): return []
    try: return json.loads(HISTORY_FILE.read_text())
    except: return []

@app.get("/api/evaluate")
async def evaluate():
    if not HISTORY_FILE.exists():
        return {"status": "ok", "updated": 0}
    try:
        hist = json.loads(HISTORY_FILE.read_text())
    except Exception:
        return {"status": "error", "message": "Could not read history"}

    # Fetch today's scoreboard for box score links
    sb = _espn_get(f"{ESPN}/scoreboard")
    events = sb.get("events", [])

    # Build a lookup: player name (lower) → actual DFS score
    actual_scores = {}
    for ev in events:
        event_id = ev.get("id")
        if not event_id:
            continue
        box_url = (f"https://site.api.espn.com/apis/site/v2/sports/"
                   f"basketball/nba/summary?event={event_id}")
        box_cache_key = f"box_{event_id}"
        box = _cg(box_cache_key)
        if not box:
            box = _espn_get(box_url)
            if box:
                _cs(box_cache_key, box)
        if not box:
            continue
        for bp in box.get("boxscore", {}).get("players", []):
            for stat_set in bp.get("statistics", []):
                labels = [l.lower() for l in stat_set.get("labels", [])]
                for athlete_row in stat_set.get("athletes", []):
                    name = athlete_row.get("athlete", {}).get("displayName", "")
                    if not name:
                        continue
                    stats_vals = athlete_row.get("stats", [])
                    s = {}
                    for lbl, val in zip(labels, stats_vals):
                        s[lbl] = _safe_float(val)
                    pts = s.get("pts", 0)
                    reb = s.get("reb", 0)
                    ast = s.get("ast", 0)
                    stl = s.get("stl", 0)
                    blk = s.get("blk", 0)
                    tov = s.get("to", s.get("tov", 0))
                    dfs = (pts + reb + (ast * 1.5)
                           + (stl * 3.5) + (blk * 3.0) - (tov * 1.2))
                    actual_scores[name.lower()] = round(dfs / 5.0, 1)

    updated = 0
    for entry in hist:
        for p in entry.get("players", []):
            if p.get("actual_score") is not None:
                continue
            key = p["name"].lower()
            if key in actual_scores:
                p["actual_score"] = actual_scores[key]
                updated += 1

    HISTORY_FILE.write_text(json.dumps(hist[-50:], indent=2))
    return {"status": "ok", "updated": updated}

@app.get("/api/refresh")
async def refresh():
    cleared = 0
    try:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink(); cleared += 1
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return {"status": "ok", "cleared": cleared, "ts": datetime.now().isoformat()}
