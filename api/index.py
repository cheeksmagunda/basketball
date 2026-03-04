import json
import copy
import math
import hashlib
import pickle
import numpy as np
import requests
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Query, Body
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

ET = timezone(timedelta(hours=-5))
def _today_et():
    """Current date in Eastern Time — NBA schedule runs on ET, not UTC."""
    return datetime.now(ET).date()

DATA_DIR = Path("/tmp/nba_data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "lineup_history.json"
CACHE_DIR = Path("/tmp/nba_cache_v24")
CACHE_DIR.mkdir(exist_ok=True)
LOCK_DIR = Path("/tmp/nba_locks_v1")
LOCK_DIR.mkdir(exist_ok=True)
LOCK_BUFFER_MINUTES = 5

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

# Players to permanently exclude (retired, washed, or known bad data)
EXCLUDED_PLAYERS = {
    "mike conley",
}

def _cp(k): return CACHE_DIR / f"{hashlib.md5(f'{_today_et().isoformat()}:{k}'.encode()).hexdigest()}.json"
def _cg(k): return json.loads(_cp(k).read_text()) if _cp(k).exists() else None
def _cs(k, v): _cp(k).write_text(json.dumps(v))
def _lp(k): return LOCK_DIR / f"{hashlib.md5(f'{_today_et().isoformat()}:{k}'.encode()).hexdigest()}.json"
def _lg(k): return json.loads(_lp(k).read_text()) if _lp(k).exists() else None
def _ls(k, v): _lp(k).write_text(json.dumps(v))

def _is_locked(start_time_iso):
    """Returns True if current UTC time is within LOCK_BUFFER_MINUTES of game start (or past it)."""
    try:
        game_start = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return now >= game_start - timedelta(minutes=LOCK_BUFFER_MINUTES)
    except:
        return False

def _all_games_ended(games):
    """Returns True if all games on the slate have likely ended (3+ hours past start)."""
    now = datetime.now(timezone.utc)
    for g in games:
        st = g.get("startTime")
        if not st:
            continue
        try:
            game_start = datetime.fromisoformat(st.replace("Z", "+00:00"))
            if now < game_start + timedelta(hours=3):
                return False
        except:
            continue
    return len(games) > 0

def _safe_float(v, default=0.0):
    try: return float(v) if v is not None else default
    except: return default

def _espn_get(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except: return {}

def fetch_games(for_date=None):
    """Fetch games from ESPN scoreboard. If for_date is provided, fetch that date's games."""
    cache_key = f"games_{for_date}" if for_date else "games"
    c = _cg(cache_key)
    if c: return c
    url = f"{ESPN}/scoreboard"
    if for_date:
        url += f"?dates={for_date.strftime('%Y%m%d')}"
    data = _espn_get(url)
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
            "startTime": ev.get("date", ""),
        })
    _cs(cache_key, games)
    return games

def fetch_roster(team_id, team_abbr):
    c = _cg(f"roster_{team_id}")
    if c: return c
    data = _espn_get(f"{ESPN}/teams/{team_id}/roster")
    players = []
    for a in data.get("athletes", []):
        inj = a.get("injuries", [])
        inj_status = inj[0].get("status", "").strip() if inj else ""
        is_out = inj_status.lower() in ["out", "injured"] if inj_status else False
        # Only surface non-healthy statuses (Questionable, Day-To-Day, Doubtful, etc.)
        show_status = inj_status if inj_status.lower() not in ("", "active", "healthy") else ""
        players.append({
            "id": a["id"], "name": a["fullName"],
            "pos": a.get("position", {}).get("abbreviation", "G"),
            "is_out": is_out, "team_abbr": team_abbr,
            "injury_status": show_status,
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
            blended["recent_min"] = recent["min"]
            blended["recent_pts"] = recent["pts"]
            blended["season_pts"] = season["pts"]
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



# ─────────────────────────────────────────────────────────────────────────────
# THE CORE MODEL
#
# DFS Scoring Formula: PTS + REB + AST×1.5 + STL×3.5 + BLK×3.0 - TOV×1.2
#
# Real Sports Value = actual_score × slot_multiplier_received
# The slot multiplier is determined by ownership — high-owned players (stars)
# always land in low-multiplier slots.
#
# We estimate ownership via projected minutes:
#   Stars (33+ min)     → everyone drafts them → low slot mult ~0.9x → AVOID
#   Starters (28-33)    → popular → slot mult ~1.5x
#   Role players(22-28) → moderate ownership → slot mult ~2.2x
#   Bench (15-22)       → low ownership, high mult ~2.8x ← SWEET SPOT
#   Deep bench (<15)    → below minutes gate, filtered out
#
# STARTING 5 = best EV at moderate risk (role players + starters)
# MOONSHOT = 5 different players — lower usage, higher ceiling, higher production floor
# ─────────────────────────────────────────────────────────────────────────────

def _ownership_mult(proj_min, closeness_coeff=1.0):
    """Ownership multiplier recalibrated for Real Score with closeness interaction.

    Base tiers are minute-based (proxy for draft ownership).
    Closeness modulates: bench players in close games = goldmine,
    bench players in blowouts = worthless (ride the bench in garbage time).
    Stars in blowouts get slight boost (guaranteed minutes, fewer people draft them).
    """
    if proj_min < 15:   base_mult = 1.5
    elif proj_min < 22: base_mult = 2.5   # bench sweet spot
    elif proj_min < 28: base_mult = 2.0   # role players
    elif proj_min < 33: base_mult = 1.4   # starters
    else:               base_mult = 0.85  # stars — heavily owned

    if proj_min < 28:  # bench and role players
        if closeness_coeff > 1.1:   return round(base_mult * 1.15, 2)  # close game boost
        elif closeness_coeff < 0.8: return round(base_mult * 0.80, 2)  # blowout penalty
    else:  # starters and stars
        if closeness_coeff > 1.1:   return round(base_mult * 1.05, 2)  # slight close-game boost
        elif closeness_coeff < 0.8: return round(base_mult * 1.10, 2)  # blowout: stars still play

    return base_mult

def _dfs_score(pts, reb, ast, stl, blk, tov):
    """Legacy DFS scoring formula — kept for backward compat in game script ratio calc."""
    return pts + reb + (ast * 1.5) + (stl * 3.5) + (blk * 3.0) - (tov * 1.2)


# ─────────────────────────────────────────────────────────────────────────────
# REAL SCORE ESTIMATION ENGINE
#
# The Real Sports App uses a proprietary "Real Score" algorithm that is
# fundamentally different from traditional DFS. Game closeness is the #1
# variable — actions in close games are worth exponentially more than
# identical actions in blowouts. This engine approximates Real Score
# priorities using pre-game spread/total data from ESPN.
# ─────────────────────────────────────────────────────────────────────────────

def _closeness_coefficient(spread):
    """Game closeness multiplier — significant but not extreme.

    Calibrated against actual Real Score data (March 3 slate):
    - Edwards scored 6.2 RS in an 18pt blowout (spread ~5) => closeness still matters
    - Thompson scored 4.0 in a 28pt blowout (exceptional efficiency)
    - Banchero 5.4 in 2pt game, Adebayo 5.2 in OT => close games DO boost

    Pick-em (spread 0)  -> ~1.30x boost
    Close (spread 3)    -> ~1.10x
    Moderate (spread 5) -> ~0.95x (near baseline)
    Large (spread 7)    -> ~0.83x
    Blowout (spread 10) -> ~0.70x
    Mega-blowout (15+)  -> ~0.58x floor

    Creates a ~2.2x ratio between pick-em and mega-blowout.
    """
    s = abs(spread) if spread is not None else 7.0
    raw = math.exp(-0.07 * (s ** 1.3))
    return round(0.50 + 0.80 * raw, 3)


def _blowout_risk(spread, total):
    """Estimate blowout probability and game variance.

    Returns (blowout_prob, variance_score):
    - blowout_prob: 0.0 to 0.85, used to penalize star minutes
    - variance_score: close games + high totals = most variance = moonshot gold
    """
    s = abs(spread) if spread is not None else 7.0
    t = total or 222

    if s <= 3:      blowout_prob = 0.05
    elif s <= 6:    blowout_prob = 0.05 + (s - 3) * 0.05
    elif s <= 10:   blowout_prob = 0.20 + (s - 6) * 0.075
    elif s <= 14:   blowout_prob = 0.50 + (s - 10) * 0.075
    else:           blowout_prob = 0.85

    closeness_var = max(0, 1.0 - s / 15)
    pace_var = max(0, (t - 210) / 40)
    variance_score = closeness_var * 0.7 + pace_var * 0.3

    return blowout_prob, round(variance_score, 3)


def _real_score_estimate(pts, reb, ast, stl, blk, tov, closeness):
    """Approximate Real Score using closeness-weighted stat values.

    Unlike flat DFS scoring:
    - All stats amplified by closeness (close game = more valuable)
    - STL/BLK get extra closeness boost (defensive plays swing win probability)
    - AST weight increased to 1.8 (momentum/playmaking valued highly)
    - TOV penalty scales with closeness (turnovers in close games are devastating)
    """
    defensive_bonus = 1.0 + (closeness - 1.0) * 0.5

    base = (
        pts * 1.0 +
        reb * 1.1 +
        ast * 1.8 +
        stl * 4.0 * defensive_bonus +
        blk * 3.5 * defensive_bonus -
        tov * 1.5 * closeness
    )
    return base * closeness


# ─────────────────────────────────────────────────────────────────────────────
# GAME SCRIPT ENGINE — stat adjustments by O/U tier
#
# Over/under tiers adjust which stat categories get boosted:
#   < 220  → Defensive Grind: boost STL/BLK, suppress PTS/AST/REB volume
#   220-235 → Balanced Pace: neutral, lean on matchup and spread
#   236-245 → Fast-Paced: boost scorers, assist props, rebounders (shot volume)
#   > 245  → Track Meet: boost PTS+AST combos, but if spread > 8 penalize
#            (blowout risk = usage collapses for starters late)
# ─────────────────────────────────────────────────────────────────────────────

def _game_script_weights(total, spread):
    """Return per-stat multipliers based on over/under and spread."""
    w = {"pts": 1.0, "reb": 1.0, "ast": 1.0, "stl": 1.0, "blk": 1.0, "tov": 1.0}
    t = total or 222

    if t < 220:
        # Defensive Grind — steals/blocks are gold, volume stats suppressed
        w["pts"] = 0.85
        w["reb"] = 0.90
        w["ast"] = 0.85
        w["stl"] = 1.40
        w["blk"] = 1.35
        w["tov"] = 1.15   # turnovers hurt more in slow games
    elif t <= 235:
        # Balanced — slight lean toward matchup, essentially neutral
        w["pts"] = 1.0
        w["reb"] = 1.0
        w["ast"] = 1.0
        w["stl"] = 1.05
        w["blk"] = 1.05
    elif t <= 245:
        # Fast-Paced — scorers and playmakers thrive, shot volume is up
        w["pts"] = 1.15
        w["reb"] = 1.10
        w["ast"] = 1.15
        w["stl"] = 0.95
        w["blk"] = 0.95
        w["tov"] = 0.90   # turnovers less costly in high-scoring games
    else:
        # Track Meet (> 245) — huge scoring upside
        w["pts"] = 1.25
        w["ast"] = 1.20
        w["reb"] = 1.05
        w["stl"] = 0.90
        w["blk"] = 0.90
        w["tov"] = 0.85
        # Blowout risk: if spread > 8, starters sit in garbage time
        if abs(spread or 0) > 8:
            w["pts"] *= 0.88
            w["ast"] *= 0.88
            w["reb"] *= 0.92

    # Universal blowout penalty — applies to ALL O/U tiers when spread > 6
    # Real Score heavily penalizes actions in non-competitive game states
    s = abs(spread or 0)
    if s > 6:
        blowout_factor = max(0.70, 1.0 - (s - 6) * 0.035)
        w["pts"] *= blowout_factor
        w["ast"] *= blowout_factor
        w["reb"] *= (1.0 - (1.0 - blowout_factor) * 0.5)  # rebounds penalized less

    return w


def _game_script_dfs(stats, total, spread):
    """DFS score adjusted by game script weights. For per-game projections only."""
    w = _game_script_weights(total, spread)
    pts = stats.get("pts", 0) * w["pts"]
    reb = stats.get("reb", 0) * w["reb"]
    ast = stats.get("ast", 0) * w["ast"]
    stl = stats.get("stl", 0) * w["stl"]
    blk = stats.get("blk", 0) * w["blk"]
    tov = stats.get("tov", 0) * w["tov"]
    return _dfs_score(pts, reb, ast, stl, blk, tov)


def _game_script_label(total):
    """Human-readable game script tier for display."""
    t = total or 222
    if t < 220:   return "Defensive Grind"
    if t <= 235:  return "Balanced Pace"
    if t <= 245:  return "Fast-Paced"
    return "Track Meet"


def project_player(pinfo, stats, spread, total, side, team_abbr="",
                   cal_bias=0.0):
    if pinfo.get("is_out"): return None
    if pinfo.get("name", "").lower() in EXCLUDED_PLAYERS: return None
    avg_min = stats.get("min", 0)
    if avg_min <= 0: return None

    proj_min = avg_min

    # Minutes gate
    if proj_min < MIN_GATE: return None

    pts = stats["pts"]
    reb = stats["reb"]
    ast = stats["ast"]
    stl = stats.get("stl", 0)
    blk = stats.get("blk", 0)
    tov = stats.get("tov", 0)
    if pts + reb + ast <= 0: return None

    # ── GAME CONTEXT ──
    closeness = _closeness_coefficient(spread)
    blowout_prob, variance = _blowout_risk(spread, total)

    # Apply game script stat weights (now for ALL projections, not just per-game)
    w = _game_script_weights(total, spread)
    adj_pts = pts * w["pts"]
    adj_reb = reb * w["reb"]
    adj_ast = ast * w["ast"]
    adj_stl = stl * w["stl"]
    adj_blk = blk * w["blk"]
    adj_tov = tov * w["tov"]

    # ── REAL SCORE ESTIMATE (closeness is the dominant factor) ──
    heuristic = _real_score_estimate(adj_pts, adj_reb, adj_ast, adj_stl, adj_blk, adj_tov, closeness)

    # Declining usage penalty
    season_min = stats.get("season_min", avg_min)
    recent_min = stats.get("recent_min", avg_min)
    decline_factor = 1.0
    if season_min > 0 and recent_min < season_min * 0.85:
        decline_factor = recent_min / season_min
        heuristic *= decline_factor

    # ── BLOWOUT STAR PENALTY ──
    # High-minute players in projected blowouts lose minutes to garbage time
    if proj_min >= 28 and blowout_prob > 0.25:
        heuristic *= 1.0 - (blowout_prob * 0.30)

    # ── AI BLEND (85% heuristic / 15% AI — AI trained on wrong target) ──
    base = heuristic
    if AI_MODEL is not None:
        try:
            usage = min(max(pts / max(avg_min, 1) * 0.8, 0.9), 1.5)
            features = np.array([[avg_min, stats.get("season_pts", pts), usage, 112.0]])
            ai_pred = AI_MODEL.predict(features)[0]
            ai_norm = ai_pred * (heuristic / max(ai_pred, 1))
            base = (heuristic * 0.85) + (ai_norm * 0.15)
        except: pass

    # ── CONTEXT MULTIPLIERS (closeness already in _real_score_estimate) ──
    pace_adj = 1.0 + (0.04 * ((total or 222) - 222) / 20)
    home_adj = 1.02 if side == "home" else 1.0

    raw_score = (base * pace_adj * home_adj) / 6.0

    # Calibration bias
    if cal_bias != 0.0:
        raw_score += cal_bias

    # ── OWNERSHIP MULT (now closeness-aware) ──
    om_chalk = _ownership_mult(proj_min, closeness)
    chalk_ev = round(raw_score * om_chalk, 2)
    expected_dp = round(raw_score * om_chalk, 1)

    # ── MOONSHOT SCORE: production-weighted ceiling estimate ──
    recent_pts = stats.get("recent_pts", pts)
    season_pts = stats.get("season_pts", pts)
    recent_trend = round(recent_pts / max(season_pts, 1), 2)
    def_upside = (stl + blk) / max(proj_min, 1) * 10
    moon_score = round(
        chalk_ev * 1.5 +             # ownership-adjusted production — avoids star overfitting
        variance * 4.0 +             # close-game upside (halved from v1)
        recent_trend * 2.0 +         # hot hand
        def_upside * 2.0             # defensive Real Score boost
    , 2)

    return {
        "id":      pinfo["id"],
        "name":    pinfo["name"],
        "pos":     pinfo["pos"],
        "team":    team_abbr,
        "rating":  round(raw_score, 1),
        "chalk_ev":chalk_ev,
        "expected_dp": expected_dp,
        "predMin": round(proj_min, 1),
        "pts":     round(pts, 1),
        "reb":     round(reb, 1),
        "ast":     round(ast, 1),
        "stl":     round(stl, 1),
        "blk":     round(blk, 1),
        "tov":     round(tov, 1),
        "est_mult": om_chalk,
        "om":      om_chalk,
        "slot":    "1.0x",
        "_base":   base,
        "injury_status": pinfo.get("injury_status", ""),
        "_decline": round(decline_factor, 2),
        "_closeness": closeness,
        "_variance": variance,
        "_blowout_prob": round(blowout_prob, 3),
        "_recent_trend": recent_trend,
        "_moon_score": moon_score,
        "_features": {"avg_min": round(avg_min, 1), "season_pts": round(season_pts, 1)},
    }

def _run_game(game, cal_bias=0.0):
    cache_key = f"game_proj_{game['gameId']}"
    if cal_bias == 0.0:
        cached = _cg(cache_key)
        if cached: return cached

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

    # Project all active players
    out = []
    for p, ab, sd in players_in:
        stats = stats_map.get(p["id"])
        if not stats:
            continue
        proj = project_player(p, stats, game["spread"], game["total"], sd, ab,
                              cal_bias=cal_bias)
        if proj:
            out.append(proj)
    _cs(cache_key, out)
    return out

CHALK_FLOOR    = 3.0  # Minimum raw rating for Starting 5 (lowered for new scoring range)
MOONSHOT_FLOOR = 3.0  # Moonshot production floor — filters out non-producers

def _build_lineups(projections):
    # STARTING 5: sorted by chalk_ev (rating × ownership mult)
    # Raw rating alone overfits for stars — ownership mult penalizes high-minute
    # players that everyone drafts (low slot multiplier) and boosts bench/role
    # players that land in high-multiplier slots
    chalk_eligible = [p for p in projections if p["rating"] >= CHALK_FLOOR]
    chalk = sorted(chalk_eligible, key=lambda x: x["chalk_ev"], reverse=True)[:5]
    for i, p in enumerate(chalk): p["slot"] = SLOT_VALUES[i]

    # MOONSHOT: 5 different players — sorted by _moon_score (production + variance)
    # Targets players with close-game upside who actually produce stats
    chalk_names = {p["name"] for p in chalk}
    moonshot_pool = [p for p in projections
                     if p["name"] not in chalk_names and p["rating"] >= MOONSHOT_FLOOR]
    upside = sorted(moonshot_pool, key=lambda x: x.get("_moon_score", 0), reverse=True)[:5]
    for i, p in enumerate(upside): p["slot"] = SLOT_VALUES[i]

    return chalk, upside

# ─────────────────────────────────────────────────────────────────────────────
# PER-GAME LINEUP BUILDER
#
# Single-game drafts are fundamentally different from full-slate:
# - Only 2 teams, so everyone is picking from the same pool
# - Must diversify across both teams (min 2 per side)
# - Stars are MORE important in single-game (smaller pool = stars stand out)
# - Ownership is more concentrated, so contrarian plays matter more
# ─────────────────────────────────────────────────────────────────────────────

GAME_CHALK_FLOOR = 2.5    # Starting 5 floor for single-game (lowered for new scoring)
GAME_MOONSHOT_FLOOR = 1.5  # Moonshot wider net — variance-based ranking does the filtering

def _pick_balanced(pool, n, min_per_team=2, sort_key="chalk_ev"):
    """Pick n players from pool ensuring at least min_per_team from each team."""
    if not pool:
        return []

    teams = {}
    for p in pool:
        t = p["team"]
        teams.setdefault(t, []).append(p)

    for t in teams:
        teams[t].sort(key=lambda x: x[sort_key], reverse=True)

    team_list = list(teams.keys())
    if len(team_list) < 2:
        return sorted(pool, key=lambda x: x[sort_key], reverse=True)[:n]

    picked = []
    used = set()

    # Phase 1: guarantee min_per_team from each team (capped by available players)
    for t in team_list:
        avail = min(min_per_team, len(teams[t]))
        for p in teams[t][:avail]:
            picked.append(p)
            used.add(p["name"])

    # Phase 2: fill remaining slots from best available across both teams
    remaining = n - len(picked)
    if remaining > 0:
        rest = sorted([p for p in pool if p["name"] not in used],
                      key=lambda x: x[sort_key], reverse=True)
        picked.extend(rest[:remaining])

    picked.sort(key=lambda x: x[sort_key], reverse=True)
    return picked[:n]


def _apply_game_script(projections, game):
    """Re-score projections using game script weights. Returns new list (deep copies, no mutation)."""
    total  = game.get("total") or 222
    spread = game.get("spread") or 0
    rescored = []
    for p in projections:
        gs_dfs = _game_script_dfs(p, total, spread)
        orig_dfs = _dfs_score(p.get("pts",0), p.get("reb",0), p.get("ast",0),
                              p.get("stl",0), p.get("blk",0), p.get("tov",0))
        if orig_dfs > 0:
            script_factor = gs_dfs / orig_dfs
        else:
            script_factor = 1.0
        new_rating  = round(p["rating"] * script_factor, 1)
        new_ev      = round(p["chalk_ev"] * script_factor, 2)
        rp = copy.deepcopy(p)
        rp["rating"]   = new_rating
        rp["chalk_ev"] = new_ev
        rp["expected_dp"] = round(new_ev, 1)
        rp["game_script"] = _game_script_label(total)
        rescored.append(rp)
    return rescored


def _build_game_lineups(projections, game):
    """Build lineups for a single-game draft with team balance.

    Game script + closeness already applied in project_player, so projections
    are used directly (no rescoring needed).
    """
    # Add game script label for display
    total = game.get("total")
    for p in projections:
        p["game_script"] = _game_script_label(total)

    chalk_eligible = [p for p in projections if p["rating"] >= GAME_CHALK_FLOOR]
    moon_eligible = [p for p in projections if p["rating"] >= GAME_MOONSHOT_FLOOR]

    # STARTING 5: best raw Real Score projection, balanced across both teams
    chalk = _pick_balanced(chalk_eligible, 5, min_per_team=2, sort_key="rating")
    for i, p in enumerate(chalk): p["slot"] = SLOT_VALUES[i]

    # MOONSHOT: best variance-weighted score, balanced (overlap OK in 2-team pool)
    moonshot = _pick_balanced(moon_eligible, 5, min_per_team=2, sort_key="_moon_score")
    for i, p in enumerate(moonshot): p["slot"] = SLOT_VALUES[i]

    return chalk, moonshot

def _get_injuries(game):
    """Get list of OUT players for a game (from cached roster data)."""
    out_players = []
    for side in ["home", "away"]:
        team = game[side]
        roster = _cg(f"roster_{team['id']}")
        if not roster:
            continue
        for p in roster:
            if p.get("is_out"):
                out_players.append({"name": p["name"], "team": team["abbr"], "pos": p["pos"]})
    return out_players

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
    games = fetch_games()
    for g in games:
        g["locked"] = _is_locked(g.get("startTime")) if g.get("startTime") else False
    return games

@app.get("/api/slate")
async def get_slate(cal_bias: float = Query(0.0)):
    today_games = fetch_games()

    # Derive slate date from games' start times, not system clock.
    # After midnight ET, ESPN still returns in-progress games from "yesterday" —
    # we want the slate date to reflect the actual games being shown.
    def _slate_date_from_games(game_list):
        starts = [g["startTime"] for g in game_list if g.get("startTime")]
        if starts:
            try:
                earliest = min(starts)
                return datetime.fromisoformat(earliest.replace("Z", "+00:00")).astimezone(ET).date()
            except Exception:
                pass
        return _today_et()

    # If today's slate is over, transition to next day
    if today_games and _all_games_ended(today_games):
        tomorrow = _today_et() + timedelta(days=1)
        next_games = fetch_games(for_date=tomorrow)
        if next_games:
            slate_date = _slate_date_from_games(next_games)
            games = next_games
        else:
            slate_date = _slate_date_from_games(today_games)
            games = today_games  # fallback to today if tomorrow has no games yet
    elif not today_games:
        # No games today — check tomorrow
        tomorrow = _today_et() + timedelta(days=1)
        next_games = fetch_games(for_date=tomorrow)
        if next_games:
            slate_date = _slate_date_from_games(next_games)
            games = next_games
        else:
            return {"date": _today_et().isoformat(), "games": [], "lineups": {"chalk": [], "upside": []}, "locked": False}
    else:
        slate_date = _slate_date_from_games(today_games)
        games = today_games

    # Check if slate is locked (5 min before earliest game)
    start_times = [g["startTime"] for g in games if g.get("startTime")]
    earliest = min(start_times) if start_times else None
    locked = _is_locked(earliest) if earliest else False

    cache_suffix = f"_{slate_date.isoformat()}" if slate_date != _today_et() else ""
    if locked:
        lock_cached = _lg(f"slate_v5_locked{cache_suffix}")
        if lock_cached:
            lock_cached["locked"] = True
            return lock_cached

    if cal_bias == 0.0:
        cached = _cg(f"slate_v5{cache_suffix}")
        if cached:
            cached["locked"] = locked
            if locked:
                _ls(f"slate_v5_locked{cache_suffix}", cached)
            return cached

    all_proj = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed({pool.submit(_run_game, g, cal_bias): g for g in games}):
            try: all_proj.extend(fut.result())
            except Exception as e: print(f"slate err: {e}")
    chalk, upside = _build_lineups(all_proj)
    result = {"date": slate_date.isoformat(), "games": games,
              "lineups": {"chalk": chalk, "upside": upside}, "locked": locked}
    if cal_bias == 0.0:
        _cs(f"slate_v5{cache_suffix}", result)
    if locked:
        _ls(f"slate_v5_locked{cache_suffix}", result)
    return result

@app.get("/api/picks")
async def get_picks(gameId: str = Query(...), cal_bias: float = Query(0.0)):
    game = next((g for g in fetch_games() if g["gameId"] == gameId), None)
    if not game:
        return JSONResponse({"error": "Game not found"}, status_code=404)

    start_time = game.get("startTime")
    locked = _is_locked(start_time) if start_time else False
    lock_key = f"picks_locked_{gameId}"

    if locked:
        lock_cached = _lg(lock_key)
        if lock_cached:
            lock_cached["locked"] = True
            return lock_cached

    projections = _run_game(game, cal_bias=cal_bias)
    if not projections:
        return JSONResponse({"error": "No projections available."}, status_code=503)
    chalk, upside = _build_game_lineups(projections, game)
    _save_history(game["label"], chalk)
    script = _game_script_label(game.get("total"))
    injuries = _get_injuries(game)
    result = {"date": _today_et().isoformat(), "game": game,
              "gameScript": script,
              "lineups": {"chalk": chalk, "upside": upside},
              "locked": locked,
              "injuries": injuries}
    if locked:
        _ls(lock_key, result)
    return result

@app.get("/api/history")
async def get_history():
    if not HISTORY_FILE.exists(): return []
    try: return json.loads(HISTORY_FILE.read_text())
    except: return []

CALIBRATION_FILE = DATA_DIR / "calibration.json"
ACTUALS_FILE = DATA_DIR / "actuals_log.json"

def _load_calibration():
    if CALIBRATION_FILE.exists():
        try: return json.loads(CALIBRATION_FILE.read_text())
        except: pass
    return {"bias": 0.0, "samples": 0}

def _update_calibration(scores):
    """Update running calibration bias from uploaded actuals."""
    cal = _load_calibration()
    errors = [s["actual_score"] - s["predicted_rating"] for s in scores
              if s.get("actual_score") is not None and s.get("predicted_rating") is not None]
    if not errors:
        return cal
    new_avg_error = sum(errors) / len(errors)
    alpha = 0.3
    if cal["samples"] > 0:
        cal["bias"] = round(cal["bias"] * (1 - alpha) + new_avg_error * alpha, 3)
    else:
        cal["bias"] = round(new_avg_error, 3)
    cal["samples"] += len(errors)
    CALIBRATION_FILE.write_text(json.dumps(cal))
    return cal

@app.post("/api/actuals")
async def save_actuals(payload: dict = Body(...)):
    date_str = payload.get("date")
    scope = payload.get("scope", "slate")
    scores = payload.get("scores", [])
    if not date_str or not scores:
        return JSONResponse({"error": "Missing date or scores"}, status_code=400)
    for s in scores:
        val = s.get("actual_score")
        if val is None or not (0 <= val <= 150):
            return JSONResponse({"error": f"Invalid score for {s.get('name', '?')}: must be 0-150"}, status_code=400)
    # Append to actuals log
    log = []
    if ACTUALS_FILE.exists():
        try: log = json.loads(ACTUALS_FILE.read_text())
        except: pass
    log.append({"date": date_str, "scope": scope, "scores": scores, "ts": datetime.now().isoformat()})
    ACTUALS_FILE.write_text(json.dumps(log[-200:], indent=2))
    # Update calibration
    cal = _update_calibration(scores)
    return {"status": "saved", "calibration": cal}

@app.get("/api/evaluate")
async def evaluate():
    """Returns current calibration stats."""
    return _load_calibration()

@app.get("/api/refresh")
async def refresh():
    cleared = 0
    try:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink(); cleared += 1
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return {"status": "ok", "cleared": cleared, "ts": datetime.now().isoformat()}
