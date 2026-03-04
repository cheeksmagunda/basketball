import json
import copy
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

DATA_DIR = Path("/tmp/nba_data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_FILE = DATA_DIR / "lineup_history.json"
CACHE_DIR = Path("/tmp/nba_cache_v19")
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

def _cp(k): return CACHE_DIR / f"{hashlib.md5(f'{date.today().isoformat()}:{k}'.encode()).hexdigest()}.json"
def _cg(k): return json.loads(_cp(k).read_text()) if _cp(k).exists() else None
def _cs(k, v): _cp(k).write_text(json.dumps(v))
def _lp(k): return LOCK_DIR / f"{hashlib.md5(f'{date.today().isoformat()}:{k}'.encode()).hexdigest()}.json"
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
            "startTime": ev.get("date", ""),
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
        inj_status = inj[0].get("status", "") if inj else ""
        is_out = inj_status.lower() in ["out", "injured"] if inj_status else False
        players.append({
            "id": a["id"], "name": a["fullName"],
            "pos": a.get("position", {}).get("abbreviation", "G"),
            "is_out": is_out, "team_abbr": team_abbr,
            "injury_status": inj_status,
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
# STARTING 5 = best EV at moderate risk (role players + starters)
# MOONSHOT = 5 different players — lower usage, higher ceiling, higher production floor
# ─────────────────────────────────────────────────────────────────────────────

def _ownership_mult_chalk(proj_min):
    """Moderate inverse-ownership mult. Penalizes stars, rewards role players."""
    if proj_min < 15:  return 1.8   # deep bench — too risky for chalk
    if proj_min < 22:  return 2.8   # bench sweet spot (Sheppard, González tier)
    if proj_min < 28:  return 2.2   # role players
    if proj_min < 33:  return 1.5   # starters
    return 0.9                      # stars — everyone drafts them, low mult

def _dfs_score(pts, reb, ast, stl, blk, tov):
    """Full DFS scoring formula — matches the leaderboard exactly."""
    return pts + reb + (ast * 1.5) + (stl * 3.5) + (blk * 3.0) - (tov * 1.2)


# ─────────────────────────────────────────────────────────────────────────────
# GAME SCRIPT ENGINE (per-game only — does NOT affect full slate)
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
                   cascade_bonus=0.0, cal_bias=0.0):
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

    # Scale heuristic by minute boost from cascade (capped at 1.4x)
    if cascade_bonus > 0 and avg_min > 0:
        min_scale = min(proj_min / avg_min, 1.4)
        heuristic *= min_scale

    # Declining usage penalty: if recent minutes dropped >15% vs season,
    # scale output proportionally (e.g. Conley post-trade: 26→19 min = 0.73x)
    season_min = stats.get("season_min", avg_min)
    recent_min = stats.get("recent_min", avg_min)
    decline_factor = 1.0
    if season_min > 0 and recent_min < season_min * 0.85:
        decline_factor = recent_min / season_min
        heuristic *= decline_factor

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

    # Apply calibration bias from user-uploaded actuals
    if cal_bias != 0.0:
        raw_score += cal_bias

    # Use projected minutes (with cascade) for ownership tiers
    om_chalk  = _ownership_mult_chalk(proj_min)

    # EV score — ownership-weighted expected value
    chalk_ev  = round(raw_score * om_chalk, 2)

    # Expected draft points (EDP) = raw_score * est_mult
    expected_dp = round(raw_score * om_chalk, 1)

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
        "is_cascade_pick": is_cascade,
        "injury_status": pinfo.get("injury_status", ""),
        "_decline": round(decline_factor, 2),
        "_features": {"avg_min": round(avg_min, 1), "season_pts": round(stats.get("season_pts", pts), 1)},
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
                              cascade_bonus=cascade_bonus, cal_bias=cal_bias)
        if proj:
            out.append(proj)
    _cs(cache_key, out)
    return out

CHALK_FLOOR    = 3.5  # Minimum raw rating for Starting 5
MOONSHOT_FLOOR = 6.0  # Higher floor for Moonshot — filters low-production bench warmers

def _build_lineups(projections):
    # STARTING 5: sorted by chalk_ev, with production floor filter
    chalk_eligible = [p for p in projections if p["rating"] >= CHALK_FLOOR]
    chalk = sorted(chalk_eligible, key=lambda x: x["chalk_ev"], reverse=True)[:5]
    for i, p in enumerate(chalk): p["slot"] = SLOT_VALUES[i]

    # MOONSHOT: 5 totally different players — higher production floor
    chalk_names = {p["name"] for p in chalk}
    moonshot_pool = [p for p in projections
                     if p["name"] not in chalk_names and p["rating"] >= MOONSHOT_FLOOR]
    upside = sorted(moonshot_pool, key=lambda x: x["chalk_ev"], reverse=True)[:5]
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

GAME_CHALK_FLOOR = 3.5    # Starting 5 floor for single-game
GAME_MOONSHOT_FLOOR = 2.5  # Lower floor for moonshot — wider net for upside plays

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
    """Build lineups for a single-game draft with team balance + game script.

    Starting 5: sorted by chalk_ev (ownership-weighted EV) — rewards role players
                 in low-ownership draft slots. Floor = GAME_CHALK_FLOOR (3.5).
    Moonshot:   sorted by raw rating (pure ceiling) — rewards stars and high-production
                 players regardless of ownership. Overlap with Starting 5 is allowed
                 because the 2-team pool is too small to force 10 unique players.
                 Floor = GAME_MOONSHOT_FLOOR (2.5) — wider net catches breakout candidates.
    """
    rescored = _apply_game_script(projections, game)
    chalk_eligible = [p for p in rescored if p["rating"] >= GAME_CHALK_FLOOR]
    moon_eligible = [p for p in rescored if p["rating"] >= GAME_MOONSHOT_FLOOR]

    # STARTING 5: best ownership-weighted EV, balanced across both teams
    chalk = _pick_balanced(chalk_eligible, 5, min_per_team=2)
    for i, p in enumerate(chalk): p["slot"] = SLOT_VALUES[i]

    # MOONSHOT: best raw ceiling, balanced — independent from Starting 5 (overlap OK)
    moonshot = _pick_balanced(moon_eligible, 5, min_per_team=2, sort_key="rating")
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
    games = fetch_games()
    if not games:
        return {"date": date.today().isoformat(), "games": [], "lineups": {"chalk": [], "upside": []}, "locked": False}

    # Check if slate is locked (5 min before earliest game)
    start_times = [g["startTime"] for g in games if g.get("startTime")]
    earliest = min(start_times) if start_times else None
    locked = _is_locked(earliest) if earliest else False

    if locked:
        lock_cached = _lg("slate_v5_locked")
        if lock_cached:
            lock_cached["locked"] = True
            return lock_cached

    if cal_bias == 0.0:
        cached = _cg("slate_v5")
        if cached:
            cached["locked"] = locked
            if locked:
                _ls("slate_v5_locked", cached)
            return cached

    all_proj = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed({pool.submit(_run_game, g, cal_bias): g for g in games}):
            try: all_proj.extend(fut.result())
            except Exception as e: print(f"slate err: {e}")
    chalk, upside = _build_lineups(all_proj)
    result = {"date": date.today().isoformat(), "games": games,
              "lineups": {"chalk": chalk, "upside": upside}, "locked": locked}
    if cal_bias == 0.0:
        _cs("slate_v5", result)
    if locked:
        _ls("slate_v5_locked", result)
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
    cascade_count = sum(1 for p in chalk + upside if p.get("is_cascade_pick"))
    result = {"date": date.today().isoformat(), "game": game,
              "gameScript": script,
              "lineups": {"chalk": chalk, "upside": upside},
              "locked": locked,
              "injuries": injuries,
              "cascadeCount": cascade_count}
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
