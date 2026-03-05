import json
import copy
import hashlib
import pickle
import os
import base64
import numpy as np
import requests
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Query, Body, File, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Real Score Ecosystem modules
try:
    from api.real_score import real_score_projection, _make_rng
    from api.asset_optimizer import optimize_lineup
    from api.line_engine import run_line_engine
except ImportError:
    from .real_score import real_score_projection, _make_rng
    from .asset_optimizer import optimize_lineup
    from .line_engine import run_line_engine

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

load_dotenv()
app = FastAPI()

# ── GitHub API helpers for persistent CSV storage ──
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "")  # e.g. "cheeksmagunda/basketball"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

def _github_get_file(path):
    """Get file content and SHA from GitHub. Returns (content_str, sha) or (None, None)."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None, None
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    r = requests.get(url, headers={
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }, timeout=10)
    if r.status_code == 200:
        data = r.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content, data["sha"]
    return None, None

def _github_list_dir(path):
    """List files in a GitHub repo directory. Returns list of {name, path} dicts."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return []
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    r = requests.get(url, headers={
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }, timeout=10)
    if r.status_code == 200:
        return r.json()
    return []

def _github_write_file(path, content, message="auto-update"):
    """Create or update a file in the GitHub repo via Contents API."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return {"error": "GITHUB_TOKEN or GITHUB_REPO not configured"}
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    _, sha = _github_get_file(path)
    payload = {
        "message": message,
        "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
    }
    if sha:
        payload["sha"] = sha
    r = requests.put(url, json=payload, headers={
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }, timeout=15)
    if r.status_code in (200, 201):
        return {"ok": True, "path": path}
    return {"error": f"GitHub API {r.status_code}: {r.text[:200]}"}

def _csv_escape(v):
    """Escape a value for CSV (quote if it contains commas or quotes)."""
    s = str(v)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s

def _predictions_to_csv(lineups, scope):
    """Convert lineup dicts to CSV rows."""
    rows = []
    for lineup_type, players in [("chalk", lineups.get("chalk", [])), ("upside", lineups.get("upside", []))]:
        for p in players:
            rows.append(",".join(_csv_escape(v) for v in [
                scope, lineup_type, p.get("slot", ""), p.get("name", ""),
                p.get("id", ""), p.get("team", ""), p.get("pos", ""),
                p.get("rating", ""), p.get("est_mult", ""),
                p.get("predMin", ""), p.get("pts", ""), p.get("reb", ""),
                p.get("ast", ""), p.get("stl", ""), p.get("blk", ""),
            ]))
    return rows

CSV_HEADER = "scope,lineup_type,slot,player_name,player_id,team,pos,predicted_rs,est_card_boost,pred_min,pts,reb,ast,stl,blk"

CACHE_DIR = Path("/tmp/nba_cache_v19")
CACHE_DIR.mkdir(exist_ok=True)
LOCK_DIR = Path("/tmp/nba_locks_v1")
LOCK_DIR.mkdir(exist_ok=True)

CONFIG_CACHE_DIR = Path("/tmp/nba_config_v1")
CONFIG_CACHE_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONFIG — Loaded from data/model-config.json on GitHub.
# Falls back to hardcoded defaults if GitHub is unreachable or file missing.
# The Lab (Phase 3) writes config updates here; changes take effect within 5 min.
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_DEFAULTS = {
    "version": 1,
    "card_boost": {
        "decay_base": 0.70, "ceiling": 3.0, "floor": 0.2,
        "base_offset": 0.3, "scalar": 3.4, "big_market_multiplier": 1.5,
        "big_market_teams": ["LAL","GSW","BOS","NYK","PHI","MIL","DAL","PHX","MIA","DEN","LAC","CHI","SA"],
    },
    "game_script": {
        "defensive_grind_ceiling": 220, "balanced_ceiling": 235, "fast_paced_ceiling": 245,
        "defensive_grind": {"pts":0.85,"reb":0.90,"ast":0.85,"stl":1.40,"blk":1.35,"tov":1.15},
        "balanced":        {"pts":1.0, "reb":1.0, "ast":1.0, "stl":1.05,"blk":1.05,"tov":1.0},
        "fast_paced":      {"pts":1.15,"reb":1.10,"ast":1.15,"stl":0.95,"blk":0.95,"tov":0.90},
        "track_meet":      {"pts":1.25,"reb":1.05,"ast":1.20,"stl":0.90,"blk":0.90,"tov":0.85},
        "blowout_spread_threshold":8,"blowout_pts_penalty":0.90,
        "blowout_ast_penalty":0.90,"blowout_reb_penalty":0.94,
    },
    "real_score": {"dfs_weights":{"pts":1.0,"reb":1.0,"ast":1.5,"stl":4.5,"blk":4.0,"tov":-1.2}},
    "cascade": {"redistribution_rate":0.70,"per_player_cap_minutes":3.0,"center_forward_share":0.30},
    "projection": {
        "min_gate_minutes":12,"lock_buffer_minutes":5,"season_recent_blend":0.5,
        "major_role_change_threshold":0.75,"major_role_change_recent_weight":0.80,
        "moderate_decline_threshold":0.90,"moderate_decline_recent_weight":0.65,
    },
    "contrarian": {
        "closeness_boost_floor":0.7,"closeness_boost_scalar":0.6,
        "underdog_bonus":1.1,"underdog_spread_min":2,"underdog_spread_max":7,
    },
}

def _load_config():
    """Load model config from GitHub (data/model-config.json), cache 5 min.
    Falls back to _CONFIG_DEFAULTS if unreachable or file missing — never breaks."""
    cache_file = CONFIG_CACHE_DIR / "model_config.json"
    # 5-minute TTL
    if cache_file.exists():
        try:
            age = datetime.now(timezone.utc).timestamp() - cache_file.stat().st_mtime
            if age < 300:
                return json.loads(cache_file.read_text())
        except: pass
    try:
        content, _ = _github_get_file("data/model-config.json")
        if content:
            cfg = json.loads(content)
            cache_file.write_text(json.dumps(cfg))
            return cfg
    except: pass
    return _CONFIG_DEFAULTS

def _cfg(path, default=None):
    """Read a dot-notation path from the loaded config with a default fallback.
    e.g. _cfg('card_boost.decay_base', 0.70)"""
    cfg = _load_config()
    keys = path.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return val

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
MIN_GATE = 12           # Minimum projected minutes — lowered from 15 to catch
                        # deep bench (Clifford, Riley) who win in garbage time
DEFAULT_TOTAL = 222     # Fallback over/under when odds unavailable

def _cp(k): return CACHE_DIR / f"{hashlib.md5(f'{date.today().isoformat()}:{k}'.encode()).hexdigest()}.json"
def _cg(k): return json.loads(_cp(k).read_text()) if _cp(k).exists() else None
def _cs(k, v): _cp(k).write_text(json.dumps(v))
def _lp(k): return LOCK_DIR / f"{hashlib.md5(f'{date.today().isoformat()}:{k}'.encode()).hexdigest()}.json"
def _lg(k): return json.loads(_lp(k).read_text()) if _lp(k).exists() else None
def _ls(k, v): _lp(k).write_text(json.dumps(v))

def _is_locked(start_time_iso):
    """Returns True if current UTC time is within lock_buffer_minutes of game start.
    Returns False for completed games (>3h past start) to avoid stale ESPN data."""
    try:
        lock_buf = _cfg("projection.lock_buffer_minutes", 5)
        game_start = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        if now > game_start + timedelta(hours=3):
            return False
        return now >= game_start - timedelta(minutes=lock_buf)
    except:
        return False

def _is_completed(start_time_iso):
    """Returns True if the game has already passed its lock window."""
    try:
        lock_buf = _cfg("projection.lock_buffer_minutes", 5)
        game_start = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return now >= game_start - timedelta(minutes=lock_buf)
    except:
        return False

def _et_date():
    """Current date in Eastern Time (handles EST/EDT)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date()
    except ImportError:
        # Fallback: EST=UTC-5 (Nov–Mar), EDT=UTC-4 (Mar–Nov)
        now_utc = datetime.now(timezone.utc)
        offset = timedelta(hours=-4 if 3 < now_utc.month < 11 else -5)
        return (now_utc + offset).date()

def _safe_float(v, default=0.0):
    try: return float(v) if v is not None else default
    except: return default

def _espn_get(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except: return {}

def _fetch_b2b_teams():
    """Detect teams on the second night of a back-to-back.

    Fetches yesterday's scoreboard and returns a set of team abbreviations
    that played yesterday. Players on these teams should be penalized —
    rest-managed players (Robinson, older players) often sit or get
    reduced minutes on B2Bs.
    """
    cache_key = "b2b_teams"
    c = _cg(cache_key)
    if c is not None: return set(c)
    yesterday = (_et_date() - timedelta(days=1)).strftime("%Y%m%d")
    data = _espn_get(f"{ESPN}/scoreboard?dates={yesterday}")
    b2b = set()
    for ev in data.get("events", []):
        comp = ev["competitions"][0]
        for cd in comp.get("competitors", []):
            abbr = cd.get("team", {}).get("abbreviation", "")
            if abbr: b2b.add(abbr)
    _cs(cache_key, list(b2b))
    return b2b

def fetch_games():
    today_et = _et_date()
    cache_key = f"games_{today_et}"
    c = _cg(cache_key)
    if c: return c
    b2b_teams = _fetch_b2b_teams()
    # Pass explicit ET date so ESPN returns the correct day's schedule regardless
    # of what ESPN considers its internal "current" day (often Pacific time).
    date_str = today_et.strftime("%Y%m%d")
    data = _espn_get(f"{ESPN}/scoreboard?dates={date_str}")
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
            "home_b2b": home["abbr"] in b2b_teams,
            "away_b2b": away["abbr"] in b2b_teams,
        })
    _cs(cache_key, games)
    return games

def fetch_roster(team_id, team_abbr):
    c = _cg(f"roster_{team_id}")
    if c: return c
    data = _espn_get(f"{ESPN}/teams/{team_id}/roster")
    # ESPN sometimes returns athletes grouped by position:
    # {"athletes": [{"position": "Guard", "items": [...]}, ...]}
    # Flatten to a single list of athlete objects before iterating.
    raw = data.get("athletes", [])
    flat = []
    for item in raw:
        if "items" in item:
            flat.extend(item["items"])
        else:
            flat.append(item)
    players = []
    for a in flat:
        try:
            inj = a.get("injuries", [])
            inj_status = inj[0].get("status", "").lower() if inj else ""
            is_out = inj_status in ["out", "injured"]
            # Capture injury status for UI badge (Questionable, Day-To-Day, Doubtful)
            injury_label = ""
            if inj and not is_out:
                raw_s = inj[0].get("status", "") or inj[0].get("type", {}).get("description", "")
                if raw_s:
                    rl = raw_s.lower()
                    if "question" in rl:       injury_label = "GTD"
                    elif "day" in rl:          injury_label = "DTD"
                    elif "doubt" in rl:        injury_label = "DOUBT"
                    elif "prob" in rl:         injury_label = ""  # Probable = fine
                    elif rl not in ["active", "healthy", ""]:
                        injury_label = raw_s[:8].upper()
            players.append({
                "id": a["id"], "name": a["fullName"],
                "pos": a.get("position", {}).get("abbreviation", "G"),
                "is_out": is_out, "team_abbr": team_abbr,
                "injury_status": injury_label,
            })
        except (KeyError, TypeError):
            continue
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
            # Minutes: when recent is significantly lower than season, override
            # with heavier recent weight. Catches role changes mid-season
            # (e.g. Drummond 25.7 season → 16.1 recent after Embiid return,
            #  Love 25.8 → 14.8 after Utah trade, Clarkson 24.2 → 18.8)
            proj = _cfg("projection", _CONFIG_DEFAULTS["projection"])
            major_thr   = proj.get("major_role_change_threshold", 0.75)
            major_w     = proj.get("major_role_change_recent_weight", 0.80)
            mod_thr     = proj.get("moderate_decline_threshold", 0.90)
            mod_w       = proj.get("moderate_decline_recent_weight", 0.65)
            blend_w     = proj.get("season_recent_blend", 0.5)

            min_ratio = recent["min"] / max(season["min"], 1)
            if min_ratio < major_thr:
                min_blend = round(season["min"] * (1 - major_w) + recent["min"] * major_w, 2)
            elif min_ratio < mod_thr:
                min_blend = round(season["min"] * (1 - mod_w) + recent["min"] * mod_w, 2)
            else:
                min_blend = round(season["min"] * (1 - blend_w) + recent["min"] * blend_w, 2)

            blended = {k: round(season[k] * (1 - blend_w) + recent[k] * blend_w, 2) for k in season}
            blended["min"] = min_blend  # Override minutes with smart blend
            blended["season_min"] = season["min"]
            blended["recent_min"] = recent["min"]
            blended["recent_pts"] = recent["pts"]
            blended["season_pts"] = season["pts"]
            blended["recent_stl"] = recent["stl"]
            blended["season_stl"] = season["stl"]
            blended["recent_blk"] = recent["blk"]
            blended["season_blk"] = season["blk"]
        else:
            blended = dict(season)
            blended["season_min"] = season["min"]
            blended["recent_min"] = season["min"]
            blended["recent_pts"] = season["pts"]
            blended["season_pts"] = season["pts"]
            blended["recent_stl"] = season["stl"]
            blended["season_stl"] = season["stl"]
            blended["recent_blk"] = season["blk"]
            blended["season_blk"] = season["blk"]
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
                cf_share = _cfg("cascade.center_forward_share", 0.30)
                freed_by_group["F"] = freed_by_group.get("F", 0) + os.get("min", 0) * cf_share

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
                redist_rate = _cfg("cascade.redistribution_rate", 0.70)
                bonus = freed_min * weight * redist_rate
                pid = rp["id"]
                if pid not in cascade_flags:
                    cascade_flags[pid] = 0.0
                cascade_flags[pid] += bonus

    # Cap per-player cascade (config: cascade.per_player_cap_minutes)
    cap_min = _cfg("cascade.per_player_cap_minutes", 3.0)
    for pid in cascade_flags:
        cascade_flags[pid] = min(cascade_flags[pid], cap_min)

    return cascade_flags


# ─────────────────────────────────────────────────────────────────────────────
# THE CORE MODEL — Optimized for the Real Sports App
#
# ADDITIVE scoring formula: Value = Real Score × (Slot_Mult + Card_Boost)
#   Jabari Walker: 4.7 RS × (2.0 + 3.0) = 23.5  ← wins
#   Anthony Edwards: 6.2 RS × (2.0 + 0.3) = 14.3  ← loses
#
# Card boost is INVERSELY driven by ownership:
#   Stars (7k+ drafts) → +0.3x | Role players (<50 drafts) → +2.5-3.0x
#
# Real Score proxy: PTS + REB + AST×1.5 + STL×4.5 + BLK×4.0 - TOV×1.2
#
# STARTING 5: MILP optimizes Σ rating × (slot + card_boost)
# MOONSHOT: same ranking, next 5 players (ranks 6-10) — different exposure,
#           same methodology. Avoids DNP risks from extreme low-ownership picks.
# ─────────────────────────────────────────────────────────────────────────────

def _est_card_boost(proj_min, pts, team_abbr):
    """Estimate ADDITIVE card boost based on predicted draft popularity.
    All parameters read from runtime config (data/model-config.json).
    Falls back to calibrated defaults if config unavailable.
    """
    cb = _cfg("card_boost", _CONFIG_DEFAULTS["card_boost"])
    decay_base     = cb.get("decay_base", 0.70)
    ceiling        = cb.get("ceiling", 3.0)
    floor_val      = cb.get("floor", 0.2)
    base_offset    = cb.get("base_offset", 0.3)
    scalar         = cb.get("scalar", 3.4)
    bm_mult        = cb.get("big_market_multiplier", 1.5)
    big_markets    = set(cb.get("big_market_teams", ["LAL","GSW","BOS","NYK","PHI","MIL","DAL","PHX","MIA","DEN","LAC","CHI","SA"]))

    hype = (pts / 10.0) ** 2 * (proj_min / 30.0) ** 0.5
    if team_abbr in big_markets:
        hype *= bm_mult
    boost = scalar * (decay_base ** hype) + base_offset
    return round(min(max(boost, floor_val), ceiling), 1)

def _dfs_score(pts, reb, ast, stl, blk, tov):
    """Real Score-aligned formula. Weights read from runtime config."""
    w = _cfg("real_score.dfs_weights", {"pts":1.0,"reb":1.0,"ast":1.5,"stl":4.5,"blk":4.0,"tov":-1.2})
    return (pts * w.get("pts", 1.0) + reb * w.get("reb", 1.0) +
            ast * w.get("ast", 1.5) + stl * w.get("stl", 4.5) +
            blk * w.get("blk", 4.0) + tov * w.get("tov", -1.2))


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
    """Return per-stat multipliers based on over/under and spread. Reads from runtime config."""
    gs = _cfg("game_script", _CONFIG_DEFAULTS["game_script"])
    t = total or DEFAULT_TOTAL

    dg_ceil  = gs.get("defensive_grind_ceiling", 220)
    bal_ceil = gs.get("balanced_ceiling", 235)
    fp_ceil  = gs.get("fast_paced_ceiling", 245)
    blow_thr = gs.get("blowout_spread_threshold", 8)

    if t < dg_ceil:
        tier = "defensive_grind"
    elif t <= bal_ceil:
        tier = "balanced"
    elif t <= fp_ceil:
        tier = "fast_paced"
    else:
        tier = "track_meet"

    defaults = _CONFIG_DEFAULTS["game_script"][tier]
    tier_cfg = gs.get(tier, defaults)
    w = {k: tier_cfg.get(k, defaults.get(k, 1.0)) for k in ["pts","reb","ast","stl","blk","tov"]}

    if tier == "track_meet" and abs(spread or 0) > blow_thr:
        w["pts"] *= gs.get("blowout_pts_penalty", 0.90)
        w["ast"] *= gs.get("blowout_ast_penalty", 0.90)
        w["reb"] *= gs.get("blowout_reb_penalty", 0.94)

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
    """Human-readable game script tier for display. Reads ceilings from runtime config."""
    gs = _cfg("game_script", _CONFIG_DEFAULTS["game_script"])
    t  = total or DEFAULT_TOTAL
    if t < gs.get("defensive_grind_ceiling", 220): return "Defensive Grind"
    if t <= gs.get("balanced_ceiling", 235):        return "Balanced Pace"
    if t <= gs.get("fast_paced_ceiling", 245):      return "Fast-Paced"
    return "Track Meet"


def project_player(pinfo, stats, spread, total, side, team_abbr="",
                   cascade_bonus=0.0, is_b2b=False):
    if pinfo.get("is_out"): return None
    # Skip day-to-day and doubtful players — high scratch risk
    if pinfo.get("injury_status") in ("DTD", "DOUBT"): return None
    avg_min = stats.get("min", 0)
    if avg_min <= 0: return None

    # Apply cascade minute boost
    proj_min = avg_min + cascade_bonus

    # Back-to-back penalty: teams on 2nd night of B2B see reduced minutes
    # and rest-managed players (older, injury-prone) often sit entirely.
    # Penalize projected minutes by 12% on B2B nights.
    if is_b2b:
        proj_min *= 0.88

    # Minutes gate: must project to at least 15 minutes
    min_gate = _cfg("projection.min_gate_minutes", MIN_GATE)
    if proj_min < min_gate: return None

    pts = stats["pts"]
    reb = stats["reb"]
    ast = stats["ast"]
    stl = stats.get("stl", 0)
    blk = stats.get("blk", 0)
    tov = stats.get("tov", 0)
    if pts + reb + ast <= 0: return None

    # Full DFS scoring formula (not just pts+reb+ast)
    heuristic = _dfs_score(pts, reb, ast, stl, blk, tov)

    # Clutch potential — playmakers, scorers, and defenders are more likely
    # to produce clutch, momentum, and streak events in the Real Score algorithm.
    # Pure rebounders (Drummond, Robinson, Jordan) inflate DFS base cheaply
    # but rarely make game-changing plays. Ball handlers with high PTS/MIN,
    # AST/MIN, and defenders with high STL+BLK/MIN generate Real Score events.
    #
    # Backtest: Marcus Smart 10/3/7/4stl/2blk = 3.4 Real Score despite only
    # 10 pts — defensive plays in close games are high-impact Real Score events.
    pts_per_min = pts / max(avg_min, 1)
    ast_per_min = ast / max(avg_min, 1)
    def_per_min = (stl + blk) / max(avg_min, 1)
    clutch_potential = 1.0 + min(
        pts_per_min * 0.10 + ast_per_min * 0.18 + def_per_min * 0.25, 0.40
    )
    heuristic *= clutch_potential

    # Scale heuristic by minute boost from cascade (capped at 1.25x)
    if cascade_bonus > 0 and avg_min > 0:
        min_scale = min(proj_min / avg_min, 1.25)
        heuristic *= min_scale

    # Declining usage penalty: if recent minutes dropped >10% vs season,
    # scale output proportionally (e.g. Conley post-trade: 26→19 min = 0.73x)
    season_min = stats.get("season_min", avg_min)
    recent_min = stats.get("recent_min", avg_min)
    decline_factor = 1.0
    if season_min > 0 and recent_min < season_min * 0.90:
        decline_factor = recent_min / season_min
        heuristic *= decline_factor

    # AI blend — use spread-derived opponent quality instead of hardcoded 112.0
    base = heuristic
    if AI_MODEL is not None:
        try:
            usage = min(max(pts / max(avg_min, 1) * 0.8, 0.9), 1.5)
            # Derive opponent defensive quality from spread and side
            # Negative spread = home favored → away faces tougher D
            sign = 1.0 if side == "away" else -1.0
            opp_def_rating = 112.0 + sign * (spread or 0) * 0.7
            features = np.array([[avg_min, stats.get("season_pts", pts), usage, opp_def_rating]])
            ai_pred = AI_MODEL.predict(features)[0]
            ai_norm = ai_pred * (heuristic / max(ai_pred, 1))
            base = (ai_norm * 0.7) + (heuristic * 0.3)
        except: pass

    # Contextual multipliers — game closeness matters BUT differently by role.
    pace_adj   = 1.0 + (0.06 * ((total or DEFAULT_TOTAL) - DEFAULT_TOTAL) / 20)

    # Spread adjustment — role-aware.
    # March 3 lesson: PHI got blown out 131-91 yet 3 of the top 8 highest-value
    # players were PHI bench guys (Raynaud +4.2, Yabusele +3.4, McCain +3.0).
    # Blowouts HURT stars (pulled early) but HELP bench (garbage-time minutes).
    #
    # Strategy: bench/role players (low PPG, low minutes) get a BOOST in
    # blowouts because they inherit extended garbage-time run. Stars still
    # get penalized because they sit Q4 in blowouts.
    abs_spread = abs(spread or 0)
    is_bench = pts <= 12 and avg_min <= 26  # role player / bench threshold
    if is_bench:
        # Bench players: blowouts = more minutes, neutral-to-positive
        if abs_spread <= 2:
            spread_adj = 1.05   # Close games — bench sits, slight penalty vs stars
        elif abs_spread <= 6:
            spread_adj = 1.0    # Moderate — neutral
        elif abs_spread <= 10:
            spread_adj = 1.08   # Lopsided — bench gets run, slight boost
        else:
            spread_adj = 1.12   # Blowout — extended garbage time, big boost
    else:
        # Stars/starters: close games are best, blowouts crush minutes
        if abs_spread <= 2:
            spread_adj = 1.15   # Pick'em — maximum Real Score environment
        elif abs_spread <= 4:
            spread_adj = 1.08   # Tight game — very good
        elif abs_spread <= 6:
            spread_adj = 1.0    # Moderate — neutral
        elif abs_spread <= 8:
            spread_adj = 0.88   # Lopsided — penalized
        else:
            spread_adj = 0.72   # Projected blowout — stars sit Q4
    home_adj   = 1.02 if side == "home" else 1.0

    s_base = base * pace_adj * spread_adj * home_adj

    # ── Real Score Engine ─────────────────────────────────────────────
    # Apply Monte Carlo-derived coefficients: Closeness, Clutch, Momentum
    # This replaces the linear volume-based approach with context-aware
    # Real Score projection aligned to the Real Sports App algorithm.
    season_pts = stats.get("season_pts", pts)
    recent_pts = stats.get("recent_pts", pts)
    player_variance = abs(recent_pts - season_pts) / max(season_pts, 1)
    usage_rate = min(max(pts / max(avg_min, 1) * 0.8, 0.5), 2.0)

    rng = _make_rng(spread or 0, total or DEFAULT_TOTAL)
    real_result, real_meta = real_score_projection(
        s_base, spread or 0, total or DEFAULT_TOTAL, usage_rate, player_variance, rng
    )

    # Raw projected score — compressed via power function to match
    # actual Real Score gaps (~1.5x star vs role, not ~3x linear).
    # Power of 0.75 compresses 23→11.2, 8→4.8 (ratio 2.3x vs 2.9x linear)
    raw_linear = real_result / 5.0
    raw_score = min(raw_linear ** 0.75, 15.0)

    # Estimated card boost (ADDITIVE, not multiplicative)
    # Real Sports formula: Value = Real Score × (Slot_Mult + Card_Boost)
    # Card boost is INVERSELY proportional to ownership — the app rewards
    # contrarian picks. Stars get crushed, obscure role players get huge boosts.
    card_boost = _est_card_boost(proj_min, pts, team_abbr)

    # EV score — card-adjusted expected value using additive formula
    # Use average slot (1.6) for ranking; MILP uses exact slots
    avg_slot = 1.6  # simple avg of [2.0, 1.8, 1.6, 1.4, 1.2]
    chalk_ev  = round(raw_score * (avg_slot + card_boost), 2)

    return {
        "id":      pinfo["id"],
        "name":    pinfo["name"],
        "pos":     pinfo["pos"],
        "team":    team_abbr,
        "rating":  round(raw_score, 1),
        "chalk_ev":chalk_ev,
        "predMin": round(proj_min, 1),
        "pts":     round(pts, 1),
        "reb":     round(reb, 1),
        "ast":     round(ast, 1),
        "stl":     round(stl, 1),
        "blk":     round(blk, 1),
        "tov":     round(tov, 1),
        "est_mult": card_boost,
        "slot":    "1.0x",
        "_decline": round(decline_factor, 2),
        "_cascade_bonus": round(cascade_bonus, 1),
        # Recent vs season stats — used by line engine for trend detection
        "season_pts": round(stats.get("season_pts", pts), 1),
        "recent_pts": round(stats.get("recent_pts", pts), 1),
        "season_stl": round(stats.get("season_stl", stl), 1),
        "recent_stl": round(stats.get("recent_stl", stl), 1),
        "season_blk": round(stats.get("season_blk", blk), 1),
        "recent_blk": round(stats.get("recent_blk", blk), 1),
        "injury_status": pinfo.get("injury_status", ""),
    }

def _run_game(game):
    cache_key = f"game_proj_{game['gameId']}"
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
        # Check if this player's team is on a back-to-back
        b2b = game.get("home_b2b") if sd == "home" else game.get("away_b2b")
        proj = project_player(p, stats, game["spread"], game["total"], sd, ab,
                              cascade_bonus=cascade_bonus, is_b2b=bool(b2b))
        if proj:
            out.append(proj)
    _cs(cache_key, out)
    return out

CHALK_FLOOR = 2.8  # Minimum raw rating for Starting 5

def _build_lineups(projections):
    # STARTING 5: MILP-optimized for highest chalk_ev = rating × (avg_slot + card_boost)
    # No team limit — Real Sports has no per-team restriction.
    chalk_eligible = [p for p in projections if p["rating"] >= CHALK_FLOOR]
    chalk = optimize_lineup(chalk_eligible, n=5, sort_key="chalk_ev",
                            rating_key="rating", card_boost_key="est_mult",
                            max_per_team=0)

    # MOONSHOT: ranks 6-10 from the same chalk EV ranking.
    # NOT a separate contrarian algo — just the next-best 5 players.
    # This avoids picking DNP-risk players with high card boosts but zero production.
    # Players with real projected RS but lower ownership naturally have high chalk_ev
    # and will appear here (e.g. Jabari Walker: 4.7 RS × +3.0 boost = 21.6 EV).
    chalk_names = {p["name"] for p in chalk}
    moonshot_pool = [p for p in chalk_eligible if p["name"] not in chalk_names]
    upside = optimize_lineup(moonshot_pool, n=5, sort_key="chalk_ev",
                             rating_key="rating", card_boost_key="est_mult",
                             max_per_team=0)

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

GAME_CHALK_FLOOR = 3.5  # Starting 5 floor for single-game

def _apply_game_script(projections, game):
    """Re-score projections using game script weights. Returns new list (deep copies, no mutation)."""
    total  = game.get("total") or DEFAULT_TOTAL
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
        rp["game_script"] = _game_script_label(total)
        rescored.append(rp)
    return rescored


def _build_game_lineups(projections, game):
    """Build lineups for a single-game draft with team balance + game script.

    Starting 5: MILP-optimized with min 2 players per team.
    Moonshot: Next 5 players by the same chalk_ev ranking (no separate contrarian algo).
    """
    rescored = _apply_game_script(projections, game)
    chalk_eligible = [p for p in rescored if p["rating"] >= GAME_CHALK_FLOOR]

    chalk = optimize_lineup(chalk_eligible, n=5, min_per_team=2, sort_key="chalk_ev",
                            rating_key="rating", card_boost_key="est_mult")

    # MOONSHOT: next 5 by chalk_ev — same ranking, different players
    chalk_names = {p["name"] for p in chalk}
    moonshot_pool = [p for p in chalk_eligible if p["name"] not in chalk_names]
    upside = optimize_lineup(moonshot_pool, n=5, min_per_team=2, sort_key="chalk_ev",
                             rating_key="rating", card_boost_key="est_mult")

    return chalk, upside

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

@app.get("/api/games")
async def get_games():
    games = fetch_games()
    for g in games:
        st = g.get("startTime", "")
        g["locked"] = _is_locked(st) if st else False
        g["draftable"] = not _is_completed(st) if st else False
    return games

@app.get("/api/slate")
async def get_slate():
    games = fetch_games()
    if not games:
        return {"date": _et_date().isoformat(), "games": [], "lineups": {"chalk": [], "upside": []}, "locked": False}

    # Only show/project games that are still draftable (not yet past lock window)
    # ESPN returns all games for the day, including already-completed ones.
    # At 1 AM EST the completed games from the evening should be invisible;
    # the upcoming day's games should be shown instead.
    draftable_games = [g for g in games if not _is_completed(g.get("startTime", ""))]

    # All games for today's ET date are already completed — this shouldn't normally
    # happen since fetch_games() explicitly requests the ET date from ESPN, but
    # handle it gracefully if ESPN returns stale data.
    if not draftable_games:
        # Find the earliest today's game start time to tell the user when to come back
        earliest_today = min((g["startTime"] for g in games if g.get("startTime")), default=None)
        available_msg = "later today"
        if earliest_today:
            try:
                gs = datetime.fromisoformat(earliest_today.replace("Z", "+00:00"))
                et_offset = timedelta(hours=-4 if 3 < gs.month < 11 else -5)
                gs_et = gs + et_offset
                available_msg = gs_et.strftime("%-I:%M %p ET")
            except: pass
        return {
            "date": _et_date().isoformat(),
            "games": games,
            "lineups": {"chalk": [], "upside": []},
            "locked": False,
            "no_games_yet": True,
            "draftable_count": 0,
            "available_after": available_msg,
        }

    # Lock is based on earliest DRAFTABLE game — completed games don't count
    start_times = [g["startTime"] for g in draftable_games if g.get("startTime")]
    earliest = min(start_times) if start_times else None
    locked = _is_locked(earliest) if earliest else False

    if locked:
        lock_cached = _lg("slate_v5_locked")
        if lock_cached:
            lock_cached["locked"] = True
            return lock_cached
        # Check regular cache and promote to lock cache
        cached = _cg("slate_v5")
        if cached:
            cached["locked"] = True
            _ls("slate_v5_locked", cached)
            return cached
        # No cache on this instance after lock — return empty rather than recomputing
        return {"date": _et_date().isoformat(), "games": games,
                "lineups": {"chalk": [], "upside": []}, "locked": True}

    cached = _cg("slate_v5")
    if cached:
        # Discard cached result if it has empty lineups but we have draftable games.
        # This clears stale cache written before the roster-fix was deployed.
        has_players = cached.get("lineups", {}).get("chalk") or cached.get("lineups", {}).get("upside")
        if has_players or not draftable_games:
            cached["locked"] = locked
            return cached

    all_proj = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed({pool.submit(_run_game, g): g for g in draftable_games}):
            try: all_proj.extend(fut.result())
            except Exception as e: print(f"slate err: {e}")
    chalk, upside = _build_lineups(all_proj)
    result = {"date": _et_date().isoformat(), "games": games,
              "lineups": {"chalk": chalk, "upside": upside}, "locked": locked,
              "draftable_count": len(draftable_games)}
    if chalk or upside:  # Don't cache empty results — allow retry on next request
        _cs("slate_v5", result)
    if locked:
        _ls("slate_v5_locked", result)
    return result

@app.get("/api/picks")
async def get_picks(gameId: str = Query(...)):
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
        # Check regular cache and promote to lock cache
        reg_key = f"picks_{gameId}"
        reg_cached = _cg(reg_key)
        if reg_cached:
            reg_cached["locked"] = True
            _ls(lock_key, reg_cached)
            return reg_cached
        # No cache on this instance after lock — return locked empty
        return {"date": _et_date().isoformat(), "game": game,
                "gameScript": None,
                "lineups": {"chalk": [], "upside": []},
                "locked": True, "injuries": []}

    projections = _run_game(game)
    if not projections:
        return JSONResponse({"error": "No projections available."}, status_code=503)
    chalk, upside = _build_game_lineups(projections, game)
    script = _game_script_label(game.get("total"))
    injuries = _get_injuries(game)

    result = {"date": _et_date().isoformat(), "game": game,
              "gameScript": script,
              "lineups": {"chalk": chalk, "upside": upside},
              "locked": locked,
              "injuries": injuries}
    # Cache picks so they survive as lock snapshot if slate locks later
    _cs(f"picks_{gameId}", result)
    return result

@app.post("/api/save-predictions")
async def save_predictions():
    """Save current predictions to GitHub as CSV."""
    today = _et_date().isoformat()
    path = f"data/predictions/{today}.csv"

    # Gather slate predictions
    rows = []
    cached_slate = _cg("slate_v5")
    if cached_slate and cached_slate.get("lineups"):
        rows.extend(_predictions_to_csv(cached_slate["lineups"], "slate"))

    # Gather per-game predictions from cache
    games = fetch_games()
    for g in games:
        gid = g["gameId"]
        cached_picks = _cg(f"picks_{gid}")
        if cached_picks and cached_picks.get("lineups"):
            rows.extend(_predictions_to_csv(cached_picks["lineups"], g.get("label", f"game_{gid}")))

    if not rows:
        return JSONResponse({"error": "No predictions cached yet"}, status_code=404)

    csv_content = CSV_HEADER + "\n" + "\n".join(rows) + "\n"
    result = _github_write_file(path, csv_content, f"predictions for {today}")
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=500)
    return {"status": "saved", "path": path, "rows": len(rows)}


@app.post("/api/parse-screenshot")
async def parse_screenshot(file: UploadFile = File(...)):
    """Parse a Real Sports app screenshot using Claude Vision API."""
    if not ANTHROPIC_API_KEY:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not configured"}, status_code=500)

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        return JSONResponse({"error": "Image too large (max 10MB)"}, status_code=413)
    b64_image = base64.b64encode(image_bytes).decode("ascii")

    # Determine media type
    ct = file.content_type or "image/png"
    if ct not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        ct = "image/png"

    prompt = """Extract ALL player data from this Real Sports app screenshot.

The screenshot may contain two sections:
1. "My draft" - the user's own drafted players
2. "Highest value" - the top performing players of the day

For EACH player, extract:
- player_name: full name
- actual_rs: the Real Score number shown with a triangle/arrow symbol (e.g. if it shows "⌃3.1" the value is 3.1, if "⌃0" the value is 0)
- actual_card_boost: the card boost shown as "+X.Xx" (e.g. "+3.0x" → 3.0, "+0.9x" → 0.9). If no + symbol is shown, set to null
- drafts: number of drafts (e.g. "31", "1.1k" → 1100, "5k" → 5000, "13.6k" → 13600)
- avg_finish: average finish position as a number (e.g. "1st" → 1, "5th" → 5). Only in "My draft" section
- total_value: the value number shown on right side (only in "Highest value" section, e.g. "⌃23.6" → 23.6)
- source: "my_draft" if from "My draft" section, "highest_value" if from "Highest value" section

If this is a Leaderboard screenshot, extract each drafter's lineup:
- For each player in a lineup: player_name, actual_rs (the number shown), card_multiplier (e.g. "4.2x" → 4.2)
- source: "leaderboard"
- Include the drafter's total_score

Return ONLY a JSON array of objects. No markdown, no explanation."""

    text = ""
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 4096,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": ct, "data": b64_image}},
                        {"type": "text", "text": prompt},
                    ]
                }]
            },
            timeout=30,
        )
        r.raise_for_status()
        resp = r.json()
        text = resp["content"][0]["text"]
        # Extract JSON from response (may be wrapped in ```json blocks)
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        parsed = json.loads(text.strip())
        return {"players": parsed}
    except json.JSONDecodeError:
        return JSONResponse({"error": "Failed to parse Claude response as JSON", "raw": text[:500]}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": f"Screenshot parsing failed: {str(e)}"}, status_code=500)


@app.post("/api/save-actuals")
async def save_actuals(payload: dict = Body(...)):
    """Save confirmed actuals to GitHub as CSV."""
    date_str = payload.get("date", _et_date().isoformat())
    players = payload.get("players", [])
    if not players:
        return JSONResponse({"error": "No player data"}, status_code=400)

    path = f"data/actuals/{date_str}.csv"
    header = "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source"

    # Check if file already exists (to append)
    existing, _ = _github_get_file(path)
    rows = []
    if existing:
        # Keep existing rows (skip header)
        lines = existing.strip().split("\n")
        rows = lines[1:] if len(lines) > 1 else []

    # Add new rows
    for p in players:
        rows.append(",".join(_csv_escape(p.get(k, "")) for k in [
            "player_name", "actual_rs", "actual_card_boost",
            "drafts", "avg_finish", "total_value", "source"
        ]))

    csv_content = header + "\n" + "\n".join(rows) + "\n"
    result = _github_write_file(path, csv_content, f"actuals for {date_str}")
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=500)
    return {"status": "saved", "path": path, "rows": len(rows)}


def _parse_csv(content, header_fields):
    """Parse a simple CSV string into list of dicts using given header fields."""
    rows = []
    if not content:
        return rows
    lines = content.strip().split("\n")
    # Skip header line
    for line in lines[1:]:
        if not line.strip():
            continue
        # Simple CSV split (handles quoted fields)
        fields = []
        current = ""
        in_quotes = False
        for ch in line:
            if ch == '"':
                in_quotes = not in_quotes
            elif ch == ',' and not in_quotes:
                fields.append(current)
                current = ""
            else:
                current += ch
        fields.append(current)
        # Pad to expected length
        while len(fields) < len(header_fields):
            fields.append("")
        rows.append(dict(zip(header_fields, fields[:len(header_fields)])))
    return rows


PRED_FIELDS = ["scope","lineup_type","slot","player_name","player_id","team","pos",
               "predicted_rs","est_card_boost","pred_min","pts","reb","ast","stl","blk"]
ACT_FIELDS = ["player_name","actual_rs","actual_card_boost","drafts","avg_finish","total_value","source"]


@app.get("/api/log/dates")
async def log_dates():
    """Return sorted list of dates that have stored prediction or actual data."""
    dates = set()
    for prefix in ["data/predictions", "data/actuals"]:
        items = _github_list_dir(prefix)
        for item in items:
            name = item.get("name", "")
            if name.endswith(".csv"):
                dates.add(name[:-4])
    return sorted(dates, reverse=True)


@app.get("/api/log/get")
async def log_get(date: str = Query(None)):
    """Get stored predictions and actuals for a date."""
    date_str = date or _et_date().isoformat()

    pred_csv, _ = _github_get_file(f"data/predictions/{date_str}.csv")
    act_csv, _ = _github_get_file(f"data/actuals/{date_str}.csv")

    predictions = _parse_csv(pred_csv, PRED_FIELDS) if pred_csv else []
    actuals = _parse_csv(act_csv, ACT_FIELDS) if act_csv else []

    # Group predictions by scope → lineup_type → players
    scopes = {}
    for row in predictions:
        scope = row.get("scope", "")
        lt = row.get("lineup_type", "chalk")
        scopes.setdefault(scope, {"chalk": [], "upside": []})[lt].append({
            "slot": row.get("slot", ""),
            "name": row.get("player_name", ""),
            "team": row.get("team", ""),
            "pos": row.get("pos", ""),
            "rating": row.get("predicted_rs", ""),
            "est_mult": row.get("est_card_boost", ""),
            "predMin": row.get("pred_min", ""),
        })

    return {
        "date": date_str,
        "has_predictions": bool(predictions),
        "has_actuals": bool(actuals),
        "scopes": scopes,
        "actuals": actuals,
    }


@app.post("/api/hindsight")
async def hindsight(payload: dict = Body(...)):
    """Given actual player RS scores, return the optimal hindsight lineup."""
    players = payload.get("players", [])
    if not players:
        return JSONResponse({"error": "No players provided"}, status_code=400)

    projections = []
    for p in players:
        rs = _safe_float(p.get("actual_rs"), 0)
        boost = _safe_float(p.get("actual_card_boost"), 0.3)
        if rs <= 0:
            continue
        avg_slot = 1.6
        projections.append({
            "name": p.get("name", p.get("player_name", "")),
            "rating": rs,
            "est_mult": boost,
            "chalk_ev": rs * (avg_slot + boost),
            "team": p.get("team", ""),
            "pos": p.get("pos", ""),
            "slot": "",
        })

    if not projections:
        return JSONResponse({"error": "No players with valid RS scores"}, status_code=400)

    lineup = optimize_lineup(projections, n=min(5, len(projections)),
                             sort_key="chalk_ev", rating_key="rating",
                             card_boost_key="est_mult")
    return {"lineup": lineup}


@app.get("/api/refresh")
async def refresh():
    cleared = 0
    try:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink(); cleared += 1
    except Exception as e:
        return {"status": "error", "message": str(e)}
    # Also clear config cache on refresh
    try:
        cfg_cache = CONFIG_CACHE_DIR / "model_config.json"
        if cfg_cache.exists():
            cfg_cache.unlink()
    except: pass
    return {"status": "ok", "cleared": cleared, "ts": datetime.now().isoformat()}


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 2: LINE OF THE DAY
# ═════════════════════════════════════════════════════════════════════════════

LINE_CSV_HEADER = "date,player_name,player_id,team,opponent,stat_type,line,direction,projection,edge,confidence,narrative,result,actual_stat"

@app.get("/api/line-of-the-day")
async def get_line_of_the_day():
    """Detect today's best player prop edge and return it with confidence score."""
    if not ODDS_API_KEY:
        return JSONResponse({"pick": None, "error": "no_api_key"}, status_code=200)

    games = fetch_games()
    draftable = [g for g in games if not _is_completed(g.get("startTime", ""))]
    if not draftable:
        return JSONResponse({"pick": None, "error": "no_games"}, status_code=200)

    # Gather all projections via the same pipeline used by /api/slate
    all_proj = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fut in as_completed({pool.submit(_run_game, g): g for g in draftable}):
            try: all_proj.extend(fut.result())
            except Exception as e: print(f"line proj err: {e}")

    if not all_proj:
        return JSONResponse({"pick": None, "error": "no_projections"}, status_code=200)

    result = run_line_engine(all_proj, draftable)
    return result


LINE_FIELDS = LINE_CSV_HEADER.split(",")

@app.post("/api/save-line")
async def save_line(payload: dict = Body(...)):
    """Save today's Line of the Day pick to data/lines/{date}.csv. Saves once per day."""
    today = _et_date().isoformat()
    path = f"data/lines/{today}.csv"

    # Check if already saved today
    existing, _ = _github_get_file(path)
    if existing:
        return {"status": "already_saved", "path": path}

    pick = payload.get("pick")
    if not pick:
        return JSONResponse({"error": "No pick provided"}, status_code=400)

    row = ",".join(_csv_escape(str(pick.get(k, ""))) for k in [
        "player_name", "player_id", "team", "opponent", "stat_type",
        "line", "direction", "projection", "edge", "confidence", "narrative",
    ])
    row = f"{today}," + row + ",pending,"  # result=pending, actual_stat=empty

    csv_content = LINE_CSV_HEADER + "\n" + row + "\n"
    result = _github_write_file(path, csv_content, f"line pick for {today}")
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=500)
    return {"status": "saved", "path": path}


@app.post("/api/resolve-line")
async def resolve_line(payload: dict = Body(...)):
    """Mark today's line pick as hit or miss given the actual stat."""
    date_str = payload.get("date", _et_date().isoformat())
    actual   = payload.get("actual_stat")
    if actual is None:
        return JSONResponse({"error": "actual_stat required"}, status_code=400)

    path = f"data/lines/{date_str}.csv"
    existing, sha = _github_get_file(path)
    if not existing:
        return JSONResponse({"error": "No line pick found for this date"}, status_code=404)

    lines = existing.strip().split("\n")
    if len(lines) < 2:
        return JSONResponse({"error": "Empty line file"}, status_code=400)

    # Parse the pick row
    rows = _parse_csv(existing, LINE_FIELDS)
    if not rows:
        return JSONResponse({"error": "Could not parse line file"}, status_code=400)

    row = rows[0]
    direction = row.get("direction", "over")
    line_val  = _safe_float(row.get("line", 0))
    actual_f  = _safe_float(actual)
    if direction == "over":
        result = "hit" if actual_f > line_val else "miss"
    else:
        result = "hit" if actual_f < line_val else "miss"

    # Rewrite CSV with result filled in
    fields_out = list(LINE_FIELDS)
    updated_row = dict(row)
    updated_row["result"]      = result
    updated_row["actual_stat"] = str(actual_f)

    new_row = ",".join(_csv_escape(str(updated_row.get(k, ""))) for k in fields_out)
    csv_content = LINE_CSV_HEADER + "\n" + new_row + "\n"
    _github_write_file(path, csv_content, f"line result for {date_str}: {result}")
    return {"status": "resolved", "result": result, "actual": actual_f}


@app.get("/api/line-history")
async def line_history():
    """Return recent Line of the Day picks with results."""
    items = _github_list_dir("data/lines")
    results = []
    for item in sorted(items, key=lambda x: x.get("name", ""), reverse=True)[:30]:
        name = item.get("name", "")
        if not name.endswith(".csv"):
            continue
        content, _ = _github_get_file(f"data/lines/{name}")
        if not content:
            continue
        rows = _parse_csv(content, LINE_FIELDS)
        if rows:
            results.append(rows[0])

    # Compute streak + hit rate
    hits   = [r for r in results if r.get("result") == "hit"]
    misses = [r for r in results if r.get("result") == "miss"]
    total_resolved = len(hits) + len(misses)
    hit_rate = round(len(hits) / total_resolved * 100, 1) if total_resolved else None

    # Streak: consecutive same result from most recent
    streak = 0
    streak_type = None
    for r in results:
        res = r.get("result", "pending")
        if res == "pending":
            continue
        if streak_type is None:
            streak_type = res
            streak = 1
        elif res == streak_type:
            streak += 1
        else:
            break

    return {
        "picks":        results,
        "hit_rate":     hit_rate,
        "total_picks":  len(results),
        "resolved":     total_resolved,
        "streak":       streak,
        "streak_type":  streak_type,
    }


# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: LAB
# ═════════════════════════════════════════════════════════════════════════════

_GAMES_FINAL_CACHE: dict = {"result": None, "ts": 0.0}

def _all_games_final(games):
    """Check ESPN scoreboard to see if all today's games are completed. Cached 3 min."""
    if not games:
        return True, 0, 0
    now_ts = datetime.now(timezone.utc).timestamp()
    if _GAMES_FINAL_CACHE["result"] is not None and now_ts - _GAMES_FINAL_CACHE["ts"] < 180:
        return tuple(_GAMES_FINAL_CACHE["result"])
    today_str = _et_date().strftime("%Y%m%d")
    data = _espn_get(f"{ESPN}/scoreboard?dates={today_str}")
    finals = 0
    remaining = 0
    for ev in data.get("events", []):
        status = ev.get("status", {}).get("type", {}).get("completed", False)
        if status:
            finals += 1
        else:
            remaining += 1
    all_final = remaining == 0 and finals > 0
    result = (all_final, remaining, finals)
    _GAMES_FINAL_CACHE["result"] = list(result)
    _GAMES_FINAL_CACHE["ts"] = now_ts
    return result


@app.get("/api/lab/status")
async def lab_status():
    """Return Lab lock status based on slate state and game completion."""
    games = fetch_games()
    draftable = [g for g in games if not _is_completed(g.get("startTime", ""))]

    # If games are currently in progress (slate locked, games not yet final)
    start_times = [g["startTime"] for g in games if g.get("startTime")]
    earliest = min(start_times) if start_times else None
    slate_locked = _is_locked(earliest) if earliest else False

    all_final, remaining, finals = _all_games_final(games)

    cfg = _load_config()
    cfg_version = cfg.get("version", 1)
    last_change = (cfg.get("changelog") or [{}])[-1]

    if all_final or not games:
        # Unlocked: all games done or no games today
        next_lock = None
        if draftable:
            lock_buf = _cfg("projection.lock_buffer_minutes", 5)
            ns = min(g["startTime"] for g in draftable if g.get("startTime"))
            gs = datetime.fromisoformat(ns.replace("Z", "+00:00"))
            next_lock = (gs - timedelta(minutes=lock_buf)).isoformat()
        return {
            "locked": False,
            "reason": "All games final" if games else "No games today",
            "current_config_version": cfg_version,
            "games_remaining": 0,
            "games_final": finals,
            "next_lock_time": next_lock,
        }
    elif slate_locked:
        # Estimate unlock: assume ~2.5h per game from earliest start
        est_unlock = None
        if earliest:
            try:
                gs = datetime.fromisoformat(earliest.replace("Z", "+00:00"))
                est_unlock = (gs + timedelta(hours=2, minutes=30)).isoformat()
            except: pass
        return {
            "locked": True,
            "reason": f"Slate in progress — {remaining} game{'s' if remaining != 1 else ''} remaining",
            "current_config_version": cfg_version,
            "games_remaining": remaining,
            "games_final": finals,
            "estimated_unlock": est_unlock,
        }
    else:
        # Pre-slate — lab is open
        return {
            "locked": False,
            "reason": "Pre-slate window",
            "current_config_version": cfg_version,
            "games_remaining": 0,
            "games_final": 0,
            "next_lock_time": None,
        }


@app.get("/api/lab/briefing")
async def lab_briefing():
    """Analyze recent prediction accuracy and return structured briefing for the Lab."""
    pred_items  = _github_list_dir("data/predictions")
    act_items   = _github_list_dir("data/actuals")
    act_dates   = {i["name"].replace(".csv","") for i in act_items if i["name"].endswith(".csv")}

    # Find dates with both predictions and actuals
    paired = []
    for item in sorted(pred_items, key=lambda x: x.get("name",""), reverse=True):
        name = item.get("name","")
        if not name.endswith(".csv"): continue
        d = name[:-4]
        if d in act_dates:
            paired.append(d)

    # Use most recent paired date for latest-slate analysis
    latest_slate = None
    rolling_errors = []

    for d in paired[:10]:  # Max 10 slates
        pred_csv, _ = _github_get_file(f"data/predictions/{d}.csv")
        act_csv, _  = _github_get_file(f"data/actuals/{d}.csv")
        if not pred_csv or not act_csv: continue

        preds   = _parse_csv(pred_csv, PRED_FIELDS)
        actuals = _parse_csv(act_csv, ACT_FIELDS)

        act_map = {r["player_name"].lower(): _safe_float(r.get("actual_rs")) for r in actuals}

        slate_errors = []
        misses = []
        for row in preds:
            pname = row.get("player_name","").lower()
            pred_rs = _safe_float(row.get("predicted_rs"))
            if pname in act_map and pred_rs > 0:
                actual_rs = act_map[pname]
                err = abs(pred_rs - actual_rs)
                slate_errors.append(err)
                rolling_errors.append(err)
                misses.append({"player": row["player_name"], "predicted_rs": pred_rs,
                               "actual_rs": actual_rs, "error": round(actual_rs - pred_rs, 2)})

        if not slate_errors: continue

        mae = round(sum(slate_errors) / len(slate_errors), 2)
        misses.sort(key=lambda x: abs(x["error"]), reverse=True)

        if latest_slate is None:
            latest_slate = {
                "date": d,
                "players_predicted": len(preds),
                "players_with_actuals": len(slate_errors),
                "mean_absolute_error": mae,
                "biggest_misses": misses[:5],
            }

    overall_mae = round(sum(rolling_errors) / len(rolling_errors), 2) if rolling_errors else None
    cfg = _load_config()

    # Simple pattern detection: check if errors correlate with game script tier
    patterns = []
    if latest_slate and latest_slate["mean_absolute_error"] > 2.5:
        patterns.append({
            "type": "high_error",
            "description": f"MAE {latest_slate['mean_absolute_error']} above 2.5 threshold — model may be over-projecting",
            "slates_observed": 1,
        })

    return {
        "latest_slate":    latest_slate,
        "rolling_accuracy": {
            "slates_with_data": len(paired),
            "overall_mae": overall_mae,
        },
        "patterns":     patterns,
        "current_config": {
            "version":         cfg.get("version", 1),
            "last_change":     (cfg.get("changelog") or [{}])[-1].get("change",""),
            "last_change_date": (cfg.get("changelog") or [{}])[-1].get("date",""),
        },
    }


def _deep_set(d, keys, value):
    """Set a nested dict value via a list of keys."""
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


@app.post("/api/lab/update-config")
async def lab_update_config(payload: dict = Body(...)):
    """Apply parameter changes to the runtime config. Increments version + appends changelog."""
    changes     = payload.get("changes", {})
    description = payload.get("change_description", "Lab update")
    if not changes:
        return JSONResponse({"error": "No changes provided"}, status_code=400)

    cfg = _load_config()
    old_version = cfg.get("version", 1)
    new_version = old_version + 1

    for dot_path, value in changes.items():
        keys = dot_path.split(".")
        _deep_set(cfg, keys, value)

    cfg["version"]    = new_version
    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    cfg["updated_by"] = "lab"

    changelog = cfg.get("changelog", [])
    changelog.append({
        "version": new_version,
        "date":    _et_date().isoformat(),
        "change":  description,
    })
    cfg["changelog"] = changelog

    content = json.dumps(cfg, indent=2)
    result  = _github_write_file("data/model-config.json", content, f"Lab config v{new_version}: {description}")
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=500)

    # Clear config cache so new values take effect immediately
    try:
        (CONFIG_CACHE_DIR / "model_config.json").unlink(missing_ok=True)
    except: pass

    return {"status": "applied", "version": new_version, "changes": changes}


@app.get("/api/lab/config-history")
async def lab_config_history():
    """Return current config + full changelog."""
    cfg = _load_config()
    return {
        "version":   cfg.get("version", 1),
        "config":    cfg,
        "changelog": cfg.get("changelog", []),
    }


@app.post("/api/lab/rollback")
async def lab_rollback(payload: dict = Body(...)):
    """Revert to a previous config version (creates new version with old values)."""
    target = payload.get("target_version")
    if target is None:
        return JSONResponse({"error": "target_version required"}, status_code=400)

    cfg = _load_config()
    changelog = cfg.get("changelog", [])
    # Find the config state at target version — we can reconstruct from the current file
    # (Full git history rollback is complex; for now we just restore defaults and note in changelog)
    # In practice: the Lab chat explains what changed and the user re-applies manually if needed.
    current_version = cfg.get("version", 1)
    if int(target) >= current_version:
        return JSONResponse({"error": "Target must be earlier than current version"}, status_code=400)

    new_version = current_version + 1
    changelog.append({
        "version": new_version,
        "date":    _et_date().isoformat(),
        "change":  f"Rollback requested to v{target} — manual parameter review needed",
    })
    cfg["version"]    = new_version
    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    cfg["updated_by"] = "lab-rollback"
    cfg["changelog"]  = changelog

    content = json.dumps(cfg, indent=2)
    result  = _github_write_file("data/model-config.json", content, f"Lab rollback to v{target}")
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=500)

    try:
        (CONFIG_CACHE_DIR / "model_config.json").unlink(missing_ok=True)
    except: pass

    return {"status": "rollback_noted", "new_version": new_version,
            "note": "Parameters were not automatically reverted — review changelog and apply changes via /api/lab/update-config"}


@app.post("/api/lab/backtest")
async def lab_backtest(payload: dict = Body(...)):
    """Replay historical slates with proposed parameter changes and compare MAE."""
    proposed_changes = payload.get("proposed_changes", {})
    description      = payload.get("description", "Backtest")
    if not proposed_changes:
        return JSONResponse({"error": "proposed_changes required"}, status_code=400)

    pred_items = _github_list_dir("data/predictions")
    act_items  = _github_list_dir("data/actuals")
    act_dates  = {i["name"].replace(".csv","") for i in act_items if i["name"].endswith(".csv")}

    # Build proposed config
    current_cfg  = _load_config()
    proposed_cfg = json.loads(json.dumps(current_cfg))  # deep copy
    for dot_path, value in proposed_changes.items():
        keys = dot_path.split(".")
        _deep_set(proposed_cfg, keys, value)

    current_mae_all  = []
    proposed_mae_all = []
    per_slate        = []
    affected_sample  = []

    for item in sorted(pred_items, key=lambda x: x.get("name",""), reverse=True)[:10]:
        name = item.get("name","")
        if not name.endswith(".csv"): continue
        d = name[:-4]
        if d not in act_dates: continue

        pred_csv, _ = _github_get_file(f"data/predictions/{d}.csv")
        act_csv, _  = _github_get_file(f"data/actuals/{d}.csv")
        if not pred_csv or not act_csv: continue

        preds   = _parse_csv(pred_csv, PRED_FIELDS)
        actuals = _parse_csv(act_csv, ACT_FIELDS)
        act_map = {r["player_name"].lower(): _safe_float(r.get("actual_rs")) for r in actuals}

        cur_errs  = []
        prop_errs = []

        for row in preds:
            pname    = row.get("player_name","").lower()
            pred_rs  = _safe_float(row.get("predicted_rs"))
            pts      = _safe_float(row.get("pts"))
            reb      = _safe_float(row.get("reb"))
            ast      = _safe_float(row.get("ast"))
            stl      = _safe_float(row.get("stl"))
            blk      = _safe_float(row.get("blk"))

            if pname not in act_map or pred_rs <= 0: continue
            actual_rs = act_map[pname]

            cur_errs.append(abs(pred_rs - actual_rs))

            # Recalculate with proposed DFS weights if changed
            if "real_score.dfs_weights" in proposed_changes or any(
                k.startswith("real_score") for k in proposed_changes
            ):
                w = proposed_cfg.get("real_score",{}).get("dfs_weights",
                    {"pts":1.0,"reb":1.0,"ast":1.5,"stl":4.5,"blk":4.0,"tov":-1.2})
                new_dfs = (pts*w.get("pts",1.0) + reb*w.get("reb",1.0) +
                           ast*w.get("ast",1.5) + stl*w.get("stl",4.5) +
                           blk*w.get("blk",4.0))
                # Scale proposed RS proportionally
                if pred_rs > 0:
                    old_w = current_cfg.get("real_score",{}).get("dfs_weights",
                        {"pts":1.0,"reb":1.0,"ast":1.5,"stl":4.5,"blk":4.0,"tov":-1.2})
                    old_dfs = (pts*old_w.get("pts",1.0) + reb*old_w.get("reb",1.0) +
                               ast*old_w.get("ast",1.5) + stl*old_w.get("stl",4.5) +
                               blk*old_w.get("blk",4.0))
                    if old_dfs > 0:
                        proposed_rs = pred_rs * (new_dfs / old_dfs)
                    else:
                        proposed_rs = pred_rs
                else:
                    proposed_rs = pred_rs
            else:
                proposed_rs = pred_rs  # No change for non-scoring params

            prop_err = abs(proposed_rs - actual_rs)
            prop_errs.append(prop_err)

            if len(affected_sample) < 5:
                improved = prop_err < abs(pred_rs - actual_rs)
                affected_sample.append({
                    "player": row.get("player_name",""),
                    "date":   d,
                    "current_projection": round(pred_rs, 2),
                    "proposed_projection": round(proposed_rs, 2),
                    "actual": round(actual_rs, 2),
                    "improved": improved,
                })

        if cur_errs:
            cm = sum(cur_errs)  / len(cur_errs)
            pm = sum(prop_errs) / len(prop_errs)
            current_mae_all.extend(cur_errs)
            proposed_mae_all.extend(prop_errs)
            per_slate.append({"date": d, "current_mae": round(cm,2), "proposed_mae": round(pm,2)})

    if not current_mae_all:
        return {"slates_tested": 0, "error": "No historical slates with both predictions and actuals"}

    cur_mae  = round(sum(current_mae_all)  / len(current_mae_all), 3)
    prop_mae = round(sum(proposed_mae_all) / len(proposed_mae_all), 3)
    improvement = round((cur_mae - prop_mae) / max(cur_mae, 0.001) * 100, 1)

    rec = "Proposed change reduces MAE." if improvement > 2 else \
          "Marginal improvement — may not be worth applying." if improvement > 0 else \
          "Proposed change worsens MAE — do not apply."

    return {
        "slates_tested":         len(per_slate),
        "current_config_mae":    cur_mae,
        "proposed_config_mae":   prop_mae,
        "improvement_pct":       improvement,
        "per_slate_comparison":  per_slate,
        "affected_players_sample": affected_sample,
        "recommendation":        rec,
    }


@app.post("/api/lab/chat")
async def lab_chat(payload: dict = Body(...)):
    """Proxy to Anthropic Messages API with Lab system prompt. Keeps API key server-side."""
    if not ANTHROPIC_API_KEY:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not configured"}, status_code=500)

    messages = payload.get("messages", [])
    system   = payload.get("system", "")

    if not messages:
        return JSONResponse({"error": "No messages provided"}, status_code=400)

    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model":      "claude-sonnet-4-6",
                "max_tokens": 2048,
                "system":     system,
                "messages":   messages,
            },
            timeout=45,
        )
        r.raise_for_status()
        resp = r.json()
        return {
            "content": resp["content"][0]["text"],
            "usage":   resp.get("usage", {}),
        }
    except Exception as e:
        return JSONResponse({"error": f"Anthropic API error: {str(e)}"}, status_code=500)
