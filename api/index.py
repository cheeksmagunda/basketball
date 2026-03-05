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
    from api.asset_optimizer import optimize_lineup, contrarian_score
    from api.temporal_risk import (
        lineup_duplication_probability, optimal_lock_time, trav_adjusted_score,
    )
except ImportError:
    from .real_score import real_score_projection, _make_rng
    from .asset_optimizer import optimize_lineup, contrarian_score
    from .temporal_risk import (
        lineup_duplication_probability, optimal_lock_time, trav_adjusted_score,
    )

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
    """Returns True if current UTC time is within LOCK_BUFFER_MINUTES of game start.
    Returns False for completed games (>3h past start) to avoid stale ESPN data."""
    try:
        game_start = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        # Completed games (>3h past start) are stale, not locked
        if now > game_start + timedelta(hours=3):
            return False
        return now >= game_start - timedelta(minutes=LOCK_BUFFER_MINUTES)
    except:
        return False

def _is_completed(start_time_iso):
    """Returns True if the game has already passed its lock window (started or about to start).
    Completed/in-progress games should not appear in draft recommendations."""
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
    yesterday = (date.today() - timedelta(days=1)).strftime("%Y%m%d")
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
    c = _cg("games")
    if c: return c
    b2b_teams = _fetch_b2b_teams()
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
            "home_b2b": home["abbr"] in b2b_teams,
            "away_b2b": away["abbr"] in b2b_teams,
        })
    _cs("games", games)
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
            min_ratio = recent["min"] / max(season["min"], 1)
            if min_ratio < 0.75:
                # Major role change: 80% recent, 20% season
                min_blend = round(season["min"] * 0.2 + recent["min"] * 0.8, 2)
            elif min_ratio < 0.90:
                # Moderate decline: 65% recent, 35% season
                min_blend = round(season["min"] * 0.35 + recent["min"] * 0.65, 2)
            else:
                min_blend = round(season["min"] * 0.5 + recent["min"] * 0.5, 2)

            blended = {k: round(season[k] * 0.5 + recent[k] * 0.5, 2) for k in season}
            blended["min"] = min_blend  # Override minutes with smart blend
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

    # Cap per-player cascade at 3 minutes to prevent unrealistic projections
    for pid in cascade_flags:
        cascade_flags[pid] = min(cascade_flags[pid], 3.0)

    return cascade_flags


# ─────────────────────────────────────────────────────────────────────────────
# THE CORE MODEL — Optimized for the Real Sports App
#
# Total Draft Score = Σ (Real Score × Card Multiplier × Slot Multiplier)
#
# Card multipliers are the DOMINANT factor in draft scoring. Yesterday's
# winners had role players with 4.0-4.6x card multipliers (Legendary +
# booster) beating superstars who only had 1.0-2.3x (General/Common).
#
# ADDITIVE formula: Value = Real Score × (Slot_Mult + Card_Boost)
# Example from actual leaderboard:
#   Jaylin Williams: 4.1 base × (2.0 + 2.7) = 4.1 × 4.7 = 19.27  ← WINNER
#   Anthony Edwards: 6.2 base × (2.0 + 0.3) = 6.2 × 2.3 = 14.26  ← loses
#
# Card boost is INVERSELY driven by draft popularity (ownership).
# Stars get 7,000+ drafts → +0.3x boost. Role players get <50 drafts → +2.5-3.0x.
# The _est_card_boost uses a "hype score" (PPG², minutes, market) to predict this.
#
# Real Score-aligned formula (proxy for raw production):
#   PTS + REB + AST×1.5 + STL×4.5 + BLK×4.0 - TOV×1.2
#
# STARTING 5 = MILP optimizes Σ rating × (slot + card_boost)
# MOONSHOT = 5 different players — close-game ceiling × card advantage
# ─────────────────────────────────────────────────────────────────────────────

# Big-market / high-profile teams that casual drafters flock to.
# These players get MORE drafts → LOWER card boosts.
_BIG_MARKET_TEAMS = {
    "LAL", "GSW", "BOS", "NYK", "PHI", "MIL", "DAL", "PHX", "MIA", "DEN",
    "LAC", "CHI", "SA",  # Wemby effect
}


def _est_card_boost(proj_min, pts, team_abbr):
    """Estimate ADDITIVE card boost based on predicted draft popularity.

    Real Sports dynamically adjusts card boosts inversely to ownership:
    stars get massive draft counts → crushed boosts (+0.3),
    obscure role players get almost no drafts → huge boosts (+2.5-3.0+).

    Uses a "hype score" — how attractive the player is to casual drafters —
    and maps it through exponential decay to a card boost.

    Calibrated against March 3 + March 4 actuals:
      Wembanyama (36m, 24p, SA):   hype 9.5 → est +0.4x  (actual +0.3)
      Ant Edwards (37m, 26p):      hype 7.5 → est +0.5x  (actual +0.3)
      Bam (34m, 21p, MIA):         hype 7.0 → est +0.6x  (actual +0.7)
      Jrue Holiday (30m, 16p, BOS):hype 3.84→ est +1.0x  (actual +1.0) ✓
      Jaylin Williams (20m, 8p):   hype 0.5 → est +2.9x  (actual +2.7)
      Marcus Smart (25m, 10p):     hype 0.9 → est +2.6x  (actual +2.5)
      Oso Ighodaro (18m, 7p):      hype 0.4 → est +3.0x  (actual +3.0) ✓
      Walter Clayton Jr (28m,14p): hype 1.79→ est +2.1x  (actual +3.0) — under
      N. Clifford (12m, 4p):       hype 0.1 → est +3.0x  (winning draft ~3.2x)

    Cap is +3.0x (confirmed max in-game). Decay base tuned from 0.74→0.70
    to differentiate stars from mid-tier players more sharply.
    """
    # Hype score — PPG² makes scoring stars disproportionately popular
    hype = (pts / 10.0) ** 2 * (proj_min / 30.0) ** 0.5
    if team_abbr in _BIG_MARKET_TEAMS:
        hype *= 1.5
    # Exponential decay: high hype → low boost, low hype → high boost
    # Decay base 0.70 (was 0.74) sharpens drop-off for mid-tier popularity.
    # Cap 3.0 (was 3.5) matches confirmed max observed in-game.
    boost = 3.4 * (0.70 ** hype) + 0.3
    return round(min(max(boost, 0.2), 3.0), 1)

def _dfs_score(pts, reb, ast, stl, blk, tov):
    """Real Score-aligned formula — boosted defensive stats.

    Backtest insight: steals and blocks correlate more strongly with Real Score
    than with traditional DFS. Marcus Smart: 10/3/7/4stl/2blk = 3.4 Real Score
    despite only 10 pts, because defensive plays in close games are high-impact
    events in the Real Score algorithm (momentum shifts, clutch stops).
    """
    return pts + reb + (ast * 1.5) + (stl * 4.5) + (blk * 4.0) - (tov * 1.2)


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
    t = total or DEFAULT_TOTAL

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
        # NOTE: this only applies to per-game script (starters context).
        # The main project_player() handles role-aware spread separately.
        if abs(spread or 0) > 8:
            w["pts"] *= 0.90
            w["ast"] *= 0.90
            w["reb"] *= 0.94

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
    t = total or DEFAULT_TOTAL
    if t < 220:   return "Defensive Grind"
    if t <= 235:  return "Balanced Pace"
    if t <= 245:  return "Fast-Paced"
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
        "_base":   base,
        "_spread": spread,
        "_decline": round(decline_factor, 2),
        "_features": {"avg_min": round(avg_min, 1), "season_pts": round(stats.get("season_pts", pts), 1)},
        "_real_meta": real_meta,
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

CHALK_FLOOR    = 2.8  # Minimum raw rating for Starting 5 — lowered from 3.5
                      # to allow high-card bench players (Clifford 2.3 RS won)
MOONSHOT_FLOOR = 3.0  # Floor for Moonshot — lowered from 4.0

def _get_recent_picks(days=3):
    """Load player names from the last N days of history for anti-repetition."""
    recent_names = {}
    if not HISTORY_FILE.exists():
        return recent_names
    try:
        hist = json.loads(HISTORY_FILE.read_text())
        today = date.today()
        for entry in reversed(hist):
            ts = entry.get("timestamp", "")
            try:
                entry_date = datetime.fromisoformat(ts).date()
            except:
                continue
            days_ago = (today - entry_date).days
            if days_ago > days:
                break
            for p in entry.get("players", []):
                name = p.get("name", "")
                if name and name not in recent_names:
                    recent_names[name] = days_ago
    except:
        pass
    return recent_names

def _apply_repetition_penalty(projections):
    """Penalize players picked in recent lineups to force roster turnover.

    Picked yesterday:   0.82x penalty
    Picked 2 days ago:  0.90x penalty
    Picked 3 days ago:  0.95x penalty

    This prevents the same 5 guys every day (Ty Jerome, Scotty Pippen problem).
    Applied to chalk_ev so MILP sees the penalized value.
    """
    recent = _get_recent_picks(days=3)
    if not recent:
        return
    penalties = {0: 0.82, 1: 0.82, 2: 0.90, 3: 0.95}
    for p in projections:
        name = p.get("name", "")
        if name in recent:
            days_ago = recent[name]
            penalty = penalties.get(days_ago, 0.95)
            p["chalk_ev"] = round(p["chalk_ev"] * penalty, 2)
            p["rating"] = round(p["rating"] * penalty, 1)
            p["_rep_penalty"] = penalty

def _build_lineups(projections):
    # Anti-repetition: penalize players picked in the last 3 days
    _apply_repetition_penalty(projections)

    # STARTING 5: MILP-optimized slot assignments using ADDITIVE formula
    # MILP maximizes: Σ rating_i × (slot_mult_j + card_boost_i)
    # No team limit — Real Sports has no per-team restriction. If the best
    # value is 5 players from the same blowout loss, so be it.
    # March 3: winner gmoneytb had 3 PHI players from a 40-pt blowout loss.
    chalk_eligible = [p for p in projections if p["rating"] >= CHALK_FLOOR]
    chalk = optimize_lineup(chalk_eligible, n=5, sort_key="chalk_ev",
                            rating_key="rating", card_boost_key="est_mult",
                            max_per_team=0)

    # CONTRARIAN: maximize leverage against the field
    chalk_names = {p["name"] for p in chalk}
    contrarian_pool = [p for p in projections
                       if p["name"] not in chalk_names and p["rating"] >= MOONSHOT_FLOOR]
    # Score by contrarian value — card-adjusted ceiling × closeness × momentum
    # Card boost is already baked into contrarian_score, so zero it for MILP
    for p in contrarian_pool:
        p["_contrarian_ev"] = contrarian_score(p)
        p["_no_boost"] = 0.0
    upside = optimize_lineup(contrarian_pool, n=5, sort_key="_contrarian_ev",
                             rating_key="_contrarian_ev",
                             card_boost_key="_no_boost")

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
GAME_MOONSHOT_FLOOR = 4.0  # Filter out low-production players

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

    Starting 5: MILP-optimized slot assignments with team balance (min 2 per team).
                 Maximizes Real Score × slot multiplier. Floor = GAME_CHALK_FLOOR (3.5).
    Contrarian: High-card-ceiling leverage play targeting Top 10% payout tier.
                 Enforces negative correlation with Starting 5 (no overlap, opposite
                 team weighting). Floor = GAME_MOONSHOT_FLOOR (4.0).
    """
    rescored = _apply_game_script(projections, game)
    chalk_eligible = [p for p in rescored if p["rating"] >= GAME_CHALK_FLOOR]

    # STARTING 5: MILP-optimized, balanced across both teams
    # Uses additive formula: rating × (slot_mult + card_boost)
    chalk = optimize_lineup(chalk_eligible, n=5, min_per_team=2, sort_key="chalk_ev",
                            rating_key="rating", card_boost_key="est_mult")

    # CONTRARIAN: leverage play — no overlap with Starting 5
    chalk_names = {p["name"] for p in chalk}
    contrarian_pool = [p for p in rescored
                       if p["name"] not in chalk_names and p["rating"] >= GAME_MOONSHOT_FLOOR]

    # Score by contrarian value with game spread awareness
    # Card boost is already baked into contrarian_score, so zero it for MILP
    game_spread = game.get("spread") or 0
    for p in contrarian_pool:
        p["_contrarian_ev"] = contrarian_score(p, spread=game_spread)
        p["_no_boost"] = 0.0

    # Determine opposite team weighting for negative correlation
    chalk_teams = {}
    for p in chalk:
        t = p.get("team", "")
        chalk_teams[t] = chalk_teams.get(t, 0) + 1
    # If chalk favors one team, contrarian should favor the other
    if len(chalk_teams) >= 2:
        dominant_team = max(chalk_teams, key=chalk_teams.get)
        for p in contrarian_pool:
            if p.get("team") != dominant_team:
                p["_contrarian_ev"] *= 1.25  # Boost opposite team

    upside = optimize_lineup(contrarian_pool, n=5, min_per_team=2,
                             sort_key="_contrarian_ev",
                             rating_key="_contrarian_ev",
                             card_boost_key="_no_boost")

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
        st = g.get("startTime", "")
        g["locked"] = _is_locked(st) if st else False
        g["draftable"] = not _is_completed(st) if st else False
    return games

@app.get("/api/slate")
async def get_slate():
    games = fetch_games()
    if not games:
        return {"date": date.today().isoformat(), "games": [], "lineups": {"chalk": [], "upside": []}, "locked": False}

    # Only show/project games that are still draftable (not yet past lock window)
    # ESPN returns all games for the day, including already-completed ones.
    # At 1 AM EST the completed games from the evening should be invisible;
    # the upcoming day's games should be shown instead.
    draftable_games = [g for g in games if not _is_completed(g.get("startTime", ""))]

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
        return {"date": date.today().isoformat(), "games": games,
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
    result = {"date": date.today().isoformat(), "games": games,
              "lineups": {"chalk": chalk, "upside": upside}, "locked": locked}
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
        return {"date": date.today().isoformat(), "game": game,
                "gameScript": None,
                "lineups": {"chalk": [], "upside": []},
                "locked": True, "injuries": [], "temporal": {}}

    projections = _run_game(game)
    if not projections:
        return JSONResponse({"error": "No projections available."}, status_code=503)
    chalk, upside = _build_game_lineups(projections, game)
    _save_history(game["label"], chalk)
    script = _game_script_label(game.get("total"))
    injuries = _get_injuries(game)
    # ── TRAV: Temporal Risk-Adjusted Value metadata ───────────────────
    # Calculate lineup duplication probability and optimal lock time.
    # Attached as extra response key — frontend ignores unknown keys.
    pool_size = len(projections)
    dup_prob = lineup_duplication_probability(chalk, pool_size)
    temporal_meta = optimal_lock_time(
        game.get("startTime", ""), chalk, pool_size
    )

    # Annotate players with TRAV-adjusted scores
    for p in chalk + upside:
        p["_trav"] = trav_adjusted_score(
            p.get("rating", 0), dup_prob,
            temporal_meta.get("tiebreaker_equity_at_optimal", 0.5)
        )
        p["_dup_prob"] = dup_prob

    result = {"date": date.today().isoformat(), "game": game,
              "gameScript": script,
              "lineups": {"chalk": chalk, "upside": upside},
              "locked": locked,
              "injuries": injuries,
              "temporal": temporal_meta}
    # Cache picks so they survive as lock snapshot if slate locks later
    _cs(f"picks_{gameId}", result)
    return result

@app.get("/api/history")
async def get_history():
    if not HISTORY_FILE.exists(): return []
    try: return json.loads(HISTORY_FILE.read_text())
    except: return []

@app.post("/api/save-predictions")
async def save_predictions():
    """Save current predictions to GitHub as CSV."""
    today = date.today().isoformat()
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
            rows.extend(_predictions_to_csv(cached_picks["lineups"], f"game_{gid}"))

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
    date_str = payload.get("date", date.today().isoformat())
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


@app.get("/api/refresh")
async def refresh():
    cleared = 0
    try:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink(); cleared += 1
    except Exception as e:
        return {"status": "error", "message": str(e)}
    return {"status": "ok", "cleared": cleared, "ts": datetime.now().isoformat()}
