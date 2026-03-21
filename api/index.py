"""
Oracle API — FastAPI backend for the Real Sports NBA draft optimizer.

Single-entry backend: all HTTP routes, projection pipeline, slate/picks computation,
Line of the Day engine, and Lab (Ben) chat live here. Uses api.real_score, api.asset_optimizer,
api.line_engine, and api.rotowire for domain logic. Config from data/model-config.json (GitHub);
secrets from environment variables only (never in code). Global exception handler returns
generic 500 to clients and logs full traceback server-side.
"""
import json
import copy
import csv
import io
import hashlib
import re
import traceback
import unicodedata
import pickle
import os
import base64
import threading
import time
import uuid
import asyncio
from statistics import mode, mean, StatisticsError
from typing import Any, Optional, Tuple

import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Query, Body, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

# Load .env before importing ecosystem modules so module-level os.getenv() calls
# (e.g. ANTHROPIC_API_KEY in line_engine.py) pick up local env vars correctly.
load_dotenv()

# Real Score Ecosystem modules
try:
    from api.real_score import real_score_projection, _make_rng, closeness_coefficient
    from api.asset_optimizer import optimize_lineup
    from api.line_engine import run_line_engine, _enrich_pick_from_projections, _game_lookup_from_games
    from api.rotowire import get_all_statuses, is_safe_to_draft, clear_cache as _rw_clear
except ImportError:
    from .real_score import real_score_projection, _make_rng, closeness_coefficient
    from .asset_optimizer import optimize_lineup
    from .line_engine import run_line_engine, _enrich_pick_from_projections, _game_lookup_from_games
    from .rotowire import get_all_statuses, is_safe_to_draft, clear_cache as _rw_clear
DOCS_SECRET = os.getenv("DOCS_SECRET", "")  # optional: require ?docs_key=DOCS_SECRET or X-Docs-Key for /docs, /redoc, /openapi.json

app = FastAPI()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch any unhandled exception; log server-side only, return generic 500 to client."""
    print(f"[unhandled] {request.method} {request.url.path}: {exc}", flush=True)
    traceback.print_exc()
    return JSONResponse(
        content={"error": "An unexpected error occurred"},
        status_code=500,
    )


@app.middleware("http")
async def docs_auth_and_log(request, call_next):
    """Optional docs auth (when DOCS_SECRET set) + structured request logging."""
    if DOCS_SECRET and request.url.path in ("/docs", "/redoc", "/openapi.json"):
        key = request.query_params.get("docs_key") or request.headers.get("x-docs-key")
        if key != DOCS_SECRET:
            return JSONResponse({"detail": "Docs require docs_key or X-Docs-Key"}, status_code=401)

    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 1)
    log_line = json.dumps({
        "request_id": request_id,
        "path": request.url.path,
        "method": request.method,
        "status": response.status_code,
        "duration_ms": duration_ms,
    })
    print(log_line, flush=True)
    return response


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _validate_date(date_str: str) -> Optional[JSONResponse]:
    """Return a 400 JSONResponse if date_str is not YYYY-MM-DD format, else None."""
    if not _DATE_RE.match(date_str):
        return JSONResponse({"error": "Invalid date format (expected YYYY-MM-DD)"}, status_code=400)
    return None


# ── GitHub API helpers for persistent CSV storage ──
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.getenv("GITHUB_REPO", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

# Startup env validation — warn on missing required vars (never crash; app degrades gracefully)
_REQUIRED_ENV = ["GITHUB_TOKEN", "GITHUB_REPO", "ANTHROPIC_API_KEY"]
_missing_env = [k for k in _REQUIRED_ENV if not os.getenv(k)]
if _missing_env:
    print(f"[WARN] Missing env vars: {_missing_env} — affected features will be degraded")

# All data writes go to main (default branch). The watchPatterns in railway.toml
# skip builds when only data/ or .github/ files change, so data commits on main
# do NOT trigger Railway rebuilds.
def _data_ref(path: str):
    """Return branch override for GitHub API calls, or None for default (main)."""
    return None

def _github_get_file(path: str, ref_override: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Get file content and SHA from GitHub. Returns (content_str, sha) or (None, None)."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None, None
    ref = ref_override if ref_override is not None else _data_ref(path)
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    if ref:
        url += f"?ref={ref}"
    r = requests.get(url, headers={
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }, timeout=10)
    if r.status_code == 200:
        data = r.json()
        content = base64.b64decode(data["content"]).decode("utf-8")
        return content, data["sha"]
    return None, None

def _github_list_dir(path: str, ref_override: Optional[str] = None) -> list:
    """List files in a GitHub repo directory."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return []
    ref = ref_override if ref_override is not None else _data_ref(path)
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    if ref:
        url += f"?ref={ref}"
    r = requests.get(url, headers={
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }, timeout=10)
    if r.status_code == 200:
        return r.json()
    return []

def _github_write_file(path: str, content: str, message: str = "auto-update", max_retries: int = 3) -> dict:
    """Create or update a file in the GitHub repo via Contents API.
    Retries on 422 Conflict (SHA mismatch) with exponential backoff."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return {"error": "GITHUB_TOKEN or GITHUB_REPO not configured"}

    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"

    for attempt in range(max_retries):
        _, sha = _github_get_file(path)
        payload = {
            "message": message,
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
        }
        if sha:
            payload["sha"] = sha

        try:
            r = requests.put(url, json=payload, headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
            }, timeout=15)

            if r.status_code in (200, 201):
                return {"ok": True, "path": path}

            # 422 = Conflict (SHA mismatch due to concurrent write)
            # Retry with fresh SHA fetch
            if r.status_code == 422 and attempt < max_retries - 1:
                backoff_sec = (2 ** attempt)  # 1, 2, 4 seconds
                time.sleep(backoff_sec)
                continue

            # For other errors or final retry exhausted: log detail, return generic (no leak to client)
            print(f"[github] write failed {path}: {r.status_code} {r.text[:200]}")
            return {"error": "GitHub write failed"}

        except Exception as e:
            if attempt < max_retries - 1:
                backoff_sec = (2 ** attempt)
                time.sleep(backoff_sec)
                continue
            print(f"[github] write exception {path}: {e}")
            return {"error": "GitHub write failed"}

    return {"error": "GitHub write failed after maximum retries"}


def _github_write_batch(files: list, message: str = "auto-update") -> dict:
    """Write multiple files in a single commit using the Git Trees API.
    files: list of {"path": str, "content": str}.
    Falls back to sequential _github_write_file if tree API fails."""
    if not GITHUB_TOKEN or not GITHUB_REPO or not files:
        return {"error": "no files or credentials"}
    branch = "main"
    h = {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github.v3+json"}
    try:
        # Get the current commit SHA for the target branch
        ref_r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/git/refs/heads/{branch}",
            headers=h, timeout=10,
        )
        if ref_r.status_code != 200:
            raise ValueError(f"ref lookup failed: {ref_r.status_code}")
        base_sha = ref_r.json()["object"]["sha"]
        # Get the tree SHA of the base commit
        commit_r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/git/commits/{base_sha}",
            headers=h, timeout=10,
        )
        if commit_r.status_code != 200:
            raise ValueError(f"commit lookup failed: {commit_r.status_code}")
        base_tree_sha = commit_r.json()["tree"]["sha"]
        # Build tree entries
        tree_entries = []
        for f in files:
            tree_entries.append({
                "path": f["path"],
                "mode": "100644",
                "type": "blob",
                "content": f["content"],
            })
        # Create new tree
        tree_r = requests.post(
            f"https://api.github.com/repos/{GITHUB_REPO}/git/trees",
            json={"base_tree": base_tree_sha, "tree": tree_entries},
            headers=h, timeout=15,
        )
        if tree_r.status_code not in (200, 201):
            raise ValueError(f"tree create failed: {tree_r.status_code}")
        new_tree_sha = tree_r.json()["sha"]
        # Create commit
        commit_create_r = requests.post(
            f"https://api.github.com/repos/{GITHUB_REPO}/git/commits",
            json={"message": message, "tree": new_tree_sha, "parents": [base_sha]},
            headers=h, timeout=15,
        )
        if commit_create_r.status_code not in (200, 201):
            raise ValueError(f"commit create failed: {commit_create_r.status_code}")
        new_commit_sha = commit_create_r.json()["sha"]
        # Update branch ref
        ref_update_r = requests.patch(
            f"https://api.github.com/repos/{GITHUB_REPO}/git/refs/heads/{branch}",
            json={"sha": new_commit_sha},
            headers=h, timeout=10,
        )
        if ref_update_r.status_code != 200:
            raise ValueError(f"ref update failed: {ref_update_r.status_code}")
        return {"ok": True, "sha": new_commit_sha}
    except Exception as e:
        print(f"[github] batch write failed, falling back to sequential: {e}")
        # Fallback: write files one at a time
        for f in files:
            try:
                _github_write_file(f["path"], f["content"], message)
            except Exception as e2:
                print(f"[github] sequential fallback err {f['path']}: {e2}")
        return {"ok": True, "fallback": True}


def _github_delete_file(path, sha, message="auto-delete"):
    """Delete a file from the GitHub repo via Contents API.
    Routes data/* paths to data branch (same as write/read)."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return False
    branch = _data_ref(path)
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    payload = {"message": message, "sha": sha}
    if branch:
        payload["branch"] = branch
    r = requests.delete(url, json=payload, headers={
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }, timeout=15)
    return r.status_code in (200, 204)


def _slate_backup_to_github(slate_data: dict):
    """Write slate response to GitHub as a locked-state backup (deduped by date).
    Called once when we promote reg_cache -> lock_cache so cold-start instances can recover."""
    try:
        today = _et_date().isoformat()
        path = f"data/locks/{today}_slate.json"
        existing, _ = _github_get_file(path)
        if existing:
            try:
                if not json.loads(existing).get("_busted"):
                    return  # Valid lock file already exists — don't overwrite
            except Exception:
                return  # Can't parse existing — treat as valid and skip
        content = json.dumps(slate_data, default=str)
        _github_write_file(path, content, f"slate lock backup {today}")
    except Exception as e:
        print(f"slate backup err: {e}")


def _slate_restore_from_github():
    """Read the locked-state slate backup from GitHub. Returns dict or None."""
    try:
        today = _et_date().isoformat()
        path = f"data/locks/{today}_slate.json"
        content, _ = _github_get_file(path)
        if content:
            data = json.loads(content)
            if data.get("_busted"):
                return None  # Tombstoned by _bust_slate_cache — treat as cache miss
            return data
    except Exception as e:
        print(f"slate restore err: {e}")
    return None


# ── GitHub-Persisted Slate & Game Projection Cache ──
# grep: SLATE CACHE
# Pre-computed predictions are persisted to data/slate/ so that cold-start
# Railway instances serve the same picks without re-running the prediction engine.
# Pattern mirrors data/locks/ and data/lines/ — date-keyed JSON files on GitHub.

def _slate_cache_to_github(slate_data: dict):
    """Persist today's generated slate to GitHub for cold-start recovery.
    Deduped by date — overwrites same-day file on regeneration (injury/config change).
    Embeds deploy_sha for Scenario 1 auto-detection (dev ships model update mid-slate)."""
    try:
        today = _et_date().isoformat()
        path = f"data/slate/{today}_slate.json"
        # Stamp deploy SHA so /api/slate can detect when a new deploy invalidates cached picks
        sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "")
        if sha:
            slate_data["deploy_sha"] = sha[:7]
        content = json.dumps(slate_data, default=str)
        _github_write_file(path, content, f"slate cache {today}")
    except Exception as e:
        print(f"[slate-cache] write err: {e}")

def _slate_cache_from_github():
    """Load today's cached slate from GitHub. Returns dict or None.
    Checks bust sentinel first (data branch and main so retrain workflow bust is seen)."""
    try:
        today = _et_date().isoformat()
        bust_path = f"data/slate/{today}_bust.json"
        for ref in (None, "main"):  # data branch first, then main (retrain writes bust to main)
            bust_content, _ = _github_get_file(bust_path, ref_override=ref)
            if bust_content:
                bust_data = json.loads(bust_content)
                if bust_data.get("_busted") or bust_data.get("at"):
                    return None
        path = f"data/slate/{today}_slate.json"
        content, _ = _github_get_file(path)
        if content:
            data = json.loads(content)
            if data.get("_busted"):
                return None
            return data
    except Exception as e:
        print(f"[slate-cache] read err: {e}")
    return None


def _github_slate_bust_active() -> bool:
    """True if today's slate bust sentinel is active on GitHub (/api/refresh, lab config).

    Railway runs multiple instances; /api/refresh only clears /tmp on one container.
    Without this check, locked slates keep serving stale lineups from slate_v5_locked.

    Uses only data/slate/{date}_bust.json (not data/locks/*): the lock backup may still
    be a tombstone briefly while bust.json is already cleared after sync regen."""
    try:
        today = _et_date().isoformat()
        bust_path = f"data/slate/{today}_bust.json"
        for ref in (None, "main"):
            bust_content, _ = _github_get_file(bust_path, ref_override=ref)
            if bust_content:
                bust_data = json.loads(bust_content)
                if bust_data.get("_busted") or bust_data.get("at"):
                    return True
    except Exception:
        pass
    return False


def _clear_local_slate_tmp_caches():
    """Remove today's slate + lock JSON from this instance's /tmp (hashed paths)."""
    try:
        _lp("slate_v5_locked").unlink(missing_ok=True)
    except Exception:
        pass
    try:
        _cp("slate_v5").unlink(missing_ok=True)
    except Exception:
        pass


def _games_cache_to_github(all_game_projections: dict):
    """Persist per-game projections {gameId: [players...]} to GitHub.
    Allows /api/picks to serve from cache without re-running _run_game()."""
    try:
        today = _et_date().isoformat()
        path = f"data/slate/{today}_games.json"
        content = json.dumps(all_game_projections, default=str)
        _github_write_file(path, content, f"game projections {today}")
    except Exception as e:
        print(f"[games-cache] write err: {e}")

def _games_cache_from_github():
    """Load per-game projections from GitHub. Returns {gameId: [...]} or None."""
    try:
        today = _et_date().isoformat()
        path = f"data/slate/{today}_games.json"
        content, _ = _github_get_file(path)
        if content:
            data = json.loads(content)
            if data.get("_busted"):
                return None
            return data
    except Exception as e:
        print(f"[games-cache] read err: {e}")
    return None

def _bust_slate_cache():
    """Clear today's slate cache from /tmp AND GitHub so next request regenerates.
    Called by /api/refresh and /api/lab/update-config.

    If Starting 5 still shows low-recent-min players (e.g. Olynyk, Plumlee, Drummond)
    after a bust: (1) Ensure this codebase is deployed with the recent_raw_min fix
    (_fetch_athlete using recent_raw_min when recent split < 10). (2) On serverless,
    each instance has its own /tmp; the tombstone on GitHub forces all instances to
    treat slate cache as miss and regenerate. (3) If the app is still on an old deploy,
    merge and deploy the fix, then trigger /api/refresh or wait for next slate load."""
    today = _et_date().isoformat()
    # Clear /tmp caches
    for key in ["slate_v5"]:
        try:
            _cp(key).unlink()
        except Exception:
            pass
    # Clear all /tmp cache files (slate + per-game); they'll regenerate with new config
    try:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink(missing_ok=True)
    except Exception:
        pass
    # Bust GitHub cache: single batched commit with all tombstones + sentinel.
    # Previously 4 separate commits per bust — now 1 commit via Git Trees API.
    bust_tombstone = json.dumps({"_busted": True})
    bust_sentinel = json.dumps({"_busted": True, "at": datetime.now(timezone.utc).isoformat()})
    bust_files = [
        {"path": f"data/slate/{today}_slate.json", "content": bust_tombstone},
        {"path": f"data/slate/{today}_games.json", "content": bust_tombstone},
        {"path": f"data/slate/{today}_bust.json", "content": bust_sentinel},
        {"path": f"data/locks/{today}_slate.json", "content": bust_tombstone},
    ]
    try:
        _github_write_batch(bust_files, f"bust slate cache {today}")
    except Exception as e:
        print(f"[bust-slate] batch write err: {e}")


def _csv_escape(v):
    """Escape a value for CSV (quote if it contains commas or quotes)."""
    s = str(v)
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s

def _predictions_to_csv(lineups, scope):
    """Convert lineup dicts to CSV rows."""
    rows = []
    for lineup_type, players in [("chalk", lineups.get("chalk", [])), ("upside", lineups.get("upside", [])), ("the_lineup", lineups.get("the_lineup", []))]:
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

# Hardcoded draft-lineup blacklist (Starting 5, Moonshot, THE LINE UP only).
# This is an application-layer override and is intentionally NOT part of
# data/model-config.json so model tuning remains config-driven.
BLACKLISTED_PLAYERS = {
    "Kevin Love",
    "Clint Capela",
    "Mike Conley",
    "Kadary Richmond",
    "Chaney Johnson",
}

# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONFIG — Loaded from data/model-config.json on GitHub.
# Falls back to hardcoded defaults if GitHub is unreachable or file missing.
# The Lab (Phase 3) writes config updates here; changes take effect within 5 min.
# ─────────────────────────────────────────────────────────────────────────────

_CONFIG_DEFAULTS = {
    "version": 1,
    "card_boost": {
        "ceiling": 3.0, "floor": 0.2,
        "big_market_teams": ["LAL","GS","GSW","BOS","NY","NYK","PHI","MIA","DEN","LAC","CHI"],
        # Sigmoid tier estimation: boost = sig_ceiling - sig_range × sigmoid((PPG - sig_midpoint) / sig_scale)
        "sig_ceiling": 3.0, "sig_range": 2.8, "sig_midpoint": 12.0, "sig_scale": 4.0,
        "big_market_discount": 0.15,
        # Player overrides from real ownership/actuals data (boost is stable per-player ±0.1x)
        "player_overrides": {
            "Aaron Nesmith": 1.9, "Ace Bailey": 2.1, "Al Horford": 2.0,
            "Amen Thompson": 0.6, "Andre Drummond": 2.3, "Anthony Edwards": 0.2,
            "Bam Adebayo": 0.6, "Brook Lopez": 3.0, "Bryce McGowens": 3.0,
            "Cade Cunningham": 0.2, "Cameron Payne": 3.0, "Clint Capela": 3.0,
            "Cody Williams": 3.0, "Collin Sexton": 2.0, "Cooper Flagg": 0.7,
            "De'Aaron Fox": 0.6, "De'Anthony Melton": 1.9, "Derik Queen": 1.4,
            "Derrick White": 0.7, "Devin Carter": 3.0, "Donovan Clingan": 1.0,
            "Grant Williams": 2.8, "Gui Santos": 2.6, "Isaiah Collier": 1.5,
            "Jalen Johnson": 0.3, "Jarace Walker": 2.3, "Jaylen Brown": 0.3,
            "Jordan Miller": 2.5, "Julian Reese": 3.0, "Julius Randle": 0.6,
            "Kam Jones": 3.0, "Kel'el Ware": 1.4, "Kevin Durant": 0.5,
            "Klay Thompson": 2.6, "Kon Knueppel": 0.8, "Kyle Filipowski": 2.1,
            "Kyle Kuzma": 1.7, "LaMelo Ball": 0.6, "Leonard Miller": 3.0,
            "Luka Dončić": 0.0, "Luke Kennard": 3.0, "Matas Buzelis": 1.4,
            "Maxime Raynaud": 2.1, "Myles Turner": 1.6, "Noah Clowney": 2.0,
            "OG Anunoby": 1.1, "Ousmane Dieng": 3.0, "Pat Spencer": 3.0,
            "Precious Achiuwa": 2.2, "Reed Sheppard": 1.5, "Robert Williams III": 2.3,
            "Ron Harper Jr.": 3.0, "Royce O'Neale": 2.0, "Rudy Gobert": 1.1,
            "Russell Westbrook": 1.1, "Scottie Barnes": 0.5, "Simone Fontecchio": 3.0,
            "Tristan da Silva": 2.8, "Tyler Herro": 0.8, "Victor Wembanyama": 0.3,
        },
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
    "real_score": {
        "dfs_weights":{"pts":2.5,"reb":0.5,"ast":1.0,"stl":2.0,"blk":1.5,"tov":-1.5},
        "compression_divisor": 4.5,
        "compression_power": 0.78,
        "rs_cap": 20.0,
        "ai_blend_weight": 0.25,
        # Optional RS bucket calibration (disabled by default).
        # Use to correct systematic bias by scoring tier without touching base model.
        "bucket_calibration": {
            "enabled": False,
            "high_pts_threshold": 18.462,
            "mid_pts_threshold": 6.324,
            "high_mult": 0.8206,
            "mid_mult": 0.9336,
            "low_mult": 1.0813,
        },
        "archetype_calibration": {
            "enabled": True,
            "archetypes": {
                "star": 0.92,
                "starter": 1.0,
                "wing_role": 1.02,
                "bench_microwave": 1.08,
                "big": 1.04,
            },
        },
        "post_lock_calibration": {
            "enabled": True,
            "require_locked_slate": True,
            "recency_strength": 0.12,
            "max_nudge": 0.12,
            "cascade_weight": 0.06,
        },
        "stat_stuffer": {
            "enabled": False,
            "pts_threshold": 15,
            "reb_threshold": 7,
            "ast_threshold": 5,
            "stl_threshold": 1.5,
            "blk_threshold": 1.0,
            "bonus_3cat": 1.08,
            "bonus_td": 1.15,
        },
        "closeness": {
            "enabled": False,
            "strength": 0.5,
            "max_mult": 1.40,
        },
        "cascade_rs": {
            "enabled": False,
            "strength": 0.6,
        },
        "role_spike_rs": {
            "enabled": False,
            "min_ratio": 1.2,
            "strength": 0.4,
        },
    },
    "cascade": {"redistribution_rate":0.70,"per_player_cap_minutes":10.0,"center_forward_share":0.30},
    "projection": {
        "min_gate_minutes":15,"lock_buffer_minutes":5,"season_recent_blend":0.5,"default_total":222,"b2b_minute_penalty":0.88,
        "major_role_change_threshold":0.75,"major_role_change_recent_weight":0.80,
        "moderate_decline_threshold":0.90,"moderate_decline_recent_weight":0.65,
        # DNP / reliability guards (added after March 4th audit)
        "gtd_minute_penalty":0.75,      # GTD players: 25% minute reduction
        "dnp_risk_min_threshold":5.0,   # recent avg min below this = dnp_risk flag (was 8; Mar 15 fix: 5-8min players were skipped entirely, losing deep bench contrarians like Quinten Post who hit for RS 3.3 / 16.4 value)
        "reliability_floor":0.70,       # minimum reliability multiplier on chalk_ev
        "chalk_boost_cap":2.5,          # was 1.5; Mar 6: winners stacked 3.0x boost players in chalk
        "chalk_season_min_floor":22.0,  # season avg floor for Starting 5 — proven rotation players (22+ min)
        "chalk_recent_min_floor":20.0,  # recent avg floor — excludes players who've fallen out of rotation
                                        # despite high season avg (e.g. demoted starter, rest-management)
        "chalk_max_stars":1,            # max players with boost < threshold allowed in chalk lineup (was 2; Mar 14: 4/6 winners had 0 stars)
        "chalk_star_boost_threshold":0.8, # boost below this = "star" (low ownership); counts toward cap (was 0.6; Bam 0.9/Reaves 0.8 weren't penalized)
        "big_man_calibration": {        # post-LGBM multiplier for rebounding bigs; see project_player()
            "reb_baseline": 6.0, "reb_scale": 0.04, "blk_scale": 0.10, "pts_cap": 20.0,
        },
        "bench_pts_threshold": 14.0,    # pts avg ceiling for "bench/role player" spread classification (was 12)
        "bench_min_threshold": 30.0,    # min avg ceiling for "bench/role player" (was 26)
        "chalk_min_boost_floor": 1.2,   # minimum card boost required for chalk eligibility;
                                        # excludes high-ownership stars (Westbrook 1.1x, Clingan 1.0x)
                                        # Mar 15 fix: without this, top chalk slots go to near-zero-boost stars
    },
    "moonshot": {
        # v6 (leaderboard-informed): gates widened to match actual winner profiles.
        # Winners are 15-25 min role players with 3x+ boost on dev teams (da Silva,
        # Ellis, Clifford, Santos, Sensabaugh). Old 25-min/3.0 rating gates blocked them.
        "min_minutes_floor":20, "min_recent_minutes_floor":20, "min_card_boost":1.5, "min_rating_floor":3.0,
        "card_boost_weight":2.5, "minutes_weight":1.0,
        "max_centers":3, "boost_leverage_power":1.2,
        "require_rotowire_clearance":True, "max_ownership_pct":3.0,
        "variance_penalty": 0.15,      # light damping — moonshot wants upside volatility
        "wildcard_min_boost": 2.0, "wildcard_min_minutes": 15.0, "wildcard_min_season_pts": 7.0,
        "role_spike_ratio": 1.4, "role_spike_recent_floor": 20.0, "role_spike_season_floor": 8.0,
        # RS-bypass: high-RS proven scorers bypass the boost floor (disabled offline for safety)
        "rs_bypass": {"enabled": False, "min_rating": 5.0, "min_season_min": 25.0, "min_boost": 0.3},
    },
    "matchup": {
        "enabled": True,
        "claude_enabled": False,  # Layer 1.5 killed — ESPN def stats in _compute_matchup_factor() sufficient
        "def_scale": 0.35,        # how strongly opponent pts_allowed affects the factor
        "pos_scale_g": 1.05,      # guards benefit most from weak defenses (pts-driver)
        "pos_scale_f": 1.00,      # forwards neutral
        "pos_scale_c": 0.90,      # centers less sensitive (RS from reb/blk not just pts)
        "chalk_adj_min": 0.92,    # narrow range for chalk — reliability first
        "chalk_adj_max": 1.10,
        "moonshot_adj_min": 0.75, # wider range for moonshot — upside signal
        "moonshot_adj_max": 1.30,
        "claude_timeout_seconds": 25,
    },
    "team_motivation": {
        # Late-season team intent signal (March/April):
        # Starting 5 should favor stable, win-now rotations.
        "enabled": True,
        "start_date": "2026-03-01",
        "seeding_gap_games": 2.0,
        "playin_gap_games": 2.0,
        "elimination_buffer_games": 3.0,
        # Soft multipliers only (no hard bans).
        "tier_a_mult_chalk": 1.08,
        "tier_b_mult_chalk": 1.00,
        "tier_c_mult_chalk": 0.90,
        # Keep moonshot neutral by default; can tune later.
        "tier_a_mult_moonshot": 1.00,
        "tier_b_mult_moonshot": 1.00,
        "tier_c_mult_moonshot": 1.00,
        "min_mult": 0.88,
        "max_mult": 1.12,
        # Optional manual overrides: {"LAL":"A","WAS":"C"}
        "team_overrides": {},
    },
    "lineup": {
        "chalk_rating_floor": 2.0,      # was 2.8; Mar 6: Ighodaro RS 2.3 was in all 3 winning lineups
        "game_chalk_rating_floor": 3.5,
        "avg_slot_multiplier": 1.6,
        "slot_multipliers": [2.0, 1.8, 1.6, 1.4, 1.2],
        # Starting 5 MILP: blend boost toward neutral so RS drives selection (0 = legacy).
        "chalk_milp_rs_focus": 0.0,
        "chalk_milp_boost_neutral": 1.0,
    },
    "core_pool": {
        "enabled": True,
        "size": 8,
        "metric": "max_ev",
        "blend_weight": 0.5,
    },
    "line": {
        "min_confidence": 50,
        "min_edge_pct": 3.0,
        "recent_form_over_ratio": 1.15,
        "recent_form_under_ratio": 0.92,
        "min_edge_pts": 2.0,
        "min_edge_other": 1.5,
        "min_edge_other_over": 2.5,
        "min_season_minutes": 20.0,
        # stat_floors must match line_engine._STAT_META defaults — used when GitHub is unreachable
        "stat_floors": {"points": 6.0, "rebounds": 5.5, "assists": 1.5},
    },
    "lab": {
        "auto_improve_threshold_pct": 3.0,
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
        except Exception as _ce:
            print(f"[WARN] Config cache read failed: {_ce}")
    try:
        content, _ = _github_get_file("data/model-config.json")
        if content:
            cfg = json.loads(content)
            cache_file.write_text(json.dumps(cfg))
            return cfg
    except Exception as _ce:
        print(f"[WARN] Config GitHub load failed, using defaults: {_ce}")
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

AI_MODEL = None  # Legacy single-head bundle
AI_MODEL_BASELINE = None  # bundle_version 2
AI_MODEL_SPIKE = None
AI_FEATURES = None  # Feature list saved alongside model to verify alignment
_LGBM_LOAD_ATTEMPTED = False
_LGBM_LOAD_LOCK = threading.Lock()

# ── Slate pipeline dedup: prevents duplicate full-pipeline runs on concurrent cold-start requests ──
# If N requests arrive simultaneously and all miss the cache, the first acquires the lock and runs;
# the rest wait (up to 120s) then serve from the warm cache populated by the first.
_SLATE_GEN_LOCK = threading.Lock()
_SLATE_GEN_IN_FLIGHT = False

_LGBM_PATHS = [
    Path(__file__).parent.parent / "lgbm_model.pkl",
    Path(__file__).parent / "lgbm_model.pkl",
    Path("lgbm_model.pkl"),
]

def _ensure_lgbm_loaded():
    """Lazy-load the LightGBM model bundle on first use.
    Supports bundle_version 2 (baseline + spike) or legacy single model."""
    global AI_MODEL, AI_MODEL_BASELINE, AI_MODEL_SPIKE, AI_FEATURES, _LGBM_LOAD_ATTEMPTED
    if _LGBM_LOAD_ATTEMPTED:
        return
    with _LGBM_LOAD_LOCK:
        if _LGBM_LOAD_ATTEMPTED:
            return
        for _p in _LGBM_PATHS:
            if _p.exists():
                try:
                    with open(_p, "rb") as _f:
                        _bundle = pickle.load(_f)
                    if not isinstance(_bundle, dict) or "features" not in _bundle:
                        continue
                    AI_FEATURES = _bundle["features"]
                    if _bundle.get("bundle_version") == 2 and "model_baseline" in _bundle and "model_spike" in _bundle:
                        AI_MODEL_BASELINE = _bundle["model_baseline"]
                        AI_MODEL_SPIKE = _bundle["model_spike"]
                        AI_MODEL = None
                        break
                    if "model" in _bundle:
                        AI_MODEL = _bundle["model"]
                        AI_MODEL_BASELINE = None
                        AI_MODEL_SPIKE = None
                        break
                except (OSError, pickle.UnpicklingError, KeyError, ValueError, ModuleNotFoundError):
                    # ModuleNotFoundError: e.g. lightgbm not installed in env but pickle references it
                    pass
        _LGBM_LOAD_ATTEMPTED = True
        if AI_MODEL is None and AI_MODEL_BASELINE is None:
            print("[WARN] LightGBM model not found or invalid bundle — using heuristic fallback for all projections")


def _lgbm_predict_rs(feat_vec: list) -> Optional[float]:
    """Return blended RS prediction from loaded bundle, or None if unavailable."""
    _ensure_lgbm_loaded()
    if AI_FEATURES is not None and len(feat_vec) != len(AI_FEATURES):
        raise ValueError(f"Feature mismatch: model expects {len(AI_FEATURES)}, got {len(feat_vec)}")
    arr = np.array([feat_vec])
    if AI_MODEL_BASELINE is not None and AI_MODEL_SPIKE is not None:
        base = float(AI_MODEL_BASELINE.predict(arr)[0])
        spike = float(AI_MODEL_SPIKE.predict(arr)[0])
        return base + max(0.0, spike)
    if AI_MODEL is not None:
        return float(AI_MODEL.predict(arr)[0])
    return None


def _lgbm_feature_vector(
    *,
    avg_min: float,
    pts: float,
    reb: float,
    ast: float,
    stl: float,
    blk: float,
    spread: Optional[float],
    side: str,
    season_pts: float,
    recent_pts: float,
    season_min: float,
    recent_min: float,
    cascade_bonus: float,
    games_played: Optional[float] = None,
) -> list:
    """Build feature vector aligned with train_lgbm.py (16 features)."""
    USAGE_TREND_MIN, USAGE_TREND_MAX = 0.90, 1.50
    # Must match train_lgbm.py: usage_trend = clip(recent_min / avg_min, ...)
    usage = float(
        np.clip(recent_min / max(avg_min, 1), USAGE_TREND_MIN, USAGE_TREND_MAX)
    )
    sign = 1.0 if side == "away" else -1.0
    opp_def_rating = 112.0 + sign * (spread or 0) * 0.7
    ast_rate_ = ast / max(avg_min, 1)
    def_rate_ = (stl + blk) / max(avg_min, 1)
    pts_per_min_ = pts / max(avg_min, 1)
    home_away_ = 1.0 if side == "home" else 0.0
    rest_days_ = 2.0
    recent_vs_season_ = float(np.clip(recent_pts / max(season_pts, 1), 0.5, 2.0))
    games_played_ = float(games_played) if games_played is not None else 40.0
    reb_per_min_ = float(np.clip(reb / max(avg_min, 1), 0.0, 1.5))
    # Inference proxies for last-3 vs last-5 scoring shape (correlate with training features).
    l3_vs_l5_pts = float(np.clip((0.55 * recent_pts + 0.45 * season_pts) / max(recent_pts, 0.1), 0.4, 2.5))
    min_volatility = float(
        np.clip(abs(recent_min - season_min) / max(season_min, 1.0), 0.0, 1.2)
    )
    starter_proxy = 1.0 if avg_min >= 26.0 else 0.0
    cascade_signal = float(np.clip(max(cascade_bonus, 0.0) / 15.0, 0.0, 1.5))
    return [
        avg_min,
        season_pts,
        usage,
        opp_def_rating,
        home_away_,
        ast_rate_,
        def_rate_,
        pts_per_min_,
        rest_days_,
        recent_vs_season_,
        games_played_,
        reb_per_min_,
        l3_vs_l5_pts,
        min_volatility,
        starter_proxy,
        cascade_signal,
    ]

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CACHE UTILITIES
# Module-level: UPPER_SNAKE = public constants; _lower = private (do not mutate).
# grep: ESPN, MIN_GATE, DEFAULT_TOTAL, _cp, _cg, _cs, _lp, _lg, _ls
# _cg/cs = prediction cache (date-keyed, /tmp); _lg/ls = lock cache (warm instance).
# ─────────────────────────────────────────────────────────────────────────────
ESPN      = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
OPUS_MODEL  = "claude-opus-4-6"
MIN_GATE  = 12          # Minimum projected minutes — lowered from 15 to catch
                        # deep bench (Clifford, Riley) who win in garbage time
DEFAULT_TOTAL = 222     # Fallback over/under when odds unavailable

# Map ESPN team abbreviations to keyword fragments in Odds API full team names
_ABBR_TO_NAME_FRAG = {
    "ATL": "atlanta",    "BOS": "boston",       "BKN": "brooklyn",    "CHA": "charlotte",
    "CHI": "chicago",    "CLE": "cleveland",    "DAL": "dallas",      "DEN": "denver",
    "DET": "detroit",    "GSW": "golden state", "HOU": "houston",     "IND": "indiana",
    "LAC": "clippers",   "LAL": "lakers",       "MEM": "memphis",     "MIA": "miami",
    "MIL": "milwaukee",  "MIN": "minnesota",    "NOP": "new orleans", "NY":  "new york",
    "NYK": "new york",   "OKC": "oklahoma",     "ORL": "orlando",     "PHI": "philadelphia",
    "PHX": "phoenix",    "POR": "portland",     "SAC": "sacramento",  "SAS": "san antonio",
    "TOR": "toronto",    "UTA": "utah",         "WAS": "washington",
}

def _abbr_matches(abbr: str, full_name: str) -> bool:
    frag = _ABBR_TO_NAME_FRAG.get(abbr.upper(), abbr.lower())
    return frag in full_name.lower()

_STAT_MARKET = {
    "points":   "player_points",
    "rebounds": "player_rebounds",
    "assists":  "player_assists",
}

def _cp(k, date_str=None):
    """Cache path for key k. date_str: optional slate date (YYYY-MM-DD) for midnight-rollover correctness."""
    d = date_str or _et_date().isoformat()
    return CACHE_DIR / f"{hashlib.md5(f'{d}:{k}'.encode()).hexdigest()}.json"
def _cg(k, date_str=None): return json.loads(_cp(k, date_str).read_text()) if _cp(k, date_str).exists() else None
def _cs(k, v, date_str=None): _cp(k, date_str).write_text(json.dumps(v))
def _lp(k, date_str=None):
    """Lock path for key k. date_str: optional slate date for midnight-rollover correctness."""
    d = date_str or _et_date().isoformat()
    return LOCK_DIR / f"{hashlib.md5(f'{d}:{k}'.encode()).hexdigest()}.json"
def _lg(k, date_str=None): return json.loads(_lp(k, date_str).read_text()) if _lp(k, date_str).exists() else None
def _ls(k, v, date_str=None): _lp(k, date_str).write_text(json.dumps(v))




# grep: LOCK HELPERS — _is_locked, _is_past_lock_window, _et_date
def _is_locked(start_time_iso: Optional[str]) -> bool:
    """Returns True if current UTC time is within lock_buffer_minutes of game start.
    Returns False only for games >6h past start (well past any possible final buzzer)."""
    try:
        lock_buf = _cfg("projection.lock_buffer_minutes", 5)
        game_start = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        # 6h ceiling: longest possible NBA game (regulation + multiple OTs) < 4h.
        # The old 3h ceiling was causing games to appear "unlocked" while still live.
        if now > game_start + timedelta(hours=6):
            return False
        return now >= game_start - timedelta(minutes=lock_buf)
    except Exception:
        return False

def _is_past_lock_window(start_time_iso):
    """Returns True if the game has passed its lock window (now >= start - lock_buf).
    Intentionally has no 6-hour ceiling: once a game is past the lock window it is
    never draftable again, even after it ends. Contrast with _is_locked(), which
    returns False after 6h so the UI stops showing the game as "locked".
    Does NOT check ESPN completion status — use _all_games_final() for that."""
    try:
        lock_buf = _cfg("projection.lock_buffer_minutes", 5)
        game_start = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return now >= game_start - timedelta(minutes=lock_buf)
    except Exception:
        return False

def _et_date() -> str:
    """Current date in Eastern Time (handles EST/EDT)."""
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date()
    except ImportError:
        # Fallback: EST=UTC-5 (Nov–Mar), EDT=UTC-4 (Mar–Nov)
        now_utc = datetime.now(timezone.utc)
        offset = timedelta(hours=-4 if 3 < now_utc.month < 11 else -5)
        return (now_utc + offset).date()

# ─────────────────────────────────────────────────────────────────────────────
# ESPN DATA FETCHERS
# grep: _espn_get, fetch_games, fetch_roster, _fetch_athlete, _fetch_b2b_teams, _fetch_team_def_stats
# _fetch_athlete: returns blended season+recent stats dict for a player
# fetch_games: returns today's game list with lock/complete status
# ─────────────────────────────────────────────────────────────────────────────
def _safe_float(v: Any, default: float = 0.0) -> float:
    try: return float(v) if v is not None else default
    except (ValueError, TypeError): return default

def _espn_get(url):
    try:
        r = requests.get(url, timeout=10)
        if not r.ok: return {}
        return r.json()
    except (requests.RequestException, ValueError): return {}

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
        comp = ev.get("competitions", [{}])[0]
        for cd in comp.get("competitors", []):
            abbr = cd.get("team", {}).get("abbreviation", "")
            if abbr: b2b.add(abbr)
    _cs(cache_key, list(b2b))
    return b2b

def _fetch_team_def_stats() -> dict:
    """Fetch NBA team defensive stats (pts allowed per game) from ESPN standings.
    Returns {abbr: {"pts_allowed": float}} e.g. {"SAC": {"pts_allowed": 118.5}}.
    Cached for the calendar day (ET). On any failure returns {} —
    callers fall back to league avg (115.0), yielding a neutral matchup factor.
    """
    cache_key = f"team_def_stats_{_et_date().strftime('%Y%m%d')}"
    cached = _cg(cache_key)
    if cached is not None:
        return cached
    data = _espn_get("https://site.api.espn.com/apis/v2/sports/basketball/nba/standings")
    result = {}
    try:
        for entry in data.get("standings", {}).get("entries", []):
            abbr = entry.get("team", {}).get("abbreviation", "")
            if not abbr:
                continue
            stats_dict = {}
            for s in entry.get("stats", []):
                if "value" in s:
                    try:
                        stats_dict[s["name"]] = float(s["value"])
                    except (TypeError, ValueError):
                        pass
            pts_allowed = (
                stats_dict.get("avgPointsAgainst")
                or stats_dict.get("pointsAgainst")
                or stats_dict.get("avgPointsAllowed")
                or stats_dict.get("opponentAvgPoints")
            )
            if pts_allowed and pts_allowed > 50:  # sanity check — must be pts/game not a rate
                result[abbr] = {"pts_allowed": pts_allowed}
    except Exception:
        pass
    if result:
        _cs(cache_key, result)
    return result


def _to_float(val, default=None):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _to_int(val, default=None):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def _team_tier_from_standings(stats_dict: dict, mot_cfg: dict) -> str:
    """Infer team motivation tier from standings stats.
    A: meaningful seeding/play-in pressure
    B: neutral
    C: low-incentive / development profile
    """
    seed = _to_int(
        stats_dict.get("playoffSeed")
        or stats_dict.get("seed")
        or stats_dict.get("rank"),
        None,
    )
    games_back = _to_float(
        stats_dict.get("gamesBack")
        or stats_dict.get("gb"),
        None,
    )
    win_pct = _to_float(
        stats_dict.get("winPercent")
        or stats_dict.get("winPct")
        or stats_dict.get("winningPercentage"),
        None,
    )

    seeding_gap = _to_float(mot_cfg.get("seeding_gap_games", 2.0), 2.0)
    playin_gap = _to_float(mot_cfg.get("playin_gap_games", 2.0), 2.0)
    elimination_buffer = _to_float(mot_cfg.get("elimination_buffer_games", 3.0), 3.0)

    # Neutral default when stats are incomplete.
    if seed is None:
        return "B"

    # Clear low-incentive bucket (bottom standings with large gap).
    if seed >= 13:
        if games_back is None or games_back > elimination_buffer:
            return "C"

    # Play-in pressure zone.
    if 7 <= seed <= 10:
        return "A"

    # Bubble teams just outside play-in.
    if 11 <= seed <= 12 and games_back is not None and games_back <= playin_gap:
        return "A"

    # Teams in top-6 that are still in a tight seeding race.
    if 1 <= seed <= 6 and games_back is not None and games_back <= seeding_gap:
        return "A"

    # Fallback low-incentive heuristic when standings are weakly informative.
    if seed >= 12 and win_pct is not None and win_pct < 0.40:
        return "C"

    return "B"


def _fetch_team_motivation_map() -> dict:
    """Build {TEAM_ABBR: {tier, chalk_mult, moonshot_mult}} for current ET date."""
    mot_cfg = _cfg("team_motivation", _CONFIG_DEFAULTS.get("team_motivation", {})) or {}
    if not mot_cfg.get("enabled", False):
        return {}

    start_date = str(mot_cfg.get("start_date", "") or "").strip()
    if start_date:
        try:
            if _et_date() < datetime.fromisoformat(start_date).date():
                return {}
        except ValueError:
            # Invalid config date should fail open (neutral behavior).
            return {}

    cache_key = f"team_motivation_{_et_date().strftime('%Y%m%d')}"
    cached = _cg(cache_key)
    if cached is not None:
        return cached

    data = _espn_get("https://site.api.espn.com/apis/v2/sports/basketball/nba/standings")
    result = {}
    try:
        for entry in data.get("standings", {}).get("entries", []):
            abbr = entry.get("team", {}).get("abbreviation", "")
            if not abbr:
                continue
            stats_dict = {}
            for s in entry.get("stats", []):
                name = s.get("name")
                if not name:
                    continue
                val = s.get("value", s.get("displayValue"))
                stats_dict[name] = val

            tier = _team_tier_from_standings(stats_dict, mot_cfg)
            tier_key = tier.lower()
            chalk_mult = _to_float(mot_cfg.get(f"tier_{tier_key}_mult_chalk", 1.0), 1.0)
            moon_mult = _to_float(mot_cfg.get(f"tier_{tier_key}_mult_moonshot", 1.0), 1.0)
            mn = _to_float(mot_cfg.get("min_mult", 0.88), 0.88)
            mx = _to_float(mot_cfg.get("max_mult", 1.12), 1.12)
            result[abbr] = {
                "tier": tier,
                "chalk_mult": max(mn, min(mx, chalk_mult)),
                "moonshot_mult": max(mn, min(mx, moon_mult)),
            }
    except Exception:
        return {}

    # Manual tier overrides win over inferred tiers.
    overrides = mot_cfg.get("team_overrides", {}) if isinstance(mot_cfg.get("team_overrides", {}), dict) else {}
    for abbr, forced in overrides.items():
        if not isinstance(abbr, str):
            continue
        tier = str(forced or "").strip().upper()
        if tier not in ("A", "B", "C"):
            continue
        tier_key = tier.lower()
        chalk_mult = _to_float(mot_cfg.get(f"tier_{tier_key}_mult_chalk", 1.0), 1.0)
        moon_mult = _to_float(mot_cfg.get(f"tier_{tier_key}_mult_moonshot", 1.0), 1.0)
        mn = _to_float(mot_cfg.get("min_mult", 0.88), 0.88)
        mx = _to_float(mot_cfg.get("max_mult", 1.12), 1.12)
        result[abbr.upper()] = {
            "tier": tier,
            "chalk_mult": max(mn, min(mx, chalk_mult)),
            "moonshot_mult": max(mn, min(mx, moon_mult)),
        }

    if result:
        _cs(cache_key, result)
    return result


def _team_motivation_multiplier(team_abbr: str, lineup_type: str, motivation_map: dict) -> float:
    """Return motivation multiplier for team/lineup type (defaults to neutral 1.0)."""
    if not team_abbr or not motivation_map:
        return 1.0
    rec = motivation_map.get(team_abbr, {})
    if lineup_type == "chalk":
        return float(rec.get("chalk_mult", 1.0))
    return float(rec.get("moonshot_mult", 1.0))

_GAMES_CACHE_TS: dict = {}  # cache_key → timestamp for TTL enforcement

def fetch_games(date=None):
    """Fetch today's (or a specific date's) NBA schedule from ESPN.
    date: a datetime.date object, or None for today ET.
    Cache TTL: 5 minutes — game list rarely changes, but stale data causes
    downstream issues (missing games in Log, wrong lock status)."""
    today_et = date or _et_date()
    cache_key = f"games_{today_et}"
    c = _cg(cache_key)
    if c:
        cached_at = _GAMES_CACHE_TS.get(cache_key, 0)
        if time.time() - cached_at < 300:  # 5 min TTL
            return c
    b2b_teams = _fetch_b2b_teams()
    date_str = today_et.strftime("%Y%m%d")
    data = _espn_get(f"{ESPN}/scoreboard?dates={date_str}")
    games = []
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
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
    _GAMES_CACHE_TS[cache_key] = time.time()
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
        if   "min" in k:                      s["min"] = _safe_float(val)
        elif "pts" in k or "point" in k:      s["pts"] = _safe_float(val)
        elif "reb" in k or "rebound" in k:    s["reb"] = _safe_float(val)
        elif "ast" in k or "assist" in k:     s["ast"] = _safe_float(val)
        elif "stl" in k or "steal" in k:      s["stl"] = _safe_float(val)
        elif "blk" in k or "block" in k:      s["blk"] = _safe_float(val)
        elif "tov" in k or "turnover" in k:   s["tov"] = _safe_float(val)
        elif k in ("gp", "g", "gamesplayed"): s["gp"]  = _safe_float(val)
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
        recent_raw_min = None  # Raw recent avg min — captured even if below usable threshold
        for split in splits[1:]:
            label = (str(split.get("displayName","")) + str(split.get("type",""))).lower()
            if any(kw in label for kw in ["last","recent","l5","l10","l3"]):
                c2 = _parse_split(names, split)
                recent_raw_min = c2.get("min", 0)  # Always capture, even if < 10
                if c2["min"] >= 10:
                    recent = c2
                break
        if recent is None and len(splits) > 1:
            c2 = _parse_split(names, splits[1])
            if recent_raw_min is None:
                recent_raw_min = c2.get("min", 0)
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
            blended["recent_reb"] = recent["reb"]
            blended["season_reb"] = season["reb"]
            blended["recent_ast"] = recent["ast"]
            blended["season_ast"] = season["ast"]
            blended["recent_stl"] = recent["stl"]
            blended["season_stl"] = season["stl"]
            blended["recent_blk"] = recent["blk"]
            blended["season_blk"] = season["blk"]
        else:
            blended = dict(season)
            blended["season_min"] = season["min"]
            # Use actual recent minutes when we have them (even if < 10). When there is
            # no recent split at all (recent_raw_min is None), do NOT use season_min as
            # proxy — that let Olynyk/Plumlee etc. pass the 20-min chalk filter. Use 0
            # so chalk/moonshot filters exclude them until we have real recent data.
            blended["recent_min"] = recent_raw_min if recent_raw_min is not None else 0.0
            blended["recent_pts"] = season["pts"]
            blended["season_pts"] = season["pts"]
            blended["recent_reb"] = season["reb"]
            blended["season_reb"] = season["reb"]
            blended["recent_ast"] = season["ast"]
            blended["season_ast"] = season["ast"]
            blended["recent_stl"] = season["stl"]
            blended["season_stl"] = season["stl"]
            blended["recent_blk"] = season["blk"]
            blended["season_blk"] = season["blk"]

        # DNP risk flag: if the most recent split showed very low avg minutes,
        # the player has been near-inactive lately (rest management, rotation bubble,
        # coach's decision). High card boost + DNP risk = the trap that lost March 4th.
        # Threshold is configurable; default 8 min = effectively on bench/DNP.
        proj_cfg = _cfg("projection", _CONFIG_DEFAULTS["projection"])
        dnp_thresh = proj_cfg.get("dnp_risk_min_threshold", 8.0)
        if recent_raw_min is not None and recent_raw_min < dnp_thresh:
            blended["dnp_risk"] = True

    except Exception as e:
        print(f"Stat parse error pid={pid}: {e}")
        return None
    _cs(f"ath3_{pid}", blended)
    return blended

# ─────────────────────────────────────────────────────────────────────────────
# LIVE NBA DATA FETCHERS — used by Ben tool use
# grep: _live_scores, _live_boxscore, _live_player_stats, BEN_TOOL
# ─────────────────────────────────────────────────────────────────────────────

def _live_scores():
    """Current NBA scoreboard: scores, quarter, time remaining, game IDs."""
    try:
        today_str = _et_date().strftime("%Y%m%d")
        data = _espn_get(f"{ESPN}/scoreboard?dates={today_str}")
        if not data:
            return "ESPN scoreboard unavailable."
        lines = []
        for e in data.get("events", []):
            game_id = e.get("id", "")
            comps   = e.get("competitions", [{}])[0]
            status  = e.get("status", {})
            detail  = status.get("type", {}).get("shortDetail", "")
            teams   = comps.get("competitors", [])
            parts   = []
            for t in sorted(teams, key=lambda x: x.get("homeAway", ""), reverse=True):
                abbr  = t.get("team", {}).get("abbreviation", "?")
                score = t.get("score", "0")
                parts.append(f"{abbr} {score}")
            score_str = " - ".join(parts)
            lines.append(f"{score_str}  [{detail}]  (game_id: {game_id})")
        return "\n".join(lines) if lines else "No games found today."
    except Exception as e:
        return f"Error fetching scores: {e}"


def _live_boxscore(game_id):
    """Live player stats for a specific ESPN game ID."""
    try:
        data = _espn_get(f"{ESPN}/summary?event={game_id}")
        if not data:
            return "Game summary unavailable."
        lines = []
        for team_block in data.get("boxscore", {}).get("players", []):
            abbr   = team_block.get("team", {}).get("abbreviation", "?")
            stats  = team_block.get("statistics", [])
            if not stats:
                continue
            labels   = stats[0].get("labels", [])
            athletes = stats[0].get("athletes", [])
            want     = {"MIN", "PTS", "REB", "AST", "STL", "BLK", "TO", "+/-"}
            idx_map  = {l: i for i, l in enumerate(labels) if l in want}
            lines.append(f"\n{abbr}:")
            for ath in athletes:
                name  = ath.get("athlete", {}).get("displayName", "?")
                vals  = ath.get("stats", [])
                parts = []
                for lbl in ["MIN", "PTS", "REB", "AST", "STL", "BLK", "TO"]:
                    if lbl in idx_map and idx_map[lbl] < len(vals):
                        parts.append(f"{lbl} {vals[idx_map[lbl]]}")
                lines.append(f"  {name}: {' | '.join(parts)}")
        return "\n".join(lines) if lines else "No boxscore data available yet."
    except Exception as e:
        return f"Error fetching boxscore: {e}"


def _live_player_stats(player_name):
    """Search all today's games for a specific player's live stats."""
    try:
        data = _espn_get(f"{ESPN}/scoreboard")
        if not data:
            return "ESPN unavailable."
        player_lower = player_name.lower().strip()
        for e in data.get("events", []):
            game_id   = e.get("id", "")
            game_name = e.get("name", "")
            box       = _live_boxscore(game_id)
            for line in box.split("\n"):
                if player_lower in line.lower():
                    return f"{game_name}:\n{line.strip()}"
        return f"No live stats found for '{player_name}'. They may not be in today's lineup or the game hasn't started."
    except Exception as e:
        return f"Error: {e}"


# Tool definition passed to Claude in lab_chat
_BEN_TOOLS = [
    {
        "name": "get_live_nba_data",
        "description": (
            "Fetch real-time NBA data from ESPN. Call this whenever the user asks about "
            "live scores, current game status, a player's stats tonight, or any information "
            "that requires up-to-the-minute data beyond what's in the system prompt."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "data_type": {
                    "type": "string",
                    "enum": ["scores", "boxscore", "player_stats"],
                    "description": (
                        "'scores' — current scoreboard for all today's games; "
                        "'boxscore' — full player stat lines for one game (requires game_id); "
                        "'player_stats' — find a specific player's live stats (requires player_name)"
                    ),
                },
                "game_id": {
                    "type": "string",
                    "description": "ESPN game ID — use for 'boxscore' queries. Get IDs from scores first if needed.",
                },
                "player_name": {
                    "type": "string",
                    "description": "Full or partial player name — used for 'player_stats' queries.",
                },
            },
            "required": ["data_type"],
        },
    },
    {
        "name": "read_repo_file",
        "description": (
            "Read any file from the GitHub repository. Use this to inspect the full source "
            "code (api/index.py, api/real_score.py, api/line_engine.py, api/asset_optimizer.py, "
            "index.html), prediction/actuals CSVs, audit JSONs, model config, or any other file. "
            "Always read a code file before proposing or making changes to it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Repo-relative path, e.g. 'api/index.py', 'data/predictions/2026-03-05.csv', 'data/model-config.json'",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "list_repo_directory",
        "description": (
            "List files and subdirectories in a GitHub repository directory. "
            "Use to discover available CSVs, audit files, or code modules."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path, e.g. 'api', 'data/predictions', 'data/actuals'",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_repo_file",
        "description": (
            "Modify the model config file only. Ben is authorized to write ONLY to "
            "data/model-config.json. Do not attempt to write api/, index.html, "
            ".github/, or any other path — those require a developer code push. "
            "ALWAYS call read_repo_file first to get the current content. "
            "Make targeted parameter changes only. "
            "Summarize what you changed and why in the commit message."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Must be 'data/model-config.json'. No other paths are permitted.",
                },
                "content": {
                    "type": "string",
                    "description": "Complete new file content",
                },
                "commit_message": {
                    "type": "string",
                    "description": "Clear commit message describing what changed and why",
                },
            },
            "required": ["path", "content", "commit_message"],
        },
    },
]


_BEN_WRITABLE_PATHS = {"data/model-config.json"}


def _execute_ben_tool(name, inp):
    """Execute a Ben tool call and return a string result."""
    if name == "get_live_nba_data":
        dt = inp.get("data_type", "scores")
        if dt == "scores":
            return _live_scores()
        if dt == "boxscore":
            gid = inp.get("game_id", "")
            return _live_boxscore(gid) if gid else "game_id is required for boxscore queries."
        if dt == "player_stats":
            pn = inp.get("player_name", "")
            return _live_player_stats(pn) if pn else "player_name is required for player_stats queries."
        return f"Unknown data_type: {dt}"

    if name == "read_repo_file":
        path = inp.get("path", "")
        if not path:
            return "path is required."
        content, _ = _github_get_file(path)
        if content is None:
            return f"File not found: {path}"
        # Truncate very large files (> 60k chars) to avoid token overflow
        if len(content) > 60000:
            return content[:60000] + f"\n\n[...TRUNCATED — file is {len(content)} chars total. Request a specific section if needed.]"
        return content

    if name == "list_repo_directory":
        path = inp.get("path", "")
        items = _github_list_dir(path)
        if not items:
            return f"Directory empty or not found: {path}"
        return json.dumps([{"name": i.get("name"), "type": i.get("type", "file"), "size": i.get("size")} for i in items], indent=2)

    if name == "write_repo_file":
        path    = inp.get("path", "")
        content = inp.get("content", "")
        message = inp.get("commit_message", "Ben: update file")
        if not path or not content:
            return "path and content are required."
        if path not in _BEN_WRITABLE_PATHS:
            return (
                f"Ben is not authorized to write to '{path}'. "
                "Ben can only modify data/model-config.json. "
                "For code or workflow changes, describe what should change "
                "and a developer will implement it via a code push."
            )
        result = _github_write_file(path, content, f"[Ben] {message}")
        if result.get("error"):
            return f"Write failed: {result['error']}"
        # Clear config cache so changes take effect immediately
        try:
            (CONFIG_CACHE_DIR / "model_config.json").unlink(missing_ok=True)
        except Exception: pass
        return f"Written successfully: {path} — config live within 5 minutes."

    return f"Unknown tool: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# INJURY CASCADE ENGINE
# grep: _cascade_minutes, _pos_group, POS_GROUPS, injury redistribution
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
# CARD BOOST & DFS SCORING
# grep: _est_card_boost, _dfs_score, card boost, ownership, Real Score formula
#
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
# STARTING 5: MILP optimizes reliability-weighted EV with capped boost exposure.
# MOONSHOT: separate contrarian MILP (two-phase) that prioritizes ceiling while
#           still enforcing floor/rotation guardrails.
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_boost_name(name):
    """Normalize player name for card boost lookup — strips diacritics, lowercases,
    removes punctuation variants (apostrophes, periods, Jr./III suffixes)."""
    import unicodedata as _ud
    n = _ud.normalize("NFKD", name).encode("ASCII", "ignore").decode("ASCII")
    return n.strip().lower()


# ── Pre-game boost ingestion (Layer 0) ─────────────────────────────────────
# Boosts are fixed daily constants published by Real Sports before drafts open.
# When ingested via /api/save-boosts, they become the ground truth for today's
# pipeline — no estimation needed. Stored in data/boosts/{date}.json on GitHub.
# grep: BOOST INGESTION, _load_daily_boosts, save-boosts
_DAILY_BOOST_CACHE = {}   # {normalized_name: boost_value}
_DAILY_BOOST_DATE = ""    # ET date string for cache validity
_DAILY_BOOST_TS = 0       # timestamp of last load

def _load_daily_boosts(date_str=None):
    """Load pre-game boosts for today from data/boosts/{date}.json.
    Returns {normalized_name: boost_value} dict. Cached 5 min."""
    global _DAILY_BOOST_CACHE, _DAILY_BOOST_DATE, _DAILY_BOOST_TS
    import time
    now = time.time()
    if date_str is None:
        date_str = _et_date().isoformat()
    if (_DAILY_BOOST_CACHE and _DAILY_BOOST_DATE == date_str
            and (now - _DAILY_BOOST_TS) < 300):
        return _DAILY_BOOST_CACHE
    # Try GitHub
    try:
        content, _ = _github_get_file(f"data/boosts/{date_str}.json")
        if content:
            data = json.loads(content)
            players = data.get("players", [])
            result = {}
            for p in players:
                name = p.get("player_name") or p.get("name", "")
                boost = p.get("boost")
                if name and boost is not None:
                    nk = _normalize_boost_name(name)
                    result[nk] = float(boost)
            _DAILY_BOOST_CACHE = result
            _DAILY_BOOST_DATE = date_str
            _DAILY_BOOST_TS = now
            return result
    except Exception as e:
        print(f"[boosts] load failed for {date_str}: {e}")
    # No boosts file for today — return empty (pipeline falls through to estimation)
    _DAILY_BOOST_CACHE = {}
    _DAILY_BOOST_DATE = date_str
    _DAILY_BOOST_TS = now
    return {}


# ── Auto-populated boost overrides from ownership/actuals data ──────────────
# Reads data/ownership/ and data/actuals/ CSVs to build a dynamic override map
# using the most recent observation per player. Refreshes every 10 minutes.
_OWNERSHIP_BOOST_CACHE = {}
_OWNERSHIP_BOOST_TS = 0

def _load_ownership_boosts():
    """Build {normalized_name: boost} from all ownership + actuals CSVs.

    Uses most recent observation per player with trend extrapolation:
    - For players with 3+ observations showing consistent decline (≥0.3 total drop),
      extrapolates one step forward at 50% of the observed decay rate.
    - Never inflates above the most-recent observed value.
    - Extrapolation capped at 0.3 below most-recent to avoid over-correcting.

    Cached 10 min.
    """
    global _OWNERSHIP_BOOST_CACHE, _OWNERSHIP_BOOST_TS
    import time
    now = time.time()
    if _OWNERSHIP_BOOST_CACHE and (now - _OWNERSHIP_BOOST_TS) < 600:
        return _OWNERSHIP_BOOST_CACHE

    # Collect all observations per player: {nkey: {date_str: (boost_val, original_name)}}
    # Using a dict keyed by date to automatically deduplicate (ownership + actuals may
    # both have data for the same player on the same date — keep the latest file's value).
    raw_obs = {}  # {normalized_name: {date_str: (boost_val, original_name)}}
    for subdir in ["ownership", "actuals"]:
        dirpath = os.path.join(os.path.dirname(__file__), "..", "data", subdir)
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            if not fname.endswith(".csv"):
                continue
            date_str = fname.replace(".csv", "")
            fpath = os.path.join(dirpath, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    import csv as _csv
                    reader = _csv.DictReader(f)
                    for row in reader:
                        # ownership CSVs use 'player', actuals use 'player_name'
                        name = row.get("player") or row.get("player_name", "")
                        if not name:
                            continue
                        boost_str = row.get("actual_card_boost", "")
                        if not boost_str or boost_str == "None":
                            continue
                        try:
                            bval = float(boost_str)
                        except (ValueError, TypeError):
                            continue
                        if bval <= 0:
                            continue
                        nkey = _normalize_boost_name(name)
                        raw_obs.setdefault(nkey, {})[date_str] = (bval, name)
            except Exception:
                continue

    result = {}
    for nkey, date_map in raw_obs.items():
        # Build deduplicated, date-sorted observation list
        obs_list = sorted(date_map.items())  # [(date_str, (bval, name)), ...]
        latest_date, (latest_boost, _name) = obs_list[-1]

        # Trend extrapolation: if 3+ observations with consistent decline ≥ 0.3 total,
        # apply 50% of average per-step decay as a forward estimate.
        if len(obs_list) >= 3:
            boosts = [bval for _, (bval, _) in obs_list[-3:]]  # last 3
            diffs = [boosts[i] - boosts[i+1] for i in range(len(boosts)-1)]
            # Consistent decline: all diffs positive (each step lower) and total ≥ 0.3
            if all(d > 0 for d in diffs) and sum(diffs) >= 0.3:
                avg_decay = sum(diffs) / len(diffs)
                # Apply 50% of avg decay as forward extrapolation
                extrapolated = latest_boost - (avg_decay * 0.5)
                # Never drop more than 0.3 below last observed, never inflate
                extrapolated = max(extrapolated, latest_boost - 0.3)
                extrapolated = min(extrapolated, latest_boost)
                result[nkey] = round(extrapolated, 1)
                continue

        result[nkey] = latest_boost

    _OWNERSHIP_BOOST_CACHE = result
    _OWNERSHIP_BOOST_TS = now
    return _OWNERSHIP_BOOST_CACHE


def _est_card_boost(proj_min, pts, team_abbr, player_name=None):
    """Get or estimate ADDITIVE card boost for Real Sports.

    Four-layer approach (v51):
    0. DAILY INGESTION — pre-game boosts from data/boosts/{date}.json (ground truth).
       When available, this is the definitive value — no estimation needed.
    1. OBSERVED DATA — auto-populated from data/actuals/ + data/ownership/ CSVs.
       Most recent observation + trend extrapolation for consistently declining players.
       Refreshed every 10 min. Takes precedence over manual config overrides because
       real observations are always more accurate than static estimates.
    2. CONFIG OVERRIDES — player_overrides from model-config.json.
       Fallback for players not yet observed in actuals (pre-season or rarely played).
    3. SIGMOID TIER FALLBACK — for completely unknown players, estimate from PPG.
    """
    cb = _cfg("card_boost", _CONFIG_DEFAULTS["card_boost"])
    ceiling   = cb.get("ceiling", 3.0)
    floor_val = cb.get("floor", 0.2)

    norm_name = _normalize_boost_name(player_name) if player_name else None

    # Layer 0: Pre-game daily boost ingestion (ground truth when available)
    # Boosts are fixed daily constants published by Real before drafts open.
    if norm_name:
        daily_boosts = _load_daily_boosts()
        if norm_name in daily_boosts:
            return round(min(max(daily_boosts[norm_name], floor_val), ceiling), 1)

    # Layer 1: Observed from actuals/ownership CSVs (most recent + trend adjustment)
    # Real observations beat static config estimates — avoids stale manual overrides.
    if norm_name:
        ownership_boosts = _load_ownership_boosts()
        if norm_name in ownership_boosts:
            return round(min(max(ownership_boosts[norm_name], floor_val), ceiling), 1)

    # Layer 2: Config overrides (fallback for players not yet observed in actuals)
    overrides = cb.get("player_overrides", {})
    if norm_name:
        for k, v in overrides.items():
            if _normalize_boost_name(k) == norm_name:
                return float(v)

    # Layer 3: Sigmoid tier estimation from PPG
    # boost = sig_ceiling - sig_range × sigmoid((PPG - sig_midpoint) / sig_scale)
    # Calibrated against 60+ observations from Mar 6/9/10 actuals:
    #   PPG 27 → 0.26, PPG 20 → 0.53, PPG 15 → 1.10, PPG 12 → 1.60,
    #   PPG 10 → 1.94, PPG 8 → 2.25, PPG 5 → 2.59
    sig_ceiling  = cb.get("sig_ceiling", 3.0)
    sig_range    = cb.get("sig_range", 2.8)
    sig_midpoint = cb.get("sig_midpoint", 12.0)
    sig_scale    = cb.get("sig_scale", 4.0)
    bm_discount  = cb.get("big_market_discount", 0.15)
    big_markets  = set(cb.get("big_market_teams", ["LAL","GS","GSW","BOS","NY","NYK","PHI","MIA","DEN","LAC","CHI"]))

    sigmoid_val = 1.0 / (1.0 + np.exp(-(pts - sig_midpoint) / sig_scale))
    boost = sig_ceiling - sig_range * sigmoid_val

    # Big market players are more recognizable → slightly lower boost
    if team_abbr in big_markets:
        boost -= bm_discount

    return round(min(max(boost, floor_val), ceiling), 1)

def _dfs_score(pts, reb, ast, stl, blk, tov):
    """Real Score-aligned formula. Weights read from runtime config."""
    w = _cfg("real_score.dfs_weights", {"pts":2.5,"reb":0.5,"ast":1.0,"stl":2.0,"blk":1.5,"tov":-1.5})
    return (pts * w.get("pts", 2.5) + reb * w.get("reb", 0.5) +
            ast * w.get("ast", 1.0) + stl * w.get("stl", 2.0) +
            blk * w.get("blk", 1.5) + tov * w.get("tov", -1.5))


# ─────────────────────────────────────────────────────────────────────────────
# GAME SCRIPT ENGINE
# grep: _game_script_weights, _game_script_dfs, _game_script_label, game pace
# Per-game only — does NOT affect full slate projections.
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


def _infer_player_archetype(pts: float, avg_min: float, reb: float, stats: dict) -> str:
    """Coarse role bucket for RS archetype calibration (not position-specific)."""
    season_pts = float(stats.get("season_pts", pts) or pts)
    recent_pts = float(stats.get("recent_pts", pts) or pts)
    reb_pm = reb / max(avg_min, 1)
    recent_vs = recent_pts / max(season_pts, 1.0)
    if season_pts >= 21.0 and avg_min >= 28.0:
        return "star"
    # Pure rebounder: high reb rate but limited scoring — Lopez, Gobert, Capela.
    # These players over-project because reb volume inflates DFS score more than RS justifies.
    if reb_pm >= 0.28 and season_pts < 12.0:
        return "pure_rebounder"
    if reb_pm >= 0.22:
        return "big"
    # Efficient scorer: high pts/min with real volume — Jalen Green, DeRozan, Nesmith.
    # These players under-project; when shots fall they generate RS 4.5–7.
    ppm = pts / max(avg_min, 1)
    if ppm >= 0.55 and season_pts >= 15.0:
        return "scorer"
    if avg_min < 22.0 and recent_vs >= 1.12:
        return "bench_microwave"
    if avg_min >= 28.0:
        return "starter"
    return "wing_role"


def _apply_post_lock_rs_calibration(projections: list, *, slate_locked: bool) -> None:
    """Re-rank tilt from recency + cascade right before MILP (config-gated)."""
    cfg = _cfg("real_score.post_lock_calibration", _CONFIG_DEFAULTS["real_score"].get("post_lock_calibration", {}))
    if not cfg.get("enabled", False):
        return
    if cfg.get("require_locked_slate", True) and not slate_locked:
        return
    strength = float(cfg.get("recency_strength", 0.12))
    max_nudge = float(cfg.get("max_nudge", 0.12))
    cascade_w = float(cfg.get("cascade_weight", 0.06))
    avg_slot = float(_cfg("lineup.avg_slot_multiplier", 1.6))
    for p in projections:
        season_pts = max(float(p.get("season_pts", 0) or p.get("pts", 0)), 0.1)
        recent_pts = float(p.get("recent_pts", season_pts))
        r = recent_pts / season_pts
        nudge = 1.0 + max(-max_nudge, min(max_nudge, (r - 1.0) * strength))
        cb = float(p.get("_cascade_bonus", 0) or 0)
        if cb > 4.0:
            nudge *= 1.0 + min(cascade_w, cb / 200.0)
        if abs(nudge - 1.0) < 0.001:
            continue
        old = float(p.get("rating", 0))
        p["rating"] = round(old * nudge, 1)
        p["chalk_ev"] = round(p["rating"] * (avg_slot + float(p.get("est_mult", 0))), 2)
        p["ceiling_score"] = round(float(p.get("ceiling_score", p["rating"])) * nudge, 1)
        p["_post_lock_rs_nudge"] = round(nudge, 4)


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER PROJECTION ENGINE
# grep: project_player, pinfo, stats, spread, total, rating, est_mult, blk, stl
# Returns projection dict: {name, team, pos, rating (RS), est_mult (card boost),
#   predMin, pts, reb, ast, stl, blk, season_*/recent_* raw stats, signals}
# ─────────────────────────────────────────────────────────────────────────────
def project_player(pinfo, stats, spread, total, side, team_abbr="",
                   cascade_bonus=0.0, is_b2b=False):
    if pinfo.get("is_out"): return None
    # Skip day-to-day and doubtful players — high scratch risk
    if pinfo.get("injury_status") in ("DTD", "DOUBT"): return None

    # DNP risk: player averaged very few minutes in recent games (rotation bubble,
    # rest management, or coach's decision). March 4th lesson: Hield/Harris/Sochan
    # all went RS=0 while the model projected them at season averages.
    # Skip unless a cascade bonus exists (teammate OUT = specific reason to play more).
    if stats.get("dnp_risk") and not cascade_bonus:
        return None

    avg_min = stats.get("min", 0)
    if avg_min <= 0: return None

    # Apply cascade minute boost
    proj_min = avg_min + cascade_bonus

    # Back-to-back penalty: teams on 2nd night of B2B see reduced minutes
    # and rest-managed players (older, injury-prone) often sit entirely.
    # Penalize projected minutes by 12% on B2B nights.
    if is_b2b:
        proj_min *= _cfg("projection.b2b_minute_penalty", 0.88)

    # GTD (game-time decision) — apply minute reduction to account for scratch risk.
    # GTD players are confirmed questionable; ~30-40% sit on any given night.
    # Reduce projected minutes rather than skip entirely (they might play).
    proj_cfg = _cfg("projection", _CONFIG_DEFAULTS["projection"])
    if pinfo.get("injury_status") == "GTD":
        proj_min *= proj_cfg.get("gtd_minute_penalty", 0.75)

    # Minutes gate — boost-aware: low-PPG contrarians get a lower threshold
    # because high card boost EV compensates for DNP risk.
    # Formula: effective_gate = max(8, min_gate - (rough_boost - 1.5) * 3)
    # Check player_overrides FIRST so override boosts (e.g. GPII 3.0, Sensabaugh 2.1,
    # Christian Braun 3.0) are reflected in the gate — otherwise those overrides are
    # unreachable for low-minute players who fail the PPG proxy threshold.
    min_gate = _cfg("projection.min_gate_minutes", MIN_GATE)
    _pts_for_gate = stats.get("pts", 0)
    _override_boost_gate = None
    _norm_for_gate = _normalize_boost_name(pinfo.get("name", ""))
    if _norm_for_gate:
        for _k, _v in _cfg("card_boost.player_overrides", {}).items():
            if _normalize_boost_name(_k) == _norm_for_gate:
                _override_boost_gate = float(_v)
                break
    _rough_boost = _override_boost_gate if _override_boost_gate is not None else max(0.2, 3.0 - _pts_for_gate * 0.12)
    effective_gate = max(8, min_gate - max(0, (_rough_boost - 1.5) * 3))
    if proj_min < effective_gate: return None

    pts = stats["pts"]
    reb = stats["reb"]
    ast = stats["ast"]
    stl = stats.get("stl", 0)
    blk = stats.get("blk", 0)
    tov = stats.get("tov", 0)
    minutes = stats.get("min", 0)

    # SCORING UPSIDE GATE 1: minimum projected points floor.
    # Universal floor uses the moonshot minimum (3.0) so low-PPG high-boost players
    # can enter the moonshot pool. Chalk pool enforces the stricter 7.0 floor separately.
    min_pts_moonshot = _cfg("scoring_thresholds.min_pts_projection_moonshot", 3.0)
    if pts < min_pts_moonshot:
        return None

    # SCORING UPSIDE GATE 2: minimum pts-per-minute efficiency.
    # Uses moonshot floor (0.15) at the universal level; chalk enforces stricter 0.28.
    min_ppm_moonshot = _cfg("scoring_thresholds.min_pts_per_minute_moonshot", 0.15)
    if minutes > 0 and (pts / minutes) < min_ppm_moonshot:
        return None

    # Base formula: PTS + REB + AST (correct for Real Sports scoring)
    base = pts + reb + ast
    if base <= 0:
        return None

    # Full DFS scoring formula (not just pts+reb+ast)
    heuristic = _dfs_score(pts, reb, ast, stl, blk, tov)

    # Stat-stuffer bonus: RS rewards breadth (pts+reb+ast+stl+blk all contribute
    # micro-ratings). Players projected in 3+ categories get an RS uplift because
    # their floor is higher and they generate more clutch-context plays.
    ss_cfg = _cfg("real_score.stat_stuffer", {})
    if ss_cfg.get("enabled", False):
        cats = 0
        if pts >= float(ss_cfg.get("pts_threshold", 15)): cats += 1
        if reb >= float(ss_cfg.get("reb_threshold", 7)): cats += 1
        if ast >= float(ss_cfg.get("ast_threshold", 5)): cats += 1
        if stl >= float(ss_cfg.get("stl_threshold", 1.5)): cats += 1
        if blk >= float(ss_cfg.get("blk_threshold", 1.0)): cats += 1
        if cats >= 4:
            heuristic *= float(ss_cfg.get("bonus_td", 1.15))
        elif cats >= 3:
            heuristic *= float(ss_cfg.get("bonus_3cat", 1.08))

    # Scale heuristic by minute boost from cascade (capped at 1.25x)
    if cascade_bonus > 0 and avg_min > 0:
        min_scale = min(proj_min / avg_min, 1.25)
        heuristic *= min_scale

    # Declining usage penalty: only triggers on significant drops (>20%).
    # Capped at 0.85 (max 15% reduction) — moderate declines (10-20%) are
    # normal rotation variance, not role changes. The old 10% trigger with
    # uncapped penalty was too aggressive (Thompson ↓17% = 0.83x penalty
    # despite still playing 25+ min and producing).
    season_min = stats.get("season_min", avg_min)
    recent_min = stats.get("recent_min", avg_min)
    decline_factor = 1.0
    if season_min > 0 and recent_min < season_min * 0.80:
        decline_factor = max(recent_min / season_min, 0.85)
        heuristic *= decline_factor

    # ── LightGBM inference — collect ai_pred only, do NOT blend yet ──────────
    # Bundle v2: baseline + spike heads (train_lgbm.py). Legacy: single regressor.
    _ensure_lgbm_loaded()
    ai_pred = None
    if AI_MODEL is not None or AI_MODEL_BASELINE is not None:
        try:
            season_pts_ = stats.get("season_pts", pts)
            recent_pts_ = stats.get("recent_pts", pts)
            season_min_ = stats.get("season_min", avg_min)
            recent_min_ = stats.get("recent_min", avg_min)
            _gp = stats.get("gp", stats.get("games_played"))
            feat_vec = _lgbm_feature_vector(
                avg_min=avg_min,
                pts=pts,
                reb=reb,
                ast=ast,
                stl=stl,
                blk=blk,
                spread=spread,
                side=side,
                season_pts=season_pts_,
                recent_pts=recent_pts_,
                season_min=season_min_,
                recent_min=recent_min_,
                cascade_bonus=cascade_bonus,
                games_played=float(_gp) if _gp is not None else None,
            )
            ai_pred = _lgbm_predict_rs(feat_vec)
        except Exception as _lgbm_e:
            print(f"[WARN] LightGBM inference failed for player, using heuristic: {_lgbm_e}")

    # Contextual multipliers — applied to heuristic only (not blend).
    # ai_pred is already in RS units; only the heuristic path needs these.
    pace_adj   = 1.0 + (0.06 * ((total or DEFAULT_TOTAL) - DEFAULT_TOTAL) / 20)

    # Spread adjustment — RS-clutch-aligned (v8).
    # RS algorithm heavily weights game closeness: tight games → every play matters →
    # higher micro-ratings. Blowouts → garbage time → devalued stats + stars sit Q4.
    # Tight spread + high total = back-and-forth shootout = RS goldmine.
    abs_spread = abs(spread or 0)
    bench_pts = float(_cfg("projection.bench_pts_threshold", 14.0))
    bench_min = float(_cfg("projection.bench_min_threshold", 30.0))
    is_bench = pts <= bench_pts and avg_min <= bench_min
    if is_bench:
        # Bench players: neutral in close games, bonus in blowouts (garbage-time minutes)
        if abs_spread <= 4:
            spread_adj = 1.0
        else:
            spread_adj = min(1.15, 1.0 + (abs_spread - 4) * 0.02)
    else:
        # Stars/starters: strong clutch bonus in tight games, steep blowout penalty.
        # RS data: Luka 9.1 in tight game, but stars in blowouts produce low RS
        # because clutch context multiplier → 0 in garbage time.
        if abs_spread <= 3:
            spread_adj = 1.25 - (abs_spread * 0.03)   # 1.25 at 0 → 1.16 at 3
        elif abs_spread <= 7:
            spread_adj = 1.16 - ((abs_spread - 3) * 0.04)  # 1.16 at 3 → 1.0 at 7
        else:
            spread_adj = max(0.55, 1.0 - (abs_spread - 7) * 0.09)  # steep: 1.0 at 7 → 0.55 at 12

    # Total interaction: high total + tight spread = shootout bonus
    _total = total or DEFAULT_TOTAL
    if not is_bench and _total >= 230 and abs_spread <= 5:
        spread_adj *= 1.0 + min(0.10, (_total - 230) * 0.005)  # up to +10% bonus
    home_adj   = 1.02 if side == "home" else 1.0

    s_base = heuristic * pace_adj * spread_adj * home_adj

    # ── RS projection: compress DFS base directly to RS scale ────────────────
    # Monte Carlo removed — closeness×clutch×momentum was multiplying ALL players
    # by 1.3-1.7x regardless of quality, inflating bench players from RS 2→4.
    # Compress the DFS base (s_base) directly. With the same div/pow params,
    # bench players (DFS ~16) now project RS ~2.3 instead of ~3.8.
    season_pts = stats.get("season_pts", pts)
    recent_pts = stats.get("recent_pts", pts)
    player_variance = abs(recent_pts - season_pts) / max(season_pts, 1)

    rs_cfg = _cfg("real_score", _CONFIG_DEFAULTS["real_score"])
    comp_div = rs_cfg.get("compression_divisor", 5.5)
    comp_pow = rs_cfg.get("compression_power", 0.72)
    raw_linear = s_base / comp_div
    heuristic_rs = raw_linear ** comp_pow
    # Asymmetric cap: high ceiling (20.0) to allow RS 6-8 projections for upside players,
    # but floor stays accurate because compression_power still dampens low values.
    rs_cap = rs_cfg.get("rs_cap", 20.0)
    heuristic_rs = min(heuristic_rs, rs_cap)

    # ── Late blend: AI (native RS units) + heuristic RS ───────────────────────
    # LightGBM outputs native RS units. 35% AI / 65% heuristic.
    ai_weight = rs_cfg.get("ai_blend_weight", 0.35)
    if ai_pred is not None:
        raw_score = min((ai_pred * ai_weight) + (heuristic_rs * (1.0 - ai_weight)), rs_cap)
    else:
        raw_score = heuristic_rs

    # Optional per-bucket RS calibration to reduce star inflation and
    # lift under-projected low-PPG role players.
    bucket_cfg = rs_cfg.get("bucket_calibration", {})
    if bucket_cfg.get("enabled", False):
        high_thr = float(bucket_cfg.get("high_pts_threshold", 16.0))
        mid_thr = float(bucket_cfg.get("mid_pts_threshold", 8.0))
        if pts >= high_thr:
            raw_score *= float(bucket_cfg.get("high_mult", 1.0))
        elif pts >= mid_thr:
            raw_score *= float(bucket_cfg.get("mid_mult", 1.0))
        else:
            raw_score *= float(bucket_cfg.get("low_mult", 1.0))
        raw_score = min(raw_score, rs_cap)

    arch_cfg = rs_cfg.get("archetype_calibration", {})
    if arch_cfg.get("enabled", False):
        arch = _infer_player_archetype(pts, avg_min, reb, stats)
        mults = arch_cfg.get("archetypes", {}) or {}
        m = float(mults.get(arch, 1.0))
        if m != 1.0:
            raw_score = min(raw_score * m, rs_cap)

    # ── Game closeness RS multiplier — selective by player usage ──────────
    # The RS algorithm HEAVILY rewards plays in tight games (clutch factor).
    # Re-integrated from real_score.py but scaled by usage so bench players
    # don't inflate (the old bug). High-usage players in tight games get the
    # full closeness bonus. Low-usage bench players get almost nothing.
    #
    # Key insight: a role player in a 2-point game with 3 steals and a scoring
    # run will outscore a superstar in a blowout EVERY TIME in RS.
    close_cfg = rs_cfg.get("closeness", {})
    if close_cfg.get("enabled", False):
        try:
            c_c = closeness_coefficient(spread, total)
            # Usage proxy: pts_per_min normalized (league avg ~0.5 pts/min)
            ppm = pts / max(avg_min, 1)
            usage_scale = min(ppm / 0.5, 1.5)  # caps at 1.5 for elite scorers
            strength = float(close_cfg.get("strength", 0.5))
            # Selective: only the (c_c - 1.0) bonus is scaled by usage
            # Bench player (ppm 0.25, usage_scale 0.5): gets 50% of bonus
            # Star (ppm 0.75, usage_scale 1.5): gets 150% of bonus (capped)
            close_mult = 1.0 + (c_c - 1.0) * usage_scale * strength
            close_mult = max(0.85, min(close_mult, float(close_cfg.get("max_mult", 1.40))))
            raw_score = min(raw_score * close_mult, rs_cap)
        except Exception:
            pass  # non-fatal: real_score.py import or math error

    # ── Cascade RS boost — usage spike from teammate injury ────────────────
    # When a starter is OUT, backups inherit not just minutes but USAGE.
    # More touches = more RS opportunity. This is additive to the minute scaling
    # already applied to the heuristic. Mar 10: Cameron Payne (Lowry OUT) went
    # from projected 2.6 to actual 7.7 — minute scaling alone can't explain this.
    cascade_cfg = rs_cfg.get("cascade_rs", {})
    if cascade_cfg.get("enabled", False) and cascade_bonus > 0:
        cascade_str = float(cascade_cfg.get("strength", 0.6))
        # Scale: +10 cascade min → ~1.10× RS boost (with strength 0.6)
        cascade_rs_mult = 1.0 + min(cascade_bonus / 15.0, 0.30) * cascade_str
        raw_score = min(raw_score * cascade_rs_mult, rs_cap)

    # ── Role-spike RS boost — hot streak / expanded role detection ─────────
    # Players whose recent production significantly exceeds their season average
    # are in an expanded role (injury, trade, coach decision). RS correlates with
    # opportunity: more minutes + more usage = exponentially more RS.
    # Mar 12: Middleton (recent 28min vs season 26min) hit RS 5.8.
    # Mar 7: Hendricks (took Achiuwa's minutes) hit RS 4.1.
    spike_cfg = rs_cfg.get("role_spike_rs", {})
    if spike_cfg.get("enabled", False):
        _spike_ratio = float(spike_cfg.get("min_ratio", 1.2))
        if season_min > 0 and recent_min >= season_min * _spike_ratio and recent_pts > 0:
            pts_surge = recent_pts / max(season_pts, 1)
            spike_str = float(spike_cfg.get("strength", 0.4))
            spike_mult = 1.0 + min(pts_surge - 1.0, 0.30) * spike_str
            if spike_mult > 1.0:
                raw_score = min(raw_score * spike_mult, rs_cap)

    # Estimated card boost (ADDITIVE, not multiplicative)
    # Real Sports formula: Value = Real Score × (Slot_Mult + Card_Boost)
    # Card boost is INVERSELY proportional to ownership — the app rewards
    # contrarian picks. Stars get crushed, obscure role players get huge boosts.
    card_boost = _est_card_boost(proj_min, pts, team_abbr, player_name=pinfo["name"])

    # EV score — card-adjusted expected value using additive formula
    # Use average slot (1.6) for ranking; MILP uses exact slots
    avg_slot = _cfg("lineup.avg_slot_multiplier", 1.6)

    # Core value formula: RS × (slot + boost). No post-hoc reliability adjustments.
    # Minute consistency is already baked into RS projection via season/recent blending.
    # Boost dominance audit (Mar 19): removed reliability multiplier that was dampening
    # high-boost players with minute variance — exactly the players that win leaderboards.
    chalk_ev  = round(raw_score * (avg_slot + card_boost), 2)

    # Ceiling score — player's upside (variance-adjusted median)
    ceiling_score = raw_score * (1.0 + (player_variance * 0.5))
    ceiling_ev = round(ceiling_score * (avg_slot + card_boost), 2)
    ceiling_score = round(ceiling_score, 1)

    return {
        "id":           pinfo["id"],
        "name":         pinfo["name"],
        "player_variance": round(player_variance, 3),
        "pos":     pinfo["pos"],
        "team":    team_abbr,
        "rating":        round(raw_score, 1),
        "chalk_ev":      chalk_ev,
        "ceiling_score": ceiling_score,
        "ceiling_ev":    ceiling_ev,
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
        "season_min": round(stats.get("season_min", avg_min), 1),
        "recent_min": round(recent_min, 1),
        "season_pts": round(stats.get("season_pts", pts), 1),
        "recent_pts": round(stats.get("recent_pts", pts), 1),
        "season_reb": round(stats.get("season_reb", reb), 1),
        "recent_reb": round(stats.get("recent_reb", reb), 1),
        "season_ast": round(stats.get("season_ast", ast), 1),
        "recent_ast": round(stats.get("recent_ast", ast), 1),
        "season_stl": round(stats.get("season_stl", stl), 1),
        "recent_stl": round(stats.get("recent_stl", stl), 1),
        "season_blk": round(stats.get("season_blk", blk), 1),
        "recent_blk": round(stats.get("recent_blk", blk), 1),
        "injury_status": pinfo.get("injury_status", ""),
        # Overperform signals — surfaced as pills on player cards.
        # _hot_streak: recent pts >= 1.15x season avg (configurable via signals.hot_streak_ratio)
        "_hot_streak": bool(
            season_pts > 0
            and recent_pts / season_pts >= float(_cfg("signals.hot_streak_ratio", 1.15))
        ),
    }

# ─────────────────────────────────────────────────────────────────────────────
# ODDS API ENRICHMENT
# grep: _enrich_projections_with_odds
# Blends sportsbook player prop lines into projections as a market signal.
# Books see information our stat model can't (teammate injuries, rotation
# changes, matchup exploitation). Called once per fresh slate generation.
# ─────────────────────────────────────────────────────────────────────────────

def _enrich_projections_with_odds(all_proj: list, games: list) -> None:
    """Enrich player projections in-place with Odds API player props.

    Reads `odds_enrichment.enabled` from model config. No-op if disabled,
    if ODDS_API_KEY is missing, or if the Odds API call fails.

    When books have a player's points line notably higher than our model's
    projection (divergence > min_divergence_pct), blends the book signal
    into our projection. Upward-only by default — books being lower than
    our model is not a reliable downgrade signal.

    Also nudges predMin upward proportionally when books expect higher scoring
    (higher points implies more minutes/usage).

    Mutates: player["pts"], player["predMin"], player["odds_pts_line"],
    player["odds_reb_line"], player["odds_ast_line"]
    """
    if not _cfg("odds_enrichment.enabled", False):
        return

    blend_weight = float(_cfg("odds_enrichment.blend_weight", 0.2))
    min_div_pct = float(_cfg("odds_enrichment.min_divergence_pct", 0.15))
    upward_only = _cfg("odds_enrichment.upward_only", True)

    # Reuse existing bulk odds fetcher (1 + N API calls, already parallelized)
    odds_map = _build_player_odds_map(games)
    if not odds_map:
        print("[odds_enrich] no odds data available — skipping")
        return

    enriched = 0
    for p in all_proj:
        name_lower = p.get("name", "").lower()
        # Attach raw odds lines for context pass / debugging
        pts_odds = odds_map.get((name_lower, "points"), {})
        reb_odds = odds_map.get((name_lower, "rebounds"), {})
        ast_odds = odds_map.get((name_lower, "assists"), {})
        if pts_odds:
            p["odds_pts_line"] = pts_odds.get("line", 0)
        if reb_odds:
            p["odds_reb_line"] = reb_odds.get("line", 0)
        if ast_odds:
            p["odds_ast_line"] = ast_odds.get("line", 0)

        # Scoring signal: blend if books diverge upward from our projection
        odds_pts = pts_odds.get("line", 0)
        model_pts = p.get("pts", 0)
        if odds_pts > 0 and model_pts > 0:
            divergence = (odds_pts - model_pts) / model_pts
            if divergence > min_div_pct:
                if not upward_only or odds_pts > model_pts:
                    new_pts = model_pts * (1 - blend_weight) + odds_pts * blend_weight
                    # Nudge predMin proportionally (higher pts → more usage/minutes)
                    pts_ratio = new_pts / max(model_pts, 0.1)
                    pred_min = p.get("predMin", 0)
                    if pred_min > 0:
                        p["predMin"] = round(pred_min * min(pts_ratio, 1.15), 1)  # cap 15% nudge
                    p["pts"] = round(new_pts, 1)
                    p["_odds_adjusted"] = True
                    enriched += 1

    print(f"[odds_enrich] enriched {enriched} players with Odds API data ({len(odds_map)} lines fetched)")


# ─────────────────────────────────────────────────────────────────────────────
# WEB INTELLIGENCE (Claude + web_search tool)
# grep: WEB INTELLIGENCE, _fetch_nba_news_context
# Uses Anthropic API with built-in web_search tool to find recent NBA news.
# No separate API key needed — uses the existing ANTHROPIC_API_KEY.
# Returns a compact text summary for injection into the Claude context pass.
# ─────────────────────────────────────────────────────────────────────────────

# Map ESPN abbreviations to full team names for better search queries
_ABBR_TO_TEAM_NAME = {
    "ATL": "Hawks", "BOS": "Celtics", "BKN": "Nets", "CHA": "Hornets",
    "CHI": "Bulls", "CLE": "Cavaliers", "DAL": "Mavericks", "DEN": "Nuggets",
    "DET": "Pistons", "GSW": "Warriors", "GS": "Warriors", "HOU": "Rockets",
    "IND": "Pacers", "LAC": "Clippers", "LAL": "Lakers", "MEM": "Grizzlies",
    "MIA": "Heat", "MIL": "Bucks", "MIN": "Timberwolves", "NOP": "Pelicans",
    "NY": "Knicks", "NYK": "Knicks", "OKC": "Thunder", "ORL": "Magic",
    "PHI": "76ers", "PHX": "Suns", "POR": "Trail Blazers", "SAC": "Kings",
    "SAS": "Spurs", "TOR": "Raptors", "UTA": "Jazz", "WAS": "Wizards",
}


def _fetch_nba_news_context(games: list, date=None, all_proj: list = None) -> str:
    """Fetch recent NBA news for teams on today's slate via Claude web_search tool.

    Layer 1 of the 3-layer Opus pipeline: once-per-slate intelligence gathering.
    Uses Opus (config: context_layer.web_search_model) with web_search so quality
    matters most. When all_proj is provided, the prompt includes key players (top
    by projected RS) so the model can prioritize news affecting likely draft picks.

    Uses the existing ANTHROPIC_API_KEY — no separate search API key needed.
    Reads `context_layer.web_search_enabled` from model config. Returns empty
    string if disabled or on any failure. Results cached per slate date in /tmp.
    """
    if not _cfg("context_layer.web_search_enabled", False):
        return ""

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        print("[web_search] ANTHROPIC_API_KEY not set — skipping")
        return ""

    today = (date or _et_date()).isoformat()
    cache_key = f"nba_news_{today}"
    cached = _cg(cache_key)
    if cached:
        return cached.get("text", "")

    # Collect unique teams from the slate
    teams = set()
    for g in games:
        home = g.get("home", {}).get("abbr", "")
        away = g.get("away", {}).get("abbr", "")
        if home:
            teams.add(home)
        if away:
            teams.add(away)

    if not teams:
        return ""

    # Build team list for the prompt
    team_names = []
    for abbr in sorted(teams):
        full = _ABBR_TO_TEAM_NAME.get(abbr, abbr)
        team_names.append(f"{abbr} ({full})")

    team_list = ", ".join(team_names)

    # Player/RS-aware: when all_proj provided, add key players (top 20–25 by rating)
    # so Opus can prioritize news that affects likely draft picks.
    key_players_blurb = ""
    if all_proj:
        top = sorted(all_proj, key=lambda p: p.get("rating", 0), reverse=True)[:25]
        lines = [f"- {p.get('name', '')} ({p.get('team', '')}): proj RS {p.get('rating', 0):.1f}" for p in top]
        key_players_blurb = (
            "\n\nKEY PLAYERS ON TONIGHT'S SLATE (prioritize news affecting these):\n"
            + "\n".join(lines)
            + "\n"
        )

    timeout_s = float(_cfg("context_layer.timeout_seconds", 20))
    web_search_model = _cfg("context_layer.web_search_model", "claude-sonnet-4-6-20250514")

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=anthropic_key)
        msg = client.messages.create(
            model=web_search_model,
            max_tokens=1500,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            messages=[{
                "role": "user",
                "content": (
                    f"Today is {today}. I need a concise NBA intelligence briefing for these teams "
                    f"playing tonight: {team_list}.{key_players_blurb}\n\n"
                    "PRIORITY: Search for confirmed OUT players and their positional backups who will "
                    f"inherit 15+ minutes tonight (e.g. 'NBA OUT tonight {today}', 'NBA injury report "
                    f"{today}'). Name the starter who is OUT and the specific backup likely to start "
                    "or play extended minutes. These cascade situations are the highest-value signals.\n\n"
                    "Also search for:\n"
                    "1. Coach press conference quotes about rotation changes or minute increases\n"
                    "2. Notable rest days, back-to-backs, or load management\n"
                    "3. Any breaking roster news (trades, call-ups, returns from injury)\n\n"
                    "Return ONLY a bullet-point summary of actionable findings. Each bullet should name "
                    "the specific player(s) affected and why. Skip teams with no relevant news. "
                    "Keep it under 2000 characters total. No preamble — start with the first bullet."
                ),
            }],
            timeout=timeout_s,
        )
        # Extract text from the response (skip tool_use blocks)
        text_parts = []
        for block in msg.content:
            if hasattr(block, "text"):
                text_parts.append(block.text.strip())
        text = "\n".join(text_parts)

        if len(text) > 2000:
            text = text[:2000] + "..."

        # Cache for the slate date
        _cs(cache_key, {"text": text, "ts": datetime.now(timezone.utc).isoformat()})
        print(f"[web_search] fetched news for {len(teams)} teams via Claude web_search ({len(text)} chars)")
        return text

    except Exception as e:
        print(f"[web_search] Claude web_search error (non-fatal): {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# MATCHUP ANALYSIS (Layer 1.5 of the projection pipeline)
# grep: MATCHUP ANALYSIS, _build_game_opp_map, _compute_matchup_factor, _fetch_matchup_intelligence
# Replaces the crude dev-team win_pct bonus with real opponent defensive quality,
# position-specific scaling, and Claude web_search for DvP intelligence.
# Runs between Layer 1 (news) and Layer 2 (context pass).
# ─────────────────────────────────────────────────────────────────────────────

def _build_game_opp_map(games: list) -> dict:
    """Return {team_abbr: opp_abbr} for every team playing today.
    Used to look up a player's opponent from their team abbreviation.
    """
    opp_map = {}
    for g in games:
        home = g.get("home", {}).get("abbr", "")
        away = g.get("away", {}).get("abbr", "")
        if home and away:
            opp_map[home] = away
            opp_map[away] = home
    return opp_map


def _compute_matchup_factor(player: dict, opp_abbr: str, def_stats: dict) -> float:
    """Compute a matchup quality multiplier [0.80, 1.25] based on opponent defense.

    Components:
    - def_factor: opponent pts_allowed vs league avg 115 → scaled by matchup.def_scale
    - pos_scale: position-specific weighting (guards see more of the signal, centers less)

    Falls back to 1.0 (neutral) when def_stats is empty or opponent unknown.
    Config keys: matchup.def_scale (0.35), matchup.pos_scale_g/f/c (1.05/1.00/0.90)
    """
    if not _cfg("matchup.enabled", True):
        return 1.0
    if not def_stats or not opp_abbr:
        return 1.0

    opp = def_stats.get(opp_abbr, {})
    pts_allowed = opp.get("pts_allowed")
    if not pts_allowed:
        return 1.0

    league_avg = 115.0
    def_scale = float(_cfg("matchup.def_scale", 0.35))
    # Weak defense (allows 120) → +1.6% ×0.35 = ~+12% at 30pts above avg
    # Elite defense (allows 108) → -7pts ×0.35 = ~-8% below avg
    def_factor = 1.0 + (pts_allowed - league_avg) / 30.0 * def_scale

    # Position scaling: guards score more pts → benefit more from a weak defense
    pos = _pos_group(player.get("pos", ""))
    pos_scales = {
        "G": float(_cfg("matchup.pos_scale_g", 1.05)),
        "F": float(_cfg("matchup.pos_scale_f", 1.00)),
        "C": float(_cfg("matchup.pos_scale_c", 0.90)),
    }
    pos_scale = pos_scales.get(pos, 1.0)
    adjusted = 1.0 + (def_factor - 1.0) * pos_scale

    return max(0.80, min(1.25, round(adjusted, 3)))


def _fetch_matchup_intelligence(games: list, all_proj: list, def_stats: dict,
                                 game_opp_map: dict, news_context: str = "") -> dict:
    """Layer 1.5: Claude analyzes tonight's matchups with web_search for DvP data.

    Returns {player_name: {"factor": float, "reason": str}} where factor is a
    [0.80, 1.20] multiplier applied on top of the math-based matchup_factor.
    Config-gated: matchup.claude_enabled (default True).
    Cached daily per slate date.
    Non-fatal: returns {} on any error so math-only factors are used instead.
    """
    if not _cfg("matchup.enabled", True) or not _cfg("matchup.claude_enabled", True):
        return {}

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        return {}

    today = _et_date().isoformat()
    cache_key = f"matchup_intel_{today}"
    cached = _cg(cache_key)
    if cached is not None:
        return cached.get("players_map", {})

    # Build matchup table for prompt
    matchup_lines = []
    for g in games:
        home = g.get("home", {}).get("abbr", "")
        away = g.get("away", {}).get("abbr", "")
        home_def = def_stats.get(home, {}).get("pts_allowed", "?")
        away_def = def_stats.get(away, {}).get("pts_allowed", "?")
        spread = g.get("spread")
        total = g.get("total")
        spread_str = f"spread {spread:+.1f}" if spread else ""
        total_str = f"total {total}" if total else ""
        extras = ", ".join(filter(None, [spread_str, total_str]))
        home_def_str = f"{home_def:.1f}" if isinstance(home_def, float) else home_def
        away_def_str = f"{away_def:.1f}" if isinstance(away_def, float) else away_def
        matchup_lines.append(
            f"{away} (allows {away_def_str} pts/g) @ {home} (allows {home_def_str} pts/g)"
            + (f" [{extras}]" if extras else "")
        )

    matchup_table = "\n".join(matchup_lines) if matchup_lines else "No games found"

    # Top projected players with opponent context
    top_players = sorted(all_proj, key=lambda p: p.get("rating", 0), reverse=True)[:30]
    player_lines = []
    for p in top_players:
        opp = game_opp_map.get(p.get("team", ""), "?")
        opp_def = def_stats.get(opp, {}).get("pts_allowed", "?")
        opp_def_str = f"{opp_def:.1f}" if isinstance(opp_def, float) else opp_def
        player_lines.append(
            f"- {p.get('name', '')} ({p.get('pos', '?')}, {p.get('team', '')} vs {opp}, "
            f"opp allows {opp_def_str}): proj RS {p.get('rating', 0):.1f}, "
            f"{p.get('season_pts', p.get('pts', 0)):.0f} pts/g avg"
        )

    player_blurb = "\n".join(player_lines)
    news_section = f"\n\nKNOWN NEWS (from earlier today):\n{news_context}\n" if news_context else ""

    timeout_s = float(_cfg("matchup.claude_timeout_seconds", 25))
    model_id = _cfg("context_layer.web_search_model", "claude-sonnet-4-6-20250514")

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=anthropic_key)
        msg = client.messages.create(
            model=model_id,
            max_tokens=2000,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 6}],
            messages=[{
                "role": "user",
                "content": (
                    f"Today is {today}. I need matchup quality analysis for tonight's NBA games "
                    f"to adjust player projections.\n\n"
                    f"TONIGHT'S MATCHUPS (pts allowed per game this season):\n{matchup_table}\n\n"
                    f"TOP PROJECTED PLAYERS:\n{player_blurb}"
                    f"{news_section}\n\n"
                    "Use web_search to find defense-vs-position (DvP) data for tonight's teams:\n"
                    "1. How many points does each team allow to guards, forwards, and centers this season?\n"
                    "2. Which teams are top-5 / bottom-5 vs each position?\n"
                    "3. Any pace or style matchup advantages (e.g. fast team vs slow defense)?\n\n"
                    "Based on your research, return ONLY a JSON object with per-player matchup factors.\n"
                    "Only include players where the matchup meaningfully helps or hurts them (skip neutral).\n"
                    "Factor range: 0.80 (very tough matchup) to 1.20 (very favorable matchup).\n"
                    "Keep reasons under 15 words each.\n\n"
                    'Respond with ONLY valid JSON: {"players": [{"name": "Player Name", "factor": 1.12, "reason": "brief reason"}]}'
                ),
            }],
            timeout=timeout_s,
        )
        # Extract JSON from text blocks
        text_parts = []
        for block in msg.content:
            if hasattr(block, "text"):
                text_parts.append(block.text.strip())
        raw_text = "\n".join(text_parts)

        # Parse JSON — be lenient (Claude may wrap in markdown)
        json_match = re.search(r'\{.*"players".*\}', raw_text, re.DOTALL)
        if not json_match:
            return {}
        parsed = json.loads(json_match.group())
        players_list = parsed.get("players", [])

        # Build name → factor map, clamp to safe range
        players_map = {}
        for entry in players_list:
            name = entry.get("name", "")
            factor = float(entry.get("factor", 1.0))
            factor = max(0.80, min(1.20, factor))
            reason = entry.get("reason", "")
            if name and factor != 1.0:
                players_map[name] = {"factor": factor, "reason": reason}

        _cs(cache_key, {"players_map": players_map, "ts": datetime.now(timezone.utc).isoformat()})
        print(f"[matchup_intel] Claude returned {len(players_map)} player adjustments")
        return players_map

    except Exception as e:
        print(f"[matchup_intel] Claude matchup analysis error (non-fatal): {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE CONTEXT PASS (Layer 2 of 3-layer Opus pipeline)
# grep: _claude_context_pass
# Optional post-projection RS adjustment using Claude (Opus) game narrative.
# Called once per fresh slate generation; config-gated.
# Applies ±40% max multiplier to rating/chalk_ev/ceiling_score based on
# context the pure stat model can't capture (blowout risk, defensive value,
# rivalry game closeness, etc).
# ─────────────────────────────────────────────────────────────────────────────

def _claude_context_pass(all_proj: list, games: list) -> None:
    """Adjust player RS projections in-place using a Claude (Opus) context call.

    Reads `context_layer.enabled` from model config. No-op if disabled or if
    the Claude call fails — slate generation must never depend on this call.

    Mutates: player["rating"], player["chalk_ev"], player["ceiling_score"]
    """
    if not _cfg("context_layer.enabled", False):
        return

    model_id = _cfg("context_layer.model", "claude-sonnet-4-6-20250514")
    max_adj = float(_cfg("context_layer.max_adjustment", 0.4))
    timeout_s = float(_cfg("context_layer.timeout_seconds", 15))

    # Fetch recent NBA news for teams on the slate (Layer 1: Opus + web_search)
    news_text = ""
    try:
        news_text = _fetch_nba_news_context(games, date=None, all_proj=all_proj)
    except Exception as _news_err:
        print(f"[context_pass] web search error (non-fatal): {_news_err}")

    # Build compact game context map: {team_abbr: {opponent, spread, total}}
    game_ctx = {}
    for g in games:
        spread = g.get("spread", 0)
        total  = g.get("total", 222)
        home   = g["home"]["abbr"]
        away   = g["away"]["abbr"]
        game_ctx[home] = {"opp": away, "spread": spread, "total": total, "side": "home"}
        game_ctx[away] = {"opp": home, "spread": -spread, "total": total, "side": "away"}

    # Fetch RotoWire statuses for cascade/role confirmation signals (uses 30-min cache)
    rw_ctx = {}
    try:
        from api.rotowire import get_all_statuses as _gcs
        rw_ctx = _gcs() or {}
    except Exception:
        pass

    # Build compact player list for prompt (top 40 by rating to keep prompt small)
    sorted_proj = sorted(all_proj, key=lambda p: p.get("rating", 0), reverse=True)[:40]
    players_payload = []
    for p in sorted_proj:
        team = p.get("team", "")
        ctx  = game_ctx.get(team, {})
        cascade_bonus = round(p.get("_cascade_bonus", 0), 1)
        roto_entry = rw_ctx.get(p["name"].lower(), {})
        roto_status = roto_entry.get("status", "unknown")
        entry = {
            "name":        p["name"],
            "team":        team,
            "opp":         ctx.get("opp", ""),
            "spread":      ctx.get("spread", 0),
            "total":       ctx.get("total", 222),
            "season_pts":  round(p.get("season_pts", 0), 1),
            "season_reb":  round(p.get("season_reb", 0), 1),
            "season_ast":  round(p.get("season_ast", 0), 1),
            "season_stl":  round(p.get("season_stl", 0), 1),
            "season_blk":  round(p.get("season_blk", 0), 1),
            "projected_rs": p.get("rating", 0),
            "card_boost":   round(p.get("est_mult", 0), 1),
            "pred_min":     round(p.get("predMin", p.get("season_min", 0)), 1),
            "cascade_bonus": cascade_bonus,
            "roto_status":  roto_status,
        }
        # Include Odds API lines if available (from _enrich_projections_with_odds)
        if p.get("odds_pts_line"):
            entry["odds_pts_line"] = p["odds_pts_line"]
        if p.get("odds_reb_line"):
            entry["odds_reb_line"] = p["odds_reb_line"]
        if p.get("odds_ast_line"):
            entry["odds_ast_line"] = p["odds_ast_line"]
        players_payload.append(entry)

    system_prompt = (
        "You adjust RS (Real Score) projections for a daily NBA Real Sports draft game. "
        "RS rewards clutch impact, defensive hustle plays, momentum swings, and multi-stat "
        "contributions in CLOSE games — NOT pure box score stats. The stat model cannot "
        "see game narrative; you can. Your adjustments are used for both Starting 5 and "
        "Moonshot; both lineups are built from the same high-confidence core pool of 7–10 "
        "players projected to pop off today.\n\n"

        "WINNING DRAFT PATTERNS (from leaderboard analysis across 8 sample dates):\n"
        "- Winning total scores range 60-76 pts depending on slate quality. Small slates "
        "(2-3 games): 60-64. Normal slates: 65-72. High-total/pace slates: 73-76+.\n"
        "- You need at least 1 RS 5.5+ anchor to win. Multiple RS 5+ players = high-score "
        "day. A lineup of all RS 3s rarely wins even with high boosts.\n"
        "- RS TIERS: 5.5+ = anchor (almost always in a winning lineup); 3.5-5.4 = solid "
        "contributor (needs 3x+ boost to drive wins); 2.5-3.4 = boost-only play (needs 4x+ "
        "boost, cannot carry a lineup alone).\n"
        "- The VALUE formula is RS × (slot_mult + card_boost). RS 4.3 × (slot + 4.6x boost) "
        "contributes ~19 pts — same as RS 6.7 with no boost. Your job is the RS side.\n"
        "- Some days, 5-6 out of 6 top drafts share the same 1-2 players ('consensus anchors'). "
        "These players are correctly popular AND high-RS — e.g., Trae Young RS 5.8 (Dec 21, "
        "in all 6 winning lineups), Aldama RS 6.0 (Dec 20, 5/6), Garland RS 4.8 (Dec 19, 6/6). "
        "These are NOT wrong to pick — when a player's RS is genuinely that high, everyone "
        "should have them.\n"
        "- Stars in blowouts (favored by 10+) consistently UNDERPERFORM. They sit early, "
        "coast, and RS drops 20-40%.\n\n"

        "TEAM/STYLE SIGNALS (adjust UP):\n"
        "- ATL Hawks: Fast pace, Trae Young drives RS through assists+clutch. Players like "
        "Ware (RS 5.1 Dec 19, RS 5.7 Dec 21), Young (RS 5.8 Dec 21) consistently beat "
        "projections. ATL in a competitive game: +10-20% on multi-stat contributors.\n"
        "- MEM Grizzlies: Development team, tight games, hustle culture. Watson (RS 6.4), "
        "Bane (RS 6.2), Aldama (RS 6.0) all big winners. MEM role players: +10-20%.\n"
        "- CHI Bulls: Development era, high-RS outputs from Buzelis (RS 5.6), Carrington "
        "(RS 3.4 with 4.1x boost), Drummond (physical presence). CHI in close game: +10-15%.\n"
        "- OKC Thunder: System-generated RS from hustle/defense. Hartenstein (RS 7.3), "
        "Ware (RS 5.1) overperform. Slight underdog situations amplify this.\n"
        "- NYK Knicks: Defensive/hustle identity. Robinson (RS 2.7 but in winning lineup Dec 21 "
        "via rebounds+defense), Brunson in close games (RS 7.3 Dec 21).\n\n"

        "KEY RS SIGNAL TYPES (adjust UP for these):\n"
        "1. Defensive anchor in a close game (spread ≤ 5): +10-25% "
        "(Draymond, GPII, Melton, Jrue Holiday, Mitchell Robinson type)\n"
        "2. ATL/MEM/CHI/OKC pace-and-hustle role player in a competitive game: +10-20%\n"
        "3. High-usage playmaker when team is slight underdog (spread -1 to -6): +10-15%\n"
        "4. Multi-stat contributor (reb+ast+stl, not just scorer) in high-total game (230+): +10-15%\n"
        "5. Player with known defensive/playmaking impact understated by pts/reb/ast alone: +10-20%\n\n"

        "KEY RS SIGNAL TYPES (adjust DOWN for these):\n"
        "1. Star on team favored by 10+: -15-30% (blowout = sits in 4th quarter)\n"
        "2. Pure one-dimensional scorer (low reb/ast/stl/blk) on heavy favorite: -20-35%\n"
        "3. Low-minute role player on favorite likely to see garbage time early: -15-25%\n\n"

        "SPORTSBOOK MARKET SIGNALS:\n"
        "Some players include odds_pts_line (sportsbook points O/U). When the books "
        "have a player notably higher than our projected_rs or season_pts, it often "
        "signals expanded role (teammate injury, matchup, coaching decision). "
        "Use this as a CONFIRMING signal — if odds + narrative both point up, that's "
        "a strong adjustment. Don't adjust purely on odds divergence alone.\n\n"

        "CASCADE & ROLE CONFIRMATION SIGNALS:\n"
        "Each player now includes `cascade_bonus` (extra minutes inherited from a teammate injury) "
        "and `roto_status` (RotoWire confirmation: 'confirmed'=in starting lineup, "
        "'expected'=likely in rotation, 'questionable'=uncertain, 'unknown'=not listed).\n"
        "- cascade_bonus >= 5.0: This player is inheriting real starter minutes tonight. "
        "If card_boost >= 2.0, apply 1.15–1.25x UP — the extra minutes create outsized RS "
        "opportunities (clutch moments, defensive plays, hustle stats in a close game). "
        "Do NOT penalize these players for blowout risk — their minutes are coach-committed.\n"
        "- roto_status='confirmed' + cascade_bonus >= 8: Apply 1.20–1.30x UP — high-confidence "
        "starter-role situation with real upside.\n"
        "- roto_status='unknown' + pred_min < 18: Apply 0.85–0.90x DOWN — player not in RotoWire "
        "rotation and has fragile minute floor; DNP or early hook risk is meaningful.\n\n"

        "CALIBRATION: Most players get NO adjustment (omit them). Only adjust when you have "
        "a clear narrative reason. Typical batch: 4-8 players out of 40. "
        "Strong signal = 1.2-1.35x up or 0.7-0.85x down. "
        "Weak signal = 1.08-1.15x up or 0.88-0.95x down. "
        "Reserve 1.35x+ for rare cases: true defensive anchor in a tight rivalry game, or "
        "a pace/hustle team player in a game projected to stay close all night.\n\n"

        "Return ONLY valid JSON:\n"
        '{"adjustments": [{"player": "Exact Name", "rs_multiplier": 1.20, "reason": "brief"}]}\n'
        "Keep multipliers between 0.6 and 1.4. Omit players with multiplier 1.0."
    )
    # Include web search results in the user prompt if available (Layer 1 output)
    news_section = ""
    if news_text:
        news_section = (
            f"\n\nRECENT NBA NEWS (last 24-48 hours):\n{news_text}\n\n"
            "Map each relevant news bullet to specific players in the list above. For each "
            "adjustment you make, the reason must cite the specific news item (e.g. 'Star X "
            "out → teammate Y minutes up'). Rotation changes, injury impacts on teammates, "
            "and coach quotes should drive explicit up/down adjustments with clear magnitude. "
            "Weight press conference quotes and official injury reports heavily."
        )
    user_prompt = (
        f"Today's slate players (top 40 by projected RS):\n"
        f"{json.dumps(players_payload, separators=(',', ':'))}"
        f"{news_section}\n\n"
        "Return JSON adjustments only."
    )

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
        msg = client.messages.create(
            model=model_id,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            timeout=timeout_s,
        )
        raw_text = msg.content[0].text.strip()
        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()
        data = json.loads(raw_text)
        adjustments = {a["player"]: a["rs_multiplier"] for a in data.get("adjustments", [])}
    except Exception as e:
        print(f"[context_pass] skipped (error: {e})")
        return

    if not adjustments:
        return

    # Apply multipliers to all matching players, capped at ±max_adj
    adj_count = 0
    name_map = {p["name"]: p for p in all_proj}
    for player_name, multiplier in adjustments.items():
        p = name_map.get(player_name)
        if not p:
            continue
        # Clamp to [1-max_adj, 1+max_adj]
        clamped = max(1.0 - max_adj, min(1.0 + max_adj, float(multiplier)))
        p["rating"]        = round(p.get("rating", 0) * clamped, 1)
        p["chalk_ev"]      = round(p.get("chalk_ev", 0) * clamped, 2)
        p["ceiling_score"] = round(p.get("ceiling_score", 0) * clamped, 1)
        p["_context_adj"]  = round(clamped, 3)
        adj_count += 1

    print(f"[context_pass] applied {adj_count} adjustments via {model_id}")


# ─────────────────────────────────────────────────────────────────────────────
# LINEUP REVIEW (Layer 3 of 3-layer Opus pipeline)
# grep: _lineup_review_opus
# Post-lineup Opus + web_search: review assembled chalk/upside, suggest swaps
# (e.g. late injury, rotation news); auto-apply valid swaps. Non-fatal: on error
# returns original lineups unchanged.
# ─────────────────────────────────────────────────────────────────────────────

def _lineup_review_opus(chalk: list, upside: list, all_proj: list, games: list, core_pool: list = None, news_context: str = "") -> tuple:
    """Review assembled lineups with Opus + web search; suggest and auto-apply swaps.

    Reads lineup_review.enabled, lineup_review.model, lineup_review.timeout_seconds.
    When core_pool is provided (list of player dicts), swap-ins are restricted to that
    core so both lineups stay configurations of the same high-confidence pool.
    When news_context is provided (from Layer 1), it's included in the prompt so Layer 3
    doesn't need to perform redundant web searches — only searches for truly late-breaking
    news (last 2-4 hours) that Layer 1 may have missed.
    Returns (chalk, upside). On any failure returns original chalk/upside unchanged.
    """
    if not _cfg("lineup_review.enabled", False):
        return chalk, upside

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        return chalk, upside

    model_id = _cfg("lineup_review.model", "claude-sonnet-4-6-20250514")
    timeout_s = float(_cfg("lineup_review.timeout_seconds", 30))

    def _describe_lineup(players: list, label: str) -> str:
        lines = []
        for p in (players or []):
            name = p.get("name", "")
            team = p.get("team", "")
            slot = p.get("slot", "")
            rs = p.get("rating", 0)
            boost = p.get("est_mult", 0)
            lines.append(f"  {slot}: {name} ({team}) — RS {rs:.1f}, boost +{boost:.1f}x")
        return f"{label}:\n" + "\n".join(lines) if lines else f"{label}: (none)"

    chalk_desc = _describe_lineup(chalk, "Starting 5 (chalk)")
    upside_desc = _describe_lineup(upside, "Moonshot (upside)")
    core_names = [p.get("name", "") for p in (core_pool or []) if p.get("name")]
    core_blurb = ""
    if core_names:
        core_blurb = (
            "\n\nThese two lineups are two configurations of the same high-confidence core pool "
            "(reliability vs ceiling); high overlap is intended. When suggesting a swap-in, prefer "
            f"players from this core when possible: {', '.join(core_names)}."
        )

    # Include cached news from Layer 1 so Layer 3 has context without redundant searches
    news_blurb = ""
    if news_context:
        news_blurb = (
            f"\n\nKNOWN NEWS FROM EARLIER TODAY:\n{news_context}\n\n"
            "The above was gathered earlier. Focus your web search on ONLY the last 2-4 hours "
            "for truly late-breaking updates (last-minute scratches, game-time decisions). "
            "If the known news already covers everything, return {\"swaps\": []}.\n"
        )

    user_content = (
        "You have two daily NBA draft lineups (Starting 5 = chalk, Moonshot = upside). "
        "Search the web for the latest news (last 2–4 hours): injuries, late scratches, "
        "rotation changes, or anything that would make a current pick wrong or a different "
        "player clearly better.\n\n"
        f"{chalk_desc}\n\n{upside_desc}\n{core_blurb}{news_blurb}\n\n"
        "If you find a reason to swap a player out (e.g. just ruled OUT, or a teammate "
        "now getting the run), return a JSON object with a 'swaps' array. Each swap: "
        '{"lineup": "chalk" or "upside", "out": "Exact Player Name", "in": "Exact Player Name"}. '
        "Only suggest swaps when the news is clear and actionable. If nothing to change, "
        'return {"swaps": []}. Return ONLY the JSON, no markdown or preamble.'
    )

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=anthropic_key)
        msg = client.messages.create(
            model=model_id,
            max_tokens=1024,
            tools=[{"type": "web_search_20250305", "name": "web_search", "max_uses": 5}],
            messages=[{"role": "user", "content": user_content}],
            timeout=timeout_s,
        )
        text = ""
        for block in msg.content:
            if hasattr(block, "text"):
                text += block.text
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        data = json.loads(text)
        swaps = data.get("swaps", [])
        if not swaps:
            return chalk, upside
    except Exception as e:
        print(f"[lineup_review] skipped (error: {e})")
        return chalk, upside

    name_to_proj = {p["name"]: p for p in all_proj}
    core_names_set = {p.get("name") for p in (core_pool or []) if p.get("name")}
    chalk_out = list(chalk)
    upside_out = list(upside)
    applied = 0

    for s in swaps:
        lineup_name = (s.get("lineup") or "").lower()
        out_name = (s.get("out") or "").strip()
        in_name = (s.get("in") or "").strip()
        if not out_name or not in_name or lineup_name not in ("chalk", "upside"):
            continue
        replacement = name_to_proj.get(in_name)
        if not replacement:
            continue
        if core_names_set and in_name not in core_names_set:
            continue
        target = chalk_out if lineup_name == "chalk" else upside_out
        idx = next((i for i, p in enumerate(target) if (p.get("name") or "").strip() == out_name), None)
        if idx is None:
            continue
        slot = target[idx].get("slot", "1.0x")
        new_entry = {**replacement, "slot": slot}
        target[idx] = new_entry
        applied += 1

    if applied:
        print(f"[lineup_review] applied {applied} swap(s) via {model_id}")
    return chalk_out, upside_out


# ─────────────────────────────────────────────────────────────────────────────
# GAME RUNNER & LINEUP BUILDER
# grep: _run_game, _build_lineups, _build_game_lineups, chalk_ev, Moonshot
# _run_game: fetches rosters, runs cascade, projects all players for one game
# _build_lineups: top-5 chalk (MILP) + moonshot (ranks 6-10 same EV)
# ─────────────────────────────────────────────────────────────────────────────
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
        # Determine opponent for matchup analysis
        opp_abbr = game["away"]["abbr"] if sd == "home" else game["home"]["abbr"]
        proj = project_player(p, stats, game["spread"], game["total"], sd, ab,
                              cascade_bonus=cascade_bonus, is_b2b=bool(b2b))
        if proj:
            proj["opp"] = opp_abbr  # store opponent for matchup factor in _build_lineups
            out.append(proj)
    _cs(cache_key, out)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE CONTRACT NORMALIZERS
# grep: _normalize_player, _normalize_line_pick
# Stable frontend API contract. Model internals can change freely; only these
# output shapes are guaranteed to the frontend. Apply at all lineup return points.
# ─────────────────────────────────────────────────────────────────────────────

# Internal-only fields never sent to the frontend
_PLAYER_INTERNAL_FIELDS = {"chalk_ev_capped", "_rw_cleared", "_matchup_factor"}

def _normalize_player(p: dict) -> dict:
    """Stable frontend contract for player projection objects.
    Model can add/remove internal fields freely.
    Guarantees all required fields are present with correct types."""
    base = {
        "id":            p.get("id", ""),
        "name":          p.get("name", ""),
        "pos":           p.get("pos", ""),
        "team":          p.get("team", ""),
        "rating":        round(float(p.get("rating") or 0), 1),
        "predMin":       round(float(p.get("predMin") or 0), 1),
        "pts":           round(float(p.get("pts") or 0), 1),
        "reb":           round(float(p.get("reb") or 0), 1),
        "ast":           round(float(p.get("ast") or 0), 1),
        "stl":           round(float(p.get("stl") or 0), 1),
        "blk":           round(float(p.get("blk") or 0), 1),
        "est_mult":      round(float(p.get("est_mult") or 0), 2),
        "slot":          p.get("slot", "1.0x"),
        "chalk_ev":      round(float(p.get("chalk_ev") or 0), 2),
        "moonshot_ev":   round(float(p.get("moonshot_ev") or 0), 2),
        "injury_status": p.get("injury_status", ""),
        "_decline":      round(float(p.get("_decline") or 0), 2),
    }
    # Pass through extras (model fields, debug fields, trend stats) — strip internal-only
    extras = {k: v for k, v in p.items()
              if k not in base and k not in _PLAYER_INTERNAL_FIELDS}
    return {**base, **extras}


_LINE_PICK_CONTRACT_FIELDS = {
    "player_name", "player_id", "team", "opponent", "direction", "line",
    "stat_type", "projection", "edge", "confidence", "narrative", "signals",
    "result", "actual_stat", "line_updated_at", "odds_over", "odds_under",
    "books_consensus", "date",
    "season_avg", "proj_min", "avg_min", "game_time", "game_start_iso",
    "recent_form_bars", "recent_form_values",
}

def _normalize_line_pick(p: dict) -> dict:
    """Stable frontend contract for line pick objects.
    Guarantees all required fields are present with correct types."""
    if p is None or not isinstance(p, dict):
        return {}
    base = {
        "player_name":     p.get("player_name", ""),
        "player_id":       p.get("player_id", ""),
        "team":            p.get("team", ""),
        "opponent":        p.get("opponent", ""),
        "direction":       p.get("direction", "over"),
        "line":            float(p.get("line") or 0),
        "stat_type":       p.get("stat_type", "points"),
        "projection":      round(float(p.get("projection") or 0), 1),
        "edge":            round(float(p.get("edge") or 0), 1),
        "confidence":      int(p.get("confidence") or 0),
        "narrative":       p.get("narrative", ""),
        "signals":         p.get("signals") or [],
        "result":          p.get("result") or "pending",
        "actual_stat":     p.get("actual_stat"),
        "line_updated_at": p.get("line_updated_at"),
        "odds_over":       p.get("odds_over"),
        "odds_under":      p.get("odds_under"),
        "books_consensus": p.get("books_consensus"),
        "date":            p.get("date", ""),
        "season_avg":      p.get("season_avg"),
        "proj_min":        p.get("proj_min"),
        "avg_min":         p.get("avg_min"),
        "game_time":          p.get("game_time", ""),
        "game_start_iso":     p.get("game_start_iso", ""),
        "recent_form_bars":   p.get("recent_form_bars"),
        "recent_form_values": p.get("recent_form_values"),
    }
    # Pass through any extra fields (e.g. _live tracker, future additions)
    extras = {k: v for k, v in p.items() if k not in _LINE_PICK_CONTRACT_FIELDS}
    return {**base, **extras}


def _build_lineups(projections, def_stats=None, matchup_intel=None):
    avg_slot   = _cfg("lineup.avg_slot_multiplier", 1.6)
    chalk_floor = _cfg("lineup.chalk_rating_floor", 2.0)
    proj_cfg = _cfg("projection", _CONFIG_DEFAULTS["projection"])
    boost_cap = proj_cfg.get("chalk_boost_cap", 2.5)

    moon_cfg = _cfg("moonshot", _CONFIG_DEFAULTS["moonshot"])
    use_rotowire = moon_cfg.get("require_rotowire_clearance", True)

    # Fetch RotoWire lineup statuses once — shared by chalk AND moonshot
    rw_statuses = {}
    if use_rotowire:
        try:
            rw_statuses = get_all_statuses()
        except Exception as e:
            print(f"RotoWire fetch failed, proceeding without: {e}")
    # Team motivation (late-season): soft reliability tilt for Starting 5.
    team_motivation = _fetch_team_motivation_map()

    # STARTING 5: MILP-optimized for chalk_ev with card boost capped.
    # chalk_boost_cap=2.5: rewards moderate-ownership role players without going full moonshot.
    # Mar 5 insight: Ace Bailey (RS 5.9 × boost 2.1) >>> Wemby (RS 7.1 × boost 0.3).
    # RotoWire filter applied here too — chalk can't include OUT/questionable players.
    chalk_eligible = []
    chalk_min_pts = float(_cfg("scoring_thresholds.min_pts_projection", 7.0))
    chalk_min_ppm = float(_cfg("scoring_thresholds.min_pts_per_minute", 0.28))
    for p in projections:
        # Application-layer blacklist: never include certain players in draft lineups.
        if p.get("name") in BLACKLISTED_PLAYERS:
            continue
        # Chalk-specific scoring floors (stricter than the universal moonshot floor).
        # A chalk player must project 7+ pts and 0.28 pts/min to be viable.
        if p.get("pts", 0) < chalk_min_pts:
            continue
        p_min = p.get("season_min", 0)
        if p_min > 0 and (p.get("pts", 0) / p_min) < chalk_min_ppm:
            continue
        if p["rating"] < chalk_floor:
            continue
        # SLATE-WIDE CHALK: Proven rotation regular OR definitive spot-starter.
        # Condition A: Both historical floors met (established starter).
        # Condition B: predMin >= 28 AND _cascade_bonus >= 10 (backup stepping into a
        #   starter role due to injury — cascade engine gave them 10+ bonus minutes).
        #   This prevents "DFS Ghosts": trickle-cascade bench warmers won't clear
        #   both gates simultaneously, only true spot-starters will.
        chalk_min_floor = _cfg("projection.chalk_season_min_floor", 20.0)
        chalk_recent_floor = _cfg("projection.chalk_recent_min_floor", 15.0)
        _recent_min_chalk = p.get("recent_min", 0) or p.get("season_min", 0)
        is_regular     = (p.get("season_min", 0) >= chalk_min_floor and
                          _recent_min_chalk >= chalk_recent_floor)
        is_spot_starter = (p.get("predMin", 0) >= 28.0 and
                           p.get("_cascade_bonus", 0) >= 10.0)
        # High-boost role player pathway for chalk: stricter thresholds than moonshot
        # because Starting 5 requires reliability. Only players with a strong boost
        # signal AND consistent recent minutes qualify. Boost >= 2.5 means Real Sports
        # considers them significantly underrated — a reliable contrarian play.
        chalk_hbr_enabled    = _cfg("projection.chalk_hbr_enabled", True)
        chalk_hbr_min_boost  = float(_cfg("projection.chalk_hbr_min_boost", 2.5))
        chalk_hbr_min_recent = float(_cfg("projection.chalk_hbr_min_recent_min", 16.0))
        chalk_hbr_min_pred   = float(_cfg("projection.chalk_hbr_min_pred_min", 16.0))
        is_chalk_high_boost_role = (
            chalk_hbr_enabled
            and p.get("est_mult", 0) >= chalk_hbr_min_boost
            and _recent_min_chalk >= chalk_hbr_min_recent
            and p.get("predMin", 0) >= chalk_hbr_min_pred
        )
        if not (is_regular or is_spot_starter or is_chalk_high_boost_role):
            continue
        # Skip players flagged OUT or questionable in RotoWire (same logic as moonshot)
        if use_rotowire and rw_statuses and not is_safe_to_draft(p["name"]):
            continue
        # Never draft a chalk player projected well below their season minute average.
        # A small tolerance band (default 2.0 min) prevents tiny projection misses from
        # excluding viable chalk players — Bub Carrington (25.8 vs 27.3 = 1.5 gap) etc.
        chalk_min_tol = float(_cfg("projection.pred_min_tolerance", 2.0))
        if p.get("predMin", 0) < (p.get("season_min", 0) - chalk_min_tol):
            continue
        # Boost floor — players below this threshold fail unless star anchor applies.
        chalk_min_boost = float(_cfg("projection.chalk_min_boost_floor", 1.0))
        est_boost = p.get("est_mult", 0)
        is_star_anchor = False
        if est_boost < chalk_min_boost:
            # Star anchor pathway: genuine scorers (season_pts >= 20) with decent boost
            # bypass the boost floor so lineups always include at least 1 high-PPG player.
            # Stars blocked: Jokic (+0.2x), Luka (+0.0x), Wemby (+0.3x) — kills EV.
            # Stars allowed: Josh Hart (+1.1x), OG Anunoby (+1.1x), Jalen Green (+1.6x).
            sa_cfg = _cfg("star_anchor", {})
            sa_enabled = sa_cfg.get("enabled", True) if isinstance(sa_cfg, dict) else True
            sa_min_pts = float(sa_cfg.get("min_season_pts", 20.0)) if isinstance(sa_cfg, dict) else 20.0
            sa_min_min = float(sa_cfg.get("min_season_min", 25.0)) if isinstance(sa_cfg, dict) else 25.0
            sa_min_boost = float(sa_cfg.get("min_boost", 0.8)) if isinstance(sa_cfg, dict) else 0.8
            sa_min_rating = float(sa_cfg.get("min_rating", 4.0)) if isinstance(sa_cfg, dict) else 4.0
            if (sa_enabled
                    and p.get("season_pts", 0) >= sa_min_pts
                    and p.get("season_min", 0) >= sa_min_min
                    and est_boost >= sa_min_boost
                    and p.get("rating", 0) >= sa_min_rating):
                is_star_anchor = True
            else:
                continue
        # Minimum rating gate: boost cannot rescue a weak base.
        # A player with rating < 3.5 doesn't generate enough RS to be a viable chalk pick
        # regardless of card boost. Kills Lopez (+3x, 11.6 pts) and Kennard (+3x, 8.9 pts)
        # archetypes where boost was masking a weak RS foundation.
        min_chalk_rating = _cfg("scoring_thresholds.min_chalk_rating", 3.5)
        if p["rating"] < min_chalk_rating:
            continue
        capped_boost = min(est_boost, boost_cap)
        p["capped_boost"] = capped_boost
        p["_is_star_anchor"] = is_star_anchor
        # Light matchup adjustment for chalk (narrower range than moonshot — reliability first)
        opp = p.get("opp", "")
        chalk_matchup = _compute_matchup_factor(p, opp, def_stats or {})
        chalk_matchup = max(
            float(_cfg("matchup.chalk_adj_min", 0.92)),
            min(float(_cfg("matchup.chalk_adj_max", 1.10)), chalk_matchup)
        )
        # MILP additive term only: pull boost toward neutral so projected RS competes with
        # high-boost role players. UI chalk_ev_capped still uses real capped_boost.
        _lu = _cfg("lineup", _CONFIG_DEFAULTS["lineup"])
        _rs_focus = max(0.0, min(1.0, float(_lu.get("chalk_milp_rs_focus", 0.0))))
        _boost_neutral = float(_lu.get("chalk_milp_boost_neutral", 1.0))
        p["chalk_milp_boost"] = round(
            (1.0 - _rs_focus) * float(capped_boost) + _rs_focus * _boost_neutral,
            4,
        )
        # Core value: RS × matchup × (slot + boost). No team motivation — disabled by audit.
        p["chalk_ev_capped"] = round(p["rating"] * chalk_matchup * (avg_slot + capped_boost), 2)
        chalk_eligible.append(p)

    # Star anchor: identify star candidate indices for the MILP constraint.
    # These are players that bypassed the boost floor via the star anchor pathway.
    # The MILP will require at least min_star_count of them in the lineup.
    sa_cfg = _cfg("star_anchor", {})
    sa_enabled = sa_cfg.get("enabled", True) if isinstance(sa_cfg, dict) else True
    sa_require = int(sa_cfg.get("require_count", 1)) if isinstance(sa_cfg, dict) else 1
    sa_max = int(sa_cfg.get("max_count", 0)) if isinstance(sa_cfg, dict) else 0
    chalk_star_indices = [
        i for i, p in enumerate(chalk_eligible)
        if p.get("_is_star_anchor", False)
    ] if sa_enabled else []

    # Lineup quality constraints — per-game cap and boost floors.
    _lu_cfg = _cfg("lineup", {})
    _chalk_max_per_game = int(_lu_cfg.get("chalk_max_per_game", 0)) if isinstance(_lu_cfg, dict) else 0
    _moon_max_per_game = int(_lu_cfg.get("moonshot_max_per_game", 0)) if isinstance(_lu_cfg, dict) else 0
    _chalk_min_high_boost = int(_lu_cfg.get("chalk_min_high_boost_count", 0)) if isinstance(_lu_cfg, dict) else 0
    _chalk_high_boost_thr = float(_lu_cfg.get("chalk_high_boost_threshold", 2.0)) if isinstance(_lu_cfg, dict) else 2.0
    _chalk_min_big_boost = int(_lu_cfg.get("chalk_min_big_boost_count", 0)) if isinstance(_lu_cfg, dict) else 0
    _chalk_big_boost_thr = float(_lu_cfg.get("chalk_big_boost_threshold", 2.8)) if isinstance(_lu_cfg, dict) else 2.8

    def _player_game_id(p):
        """Derive a canonical game ID from team + opp pair (order-independent)."""
        return "_vs_".join(sorted([p.get("team", ""), p.get("opp", "")]))

    # Core pool: when enabled, both lineups are built from the same 7–10 player core (two configurations).
    core_pool_cfg = _cfg("core_pool", _CONFIG_DEFAULTS.get("core_pool", {}))
    core_pool_enabled = core_pool_cfg.get("enabled", False) if isinstance(core_pool_cfg, dict) else False

    if not core_pool_enabled:
        _chalk_elig_games = [_player_game_id(p) for p in chalk_eligible]
        chalk = optimize_lineup(chalk_eligible, n=5, sort_key="chalk_ev_capped",
                                rating_key="rating", card_boost_key="chalk_milp_boost",
                                max_per_team=2,
                                star_indices=chalk_star_indices if chalk_star_indices else None,
                                min_star_count=sa_require if chalk_star_indices else 0,
                                max_star_count=sa_max if sa_max > 0 else 0,
                                max_per_game=_chalk_max_per_game,
                                player_games=_chalk_elig_games,
                                min_high_boost_count=_chalk_min_high_boost,
                                high_boost_threshold=_chalk_high_boost_thr,
                                min_big_boost_count=_chalk_min_big_boost,
                                big_boost_threshold=_chalk_big_boost_thr)

    # ── MOONSHOT: Contrarian EV strategy (v6 — matchup-aware) ───────────────
    # 5-day leaderboard analysis (Mar 8-13): winning moonshots are role players
    # with high card boost. Stars almost NEVER win moonshot (4/5 days pure role
    # players). The formula: maximize RS × (slot + boost) with matchup adjustment.
    #
    # Approach:
    #   - Card boost >= 1.5 (contrarian — low ownership)
    #   - Season/recent avg >= 15 min (was 25; Santos/Clifford/Ellis play 15-20)
    #   - Rating >= 2.0 (was 3.0; Ellis RS 2.2, Clifford RS 2.1 were winners)
    #   - RotoWire: only exclude confirmed OUT
    #   - Matchup factor: opponent def quality × Claude DvP intelligence (replaces dev-team bonus)
    #   - Boost leverage baked into adj_ceiling for MILP
    #   - No center penalty (Poeltl, Queta, Achiuwa all win)
    #   - Light variance damping (moonshot wants upside)
    # ─────────────────────────────────────────────────────────────────────────
    min_floor        = moon_cfg.get("min_minutes_floor", 15)
    min_recent_floor = moon_cfg.get("min_recent_minutes_floor", 15)
    min_boost        = moon_cfg.get("min_card_boost", 1.5)

    # Wildcard thresholds — read once outside the loop
    # Wildcard gate: ultra-high-boost deep bench players bypass season/recent min floors.
    wildcard_boost        = moon_cfg.get("wildcard_min_boost", 2.5)
    wildcard_min          = moon_cfg.get("wildcard_min_minutes", 10.0)
    wildcard_min_pts      = moon_cfg.get("wildcard_min_season_pts", 5.0)
    variance_penalty_coef = moon_cfg.get("variance_penalty", 0.15)
    # Matchup intel from Claude Layer 1.5 (pre-fetched, passed in)
    _matchup_intel = matchup_intel or {}

    # v5: Starting 5 and Moonshot can share players. They represent two independent
    # draft strategies (reliability vs ceiling) — if a player is the best pick for
    # both, they should appear in both. The two lineups together predict the two
    # best possible drafts of the day.
    moonshot_pool = []
    for p in projections:
        if p.get("name") in BLACKLISTED_PLAYERS:
            continue

        # Proven rotation regular OR definitive spot-starter due to injury cascade.
        # Condition A: Both historical floors met.
        # Condition B: predMin >= 28 AND _cascade_bonus >= 10 — prevents DFS Ghosts
        #   (trickle-cascade bench warmers) from clearing both gates simultaneously.
        # Condition C: Wildcard — ultra-high boost with higher minutes and a minimum
        #   season scoring floor so we don't chase pure cardio bench archetypes.
        season_min = p.get("season_min", 0)
        recent_min = p.get("recent_min", 0) or season_min  # fallback to season_min if missing/zero
        pred_min   = p.get("predMin", 0)
        season_pts = p.get("season_pts", p.get("pts", 0))
        est_mult   = p.get("est_mult", 0.3)

        # ── Moonshot eligibility ─────────────────────────────────────────────
        # 3 pathways (star anchor removed — boost dominance audit Mar 19):
        #   Regular:      season AND recent min meet floor
        #   Spot-starter: cascade injury gave them a starter role
        #   Role spike:   recent minutes >> season (role change/injury)
        is_moonshot_regular      = (season_min >= min_floor and recent_min >= min_recent_floor)
        is_moonshot_spot_starter = (pred_min >= 28.0 and p.get("_cascade_bonus", 0) >= 10.0)
        role_spike_ratio  = moon_cfg.get("role_spike_ratio", 1.4)
        role_spike_recent = moon_cfg.get("role_spike_recent_floor", 20.0)
        role_spike_season = moon_cfg.get("role_spike_season_floor", 10.0)
        is_role_spike = (
            recent_min >= role_spike_recent
            and season_min >= role_spike_season
            and recent_min >= season_min * role_spike_ratio
            and est_mult >= min_boost
        )
        # High-boost role player pathway: consistent rotation player whose value comes
        # from a high card boost + real RS projection, not from minutes volume.
        # Example: 16 min/game player with +2.5x boost and RS 4.5 → chalk_ev 18.45 —
        # beats many starters on EV. The boost floor IS the quality gate here.
        hbr_cfg = moon_cfg.get("high_boost_role", {})
        hbr_enabled     = hbr_cfg.get("enabled", True)
        hbr_min_boost   = float(hbr_cfg.get("min_boost", 2.0))
        hbr_min_recent  = float(hbr_cfg.get("min_recent_min", 14.0))
        hbr_min_pred    = float(hbr_cfg.get("min_pred_min", 14.0))
        is_high_boost_role = (
            hbr_enabled
            and est_mult >= hbr_min_boost
            and recent_min >= hbr_min_recent
            and pred_min >= hbr_min_pred
        )
        if not (is_moonshot_regular or is_moonshot_spot_starter
                or is_role_spike or is_high_boost_role):
            continue

        # Never draft a moonshot player projected well below their season minute average.
        # Wider tolerance than chalk (default 3.0 min) — moonshot is contrarian.
        moon_min_tol = float(moon_cfg.get("pred_min_tolerance", 3.0))
        if pred_min < (season_min - moon_min_tol):
            continue

        # Hard boost floor — with RS-bypass for high-RS proven scorers.
        if est_mult < min_boost:
            rs_bypass = moon_cfg.get("rs_bypass", {})
            if (rs_bypass.get("enabled", False)
                    and p.get("rating", 0) >= float(rs_bypass.get("min_rating", 5.0))
                    and season_min >= float(rs_bypass.get("min_season_min", 25.0))
                    and est_mult >= float(rs_bypass.get("min_boost", 0.3))):
                pass  # high-RS bypass
            else:
                continue

        # RotoWire: only exclude confirmed OUT players.
        if use_rotowire and rw_statuses and p.get("injury_status", "").upper() == "OUT":
            continue

        # Minimum rating floor — all pathways use the same floor.
        # Exception: confirmed rotation players with high boost pass a lower floor.
        # This captures cascade-elevated backups (e.g. Garza when Porzingis is OUT)
        # and confirmed role players with very high boosts (e.g. Taylor Hendricks +3.0x).
        min_rating_floor = moon_cfg.get("min_rating_floor", 3.0)
        if p["rating"] < min_rating_floor:
            roto_confirmed_min_rating = float(moon_cfg.get("roto_confirmed_min_rating", 2.2))
            roto_confirmed_min_boost = float(moon_cfg.get("roto_confirmed_min_boost", 2.5))
            _roto_entry = (rw_statuses or {}).get(p["name"].lower(), {})
            _roto_status = _roto_entry.get("status", "unknown")
            _is_roto_confirmed = _roto_status in ("confirmed", "expected")
            _has_cascade = p.get("_cascade_bonus", 0) >= 5.0
            if (_is_roto_confirmed or _has_cascade) and est_mult >= roto_confirmed_min_boost and p["rating"] >= roto_confirmed_min_rating:
                pass  # confirmed rotation player with meaningful boost — allow through
            else:
                continue

        # ── Moonshot EV (boost-dominance formula) ──────────────────────────────
        # Core formula: RS × boost_leverage × (slot + boost)
        # Boost dominance audit (Mar 19): removed variance penalty (moonshot IS variance),
        # removed Claude matchup factor (noise). Light math matchup kept.
        opp_abbr = p.get("opp", "")
        matchup_factor = _compute_matchup_factor(p, opp_abbr, def_stats or {})
        matchup_factor = max(
            float(_cfg("matchup.moonshot_adj_min", 0.90)),
            min(float(_cfg("matchup.moonshot_adj_max", 1.15)), matchup_factor)
        )

        # Boost leverage — the core moonshot signal. High-boost players get
        # exponentially more weight so the MILP strongly favors them.
        boost_power = moon_cfg.get("boost_leverage_power", 1.6)
        boost_leverage = max(est_mult, 0.2) ** boost_power

        adj_ceiling = round(p["rating"] * matchup_factor * boost_leverage, 3)

        # Scoring bias: reward players who actually score over pure rebounders/defenders.
        # Leaderboard data (Mar 15-19): winners are scorers (Harkless RS 4.2, Bailey RS 5.0,
        # Williams RS 4.0, Gillespie RS 4.5) — not big men accumulating reb/blk only.
        pts_bias_threshold = float(moon_cfg.get("scoring_pts_bias_threshold", 10.0))
        pts_bias_scale = float(moon_cfg.get("scoring_pts_bias_scale", 0.0))
        if pts_bias_scale > 0 and p.get("pts", 0) > pts_bias_threshold:
            pts_bias = 1.0 + (p["pts"] - pts_bias_threshold) * pts_bias_scale
            adj_ceiling = round(adj_ceiling * pts_bias, 3)

        # Scorer upside: efficient scorers with real volume get a moonshot ceiling boost.
        # Data: Jalen Green, Aaron Nesmith, DeRozan under-projected on multiple dates —
        # they generate RS 4.5–6.8 when hot but project 2.6–4.1 from season averages.
        # Only applies when season_min data is present (not inferred projections).
        scorer_upside_cfg = moon_cfg.get("scorer_upside", {})
        if scorer_upside_cfg.get("enabled", True):
            su_min_ppm = float(scorer_upside_cfg.get("min_pts_per_min", 0.55))
            su_min_pts = float(scorer_upside_cfg.get("min_season_pts", 15.0))
            su_mult = float(scorer_upside_cfg.get("multiplier", 1.10))
            su_ppm = p.get("pts", 0) / max(p.get("season_min", 1) or 1, 1)
            su_season_pts = p.get("season_pts", p.get("pts", 0)) or 0
            if su_ppm >= su_min_ppm and su_season_pts >= su_min_pts:
                adj_ceiling = round(adj_ceiling * su_mult, 3)

        # Moonshot EV: MILP will optimize slot assignment on top
        moonshot_ev = round(adj_ceiling * (avg_slot + est_mult), 2)

        moonshot_pool.append({
            **p,
            "moonshot_ev":    moonshot_ev,
            "adj_ceiling":    adj_ceiling,
            "_matchup_factor": round(matchup_factor, 3),
            "_rw_cleared":    True,
        })

    # No center cap — position balancing removed (boost dominance audit Mar 19).
    # Poeltl, Queta, Achiuwa all appear in winning lineups.
    moonshot_max_team = int(moon_cfg.get("max_per_team", 2))

    if core_pool_enabled:
        # ── Core pool path: one 7–10 player core; both lineups are configurations of it ──
        def _moonshot_ev_for_player(p, _avg_slot, _moon_cfg, _def_stats, _matchup_intel_map):
            _est = p.get("est_mult", 0.3)
            _boost_power = _moon_cfg.get("boost_leverage_power", 1.6)
            _boost_leverage = max(_est, 0.2) ** _boost_power
            # Simplified: rating × matchup × boost_leverage (no variance, no Claude)
            _opp = p.get("opp", "")
            _matchup = _compute_matchup_factor(p, _opp, _def_stats or {})
            _matchup = max(
                float(_cfg("matchup.moonshot_adj_min", 0.90)),
                min(float(_cfg("matchup.moonshot_adj_max", 1.15)), _matchup)
            )
            _adj = round(p["rating"] * _matchup * _boost_leverage, 3)
            _pts_threshold = float(_moon_cfg.get("scoring_pts_bias_threshold", 10.0))
            _pts_scale = float(_moon_cfg.get("scoring_pts_bias_scale", 0.0))
            if _pts_scale > 0 and p.get("pts", 0) > _pts_threshold:
                _adj = round(_adj * (1.0 + (p["pts"] - _pts_threshold) * _pts_scale), 3)
            return round(_adj * (_avg_slot + _est), 2), _adj

        moonshot_by_name = {p["name"]: p for p in moonshot_pool}
        eligible_union = []
        # Core pool = players eligible for BOTH chalk and moonshot (intersection),
        # plus chalk-only and moonshot-only with computed cross-EVs.
        # Both Starting 5 and Moonshot MILP run on the same core pool — one pool,
        # two configurations (reliability vs ceiling). No separate filtering needed.
        for p in chalk_eligible:
            rec = {**p, "chalk_ev_capped": p["chalk_ev_capped"], "capped_boost": p["capped_boost"]}
            if p["name"] in moonshot_by_name:
                rec["moonshot_ev"] = moonshot_by_name[p["name"]]["moonshot_ev"]
                rec["adj_ceiling"] = moonshot_by_name[p["name"]]["adj_ceiling"]
            else:
                _mev, _adj = _moonshot_ev_for_player(p, avg_slot, moon_cfg, def_stats, _matchup_intel)
                rec["moonshot_ev"] = _mev
                rec["adj_ceiling"] = _adj
            eligible_union.append(rec)
        # Moonshot-only players (pass moonshot gates but not chalk gates) are NOT added
        # to the core pool. The core pool is one shared pool — if a player can't make
        # Starting 5, they shouldn't be in the pool at all. This prevents the chalk gate
        # bypass where sub-4.0-RS moonshot-only players leaked into Starting 5.

        core_size = min(int(core_pool_cfg.get("size", 8)), max(5, len(eligible_union)))
        core_metric = (core_pool_cfg.get("metric") or "max_ev").lower()
        blend_w = float(core_pool_cfg.get("blend_weight", 0.5))
        for r in eligible_union:
            ce, me = r.get("chalk_ev_capped", 0), r.get("moonshot_ev", 0)
            if core_metric == "rs":
                r["_core_score"] = r.get("rating", 0)
            elif core_metric == "max_ev":
                r["_core_score"] = max(ce, me)
            else:
                r["_core_score"] = blend_w * ce + (1 - blend_w) * me
        eligible_union.sort(key=lambda x: x.get("_core_score", 0), reverse=True)
        core_pool = eligible_union[:core_size]

        # Core pool is built exclusively from chalk_eligible — all players passed
        # chalk gates (4.0 RS, 22 min, 7 pts, 0.28 ppm). Both MILP runs use core_pool.
        # If core pool has <5 players, fall back to full chalk_eligible.
        chalk_source = core_pool if len(core_pool) >= 5 else chalk_eligible
        # Star indices relative to chalk_source
        _chalk_source_names = [p["name"] for p in chalk_source]
        _core_star_indices = [
            i for i, p in enumerate(chalk_source)
            if p.get("_is_star_anchor", False)
        ] if sa_enabled else []
        _chalk_source_games = [_player_game_id(p) for p in chalk_source]
        chalk = optimize_lineup(chalk_source, n=5, sort_key="chalk_ev_capped",
                                rating_key="rating", card_boost_key="chalk_milp_boost",
                                max_per_team=2,
                                objective_mode="chalk",
                                variance_penalty=0.5,
                                star_indices=_core_star_indices if _core_star_indices else None,
                                min_star_count=sa_require if _core_star_indices else 0,
                                max_star_count=sa_max if sa_max > 0 else 0,
                                max_per_game=_chalk_max_per_game,
                                player_games=_chalk_source_games,
                                min_high_boost_count=_chalk_min_high_boost,
                                high_boost_threshold=_chalk_high_boost_thr,
                                min_big_boost_count=_chalk_min_big_boost,
                                big_boost_threshold=_chalk_big_boost_thr)
        chalk_ids = [p.get("id") for p in chalk if p.get("id")]
        # Moonshot uses the full moonshot_pool (boost-ranked contrarians), NOT core_pool.
        # core_pool is built exclusively from chalk_eligible — it excludes moonshot-only
        # players (high-boost role players who fail chalk gates like 7 PPG or 20 min floor).
        # Using core_pool for moonshot caused it to pick the same RS-ranked stars as chalk.
        # moonshot_pool has its own eligibility gates (boost >= min_card_boost, RS >= 3.0,
        # high-boost-role pathway for 2.0x+/14min players) and ranks by moonshot_ev which
        # uses boost_leverage_power to strongly favor 3.0x boost players.
        # overlap_cap=3 ensures Moonshot differentiates from Starting 5 (max 3 shared players).
        # No star_count constraints — contrarian moonshot skips the star anchor concept.
        _moon_pool_games = [_player_game_id(p) for p in moonshot_pool]
        upside = optimize_lineup(moonshot_pool, n=5, sort_key="moonshot_ev",
                                 rating_key="adj_ceiling",
                                 card_boost_key="est_mult",
                                 max_per_team=moonshot_max_team,
                                 objective_mode="moonshot",
                                 variance_uplift=0.35,
                                 boost_leverage_extra_power=0.2,
                                 overlap_player_ids=chalk_ids,
                                 overlap_cap=3,
                                 two_phase=True,
                                 raw_rating_key="rating",
                                 max_per_game=_moon_max_per_game,
                                 player_games=_moon_pool_games)
    else:
        _moon_pool_games = [_player_game_id(p) for p in moonshot_pool]
        upside = optimize_lineup(moonshot_pool, n=5, sort_key="moonshot_ev",
                                 rating_key="adj_ceiling",
                                 card_boost_key="est_mult",
                                 max_per_team=moonshot_max_team,
                                 objective_mode="moonshot",
                                 variance_uplift=0.35,
                                 boost_leverage_extra_power=0.2,
                                 two_phase=True,
                                 raw_rating_key="rating",
                                 max_per_game=_moon_max_per_game,
                                 player_games=_moon_pool_games)
        core_pool = None

    return [_normalize_player(p) for p in chalk], [_normalize_player(p) for p in upside], core_pool


# grep: WATCHLIST — _build_watchlist, lineup-sensitive players, Pass 2 triggers
def _build_watchlist(chalk, upside, all_proj, games):
    """Identify players whose value is sensitive to late-breaking news.

    A watchlist player = someone NOT in the lineup whose projected value would
    spike if a specific event occurs (injury to a teammate/starter, etc.).

    Returns list of {player, trigger_event, depends_on, current_rating,
    projected_boost_rating, game_id} dicts.
    """
    watchlist = []
    lineup_names = set()
    for p in (chalk or []) + (upside or []):
        lineup_names.add(p.get("name", ""))

    # Build a map of team → players for cascade analysis
    team_players = {}
    for p in all_proj:
        t = p.get("team", "")
        team_players.setdefault(t, []).append(p)

    # For each player NOT in the lineup, check if an injury to a lineup player
    # on the same team would cause a cascade that makes them lineup-worthy
    for p in all_proj:
        pname = p.get("name", "")
        if pname in lineup_names or not pname:
            continue

        rating = p.get("rating", 0)
        est_mult = p.get("est_mult", 0)
        team = p.get("team", "")
        season_min = p.get("season_min", 0)
        pos = p.get("pos", "")

        # Skip players with very low baseline (won't help even with cascade)
        if rating < 2.0 or season_min < 12:
            continue

        # Check each lineup player on the same team — if they went OUT,
        # this player would get a ~5-10 minute cascade bonus
        for lp in (chalk or []) + (upside or []):
            if lp.get("team") != team:
                continue
            lp_name = lp.get("name", "")
            lp_min = lp.get("season_min", 0) or lp.get("predMin", 0)
            if lp_min < 15:
                continue

            # Estimate cascade: player inherits ~30% of the OUT player's minutes
            # (conservative — real cascade depends on position/rotation)
            cascade_min = lp_min * 0.30
            boosted_rating = rating * min((season_min + cascade_min) / max(season_min, 1), 1.35)

            # Only add to watchlist if cascade would push them above lineup threshold
            min_chalk = float(_cfg("scoring_thresholds.min_chalk_rating", 3.5))
            if boosted_rating >= min_chalk and est_mult >= 1.0:
                watchlist.append({
                    "player": pname,
                    "trigger_event": "injury_cascade",
                    "depends_on": lp_name,
                    "current_rating": round(rating, 2),
                    "projected_boost_rating": round(boosted_rating, 2),
                    "est_mult": round(est_mult, 1),
                    "team": team,
                })
                break  # One watchlist entry per player is enough

    # Sort by projected upside (highest potential first), limit to top 10
    watchlist.sort(key=lambda w: w.get("projected_boost_rating", 0), reverse=True)
    return watchlist[:10]


# ─────────────────────────────────────────────────────────────────────────────
# PER-GAME LINEUP BUILDER
#
# Single-game drafts are fundamentally different from full-slate:
# - Only 2 teams, so everyone is picking from the same pool
# - Must diversify across both teams (min 2 per side)
# - Stars are MORE important in single-game (smaller pool = stars stand out)
# - Ownership is more concentrated, so contrarian plays matter more
# ─────────────────────────────────────────────────────────────────────────────

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
    """Build exactly ONE lineup ('THE LINE UP') for a single-game draft.

    Per-game drafts restrict to a single 5-player format. No Starting 5 / Moonshot
    split — both users draft from the same 2-team pool, so card boost is irrelevant.
    Optimized purely by projected Real Score × slot multiplier.
    """
    game_chalk_floor = _cfg("lineup.game_chalk_rating_floor", 3.5)
    rescored = _apply_game_script(projections, game)

    # PER-GAME: Requires min recent minutes — configurable via lineup.game_recent_min_floor
    # (default 15; was 20 which excluded role players like Braun/GPII who pop in single-game).
    # Also enforces pts floor: a player projecting < 8 pts is a ceiling liability
    # in single-game format where card boost is irrelevant.
    game_min_floor = _cfg("lineup.game_recent_min_floor", 15.0)
    min_game_pts = _cfg("scoring_thresholds.min_game_pts", 8.0)
    eligible_pool = [
        p for p in rescored
        if p.get("recent_min", 0) >= game_min_floor
        and p["rating"] >= game_chalk_floor
        and p.get("pts", 0) >= min_game_pts
        and p.get("name") not in BLACKLISTED_PLAYERS
    ]

    # Per-game: card boost is irrelevant (everyone drafts from the same pool).
    # Optimize purely by RS × slot multiplier — zero out est_mult for MILP.
    no_boost = [{**p, "est_mult": 0} for p in eligible_pool]
    the_lineup = optimize_lineup(no_boost, n=5, min_per_team=2, sort_key="rating",
                                 rating_key="rating", card_boost_key="est_mult")

    # Fill to 5 if the pool was smaller than 5 after floor filtering
    if len(the_lineup) < 5:
        lineup_names = {p["name"] for p in the_lineup}
        fill_pool = sorted(
            [
                p for p in rescored
                if p["name"] not in lineup_names
                and p.get("recent_min", 0) >= 20.0
                and p.get("name") not in BLACKLISTED_PLAYERS
            ],
            key=lambda p: p.get("rating", 0), reverse=True
        )
        for p in fill_pool:
            if len(the_lineup) >= 5:
                break
            the_lineup.append(p)

    # Per-game: zero out est_mult in returned data too (not just MILP input)
    # so the frontend never renders a misleading card boost pill.
    return {"the_lineup": [_normalize_player({**p, "est_mult": 0}) for p in the_lineup]}


# ─────────────────────────────────────────────────────────────────────────────
# SCORE BOUNDARY VALIDATION
# grep: _lineup_ev_total, score_bounds, _get_mock_slate
#
# "Total projected draft score" for audit/monitoring purposes:
#   - Slate-Wide (chalk/moonshot): sum of rating × (slot_numeric + est_mult) per player
#   - Per-Game (the_lineup):       sum of rating only (no card boost in per-game)
#
# Expected ranges:
#   - chalk:       70–100
#   - upside:      70–100
#   - the_lineup:  25–35
# ─────────────────────────────────────────────────────────────────────────────

_SLOT_NUMS = {"2.0x": 2.0, "1.8x": 1.8, "1.6x": 1.6, "1.4x": 1.4, "1.2x": 1.2}
_SCORE_BOUNDS = {
    "chalk":      (70.0, 100.0),
    "upside":     (70.0, 100.0),
    "the_lineup": (25.0, 35.0),
}

def _lineup_ev_total(lineup: list, mode: str) -> float:
    """Compute total projected draft score for a 5-player lineup.
    Mode 'chalk'/'upside': sum(rating × (slot_numeric + est_mult)).
    Mode 'the_lineup': sum(rating) — no card boost in per-game drafts."""
    total = 0.0
    for p in lineup:
        r = float(p.get("rating") or 0)
        if mode == "the_lineup":
            total += r
        else:
            slot = _SLOT_NUMS.get(p.get("slot", ""), 1.6)
            boost = float(p.get("est_mult") or 0)
            total += r * (slot + boost)
    return round(total, 1)


def _score_bounds_for_lineups(lineups: dict) -> dict:
    """Return total EV and in-range flag for each lineup type in a lineups dict."""
    bounds = {}
    for mode, players in lineups.items():
        if not players:
            continue
        total = _lineup_ev_total(players, mode)
        lo, hi = _SCORE_BOUNDS.get(mode, (0, 9999))
        in_range = lo <= total <= hi
        if not in_range:
            print(f"[score-bounds] {mode} total {total} outside expected {lo}–{hi}")
        bounds[mode] = {"total": total, "lo": lo, "hi": hi, "in_range": in_range}
    return bounds


# ─────────────────────────────────────────────────────────────────────────────
# MOCK SLATE — deterministic test data for Phase 0 audit validation
# Returned by GET /api/slate?mock=true. No ESPN, LightGBM, or cache I/O.
# Scores are pre-computed to fall within expected bounds:
#   chalk / upside totals ≈ 76.6 / 71.3 (both in 70–100)
#   per-game total ≈ 28.0 (in 25–35)
# ─────────────────────────────────────────────────────────────────────────────

def _get_mock_slate() -> dict:
    """Return a fully-formed mock slate for testing. No side effects."""
    chalk = [
        {"id":"mock1","name":"Mock Player 1","pos":"PG","team":"LAL","rating":7.0,"predMin":36.0,
         "pts":22.0,"reb":4.0,"ast":7.0,"stl":1.2,"blk":0.3,"est_mult":0.8,"slot":"2.0x",
         "chalk_ev":19.6,"moonshot_ev":0.0,"injury_status":"","_decline":0.0},
        {"id":"mock2","name":"Mock Player 2","pos":"SF","team":"BOS","rating":5.5,"predMin":34.0,
         "pts":18.0,"reb":6.0,"ast":3.0,"stl":1.0,"blk":0.5,"est_mult":1.2,"slot":"1.8x",
         "chalk_ev":16.5,"moonshot_ev":0.0,"injury_status":"","_decline":0.0},
        {"id":"mock3","name":"Mock Player 3","pos":"C","team":"LAL","rating":4.5,"predMin":30.0,
         "pts":14.0,"reb":10.0,"ast":2.0,"stl":0.5,"blk":1.8,"est_mult":1.5,"slot":"1.6x",
         "chalk_ev":13.95,"moonshot_ev":0.0,"injury_status":"","_decline":0.0},
        {"id":"mock4","name":"Mock Player 4","pos":"SG","team":"BOS","rating":4.0,"predMin":28.0,
         "pts":13.0,"reb":3.0,"ast":4.0,"stl":1.0,"blk":0.2,"est_mult":2.0,"slot":"1.4x",
         "chalk_ev":13.6,"moonshot_ev":0.0,"injury_status":"","_decline":0.0},
        {"id":"mock5","name":"Mock Player 5","pos":"PF","team":"LAL","rating":3.5,"predMin":24.0,
         "pts":10.0,"reb":7.0,"ast":1.5,"stl":0.8,"blk":0.6,"est_mult":2.5,"slot":"1.2x",
         "chalk_ev":12.95,"moonshot_ev":0.0,"injury_status":"","_decline":0.0},
    ]
    upside = [
        {"id":"mock6","name":"Mock Moonshot 1","pos":"G","team":"IND","rating":3.5,"predMin":22.0,
         "pts":9.0,"reb":3.0,"ast":4.0,"stl":1.0,"blk":0.1,"est_mult":3.5,"slot":"2.0x",
         "chalk_ev":9.45,"moonshot_ev":194.3,"injury_status":"","_decline":0.0},
        {"id":"mock7","name":"Mock Moonshot 2","pos":"F","team":"UTA","rating":3.0,"predMin":20.0,
         "pts":8.0,"reb":5.0,"ast":2.0,"stl":0.7,"blk":0.4,"est_mult":3.0,"slot":"1.8x",
         "chalk_ev":7.2,"moonshot_ev":126.0,"injury_status":"","_decline":0.0},
        {"id":"mock8","name":"Mock Moonshot 3","pos":"G","team":"BKN","rating":2.5,"predMin":20.0,
         "pts":7.0,"reb":2.0,"ast":3.5,"stl":1.2,"blk":0.0,"est_mult":4.0,"slot":"1.6x",
         "chalk_ev":6.0,"moonshot_ev":100.8,"injury_status":"","_decline":0.0},
        {"id":"mock9","name":"Mock Moonshot 4","pos":"F","team":"SAC","rating":2.5,"predMin":19.0,
         "pts":7.0,"reb":4.0,"ast":1.5,"stl":0.6,"blk":0.3,"est_mult":3.5,"slot":"1.4x",
         "chalk_ev":5.5,"moonshot_ev":82.6,"injury_status":"","_decline":0.0},
        {"id":"mock10","name":"Mock Moonshot 5","pos":"C","team":"MEM","rating":2.0,"predMin":18.0,
         "pts":5.0,"reb":7.0,"ast":0.5,"stl":0.3,"blk":1.5,"est_mult":4.5,"slot":"1.2x",
         "chalk_ev":4.2,"moonshot_ev":50.9,"injury_status":"","_decline":0.0},
    ]
    mock_games = [
        {"gameId":"mock_game_1","label":"LAL vs BOS","home":{"abbr":"LAL"},"away":{"abbr":"BOS"},
         "startTime":None,"locked":False,"draftable":True},
        {"gameId":"mock_game_2","label":"IND vs UTA","home":{"abbr":"IND"},"away":{"abbr":"UTA"},
         "startTime":None,"locked":False,"draftable":True},
    ]
    lineups = {"chalk": chalk, "upside": upside}
    return {
        "date": _et_date().isoformat(),
        "mock": True,
        "games": mock_games,
        "lineups": lineups,
        "locked": False,
        "all_complete": False,
        "draftable_count": 2,
        "lock_time": None,
        "score_bounds": _score_bounds_for_lineups(lineups),
    }


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

# ═════════════════════════════════════════════════════════════════════════════
# CORE API ENDPOINTS
# grep: /api/games, /api/slate, /api/picks, /api/save-predictions, /api/refresh
# /api/hindsight, /api/log, /api/parse-screenshot, /api/save-actuals, /api/health
# ═════════════════════════════════════════════════════════════════════════════
CRON_SECRET = os.getenv("CRON_SECRET", "")

# Rate limiting: in-memory sliding window per IP for expensive endpoints (thread-safe)
_RATE_LIMIT_STORE = {}  # (ip, path_key) -> [timestamps]
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMITS = {"parse-screenshot": 5, "lab/chat": 20, "line-of-the-day": 10}

def _client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return (forwarded.split(",")[0].strip() or (request.client.host if request.client else "") or "unknown")

def _check_rate_limit(request: Request, path_key: str):
    """Return JSONResponse(429) if over limit, else None."""
    if path_key not in _RATE_LIMITS:
        return None
    ip = _client_ip(request)
    key = (ip, path_key)
    now = datetime.now(timezone.utc).timestamp()
    with _RATE_LIMIT_LOCK:
        if key not in _RATE_LIMIT_STORE:
            _RATE_LIMIT_STORE[key] = []
        times = _RATE_LIMIT_STORE[key]
        times[:] = [t for t in times if now - t < _RATE_LIMIT_WINDOW]
        limit = _RATE_LIMITS[path_key]
        if len(times) >= limit:
            return JSONResponse({"error": "Too many requests", "retry_after": _RATE_LIMIT_WINDOW}, status_code=429)
        times.append(now)
    return None


def _require_cron_secret(request: Request):
    """Return True if request is authorized (cron or no CRON_SECRET set). Used by cron-only endpoints.
    Accepts Authorization: Bearer <CRON_SECRET> or ?key=<CRON_SECRET> so manual GET /api/refresh works
    (e.g. myurl/api/refresh?key=SECRET). Keep the URL private — it is sensitive."""
    if not CRON_SECRET:
        return True  # backward compat: no secret configured => allow
    auth = request.headers.get("authorization", "")
    if auth == f"Bearer {CRON_SECRET}":
        return True
    if request.query_params.get("key") == CRON_SECRET:
        return True
    return False


@app.get("/api/health")
async def health() -> dict:
    """Lightweight health check for uptime monitoring. Returns 200 and optionally checks GitHub + config."""
    out = {"status": "ok"}
    try:
        _load_config()
        out["config"] = "ok"
    except Exception as e:
        out["config"] = "error"
        print(f"[health] config load: {e}")
    if GITHUB_TOKEN and GITHUB_REPO:
        c, _ = _github_get_file("data/model-config.json")
        out["github"] = "ok" if c is not None else "unreachable"
    else:
        out["github"] = "skipped"
    return out


@app.get("/api/version")
async def version() -> dict:
    """Return build/deploy identifier for 'what is deployed' checks. Set RAILWAY_GIT_COMMIT_SHA at build time."""
    sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "")
    return {"version": sha[:7] if sha else "unknown"}


@app.get("/api/games")
async def get_games():
    """Never returns 500; on exception returns 200 with empty list."""
    try:
        games = fetch_games()
        for g in games:
            st = g.get("startTime", "")
            g["locked"] = _is_locked(st) if st else False
            g["draftable"] = not _is_past_lock_window(st) if st else False
        return games
    except Exception as e:
        print(f"[games] error: {e}")
        return []

def _force_regenerate_bg(*_args):
    """Sentinel function used as attribute namespace for _in_flight flag."""
    pass

def _force_regenerate_bg_worker():
    """Background thread: runs _force_regenerate_sync("full") after deploy SHA mismatch.
    Sets _force_regenerate_bg._in_flight = False when done (so a subsequent deploy
    can re-trigger if needed)."""
    try:
        print("[force-regen-bg] starting background regeneration (deploy SHA mismatch)")
        result = _force_regenerate_sync("full")
        print(f"[force-regen-bg] done: {result.get('status')}, games={result.get('games_regenerated', 0)}")
    except Exception as e:
        print(f"[force-regen-bg] error: {e}")
        traceback.print_exc()
    finally:
        _force_regenerate_bg._in_flight = False


def _get_slate_impl():
    """Inner slate computation; get_slate() wraps this in try/except so we never return 500."""
    # Fast path: if today's locked slate is already in memory, skip ALL external calls.
    today_str = _et_date().isoformat()
    _lk_pre = _lg("slate_v5_locked")
    if _lk_pre and _lk_pre.get("date") == today_str:
        _lk_pre["locked"] = True
        _lk_pre.setdefault("draftable_count", 0)
        return _lk_pre
    # Cold-start fast path: check GitHub backup and fetch games in parallel.
    # The backup is written at lock-promotion time so it exists on most cold starts.
    # Running both concurrently saves 1-2s vs sequential on every cold start.
    with ThreadPoolExecutor(max_workers=2) as _pre_pool:
        _gh_pre_fut = _pre_pool.submit(_slate_restore_from_github)
        _games_fut  = _pre_pool.submit(fetch_games)
        _gh_pre = _gh_pre_fut.result()
        games   = _games_fut.result()
    if _gh_pre and _gh_pre.get("date") == today_str and _gh_pre.get("locked"):
        _gh_pre["locked"] = True
        _gh_pre.setdefault("draftable_count", 0)
        _ls("slate_v5_locked", _gh_pre)
        return _gh_pre

    # Midnight rollover guard: if none of today's games have started yet but
    # yesterday's games are still in progress, hold yesterday's locked slate.
    # Handles the common case where a 10 PM ET tip-off runs past midnight and
    # _et_date() has already advanced to the next day.
    # Time-gate: only applies before 6 AM ET — after that, today is a new slate day
    # even if no games have started yet. Without this gate, the guard fires all day
    # pre-tip-off (e.g. 2 PM ET) and incorrectly serves yesterday's stale slate.
    try:
        from zoneinfo import ZoneInfo as _ZI
        _et_hour = datetime.now(_ZI("America/New_York")).hour
    except ImportError:
        now_utc = datetime.now(timezone.utc)
        _et_hour = (now_utc + timedelta(hours=-4 if 3 < now_utc.month < 11 else -5)).hour
    any_today_started = any(_is_past_lock_window(g.get("startTime", "")) for g in games)
    if not any_today_started and _et_hour < 6:
        _, remaining_yesterday, _, _ = _all_games_final(games)
        if remaining_yesterday > 0:
            yesterday = (_et_date() - timedelta(days=1)).isoformat()
            lock_cached = _lg("slate_v5_locked", yesterday)
            if lock_cached:
                lock_cached["locked"] = True
                lock_cached.setdefault("all_complete", False)
                return lock_cached
            try:
                content, _ = _github_get_file(f"data/locks/{yesterday}_slate.json")
                if content:
                    gh = json.loads(content)
                    gh["locked"] = True
                    gh.setdefault("all_complete", False)
                    _ls("slate_v5_locked", gh, yesterday)
                    return gh
            except Exception:
                pass
            # No yesterday cache available — fall through to today's generation

    if not games:
        today = _et_date()
        next_slate = _find_next_slate_date(today + timedelta(days=1))
        return {
            "date": today.isoformat(),
            "games": [],
            "lineups": {"chalk": [], "upside": []},
            "locked": False,
            "draftable_count": 0,
            "no_games": True,
            "next_slate_date": next_slate.isoformat() if next_slate else None,
        }

    # Only project games that are still draftable (not yet past lock window).
    draftable_games = [g for g in games if not _is_past_lock_window(g.get("startTime", ""))]

    if not draftable_games:
        # All today's games have passed their lock window (draftable_games is empty).
        # This means tip-off has occurred — slate is locked regardless of how many games
        # are final. Do NOT gate on finals>0: that causes an unlock bug during the first
        # ~30 min of play (before any game finishes) and on ESPN API failures.

        # Check in-memory lock cache FIRST — avoids a full pipeline re-run.
        # BUT: always re-check ESPN for all_complete so the frontend can detect
        # game completion and transition to the next-day slate. _all_games_final
        # has its own 60s internal cache, so this is cheap on warm instances.
        lock_cached = _lg("slate_v5_locked")
        if lock_cached:
            lock_cached["locked"] = True
            lock_cached.setdefault("draftable_count", 0)
            # Refresh all_complete from ESPN (cached 60s) so warm instances
            # detect game completion instead of serving stale all_complete=False.
            if not lock_cached.get("all_complete"):
                all_final, _rem, _fin, _lrs = _all_games_final(games)
                if all_final and _fin > 0:
                    lock_cached["all_complete"] = True
                    _ls("slate_v5_locked", lock_cached)
            return lock_cached

        # Cache miss (cold start) — check regular /tmp cache and GitHub backup BEFORE
        # calling ESPN. _cg("slate_v5") survives longer than the lock cache on partial
        # warm instances; GitHub backup exists on true cold starts after lock-promotion.
        reg_cached = _cg("slate_v5")
        gh_backup = None if reg_cached else _slate_restore_from_github()
        # Now call ESPN once to get all_complete status (only reached on cold start).
        all_final, remaining, finals, _lrs = _all_games_final(games)
        all_complete = all_final and finals > 0

        if reg_cached:
            reg_cached["locked"] = True
            reg_cached["all_complete"] = all_complete
            reg_cached.setdefault("draftable_count", 0)
            _ls("slate_v5_locked", reg_cached)
            _slate_backup_to_github(reg_cached)
            return reg_cached
        if gh_backup is None:
            gh_backup = _slate_restore_from_github()
        if gh_backup:
            gh_backup["locked"] = True
            gh_backup["all_complete"] = all_complete
            gh_backup.setdefault("draftable_count", 0)
            _ls("slate_v5_locked", gh_backup)
            return gh_backup
        # All caches empty — forced regeneration after config bust during a locked slate.
        # (e.g. model config changed post-lock, user hit Refresh to get new picks)
        # Fall through to the full pipeline using all today's games.
        draftable_games = games

    # lock_time and locked status both use the FIRST game of the entire day (all games,
    # not just draftable ones). Once the 6 PM game locks at 5:55 PM, the slate stays
    # locked for the rest of the day — even between game windows when mid-day games
    # are already in progress but the next batch hasn't hit their lock window yet.
    start_times = [g["startTime"] for g in draftable_games if g.get("startTime")]
    earliest = min(start_times) if start_times else None
    all_start_times = [g["startTime"] for g in games if g.get("startTime")]
    earliest_all = min(all_start_times) if all_start_times else earliest
    # any() not min() — on split-window days (e.g. 2 PM + 9 PM CT), the earliest
    # game's 6h ceiling can expire while late games are still live. any() stays
    # locked as long as ANY game is within its lock window.
    locked = any(_is_locked(st) for st in all_start_times) if all_start_times else False
    lock_time = None
    if earliest_all:
        try:
            lock_buf = _cfg("projection.lock_buffer_minutes", 5)
            gs = datetime.fromisoformat(earliest_all.replace("Z", "+00:00")).astimezone(timezone.utc)
            lock_dt = gs - timedelta(minutes=lock_buf)
            lock_time = lock_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass

    if locked:
        # Multi-instance: refresh busts GitHub but other containers still have
        # stale slate_v5_locked in /tmp — drop local files when GitHub says busted.
        try:
            if _github_slate_bust_active():
                _clear_local_slate_tmp_caches()
        except Exception as _bust_chk_err:
            print(f"[slate] github bust check err: {_bust_chk_err}")
        lock_cached = _lg("slate_v5_locked")
        if lock_cached:
            lock_cached["locked"] = True
            lock_cached.setdefault("draftable_count", len(draftable_games))
            if lock_time: lock_cached.setdefault("lock_time", lock_time)
            # Scenario 1 auto-detect: if a new deploy landed mid-slate, the cached
            # picks were built with the old model. Detect SHA mismatch and regenerate
            # in the background — user gets the stale-but-functional slate immediately,
            # and the next request will serve the freshly regenerated picks.
            _current_sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "")
            _cached_sha = lock_cached.get("deploy_sha", "")
            if _current_sha and _cached_sha and _current_sha[:7] != _cached_sha[:7]:
                if not getattr(_force_regenerate_bg, "_in_flight", False):
                    _force_regenerate_bg._in_flight = True
                    print(f"[slate] deploy SHA mismatch: cached={_cached_sha} current={_current_sha[:7]} — background regeneration triggered")
                    threading.Thread(target=_force_regenerate_bg_worker, daemon=True).start()
            return lock_cached
        # Check regular cache and promote to lock cache
        cached = _cg("slate_v5")
        if cached:
            cached["locked"] = True
            cached.setdefault("draftable_count", len(draftable_games))
            if lock_time: cached.setdefault("lock_time", lock_time)
            _ls("slate_v5_locked", cached)
            _slate_backup_to_github(cached)
            # Inline slate prediction save at lock-promotion time.
            # Cold-start Lambdas handling the follow-up /api/save-predictions won't
            # have the slate cache — write slate rows to GitHub now while we have them.
            try:
                today_str = _et_date().isoformat()
                pred_path = f"data/predictions/{today_str}.csv"
                pred_existing, _ = _github_get_file(pred_path)
                if not pred_existing:
                    lineups = cached.get("lineups")
                    if not lineups:
                        print(f"[slate lock] WARNING: no lineups to save (lineups={lineups})")
                    else:
                        slate_rows = _predictions_to_csv(lineups, "slate")
                        if slate_rows:
                            csv_init = CSV_HEADER + "\n" + "\n".join(slate_rows) + "\n"
                            _github_write_file(pred_path, csv_init,
                                               f"slate predictions lock {today_str}")
                        else:
                            print(f"[slate lock] WARNING: lineups exist but _predictions_to_csv returned empty")
            except Exception as _e:
                print(f"[slate lock] inline pred save err: {_e}")
            return cached
        # Cold-start: no local cache. Try GitHub backup written at lock promotion time.
        gh_backup = _slate_restore_from_github()
        if gh_backup:
            gh_backup["locked"] = True
            gh_backup.setdefault("draftable_count", len(draftable_games))
            _ls("slate_v5_locked", gh_backup)
            return gh_backup
        # Lock file missing or busted (tombstoned by _bust_slate_cache) — fall through
        # to the full pipeline for forced regeneration. This handles the case where an
        # admin bust was requested to fix bad lineups in the lock file (e.g. predMin
        # filter change). The pipeline runs with draftable games still in progress.

    # ── Layer 1: /tmp cache (warm Railway instance) ──
    cached = _cg("slate_v5")
    if cached:
        # Discard cached result if it has empty lineups but we have draftable games.
        has_players = cached.get("lineups", {}).get("chalk") or cached.get("lineups", {}).get("upside")
        if has_players or not draftable_games:
            cached["locked"] = locked
            cached.setdefault("draftable_count", len(draftable_games))
            return cached

    # ── Layer 2: GitHub persistent cache (cold-start recovery) ──
    gh_cached = _slate_cache_from_github()
    if gh_cached:
        has_players = gh_cached.get("lineups", {}).get("chalk") or gh_cached.get("lineups", {}).get("upside")
        if has_players or not draftable_games:
            gh_cached["locked"] = locked
            gh_cached.setdefault("draftable_count", len(draftable_games))
            if lock_time:
                gh_cached.setdefault("lock_time", lock_time)
            # Warm /tmp cache for subsequent requests on this instance
            _cs("slate_v5", gh_cached)
            # Also warm per-game /tmp caches from GitHub
            try:
                gh_games = _games_cache_from_github()
                if gh_games:
                    for gid, projs in gh_games.items():
                        _cs(f"game_proj_{gid}", projs)
            except Exception:
                pass
            return gh_cached

    # ── Layer 3: First run of the day — generate fresh, then persist ──
    # Concurrent cold-start guard: if another thread is already generating, wait for it
    # and then serve from the cache it populated. Prevents N×(ESPN + LightGBM + MILP) on
    # simultaneous cold-start requests (common when the app is opened in multiple tabs).
    global _SLATE_GEN_IN_FLIGHT
    with _SLATE_GEN_LOCK:
        if _SLATE_GEN_IN_FLIGHT:
            # Another thread is already running the pipeline — release lock and wait.
            already_running = True
        else:
            _SLATE_GEN_IN_FLIGHT = True
            already_running = False

    if already_running:
        # Poll /tmp cache until the other thread finishes (max ~90s).
        for _ in range(45):
            time.sleep(2)
            _warm = _cg("slate_v5")
            if _warm:
                _warm["locked"] = locked
                _warm.setdefault("draftable_count", len(draftable_games))
                return _warm
        # Fallback: if it never appeared, fall through to run our own pipeline
        # (the other thread may have crashed).

    try:
        all_proj = []
        game_proj_map = {}  # {gameId: [projections...]} for GitHub persistence
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(_run_game, g): g for g in draftable_games}
            for fut in as_completed(futs):
                try:
                    game = futs[fut]
                    projs = fut.result()
                    all_proj.extend(projs)
                    game_proj_map[game["gameId"]] = projs
                except Exception as e:
                    print(f"slate err: {e}")
        # Optional Odds API enrichment: blend sportsbook player props into projections.
        # Books see information our model can't (rotation changes, matchup exploitation).
        try:
            _enrich_projections_with_odds(all_proj, draftable_games)
        except Exception as _odds_err:
            print(f"[odds_enrich] call-site error: {_odds_err}")
        # Matchup data: opponent defensive stats + game opponent map (used by Layer 1.5 and _build_lineups)
        _def_stats = {}
        try:
            _def_stats = _fetch_team_def_stats()
        except Exception as _def_err:
            print(f"[matchup] def stats fetch error (non-fatal): {_def_err}")
        _game_opp_map = _build_game_opp_map(draftable_games)
        # Optional Claude context pass: adjust RS projections for game narrative
        # (blowout risk, defensive value, rivalry closeness). No-op when disabled.
        # Capture news_text so Layer 1.5 and Layer 3 can reuse it.
        _slate_news_text = ""
        try:
            _slate_news_text = _fetch_nba_news_context(draftable_games, all_proj=all_proj)
        except Exception:
            pass
        # Layer 1.5: Claude matchup intelligence — DISABLED (cost reduction).
        # ESPN def stats in _compute_matchup_factor() provide equivalent signal at zero cost.
        # Re-enable via matchup.claude_enabled=true in model-config.json if needed.
        _matchup_intel = {}
        try:
            _claude_context_pass(all_proj, draftable_games)
        except Exception as _ctx_err:
            print(f"[context_pass] call-site error: {_ctx_err}")
        _apply_post_lock_rs_calibration(all_proj, slate_locked=locked)
        chalk, upside, core_pool = _build_lineups(all_proj, def_stats=_def_stats, matchup_intel=_matchup_intel)
        try:
            chalk, upside = _lineup_review_opus(chalk, upside, all_proj, draftable_games, core_pool=core_pool, news_context=_slate_news_text)
        except Exception as _rev_err:
            print(f"[lineup_review] call-site error: {_rev_err}")
        lineups = {"chalk": chalk, "upside": upside}
        # Watchlist: players near the lineup bubble sensitive to late-breaking news
        _watchlist = []
        try:
            _watchlist = _build_watchlist(chalk, upside, all_proj, draftable_games)
        except Exception as _wl_err:
            print(f"[watchlist] build error: {_wl_err}")
        _deploy_sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "")
        _boosts_ingested = bool(_load_daily_boosts())
        result = {"date": _et_date().isoformat(), "games": games,
                  "lineups": lineups, "locked": locked,
                  "all_complete": False, "draftable_count": len(draftable_games),
                  "lock_time": lock_time,
                  "watchlist": _watchlist,
                  "boosts_ingested": _boosts_ingested,
                  "pass": 1,
                  "score_bounds": _score_bounds_for_lineups(lineups),
                  "deploy_sha": _deploy_sha[:7] if _deploy_sha else ""}
        if chalk or upside:  # Don't cache empty results — allow retry on next request
            _cs("slate_v5", result)
            # GitHub writes are fire-and-forget: store to /tmp first so any concurrent
            # request is served immediately, then persist to GitHub in the background.
            # This removes 2-3s of blocking I/O from the hot path — the user gets
            # the slate response as soon as the pipeline finishes, not after writes.
            _bg_result       = result
            _bg_game_proj    = game_proj_map
            _bg_today        = _et_date().isoformat()
            def _slate_persist_bg():
                try:
                    _slate_cache_to_github(_bg_result)
                except Exception as _e:
                    print(f"[slate-cache] bg write err: {_e}")
                try:
                    _github_write_file(f"data/slate/{_bg_today}_bust.json", "{}", f"clear bust {_bg_today}")
                except Exception as _e:
                    print(f"[slate-cache] bg clear bust err: {_e}")
                if _bg_game_proj:
                    try:
                        _games_cache_to_github(_bg_game_proj)
                    except Exception as _e:
                        print(f"[slate-cache] bg games write err: {_e}")
                try:
                    _slate_backup_to_github(_bg_result)
                except Exception as _e:
                    print(f"[slate-cache] bg backup err: {_e}")
            threading.Thread(target=_slate_persist_bg, daemon=True).start()
        if locked:
            _ls("slate_v5_locked", result)
        # Sync-clear bust so other Railway instances stop dropping /tmp on every
        # locked request (bg-only clear races multi-instance).
        if chalk or upside:
            try:
                _st = _et_date().isoformat()
                _github_write_file(
                    f"data/slate/{_st}_bust.json",
                    "{}",
                    f"clear bust sync after slate gen {_st}",
                )
            except Exception as _e:
                print(f"[slate-cache] sync clear bust err: {_e}")
        return result
    finally:
        if not already_running:
            with _SLATE_GEN_LOCK:
                _SLATE_GEN_IN_FLIGHT = False


@app.get("/api/slate")
async def get_slate(mock: bool = Query(False, description="Return deterministic mock data for testing (no ESPN/model calls)")) -> dict:
    """Slate endpoint: never returns 500; on exception returns 200 with error key for graceful frontend handling.
    Pass ?mock=true to get static test data suitable for UI/audit validation without hitting live systems."""
    if mock:
        return _get_mock_slate()
    try:
        return _get_slate_impl()
    except Exception as e:
        print(f"[slate] error: {e}")
        return JSONResponse(
            content={
                "error": "slate_failed",
                "date": _et_date().isoformat(),
                "games": [],
                "lineups": {"chalk": [], "upside": []},
                "locked": False,
                "draftable_count": 0,
            },
            status_code=200,
        )


def _compute_game_picks(game):
    """Compute game-specific projections and cache under both regular and lock keys.
    Returns the result dict, or None if projections unavailable. Skips if already cached."""
    gid = game["gameId"]
    existing = _lg(f"picks_locked_{gid}") or _cg(f"picks_{gid}")
    if existing:
        return existing
    try:
        projections = _run_game(game)
        if not projections:
            return None
        lineups_dict = _build_game_lineups(projections, game)
        result = {
            "date": _et_date().isoformat(), "game": game,
            "gameScript": _game_script_label(game.get("total")),
            "lineups": lineups_dict,
            "locked": True, "injuries": _get_injuries(game),
        }
        _cs(f"picks_{gid}", result)
        _ls(f"picks_locked_{gid}", result)
        return result
    except Exception as e:
        print(f"[auto-lock] game {gid} picks err: {e}")
        traceback.print_exc()
        return None


@app.get("/api/picks")
async def get_picks(gameId: str = Query(...)):
    # Mock per-game response for audit/testing — gameId "mock_game_1" or "mock_game_2"
    if gameId.startswith("mock_game_"):
        mock_slate = _get_mock_slate()
        game_meta = next((g for g in mock_slate["games"] if g["gameId"] == gameId), None)
        if not game_meta:
            return JSONResponse({"error": "Mock game not found"}, status_code=404)
        # Return mock per-game lineup (subset of chalk players from different teams)
        the_lineup = [
            {"id":"mock_pg1","name":"Mock Game PG","pos":"PG","team":"CHI","rating":7.0,"predMin":36.0,
             "pts":22.0,"reb":4.0,"ast":7.0,"stl":1.2,"blk":0.3,"est_mult":0.0,"slot":"2.0x",
             "chalk_ev":14.0,"moonshot_ev":0.0,"injury_status":"","_decline":0.0},
            {"id":"mock_pg2","name":"Mock Game SG","pos":"SG","team":"MIN","rating":6.0,"predMin":34.0,
             "pts":18.0,"reb":3.0,"ast":4.0,"stl":1.0,"blk":0.2,"est_mult":0.0,"slot":"1.8x",
             "chalk_ev":10.8,"moonshot_ev":0.0,"injury_status":"","_decline":0.0},
            {"id":"mock_pg3","name":"Mock Game SF","pos":"SF","team":"CHI","rating":5.5,"predMin":32.0,
             "pts":16.0,"reb":6.0,"ast":3.0,"stl":1.0,"blk":0.5,"est_mult":0.0,"slot":"1.6x",
             "chalk_ev":8.8,"moonshot_ev":0.0,"injury_status":"","_decline":0.0},
            {"id":"mock_pg4","name":"Mock Game PF","pos":"PF","team":"MIN","rating":5.0,"predMin":30.0,
             "pts":14.0,"reb":8.0,"ast":2.0,"stl":0.5,"blk":1.0,"est_mult":0.0,"slot":"1.4x",
             "chalk_ev":7.0,"moonshot_ev":0.0,"injury_status":"","_decline":0.0},
            {"id":"mock_pg5","name":"Mock Game C","pos":"C","team":"CHI","rating":4.5,"predMin":28.0,
             "pts":12.0,"reb":10.0,"ast":1.5,"stl":0.3,"blk":1.8,"est_mult":0.0,"slot":"1.2x",
             "chalk_ev":5.4,"moonshot_ev":0.0,"injury_status":"","_decline":0.0},
        ]
        # Per-game total = sum of ratings (25-35 range): 7+6+5.5+5+4.5 = 28.0 ✓
        lineups = {"the_lineup": the_lineup}
        return {"date": _et_date().isoformat(), "game": game_meta, "mock": True,
                "gameScript": "balanced", "lineups": lineups, "locked": False,
                "injuries": [], "score_bounds": _score_bounds_for_lineups(lineups)}

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
            # Inline per-game save at lock-promotion time — ensures per-game
            # predictions are persisted to GitHub as soon as the game locks,
            # not just when the slate-level save fires later.
            try:
                if reg_cached.get("lineups"):
                    label = game.get("label", f"game_{gameId}")
                    game_rows = _predictions_to_csv(reg_cached["lineups"], label)
                    if game_rows:
                        today = _et_date().isoformat()
                        pred_path = f"data/predictions/{today}.csv"
                        existing, _ = _github_get_file(pred_path)
                        if existing:
                            existing_scopes = {line.split(",")[0] for line in existing.strip().split("\n")[1:] if line.split(",")[0]}
                            if label not in existing_scopes:
                                csv_content = existing.rstrip("\n") + "\n" + "\n".join(game_rows) + "\n"
                                _github_write_file(pred_path, csv_content, f"per-game lock {label} {today}")
                        else:
                            csv_content = CSV_HEADER + "\n" + "\n".join(game_rows) + "\n"
                            _github_write_file(pred_path, csv_content, f"per-game lock {label} {today}")
            except Exception as _e:
                print(f"[picks lock] inline save err {gameId}: {_e}")
            return reg_cached
        # No cache on this instance after lock — auto-compute so user sees picks
        auto = _compute_game_picks(game)
        if auto:
            return auto
        return {"date": _et_date().isoformat(), "game": game,
                "gameScript": None,
                "lineups": {"the_lineup": []},
                "locked": True, "injuries": []}

    # Try /tmp cache first (populated by /api/slate or previous /api/picks)
    cache_key = f"game_proj_{gameId}"
    projections = _cg(cache_key)
    if not projections:
        # Try GitHub persistent cache (populated by first slate run of the day)
        gh_games = _games_cache_from_github()
        if gh_games and gameId in gh_games:
            projections = gh_games[gameId]
            _cs(cache_key, projections)  # warm /tmp for next call
    if not projections:
        # True cold start with no cache anywhere — run engine (rare after first daily run)
        projections = _run_game(game)
    if not projections:
        return JSONResponse({"error": "No projections available."}, status_code=503)
    lineups_dict = _build_game_lineups(projections, game)
    script = _game_script_label(game.get("total"))
    injuries = _get_injuries(game)

    result = {"date": _et_date().isoformat(), "game": game,
              "gameScript": script,
              "lineups": lineups_dict,
              "locked": locked,
              "injuries": injuries,
              "score_bounds": _score_bounds_for_lineups(lineups_dict)}
    # Cache picks so they survive as lock snapshot if slate locks later
    _cs(f"picks_{gameId}", result)
    return result

@app.post("/api/save-predictions")
async def save_predictions():
    """Save current predictions to GitHub as CSV."""
    today = _et_date().isoformat()
    path = f"data/predictions/{today}.csv"

    # Guard: only write predictions after the slate has locked.
    # Pre-lock projections are not finalized — saving them would pollute the log
    # with data that changes as injury news, lineups, and odds shift.
    # Uses any() instead of min() — on split-window days the earliest game's 6h
    # ceiling can expire while late games are still live. any() stays locked as
    # long as ANY game is within its lock window.
    _games_now = fetch_games()
    _start_times = [g["startTime"] for g in _games_now if g.get("startTime")]
    if _start_times and not any(_is_locked(st) for st in _start_times):
        return JSONResponse({"error": "Slate not locked yet — predictions not finalized"}, status_code=409)

    # Gather slate predictions — try all cache layers before giving up
    rows = []
    cached_slate = _cg("slate_v5") or _lg("slate_v5_locked")
    if not cached_slate:
        cached_slate = _slate_restore_from_github()
    if cached_slate and cached_slate.get("lineups"):
        rows.extend(_predictions_to_csv(cached_slate["lineups"], "slate"))

    # Gather per-game predictions. Prefer explicit Game Analysis picks (user-triggered),
    # then lock cache, then raw slate projections. For locked games with no cache at all,
    # auto-compute now so the log always has a full set of predictions.
    games = fetch_games()
    locked_games_to_compute = []
    for g in games:
        gid = g["gameId"]
        label = g.get("label", f"game_{gid}")
        cached_picks = _cg(f"picks_{gid}") or _lg(f"picks_locked_{gid}")
        if cached_picks and cached_picks.get("lineups"):
            rows.extend(_predictions_to_csv(cached_picks["lineups"], label))
        elif _is_locked(g.get("startTime", "")):
            locked_games_to_compute.append(g)

    # Auto-compute picks for locked games the user never manually analyzed.
    # Run in parallel so this doesn't add significant latency per game.
    if locked_games_to_compute:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_compute_game_picks, g): g for g in locked_games_to_compute}
            for fut in as_completed(futures):
                g = futures[fut]
                label = g.get("label", f"game_{g['gameId']}")
                try:
                    result = fut.result()
                    if result and result.get("lineups"):
                        rows.extend(_predictions_to_csv(result["lineups"], label))
                except Exception as e:
                    print(f"save-predictions auto-lock {g['gameId']}: {e}")
                    traceback.print_exc()

    if not rows:
        return JSONResponse({"error": "No predictions cached yet"}, status_code=404)

    # Merge with existing CSV — on split-window days, later-locking games need to be
    # appended without overwriting earlier predictions (e.g. 5 PM game saved first,
    # 7:30 PM game locks later and gets added by /api/refresh cron).
    existing, _ = _github_get_file(path)
    if existing:
        existing_scopes = set()
        for line in existing.strip().split("\n")[1:]:  # skip header
            fields = line.split(",")
            if fields:
                existing_scopes.add(fields[0])
        new_rows = [r for r in rows if r.split(",")[0] not in existing_scopes]
        if not new_rows:
            return {"status": "unchanged", "path": path, "rows": 0}
        csv_content = existing.rstrip("\n") + "\n" + "\n".join(new_rows) + "\n"
    else:
        csv_content = CSV_HEADER + "\n" + "\n".join(rows) + "\n"

    result = _github_write_file(path, csv_content, f"predictions for {today}")
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=500)

    # Also write the slate backup now so cold-start instances can recover after lock,
    # even if this Railway instance dies before the lock window promotes the reg cache.
    cached_slate_for_backup = _cg("slate_v5")
    if cached_slate_for_backup:
        _slate_backup_to_github(cached_slate_for_backup)

    return {"status": "saved", "path": path, "rows": len(rows)}


# grep: FORCE REGENERATE — mid-slate prediction update for dev deploys + late drafts
def _force_write_predictions(rows, replace_all=False, replace_scopes=None):
    """Write prediction CSV rows to GitHub, replacing specific scopes instead of deduplicating.

    Unlike save_predictions() (which skips existing scopes), this helper:
    - replace_all=True: discards ALL existing rows, writes fresh (Scenario 1: dev deploy)
    - replace_scopes=[...]: removes rows matching those scopes, appends new (Scenario 2: late draft)

    Returns dict with status + path + rows count.
    """
    today = _et_date().isoformat()
    path = f"data/predictions/{today}.csv"
    if not rows:
        return {"error": "no_rows", "path": path}

    if replace_all:
        csv_content = CSV_HEADER + "\n" + "\n".join(rows) + "\n"
    else:
        existing, _ = _github_get_file(path)
        if existing and replace_scopes:
            replace_set = set(replace_scopes)
            kept = [existing.strip().split("\n")[0]]  # header
            for line in existing.strip().split("\n")[1:]:
                fields = line.split(",")
                if fields and fields[0] not in replace_set:
                    kept.append(line)
            csv_content = "\n".join(kept) + "\n" + "\n".join(rows) + "\n"
        elif existing:
            # No replace_scopes — just append (same as save_predictions merge)
            csv_content = existing.rstrip("\n") + "\n" + "\n".join(rows) + "\n"
        else:
            csv_content = CSV_HEADER + "\n" + "\n".join(rows) + "\n"

    result = _github_write_file(path, csv_content, f"force-regenerate predictions {today}")
    if result.get("error"):
        return {"error": result["error"], "path": path}
    return {"status": "saved", "path": path, "rows": len(rows)}


# grep: FORCE REGENERATE SYNC — _force_regenerate_sync, scope=full|remaining, deploy SHA mismatch
def _force_regenerate_sync(scope: str):
    """Run the full prediction pipeline for a given scope and update predictions CSV.

    scope="full": ALL today's games (dev shipped mid-slate — as if 5 min before lock).
    scope="remaining": only games NOT yet locked (user woke up late).

    Returns summary dict.
    """
    today_str = _et_date().isoformat()
    games = fetch_games()
    if not games:
        return {"status": "no_games", "scope": scope, "date": today_str}

    # Determine game pool based on scope
    if scope == "remaining":
        game_pool = [g for g in games if not _is_locked(g.get("startTime", ""))]
        if not game_pool:
            return {"status": "no_remaining_games", "scope": scope, "date": today_str,
                    "message": "All games have already started"}
    else:
        # "full" scope — use all today's games regardless of lock status
        game_pool = games

    # Step 1: Run projection pipeline on the game pool (same as _get_slate_impl Layer 3)
    # NOTE: Cache bust intentionally deferred until we have valid results — if the pipeline
    # fails we preserve the existing cache so cold-start recovery still works.
    all_proj = []
    game_proj_map = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_run_game, g): g for g in game_pool}
        for fut in as_completed(futs):
            try:
                game = futs[fut]
                projs = fut.result()
                all_proj.extend(projs)
                game_proj_map[game["gameId"]] = projs
            except Exception as e:
                print(f"[force-regen] game err: {e}")

    if not all_proj:
        return {"status": "no_projections", "scope": scope, "date": today_str}

    # Step 2: Build slate-wide lineups (Starting 5 + Moonshot) with matchup data
    _fr_def_stats = {}
    try:
        _fr_def_stats = _fetch_team_def_stats()
    except Exception:
        pass
    _fr_starts = [g["startTime"] for g in games if g.get("startTime")]
    _fr_any_locked = bool(_fr_starts) and any(_is_locked(st) for st in _fr_starts)
    _apply_post_lock_rs_calibration(all_proj, slate_locked=_fr_any_locked)
    chalk, upside, core_pool = _build_lineups(all_proj, def_stats=_fr_def_stats)
    try:
        chalk, upside = _lineup_review_opus(chalk, upside, all_proj, game_pool, core_pool=core_pool)
    except Exception as _rev_err:
        print(f"[lineup_review] call-site error: {_rev_err}")
    lineups = {"chalk": chalk, "upside": upside}
    # Watchlist for force-regen (Pass 2)
    _fr_watchlist = []
    try:
        _fr_watchlist = _build_watchlist(chalk, upside, all_proj, game_pool)
    except Exception:
        pass

    # Step 3: Build per-game lineups
    per_game_results = {}
    for g in game_pool:
        gid = g["gameId"]
        game_projs = game_proj_map.get(gid, [])
        if game_projs:
            try:
                game_lineups = _build_game_lineups(game_projs, g)
                per_game_results[gid] = {
                    "game": g,
                    "lineups": game_lineups,
                }
            except Exception as e:
                print(f"[force-regen] game lineups err {gid}: {e}")

    # Step 4: Pipeline succeeded — now bust old cache and write fresh data to all layers
    _bust_slate_cache()

    # Step 5: Build the slate cache object and persist to all layers
    deploy_sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "")
    result = {
        "date": today_str, "games": games,
        "lineups": lineups, "locked": True,
        "all_complete": False, "draftable_count": len(game_pool),
        "score_bounds": _score_bounds_for_lineups(lineups),
        "deploy_sha": deploy_sha[:7] if deploy_sha else "",
        "watchlist": _fr_watchlist,
        "boosts_ingested": bool(_load_daily_boosts()),
        "pass": 2,
    }
    if chalk or upside:
        _cs("slate_v5", result)
        _ls("slate_v5_locked", result)
        _slate_cache_to_github(result)
        _slate_backup_to_github_force(result)
        try:
            _github_write_file(f"data/slate/{today_str}_bust.json", "{}",
                               f"clear bust after force-regen {today_str}")
        except Exception:
            pass
        if game_proj_map:
            _games_cache_to_github(game_proj_map)

    # Step 6: Cache per-game picks too
    for gid, gdata in per_game_results.items():
        _cs(f"picks_{gid}", gdata)
        _ls(f"picks_locked_{gid}", gdata)

    # Step 7: Write predictions CSV
    rows = _predictions_to_csv(lineups, "slate")
    for gid, gdata in per_game_results.items():
        game = gdata["game"]
        label = game.get("label", f"game_{gid}")
        rows.extend(_predictions_to_csv(gdata["lineups"], label))

    if scope == "full":
        csv_result = _force_write_predictions(rows, replace_all=True)
    else:
        replace_scopes = ["slate"] + [
            g.get("label", f"game_{g['gameId']}") for g in game_pool
        ]
        csv_result = _force_write_predictions(rows, replace_scopes=replace_scopes)

    return {
        "status": "regenerated",
        "scope": scope,
        "date": today_str,
        "games_regenerated": len(game_pool),
        "games_total": len(games),
        "scopes_updated": ["slate"] + list(per_game_results.keys()),
        "csv": csv_result,
        "deploy_sha": deploy_sha[:7] if deploy_sha else "",
        "lineups": lineups,
    }


def _slate_backup_to_github_force(slate_data: dict):
    """Write slate lock backup to GitHub, OVERWRITING any existing file.
    Used by force-regenerate (unlike _slate_backup_to_github which skips if existing)."""
    try:
        today = _et_date().isoformat()
        path = f"data/locks/{today}_slate.json"
        content = json.dumps(slate_data, default=str)
        _github_write_file(path, content, f"force-regen lock backup {today}")
    except Exception as e:
        print(f"[force-regen] lock backup err: {e}")


# ── TWO-PASS PIPELINE MONITORING ───────────────────────────────────────────
# grep: PASS 2 MONITOR, slate-check, material change detection
# Pass 1 runs via /api/slate (morning). Pass 2 runs via /api/force-regenerate
# (pre-game, conditional). This endpoint checks if material changes have occurred
# since Pass 1 that warrant a Pass 2 re-run.
@app.get("/api/slate-check")
async def slate_check(request: Request):
    """Check if material inputs have changed since Pass 1 lineup was generated.

    Checks for:
    1. Injury status changes for players in the current lineup
    2. Injury status changes for players on the watchlist
    3. Vegas line movements (spread/total) beyond configured thresholds
    4. New starter ruled OUT on any game (cascade opportunity)

    Returns: {changed: bool, triggers: [...], recommendation: "hold"|"rerun"}
    """
    today = _et_date().isoformat()
    games = fetch_games()
    if not games:
        return {"changed": False, "triggers": [], "recommendation": "hold",
                "reason": "no_games"}

    # Don't check after lock — picks are frozen
    start_times = [g["startTime"] for g in games if g.get("startTime")]
    if start_times and any(_is_locked(st) for st in start_times):
        return {"changed": False, "triggers": [], "recommendation": "hold",
                "reason": "locked"}

    # Load Pass 1 cached slate
    cached_slate = _cg("slate_v5") or _slate_cache_from_github()
    if not cached_slate or not cached_slate.get("lineups"):
        return {"changed": False, "triggers": [], "recommendation": "hold",
                "reason": "no_pass1"}

    triggers = []

    # ── Trigger 1: Injury changes for lineup players ──────────────────────
    try:
        rw_statuses = get_all_statuses()
        if rw_statuses:
            for lineup_type in ["chalk", "upside"]:
                for player in cached_slate["lineups"].get(lineup_type, []):
                    pname = player.get("name", "")
                    if pname and not is_safe_to_draft(pname):
                        triggers.append({
                            "type": "injury_lineup",
                            "player": pname,
                            "lineup": lineup_type,
                            "severity": "high",
                        })

            # ── Trigger 2: Watchlist player status changes ────────────────
            watchlist = cached_slate.get("watchlist", [])
            for wp in watchlist:
                wname = wp.get("player", "")
                watch_event = wp.get("trigger_event", "")
                if wname and watch_event == "injury_cascade":
                    # The player benefits if someone is OUT — check if that happened
                    dep_player = wp.get("depends_on", "")
                    if dep_player and not is_safe_to_draft(dep_player):
                        triggers.append({
                            "type": "watchlist_activated",
                            "player": wname,
                            "depends_on": dep_player,
                            "severity": "medium",
                        })

            # ── Trigger 3: New OUT starter (not in lineup) ────────────────
            for g in games:
                for team_key in ["homeTeam", "awayTeam"]:
                    team_info = g.get(team_key, {})
                    if isinstance(team_info, dict):
                        team_abbr = team_info.get("abbreviation", "")
                    elif isinstance(team_info, str):
                        team_abbr = team_info
                    else:
                        continue
                    # Check if any known players on this team are newly OUT
                    # (RotoWire integration catches this)
    except Exception as e:
        print(f"[slate-check] injury check err: {e}")

    # ── Trigger 4: Vegas line movement ────────────────────────────────────
    try:
        pass1_games = {g.get("gameId"): g for g in cached_slate.get("games", [])}
        vegas_threshold = float(_cfg("pass2.vegas_total_threshold", 3.0))
        for g in games:
            gid = g.get("gameId")
            if gid and gid in pass1_games:
                old_total = pass1_games[gid].get("total") or 0
                new_total = g.get("total") or 0
                if old_total and new_total and abs(new_total - old_total) >= vegas_threshold:
                    triggers.append({
                        "type": "vegas_movement",
                        "game": gid,
                        "old_total": old_total,
                        "new_total": new_total,
                        "delta": round(new_total - old_total, 1),
                        "severity": "medium",
                    })
    except Exception as e:
        print(f"[slate-check] vegas check err: {e}")

    has_high = any(t.get("severity") == "high" for t in triggers)
    recommendation = "rerun" if has_high or len(triggers) >= 2 else "hold"

    return {
        "changed": len(triggers) > 0,
        "triggers": triggers,
        "trigger_count": len(triggers),
        "recommendation": recommendation,
        "date": today,
        "boosts_ingested": bool(_load_daily_boosts()),
    }


@app.get("/api/force-regenerate")
async def force_regenerate(request: Request, scope: str = Query("full")):
    """Force-regenerate predictions mid-slate.

    scope=full: Re-run ALL games (dev shipped model update mid-slate). CRON_SECRET-gated.
    scope=remaining: Only games not yet started (user woke up late). User-facing, no auth.
    """
    if scope == "full" and not _require_cron_secret(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if scope not in ("full", "remaining"):
        return JSONResponse({"error": "scope must be 'full' or 'remaining'"}, status_code=400)

    try:
        result = await asyncio.to_thread(_force_regenerate_sync, scope)
        return result
    except Exception as e:
        print(f"[force-regenerate] error: {e}")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)



async def reset_uploads(body: dict):
    """Delete actuals + audit files for a given date from GitHub.
    Used to undo an incorrect/early upload so Ben can re-prompt for the right date."""
    date_str = body.get("date")
    if not date_str:
        return JSONResponse({"error": "date required"}, status_code=400)
    bad = _validate_date(date_str)
    if bad: return bad
    deleted = []
    for path in [f"data/actuals/{date_str}.csv", f"data/audit/{date_str}.json"]:
        _, sha = _github_get_file(path)
        if sha:
            ok = _github_delete_file(path, sha, f"reset uploads for {date_str}")
            if ok:
                deleted.append(path)
    return {"deleted": deleted, "date": date_str}


@app.post("/api/parse-screenshot")
async def parse_screenshot(
    request: Request,
    file: UploadFile = File(...),
    screenshot_type: str = Form(default="actuals"),
):
    """Parse a Real Sports app screenshot using Claude Vision API.

    screenshot_type:
      "actuals"      — default; extracts My Draft / Highest Value player RS data
      "most_drafted" — extracts ownership leaderboard (player + draft_pct)
      "boosts"       — extracts pre-game player list with boosts (fixed daily constants)
    """
    rl = _check_rate_limit(request, "parse-screenshot")
    if rl is not None:
        return rl
    if not ANTHROPIC_API_KEY:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not configured"}, status_code=500)

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        return JSONResponse({"error": "Image too large (max 10MB)"}, status_code=413)
    b64_image = base64.b64encode(image_bytes).decode("ascii")

    # Determine media type
    _ALLOWED_IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")
    ct = file.content_type or ""
    if ct not in _ALLOWED_IMAGE_TYPES:
        return JSONResponse({"error": f"Unsupported image type: {ct or 'unknown'}. Allowed: png, jpeg, gif, webp"}, status_code=415)

    if screenshot_type == "boosts":
        prompt = """Extract player boost data from this Real Sports pre-game screenshot.

This shows available players for today's draft with their boosts displayed.
Boosts appear as "+X.Xx" (e.g. "+3.0x", "+0.5x", "+2.1x") next to each player.

For EACH player listed, extract:
- player_name: full player name exactly as shown
- boost: the card boost number (e.g. "+3.0x" → 3.0, "+0.5x" → 0.5). If no boost shown, set to null
- team: team abbreviation if visible (e.g. LAL, GSW, BOS), otherwise null
- rax_cost: the Rax cost if shown as a number with triangle symbol (e.g. "▽5.4" → 5.4), otherwise null

Return ONLY a JSON array of objects. No markdown, no explanation."""
    elif screenshot_type == "most_drafted":
        prompt = """Extract player ownership data from this Real Sports 'Most popular' or 'Most drafted' screenshot.

For EACH player listed, extract:
- rank: their position in the list as an integer (1, 2, 3...)
- player_name: full player name exactly as shown
- team: team abbreviation if visible (e.g. LAL, GSW, BOS), otherwise null
- draft_count: number of drafts shown (e.g. "8.3k" → 8300, "492" → 492, "1.1k" → 1100)
- actual_rs: the Real Score number shown with triangle symbol (e.g. "▽5.4" → 5.4, "▽3.7" → 3.7)
- actual_card_boost: the card boost shown as "+X.Xx" (e.g. "+0.3x" → 0.3, "+2.2x" → 2.2). If no + symbol shown, set to null
- avg_finish: average finish shown (e.g. "1st" → 1, "3rd" → 3), or null if not visible

Return ONLY a JSON array of objects. No markdown, no explanation."""
    else:
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
            f"{ANTHROPIC_API_BASE}/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": HAIKU_MODEL,
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


def _compute_audit(date_str):
    """Compare predictions vs actuals for a date. Returns audit dict or None if no data."""
    pred_csv, _ = _github_get_file(f"data/predictions/{date_str}.csv")
    act_csv,  _ = _github_get_file(f"data/actuals/{date_str}.csv")
    if not pred_csv or not act_csv:
        return None

    preds   = _parse_csv(pred_csv, PRED_FIELDS)
    actuals = _parse_csv(act_csv, ACT_FIELDS)

    act_map = {r["player_name"].lower(): r for r in actuals}

    errors, dir_hits, misses = [], [], []
    for row in preds:
        pname    = row.get("player_name", "").lower()
        pred_rs  = _safe_float(row.get("predicted_rs"))
        if pname not in act_map or pred_rs <= 0:
            continue
        a        = act_map[pname]
        actual_rs = _safe_float(a.get("actual_rs"))
        if actual_rs <= 0:
            continue
        err = actual_rs - pred_rs
        errors.append(abs(err))
        dir_hits.append(1 if (err >= 0) == (pred_rs >= 3.0) else 0)  # directional: above/below avg
        misses.append({
            "player":       row["player_name"],
            "team":         row.get("team", ""),
            "predicted_rs": round(pred_rs, 2),
            "actual_rs":    round(actual_rs, 2),
            "error":        round(err, 2),
            "drafts":       a.get("drafts", ""),
            "actual_card_boost": a.get("actual_card_boost", ""),
        })

    if not errors:
        return None

    misses.sort(key=lambda x: abs(x["error"]), reverse=True)
    mae = round(sum(errors) / len(errors), 3)
    dir_acc = round(sum(dir_hits) / len(dir_hits), 3) if dir_hits else None

    # Over- vs under-projection breakdown
    over  = [e for e in misses if e["error"] < 0]
    under = [e for e in misses if e["error"] > 0]

    # Simulated draft score — measures progress toward 60+ goal.
    # Optimal hindsight: sort actuals by RS, assign slots 2.0x→1.2x to top 5,
    # compute RS × (slot + actual_card_boost) for each. Shows what was achievable.
    slot_mults = [2.0, 1.8, 1.6, 1.4, 1.2]
    top5_actuals = sorted(
        [a for a in actuals if _safe_float(a.get("actual_rs")) > 0],
        key=lambda a: _safe_float(a.get("actual_rs")),
        reverse=True
    )[:5]
    simulated_draft_score = None
    if len(top5_actuals) >= 3:
        simulated_draft_score = round(sum(
            _safe_float(p.get("actual_rs")) * (slot_mults[i] + _safe_float(p.get("actual_card_boost", 0)))
            for i, p in enumerate(top5_actuals)
        ), 1)

    return {
        "date":               date_str,
        "players_compared":   len(errors),
        "mae":                mae,
        "directional_accuracy": dir_acc,
        "over_projected":     len(over),
        "under_projected":    len(under),
        "biggest_misses":     misses[:8],
        "simulated_draft_score": simulated_draft_score,
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/save-actuals")
async def save_actuals(payload: dict = Body(...)):
    """Save confirmed actuals to GitHub as CSV.

    Safety: Checks if user has skipped uploads for this date.
    If skipped, returns early without processing screenshots.
    """
    date_str = payload.get("date", _et_date().isoformat())
    bad = _validate_date(date_str)
    if bad: return bad
    players = payload.get("players", [])
    if not players:
        return JSONResponse({"error": "No player data"}, status_code=400)

    # Check if this date was marked as skipped by user
    try:
        skipped_content, _ = _github_get_file("data/skipped-uploads.json")
        if skipped_content:
            skipped_data = json.loads(skipped_content)
            if date_str in skipped_data.get("skipped_dates", []):
                print(f"[save-actuals] Skipping upload for {date_str} (user marked as skipped)")
                return {"status": "skipped", "date": date_str, "reason": "User skipped uploads for this date"}
    except Exception:
        pass  # If check fails, continue with normal processing

    path = f"data/actuals/{date_str}.csv"
    header = "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source"

    # Check if file already exists (to append / overwrite with dedup)
    existing, _ = _github_get_file(path)
    rows = []
    if existing:
        # Keep existing rows, but drop any row whose player_name matches a new upload
        # (prevents duplicates if user uploads the same screenshot twice)
        new_names = {str(p.get("player_name", "")).strip().lower() for p in players}
        lines = existing.strip().split("\n")
        for line in (lines[1:] if len(lines) > 1 else []):
            if not line.strip():
                continue
            # Use csv.reader so quoted names containing commas parse correctly
            try:
                first_field = next(csv.reader(io.StringIO(line)))[0].strip().lower()
            except Exception:
                first_field = line.split(",")[0].strip().strip('"').lower()
            if first_field not in new_names:
                rows.append(line)

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

    # Auto-generate audit JSON — only when real_scores data is present.
    # Audit compares predicted RS vs actual RS; meaningless without RS actuals.
    upload_source = players[0].get("source", "") if players else ""
    rs_in_csv = "real_scores" in (existing or "")
    audit = None
    if upload_source == "real_scores" or rs_in_csv:
        audit = _compute_audit(date_str)
        if audit:
            audit_path = f"data/audit/{date_str}.json"
            _github_write_file(audit_path, json.dumps(audit, indent=2), f"audit for {date_str}")

    return {"status": "saved", "path": path, "rows": len(rows), "audit": audit}


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
    """Return sorted list of dates that have stored prediction or actual data. Never returns 500."""
    try:
        cached = _cg("log_dates_v1")
        if cached is not None and isinstance(cached, dict) and "data" in cached:
            if time.time() - cached.get("ts", 0) < 180:
                return cached["data"]
        dates = set()
        with ThreadPoolExecutor(max_workers=2) as pool:
            dir_results = list(pool.map(_github_list_dir, ["data/predictions", "data/actuals"]))
        for items in dir_results:
            for item in items:
                name = item.get("name", "")
                if name.endswith(".csv"):
                    dates.add(name[:-4])
        result = sorted(dates, reverse=True)
        _cs("log_dates_v1", {"data": result, "ts": time.time()})
        return result
    except Exception as e:
        print(f"[log/dates] error: {e}")
        return []


@app.get("/api/log/get")
async def log_get(date: str = Query(None)):
    """Get stored predictions and actuals for a date."""
    date_str = date or _et_date().isoformat()
    bad = _validate_date(date_str)
    if bad: return bad

    with ThreadPoolExecutor(max_workers=2) as pool:
        (pred_csv, _), (act_csv, _) = list(pool.map(
            _github_get_file,
            [f"data/predictions/{date_str}.csv", f"data/actuals/{date_str}.csv"]
        ))

    predictions = _parse_csv(pred_csv, PRED_FIELDS) if pred_csv else []
    actuals = _parse_csv(act_csv, ACT_FIELDS) if act_csv else []

    # Group predictions by scope → lineup_type → players
    scopes = {}
    for row in predictions:
        scope = row.get("scope", "")
        lt = row.get("lineup_type", "chalk")
        scopes.setdefault(scope, {"chalk": [], "upside": [], "the_lineup": []})[lt].append(_normalize_player({
            "slot": row.get("slot", ""),
            "name": row.get("player_name", ""),
            "team": row.get("team", ""),
            "pos": row.get("pos", ""),
            "rating": row.get("predicted_rs", ""),
            "est_mult": row.get("est_card_boost", ""),
            "predMin": row.get("pred_min", ""),
            "pts": row.get("pts", ""),
            "reb": row.get("reb", ""),
            "ast": row.get("ast", ""),
            "stl": row.get("stl", ""),
            "blk": row.get("blk", ""),
        }))

    return {
        "date": date_str,
        "has_predictions": bool(predictions),
        "has_actuals": bool(actuals),
        "scopes": scopes,
        "actuals": actuals,
    }


@app.get("/api/log/actuals-stats")
async def log_actuals_stats(date: str = Query(None)):
    """Fetch actual box score stats (PTS, REB, AST, STL, BLK, MIN) from ESPN
    for all players on a given date's completed games. Returns a map of
    player_name -> {pts, reb, ast, stl, blk, min}. Cached for 24h since
    historical box scores don't change."""
    date_str = date or _et_date().isoformat()
    bad = _validate_date(date_str)
    if bad: return bad
    cache_key = f"actuals_stats_{date_str}"
    cached = _cg(cache_key)
    if cached is not None:
        return cached

    from datetime import date as date_cls
    try:
        d = date_cls.fromisoformat(date_str)
    except (ValueError, TypeError):
        return {"error": "Invalid date format", "players": {}}

    games = fetch_games(d)
    if not games:
        result = {"date": date_str, "players": {}}
        _cs(cache_key, result)
        return result

    player_stats = {}
    want_labels = {"MIN", "PTS", "REB", "AST", "STL", "BLK"}

    def _fetch_game_box(game):
        game_id = game.get("gameId")
        if not game_id:
            return {}
        data = _espn_get(f"{ESPN}/summary?event={game_id}")
        if not data:
            return {}
        result = {}
        for team_block in data.get("boxscore", {}).get("players", []):
            stats_sections = team_block.get("statistics", [])
            if not stats_sections:
                continue
            labels = stats_sections[0].get("labels", [])
            idx_map = {l: i for i, l in enumerate(labels) if l in want_labels}
            for ath in stats_sections[0].get("athletes", []):
                name = ath.get("athlete", {}).get("displayName", "")
                if not name:
                    continue
                vals = ath.get("stats", [])
                pdata = {}
                for lbl, key in [("PTS","pts"),("REB","reb"),("AST","ast"),
                                  ("STL","stl"),("BLK","blk"),("MIN","min")]:
                    if lbl in idx_map and idx_map[lbl] < len(vals):
                        try:
                            raw = vals[idx_map[lbl]]
                            pdata[key] = float(raw.split(":")[0]) if ":" in str(raw) else float(raw)
                        except (ValueError, TypeError):
                            pdata[key] = 0.0
                if pdata:
                    result[name] = pdata
        return result

    with ThreadPoolExecutor(max_workers=8) as pool:
        for game_result in pool.map(_fetch_game_box, games):
            player_stats.update(game_result)

    result = {"date": date_str, "players": player_stats}
    _cs(cache_key, result)
    return result


@app.get("/api/audit/get")
async def audit_get(date: str = Query(None)):
    """Return pre-computed audit JSON for a date (or compute live if missing)."""
    date_str = date or _et_date().isoformat()
    bad = _validate_date(date_str)
    if bad: return bad
    # Try cached audit first
    cached_json, _ = _github_get_file(f"data/audit/{date_str}.json")
    if cached_json:
        try:
            return json.loads(cached_json)
        except Exception:
            pass
    # Fall back to live computation
    audit = _compute_audit(date_str)
    return audit or {"error": "No paired prediction+actuals data for this date"}


@app.post("/api/hindsight")
async def hindsight(payload: dict = Body(...)):
    """Given actual player RS scores, return the optimal hindsight lineup."""
    players = payload.get("players", [])
    if not players:
        return JSONResponse({"error": "No players provided"}, status_code=400)

    avg_slot = _cfg("lineup.avg_slot_multiplier", 1.6)
    projections = []
    for p in players:
        rs = _safe_float(p.get("actual_rs"), 0)
        boost = _safe_float(p.get("actual_card_boost"), 0.3)
        if rs <= 0:
            continue
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
    return {"lineup": [_normalize_player(p) for p in lineup]}


@app.get("/api/refresh")
async def refresh(request: Request):
    # No auth required — cache clearing is non-destructive and user-facing.
    # Auto-save predictions BEFORE clearing cache, if the slate is currently locked.
    # Cron safety net: ensures predictions persist even if no user visits at lock time.
    # Must run first — save_predictions() reads from _cg("slate_v5") which gets wiped below.
    auto_saved = False
    try:
        games = fetch_games()
        draftable = [g for g in games if g.get("startTime")]
        start_times = [g["startTime"] for g in draftable]
        if start_times and any(_is_locked(st) for st in start_times):
            await save_predictions()
            auto_saved = True
            # End-of-slate: wipe daily Ben chat history so next cycle starts fresh.
            try:
                with _BEN_CHAT_HISTORY_LOCK:
                    _BEN_CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
                    _BEN_CHAT_HISTORY_PATH.write_text("[]")
                    _BEN_CHAT_HISTORY_DATE_PATH.write_text(
                        json.dumps({"et_date": _et_date().isoformat()}, indent=2)
                    )
            except Exception as e:
                print(f"[ben-chat] reset on refresh failed: {e}")
    except Exception as e:
        print(f"[refresh] auto-save skipped: {e}")

    cleared = 0
    try:
        for f in CACHE_DIR.glob("*.json"):
            f.unlink(); cleared += 1
    except Exception as e:
        return {"status": "error", "message": str(e)}
    # Clear lock cache (LOCK_DIR) — stale lock files must not survive a manual refresh
    try:
        for f in LOCK_DIR.glob("*.json"):
            f.unlink(missing_ok=True); cleared += 1
    except Exception: pass
    # Also clear config cache on refresh
    try:
        cfg_cache = CONFIG_CACHE_DIR / "model_config.json"
        if cfg_cache.exists():
            cfg_cache.unlink()
    except Exception: pass
    # Clear RotoWire cache so next slate load gets fresh lineup data
    try:
        _rw_clear()
    except Exception: pass
    # Bust GitHub-persisted slate cache so next request regenerates fresh
    try:
        _bust_slate_cache()
    except Exception: pass
    # Clear daily boost cache so next load re-reads from GitHub
    global _DAILY_BOOST_CACHE, _DAILY_BOOST_TS
    _DAILY_BOOST_CACHE = {}
    _DAILY_BOOST_TS = 0
    return {"status": "ok", "cleared": cleared, "auto_saved": auto_saved, "ts": datetime.now().isoformat()}


# ═════════════════════════════════════════════════════════════════════════════
# Weekly MAE drift check (backend-only)
# Writes a flag for ops/diagnostics; does NOT affect auto-improve behavior.
# ═════════════════════════════════════════════════════════════════════════════
_MAE_DRIFT_FLAG_PATH = Path("data/mae_drift_flag.json")


@app.get("/api/mae-drift-check")
async def mae_drift_check(request: Request):
    """Weekly cron: compute last 7 calendar days MAE and write a backend-only flag."""
    if not _require_cron_secret(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    THRESHOLD = 2.5
    today = _et_date()
    start = today - timedelta(days=6)
    window_dates = [(start + timedelta(days=i)).isoformat() for i in range(7)]

    total_weighted_abs_error = 0.0
    total_players_compared = 0
    per_date = []

    try:
        for d in window_dates:
            audit = _compute_audit(d)
            if not audit:
                continue
            pc = int(audit.get("players_compared") or 0)
            mae = _safe_float(audit.get("mae"), 0.0)
            if pc > 0 and mae >= 0:
                total_weighted_abs_error += mae * pc
                total_players_compared += pc
                per_date.append({"date": d, "mae": mae, "players_compared": pc})
    except Exception as e:
        print(f"[mae-drift-check] compute failed: {e}")
        return {"status": "error", "reason": str(e)}

    if total_players_compared > 0:
        computed_mae = total_weighted_abs_error / total_players_compared
    else:
        computed_mae = None

    triggered = (computed_mae is not None) and (computed_mae > THRESHOLD)
    payload = {
        "status": "ok",
        "threshold": THRESHOLD,
        "computed_mae": computed_mae,
        "triggered": triggered,
        "window_start": window_dates[0],
        "window_end": window_dates[-1],
        "players_compared": total_players_compared,
        "per_date": per_date,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        _MAE_DRIFT_FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _MAE_DRIFT_FLAG_PATH.write_text(json.dumps(payload, indent=2))
    except Exception as e:
        print(f"[mae-drift-check] write failed: {e}")

    print(f"[mae-drift-check] MAE={computed_mae} threshold={THRESHOLD} triggered={triggered} n={total_players_compared}")
    return payload


# ═════════════════════════════════════════════════════════════════════════════
# INJURY CHECK CRON
# grep: /api/injury-check
# Every 2h: check RotoWire for newly OUT/questionable players in cached picks.
# If any cached player is affected, regenerate only that game's projections.
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/api/injury-check")
async def injury_check(request: Request):
    """Cron (every 2h): check RotoWire for newly OUT/questionable players in cached picks.
    If any cached player is OUT/questionable, regenerate affected games only."""
    if not _require_cron_secret(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    today = _et_date().isoformat()

    # Don't run if slate is locked — picks are frozen post-lock
    games = fetch_games()
    start_times = [g["startTime"] for g in games if g.get("startTime")]
    if start_times and any(_is_locked(st) for st in start_times):
        return {"status": "locked", "skipped": True}

    if not games:
        return {"status": "no_games", "skipped": True}

    # Load current cached slate (try /tmp first, then GitHub)
    cached_slate = _cg("slate_v5") or _slate_cache_from_github()
    if not cached_slate or not cached_slate.get("lineups"):
        return {"status": "no_cache", "skipped": True}

    # Get fresh RotoWire statuses (bust cache to get latest injury news)
    _rw_clear()
    rw_statuses = get_all_statuses()
    if not rw_statuses:
        # RotoWire down — can't check injuries, serve existing cache
        return {"status": "rotowire_unavailable", "skipped": True}

    # Check if any player in Starting 5 or Moonshot is newly OUT/questionable
    injured_games = set()
    affected_players = []
    for lineup_type in ["chalk", "upside"]:
        for player in cached_slate["lineups"].get(lineup_type, []):
            pname = player.get("name", "")
            if pname and not is_safe_to_draft(pname):
                affected_players.append(pname)
                # Find which game this player belongs to
                pteam = player.get("team", "")
                for g in games:
                    if pteam in (g.get("home", {}).get("abbr", ""),
                                 g.get("away", {}).get("abbr", "")):
                        injured_games.add(g["gameId"])
                        break

    if not injured_games:
        return {"status": "ok", "injuries_found": 0, "checked_players": len(
            [p for lt in ["chalk", "upside"]
             for p in cached_slate["lineups"].get(lt, [])])}

    print(f"[injury-check] {len(injured_games)} games affected by injuries: {affected_players}")

    # Load existing per-game projections from GitHub
    all_game_projs = _games_cache_from_github() or {}

    # Regenerate only affected games
    games_map = {g["gameId"]: g for g in games}
    for gid in injured_games:
        game = games_map.get(gid)
        if not game:
            continue
        # Clear per-game /tmp cache to force fresh computation
        try:
            _cp(f"game_proj_{gid}").unlink()
        except Exception:
            pass
        try:
            projections = _run_game(game)
            if projections:
                all_game_projs[gid] = projections
                _cs(f"game_proj_{gid}", projections)
        except Exception as e:
            print(f"[injury-check] game {gid} regen err: {e}")

    # Rebuild full slate lineups with updated projections
    all_proj = []
    for gid, projs in all_game_projs.items():
        all_proj.extend(projs)

    if not all_proj:
        return {"status": "regen_failed", "injuries_found": len(injured_games)}

    _inj_starts = [g["startTime"] for g in games if g.get("startTime")]
    _inj_locked = bool(_inj_starts) and any(_is_locked(st) for st in _inj_starts)
    _apply_post_lock_rs_calibration(all_proj, slate_locked=_inj_locked)
    chalk, upside, core_pool = _build_lineups(all_proj)
    draftable_for_review = cached_slate.get("games", [])
    try:
        chalk, upside = _lineup_review_opus(chalk, upside, all_proj, draftable_for_review, core_pool=core_pool)
    except Exception as _rev_err:
        print(f"[lineup_review] call-site error: {_rev_err}")
    result = {**cached_slate, "lineups": {"chalk": chalk, "upside": upside}}

    # Persist updated slate to /tmp + GitHub
    _cs("slate_v5", result)
    _slate_cache_to_github(result)
    _games_cache_to_github(all_game_projs)

    return {"status": "ok", "injuries_found": len(injured_games),
            "affected_players": affected_players,
            "regenerated_games": list(injured_games)}


# ═════════════════════════════════════════════════════════════════════════════
# LINE OF THE DAY ENGINE — Phase 2
# grep: /api/line-of-the-day, /api/save-line, /api/resolve-line, /api/line-history
# grep: LINE_CSV_HEADER, run_line_engine, line_engine, Odds API, prop edge
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_odds_line(player_name: str, stat_type: str, team_abbr: str, opponent_abbr: str):
    """Fetch the current bookmaker consensus line for a player prop from The Odds API.
    Returns {"line", "odds_over", "odds_under", "books_consensus"} or None on failure."""
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        return None
    market = _STAT_MARKET.get(stat_type, "player_points")
    try:
        # Step 1: find today's event ID for this matchup
        ev_r = requests.get(
            f"{ODDS_API_BASE}/sports/basketball_nba/events",
            params={"apiKey": api_key, "dateFormat": "iso"},
            timeout=10,
        )
        if not ev_r.ok:
            return None
        event_id = None
        for ev in ev_r.json():
            home, away = ev.get("home_team", ""), ev.get("away_team", "")
            if (_abbr_matches(team_abbr, home) or _abbr_matches(team_abbr, away)) and \
               (_abbr_matches(opponent_abbr, home) or _abbr_matches(opponent_abbr, away)):
                event_id = ev["id"]
                break
        if not event_id:
            return None
        # Step 2: fetch player props for that event
        odds_r = requests.get(
            f"{ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds",
            params={
                "apiKey":     api_key,
                "regions":    "us",
                "markets":    market,
                "oddsFormat": "american",
                "bookmakers": "draftkings,fanduel,betmgm,pointsbet",
            },
            timeout=10,
        )
        if not odds_r.ok:
            return None
        # Step 3: collect lines for the matching player across all bookmakers
        lines_over, lines_under, over_prices, under_prices = [], [], [], []
        pname_lower = player_name.lower()
        for book in odds_r.json().get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt["key"] != market:
                    continue
                for outcome in mkt.get("outcomes", []):
                    if pname_lower not in outcome.get("name", "").lower():
                        continue
                    pt = outcome.get("point")
                    if pt is None:
                        continue
                    if outcome.get("description", "").lower() == "over":
                        lines_over.append(pt)
                        over_prices.append(outcome.get("price", -110))
                    else:
                        lines_under.append(pt)
                        under_prices.append(outcome.get("price", -110))
        if not lines_over:
            return None
        # Consensus line = mode across books; fallback to average rounded to 0.5
        try:
            consensus = mode(lines_over)
        except StatisticsError:
            consensus = round(round(mean(lines_over) * 2) / 2, 1)
        return {
            "line":            consensus,
            "odds_over":       int(mean(over_prices))  if over_prices  else -110,
            "odds_under":      int(mean(under_prices)) if under_prices else -110,
            "books_consensus": len(lines_over),
        }
    except Exception:
        return None


def _build_player_odds_map(games):
    """Bulk-fetch Odds API player props for all slate games.

    Makes 1 + N calls (events list + one props call per game) instead of
    2 calls per player — far more API-efficient.

    Returns {(player_name_lower, stat_type): {"line", "odds_over", "odds_under",
    "books_consensus"}} or {} on any failure.
    """
    api_key = os.environ.get("ODDS_API_KEY")
    if not api_key:
        return {}

    # Degraded mode: if the Odds API is rate-limiting / failing, fall back to the
    # last successfully built map (previous hour). This keeps lineup generation alive.
    cache_key = "odds_last_success_map_v1"
    cached_odds_map = {}
    try:
        cp = _cp(cache_key)
        if cp.exists():
            age = time.time() - cp.stat().st_mtime
            if age < 3600:  # 1h
                obj = json.loads(cp.read_text()) if cp.read_text() else {}
                if isinstance(obj, dict) and isinstance(obj.get("data"), dict):
                    cached_odds_map = obj.get("data") or {}
    except Exception:
        cached_odds_map = {}

    markets_param = ",".join(_STAT_MARKET.values())  # player_points,player_rebounds,player_assists
    _market_to_stat = {v: k for k, v in _STAT_MARKET.items()}
    try:
        # Step 1: fetch all NBA events today (1 call)
        ev_r = requests.get(
            f"{ODDS_API_BASE}/sports/basketball_nba/events",
            params={"apiKey": api_key, "dateFormat": "iso"},
            timeout=10,
        )
        if not ev_r.ok:
            print(f"[odds_map] events fetch failed: {ev_r.status_code}")
            return cached_odds_map or {}
        all_events = ev_r.json()
    except Exception as e:
        print(f"[odds_map] events fetch error: {e}")
        return cached_odds_map or {}

    # Map gameId → Odds API event_id
    game_event_ids = []
    for g in games:
        home_abbr = g.get("home", {}).get("abbr", "") or g.get("homeTeam", "")
        away_abbr = g.get("away", {}).get("abbr", "") or g.get("awayTeam", "")
        for ev in all_events:
            ev_home = ev.get("home_team", "")
            ev_away = ev.get("away_team", "")
            if ((_abbr_matches(home_abbr, ev_home) or _abbr_matches(home_abbr, ev_away)) and
                    (_abbr_matches(away_abbr, ev_home) or _abbr_matches(away_abbr, ev_away))):
                game_event_ids.append(ev["id"])
                break

    if not game_event_ids:
        print("[odds_map] no matching Odds API events for slate games")
        return cached_odds_map or {}

    # Step 2: fetch props for each game in parallel (1 call per game)
    def _fetch_event_props(event_id):
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/sports/basketball_nba/events/{event_id}/odds",
                params={
                    "apiKey":     api_key,
                    "regions":    "us",
                    "markets":    markets_param,
                    "oddsFormat": "american",
                    "bookmakers": "draftkings,fanduel,betmgm,pointsbet",
                },
                timeout=10,
            )
            return r.json() if r.ok else {}
        except Exception:
            return {}

    with ThreadPoolExecutor(max_workers=8) as pool:
        event_results = list(pool.map(_fetch_event_props, game_event_ids))

    # Step 3: aggregate lines per (player_name_lower, stat_type)
    raw = {}  # (player_key, stat_type) -> {over_lines, under_lines, over_prices, under_prices}
    for data in event_results:
        for book in data.get("bookmakers", []):
            for mkt in book.get("markets", []):
                stat_type = _market_to_stat.get(mkt["key"])
                if not stat_type:
                    continue
                for outcome in mkt.get("outcomes", []):
                    pt = outcome.get("point")
                    if pt is None:
                        continue
                    player_key = outcome.get("name", "").lower()
                    direction  = outcome.get("description", "").lower()
                    price      = outcome.get("price", -110)
                    key = (player_key, stat_type)
                    if key not in raw:
                        raw[key] = {"over_lines": [], "under_lines": [], "over_prices": [], "under_prices": []}
                    if direction == "over":
                        raw[key]["over_lines"].append(pt)
                        raw[key]["over_prices"].append(price)
                    else:
                        raw[key]["under_lines"].append(pt)
                        raw[key]["under_prices"].append(price)

    # Step 4: compute consensus line per player+stat
    result_map = {}
    for (player_key, stat_type), d in raw.items():
        if not d["over_lines"]:
            continue
        try:
            consensus = mode(d["over_lines"])
        except StatisticsError:
            consensus = round(round(mean(d["over_lines"]) * 2) / 2, 1)
        result_map[(player_key, stat_type)] = {
            "line":            consensus,
            "odds_over":       int(mean(d["over_prices"])) if d["over_prices"] else -110,
            "odds_under":      int(mean(d["under_prices"])) if d["under_prices"] else -110,
            "books_consensus": len(d["over_lines"]),
        }

    print(f"[odds_map] fetched {len(result_map)} player+stat lines from Odds API")
    if not result_map and cached_odds_map:
        return cached_odds_map

    try:
        _cs(cache_key, {"ts": time.time(), "data": result_map})
    except Exception:
        pass

    return result_map


LINE_CSV_HEADER = "date,player_name,player_id,team,opponent,stat_type,line,direction,projection,edge,confidence,narrative,result,actual_stat"


def _get_projections_for_date(date_obj):
    """Return (all_proj, draftable_games) for the given date for line-engine enrichment.
    Checks /tmp cache, then GitHub persistent cache, then runs the full pipeline as a last resort."""
    games = fetch_games(date_obj)
    draftable = [g for g in games if not _is_past_lock_window(g.get("startTime", ""))]
    if not draftable:
        return [], []
    all_proj = []
    # Layer 1: /tmp per-game cache
    for g in draftable:
        gp = _cg(f"game_proj_{g['gameId']}")
        if gp:
            all_proj.extend(gp)
    # Full pipeline if no /tmp cache (skip GitHub games cache here — it adds
    # latency and the line engine still needs Haiku calls regardless)
    if not all_proj:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for fut in as_completed({pool.submit(_run_game, g): g for g in draftable}):
                try:
                    all_proj.extend(fut.result())
                except Exception as e:
                    print(f"[line] proj err: {e}")
    return all_proj, draftable


def _pick_has_display_fields(pick):
    """Check if a pick already has all fields needed for frontend display.
    When True, skip the expensive enrichment pipeline entirely."""
    if not pick:
        return True  # null pick needs no enrichment
    _required = ("season_avg", "proj_min", "avg_min")
    return all(pick.get(f) is not None for f in _required)


def _enrich_loaded_line_picks(picks_dict, date_obj):
    """Enrich over_pick and under_pick with season_avg, proj_min, avg_min, game_time, recent_form_bars when loaded from GitHub.
    Only runs the full projection pipeline when enrichment fields are actually missing — skips it
    entirely when the stored picks already have all display fields (avoids 30-60s cold-start hit)."""
    if not picks_dict:
        return
    _enrich_fields = ("season_avg", "proj_min", "avg_min", "game_time", "game_start_iso", "recent_form_bars")
    def _needs_proj(pick):
        return bool(pick) and any(pick.get(f) is None for f in _enrich_fields)
    needs_proj = any(_needs_proj(picks_dict.get(k)) for k in ("over_pick", "under_pick"))
    if needs_proj:
        all_proj, draftable = _get_projections_for_date(date_obj)
        if all_proj and draftable:
            game_ctx_map = _game_lookup_from_games(draftable)
            for key in ("over_pick", "under_pick"):
                pick = picks_dict.get(key)
                if pick:
                    _enrich_pick_from_projections(pick, all_proj, game_ctx_map)



def _load_line_pick_for_date(date_str: str):
    """Load saved line picks for the given ISO date string (from JSON then CSV).
    Returns {"over_pick": ..., "under_pick": ...} or None.
    Handles legacy single-pick JSON format transparently."""
    json_path = f"data/lines/{date_str}_pick.json"
    saved_raw, _ = _github_get_file(json_path)
    if saved_raw:
        try:
            data = json.loads(saved_raw)
            # New dual-pick format
            if "over_pick" in data or "under_pick" in data:
                return data
            # Legacy single-pick format — wrap by direction
            direction = data.get("direction", "over").lower()
            result = {"over_pick": None, "under_pick": None}
            result[f"{direction}_pick"] = data
            return result
        except Exception:
            pass
    csv_path = f"data/lines/{date_str}.csv"
    csv_raw, _ = _github_get_file(csv_path)
    if csv_raw:
        try:
            rows = _parse_csv(csv_raw, LINE_FIELDS)
            if rows:
                r = rows[0]
                pick = {
                    "player_name": r.get("player_name", ""),
                    "player_id":   r.get("player_id", ""),
                    "team":        r.get("team", ""),
                    "opponent":    r.get("opponent", ""),
                    "stat_type":   r.get("stat_type", "points"),
                    "line":        _safe_float(r.get("line", 0)),
                    "direction":   r.get("direction", "over"),
                    "projection":  _safe_float(r.get("projection", 0)),
                    "edge":        _safe_float(r.get("edge", 0)),
                    "confidence":  int(_safe_float(r.get("confidence", 70))),
                    "narrative":   r.get("narrative", ""),
                    "result":      r.get("result", ""),
                    "actual_stat": r.get("actual_stat", ""),
                    "signals":     [],
                    "model_only":  True,
                }
                direction = pick["direction"].lower()
                result = {"over_pick": None, "under_pick": None}
                result[f"{direction}_pick"] = pick
                # Back-fill JSON in new format
                _github_write_file(json_path, json.dumps(result), f"backfill line pick json {date_str}")
                return result
        except Exception:
            pass
    return None


def _primary_pick(picks):
    """Return the highest-confidence pick from a dual-pick dict, for resolved/rotation checks."""
    if not picks:
        return None
    over  = picks.get("over_pick")
    under = picks.get("under_pick")
    if over and under:
        return over if over.get("confidence", 0) >= under.get("confidence", 0) else under
    return over or under


# grep: NEXT SLATE DATE — _find_next_slate_date, multi-day gap, All-Star break
def _find_next_slate_date(start_date, max_days=30):
    """Find the next date with NBA games, starting from start_date (inclusive).
    Returns a datetime.date or None if no games found within max_days.
    max_days=30 covers All-Star breaks (~10 days) and playoff gaps.
    fetch_games has a 5-min cache per date, so scanning is cheap."""
    for i in range(max_days):
        candidate = start_date + timedelta(days=i)
        games = fetch_games(candidate)
        if games:
            return candidate
    return None


def _run_line_engine_for_date(date):
    """Run the full line engine pipeline for a given date (datetime.date).
    Blocking — call via asyncio.to_thread() from async endpoints."""
    games = fetch_games(date)
    if not games:
        return None, "no_games"
    draftable = [g for g in games if not _is_past_lock_window(g.get("startTime", ""))]
    # Post-lock (slate locked, all games past their lock window): fall back to all games
    # so line picks can still be generated after tip-off. Projections come from cache
    # (_run_game cache is keyed by gameId, survives across lock).
    target_games = draftable if draftable else games
    all_proj = []
    # /tmp per-game cache (warm instance — populated by /api/slate or previous calls)
    for g in target_games:
        gp = _cg(f"game_proj_{g['gameId']}")
        if gp:
            all_proj.extend(gp)
    # Full pipeline if no /tmp cache (skip GitHub games cache here — it adds latency
    # and the line engine still needs Haiku calls regardless, so the savings are minimal
    # vs the risk of timing out on cold start)
    if not all_proj:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(_run_game, g): g for g in target_games}
            try:
                for fut in as_completed(futs, timeout=60):
                    try: all_proj.extend(fut.result(timeout=10))
                    except Exception as e: print(f"line proj err: {e}")
            except TimeoutError:
                print(f"[line] game projection pool timed out after 60s, got {len(all_proj)} projections")
    if not all_proj:
        return None, "no_projections"
    line_config = _cfg("line", _CONFIG_DEFAULTS.get("line", {}))
    # Fetch bookmaker lines from Odds API for all slate games — used as the
    # primary "line" value in picks instead of ESPN season averages.
    # Falls back to {} if ODDS_API_KEY is missing or API fails (graceful degradation).
    player_odds_map = _build_player_odds_map(target_games)
    # Fetch web search news context for the line engine (reuses Layer 1 cache).
    # This gives Claude injury/rotation/rest intel that can improve pick quality.
    news_context = ""
    try:
        news_context = _fetch_nba_news_context(target_games, date=date, all_proj=all_proj)
    except Exception as _news_err:
        print(f"[line] web search for news context failed (non-fatal): {_news_err}")
    result = run_line_engine(all_proj, target_games, line_config, player_odds_map, news_context=news_context)
    return result, None


def _picks_response(picks, **extra):
    """Build the standard /api/line-of-the-day response dict from a dual-pick dict."""
    primary = _primary_pick(picks) if picks else None
    over  = picks.get("over_pick")  if picks else None
    under = picks.get("under_pick") if picks else None
    return {
        "pick":       _normalize_line_pick(primary) if primary else None,
        "over_pick":  _normalize_line_pick(over)    if over    else None,
        "under_pick": _normalize_line_pick(under)   if under   else None,
        **extra,
    }


def _get_mock_line_picks() -> dict:
    """Return a fully-formed mock Line of the Day response for testing.
    No Odds API, ESPN, LightGBM, or GitHub I/O. Safe to call anytime."""
    today = _et_date().isoformat()
    _mock_over = {
        "player_name": "Mock Over Player", "player_id": "mock_over_001",
        "team": "BOS", "opponent": "LAL", "direction": "over",
        "stat_type": "points", "line": 24.5, "projection": 27.3,
        "edge": 2.8, "confidence": 72, "date": today,
        "narrative": "Mock over pick — trending above baseline in recent games.",
        "signals": ["3 of last 5 over", "favorable matchup"],
        "result": "pending", "actual_stat": None,
        "season_avg": 23.1, "proj_min": 36.0, "avg_min": 34.5,
        "game_time": "7:30 PM ET",
        "recent_form_bars": [0.8, 1.1, 0.9, 1.2, 1.0],
        "recent_form_values": [22, 28, 23, 31, 25],
        "odds_over": -115, "odds_under": -105,
        "books_consensus": 24.5, "line_updated_at": f"{today}T18:00:00Z",
    }
    _mock_under = {
        "player_name": "Mock Under Player", "player_id": "mock_under_001",
        "team": "LAL", "opponent": "BOS", "direction": "under",
        "stat_type": "rebounds", "line": 8.5, "projection": 6.8,
        "edge": -1.7, "confidence": 68, "date": today,
        "narrative": "Mock under pick — recent opponents have slowed this player's rebounding.",
        "signals": ["B2B tonight", "4 of last 5 under"],
        "result": "pending", "actual_stat": None,
        "season_avg": 8.2, "proj_min": 32.0, "avg_min": 33.0,
        "game_time": "7:30 PM ET",
        "recent_form_bars": [0.9, 0.7, 0.8, 0.75, 0.85],
        "recent_form_values": [7, 6, 7, 6, 7],
        "odds_over": -110, "odds_under": -110,
        "books_consensus": 8.5, "line_updated_at": f"{today}T18:00:00Z",
    }
    over_norm  = _normalize_line_pick(_mock_over)
    under_norm = _normalize_line_pick(_mock_under)
    # Primary = higher confidence
    primary = over_norm if over_norm["confidence"] >= under_norm["confidence"] else under_norm
    return {
        "mock": True,
        "pick":       primary,
        "over_pick":  over_norm,
        "under_pick": under_norm,
        "slate_summary": {
            "games_evaluated": 2, "props_scanned": 12, "edges_found": 2,
            "timestamp": f"{today}T00:00:00Z", "model_only": True,
        },
    }


def _is_pick_resolved(pick):
    """Return True if pick is resolved (hit/miss) — not pending/null."""
    if not pick or not isinstance(pick, dict):
        return False
    return pick.get("result") not in (None, "", "pending")


async def _get_or_generate_next_slate_pick(today, direction: str):
    """Get or generate the next-slate pick for a single direction.
    Returns (pick_dict, next_slate_date_obj) or (None, None) on failure."""
    next_slate = _find_next_slate_date(today + timedelta(days=1))
    if not next_slate:
        return None, None
    next_str = next_slate.isoformat()
    next_picks = _load_line_pick_for_date(next_str)

    dir_key = f"{direction}_pick"
    existing = (next_picks or {}).get(dir_key)

    if existing and isinstance(existing, dict) and existing.get("player_name"):
        # Enrich if missing display fields
        if not _pick_has_display_fields(existing):
            _tmp = {dir_key: existing}
            _enrich_loaded_line_picks(_tmp, next_slate)
            existing = _tmp[dir_key]
        existing.setdefault("date", next_str)
        return existing, next_slate

    # Direction missing from next-day file — generate it
    eng_result, err = await asyncio.to_thread(_run_line_engine_for_date, next_slate)
    if err or not eng_result:
        return None, None
    new_pick = eng_result.get(dir_key)
    if not new_pick:
        return None, None
    new_pick["date"] = next_str

    # Persist to GitHub (merge into existing file if it has the other direction)
    try:
        other_key = "under_pick" if dir_key == "over_pick" else "over_pick"
        saves = {"over_pick": (next_picks or {}).get("over_pick"),
                 "under_pick": (next_picks or {}).get("under_pick")}
        saves[dir_key] = new_pick
        # Also save the other direction from engine result if currently missing in file —
        # avoids a wasted second engine run when the other direction is needed.
        if not saves.get(other_key) and eng_result.get(other_key):
            other = dict(eng_result[other_key])
            other.setdefault("date", next_str)
            saves[other_key] = other
        _github_write_file(f"data/lines/{next_str}_pick.json",
                           json.dumps(saves), f"line picks for {next_str}")
    except Exception:
        pass

    return new_pick, next_slate


@app.get("/api/line-of-the-day")
async def get_line_of_the_day(request: Request, mock: bool = Query(False, description="Return deterministic mock picks for testing"), nocache: bool = Query(False, description="Bypass /tmp cache and inline-resolve finished picks (sent by frontend after game-final detection)")):
    """Best player prop picks (over + under), with per-direction independent rotation.
    Each direction rotates to the next slate independently when its game finishes.
    Never returns 500."""
    if mock:
        return _get_mock_line_picks()
    rl = _check_rate_limit(request, "line-of-the-day")
    if rl is not None:
        return rl
    try:
        today = _et_date()
        today_str = today.isoformat()

        # ── Cache check ──
        # Serve cache only if BOTH directions are still unresolved (no rotation needed).
        # Once any direction resolves, bust cache so rotation logic runs.
        # Skip entirely when nocache=True (frontend post-game-final re-fetch).
        cached = None if nocache else _cg("line_v1")
        if cached and cached.get("pick"):
            # Strong date guard: never serve line cache across ET-day boundaries.
            # Legacy caches may omit pick.date; treat those as stale.
            _cache_date = cached.get("_cache_date")
            if _cache_date and _cache_date != today_str:
                cached = None
            elif _cache_date is None and not cached["pick"].get("date"):
                cached = None
        if cached and cached.get("pick"):
            _c_over  = cached.get("over_pick") or {}
            _c_under = cached.get("under_pick") or {}
            _c_over_ok  = not _c_over  or not _is_pick_resolved(_c_over)
            _c_under_ok = not _c_under or not _is_pick_resolved(_c_under)
            if _c_over_ok and _c_under_ok:
                _c_date = cached["pick"].get("date", today_str)
                if _c_date >= today_str:
                    _cached_at = cached.get("_cached_at")
                    if _cached_at:
                        try:
                            _age_s = (datetime.utcnow() - datetime.fromisoformat(_cached_at)).total_seconds()
                            if _age_s > 1800:  # 30 min — was 2h; fresher line picks
                                cached = None
                        except Exception:
                            pass
                    if cached:
                        return JSONResponse(cached)

        # ── Load today's picks ──
        today_picks = _load_line_pick_for_date(today_str)

        # If both directions are null, treat as no saved picks
        if today_picks and not today_picks.get("over_pick") and not today_picks.get("under_pick"):
            today_picks = None
        # If both saved direction dates are clearly old, regenerate for today.
        if today_picks:
            _saved_dates = []
            for _k in ("over_pick", "under_pick"):
                _p = today_picks.get(_k)
                if _p and isinstance(_p, dict):
                    _d = _p.get("date")
                    if _d:
                        _saved_dates.append(_d)
            if _saved_dates and all(_d < today_str for _d in _saved_dates):
                print(f"[line-of-the-day] stale saved picks detected ({_saved_dates}) — regenerating")
                today_picks = None

        # ── No saved picks — generate fresh ──
        if not today_picks:
            eng_result, err = await asyncio.to_thread(_run_line_engine_for_date, today)
            if err or not eng_result or not eng_result.get("pick"):
                print(f"[line-of-the-day] first attempt failed ({err}), retrying...")
                eng_result, err = await asyncio.to_thread(_run_line_engine_for_date, today)
            if err or not eng_result or not eng_result.get("pick"):
                return JSONResponse({"pick": None, "over_pick": None, "under_pick": None,
                                     "error": err or "no_projections"}, status_code=200)
            try:
                json_path = f"data/lines/{today_str}_pick.json"
                existing_json, _ = _github_get_file(json_path)
                if not existing_json:
                    saves = {"over_pick": eng_result.get("over_pick"), "under_pick": eng_result.get("under_pick")}
                    _github_write_file(json_path, json.dumps(saves), f"line picks for {today_str}")
            except Exception as _save_err:
                print(f"[line-of-the-day] auto-save err: {_save_err}")
            eng_result["_cached_at"] = datetime.utcnow().isoformat()
            eng_result["_cache_date"] = today_str
            _cs("line_v1", eng_result)
            return JSONResponse(eng_result)

        # ── Fill missing directions (legacy single-pick) ──
        missing_over  = not today_picks.get("over_pick")
        missing_under = not today_picks.get("under_pick")
        if missing_over or missing_under:
            eng_result, err = await asyncio.to_thread(_run_line_engine_for_date, today)
            if not err and eng_result:
                if missing_over and eng_result.get("over_pick"):
                    today_picks["over_pick"] = eng_result["over_pick"]
                    today_picks["over_pick"].setdefault("date", today_str)
                if missing_under and eng_result.get("under_pick"):
                    today_picks["under_pick"] = eng_result["under_pick"]
                    today_picks["under_pick"].setdefault("date", today_str)
                try:
                    _github_write_file(f"data/lines/{today_str}_pick.json",
                                       json.dumps(today_picks),
                                       f"backfill missing direction for {today_str}")
                except Exception: pass

        # ── Inline resolve: game finished but result not yet set by cron ──
        # Only runs when nocache=True — meaning the frontend explicitly detected
        # game-final via the live-stat poll (which already called ESPN to confirm
        # completed=true). Skipped on normal cache misses to avoid redundant ESPN
        # calls while games are still in progress.
        if nocache:
            _inline_resolved = False
            for _dir in ("over", "under"):
                _dir_key = f"{_dir}_pick"
                _p = today_picks.get(_dir_key)
                if not _p or _is_pick_resolved(_p):
                    continue
                _actual = _fetch_player_final_stat(
                    _p.get("player_name", ""), _p.get("stat_type", "points"),
                    date_str=today.strftime("%Y%m%d"), team=_p.get("team")
                )
                if _actual is not None:
                    _line_val = _safe_float(_p.get("line", 0))
                    _res = "hit" if (_actual > _line_val if _dir == "over" else _actual < _line_val) else "miss"
                    today_picks[_dir_key]["result"] = _res
                    today_picks[_dir_key]["actual_stat"] = _actual
                    _inline_resolved = True
                    print(f"[line-of-the-day] inline-resolved {_dir} ({_p.get('player_name')}) → {_res}")
            if _inline_resolved:
                try:
                    _github_write_file(f"data/lines/{today_str}_pick.json", json.dumps(today_picks),
                                       f"inline-resolve line {today_str}")
                    try:
                        _lc = _cp("line_v1", today_str)
                        if _lc.exists(): _lc.unlink()
                    except Exception: pass
                    try:
                        _hc = _cp("line_history_v1")
                        if _hc.exists(): _hc.unlink()
                    except Exception: pass
                except Exception as _ie:
                    print(f"[line-of-the-day] inline-resolve persist error: {_ie}")

        # ── Per-direction independent rotation ──
        # Check each direction: if resolved, swap in next-slate pick for that direction.
        over_pick  = today_picks.get("over_pick")
        under_pick = today_picks.get("under_pick")
        over_resolved  = _is_pick_resolved(over_pick)
        under_resolved = _is_pick_resolved(under_pick)

        final_over  = over_pick
        final_under = under_pick

        if over_resolved:
            next_over, _ = await _get_or_generate_next_slate_pick(today, "over")
            if next_over:
                final_over = next_over

        if under_resolved:
            next_under, _ = await _get_or_generate_next_slate_pick(today, "under")
            if next_under:
                final_under = next_under

        # ── Enrich if needed ──
        combo = {"over_pick": final_over, "under_pick": final_under}
        if not (_pick_has_display_fields(final_over) and _pick_has_display_fields(final_under)):
            # Determine correct date for enrichment — picks may be on different dates
            for _dir_key in ("over_pick", "under_pick"):
                _p = combo.get(_dir_key)
                if _p and not _pick_has_display_fields(_p):
                    _p_date_str = _p.get("date", today_str)
                    try:
                        _p_date = datetime.strptime(_p_date_str, "%Y-%m-%d").date()
                    except Exception:
                        _p_date = today
                    _tmp = {_dir_key: _p}
                    _enrich_loaded_line_picks(_tmp, _p_date)
                    combo[_dir_key] = _tmp[_dir_key]

        result = _picks_response(combo, from_github=True, slate_summary=None)
        result["_cached_at"] = datetime.utcnow().isoformat()
        result["_cache_date"] = today_str
        _cs("line_v1", result)
        return JSONResponse(result)
    except Exception as e:
        print(f"[line-of-the-day] error: {e}")
        traceback.print_exc()
        return JSONResponse(
            {"pick": None, "over_pick": None, "under_pick": None, "error": "server_error"},
            status_code=200,
        )


@app.get("/api/line-force-regenerate")
async def line_force_regenerate():
    """Force-generate today's line picks and overwrite stale line artifacts."""
    today = _et_date()
    today_str = today.isoformat()

    eng_result, err = await asyncio.to_thread(_run_line_engine_for_date, today)
    if err or not eng_result or not eng_result.get("pick"):
        return JSONResponse({"error": err or "no_projections"}, status_code=503)

    saves = {"over_pick": eng_result.get("over_pick"), "under_pick": eng_result.get("under_pick")}
    _github_write_file(
        f"data/lines/{today_str}_pick.json",
        json.dumps(saves),
        f"force regenerate line picks {today_str}",
    )

    # Ensure a CSV row exists for history/resolve compatibility.
    primary = _primary_pick(saves)
    if primary:
        row = ",".join(_csv_escape(str(primary.get(k, ""))) for k in [
            "player_name", "player_id", "team", "opponent", "stat_type",
            "line", "direction", "projection", "edge", "confidence", "narrative",
        ])
        csv_content = LINE_CSV_HEADER + "\n" + f"{today_str}," + row + ",pending,\n"
        _github_write_file(f"data/lines/{today_str}.csv", csv_content, f"force regenerate line csv {today_str}")

    out = _picks_response(saves, from_github=False, slate_summary=eng_result.get("slate_summary"))
    out["_cached_at"] = datetime.utcnow().isoformat()
    out["_cache_date"] = today_str
    _cs("line_v1", out)

    # Bust history cache so the fresh pick card is immediately reflected.
    try:
        _hc = _cp("line_history_v1")
        if _hc.exists():
            _hc.unlink()
    except Exception:
        pass

    return {"status": "ok", "forced": True, "date": today_str, "pick": out.get("pick")}


LINE_FIELDS = LINE_CSV_HEADER.split(",")

@app.post("/api/save-line")
async def save_line(payload: dict = Body(...)):
    """Save today's Line of the Day pick to data/lines/{date}.csv and a companion JSON."""
    today = _et_date().isoformat()
    csv_path  = f"data/lines/{today}.csv"
    json_path = f"data/lines/{today}_pick.json"

    pick       = payload.get("pick")
    over_pick  = payload.get("over_pick")
    under_pick = payload.get("under_pick")
    primary    = pick or over_pick or under_pick
    if not primary:
        return JSONResponse({"error": "No pick provided"}, status_code=400)

    # Always ensure CSV exists (for history / resolve compatibility)
    existing_csv, _ = _github_get_file(csv_path)
    if not existing_csv:
        row = ",".join(_csv_escape(str(primary.get(k, ""))) for k in [
            "player_name", "player_id", "team", "opponent", "stat_type",
            "line", "direction", "projection", "edge", "confidence", "narrative",
        ])
        row = f"{today}," + row + ",pending,"
        csv_content = LINE_CSV_HEADER + "\n" + row + "\n"
        result = _github_write_file(csv_path, csv_content, f"line pick for {today}")
        if result.get("error"):
            return JSONResponse({"error": result["error"]}, status_code=500)

    # Dedup on JSON — if JSON already exists, CSV is now guaranteed to exist too
    existing_json, _ = _github_get_file(json_path)
    if existing_json:
        return {"status": "already_saved", "path": json_path}

    # Write JSON with both picks (new dual-pick format).
    # Direction-aware fallback: if directional picks are absent (legacy single-pick payload),
    # slot the primary pick into the correct direction field based on its direction field
    # rather than blindly assigning it to over_pick regardless of direction.
    _over  = over_pick  or (pick if pick and pick.get("direction") == "over"  else None)
    _under = under_pick or (pick if pick and pick.get("direction") == "under" else None)
    saves = {"over_pick": _over, "under_pick": _under}
    _github_write_file(json_path, json.dumps(saves), f"line picks json for {today}")

    return {"status": "saved", "path": json_path}


@app.get("/api/refresh-line-odds")
async def refresh_line_odds():
    """Hourly cron + Line tab Refresh button: sync current bookmaker line from Odds API.
    No-op if slate is locked — odds freeze at the same boundary as picks (5 min before tip)."""
    today_str = _et_date().isoformat()

    # Respect slate lock — stop updating once ANY game on the slate is locked.
    # Uses any() pattern (not min()) to match /api/slate and /api/save-predictions:
    # on split-window days, once the first game locks, odds freeze for the whole slate.
    games = fetch_games()
    all_start_times = [g["startTime"] for g in games if g.get("startTime")]
    if any(_is_locked(st) for st in all_start_times):
        return {"status": "locked", "message": "Slate locked — odds frozen"}

    picks = _load_line_pick_for_date(today_str)
    if not picks:
        return {"status": "no_pick"}

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    updated = False

    def _pick_odds_key(p):
        return (p.get("player_name", ""), p.get("stat_type", "points"), p.get("team", ""), p.get("opponent", ""))

    odds_cache = {}  # (player_name, stat_type, team, opponent) -> result dict
    for key in ("over_pick", "under_pick"):
        pick = picks.get(key)
        if not pick:
            continue
        pk = _pick_odds_key(pick)
        if pk not in odds_cache:
            odds_cache[pk] = _fetch_odds_line(pick.get("player_name", ""), pick.get("stat_type", "points"), pick.get("team", ""), pick.get("opponent", ""))
        result = odds_cache[pk]
        if result:
            pick.update(result)
            pick["model_only"] = False
        pick["line_updated_at"] = now_utc
        updated = True

    if updated:
        json_path = f"data/lines/{today_str}_pick.json"
        write_result = _github_write_file(json_path, json.dumps(picks), f"odds refresh {today_str}")
        if write_result.get("error"):
            return JSONResponse({"status": "error", "message": write_result["error"]}, status_code=500)
        # Clear /tmp line cache so next /api/line-of-the-day reloads from GitHub (fresh odds).
        # Also clear line_history cache — it embeds books_consensus and odds_* fields.
        for _cache_key in ("line_v1", "line_history_v1"):
            _cf = _cp(_cache_key, today_str)
            if _cf.exists():
                _cf.unlink()

    return {"status": "ok", "updated": updated, "timestamp": now_utc}


@app.get("/api/line-live-stat")
async def get_line_live_stat(
    player_id: str = Query(""), player_name: str = Query(""),
    team: str = Query(""), stat_type: str = Query("points")
):
    """Live in-game stat for a Line pick. No cache — must be fresh for 30s frontend polling."""
    games = fetch_games()
    game = next(
        (g for g in games if team and (
            g["home"]["abbr"].upper() == team.upper() or
            g["away"]["abbr"].upper() == team.upper()
        )), None
    )
    if not game:
        return {"status": "no_game"}

    start_time = game.get("startTime", "")
    if not _is_locked(start_time):
        return {"status": "pregame", "game_id": game["gameId"]}

    data = _espn_get(f"{ESPN}/summary?event={game['gameId']}")
    if not data:
        return {"status": "unavailable"}

    ev = data.get("header", {}).get("competitions", [{}])[0]
    game_status = ev.get("status", {})
    completed = game_status.get("type", {}).get("completed", False)
    period = game_status.get("period", 0)
    clock = game_status.get("displayClock", "")

    if completed:
        return {"status": "final", "game_id": game["gameId"]}
    if not period:
        return {"status": "pregame", "game_id": game["gameId"]}

    # Find the player's current stat in the live box score
    _STAT_BOX_LABEL = {"points": "PTS", "rebounds": "REB", "assists": "AST"}
    label = _STAT_BOX_LABEL.get(stat_type.lower(), "PTS")
    stat_current = None
    for team_block in data.get("boxscore", {}).get("players", []):
        for stat_block in team_block.get("statistics", []):
            labels = stat_block.get("labels", [])
            if label not in labels:
                continue
            idx = labels.index(label)
            for ath in stat_block.get("athletes", []):
                pid = ath.get("athlete", {}).get("id", "")
                name = ath.get("athlete", {}).get("displayName", "")
                if (player_id and pid == player_id) or \
                   (player_name and player_name.lower() in name.lower()):
                    stats = ath.get("stats", [])
                    if idx < len(stats):
                        try: stat_current = float(stats[idx])
                        except (ValueError, TypeError): pass
                    break

    # Pace: project current stat rate to a full 48-minute game baseline
    pace = None
    try:
        parts = clock.split(":")
        clock_mins = int(parts[0]) + int(parts[1]) / 60
        mins_per_period = 12 if period <= 4 else 5
        elapsed = (period - 1) * mins_per_period + (mins_per_period - clock_mins)
        if elapsed > 0 and stat_current is not None:
            pace = round(stat_current / elapsed * 48, 1)
    except Exception:
        pass

    return {
        "status": "live", "stat_current": stat_current, "stat_type": stat_type,
        "period": period, "clock": clock, "pace": pace, "game_id": game["gameId"],
    }


@app.post("/api/resolve-line")
async def resolve_line(payload: dict = Body(...)):
    """Mark today's line pick as hit or miss given the actual stat."""
    date_str = payload.get("date", _et_date().isoformat())
    bad = _validate_date(date_str)
    if bad: return bad
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

    # Also update the _pick.json so rotation logic sees the result
    try:
        json_p = f"data/lines/{date_str}_pick.json"
        pick_raw, _ = _github_get_file(json_p)
        if pick_raw:
            pick_data = json.loads(pick_raw)
            dir_key = f"{direction}_pick"
            if pick_data.get(dir_key):
                pick_data[dir_key]["result"] = result
                pick_data[dir_key]["actual_stat"] = str(actual_f)
            other_dir = "under" if direction == "over" else "over"
            other_key = f"{other_dir}_pick"
            other = pick_data.get(other_key)
            player_name_r = row.get("player_name", "")
            if other and other.get("player_name", "").lower() == player_name_r.lower():
                other_line = _safe_float(other.get("line", 0))
                other_res = "hit" if (actual_f > other_line if other_dir == "over" else actual_f < other_line) else "miss"
                pick_data[other_key]["result"] = other_res
                pick_data[other_key]["actual_stat"] = str(actual_f)
            _github_write_file(json_p, json.dumps(pick_data),
                               f"resolve line json {date_str}: {result}")
    except Exception as e:
        print(f"[resolve-line] json update err: {e}")

    # Bust the line cache for this date
    try:
        lc = _cp("line_v1", date_str)
        if lc.exists():
            lc.unlink()
    except Exception: pass

    return {"status": "resolved", "result": result, "actual": actual_f}


def _strip_name_suffix(n: str) -> str:
    """Strip common name suffixes (Jr., Sr., III, II, IV) for fuzzy matching."""
    return re.sub(r'\s+(jr\.?|sr\.?|iii|ii|iv)\s*$', '', n, flags=re.IGNORECASE).strip()

def _name_matches(player_lower: str, espn_name_lower: str) -> bool:
    """Check if player name matches ESPN name, with suffix-stripped fallback."""
    if player_lower in espn_name_lower or espn_name_lower in player_lower:
        return True
    stripped_player = _strip_name_suffix(player_lower)
    stripped_espn = _strip_name_suffix(espn_name_lower)
    if stripped_player and stripped_espn:
        return stripped_player in stripped_espn or stripped_espn in stripped_player
    return False

def _fetch_player_final_stat(player_name: str, stat_type: str, date_str: str = None, team: str = None) -> "float | None":
    """Fetch a player's final boxscore stat from ESPN for today's (or a given) games.
    stat_type is the line pick type: 'points', 'rebounds', 'assists', etc.
    date_str: optional YYYYMMDD string; defaults to today in ET.
    team: optional team abbreviation (e.g. 'CLE') — when provided, returns 0.0 for DNP
    players whose team's game is final but they don't appear in the boxscore.
    Returns the numeric value or None if not found / game not final."""
    label_map = {
        "points": "PTS", "pts": "PTS",
        "rebounds": "REB", "reb": "REB",
        "assists": "AST", "ast": "AST",
        "steals": "STL", "stl": "STL",
        "blocks": "BLK", "blk": "BLK",
        "turnovers": "TO", "tov": "TO",
    }
    espn_label = label_map.get(stat_type.lower(), "PTS")
    player_lower = player_name.lower().strip()

    query_date = date_str if date_str else _et_date().strftime("%Y%m%d")
    data = _espn_get(f"{ESPN}/scoreboard?dates={query_date}")
    team_played_in_final = False
    for ev in data.get("events", []):
        # Only use completed games
        completed = ev.get("status", {}).get("type", {}).get("completed", False)
        if not completed:
            continue
        game_id = ev.get("id", "")
        box = _espn_get(f"{ESPN}/summary?event={game_id}")
        for team_block in box.get("boxscore", {}).get("players", []):
            team_abbr = team_block.get("team", {}).get("abbreviation", "")
            if team and team.upper() == team_abbr.upper():
                team_played_in_final = True
            stats = team_block.get("statistics", [])
            if not stats:
                continue
            labels = stats[0].get("labels", [])
            if espn_label not in labels:
                continue
            idx = labels.index(espn_label)
            for ath in stats[0].get("athletes", []):
                name = ath.get("athlete", {}).get("displayName", "")
                if _name_matches(player_lower, name.lower()):
                    vals = ath.get("stats", [])
                    if idx < len(vals):
                        try:
                            return float(vals[idx])
                        except (ValueError, TypeError):
                            return None
    # DNP fallback: player's team played in a completed game but player not in boxscore
    if team and team_played_in_final:
        print(f"[resolve] {player_name} not in boxscore (DNP) — returning 0.0 for {stat_type}")
        return 0.0
    return None


@app.get("/api/auto-resolve-line")
async def auto_resolve_line(request: Request):
    """Auto-resolve today's line picks independently as each game finishes.
    Checks BOTH over and under picks — resolves whichever game is final,
    even if the other game is still in progress. Runs on cron every 15 min.

    Midnight rollover: a late game (e.g. 10 PM ET tip) can finish after midnight
    ET, advancing _et_date() to the next day before the pick resolves. In that
    case today's pick file doesn't exist yet, so we fall back to yesterday's file.
    We also pass the pick's actual date to _fetch_player_final_stat so it queries
    the right ESPN scoreboard, and compute the next-day target from pick_date+1
    (not _et_date()+1, which would skip a day after midnight rollover)."""
    if not _require_cron_secret(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    today = _et_date().isoformat()

    # Determine which date's pick file to use — midnight rollover support.
    pick_date = today
    json_path = f"data/lines/{pick_date}_pick.json"
    pick_data_raw, _ = _github_get_file(json_path)
    if not pick_data_raw:
        # Check yesterday: a 10 PM ET game finishing at 12:30 AM is stored
        # under yesterday's date but the cron now runs on the new ET day.
        yesterday = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).date().isoformat()
        ypath = f"data/lines/{yesterday}_pick.json"
        pick_data_raw, _ = _github_get_file(ypath)
        if pick_data_raw:
            pick_date = yesterday
            json_path = ypath
        else:
            return {"status": "no_pick"}

    csv_path = f"data/lines/{pick_date}.csv"
    try:
        pick_data = json.loads(pick_data_raw)
    except Exception:
        return {"status": "no_pick"}

    resolved_any = False
    results = {}

    # Pass the pick's actual date to ESPN boxscore queries so games that ran
    # past midnight are found on the date they started, not today's new date.
    espn_date = pick_date.replace("-", "")  # YYYYMMDD

    for direction in ("over", "under"):
        dir_key = f"{direction}_pick"
        pick = pick_data.get(dir_key)
        if not pick or not isinstance(pick, dict):
            continue
        if pick.get("result") and pick["result"] not in ("pending", ""):
            results[direction] = {"status": "already_resolved", "result": pick["result"]}
            continue

        player_name = pick.get("player_name", "")
        stat_type   = pick.get("stat_type", "points")
        if not player_name:
            continue

        actual = _fetch_player_final_stat(player_name, stat_type, date_str=espn_date, team=pick.get("team"))
        if actual is None:
            results[direction] = {"status": "game_not_final", "player": player_name}
            continue

        line_val = _safe_float(pick.get("line", 0))
        result = "hit" if (actual > line_val if direction == "over" else actual < line_val) else "miss"

        pick_data[dir_key]["result"]      = result
        pick_data[dir_key]["actual_stat"] = actual
        resolved_any = True
        results[direction] = {"status": "resolved", "result": result,
                              "actual_stat": actual, "player": player_name,
                              "line": line_val}

    if not resolved_any:
        return {"status": "no_change", "details": results}

    # Persist updated JSON
    _github_write_file(json_path, json.dumps(pick_data),
                       f"auto-resolve line {pick_date}")

    # Update or create CSV for the primary pick (first row matches primary direction)
    csv_existing, _ = _github_get_file(csv_path)
    if not csv_existing:
        # CSV was never created (save_line dedup returned early) — create it now
        primary = _primary_pick(pick_data)
        if primary:
            row_vals = [pick_date] + [_csv_escape(str(primary.get(k, ""))) for k in LINE_FIELDS[1:]]
            csv_existing = LINE_CSV_HEADER + "\n" + ",".join(row_vals) + "\n"
            _github_write_file(csv_path, csv_existing, f"line csv for {pick_date}")
            print(f"[auto-resolve] created missing CSV for {pick_date}")
    if csv_existing:
        rows = _parse_csv(csv_existing, LINE_FIELDS)
        if rows:
            row = rows[0]
            csv_dir = row.get("direction", "over")
            dir_key = f"{csv_dir}_pick"
            pick = pick_data.get(dir_key, {})
            if pick.get("result") and pick["result"] not in ("pending", ""):
                updated = dict(row)
                updated["result"]      = pick["result"]
                updated["actual_stat"] = str(pick.get("actual_stat", ""))
                new_row = ",".join(_csv_escape(str(updated.get(k, ""))) for k in LINE_FIELDS)
                _github_write_file(csv_path, LINE_CSV_HEADER + "\n" + new_row + "\n",
                                   f"auto-resolve line csv {pick_date}")

    # Bust the line cache so /api/line-of-the-day sees the resolution and rotates.
    # Bust both pick_date and today (they differ on midnight rollover).
    for _bust_date in set([pick_date, today]):
        try:
            line_cache = _cp("line_v1", _bust_date)
            if line_cache.exists():
                line_cache.unlink()
        except Exception: pass
    # Bust history cache so resolved picks appear immediately in /api/line-history
    try:
        hist_cache = _cp("line_history_v1")
        if hist_cache.exists():
            hist_cache.unlink()
    except Exception: pass

    # Pre-generate next day's picks when any pick resolves.
    # Runs the engine if: (a) no next-day file exists, OR (b) file exists but is missing
    # a direction (partial generation from an earlier resolve). This ensures both
    # directions are always available for per-direction independent rotation.
    over_done  = pick_data.get("over_pick", {}).get("result") not in (None, "", "pending")
    under_done = pick_data.get("under_pick", {}).get("result") not in (None, "", "pending")
    if over_done or under_done or resolved_any:
        try:
            _next_candidate = (datetime.strptime(pick_date, "%Y-%m-%d") + timedelta(days=1)).date()
            next_day = _find_next_slate_date(_next_candidate) or _next_candidate
            next_day_str = next_day.isoformat()
            tomorrow_json = f"data/lines/{next_day_str}_pick.json"
            existing_tomorrow_raw, _ = _github_get_file(tomorrow_json)
            existing_tomorrow = None
            if existing_tomorrow_raw:
                try:
                    existing_tomorrow = json.loads(existing_tomorrow_raw)
                except Exception:
                    pass
            # Check if either direction is missing from the existing file
            _need_gen = (not existing_tomorrow
                         or not existing_tomorrow.get("over_pick")
                         or not existing_tomorrow.get("under_pick"))
            if _need_gen:
                try:
                    eng_result, err = await asyncio.wait_for(
                        asyncio.to_thread(_run_line_engine_for_date, next_day),
                        timeout=120.0
                    )
                except asyncio.TimeoutError:
                    err = "timeout"
                    eng_result = None
                    print(f"[auto-resolve] next-day generation TIMEOUT for {next_day_str}")

                if not err and eng_result and eng_result.get("pick"):
                    # Merge with existing file (preserve any direction already saved)
                    saves = {
                        "over_pick":  (existing_tomorrow or {}).get("over_pick") or eng_result.get("over_pick"),
                        "under_pick": (existing_tomorrow or {}).get("under_pick") or eng_result.get("under_pick"),
                    }
                    # Stamp date on freshly generated picks so the card shows the right date
                    for _dk in ("over_pick", "under_pick"):
                        if saves.get(_dk) and isinstance(saves[_dk], dict) and not saves[_dk].get("date"):
                            saves[_dk]["date"] = next_day_str
                    _github_write_file(tomorrow_json, json.dumps(saves),
                                       f"line picks for {next_day_str}")
                    results["next_day"] = next_day_str
                elif err:
                    print(f"[auto-resolve] next-day generation failed for {next_day_str}: {err}")
        except Exception as e:
            print(f"[auto-resolve] next-day generation err: {e}")

    # ── Backfill: resolve pending picks from past 7 days ──
    # auto_resolve_line only checks today/yesterday. Picks that couldn't be resolved
    # (ESPN lag, rate limit, midnight rollover) stay pending forever and never appear
    # in history. Scan back 7 days and attempt resolution for any still-pending pick.
    try:
        for _back_days in range(2, 8):
            _back_date = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=_back_days)).date().isoformat()
            _back_json = f"data/lines/{_back_date}_pick.json"
            _back_raw, _ = _github_get_file(_back_json)
            if not _back_raw:
                continue
            try:
                _back_data = json.loads(_back_raw)
            except Exception:
                continue
            _back_espn_date = _back_date.replace("-", "")
            _back_changed = False
            for _bdir in ("over", "under"):
                _bdk = f"{_bdir}_pick"
                _bp = _back_data.get(_bdk)
                if not _bp or not isinstance(_bp, dict):
                    continue
                if _bp.get("result") and _bp["result"] not in ("pending", ""):
                    continue  # already resolved
                _bpn = _bp.get("player_name", "")
                _bst = _bp.get("stat_type", "points")
                if not _bpn:
                    continue
                _bactual = _fetch_player_final_stat(_bpn, _bst, date_str=_back_espn_date, team=_bp.get("team"))
                if _bactual is None:
                    continue
                _bline = _safe_float(_bp.get("line", 0))
                _bres = "hit" if (_bactual > _bline if _bdir == "over" else _bactual < _bline) else "miss"
                _back_data[_bdk]["result"] = _bres
                _back_data[_bdk]["actual_stat"] = _bactual
                _back_changed = True
                print(f"[auto-resolve] backfill {_bdir} ({_bpn}) on {_back_date} → {_bres}")
            if _back_changed:
                try:
                    _github_write_file(_back_json, json.dumps(_back_data), f"backfill resolve {_back_date}")
                    # Bust history cache so resolved picks appear immediately
                    try:
                        _hc2 = _cp("line_history_v1")
                        if _hc2.exists(): _hc2.unlink()
                    except Exception: pass
                except Exception as _be:
                    print(f"[auto-resolve] backfill write err {_back_date}: {_be}")
    except Exception as _backfill_err:
        print(f"[auto-resolve] backfill err: {_backfill_err}")

    # ── Piggyback: save predictions for any newly-locked games ──
    # This cron runs every 30 min. On split-window days (e.g. 1 PM + 7 PM games),
    # the /api/refresh cron (once at 2 PM EST) misses late-locking games.
    # By calling save_predictions here, later games get their predictions written
    # to the CSV as soon as they lock, within the next 30-min window.
    try:
        _sp_games = fetch_games()
        _sp_starts = [g["startTime"] for g in _sp_games if g.get("startTime")]
        if _sp_starts and any(_is_locked(st) for st in _sp_starts):
            sp_result = await save_predictions()
            # save_predictions returns dict on success, JSONResponse on error
            if isinstance(sp_result, dict):
                sp_status = sp_result.get("status", "ok")
                if sp_status != "unchanged":
                    print(f"[auto-resolve] piggyback save-predictions: {sp_status}, rows={sp_result.get('rows', '?')}")
    except Exception as e:
        print(f"[auto-resolve] piggyback save-predictions err: {e}")

    return {"status": "resolved", "details": results}


@app.get("/api/line-history")
async def line_history():
    """Return recent Line of the Day picks with results."""
    # 10-min cache TTL — history only changes when auto-resolve-line fires (every 30 min cron)
    # or resolve-line is called manually (both clear line_history_v1).
    _hist_cached = _cg("line_history_v1")
    if _hist_cached and isinstance(_hist_cached, dict) and "data" in _hist_cached:
        if time.time() - _hist_cached.get("ts", 0) < 600:
            return _hist_cached["data"]

    items = _github_list_dir("data/lines")
    # Collect unique dates from both CSV files and JSON pick files so JSON-only
    # dates (where CSV write failed or save_line was never called from the frontend)
    # still appear in history — the JSON has the resolved results.
    csv_dates  = {i["name"][:-4]  for i in items if i.get("name", "").endswith(".csv")}
    json_dates = {i["name"][:-10] for i in items if i.get("name", "").endswith("_pick.json")}
    # Cap at 30 days for more comprehensive history; parallel GitHub fetches keep it fast
    all_dates  = sorted(csv_dates | json_dates, reverse=True)[:30]

    # Fetch CSV + JSON for each date in parallel
    def _fetch_date_files(date_str):
        csv_raw,  _ = _github_get_file(f"data/lines/{date_str}.csv")
        json_raw, _ = _github_get_file(f"data/lines/{date_str}_pick.json")
        return date_str, csv_raw, json_raw

    fetched = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for date_str, csv_raw, json_raw in pool.map(_fetch_date_files, all_dates):
            fetched[date_str] = (csv_raw, json_raw)

    results = []
    # ESPN backfill tasks: collected in first pass, then run in parallel to avoid sequential I/O
    # Each task: (p_copy, p_line, p_dir, date_str, json_key, jd_ref)
    espn_queue = []

    for date_str in all_dates:
        content, json_raw = fetched.get(date_str, (None, None))
        if not content and not json_raw:
            continue
        rows = _parse_csv(content, LINE_FIELDS) if content else []
        csv_primary = rows[0] if rows else {}

        # Try JSON for both-direction picks (over + under per day).
        # CSV only stores the primary pick; JSON has both.
        if json_raw:
            try:
                jd = json.loads(json_raw)
                added_dirs = set()
                for jkey in ("over_pick", "under_pick"):
                    p = jd.get(jkey)
                    if not (p and isinstance(p, dict) and p.get("player_name")):
                        continue
                    p = dict(p)
                    p.setdefault("date", date_str)
                    if not p.get("result") or p.get("result") == "pending":
                        csv_actual = _safe_float(csv_primary.get("actual_stat", 0))
                        if p.get("direction") == csv_primary.get("direction"):
                            # Primary direction — copy result directly from CSV
                            p["result"]      = csv_primary.get("result", "pending")
                            p["actual_stat"] = csv_primary.get("actual_stat", "")
                        elif (p.get("player_name", "").lower() == csv_primary.get("player_name", "").lower()
                              and csv_primary.get("result") in ("hit", "miss")):
                            # Other direction, same player — infer result from CSV
                            if csv_primary.get("actual_stat"):
                                # Compute directly from actual stat
                                p_line = _safe_float(p.get("line", 0))
                                p_dir  = p.get("direction", "over")
                                p["result"] = "hit" if (csv_actual > p_line if p_dir == "over" else csv_actual < p_line) else "miss"
                            else:
                                # No actual_stat — invert the CSV direction's result
                                p["result"] = "miss" if csv_primary["result"] == "hit" else "hit"
                            p["actual_stat"] = csv_primary.get("actual_stat", "")
                        elif date_str < _et_date().isoformat():
                            # Different player, historical date — queue for parallel ESPN lookup
                            espn_queue.append((p, _safe_float(p.get("line", 0)), p.get("direction", "over"), date_str, jkey, jd))
                            added_dirs.add(p.get("direction"))
                            continue
                    # Only add resolved picks to history — pending picks belong
                    # on the main card, not in Recent Picks.
                    if p.get("result") and p["result"] not in ("pending", ""):
                        results.append(_normalize_line_pick(p))
                    added_dirs.add(p.get("direction"))
                # Fallback: if JSON didn't cover the primary direction, add CSV row
                if csv_primary.get("direction") not in added_dirs:
                    if csv_primary.get("result") and csv_primary["result"] not in ("pending", ""):
                        results.append(_normalize_line_pick(csv_primary))
                continue
            except Exception:
                pass
        # Only add resolved picks — pending picks stay on the main card
        if csv_primary.get("result") and csv_primary["result"] not in ("pending", ""):
            results.append(_normalize_line_pick(csv_primary))

    # Parallel ESPN backfill for secondary-direction picks with different players.
    # These are rare (only fires when over/under are different players AND unresolved),
    # but when they do occur they write the result back so future loads skip ESPN entirely.
    if espn_queue:
        def _backfill_espn(task):
            p, p_line, p_dir, date_str, jkey, jd = task
            actual = _fetch_player_final_stat(
                p.get("player_name", ""), p.get("stat_type", "points"), date_str.replace("-", "")
            )
            return p, p_line, p_dir, actual, date_str, jkey, jd

        with ThreadPoolExecutor(max_workers=min(len(espn_queue), 8)) as pool:
            for p, p_line, p_dir, actual, date_str, jkey, jd in pool.map(_backfill_espn, espn_queue):
                if actual is not None:
                    p["result"]      = "hit" if (actual > p_line if p_dir == "over" else actual < p_line) else "miss"
                    p["actual_stat"] = str(actual)
                    # Write back to JSON so future history loads skip this ESPN call
                    try:
                        jd[jkey]["result"]      = p["result"]
                        jd[jkey]["actual_stat"] = p["actual_stat"]
                        _github_write_file(
                            f"data/lines/{date_str}_pick.json",
                            json.dumps(jd),
                            f"backfill {date_str} {p.get('direction')} result"
                        )
                    except Exception:
                        pass
                if p.get("result") and p["result"] not in ("pending", ""):
                    results.append(_normalize_line_pick(p))

    # Deduplicate by (player_name, direction): allow same player to appear as both
    # over and under on the same day, but prevent duplicates of the exact same pick.
    seen: set = set()
    deduped = []
    for r in results:
        dedup_key = (r.get("player_name", ""), r.get("direction", ""))
        if dedup_key not in seen:
            seen.add(dedup_key)
            deduped.append(r)
    results = deduped

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

    out = {
        "picks":        results,
        "hit_rate":     hit_rate,
        "total_picks":  len(results),
        "resolved":     total_resolved,
        "streak":       streak,
        "streak_type":  streak_type,
    }
    _cs("line_history_v1", {"data": out, "ts": time.time()})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# BEN / LAB ENGINE — Phase 3
# grep: /api/lab/status, /api/lab/briefing, /api/lab/update-config, /api/lab/chat
# grep: /api/lab/backtest, /api/lab/rollback, _all_games_final, Lab lock system
# grep: _GAMES_FINAL_CACHE, buildLabSystemPrompt, claude-opus-4-6, Ben
# ═════════════════════════════════════════════════════════════════════════════
# PHASE 3: LAB
# ═════════════════════════════════════════════════════════════════════════════

_GAMES_FINAL_CACHE: dict = {"result": None, "ts": 0.0, "date": ""}

# grep: ALL GAMES FINAL — _all_games_final, ESPN scoreboard poll, midnight rollover, 4.5h fallback
def _all_games_final(games):
    """Check ESPN scoreboard to see if all today's games are completed.
    Cached with adaptive TTL: 60s when locked slate, 180s pre-slate.
    Returns (all_final, remaining, finals, latest_remaining_start_iso).

    Handles midnight-rollover: late games (e.g. 10 PM ET tip-off) can still be
    running past midnight when _et_date() has already advanced to the next day.
    If today's ESPN scoreboard has no started/completed games, we also check
    yesterday's scoreboard so those late games are not missed.

    Future/pregame games on the next day's slate are NOT counted as remaining —
    only games that have actually passed their tip-off time matter here."""
    now_ts = datetime.now(timezone.utc).timestamp()
    today_str = _et_date().strftime("%Y%m%d")

    # Determine if slate is locked. During locked slate, use 60s TTL for responsiveness.
    # This enables event-driven unlock detection without hammering ESPN API.
    slate_locked = any(_is_locked(g.get("startTime", "")) for g in games if g.get("startTime"))
    cache_ttl = 60 if slate_locked else 180

    # Cache valid only if within TTL AND still the same ET date
    if (_GAMES_FINAL_CACHE["result"] is not None
            and now_ts - _GAMES_FINAL_CACHE["ts"] < cache_ttl
            and _GAMES_FINAL_CACHE.get("date") == today_str):
        return tuple(_GAMES_FINAL_CACHE["result"])

    def _tally(scoreboard_data):
        fins, rem, latest = 0, 0, None
        for ev in scoreboard_data.get("events", []):
            completed = ev.get("status", {}).get("type", {}).get("completed", False)
            start = ev.get("date", "")
            if completed:
                fins += 1
            else:
                # Only count games that have actually started. Future pregame games
                # (tomorrow's slate already in ESPN) must NOT block unlock.
                try:
                    start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
                    started = now_ts > start_dt.timestamp() - 300  # 5-min grace
                except Exception:
                    started = False
                if started:
                    rem += 1
                    if start and (latest is None or start > latest):
                        latest = start
        return fins, rem, latest

    data = _espn_get(f"{ESPN}/scoreboard?dates={today_str}")
    finals, remaining, latest_remaining = _tally(data)

    # Midnight rollover: if today has no started or completed games yet,
    # check yesterday — late games may still be running past midnight ET.
    if finals == 0 and remaining == 0:
        yesterday_str = (_et_date() - timedelta(days=1)).strftime("%Y%m%d")
        ydata = _espn_get(f"{ESPN}/scoreboard?dates={yesterday_str}")
        finals, remaining, latest_remaining = _tally(ydata)

    # All done when no started games are still in progress AND we actually saw
    # completed games. If ESPN returns empty data (outage/rate-limit), both counts
    # are 0 — that's NOT confirmation games are final. Only unlock when we have
    # positive evidence (at least one completed game) or it's genuinely a no-game day.
    all_final = (remaining == 0 and finals > 0)

    # AGGRESSIVE FALLBACK: If ESPN isn't marking games as complete but they've been
    # running for 4.5+ hours, treat as final anyway. This handles ESPN API delays.
    # NBA games max ~3.5h; 4.5h buffer includes OT, 2OT, and processing delays.
    # KEY: Removed the `finals > 0` requirement. Now fires even if ESPN is completely
    # lagged on all game statuses. This prevents 6-hour lock ceiling waits when ESPN
    # is slow during high-traffic periods (evening games on Saturdays).
    if not all_final and remaining > 0 and latest_remaining:
        try:
            latest_dt = datetime.fromisoformat(latest_remaining.replace("Z", "+00:00"))
            hours_since_start = (now_ts - latest_dt.timestamp()) / 3600
            if hours_since_start >= 4.5:
                all_final = True
                print(f"[espn fallback] latest_remaining running {hours_since_start:.1f}h — forcing all_final=True (ESPN lagged)")
        except Exception:
            pass

    # Safety: if ESPN returned no data (both counts 0), we cannot confirm games
    # are final even if it's past game time. Only lock files and manual override
    # can force unlock when ESPN is unreachable. This prevents false unlock during
    # ESPN outages that coincide with off-days or startup delays.
    if finals == 0 and remaining == 0:
        all_final = False

    result = (all_final, remaining, finals, latest_remaining)
    _GAMES_FINAL_CACHE.update({"result": list(result), "ts": now_ts, "date": today_str})
    return result


@app.get("/api/lab/status")
async def lab_status():
    """Return Lab lock status based on slate state and game completion.
    On any exception (ESPN timeout, config load, etc.) returns 200 with locked=True
    so the frontend shows a retryable message instead of a generic fetch failure."""
    try:
        games = fetch_games()
        draftable = [g for g in games if not _is_past_lock_window(g.get("startTime", ""))]

        # If games are currently in progress (slate locked, games not yet final).
        # Check ALL game start times — not just earliest. On a 6-game Saturday slate,
        # the earliest game (e.g. 2 PM) expires its 6h lock window by 8 PM, but late
        # games (7-10 PM tips) are still live. Using min() caused Ben to unlock mid-slate.
        start_times = [g["startTime"] for g in games if g.get("startTime")]
        slate_locked = any(_is_locked(st) for st in start_times) if start_times else False
        earliest = min(start_times) if start_times else None

        # Fast path: pre-slate (no game has started). Return unlocked without calling ESPN scoreboard.
        if not slate_locked and earliest:
            try:
                lock_buf = _cfg("projection.lock_buffer_minutes", 5)
                earliest_dt = datetime.fromisoformat(earliest.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if now < earliest_dt - timedelta(minutes=lock_buf):
                    cfg = _load_config()
                    cfg_version = cfg.get("version", 1)
                    return {
                        "locked": False,
                        "reason": "Pre-slate window",
                        "current_config_version": cfg_version,
                        "games_remaining": 0,
                        "games_final": 0,
                        "next_lock_time": earliest,
                    }
            except Exception:
                pass

        all_final, remaining, finals, latest_remaining_start = _all_games_final(games)

        cfg = _load_config()
        cfg_version = cfg.get("version", 1)
        last_change = (cfg.get("changelog") or [{}])[-1]

        if all_final:
            # All started games confirmed done (requires finals > 0 from _all_games_final,
            # so ESPN outages/empty responses won't falsely trigger this path).
            next_lock = None
            upcoming = 0
            if draftable:
                lock_buf = _cfg("projection.lock_buffer_minutes", 5)
                ns = min(g["startTime"] for g in draftable if g.get("startTime"))
                gs = datetime.fromisoformat(ns.replace("Z", "+00:00"))
                next_lock = (gs - timedelta(minutes=lock_buf)).isoformat()
                upcoming = len(draftable)
            reason = "All games final" if not upcoming else f"Break — {upcoming} game{'s' if upcoming != 1 else ''} later today"
            return {
                "locked": False,
                "reason": reason,
                "current_config_version": cfg_version,
                "games_remaining": upcoming,
                "games_final": finals,
                "next_lock_time": next_lock,
            }
        elif slate_locked:
            # Total remaining = all games on slate minus finals (includes in-progress + scheduled).
            total_remaining = len(games) - finals
            est_unlock = None
            latest_start = max(start_times) if start_times else earliest
            if latest_start:
                try:
                    gs = datetime.fromisoformat(latest_start.replace("Z", "+00:00"))
                    est_unlock = (gs + timedelta(hours=2, minutes=30)).isoformat()
                except Exception: pass
            return {
                "locked": True,
                "reason": f"Slate in progress — {total_remaining} game{'s' if total_remaining != 1 else ''} remaining",
                "current_config_version": cfg_version,
                "games_remaining": total_remaining,
                "games_final": finals,
                "estimated_unlock": est_unlock,
            }
        else:
            # Pre-slate — lab is open. If no games and no finals, check GitHub lock file (ESPN down).
            if not games and not finals:
                try:
                    today_str = _et_date().strftime("%Y-%m-%d")
                    lock_content, _ = _github_get_file(f"data/locks/{today_str}_slate.json")
                    if lock_content:
                        return {
                            "locked": True,
                            "reason": "ESPN unavailable — defaulting to locked (games scheduled today)",
                            "current_config_version": cfg_version,
                            "games_remaining": 0,
                            "games_final": 0,
                        }
                except Exception:
                    pass
            return {
                "locked": False,
                "reason": "Pre-slate window",
                "current_config_version": cfg_version,
                "games_remaining": 0,
                "games_final": 0,
                "next_lock_time": None,
            }
    except Exception as e:
        print(f"[lab/status] error: {e}")
        try:
            cfg = _load_config()
            cfg_version = cfg.get("version", 1)
        except Exception:
            cfg_version = 1
        return {
            "locked": True,
            "reason": "Server temporarily unavailable — try again",
            "current_config_version": cfg_version,
            "games_remaining": 0,
            "games_final": 0,
        }


@app.get("/api/lab/briefing")
async def lab_briefing():
    """Analyze recent prediction accuracy and return structured briefing for the Lab."""
    pred_items = _github_list_dir("data/predictions")
    act_items  = _github_list_dir("data/actuals")
    act_dates  = {i["name"].replace(".csv","") for i in act_items if i["name"].endswith(".csv")}

    # Find dates with both predictions and actuals
    paired = []
    for item in sorted(pred_items, key=lambda x: x.get("name",""), reverse=True):
        name = item.get("name","")
        if not name.endswith(".csv"): continue
        d = name[:-4]
        if d in act_dates:
            paired.append(d)

    # Gather audits — use pre-computed JSON when cached, else compute live
    audits = []
    for d in paired[:10]:
        cached_json, _ = _github_get_file(f"data/audit/{d}.json")
        if cached_json:
            try:
                audits.append(json.loads(cached_json))
                continue
            except Exception:
                pass
        a = _compute_audit(d)
        if a:
            audits.append(a)

    latest_slate = None
    rolling_errors = []
    for a in audits:
        rolling_errors.extend([a["mae"]] * a.get("players_compared", 1))
        if latest_slate is None:
            latest_slate = {
                "date":                  a["date"],
                "players_with_actuals":  a["players_compared"],
                "mean_absolute_error":   a["mae"],
                "directional_accuracy":  a.get("directional_accuracy"),
                "over_projected":        a.get("over_projected", 0),
                "under_projected":       a.get("under_projected", 0),
                "biggest_misses":        a.get("biggest_misses", [])[:5],
                "simulated_draft_score": a.get("simulated_draft_score"),
            }

    overall_mae = round(sum(rolling_errors) / len(rolling_errors), 2) if rolling_errors else None
    cfg = _load_config()

    # Pattern detection
    patterns = []
    if latest_slate and latest_slate["mean_absolute_error"] > 2.5:
        patterns.append({
            "type": "high_error",
            "description": f"MAE {latest_slate['mean_absolute_error']} above 2.5 — model may be over-projecting",
            "slates_observed": 1,
        })
    if latest_slate and latest_slate.get("over_projected", 0) > latest_slate.get("under_projected", 0) * 1.5:
        patterns.append({
            "type": "systematic_over_projection",
            "description": f"Over-projecting {latest_slate['over_projected']} vs under-projecting {latest_slate['under_projected']} players — scores running lower than model expects",
            "slates_observed": 1,
        })

    # Determine the most recent date that has predictions but no actuals yet —
    # this is the date the user should upload actuals for.
    # IMPORTANT: exclude today — games may still be in progress and any upload
    # would be yesterday's screenshots misfiled under today's date.
    today_iso = _et_date().isoformat()
    pred_dates = sorted(
        [i["name"].replace(".csv","") for i in pred_items if i["name"].endswith(".csv")],
        reverse=True
    )
    pending_upload_date = next((d for d in pred_dates if d not in act_dates and d != today_iso), None)

    # Check for ownership calibration data
    try:
        own_items = _github_list_dir("data/ownership") or []
        own_dates = sorted(
            [i["name"].replace(".csv", "") for i in own_items if i["name"].endswith(".csv")],
            reverse=True,
        )
    except Exception:
        own_dates = []

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
        "pending_upload_date": pending_upload_date,
        "ownership_calibration_available": len(own_dates) > 0,
        "ownership_dates": own_dates[:5],
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
    # Block config changes during active slate — picks are frozen until all games finish
    try:
        _uc_games = fetch_games()
        _uc_starts = [g["startTime"] for g in _uc_games if g.get("startTime")]
        if _uc_starts and any(_is_locked(st) for st in _uc_starts):
            return JSONResponse({"error": "Slate is active — config changes are locked until all games finish"}, status_code=423)
    except Exception:
        pass  # if games check fails, allow the write
    # Security: reject keys with non-alphanumeric path segments (prevents path traversal)
    import re as _re
    for key in changes:
        if not _re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*([.][a-zA-Z_][a-zA-Z0-9_]*)*$', str(key)):
            return JSONResponse({"error": f"Invalid key format: {key!r}"}, status_code=400)

    cfg = _load_config()
    old_version = cfg.get("version", 1)
    new_version = old_version + 1

    for dot_path, value in changes.items():
        keys = dot_path.split(".")
        _deep_set(cfg, keys, value)

    cfg["version"]    = new_version
    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    cfg["updated_by"] = "lab"

    # Snapshot the pre-change param values for the keys being modified,
    # so rollback can restore them without needing git history.
    snapshot = {}
    for dot_path in changes:
        keys = dot_path.split(".")
        v = cfg
        try:
            for k in keys:
                v = v[k]
            snapshot[dot_path] = v
        except (KeyError, TypeError):
            pass  # key didn't exist before — rollback will just unset it

    changelog = cfg.get("changelog", [])
    changelog.append({
        "version":  new_version,
        "date":     _et_date().isoformat(),
        "change":   description,
        "snapshot": snapshot,   # previous values — used by lab/rollback
    })
    cfg["changelog"] = changelog

    content = json.dumps(cfg, indent=2)
    # Note: _github_write_file now handles 422 conflicts with exponential backoff retry.
    # If another concurrent request modified config, this write will retry with fresh SHA.
    # Changelog is built from config at this moment; if multiple writes race, the one that
    # succeeds retains its changelog entry, others are retried and see the updated config.
    result  = _github_write_file("data/model-config.json", content, f"Lab config v{new_version}: {description}")
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=500)

    # Clear config cache so new values take effect immediately
    try:
        (CONFIG_CACHE_DIR / "model_config.json").unlink(missing_ok=True)
    except Exception: pass
    # Bust slate cache so next request regenerates with new config params
    try:
        _bust_slate_cache()
    except Exception: pass

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
    # Block rollbacks during active slate — same lock guard as update-config
    try:
        _rb_games = fetch_games()
        _rb_starts = [g["startTime"] for g in _rb_games if g.get("startTime")]
        if _rb_starts and any(_is_locked(st) for st in _rb_starts):
            return JSONResponse({"error": "Slate is active — config changes are locked until all games finish"}, status_code=423)
    except Exception:
        pass

    cfg = _load_config()
    changelog = cfg.get("changelog", [])
    current_version = cfg.get("version", 1)
    if int(target) >= current_version:
        return JSONResponse({"error": "Target must be earlier than current version"}, status_code=400)

    # Find the changelog entry FOR the target version — its snapshot holds the
    # pre-change values that were overwritten when that version was applied.
    # We want to restore those values (i.e., what existed at v{target-1}).
    target_entry = next((e for e in changelog if e.get("version") == int(target) + 1), None)
    snapshot = target_entry.get("snapshot") if target_entry else None

    if not snapshot:
        return JSONResponse({
            "error": (
                f"No snapshot available for v{target} — it predates snapshot tracking. "
                "Manually apply the previous values via /api/lab/update-config."
            )
        }, status_code=400)

    # Apply the snapshot (restores the values that were in effect before v{target} was applied)
    for dot_path, value in snapshot.items():
        keys = dot_path.split(".")
        _deep_set(cfg, keys, value)

    new_version = current_version + 1
    cfg["version"]    = new_version
    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    cfg["updated_by"] = "lab-rollback"
    changelog.append({
        "version":  new_version,
        "date":     _et_date().isoformat(),
        "change":   f"Rollback to v{target}: restored {list(snapshot.keys())}",
        "snapshot": {k: cfg_val for k, cfg_val in snapshot.items()},
    })
    cfg["changelog"] = changelog

    content = json.dumps(cfg, indent=2)
    result  = _github_write_file("data/model-config.json", content, f"Lab rollback to v{target}")
    if result.get("error"):
        return JSONResponse({"error": result["error"]}, status_code=500)

    try:
        (CONFIG_CACHE_DIR / "model_config.json").unlink(missing_ok=True)
    except Exception: pass
    try:
        _bust_slate_cache()
    except Exception: pass

    return {"status": "rolled_back", "new_version": new_version,
            "restored": snapshot, "from_version": target}


@app.post("/api/lab/backtest")
async def lab_backtest(payload: dict = Body(...)):
    """Replay historical slates with proposed parameter changes and compare MAE.

    Safety: Backtest is limited to last 10 historical slates and will timeout
    after 240 seconds to stay within Railway's timeout limit."""
    proposed_changes = payload.get("proposed_changes", {})
    description      = payload.get("description", "Backtest")
    if not proposed_changes:
        return JSONResponse({"error": "proposed_changes required"}, status_code=400)

    # Card boost and other non-RS params don't affect predicted_rs — they control
    # lineup selection EV (which players get picked), not projection accuracy.
    # MAE backtest is the wrong tool here; return an explanatory response instead.
    rs_params = [k for k in proposed_changes if k.startswith("real_score")]
    non_rs_only = rs_params == [] and proposed_changes
    card_boost_only = all(k.startswith("card_boost") or k.startswith("moonshot")
                          or k.startswith("projection") for k in proposed_changes)
    if non_rs_only or (card_boost_only and not rs_params):
        affected = list(proposed_changes.keys())
        return {
            "slates_tested": 0,
            "note": (
                f"MAE backtest does not apply to {', '.join(affected)}. "
                "These params control lineup selection EV (which players get chosen), "
                "not RS projection accuracy — changing them won't move MAE. "
                "Apply the change and evaluate results from the next slate instead."
            ),
            "recommendation": "Apply and evaluate on next slate — MAE backtest not applicable here.",
        }

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
                    {"pts":2.5,"reb":0.5,"ast":1.0,"stl":2.0,"blk":1.5,"tov":-1.5})
                new_dfs = (pts*w.get("pts",2.5) + reb*w.get("reb",0.5) +
                           ast*w.get("ast",1.0) + stl*w.get("stl",2.0) +
                           blk*w.get("blk",1.5))
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


@app.get("/api/lab/auto-improve")
async def lab_auto_improve(request: Request):
    """
    Autonomous model improvement cron endpoint.
    1. Fetches briefing data to assess current accuracy
    2. If MAE > 2.0 or patterns detected, asks Claude Haiku to propose config changes
    3. Backtests proposed changes against historical data (limited to 10 slates)
    4. Auto-applies if improvement >= 3%
    5. Returns a log of actions taken (safe to run even if no data yet)

    Safety: Runs at 9 AM UTC daily. Backtest internally limited to 10 slates
    and will timeout after 240s to prevent exceeding Railway's timeout limit.
    If timeout occurs, returns error log without auto-applying.
    """
    if not _require_cron_secret(request):
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    if not ANTHROPIC_API_KEY:
        return {"status": "skipped", "reason": "ANTHROPIC_API_KEY not configured"}

    # Step 1: Get briefing
    try:
        briefing = await lab_briefing()
    except Exception as e:
        return {"status": "error", "reason": f"Briefing failed: {e}"}

    latest = briefing.get("latest_slate")
    patterns = briefing.get("patterns", [])
    rolling = briefing.get("rolling_accuracy", {})
    current_cfg = briefing.get("current_config", {})

    # Skip if no data or accuracy is already good
    mae = latest.get("mean_absolute_error") if latest else None
    overall_mae = rolling.get("overall_mae")
    effective_mae = mae or overall_mae

    action_log = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "latest_mae": mae,
        "overall_mae": overall_mae,
        "patterns_detected": len(patterns),
    }

    if effective_mae is None:
        return {"status": "skipped", "reason": "No historical data yet", "log": action_log}

    if effective_mae <= 1.8 and len(patterns) == 0:
        action_log["decision"] = f"MAE {effective_mae} is good — no changes needed"
        return {"status": "skipped", "reason": "Model accuracy is satisfactory", "log": action_log}

    # Step 2: Ask Claude Haiku to propose changes
    cfg = _load_config()
    dfs_weights = cfg.get("real_score", {}).get("dfs_weights", {})
    card_boost_cfg = cfg.get("card_boost", {})

    haiku_prompt = f"""You are an NBA fantasy model optimizer. Analyze the accuracy data and propose ONE specific parameter change.

CURRENT ACCURACY:
- Latest slate MAE: {mae}
- Overall rolling MAE: {overall_mae}
- Over-projected players: {latest.get('over_projected', 'N/A') if latest else 'N/A'}
- Under-projected players: {latest.get('under_projected', 'N/A') if latest else 'N/A'}
- Patterns: {json.dumps(patterns)}

CURRENT CONFIG:
- DFS weights: {json.dumps(dfs_weights)}
- Card boost scalar: {card_boost_cfg.get('scalar', 'N/A')}
- Card boost decay: {card_boost_cfg.get('decay_base', 'N/A')}
- Model version: {current_cfg.get('version', 1)}

Biggest misses from latest slate: {json.dumps(latest.get('biggest_misses', []) if latest else [])}

RULES:
- Propose exactly ONE change to ONE parameter using dot notation
- Valid paths: real_score.dfs_weights.pts, real_score.dfs_weights.reb, real_score.dfs_weights.ast, real_score.dfs_weights.stl, real_score.dfs_weights.blk, card_boost.scalar, card_boost.decay_base
- Keep changes small: ±5-10% max from current values
- If over-projecting consistently, suggest reducing a weight. If under-projecting, increase.
- Respond ONLY with valid JSON in this exact format:
{{"dot_path": "real_score.dfs_weights.pts", "new_value": 1.05, "reasoning": "Brief reason"}}"""

    try:
        haiku_resp = requests.post(
            f"{ANTHROPIC_API_BASE}/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": HAIKU_MODEL,
                "max_tokens": 256,
                "messages": [{"role": "user", "content": haiku_prompt}],
            },
            timeout=30,
        )
        haiku_resp.raise_for_status()
        haiku_data = haiku_resp.json()
        raw_text = next(
            (b["text"] for b in haiku_data.get("content", []) if b.get("type") == "text"), ""
        ).strip()

        # Parse JSON from response (may be wrapped in ```json ... ```)
        if "```" in raw_text:
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
        proposal = json.loads(raw_text.strip())
        dot_path = proposal["dot_path"]
        new_value = proposal["new_value"]
        reasoning = proposal.get("reasoning", "Auto-improve suggestion")

    except Exception as e:
        action_log["decision"] = f"Claude proposal failed: {e}"
        return {"status": "error", "reason": f"Haiku proposal error: {e}", "log": action_log}

    action_log["proposed_change"] = {dot_path: new_value}
    action_log["reasoning"] = reasoning

    # Step 3: Backtest the proposed change
    try:
        backtest_payload = {
            "proposed_changes": {dot_path: new_value},
            "description": f"Auto-improve: {reasoning}",
        }
        backtest_result = await lab_backtest(backtest_payload)
    except Exception as e:
        action_log["decision"] = f"Backtest failed: {e}"
        return {"status": "error", "reason": f"Backtest error: {e}", "log": action_log}

    improvement_pct = backtest_result.get("improvement_pct", 0)
    current_mae_bt = backtest_result.get("current_config_mae")
    proposed_mae_bt = backtest_result.get("proposed_config_mae")

    action_log["backtest"] = {
        "current_mae": current_mae_bt,
        "proposed_mae": proposed_mae_bt,
        "improvement_pct": improvement_pct,
    }

    # Step 4: Apply if improvement >= threshold (tunable via lab.auto_improve_threshold_pct)
    IMPROVEMENT_THRESHOLD = _cfg("lab.auto_improve_threshold_pct", 3.0)
    if improvement_pct >= IMPROVEMENT_THRESHOLD:
        try:
            update_payload = {
                "changes": {dot_path: new_value},
                "change_description": f"[auto-improve] {reasoning} (backtest: {improvement_pct:.1f}% MAE improvement)",
            }
            update_result = await lab_update_config(update_payload)
            action_log["decision"] = "applied"
            action_log["new_version"] = update_result.get("version")
            return {
                "status": "applied",
                "change": {dot_path: new_value},
                "improvement_pct": improvement_pct,
                "reasoning": reasoning,
                "log": action_log,
            }
        except Exception as e:
            action_log["decision"] = f"Apply failed: {e}"
            return {"status": "error", "reason": f"Config update error: {e}", "log": action_log}
    else:
        action_log["decision"] = f"Improvement {improvement_pct:.1f}% below threshold {IMPROVEMENT_THRESHOLD}% — not applied"
        return {
            "status": "no_change",
            "reason": f"Backtest improvement {improvement_pct:.1f}% below {IMPROVEMENT_THRESHOLD}% threshold",
            "proposed_change": {dot_path: new_value},
            "reasoning": reasoning,
            "log": action_log,
        }


def _tool_status_label(name: str, inputs: dict) -> str:
    """Human-readable status label for a Ben tool call."""
    if name == "get_live_nba_data":
        dt = inputs.get("data_type", "")
        if dt == "scores":
            return "Checking live scores..."
        if dt == "boxscore":
            return "Reading box score..."
        if dt == "player_stats":
            player = inputs.get("player_name", "player")
            return f"Looking up {player} stats..."
    return "Fetching data..."


# ═════════════════════════════════════════════════════════════════════════════
# BEN (Lab) chat history persistence
# ═════════════════════════════════════════════════════════════════════════════
_BEN_CHAT_HISTORY_PATH = Path("data/ben_chat_history.json")
_BEN_CHAT_HISTORY_DATE_PATH = Path("data/ben_chat_history_last_et_date.json")
_BEN_CHAT_HISTORY_LOCK = threading.Lock()


def _ben_chat_extract_last_user_text(msgs: list) -> str:
    """Extract the latest user message text (ignoring any attached image blocks)."""
    if not msgs or not isinstance(msgs, list):
        return ""
    for m in reversed(msgs):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c.strip()
        if isinstance(c, list):
            parts = []
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text", "")
                    if t:
                        parts.append(str(t).strip())
            return "\n".join([p for p in parts if p]).strip()
        if c is None:
            return ""
        return str(c).strip()
    return ""


def _ben_chat_sanitize_assistant_text(text: str) -> str:
    """Store only user-visible assistant text (strip <action> tags)."""
    if not text:
        return ""
    return re.sub(r"<action>[\s\S]*?</action>", "", str(text)).strip()


def _ben_chat_maybe_reset_for_today_locked() -> None:
    """Reset chat history when ET day changes."""
    today = _et_date().isoformat()

    last_et = ""
    try:
        if _BEN_CHAT_HISTORY_DATE_PATH.exists():
            raw = _BEN_CHAT_HISTORY_DATE_PATH.read_text()
            obj = json.loads(raw) if raw else {}
            last_et = (obj or {}).get("et_date", "") or ""
    except Exception:
        last_et = ""

    if last_et == today:
        return

    try:
        _BEN_CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BEN_CHAT_HISTORY_PATH.write_text("[]")
        _BEN_CHAT_HISTORY_DATE_PATH.write_text(json.dumps({"et_date": today}, indent=2))
    except Exception as e:
        print(f"[ben-chat] reset failed: {e}")


def _ben_chat_read_history_locked() -> list:
    _ben_chat_maybe_reset_for_today_locked()
    try:
        if not _BEN_CHAT_HISTORY_PATH.exists():
            return []
        raw = _BEN_CHAT_HISTORY_PATH.read_text()
        data = json.loads(raw) if raw else []
        if not isinstance(data, list):
            return []
        # Trim trailing user messages — these are orphaned when a previous API call
        # failed after the user message was persisted but before the assistant responded.
        # Sending consecutive user messages causes a 400 from Anthropic on the next call.
        while data and data[-1].get("role") == "user":
            data.pop()
        return data
    except Exception:
        return []


def _ben_chat_write_history_locked(messages: list) -> None:
    try:
        _BEN_CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BEN_CHAT_HISTORY_PATH.write_text(json.dumps(messages, indent=2))
    except Exception as e:
        print(f"[ben-chat] write failed: {e}")


def _ben_chat_append_message(role: str, content: str) -> None:
    if role not in ("user", "assistant"):
        return
    content = str(content or "").strip()
    if not content:
        return
    with _BEN_CHAT_HISTORY_LOCK:
        history = _ben_chat_read_history_locked()
        history.append({"role": role, "content": content})
        _ben_chat_write_history_locked(history)


@app.get("/api/lab/chat-history")
async def lab_chat_history():
    """Return the persisted daily Ben chat history as an array [{role,content}, ...]."""
    with _BEN_CHAT_HISTORY_LOCK:
        history = _ben_chat_read_history_locked()
    return history


@app.post("/api/lab/chat")
async def lab_chat(request: Request, payload: dict = Body(...)):
    """Proxy to Anthropic Messages API — streams SSE events so the UI can show live status."""
    rl = _check_rate_limit(request, "lab/chat")
    if rl is not None:
        return rl
    if not ANTHROPIC_API_KEY:
        return JSONResponse({"error": "ANTHROPIC_API_KEY not configured"}, status_code=500)

    messages = payload.get("messages", [])
    system   = payload.get("system", "")

    if not messages:
        return JSONResponse({"error": "No messages provided"}, status_code=400)

    # Persist the latest user message before sending the Claude payload.
    try:
        last_user_text = _ben_chat_extract_last_user_text(messages)
        if last_user_text:
            _ben_chat_append_message("user", last_user_text)
    except Exception as e:
        # Non-fatal: the chat should still work even if persistence fails.
        print(f"[ben-chat] append user failed: {e}")

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    base_body = {
        "model":      OPUS_MODEL,
        "max_tokens": 2048,
        "system":     system,
        "tools":      _BEN_TOOLS,
        "messages":   messages,
    }

    def _sse(obj) -> str:
        return f"data: {json.dumps(obj)}\n\n"

    def event_stream():
        try:
            r = requests.post(f"{ANTHROPIC_API_BASE}/messages",
                              headers=headers, json=base_body, timeout=45)
            r.raise_for_status()
            resp = r.json()

            # Handle up to 5 rounds of tool use — Claude often chains multiple
            # read_repo_file calls (e.g. read actuals → read predictions → respond).
            # Without looping, the second tool_use response has no text block and
            # renders as an empty bubble in the UI.
            current_messages = list(messages)
            for _round in range(5):
                if resp.get("stop_reason") != "tool_use":
                    break
                tool_results = []
                for block in resp.get("content", []):
                    if block.get("type") == "tool_use":
                        status = _tool_status_label(block["name"], block.get("input", {}))
                        yield _sse({"type": "status", "text": status})
                        result = _execute_ben_tool(block["name"], block.get("input", {}))
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block["id"],
                            "content":     result,
                        })
                current_messages = current_messages + [
                    {"role": "assistant", "content": resp["content"]},
                    {"role": "user",      "content": tool_results},
                ]
                r_next = requests.post(
                    f"{ANTHROPIC_API_BASE}/messages",
                    headers=headers,
                    json={**base_body, "messages": current_messages},
                    timeout=45,
                )
                r_next.raise_for_status()
                resp = r_next.json()

            text = next((b["text"] for b in resp.get("content", []) if b.get("type") == "text"), "")
            # Persist the assistant response (strip any <action> tags) before yielding to the client.
            try:
                safe_text = _ben_chat_sanitize_assistant_text(text)
                if safe_text:
                    _ben_chat_append_message("assistant", safe_text)
            except Exception as e:
                print(f"[ben-chat] append assistant failed: {e}")

            yield _sse({"type": "content", "text": text})

        except Exception as e:
            err_body = ""
            if hasattr(e, "response") and e.response is not None:
                try:
                    err_body = e.response.text[:400]
                except Exception:
                    pass
            error_msg = f"Anthropic API error: {str(e)}"
            if err_body:
                error_msg += f" — {err_body}"
            yield _sse({"type": "content", "error": error_msg, "text": ""})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/lab/skip-uploads")
async def lab_skip_uploads(payload: dict = Body(...)):
    """Mark a specific date's screenshots as skipped (user doesn't want to upload).

    Does NOT process the screenshots, does NOT store them, does NOT affect learning.
    Simply records in GitHub that this date was skipped by the user.

    Frontend calls this when user clicks "Skip All Uploads" button on the banner.
    Stores indication in data/skipped-uploads.json for audit purposes.
    """
    date_str = payload.get("date", "").strip()
    if not date_str:
        return JSONResponse({"error": "date required"}, status_code=400)
    bad = _validate_date(date_str)
    if bad: return bad

    try:
        # Load existing skipped dates
        skipped_file = "data/skipped-uploads.json"
        skipped_content, _ = _github_get_file(skipped_file)
        skipped_data = json.loads(skipped_content) if skipped_content else {"skipped_dates": []}

        # Add this date if not already present
        if date_str not in skipped_data.get("skipped_dates", []):
            skipped_data.setdefault("skipped_dates", []).append(date_str)
            skipped_data["last_skipped_at"] = datetime.now(timezone.utc).isoformat()

            # Write back to GitHub
            _github_write_file(skipped_file, json.dumps(skipped_data, indent=2),
                              f"Skip uploads for {date_str}")

        return {"status": "skipped", "date": date_str}

    except Exception as e:
        # Non-critical — don't fail if we can't record skip
        print(f"[skip-uploads] Error recording skip for {date_str}: {e}")
        return {"status": "recorded_locally", "date": date_str}


# grep: BOOST INGESTION ENDPOINT
@app.post("/api/save-boosts")
async def save_boosts(payload: dict = Body(...)):
    """Save pre-game player boosts for today's slate.

    Boosts are fixed daily constants published by Real Sports before drafts open.
    This endpoint stores them as ground truth — the pipeline uses these directly
    instead of estimating. Busts the slate cache so the next /api/slate call
    regenerates with real boosts.

    Body: { date: str (optional), players: [{player_name, boost, team?, rax_cost?}] }
    """
    global _DAILY_BOOST_CACHE, _DAILY_BOOST_DATE, _DAILY_BOOST_TS
    date_str = payload.get("date", _et_date().isoformat())
    bad = _validate_date(date_str)
    if bad: return bad

    players = payload.get("players", [])
    if not players:
        return {"saved": 0, "date": date_str}

    ts = datetime.now(timezone.utc).isoformat()
    clean_players = []
    for p in players:
        name = str(p.get("player_name") or p.get("name", "")).strip()
        boost = _safe_float(p.get("boost") or p.get("actual_card_boost"))
        if not name or boost is None:
            continue
        entry = {"player_name": name, "boost": round(float(boost), 1)}
        if p.get("team"):
            entry["team"] = str(p["team"]).upper()
        if p.get("rax_cost") is not None:
            entry["rax_cost"] = _safe_float(p["rax_cost"])
        clean_players.append(entry)

    if not clean_players:
        return {"saved": 0, "date": date_str}

    data = {
        "date": date_str,
        "saved_at": ts,
        "player_count": len(clean_players),
        "players": clean_players,
    }
    content = json.dumps(data, indent=2)
    path = f"data/boosts/{date_str}.json"
    try:
        _github_write_file(path, content, f"pre-game boosts {date_str} ({len(clean_players)} players)")
    except Exception as e:
        print(f"[save-boosts] GitHub write failed: {e}")
        return JSONResponse({"error": f"Failed to save boosts: {str(e)}"}, status_code=500)

    # Bust caches so next slate request uses real boosts
    _DAILY_BOOST_CACHE = {_normalize_boost_name(p["player_name"]): p["boost"] for p in clean_players}
    _DAILY_BOOST_DATE = date_str
    _DAILY_BOOST_TS = time.time()
    _bust_slate_cache()

    return {"saved": len(clean_players), "date": date_str, "cache_busted": True}


@app.post("/api/save-ownership")
async def save_ownership(payload: dict = Body(...)):
    """Save parsed Most Drafted / ownership data to GitHub as CSV.

    Stores actual draft counts and card boosts per player for a given date.
    Used by the calibrate-boost endpoint to build the player_overrides lookup
    against real ownership data.

    Body: { date: str, players: [{player_name|name, team, draft_count, actual_rs,
                                   actual_card_boost, avg_finish, rank}] }
    """
    date_str = payload.get("date", _et_date().isoformat())
    bad = _validate_date(date_str)
    if bad: return bad

    players = payload.get("players", [])
    if not players:
        return {"saved": 0, "date": date_str}

    rows = ["player,team,draft_count,actual_rs,actual_card_boost,avg_finish,rank,saved_at"]
    ts = datetime.now(timezone.utc).isoformat()
    saved = 0
    for p in players:
        name = str(p.get("player_name") or p.get("name", "")).strip().replace(",", " ")
        if not name:
            continue
        team           = str(p.get("team") or "").upper().replace(",", "")
        draft_count    = _safe_float(p.get("draft_count")) or 0
        actual_rs      = _safe_float(p.get("actual_rs"))
        actual_boost   = _safe_float(p.get("actual_card_boost"))
        avg_finish     = _safe_float(p.get("avg_finish"))
        rank           = int(p.get("rank") or 0)
        rows.append(
            f"{name},{team},{int(draft_count)},"
            f"{actual_rs if actual_rs is not None else ''},"
            f"{actual_boost if actual_boost is not None else ''},"
            f"{avg_finish if avg_finish is not None else ''},"
            f"{rank},{ts}"
        )
        saved += 1

    if saved == 0:
        return {"saved": 0, "date": date_str}

    content = "\n".join(rows) + "\n"
    path    = f"data/ownership/{date_str}.csv"
    try:
        _github_write_file(path, content, f"ownership data {date_str} ({saved} players)")
    except Exception as e:
        print(f"[save-ownership] GitHub write failed: {e}")
        return JSONResponse({"error": f"Failed to save ownership data: {str(e)}"}, status_code=500)

    return {"saved": saved, "date": date_str}


@app.get("/api/lab/calibrate-boost")
async def lab_calibrate_boost():
    """Build player_overrides lookup from real ownership + actuals data.

    Reads data/ownership/ and data/actuals/ CSVs to collect per-player boost
    values. Averages across dates (boost is stable per-player +-0.1x).

    Returns proposed player_overrides dict. Does NOT auto-apply.
    To apply: POST /api/lab/update-config with card_boost.player_overrides dict.
    """
    try:
        from collections import defaultdict

        ceiling = _cfg("card_boost.ceiling", 3.0)
        floor_  = _cfg("card_boost.floor", 0.2)
        current_overrides = _cfg("card_boost.player_overrides", {})

        player_boosts = defaultdict(list)
        dates_used = []

        # Collect from ownership CSVs
        own_items = _github_list_dir("data/ownership") or []
        own_dates = sorted(
            [i["name"].replace(".csv", "") for i in own_items if i["name"].endswith(".csv")],
            reverse=True,
        )
        for date_str in own_dates:
            own_csv, _ = _github_get_file(f"data/ownership/{date_str}.csv")
            if not own_csv:
                continue
            own_rows = _parse_csv(own_csv, ["player", "team", "draft_count", "actual_rs",
                                             "actual_card_boost", "avg_finish", "rank", "saved_at"])
            added = 0
            for row in own_rows:
                boost = _safe_float(row.get("actual_card_boost"))
                name  = row.get("player", "").strip()
                if not name or boost is None or boost < 0:
                    continue
                player_boosts[name].append(min(ceiling, max(floor_, boost)))
                added += 1
            if added:
                dates_used.append(f"ownership/{date_str}")

        # Collect from actuals CSVs
        act_items = _github_list_dir("data/actuals") or []
        act_dates = sorted(
            [i["name"].replace(".csv", "") for i in act_items if i["name"].endswith(".csv")],
            reverse=True,
        )
        for date_str in act_dates:
            act_csv, _ = _github_get_file(f"data/actuals/{date_str}.csv")
            if not act_csv:
                continue
            act_rows = _parse_csv(act_csv, ["player_name", "actual_rs", "actual_card_boost",
                                             "drafts", "avg_finish", "total_value", "source"])
            added = 0
            for row in act_rows:
                boost = _safe_float(row.get("actual_card_boost"))
                name  = row.get("player_name", "").strip()
                if not name or boost is None or boost < 0:
                    continue
                player_boosts[name].append(min(ceiling, max(floor_, boost)))
                added += 1
            if added:
                dates_used.append(f"actuals/{date_str}")

        n_players = len(player_boosts)
        n_samples = sum(len(v) for v in player_boosts.values())

        if n_players < 4:
            return {
                "error": f"Not enough players (found {n_players}, need >= 4). "
                         "Upload more Most Drafted or Real Scores screenshots.",
                "n_players": n_players,
                "dates_with_data": dates_used,
            }

        # Average boost per player (stable +-0.1x across dates)
        proposed = {}
        for name, boosts in sorted(player_boosts.items()):
            proposed[name] = round(sum(boosts) / len(boosts), 1)

        # Count new/changed players vs current overrides
        new_players = [n for n in proposed if n not in current_overrides]
        changed_players = [n for n in proposed if n in current_overrides
                           and abs(proposed[n] - current_overrides[n]) >= 0.1]

        return {
            "current_count":   len(current_overrides),
            "proposed_count":  n_players,
            "proposed":        proposed,
            "n_samples":       n_samples,
            "dates_used":      dates_used,
            "new_players":     new_players,
            "changed_players": changed_players,
            "note": "To apply: POST /api/lab/update-config with card_boost.player_overrides dict",
        }

    except Exception as e:
        print(f"[calibrate-boost] Error: {e}")
        return {"error": str(e)}
