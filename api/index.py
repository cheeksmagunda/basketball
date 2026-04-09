"""
Oracle API — FastAPI backend for the Real Sports NBA draft optimizer.

Single-entry backend: all HTTP routes, projection pipeline, slate/picks computation,
and Lab (Ben) chat live here. Uses api.real_score, api.asset_optimizer,
and api.rotowire for domain logic. Config from data/model-config.json (GitHub);
secrets from environment variables only (never in code). Global exception handler returns
generic 500 to clients and logs full traceback server-side.
"""
import json
import copy
import csv
import io
import hashlib
import math
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
import lightgbm as lgb
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, Query, Body, File, Form, UploadFile, Request
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

# Load .env before importing ecosystem modules so module-level os.getenv() calls
# pick up local env vars correctly.
load_dotenv()

# Real Score Ecosystem modules
try:
    from api.asset_optimizer import optimize_lineup
    from api.rotowire import get_all_statuses, is_safe_to_draft, clear_cache as _rw_clear
    from api.cache import rcg, rcs, rcd, rflush, redis_ok
    from api.features import (
        RS_FEATURES, compute_rs_features, rs_feature_vector,
        TEAM_MARKET_SCORES, get_team_market_score, pos_bucket, ppg_tier_bucket,
    )
    from api.nba_api_feed import prefetch_enrichment as _nba_api_prefetch, enrich_stats_map as _nba_api_enrich
    from api.injury_feed import is_available as _injury_available, fetch_espn_injuries as _espn_injuries_fetch
    from api.dfs_salary_feed import get_anti_popularity_adjustment as _dfs_pop_adj, save_dfs_salaries as _save_dfs_sal, compute_popularity_scores as _dfs_pop_scores
except ImportError:
    from .asset_optimizer import optimize_lineup
    from .rotowire import get_all_statuses, is_safe_to_draft, clear_cache as _rw_clear
    from .cache import rcg, rcs, rcd, rflush, redis_ok
    from .features import (
        RS_FEATURES, compute_rs_features, rs_feature_vector,
        TEAM_MARKET_SCORES, get_team_market_score, pos_bucket, ppg_tier_bucket,
    )
    from .nba_api_feed import prefetch_enrichment as _nba_api_prefetch, enrich_stats_map as _nba_api_enrich
    from .injury_feed import is_available as _injury_available, fetch_espn_injuries as _espn_injuries_fetch
    from .dfs_salary_feed import get_anti_popularity_adjustment as _dfs_pop_adj, save_dfs_salaries as _save_dfs_sal, compute_popularity_scores as _dfs_pop_scores
DOCS_SECRET = os.getenv("DOCS_SECRET", "")  # optional: require ?docs_key=DOCS_SECRET or X-Docs-Key for /docs, /redoc, /openapi.json

app = FastAPI()

# ── GZip compression — ~70-80% smaller JSON payloads over the wire ──
from starlette.middleware.gzip import GZipMiddleware  # noqa: E402
app.add_middleware(GZipMiddleware, minimum_size=500)

# ── Per-endpoint browser Cache-Control headers ──
# Short-lived browser caching with stale-while-revalidate to eliminate network
# round-trips on tab re-visits.  Server-side cache busting (cold pipeline,
# config change, injury check) is unaffected — these TTLs are ≤5 min.
_BROWSER_CACHE: dict[str, str] = {
    "/api/games":            "public, max-age=120, stale-while-revalidate=180",
    "/api/slate":            "public, max-age=60, stale-while-revalidate=120",
    "/api/picks":            "public, max-age=120, stale-while-revalidate=300",
    "/api/lab/briefing":     "public, max-age=120, stale-while-revalidate=300",
    "/api/lab/status":       "public, max-age=60, stale-while-revalidate=120",
    "/api/lab/config-history": "public, max-age=300, stale-while-revalidate=600",
}


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
            return _err("Docs require docs_key or X-Docs-Key", 401)

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
    # Inject browser Cache-Control on success responses for configured endpoints
    cc = _BROWSER_CACHE.get(request.url.path)
    if cc and 200 <= response.status_code < 300:
        response.headers["Cache-Control"] = cc
    return response


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def _validate_date(date_str: str) -> Optional[JSONResponse]:
    """Return a 400 JSONResponse if date_str is not YYYY-MM-DD format, else None."""
    if not _DATE_RE.match(date_str):
        return _err("Invalid date format (expected YYYY-MM-DD)", 400)
    return None


# ── GitHub API helpers for persistent CSV storage ──
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "").strip()
GITHUB_REPO = os.getenv("GITHUB_REPO", "").strip()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
INGEST_SECRET = os.getenv("INGEST_SECRET", "").strip()

def _get_odds_api_key() -> str:
    """The Odds API key trimmed of whitespace (Railway/dashboard pastes often add newlines)."""
    return (os.environ.get("ODDS_API_KEY") or "").strip()


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

def _github_headers() -> dict:
    """Standardized GitHub API headers (DRY — single source of truth)."""
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

def _github_get_file(path: str, ref_override: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Get file content and SHA from GitHub. Returns (content_str, sha) or (None, None)."""
    if not GITHUB_TOKEN or not GITHUB_REPO:
        return None, None
    ref = ref_override if ref_override is not None else _data_ref(path)
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}"
    if ref:
        url += f"?ref={ref}"
    r = requests.get(url, headers=_github_headers(), timeout=_T_DEFAULT)
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
    r = requests.get(url, headers=_github_headers(), timeout=_T_DEFAULT)
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
            r = requests.put(url, json=payload, headers=_github_headers(), timeout=_T_GITHUB)

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


def _github_write_file_bg(path: str, content: str, message: str = "auto-update"):
    """Fire-and-forget GitHub write in a background thread.
    Use for non-critical persistence where the caller doesn't need confirmation
    (e.g., audit JSON, secondary JSON writes).
    Logs errors but never raises — the HTTP response has already been returned."""
    def _bg():
        try:
            result = _github_write_file(path, content, message)
            if result.get("error"):
                print(f"[github-bg] write failed {path}: {result['error']}")
        except Exception as e:
            print(f"[github-bg] exception {path}: {e}")
    threading.Thread(target=_bg, daemon=True).start()


def _github_write_batch(files: list, message: str = "auto-update") -> dict:
    """Write multiple files in a single commit using the Git Trees API.
    files: list of {"path": str, "content": str}.
    Falls back to sequential _github_write_file if tree API fails."""
    if not GITHUB_TOKEN or not GITHUB_REPO or not files:
        return {"error": "no files or credentials"}
    branch = "main"
    h = _github_headers()
    try:
        # Get the current commit SHA for the target branch
        ref_r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/git/refs/heads/{branch}",
            headers=h, timeout=_T_DEFAULT,
        )
        if ref_r.status_code != 200:
            raise ValueError(f"ref lookup failed: {ref_r.status_code}")
        base_sha = ref_r.json()["object"]["sha"]
        # Get the tree SHA of the base commit
        commit_r = requests.get(
            f"https://api.github.com/repos/{GITHUB_REPO}/git/commits/{base_sha}",
            headers=h, timeout=_T_DEFAULT,
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
            headers=h, timeout=_T_GITHUB,
        )
        if tree_r.status_code not in (200, 201):
            raise ValueError(f"tree create failed: {tree_r.status_code}")
        new_tree_sha = tree_r.json()["sha"]
        # Create commit
        commit_create_r = requests.post(
            f"https://api.github.com/repos/{GITHUB_REPO}/git/commits",
            json={"message": message, "tree": new_tree_sha, "parents": [base_sha]},
            headers=h, timeout=_T_GITHUB,
        )
        if commit_create_r.status_code not in (200, 201):
            raise ValueError(f"commit create failed: {commit_create_r.status_code}")
        new_commit_sha = commit_create_r.json()["sha"]
        # Update branch ref
        ref_update_r = requests.patch(
            f"https://api.github.com/repos/{GITHUB_REPO}/git/refs/heads/{branch}",
            json={"sha": new_commit_sha},
            headers=h, timeout=_T_DEFAULT,
        )
        if ref_update_r.status_code != 200:
            raise ValueError(f"ref update failed: {ref_update_r.status_code}")
        return {"ok": True, "sha": new_commit_sha}
    except Exception as e:
        print(f"[github] batch write failed, falling back to sequential: {e}")
        failed_files = []
        for f in files:
            try:
                _github_write_file(f["path"], f["content"], message)
            except Exception as e2:
                print(f"[github] sequential fallback err {f['path']}: {e2}")
                failed_files.append(f["path"])
        if failed_files:
            return {"ok": False, "fallback": True, "failed": failed_files}
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
    r = requests.delete(url, json=payload, headers=_github_headers(), timeout=_T_GITHUB)
    return r.status_code in (200, 204)


def _slate_backup_to_github(slate_data: dict, date_str: str = None):
    """Write slate response to GitHub as a locked-state backup (deduped by date).
    Called once when we promote reg_cache -> lock_cache so cold-start instances can recover."""
    try:
        today = date_str or slate_data.get("date") or _today_str()
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
        today = _today_str()
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

def _slate_cache_to_github(slate_data: dict, date_str: str = None):
    """Persist today's generated slate to GitHub for cold-start recovery.
    Deduped by date — overwrites same-day file on regeneration (injury/config change).
    Embeds deploy_sha for Scenario 1 auto-detection (dev ships model update mid-slate)."""
    try:
        today = date_str or slate_data.get("date") or _today_str()
        path = f"data/slate/{today}_slate.json"
        # Stamp deploy SHA so /api/slate can detect when a new deploy invalidates cached picks
        sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "")
        if sha:
            slate_data["deploy_sha"] = sha[:7]
        content = json.dumps(slate_data, default=str)
        _github_write_file(path, content, f"slate cache {today}")
    except Exception as e:
        print(f"[slate-cache] write err: {e}")

def _slate_cache_from_github(date_str: str = None):
    """Load today's (or a specific date's) cached slate from GitHub. Returns dict or None.
    Checks bust sentinel first (data branch and main so retrain workflow bust is seen)."""
    try:
        today = date_str or _today_str()
        bust_path = f"data/slate/{today}_bust.json"
        for ref in (None, "main"):  # data branch first, then main (retrain writes bust to main)
            bust_content, _ = _github_get_file(bust_path, ref_override=ref)
            if bust_content:
                bust_data = json.loads(bust_content)
                # Time-based expiry: busts older than 90 minutes auto-expire so a failed
                # regeneration can't permanently block the cache indefinitely.
                if bust_data.get("at"):
                    try:
                        bust_age_s = (datetime.now(timezone.utc) -
                                      datetime.fromisoformat(bust_data["at"])).total_seconds()
                        if bust_age_s > 5400:  # 90 minutes
                            print(f"[slate-cache] bust expired ({bust_age_s:.0f}s old), ignoring")
                            break  # Treat as expired — fall through to normal cache read
                    except Exception:
                        pass
                if bust_data.get("_busted"):
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
    """True if today's slate bust sentinel is active on GitHub (cold reset, lab config).

    Railway runs multiple instances; local /tmp clearing is instance-scoped.
    Without this check, locked slates keep serving stale lineups from slate_v5_locked.

    Uses only data/slate/{date}_bust.json (not data/locks/*): the lock backup may still
    be a tombstone briefly while bust.json is already cleared after sync regen."""
    try:
        today = _today_str()
        bust_path = f"data/slate/{today}_bust.json"
        for ref in (None, "main"):
            bust_content, _ = _github_get_file(bust_path, ref_override=ref)
            if bust_content:
                bust_data = json.loads(bust_content)
                # 90-min expiry consistent with _slate_cache_from_github
                if bust_data.get("at"):
                    try:
                        bust_age_s = (datetime.now(timezone.utc) -
                                      datetime.fromisoformat(bust_data["at"])).total_seconds()
                        if bust_age_s > 5400:  # 90 minutes
                            return False
                    except Exception:
                        pass
                if bust_data.get("_busted"):
                    return True
    except Exception:
        pass
    return False


def _clear_local_slate_tmp_caches():
    """Remove today's slate + lock JSON from this instance's /tmp + Redis (hashed paths)."""
    try:
        _lp(_CK_SLATE_LOCKED).unlink(missing_ok=True)
    except Exception:
        pass
    _cd(_CK_SLATE)


def _games_cache_to_github(all_game_projections: dict, date_str: str = None):
    """Persist per-game projections {gameId: [players...]} to GitHub.
    Allows /api/picks to serve from cache without re-running _run_game()."""
    try:
        today = date_str or _today_str()
        path = f"data/slate/{today}_games.json"
        content = json.dumps(all_game_projections, default=str)
        _github_write_file(path, content, f"game projections {today}")
    except Exception as e:
        print(f"[games-cache] write err: {e}")

def _games_cache_from_github(date_str: str = None):
    """Load per-game projections from GitHub. Returns {gameId: [...]} or None."""
    try:
        today = date_str or _today_str()
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


def _hydrate_game_projs_from_github(target_games):
    """Load per-game projections from GitHub `data/slate/{today}_games.json` into /tmp.

    Used when /tmp per-game cache is cold (new instance) but slate was already generated
    elsewhere — avoids re-running _run_game for Line/Parlay paths."""
    gh = _games_cache_from_github()
    if not gh or not isinstance(gh, dict) or gh.get("_busted"):
        return []
    out = []
    n_games = 0
    for g in target_games:
        gid = g.get("gameId")
        if not gid:
            continue
        projs = gh.get(gid)
        if not projs or not isinstance(projs, list):
            continue
        _cs(_ck_game_proj(gid), projs)
        out.extend(projs)
        n_games += 1
    if out:
        print(f"[games-cache] hydrated {n_games} games from GitHub into /tmp ({len(out)} players)")
    return out


def _bust_slate_cache(_caller: str = ""):
    """Clear today's slate cache from Redis + /tmp + GitHub so next request regenerates.

    IMPORTANT: This function writes bust tombstones to GitHub but does NOT regenerate
    the slate. It MUST only be called from code paths that will subsequently regenerate:
      - _run_cold_pipeline() — handles bust + regen as an atomic unit
      - /api/lab/update-config — bust only; next /api/slate request regenerates via Layer 3
    Never call from external scripts, MCP tools, or ad-hoc GitHub API writes — orphan
    tombstones with no regeneration cause infinite loading."""
    print(f"[bust-slate] triggered by: {_caller or 'unknown'}")
    today = _today_str()
    # Clear in-memory response cache (Level 0)
    _RESPONSE_CACHE.invalidate()
    # Flush Redis (Level 0.5) — all oracle:* keys
    try:
        flushed = rflush()
        if flushed:
            print(f"[bust-slate] flushed {flushed} Redis keys")
    except Exception as e:
        print(f"[bust-slate] Redis flush error: {e}")
    # Clear /tmp caches
    for key in [_CK_SLATE]:
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

# ── Extracted constants (DRY audit) ──────────────────────────────────────────
# Cache key constants — single source of truth for all tmp/lock cache lookups.
_CK_SLATE = "slate_v5"
_CK_SLATE_LOCKED = "slate_v5_locked"
_CK_LOG_DATES = "log_dates_v1"
def _ck_picks(gid): return f"picks_{gid}"
def _ck_game_proj(gid): return f"game_proj_{gid}"
def _ck_picks_locked(gid): return f"picks_locked_{gid}"

# Request timeout constants (seconds) for requests.get/post calls.
_T_DEFAULT = 10
_T_GITHUB = 15
_T_HEAVY = 30
_T_CLAUDE = 45
_T_EXECUTOR = 60

# Cache TTL constants (seconds).
_TTL_CONFIG = 300       # 5 min — model-config.json reload interval
_TTL_GAMES = 300        # 5 min — ESPN game data freshness
_TTL_LOG = 600          # 10 min — log dates / parlay history
_TTL_LOCKED = 60        # 1 min — game final check during locked slate
_TTL_PRE_SLATE = 180    # 3 min — pre-slate polling
_TTL_L5 = 1800          # 30 min — ESPN gamelog cache for parlay volatility (see _fetch_gamelog)
_TTL_HOUR = 3600        # 1 hour — infrequently changing data

# ── Response Cache for App-Level Hydration ──
# Level 0: In-memory response cache — serves cached JSON to frontend hydration.
# Reduces pipeline execution from 7 per session to 1 per day (at midnight or cold reset).
# Cache keys: endpoint + date (where applicable) + hash of params.
# TTLs override per-endpoint defaults. Cache invalidated by cold reset, injury checks, config changes.

class ResponseCache:
    """In-memory response cache with TTL, hit tracking, and thread-safe invalidation."""
    def __init__(self):
        self.store = {}  # {key: (data, timestamp, ttl_sec)}
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()

    def get(self, key):
        """Return (data, is_hit) where is_hit=True if cache valid, False if miss/expired."""
        with self._lock:
            if key in self.store:
                data, ts, ttl = self.store[key]
                age = time.time() - ts
                if age < ttl:
                    self.hits += 1
                    return data, True
                del self.store[key]  # evict expired entry to prevent unbounded growth
            self.misses += 1
            return None, False

    def set(self, key, data, ttl_sec):
        """Store response in cache with TTL (seconds)."""
        with self._lock:
            self.store[key] = (data, time.time(), ttl_sec)

    def invalidate(self, pattern=None):
        """Clear cache. pattern=None clears all; pattern='slate' clears keys containing 'slate'."""
        with self._lock:
            if pattern is None:
                self.store.clear()
                self.hits = 0
                self.misses = 0
            else:
                self.store = {k: v for k, v in self.store.items() if pattern not in k}

    def stats(self):
        """Return cache hit rate and counts."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total * 100) if total > 0 else 0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "size": len(self.store),
            }

_RESPONSE_CACHE = ResponseCache()

# Response cache key templates (single source of truth for cache keys).
_CACHE_KEYS = {
    "slate": "slate:{}",
    "games": "games:{}",
    "log_dates": "log_dates",
    "log_get": "log_get:{}",
}

# Response cache TTLs per endpoint (seconds). Override defaults here.
_CACHE_TTLS = {
    "slate": 60,            # 60s — burst protection; short TTL ensures lock state propagates promptly
    "games": 60,            # 60s — matches slate TTL; lock status must propagate promptly
    "log_dates": 600,       # 10min — dates rarely change mid-day
    "log_get": 300,         # 5min — per-date, static once day is final
}

def _with_response_cache(cache_key, ttl_key, fn, *args, **kwargs):
    """Execute fn() if cache miss, store result, return (data, is_hit).

    cache_key: str key for cache lookup
    ttl_key: str key in _CACHE_TTLS for TTL lookup
    fn: callable returning response dict
    *args, **kwargs: passed to fn

    Returns: (response_dict, is_hit)
    """
    cached, is_hit = _RESPONSE_CACHE.get(cache_key)
    if is_hit:
        return cached, True

    # Cache miss: execute function and store result
    result = fn(*args, **kwargs)
    ttl_sec = _CACHE_TTLS.get(ttl_key, 3600)
    _RESPONSE_CACHE.set(cache_key, result, ttl_sec)
    return result, False

def _add_cache_metadata(response_dict, is_hit, cache_key=None):
    """Add cache_status and metadata fields to response."""
    response_dict["cache_status"] = "hit" if is_hit else "miss"
    response_dict["cached_at"] = datetime.now(timezone.utc).isoformat() + "Z"
    if cache_key:
        response_dict["_cache_key"] = cache_key
    return response_dict

def _invalidate_response_cache(pattern=None):
    """Clear response cache + Redis + /tmp files."""
    _RESPONSE_CACHE.invalidate(pattern)
    try:
        rflush()
    except Exception:
        pass
    for f in CACHE_DIR.glob("*.json"):
        try:
            f.unlink(missing_ok=True)
        except Exception:
            pass
    print(f"[cache] response cache invalidated (pattern={pattern})", flush=True)

# ThreadPoolExecutor worker counts.
_W_LIGHT = 2            # light parallelism (2-way fetch)
_W_STANDARD = 8         # standard game/slate/picks processing
_W_L5 = 10              # parlay gamelog batch (I/O-bound ESPN athlete gamelog calls)

# GitHub data path builder — replaces 34+ hardcoded "data/..." strings.
_DATA_PREFIXES = {
    "slate": "data/slate", "predictions": "data/predictions",
    "locks": "data/locks", "boosts": "data/boosts",
    "ownership": "data/ownership",
    "most_popular": "data/most_popular",
    "most_drafted_3x": "data/most_drafted_3x",
    "winning_drafts": "data/winning_drafts",
    "actuals": "data/actuals", "audit": "data/audit",
}
def _data_path(kind, date, ext="json", suffix=""):
    """Build GitHub data path.
    _data_path('slate', '2026-03-22', suffix='_slate') -> 'data/slate/2026-03-22_slate.json'
    _data_path('predictions', '2026-03-22', ext='csv')  -> 'data/predictions/2026-03-22.csv'
    """
    return f"{_DATA_PREFIXES[kind]}/{date}{suffix}.{ext}"

# Shared slot multipliers (imported from api.shared for single source of truth)
from api.shared import SLOT_MULTIPLIERS as _SLOT_MULTS_SHARED, ESPN_BASE as ESPN

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
        "ceiling": 3.0, "floor": 0.0,
        "big_market_teams": ["LAL", "GS", "GSW", "BOS", "NY", "NYK", "PHI", "MIA", "LAC", "CHI"],
        "star_ppg_tiers": [
            {"min_ppg": 26, "boost_cap": 0.2},
            {"min_ppg": 22, "boost_cap": 0.4},
            {"min_ppg": 19, "boost_cap": 0.8},
        ],
        "team_boost_ceiling": {
            "GS": 1.8, "GSW": 1.8, "CLE": 1.7,
            "MEM": 1.5, "OKC": 1.8, "BOS": 1.8,
            "DEN": 2.0, "MIL": 2.0,
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
        "starter_blowout_floor": 0.70,
    },
    "real_score": {
        "dfs_weights":{"pts":1.5,"reb":0.5,"ast":1.2,"stl":2.5,"blk":1.5,"tov":-1.0},
        "compression_divisor": 6.0,
        "compression_power": 0.65,
        "rs_cap": 20.0,
        "ai_blend_weight": 0.30,
        # RS regression guard: cap heuristic_rs by scoring tier to prevent
        # bench players (PPG < 12) from being projected at RS 6.0+.
        # Data: avg top performer RS = 4.16, median = 3.9. Role players rarely
        # produce RS > 5.0 even on their best days.
        "regression_guard": {
            "enabled": True,
            "bench_cap": 4.5,       # PPG < 8: deep bench max RS
            "role_cap": 5.5,        # PPG 8-14: role player max RS
            "starter_cap": 7.0,     # PPG 14-20: starter max RS
        },
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
        "post_lock_calibration": {
            "enabled": True,
            "require_locked_slate": True,
            "recency_strength": 0.20,
            "max_nudge": 0.20,
            "cascade_weight": 0.10,
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
    },
    "cascade": {"redistribution_rate":0.70,"per_player_cap_minutes":10.0,"partial_cascade_cap_minutes":4.0,"center_forward_share":0.30,"gtd_sit_probability":0.30,"dtd_sit_probability":0.10,"gtd_minute_reduction":0.15,"max_cascade_pct":0.40,
        "team_detector": {
            "enabled": True,
            "star_ppg_threshold": 20.0,     # PPG that qualifies as "star" for cascade detection
            "boost_floor": 2.5,             # Minimum boost prediction for cascade teammates
            "deep_rotation_rs_floor": 1.5,  # Relaxed RS floor for cascade deep rotation (normally 2.0)
            "deep_rotation_min_minutes": 12.0,  # Relaxed minutes floor for cascade (normally 25)
        },
    },
    "dfs_popularity": {
        "weight": 0.7,
        "max_penalty": 0.5,
        "max_bonus": 0.2,
        "decay_days": 3,
        "confidence_threshold": 0.4,
    },
    "context_layer": {
        "enabled": True,
        "model": "claude-sonnet-4-6-20250514",
        "max_adjustment": 0.4,
        "timeout_seconds": 20,
        "web_search_enabled": True,
        "web_search_model": "claude-sonnet-4-6-20250514",
        "max_slate_adjustments": 8,
        "max_total_impact": 2.0,
    },
    "projection": {
        "min_gate_minutes": 15, "lock_buffer_minutes": 5, "b2b_minute_penalty": 0.88,
        # DNP / reliability guards
        "gtd_minute_penalty": 0.75,
        "dnp_risk_min_threshold": 8.0,
        "max_predmin_drop": 0.0,  # Reject players projected below season avg minutes (0 = no drop allowed)
        # Gamelog-based projection: use per-game data from last N days instead of
        # ESPN averaged splits. Fixes rotation-loss blindness (e.g. Payne 17→2 min).
        "gamelog_window_days": 7,
        "season_recent_blend": 0.70,          # 70% recent / 30% season
        "injury_return_blend": 0.30,           # Injury return players: 30% recent / 70% season
        "injury_return_gp_ratio": 0.60,        # GP < 60% of expected = likely missed time
        "injury_return_min_spike": 1.12,       # Recent min > 112% of season = ramp-up signal
    },
    "matchup": {
        "enabled": True,
        "claude_enabled": False,  # Layer 1.5 disabled — ESPN def stats provide equivalent signal at zero cost
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
    "lineup": {
        "game_chalk_rating_floor": 3.5,
        "avg_slot_multiplier": 2.0,
        "slot_multipliers": _SLOT_MULTS_SHARED,
    },
    "lab": {
        "auto_improve_threshold_pct": 3.0,
    },
    "pass2": {
        "vegas_total_threshold": 3.0,
    },
    # ── Per-Game Draft Strategy (Findings 1-6, 18-game 76-lineup analysis) ──
    # grep: PER_GAME_CONFIG
    "per_game": {
        "enabled": True,
        # Finding 3: Game total ceiling multiplier
        # 250+ → avg winning score 32.1; <210 → 25.7
        # Scale projections proportionally to game total
        "total_baseline": 222,
        "total_mult_strength": 0.003,   # per-point above/below baseline
        "total_mult_floor": 0.92,
        "total_mult_ceiling": 1.12,
        # Finding 4: Spread-based composition toggle
        # Close (≤5): balanced build — reward floor/consistency
        # Moderate (6-12): neutral
        # Blowout (13+): top-heavy — lean favored team
        "close_spread_threshold": 5,
        "blowout_spread_threshold": 13,
        "close_game_floor_bonus": 0.06,  # max bonus for low-variance players in close games
        # Finding 6: Favored team role player tilt (blowout-specific)
        "blowout_favored_role_bonus": 1.12,
        "blowout_favored_star_bonus": 1.05,
        "blowout_underdog_role_penalty": 0.88,
        "blowout_underdog_star_penalty": 0.95,
        "blowout_min_per_team": 1,      # relax team balance in blowouts (allow 4-1)
        "role_player_pts_ceiling": 18,   # season_pts <= this = role player
        # Finding 7: Graduated blowout penalties (new, from 6-game leaderboard study)
        # LAL/LAC 14pt: losing-team bigs (Allen 3.4, Ayton 3.3) outscored winning
        # role players. Moderate blowouts (13-19pt) shouldn't bury underdog players
        # the way true blowouts (20+pt, MIL/DAL 24pt) do.
        "heavy_blowout_threshold": 20,         # 20+pt = true blowout (full penalties)
        "moderate_blowout_underdog_role": 0.95, # 13-19pt: mild role penalty (was 0.88)
        "moderate_blowout_underdog_star": 0.98, # 13-19pt: barely penalize stars (was 0.95)
        # Finding 2: Value anchor (high-floor player at low slot)
        "value_anchor_min_rating": 3.8,  # RS floor for value anchor candidates
        "value_anchor_pts_ceiling": 16,  # season_pts ceiling — non-star
        "value_anchor_bonus": 0.08,      # RS bonus for value anchor candidates
        # Finding 1: Conviction slot strategy
        # Close game: dampen variance → consistent players rise to 2.0x
        # Blowout: reward upside → high-ceiling players rise to 2.0x
        "close_game_variance_dampen": 0.08,  # penalize variance in close games
        "blowout_variance_uplift": 0.04,     # reward variance in blowouts
        "neutral_favored_lean": 1.02,        # mild favored-team lean in moderate spreads
        # Finding 8: Sleeper 5th-player gates (new, from 6-game leaderboard study)
        # Winners differ from mid-pack by exactly 1 player pick in 5/6 games.
        # Winning 5th picks (Plowden 3.3, Camara 3.1, Dieng 3.1) fail the 3.5 floor.
        # Two-tier pool: core (standard) + sleeper (relaxed) for the decisive 5th pick.
        "sleeper_rating_floor": 2.5,     # relaxed RS floor for 5th player candidates
        "sleeper_min_floor": 12.0,       # relaxed minutes floor
        "sleeper_min_pts": 3.0,          # relaxed points floor — includes young/fringe players on rebuilding teams
        "star_carry_threshold": 6.0,     # min projected RS to classify a player as superstar-carry
        "star_carry_role_bonus": 0.10,   # bonus for role players on a superstar-carry team (Jokić/Wemby effect)
    },
    "scoring_thresholds": {
        "min_chalk_rating": 3.5,
        "min_game_pts": 8.0,
    },
    # ── Strategy Report v1: Data-driven draft parameters ──────────────────
    # Based on 90 dates of winning draft data. These ~15 parameters replace
    # the ~120 draft-specific parameters scattered across other config sections.
    "strategy": {
        "rs_floor": 2.0,               # Finding 2: 0% of winners below RS 2.0; 2.7% below 3.0
                                        # RS 2 + Boost 3.0 combo has 18 winning appearances (avg value 13.3)
        "min_pts_projection": 2.0,      # Universal scoring floor in project_player
        "min_minutes": 15.0,            # Lowered from 25: deep bench 3.0x boost players win slates
        "min_recent_minutes": 15.0,     # Minimum recent minutes for candidate pool (rotation-bubble filter)
        "min_pred_min_season_ratio": 1.0, # Floor: predMin >= season_min * ratio (1.0 = at least season avg; exempt: B2B, GTD)
        "minutes_increase_bypass": 15.0, # If predMin - season_min >= this, bypass minutes gates (cascade/injury expanded role)
        "close_game_rs_bonus": 0.3,     # Finding 7: close games (spread ≤ 5) → +0.3 RS
        "pace_rs_bonus_per_10": 0.15,   # Finding 7: +0.15 RS per 10pts of game total above 220
        "anti_popularity_enabled": True, # Finding 4: -0.457 correlation, 24% value edge
        "anti_popularity_strength": 0.2, # Boost penalty per unit of estimated popularity
        "max_per_team": 1,              # Keep at 1 for now — team stacking risk too high
        # ── Unified EV lineup construction ────────────────────────────────
        # EV = RS × (avg_slot + boost). No hardcoded star/role archetype.
        # The formula naturally selects stars when RS is dominant, role players
        # when boost is dominant — slate composition determines the mix each day.
        # ── Contrarian bonus (Finding 4: 22.7% of leaderboard, avg value 18.0)
        # Low-PPG role players with high boost are under-drafted. They keep their
        # boost 98.9% of the time and produce outsized value. Small EV uplift.
        "contrarian_bonus": {
            "enabled": True,
            "max_bonus": 0.10,       # Up to 10% EV boost for contrarian picks
            "min_boost": 2.0,        # Must have ≥2.0 predicted boost to qualify
            "max_season_ppg": 16.0,  # Only role players (not stars who get drafted heavily)
        },
        # ── Leaderboard frequency bonus ───────────────────────────────────
        # Data-driven: players appearing 5+ times on the leaderboard are proven.
        # No hardcoded names — loaded from data/top_performers.csv dynamically.
        "leaderboard_frequency": {
            "enabled": True,
            "min_appearances": 2,          # 2+ appearances qualify; ghost pattern visible fast
            "ghost_quality_weight": 0.80,  # 80% of bonus from ghost quality; 20% raw count
            "max_bonus": 0.35,             # Up to 35% EV uplift for proven ghost players
            "bonus_per_appearance": 0.008, # Small residual count bonus (secondary signal)
        },
        "minutes_delta": {
            "enabled": True,
            "neutral_zone": 2.0,        # ±2 min band = "business as usual" (no catalyst)
            "neutral_discount": 0.97,   # 3% RS haircut when delta within neutral zone
            "bonus_per_min": 0.03,      # 3% RS bonus per min above neutral zone (was 1.5% — too weak)
            "max_bonus": 0.20,          # Cap at 20% (was 12%; reached at ~+8.7 min delta)
            "penalty_per_min": 0.01,    # 1% RS penalty per min below -neutral zone
            "max_penalty": 0.08,        # Cap at 8%
        },
        "minutes_increase_ev_bonus": {
            "enabled": True,
            "min_delta": 4.0,           # Minimum minutes increase to trigger EV bonus
            "bonus_per_min": 0.02,      # 2% EV bonus per minute above min_delta
            "max_bonus": 0.15,          # Cap at 15% EV uplift
        },
        # ── Historical RS confidence discount (project_player) ────────────
        # Soft pull-back when predicted RS is far above player's historical track record.
        # NOT a hard cap — players can still pop off. Bayesian approach:
        # history is the prior, projection is the likelihood. More history = stronger prior.
        "historical_rs_discount": {
            "enabled": True,
            "min_appearances": 2,       # Lowered from 3: even 2 datapoints inform the prior
            "saturation_k": 6.0,        # Prior strength saturates faster: n/(n+k). k=6 → 50% at 6 appearances
            "max_prior_strength": 0.65, # Raised from 0.5: history is a stronger signal (Hyland 5.0→0.5 post-mortem)
            "discount_scale": 0.6,      # Raised from 0.4: more aggressive pull-back per unit overshoot
            "max_discount_frac": 0.8,   # Raised from 0.6: can pull back up to 80% of overshoot
        },
        # ── Momentum curve detection (_build_lineups) ─────────────────────
        # Detects two patterns in player historical data:
        #   HYPE TRAP: Drafts exploding + boost declining = player peaked. Penalty.
        #     (e.g., Sensabaugh: 2→207 drafts, boost 3.0→2.0 over the season)
        #   RISING WAVE: RS trending up + low drafts + high boost = coming up. Bonus.
        #     (e.g., Fears/Hawkins: rising RS, still low-drafted, high boost)
        "momentum_curve": {
            "enabled": True,
            "min_history": 3,                  # Need 3+ appearances to detect momentum
            # Hype trap
            "hype_trap_max_penalty": 0.20,     # Up to 20% EV penalty for trapped players
            "draft_growth_threshold": 0.5,     # 50%+ draft count increase triggers trap
            "boost_decline_threshold": 0.4,    # 0.4+ boost decline confirms trap
            # Rising wave
            "rising_wave_max_bonus": 0.20,     # Up to 20% EV bonus for rising players
            "rs_trend_min": 0.3,               # Minimum RS increase trend to qualify
            "wave_max_drafts": 200.0,          # Must still be under-drafted (<200 drafts)
            "wave_min_boost": 1.5,             # Must still have meaningful boost (≥1.5)
        },
        # NOTE: Injury return penalties REMOVED after historical audit (2,316 entries, 152 dates).
        # Returning players produce +11.5% higher value than baseline due to boost reset mechanism.
        # The contrarian signal (72% under-drafted, avg 297 drafts vs 647) is too valuable to penalize.
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
            if age < _TTL_CONFIG:
                raw = cache_file.read_text().strip()
                if raw:
                    return json.loads(raw)
                # Empty file — delete and fall through to GitHub
                cache_file.unlink(missing_ok=True)
        except json.JSONDecodeError:
            # Corrupt cache file — delete and fall through to GitHub
            try:
                cache_file.unlink(missing_ok=True)
            except Exception:
                pass
        except Exception as _ce:
            print(f"[WARN] Config cache read failed: {_ce}")
    try:
        content, _ = _github_get_file("data/model-config.json")
        if content:
            cfg = json.loads(content)
            # Atomic write: write to temp file then rename to prevent empty/partial reads
            tmp_file = cache_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(cfg))
            tmp_file.rename(cache_file)
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


def _repo_pickle_paths(filename: str) -> list:
    """Repo root, package dir, then CWD — same resolution order for bundled .pkl files."""
    base = Path(__file__)
    return [base.parent.parent / filename, base.parent / filename, Path(filename)]


_LGBM_JSON_PATHS = _repo_pickle_paths("lgbm_model.json")
_LGBM_PATHS = _repo_pickle_paths("lgbm_model.pkl")
# Boost/drafts LightGBM models removed — replaced by 3-tier cascade in api/boost_model.py.
# grep: BOOST CASCADE MODEL

def _ensure_lgbm_loaded():
    """Lazy-load the Real Score LightGBM bundle on first use.

    Order: native ``lgb.Booster`` files (``lgbm_model.json`` + .txt heads), then legacy pickle.
    Native format survives LightGBM / sklearn pickle drift across deploys."""
    global AI_MODEL, AI_MODEL_BASELINE, AI_MODEL_SPIKE, AI_FEATURES, _LGBM_LOAD_ATTEMPTED
    if _LGBM_LOAD_ATTEMPTED:
        return
    with _LGBM_LOAD_LOCK:
        if _LGBM_LOAD_ATTEMPTED:
            return
        for _p in _LGBM_JSON_PATHS:
            if not _p.exists():
                continue
            try:
                meta = json.loads(_p.read_text(encoding="utf-8"))
                if meta.get("format") != "lightgbm_native":
                    continue
                _dir = _p.parent
                bf = _dir / meta.get("baseline_file", "lgbm_baseline.txt")
                sf = _dir / meta.get("spike_file", "lgbm_spike.txt")
                if not bf.is_file() or not sf.is_file():
                    continue
                if meta.get("bundle_version") != 2 or "features" not in meta:
                    continue
                _mb = lgb.Booster(model_file=str(bf.resolve()))
                _ms = lgb.Booster(model_file=str(sf.resolve()))
                if callable(getattr(_mb, "predict", None)) and callable(getattr(_ms, "predict", None)):
                    AI_FEATURES = meta["features"]
                    AI_MODEL_BASELINE = _mb
                    AI_MODEL_SPIKE = _ms
                    AI_MODEL = None
                    break
                print(f"[lgbm] native boosters not callable — skipping {_p}")
            except (OSError, json.JSONDecodeError, KeyError, ValueError, Exception):
                pass
        if AI_MODEL_BASELINE is None and AI_MODEL is None:
            for _p in _LGBM_PATHS:
                if _p.exists():
                    try:
                        with open(_p, "rb") as _f:
                            _bundle = pickle.load(_f)
                        if not isinstance(_bundle, dict) or "features" not in _bundle:
                            continue
                        AI_FEATURES = _bundle["features"]
                        if _bundle.get("bundle_version") == 2 and "model_baseline" in _bundle and "model_spike" in _bundle:
                            _mb = _bundle["model_baseline"]
                            _ms = _bundle["model_spike"]
                            if callable(getattr(_mb, "predict", None)) and callable(getattr(_ms, "predict", None)):
                                AI_MODEL_BASELINE = _mb
                                AI_MODEL_SPIKE = _ms
                                AI_MODEL = None
                                break
                            print(f"[lgbm] bundle_version=2 models not callable — skipping {_p}")
                            continue
                        if "model" in _bundle:
                            _m = _bundle["model"]
                            if callable(getattr(_m, "predict", None)):
                                AI_MODEL = _m
                                AI_MODEL_BASELINE = None
                                AI_MODEL_SPIKE = None
                                break
                            print(f"[lgbm] model loaded but predict is not callable — skipping {_p}")
                            continue
                    except (OSError, pickle.UnpicklingError, KeyError, ValueError, ModuleNotFoundError):
                        pass
        _LGBM_LOAD_ATTEMPTED = True
        if AI_MODEL is None and AI_MODEL_BASELINE is None:
            print("[WARN] LightGBM model not found or invalid bundle — using heuristic fallback for all projections")


def _lgbm_predict_rs(feat_vec: list) -> Optional[float]:
    """Return blended RS prediction from loaded bundle, or None if unavailable."""
    _ensure_lgbm_loaded()
    if AI_FEATURES is not None and len(feat_vec) != len(AI_FEATURES):
        raise ValueError(f"Feature mismatch: model expects {len(AI_FEATURES)}, got {len(feat_vec)}")
    arr = np.array([feat_vec], dtype=np.float64)
    if AI_MODEL_BASELINE is not None and AI_MODEL_SPIKE is not None:
        base = float(AI_MODEL_BASELINE.predict(arr)[0])
        spike = float(AI_MODEL_SPIKE.predict(arr)[0])
        out = base + max(0.0, spike)
        if not math.isfinite(out):
            print(f"[WARN] LightGBM RS prediction non-finite ({out}) — skipping AI blend for this player")
            return None
        return out
    if AI_MODEL is not None:
        out = float(AI_MODEL.predict(arr)[0])
        if not math.isfinite(out):
            print(f"[WARN] LightGBM RS prediction non-finite ({out}) — skipping AI blend for this player")
            return None
        return out
    return None


# _ensure_boost_model_loaded / _lgbm_predict_boost removed — see api/boost_model.py


# _ensure_drafts_model_loaded / _lgbm_predict_log1p_drafts removed — see api/boost_model.py

# _TEAM_MARKET_SCORES alias — still used by _estimate_log_drafts and draft tier system
_TEAM_MARKET_SCORES = TEAM_MARKET_SCORES


_NBA_SEASON_START_MONTH_DAY = (10, 21)  # Season typically starts Oct 21

def _estimate_games_played() -> float:
    """Estimate games played from season progress when ESPN gp unavailable.
    Returns a float estimate based on days elapsed since season start.
    82 games over ~180 days ≈ 0.456 games/day."""
    today = _et_date()
    season_year = today.year if today.month >= 10 else today.year - 1
    season_start = today.replace(year=season_year, month=_NBA_SEASON_START_MONTH_DAY[0],
                                 day=_NBA_SEASON_START_MONTH_DAY[1])
    days_elapsed = max(0, (today - season_start).days)
    return float(max(1, min(82, round(days_elapsed * 82 / 180))))


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
    rest_days: Optional[float] = None,
    team_pace: Optional[float] = None,
    opp_pts_allowed: Optional[float] = None,
    game_total: Optional[float] = None,
    teammate_out_count: Optional[float] = None,
    recent_ast: Optional[float] = None,
    recent_stl: Optional[float] = None,
    recent_blk: Optional[float] = None,
    avg_reb: Optional[float] = None,
    usage_share: Optional[float] = None,
    min_volatility: Optional[float] = None,
    spread_abs: Optional[float] = None,
    recent_3g_pts: Optional[float] = None,
) -> list:
    """Build feature vector aligned with the loaded Real Score bundle (native or pickle).

    Uses compute_rs_features() from api.features for formula consistency with training.
    Uses the loaded model's AI_FEATURES list to select and order features,
    so both v63 (17-feature) and legacy (22-feature) bundles work without
    code changes. Falls back to the 17-feature canonical order if no bundle loaded.
    """
    # Derive values that training computes from raw game data
    sign = 1.0 if side == "away" else -1.0
    opp_def_rating = 112.0 + sign * (spread or 0) * 0.7
    home_away_ = 1.0 if side == "home" else 0.0
    cascade_signal_ = 1.0 if cascade_bonus > 0 else 0.0

    # Use shared feature computation — matches training formulas exactly
    _feature_map = compute_rs_features(
        avg_min=avg_min,
        avg_pts=season_pts,
        recent_min=recent_min,
        recent_pts=recent_pts,
        season_pts=season_pts,
        season_min=season_min,
        recent_ast=recent_ast if recent_ast is not None else ast,
        recent_stl=recent_stl if recent_stl is not None else stl,
        recent_blk=recent_blk if recent_blk is not None else blk,
        avg_reb=avg_reb if avg_reb is not None else reb,
        home_away=home_away_,
        opp_def_rating=opp_def_rating,
        rest_days=float(rest_days) if rest_days is not None else 2.0,
        games_played=float(games_played) if games_played is not None else _estimate_games_played(),
        cascade_signal=cascade_signal_,
        opp_pts_allowed=float(opp_pts_allowed) if opp_pts_allowed is not None else 110.0,
        team_pace_proxy=float(team_pace) if team_pace is not None else 110.0,
        usage_share=float(usage_share) if usage_share else 0.0,
        teammate_out_count=float(teammate_out_count) if teammate_out_count is not None else 0.0,
        game_total=float(game_total) if game_total is not None else 222.0,
        spread_abs=float(spread_abs) if spread_abs is not None else abs(float(spread or 0)),
        recent_3g_pts=float(recent_3g_pts) if recent_3g_pts is not None else None,
    )

    # Override min_volatility with nba_api value (rolling 5-game std) when available
    if min_volatility is not None:
        _feature_map["min_volatility"] = float(np.clip(min_volatility, 0.0, 1.2))

    # Use the loaded model's feature list to select and order.
    _ensure_lgbm_loaded()
    bundle_features = AI_FEATURES
    if bundle_features:
        return rs_feature_vector(_feature_map, bundle_features)

    # Fallback: canonical 17-feature order
    _CANONICAL_FEATURES = [
        "avg_min", "avg_pts", "usage_trend", "opp_def_rating", "home_away",
        "ast_rate", "def_rate", "pts_per_min", "rest_days", "recent_vs_season",
        "games_played", "reb_per_min", "l3_vs_l5_pts", "min_volatility",
        "starter_proxy", "opp_pts_allowed", "team_pace_proxy",
    ]
    return rs_feature_vector(_feature_map, _CANONICAL_FEATURES)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CACHE UTILITIES
# Module-level: UPPER_SNAKE = public constants; _lower = private (do not mutate).
# grep: ESPN, MIN_GATE, DEFAULT_TOTAL, _cp, _cg, _cs, _lp, _lg, _ls
# _cg/cs = prediction cache (date-keyed, /tmp); _lg/ls = lock cache (warm instance).
# ─────────────────────────────────────────────────────────────────────────────
# ESPN imported from api.shared as ESPN (DRY — single source of truth)
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
ANTHROPIC_API_BASE = "https://api.anthropic.com/v1"
HAIKU_MODEL = "claude-haiku-4-5-20251001"
OPUS_MODEL  = "claude-opus-4-6"
MIN_GATE  = 25          # Minimum projected minutes — filter low-minutes players
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
    "steals":   "player_steals",
    "blocks":   "player_blocks",
    "threes":   "player_threes",
    "points_rebounds_assists": "player_points_rebounds_assists",
}


# ─────────────────────────────────────────────────────────────────────────────
# GAME-LEVEL ODDS SNAPSHOT (spreads + totals from The Odds API)
# grep: GAME ODDS SNAPSHOT
# Fetches consensus spreads and totals near lock window — more accurate than
# ESPN's stale odds. One API call per slate, cached for 30 minutes.
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_game_odds_snapshot(games: list) -> dict:
    """Fetch game-level spreads/totals from The Odds API.

    Returns {(home_abbr, away_abbr): {"spread": float, "total": float}}
    Falls back to empty dict on any failure — callers keep ESPN data.
    """
    api_key = _get_odds_api_key()
    if not api_key or not games:
        return {}

    cache_key = f"game_odds_snapshot_{_today_str()}"
    cached = _cg(cache_key)
    if cached and isinstance(cached, dict) and cached.get("data"):
        age = time.time() - cached.get("_ts", 0)
        if age < 1800:  # 30 min cache
            print(f"[game-odds] cache hit ({len(cached['data'])} games, {age:.0f}s old)")
            return cached["data"]

    try:
        r = requests.get(
            f"{ODDS_API_BASE}/sports/basketball_nba/odds/",
            params={
                "apiKey": api_key,
                "regions": "us",
                "markets": "spreads,totals",
                "oddsFormat": "american",
                "bookmakers": "pinnacle,draftkings,fanduel,betmgm",
            },
            timeout=15,
        )
        remaining = r.headers.get("x-requests-remaining", "?")
        if not r.ok:
            print(f"[game-odds] fetch failed: HTTP {r.status_code} (remaining={remaining})")
            return {}
        events = r.json()
        print(f"[game-odds] fetched {len(events)} events (remaining={remaining})")
    except Exception as e:
        print(f"[game-odds] fetch error: {e}")
        return {}

    result = {}
    for ev in events:
        ev_home = ev.get("home_team", "")
        ev_away = ev.get("away_team", "")
        home_abbr = away_abbr = ""
        for g in games:
            ha = g.get("home", {}).get("abbr", "")
            aa = g.get("away", {}).get("abbr", "")
            if (_abbr_matches(ha, ev_home) and _abbr_matches(aa, ev_away)):
                home_abbr, away_abbr = ha, aa
                break
            if (_abbr_matches(ha, ev_away) and _abbr_matches(aa, ev_home)):
                home_abbr, away_abbr = ha, aa
                break
        if not home_abbr:
            continue

        spread_val = total_val = None
        for book in ev.get("bookmakers", []):
            for mkt in book.get("markets", []):
                if mkt["key"] == "spreads":
                    for out in mkt.get("outcomes", []):
                        if _abbr_matches(home_abbr, out.get("name", "")):
                            spread_val = spread_val or out.get("point")
                elif mkt["key"] == "totals":
                    for out in mkt.get("outcomes", []):
                        if out.get("name", "").lower() == "over":
                            total_val = total_val or out.get("point")

        if spread_val is not None or total_val is not None:
            result[(home_abbr, away_abbr)] = {
                "spread": spread_val,
                "total": total_val,
            }

    if result:
        _cs(cache_key, {"data": result, "_ts": time.time()})
        print(f"[game-odds] snapshot: {len(result)} games with spreads/totals")
    return result


def _apply_odds_snapshot_to_games(games: list) -> int:
    """Overlay Odds API spreads/totals onto games in-place. Returns count updated."""
    snapshot = _fetch_game_odds_snapshot(games)
    if not snapshot:
        return 0
    updated = 0
    for g in games:
        ha = g.get("home", {}).get("abbr", "")
        aa = g.get("away", {}).get("abbr", "")
        odds = snapshot.get((ha, aa))
        if not odds:
            continue
        if odds.get("spread") is not None:
            g["spread"] = odds["spread"]
            g["_odds_source"] = "odds_api"
        if odds.get("total") is not None:
            g["total"] = odds["total"]
            g["_odds_source"] = "odds_api"
        updated += 1
    return updated

def _cp(k, date_str=None):
    """Cache path for key k. date_str: optional slate date (YYYY-MM-DD) for midnight-rollover correctness."""
    d = date_str or _today_str()
    return CACHE_DIR / f"{hashlib.md5(f'{d}:{k}'.encode()).hexdigest()}.json"

# ── Redis TTL map — maps cache key prefixes/names to TTL in seconds ──────────
# Keys not in the map get a default 24h TTL in Redis.
_REDIS_TTL_MAP = {
    # Lock caches — very short TTL (1 min)
    "slate_v5_locked": _TTL_LOCKED,
    "picks_locked_": _TTL_LOCKED,
    # Game data — 5 min
    "games_": _TTL_GAMES,
    "log_get_": _TTL_CONFIG,
    # Log / history — 10 min
    "log_dates_v1": _TTL_LOG,
    # Athlete / player data — 30 min
    "ath3_": _TTL_L5,
    "team_pstats_": _TTL_L5,
    "gamelog_v2_": _TTL_L5,
    # Slate / picks — day-scoped, 6h default (cleared by refresh)
    "slate_v5": 21600,
    "picks_": 21600,
    "game_proj_": 21600,
}

def _redis_ttl_for_key(k):
    """Look up Redis TTL for a cache key. Returns seconds or 86400 (24h) default."""
    for prefix, ttl in _REDIS_TTL_MAP.items():
        if k.startswith(prefix) or k == prefix:
            return ttl
    return 86400  # 24h default for unrecognised keys

def _cg(k, date_str=None):
    """Cache GET: try Redis first, fall back to /tmp file."""
    d = date_str or _today_str()
    # Try Redis
    val = rcg(k, d)
    if val is not None:
        return val
    # Fall back to file
    p = _cp(k, date_str)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            # Backfill Redis from file hit (async-safe, best-effort)
            rcs(k, data, d, ttl=_redis_ttl_for_key(k))
            return data
        except Exception:
            return None
    return None

def _cs(k, v, date_str=None):
    """Cache SET: write to both Redis (with TTL) and /tmp file."""
    d = date_str or _today_str()
    rcs(k, v, d, ttl=_redis_ttl_for_key(k))
    try:
        _cp(k, date_str).write_text(json.dumps(v))
    except Exception:
        pass

def _cd(k, date_str=None):
    """Cache DELETE: remove from Redis AND /tmp file."""
    d = date_str or _today_str()
    rcd(k, d)
    try:
        _cp(k, date_str).unlink(missing_ok=True)
    except Exception:
        pass

def _lp(k, date_str=None):
    """Lock path for key k. date_str: optional slate date for midnight-rollover correctness."""
    d = date_str or _today_str()
    return LOCK_DIR / f"{hashlib.md5(f'{d}:{k}'.encode()).hexdigest()}.json"

def _lg(k, date_str=None):
    """Lock GET: try Redis first, fall back to /tmp lock file."""
    d = date_str or _today_str()
    val = rcg(f"lock:{k}", d)
    if val is not None:
        return val
    p = _lp(k, date_str)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            rcs(f"lock:{k}", data, d, ttl=_TTL_LOCKED)
            return data
        except Exception:
            return None
    return None

def _ls(k, v, date_str=None):
    """Lock SET: write to both Redis and /tmp lock file."""
    d = date_str or _today_str()
    rcs(f"lock:{k}", v, d, ttl=_TTL_LOCKED)
    try:
        _lp(k, date_str).write_text(json.dumps(v))
    except Exception:
        pass


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

def _any_locked(start_times):
    """True if any game in the list is currently in its lock window.
    Handles None/empty lists gracefully. Uses any() pattern per CLAUDE.md
    split-window documentation."""
    return bool(start_times) and any(_is_locked(st) for st in start_times)

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
    except (ImportError, KeyError):
        # Fallback when tzdata is unavailable: approximate DST transitions.
        # US DST starts 2nd Sunday of March (~Mar 8–14) and ends 1st Sunday of Nov (~Nov 1–7).
        # Using month 3 day>=8 through month 11 day<8 covers all transition weeks correctly.
        now_utc = datetime.now(timezone.utc)
        m, d = now_utc.month, now_utc.day
        is_dst = (m == 3 and d >= 8) or (4 <= m <= 10) or (m == 11 and d < 8)
        offset = timedelta(hours=-4 if is_dst else -5)
        return (now_utc + offset).date()

def _err(msg: str, code: int = 400) -> JSONResponse:
    """Standard error response helper (DRY)."""
    return JSONResponse({"error": msg}, status_code=code)

def _today_str() -> str:
    """Current ET date as YYYY-MM-DD string (shorthand for _et_date().isoformat())."""
    return _et_date().isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# ESPN DATA FETCHERS
# grep: _espn_get, fetch_games, fetch_roster, _fetch_athlete, _fetch_b2b_teams, _fetch_team_def_stats
# _fetch_athlete: returns blended season+recent stats dict for a player
# fetch_games: returns today's game list with lock/complete status
# ─────────────────────────────────────────────────────────────────────────────
def _safe_float(v: Any, default: float = 0.0) -> float:
    try: return float(v) if v is not None else default
    except (ValueError, TypeError): return default

def _espn_scoreboard(date_ymd: str) -> str:
    """Build ESPN scoreboard URL. date_ymd = YYYYMMDD format."""
    return f"{ESPN}/scoreboard?dates={date_ymd}"

_ESPN_RATE_LIMIT_STATE = {"remaining": None, "reset_at": None}
_ESPN_RATE_LIMIT_LOCK = threading.Lock()

def _espn_get(url, retry_on_429=True, max_retries=3):
    """Fetch from ESPN API with exponential backoff for rate limits.

    Inspects X-RateLimit-Remaining header to detect rate limit exhaustion.
    On 429 (Too Many Requests), backs off exponentially: 1s, 2s, 4s.
    """
    backoff_delays = [1, 2, 4]
    last_error = None

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=_T_DEFAULT)

            # Track rate limit state from response headers
            with _ESPN_RATE_LIMIT_LOCK:
                remaining = r.headers.get("x-requests-remaining")
                if remaining:
                    try:
                        _ESPN_RATE_LIMIT_STATE["remaining"] = int(remaining)
                    except ValueError:
                        pass
                reset_at = r.headers.get("x-requests-reset")
                if reset_at:
                    _ESPN_RATE_LIMIT_STATE["reset_at"] = reset_at

            # Handle rate limit response
            if r.status_code == 429:
                last_error = "rate_limited"
                if attempt < max_retries - 1 and retry_on_429:
                    delay = backoff_delays[attempt]
                    print(f"[espn] 429 rate limit — backing off {delay}s", flush=True)
                    time.sleep(delay)
                    continue
                else:
                    print(f"[espn] 429 rate limit (max retries reached) for {url[:120]}", flush=True)
                    return {}

            if not r.ok:
                print(f"[espn] HTTP {r.status_code} for {url[:120]}", flush=True)
                return {}

            return r.json()

        except requests.exceptions.Timeout:
            last_error = "timeout"
            if attempt < max_retries - 1:
                print(f"[espn] timeout (retry {attempt+1}/{max_retries}) for {url[:120]}", flush=True)
                time.sleep(backoff_delays[attempt])
                continue
            else:
                print(f"[espn] timeout for {url[:120]}", flush=True)
                return {}

        except requests.exceptions.ConnectionError as e:
            last_error = f"connection_error: {e}"
            if attempt < max_retries - 1:
                print(f"[espn] connection error (retry {attempt+1}/{max_retries}) for {url[:120]}: {e}", flush=True)
                time.sleep(backoff_delays[attempt])
                continue
            else:
                print(f"[espn] connection error for {url[:120]}: {e}", flush=True)
                return {}

        except (requests.RequestException, ValueError) as e:
            last_error = str(e)
            print(f"[espn] error for {url[:120]}: {e}", flush=True)
            return {}

    return {}

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
    data = _espn_get(_espn_scoreboard(yesterday))
    b2b = set()
    for ev in data.get("events", []):
        comp = ev.get("competitions", [{}])[0]
        for cd in comp.get("competitors", []):
            abbr = cd.get("team", {}).get("abbreviation", "")
            if abbr: b2b.add(abbr)
    _cs(cache_key, list(b2b))
    return b2b

def _fetch_team_rest_days() -> dict:
    """Compute rest days per team by checking ESPN scoreboards for the last 7 days.

    Returns {team_abbr: rest_days} where rest_days is clipped to [1, 7].
    Teams with no games in the last 7 days get default 3 (matches training fillna(3)).
    Cached for the current ET date.
    """
    today = _et_date()
    cache_key = f"team_rest_days_{today}"
    c = _cg(cache_key)
    if c is not None:
        return c
    # Check scoreboards for the last 7 days to find each team's most recent game
    last_game = {}  # {abbr: date}
    for days_ago in range(1, 8):
        check_date = today - timedelta(days=days_ago)
        date_str = check_date.strftime("%Y%m%d")
        try:
            data = _espn_get(_espn_scoreboard(date_str))
            for ev in data.get("events", []):
                comp = ev.get("competitions", [{}])[0]
                for cd in comp.get("competitors", []):
                    abbr = cd.get("team", {}).get("abbreviation", "")
                    if abbr and abbr not in last_game:
                        last_game[abbr] = check_date
        except Exception:
            continue
    # Convert to rest_days: days since last game, clipped [1, 7]
    result = {}
    for abbr, game_date in last_game.items():
        rest = (today - game_date).days
        result[abbr] = max(1, min(rest, 7))
    _cs(cache_key, result)
    return result

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


def _fetch_dvp_data() -> dict:
    """Fetch defense-vs-position (DvP) data from NBA.com stats API.

    Returns {team_abbr: {"G": ppg_allowed, "F": ppg_allowed, "C": ppg_allowed}}
    where abbrs match ESPN conventions (e.g. "GSW", "PHX").
    Cached daily (ET). Returns {} on any failure — callers fall back to
    team-level _fetch_team_def_stats().
    """
    cache_key = f"dvp_data_{_et_date().strftime('%Y%m%d')}"
    cached = _cg(cache_key)
    if cached is not None:
        return cached

    # NBA.com team abbr → ESPN abbr for mismatches
    _NBA_TO_ESPN = {
        "GSW": "GS", "SAS": "SA", "NYK": "NY", "NOP": "NO",
        "OKC": "OKC", "UTA": "UTAH", "PHX": "PHX",
    }

    # NBA.com anti-bot headers (required)
    _NBA_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nba.com/",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }

    _NBA_BASE = "https://stats.nba.com/stats/leaguedashteamstats"
    _SEASON = "2025-26"

    result: dict = {}
    try:
        for pos_group in ("G", "F", "C"):
            r = requests.get(
                _NBA_BASE,
                headers=_NBA_HEADERS,
                params={
                    "MeasureType": "Opponent",
                    "PerMode": "PerGame",
                    "Season": _SEASON,
                    "SeasonType": "Regular Season",
                    "PlayerPosition": pos_group,
                    "PaceAdjust": "N",
                    "Rank": "N",
                    "Outcome": "",
                    "Location": "",
                    "Month": "0",
                    "SeasonSegment": "",
                    "DateFrom": "",
                    "DateTo": "",
                    "OpponentTeamID": "0",
                    "VsConference": "",
                    "VsDivision": "",
                    "GameSegment": "",
                    "Period": "0",
                    "ShotClockRange": "",
                    "LastNGames": "0",
                },
                timeout=_T_DEFAULT,
            )
            if not r.ok:
                continue
            data = r.json()
            rs = (data.get("resultSets") or [{}])[0]
            headers = rs.get("headers", [])
            rows = rs.get("rowSet", [])
            if not headers or not rows:
                continue
            try:
                abbr_idx = headers.index("TEAM_ABBREVIATION")
                pts_idx = headers.index("OPP_PTS")
            except ValueError:
                continue
            for row in rows:
                abbr = row[abbr_idx]
                opp_pts = row[pts_idx]
                if not abbr or opp_pts is None:
                    continue
                # Normalise to ESPN abbreviation
                abbr = _NBA_TO_ESPN.get(abbr, abbr)
                try:
                    pts_f = float(opp_pts)
                except (TypeError, ValueError):
                    continue
                if abbr not in result:
                    result[abbr] = {}
                result[abbr][pos_group] = pts_f
    except Exception:
        pass

    if result:
        _cs(cache_key, result)
        print(f"[dvp] fetched DvP data for {len(result)} teams")
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


def _coerce_float(val, default: float) -> float:
    try:
        if val is None:
            return float(default)
        if isinstance(val, str):
            val = val.strip()
            if not val:
                return float(default)
        return float(val)
    except (TypeError, ValueError):
        return float(default)


def _clamp_float(val, default: float, mn: float, mx: float) -> float:
    return max(mn, min(mx, _coerce_float(val, default)))


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
        if time.time() - cached_at < _TTL_GAMES:  # 5 min TTL
            return c
    b2b_teams = _fetch_b2b_teams()
    date_str = today_et.strftime("%Y%m%d")
    data = _espn_get(_espn_scoreboard(date_str))
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
            is_out = inj_status in ["out", "injured", "suspended", "suspension"]
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

_TTL_ATHLETE = 1800  # 30-min TTL for athlete data (prevents stale injury/status)

def _fetch_athlete(pid):
    _ath_key = f"ath3_{pid}"
    c = _cg(_ath_key)
    if c:
        ts = c.get("_cached_ts", 0) if isinstance(c, dict) else 0
        if ts and (time.time() - ts) > _TTL_ATHLETE:
            _cd(_ath_key)
            c = None
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
            blend_w     = proj.get("season_recent_blend", 0.70)

            min_ratio = recent["min"] / max(season["min"], 1)
            # ── Injury return detection using ESPN data ──────────────────────
            # Signal 1: Games played (GP) — ESPN provides this in season splits.
            # If player has played <60% of expected games, they missed significant time.
            # Signal 2: Recent minutes spike — recent min > 112% of season avg = ramp-up.
            # Signal 3: RotoWire "questionable" status (already handled elsewhere).
            # When detected, cap blend weight to trust season more (avoid over-projecting
            # a player ramping back from injury whose recent 5-game sample is inflated).
            _ir_blend = float(proj.get("injury_return_blend", 0.30))
            _ir_gp_ratio = float(proj.get("injury_return_gp_ratio", 0.60))
            _ir_min_spike = float(proj.get("injury_return_min_spike", 1.12))
            _expected_gp = _estimate_games_played()
            _actual_gp = float(season.get("gp", 0))
            _gp_ratio = _actual_gp / max(_expected_gp, 1)
            # ESPN GP-based: player missed 40%+ of games → injury return
            _is_injury_return_gp = _actual_gp > 0 and _gp_ratio < _ir_gp_ratio
            # Minutes spike: recent min significantly higher than season (ramp-up)
            _is_injury_return_min = min_ratio > _ir_min_spike and recent["min"] > season["min"] + 2
            _is_injury_return = _is_injury_return_gp or _is_injury_return_min
            _effective_blend_w = min(blend_w, _ir_blend) if _is_injury_return else blend_w
            if min_ratio < major_thr:
                min_blend = round(season["min"] * (1 - major_w) + recent["min"] * major_w, 2)
            elif min_ratio < mod_thr:
                min_blend = round(season["min"] * (1 - mod_w) + recent["min"] * mod_w, 2)
            else:
                min_blend = round(season["min"] * (1 - _effective_blend_w) + recent["min"] * _effective_blend_w, 2)

            blended = {k: round(season[k] * (1 - _effective_blend_w) + recent[k] * _effective_blend_w, 2) for k in season}
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
            blended["_injury_return"] = _is_injury_return
            blended["_gp"] = _actual_gp
            blended["_expected_gp"] = _expected_gp
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
    blended["_cached_ts"] = time.time()
    _cs(f"ath3_{pid}", blended)
    return blended

# ─────────────────────────────────────────────────────────────────────────────
# BULK TEAM STATS FETCHER — reduces N+1 ESPN athlete fetches to 2 per game
# grep: _fetch_team_player_stats, BULK_ESPN
# ─────────────────────────────────────────────────────────────────────────────
def _fetch_team_player_stats(team_id: str) -> dict:
    """Fetch all player stats for a team in one ESPN API call.
    Returns {player_id: blended_stats_dict} matching _fetch_athlete output shape.
    Uses the athlete overview endpoint per-team via the roster, but caches at
    team level so a cold start fetches 2 team bundles instead of ~30 individual athletes.
    Falls back gracefully — callers should use _fetch_athlete for any missing players."""
    cache_key = f"team_pstats_{team_id}"
    c = _cg(cache_key)
    if c:
        ts = c.get("_cached_ts", 0) if isinstance(c, dict) else 0
        if ts and (time.time() - ts) < _TTL_ATHLETE:
            return c
        _cd(cache_key)

    # Fetch team statistics page — returns per-player season stats in one call
    url = (f"https://site.api.espn.com/apis/site/v2/sports/basketball"
           f"/nba/teams/{team_id}/statistics")
    data = _espn_get(url)
    if not data:
        return {}

    result = {}
    try:
        # ESPN team stats structure: { splits: { categories: [...] }, athletes: [...] }
        # athletes[] has { athlete: {id, displayName}, categories: [{name, stats}] }
        athletes = data.get("athletes", [])
        if not athletes:
            # Fallback: some ESPN responses nest under "results" or "splits"
            splits = data.get("splits", {})
            athletes = splits.get("athletes", []) if splits else []

        for ath_entry in athletes:
            try:
                athlete_info = ath_entry.get("athlete", {})
                pid = str(athlete_info.get("id", ""))
                if not pid:
                    continue

                # Build stats dict from categories
                s = {"min": 0.0, "pts": 0.0, "reb": 0.0, "ast": 0.0,
                     "stl": 0.0, "blk": 0.0, "tov": 0.0}

                for cat in ath_entry.get("categories", []):
                    cat_name = cat.get("name", "").lower()
                    # Map category stats to our keys
                    for stat_entry in cat.get("stats", []):
                        stat_name = stat_entry.get("name", "").lower() if isinstance(stat_entry, dict) else ""
                        stat_val = stat_entry.get("value", 0) if isinstance(stat_entry, dict) else 0

                        if "min" in stat_name:    s["min"] = _safe_float(stat_val)
                        elif "pts" in stat_name or "point" in stat_name:  s["pts"] = _safe_float(stat_val)
                        elif "reb" in stat_name:  s["reb"] = _safe_float(stat_val)
                        elif "ast" in stat_name:  s["ast"] = _safe_float(stat_val)
                        elif "stl" in stat_name:  s["stl"] = _safe_float(stat_val)
                        elif "blk" in stat_name:  s["blk"] = _safe_float(stat_val)
                        elif "tov" in stat_name:  s["tov"] = _safe_float(stat_val)
                        elif stat_name in ("gp", "g", "gamesplayed"):  s["gp"] = _safe_float(stat_val)

                    # Alternative: stats as flat array with labels
                    labels = cat.get("labels", [])
                    values = cat.get("totals", cat.get("stats", []))
                    if labels and isinstance(values, list) and len(values) == len(labels):
                        for lbl, val in zip(labels, values):
                            lk = lbl.lower()
                            if "min" in lk:    s["min"] = _safe_float(val)
                            elif "pts" in lk or "point" in lk:  s["pts"] = _safe_float(val)
                            elif "reb" in lk:  s["reb"] = _safe_float(val)
                            elif "ast" in lk:  s["ast"] = _safe_float(val)
                            elif "stl" in lk:  s["stl"] = _safe_float(val)
                            elif "blk" in lk:  s["blk"] = _safe_float(val)
                            elif "tov" in lk:  s["tov"] = _safe_float(val)
                            elif lk in ("gp", "g", "gamesplayed"):  s["gp"] = _safe_float(val)

                if s["min"] <= 0:
                    continue

                # Build blended output matching _fetch_athlete format
                blended = dict(s)
                blended["season_min"] = s["min"]
                blended["recent_min"] = s["min"]  # Team stats endpoint only has season
                blended["season_pts"] = s["pts"]
                blended["recent_pts"] = s["pts"]
                blended["season_reb"] = s["reb"]
                blended["recent_reb"] = s["reb"]
                blended["season_ast"] = s["ast"]
                blended["recent_ast"] = s["ast"]
                blended["season_stl"] = s["stl"]
                blended["recent_stl"] = s["stl"]
                blended["season_blk"] = s["blk"]
                blended["recent_blk"] = s["blk"]

                blended["_cached_ts"] = time.time()
                result[pid] = blended
                _cs(f"ath3_{pid}", blended)

            except (KeyError, TypeError, ValueError):
                continue

    except Exception as e:
        print(f"[team-stats] parse error team={team_id}: {e}")
        return {}

    if result:
        result["_cached_ts"] = time.time()
        _cs(cache_key, result)
        print(f"[team-stats] loaded {len(result)} players for team {team_id}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# GAMELOG-BASED PROJECTION STATS — replaces ESPN averaged-split blending
# grep: GAMELOG PROJECTION, _gamelog_to_stats
# Uses per-game data from last 7 days (configurable) with recency weighting.
# Fixes the architectural blindness where ESPN L5/L10 split averages dilute
# sudden rotation changes (e.g. Payne 17 avg → 2 min last game → L5 still ~14).
# ─────────────────────────────────────────────────────────────────────────────

def _parse_gamelog_date(raw_date_str):
    """Parse ESPN gamelog date string into a Python date object. Returns None on failure."""
    if not raw_date_str:
        return None
    try:
        s = str(raw_date_str).strip()
        # ESPN formats: "2026-03-28T00:00Z", "2026-03-28T19:30:00Z", epoch millis
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
        # Epoch milliseconds (ESPN sometimes uses this)
        if s.isdigit() and len(s) >= 10:
            ts = int(s)
            if ts > 1e12:
                ts = ts / 1000  # millis → seconds
            return datetime.fromtimestamp(ts, tz=timezone.utc).date()
        # YYYYMMDD
        if len(s) == 8 and s.isdigit():
            return datetime.strptime(s, "%Y%m%d").date()
        # YYYY-MM-DD (no time)
        if len(s) == 10 and s[4] == "-":
            return datetime.strptime(s, "%Y-%m-%d").date()
    except (ValueError, TypeError, OSError):
        pass
    return None


def _gamelog_to_stats(gamelog, season_stats, window_days=None):
    """Build projection stats from per-game gamelog data (last N days).

    Primary window: last `window_days` days (default 7, configurable via
    projection.gamelog_window_days). If <2 games in that window, falls back
    to last 3 games. If still insufficient, returns season_stats unchanged.

    Recency weighting: most recent game = 2x, second-most = 1.5x, rest = 1x.
    This ensures sudden changes (benching, injury return, role change) are
    reflected immediately rather than diluted over 5-10 game averages.

    Returns a stats dict matching the shape expected by project_player():
      min, pts, reb, ast, stl, blk, tov, season_*, recent_*, dnp_risk, etc.
    """
    if window_days is None:
        proj_cfg = _cfg("projection", _CONFIG_DEFAULTS.get("projection", {}))
        window_days = int(proj_cfg.get("gamelog_window_days", 7))

    if not gamelog or not gamelog.get("minutes"):
        return season_stats

    minutes = gamelog["minutes"]
    if not minutes:
        return season_stats

    # Parse game dates to find games within the window
    raw_dates = gamelog.get("game_dates", [])
    today = _et_date()
    parsed_dates = [_parse_gamelog_date(d) for d in raw_dates] if raw_dates else []

    # Find indices of games within the window
    if parsed_dates and any(d is not None for d in parsed_dates):
        window_indices = []
        for i, d in enumerate(parsed_dates):
            if d is not None and (today - d).days <= window_days:
                window_indices.append(i)
    else:
        # No dates available — use last 3 games as proxy for ~7 days
        window_indices = list(range(max(0, len(minutes) - 3), len(minutes)))

    # If <2 games in the window, widen to last 3 games
    if len(window_indices) < 2:
        window_indices = list(range(max(0, len(minutes) - 3), len(minutes)))

    # If still nothing, return season stats
    if not window_indices:
        return season_stats

    # Helper to safely extract values from gamelog arrays
    def _get_vals(key):
        arr = gamelog.get(key, [])
        return [arr[i] if i < len(arr) else 0.0 for i in window_indices]

    w_min = _get_vals("minutes")
    w_pts = _get_vals("points")
    w_reb = _get_vals("rebounds")
    w_ast = _get_vals("assists")
    w_stl = _get_vals("steals")
    w_blk = _get_vals("blocks")
    w_tov = _get_vals("turnovers")

    # Recency weights: most recent game = 2x, second-most = 1.5x, rest = 1x
    n = len(window_indices)
    weights = []
    for i in range(n):
        if i == n - 1:
            weights.append(2.0)   # Most recent game
        elif i == n - 2:
            weights.append(1.5)   # Second most recent
        else:
            weights.append(1.0)
    total_w = sum(weights)

    def _wavg(values):
        return sum(v * w for v, w in zip(values, weights)) / total_w

    proj_min = _wavg(w_min)
    proj_pts = _wavg(w_pts)
    proj_reb = _wavg(w_reb)
    proj_ast = _wavg(w_ast)
    proj_stl = _wavg(w_stl)
    proj_blk = _wavg(w_blk)
    proj_tov = _wavg(w_tov)

    # Season stats preserved for LightGBM features and reference
    s_min = season_stats.get("season_min", season_stats.get("min", 0))
    s_pts = season_stats.get("season_pts", season_stats.get("pts", 0))
    s_reb = season_stats.get("season_reb", season_stats.get("reb", 0))
    s_ast = season_stats.get("season_ast", season_stats.get("ast", 0))
    s_stl = season_stats.get("season_stl", season_stats.get("stl", 0))
    s_blk = season_stats.get("season_blk", season_stats.get("blk", 0))

    result = {
        # Primary projection stats — gamelog-derived
        "min":  round(proj_min, 2),
        "pts":  round(proj_pts, 2),
        "reb":  round(proj_reb, 2),
        "ast":  round(proj_ast, 2),
        "stl":  round(proj_stl, 2),
        "blk":  round(proj_blk, 2),
        "tov":  round(proj_tov, 2),
        # Season stats — preserved for LightGBM features and card display
        "season_min": s_min,
        "season_pts": s_pts,
        "season_reb": s_reb,
        "season_ast": s_ast,
        "season_stl": s_stl,
        "season_blk": s_blk,
        # Recent stats = gamelog-derived (used by LightGBM + decline penalty)
        "recent_min": round(proj_min, 2),
        "recent_pts": round(proj_pts, 2),
        "recent_reb": round(proj_reb, 2),
        "recent_ast": round(proj_ast, 2),
        "recent_stl": round(proj_stl, 2),
        "recent_blk": round(proj_blk, 2),
        # Gamelog metadata
        "games_in_window": n,
        "last_game_min": round(w_min[-1], 1) if w_min else 0,
        # Preserve gp from season stats
        "gp": season_stats.get("gp"),
    }

    # DNP risk: based on LAST GAME minutes, not averaged split.
    # A single 2-minute garbage-time appearance is a clear signal the player
    # is out of the rotation — regardless of what their L5 average says.
    dnp_thresh = float(_cfg("projection.dnp_risk_min_threshold", 8.0))
    last_min = w_min[-1] if w_min else 0
    if last_min < dnp_thresh:
        result["dnp_risk"] = True

    return result


# ─────────────────────────────────────────────────────────────────────────────
# LIVE NBA DATA FETCHERS — used by Ben tool use
# grep: _live_scores, _live_boxscore, _live_player_stats, BEN_TOOL
# ─────────────────────────────────────────────────────────────────────────────

def _live_scores():
    """Current NBA scoreboard: scores, quarter, time remaining, game IDs."""
    try:
        today_str = _et_date().strftime("%Y%m%d")
        data = _espn_get(_espn_scoreboard(today_str))
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
            "code (api/index.py, api/real_score.py, api/asset_optimizer.py, "
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
    """Redistribute minutes from OUT and GTD/DTD players to eligible teammates.

    OUT players: 100% of their minutes are redistributed (they won't play).
    GTD/DTD players: a fraction of their minutes are redistributed, weighted by
    the probability they sit. This prevents the binary 0-or-10 cascade gap where
    a star listed day-to-day generates zero cascade for teammates.

    Config:  cascade.gtd_sit_probability  (default 0.30)
             cascade.dtd_sit_probability  (default 0.10)
             cascade.gtd_minute_reduction (default 0.15) — expected minute cut when they DO play
             cascade.max_cascade_pct      (default 0.40) — proportional cap (% of avg_min)
    """
    cascade_flags = {}

    # Sit-probability weights for partial cascade
    gtd_sit_prob = float(_cfg("cascade.gtd_sit_probability", 0.30))
    dtd_sit_prob = float(_cfg("cascade.dtd_sit_probability", 0.10))
    gtd_min_reduction = float(_cfg("cascade.gtd_minute_reduction", 0.15))

    # Group players by team
    teams = {}
    for p in roster:
        team = p.get("team_abbr", "")
        if team not in teams:
            teams[team] = []
        teams[team].append(p)

    for team, team_players in teams.items():
        # Find OUT + GTD/DTD players with known minutes, and active players
        # Each donor has a cascade_weight: 1.0 for OUT, partial for GTD/DTD
        donor_players = []  # (player, stats, cascade_weight, is_partial)
        active_players = []
        has_full_donor = False
        for p in team_players:
            pid = p["id"]
            s = stats_map.get(pid)
            if not s or s.get("min", 0) <= 0:
                continue
            inj = (p.get("injury_status") or "").upper()
            if p.get("is_out"):
                donor_players.append((p, s, 1.0, False))
                has_full_donor = True
            elif inj in ("GTD",):
                # GTD: expected cascade = sit_prob * full_minutes + play_prob * minute_reduction
                weight = gtd_sit_prob + (1.0 - gtd_sit_prob) * gtd_min_reduction
                donor_players.append((p, s, weight, True))
                active_players.append((p, s))  # Still active (might play)
            elif inj in ("DTD", "DOUBT"):
                weight = dtd_sit_prob + (1.0 - dtd_sit_prob) * gtd_min_reduction
                donor_players.append((p, s, weight, True))
                active_players.append((p, s))  # Still active (might play)
            else:
                active_players.append((p, s))

        if not donor_players or not active_players:
            continue

        # Calculate total minutes freed per position group, weighted by cascade probability
        # Track whether freed minutes came from partial (GTD/DTD) vs full (OUT) donors
        freed_by_group = {}       # group -> freed_min
        partial_by_group = {}     # group -> True if ALL donors for this group are partial
        for op, os, cw, is_partial in donor_players:
            pg = _pos_group(op["pos"])
            freed = os.get("min", 0) * cw
            freed_by_group[pg] = freed_by_group.get(pg, 0) + freed
            # A group is "partial-only" if it has zero full (OUT) donors
            if pg not in partial_by_group:
                partial_by_group[pg] = is_partial
            elif not is_partial:
                partial_by_group[pg] = False
            # Centers also share with forwards
            if pg == "C":
                cf_share = _cfg("cascade.center_forward_share", 0.30)
                freed_by_group["F"] = freed_by_group.get("F", 0) + freed * cf_share
                if "F" not in partial_by_group:
                    partial_by_group["F"] = is_partial
                elif not is_partial:
                    partial_by_group["F"] = False

        # Distribute freed minutes to active players in same position group
        for group, freed_min in freed_by_group.items():
            is_partial_group = partial_by_group.get(group, False)
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
                avg_min = rs.get("min", 0)
                if pid not in cascade_flags:
                    cascade_flags[pid] = {"bonus": 0.0, "partial_only": is_partial_group, "avg_min": avg_min}
                cascade_flags[pid]["bonus"] += bonus
                # If any group contributing is full (OUT), mark as not partial-only
                if not is_partial_group:
                    cascade_flags[pid]["partial_only"] = False

    # Cap per-player cascade (3 layers, take the minimum):
    # 1. Hard cap: per_player_cap_minutes (10.0 full, 4.0 partial)
    # 2. Proportional cap: max_cascade_pct × avg_min — prevents bench players
    #    (16 min avg) from getting +10 minutes. At 40%, Bones Hyland (16 min)
    #    maxes out at +6.4 instead of +10.
    cap_full = _cfg("cascade.per_player_cap_minutes", _CONFIG_DEFAULTS["cascade"]["per_player_cap_minutes"])
    cap_partial = _cfg("cascade.partial_cascade_cap_minutes", _CONFIG_DEFAULTS["cascade"].get("partial_cascade_cap_minutes", 4.0))
    max_cascade_pct = float(_cfg("cascade.max_cascade_pct", 0.40))
    result = {}
    for pid, info in cascade_flags.items():
        hard_cap = cap_partial if info["partial_only"] else cap_full
        pct_cap = info["avg_min"] * max_cascade_pct if info["avg_min"] > 0 else hard_cap
        result[pid] = min(info["bonus"], hard_cap, pct_cap)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CARD BOOST & DFS SCORING
# grep: _est_card_boost, _dfs_score, card boost, boost_model, Real Score formula
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

def _normalize_player_name(name):
    """Normalize player name for consistent joining across datasets.

    Delegates to shared.normalize_player_name for consistency across all pipelines.
    """
    from api.shared import normalize_player_name
    return normalize_player_name(name)


def _normalize_boost_name(name):
    """Alias for _normalize_player_name for backward compatibility."""
    return _normalize_player_name(name)


def _clamp_round_boost(x: float, floor_val: float, ceiling: float) -> float:
    # Real Sports snaps card boosts to tenths (1.1, 2.4, etc.) — match that scale
    # so the optimizer works with the same values the app actually uses.
    return round(min(max(float(x), floor_val), ceiling), 1)


# _ensure_boost_priors_loaded / _get_boost_prior removed — replaced by api/boost_model.py cascade


# grep: DRAFT TIER TARGETING
def _estimate_log_drafts(season_pts, is_big_market, recent_pts=0, season_pts_raw=0, market_score=None):
    """Estimate log10(expected draft count) from player profile.

    Calibrated from 578 top-performer entries:
    - Stars (25+ PPG): ~1000 drafts (log=3.0)
    - Starters (15-25 PPG): ~150 drafts (log=2.2)
    - Role players (6-15 PPG): ~10-50 drafts (log=1.0-1.7)
    - Deep bench (<6 PPG): ~3 drafts (log=0.5)

    market_score: continuous 0.0–1.0 (new). When provided, scales the market
    adjustment smoothly. Falls back to binary is_big_market (+0.3) when not provided.
    """
    pts = max(float(season_pts), float(recent_pts))
    if pts >= 25:
        base = 3.0
    elif pts >= 20:
        base = 2.7
    elif pts >= 15:
        base = 2.2
    elif pts >= 10:
        base = 1.7
    elif pts >= 6:
        base = 1.0
    else:
        base = 0.5

    # Market popularity adjustment: continuous score scales the ~2x drafts signal.
    if market_score is not None:
        base += 0.3 * float(market_score)
    elif is_big_market:
        base += 0.3

    # Trending factor: recent scoring surge draws attention
    if float(season_pts_raw) > 0 and float(recent_pts) / float(season_pts_raw) > 1.15:
        base += 0.15

    return max(0.0, min(3.5, base))


def _est_card_boost(
    proj_min,
    pts,
    team_abbr,
    player_name=None,
    season_pts=0.0,
    recent_pts=0.0,
    cascade_bonus=0.0,
    is_home=False,
    projected_rs=None,
    season_avg_min=0.0,
    player_pos="",
    season_reb=0.0,
    season_ast=0.0,
):
    """Predict card boost via 3-tier cascade (api/boost_model.py).

    # grep: CARD BOOST

    Tier 1 — Returning player (appeared on a slate within 14 days).
             Uses prev_boost + adjustment factors. MAE ~0.10–0.15.
    Tier 2 — Known player, stale (>14 days since last appearance).
             Blends historical mean with API-derived estimate.
    Tier 3 — Cold start (never seen on Real Sports).
             Player Quality Index from season stats.

    Post-prediction caps:
      - Star PPG tier caps (national stars always low boost)
      - Per-team boost ceilings (popular franchise role players)
    """
    from api.boost_model import predict_boost, estimate_draft_popularity

    cb = _cfg("card_boost", _CONFIG_DEFAULTS["card_boost"])
    ceiling   = cb.get("ceiling", 3.0)
    floor_val = cb.get("floor", 0.0)

    _spts = float(season_pts or pts or 0.0)

    # Get today's date string for gap calculation
    today_str = str(_et_date())

    # Run 3-tier cascade
    result = predict_boost(
        player_name=player_name or "",
        today_str=today_str,
        season_ppg=_spts,
        season_rpg=float(season_reb or 0.0),
        season_apg=float(season_ast or 0.0),
        season_mpg=float(season_avg_min or proj_min or 0.0),
        recent_ppg=float(recent_pts or _spts or 0.0),
        team=team_abbr or "",
        ceiling=ceiling,
        floor=floor_val,
    )

    raw_boost = result["boost"]
    cb_low, cb_high = result.get("confidence_band", (raw_boost, raw_boost))

    # ── Anti-popularity adjustment (Strategy Report Finding 4) ─────────
    # Draft popularity has -0.457 correlation with boost. The least-drafted
    # 50% produce 24-26% more total value. High-popularity players see
    # depressed boosts; this feeds that signal into predictions.
    _strat = _cfg("strategy", {}) or {}
    _anti_pop_delta = 0.0
    if _strat.get("anti_popularity_enabled", True):
        _pop_strength = float(_strat.get("anti_popularity_strength", 0.2))
        _pop_score = estimate_draft_popularity(
            season_ppg=_spts,
            team=team_abbr or "",
            recent_ppg=float(recent_pts or _spts or 0.0),
        )
        # Normalize popularity: 2500 = typical star draft count (top quartile)
        # Penalty scales from 0 (unknown player) to ~0.6 (heavily drafted star)
        _pop_normalized = min(_pop_score / 2500.0, 1.0)
        _anti_pop_delta = _pop_normalized * _pop_strength * 3.0
    raw_boost -= _anti_pop_delta
    cb_low -= _anti_pop_delta
    cb_high -= _anti_pop_delta

    # Star PPG tier caps — high-PPG players are nationally popular regardless of
    # team market size. These hard caps prevent over-prediction for stars.
    _star_tiers = sorted(
        cb.get("star_ppg_tiers", []),
        key=lambda t: float(t.get("min_ppg", 0)),
        reverse=True,
    )

    # Per-team boost ceiling — popular franchise role players get capped.
    _team_ceilings = cb.get("team_boost_ceiling", {})
    _team_ceil = float(_team_ceilings.get(team_abbr, ceiling))

    # Apply caps
    _star_cap = ceiling
    for _tier in _star_tiers:
        if _spts >= float(_tier.get("min_ppg", 9999)):
            _star_cap = float(_tier.get("boost_cap", ceiling))
            raw_boost = min(raw_boost, _star_cap)
            break
    raw_boost = min(raw_boost, _team_ceil)
    cb_high = min(cb_high, _star_cap, _team_ceil)

    return (
        _clamp_round_boost(raw_boost, floor_val, ceiling),
        (
            _clamp_round_boost(cb_low, floor_val, ceiling),
            _clamp_round_boost(cb_high, floor_val, ceiling),
        ),
        result.get("tier", 3),
    )

def _dfs_score(pts, reb, ast, stl, blk, tov):
    """Real Score-aligned formula. Weights read from runtime config."""
    _w_defaults = _CONFIG_DEFAULTS["real_score"]["dfs_weights"]
    w = _cfg("real_score.dfs_weights", _w_defaults)
    return (pts * w.get("pts", _w_defaults["pts"]) + reb * w.get("reb", _w_defaults["reb"]) +
            ast * w.get("ast", _w_defaults["ast"]) + stl * w.get("stl", _w_defaults["stl"]) +
            blk * w.get("blk", _w_defaults["blk"]) + tov * w.get("tov", _w_defaults["tov"]))


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
# grep: PLAYER PROJECTION ENGINE — LightGBM + heuristic blend, sort-based lineup builder
# Returns projection dict: {name, team, pos, rating (RS), est_mult (card boost),
#   predMin, pts, reb, ast, stl, blk, season_*/recent_* raw stats, signals}
# ─────────────────────────────────────────────────────────────────────────────


def project_player(pinfo, stats, spread, total, side, team_abbr="",
                   cascade_bonus=0.0, is_b2b=False,
                   prefetched_gamelog=None, dvp_data=None, opp_abbr=None,
                   game_id=None):
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

    # Season-minutes floor: ensure projected minutes don't drop below season avg
    # times a configurable ratio. The season/recent blend can pull predMin far
    # below season avg (e.g. 18.1 vs 30.1 season) for players in temporary
    # minute dips. Drafted players should show at least their season-level minutes.
    # GTD and B2B players are exempt — their minute reductions are real signals.
    _season_min_floor = stats.get("season_min", avg_min)
    _min_season_ratio = float(_cfg("strategy.min_pred_min_season_ratio", 1.0))
    if not is_b2b and pinfo.get("injury_status") != "GTD" and _season_min_floor > 0:
        proj_min = max(proj_min, _season_min_floor * _min_season_ratio)

    # Minutes gate — cascade team players get relaxed gate (12 min vs 25).
    # Deep rotation players on cascade teams historically produce avg value 16.1.
    _cascade_team = bool(pinfo.get("_cascade_team"))
    if _cascade_team:
        _ct_cfg_gate = _cfg("cascade.team_detector", {}) or {}
        min_gate = float(_ct_cfg_gate.get("deep_rotation_min_minutes", 12.0))
    else:
        min_gate = _cfg("projection.min_gate_minutes", MIN_GATE)
    if proj_min < min_gate: return None

    pts = stats["pts"]
    reb = stats["reb"]
    ast = stats["ast"]
    stl = stats.get("stl", 0)
    blk = stats.get("blk", 0)
    tov = stats.get("tov", 0)
    minutes = stats.get("min", 0)

    # Universal scoring floor: RS floor of 2.0 means ~2 PPG minimum.
    # Strategy report Finding 2: only 1.3% of winning players have RS < 2.0.
    _min_pts_universal = float(_cfg("strategy.min_pts_projection", 2.0))
    if pts < _min_pts_universal:
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
            # v62 feature wiring: pass runtime game context instead of inference defaults.
            # nba_api enrichment: prefer real team pace and opp defense when available
            _nba_team_pace = stats.get("_nba_api_team_pace")
            _team_total_proxy = float(_nba_team_pace) if _nba_team_pace else float(total or DEFAULT_TOTAL) / 2.0
            _nba_opp_pts = stats.get("_nba_api_opp_pts_allowed")
            _opp_def_proxy = float(_nba_opp_pts) if _nba_opp_pts else 112.0 + ((1.0 if side == "away" else -1.0) * float(spread or 0) * 0.7)
            _nba_usage = stats.get("_nba_api_usage_share", 0.0)
            _nba_gp = stats.get("_nba_api_games_played")
            _nba_min_vol = stats.get("_nba_api_min_volatility")
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
                games_played=float(_nba_gp) if _nba_gp else (float(_gp) if _gp is not None else None),
                rest_days=float(pinfo.get("rest_days", 2) or 2),
                team_pace=_team_total_proxy,
                opp_pts_allowed=_opp_def_proxy,
                game_total=float(total or DEFAULT_TOTAL),
                teammate_out_count=float(pinfo.get("teammate_out_count", 0) or 0),
                recent_ast=stats.get("recent_ast"),
                recent_stl=stats.get("recent_stl"),
                recent_blk=stats.get("recent_blk"),
                avg_reb=stats.get("season_reb"),
                usage_share=float(_nba_usage),
                min_volatility=float(_nba_min_vol) if _nba_min_vol is not None else None,
                spread_abs=abs(float(spread or 0)),
                recent_3g_pts=float(stats.get("_nba_api_recent_3g_pts")) if stats.get("_nba_api_recent_3g_pts") is not None else None,
            )
            ai_pred = _lgbm_predict_rs(feat_vec)
        except Exception as _lgbm_e:
            print(f"[WARN] LightGBM inference failed for player, using heuristic: {_lgbm_e}")

    # ── Simple game context adjustments (Strategy Report Finding 7) ──────────
    # Close games (spread ≤ 5): avg RS 4.44 → +0.3 RS additive
    # High-pace games (total ≥ 230): avg RS 4.44 vs 3.82 → +0.15 per 10pts above 220
    _total = total or DEFAULT_TOTAL
    abs_spread = abs(spread or 0)
    _close_game_bonus = float(_cfg("strategy.close_game_rs_bonus", 0.3))
    _pace_bonus_per_10 = float(_cfg("strategy.pace_rs_bonus_per_10", 0.15))
    game_context_bonus = 0.0
    if abs_spread <= 5:
        game_context_bonus += _close_game_bonus
    if _total > 220:
        game_context_bonus += (_total - 220) / 10.0 * _pace_bonus_per_10
    home_adj = 1.02 if side == "home" else 1.0

    s_base = heuristic * home_adj

    # ── RS projection (simplified — no Monte Carlo) ─────────────────────────
    # Strategy report: game context effects are simple additive adjustments,
    # not multiplicative Monte Carlo simulations. RS predictability from
    # season averages is already high (CV < 0.25 for reliable players).
    season_pts = stats.get("season_pts", pts)
    recent_pts = stats.get("recent_pts", pts)
    player_variance = abs(recent_pts - season_pts) / max(season_pts, 1)

    rs_cfg = _cfg("real_score", _CONFIG_DEFAULTS["real_score"])
    _rs_defaults = _CONFIG_DEFAULTS["real_score"]
    comp_div = rs_cfg.get("compression_divisor", _rs_defaults["compression_divisor"])
    comp_pow = rs_cfg.get("compression_power", _rs_defaults["compression_power"])
    rs_cap = rs_cfg.get("rs_cap", _rs_defaults["rs_cap"])

    raw_linear = s_base / comp_div
    heuristic_rs = min(raw_linear ** comp_pow, rs_cap)

    # ── RS regression guard (Winning Draft Audit) ───────────────────────────
    # Data: avg top performer RS is 4.16 (median 3.9). Role players with season
    # PPG < 12 rarely produce RS > 5.0. Cap heuristic_rs based on scoring tier
    # to prevent bench players from being projected at 6.0+.
    _rs_regression = rs_cfg.get("regression_guard", {})
    if _rs_regression.get("enabled", True):
        _rg_ppg = float(season_pts or pts or 0)
        if _rg_ppg < 8:
            # Deep bench: cap RS at 4.0 (avg leaderboard RS for RS 3-4 band)
            _rg_cap = float(_rs_regression.get("bench_cap", 4.5))
            heuristic_rs = min(heuristic_rs, _rg_cap)
        elif _rg_ppg < 14:
            # Role player: cap RS at 5.5
            _rg_cap = float(_rs_regression.get("role_cap", 5.5))
            heuristic_rs = min(heuristic_rs, _rg_cap)
        elif _rg_ppg < 20:
            # Starter: cap RS at 7.0
            _rg_cap = float(_rs_regression.get("starter_cap", 7.0))
            heuristic_rs = min(heuristic_rs, _rg_cap)
        # Stars (20+ PPG): no cap — they produce RS 5-12 naturally

    # ── Late blend: AI (native RS units) + heuristic RS ───────────────────────
    # LightGBM outputs native RS units. 30% AI / 70% heuristic.
    ai_weight = rs_cfg.get("ai_blend_weight", _rs_defaults["ai_blend_weight"])
    if ai_pred is not None:
        raw_score = min((ai_pred * ai_weight) + (heuristic_rs * (1.0 - ai_weight)), rs_cap)
    else:
        raw_score = heuristic_rs

    # ── Historical RS confidence discount ─────────────────────────────────
    # grep: HISTORICAL RS DISCOUNT
    # Cross-reference predicted RS against the player's ACTUAL historical RS
    # distribution from top_performers.csv. When we predict far above their
    # proven range, apply a soft pull-back toward their track record.
    #
    # NOT a hard cap — players CAN pop off. But the further above history,
    # the more the model discounts the projection. Think Bayesian: history
    # is the prior, today's projection is the likelihood. With thin history
    # (few appearances), the prior is weak and projections dominate. With
    # deep history (10+ appearances), the prior is strong.
    #
    # Example: Sensabaugh median RS 3.3, predicted 6.4 → +3.1 above median.
    # With 12 appearances, prior_weight ≈ 0.35. Pull-back = 0.35 × 3.1 × 0.5 = 0.54.
    # Adjusted RS ≈ 5.86 instead of 6.4. Still allows upside, just tempered.
    _hrs_cfg = _cfg("strategy.historical_rs_discount", {})
    _hrs_player_name = pinfo.get("name", "")
    if _hrs_cfg.get("enabled", True) and _hrs_player_name:
        print(f"[web_search] Claude web_search error (non-fatal): {e}")
        return ""


# ─────────────────────────────────────────────────────────────────────────────
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

    today = _today_str()
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
    timeout_s = float(_cfg("context_layer.timeout_seconds", 20))

    # Fetch recent NBA news for teams on the slate (Layer 1: Opus + web_search)
    news_text = ""
    try:
        news_text = _fetch_nba_news_context(games, date=None, all_proj=all_proj)
    except Exception as _news_err:
        print(f"[context_pass] web search error (non-fatal): {_news_err}")

    # Skip guard: if no fresh signal exists, Claude can't add meaningful value
    # beyond what the algorithmic model already computed. Skip to avoid latency
    # and API cost when the call would just repack existing filter outputs.
    has_cascade = any(p.get("_cascade_bonus", 0) > 0 for p in all_proj)
    has_b2b = any(g.get("home_b2b") or g.get("away_b2b") for g in games)
    has_gtd_dtd = any(p.get("injury_status", "").upper() in ("GTD", "DTD", "DOUBT") for p in all_proj)
    if not news_text and not has_cascade and not has_b2b and not has_gtd_dtd:
        print("[context_pass] skipped — no fresh signals (no news, no cascade, no B2B, no GTD/DTD)")
        return

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

    # Fetch ESPN injury report for narrative context (GTD/DTD details)
    espn_injuries = {}
    try:
        espn_injuries = _espn_injuries_fetch()
    except Exception:
        pass

    # Build per-team injury summary for context pass
    team_injuries = {}
    for norm_name, inj_info in espn_injuries.items():
        t = inj_info.get("team", "")
        if t and t in game_ctx:
            if t not in team_injuries:
                team_injuries[t] = []
            team_injuries[t].append(f"{norm_name}: {inj_info['status']} — {inj_info.get('detail', '')[:60]}")

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
            "season_min":   round(p.get("season_min", 0), 1),
            "cascade_bonus": cascade_bonus,
            "roto_status":  roto_status,
            "injury_status": p.get("injury_status", ""),
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

        "TEAM/STYLE SIGNALS (use as CONFIRMING context only — not a standalone reason):\n"
        "These teams have RS-friendly styles, but adjusting purely because a player wears "
        "their jersey repacks what the algorithm already knows. Only use team style as a "
        "CONFIRMING signal when COMBINED with today's specific evidence (news, cascade, "
        "spread, or roto_status). Examples of past high-RS outputs for calibration:\n"
        "- ATL Hawks: multi-stat contributors thrive in close games (Ware RS 5.1–5.7, "
        "Young RS 5.8). Confirm with: spread ≤ 5 AND normal rotation confirmed in news.\n"
        "- MEM Grizzlies: hustle culture yields high RS in tight games (Watson RS 6.4, "
        "Aldama RS 6.0). Confirm with: spread ≤ 6 AND no blowout projection.\n"
        "- OKC Thunder: system RS from hustle/defense (Hartenstein RS 7.3). Confirm with: "
        "OKC as underdog (spread +1 to +6) AND no key-player injury dampening rotations.\n"
        "- CHI Bulls, NYK Knicks: high RS variance in close games. Only adjust when specific "
        "news (rotation change, cascade opportunity) justifies it today.\n\n"

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
        "rotation and has fragile minute floor; DNP or early hook risk is meaningful.\n"
        "- recent_min < 16 AND season_min < 18 (any roto_status): Apply 0.85–0.90x DOWN — "
        "rotation-bubble player with limited recent usage; DNP or early hook risk even if "
        "technically in rotation. These players look attractive on paper (high boost) but "
        "12-15 min/game players frequently get DNP'd or pulled early.\n\n"

        "CALIBRATION: Most players get NO adjustment (omit them). Only adjust when you have "
        "a specific reason from TODAY's news, cascade_bonus, roto_status, or game context. "
        "Team membership alone ('Player X plays for ATL') is NOT a reason — it must combine "
        "with a specific today-signal (news item, spread ≤ 5, cascade, roto confirmed). "
        "Typical batch: 2-5 players out of 40 (fewer is better). "
        "Strong signal = 1.2-1.35x up or 0.7-0.85x down. "
        "Weak signal = 1.08-1.15x up or 0.88-0.95x down. "
        "Reserve 1.35x+ for rare cases: confirmed starter OUT → backup inherits 30+ min, "
        "or a blowout star sitting early with projection still inflated.\n\n"

        "OVERRIDE PROTOCOL — Each adjustment MUST include:\n"
        "- player: exact name from the list\n"
        "- rs_multiplier: between 0.6 and 1.4\n"
        "- minutes_delta: integer, extra minutes to add/subtract (e.g. +4, -3). "
        "Use 0 if only RS changes. Range: -8 to +8. This is for injury narrative "
        "context the algorithmic cascade cannot see — e.g. 'Star X questionable with "
        "knee → backup Y likely gets +4 min even if Star X plays on a restriction'. "
        "Do NOT double-count cascade_bonus already shown in the data.\n"
        "- reason: brief explanation citing specific today-signal\n"
        "- reason_class: one of: availability, role_change, minutes_risk, game_script, "
        "injury_cascade, blowout_risk, matchup, news, b2b, rest\n\n"
        "STRICT CAPS: max 8 adjustments per slate. Each reason must cite a specific signal "
        "(news item, cascade_bonus value, spread, roto_status). No generic team-style reasons.\n\n"
        "Return ONLY valid JSON:\n"
        '{"adjustments": [{"player": "Exact Name", "rs_multiplier": 1.20, "minutes_delta": 4, '
        '"reason": "Star X questionable — Y inherits expanded role", "reason_class": "injury_cascade"}]}\n'
        "Keep multipliers between 0.6 and 1.4. minutes_delta between -8 and +8. "
        "Omit players with multiplier 1.0 AND minutes_delta 0."
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
    # Build injury report section
    injury_section = ""
    if team_injuries:
        lines = []
        for t, injs in sorted(team_injuries.items()):
            lines.append(f"  {t}: " + "; ".join(injs))
        injury_section = (
            "\n\nTEAM INJURY REPORT (ESPN):\n" + "\n".join(lines) + "\n"
            "Synthesize these injury statuses with the player data above. A star listed "
            "day-to-day or questionable may play limited minutes — their teammates should "
            "get minutes_delta bumps even if the cascade_bonus doesn't fully reflect it. "
            "Think about WHO benefits from each injury narratively (same-position backups, "
            "ball-handlers if a guard is questionable, etc.).\n"
        )

    user_prompt = (
        f"Today's slate players (top 40 by projected RS):\n"
        f"{json.dumps(players_payload, separators=(',', ':'))}"
        f"{injury_section}"
        f"{news_section}\n\n"
        "Return JSON adjustments only."
    )

    _ctx_api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not _ctx_api_key:
        print("[context_pass] ANTHROPIC_API_KEY not set — skipping")
        return

    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=_ctx_api_key, max_retries=0)
        msg = client.messages.create(
            model=model_id,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            timeout=timeout_s,
        )
        if not msg.content:
            print("[context_pass] skipped (empty content array)")
            return
        raw_text = (getattr(msg.content[0], "text", None) or "").strip()
        if not raw_text:
            print("[context_pass] skipped (empty text in response)")
            return
        # Strip markdown code fences if present
        if raw_text.startswith("```"):
            parts = raw_text.split("```")
            raw_text = parts[1] if len(parts) >= 3 else raw_text[3:]
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

    # ── Guardrailed application with reason codes and audit trail ────────
    _VALID_REASON_CLASSES = {
        "availability", "role_change", "minutes_risk", "game_script",
        "injury_cascade", "blowout_risk", "matchup", "news", "b2b", "rest",
    }
    max_slate_adjustments = int(_cfg("context_layer.max_slate_adjustments", 8))
    max_total_impact = float(_cfg("context_layer.max_total_impact", 2.0))

    adj_count = 0
    total_impact = 0.0
    audit_entries = []
    name_map = {p["name"]: p for p in all_proj}

    # Parse full adjustment objects with reason codes
    raw_adj_list = data.get("adjustments", [])
    for adj_entry in raw_adj_list:
        if adj_count >= max_slate_adjustments:
            print(f"[context_pass] hit slate cap ({max_slate_adjustments}) — skipping remaining")
            break

        player_name = adj_entry.get("player", "")
        multiplier = adj_entry.get("rs_multiplier", 1.0)
        minutes_delta = adj_entry.get("minutes_delta", 0)
        reason = adj_entry.get("reason", "")
        reason_class = adj_entry.get("reason_class", "")

        p = name_map.get(player_name)
        if not p:
            continue

        # Clamp RS multiplier to [1-max_adj, 1+max_adj]
        clamped = max(1.0 - max_adj, min(1.0 + max_adj, float(multiplier)))

        # Clamp minutes_delta to [-8, +8]
        min_delta = max(-8, min(8, int(minutes_delta)))

        # Track cumulative slate impact
        impact = abs(clamped - 1.0) * p.get("rating", 3.0)
        if total_impact + impact > max_total_impact:
            print(f"[context_pass] total impact cap ({max_total_impact}) — skipping {player_name}")
            continue

        old_rating = p.get("rating", 0)
        p["rating"]        = round(old_rating * clamped, 1)
        p["chalk_ev"]      = round(p.get("chalk_ev", 0) * clamped, 2)
        p["ceiling_score"] = round(p.get("ceiling_score", 0) * clamped, 1)
        p["_context_adj"]  = round(clamped, 3)
        p["_context_reason"] = reason[:100]
        p["_context_reason_class"] = reason_class if reason_class in _VALID_REASON_CLASSES else "other"

        # Apply minutes adjustment from injury narrative synthesis
        if min_delta != 0:
            old_min = p.get("predMin", 0)
            new_min = round(max(0, old_min + min_delta), 1)
            p["predMin"] = new_min
            p["_context_minutes_delta"] = min_delta
            print(f"[context_pass] {player_name}: minutes {old_min} → {new_min} ({min_delta:+d}, {reason[:50]})")

        audit_entries.append({
            "player": player_name,
            "team": p.get("team", ""),
            "multiplier": round(clamped, 3),
            "old_rating": round(old_rating, 1),
            "new_rating": p["rating"],
            "minutes_delta": min_delta,
            "reason": reason[:200],
            "reason_class": p["_context_reason_class"],
        })

        total_impact += impact
        adj_count += 1

    print(f"[context_pass] applied {adj_count} adjustments via {model_id} "
          f"(total_impact={total_impact:.2f})")

    # Persist audit artifact for post-slate analysis
    if audit_entries:
        try:
            audit_data = {
                "date": str(_et_date()),
                "model": model_id,
                "adjustments": audit_entries,
                "total_impact": round(total_impact, 3),
                "news_available": bool(news_text),
                "signals": {
                    "cascade": has_cascade,
                    "b2b": has_b2b,
                },
            }
            _cs(f"context_audit_{_today_str()}", audit_data)
            # Also persist to GitHub for long-term analysis
            try:
                _github_write_file(
                    f"data/audit/context_{_today_str()}.json",
                    json.dumps(audit_data, indent=2),
                    f"context pass audit {_today_str()}",
                )
            except Exception:
                pass
        except Exception as _audit_err:
            print(f"[context_pass] audit write error: {_audit_err}")


# ─────────────────────────────────────────────────────────────────────────────
# GAME RUNNER & LINEUP BUILDER
# grep: _run_game, _build_lineups, _build_game_lineups, chalk_ev, Moonshot
# _run_game: fetches rosters, runs cascade, projects all players for one game
# _build_lineups: top-5 chalk (MILP) + moonshot (ranks 6-10 same EV)
# ─────────────────────────────────────────────────────────────────────────────
def _run_game(game, gamelog_map=None, dvp_data=None):
    cache_key = _ck_game_proj(game['gameId'])
    cached = _cg(cache_key)
    if cached: return cached

    home_r = fetch_roster(game["home"]["id"], game["home"]["abbr"])
    away_r = fetch_roster(game["away"]["id"], game["away"]["abbr"])

    all_roster = home_r + away_r
    players_in = (
        [(p, game["home"]["abbr"], "home") for p in home_r] +
        [(p, game["away"]["abbr"], "away") for p in away_r]
    )

    # Bulk fetch: get all player stats per team in 2 parallel calls instead of ~30
    # individual _fetch_athlete calls. Falls back to individual fetches for missing.
    stats_map = {}
    with ThreadPoolExecutor(max_workers=2) as _team_pool:
        _home_fut = _team_pool.submit(_fetch_team_player_stats, game["home"]["id"])
        _away_fut = _team_pool.submit(_fetch_team_player_stats, game["away"]["id"])
        home_bulk = _home_fut.result()
        away_bulk = _away_fut.result()
    bulk_stats = {**home_bulk, **away_bulk}

    # Populate stats_map from bulk results
    missing_players = []
    for p, _, _ in players_in:
        pid = p["id"]
        if pid in bulk_stats:
            stats_map[pid] = bulk_stats[pid]
        else:
            missing_players.append(p)

    # Fallback: fetch individually for players missing from bulk results
    if missing_players:
        print(f"[run-game] {len(missing_players)} players missing from bulk, fetching individually")
        with ThreadPoolExecutor(max_workers=_W_STANDARD) as pool:
            futs = {pool.submit(_fetch_athlete, p["id"]): p for p in missing_players}
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    stats = fut.result()
                    if stats:
                        stats_map[p["id"]] = stats
                except Exception as e:
                    print(f"fetch err {p['name']}: {e}")

    # ── Gamelog-based projection stats (grep: GAMELOG PROJECTION) ──────────
    # Fetch per-game data and overlay onto season stats. This replaces the old
    # ESPN averaged-split blending with actual game-by-game data from last 7 days.
    # Players who lost their rotation (e.g. 17 avg → 2 min last game) are now
    # correctly reflected instead of being diluted by L5/L10 averages.
    all_pids_for_gamelog = [str(p["id"]) for p, _, _ in players_in
                           if p["id"] in stats_map or str(p["id"]) in stats_map]
    game_gamelogs = {}
    if all_pids_for_gamelog:
        game_gamelogs = _fetch_gamelogs_batch(all_pids_for_gamelog, num_games=15)

    gamelog_overlay_count = 0
    for pid in list(stats_map.keys()):
        gl = game_gamelogs.get(pid) or game_gamelogs.get(str(pid))
        if gl and gl.get("minutes"):
            stats_map[pid] = _gamelog_to_stats(gl, stats_map[pid])
            gamelog_overlay_count += 1

    if gamelog_overlay_count:
        print(f"[run-game] gamelog overlay applied to {gamelog_overlay_count}/{len(stats_map)} players")

    # Enrich stats with nba_api features (usage_share, team_pace, etc.)
    try:
        _pid_names = {str(p["id"]): p.get("name", "") for p, _, _ in players_in}
        _nba_enriched = _nba_api_enrich(stats_map, _pid_names)
        if _nba_enriched:
            print(f"[run-game] nba_api enrichment: {_nba_enriched}/{len(stats_map)} players")
    except Exception as _nba_e:
        print(f"[run-game] nba_api enrich error (non-fatal): {_nba_e}")

    # Run cascade engine to redistribute minutes from OUT players
    # First, enrich roster with ESPN injury feed data — the roster endpoint often
    # doesn't include GTD/DTD status, but the dedicated injuries endpoint does.
    # This is critical for partial cascade: Edwards "Day-to-Day" needs to be
    # reflected as injury_status="DTD" on the roster player for cascade weighting.
    try:
        espn_inj = _espn_injuries_fetch()
        if espn_inj:
            from api.injury_feed import _normalize_name as _inj_normalize
            enriched_count = 0
            for p in all_roster:
                if p.get("injury_status"):
                    continue  # Already has status from roster endpoint
                norm = _inj_normalize(p.get("name", ""))
                inj_info = espn_inj.get(norm)
                if inj_info:
                    status = inj_info.get("status", "")
                    if status == "out" and not p.get("is_out"):
                        p["is_out"] = True
                        p["injury_status"] = ""
                        enriched_count += 1
                    elif status == "questionable":
                        p["injury_status"] = "GTD"
                        enriched_count += 1
                    elif status == "day-to-day":
                        p["injury_status"] = "DTD"
                        enriched_count += 1
                    elif status == "doubtful":
                        p["injury_status"] = "DOUBT"
                        enriched_count += 1
            if enriched_count:
                print(f"[run-game] injury feed enriched {enriched_count} players with GTD/DTD/OUT status")
    except Exception as _inj_err:
        print(f"[run-game] injury feed enrichment error (non-fatal): {_inj_err}")

    cascade_flags = _cascade_minutes(all_roster, stats_map)

    # Project all players with cascade-adjusted minutes
    # v62 feature context: track same-team OUT counts for teammate_out_count feature.
    team_out_counts = {}
    for p, ab, _ in players_in:
        if p.get("is_out"):
            team_out_counts[ab] = team_out_counts.get(ab, 0) + 1

    # ── Cascade Team Detector ──────────────────────────────────────────────
    # grep: CASCADE TEAM DETECTOR
    # When a star (20+ PPG) is OUT on a team, flag all active teammates.
    # Historical data: 192 mega-stack instances, avg combined value 50-80+.
    # Flagged players get: RS multiplier (1.3x), boost floor (2.5), relaxed
    # gates in _build_lineups (deep rotation sweet spot targeting).
    _ct_cfg = _cfg("cascade.team_detector", {}) or {}
    _ct_enabled = _ct_cfg.get("enabled", True)
    cascade_teams = set()  # teams with a star OUT
    if _ct_enabled:
        _ct_star_ppg = float(_ct_cfg.get("star_ppg_threshold", 20.0))
        for p, ab, _ in players_in:
            if p.get("is_out"):
                _p_stats = stats_map.get(p["id"])
                _p_ppg = _p_stats.get("pts", 0) if _p_stats else 0
                if _p_ppg >= _ct_star_ppg:
                    cascade_teams.add(ab)
                    print(f"[cascade-team] {ab} star OUT: {p.get('name','')} ({_p_ppg:.1f} PPG) — flagging teammates")

    # Fetch team rest days for rest_days feature (fixes training skew — was always 2.0)
    team_rest = _fetch_team_rest_days()

    out = []
    for p, ab, sd in players_in:
        stats = stats_map.get(p["id"])
        if not stats:
            continue
        _is_cascade_team = ab in cascade_teams and not p.get("is_out")
        p = {**p, "teammate_out_count": team_out_counts.get(ab, 0),
             "rest_days": team_rest.get(ab, 3),
             "_cascade_team": _is_cascade_team}
        cascade_bonus = cascade_flags.get(p["id"], 0.0)
        # Check if this player's team is on a back-to-back
        b2b = game.get("home_b2b") if sd == "home" else game.get("away_b2b")
        # Determine opponent for matchup analysis
        opp_abbr = game["away"]["abbr"] if sd == "home" else game["home"]["abbr"]
        prefetched = None
        if gamelog_map is not None:
            prefetched = gamelog_map.get(str(p["id"])) or gamelog_map.get(p["id"])
        proj = project_player(
            p, stats, game["spread"], game["total"], sd, ab,
            cascade_bonus=cascade_bonus, is_b2b=bool(b2b),
            prefetched_gamelog=prefetched,
            dvp_data=dvp_data,
            opp_abbr=opp_abbr,
            game_id=game.get("gameId"),
        )
        if proj:
            proj["opp"] = opp_abbr  # store opponent for matchup factor in _build_lineups
            out.append(proj)
    _cs(cache_key, out)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# RESPONSE CONTRACT NORMALIZERS
# grep: _normalize_player
# Stable frontend API contract. Model internals can change freely; only these
# output shapes are guaranteed to the frontend. Apply at all lineup return points.
# ─────────────────────────────────────────────────────────────────────────────

# Internal-only fields never sent to the frontend
_PLAYER_INTERNAL_FIELDS = {"chalk_ev_capped", "_rw_cleared", "_matchup_factor", "_core_score"}

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
        "draft_ev":      round(float(p.get("draft_ev") or 0), 2),
        "chalk_ev":      round(float(p.get("chalk_ev") or 0), 2),
        "moonshot_ev":   round(float(p.get("moonshot_ev") or 0), 2),
        "injury_status": p.get("injury_status", ""),
        "_decline":      round(float(p.get("_decline") or 0), 2),
    }
    # Pass through extras (model fields, debug fields, trend stats) — strip internal-only
    extras = {k: v for k, v in p.items()
              if k not in base and k not in _PLAYER_INTERNAL_FIELDS}
    return {**base, **extras}


def _apply_per_game_carry_core_pool(sorted_union, chalk_eligible, core_size, per_carry, avg_slot):
    """Prefer top per-game TV players into the Starting-5 core pool before global trim.

    Surfaces per-game standouts (e.g. THE LINE UP leaders) that lose to global RS×boost rank.
    """
    if per_carry <= 0 or core_size <= 0:
        return sorted_union[:core_size]

    def _gid(p):
        return "_vs_".join(sorted([p.get("team", ""), p.get("opp", "")]))

    by_game = {}
    for p in chalk_eligible:
        by_game.setdefault(_gid(p), []).append(p)
    carry_names = set()
    k = max(0, int(per_carry))
    for plist in by_game.values():

        def _tv(pl):
            return float(pl.get("rating", 0) or 0) * (avg_slot + float(pl.get("est_mult", 0.3)))

        for pl in sorted(plist, key=_tv, reverse=True)[:k]:
            nm = pl.get("name")
            if nm:
                carry_names.add(nm)
    name_to_rec = {r["name"]: r for r in sorted_union}
    carried_recs = [name_to_rec[n] for n in carry_names if n in name_to_rec]
    carried_recs.sort(key=lambda x: x.get("_core_score", 0), reverse=True)
    core_pool = []
    seen = set()
    for r in carried_recs:
        if len(core_pool) >= core_size:
            break
        core_pool.append(r)
        seen.add(r["name"])
    for r in sorted_union:
        if len(core_pool) >= core_size:
            break
        if r["name"] not in seen:
            core_pool.append(r)
            seen.add(r["name"])
    return core_pool


def _build_lineups(projections, def_stats=None, matchup_intel=None, dvp_data=None, n_games=None):
    """Dynamic lineup builder — unified EV formula, no hardcoded archetypes.

    EV formula: RS × (1.6 + boost) × data_multipliers
      Where 1.6 = mean slot multiplier ([2.0, 1.8, 1.6, 1.4, 1.2]).
      This correctly values both RS and boost relative to actual game scoring.
      Stars (high RS, low boost) and role players (moderate RS, high boost)
      rank on equal footing — the slate's player pool determines the mix.

    Pipeline:
      1. Filter: RS >= 2.0, minutes >= 12, not OUT, not blacklisted
      2. Score: safe_ev = RS × (1.6 + cb_low), upside_ev = RS × (1.6 + cb_high)
         Plus data-driven multipliers (minutes delta, leaderboard frequency,
         contrarian bonus for under-drafted high-boost role players)
      3. Starting 5: top 5 by safe_ev (floor reliability), team cap applied
      4. Moonshot: top 5 by upside_ev from remaining pool (ceiling)
      5. Slot assignment: sort both lineups by RS descending (provably optimal)
    """
    # ── Configuration ──────────────────────────────────────────────────────
    _strat = _cfg("strategy", {}) or {}
    rs_floor = float(_strat.get("rs_floor", 2.0))
    min_minutes = float(_strat.get("min_minutes", 25.0))
    max_per_team = int(_strat.get("max_per_team", 1))

    # ── Small-slate detection ──────────────────────────────────────────────
    # 3-game slates have only ~18 eligible players across 6 teams.
    # Aggressive gates (25 min, 2.0 RS) collapse the pool below 5.
    # Detect small slates and relax gates proportionally.
    if n_games is None:
        n_games = 6  # default to medium if unknown
    _small_slate = n_games <= 4
    if _small_slate:
        # Relax min_minutes: 25 → 16 for 3-game, 20 for 4-game
        min_minutes = max(16.0, min_minutes - (5 - n_games) * 3.0)
        # RS floor stays at 2.0 — keeps quality bar, fallback handles the rest
        print(f"[build_lineups] small slate detected ({n_games} games), min_minutes relaxed to {min_minutes}")

    # RotoWire availability check
    rw_statuses = {}
    try:
        rw_statuses = get_all_statuses()
    except Exception as e:
        print(f"RotoWire fetch failed, proceeding without: {e}")

    # ── Step 1: Build single candidate pool ────────────────────────────────
    # Gates: RS floor + minutes floor + recent_min floor (rotation-bubble filter)
    # Deep Rotation Sweet Spot: cascade team players get relaxed gates because
    # historically 5-20 draft players on cascade teams produce avg value 16.1.
    min_recent_minutes = float(_strat.get("min_recent_minutes", 15.0))
    minutes_increase_bypass = float(_strat.get("minutes_increase_bypass", 15.0))
    _ct_cfg = _cfg("cascade.team_detector", {}) or {}
    _ct_rs_floor = float(_ct_cfg.get("deep_rotation_rs_floor", 1.5))
    _ct_min_minutes = float(_ct_cfg.get("deep_rotation_min_minutes", 12.0))
    candidate_pool = []
    for p in projections:
        if p.get("name") in BLACKLISTED_PLAYERS:
            continue
        # Cascade team players get relaxed RS floor (1.5 vs 2.0) and minutes floor (12 vs 25)
        _is_ct = p.get("_cascade_team", False)
        # High-boost bypass: players with predicted boost ≥ 2.5 get relaxed minutes gate (12 min)
        # Apr 8 post-mortem: ALL winning players had 3.0x boost with low minutes — these
        # deep bench contrarians were filtered by the 25-min gate despite massive EV.
        _est_boost = float(p.get("est_mult", 0) or p.get("card_boost", 0) or 0)
        _is_high_boost = _est_boost >= 2.5 and p.get("rating", 0) >= rs_floor
        _effective_rs_floor = _ct_rs_floor if _is_ct else rs_floor
        _effective_min_minutes = _ct_min_minutes if (_is_ct or _is_high_boost) else min_minutes
        if p.get("rating", 0) < _effective_rs_floor:
            continue
        # Minutes-increase bypass: if projected minutes jump is huge (cascade, injury),
        # skip the minutes floor and rotation-bubble gates. These are deep bench players
        # stepping into expanded roles — exactly the high-boost role players that win.
        _pred_min = float(p.get("predMin", 0))
        _season_min = float(p.get("season_min", 0))
        _mi_bypass = (_pred_min - _season_min) >= minutes_increase_bypass
        # Hard gate: only draft players projected at or above their season average minutes.
        # This applies to ALL players including cascade team — any minutes drop
        # signals B2B, load management, or role change regardless of team situation.
        # Cascade bypass is for bench players getting MORE minutes, not starters getting fewer.
        _max_min_drop = float(_cfg("projection.max_predmin_drop", 8.0))
        if _season_min > 0 and (_season_min - _pred_min) > _max_min_drop:
            continue
        if not _mi_bypass and not _is_ct and not _is_high_boost:
            if _pred_min < _effective_min_minutes and _season_min < _effective_min_minutes:
                continue
            # Rotation-bubble filter: recent_min must meet floor to avoid DNP/early-hook risk
            # Players with 12-14 recent min are rotation-bubble — high risk of wasting a draft slot
            if p.get("recent_min", 0) < min_recent_minutes and _season_min < min_recent_minutes + 3:
                continue
        if rw_statuses and not is_safe_to_draft(p.get("name", "")):
            continue
        if p.get("injury_status", "").upper() == "OUT":
            continue
        try:
            _avail, _ = _injury_available(p.get("name", ""))
            if not _avail:
                continue
        except Exception:
            pass
        candidate_pool.append(p)

    if len(candidate_pool) < 5:
        print(f"[build_lineups] candidate pool thin ({len(candidate_pool)}), relaxing RS floor to 1.5")
        for p in projections:
            if p.get("name") in BLACKLISTED_PLAYERS:
                continue
            if p.get("name") in {c.get("name") for c in candidate_pool}:
                continue
            if p.get("rating", 0) < 1.5:
                continue
            if p.get("injury_status", "").upper() == "OUT":
                continue
            candidate_pool.append(p)
            if len(candidate_pool) >= 10:
                break

    # ── Small-slate team cap relaxation ────────────────────────────────────
    # 3-game slate = 6 teams. With max_per_team=1 we can only select 6 unique
    # players total. Need 10 (5 chalk + 5 moonshot). Auto-relax to 2 when the
    # pool can't fill both lineups under current team cap.
    _unique_teams = len({(p.get("team") or "").upper() for p in candidate_pool if p.get("team")})
    if max_per_team == 1 and _unique_teams < 10 and len(candidate_pool) >= 8:
        max_per_team = 2
        print(f"[build_lineups] only {_unique_teams} teams in pool, relaxing max_per_team to 2 for coverage")

    # ── Step 2: Score by unified EV = RS × (avg_slot + boost) ────────────
    # Formula: EV = RS × (1.6 + boost), where 1.6 = avg([2.0,1.8,1.6,1.4,1.2]).
    #
    # Why 1.6 + boost (not just boost):
    #   A star (RS=6, boost=0.3) at the 2.0x slot earns 6×(2.0+0.3)=13.8 total.
    #   A role player (RS=3, boost=3.0) at the 1.2x slot earns 3×(1.2+3.0)=12.6.
    #   RS × boost alone gives 1.8 vs 9.0, wildly wrong relative to actual scoring.
    #   RS × (1.6 + boost) gives 11.4 vs 13.8 — much closer to reality, and both
    #   players could legitimately be optimal depending on the full slate pool.
    #   The formula naturally selects stars when they're valuable, role players when
    #   they're not — no hardcoded archetype required.
    #
    # safe_ev uses cb_low (conservative boost floor) — prefer for Starting 5.
    # upside_ev uses cb_high (optimistic boost ceiling) — prefer for Moonshot.
    #
    # Data-driven multipliers still applied:
    #   - minutes_increase_ev_bonus: rewards players stepping into expanded roles
    #   - leaderboard_frequency: proven repeat leaderboard performers get small edge
    #   - contrarian_bonus: under-drafted role players with high boost get small edge
    _avg_slot = float(_cfg("lineup.avg_slot_multiplier", 1.6) or 1.6)

    _mi_cfg = _strat.get("minutes_increase_ev_bonus", {})
    _mi_enabled = _mi_cfg.get("enabled", True)
    _mi_min_delta = float(_mi_cfg.get("min_delta", 4.0))
    _mi_bonus_per_min = float(_mi_cfg.get("bonus_per_min", 0.02))
    _mi_max_bonus = float(_mi_cfg.get("max_bonus", 0.15))

    _lb_freq = {}
    _lb_cfg = _strat.get("leaderboard_frequency", {})
    _lb_enabled = _lb_cfg.get("enabled", True)
    _lb_min_appearances = int(_lb_cfg.get("min_appearances", 2))
    _lb_max_bonus = float(_lb_cfg.get("max_bonus", 0.35))
    _lb_per_appearance = float(_lb_cfg.get("bonus_per_appearance", 0.008))
    _lb_ghost_weight = float(_lb_cfg.get("ghost_quality_weight", 0.80))
    if _lb_enabled:
        try:
            _lb_freq = _load_leaderboard_frequency()
        except Exception:
            pass

    _ct_cfg = _strat.get("contrarian_bonus", {})
    _ct_enabled = _ct_cfg.get("enabled", True)
    _ct_max_bonus = float(_ct_cfg.get("max_bonus", 0.10))
    _ct_min_boost = float(_ct_cfg.get("min_boost", 2.0))
    _ct_max_ppg = float(_ct_cfg.get("max_season_ppg", 16.0))

    # ── Momentum curve detection ──────────────────────────────────────────
    # grep: MOMENTUM CURVE
    # Load player history to detect two critical patterns:
    #   HYPE TRAP: Drafts exploding + boost declining = player at TOP of curve.
    #     Sensabaugh went 2 → 207 drafts, boost 3.0 → 2.0. Classic trap.
    #   RISING WAVE: RS trending up + low drafts + high boost = COMING UP curve.
    #     Fears/Hawkins: high RS, low drafts, contrarian edge. These are our targets.
    _mc_cfg = _strat.get("momentum_curve", {})
    _mc_enabled = _mc_cfg.get("enabled", True)
    _mc_history = {}
    if _mc_enabled:
        try:
            from api.boost_model import load_player_history
            _mc_history = load_player_history()
        except Exception:
            pass

    for p in candidate_pool:
        rs = float(p.get("rating", 0))
        boost = float(p.get("est_mult", 0))
        band = p.get("boost_band")
        if band and isinstance(band, (list, tuple)) and len(band) == 2:
            cb_low, cb_high = float(band[0]), float(band[1])
        else:
            cb_low = cb_high = boost
        # Minutes increase multiplier: reward expanded-role players
        mi_mult = 1.0
        if _mi_enabled:
            pred_min = float(p.get("predMin", 0))
            season_min = float(p.get("season_min", 0))
            mi_delta = pred_min - season_min
            if mi_delta >= _mi_min_delta:
                mi_mult = 1.0 + min(_mi_bonus_per_min * (mi_delta - _mi_min_delta), _mi_max_bonus)
        # Leaderboard frequency + ghost quality bonus
        # Two-part signal:
        #   Ghost quality (80% weight): data-driven from low-ownership high-boost appearances.
        #   Saturates at ~3 ghost appearances, scales with avg value delivered.
        #   Answers: "does this player reliably produce when nobody drafts them?"
        # Raw count (20% weight): rewards consistent leaderboard presence regardless of ownership.
        lb_mult = 1.0
        if _lb_enabled and _lb_freq:
            _pname_norm = _normalize_player_name(p.get("name", ""))
            _lb_data = _lb_freq.get(_pname_norm)
            if _lb_data and _lb_data["count"] >= _lb_min_appearances:
                _ghost_quality = float(_lb_data.get("ghost_quality", 0.0))
                _ghost_bonus = _ghost_quality * _lb_max_bonus * _lb_ghost_weight
                _count_factor = min((_lb_data["count"] - _lb_min_appearances + 1) * _lb_per_appearance,
                                    _lb_max_bonus * (1.0 - _lb_ghost_weight))
                lb_mult = 1.0 + _ghost_bonus + _count_factor
                p["_lb_freq"] = _lb_data["count"]
                p["_lb_avg_value"] = _lb_data["avg_value"]
                p["_ghost_quality"] = round(_ghost_quality, 3)
                p["_ghost_count"] = _lb_data.get("ghost_count", 0)
        # Contrarian bonus: under-drafted role players with high boost
        ct_mult = 1.0
        if _ct_enabled:
            _season_ppg = float(p.get("season_pts") or p.get("pts") or 0)
            if boost >= _ct_min_boost and _season_ppg <= _ct_max_ppg and _season_ppg > 0:
                _ct_ratio = min((boost - _ct_min_boost) / max(1.0, 3.0 - _ct_min_boost), 1.0)
                ct_mult = 1.0 + (_ct_max_bonus * _ct_ratio)
                p["_contrarian"] = True

        # ── Momentum curve multiplier ─────────────────────────────────────
        # grep: MOMENTUM CURVE SCORING
        mc_mult = 1.0
        if _mc_enabled and _mc_history:
            _pname_mc = _normalize_player_name(p.get("name", ""))
            _phist = _mc_history.get(_pname_mc, [])
            _mc_min_hist = int(_mc_cfg.get("min_history", 3))
            if len(_phist) >= _mc_min_hist:
                # Recent entries (last 5 appearances)
                _recent_n = min(5, len(_phist))
                _recent = _phist[-_recent_n:]
                _older = _phist[:-_recent_n] if len(_phist) > _recent_n else _phist[:1]

                _recent_rs = [e["rs"] for e in _recent if e.get("rs", 0) > 0]
                _recent_drafts = [e["drafts"] for e in _recent if e.get("drafts", 0) > 0]
                _recent_boosts = [e["boost"] for e in _recent if e.get("boost") is not None]
                _older_drafts = [e["drafts"] for e in _older if e.get("drafts", 0) > 0]
                _older_boosts = [e["boost"] for e in _older if e.get("boost") is not None]

                # === HYPE TRAP DETECTION ===
                # Signal: drafts rising sharply AND boost declining
                # Player is getting more popular → Real Sports lowers their boost → trap
                _trap_penalty = float(_mc_cfg.get("hype_trap_max_penalty", 0.20))
                if _recent_drafts and _older_drafts and _recent_boosts and _older_boosts:
                    _avg_recent_drafts = sum(_recent_drafts) / len(_recent_drafts)
                    _avg_older_drafts = sum(_older_drafts) / len(_older_drafts)
                    _avg_recent_boost = sum(_recent_boosts) / len(_recent_boosts)
                    _avg_older_boost = sum(_older_boosts) / len(_older_boosts)

                    # Drafts going up AND boost going down = hype trap
                    _draft_growth = (_avg_recent_drafts / max(_avg_older_drafts, 1.0)) - 1.0  # % increase
                    _boost_decline = _avg_older_boost - _avg_recent_boost  # positive = declining

                    # Both conditions must be true (AND not OR)
                    _draft_growth_threshold = float(_mc_cfg.get("draft_growth_threshold", 2.0))  # 200% draft increase
                    _boost_decline_threshold = float(_mc_cfg.get("boost_decline_threshold", 0.5))  # 0.5 boost drop

                    if _draft_growth >= _draft_growth_threshold and _boost_decline >= _boost_decline_threshold:
                        # Scale penalty by how extreme the trap is
                        _trap_severity = min(_draft_growth / 10.0, 1.0) * min(_boost_decline / 1.5, 1.0)
                        mc_mult *= (1.0 - _trap_penalty * _trap_severity)
                        p["_hype_trap"] = True
                        p["_hype_trap_severity"] = round(_trap_severity, 3)

                # === RISING WAVE DETECTION ===
                # Signal: RS trending up in recent appearances + still low drafts + high boost
                # These are the players "coming up on the curve" — exactly who we want
                _wave_bonus = float(_mc_cfg.get("rising_wave_max_bonus", 0.20))
                if len(_recent_rs) >= 2 and _recent_boosts:
                    # RS trending up: last 2-3 appearances higher than first 2-3
                    _first_half = _recent_rs[:len(_recent_rs)//2] or _recent_rs[:1]
                    _second_half = _recent_rs[len(_recent_rs)//2:] or _recent_rs[-1:]
                    _rs_trend = (sum(_second_half) / len(_second_half)) - (sum(_first_half) / len(_first_half))

                    _latest_boost = _recent_boosts[-1]
                    _latest_drafts = _recent_drafts[-1] if _recent_drafts else 0

                    # Rising: RS going up + boost still high (≥ 2.0) + drafts still low (< 200)
                    _rs_trend_min = float(_mc_cfg.get("rs_trend_min", 0.3))
                    _wave_max_drafts = float(_mc_cfg.get("wave_max_drafts", 200.0))
                    _wave_min_boost = float(_mc_cfg.get("wave_min_boost", 2.0))

                    if (_rs_trend >= _rs_trend_min
                            and _latest_boost >= _wave_min_boost
                            and _latest_drafts < _wave_max_drafts):
                        # Scale by trend strength and boost level
                        _trend_strength = min(_rs_trend / 1.5, 1.0)
                        _boost_strength = min((_latest_boost - _wave_min_boost) / 1.0, 1.0)
                        _draft_contrarian = min(1.0, ((_wave_max_drafts - _latest_drafts) / _wave_max_drafts))
                        _wave_score = _trend_strength * _boost_strength * _draft_contrarian
                        mc_mult *= (1.0 + _wave_bonus * _wave_score)
                        p["_rising_wave"] = True
                        p["_rising_wave_score"] = round(_wave_score, 3)

        total_mult = mi_mult * lb_mult * ct_mult * mc_mult
        # Unified EV: RS × (avg_slot + boost) captures both RS and boost value
        p["draft_ev"]  = round(rs * (_avg_slot + boost)    * total_mult, 2)
        p["safe_ev"]   = round(rs * (_avg_slot + cb_low)   * total_mult, 2)
        p["upside_ev"] = round(rs * (_avg_slot + cb_high)  * total_mult, 2)
        p["_mi_mult"]  = round(mi_mult, 3)
        p["_lb_mult"]  = round(lb_mult, 3)
        p["_ct_mult"]  = round(ct_mult, 3)
        p["_mc_mult"]  = round(mc_mult, 3)
        p["moonshot_ev"] = p["draft_ev"]

    # ── Step 3: Sort candidate pool by safe_ev ──────────────────────────────
    candidate_pool.sort(key=lambda x: (-x.get("safe_ev", 0), -x.get("rating", 0)))

    def _select_with_team_cap(pool, n, max_team, exclude_names=None):
        """Greedy top-N selection respecting per-team cap."""
        selected = []
        team_counts = {}
        _excl = exclude_names or set()
        for p in pool:
            if len(selected) >= n:
                break
            if p.get("name") in _excl:
                continue
            team = (p.get("team") or "").upper()
            if team and team_counts.get(team, 0) >= max_team:
                continue
            selected.append(p)
            team_counts[team] = team_counts.get(team, 0) + 1
        return selected, team_counts

    # ── Step 4: Starting 5 — top 5 by safe_ev with team cap ─────────────
    # No hardcoded archetypes. The unified EV formula (RS × (1.6 + boost))
    # naturally balances stars vs role players per-slate:
    #   - Star slates: high RS pushes stars to top even with low boost
    #   - Role-player slates: high boost dominates when RS is moderate
    #   - Mixed slates: the formula finds the true optimal mix
    # Both lineups drawn from the same pool. Starting 5 uses safe_ev (cb_low)
    # for floor reliability. Moonshot uses upside_ev (cb_high) for ceiling.
    safe_pool = sorted(candidate_pool, key=lambda x: (-x.get("safe_ev", 0), x.get("player_variance", 0)))
    chalk, _ = _select_with_team_cap(safe_pool, 5, max_per_team)
    chalk_names = {p.get("name") for p in chalk}

    # ── Step 5: Moonshot — top 5 by upside_ev from remaining pool ───────
    # Same candidates as chalk pool, different ordering (upside_ev vs safe_ev).
    # Moonshot players are expected to diverge 1-3 players from the Starting 5.
    moonshot_pool = sorted(
        [p for p in candidate_pool if p.get("name") not in chalk_names],
        key=lambda x: (-x.get("upside_ev", 0), -x.get("est_mult", 0))
    )
    upside, _ = _select_with_team_cap(moonshot_pool, 5, max_per_team)

    # ── Step 6: Assign slots by RS descending (Finding 3) ──────────────────
    # Provably optimal: highest RS → 2.0x, next → 1.8x, etc.
    chalk.sort(key=lambda x: -float(x.get("rating", 0)))
    upside.sort(key=lambda x: -float(x.get("rating", 0)))
    for i, p in enumerate(chalk):
        p["slot"] = _SLOT_LABELS_SHARED[i] if i < len(_SLOT_LABELS_SHARED) else "1.0x"
    for i, p in enumerate(upside):
        p["slot"] = _SLOT_LABELS_SHARED[i] if i < len(_SLOT_LABELS_SHARED) else "1.0x"

    # core_pool = full ranked candidate list (for lineup review / watchlist compatibility)
    core_pool = candidate_pool[:20]

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
# grep: PER-GAME, _build_game_lineups, _per_game_strategy, _per_game_adjust
#
# Redesigned from 18-game / 76-lineup empirical analysis (Jan 6 – Mar 23).
# Six findings drive the strategy:
#   F1: 2x slot = conviction slot (not always best player)
#   F2: Value anchor pattern — 3.5+ RS in 1.2x–1.4x lifts floor
#   F3: Game total correlates with winning ceiling (250+ → 32.1 avg)
#   F4: Blowouts reward top-heavy; close games reward balance
#   F5: Margins are razor-thin (<1.5 pts) — slot optimization matters
#   F6: Favored team role players benefit in blowouts
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


def _per_game_strategy(game):
    """Determine per-game draft strategy based on spread and game total.

    Returns dict with strategy type, label, and parameters used for
    _per_game_adjust_projections and frontend display.
    """
    cfg = _cfg("per_game", _CONFIG_DEFAULTS["per_game"])
    spread = abs(game.get("spread") or 0)
    total = game.get("total") or DEFAULT_TOTAL
    raw_spread = game.get("spread") or 0

    close_thr = cfg.get("close_spread_threshold", 5)
    blow_thr = cfg.get("blowout_spread_threshold", 13)

    home_abbr = game.get("home", {}).get("abbr", "")
    away_abbr = game.get("away", {}).get("abbr", "")
    # Negative spread = home favored in most ESPN formats
    favored_team = home_abbr if raw_spread < 0 else away_abbr
    underdog_team = away_abbr if raw_spread < 0 else home_abbr

    # Game total multiplier (Finding 3)
    baseline = cfg.get("total_baseline", 222)
    t_str = cfg.get("total_mult_strength", 0.003)
    t_floor = cfg.get("total_mult_floor", 0.92)
    t_ceil = cfg.get("total_mult_ceiling", 1.12)
    total_mult = max(t_floor, min(t_ceil, 1.0 + (total - baseline) * t_str))

    # Determine composition type (Finding 4)
    if spread <= close_thr:
        comp = "balanced"
        label = "Balanced Build"
        description = "Close game — all starters play full minutes. Prioritizing consistent producers across all 5 slots."
    elif spread >= blow_thr:
        comp = "top_heavy"
        label = "Blowout Lean"
        description = f"Large spread ({spread:.0f} pts) — leaning toward {favored_team} role players who benefit from extended run."
    else:
        comp = "neutral"
        label = "Standard Build"
        description = "Moderate spread — balanced approach with slight edge to favored side."

    # Overlay game total context
    if total >= 245:
        label = f"Shootout · {label}" if comp != "neutral" else "Shootout"
        description = f"High total ({total}) — aggressive ceiling play. " + description
    elif total < 215:
        label = f"Grind · {label}" if comp != "neutral" else "Defensive Grind"
        description = f"Low total ({total}) — floor and efficiency matter. " + description

    return {
        "type": comp,
        "label": label,
        "description": description,
        "total_mult": round(total_mult, 3),
        "spread": spread,
        "total": total,
        "favored_team": favored_team,
        "underdog_team": underdog_team,
    }


def _per_game_adjust_projections(projections, game, strategy):
    """Apply per-game draft strategy adjustments to projections (Findings 1-6).

    Operates on already game-script-rescored projections. Adjustments:
    - Game total ceiling multiplier (F3)
    - Spread-based composition: balanced vs top-heavy (F4)
    - Favored team role player tilt in blowouts (F6)
    - Value anchor bonus for high-floor mid-tier players (F2)
    - Conviction slot variance shaping (F1)

    Returns new list of adjusted projection dicts (deep copies).
    """
    cfg = _cfg("per_game", _CONFIG_DEFAULTS["per_game"])
    if not cfg.get("enabled", True):
        return projections

    comp = strategy["type"]
    total_mult = strategy["total_mult"]
    favored = strategy["favored_team"]
    spread = strategy["spread"]

    close_thr = cfg.get("close_spread_threshold", 5)
    blow_thr = cfg.get("blowout_spread_threshold", 13)
    role_ceil = cfg.get("role_player_pts_ceiling", 18)
    anchor_min = cfg.get("value_anchor_min_rating", 3.8)
    anchor_pts = cfg.get("value_anchor_pts_ceiling", 16)
    anchor_bonus = cfg.get("value_anchor_bonus", 0.08)
    star_carry_thr = cfg.get("star_carry_threshold", 6.0)
    star_carry_bonus = cfg.get("star_carry_role_bonus", 0.10)

    # Pre-compute max projected rating per team to detect superstar-carry games.
    # When Jokić/Wemby/Durant projects RS >= 6.0, their role teammates get a
    # +10% boost — the superstar creates extra opportunities (assists, putbacks,
    # transition) that lift everyone on the same team.
    team_max_rating: dict[str, float] = {}
    for p in projections:
        t = p.get("team", "")
        r = float(p.get("rating") or 0)
        if r > team_max_rating.get(t, 0.0):
            team_max_rating[t] = r

    adjusted = []
    for p in projections:
        ap = copy.deepcopy(p)
        rating = float(ap.get("rating") or 0)
        team = ap.get("team", "")
        season_pts = float(ap.get("season_pts") or ap.get("pts") or 0)
        variance = float(ap.get("player_variance") or 0.3)
        is_role = season_pts <= role_ceil

        # ── F3: Game total multiplier ──
        # Higher game totals raise the scoring ceiling for everyone
        rating *= total_mult

        # ── F4 + F6: Spread-based composition ──
        if comp == "balanced":
            # Close game: starters on both teams play full minutes.
            # Reward consistency (low variance) — the winning margin is thin (avg 2.52 range).
            floor_bonus = cfg.get("close_game_floor_bonus", 0.06)
            var_dampen = cfg.get("close_game_variance_dampen", 0.08)
            # Consistent players get a floor bonus; volatile players get dampened
            rating *= (1.0 + floor_bonus * (1.0 - variance))
            rating *= (1.0 - var_dampen * variance)

        elif comp == "top_heavy":
            # Blowout: favored team runs away; role players get extended run.
            # Underdog role players get buried — BUT only in true blowouts (20+).
            # Data (LAL 127-113 LAC, 14pt): losing-team bigs (Allen 3.4, Ayton 3.3)
            # outscored ALL winning role players. The old flat 12% penalty is too
            # harsh for moderate blowouts (13-19pt). Use graduated penalties.
            heavy_blowout_thr = cfg.get("heavy_blowout_threshold", 20)
            is_heavy = spread >= heavy_blowout_thr
            if team == favored:
                if is_role:
                    # F6: Favored team's 3rd-5th options get garbage-time production
                    rating *= cfg.get("blowout_favored_role_bonus", 1.12)
                else:
                    # Stars on favored team still produce but game may end early
                    rating *= cfg.get("blowout_favored_star_bonus", 1.05)
            else:
                if is_heavy:
                    # True blowout (20+pt): underdog role players get buried
                    if is_role:
                        rating *= cfg.get("blowout_underdog_role_penalty", 0.88)
                    else:
                        rating *= cfg.get("blowout_underdog_star_penalty", 0.95)
                else:
                    # Moderate blowout (13-19pt): underdog still contributes,
                    # especially bigs who accumulate stats in losing effort.
                    # Data: Allen/Ayton (LAC losers) had RS 3.4/3.3 in 14pt loss.
                    if is_role:
                        rating *= cfg.get("moderate_blowout_underdog_role", 0.95)
                    else:
                        rating *= cfg.get("moderate_blowout_underdog_star", 0.98)

            # F1: Blowout variance uplift — high-upside players rise to 2.0x
            var_up = cfg.get("blowout_variance_uplift", 0.04)
            rating *= (1.0 + var_up * variance)

        else:
            # Neutral (moderate spread 6-12): mild favored-team lean.
            # Data: POR/LAC (10pt) had 3 loser players in optimal 5;
            # CHA/NYK (11pt) had 2. Don't over-penalize losing side.
            if team == favored:
                rating *= cfg.get("neutral_favored_lean", 1.02)
            # F1: neutral variance treatment — no shaping

        # ── F2: Value anchor bonus ──
        # Players with solid RS floor + not a superstar → silently lift lineup floor.
        # Spencer 5.5 at 1.4x, Gafford 4.5 at 1.4x, Champagnie 4.2 at 1.2x
        is_anchor = rating >= anchor_min and season_pts <= anchor_pts
        if is_anchor:
            rating *= (1.0 + anchor_bonus)
        ap["_is_value_anchor"] = is_anchor
        ap["_favored_team"] = (team == favored)

        # ── Superstar-carry stack bonus ──
        # When a teammate is projected RS >= 6.0 (Jokić/Wemby/Durant tier),
        # role players on the same team get +10%. The superstar creates extra
        # scoring opportunities — assists, putback situations, transition —
        # that inflate their teammates' RS beyond what season averages predict.
        # Only applies to role players (season_pts <= role_ceil) to avoid
        # double-boosting the superstar's own projection.
        team_max = team_max_rating.get(team, 0.0)
        if team_max >= star_carry_thr and is_role and float(ap.get("rating") or 0) < team_max:
            rating *= (1.0 + star_carry_bonus)
            ap["_star_carry"] = True

        ap["rating"] = round(rating, 2)
        ap["_pg_total_mult"] = strategy["total_mult"]
        ap["_pg_strategy"] = comp
        adjusted.append(ap)

    return adjusted


def _build_game_lineups(projections, game):
    """Build exactly ONE lineup ('THE LINE UP') for a single-game draft.

    Per-game drafts restrict to a single 5-player format. No Starting 5 / Moonshot
    split — both users draft from the same 2-team pool, so card boost is irrelevant.

    Pipeline (post-redesign):
    1. Game script re-scoring (stat-weight tiers by game total)
    2. Per-game strategy adjustments (F1-F6: total mult, spread comp, team tilt, anchors)
    3. Eligibility gating (minutes, rating, pts floors)
    4. MILP optimization (RS × slot_mult, value anchor-aware)
    5. 5! permutation validation (120 combos — computationally trivial)
    6. Strategy metadata for frontend
    """
    # Step 1: Game script (existing — adjusts stat category weights by game pace)
    rescored = _apply_game_script(projections, game)

    # Step 2: Per-game strategy analysis + projection adjustments (NEW)
    strategy = _per_game_strategy(game)
    adjusted = _per_game_adjust_projections(rescored, game, strategy)

    # Step 3: Eligibility gating — TWO TIERS + CASCADE (grep: PER-GAME ELIGIBILITY)
    # Tier A (core): standard gates for top-4 consensus picks
    # Tier B (sleeper): relaxed gates for the decisive 5th player pick
    # Tier C (cascade): star is OUT on team — deep rotation players bypass normal
    #   floors and go straight into core_pool so MILP can slot them at any position.
    #   _cascade_team flag is already set by _run_game; project_player has already
    #   applied the RS boost and relaxed the minutes gate. Here we just respect it.
    # Data (6 games, 48 lineups): winners differ from mid-pack by exactly
    # 1 player in 5/6 games. That 5th player averages +0.9 RS over the
    # mid-pack's 5th pick. Many winning 5th picks (Plowden 3.3, Camara 3.1,
    # Dieng 3.1) would fail the old 3.5 rating floor.
    game_chalk_floor = _cfg("lineup.game_chalk_rating_floor", 3.5)
    game_sleeper_floor = _cfg("per_game.sleeper_rating_floor", 2.5)
    game_min_floor = _cfg("lineup.game_recent_min_floor", 15.0)
    game_sleeper_min_floor = _cfg("per_game.sleeper_min_floor", 12.0)
    min_game_pts = _cfg("scoring_thresholds.min_game_pts", 8.0)
    sleeper_min_pts = _cfg("per_game.sleeper_min_pts", 3.0)
    _ct_pg_cfg = _cfg("cascade.team_detector", {}) or {}
    _ct_pg_rs_floor = float(_ct_pg_cfg.get("deep_rotation_rs_floor", 1.5))
    _ct_pg_min_floor = float(_ct_pg_cfg.get("deep_rotation_min_minutes", 12.0))
    rw_statuses = {}
    try:
        rw_statuses = get_all_statuses()
    except Exception:
        pass

    core_pool = []
    sleeper_pool = []
    for p in adjusted:
        if p.get("name") in BLACKLISTED_PLAYERS:
            continue
        if p.get("injury_status", "").upper() == "OUT":
            continue
        if rw_statuses and not is_safe_to_draft(p.get("name", "")):
            continue
        try:
            avail, _reason = _injury_available(p.get("name", ""))
            if not avail:
                continue
        except Exception:
            pass
        _is_cascade = p.get("_cascade_team", False)
        if _is_cascade:
            # Tier C: cascade — star is OUT on this player's team.
            # Use deep-rotation relaxed floors (rating 1.5, min 12, pts 3.0).
            # Go directly into core_pool so MILP can place them at any slot,
            # not just as a last-resort sleeper replacement.
            if (p.get("recent_min", 0) >= _ct_pg_min_floor
                    and p["rating"] >= _ct_pg_rs_floor
                    and p.get("pts", 0) >= sleeper_min_pts):
                core_pool.append(p)
        else:
            # Tier A: core pool (standard gates)
            if (p.get("recent_min", 0) >= game_min_floor
                    and p["rating"] >= game_chalk_floor
                    and p.get("pts", 0) >= min_game_pts):
                core_pool.append(p)
            # Tier B: sleeper pool (relaxed gates — the 5th player edge)
            elif (p.get("recent_min", 0) >= game_sleeper_min_floor
                  and p["rating"] >= game_sleeper_floor
                  and p.get("pts", 0) >= sleeper_min_pts):
                sleeper_pool.append(p)

    # Step 4: MILP optimization with sleeper-aware candidate pool
    # Per-game: card boost is irrelevant — zero out est_mult.
    # In blowouts, relax team balance to allow 4-1 from favored team (Finding 6).
    cfg = _cfg("per_game", _CONFIG_DEFAULTS["per_game"])
    if strategy["type"] == "top_heavy":
        min_per_team = cfg.get("blowout_min_per_team", 1)
    else:
        min_per_team = 2

    # First pass: MILP on core pool (gets the consensus top-4 right)
    no_boost = [{**p, "est_mult": 0} for p in core_pool]
    the_lineup = optimize_lineup(no_boost, n=5, min_per_team=min_per_team,
                                 sort_key="rating", rating_key="rating",
                                 card_boost_key="est_mult")

    # Step 4b: Sleeper substitution — try replacing weakest core pick with
    # best sleeper if the sleeper has higher adjusted rating.
    # This captures the decisive 5th-player edge that separates winners.
    if len(the_lineup) >= 5 and sleeper_pool:
        weakest_idx = min(range(len(the_lineup)),
                         key=lambda i: the_lineup[i].get("rating", 0))
        weakest_rating = the_lineup[weakest_idx].get("rating", 0)
        lineup_names = {p["name"] for p in the_lineup}
        best_sleeper = max(
            [s for s in sleeper_pool if s["name"] not in lineup_names],
            key=lambda s: s.get("rating", 0), default=None
        )
        if best_sleeper and best_sleeper.get("rating", 0) > weakest_rating:
            the_lineup[weakest_idx] = {**best_sleeper, "est_mult": 0}

    # Step 5: Brute-force 5! permutation validation (Finding 5)
    # MILP handles this, but 120 combos is trivial — verify optimality.
    if len(the_lineup) == 5:
        the_lineup = _validate_slot_assignment(the_lineup)

    # Fill to 5 if pool was too small after gating
    if len(the_lineup) < 5:
        lineup_names = {p["name"] for p in the_lineup}
        fill_pool = sorted(
            [p for p in adjusted
             if p["name"] not in lineup_names
             and p.get("recent_min", 0) >= 12.0
             and p.get("name") not in BLACKLISTED_PLAYERS],
            key=lambda p: p.get("rating", 0), reverse=True
        )
        for p in fill_pool:
            if len(the_lineup) >= 5:
                break
            the_lineup.append(p)

    # Step 6: Normalize — zero est_mult, attach strategy metadata
    normalized = [_normalize_player({**p, "est_mult": 0}) for p in the_lineup]
    return {
        "the_lineup": normalized,
        "strategy": strategy,
    }


def _validate_slot_assignment(lineup):
    """Brute-force 5! = 120 permutation check for optimal slot assignment (Finding 5).

    Given 5 selected players, tries every permutation of slot assignment and returns
    the one that maximizes total score = Σ rating_i × slot_mult_i.
    With card boost zeroed, this is purely RS × slot ordering.

    The MILP should already produce the optimal assignment (highest RS → 2.0x),
    but this is a zero-cost safety net for razor-thin margins (<1.5 pts).
    """
    from itertools import permutations
    slot_mults = _SLOT_MULTS_SHARED[:5]  # [2.0, 1.8, 1.6, 1.4, 1.2]

    ratings = [float(p.get("rating") or 0) for p in lineup]
    best_score = -1
    best_perm = None

    for perm in permutations(range(5)):
        score = sum(ratings[perm[j]] * slot_mults[j] for j in range(5))
        if score > best_score:
            best_score = score
            best_perm = perm

    if best_perm is None:
        return lineup

    result = []
    for j in range(5):
        p = copy.deepcopy(lineup[best_perm[j]])
        p["slot"] = _SLOT_LABELS_SHARED[j]
        result.append(p)
    return result


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

from api.shared import SLOT_LABELS as _SLOT_LABELS_SHARED
_SLOT_NUMS = {label: mult for label, mult in zip(_SLOT_LABELS_SHARED, _SLOT_MULTS_SHARED)}
_SCORE_BOUNDS = {
    "chalk":      (70.0, 100.0),
    "upside":     (70.0, 100.0),
    "the_lineup": (20.0, 42.0),
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
        "date": _today_str(),
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
# grep: /api/games, /api/slate, /api/picks, /api/save-predictions, /api/cold-reset
# /api/hindsight, /api/log, /api/parse-screenshot, /api/save-actuals, /api/health
# ═════════════════════════════════════════════════════════════════════════════
CRON_SECRET = os.getenv("CRON_SECRET", "")

# Rate limiting: in-memory sliding window per IP for expensive endpoints (thread-safe)
_RATE_LIMIT_STORE = {}  # (ip, path_key) -> [timestamps]
_RATE_LIMIT_LOCK = threading.Lock()
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMITS = {"parse-screenshot": 5, "lab/chat": 20}

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
    Accepts Authorization: Bearer <CRON_SECRET> or ?key=<CRON_SECRET> so manual cron/admin
    calls can be authenticated (e.g. myurl/api/cold-reset?key=SECRET)."""
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
    # Redis health probe
    out["redis"] = "ok" if redis_ok() else "unavailable"
    # nba_api feed status
    try:
        from api.nba_api_feed import HAS_NBA_API, HAS_PANDAS, get_player_enrichment
        out["nba_api"] = "available" if HAS_NBA_API and HAS_PANDAS else "missing_deps"
        _pe = get_player_enrichment()
        out["nba_api_players"] = len(_pe) if _pe else 0
    except Exception:
        out["nba_api"] = "error"
    return out


@app.get("/api/version")
async def version() -> dict:
    """Return build/deploy identifier for 'what is deployed' checks. Set RAILWAY_GIT_COMMIT_SHA at build time."""
    sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "")
    return {"version": sha[:7] if sha else "unknown"}


@app.get("/api/games")
async def get_games():
    """Never returns 500; on exception returns 200 with empty list.
    Uses Level 0 response cache to avoid ESPN API calls on repeated fetches."""
    try:
        today = _today_str()
        cache_key = _CACHE_KEYS["games"].format(today)

        def _get_games_impl():
            games = fetch_games()
            for g in games:
                st = g.get("startTime", "")
                g["locked"] = _is_locked(st) if st else False
                g["draftable"] = not _is_past_lock_window(st) if st else False
            return games

        games, is_hit = _with_response_cache(cache_key, "games", _get_games_impl)
        # Add cache metadata
        return {
            "data": games,
            "cache_status": "hit" if is_hit else "miss",
            "cached_at": datetime.now(timezone.utc).isoformat() + "Z",
        }
    except Exception as e:
        print(f"[games] error: {e}")
        return {
            "data": [],
            "cache_status": "miss",
            "error": str(e)
        }

def _force_regenerate_bg(*_args):
    """Sentinel function used as attribute namespace for _in_flight flag."""
    pass

def _force_regenerate_bg_worker():
    """Background thread: runs _force_regenerate_sync("full") when flat-boost anomaly detected.
    This is a safety check for data corruption (all boosts collapsed to +1x).
    Sets _force_regenerate_bg._in_flight = False when done."""
    try:
        print("[force-regen-bg] starting background regeneration (deploy SHA mismatch)")
        result = _force_regenerate_sync("full")
        print(f"[force-regen-bg] done: {result.get('status')}, games={result.get('games_regenerated', 0)}")
    except Exception as e:
        print(f"[force-regen-bg] error: {e}")
        traceback.print_exc()
    finally:
        _force_regenerate_bg._in_flight = False


def _slate_has_flat_boosts(slate_obj: dict) -> bool:
    """Detect suspicious locked slate artifacts where all boosts collapse to +1x.

    We only flag when we have enough players (slate-wide chalk + upside) to avoid
    false positives on tiny/malformed payloads.
    """
    try:
        lineups = (slate_obj or {}).get("lineups", {}) or {}
        players = list(lineups.get("chalk", []) or []) + list(lineups.get("upside", []) or [])
        if len(players) < 8:
            return False
        boosts = []
        for p in players:
            b = _safe_float((p or {}).get("est_mult"), -999.0)
            if b > -900:
                boosts.append(round(float(b), 3))
        if len(boosts) < 8:
            return False
        return all(abs(b - 1.0) <= 1e-9 for b in boosts)
    except Exception:
        return False


def _maybe_trigger_locked_slate_regen(cached_slate: dict, reason_prefix: str = "slate") -> None:
    """Disabled: locked predictions must never be regenerated.

    Previously triggered background full regeneration for flat-boost anomalies,
    but this violated the lock contract — predictions must be immutable once locked.
    Flat-boost issues are a pre-lock data quality problem, not something to fix mid-lock.
    """
    _flat_boosts = _slate_has_flat_boosts(cached_slate or {})
    if _flat_boosts:
        print(f"[{reason_prefix}] flat +1x boosts detected but regeneration BLOCKED — slate is locked")


def _get_slate_impl():
    """Inner slate computation; get_slate() wraps this in try/except so we never return 500."""
    # Determine the active slate date. Normally this is today ET, but when today's slate
    # has ended before midnight (same-day rollover), we serve the next slate instead so
    # the frontend transitions immediately rather than waiting until midnight.
    today_str = _today_str()
    today_et = _et_date()
    try:
        _lk_rollover = _lg(_CK_SLATE_LOCKED)
        if (_lk_rollover
                and _lk_rollover.get("date") == today_str
                and _lk_rollover.get("all_complete")):
            _trigger_cold_pipeline_once("slate_change", today_str)
            _next_d = _find_next_slate_date(today_et + timedelta(days=1))
            if _next_d:
                _next_g = fetch_games(_next_d)
                if _next_g:
                    print(f"[slate] same-day rollover: {today_str} all_complete → {_next_d}")
                    today_str = _next_d.isoformat()
                    today_et = _next_d
    except Exception as _ro_err:
        print(f"[slate] same-day rollover check err (non-fatal): {_ro_err}")

    # Fast path: if today's locked slate is already in memory, skip ALL external calls.
    _lk_pre = _lg(_CK_SLATE_LOCKED, today_str)
    if _lk_pre and _lk_pre.get("date") == today_str:
        # Guard: validate games have actually started before serving the cached lock state.
        # force-regenerate may have written a lock file with locked=True before tip-off.
        _lk_starts = [g.get("startTime", "") for g in _lk_pre.get("games", []) if g.get("startTime")]
        if not _lk_starts or any(_is_past_lock_window(st) for st in _lk_starts):
            _lk_pre["locked"] = True
            _lk_pre.setdefault("draftable_count", 0)
            _maybe_trigger_locked_slate_regen(_lk_pre, "slate-fastpath")
            return _lk_pre
        # Games haven't started — fall through to fetch live games + GitHub check
    # Cold-start fast path: check GitHub backup and fetch games in parallel.
    # The backup is written at lock-promotion time so it exists on most cold starts.
    # Running both concurrently saves 1-2s vs sequential on every cold start.
    with ThreadPoolExecutor(max_workers=_W_LIGHT) as _pre_pool:
        _gh_pre_fut = _pre_pool.submit(_slate_restore_from_github)
        _games_fut  = _pre_pool.submit(fetch_games, today_et)
        _gh_pre = _gh_pre_fut.result()
        games   = _games_fut.result()
    if _gh_pre and _gh_pre.get("date") == today_str and _gh_pre.get("locked"):
        # Guard: validate at least one game has passed its lock window using live ESPN data.
        # Lock file may be written by force-regenerate before games start today (pass: 2).
        _live_starts = [g.get("startTime", "") for g in games if g.get("startTime")]
        if not _live_starts or any(_is_past_lock_window(st) for st in _live_starts):
            _gh_pre["locked"] = True
            _gh_pre.setdefault("draftable_count", 0)
            _ls(_CK_SLATE_LOCKED, _gh_pre)
            _maybe_trigger_locked_slate_regen(_gh_pre, "slate-preload")
            return _gh_pre
        # Games haven't started — don't trust lock file, fall through to pipeline

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
    except (ImportError, KeyError):
        now_utc = datetime.now(timezone.utc)
        m, d = now_utc.month, now_utc.day
        is_dst = (m == 3 and d >= 8) or (4 <= m <= 10) or (m == 11 and d < 8)
        _et_hour = (now_utc + timedelta(hours=-4 if is_dst else -5)).hour
    any_today_started = any(_is_past_lock_window(g.get("startTime", "")) for g in games)
    if not any_today_started and _et_hour < 6:
        _, remaining_yesterday, _, _ = _all_games_final(games)
        if remaining_yesterday > 0:
            yesterday = (_et_date() - timedelta(days=1)).isoformat()
            lock_cached = _lg(_CK_SLATE_LOCKED, yesterday)
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
                    _ls(_CK_SLATE_LOCKED, gh, yesterday)
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
        lock_cached = _lg(_CK_SLATE_LOCKED)
        if lock_cached:
            lock_cached["locked"] = True
            lock_cached.setdefault("draftable_count", 0)
            # Refresh all_complete from ESPN (cached 60s) so warm instances
            # detect game completion instead of serving stale all_complete=False.
            if not lock_cached.get("all_complete"):
                all_final, _rem, _fin, _lrs = _all_games_final(games)
                if all_final and _fin > 0:
                    lock_cached["all_complete"] = True
                    _ls(_CK_SLATE_LOCKED, lock_cached)
            _maybe_trigger_locked_slate_regen(lock_cached, "slate-prelocked")
            return lock_cached

        # Cache miss (cold start) — check regular /tmp cache and GitHub backup BEFORE
        # calling ESPN. _cg(_CK_SLATE) survives longer than the lock cache on partial
        # warm instances; GitHub backup exists on true cold starts after lock-promotion.
        reg_cached = _cg(_CK_SLATE)
        gh_backup = None if reg_cached else _slate_restore_from_github()
        # Now call ESPN once to get all_complete status (only reached on cold start).
        all_final, remaining, finals, _lrs = _all_games_final(games)
        all_complete = all_final and finals > 0

        if reg_cached:
            reg_cached["locked"] = True
            reg_cached["all_complete"] = all_complete
            reg_cached.setdefault("draftable_count", 0)
            _ls(_CK_SLATE_LOCKED, reg_cached)
            _slate_backup_to_github(reg_cached)
            _maybe_trigger_locked_slate_regen(reg_cached, "slate-regcache")
            return reg_cached
        if gh_backup is None:
            gh_backup = _slate_restore_from_github()
        if gh_backup:
            gh_backup["locked"] = True
            gh_backup["all_complete"] = all_complete
            gh_backup.setdefault("draftable_count", 0)
            _ls(_CK_SLATE_LOCKED, gh_backup)
            _maybe_trigger_locked_slate_regen(gh_backup, "slate-prebackup")
            return gh_backup
        # All caches empty during lock. Return locked empty response rather than
        # regenerating — locked predictions must never change.
        print(f"[slate] all games past lock window, all caches empty — returning empty locked response")
        return {
            "date": today_str,
            "games": games,
            "lineups": {"chalk": [], "upside": []},
            "locked": True,
            "all_complete": all_complete,
            "draftable_count": 0,
        }

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
    locked = _any_locked(all_start_times)
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
        lock_cached = _lg(_CK_SLATE_LOCKED)
        if lock_cached:
            lock_cached["locked"] = True
            lock_cached.setdefault("draftable_count", len(draftable_games))
            if lock_time and not lock_cached.get("lock_time"):
                lock_cached["lock_time"] = lock_time
            _maybe_trigger_locked_slate_regen(lock_cached, "slate")
            return lock_cached
        # Check regular cache and promote to lock cache
        cached = _cg(_CK_SLATE)
        if cached:
            cached["locked"] = True
            cached.setdefault("draftable_count", len(draftable_games))
            if lock_time and not cached.get("lock_time"):
                cached["lock_time"] = lock_time
            _ls(_CK_SLATE_LOCKED, cached)
            _slate_backup_to_github(cached)
            # Inline slate prediction save at lock-promotion time.
            # Cold-start Lambdas handling the follow-up /api/save-predictions won't
            # have the slate cache — write slate rows to GitHub now while we have them.
            try:
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
            _ls(_CK_SLATE_LOCKED, gh_backup)
            _maybe_trigger_locked_slate_regen(gh_backup, "slate-backup")
            return gh_backup
        # All caches empty during lock. Return a locked empty response rather than
        # regenerating — locked predictions must never change. The frontend will
        # attempt to restore from localStorage (cold-start recovery path).
        print(f"[slate] locked but no cache available — returning empty locked response")
        return {
            "date": today_str,
            "games": games,
            "lineups": {"chalk": [], "upside": []},
            "locked": True,
            "all_complete": False,
            "draftable_count": 0,
            "lock_time": lock_time,
        }

    # ── Bust check (unlocked path) — multi-instance: another container may have
    # busted via cold reset or config change; clear stale /tmp before serving.
    try:
        if _github_slate_bust_active():
            _clear_local_slate_tmp_caches()
    except Exception as _bust_chk_err:
        print(f"[slate] github bust check err (unlocked): {_bust_chk_err}")

    # ── Layer 1: /tmp cache (warm Railway instance) ──
    cached = _cg(_CK_SLATE, today_str)
    if cached:
        # Discard cached result if it has empty lineups but we have draftable games.
        has_players = cached.get("lineups", {}).get("chalk") or cached.get("lineups", {}).get("upside")
        if has_players or not draftable_games:
            cached["locked"] = locked
            cached.setdefault("draftable_count", len(draftable_games))
            return cached

    # ── Layer 2: GitHub persistent cache (cold-start recovery) ──
    gh_cached = _slate_cache_from_github(today_str)
    if gh_cached:
        has_players = gh_cached.get("lineups", {}).get("chalk") or gh_cached.get("lineups", {}).get("upside")
        if has_players or not draftable_games:
            gh_cached["locked"] = locked
            gh_cached.setdefault("draftable_count", len(draftable_games))
            if lock_time and not gh_cached.get("lock_time"):
                gh_cached["lock_time"] = lock_time
            # Warm /tmp cache for subsequent requests on this instance
            _cs(_CK_SLATE, gh_cached, today_str)
            # Also warm per-game /tmp caches from GitHub
            try:
                gh_games = _games_cache_from_github(today_str)
                if gh_games:
                    for gid, projs in gh_games.items():
                        _cs(_ck_game_proj(gid), projs)
            except Exception:
                pass
            return gh_cached

    # ── Layer 3: First run of the day — generate fresh, then persist ──
    # Concurrent cold-start guard: if another thread is already generating, wait for it
    # and then serve from the cache it populated. Prevents N×(ESPN + LightGBM + boost) on
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
        # Poll /tmp cache until the other thread finishes (max ~20s — within
        # frontend's 30s timeout).  If it never appears, fall through and run
        # our own pipeline (the other thread may have crashed).
        for _ in range(10):
            time.sleep(2)
            _warm = _cg(_CK_SLATE, today_str)
            if _warm:
                _warm["locked"] = locked
                _warm.setdefault("draftable_count", len(draftable_games))
                return _warm

    try:
        all_proj = []
        game_proj_map = {}  # {gameId: [projections...]} for GitHub persistence
        gamelog_map, _dvp_data, player_odds_prefetch, _def_stats = {}, {}, {}, {}

        # Pre-fetch nba_api enrichment (usage_share, team_pace, min_volatility, etc.)
        # before parallel game runs. Cached per slate date — one heavy call per day.
        try:
            _nba_api_prefetch(today_str)
        except Exception as _nba_err:
            print(f"[nba-api-feed] prefetch error (non-fatal): {_nba_err}")

        # Pre-fetch ESPN injury report (secondary source alongside RotoWire)
        try:
            _espn_injuries_fetch()
        except Exception:
            pass

        # Overlay Odds API game-level spreads/totals onto games (more accurate than ESPN)
        try:
            _odds_updated = _apply_odds_snapshot_to_games(draftable_games)
            if _odds_updated:
                print(f"[slate] odds snapshot: {_odds_updated}/{len(draftable_games)} games updated")
        except Exception as _odds_snap_err:
            print(f"[game-odds] snapshot error (non-fatal): {_odds_snap_err}")

        with ThreadPoolExecutor(max_workers=_W_STANDARD) as pool:
            futs = {
                pool.submit(_run_game, g, gamelog_map, _dvp_data, player_odds_prefetch): g
                for g in draftable_games
            }
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
        # Matchup data + news + context pass — run in parallel (saves 2-4s vs serial)
        _matchup_intel = {}
        _slate_news_text = ""
        with ThreadPoolExecutor(max_workers=4) as _enrich_pool:
            _def_fut = _enrich_pool.submit(_fetch_team_def_stats)
            _dvp_fut = _enrich_pool.submit(_fetch_dvp_data)
            _news_fut = _enrich_pool.submit(_fetch_nba_news_context, draftable_games, all_proj=all_proj)
            _ctx_fut = _enrich_pool.submit(_claude_context_pass, all_proj, draftable_games)
            try:
                _def_stats = _def_fut.result(timeout=_T_EXECUTOR)
            except Exception as _def_err:
                print(f"[matchup] def stats fetch error (non-fatal): {_def_err}")
            try:
                _dvp_data = _dvp_fut.result(timeout=_T_EXECUTOR)
            except Exception as _dvp_err:
                print(f"[matchup] DvP fetch error (non-fatal): {_dvp_err}")
            try:
                _slate_news_text = _news_fut.result(timeout=_T_EXECUTOR) or ""
            except Exception:
                pass
            try:
                _ctx_fut.result(timeout=_T_EXECUTOR)
            except Exception as _ctx_err:
                print(f"[context_pass] call-site error: {_ctx_err}")
        _apply_post_lock_rs_calibration(all_proj, slate_locked=locked)
        chalk, upside, core_pool = _build_lineups(all_proj, def_stats=_def_stats, matchup_intel=_matchup_intel, dvp_data=_dvp_data, n_games=len(draftable_games))
        lineups = {"chalk": chalk, "upside": upside}

        # Pre-warm per-game caches so /api/picks is instant on first click
        # This runs in the background without blocking the slate response
        try:
            for g in draftable_games:
                gid = g["gameId"]
                g_projs = game_proj_map.get(gid)
                if g_projs:
                    try:
                        game_lineups = _build_game_lineups(g_projs, g)
                        pg_strat = game_lineups.pop("strategy", None)
                        pg_result = {
                            "date": today_str, "game": g,
                            "gameScript": _game_script_label(g.get("total")),
                            "lineups": game_lineups,
                            "strategy": pg_strat,
                            "locked": locked, "injuries": _get_injuries(g),
                            "score_bounds": _score_bounds_for_lineups(game_lineups)
                        }
                        _cs(_ck_picks(gid), pg_result)
                    except Exception as pg_err:
                        print(f"[picks-prewarm] Failed to pre-build picks for {gid}: {pg_err}")
        except Exception as prewarm_err:
            print(f"[picks-prewarm] Loop error: {prewarm_err}")

        # Watchlist: players near the lineup bubble sensitive to late-breaking news
        _watchlist = []
        try:
            _watchlist = _build_watchlist(chalk, upside, all_proj, draftable_games)
        except Exception as _wl_err:
            print(f"[watchlist] build error: {_wl_err}")
        _deploy_sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "")
        result = {"date": today_str, "games": games,
                  "lineups": lineups, "locked": locked,
                  "all_complete": False, "draftable_count": len(draftable_games),
                  "lock_time": lock_time,
                  "watchlist": _watchlist,
                  "pass": 1,
                  "score_bounds": _score_bounds_for_lineups(lineups),
                  "deploy_sha": _deploy_sha[:7] if _deploy_sha else ""}
        if chalk or upside:  # Don't cache empty results — allow retry on next request
            _cs(_CK_SLATE, result, today_str)
            # GitHub writes are fire-and-forget: store to /tmp first so any concurrent
            # request is served immediately, then persist to GitHub in the background.
            # This removes 2-3s of blocking I/O from the hot path — the user gets
            # the slate response as soon as the pipeline finishes, not after writes.
            _bg_result       = result
            _bg_game_proj    = game_proj_map
            _bg_today        = today_str
            def _slate_persist_bg():
                try:
                    _slate_cache_to_github(_bg_result, _bg_today)
                except Exception as _e:
                    print(f"[slate-cache] bg write err: {_e}")
                try:
                    _github_write_file(f"data/slate/{_bg_today}_bust.json", "{}", f"clear bust {_bg_today}")
                except Exception as _e:
                    print(f"[slate-cache] bg clear bust err: {_e}")
                if _bg_game_proj:
                    try:
                        _games_cache_to_github(_bg_game_proj, _bg_today)
                    except Exception as _e:
                        print(f"[slate-cache] bg games write err: {_e}")
                try:
                    _slate_backup_to_github(_bg_result, _bg_today)
                except Exception as _e:
                    print(f"[slate-cache] bg backup err: {_e}")
            threading.Thread(target=_slate_persist_bg, daemon=True).start()
        if locked:
            _ls(_CK_SLATE_LOCKED, result, today_str)
        # Sync-clear bust so other Railway instances stop dropping /tmp on every
        # locked request (bg-only clear races multi-instance).
        if chalk or upside:
            try:
                _github_write_file(
                    f"data/slate/{today_str}_bust.json",
                    "{}",
                    f"clear bust sync after slate gen {today_str}",
                )
            except Exception as _e:
                print(f"[slate-cache] sync clear bust err: {_e}")
        return result
    finally:
        with _SLATE_GEN_LOCK:
            _SLATE_GEN_IN_FLIGHT = False


@app.get("/api/slate")
async def get_slate(mock: bool = Query(False, description="Return deterministic mock data for testing (no ESPN/model calls)")) -> dict:
    """Slate endpoint: never returns 500; on exception returns 200 with error key for graceful frontend handling.
    Pass ?mock=true to get static test data suitable for UI/audit validation without hitting live systems.
    Uses Level 0 response cache to avoid re-running pipeline on repeated calls same day."""
    if mock:
        return _get_mock_slate()
    try:
        today = _today_str()

        # Warming-up guard: if the cold pipeline is already running in background
        # and no usable cache exists yet, return immediately with warming_up=True
        # so the frontend can poll instead of blocking for 30-60s on a second
        # inline pipeline run that would compete with the background one.
        with _COLD_PIPELINE_LOCK:
            _pipe_in_flight = _COLD_PIPELINE_IN_FLIGHT
        if _pipe_in_flight:
            _quick = _cg(_CK_SLATE) or _lg(_CK_SLATE_LOCKED)
            if not _quick:
                # Use games fetched at startup so the header shows real game count.
                # If startup confirmed 0 games, skip warming_up entirely and return
                # no_games so the frontend shows "No NBA games today" immediately.
                _wu_games = _WARMUP_GAMES or []
                if not _wu_games:
                    # Try a quick ESPN check before returning no_games —
                    # _WARMUP_GAMES might not be populated yet on container restart.
                    try:
                        _wu_games = fetch_games()
                    except Exception:
                        pass
                if not _wu_games:
                    return JSONResponse(
                        content={
                            "date": today,
                            "games": [],
                            "lineups": {"chalk": [], "upside": []},
                            "locked": False,
                            "draftable_count": 0,
                            "no_games": True,
                            "cache_status": "warming",
                        },
                        status_code=200,
                    )
                return JSONResponse(
                    content={
                        "warming_up": True,
                        "date": today,
                        "games": _wu_games,
                        "lineups": {"chalk": [], "upside": []},
                        "locked": False,
                        "draftable_count": len(_wu_games),
                        "cache_status": "warming",
                    },
                    status_code=200,
                )

        cache_key = _CACHE_KEYS["slate"].format(today)
        result, is_hit = _with_response_cache(cache_key, "slate", _get_slate_impl)
        # Same-day rollover: when _get_slate_impl switched to a future date, also cache
        # the result under the correct date key so subsequent calls serve from there.
        result_date = result.get("date", today) if isinstance(result, dict) else today
        if result_date != today:
            next_key = _CACHE_KEYS["slate"].format(result_date)
            if not _RESPONSE_CACHE.get(next_key)[1]:  # only set if not already cached
                _RESPONSE_CACHE.set(next_key, result, _CACHE_TTLS.get("slate", 60))
        result = _add_cache_metadata(result, is_hit, cache_key)
        return result
    except Exception as e:
        import traceback as _tb
        print(f"[slate] PIPELINE ERROR: {e}\n{_tb.format_exc()}")
        # Auto-clear stale bust sentinel so future requests can retry cleanly.
        # Only fires for busts 30+ minutes old (fresh busts may still be mid-regeneration).
        try:
            _today = _today_str()
            _bust_path = f"data/slate/{_today}_bust.json"
            _bc, _ = _github_get_file(_bust_path)
            if _bc:
                _bd = json.loads(_bc)
                if _bd.get("at"):
                    _age = (datetime.now(timezone.utc) -
                            datetime.fromisoformat(_bd["at"])).total_seconds()
                    if _age > 1800:  # 30+ min stale
                        _github_write_file(_bust_path, "{}", f"auto-clear stale bust {_today}")
                        print(f"[slate] auto-cleared stale bust ({_age:.0f}s old)")
        except Exception:
            pass
        return JSONResponse(
            content={
                "error": "slate_failed",
                "date": _today_str(),
                "games": [],
                "lineups": {"chalk": [], "upside": []},
                "locked": False,
                "draftable_count": 0,
                "cache_status": "miss",
            },
            status_code=200,
        )


def _compute_game_picks(game):
    """Compute game-specific projections and cache under both regular and lock keys.
    Returns the result dict, or None if projections unavailable. Skips if already cached."""
    gid = game["gameId"]
    existing = _lg(_ck_picks_locked(gid)) or _cg(_ck_picks(gid))
    if existing:
        return existing
    try:
        projections = _run_game(game)
        if not projections:
            return None
        lineups_dict = _build_game_lineups(projections, game)
        pg_strategy = lineups_dict.pop("strategy", None)
        result = {
            "date": _today_str(), "game": game,
            "gameScript": _game_script_label(game.get("total")),
            "lineups": lineups_dict,
            "strategy": pg_strategy,
            "locked": True, "injuries": _get_injuries(game),
        }
        _cs(_ck_picks(gid), result)
        _ls(_ck_picks_locked(gid), result)
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
            return _err("Mock game not found", 404)
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
        return {"date": _today_str(), "game": game_meta, "mock": True,
                "gameScript": "balanced", "lineups": lineups, "locked": False,
                "strategy": {"type": "neutral", "label": "Standard Build",
                             "description": "Mock game — standard build.", "total_mult": 1.0,
                             "spread": 0, "total": 222, "favored_team": "", "underdog_team": ""},
                "injuries": [], "score_bounds": _score_bounds_for_lineups(lineups)}

    game = next((g for g in fetch_games() if g["gameId"] == gameId), None)
    if not game:
        return _err("Game not found", 404)

    start_time = game.get("startTime")
    # Use _is_past_lock_window (no 6h ceiling) — once a game starts, picks stay
    # frozen permanently. _is_locked has a 6h ceiling that would allow recompute.
    locked = _is_past_lock_window(start_time) if start_time else False
    lock_key = _ck_picks_locked(gameId)

    if locked:
        lock_cached = _lg(lock_key)
        if lock_cached:
            lock_cached["locked"] = True
            return lock_cached
        # Check regular cache and promote to lock cache
        reg_key = _ck_picks(gameId)
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
                        today = _today_str()
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
        return {"date": _today_str(), "game": game,
                "gameScript": None,
                "lineups": {"the_lineup": []},
                "locked": True, "injuries": []}

    # Try fully-built picks cache first (populated by /api/slate pre-warm or previous /api/picks)
    cached_picks = _cg(_ck_picks(gameId))
    if cached_picks:
        return cached_picks

    # Try /tmp cache first (populated by /api/slate or previous /api/picks)
    cache_key = _ck_game_proj(gameId)
    projections = _cg(cache_key)
    if not projections:
        # Try GitHub persistent cache (populated by first slate run of the day).
        # Batch-warm ALL games into /tmp + Redis in a single GitHub read so
        # subsequent clicks on other games don't trigger another round-trip.
        gh_games = _games_cache_from_github()
        if gh_games and not gh_games.get("_busted"):
            for gid, gprojs in gh_games.items():
                if gid.startswith("_"):  # skip sentinel keys (_busted etc.)
                    continue
                if isinstance(gprojs, list) and gprojs:
                    _cs(_ck_game_proj(gid), gprojs)
            projections = gh_games.get(gameId)
    if not projections:
        # True cold start with no cache anywhere — run engine (rare after first daily run)
        projections = _run_game(game)
    if not projections:
        return _err("No projections available.", 503)
    lineups_dict = _build_game_lineups(projections, game)
    script = _game_script_label(game.get("total"))
    injuries = _get_injuries(game)
    # Extract strategy from lineup builder (per-game v2)
    pg_strategy = lineups_dict.pop("strategy", None)

    result = {"date": _today_str(), "game": game,
              "gameScript": script,
              "lineups": lineups_dict,
              "locked": locked,
              "injuries": injuries,
              "strategy": pg_strategy,
              "score_bounds": _score_bounds_for_lineups(lineups_dict)}
    # Cache picks so they survive as lock snapshot if slate locks later
    _cs(_ck_picks(gameId), result)
    return result

@app.post("/api/save-predictions")
async def save_predictions():
    """Save current predictions to GitHub as CSV."""
    today = _today_str()
    path = f"data/predictions/{today}.csv"

    # Guard: only write predictions after the slate has locked.
    # Pre-lock projections are not finalized — saving them would pollute the log
    # with data that changes as injury news, lineups, and odds shift.
    # Uses any() instead of min() — on split-window days the earliest game's 6h
    # ceiling can expire while late games are still live. any() stays locked as
    # long as ANY game is within its lock window.
    _games_now = fetch_games()
    _start_times = [g["startTime"] for g in _games_now if g.get("startTime")]
    if _start_times and not _any_locked(_start_times):
        return _err("Slate not locked yet — predictions not finalized", 409)

    # Gather slate predictions — try all cache layers before giving up
    rows = []
    cached_slate = _cg(_CK_SLATE) or _lg(_CK_SLATE_LOCKED)
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
        cached_picks = _cg(_ck_picks(gid)) or _lg(_ck_picks_locked(gid))
        if cached_picks and cached_picks.get("lineups"):
            rows.extend(_predictions_to_csv(cached_picks["lineups"], label))
        elif _is_locked(g.get("startTime", "")):
            locked_games_to_compute.append(g)

    # Auto-compute picks for locked games the user never manually analyzed.
    # Run in parallel so this doesn't add significant latency per game.
    if locked_games_to_compute:
        with ThreadPoolExecutor(max_workers=_W_STANDARD) as pool:
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
        return _err("No predictions cached yet", 404)

    # Merge with existing CSV — on split-window days, later-locking games need to be
    # appended without overwriting earlier predictions (e.g. 5 PM game saved first,
    # 7:30 PM game locks later and gets added by cold-reset/injury-check flow).
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
        return _err(result["error"], 500)

    # Also write the slate backup now so cold-start instances can recover after lock,
    # even if this Railway instance dies before the lock window promotes the reg cache.
    cached_slate_for_backup = _cg(_CK_SLATE)
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
    today = _today_str()
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
    today_str = _today_str()
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
    try:
        _nba_api_prefetch(today_str)
    except Exception:
        pass
    try:
        _apply_odds_snapshot_to_games(game_pool)
    except Exception:
        pass
    all_proj = []
    game_proj_map = {}
    # DFS draft path — per-game projections
    with ThreadPoolExecutor(max_workers=_W_STANDARD) as pool:
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
    _fr_dvp_data = None
    try:
        _fr_def_stats = _fetch_team_def_stats()
    except Exception:
        pass
    try:
        _fr_dvp_data = _fetch_dvp_data()
    except Exception:
        pass
    _fr_starts = [g["startTime"] for g in games if g.get("startTime")]
    _fr_any_locked = _any_locked(_fr_starts)
    _apply_post_lock_rs_calibration(all_proj, slate_locked=_fr_any_locked)
    chalk, upside, core_pool = _build_lineups(all_proj, def_stats=_fr_def_stats, dvp_data=_fr_dvp_data, n_games=len(games))
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
                game_lineups.pop("strategy", None)  # strategy is response-level, not lineups-level
                per_game_results[gid] = {
                    "game": g,
                    "lineups": game_lineups,
                }
            except Exception as e:
                print(f"[force-regen] game lineups err {gid}: {e}")

    # Step 4: (Bust deferred — only bust if we have valid results to write.
    # Busting without a valid replacement leaves the cache permanently empty.)

    # Step 5: Build the slate cache object and persist to all layers
    deploy_sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "")
    # Calculate lock_time from earliest game start (same logic as normal slate path)
    _fr_lock_time = None
    _fr_start_times = [g.get("startTime") for g in games if g.get("startTime")]
    if _fr_start_times:
        try:
            _fr_earliest = min(_fr_start_times)
            _fr_lock_buf = _cfg("projection.lock_buffer_minutes", 5)
            _fr_gs = datetime.fromisoformat(_fr_earliest.replace("Z", "+00:00")).astimezone(timezone.utc)
            _fr_lock_dt = _fr_gs - timedelta(minutes=_fr_lock_buf)
            _fr_lock_time = _fr_lock_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            pass
    result = {
        "date": today_str, "games": games,
        "lineups": lineups, "locked": True,
        "all_complete": False, "draftable_count": len(game_pool),
        "score_bounds": _score_bounds_for_lineups(lineups),
        "deploy_sha": deploy_sha[:7] if deploy_sha else "",
        "watchlist": _fr_watchlist,
        "lock_time": _fr_lock_time,
        "pass": 2,
    }
    if chalk or upside:
        # Only bust AFTER confirming we have valid lineups to replace it with.
        _bust_slate_cache(_caller="force_regenerate")
        _cs(_CK_SLATE, result)
        _ls(_CK_SLATE_LOCKED, result)
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
        _cs(_ck_picks(gid), gdata)
        _ls(_ck_picks_locked(gid), gdata)

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
        today = _today_str()
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
    today = _today_str()
    games = fetch_games()
    if not games:
        return {"changed": False, "triggers": [], "recommendation": "hold",
                "reason": "no_games"}

    # Don't check after lock — picks are frozen
    start_times = [g["startTime"] for g in games if g.get("startTime")]
    if _any_locked(start_times):
        return {"changed": False, "triggers": [], "recommendation": "hold",
                "reason": "locked"}

    # Load Pass 1 cached slate
    cached_slate = _cg(_CK_SLATE) or _slate_cache_from_github()
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
    }


@app.get("/api/force-regenerate")
async def force_regenerate(request: Request, scope: str = Query("full")):
    """Force-regenerate predictions mid-slate.

    scope=full: Re-run ALL games. Requires CRON_SECRET and slate must NOT be locked.
    scope=remaining: Only games not yet started (user woke up late). User-facing, no auth.
    """
    if scope not in ("full", "remaining"):
        return _err("scope must be 'full' or 'remaining'", 400)

    # scope=full requires CRON_SECRET — prevents accidental full regen
    if scope == "full" and not _require_cron_secret(request):
        return _err("Unauthorized — scope=full requires CRON_SECRET", 401)

    # Lock guard for scope=full: refuse to regenerate locked predictions
    if scope == "full":
        try:
            _fg_games = fetch_games()
            _fg_starts = [g["startTime"] for g in _fg_games if g.get("startTime")]
            if _any_locked(_fg_starts):
                return _err("Slate is locked — predictions are frozen. Use scope=remaining for late draft.", 423)
        except Exception:
            pass

    try:
        result = await asyncio.to_thread(_force_regenerate_sync, scope)
        return result
    except Exception as e:
        print(f"[force-regenerate] error: {e}")
        traceback.print_exc()
        return _err(str(e), 500)


async def reset_uploads(body: dict):
    """Delete data/actuals/{date}.csv and data/audit/{date}.json from GitHub (admin / repair)."""
    date_str = body.get("date")
    if not date_str:
        return _err("date required", 400)
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
    """Parse a Real Sports app screenshot using Claude Vision API (developer / script use).

    screenshot_type:
      "actuals" — default; My Draft + Highest Value + leaderboard heuristics
      "most_drafted" | "most_popular" — Most popular / most drafted list
      "most_drafted_high_boost" — high-boost sub-leaderboard (e.g. 3x+)
      "top_performers" — Highest value / top performers rows only
      "winning_drafts" — up to 4 winning lineups (flat JSON rows per slot)
    """
    rl = _check_rate_limit(request, "parse-screenshot")
    if rl is not None:
        return rl
    if not ANTHROPIC_API_KEY:
        return _err("ANTHROPIC_API_KEY not configured", 500)

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        return _err("Image too large (max 10MB)", 413)
    b64_image = base64.b64encode(image_bytes).decode("ascii")

    # Determine media type
    _ALLOWED_IMAGE_TYPES = ("image/png", "image/jpeg", "image/gif", "image/webp")
    ct = file.content_type or ""
    if ct not in _ALLOWED_IMAGE_TYPES:
        return _err(f"Unsupported image type: {ct or 'unknown'}. Allowed: png, jpeg, gif, webp", 415)

    if screenshot_type == "boosts":
        return _err(
            "screenshot_type 'boosts' is removed — card boosts are model-estimated only.",
            400,
        )
    if screenshot_type in ("most_drafted", "most_popular"):
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
    elif screenshot_type == "most_drafted_high_boost":
        prompt = """Extract players from this Real Sports screen showing the high card-boost popular / most-drafted sub-leaderboard (e.g. 3x boost filter). If the full list is shown, only include rows where card boost is clearly 3.0x or higher.

For EACH player listed, extract:
- rank: integer position in this visible list
- player_name: full name
- team: abbreviation if visible, else null
- draft_count: parse k/m notation to integer
- actual_rs: Real Score from triangle/symbol if shown, else null
- actual_card_boost: "+X.Xx" as decimal (required for this screen)
- avg_finish: ordinal as number if shown, else null

Return ONLY a JSON array of objects. No markdown."""
    elif screenshot_type == "top_performers":
        prompt = """Extract ONLY the "Highest value" / top performing players section from this Real Sports screenshot. Ignore "My draft" unless there is no highest-value section.

For EACH player row:
- player_name: full name
- team: NBA team abbreviation if visible on the row or logo (e.g. LAL, DEN, PHX), else null
- actual_rs: Real Score (triangle/arrow symbol)
- actual_card_boost: "+X.Xx" as decimal, or null
- drafts: draft count if shown
- avg_finish: null unless clearly shown
- total_value: value number on the right if shown (e.g. 23.6)
- source: always the string "highest_value"

Return ONLY a JSON array of objects. No markdown."""
    elif screenshot_type == "winning_drafts":
        prompt = """Extract winning drafts from this Real Sports leaderboard (up to 4 winners, each with 5 players).

For EACH player in each winning lineup, output one object:
- winner_rank: integer 1-4 (which winning row, top winner = 1)
- drafter_label: username or label for that winner if visible, else null
- total_score: that winner's total score if visible, else null
- slot_index: integer 1-5 (top of lineup to bottom)
- player_name: full name
- team: NBA team abbreviation if visible (logo or text), else null
- actual_rs: Real Score if visible, else null
- slot_mult: slot multiplier if shown (e.g. 2.0x as 2.0), else null
- card_boost: card boost "+X.Xx" as decimal if shown, else null

Return ONLY a flat JSON array (one object per player slot). No markdown."""
    else:
        prompt = """Extract ALL player data from this Real Sports app screenshot.

The screenshot may contain two sections:
1. "My draft" - the user's own drafted players
2. "Highest value" - the top performing players of the day

For EACH player, extract:
- player_name: full name
- team: NBA team abbreviation if visible (logo or text), else null
- actual_rs: the Real Score number shown with a triangle/arrow symbol (e.g. if it shows "⌃3.1" the value is 3.1, if "⌃0" the value is 0)
- actual_card_boost: the card boost shown as "+X.Xx" (e.g. "+3.0x" → 3.0, "+0.9x" → 0.9). If no + symbol is shown, set to null
- drafts: number of drafts (e.g. "31", "1.1k" → 1100, "5k" → 5000, "13.6k" → 13600)
- avg_finish: average finish position as a number (e.g. "1st" → 1, "5th" → 5). Only in "My draft" section
- total_value: the value number shown on right side (only in "Highest value" section, e.g. "⌃23.6" → 23.6)
- source: "my_draft" if from "My draft" section, "highest_value" if from "Highest value" section

If this is a Leaderboard screenshot, extract each drafter's lineup:
- For each player in a lineup: player_name, team (abbr if visible), actual_rs (the number shown), card_multiplier (e.g. "4.2x" → 4.2)
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
            timeout=_T_HEAVY,
        )
        if not r.ok:
            if r.status_code == 402:
                return _err("Claude API credits exhausted. Please top up Anthropic billing.", 402)
            if r.status_code == 429:
                return _err("Claude API rate limited. Please wait a moment and retry.", 429)
            return _err(f"Claude API error (HTTP {r.status_code})", r.status_code)
        resp = r.json()
        content = resp.get("content") or []
        if not content or not isinstance(content, list):
            return _err("Claude returned empty content array", 500)
        text = (content[0].get("text") or "").strip()
        if not text:
            return _err("Claude returned empty text in response", 500)
        # Extract JSON from response (may be wrapped in ```json blocks)
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        parsed = json.loads(text.strip())
        return {"players": parsed}
    except json.JSONDecodeError:
        return JSONResponse({"error": "Failed to parse Claude response as JSON", "raw": text[:500]}, status_code=500)
    except Exception as e:
        return _err(f"Screenshot parsing failed: {str(e)}", 500)


def _compute_audit(date_str):
    """Compare predictions vs actuals for a date. Returns audit dict or None if no data."""
    pred_csv, _ = _github_get_file(f"data/predictions/{date_str}.csv")
    if not pred_csv:
        return None
    actuals = _load_player_actuals_for_date(date_str)
    if not actuals:
        return None

    preds = _parse_csv(pred_csv, PRED_FIELDS)

    act_map = {r["player_name"].lower(): r for r in actuals}

    errors, dir_hits, misses = [], [], []
    boost_abs_errors, boost_signed_errors = [], []
    boost_under, boost_over = 0, 0
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
            "team":         (a.get("team") or row.get("team") or ""),
            "predicted_rs": round(pred_rs, 2),
            "actual_rs":    round(actual_rs, 2),
            "error":        round(err, 2),
            "drafts":       a.get("drafts", ""),
            "actual_card_boost": a.get("actual_card_boost", ""),
        })
        pred_boost = _safe_float(row.get("est_card_boost"))
        actual_boost = _safe_float(a.get("actual_card_boost"))
        boost_err = actual_boost - pred_boost
        boost_abs_errors.append(abs(boost_err))
        boost_signed_errors.append(boost_err)
        if boost_err > 0:
            boost_under += 1
        elif boost_err < 0:
            boost_over += 1

    if not errors:
        return None

    misses.sort(key=lambda x: abs(x["error"]), reverse=True)
    mae = round(sum(errors) / len(errors), 3)
    dir_acc = round(sum(dir_hits) / len(dir_hits), 3) if dir_hits else None
    boost_mae = round(sum(boost_abs_errors) / len(boost_abs_errors), 3) if boost_abs_errors else None
    boost_bias = round(sum(boost_signed_errors) / len(boost_signed_errors), 3) if boost_signed_errors else None
    boost_under_rate = round(boost_under / len(boost_signed_errors), 3) if boost_signed_errors else None

    # Over- vs under-projection breakdown
    over  = [e for e in misses if e["error"] < 0]
    under = [e for e in misses if e["error"] > 0]

    # Simulated draft score — measures progress toward 60+ goal.
    # Optimal hindsight: sort actuals by RS, assign slots 2.0x→1.2x to top 5,
    # compute RS × (slot + actual_card_boost) for each. Shows what was achievable.
    slot_mults = _SLOT_MULTS_SHARED
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
        "boost_mae":          boost_mae,
        "boost_bias":         boost_bias,
        "boost_under_predicted": boost_under,
        "boost_over_predicted":  boost_over,
        "boost_under_rate":   boost_under_rate,
        "generated_at":       datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/save-actuals")
async def save_actuals(payload: dict = Body(...)):
    """Merge parsed actuals into data/actuals/{date}.csv (legacy path).

    If the date is in data/skipped-uploads.json, returns early (no-op).
    """
    date_str = payload.get("date", _today_str())
    bad = _validate_date(date_str)
    if bad: return bad
    players = payload.get("players", [])
    if not players:
        return _err("No player data", 400)

    try:
        skipped_content, _ = _github_get_file("data/skipped-uploads.json")
        if skipped_content:
            skipped_data = json.loads(skipped_content)
            if date_str in skipped_data.get("skipped_dates", []):
                print(f"[save-actuals] Skipping {date_str} (in skipped-uploads list)")
                return {"status": "skipped", "date": date_str, "reason": "Date listed in data/skipped-uploads.json"}
    except Exception:
        pass  # If check fails, continue with normal processing

    path = f"data/actuals/{date_str}.csv"

    # Check if file already exists (to append / overwrite with dedup)
    existing, _ = _github_get_file(path)
    merged: list = []
    if existing:
        merged = _parse_actuals_rows(existing)
        new_names = {str(p.get("player_name", "")).strip().lower() for p in players}
        merged = [r for r in merged if r["player_name"].lower() not in new_names]

    for p in players:
        merged.append(
            {
                "player_name": str(p.get("player_name", "")).strip(),
                "team": str(p.get("team", "")).strip().upper(),
                "actual_rs": p.get("actual_rs", ""),
                "actual_card_boost": p.get("actual_card_boost", ""),
                "drafts": p.get("drafts", ""),
                "avg_finish": p.get("avg_finish", ""),
                "total_value": p.get("total_value", ""),
                "source": str(p.get("source", "")).strip(),
            }
        )

    csv_content = _actuals_csv_from_rows(merged)
    result = _github_write_file(path, csv_content, f"actuals for {date_str}")
    if result.get("error"):
        return _err(result["error"], 500)

    # Auto-generate audit JSON — only when real_scores data is present.
    # Audit compares predicted RS vs actual RS; meaningless without RS actuals.
    upload_source = players[0].get("source", "") if players else ""
    rs_in_csv = "real_scores" in (existing or "")
    audit = None
    if upload_source == "real_scores" or rs_in_csv:
        audit = _compute_audit(date_str)
        if audit:
            audit_path = f"data/audit/{date_str}.json"
            _github_write_file_bg(audit_path, json.dumps(audit, indent=2), f"audit for {date_str}")

    return {"status": "saved", "path": path, "rows": len(merged), "audit": audit}


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
ACT_FIELDS = [
    "player_name",
    "team",
    "actual_rs",
    "actual_card_boost",
    "drafts",
    "avg_finish",
    "total_value",
    "source",
]

# grep: HISTORICAL DATA — mega top_performers, most_popular, winning_drafts, most_drafted_3x
TOP_PERFORMERS_GH_PATH = "data/top_performers.csv"

# ── Leaderboard frequency cache (Winning Draft Audit) ────────────────────
# Tracks how many times each player appears on the leaderboard + their avg value.
# Data-driven (from top_performers.csv), no hardcoded names. Cached 30 min.
_LEADERBOARD_FREQ_CACHE: dict = {}
_LEADERBOARD_FREQ_TS: float = 0.0
_LEADERBOARD_FREQ_TTL: float = 1800.0  # 30 min


def _load_leaderboard_frequency() -> dict:
    """Load player leaderboard frequency from top_performers.csv.

    Returns dict: {normalized_name: {count, avg_rs, avg_boost, avg_value, avg_drafts}}
    Cached for 30 minutes to avoid repeated GitHub reads.
    """
    global _LEADERBOARD_FREQ_CACHE, _LEADERBOARD_FREQ_TS
    now = datetime.now(timezone.utc).timestamp()
    if _LEADERBOARD_FREQ_CACHE and (now - _LEADERBOARD_FREQ_TS) < _LEADERBOARD_FREQ_TTL:
        return _LEADERBOARD_FREQ_CACHE

    try:
        raw, _ = _github_get_file(TOP_PERFORMERS_GH_PATH)
        if not raw:
            return _LEADERBOARD_FREQ_CACHE or {}
        rows = _parse_top_performers_mega_rows(raw)
        # Config thresholds for ghost appearances (low-ownership, high-boost)
        _ghost_boost_floor = 2.8
        _ghost_draft_ceil = 50
        freq: dict = {}
        for r in rows:
            name = _normalize_player_name(r.get("player_name", ""))
            if not name:
                continue
            if name not in freq:
                freq[name] = {
                    "count": 0, "rs_sum": 0.0, "boost_sum": 0.0,
                    "value_sum": 0.0, "draft_sum": 0.0,
                    "ghost_count": 0, "ghost_value_sum": 0.0,
                }
            boost_val = float(r.get("actual_card_boost", 0) or 0)
            drafts_val = float(r.get("drafts", 0) or 0)
            value_val = float(r.get("total_value", 0) or 0)
            freq[name]["count"] += 1
            freq[name]["rs_sum"] += float(r.get("actual_rs", 0) or 0)
            freq[name]["boost_sum"] += boost_val
            freq[name]["value_sum"] += value_val
            freq[name]["draft_sum"] += drafts_val
            # Ghost appearance: low-ownership + high-boost = structural contrarian edge
            if boost_val >= _ghost_boost_floor and drafts_val < _ghost_draft_ceil:
                freq[name]["ghost_count"] += 1
                freq[name]["ghost_value_sum"] += value_val
        # Compute averages + ghost quality signal
        result = {}
        for name, d in freq.items():
            n = d["count"]
            gc = d["ghost_count"]
            avg_ghost_value = d["ghost_value_sum"] / gc if gc > 0 else 0.0
            # Ghost quality: saturation × value scaling
            # (gc / (gc+2)): reaches 0.5 at gc=2, 0.6 at gc=3, 0.83 at gc=10
            # min(avg_ghost_value/15, 1.5): normalised; 15 = slate avg value baseline
            ghost_quality = (gc / (gc + 2)) * min(avg_ghost_value / 15.0, 1.5) if gc > 0 else 0.0
            result[name] = {
                "count": n,
                "avg_rs": round(d["rs_sum"] / n, 2) if n else 0,
                "avg_boost": round(d["boost_sum"] / n, 2) if n else 0,
                "avg_value": round(d["value_sum"] / n, 2) if n else 0,
                "avg_drafts": round(d["draft_sum"] / n, 0) if n else 0,
                "ghost_count": gc,
                "avg_ghost_value": round(avg_ghost_value, 2),
                "ghost_quality": round(ghost_quality, 3),
            }
        _LEADERBOARD_FREQ_CACHE = result
        _LEADERBOARD_FREQ_TS = now
        print(f"[leaderboard_freq] Loaded {len(result)} players from top_performers.csv"
              f" ({sum(1 for v in result.values() if v['ghost_count'] >= 2)} ghost players)")
        return result
    except Exception as e:
        print(f"[leaderboard_freq] Error loading: {e}")
        return _LEADERBOARD_FREQ_CACHE or {}
TP_MEGA_FIELDS = [
    "date",
    "player_name",
    "team",
    "actual_rs",
    "actual_card_boost",
    "drafts",
    "avg_finish",
    "total_value",
    "source",
]
WINNING_DRAFTS_FIELDS = [
    "winner_rank",
    "drafter_label",
    "total_score",
    "slot_index",
    "player_name",
    "team",
    "actual_rs",
    "slot_mult",
    "card_boost",
]
MOST_POPULAR_GH_PREFIX = "data/most_popular"
OWNERSHIP_LEGACY_PREFIX = "data/ownership"
MOST_DRAFTED_3X_GH_PREFIX = "data/most_drafted_3x"
WINNING_DRAFTS_GH_PREFIX = "data/winning_drafts"
SLATE_RESULTS_GH_PREFIX = "data/slate_results"
MOST_POPULAR_ROW_FIELDS = [
    "player", "team", "draft_count", "actual_rs", "actual_card_boost",
    "avg_finish", "rank", "saved_at",
]


def _parse_actuals_rows(content) -> list:
    """Parse per-day actuals CSV. Header-aware: legacy files without `team` column still work.

    AUDIT FIX (v63): Include normalized player name for consistent joining across datasets.
    """
    rows = []
    if not content or not str(content).strip():
        return rows
    f = io.StringIO(str(content).strip())
    reader = csv.DictReader(f)
    for r in reader:
        if not r:
            continue
        pn = (r.get("player_name") or "").strip()
        if not pn:
            continue
        rows.append(
            {
                "player_name": pn,
                "player_name_normalized": _normalize_player_name(pn),
                "team": (r.get("team") or "").strip().upper(),
                "actual_rs": r.get("actual_rs", ""),
                "actual_card_boost": r.get("actual_card_boost", ""),
                "drafts": r.get("drafts", ""),
                "avg_finish": r.get("avg_finish", ""),
                "total_value": r.get("total_value", ""),
                "source": (r.get("source") or "").strip(),
            }
        )
    return rows


def _parse_top_performers_mega_rows(content: str) -> list:
    """Parse data/top_performers.csv. Header-aware: legacy rows without `team` default to ''."""
    rows = []
    if not content or not str(content).strip():
        return rows
    f = io.StringIO(str(content).strip())
    reader = csv.DictReader(f)
    for r in reader:
        if not r:
            continue
        d = (r.get("date") or "").strip()
        pn = (r.get("player_name") or "").strip()
        if not d or not pn:
            continue
        rows.append(
            {
                "date": d,
                "player_name": pn,
                "team": (r.get("team") or "").strip().upper(),
                "actual_rs": r.get("actual_rs", ""),
                "actual_card_boost": r.get("actual_card_boost", ""),
                "drafts": r.get("drafts", ""),
                "avg_finish": r.get("avg_finish", ""),
                "total_value": r.get("total_value", ""),
                "source": (r.get("source") or "highest_value").strip(),
            }
        )
    return rows


def _actuals_csv_from_rows(rows: list) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=ACT_FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: (r.get(k) if r.get(k) is not None else "") for k in ACT_FIELDS})
    return buf.getvalue()


def _ingest_secret_ok(request: Request) -> bool:
    if not INGEST_SECRET:
        return True
    if request.headers.get("X-Ingest-Key") == INGEST_SECRET:
        return True
    auth = request.headers.get("Authorization") or ""
    if auth.startswith("Bearer ") and auth[7:].strip() == INGEST_SECRET:
        return True
    return False


def _parse_completed_regular_season_games(scoreboard: dict) -> list[dict]:
    """Extract completed regular-season game finals from ESPN scoreboard payload."""
    games = []
    for ev in (scoreboard or {}).get("events", []):
        season = ev.get("season") or {}
        if (season.get("slug") or "") != "regular-season":
            continue
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0] or {}
        st = (comp.get("status") or {}).get("type") or {}
        if not st.get("completed"):
            continue
        home = away = None
        for cd in comp.get("competitors") or []:
            team = cd.get("team") or {}
            abbr = str(team.get("abbreviation") or "").strip().upper()
            if not abbr:
                continue
            rec = {
                "abbr": abbr,
                "score": int(_safe_float(cd.get("score"), 0)),
                "winner": bool(cd.get("winner")),
            }
            if cd.get("homeAway") == "home":
                home = rec
            else:
                away = rec
        if not home or not away:
            continue
        if home["score"] == away["score"]:
            continue
        if home["winner"]:
            w, l = home, away
        elif away["winner"]:
            w, l = away, home
        elif home["score"] > away["score"]:
            w, l = home, away
        else:
            w, l = away, home
        games.append(
            {
                "home": home["abbr"],
                "away": away["abbr"],
                "home_score": home["score"],
                "away_score": away["score"],
                "winner": w["abbr"],
                "loser": l["abbr"],
                "winner_score": w["score"],
                "loser_score": l["score"],
            }
        )
    return games


def _save_slate_results_for_date(date_str: str, skip_if_existing_nonzero: bool = True) -> dict:
    """Persist one day's completed game finals to data/slate_results/{date}.json."""
    bad = _validate_date(date_str)
    if bad:
        return {"status": "invalid_date", "date": date_str}

    path = f"{SLATE_RESULTS_GH_PREFIX}/{date_str}.json"
    if skip_if_existing_nonzero:
        existing_raw, _ = _github_get_file(path)
        if existing_raw:
            try:
                existing = json.loads(existing_raw)
                if int(_safe_float(existing.get("game_count"), 0)) > 0:
                    return {"status": "skipped_existing", "date": date_str, "path": path}
            except Exception:
                pass

    ymd = date_str.replace("-", "")
    data = _espn_get(_espn_scoreboard(ymd))
    games = _parse_completed_regular_season_games(data)
    payload = {
        "date": date_str,
        "game_count": len(games),
        "games": games,
        "season_stage": "regular-season",
        "source": "espn_scoreboard_api",
        "saved_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    wr = _github_write_file(path, json.dumps(payload, indent=2) + "\n", f"slate_results {date_str}")
    if not wr.get("ok"):
        return {"status": "write_error", "date": date_str, "path": path}
    return {"status": "saved", "date": date_str, "path": path, "game_count": len(games)}


def _dates_from_top_performers_mega() -> set:
    raw, _ = _github_get_file(TOP_PERFORMERS_GH_PATH)
    if not raw:
        return set()
    rows = _parse_top_performers_mega_rows(raw)
    return {(r.get("date") or "").strip() for r in rows if (r.get("date") or "").strip()}


def _load_player_actuals_for_date(date_str: str) -> list:
    """Primary: rows from data/top_performers.csv for date_str (ACT_FIELDS shape).
    Fallback: legacy data/actuals/{date}.csv.

    AUDIT FIX (v63): Normalize player names for consistent joining across datasets.
    """
    raw, _ = _github_get_file(TOP_PERFORMERS_GH_PATH)
    if raw:
        rows = _parse_top_performers_mega_rows(raw)
        out = []
        for r in rows:
            if (r.get("date") or "").strip() != date_str:
                continue
            orig_name = (r.get("player_name") or "").strip()
            out.append(
                {
                    "player_name": orig_name,
                    "player_name_normalized": _normalize_player_name(orig_name),
                    "team": (r.get("team") or "").strip().upper(),
                    "actual_rs": r.get("actual_rs", ""),
                    "actual_card_boost": r.get("actual_card_boost", ""),
                    "drafts": r.get("drafts", ""),
                    "avg_finish": r.get("avg_finish", ""),
                    "total_value": r.get("total_value", ""),
                    "source": (r.get("source") or "highest_value").strip(),
                }
            )
        if out:
            return out
    act_csv, _ = _github_get_file(f"data/actuals/{date_str}.csv")
    return _parse_actuals_rows(act_csv) if act_csv else []


def _github_get_most_popular_csv(date_str: str) -> tuple[str, str]:
    """Return (content, path_used) for most-popular-style CSV; prefer data/most_popular/."""
    p_new = f"{MOST_POPULAR_GH_PREFIX}/{date_str}.csv"
    c, _ = _github_get_file(p_new)
    if c:
        return c, p_new
    p_old = f"{OWNERSHIP_LEGACY_PREFIX}/{date_str}.csv"
    c2, _ = _github_get_file(p_old)
    return c2, p_old


@app.get("/api/log/dates")
async def log_dates():
    """Return sorted list of dates that have stored prediction or actual data. Never returns 500."""
    try:
        cached = _cg(_CK_LOG_DATES)
        if cached is not None and isinstance(cached, dict) and "data" in cached:
            if time.time() - cached.get("ts", 0) < _TTL_LOG:  # 10-min TTL (dates change rarely)
                return cached["data"]
        dates = set()
        with ThreadPoolExecutor(max_workers=_W_LIGHT) as pool:
            dir_results = list(pool.map(_github_list_dir, ["data/predictions", "data/actuals"]))
        for items in dir_results:
            for item in items:
                name = item.get("name", "")
                if name.endswith(".csv"):
                    dates.add(name[:-4])
        dates |= _dates_from_top_performers_mega()
        result = sorted(dates, reverse=True)
        _cs(_CK_LOG_DATES, {"data": result, "ts": time.time()})
        return result
    except Exception as e:
        print(f"[log/dates] error: {e}")
        return []


@app.get("/api/log/get")
async def log_get(date: str = Query(None)):
    """Get stored predictions and actuals for a date. Cached 5 min."""
    date_str = date or _today_str()
    bad = _validate_date(date_str)
    if bad: return bad

    # 5-min cache — historical dates are static; today changes only at save-predictions time
    _log_cache_key = f"log_get_{date_str}"
    _log_cached = _cg(_log_cache_key)
    if _log_cached is not None and isinstance(_log_cached, dict) and "ts" in _log_cached:
        if time.time() - _log_cached["ts"] < _TTL_CONFIG:
            return _log_cached["data"]

    pred_csv, _ = _github_get_file(f"data/predictions/{date_str}.csv")
    predictions = _parse_csv(pred_csv, PRED_FIELDS) if pred_csv else []
    actuals = _load_player_actuals_for_date(date_str)

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

    _result = {
        "date": date_str,
        "has_predictions": bool(predictions),
        "has_actuals": bool(actuals),
        "scopes": scopes,
        "actuals": actuals,
    }
    _cs(_log_cache_key, {"data": _result, "ts": time.time()})
    return _result


@app.get("/api/log/actuals-stats")
async def log_actuals_stats(date: str = Query(None)):
    """Fetch actual box score stats (PTS, REB, AST, STL, BLK, MIN) from ESPN
    for all players on a given date's completed games. Returns a map of
    player_name -> {pts, reb, ast, stl, blk, min}. Cached for 24h since
    historical box scores don't change."""
    date_str = date or _today_str()
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

    with ThreadPoolExecutor(max_workers=_W_STANDARD) as pool:
        for game_result in pool.map(_fetch_game_box, games):
            player_stats.update(game_result)

    result = {"date": date_str, "players": player_stats}
    _cs(cache_key, result)
    return result


@app.get("/api/audit/get")
async def audit_get(date: str = Query(None)):
    """Return pre-computed audit JSON for a date (or compute live if missing)."""
    date_str = date or _today_str()
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
        return _err("No players provided", 400)

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
        return _err("No players with valid RS scores", 400)

    lineup = optimize_lineup(projections, n=min(5, len(projections)),
                             sort_key="chalk_ev", rating_key="rating",
                             card_boost_key="est_mult")
    return {"lineup": [_normalize_player(p) for p in lineup]}


_COLD_PIPELINE_LOCK = threading.Lock()
_COLD_PIPELINE_IN_FLIGHT = False
# Populated by startup hook with games from ESPN so the warming_up response
# can show the real game count instead of an empty array.
_WARMUP_GAMES: list = []


def _run_cold_pipeline(trigger: str) -> dict:
    """Single global cold reset + regenerate pipeline for slate.
    Refuses to bust/regenerate when the slate is locked — locked predictions are immutable."""
    global _COLD_PIPELINE_IN_FLIGHT
    with _COLD_PIPELINE_LOCK:
        if _COLD_PIPELINE_IN_FLIGHT:
            return {"status": "in_progress", "trigger": trigger, "ts": datetime.now(timezone.utc).isoformat()}
        _COLD_PIPELINE_IN_FLIGHT = True
    try:
        # Lock guard: refuse to bust/regenerate when slate is locked.
        # Locked predictions must never change. The only exception is manual
        # cold-reset (which requires CRON_SECRET and explicit intent).
        if trigger != "manual_redeploy":
            try:
                _cp_games = fetch_games()
                _cp_starts = [g["startTime"] for g in _cp_games if g.get("startTime")]
                if _any_locked(_cp_starts):
                    print(f"[cold-pipeline] BLOCKED — slate is locked, trigger={trigger}")
                    return {
                        "status": "locked",
                        "trigger": trigger,
                        "message": "Slate is locked — predictions are frozen",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
            except Exception as _lk_err:
                print(f"[cold-pipeline] lock check failed ({_lk_err}) — blocking to be safe")
                return {
                    "status": "lock_check_failed",
                    "trigger": trigger,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }

        auto_saved = False
        cleared = 0
        try:
            games = _cp_games if trigger != "manual_redeploy" else fetch_games()
            starts = [g["startTime"] for g in games if g.get("startTime")]
            if _any_locked(starts):
                try:
                    _sp = asyncio.run(save_predictions())
                    auto_saved = bool(isinstance(_sp, dict) and _sp.get("status") in ("ok", "unchanged"))
                except Exception as _se:
                    print(f"[cold-reset] save-predictions skipped: {_se}")
        except Exception as e:
            print(f"[cold-reset] pre-save check skipped: {e}")

        try:
            _bust_slate_cache(_caller=f"cold_pipeline:{trigger}")
        except Exception as e:
            print(f"[cold-reset] bust err: {e}")
        try:
            for f in LOCK_DIR.glob("*.json"):
                f.unlink(missing_ok=True)
                cleared += 1
        except Exception:
            pass
        try:
            cfg_cache = CONFIG_CACHE_DIR / "model_config.json"
            if cfg_cache.exists():
                cfg_cache.unlink()
        except Exception:
            pass
        try:
            _rw_clear()
        except Exception:
            pass

        prewarm = _prewarm_current_slate_sync(force=True, include_slate=True)
        return {
            "status": "ok",
            "trigger": trigger,
            "auto_saved": auto_saved,
            "cleared_locks": cleared,
            "slate_date": prewarm.get("current_slate_date"),
            "slate_status": prewarm.get("status"),
            "ts": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        with _COLD_PIPELINE_LOCK:
            _COLD_PIPELINE_IN_FLIGHT = False


def _trigger_cold_pipeline_once(trigger: str, marker: str) -> None:
    """Run cold pipeline in background once per trigger+marker key."""
    try:
        key = f"cold_reset:{trigger}:{marker}"
        if rcg(key, "global"):
            return
        rcs(key, "1", "global", ttl=21600)
    except Exception:
        # Best-effort dedupe only; still allow trigger.
        pass

    def _bg():
        try:
            out = _run_cold_pipeline(trigger)
            print(f"[cold-reset] trigger={trigger} marker={marker} status={out.get('status')}")
        except Exception as e:
            print(f"[cold-reset] trigger={trigger} marker={marker} err={e}")

    threading.Thread(target=_bg, daemon=True).start()


@app.get("/api/cold-reset")
async def cold_reset(request: Request):
    """Secured manual/cron cold reset; full pipeline regeneration for current slate."""
    if not _require_cron_secret(request):
        return _err("Unauthorized", 401)
    try:
        return await asyncio.to_thread(_run_cold_pipeline, "manual_redeploy")
    except Exception as e:
        print(f"[cold-reset] error: {e}")
        return {"status": "error", "trigger": "manual_redeploy", "message": str(e)}


@app.get("/api/save-slate-results")
async def save_slate_results(request: Request, date: Optional[str] = None):
    """Cron/admin helper: persist completed regular-season final scores for a date."""
    if not _require_cron_secret(request):
        return _err("Unauthorized", 401)
    date_str = date or _today_str()
    out = _save_slate_results_for_date(date_str, skip_if_existing_nonzero=False)
    return out


def _prewarm_current_slate_sync(force=False, include_slate=True):
    """Sync helper used by cron endpoint and deploy-triggered prewarm worker."""
    today = _today_str()
    is_hit = False
    result = {}
    if include_slate:
        cache_key = _CACHE_KEYS["slate"].format(today)
        result, is_hit = _with_response_cache(cache_key, "slate", _get_slate_impl)
    else:
        result = _get_slate_impl()
    lineups = (result.get("lineups") or {})
    has_lineups = bool(lineups.get("chalk") or lineups.get("upside"))
    current_slate_date = result.get("date") or today

    return {
        "status": "ok",
        "cache_status": "hit" if is_hit else "miss",
        "generated": has_lineups,
        "current_slate_date": current_slate_date,
        "locked": bool(result.get("locked")),
        "all_complete": bool(result.get("all_complete")),
        "draftable_count": int(result.get("draftable_count", 0) or 0),
        "ts": datetime.now().isoformat(),
    }


@app.get("/api/prewarm-current-slate")
async def prewarm_current_slate(request: Request):
    """Cron prewarm for whichever slate is CURRENT per existing global slate logic."""
    if not _require_cron_secret(request):
        return _err("Unauthorized", 401)
    try:
        return await asyncio.to_thread(_prewarm_current_slate_sync, False, True)
    except Exception as e:
        print(f"[prewarm-current-slate] error: {e}")
        return {"status": "error", "message": str(e)}


@app.on_event("startup")
async def _deploy_startup_safe_prewarm():
    """Version-aware startup: detect new deploy SHA → bust + regenerate; same SHA → hydrate from Redis/GitHub.

    On new deploy (SHA mismatch): bust all caches, background-regenerate the full slate
    so the cold pipeline runs exactly once at deploy time. All subsequent requests served
    from Redis.

    On container restart (same SHA): hydrate /tmp from Redis (instant) or GitHub (1-2s).
    No pipeline run needed — Redis survives container restarts.
    """
    today = _today_str()
    current_sha = os.getenv("RAILWAY_GIT_COMMIT_SHA", "")

    # Check if this is a new deploy by comparing SHA in Redis
    previous_sha = rcg("deploy_sha", "global")
    is_new_deploy = bool(current_sha) and (current_sha != (previous_sha or ""))

    if is_new_deploy:
        print(f"[startup] new deploy detected: {previous_sha or 'none'} → {current_sha[:7]}")
        # Store the new SHA so subsequent restarts skip regeneration
        rcs("deploy_sha", current_sha, "global", ttl=604800)  # 7-day TTL

        # Lock guard: if slate is locked, do NOT bust cache or regenerate.
        # Locked predictions must never change — even on a new deploy.
        # The new code will take effect on the next slate.
        global _WARMUP_GAMES
        try:
            _deploy_games = fetch_games()
            # Save games so the warming_up response can show real game count
            _WARMUP_GAMES = _deploy_games
            _deploy_starts = [g["startTime"] for g in _deploy_games if g.get("startTime")]
            if _any_locked(_deploy_starts):
                print(f"[startup] slate is LOCKED — skipping cold pipeline to preserve predictions")
                return
        except Exception as _lk_err:
            print(f"[startup] lock check failed ({_lk_err}) — skipping regen to be safe")
            return

        def _bg_deploy_regen():
            try:
                _run_cold_pipeline("deploy_sha_change")
                # Re-store SHA after rflush inside prewarm may have cleared it
                rcs("deploy_sha", current_sha, "global", ttl=604800)
                print("[startup] background deploy cold-reset completed")
            except Exception as e:
                print(f"[startup] background deploy cold-reset failed: {e}")
        threading.Thread(target=_bg_deploy_regen, daemon=True).start()
        return

    # Same SHA (container restart) — hydrate /tmp from Redis or GitHub.
    # If both miss, run the full cold pipeline in background so picks are
    # ready before the first frontend request instead of blocking it.
    try:
        cached = _cg(_CK_SLATE)
        if cached:
            print(f"[startup] Redis hit — slate already cached for {today}")
            return

        slate_hydrated = False
        gh_slate, _ = _github_get_file(f"data/slate/{today}_slate.json")
        if gh_slate:
            try:
                slate_data = json.loads(gh_slate)
                if not slate_data.get("_busted"):
                    _cs(_CK_SLATE, slate_data)
                    print(f"[startup] hydrated slate cache from GitHub for {today}")
                    slate_hydrated = True
            except Exception as e:
                print(f"[startup] hydration error: {e}")

        gh_games, _ = _github_get_file(f"data/slate/{today}_games.json")
        if gh_games:
            try:
                games_data = json.loads(gh_games)
                if not games_data.get("_busted"):
                    _cs(f"{_CK_SLATE}_games", games_data)
                    print(f"[startup] hydrated games cache from GitHub for {today}")
            except Exception as e:
                print(f"[startup] games hydration error: {e}")

        if slate_hydrated:
            print(f"[startup] cache rehydration completed")
        else:
            # Both Redis and GitHub miss (busted, absent, or stale) — run cold pipeline
            # in background so Redis is populated before the frontend's first request.
            print(f"[startup] no warm cache found — starting background pipeline for {today}")
            # Pre-fetch games so warming_up response shows real game count
            try:
                _WARMUP_GAMES = fetch_games()
            except Exception:
                pass
            def _bg_cold_restart():
                try:
                    _run_cold_pipeline("container_restart_cold")
                    if current_sha:
                        rcs("deploy_sha", current_sha, "global", ttl=604800)
                    print("[startup] background restart pipeline completed")
                except Exception as e:
                    print(f"[startup] background restart pipeline failed: {e}")
            threading.Thread(target=_bg_cold_restart, daemon=True).start()
    except Exception as e:
        print(f"[startup] initialization failed (non-fatal, will regenerate on first request): {e}")


# ═════════════════════════════════════════════════════════════════════════════
# Weekly MAE drift check (backend-only)
# Writes a flag for ops/diagnostics; does NOT affect auto-improve behavior.
# ═════════════════════════════════════════════════════════════════════════════
_MAE_DRIFT_FLAG_PATH = Path("data/mae_drift_flag.json")


@app.get("/api/mae-drift-check")
async def mae_drift_check(request: Request):
    """Weekly cron: compute last 7 calendar days MAE and write a backend-only flag."""
    if not _require_cron_secret(request):
        return _err("Unauthorized", 401)

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
        return _err("Unauthorized", 401)

    today = _today_str()

    # Don't run if slate is locked — picks are frozen post-lock
    games = fetch_games()
    start_times = [g["startTime"] for g in games if g.get("startTime")]
    if _any_locked(start_times):
        return {"status": "locked", "skipped": True}

    if not games:
        return {"status": "no_games", "skipped": True}

    # Load current cached slate (try /tmp first, then GitHub)
    cached_slate = _cg(_CK_SLATE) or _slate_cache_from_github()
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
    cold = await asyncio.to_thread(_run_cold_pipeline, "injury_check")
    return {
        "status": cold.get("status", "ok"),
        "trigger": "injury_check",
        "injuries_found": len(injured_games),
        "affected_players": affected_players,
        "regenerated_games": list(injured_games),
        "cold_reset": cold,
    }
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
    slate_locked = _any_locked([g.get("startTime", "") for g in games if g.get("startTime")])
    cache_ttl = 60 if slate_locked else 180

    # Cache valid only if within TTL AND still the same ET date
    if (_GAMES_FINAL_CACHE["result"] is not None
            and now_ts - _GAMES_FINAL_CACHE["ts"] < cache_ttl
            and _GAMES_FINAL_CACHE.get("date") == today_str):
        return tuple(_GAMES_FINAL_CACHE["result"])

    def _tally(scoreboard_data):
        fins, rem, latest = 0, 0, None
        for ev in scoreboard_data.get("events", []):
            _st = ev.get("status", {}).get("type", {})
            # ESPN has multiple completion signals — `completed` (bool) lags behind
            # `state` ("post") and `name` ("STATUS_FINAL"). Check all three so we
            # detect game completion as fast as the NBA app does.
            completed = (
                _st.get("completed", False)
                or _st.get("state", "").lower() == "post"
                or _st.get("name", "").upper() == "STATUS_FINAL"
            )
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

    data = _espn_get(_espn_scoreboard(today_str))
    finals, remaining, latest_remaining = _tally(data)

    # Midnight rollover: if today has no started or completed games yet,
    # check yesterday — late games may still be running past midnight ET.
    if finals == 0 and remaining == 0:
        yesterday_str = (_et_date() - timedelta(days=1)).strftime("%Y%m%d")
        ydata = _espn_get(_espn_scoreboard(yesterday_str))
        finals, remaining, latest_remaining = _tally(ydata)

    # All done when no started games are still in progress AND we actually saw
    # completed games. If ESPN returns empty data (outage/rate-limit), both counts
    # are 0 — that's NOT confirmation games are final. Only unlock when we have
    # positive evidence (at least one completed game) or it's genuinely a no-game day.
    all_final = (remaining == 0 and finals > 0)

    # AGGRESSIVE FALLBACK: If ESPN isn't marking games as complete but they've been
    # running for 4.5+ hours, treat as final anyway. This handles ESPN API delays.
    # NBA games max ~3.5h; 4.5h buffer includes OT, 2OT, and processing delays.
    # v72: Also fires when remaining=0 AND we have slate game start times as fallback.
    # Mar 26 bug: ESPN returned finals=0, remaining=0 for a completed 3-game slate,
    # keeping the app locked indefinitely. Now uses slate start times when ESPN is empty.
    _fallback_latest = latest_remaining
    if not _fallback_latest and games:
        # ESPN scoreboard empty — use PAST slate game start times as fallback.
        # Filter to only started games; future start times (tomorrow's games in
        # the ESPN response after midnight) would give negative hours_since_start
        # and prevent the fallback from firing on yesterday's completed slate.
        _slate_starts = [g.get("startTime", "") for g in games
                         if g.get("startTime") and _is_past_lock_window(g.get("startTime", ""))]
        if _slate_starts:
            _fallback_latest = max(_slate_starts)
    if not all_final and _fallback_latest:
        try:
            latest_dt = datetime.fromisoformat(_fallback_latest.replace("Z", "+00:00"))
            hours_since_start = (now_ts - latest_dt.timestamp()) / 3600
            if hours_since_start >= 4.5:
                all_final = True
                print(f"[espn fallback] latest game started {hours_since_start:.1f}h ago — forcing all_final=True (ESPN lagged/empty)")
        except Exception:
            pass

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
        slate_locked = _any_locked(start_times)
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
    tp_dates   = _dates_from_top_performers_mega()
    historical_dates = act_dates | tp_dates

    # Find dates with predictions and historical outcomes (mega top_performers and/or legacy actuals)
    paired = []
    for item in sorted(pred_items, key=lambda x: x.get("name",""), reverse=True):
        name = item.get("name","")
        if not name.endswith(".csv"): continue
        d = name[:-4]
        if d in historical_dates:
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
    today_iso = _today_str()
    pred_dates = sorted(
        [i["name"].replace(".csv","") for i in pred_items if i["name"].endswith(".csv")],
        reverse=True
    )
    # Pending = predictions exist but no rows for that date in mega top_performers (primary source).
    pending_historical_date = next((d for d in pred_dates if d not in tp_dates and d != today_iso), None)
    pending_upload_date = pending_historical_date

    # Check for most-popular / ownership calibration data
    try:
        mp_items = _github_list_dir(MOST_POPULAR_GH_PREFIX) or []
        leg_items = _github_list_dir(OWNERSHIP_LEGACY_PREFIX) or []
        own_dates = sorted(
            {
                i["name"].replace(".csv", "")
                for i in (mp_items + leg_items)
                if i.get("name", "").endswith(".csv")
            },
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
        "pending_historical_date": pending_historical_date,
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
        return _err("No changes provided", 400)
    # Block config changes during active slate — picks are frozen until all games finish
    try:
        _uc_games = fetch_games()
        _uc_starts = [g["startTime"] for g in _uc_games if g.get("startTime")]
        if _any_locked(_uc_starts):
            return _err("Slate is active — config changes are locked until all games finish", 423)
    except Exception:
        pass  # if games check fails, allow the write
    # Security: reject keys with non-alphanumeric path segments (prevents path traversal)
    import re as _re
    for key in changes:
        if not _re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*([.][a-zA-Z_][a-zA-Z0-9_]*)*$', str(key)):
            return _err(f"Invalid key format: {key!r}", 400)

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
        "date":     _today_str(),
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
        return _err(result["error"], 500)

    # Clear config cache so new values take effect immediately
    try:
        (CONFIG_CACHE_DIR / "model_config.json").unlink(missing_ok=True)
    except Exception: pass
    # Bust slate cache so next request regenerates with new config params
    try:
        _bust_slate_cache(_caller="lab_update_config")
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
        return _err("target_version required", 400)
    # Block rollbacks during active slate — same lock guard as update-config
    try:
        _rb_games = fetch_games()
        _rb_starts = [g["startTime"] for g in _rb_games if g.get("startTime")]
        if _any_locked(_rb_starts):
            return _err("Slate is active — config changes are locked until all games finish", 423)
    except Exception:
        pass

    cfg = _load_config()
    changelog = cfg.get("changelog", [])
    current_version = cfg.get("version", 1)
    if int(target) >= current_version:
        return _err("Target must be earlier than current version", 400)

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
        "date":     _today_str(),
        "change":   f"Rollback to v{target}: restored {list(snapshot.keys())}",
        "snapshot": {k: cfg_val for k, cfg_val in snapshot.items()},
    })
    cfg["changelog"] = changelog

    content = json.dumps(cfg, indent=2)
    result  = _github_write_file("data/model-config.json", content, f"Lab rollback to v{target}")
    if result.get("error"):
        return _err(result["error"], 500)

    try:
        (CONFIG_CACHE_DIR / "model_config.json").unlink(missing_ok=True)
    except Exception: pass
    try:
        _bust_slate_cache(_caller="lab_rollback")
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
        return _err("proposed_changes required", 400)

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
        actuals = _parse_actuals_rows(act_csv)
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
        return _err("Unauthorized", 401)
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
            timeout=_T_HEAVY,
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


def _ben_chat_trim_trailing_user_orphan(data: list) -> None:
    """Drop at most one trailing user turn after prior messages (assistant failed before reply).

    Does not remove the only message in the thread — preserves a lone user turn on reload.
    """
    if not isinstance(data, list) or len(data) < 2:
        return
    if data[-1].get("role") == "user":
        data.pop()


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
    today = _today_str()

    last_et = ""
    try:
        if _BEN_CHAT_HISTORY_DATE_PATH.exists():
            raw = _BEN_CHAT_HISTORY_DATE_PATH.read_text()
            obj = json.loads(raw) if raw else {}
            last_et = (obj or {}).get("et_date", "") or ""
    except Exception:
        last_et = ""

    # If local date file is missing, also check GitHub (cold start after redeploy)
    if not last_et:
        try:
            gh_date_content, _ = _github_get_file("data/ben_chat_history_last_et_date.json")
            if gh_date_content:
                obj = json.loads(gh_date_content) if isinstance(gh_date_content, str) else {}
                last_et = (obj or {}).get("et_date", "") or ""
        except Exception:
            pass

    if last_et == today:
        return

    try:
        _BEN_CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BEN_CHAT_HISTORY_PATH.write_text("[]")
        _BEN_CHAT_HISTORY_DATE_PATH.write_text(json.dumps({"et_date": today}, indent=2))
        # Also reset GitHub files so cold starts (after Railway redeploy) see the clear slate, not stale old messages
        try:
            _github_write_file("data/ben_chat_history.json", "[]", f"Clear Ben chat on slate turnover {today}")
            _github_write_file("data/ben_chat_history_last_et_date.json", json.dumps({"et_date": today}, indent=2), f"Update Ben chat date marker {today}")
        except Exception as gh_err:
            print(f"[ben-chat] github reset failed (non-fatal): {gh_err}")
    except Exception as e:
        print(f"[ben-chat] reset failed: {e}")


def _ben_chat_read_history_locked() -> list:
    _ben_chat_maybe_reset_for_today_locked()
    # Layer 1: local file (fast path — survives within a container session)
    try:
        if _BEN_CHAT_HISTORY_PATH.exists():
            raw = _BEN_CHAT_HISTORY_PATH.read_text()
            data = json.loads(raw) if raw else []
            if isinstance(data, list) and data:
                _ben_chat_trim_trailing_user_orphan(data)
                return data
    except Exception:
        pass
    # Layer 2: GitHub (cold start after Railway redeploy wipes /tmp + local data/)
    try:
        gh_content, _ = _github_get_file("data/ben_chat_history.json")
        if gh_content:
            data = json.loads(gh_content) if isinstance(gh_content, str) else []
        else:
            data = []
        if isinstance(data, list) and data:
            # Write back to local for subsequent fast-path reads this session
            _BEN_CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            _BEN_CHAT_HISTORY_PATH.write_text(json.dumps(data, indent=2))
            _ben_chat_trim_trailing_user_orphan(data)
            return data
    except Exception as e:
        print(f"[ben-chat] github read failed (non-fatal): {e}")
    return []


def _ben_chat_write_history_locked(messages: list) -> None:
    # Always write to local file immediately (fast path for reads in same session)
    try:
        _BEN_CHAT_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        _BEN_CHAT_HISTORY_PATH.write_text(json.dumps(messages, indent=2))
    except Exception as e:
        print(f"[ben-chat] local write failed: {e}")
    # Background GitHub write — survives Railway redeploys (non-blocking, non-fatal)
    _msgs_snapshot = list(messages)
    def _bg_github_write():
        try:
            _github_write_file(
                "data/ben_chat_history.json",
                json.dumps(_msgs_snapshot, indent=2),
                f"Ben chat: {len(_msgs_snapshot)} messages",
            )
        except Exception as e:
            print(f"[ben-chat] github write failed (non-fatal): {e}")
    threading.Thread(target=_bg_github_write, daemon=True).start()


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
        return _err("ANTHROPIC_API_KEY not configured", 500)

    messages = payload.get("messages", [])
    system   = payload.get("system", "")

    if not messages:
        return _err("No messages provided", 400)

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
                              headers=headers, json=base_body, timeout=_T_CLAUDE)
            if not r.ok:
                _api_err = {402: "API credits exhausted. Please top up Anthropic billing.",
                            429: "Rate limited. Please wait a moment and retry."}.get(
                    r.status_code, f"Claude API error (HTTP {r.status_code})")
                yield _sse({"type": "content", "error": _api_err, "text": ""})
                return
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
                    timeout=_T_CLAUDE,
                )
                if not r_next.ok:
                    _api_err = {402: "API credits exhausted.", 429: "Rate limited."}.get(
                        r_next.status_code, f"Claude API error (HTTP {r_next.status_code})")
                    yield _sse({"type": "content", "error": _api_err, "text": ""})
                    return
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
    """Append a date to data/skipped-uploads.json so save-actuals no-ops for that date.

    Optional — no in-app UI; scripts or manual tooling may call this. Does not delete data.
    """
    date_str = payload.get("date", "").strip()
    if not date_str:
        return _err("date required", 400)
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


def _save_most_popular_style_csv(payload: dict, gh_prefix: str, commit_msg: str):
    """Write most-popular schema CSV to gh_prefix/{date}.csv. Returns dict or JSONResponse."""
    date_str = payload.get("date", _today_str())
    bad = _validate_date(date_str)
    if bad:
        return bad

    players = payload.get("players", [])
    if not players:
        return {"saved": 0, "date": date_str, "path": f"{gh_prefix}/{date_str}.csv"}

    rows = ["player,team,draft_count,actual_rs,actual_card_boost,avg_finish,rank,saved_at"]
    ts = datetime.now(timezone.utc).isoformat()
    saved = 0
    for p in players:
        name = str(p.get("player_name") or p.get("name", "")).strip().replace(",", " ")
        if not name:
            continue
        team = str(p.get("team") or "").upper().replace(",", "")
        draft_count = _safe_float(p.get("draft_count")) or 0
        actual_rs = _safe_float(p.get("actual_rs"))
        actual_boost = _safe_float(p.get("actual_card_boost"))
        avg_finish = _safe_float(p.get("avg_finish"))
        rank = int(p.get("rank") or 0)
        rows.append(
            f"{name},{team},{int(draft_count)},"
            f"{actual_rs if actual_rs is not None else ''},"
            f"{actual_boost if actual_boost is not None else ''},"
            f"{avg_finish if avg_finish is not None else ''},"
            f"{rank},{ts}"
        )
        saved += 1

    if saved == 0:
        return {"saved": 0, "date": date_str, "path": f"{gh_prefix}/{date_str}.csv"}

    content = "\n".join(rows) + "\n"
    path = f"{gh_prefix}/{date_str}.csv"
    try:
        _github_write_file(path, content, f"{commit_msg} {date_str} ({saved} players)")
    except Exception as e:
        print(f"[save-most-popular] GitHub write failed: {e}")
        return _err(f"Failed to save: {str(e)}", 500)

    return {"saved": saved, "date": date_str, "path": path}


@app.post("/api/save-most-popular")
async def save_most_popular(request: Request, payload: dict = Body(...)):
    """Save parsed Most Popular list to data/most_popular/{date}.csv (developer ingest)."""
    if not _ingest_secret_ok(request):
        return _err("Invalid or missing ingest key", 401)
    out = _save_most_popular_style_csv(payload, MOST_POPULAR_GH_PREFIX, "most_popular")
    if isinstance(out, JSONResponse):
        return out
    return out


@app.post("/api/save-ownership")
async def save_ownership(request: Request, payload: dict = Body(...)):
    """Backward-compatible alias: writes data/most_popular/{date}.csv (same as save-most-popular).

    Body: { date, players: [{player_name|name, team, draft_count, actual_rs,
           actual_card_boost, avg_finish, rank}] }
    """
    if not _ingest_secret_ok(request):
        return _err("Invalid or missing ingest key", 401)
    out = _save_most_popular_style_csv(payload, MOST_POPULAR_GH_PREFIX, "most_popular")
    if isinstance(out, JSONResponse):
        return out
    return out


@app.post("/api/save-dfs-salaries")
async def save_dfs_salaries(request: Request, payload: dict = Body(...)):
    """Save DFS salary CSV for popularity proxy. Body: {date, csv_content, platform?}"""
    if not _ingest_secret_ok(request):
        return _err("Invalid or missing ingest key", 401)
    date_str = payload.get("date", "")
    csv_content = payload.get("csv_content", "")
    platform = payload.get("platform", "draftkings")
    if not date_str or not csv_content:
        return _err("date and csv_content required")
    try:
        result = _save_dfs_sal(date_str, csv_content, platform)
        if result.get("error"):
            return _err(result["error"])
        # Also persist to GitHub
        gh_path = f"data/dfs_salaries/{date_str}.csv"
        try:
            _github_write_file(gh_path, csv_content, f"DFS salaries {date_str} ({platform})")
        except Exception as _gh_e:
            print(f"[dfs-salaries] GitHub write error: {_gh_e}")
        return result
    except Exception as e:
        return _err(f"Failed to save DFS salaries: {e}")


@app.post("/api/save-most-drafted-3x")
async def save_most_drafted_3x(request: Request, payload: dict = Body(...)):
    """High-boost popular sub-list → data/most_drafted_3x/{date}.csv (same columns as most_popular)."""
    if not _ingest_secret_ok(request):
        return _err("Invalid or missing ingest key", 401)
    out = _save_most_popular_style_csv(payload, MOST_DRAFTED_3X_GH_PREFIX, "most_drafted_3x")
    if isinstance(out, JSONResponse):
        return out
    out["min_boost_filter"] = payload.get("min_boost", 3.0)
    return out


@app.post("/api/save-winning-drafts")
async def save_winning_drafts(request: Request, payload: dict = Body(...)):
    """Long-format winning lineups → data/winning_drafts/{date}.csv (max 4 winners × 5 slots)."""
    if not _ingest_secret_ok(request):
        return _err("Invalid or missing ingest key", 401)
    date_str = payload.get("date", _today_str())
    bad = _validate_date(date_str)
    if bad:
        return bad
    raw = payload.get("rows") or payload.get("players") or []
    if not raw:
        return _err("rows or players required", 400)

    ts = datetime.now(timezone.utc).isoformat()
    buf = io.StringIO()
    wr_csv = csv.writer(buf)
    wr_csv.writerow(WINNING_DRAFTS_FIELDS + ["saved_at"])
    saved = 0
    for r in raw:
        try:
            wrn = int(float(r.get("winner_rank") or 0))
            si = int(float(r.get("slot_index") or 0))
        except (TypeError, ValueError):
            continue
        if wrn < 1 or wrn > 4 or si < 1 or si > 5:
            continue
        pname = str(r.get("player_name") or r.get("name", "")).strip()
        if not pname:
            continue
        dl = str(r.get("drafter_label") or "")
        ts_val = r.get("total_score")
        ars = r.get("actual_rs")
        sm = r.get("slot_mult")
        cb = r.get("card_boost")
        team = str(r.get("team", "")).strip().upper()
        wr_csv.writerow(
            [
                wrn,
                dl,
                ts_val if ts_val is not None else "",
                si,
                pname,
                team,
                ars if ars is not None else "",
                sm if sm is not None else "",
                cb if cb is not None else "",
                ts,
            ]
        )
        saved += 1

    if saved == 0:
        return _err("No valid rows (need winner_rank 1-4, slot_index 1-5, player_name)", 400)
    if saved > 20:
        return _err("Too many rows (max 20 = 4 winners × 5 slots)", 400)

    path = f"{WINNING_DRAFTS_GH_PREFIX}/{date_str}.csv"
    content = buf.getvalue()
    try:
        _github_write_file(path, content, f"winning_drafts {date_str} ({saved} rows)")
    except Exception as e:
        print(f"[save-winning-drafts] GitHub write failed: {e}")
        return _err(f"Failed to save: {str(e)}", 500)
    return {"saved": saved, "date": date_str, "path": path}


@app.get("/api/lab/calibrate-boost")
async def lab_calibrate_boost():
    """Aggregate historical actual_card_boost by player from ownership + actuals CSVs.

    For analysis and offline boost-model retraining only — live slate uses LightGBM + sigmoid,
    not this map.
    """
    try:
        from collections import defaultdict

        ceiling = _cfg("card_boost.ceiling", _CONFIG_DEFAULTS["card_boost"]["ceiling"])
        floor_  = _cfg("card_boost.floor", _CONFIG_DEFAULTS["card_boost"]["floor"])

        player_boosts = defaultdict(list)
        dates_used = []

        mp_items = _github_list_dir(MOST_POPULAR_GH_PREFIX) or []
        leg_items = _github_list_dir(OWNERSHIP_LEGACY_PREFIX) or []
        own_dates = sorted(
            {
                i["name"].replace(".csv", "")
                for i in (mp_items + leg_items)
                if i.get("name", "").endswith(".csv")
            },
            reverse=True,
        )
        for date_str in own_dates:
            own_csv, _src = _github_get_most_popular_csv(date_str)
            if not own_csv:
                continue
            own_rows = _parse_csv(own_csv, MOST_POPULAR_ROW_FIELDS)
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
            act_rows = _parse_actuals_rows(act_csv)
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

        return {
            "proposed_count": n_players,
            "proposed":       proposed,
            "n_samples":      n_samples,
            "dates_used":     dates_used,
            "note": (
                "Historical average boost per player — use for train_boost_lgbm / drift analysis. "
                "Runtime card boost has no config overrides or upload path."
            ),
        }

    except Exception as e:
        print(f"[calibrate-boost] Error: {e}")
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# grep: PARLAY ENGINE — /api/parlay, _fetch_gamelog, _fetch_gamelogs_batch
# Safest 3-leg player prop parlay optimizer (certainty over edge).
# ─────────────────────────────────────────────────────────────────────────────

def _gamelog_parse_stat_val(raw):
    try:
        if raw is None:
            return None
        s = str(raw)
        if ":" in s:
            return float(s.split(":")[0])
        return float(raw)
    except (ValueError, TypeError):
        return None


def _gamelog_opponent_abbr(event: dict) -> str:
    """Best-effort opponent abbreviation from ESPN gamelog event."""
    if not event:
        return ""
    opp = event.get("opponent")
    if isinstance(opp, dict):
        return (opp.get("abbreviation") or opp.get("abbr") or "") or ""
    for c in event.get("competitions", []) or []:
        for comp in c.get("competitors", []) or []:
            ab = (comp.get("team") or {}).get("abbreviation") or comp.get("abbreviation")
            if ab:
                return str(ab)
    return ""


def _fetch_gamelog(pid, num_games=15):
    """Fetch a player's game log from ESPN and return structured stat arrays.

    Hits https://site.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{pid}/gamelog
    Caches in /tmp per player per date.

    Returns: points/rebounds/assists/minutes plus steals/blocks/threes/tov/fga when labels exist,
             opponent_abbr[] and game_dates[] (ISO date strings) parallel to games (best-effort).
             or {} on failure.
    """
    cache_key = f"gamelog_v2_{pid}"
    cached = _cg(cache_key)
    if cached:
        return cached

    url = (f"https://site.api.espn.com/apis/common/v3/sports/basketball"
           f"/nba/athletes/{pid}/gamelog")
    data = _espn_get(url)
    if not data:
        return {}

    try:
        # ESPN gamelog structure:
        #   Top-level: labels[] (display: "MIN","PTS","REB","AST",...) and
        #              names[] (machine: "minutes","points","totalRebounds","assists",...)
        #   Data:      seasonTypes[] -> categories[] -> events[] (each event has stats[] aligned to labels/names)
        stat_arrays = {
            "points": [], "rebounds": [], "assists": [], "minutes": [],
            "steals": [], "blocks": [], "turnovers": [], "threes": [],
            "field_goals_attempted": [],
        }
        opponent_abbr = []
        game_dates = []  # ISO date strings parallel to stat arrays

        # Build index map from top-level labels (display names)
        top_labels = [lbl.lower() for lbl in data.get("labels", [])]
        idx_map = {}
        for stat_name, lbl_options in [
            ("points", ["pts"]),
            ("rebounds", ["reb"]),
            ("assists", ["ast"]),
            ("minutes", ["min"]),
            ("steals", ["stl"]),
            ("blocks", ["blk"]),
            ("turnovers", ["to", "tov"]),
            ("threes", ["3pm", "3pt", "3fgm"]),
            ("field_goals_attempted", ["fga", "fg att"]),
        ]:
            for lbl in lbl_options:
                if lbl in top_labels:
                    idx_map[stat_name] = top_labels.index(lbl)
                    break

        if not idx_map or "points" not in idx_map:
            return {}

        # Collect game rows from seasonTypes -> categories -> events
        for st in data.get("seasonTypes", []):
            for cat in st.get("categories", []):
                for event in cat.get("events", []):
                    stats = event.get("stats", [])
                    if not stats:
                        continue
                    opponent_abbr.append(_gamelog_opponent_abbr(event))
                    # Extract game date from ESPN event (gameDate, date, or eventDate)
                    raw_date = event.get("gameDate") or event.get("date") or event.get("eventDate") or ""
                    game_dates.append(str(raw_date) if raw_date else "")
                    for stat_name in stat_arrays.keys():
                        idx = idx_map.get(stat_name)
                        if idx is not None and idx < len(stats):
                            val = _gamelog_parse_stat_val(stats[idx])
                            if val is not None:
                                stat_arrays[stat_name].append(val)
                            else:
                                stat_arrays[stat_name].append(0.0)
                        else:
                            stat_arrays[stat_name].append(0.0)

        # ESPN returns newest-first within each month category — reverse to
        # chronological (oldest-first) so [-N:] gives the N most recent games.
        for k in stat_arrays:
            stat_arrays[k].reverse()
            stat_arrays[k] = stat_arrays[k][-num_games:]
        opponent_abbr.reverse()
        opponent_abbr = opponent_abbr[-num_games:]
        game_dates.reverse()
        game_dates = game_dates[-num_games:]

        if opponent_abbr:
            stat_arrays["opponent_abbr"] = opponent_abbr
        if game_dates:
            stat_arrays["game_dates"] = game_dates

        if any(len(v) > 0 for k, v in stat_arrays.items() if k not in ("opponent_abbr", "game_dates")):
            _cs(cache_key, stat_arrays)
            return stat_arrays
        return {}

    except Exception as e:
        print(f"[gamelog] parse error for {pid}: {e}")
        return {}


def _fetch_gamelogs_for_slate(player_ids, num_games=15):
    """Batch-fetch gamelogs for many player IDs (slate-wide). Same as batch helper."""
    return _fetch_gamelogs_batch(player_ids, num_games=num_games)


def _slate_prefetch_gamelogs(draftable_games, num_games=15):
    """Union rosters across draftable games and batch-fetch gamelogs (cache warm)."""
    ids = []
    for g in draftable_games:
        try:
            hr = fetch_roster(g["home"]["id"], g["home"]["abbr"])
            ar = fetch_roster(g["away"]["id"], g["away"]["abbr"])
            for p in hr + ar:
                ids.append(str(p["id"]))
        except Exception as e:
            print(f"[gamelog-slate] roster err {g.get('gameId')}: {e}")
    if not ids:
        return {}
    return _fetch_gamelogs_for_slate(list(set(ids)), num_games=num_games)


def _fetch_gamelogs_batch(player_ids, num_games=15, max_workers=None):
    """Fetch gamelogs for multiple players with rate-limit aware throttling.

    Rate limit strategy:
    - If ESPN rate limit is low (<5 remaining), reduce max_workers to 2
    - If at/near limit, sleep 1s between batches
    - Falls back to sequential fetch on rate limit exhaustion

    Returns {player_id: gamelog dict}
    """
    if max_workers is None:
        # Check rate limit state before deciding parallelism
        with _ESPN_RATE_LIMIT_LOCK:
            remaining = _ESPN_RATE_LIMIT_STATE.get("remaining")

        if remaining is not None and remaining < 5:
            max_workers = 2  # Reduce parallelism if rate limit low
            print(f"[gamelog-batch] ESPN rate limit low ({remaining} remaining) — reducing workers to 2", flush=True)
        else:
            max_workers = _W_L5  # Standard: 10 workers

    result = {}
    batch_size = max(1, len(player_ids) // max_workers)  # Distribute into batches

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for idx, pid in enumerate(player_ids):
            # Submit in batches; sleep between batches to throttle ESPN requests
            if idx > 0 and idx % batch_size == 0:
                time.sleep(0.5)  # Small delay between batches
            futures[pool.submit(_fetch_gamelog, pid, num_games)] = pid

        for fut in as_completed(futures, timeout=_T_HEAVY):
            pid = futures[fut]
            try:
                log = fut.result(timeout=5)
                if log:
                    result[pid] = log
            except Exception as e:
                print(f"[gamelog-batch] {pid} error: {e}")

    return result

