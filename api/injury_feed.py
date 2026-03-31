"""Unified injury/availability feed — multi-source with consistent veto logic.

Primary: RotoWire (scraper, 30-min cache)
Secondary: ESPN injury page (corroboration, 30-min cache)

Single decision function: is_available(player_name) → (available: bool, reason: str)
Ensures consistent filtering across _build_lineups (slate-wide) AND
_build_game_lineups (per-game), closing the per-game RotoWire gap.

grep: INJURY FEED
"""

import re
import json
import time
import requests
from pathlib import Path
from typing import Optional


CACHE_DIR = Path("/tmp/nba_cache_v19")
_CACHE_TTL = 1800  # 30 minutes

ESPN_INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
CBS_INJURY_URL = "https://www.cbssports.com/nba/injuries/"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
}


def _normalize_name(name: str) -> str:
    n = str(name).lower().strip()
    n = re.sub(r"['\.\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"injury_{key}.json"


def _read_cache(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if time.time() - data.get("_ts", 0) > _CACHE_TTL:
            return None
        return data
    except Exception:
        return None


def _write_cache(key: str, data: dict):
    try:
        data["_ts"] = time.time()
        _cache_path(key).write_text(json.dumps(data, default=str))
    except Exception:
        pass


def fetch_espn_injuries() -> dict:
    """Fetch NBA-wide injury report from ESPN's public API.

    Returns {normalized_name: {"status": str, "team": str, "detail": str}}
    Status values: "out", "doubtful", "questionable", "day-to-day", "active"
    """
    cached = _read_cache("espn_injuries")
    if cached and "players" in cached:
        return cached["players"]

    try:
        r = requests.get(ESPN_INJURY_URL, timeout=10)
        if not r.ok:
            print(f"[injury-feed] ESPN injury fetch failed: HTTP {r.status_code}")
            return {}
        data = r.json()
    except Exception as e:
        print(f"[injury-feed] ESPN injury fetch error: {e}")
        return {}

    players = {}
    try:
        for team_entry in data.get("items", []):
            team_abbr = team_entry.get("team", {}).get("abbreviation", "")
            for inj in team_entry.get("injuries", []):
                athlete = inj.get("athlete", {})
                name = athlete.get("displayName", "")
                if not name:
                    continue
                status_raw = (inj.get("status", "") or "").lower().strip()
                detail = inj.get("shortComment", "") or inj.get("longComment", "")
                norm = _normalize_name(name)
                players[norm] = {
                    "status": _normalize_status(status_raw),
                    "team": team_abbr,
                    "detail": detail,
                    "source": "espn",
                }
    except Exception as e:
        print(f"[injury-feed] ESPN parse error: {e}")
        return {}

    if players:
        _write_cache("espn_injuries", {"players": players})
        print(f"[injury-feed] ESPN injuries: {len(players)} players")
    return players


def _normalize_status(raw: str) -> str:
    """Map raw status strings to standard tiers."""
    r = raw.lower().strip()
    if r in ("out", "ruled out", "o", "injured", "suspended"):
        return "out"
    if r in ("doubtful", "doubt"):
        return "doubtful"
    if r in ("questionable", "gtd", "game-time decision", "game time decision"):
        return "questionable"
    if r in ("day-to-day", "dtd", "day to day"):
        return "day-to-day"
    if r in ("probable", "active", "available"):
        return "active"
    return raw or "unknown"


def get_combined_availability() -> dict:
    """Build combined availability from RotoWire (primary) + ESPN (secondary).

    Returns {normalized_name: {
        "available": bool,
        "status": str,
        "source": str,  # "rotowire", "espn", "both"
        "confidence": str,  # "high" (both agree), "medium" (one source), "low" (conflict)
        "detail": str,
    }}
    """
    try:
        from api.rotowire import get_all_statuses
        rw = get_all_statuses() or {}
    except Exception:
        rw = {}

    espn = fetch_espn_injuries()
    combined = {}

    # Process RotoWire entries first (primary source)
    for norm, rw_info in rw.items():
        rw_status = rw_info.get("status", "unknown")
        rw_out = rw_status in ("out",)
        rw_questionable = rw_status in ("questionable",)
        espn_info = espn.get(norm, {})
        espn_status = espn_info.get("status", "")
        espn_out = espn_status in ("out", "doubtful")

        if rw_out and espn_out:
            available, source, confidence = False, "both", "high"
        elif rw_out:
            available, source, confidence = False, "rotowire", "medium"
        elif espn_out:
            available, source, confidence = False, "espn", "medium"
        elif rw_questionable:
            available, source, confidence = True, "rotowire", "low"
        else:
            available, source, confidence = True, "rotowire", "high"

        combined[norm] = {
            "available": available,
            "status": rw_status if rw_status != "unknown" else espn_status,
            "source": source,
            "confidence": confidence,
            "detail": espn_info.get("detail", rw_info.get("injury_note", "")),
        }

    # Add ESPN-only entries (not in RotoWire)
    for norm, espn_info in espn.items():
        if norm in combined:
            continue
        espn_status = espn_info.get("status", "")
        espn_out = espn_status in ("out", "doubtful")
        combined[norm] = {
            "available": not espn_out,
            "status": espn_status,
            "source": "espn",
            "confidence": "medium",
            "detail": espn_info.get("detail", ""),
        }

    return combined


def is_available(player_name: str) -> tuple:
    """Unified availability check combining all injury sources.

    Returns (available: bool, reason: str).
    Use this instead of is_safe_to_draft() for consistent veto logic.
    """
    combined = get_combined_availability()
    norm = _normalize_name(player_name)
    info = combined.get(norm)
    if not info:
        return True, ""  # Not in any injury report
    if not info["available"]:
        return False, f"{info['status']} ({info['source']}: {info['detail'][:80]})"
    return True, ""


def get_unavailable_players() -> list:
    """Return list of normalized names of all currently unavailable players."""
    combined = get_combined_availability()
    return [name for name, info in combined.items() if not info["available"]]
