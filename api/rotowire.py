# ─────────────────────────────────────────────────────────────────────────────
# ROTOWIRE LINEUP INTELLIGENCE — Free-Tier Scraper
#
# Scrapes the RotoWire NBA daily lineups page to answer the single most
# important question the model gets wrong: "Will this player actually play?"
#
# Data extracted (all free, no subscription):
#   - Confirmed / Expected / Questionable / OUT status per player
#   - Injury notes and color-coded uncertainty flags
#   - Starting lineup positions
#
# Called ~30 min before first tip-off by the slate projection pipeline.
# Results cached for the day to avoid repeated scrapes.
#
# Integration points:
#   - project_player(): skip or penalize players RotoWire flags as OUT/uncertain
#   - _build_lineups(): moonshot hard-filters on lineup confirmation
#   - Minute projections: override ESPN blend with RotoWire when available
# ─────────────────────────────────────────────────────────────────────────────

import re
import json
import hashlib
import unicodedata
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta
from api.shared import et_date as _shared_et_date

# Cache directory — same pattern as index.py
ROTOWIRE_CACHE_DIR = Path("/tmp/nba_rotowire_v1")
ROTOWIRE_CACHE_DIR.mkdir(exist_ok=True)

LINEUPS_URL = "https://www.rotowire.com/basketball/nba-lineups.php"

# User-agent to avoid bot blocking
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Player status tiers — ordered from most to least certain to play
STATUS_CONFIRMED = "confirmed"    # In confirmed starting lineup
STATUS_EXPECTED = "expected"      # In expected lineup, not yet confirmed
STATUS_QUESTIONABLE = "questionable"  # Color-flagged as uncertain
STATUS_OUT = "out"                # Ruled out / not in lineup
STATUS_UNKNOWN = "unknown"        # Not found on RotoWire page


def _et_date():
    """Current date in Eastern Time — mirrors api/index.py logic."""
    return _shared_et_date()


def _cache_path():
    """Date-keyed cache file for today's RotoWire data."""
    today = _et_date().isoformat()
    h = hashlib.md5(f"rotowire_{today}".encode()).hexdigest()
    return ROTOWIRE_CACHE_DIR / f"{h}.json"


def _normalize_name(name):
    """Normalize player name for fuzzy matching.
    Strips accents, periods, suffixes, lowercases.
    'Nikola Jokić' → 'nikola jokic'
    'P.J. Washington' → 'pj washington'
    'Marcus Morris Sr.' → 'marcus morris'
    """
    n = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
    n = n.lower().strip()
    n = n.replace(".", "").replace("'", "").replace("-", " ")
    # Remove common suffixes
    for suffix in [" jr", " sr", " iii", " ii", " iv"]:
        if n.endswith(suffix):
            n = n[:-len(suffix)].strip()
    return n


def fetch_rotowire_lineups():
    """Scrape RotoWire NBA lineups page and extract player availability.

    Returns dict keyed by normalized player name:
    {
        "nikola jokic": {
            "name": "Nikola Jokic",
            "team": "DEN",
            "status": "confirmed",    # confirmed | expected | questionable | out
            "is_starter": True,
            "injury_note": "",
            "position": "C",
        },
        ...
    }

    Caches result for 30 minutes (re-scrape on next call after TTL).
    """
    # Check cache (30-min TTL)
    cp = _cache_path()
    if cp.exists():
        try:
            age = datetime.now(timezone.utc).timestamp() - cp.stat().st_mtime
            if age < 1800:  # 30 minutes
                return json.loads(cp.read_text())
        except Exception:
            pass

    # Scrape the page
    try:
        resp = requests.get(LINEUPS_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"RotoWire fetch error: {e}")
        # Return cached data if available (even if stale)
        if cp.exists():
            try:
                return json.loads(cp.read_text())
            except Exception:
                pass
        return {}

    try:
        players = _parse_lineups_html(html)
    except Exception as e:
        print(f"RotoWire parse error: {e}")
        # Degraded mode: return cached data if available, otherwise empty.
        if cp.exists():
            try:
                return json.loads(cp.read_text())
            except Exception:
                pass
        return {}

    # Cache result
    try:
        cp.write_text(json.dumps(players))
    except Exception as e:
        print(f"RotoWire cache write error: {e}")

    return players


def _parse_lineups_html(html):
    """Parse the RotoWire lineups HTML to extract player data.

    The RotoWire lineups page uses a structured layout:
    - Each game is in a div.lineup__main or similar container
    - Players are listed with their status indicated by CSS classes
    - Injured/questionable players have colored indicator bars

    This parser uses regex patterns since we want to avoid requiring
    BeautifulSoup as a dependency (keep the serverless bundle small).
    """
    players = {}

    # ── Strategy: extract data from the embedded JavaScript/JSON ──
    # RotoWire often embeds lineup data in script tags or data attributes.
    # We'll try multiple extraction strategies.

    # Strategy 1: Look for structured data in script tags
    # RotoWire sometimes embeds JSON data for client-side rendering
    json_match = re.search(r'var\s+lineups\s*=\s*(\[.*?\]);', html, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return _parse_json_lineups(data)
        except Exception:
            pass

    # Strategy 2: Parse the HTML structure directly
    # Look for player entries in the lineup cards
    # RotoWire uses patterns like:
    #   <a ... class="lineup__player" ...>Player Name</a>
    #   with nearby status indicators

    # Extract team abbreviations from game headers
    # Pattern: "TEAM @ TEAM" or "TEAM vs TEAM"
    # Extract individual player entries
    # RotoWire player links typically look like:
    # <a href="/basketball/player/..." class="lineup__player">Name</a>
    player_pattern = re.compile(
        r'<a[^>]*href="/basketball/player[^"]*"[^>]*>'
        r'\s*([^<]+?)\s*</a>',
        re.IGNORECASE
    )

    # Look for team context near each player
    # Teams appear as 3-letter abbreviations in various elements
    team_pattern = re.compile(
        r'<(?:div|span|a)[^>]*class="[^"]*lineup__(?:team|abbr)[^"]*"[^>]*>'
        r'\s*([A-Z]{2,3})\s*</(?:div|span|a)>',
        re.IGNORECASE
    )

    # Status indicators — RotoWire uses CSS classes and color bars
    # Classes like "is-injured", "is-questionable", "is-out", "is-confirmed"
    status_pattern = re.compile(
        r'class="[^"]*(?:is-(injured|questionable|out|doubtful|probable|confirmed))[^"]*"',
        re.IGNORECASE
    )

    # Alternative: look for player containers with all data together
    # This broader pattern catches the player block including status
    block_pattern = re.compile(
        r'<(?:li|div)[^>]*class="[^"]*lineup__player[^"]*"[^>]*>'
        r'(.*?)'
        r'</(?:li|div)>',
        re.IGNORECASE | re.DOTALL
    )

    # If we can't parse structured blocks, fall back to a simpler approach:
    # Find all player names and try to determine their status from context
    # Also try to extract from the more modern RotoWire format
    # which uses data attributes
    data_player_pattern = re.compile(
        r'data-player="([^"]+)"[^>]*'
        r'data-team="([^"]*)"[^>]*'
        r'data-status="([^"]*)"',
        re.IGNORECASE
    )

    for match in data_player_pattern.finditer(html):
        name, team, status = match.groups()
        norm = _normalize_name(name)
        status_mapped = _map_status(status)
        players[norm] = {
            "name": name.strip(),
            "team": team.strip().upper(),
            "status": status_mapped,
            "is_starter": status_mapped in (STATUS_CONFIRMED, STATUS_EXPECTED),
            "injury_note": "",
            "position": "",
        }

    # Strategy 3: Parse the visible lineup cards
    # Each game has home/away lineup sections with ordered player lists
    # Starters are listed first (positions 1-5), bench after

    # Look for lineup sections with team headers
    section_pattern = re.compile(
        r'<div[^>]*class="[^"]*lineup__list[^"]*"[^>]*>(.*?)</div>',
        re.IGNORECASE | re.DOTALL
    )

    current_team = ""
    for section in section_pattern.finditer(html):
        block = section.group(1)
        # Try to find team in surrounding context
        team_match = team_pattern.search(html[max(0, section.start()-500):section.start()])
        if team_match:
            current_team = team_match.group(1).upper()

        # Find players in this section
        section_players = player_pattern.findall(block)
        for i, pname in enumerate(section_players):
            pname = pname.strip()
            if not pname or len(pname) < 3:
                continue
            norm = _normalize_name(pname)
            if norm in players:
                continue  # Already found via data attributes

            # Check for status indicators near this player's name
            # Search a window around the player name in the block
            name_pos = block.find(pname)
            context = block[max(0, name_pos-200):name_pos+200] if name_pos >= 0 else ""

            status = STATUS_EXPECTED
            if re.search(r'(?:is-out|ruled.out|out\b)', context, re.IGNORECASE):
                status = STATUS_OUT
            elif re.search(r'(?:is-questionable|game.time|GTD|questionable)', context, re.IGNORECASE):
                status = STATUS_QUESTIONABLE
            elif re.search(r'(?:is-confirmed|confirmed)', context, re.IGNORECASE):
                status = STATUS_CONFIRMED

            # First 5 players in a section are typically starters
            is_starter = i < 5 and status != STATUS_OUT

            players[norm] = {
                "name": pname,
                "team": current_team,
                "status": status,
                "is_starter": is_starter,
                "injury_note": "",
                "position": "",
            }

    # If HTML parsing yielded nothing, try one more approach:
    # Extract from any embedded JSON-LD or inline data
    if not players:
        players = _fallback_parse(html)

    return players


def _parse_json_lineups(data):
    """Parse lineup data from embedded JSON structure."""
    players = {}
    for game in data:
        for side in ["home", "away"]:
            team = game.get(side, {})
            team_abbr = team.get("abbr", team.get("team", "")).upper()
            for i, p in enumerate(team.get("players", team.get("lineup", []))):
                name = p.get("name", p.get("player_name", ""))
                if not name:
                    continue
                status_raw = p.get("status", p.get("injury_status", ""))
                status = _map_status(status_raw)
                norm = _normalize_name(name)
                players[norm] = {
                    "name": name,
                    "team": team_abbr,
                    "status": status,
                    "is_starter": i < 5 and status != STATUS_OUT,
                    "injury_note": p.get("injury", ""),
                    "position": p.get("pos", p.get("position", "")),
                }
    return players


def _map_status(raw):
    """Map a raw status string to our standard status tiers."""
    if not raw:
        return STATUS_EXPECTED
    r = raw.lower().strip()
    if r in ("out", "ruled out", "o", "injured", "inj", "suspended", "suspension"):
        return STATUS_OUT
    if r in ("questionable", "gtd", "game-time", "game time", "doubtful"):
        return STATUS_QUESTIONABLE
    if r in ("confirmed", "active", "probable", "available", "expected to play"):
        return STATUS_CONFIRMED
    if r in ("expected", "likely", "starting"):
        return STATUS_EXPECTED
    return STATUS_UNKNOWN


def _fallback_parse(html):
    """Last-resort parser: extract any player names with OUT/injury tags.
    Even if we can't get full lineup data, knowing who is OUT is valuable."""
    players = {}

    # Look for common OUT patterns in the HTML
    out_pattern = re.compile(
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\s*'
        r'(?:<[^>]*>)*\s*'
        r'(Out|OUT|O|GTD|Questionable|Doubtful)',
        re.IGNORECASE
    )

    for match in out_pattern.finditer(html):
        name = match.group(1).strip()
        status_raw = match.group(2).strip()
        if len(name) < 5 or len(name) > 40:
            continue
        norm = _normalize_name(name)
        players[norm] = {
            "name": name,
            "team": "",
            "status": _map_status(status_raw),
            "is_starter": False,
            "injury_note": status_raw,
            "position": "",
        }

    return players


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API — Called by api/index.py
# ─────────────────────────────────────────────────────────────────────────────

def get_player_status(player_name):
    """Look up a single player's lineup status from RotoWire.

    Args:
        player_name: Full player name (e.g. "Nikola Jokic")

    Returns:
        dict with status info, or None if player not found.
        {
            "status": "confirmed" | "expected" | "questionable" | "out" | "unknown",
            "is_starter": True/False,
            "injury_note": "...",
        }
    """
    lineups = fetch_rotowire_lineups()
    if not lineups:
        return None

    norm = _normalize_name(player_name)

    # Exact match
    if norm in lineups:
        return lineups[norm]

    # Fuzzy: try last name match if exact fails
    last = norm.split()[-1] if norm.split() else ""
    if last and len(last) > 3:
        candidates = [(k, v) for k, v in lineups.items() if k.endswith(last)]
        if len(candidates) == 1:
            return candidates[0][1]

    return None


def is_safe_to_draft(player_name):
    """Quick check: is this player safe for moonshot consideration?

    Returns True if:
      - Player is confirmed or expected in lineup
      - Player is not found (benefit of the doubt — ESPN data will gate)
    Returns False if:
      - Player is OUT or questionable
    """
    info = get_player_status(player_name)
    if info is None:
        return True  # Not found = can't disqualify
    return info["status"] not in (STATUS_OUT, STATUS_QUESTIONABLE)


def get_all_statuses():
    """Return the full RotoWire lineup dict for batch processing.
    Called once per slate build, then individual lookups are O(1)."""
    return fetch_rotowire_lineups()


def clear_cache():
    """Clear RotoWire cache — called by /api/refresh endpoint."""
    cp = _cache_path()
    if cp.exists():
        try:
            cp.unlink()
        except Exception:
            pass
