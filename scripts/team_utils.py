"""Shared NBA team abbreviation normalization backed by data/teams.json.

Single source of truth for all scripts that need to resolve team abbreviations
(e.g. GSW -> GS, NYK -> NY). Lazy-loads the JSON on first call with a hardcoded
fallback if the file is missing.

Usage:
    from scripts.team_utils import normalize_team, canonical_teams, load_team_aliases
"""
from __future__ import annotations

import json
from pathlib import Path

_TEAMS_JSON = Path(__file__).resolve().parent.parent / "data" / "teams.json"

# Module-level cache (populated on first call)
_alias_map: dict[str, str] | None = None
_canonical_set: set[str] | None = None

# Hardcoded fallback if data/teams.json is missing
_FALLBACK_ALIASES: dict[str, str] = {
    "ATL": "ATL", "BKN": "BKN", "BOS": "BOS", "CHA": "CHA", "CHI": "CHI",
    "CLE": "CLE", "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GS": "GS",
    "HOU": "HOU", "IND": "IND", "LAC": "LAC", "LAL": "LAL", "MEM": "MEM",
    "MIA": "MIA", "MIL": "MIL", "MIN": "MIN", "NO": "NO", "NY": "NY",
    "OKC": "OKC", "ORL": "ORL", "PHI": "PHI", "PHX": "PHX", "POR": "POR",
    "SA": "SA", "SAC": "SAC", "TOR": "TOR", "UTAH": "UTAH", "WSH": "WSH",
    # Common aliases
    "GSW": "GS", "NYK": "NY", "SAS": "SA", "NOP": "NO", "NOH": "NO",
    "WAS": "WSH", "UTA": "UTAH", "UTH": "UTAH", "PHO": "PHX",
    "BRO": "BKN", "NJN": "BKN", "CHO": "CHA",
}


def _load() -> None:
    """Load alias map and canonical set from data/teams.json (or fallback)."""
    global _alias_map, _canonical_set
    if _alias_map is not None:
        return

    if _TEAMS_JSON.exists():
        try:
            with _TEAMS_JSON.open("r", encoding="utf-8") as f:
                data = json.load(f)
            mapping: dict[str, str] = {}
            teams: set[str] = set()
            for canon, info in data.get("canonical", {}).items():
                mapping[canon] = canon
                teams.add(canon)
                for alias in info.get("aliases", []):
                    mapping[alias] = canon
            _alias_map = mapping
            _canonical_set = teams
            return
        except (json.JSONDecodeError, KeyError):
            pass  # fall through to hardcoded fallback

    _alias_map = dict(_FALLBACK_ALIASES)
    _canonical_set = {v for v in _FALLBACK_ALIASES.values()}


def load_team_aliases() -> dict[str, str]:
    """Return alias -> canonical mapping (includes canonical -> canonical identity entries)."""
    _load()
    assert _alias_map is not None
    return dict(_alias_map)


def normalize_team(raw: str) -> str:
    """Normalize any team abbreviation to its canonical form.

    Returns the canonical abbreviation if recognized, or the uppercased input
    if not found in the alias map.
    """
    _load()
    assert _alias_map is not None
    t = (raw or "").strip().upper()
    if not t:
        return ""
    return _alias_map.get(t, t)


def canonical_teams() -> set[str]:
    """Return the set of all 30 canonical team abbreviations."""
    _load()
    assert _canonical_set is not None
    return set(_canonical_set)
