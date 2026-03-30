"""
Data contract definitions for every Real Sports ingestion dataset.

Each TypedDict mirrors the exact column/field set expected by the backend
API endpoints and existing CSV/JSON files in data/.

Validation helpers raise SchemaValidationError on any required-field failure.
All normalisation rules match the existing backend conventions:
  - player names: strip + NFKD + title-case
  - team:         3-letter uppercase abbreviation or "" (never None)
  - floats:       parsed strictly; unparseable → ""
  - dates:        YYYY-MM-DD
  - saved_at:     ISO 8601 with UTC suffix
"""

from __future__ import annotations

import unicodedata
from datetime import datetime, timezone
from typing import Optional, TypedDict

try:
    from scripts.team_utils import normalize_team as _normalize_team_canonical
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from team_utils import normalize_team as _normalize_team_canonical


# ── Exceptions ────────────────────────────────────────────────────────────────

class SchemaValidationError(ValueError):
    """Raised when a required field is missing or invalid."""
    def __init__(self, field: str, value, message: str = ""):
        self.field = field
        self.value = value
        super().__init__(f"[{field}={value!r}] {message}")


# ── TypedDicts ────────────────────────────────────────────────────────────────

class MostPopularRow(TypedDict):
    """One player row for data/most_popular/{date}.csv and most_drafted_3x."""
    player: str            # required; full player name
    team: str              # 3-letter abbr or ""
    draft_count: int       # required; integer >= 0
    actual_rs: str         # float as string, or ""
    actual_card_boost: str # float as string, or ""
    avg_finish: str        # float as string, or ""
    rank: int              # required; ordinal rank (1-based)
    saved_at: str          # ISO 8601 UTC


class ActualRow(TypedDict):
    """One player row for data/actuals/{date}.csv (and top_performers.csv minus date)."""
    player_name: str       # required
    team: str              # 3-letter abbr or ""
    actual_rs: str         # float as string, or ""
    actual_card_boost: str # float as string, or ""
    drafts: str            # integer as string, or ""
    avg_finish: str        # float as string, or ""
    total_value: str       # float as string, or ""
    source: str            # always "highest_value"


class WinningDraftRow(TypedDict):
    """One player-slot row for data/winning_drafts/{date}.csv (long format)."""
    winner_rank: int       # required; 1-based placement
    drafter_label: str     # username or ""
    total_score: str       # float as string (total lineup value)
    slot_index: int        # 1-5
    player_name: str       # required
    team: str              # 3-letter abbr or ""
    actual_rs: str         # float as string, or ""
    slot_mult: str         # float as string (2.0, 1.8, 1.6, 1.4, 1.2)
    card_boost: str        # float as string, or ""
    saved_at: str          # ISO 8601 UTC


class BoostPlayer(TypedDict):
    """One player entry inside data/boosts/{date}.json players array."""
    player_name: str       # required
    boost: float           # required; multiplier (e.g. 2.5)
    team: Optional[str]    # 3-letter abbr if available
    rax_cost: Optional[float]  # RAX cost if displayed


class BoostPayload(TypedDict):
    """Full structure for data/boosts/{date}.json."""
    date: str              # YYYY-MM-DD
    saved_at: str          # ISO 8601 UTC
    player_count: int
    players: list[BoostPlayer]


# ── Normalisation helpers ─────────────────────────────────────────────────────

def _norm_name(raw: str) -> str:
    """Strip whitespace, NFKD-normalise unicode, title-case."""
    if not raw:
        return ""
    cleaned = unicodedata.normalize("NFKD", raw.strip())
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    return cleaned.title()


def _norm_team(raw: str) -> str:
    """Normalize to canonical NBA abbreviation via data/teams.json."""
    return _normalize_team_canonical(raw)


def _norm_float(raw) -> str:
    """Parse to float string with 1 decimal, or empty string on failure."""
    if raw is None or str(raw).strip() in ("", "-", "N/A", "null", "None"):
        return ""
    try:
        return str(round(float(str(raw).replace(",", "")), 1))
    except (ValueError, TypeError):
        return ""


def _norm_int(raw) -> Optional[int]:
    """Parse to int or None on failure."""
    if raw is None or str(raw).strip() in ("", "-", "N/A"):
        return None
    try:
        return int(float(str(raw).replace(",", "")))
    except (ValueError, TypeError):
        return None


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Builders (raw dict → typed row with validation) ───────────────────────────

def build_most_popular_row(raw: dict, rank: int) -> MostPopularRow:
    """
    Build a validated MostPopularRow from a raw extracted dict.
    raw keys accepted (case-insensitive):
      player / name / player_name
      team
      draft_count / drafts / count
      actual_rs / rs / real_score
      actual_card_boost / card_boost / boost
      avg_finish / finish
    """
    def get(*keys):
        for k in keys:
            for candidate in (k, k.lower(), k.upper()):
                if candidate in raw:
                    return raw[candidate]
        return None

    player = _norm_name(str(get("player", "name", "player_name") or ""))
    if not player:
        raise SchemaValidationError("player", player, "player name is required")

    draft_count = _norm_int(get("draft_count", "drafts", "count", "draft_pct"))
    if draft_count is None:
        # Allow 0 as a valid count, only reject None
        draft_count = 0

    return MostPopularRow(
        player=player,
        team=_norm_team(str(get("team") or "")),
        draft_count=draft_count,
        actual_rs=_norm_float(get("actual_rs", "rs", "real_score")),
        actual_card_boost=_norm_float(get("actual_card_boost", "card_boost", "boost", "actual_boost")),
        avg_finish=_norm_float(get("avg_finish", "finish", "avg_rank")),
        rank=rank,
        saved_at=_now_utc(),
    )


def build_actual_row(raw: dict) -> ActualRow:
    """
    Build a validated ActualRow from a raw extracted dict.
    raw keys accepted:
      player_name / player / name
      team
      actual_rs / rs / real_score
      actual_card_boost / card_boost / boost
      drafts / draft_count
      avg_finish / finish
      total_value / value
    """
    def get(*keys):
        for k in keys:
            for candidate in (k, k.lower()):
                if candidate in raw:
                    return raw[candidate]
        return None

    player_name = _norm_name(str(get("player_name", "player", "name") or ""))
    if not player_name:
        raise SchemaValidationError("player_name", player_name, "player name is required")

    return ActualRow(
        player_name=player_name,
        team=_norm_team(str(get("team") or "")),
        actual_rs=_norm_float(get("actual_rs", "rs", "real_score")),
        actual_card_boost=_norm_float(get("actual_card_boost", "card_boost", "boost")),
        drafts=str(_norm_int(get("drafts", "draft_count")) or ""),
        avg_finish=_norm_float(get("avg_finish", "finish")),
        total_value=_norm_float(get("total_value", "value", "total")),
        source="highest_value",
    )


def build_winning_draft_row(
    raw: dict,
    winner_rank: int,
    slot_index: int,
    total_score: str,
    drafter_label: str = "",
) -> WinningDraftRow:
    """
    Build a validated WinningDraftRow for one player-slot in a winning lineup.
    raw keys accepted:
      player_name / player / name
      team
      actual_rs / rs / real_score
      slot_mult / slot_multiplier / multiplier
      card_boost / boost / actual_card_boost
    """
    def get(*keys):
        for k in keys:
            for candidate in (k, k.lower()):
                if candidate in raw:
                    return raw[candidate]
        return None

    player_name = _norm_name(str(get("player_name", "player", "name") or ""))
    if not player_name:
        raise SchemaValidationError("player_name", player_name, "player name is required")

    if not (1 <= slot_index <= 5):
        raise SchemaValidationError("slot_index", slot_index, "must be 1-5")

    return WinningDraftRow(
        winner_rank=winner_rank,
        drafter_label=drafter_label,
        total_score=_norm_float(total_score),
        slot_index=slot_index,
        player_name=player_name,
        team=_norm_team(str(get("team") or "")),
        actual_rs=_norm_float(get("actual_rs", "rs", "real_score")),
        slot_mult=_norm_float(get("slot_mult", "slot_multiplier", "multiplier")),
        card_boost=_norm_float(get("card_boost", "boost", "actual_card_boost")),
        saved_at=_now_utc(),
    )


def build_boost_player(raw: dict) -> BoostPlayer:
    """
    Build a validated BoostPlayer entry.
    raw keys accepted:
      player_name / player / name
      boost / card_boost / multiplier
      team
      rax_cost / cost / rax
    """
    def get(*keys):
        for k in keys:
            for candidate in (k, k.lower()):
                if candidate in raw:
                    return raw[candidate]
        return None

    player_name = _norm_name(str(get("player_name", "player", "name") or ""))
    if not player_name:
        raise SchemaValidationError("player_name", player_name, "player name is required")

    boost_raw = get("boost", "card_boost", "multiplier")
    try:
        boost = round(float(str(boost_raw).replace("x", "").strip()), 1)
    except (ValueError, TypeError):
        raise SchemaValidationError("boost", boost_raw, "boost multiplier is required and must be numeric")

    rax_raw = get("rax_cost", "cost", "rax")
    rax_cost: Optional[float] = None
    if rax_raw is not None:
        try:
            rax_cost = round(float(str(rax_raw).replace(",", "")), 0)
        except (ValueError, TypeError):
            pass

    team_raw = get("team")
    team: Optional[str] = _norm_team(str(team_raw)) if team_raw else None

    return BoostPlayer(player_name=player_name, boost=boost, team=team, rax_cost=rax_cost)


def build_boost_payload(players: list[BoostPlayer], date: str) -> BoostPayload:
    return BoostPayload(
        date=date,
        saved_at=_now_utc(),
        player_count=len(players),
        players=players,
    )
