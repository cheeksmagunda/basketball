"""
Extractor: Winning Drafts / Top Lineups.

Extracts up to 4 winning lineups in long format (5 rows per winner).
Maps to data/winning_drafts/{date}.csv.

Each winning lineup has:
  - winner_rank (1–4)
  - drafter_label (username or "")
  - total_score (sum of RS × slot multipliers)
  - 5 player rows: player_name, team, actual_rs, slot_mult, card_boost, slot_index

After discover.py, update SCREEN_CONFIG with confirmed selectors.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from scripts.real_ingest.extractors.base import (
    extract_all_text_blocks,
    intercept_json_responses,
    navigate_to_screen,
    wait_for_content,
)
from scripts.real_ingest.schemas import SchemaValidationError, build_winning_draft_row

logger = logging.getLogger(__name__)

SCREEN_CONFIG: dict = {
    "nav_selectors": [],
    "url": "",
}

# Real Sports slot multipliers (slot 1 = highest = 2.0x, slot 5 = lowest = 1.2x)
# The app may present players sorted by slot or by RS — we rank by slot_mult
SLOT_MULTIPLIERS = [2.0, 1.8, 1.6, 1.4, 1.2]

JSON_URL_PATTERNS = [
    "winning", "winner", "top-draft", "top_draft", "best-lineup",
    "champion", "leaderboard", "lineup",
]


def _parse_float(text: str) -> str:
    m = re.search(r"[\d]+\.?[\d]*", str(text).replace(",", ""))
    return m.group() if m else ""


def _try_json(captured: list[dict], max_lineups: int = 4) -> list[list[dict]]:
    """
    Try to parse winning lineups from intercepted JSON.
    Returns list of lineups, each lineup = list of player dicts.
    """
    for body in captured:
        # Common response shapes for winning lineups
        lineups_raw = None
        for key in ("lineups", "winning_lineups", "winners", "top_drafts", "entries", "_list"):
            if key in body:
                lineups_raw = body[key]
                break

        if lineups_raw is None and "_list" in body:
            lineups_raw = body["_list"]

        if not lineups_raw or not isinstance(lineups_raw, list):
            continue

        lineups: list[list[dict]] = []
        for lineup_data in lineups_raw[:max_lineups]:
            if not isinstance(lineup_data, dict):
                continue
            # Try to find players array inside the lineup
            players_raw = None
            for key in ("players", "picks", "lineup", "slots", "entries"):
                if key in lineup_data and isinstance(lineup_data[key], list):
                    players_raw = lineup_data[key]
                    break

            if not players_raw:
                continue

            lineup_players: list[dict] = []
            for player in players_raw:
                if not isinstance(player, dict):
                    continue
                name = player.get("player_name") or player.get("player") or player.get("name") or ""
                if not name:
                    continue
                lineup_players.append({
                    "player_name": str(name),
                    "team": str(player.get("team") or ""),
                    "actual_rs": _parse_float(str(player.get("actual_rs") or player.get("rs") or "")),
                    "slot_mult": _parse_float(str(player.get("slot_mult") or player.get("multiplier") or "")),
                    "card_boost": _parse_float(str(player.get("card_boost") or player.get("boost") or "")),
                    "_total_score": str(lineup_data.get("total_score") or lineup_data.get("score") or ""),
                    "_drafter": str(lineup_data.get("username") or lineup_data.get("drafter") or ""),
                })

            if len(lineup_players) >= 2:
                lineups.append(lineup_players)

        if lineups:
            logger.info("[winning_drafts] Extracted %d lineups from JSON intercept", len(lineups))
            return lineups

    return []


def _try_dom_text_blocks(page) -> list[list[dict]]:
    """
    Fallback: use text blocks to identify player names in winning lineups.
    This is heuristic and will need refinement after seeing the actual app.
    """
    blocks = extract_all_text_blocks(page)
    logger.info("[winning_drafts] Text blocks for debugging (%d total):", len(blocks))
    for b in blocks[:15]:
        logger.debug("  Block: %s", b.get("text", "")[:120])
    # Cannot parse winning drafts reliably without knowing the exact DOM structure.
    # Return empty and let the caller know to update selectors.
    return []


def extract(page, date: str, screen: dict | None = None, max_lineups: int = 4) -> list[dict]:
    """
    Navigate to the Winning Drafts screen and extract lineups in long format.

    Returns list of WinningDraftRow dicts (5 rows per lineup × up to 4 lineups = up to 20 rows).
    """
    cfg = screen or SCREEN_CONFIG

    intercepted: list[dict] = intercept_json_responses(page, JSON_URL_PATTERNS)

    logger.info("[winning_drafts] Navigating to Winning Drafts screen for %s", date)
    if cfg.get("nav_selectors") or cfg.get("url"):
        ok = navigate_to_screen(page, cfg)
        if not ok:
            logger.error("[winning_drafts] Navigation failed.")
            return []
    else:
        logger.warning("[winning_drafts] No nav_selectors configured. Run discover.py first.")
        return []

    time.sleep(2.5)
    for sel in ["[class*='lineup']", "[class*='winner']", "[class*='draft']", "table"]:
        if wait_for_content(page, sel, timeout_ms=6_000):
            break

    lineups = _try_json(intercepted, max_lineups=max_lineups)
    if not lineups:
        lineups = _try_dom_text_blocks(page)

    if not lineups:
        logger.error("[winning_drafts] No lineups extracted.")
        return []

    # Convert to long-format rows
    long_rows: list[dict] = []
    for rank, lineup_players in enumerate(lineups, start=1):
        total_score = lineup_players[0].get("_total_score", "") if lineup_players else ""
        drafter = lineup_players[0].get("_drafter", "") if lineup_players else ""

        for slot_idx, player in enumerate(lineup_players[:5], start=1):
            # Assign slot multiplier: if slot_mult not in data, use position-based default
            slot_mult = player.get("slot_mult", "")
            if not slot_mult and slot_idx <= len(SLOT_MULTIPLIERS):
                slot_mult = str(SLOT_MULTIPLIERS[slot_idx - 1])

            raw = {
                "player_name": player.get("player_name", ""),
                "team": player.get("team", ""),
                "actual_rs": player.get("actual_rs", ""),
                "slot_mult": slot_mult,
                "card_boost": player.get("card_boost", ""),
            }
            try:
                row = build_winning_draft_row(
                    raw=raw,
                    winner_rank=rank,
                    slot_index=slot_idx,
                    total_score=total_score,
                    drafter_label=drafter,
                )
                long_rows.append(dict(row))
            except SchemaValidationError as e:
                logger.warning("[winning_drafts] Rank %d slot %d error: %s", rank, slot_idx, e)

    logger.info("[winning_drafts] Final rows: %d (%d lineups) for %s",
                len(long_rows), len(lineups), date)
    return long_rows
