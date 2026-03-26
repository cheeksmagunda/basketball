"""
Extractor: Pre-game player boost list.

Navigates to the screen where Real Sports shows today's player boosts
(card multipliers) before the draft closes. Extracts player name + boost
multiplier + team + rax_cost (if shown).

Maps to data/boosts/{date}.json.

This extractor should be run BEFORE the draft closes for the day
(i.e. before games start) so that pre-game boosts are captured.

After discover.py, update SCREEN_CONFIG with confirmed selectors.
"""

from __future__ import annotations

import logging
import re
import time

from scripts.real_ingest.extractors.base import (
    extract_all_text_blocks,
    extract_table_rows,
    intercept_json_responses,
    navigate_to_screen,
    wait_for_content,
)
from scripts.real_ingest.schemas import (
    SchemaValidationError,
    build_boost_payload,
    build_boost_player,
)

logger = logging.getLogger(__name__)

SCREEN_CONFIG: dict = {
    "nav_selectors": [],
    "url": "",
}

ROW_SELECTORS = [
    "[class*='player-row']",
    "[class*='boost-row']",
    "[class*='card-row']",
    "[class*='player-card']",
    "[class*='player-item']",
    "li[class*='player']",
    "tr",
]

FIELD_SELECTORS = {
    "player_name": "[class*='name'], [class*='player-name']",
    "boost": "[class*='boost'], [class*='multiplier'], [class*='card']",
    "team": "[class*='team']",
    "rax_cost": "[class*='rax'], [class*='cost'], [class*='price']",
}

JSON_URL_PATTERNS = [
    "boost", "card", "multiplier", "players", "draft", "roster",
    "pre-game", "lineup",
]


def _parse_boost(text: str) -> str:
    """Extract boost multiplier from text like '2.5x', '3.0', '+2x'."""
    text = str(text).replace("+", "").replace("x", "").strip()
    m = re.search(r"[\d]+\.?[\d]*", text)
    return m.group() if m else ""


def _try_json(captured: list[dict]) -> list[dict]:
    for body in captured:
        items = body.get("_list") if "_list" in body else None
        if items is None:
            for key in ("players", "data", "roster", "boosts", "entries"):
                if key in body and isinstance(body[key], list):
                    items = body[key]
                    break
        if not items:
            continue

        rows: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("player_name") or item.get("player") or item.get("name") or ""
            boost_raw = item.get("boost") or item.get("card_boost") or item.get("multiplier") or ""
            if not name or not boost_raw:
                continue
            rows.append({
                "player_name": str(name),
                "boost": _parse_boost(str(boost_raw)),
                "team": str(item.get("team") or item.get("team_abbr") or ""),
                "rax_cost": str(item.get("rax_cost") or item.get("rax") or item.get("cost") or ""),
            })

        if rows:
            logger.info("[boosts] Extracted %d players from JSON intercept", len(rows))
            return rows

    return []


def _try_dom(page) -> list[dict]:
    for sel in ROW_SELECTORS:
        rows = extract_table_rows(page, sel, FIELD_SELECTORS)
        if len(rows) >= 3:
            logger.info("[boosts] Extracted %d rows via DOM: %s", len(rows), sel)
            # Normalise boost text
            for r in rows:
                r["boost"] = _parse_boost(r.get("boost", ""))
            return rows

    logger.info("[boosts] DOM fallback — logging text blocks")
    blocks = extract_all_text_blocks(page)
    for b in blocks[:10]:
        logger.debug("[boosts] Block: %s", b.get("text", "")[:120])
    return []


def extract(page, date: str, screen: dict | None = None) -> dict:
    """
    Navigate to the pre-game boosts screen and extract the boost payload.

    Returns a BoostPayload dict ready for POST /api/save-boosts.
    Returns empty payload dict on failure.
    """
    cfg = screen or SCREEN_CONFIG

    intercepted: list[dict] = intercept_json_responses(page, JSON_URL_PATTERNS)

    logger.info("[boosts] Navigating to pre-game boosts screen for %s", date)
    if cfg.get("nav_selectors") or cfg.get("url"):
        ok = navigate_to_screen(page, cfg)
        if not ok:
            logger.error("[boosts] Navigation failed.")
            return {}
    else:
        logger.warning("[boosts] No nav_selectors configured. Run discover.py first.")
        return {}

    time.sleep(2.5)
    for sel in ["[class*='boost']", "[class*='player']", "[class*='card']", "table"]:
        if wait_for_content(page, sel, timeout_ms=6_000):
            break

    raw_rows = _try_json(intercepted)
    if not raw_rows:
        raw_rows = _try_dom(page)

    if not raw_rows:
        logger.error("[boosts] No player data extracted.")
        return {}

    players = []
    for raw in raw_rows:
        try:
            player = build_boost_player(raw)
            players.append(player)
        except SchemaValidationError as e:
            logger.warning("[boosts] Validation error: %s — skipping", e)

    if not players:
        logger.error("[boosts] No valid players after validation.")
        return {}

    payload = build_boost_payload(players, date)
    logger.info("[boosts] Final boost payload: %d players for %s", len(players), date)
    return dict(payload)
