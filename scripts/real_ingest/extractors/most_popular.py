"""
Extractor: Most Popular / Most Drafted leaderboard.

Navigates to the Most Drafted screen, extracts player name + draft count +
boost + RS, and returns validated MostPopularRow dicts.

Strategy (in order):
  1. Network intercept — capture any JSON response with player draft data
  2. DOM extraction — find player rows via common selectors
  3. Text block fallback — parse visible text if table selectors miss

After running discover.py and confirming flow_map.json, update the
SCREEN_CONFIG below with the confirmed selectors.
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
from scripts.real_ingest.schemas import SchemaValidationError, build_most_popular_row

logger = logging.getLogger(__name__)

# ── Update these after running discover.py ────────────────────────────────────
# nav_selectors: list of {type, value} steps to reach this screen.
# Set via flow_map.json confirmed_dataset="most_popular" entry.
SCREEN_CONFIG: dict = {
    "nav_selectors": [],  # populated from flow_map.json at runtime
    "url": "",
}

# Candidate CSS selectors for player rows (try in order)
ROW_SELECTORS = [
    "[class*='player-row']",
    "[class*='playerRow']",
    "[class*='player-card']",
    "[class*='leaderboard-row']",
    "[class*='list-item']",
    "li[class*='player']",
    "tr[class*='player']",
]

# Field selectors within each row
FIELD_SELECTORS_CANDIDATES = [
    {
        "player": "[class*='player-name'], [class*='playerName'], .name",
        "draft_count": "[class*='draft-count'], [class*='draftCount'], [class*='count'], [class*='pct']",
        "actual_card_boost": "[class*='boost'], [class*='multiplier'], [class*='card-boost']",
        "actual_rs": "[class*='rs'], [class*='real-score'], [class*='score']",
        "avg_finish": "[class*='finish'], [class*='avg-finish'], [class*='rank']",
        "team": "[class*='team'], [class*='team-abbr']",
    }
]

# URL patterns to intercept for JSON data
JSON_URL_PATTERNS = [
    "popular", "drafted", "leaderboard", "players", "draft",
    "most-drafted", "most-popular", "ownership",
]


def _parse_number(text: str) -> str:
    """Extract first number from text (handles '1,234', '12.3K', '45%')."""
    text = text.replace(",", "").replace("%", "").strip()
    # Handle K suffix (e.g. "1.2K" → "1200")
    if text.upper().endswith("K"):
        try:
            return str(int(float(text[:-1]) * 1000))
        except ValueError:
            pass
    m = re.search(r"[\d]+\.?[\d]*", text)
    return m.group() if m else ""


def _try_extract_from_json(captured: list[dict]) -> list[dict]:
    """
    Try to extract player rows from intercepted JSON responses.
    Returns list of raw dicts, or empty list if no usable JSON found.
    """
    for body in captured:
        # Handle _list wrapper
        items = body.get("_list", []) if "_list" in body else None

        # Common response shapes
        if items is None:
            for key in ("players", "data", "results", "leaderboard", "entries"):
                if key in body and isinstance(body[key], list):
                    items = body[key]
                    break

        if not items:
            continue

        rows: list[dict] = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            # Try to find player name
            name = (
                item.get("player_name") or item.get("player") or
                item.get("name") or item.get("username") or ""
            )
            if not name:
                continue

            draft_count = (
                item.get("draft_count") or item.get("drafts") or
                item.get("count") or item.get("picks") or 0
            )
            rows.append({
                "player": str(name),
                "team": str(item.get("team") or item.get("team_abbr") or ""),
                "draft_count": _parse_number(str(draft_count)),
                "actual_rs": str(item.get("actual_rs") or item.get("rs") or item.get("score") or ""),
                "actual_card_boost": str(item.get("actual_card_boost") or item.get("boost") or item.get("card_boost") or ""),
                "avg_finish": str(item.get("avg_finish") or item.get("finish") or ""),
            })

        if rows:
            logger.info("[most_popular] Extracted %d rows from JSON intercept", len(rows))
            return rows

    return []


def _try_extract_from_dom(page) -> list[dict]:
    """Try to extract player rows from DOM using candidate selectors."""
    for row_sel in ROW_SELECTORS:
        for field_map in FIELD_SELECTORS_CANDIDATES:
            rows = extract_table_rows(page, row_sel, field_map)
            if len(rows) >= 3:
                logger.info("[most_popular] Extracted %d rows via DOM selector: %s", len(rows), row_sel)
                return rows

    # Fallback: generic text block extraction
    logger.info("[most_popular] Falling back to text block extraction")
    blocks = extract_all_text_blocks(page)
    # Log first 10 blocks for debugging
    for b in blocks[:10]:
        logger.debug("[most_popular] Text block: %s", b.get("text", "")[:100])
    return []


def extract(page, date: str, screen: dict | None = None) -> list[dict]:
    """
    Navigate to the Most Popular screen and extract player data.

    Args:
        page:   Authenticated Playwright Page.
        date:   YYYY-MM-DD (for logging only — not used in navigation).
        screen: Screen config dict from flow_map.json. If None, uses SCREEN_CONFIG.

    Returns:
        List of validated MostPopularRow dicts.
    """
    cfg = screen or SCREEN_CONFIG

    # Set up network intercept BEFORE navigation
    captured: list[dict] = []
    intercepted = intercept_json_responses(page, JSON_URL_PATTERNS)
    # intercepted list is mutated in-place by the event handler

    # Navigate
    logger.info("[most_popular] Navigating to Most Popular screen for %s", date)
    if cfg.get("nav_selectors") or cfg.get("url"):
        ok = navigate_to_screen(page, cfg)
        if not ok:
            logger.error("[most_popular] Navigation failed — aborting extraction.")
            return []
    else:
        logger.warning(
            "[most_popular] No nav_selectors configured. "
            "Run discover.py first and update flow_map.json."
        )
        return []

    # Wait for content
    time.sleep(2)
    for content_sel in [
        "[class*='player']", "[class*='leaderboard']", "[class*='drafted']",
        "table", "ul li", "ol li"
    ]:
        if wait_for_content(page, content_sel, timeout_ms=5_000):
            break

    # Try JSON first (intercepted may have data now)
    raw_rows = _try_extract_from_json(intercepted)

    # DOM fallback
    if not raw_rows:
        raw_rows = _try_extract_from_dom(page)

    if not raw_rows:
        logger.error("[most_popular] No data extracted. Check discover.py report for this screen.")
        return []

    # Validate and build typed rows
    validated: list[dict] = []
    for i, raw in enumerate(raw_rows, start=1):
        try:
            row = build_most_popular_row(raw, rank=i)
            validated.append(dict(row))
        except SchemaValidationError as e:
            logger.warning("[most_popular] Row %d validation error: %s — skipping", i, e)

    logger.info("[most_popular] Final validated rows: %d for %s", len(validated), date)
    return validated
