"""
Extractor: Top Performers / Real Scores leaderboard (actuals).

This is the "Highest Value" leaderboard — the primary source of
actual_rs, actual_card_boost, drafts, avg_finish, and total_value.

Maps to data/actuals/{date}.csv and ultimately data/top_performers.csv.

Strategy:
  1. Network intercept — capture the JSON payload for the leaderboard
  2. DOM extraction — parse the visible leaderboard table/list
  3. Text fallback — extract text blocks and parse heuristically

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
from scripts.real_ingest.schemas import SchemaValidationError, build_actual_row

logger = logging.getLogger(__name__)

SCREEN_CONFIG: dict = {
    "nav_selectors": [],
    "url": "",
}

ROW_SELECTORS = [
    "[class*='performer']",
    "[class*='leaderboard-row']",
    "[class*='result-row']",
    "[class*='player-row']",
    "[class*='player-card']",
    "tr[class*='player']",
    "li[class*='player']",
]

FIELD_SELECTORS = {
    "player_name": "[class*='name'], [class*='player-name'], [class*='playerName']",
    "actual_rs": "[class*='rs'], [class*='real-score'], [class*='score'], [class*='pts']",
    "actual_card_boost": "[class*='boost'], [class*='card-boost'], [class*='multiplier']",
    "drafts": "[class*='draft'], [class*='count'], [class*='picks']",
    "avg_finish": "[class*='finish'], [class*='position'], [class*='rank']",
    "total_value": "[class*='value'], [class*='total'], [class*='points']",
    "team": "[class*='team']",
}

JSON_URL_PATTERNS = [
    "leaderboard", "top-performer", "top_performer", "results", "actuals",
    "real-score", "highest-value", "players",
]


def _parse_float(text: str) -> str:
    text = str(text).strip()
    m = re.search(r"[\d]+\.?[\d]*", text)
    return m.group() if m else ""


def _parse_count(text: str) -> str:
    text = str(text).replace(",", "").strip()
    if text.upper().endswith("K"):
        try:
            return str(int(float(text[:-1]) * 1000))
        except ValueError:
            pass
    m = re.search(r"\d+", text)
    return m.group() if m else ""


def _try_json(captured: list[dict]) -> list[dict]:
    for body in captured:
        items = body.get("_list") if "_list" in body else None
        if items is None:
            for key in ("players", "data", "results", "leaderboard", "performers", "entries"):
                if key in body and isinstance(body[key], list):
                    items = body[key]
                    break
        if not items:
            continue

        rows: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = (
                item.get("player_name") or item.get("player") or
                item.get("name") or ""
            )
            if not name:
                continue
            rows.append({
                "player_name": str(name),
                "team": str(item.get("team") or item.get("team_abbr") or ""),
                "actual_rs": _parse_float(str(item.get("actual_rs") or item.get("rs") or item.get("real_score") or "")),
                "actual_card_boost": _parse_float(str(item.get("actual_card_boost") or item.get("boost") or item.get("card_boost") or "")),
                "drafts": _parse_count(str(item.get("drafts") or item.get("draft_count") or item.get("count") or "")),
                "avg_finish": _parse_float(str(item.get("avg_finish") or item.get("finish") or "")),
                "total_value": _parse_float(str(item.get("total_value") or item.get("value") or item.get("total") or "")),
            })

        if rows:
            logger.info("[actuals] Extracted %d rows from JSON intercept", len(rows))
            return rows

    return []


def _try_dom(page) -> list[dict]:
    for sel in ROW_SELECTORS:
        rows = extract_table_rows(page, sel, FIELD_SELECTORS)
        if len(rows) >= 3:
            logger.info("[actuals] Extracted %d rows via DOM: %s", len(rows), sel)
            return rows

    logger.info("[actuals] No DOM rows found — logging text blocks for debugging")
    blocks = extract_all_text_blocks(page)
    for b in blocks[:10]:
        logger.debug("[actuals] Block: %s", b.get("text", "")[:120])
    return []


def extract(page, date: str, screen: dict | None = None) -> list[dict]:
    """
    Navigate to the Top Performers screen and extract actuals data.

    Returns list of validated ActualRow dicts.
    """
    cfg = screen or SCREEN_CONFIG

    intercepted: list[dict] = intercept_json_responses(page, JSON_URL_PATTERNS)

    logger.info("[actuals] Navigating to Top Performers screen for %s", date)
    if cfg.get("nav_selectors") or cfg.get("url"):
        ok = navigate_to_screen(page, cfg)
        if not ok:
            logger.error("[actuals] Navigation failed.")
            return []
    else:
        logger.warning(
            "[actuals] No nav_selectors configured. Run discover.py first."
        )
        return []

    time.sleep(2)
    for sel in ["[class*='performer']", "[class*='leaderboard']", "[class*='result']", "table"]:
        if wait_for_content(page, sel, timeout_ms=6_000):
            break

    raw_rows = _try_json(intercepted)
    if not raw_rows:
        raw_rows = _try_dom(page)

    if not raw_rows:
        logger.error("[actuals] No data extracted.")
        return []

    validated: list[dict] = []
    for i, raw in enumerate(raw_rows):
        try:
            row = build_actual_row(raw)
            validated.append(dict(row))
        except SchemaValidationError as e:
            logger.warning("[actuals] Row %d validation error: %s — skipping", i + 1, e)

    logger.info("[actuals] Final validated rows: %d for %s", len(validated), date)
    return validated
