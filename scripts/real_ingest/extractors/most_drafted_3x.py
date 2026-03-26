"""
Extractor: Most Drafted 3x+ (high-boost sub-leaderboard).

Reuses most_popular extraction logic but filters for card_boost >= min_boost,
or navigates to a dedicated high-boost sub-tab if the app has one.

After running discover.py, update SCREEN_CONFIG with the confirmed selectors
for the high-boost screen (or leave nav_selectors empty to derive from
most_popular data via client-side filtering).
"""

from __future__ import annotations

import logging

from scripts.real_ingest.extractors.most_popular import extract as extract_most_popular
from scripts.real_ingest.schemas import build_most_popular_row, SchemaValidationError

logger = logging.getLogger(__name__)

# Minimum boost threshold for the 3x+ dataset
DEFAULT_MIN_BOOST = 3.0

# If the app has a dedicated high-boost screen, configure it here after discovery.
# If empty, we fall back to filtering from the most_popular data.
SCREEN_CONFIG: dict = {
    "nav_selectors": [],
    "url": "",
}


def extract(
    page,
    date: str,
    screen: dict | None = None,
    min_boost: float = DEFAULT_MIN_BOOST,
    most_popular_rows: list[dict] | None = None,
) -> list[dict]:
    """
    Extract high-boost (3x+) player data.

    Strategy:
      1. If a dedicated screen is configured (nav_selectors or url), use it.
      2. Otherwise, filter most_popular_rows by card_boost >= min_boost.
         If most_popular_rows is provided, use those directly (saves a second
         navigation). Otherwise, re-run the most_popular extractor.

    Args:
        page:               Authenticated Playwright Page.
        date:               YYYY-MM-DD.
        screen:             Screen config from flow_map.json. If None, uses SCREEN_CONFIG.
        min_boost:          Minimum card boost threshold (default 3.0).
        most_popular_rows:  Pre-extracted most_popular rows (to avoid double navigation).

    Returns:
        List of validated MostPopularRow dicts with boost >= min_boost.
    """
    cfg = screen or SCREEN_CONFIG
    has_dedicated_screen = bool(cfg.get("nav_selectors") or cfg.get("url"))

    if has_dedicated_screen:
        # Navigate to the dedicated high-boost screen and extract fresh
        logger.info("[most_drafted_3x] Using dedicated screen for %s (min_boost=%.1f)", date, min_boost)
        rows = extract_most_popular(page, date, screen=cfg)
    else:
        # Derive from most_popular data
        logger.info("[most_drafted_3x] Deriving from most_popular data (min_boost=%.1f)", min_boost)
        source_rows = most_popular_rows if most_popular_rows is not None else extract_most_popular(page, date)
        rows = source_rows

    # Filter by boost threshold
    filtered: list[dict] = []
    for row in rows:
        boost_str = str(row.get("actual_card_boost", "") or "")
        if not boost_str:
            continue
        try:
            boost_val = float(boost_str)
            if boost_val >= min_boost:
                filtered.append(row)
        except ValueError:
            continue

    # Re-rank after filtering
    for i, row in enumerate(filtered, start=1):
        row["rank"] = i

    logger.info("[most_drafted_3x] %d/%d rows pass min_boost=%.1f for %s",
                len(filtered), len(rows), min_boost, date)

    if not filtered:
        logger.warning(
            "[most_drafted_3x] No rows with boost >= %.1f found. "
            "Boost data may not be available yet (pre-game) or the screen "
            "does not show boost values — check discovery_report.", min_boost
        )

    return filtered
