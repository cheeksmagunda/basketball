"""
Shared utilities for all extractors.

Navigation + network interception + DOM fallback pattern.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Default wait after navigation (seconds)
NAV_WAIT = 2.5


def navigate_to_screen(page, screen: dict) -> bool:
    """
    Navigate to a screen using its confirmed nav_selectors from flow_map.json.

    screen dict fields (from flow_map.json confirmed_dataset entry):
        nav_selectors: list of {type, value} — e.g.
            [{"type": "click_text", "value": "Leaderboard"},
             {"type": "click_text", "value": "Most Drafted"}]
        url: optional direct URL to navigate to

    Selector types supported:
        click_text   — page.get_by_text(value, exact=True).click()
        click_css    — page.locator(value).click()
        goto         — page.goto(value)

    Returns True if navigation succeeded.
    """
    selectors = screen.get("nav_selectors", [])
    direct_url = screen.get("url", "")

    if not selectors and not direct_url:
        logger.warning("Screen %s has no nav_selectors or url — cannot navigate.",
                       screen.get("screen_id", "?"))
        return False

    # Try direct URL if present
    if direct_url and not selectors:
        try:
            page.goto(direct_url, wait_until="domcontentloaded", timeout=20_000)
            time.sleep(NAV_WAIT)
            return True
        except Exception as e:
            logger.warning("goto %s failed: %s", direct_url, e)
            return False

    # Execute selector chain
    for step in selectors:
        sel_type = step.get("type", "click_text")
        value = step.get("value", "")
        if not value:
            continue
        try:
            if sel_type == "goto":
                page.goto(value, wait_until="domcontentloaded", timeout=20_000)
                time.sleep(NAV_WAIT)
            elif sel_type == "click_text":
                page.get_by_text(value, exact=True).first.click(timeout=8_000)
                page.wait_for_load_state("domcontentloaded", timeout=10_000)
                time.sleep(NAV_WAIT)
            elif sel_type == "click_css":
                page.locator(value).first.click(timeout=8_000)
                page.wait_for_load_state("domcontentloaded", timeout=10_000)
                time.sleep(NAV_WAIT)
            else:
                logger.warning("Unknown selector type: %s", sel_type)
        except Exception as e:
            logger.warning("Nav step [%s=%s] failed: %s", sel_type, value, e)
            return False

    return True


def intercept_json_responses(page, url_patterns: list[str]) -> list[dict]:
    """
    Set up a response interceptor and wait for JSON responses matching any pattern.

    Returns list of captured response bodies (dicts or lists-wrapped-as-dicts).
    This should be called BEFORE navigation so the interceptor is ready.
    """
    captured: list[dict] = []

    def on_response(response):
        url = response.url
        if not any(pat in url for pat in url_patterns):
            return
        ct = response.headers.get("content-type", "")
        if "json" not in ct:
            return
        try:
            body = response.json()
            if isinstance(body, list):
                captured.append({"_list": body})
            elif isinstance(body, dict):
                captured.append(body)
        except Exception:
            pass

    page.on("response", on_response)
    return captured  # caller can check this list after navigation


def extract_table_rows(page, row_selector: str, field_map: dict[str, str]) -> list[dict]:
    """
    Extract rows from a table or list using CSS selectors.

    field_map: {output_field: css_selector_within_row}
    e.g. {"player": ".player-name", "draft_count": ".draft-count"}

    Returns list of raw dicts with string values.
    """
    try:
        rows = page.locator(row_selector).all()
    except Exception as e:
        logger.warning("extract_table_rows: locator failed: %s", e)
        return []

    results: list[dict] = []
    for row in rows:
        record: dict[str, str] = {}
        for field, sel in field_map.items():
            try:
                el = row.locator(sel).first
                record[field] = el.inner_text().strip() if el.count() > 0 else ""
            except Exception:
                record[field] = ""
        if any(v for v in record.values()):  # skip empty rows
            results.append(record)

    logger.info("extract_table_rows: found %d rows via %s", len(results), row_selector)
    return results


def extract_all_text_blocks(page) -> list[dict]:
    """
    Generic DOM fallback: extract all visible text blocks with their structure
    for manual parsing when table selectors are unknown.
    """
    try:
        return page.evaluate("""() => {
            const results = [];
            const candidates = document.querySelectorAll(
                'li, tr, [class*="player"], [class*="row"], [class*="item"], [class*="card"]'
            );
            for (const el of candidates) {
                const rect = el.getBoundingClientRect();
                if (rect.height < 10 || rect.width < 50) continue;
                const text = el.innerText.replace(/\\s+/g, ' ').trim();
                if (text.length < 3) continue;
                results.push({
                    tag: el.tagName.toLowerCase(),
                    classes: el.className,
                    text: text.slice(0, 200),
                    childCount: el.children.length,
                });
            }
            return results.slice(0, 200);
        }""")
    except Exception as e:
        logger.warning("extract_all_text_blocks failed: %s", e)
        return []


def wait_for_content(page, selector: str, timeout_ms: int = 10_000) -> bool:
    """Wait for a selector to be visible. Returns True if found."""
    try:
        page.wait_for_selector(selector, timeout=timeout_ms, state="visible")
        return True
    except Exception:
        return False
