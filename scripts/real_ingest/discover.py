"""
Real Sports App Reconnaissance Script.

Run this ONCE (manually, headless=False) after first setup to map the app's
navigation structure. It logs in, clicks through every visible nav item,
captures screenshots + DOM snapshots + network calls, and produces:

    scripts/real_ingest/discovery_report/
    ├── screenshots/           — one PNG per screen visited
    ├── dom_snapshots/         — one HTML per screen visited
    ├── network_log.jsonl      — every XHR/fetch request captured
    └── flow_map.json          — candidate screen→dataset mappings (edit & confirm)

After reviewing the report, update flow_map.json with the confirmed
`confirmed_dataset` field for each screen. The extractors then use
flow_map.json as their navigation source of truth.

Usage:
    cd /home/user/basketball
    python -m scripts.real_ingest.discover
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REPORT_DIR = Path("scripts/real_ingest/discovery_report")
BASE_URL = os.environ.get("REAL_SPORTS_BASE_URL", "https://www.realsports.io").rstrip("/")

# Keywords that help guess which dataset a screen relates to
DATASET_HINTS = {
    "most_popular": ["most popular", "most drafted", "popular", "draft %", "ownership"],
    "most_drafted_3x": ["3x", "high boost", "big boost", "top boost"],
    "actuals": ["top performer", "real score", "leaderboard", "highest value", "results"],
    "winning_drafts": ["winning", "winner", "top draft", "best lineup", "champion"],
    "boosts": ["boost", "card boost", "multiplier", "pre-game", "pre game"],
}

# Nav selectors to try clicking (in order)
NAV_CLICK_TARGETS = [
    # Bottom / top navigation
    "nav a",
    "nav button",
    "[role='navigation'] a",
    "[role='navigation'] button",
    # Tab-style navigation
    "[role='tab']",
    "[role='tablist'] button",
    # Common label patterns
    "a:has-text('Leaderboard')",
    "a:has-text('Results')",
    "a:has-text('History')",
    "a:has-text('Stats')",
    "a:has-text('Draft')",
    "a:has-text('My Drafts')",
    "a:has-text('Profile')",
    "button:has-text('Leaderboard')",
    "button:has-text('Results')",
    "button:has-text('History')",
    "button:has-text('Stats')",
]

# Network request types to capture (ignore static assets)
CAPTURE_RESOURCE_TYPES = {"fetch", "xhr"}
IGNORE_URL_PATTERNS = [
    ".png", ".jpg", ".jpeg", ".svg", ".gif", ".webp", ".ico",
    ".woff", ".woff2", ".ttf", ".otf",
    ".css", ".js",
    "analytics", "tracking", "segment", "mixpanel", "amplitude",
    "sentry", "hotjar", "intercom", "crisp",
]


def _should_capture(url: str, resource_type: str) -> bool:
    if resource_type not in CAPTURE_RESOURCE_TYPES:
        return False
    url_lower = url.lower()
    return not any(pat in url_lower for pat in IGNORE_URL_PATTERNS)


def _guess_dataset(text: str) -> str:
    """Guess which dataset a screen relates to based on visible text."""
    text_lower = text.lower()
    for dataset, hints in DATASET_HINTS.items():
        if any(h in text_lower for h in hints):
            return dataset
    return "unknown"


def _screen_id(url: str, label: str) -> str:
    """Generate a filesystem-safe screen identifier."""
    path = urlparse(url).path.strip("/").replace("/", "_") or "home"
    label_clean = "".join(c if c.isalnum() else "_" for c in label[:30]).strip("_")
    return f"{path}__{label_clean}" if label_clean else path


def _save_screenshot(page, name: str) -> str:
    out_dir = REPORT_DIR / "screenshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    logger.info("Screenshot: %s", path)
    return str(path)


def _save_dom(page, name: str) -> str:
    out_dir = REPORT_DIR / "dom_snapshots"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.html"
    path.write_text(page.content(), encoding="utf-8")
    return str(path)


def _get_visible_text(page) -> str:
    """Extract a condensed version of visible page text for hinting."""
    try:
        return page.evaluate("""() => {
            const walker = document.createTreeWalker(
                document.body,
                NodeFilter.SHOW_TEXT,
                null,
                false
            );
            const texts = [];
            let node;
            while ((node = walker.nextNode())) {
                const t = node.textContent.trim();
                if (t.length > 2) texts.push(t);
            }
            return texts.slice(0, 200).join(' ');
        }""")
    except Exception:
        return ""


def _get_clickable_nav_elements(page) -> list[dict]:
    """Return list of {label, selector, tag, href} for visible nav-ish elements."""
    try:
        return page.evaluate("""() => {
            const results = [];
            const selectors = [
                'nav a', 'nav button',
                '[role=navigation] a', '[role=navigation] button',
                '[role=tab]', '[role=tablist] button',
                'a[href]', 'button[type=button]'
            ];
            const seen = new Set();
            for (const sel of selectors) {
                for (const el of document.querySelectorAll(sel)) {
                    const rect = el.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;
                    const label = (el.textContent || el.getAttribute('aria-label') || '').trim();
                    if (!label || seen.has(label)) continue;
                    seen.add(label);
                    results.push({
                        label: label.slice(0, 80),
                        tag: el.tagName.toLowerCase(),
                        href: el.getAttribute('href') || '',
                        ariaLabel: el.getAttribute('aria-label') || '',
                    });
                }
            }
            return results;
        }""")
    except Exception:
        return []


def run_discovery(headless: bool = False) -> dict:
    """
    Main reconnaissance routine.

    Returns the flow_map dict that is also saved to discovery_report/flow_map.json.
    """
    from playwright.sync_api import sync_playwright
    from scripts.real_ingest.session import get_page

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    network_log_path = REPORT_DIR / "network_log.jsonl"
    network_records: list[dict] = []

    logger.info("=== Real Sports Discovery Run ===")
    logger.info("Report will be written to: %s", REPORT_DIR.resolve())

    with sync_playwright() as pw:
        page = get_page(pw, headless=headless)

        # Attach network interceptor
        def on_request(request):
            if _should_capture(request.url, request.resource_type):
                record = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "method": request.method,
                    "url": request.url,
                    "resource_type": request.resource_type,
                    "headers": dict(request.headers),
                    "post_data": request.post_data or "",
                    "screen": None,  # filled in below per screen
                }
                network_records.append(record)

        def on_response(response):
            # Try to capture JSON responses for API endpoints
            if _should_capture(response.url, response.request.resource_type):
                try:
                    if "application/json" in (response.headers.get("content-type", "")):
                        body = response.json()
                        # attach body to most recent matching network record
                        for rec in reversed(network_records):
                            if rec["url"] == response.url:
                                rec["response_json"] = body if isinstance(body, dict) else {"_list": body[:20] if isinstance(body, list) else body}
                                break
                except Exception:
                    pass

        page.on("request", on_request)
        page.on("response", on_response)

        screens: list[dict] = []
        visited_urls: set[str] = set()

        def capture_screen(label: str) -> dict:
            """Capture current page state and return a screen record."""
            time.sleep(2)  # let dynamic content settle
            url = page.url
            sid = _screen_id(url, label)
            visible_text = _get_visible_text(page)
            nav_elements = _get_clickable_nav_elements(page)

            # Mark network records with this screen
            for rec in network_records:
                if rec["screen"] is None:
                    rec["screen"] = sid

            screenshot_path = _save_screenshot(page, sid)
            dom_path = _save_dom(page, sid)

            screen = {
                "screen_id": sid,
                "label": label,
                "url": url,
                "screenshot": screenshot_path,
                "dom_snapshot": dom_path,
                "visible_text_sample": visible_text[:500],
                "nav_elements": nav_elements,
                "candidate_dataset": _guess_dataset(visible_text),
                "confirmed_dataset": None,  # developer fills this in
                "nav_selectors": [],         # developer fills in confirmed selectors
                "notes": "",
            }
            screens.append(screen)
            logger.info(
                "Screen captured: [%s] url=%s candidate=%s",
                label, url, screen["candidate_dataset"]
            )
            return screen

        # Capture the initial authenticated screen
        page.wait_for_load_state("domcontentloaded", timeout=15_000)
        capture_screen("home")
        visited_urls.add(page.url)

        # Discover nav elements and click each one
        nav_elements = _get_clickable_nav_elements(page)
        logger.info("Found %d nav-ish elements on home screen", len(nav_elements))

        for elem in nav_elements:
            label = elem.get("label", "").strip()
            href = elem.get("href", "")
            if not label:
                continue

            # Skip external links and anchors
            if href.startswith("http") and BASE_URL not in href:
                logger.info("Skipping external link: %s → %s", label, href)
                continue

            logger.info("Clicking nav element: [%s]", label)
            try:
                # Use text matching to find the element
                target = page.locator(f"text='{label}'").first
                if not target.is_visible(timeout=2000):
                    target = page.get_by_text(label, exact=True).first
                if not target.is_visible(timeout=1000):
                    logger.info("  Element not visible, skipping: [%s]", label)
                    continue

                target.click(timeout=5000)
                page.wait_for_load_state("domcontentloaded", timeout=10_000)
                time.sleep(2)

                new_url = page.url
                if new_url in visited_urls:
                    logger.info("  Already visited: %s", new_url)
                    # Still capture if label differs (could be same URL, different state)
                    # but skip to avoid redundant screenshots
                    continue

                visited_urls.add(new_url)
                capture_screen(label)

                # Look for sub-navigation on this screen
                sub_nav = _get_clickable_nav_elements(page)
                for sub in sub_nav:
                    sub_label = sub.get("label", "").strip()
                    if not sub_label or sub_label == label:
                        continue
                    # Only explore one level deep
                    try:
                        sub_target = page.get_by_text(sub_label, exact=True).first
                        if sub_target.is_visible(timeout=1000):
                            sub_target.click(timeout=3000)
                            page.wait_for_load_state("domcontentloaded", timeout=8_000)
                            time.sleep(1.5)
                            sub_url = page.url
                            if sub_url not in visited_urls:
                                visited_urls.add(sub_url)
                                capture_screen(f"{label} > {sub_label}")
                            # Go back to parent
                            page.go_back()
                            page.wait_for_load_state("domcontentloaded", timeout=8_000)
                            time.sleep(1)
                    except Exception:
                        pass

                # Navigate back to home between top-level items
                page.goto(BASE_URL, wait_until="domcontentloaded", timeout=20_000)
                time.sleep(2)

            except Exception as e:
                logger.warning("Could not click [%s]: %s", label, e)
                # Navigate back to home on error
                try:
                    page.goto(BASE_URL, wait_until="domcontentloaded", timeout=15_000)
                    time.sleep(2)
                except Exception:
                    pass

        # Save network log
        with network_log_path.open("w", encoding="utf-8") as f:
            for rec in network_records:
                f.write(json.dumps(rec) + "\n")
        logger.info("Network log: %s (%d records)", network_log_path, len(network_records))

        # Build and save flow_map
        flow_map = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "base_url": BASE_URL,
            "screens": screens,
            "instructions": (
                "Review each screen's screenshot and candidate_dataset. "
                "Set confirmed_dataset to one of: "
                "most_popular | most_drafted_3x | actuals | winning_drafts | boosts | null. "
                "Also fill nav_selectors with the CSS/text selectors needed to navigate to this screen. "
                "Then re-run the extractors — they will use this file as their navigation source."
            ),
        }
        flow_map_path = REPORT_DIR / "flow_map.json"
        flow_map_path.write_text(json.dumps(flow_map, indent=2, default=str), encoding="utf-8")
        logger.info("Flow map saved: %s (%d screens)", flow_map_path, len(screens))

        page.context.browser.close()

    logger.info("=== Discovery complete ===")
    logger.info("Next steps:")
    logger.info("  1. Open %s/screenshots/ and review each PNG", REPORT_DIR)
    logger.info("  2. Edit %s/flow_map.json:", flow_map_path)
    logger.info("     - Set 'confirmed_dataset' for each screen")
    logger.info("     - Set 'nav_selectors' with confirmed click path")
    logger.info("  3. Run: python -m scripts.real_ingest.runner --date YYYY-MM-DD --dry-run")

    return flow_map


if __name__ == "__main__":
    import sys
    headless = "--headless" in sys.argv
    run_discovery(headless=headless)
