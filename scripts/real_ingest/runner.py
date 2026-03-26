"""
Real Sports Data Ingestion Runner — CLI entrypoint.

Orchestrates the full ingestion pipeline:
  1. Login (or restore session)
  2. Navigate to each requested dataset screen
  3. Extract data
  4. Publish to backend API endpoints
  5. Verify

Usage:
    # Reconnaissance (one-time setup)
    python -m scripts.real_ingest.discover

    # Dry run (parse + validate, no HTTP POSTs)
    python -m scripts.real_ingest.runner --date 2026-03-24 --dry-run

    # Full run — all datasets
    python -m scripts.real_ingest.runner --date 2026-03-24

    # Specific datasets
    python -m scripts.real_ingest.runner --date 2026-03-24 --datasets most_popular,actuals

    # Pre-game boosts (before tip-off)
    python -m scripts.real_ingest.runner --date 2026-03-25 --datasets boosts

    # Headless (CI/automation)
    python -m scripts.real_ingest.runner --date 2026-03-24 --headless

Credentials are read from environment variables (see .env.example):
    REAL_SPORTS_USERNAME
    REAL_SPORTS_PASSWORD
    BASKETBALL_API_BASE (default: http://localhost:8000)
    INGEST_SECRET       (optional)
    INGEST_AUTOMATION_ENABLED (default: true; set to false to abort)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("runner")

# ── Lock file to prevent concurrent runs ─────────────────────────────────────

LOCK_DIR = Path("/tmp")

def _acquire_lock(date: str) -> Path | None:
    lock_file = LOCK_DIR / f"real_ingest_{date}.lock"
    if lock_file.exists():
        age_s = time.time() - lock_file.stat().st_mtime
        if age_s < 3600:  # 1-hour stale threshold
            logger.error(
                "Lock file exists: %s (age %.0fs). "
                "Another run may be in progress. Delete it to force a new run.",
                lock_file, age_s
            )
            return None
        else:
            logger.warning("Stale lock file removed: %s", lock_file)
            lock_file.unlink()
    lock_file.write_text(datetime.now(timezone.utc).isoformat())
    return lock_file


def _release_lock(lock_file: Path) -> None:
    try:
        lock_file.unlink(missing_ok=True)
    except Exception:
        pass


# ── Flow map loader ───────────────────────────────────────────────────────────

FLOW_MAP_PATH = Path("scripts/real_ingest/discovery_report/flow_map.json")

def _load_flow_map() -> dict:
    """Load the confirmed flow_map.json. Returns empty dict if not found."""
    if not FLOW_MAP_PATH.exists():
        logger.warning(
            "flow_map.json not found at %s. "
            "Run 'python -m scripts.real_ingest.discover' first.",
            FLOW_MAP_PATH
        )
        return {}
    try:
        return json.loads(FLOW_MAP_PATH.read_text())
    except Exception as e:
        logger.error("Could not parse flow_map.json: %s", e)
        return {}


def _get_screen_for_dataset(flow_map: dict, dataset: str) -> dict | None:
    """Find the confirmed screen config for a given dataset name."""
    for screen in flow_map.get("screens", []):
        if screen.get("confirmed_dataset") == dataset:
            return screen
    return None


# ── Audit log ─────────────────────────────────────────────────────────────────

def _write_audit_log(date: str, run_result: dict) -> None:
    log_path = Path(f"/tmp/real_ingest_{date}.log")
    try:
        log_path.write_text(json.dumps(run_result, indent=2, default=str))
        logger.info("Audit log written: %s", log_path)
    except Exception as e:
        logger.warning("Could not write audit log: %s", e)


# ── Main pipeline ─────────────────────────────────────────────────────────────

ALL_DATASETS = ["most_popular", "most_drafted_3x", "actuals", "winning_drafts", "boosts"]

def run(
    date: str,
    datasets: list[str] | None = None,
    dry_run: bool = False,
    headless: bool = False,
) -> dict:
    """
    Run the full ingestion pipeline for a given date.

    Args:
        date:     YYYY-MM-DD
        datasets: List of dataset names to collect. None = all.
        dry_run:  If True, extract and validate but do not POST to API.
        headless: Run Playwright in headless mode.

    Returns dict with run summary.
    """
    from playwright.sync_api import sync_playwright
    from scripts.real_ingest.session import get_page
    from scripts.real_ingest import publish, verify
    from scripts.real_ingest.extractors import (
        actuals,
        boosts,
        most_drafted_3x,
        most_popular,
        winning_drafts,
    )

    requested = set(datasets or ALL_DATASETS)
    invalid = requested - set(ALL_DATASETS)
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}. Valid: {ALL_DATASETS}")

    # Kill switch
    enabled = os.environ.get("INGEST_AUTOMATION_ENABLED", "true").strip().lower()
    if enabled in ("false", "0", "no", "off"):
        logger.error("INGEST_AUTOMATION_ENABLED=false — aborting.")
        return {"ok": False, "reason": "kill_switch"}

    # Lock
    lock = _acquire_lock(date)
    if lock is None:
        return {"ok": False, "reason": "lock_exists"}

    flow_map = _load_flow_map()
    if not flow_map:
        logger.warning(
            "No flow_map.json found. Extractors will use their built-in SCREEN_CONFIG defaults "
            "(empty nav_selectors) and will likely return no data. "
            "Run 'python -m scripts.real_ingest.discover' first."
        )

    start_ts = datetime.now(timezone.utc).isoformat()
    extracted: dict[str, list[dict] | dict] = {}
    publish_results: dict[str, dict] = {}
    submitted: dict[str, list[dict]] = {}

    logger.info("=== Real Sports Ingestion Run ===")
    logger.info("Date: %s | Datasets: %s | Dry-run: %s | Headless: %s",
                date, sorted(requested), dry_run, headless)

    try:
        with sync_playwright() as pw:
            page = get_page(pw, headless=headless)

            # ── most_popular ──────────────────────────────────────────────────
            if "most_popular" in requested:
                screen = _get_screen_for_dataset(flow_map, "most_popular")
                rows = most_popular.extract(page, date, screen=screen)
                extracted["most_popular"] = rows
                res = publish.publish_most_popular(date, rows, dry_run=dry_run)
                publish_results["most_popular"] = res
                submitted["most_popular"] = rows
                time.sleep(1.5)

            # ── most_drafted_3x ───────────────────────────────────────────────
            if "most_drafted_3x" in requested:
                screen = _get_screen_for_dataset(flow_map, "most_drafted_3x")
                # Pass already-extracted most_popular rows to avoid re-navigation if possible
                mp_rows = extracted.get("most_popular")
                rows = most_drafted_3x.extract(
                    page, date,
                    screen=screen,
                    most_popular_rows=mp_rows if isinstance(mp_rows, list) else None,
                )
                extracted["most_drafted_3x"] = rows
                res = publish.publish_most_drafted_3x(date, rows, dry_run=dry_run)
                publish_results["most_drafted_3x"] = res
                submitted["most_drafted_3x"] = rows
                time.sleep(1.5)

            # ── actuals ───────────────────────────────────────────────────────
            if "actuals" in requested:
                screen = _get_screen_for_dataset(flow_map, "actuals")
                rows = actuals.extract(page, date, screen=screen)
                extracted["actuals"] = rows
                res = publish.publish_actuals(date, rows, dry_run=dry_run)
                publish_results["actuals"] = res
                submitted["actuals"] = rows
                time.sleep(1.5)

            # ── winning_drafts ────────────────────────────────────────────────
            if "winning_drafts" in requested:
                screen = _get_screen_for_dataset(flow_map, "winning_drafts")
                rows = winning_drafts.extract(page, date, screen=screen)
                extracted["winning_drafts"] = rows
                res = publish.publish_winning_drafts(date, rows, dry_run=dry_run)
                publish_results["winning_drafts"] = res
                submitted["winning_drafts"] = rows
                time.sleep(1.5)

            # ── boosts ────────────────────────────────────────────────────────
            if "boosts" in requested:
                screen = _get_screen_for_dataset(flow_map, "boosts")
                payload = boosts.extract(page, date, screen=screen)
                extracted["boosts"] = payload
                if isinstance(payload, dict) and payload:
                    res = publish.publish_boosts(date, payload, dry_run=dry_run)
                    publish_results["boosts"] = res
                    submitted["boosts"] = payload.get("players", [])
                else:
                    publish_results["boosts"] = {"ok": True, "error": "no data"}

            page.context.browser.close()

    except Exception as e:
        logger.exception("Runner failed: %s", e)
        _release_lock(lock)
        return {"ok": False, "reason": str(e), "extracted": extracted}

    # Print publish summary
    publish.summarise_results(publish_results)

    # Verify
    verify_result = verify.verify_run(date, submitted, dry_run=dry_run)
    verify.print_verify_report(verify_result)

    run_result = {
        "ok": True,
        "date": date,
        "datasets": sorted(requested),
        "dry_run": dry_run,
        "start_ts": start_ts,
        "end_ts": datetime.now(timezone.utc).isoformat(),
        "row_counts": {k: len(v) if isinstance(v, list) else (len(v.get("players", [])) if isinstance(v, dict) else 0) for k, v in extracted.items()},
        "publish_results": publish_results,
        "verify": verify_result,
    }

    _write_audit_log(date, run_result)
    _release_lock(lock)

    logger.info("=== Run complete for %s ===", date)
    logger.info("Row counts: %s", run_result["row_counts"])

    if not dry_run and any(r.get("player_name") or r.get("player") for rows in submitted.values() for r in (rows if isinstance(rows, list) else [])):
        logger.info(
            "Next step: python scripts/rebuild_top_performers_mega.py "
            "  # merges actuals into top_performers.csv"
        )

    return run_result


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Real Sports data ingestion runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--date",
        required=True,
        metavar="YYYY-MM-DD",
        help="Slate date to ingest",
    )
    parser.add_argument(
        "--datasets",
        metavar="DS1,DS2",
        default=None,
        help=f"Comma-separated list of datasets. Default: all. Options: {','.join(ALL_DATASETS)}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Extract and validate but do not POST to API",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run browser in headless mode (for CI / cron)",
    )

    args = parser.parse_args()

    # Validate date format
    try:
        datetime.strptime(args.date, "%Y-%m-%d")
    except ValueError:
        parser.error(f"--date must be YYYY-MM-DD, got: {args.date}")

    datasets = [d.strip() for d in args.datasets.split(",")] if args.datasets else None

    result = run(
        date=args.date,
        datasets=datasets,
        dry_run=args.dry_run,
        headless=args.headless,
    )

    sys.exit(0 if result.get("ok") else 1)


if __name__ == "__main__":
    main()
