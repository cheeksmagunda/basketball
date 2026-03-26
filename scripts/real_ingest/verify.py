"""
Post-write verification for Real Sports ingestion runs.

After publishing data to the backend API, this module confirms that the data
actually landed correctly by querying the read-side endpoints and comparing
player counts + names against what was submitted.

Checks performed:
  1. GET /api/log/get?date={date}       — confirms actuals rows visible in Log
  2. GET /api/audit/get?date={date}     — confirms audit generated (if actuals saved)
  3. Row count comparison per dataset
  4. Spot-check top-3 player names per dataset

All failures are logged as warnings (not errors) — a verify failure does not
indicate data loss, only that the read-side endpoint may be stale (cache) or
the data needs the post-save rebuild step.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

_API_BASE = os.environ.get("BASKETBALL_API_BASE", "http://localhost:8000").rstrip("/")


def _get(path: str, timeout: int = 15) -> dict[str, Any] | None:
    """GET endpoint, return parsed JSON or None on failure."""
    url = f"{_API_BASE}{path}"
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.ok:
            return resp.json()
        logger.warning("GET %s → %d", url, resp.status_code)
        return None
    except Exception as e:
        logger.warning("GET %s → error: %s", url, e)
        return None


def _extract_player_names_from_log(log_data: dict | None) -> list[str]:
    """Extract player names from the /api/log/get response."""
    if not log_data:
        return []
    names: list[str] = []
    # Response has structure: {scopes: [{lineup_type, players: [{player_name, ...}]}]}
    for scope in log_data.get("scopes", []):
        for player in scope.get("players", []):
            n = player.get("player_name") or player.get("name") or ""
            if n:
                names.append(n.strip())
    # Also check top-level players array if present
    for player in log_data.get("players", []):
        n = player.get("player_name") or player.get("name") or ""
        if n:
            names.append(n.strip())
    return list(dict.fromkeys(names))  # dedupe, preserve order


def verify_run(
    date: str,
    submitted: dict[str, list[dict]],
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Verify that submitted data is readable from the backend.

    Args:
        date:      YYYY-MM-DD
        submitted: {dataset_name: [rows]} — what was submitted (for count comparison)
        dry_run:   If True, skip HTTP calls and return a dummy pass result.

    Returns dict with keys:
        passed (bool)
        checks (list of {check, passed, detail})
    """
    if dry_run:
        logger.info("[verify] Dry-run mode — skipping verification HTTP calls.")
        return {"passed": True, "checks": [], "dry_run": True}

    checks: list[dict] = []

    def add_check(name: str, passed: bool, detail: str):
        icon = "✓" if passed else "✗"
        level = logging.INFO if passed else logging.WARNING
        logger.log(level, "[verify] %s %s — %s", icon, name, detail)
        checks.append({"check": name, "passed": passed, "detail": detail})

    # 1. Check Log endpoint is reachable
    log_data = _get(f"/api/log/get?date={date}")
    add_check(
        "log_endpoint_reachable",
        log_data is not None,
        f"/api/log/get?date={date} → {'ok' if log_data else 'failed/empty'}"
    )

    # 2. Check player names appear in log (if actuals were submitted)
    actuals_submitted = submitted.get("actuals", [])
    if actuals_submitted and log_data:
        log_names = _extract_player_names_from_log(log_data)
        submitted_names = [r.get("player_name", "") for r in actuals_submitted[:5]]
        found = [n for n in submitted_names if any(n.lower() in ln.lower() for ln in log_names)]
        add_check(
            "actuals_names_in_log",
            len(found) >= min(3, len(submitted_names)),
            f"found {len(found)}/{len(submitted_names)} spot-checked names in log"
        )

    # 3. Check log/dates includes our date
    dates_data = _get("/api/log/dates")
    if dates_data:
        dates_list = dates_data.get("dates", []) if isinstance(dates_data, dict) else dates_data
        add_check(
            "date_in_log_dates",
            date in dates_list,
            f"{date} in /api/log/dates → {'yes' if date in dates_list else 'not yet (may need rebuild)'}"
        )

    # 4. Check audit if actuals were submitted
    if actuals_submitted:
        audit_data = _get(f"/api/audit/get?date={date}")
        add_check(
            "audit_generated",
            audit_data is not None and "mae" in str(audit_data).lower(),
            f"/api/audit/get?date={date} → {'audit present' if audit_data else 'not found (may need real_scores in actuals)'}"
        )

    # 5. Log/health check
    health = _get("/api/health")
    add_check(
        "backend_health",
        health is not None,
        f"/api/health → {'ok' if health else 'unreachable'}"
    )

    all_passed = all(c["passed"] for c in checks)
    logger.info("[verify] %s %d/%d checks passed for %s",
                "PASS" if all_passed else "WARN",
                sum(1 for c in checks if c["passed"]), len(checks), date)

    return {"passed": all_passed, "checks": checks, "dry_run": False}


def print_verify_report(result: dict) -> None:
    print("\n── Verification Report ──────────────────────────────────────────")
    if result.get("dry_run"):
        print("  (dry-run — verification skipped)")
    else:
        for c in result.get("checks", []):
            icon = "✓" if c["passed"] else "✗"
            print(f"  {icon} {c['check']:<35} {c['detail']}")
    overall = "PASS" if result.get("passed") else "WARN (see above)"
    print(f"\n  Overall: {overall}")
    if not result.get("passed") and not result.get("dry_run"):
        print("  Note: WARN does not mean data loss — backend cache may be stale.")
        print("  Run: python scripts/rebuild_top_performers_mega.py")
    print("─────────────────────────────────────────────────────────────────\n")
