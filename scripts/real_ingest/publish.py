"""
Publisher: POST extracted data to the existing basketball backend API endpoints.

All validation and deduplication logic lives in the backend — this module
is purely the HTTP transport layer. It:
  - Reads BASKETBALL_API_BASE and INGEST_SECRET from the environment
  - Posts each dataset to the correct endpoint
  - Handles non-200 responses gracefully (logs and continues)
  - Supports dry_run=True to log payloads without any HTTP calls

Endpoint reference (from CLAUDE.md):
    POST /api/save-most-popular   → data/most_popular/{date}.csv
    POST /api/save-most-drafted-3x → data/most_drafted_3x/{date}.csv
    POST /api/save-winning-drafts → data/winning_drafts/{date}.csv
    POST /api/save-actuals        → data/actuals/{date}.csv
    POST /api/save-boosts         → data/boosts/{date}.json
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)

_API_BASE = os.environ.get("BASKETBALL_API_BASE", "http://localhost:8000").rstrip("/")
_INGEST_SECRET = os.environ.get("INGEST_SECRET", "")


def _headers() -> dict[str, str]:
    h: dict[str, str] = {"Content-Type": "application/json"}
    if _INGEST_SECRET:
        h["X-Ingest-Key"] = _INGEST_SECRET
    return h


def _post(endpoint: str, payload: dict, dry_run: bool = False) -> dict[str, Any]:
    """
    POST payload to endpoint. Returns response dict with keys:
        ok (bool), status (int|None), body (dict), error (str|None)
    """
    url = f"{_API_BASE}{endpoint}"
    if dry_run:
        logger.info("[DRY RUN] POST %s — payload preview:\n%s",
                    url, json.dumps(payload, indent=2, default=str)[:800])
        return {"ok": True, "status": None, "body": {}, "error": None}

    try:
        resp = requests.post(url, json=payload, headers=_headers(), timeout=30)
        body: dict = {}
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:500]}

        if resp.ok:
            logger.info("POST %s → %d OK", url, resp.status_code)
        else:
            logger.error("POST %s → %d ERROR: %s", url, resp.status_code, body)

        return {"ok": resp.ok, "status": resp.status_code, "body": body, "error": None}

    except Exception as e:
        logger.error("POST %s → network error: %s", url, e)
        return {"ok": False, "status": None, "body": {}, "error": str(e)}


# ── Dataset-specific publishers ───────────────────────────────────────────────

def publish_most_popular(date: str, rows: list[dict], dry_run: bool = False) -> dict:
    """
    POST to /api/save-most-popular.

    Body shape: {date, players: [{player, team, draft_count, actual_rs,
                                   actual_card_boost, avg_finish, rank}]}
    """
    if not rows:
        logger.warning("[most_popular] No rows to publish for %s — skipping.", date)
        return {"ok": True, "status": None, "body": {}, "error": "no rows"}

    # Convert any numeric fields that may be strings
    cleaned = []
    for r in rows:
        cleaned.append({
            "player": r.get("player", ""),
            "team": r.get("team", ""),
            "draft_count": int(r.get("draft_count") or 0),
            "actual_rs": r.get("actual_rs", ""),
            "actual_card_boost": r.get("actual_card_boost", ""),
            "avg_finish": r.get("avg_finish", ""),
            "rank": int(r.get("rank") or 0),
        })

    payload = {"date": date, "players": cleaned}
    logger.info("[most_popular] Publishing %d rows for %s", len(cleaned), date)
    return _post("/api/save-most-popular", payload, dry_run=dry_run)


def publish_most_drafted_3x(date: str, rows: list[dict], min_boost: float = 3.0, dry_run: bool = False) -> dict:
    """
    POST to /api/save-most-drafted-3x.

    Body shape: same as most_popular + optional min_boost field.
    """
    if not rows:
        logger.warning("[most_drafted_3x] No rows to publish for %s — skipping.", date)
        return {"ok": True, "status": None, "body": {}, "error": "no rows"}

    cleaned = []
    for r in rows:
        cleaned.append({
            "player": r.get("player", ""),
            "team": r.get("team", ""),
            "draft_count": int(r.get("draft_count") or 0),
            "actual_rs": r.get("actual_rs", ""),
            "actual_card_boost": r.get("actual_card_boost", ""),
            "avg_finish": r.get("avg_finish", ""),
            "rank": int(r.get("rank") or 0),
        })

    payload = {"date": date, "players": cleaned, "min_boost": min_boost}
    logger.info("[most_drafted_3x] Publishing %d rows for %s", len(cleaned), date)
    return _post("/api/save-most-drafted-3x", payload, dry_run=dry_run)


def publish_winning_drafts(date: str, rows: list[dict], dry_run: bool = False) -> dict:
    """
    POST to /api/save-winning-drafts.

    Body shape: {date, rows: [{winner_rank, drafter_label, total_score,
                                slot_index, player_name, team, actual_rs,
                                slot_mult, card_boost}]}
    """
    if not rows:
        logger.warning("[winning_drafts] No rows to publish for %s — skipping.", date)
        return {"ok": True, "status": None, "body": {}, "error": "no rows"}

    cleaned = []
    for r in rows:
        cleaned.append({
            "winner_rank": int(r.get("winner_rank") or 1),
            "drafter_label": r.get("drafter_label", ""),
            "total_score": r.get("total_score", ""),
            "slot_index": int(r.get("slot_index") or 0),
            "player_name": r.get("player_name", ""),
            "team": r.get("team", ""),
            "actual_rs": r.get("actual_rs", ""),
            "slot_mult": r.get("slot_mult", ""),
            "card_boost": r.get("card_boost", ""),
        })

    payload = {"date": date, "rows": cleaned}
    logger.info("[winning_drafts] Publishing %d rows (%d lineups) for %s",
                len(cleaned), len({r["winner_rank"] for r in cleaned}), date)
    return _post("/api/save-winning-drafts", payload, dry_run=dry_run)


def publish_actuals(date: str, rows: list[dict], dry_run: bool = False) -> dict:
    """
    POST to /api/save-actuals.

    Body shape: {date, players: [{player_name, team, actual_rs, actual_card_boost,
                                   drafts, avg_finish, total_value, source}]}
    Note: save-actuals does NOT require INGEST_SECRET header.
    """
    if not rows:
        logger.warning("[actuals] No rows to publish for %s — skipping.", date)
        return {"ok": True, "status": None, "body": {}, "error": "no rows"}

    cleaned = []
    for r in rows:
        cleaned.append({
            "player_name": r.get("player_name", ""),
            "team": r.get("team", ""),
            "actual_rs": r.get("actual_rs", ""),
            "actual_card_boost": r.get("actual_card_boost", ""),
            "drafts": r.get("drafts", ""),
            "avg_finish": r.get("avg_finish", ""),
            "total_value": r.get("total_value", ""),
            "source": r.get("source", "highest_value"),
        })

    payload = {"date": date, "players": cleaned}
    logger.info("[actuals] Publishing %d rows for %s", len(cleaned), date)

    url = f"{_API_BASE}/api/save-actuals"
    if dry_run:
        logger.info("[DRY RUN] POST %s — payload preview:\n%s",
                    url, json.dumps(payload, indent=2, default=str)[:800])
        return {"ok": True, "status": None, "body": {}, "error": None}

    try:
        # save-actuals does NOT require INGEST_SECRET (per CLAUDE.md)
        h = {"Content-Type": "application/json"}
        resp = requests.post(url, json=payload, headers=h, timeout=30)
        body = {}
        try:
            body = resp.json()
        except Exception:
            body = {"raw": resp.text[:500]}
        if resp.ok:
            logger.info("POST %s → %d OK", url, resp.status_code)
        else:
            logger.error("POST %s → %d ERROR: %s", url, resp.status_code, body)
        return {"ok": resp.ok, "status": resp.status_code, "body": body, "error": None}
    except Exception as e:
        logger.error("POST %s → network error: %s", url, e)
        return {"ok": False, "status": None, "body": {}, "error": str(e)}


def publish_boosts(date: str, payload: dict, dry_run: bool = False) -> dict:
    """
    POST to /api/save-boosts.

    Body shape: {date, players: [{player_name, boost, team?, rax_cost?}]}
    """
    if not payload.get("players"):
        logger.warning("[boosts] No players in payload for %s — skipping.", date)
        return {"ok": True, "status": None, "body": {}, "error": "no players"}

    logger.info("[boosts] Publishing %d players for %s", len(payload["players"]), date)
    return _post("/api/save-boosts", payload, dry_run=dry_run)


# ── Summary ───────────────────────────────────────────────────────────────────

def summarise_results(results: dict[str, dict]) -> None:
    """Print a pass/fail summary table for all published datasets."""
    print("\n── Publish Results ──────────────────────────────────────────────")
    for dataset, result in results.items():
        if result.get("status") is None and result.get("error") == "no rows":
            status = "SKIP (no rows)"
        elif result.get("ok"):
            status = f"OK   (HTTP {result.get('status', 'dry-run')})"
        else:
            status = f"FAIL (HTTP {result.get('status')} — {result.get('error') or result.get('body')})"
        print(f"  {dataset:<20} {status}")
    print("─────────────────────────────────────────────────────────────────\n")
