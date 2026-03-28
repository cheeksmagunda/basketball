#!/usr/bin/env python3
"""Batch POST /api/parse-screenshot + save-* for docs/historical-ingest raster PNGs.

Maps each PNG to date + screenshot_type from the Historical data input PDFs (Oct 21–Nov 2
slates, 2025-26 season dates as YYYY-MM-DD).

Types used (boosts API removed in backend):
  most_popular, most_drafted_high_boost, top_performers, winning_drafts

Omitted vs PDFs:
  Games/scoreboard-only PNGs — no player parse (skipped).
  Blank pt-1 page-020 — skipped.
  actuals — not run on these screens; they are Highest value / Most popular only; top_performers
    → save-actuals covers highest-value rows. Use actuals only for true My Draft + HV composites.
  winning_drafts for 2025-11-01 — not present in the PDFs (add a row to batch_manifest.json if you capture it).

Dates use 2025-10-xx / 2025-11-xx (NBA 25–26 season). PDF captions said 2026-10-xx; manifest normalizes to 2025.

parse-screenshot is rate-limited (5/min) — default 13s delay between parse calls.

Usage:
  export INGEST_SECRET=...   # if your server requires it
  python scripts/batch_historical_ingest.py --base-url http://127.0.0.1:8000
  python scripts/batch_historical_ingest.py --base-url https://the-oracle.up.railway.app --dry-run

Requires: requests (already in requirements.txt)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests


def load_manifest(ingest_dir: Path) -> list[tuple[str, str, str]]:
    path = ingest_dir / "batch_manifest.json"
    raw = json.loads(path.read_text())
    out: list[tuple[str, str, str]] = []
    for row in raw:
        out.append(
            (
                str(row["image"]),
                str(row["date"]),
                str(row["screenshot_type"]),
            )
        )
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ingest_dir() -> Path:
    return _repo_root() / "docs" / "historical-ingest"


def normalize_players(players: list, screenshot_type: str) -> list:
    out = []
    for p in players:
        if not isinstance(p, dict):
            continue
        row = dict(p)
        if screenshot_type in ("most_popular", "most_drafted", "most_drafted_high_boost"):
            dc = row.get("draft_count")
            if dc is None and row.get("drafts") is not None:
                row["draft_count"] = row["drafts"]
        if screenshot_type == "winning_drafts":
            if row.get("card_boost") is None and row.get("card_multiplier") is not None:
                row["card_boost"] = row["card_multiplier"]
            if row.get("slot_mult") is None and row.get("multiplier") is not None:
                sm = row["multiplier"]
                if isinstance(sm, str) and sm.endswith("x"):
                    sm = sm[:-1].strip()
                try:
                    row["slot_mult"] = float(sm)
                except (TypeError, ValueError):
                    pass
        out.append(row)
    return out


def save_endpoint_and_body(
    date: str, screenshot_type: str, players: list
) -> tuple[str, dict]:
    if screenshot_type in ("most_popular", "most_drafted"):
        return "/api/save-most-popular", {"date": date, "players": players}
    if screenshot_type == "most_drafted_high_boost":
        return "/api/save-most-drafted-3x", {
            "date": date,
            "players": players,
            "min_boost": 3.0,
        }
    if screenshot_type == "top_performers":
        return "/api/save-actuals", {"date": date, "players": players}
    if screenshot_type == "winning_drafts":
        return "/api/save-winning-drafts", {"date": date, "players": players}
    raise ValueError(f"unknown type {screenshot_type}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8000")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sleep", type=float, default=13.0, help="seconds between parse calls")
    ap.add_argument("--ingest-secret", default="", help="or set INGEST_SECRET env")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    secret = args.ingest_secret or os.environ.get("INGEST_SECRET", "")
    headers = {}
    if secret:
        headers["X-Ingest-Key"] = secret

    indir = _ingest_dir()
    manifest = load_manifest(indir)
    ok, failed = 0, 0
    for i, (rel, date, stype) in enumerate(manifest):
        path = indir / rel
        if not path.is_file():
            print(f"MISSING FILE {path}", file=sys.stderr)
            failed += 1
            continue
        label = f"[{i+1}/{len(manifest)}] {rel} | {date} | {stype}"
        if args.dry_run:
            print(f"DRY {label}")
            ok += 1
            continue
        if i > 0 and args.sleep > 0:
            time.sleep(args.sleep)
        with open(path, "rb") as f:
            pr = requests.post(
                f"{base}/api/parse-screenshot",
                files={"file": (path.name, f, "image/png")},
                data={"screenshot_type": stype},
                timeout=120,
            )
        if not pr.ok:
            print(f"FAIL parse {label} HTTP {pr.status_code} {pr.text[:300]}", file=sys.stderr)
            failed += 1
            continue
        data = pr.json()
        if data.get("error"):
            print(f"FAIL parse {label} {data}", file=sys.stderr)
            failed += 1
            continue
        players = normalize_players(data.get("players") or [], stype)
        if not players:
            print(f"FAIL empty players {label}", file=sys.stderr)
            failed += 1
            continue
        ep, body = save_endpoint_and_body(date, stype, players)
        sr = requests.post(
            f"{base}{ep}",
            json=body,
            headers={**headers, "Content-Type": "application/json"},
            timeout=120,
        )
        if not sr.ok:
            print(f"FAIL save {label} {ep} HTTP {sr.status_code} {sr.text[:400]}", file=sys.stderr)
            failed += 1
            continue
        print(f"OK {label} -> {ep} ({len(players)} rows)")
        ok += 1

    print(json.dumps({"ok": ok, "failed": failed, "total": len(manifest)}))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
