#!/usr/bin/env python3
"""Sanity-check canonical historical datasets: row counts, date overlap, and schema validation.

Usage (repo root):
  python scripts/verify_historical_datasets.py
  python scripts/verify_historical_datasets.py --strict
  python scripts/verify_historical_datasets.py --no-validate   # skip schema validation

Exit code: 0 always unless --strict and a check fails (then 2).
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

try:
    from scripts.team_utils import canonical_teams as _canonical_teams
except ImportError:
    from team_utils import canonical_teams as _canonical_teams

REPO = Path(__file__).resolve().parent.parent

# Expected columns in top_performers.csv
REQUIRED_COLUMNS = {"date", "player_name", "team", "actual_rs", "actual_card_boost",
                    "drafts", "avg_finish", "total_value", "source"}

# Allowed source values (each represents a valid ingest path)
VALID_SOURCES = {"highest_value", "real_scores", "most_popular", "winning_drafts",
                 "most_drafted_3x", "leaderboard"}

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _load_canonical_teams() -> set[str]:
    """Load canonical team abbreviations via shared team_utils module."""
    return _canonical_teams()


def _dates_from_top_performers(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r.get("date") or "").strip()
            if d:
                out.add(d)
    return out


def _csv_stems(d: Path) -> set[str]:
    if not d.is_dir():
        return set()
    return {p.stem for p in d.glob("*.csv")}


def _validate_top_performers(path: Path, canonical_teams: set[str]) -> tuple[int, int]:
    """Validate schema of top_performers.csv. Returns (error_count, warning_count)."""
    if not path.is_file():
        print("[validate] SKIP: top_performers.csv not found")
        return 0, 0

    errors = 0
    warnings = 0
    null_counts: dict[str, int] = {col: 0 for col in REQUIRED_COLUMNS}
    bad_dates: list[int] = []
    bad_teams: dict[str, int] = {}
    bad_sources: dict[str, int] = {}
    bad_rs: list[int] = []
    total_rows = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        # Check columns
        if reader.fieldnames:
            actual_cols = set(reader.fieldnames)
            missing = REQUIRED_COLUMNS - actual_cols
            extra = actual_cols - REQUIRED_COLUMNS
            if missing:
                print(f"[validate] ERROR: missing columns: {missing}")
                errors += len(missing)
            if extra:
                print(f"[validate] WARN: extra columns: {extra}")
                warnings += 1

        for i, row in enumerate(reader, start=2):  # line 2 = first data row
            total_rows += 1

            # Track nulls/empties
            for col in REQUIRED_COLUMNS:
                val = (row.get(col) or "").strip()
                if not val:
                    null_counts[col] += 1

            # Validate date format
            date_val = (row.get("date") or "").strip()
            if date_val and not DATE_RE.match(date_val):
                bad_dates.append(i)

            # Validate team abbreviation
            team_val = (row.get("team") or "").strip()
            if team_val and team_val not in canonical_teams:
                bad_teams[team_val] = bad_teams.get(team_val, 0) + 1

            # Validate actual_rs is numeric
            rs_val = (row.get("actual_rs") or "").strip()
            if rs_val:
                try:
                    float(rs_val)
                except ValueError:
                    bad_rs.append(i)

            # Validate source
            src_val = (row.get("source") or "").strip()
            if src_val and src_val not in VALID_SOURCES:
                bad_sources[src_val] = bad_sources.get(src_val, 0) + 1

    # Report null counts ("swamps of nulls")
    print(f"\n[validate] Schema validation for top_performers.csv ({total_rows} rows):")
    null_cols = {col: ct for col, ct in null_counts.items() if ct > 0}
    if null_cols:
        print(f"  Null/empty counts: {null_cols}")
        # Required fields that should never be null
        for col in ("date", "player_name", "actual_rs"):
            if null_counts[col] > 0:
                print(f"  ERROR: {null_counts[col]} rows with empty '{col}' (required field)")
                errors += null_counts[col]
        # Team can be empty (backfill scripts handle it) but warn
        if null_counts["team"] > 0:
            print(f"  WARN: {null_counts['team']} rows with empty 'team'")
            warnings += null_counts["team"]
    else:
        print("  No null/empty values detected")

    if bad_dates:
        print(f"  ERROR: {len(bad_dates)} rows with invalid date format (expected YYYY-MM-DD), lines: {bad_dates[:10]}")
        errors += len(bad_dates)

    if bad_teams:
        print(f"  WARN: non-canonical team abbreviations: {bad_teams}")
        print(f"    (Run scripts/audit_backfill_teams.py to normalize)")
        warnings += sum(bad_teams.values())

    if bad_rs:
        print(f"  ERROR: {len(bad_rs)} rows with non-numeric actual_rs, lines: {bad_rs[:10]}")
        errors += len(bad_rs)

    if bad_sources:
        print(f"  WARN: unexpected source values: {bad_sources}")
        print(f"    Valid sources: {VALID_SOURCES}")
        warnings += sum(bad_sources.values())

    # Summary
    if errors == 0 and warnings == 0:
        print("  PASS: all schema checks passed")
    else:
        print(f"  Summary: {errors} error(s), {warnings} warning(s)")

    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="Exit 2 if predictions exist but top_performers has no overlapping dates")
    ap.add_argument("--no-validate", action="store_true", help="Skip schema validation")
    args = ap.parse_args()

    tp = REPO / "data" / "top_performers.csv"
    pred_dir = REPO / "data" / "predictions"
    mp = REPO / "data" / "most_popular"
    m3 = REPO / "data" / "most_drafted_3x"
    wd = REPO / "data" / "winning_drafts"
    own = REPO / "data" / "ownership"
    slate = REPO / "data" / "slate_results"

    tp_dates = _dates_from_top_performers(tp)
    pred_dates = _csv_stems(pred_dir)
    mp_dates = _csv_stems(mp)
    m3_dates = _csv_stems(m3)
    wd_dates = _csv_stems(wd)
    own_dates = _csv_stems(own)
    slate_dates: set[str] = set()
    if slate.is_dir():
        slate_dates = {p.stem for p in slate.glob("*.json")}

    ovl = tp_dates & pred_dates
    print(
        f"[historical] top_performers dates: {len(tp_dates)} | rows file: {tp.is_file()}\n"
        f"  predictions dates: {len(pred_dates)} | overlap(tp ∩ pred): {len(ovl)}\n"
        f"  most_popular/*.csv: {len(mp_dates)} | most_drafted_3x/*.csv: {len(m3_dates)}\n"
        f"  winning_drafts/*.csv: {len(wd_dates)} | legacy ownership/*.csv: {len(own_dates)}\n"
        f"  slate_results/*.json: {len(slate_dates)}"
    )

    # Schema validation
    if not args.no_validate:
        canonical_teams = _load_canonical_teams()
        err_count, warn_count = _validate_top_performers(tp, canonical_teams)
        if args.strict and err_count > 0:
            print(f"\n[historical] STRICT FAIL: {err_count} schema validation error(s)", file=sys.stderr)
            return 2

    if args.strict and pred_dates and not ovl:
        print("[historical] STRICT FAIL: no date overlap between top_performers and predictions", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
