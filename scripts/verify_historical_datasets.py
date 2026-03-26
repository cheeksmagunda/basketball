#!/usr/bin/env python3
"""Sanity-check canonical historical datasets: row counts and date overlap with predictions.

Usage (repo root):
  python scripts/verify_historical_datasets.py
  python scripts/verify_historical_datasets.py --strict

Exit code: 0 always unless --strict and a check fails (then 2).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="Exit 2 if predictions exist but top_performers has no overlapping dates")
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

    if args.strict and pred_dates and not ovl:
        print("[historical] STRICT FAIL: no date overlap between top_performers and predictions", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
