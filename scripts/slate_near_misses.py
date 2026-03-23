#!/usr/bin/env python3
"""List per-game standouts who did not appear in slate-wide chalk or moonshot.

Uses only data/predictions/{date}.csv (no slate JSON / GitHub).
For each game scope (e.g. 'MIN @ BOS'), ranks THE_LINE_UP rows by predicted_rs
and prints the top K names that are absent from slate chalk ∪ moonshot.

Usage:
  python scripts/slate_near_misses.py 2026-03-22
  python scripts/slate_near_misses.py 2026-03-22 --top 5
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "predictions"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("date", help="YYYY-MM-DD")
    ap.add_argument("--top", type=int, default=3, help="Top K per game (default 3)")
    args = ap.parse_args()

    path = PRED_DIR / f"{args.date}.csv"
    if not path.is_file():
        print(f"Missing {path}", file=sys.stderr)
        return 1

    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    slate_names = set()
    for r in rows:
        if r.get("scope") == "slate" and r.get("lineup_type") in ("chalk", "upside"):
            slate_names.add(r.get("player_name", "").strip().lower())

    by_scope: dict[str, list[dict]] = {}
    for r in rows:
        sc = r.get("scope", "")
        if sc in ("", "slate") or r.get("lineup_type") != "the_lineup":
            continue
        by_scope.setdefault(sc, []).append(r)

    print(f"Date {args.date} — slate-wide roster ({len(slate_names)} names)\n")
    for scope in sorted(by_scope.keys()):
        game_rows = by_scope[scope]
        game_rows.sort(key=lambda x: float(x.get("predicted_rs") or 0), reverse=True)
        misses = []
        for r in game_rows:
            nm = r.get("player_name", "").strip().lower()
            if nm not in slate_names:
                misses.append(r)
            if len(misses) >= args.top:
                break
        print(f"{scope}")
        if not misses:
            print("  (all top {k} in slate or fewer rows)".format(k=args.top))
            continue
        for r in misses:
            print(
                f"  {r.get('player_name'):<22} pred_rs={float(r.get('predicted_rs') or 0):.1f}  "
                f"pts={r.get('pts')}  boost~{r.get('est_card_boost')}"
            )
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
