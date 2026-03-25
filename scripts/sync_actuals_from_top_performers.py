#!/usr/bin/env python3
"""Create data/actuals/{date}.csv for dates that appear in data/top_performers.csv
but have no actuals file yet. Does not overwrite existing CSVs.

Same column layout as hand-uploaded actuals: player_name, actual_rs, actual_card_boost,
drafts, avg_finish, total_value, source.

Usage (repo root):  python scripts/sync_actuals_from_top_performers.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOP = REPO / "data" / "top_performers.csv"
ACT = REPO / "data" / "actuals"
OUT_FIELDS = [
    "player_name",
    "actual_rs",
    "actual_card_boost",
    "drafts",
    "avg_finish",
    "total_value",
    "source",
]


def main() -> int:
    if not TOP.exists():
        print(f"[sync-actuals] missing {TOP}")
        return 1
    ACT.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in ACT.glob("*.csv")}

    rows_by_date: dict[str, list[dict]] = {}
    with TOP.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r.get("date") or "").strip()
            if not d:
                continue
            rows_by_date.setdefault(d, []).append(r)

    missing = sorted(set(rows_by_date.keys()) - have)
    n = 0
    for d in missing:
        out = ACT / f"{d}.csv"
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in rows_by_date[d]:
                w.writerow({k: r.get(k, "") for k in OUT_FIELDS})
        n += 1
    print(f"[sync-actuals] wrote {n} files (skipped {len(have)} existing)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
