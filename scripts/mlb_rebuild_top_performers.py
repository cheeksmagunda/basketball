#!/usr/bin/env python3
"""Rebuild data/mlb/top_performers.csv as the union of itself and every data/mlb/actuals/*.csv.

Mirrors scripts/rebuild_top_performers_mega.py exactly, but rooted at data/mlb/.

- Date on each actuals row comes from the filename (YYYY-MM-DD.csv).
- Dedupe key: (date, normalized player_name). Later sources overwrite earlier.
- Order: load existing top_performers first, then overlay actuals in date-sorted order
  so hand-uploaded actuals win on conflicts.
- Also outputs data/mlb/top_performers.parquet for faster model training (requires pyarrow).

Usage (repo root):  python scripts/mlb_rebuild_top_performers.py
"""
from __future__ import annotations

import csv
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MLB = REPO / "data" / "mlb"
MEGA = MLB / "top_performers.csv"
PARQUET = MLB / "top_performers.parquet"
ACT = MLB / "actuals"

# MLB uses standard 3-letter abbreviations (NYY, LAD, BOS, etc.)
# No normalization map needed until abbreviation conflicts arise.
MLB_ABBR_FIXES: dict[str, str] = {
    # Add any known alias → canonical mappings here as data accumulates.
    # e.g. "CWS": "CHW",
}

FIELDS = [
    "date",
    "player_name",
    "team",
    "actual_rs",
    "actual_card_boost",
    "drafts",
    "avg_finish",
    "total_value",
    "source",
]


def _norm_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", name or "").encode("ASCII", "ignore").decode("ASCII")
    return n.strip().lower()


def _norm_team(team: str) -> str:
    t = (team or "").strip().upper()
    return MLB_ABBR_FIXES.get(t, t)


def _row_from_actuals(date_str: str, r: dict) -> dict:
    out = {"date": date_str}
    for k in FIELDS[1:]:
        out[k] = (r.get(k) or "").strip() if isinstance(r.get(k), str) else r.get(k, "")
    if not out.get("source"):
        out["source"] = "highest_value"
    if out.get("team"):
        out["team"] = _norm_team(out["team"])
    return out


def main() -> int:
    if not MEGA.exists():
        print(f"[mlb-mega] missing {MEGA} — create it first (headers only is fine)")
        return 1

    mega: dict[tuple[str, str], dict] = {}

    with MEGA.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r.get("date") or "").strip()
            name = (r.get("player_name") or "").strip()
            if not d or not name:
                continue
            row = {k: r.get(k, "") for k in FIELDS}
            if row.get("team"):
                row["team"] = _norm_team(row["team"])
            mega[(d, _norm_name(name))] = row

    for path in sorted(ACT.glob("*.csv")):
        d = path.stem
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                name = (r.get("player_name") or "").strip()
                if not name:
                    continue
                mega[(d, _norm_name(name))] = _row_from_actuals(d, r)

    rows = sorted(mega.values(), key=lambda x: (x["date"], x["player_name"]))

    with MEGA.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in FIELDS})

    print(f"[mlb-mega] wrote {len(rows)} rows to {MEGA.relative_to(REPO)}")

    try:
        import pandas as pd

        df = pd.DataFrame(rows, columns=FIELDS)
        for col in ("actual_rs", "actual_card_boost", "drafts", "avg_finish", "total_value"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("team", "source"):
            df[col] = df[col].astype("category")
        df.to_parquet(PARQUET, index=False, engine="pyarrow")
        print(f"[mlb-mega] wrote parquet to {PARQUET.relative_to(REPO)} ({len(df)} rows)")
    except ImportError:
        print("[mlb-mega] SKIP parquet: pandas/pyarrow not installed")
    except Exception as e:
        print(f"[mlb-mega] WARN parquet write failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
