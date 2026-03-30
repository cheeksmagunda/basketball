#!/usr/bin/env python3
"""Rebuild data/top_performers.csv as the union of itself and every data/actuals/*.csv.

- Date on each actuals row comes from the filename (YYYY-MM-DD.csv).
- Dedupe key: (date, normalized player_name). Later sources overwrite earlier.
- Order: load existing top_performers first, then overlay actuals in date-sorted order
  so hand-uploaded actuals (e.g. mid-March) win on conflicts.
- Also outputs data/top_performers.parquet for faster model training (requires pyarrow).

Usage (repo root):  python scripts/rebuild_top_performers_mega.py
"""
from __future__ import annotations

import csv
import json
import sys
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MEGA = REPO / "data" / "top_performers.csv"
PARQUET = REPO / "data" / "top_performers.parquet"
ACT = REPO / "data" / "actuals"
TEAMS_JSON = REPO / "data" / "teams.json"

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


def _load_team_aliases() -> dict[str, str]:
    """Load alias->canonical mapping from data/teams.json."""
    if not TEAMS_JSON.exists():
        return {}
    with TEAMS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    mapping: dict[str, str] = {}
    for canon, info in data.get("canonical", {}).items():
        mapping[canon] = canon
        for alias in info.get("aliases", []):
            mapping[alias] = canon
    return mapping


_TEAM_MAP: dict[str, str] = {}


def _normalize_team(raw: str) -> str:
    """Normalize team abbreviation to canonical form using data/teams.json."""
    global _TEAM_MAP
    if not _TEAM_MAP:
        _TEAM_MAP = _load_team_aliases()
    t = (raw or "").strip().upper()
    return _TEAM_MAP.get(t, t)


def _norm_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", name or "").encode("ASCII", "ignore").decode("ASCII")
    return n.strip().lower()


def _row_from_actuals(date_str: str, r: dict) -> dict:
    out = {"date": date_str}
    for k in FIELDS[1:]:
        out[k] = (r.get(k) or "").strip() if isinstance(r.get(k), str) else r.get(k, "")
    if not out.get("source"):
        out["source"] = "highest_value"
    # Normalize team abbreviation to canonical form
    if out.get("team"):
        out["team"] = _normalize_team(out["team"])
    return out


def main() -> int:
    if not MEGA.exists():
        print(f"[mega] missing {MEGA}")
        return 1
    mega: dict[tuple[str, str], dict] = {}

    with MEGA.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = (r.get("date") or "").strip()
            name = (r.get("player_name") or "").strip()
            if not d or not name:
                continue
            row = {k: r.get(k, "") for k in FIELDS}
            # Normalize team abbreviation to canonical form
            if row.get("team"):
                row["team"] = _normalize_team(row["team"])
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

    print(f"[mega] wrote {len(rows)} rows to {MEGA.relative_to(REPO)}")

    # Also output Parquet for faster model training (columnar format)
    try:
        import pandas as pd
        df = pd.read_csv(MEGA)
        df.to_parquet(PARQUET, index=False, engine="pyarrow")
        print(f"[mega] wrote parquet to {PARQUET.relative_to(REPO)} ({len(df)} rows)")
    except ImportError:
        print("[mega] SKIP parquet: pandas/pyarrow not installed (pip install -r requirements-train.txt)")
    except Exception as e:
        print(f"[mega] WARN parquet write failed: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
