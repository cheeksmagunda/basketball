#!/usr/bin/env python3
"""Add `team` (NBA abbr) to historical CSVs; backfill from data/predictions/{date}.csv when blank.

Updates:
  - data/top_performers.csv
  - data/actuals/*.csv
  - data/winning_drafts/*.csv

Idempotent: re-run safe. Legacy files without a team column are read via DictReader.

Usage (repo root):  python scripts/migrate_historical_add_team.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TP_FIELDS = [
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
ACT_FIELDS = [
    "player_name",
    "team",
    "actual_rs",
    "actual_card_boost",
    "drafts",
    "avg_finish",
    "total_value",
    "source",
]
WD_FIELDS = [
    "winner_rank",
    "drafter_label",
    "total_score",
    "slot_index",
    "player_name",
    "team",
    "actual_rs",
    "slot_mult",
    "card_boost",
    "saved_at",
]


def _pred_team_map(repo: Path, date_str: str) -> dict[str, str]:
    p = repo / "data" / "predictions" / f"{date_str}.csv"
    if not p.is_file():
        return {}
    out: dict[str, str] = {}
    with p.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nm = (r.get("player_name") or "").strip().lower()
            t = (r.get("team") or "").strip().upper()
            if nm and t:
                out[nm] = t
    return out


def _migrate_top_performers(repo: Path) -> int:
    path = repo / "data" / "top_performers.csv"
    if not path.is_file():
        return 0
    rows_in = list(csv.DictReader(path.open(encoding="utf-8")))
    out: list[dict] = []
    for r in rows_in:
        d = (r.get("date") or "").strip()
        pn = (r.get("player_name") or "").strip()
        if not d or not pn:
            continue
        team = (r.get("team") or "").strip().upper()
        if not team:
            team = _pred_team_map(repo, d).get(pn.lower(), "")
        src = (r.get("source") or "").strip() or "highest_value"
        out.append(
            {
                "date": d,
                "player_name": pn,
                "team": team,
                "actual_rs": r.get("actual_rs", ""),
                "actual_card_boost": r.get("actual_card_boost", ""),
                "drafts": r.get("drafts", ""),
                "avg_finish": r.get("avg_finish", ""),
                "total_value": r.get("total_value", ""),
                "source": src,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TP_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in out:
            w.writerow({k: row.get(k, "") for k in TP_FIELDS})
    return len(out)


def _migrate_actuals(repo: Path) -> int:
    act_dir = repo / "data" / "actuals"
    if not act_dir.is_dir():
        return 0
    n = 0
    for path in sorted(act_dir.glob("*.csv")):
        date_str = path.stem
        pmap = _pred_team_map(repo, date_str)
        rows_in = list(csv.DictReader(path.open(encoding="utf-8")))
        out_rows = []
        for r in rows_in:
            pn = (r.get("player_name") or "").strip()
            if not pn:
                continue
            team = (r.get("team") or "").strip().upper()
            if not team:
                team = pmap.get(pn.lower(), "")
            out_rows.append(
                {
                    "player_name": pn,
                    "team": team,
                    "actual_rs": r.get("actual_rs", ""),
                    "actual_card_boost": r.get("actual_card_boost", ""),
                    "drafts": r.get("drafts", ""),
                    "avg_finish": r.get("avg_finish", ""),
                    "total_value": r.get("total_value", ""),
                    "source": (r.get("source") or "").strip(),
                }
            )
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ACT_FIELDS, extrasaction="ignore")
            w.writeheader()
            for row in out_rows:
                w.writerow(row)
        n += len(out_rows)
    return n


def _migrate_winning_drafts(repo: Path) -> int:
    wd = repo / "data" / "winning_drafts"
    if not wd.is_dir():
        return 0
    n = 0
    for path in sorted(wd.glob("*.csv")):
        date_str = path.stem
        pmap = _pred_team_map(repo, date_str)
        rows_in = list(csv.DictReader(path.open(encoding="utf-8")))
        out_rows = []
        for r in rows_in:
            pname = (r.get("player_name") or "").strip()
            if not pname:
                continue
            team = (r.get("team") or "").strip().upper()
            if not team:
                team = pmap.get(pname.lower(), "")
            out_rows.append(
                {
                    "winner_rank": r.get("winner_rank", ""),
                    "drafter_label": r.get("drafter_label", ""),
                    "total_score": r.get("total_score", ""),
                    "slot_index": r.get("slot_index", ""),
                    "player_name": pname,
                    "team": team,
                    "actual_rs": r.get("actual_rs", ""),
                    "slot_mult": r.get("slot_mult", ""),
                    "card_boost": r.get("card_boost", ""),
                    "saved_at": r.get("saved_at", ""),
                }
            )
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=WD_FIELDS, extrasaction="ignore")
            w.writeheader()
            for row in out_rows:
                w.writerow(row)
        n += len(out_rows)
    return n


def main() -> int:
    tp = _migrate_top_performers(REPO)
    ac = _migrate_actuals(REPO)
    wd = _migrate_winning_drafts(REPO)
    print(f"[migrate-team] top_performers rows: {tp} | actuals row-writes: {ac} | winning_drafts row-writes: {wd}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
