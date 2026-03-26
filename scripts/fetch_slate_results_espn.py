#!/usr/bin/env python3
"""Download NBA regular-season final scores per calendar day from ESPN scoreboard API.

Writes one JSON per date under data/slate_results/{YYYY-MM-DD}.json with home/away,
scores, and winner/loser (NBA-standard 3-letter abbreviations).

Default range: first 2025-26 regular-season day (2025-10-21) through 2026-03-24 inclusive.
Days with no regular-season games get game_count 0 and games [].

Usage (repo root):
  python scripts/fetch_slate_results_espn.py
  python scripts/fetch_slate_results_espn.py --start 2025-10-21 --end 2026-03-24
  python scripts/fetch_slate_results_espn.py --dry-run  # print summary only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "data" / "slate_results"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
DEFAULT_START = date(2025, 10, 21)
DEFAULT_END = date(2026, 3, 24)
REQUEST_PAUSE_SEC = 0.12


def _daterange(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def _fetch_day(ymd: str) -> dict[str, Any]:
    url = f"{ESPN_SCOREBOARD}?dates={ymd}"
    req = urllib.request.Request(url, headers={"User-Agent": "basketball-slate-results/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _write_flat_csv(out_dir: Path, start: date, end: date) -> Path:
    import csv

    manifest = out_dir / "regular_season_games_flat.csv"
    fields = [
        "date",
        "home",
        "away",
        "home_score",
        "away_score",
        "winner",
        "loser",
        "winner_score",
        "loser_score",
    ]
    rows: list[dict[str, Any]] = []
    for d in _daterange(start, end):
        p = out_dir / f"{d.isoformat()}.json"
        if not p.is_file():
            continue
        day = json.loads(p.read_text(encoding="utf-8"))
        for g in day.get("games") or []:
            rows.append({"date": day["date"], **g})
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return manifest


def _parse_regular_finals(data: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in data.get("events") or []:
        season = ev.get("season") or {}
        if (season.get("slug") or "") != "regular-season":
            continue
        comps = ev.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]
        status = (comp.get("status") or {}).get("type") or {}
        if not status.get("completed"):
            continue
        home = away = None
        for cd in comp.get("competitors") or []:
            team = cd.get("team") or {}
            abbr = (team.get("abbreviation") or "").strip().upper()
            if not abbr:
                continue
            rec = {
                "abbr": abbr,
                "score": int(float(str(cd.get("score", 0) or 0))),
                "winner": bool(cd.get("winner")),
            }
            if cd.get("homeAway") == "home":
                home = rec
            else:
                away = rec
        if not home or not away:
            continue
        if home["score"] == away["score"]:
            continue
        if home["winner"]:
            w, l = home, away
        elif away["winner"]:
            w, l = away, home
        elif home["score"] > away["score"]:
            w, l = home, away
        else:
            w, l = away, home
        out.append(
            {
                "home": home["abbr"],
                "away": away["abbr"],
                "home_score": home["score"],
                "away_score": away["score"],
                "winner": w["abbr"],
                "loser": l["abbr"],
                "winner_score": w["score"],
                "loser_score": l["score"],
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=DEFAULT_START.isoformat())
    ap.add_argument("--end", type=str, default=DEFAULT_END.isoformat())
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--manifest-only",
        action="store_true",
        help="Rebuild regular_season_games_flat.csv from existing JSON only (no HTTP).",
    )
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.manifest_only:
        mf = _write_flat_csv(out_dir, start, end)
        n = sum(1 for _ in mf.open(encoding="utf-8")) - 1
        print(f"[slate-results] manifest-only: {mf} ({n} game rows)")
        return 0

    total_games = 0
    days_with_games = 0
    errors = 0

    for d in _daterange(start, end):
        ymd = d.strftime("%Y%m%d")
        iso = d.isoformat()
        try:
            raw = _fetch_day(ymd)
            games = _parse_regular_finals(raw)
        except Exception as e:
            print(f"[slate-results] ERROR {iso}: {e}", file=sys.stderr)
            errors += 1
            continue
        total_games += len(games)
        if games:
            days_with_games += 1
        payload = {
            "date": iso,
            "game_count": len(games),
            "games": games,
            "season_stage": "regular-season",
            "source": "espn_scoreboard_api",
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if not args.dry_run:
            path = out_dir / f"{iso}.json"
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        if not args.dry_run:
            time.sleep(REQUEST_PAUSE_SEC)

    print(
        f"[slate-results] range {start} .. {end} | days scanned: {(end - start).days + 1} | "
        f"days with games: {days_with_games} | total games: {total_games} | errors: {errors} | "
        f"out: {out_dir}"
    )
    if not args.dry_run and not errors:
        mf = _write_flat_csv(out_dir, start, end)
        print(f"[slate-results] flat manifest: {mf} ({total_games} rows)")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
