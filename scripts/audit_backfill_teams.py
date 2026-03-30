#!/usr/bin/env python3
"""Audit and backfill `team` columns across historical player-row datasets.

Scope:
  - data/top_performers.csv
  - data/actuals/*.csv
  - data/most_popular/*.csv
  - data/most_drafted_3x/*.csv
  - data/winning_drafts/*.csv

Backfill priority (date-aware):
  1) data/predictions/{date}.csv player -> team
  2) same-date rows from historical datasets (cross-dataset union)
  3) ESPN scoreboard + boxscore player -> team for that date

Team normalization:
  Canonical ESPN-ish abbreviations aligned with prediction CSVs:
  GS, NY, SA, NO, UTAH, WSH, etc.

Why this handles trades:
  Team assignment is resolved per date, so a player traded mid-season can map
  to different teams on different dates.

Usage:
  python scripts/audit_backfill_teams.py
  python scripts/audit_backfill_teams.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
TEAMS_JSON = DATA / "teams.json"


def _load_teams_from_json() -> tuple[set[str], dict[str, str]]:
    """Load canonical teams and alias mapping from data/teams.json."""
    if not TEAMS_JSON.exists():
        # Hardcoded fallback for backward compatibility
        canonical = {
            "ATL", "BKN", "BOS", "CHA", "CHI", "CLE", "DAL", "DEN", "DET",
            "GS", "HOU", "IND", "LAC", "LAL", "MEM", "MIA", "MIL", "MIN",
            "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SA", "SAC",
            "TOR", "UTAH", "WSH",
        }
        aliases = {
            "GSW": "GS", "NYK": "NY", "SAS": "SA", "NOP": "NO", "NOH": "NO",
            "WAS": "WSH", "UTA": "UTAH", "UTH": "UTAH", "PHO": "PHX",
            "BRO": "BKN", "NJN": "BKN", "CHO": "CHA",
        }
        return canonical, aliases
    with TEAMS_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    canonical: set[str] = set()
    aliases: dict[str, str] = {}
    for canon, info in data.get("canonical", {}).items():
        canonical.add(canon)
        for alias in info.get("aliases", []):
            aliases[alias] = canon
    return canonical, aliases


# Load from centralized data/teams.json (single source of truth)
CANONICAL_TEAMS, TEAM_ALIASES = _load_teams_from_json()


def _norm_name(raw: str) -> str:
    s = unicodedata.normalize("NFKD", (raw or "").strip())
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def _name_tokens(raw: str) -> list[str]:
    s = unicodedata.normalize("NFKD", (raw or "").strip())
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return [t for t in s.split() if t]


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _name_key_initial_surname(raw: str) -> tuple[str, str]:
    toks = _name_tokens(raw)
    if not toks:
        return "", ""
    first = toks[0][0] if toks[0] else ""
    surname = ""
    for t in reversed(toks):
        if t not in _SUFFIXES:
            surname = t
            break
    return first, surname


def _norm_team(raw: str) -> str:
    t = (raw or "").strip().upper()
    if not t:
        return ""
    t = TEAM_ALIASES.get(t, t)
    if t in CANONICAL_TEAMS:
        return t
    return ""


def _read_csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _parse_dates_arg(raw: str) -> Set[str]:
    out: Set[str] = set()
    for part in (raw or "").split(","):
        d = part.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            out.add(d)
    return out


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "team-audit/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _espn_player_map_for_date(date_str: str, cache: dict[str, dict[str, str]]) -> dict[str, str]:
    if date_str in cache:
        return cache[date_str]
    ymd = date_str.replace("-", "")
    out: dict[str, str] = {}
    try:
        board = _fetch_json(f"{ESPN}/scoreboard?dates={ymd}")
    except Exception:
        cache[date_str] = out
        return out

    for ev in board.get("events") or []:
        gid = str(ev.get("id") or "")
        if not gid:
            continue
        try:
            summ = _fetch_json(f"{ESPN}/summary?event={gid}")
        except Exception:
            continue
        for tb in (summ.get("boxscore") or {}).get("players", []):
            team = _norm_team((tb.get("team") or {}).get("abbreviation", ""))
            if not team:
                continue
            for stat_block in tb.get("statistics") or []:
                for ath in stat_block.get("athletes") or []:
                    name = ((ath.get("athlete") or {}).get("displayName") or "").strip()
                    key = _norm_name(name)
                    if key and key not in out:
                        out[key] = team
    cache[date_str] = out
    return out


def _prediction_maps(date_filter: Optional[Set[str]] = None) -> dict[str, dict[str, str]]:
    maps: dict[str, dict[str, str]] = {}
    for p in sorted((DATA / "predictions").glob("*.csv")):
        date_str = p.stem
        if date_filter and date_str not in date_filter:
            continue
        d: dict[str, str] = {}
        for r in _read_csv_rows(p):
            nm = _norm_name(r.get("player_name", ""))
            tm = _norm_team(r.get("team", ""))
            if nm and tm:
                d[nm] = tm
        maps[date_str] = d
    return maps


def _cross_dataset_maps(date_filter: Optional[Set[str]] = None) -> dict[str, dict[str, str]]:
    per_date: dict[str, dict[str, str]] = defaultdict(dict)

    # actuals
    for p in sorted((DATA / "actuals").glob("*.csv")):
        date_str = p.stem
        if date_filter and date_str not in date_filter:
            continue
        for r in _read_csv_rows(p):
            nm = _norm_name(r.get("player_name", ""))
            tm = _norm_team(r.get("team", ""))
            if nm and tm:
                per_date[date_str][nm] = tm

    # most_popular + most_drafted_3x (column: player)
    for sub in ("most_popular", "most_drafted_3x"):
        for p in sorted((DATA / sub).glob("*.csv")):
            date_str = p.stem
            if date_filter and date_str not in date_filter:
                continue
            for r in _read_csv_rows(p):
                nm = _norm_name(r.get("player", ""))
                tm = _norm_team(r.get("team", ""))
                if nm and tm:
                    per_date[date_str][nm] = tm

    # winning_drafts
    for p in sorted((DATA / "winning_drafts").glob("*.csv")):
        date_str = p.stem
        if date_filter and date_str not in date_filter:
            continue
        for r in _read_csv_rows(p):
            nm = _norm_name(r.get("player_name", ""))
            tm = _norm_team(r.get("team", ""))
            if nm and tm:
                per_date[date_str][nm] = tm

    # top_performers (date in row)
    tp = DATA / "top_performers.csv"
    if tp.is_file():
        for r in _read_csv_rows(tp):
            date_str = (r.get("date") or "").strip()
            if date_filter and date_str not in date_filter:
                continue
            nm = _norm_name(r.get("player_name", ""))
            tm = _norm_team(r.get("team", ""))
            if date_str and nm and tm:
                per_date[date_str][nm] = tm

    return per_date


def _resolve_team(
    date_str: str,
    player_name: str,
    pred: dict[str, dict[str, str]],
    cross: dict[str, dict[str, str]],
    espn_cache: dict[str, dict[str, str]],
) -> str:
    key = _norm_name(player_name)
    if not key:
        return ""
    pred_map = pred.get(date_str) or {}
    cross_map = cross.get(date_str) or {}
    espn_map = _espn_player_map_for_date(date_str, espn_cache)

    t = pred_map.get(key)
    if t:
        return t
    t = cross_map.get(key)
    if t:
        return t
    t = espn_map.get(key, "")
    if t:
        return _norm_team(t)

    # Fallback for abbreviated names in winning_drafts (e.g. "G. Payton II").
    # Build a unique (first initial + surname) index from all date-aware sources.
    idx: dict[tuple[str, str], set[str]] = defaultdict(set)
    for nm, tm in {**pred_map, **cross_map, **espn_map}.items():
        fi, sn = _name_key_initial_surname(nm)
        if fi and sn and tm:
            idx[(fi, sn)].add(tm)
    fi, sn = _name_key_initial_surname(player_name)
    if fi and sn:
        teams = idx.get((fi, sn), set())
        if len(teams) == 1:
            return _norm_team(next(iter(teams)))
    return _norm_team(t)


@dataclass
class Stats:
    rows: int = 0
    blank_before: int = 0
    blank_after: int = 0
    filled: int = 0
    normalized: int = 0
    corrected: int = 0
    unresolved: int = 0


def _process_file(
    path: Path,
    date_str: str,
    name_col: str,
    team_col: str,
    fieldnames: list[str],
    pred: dict[str, dict[str, str]],
    cross: dict[str, dict[str, str]],
    espn_cache: dict[str, dict[str, str]],
    dry_run: bool,
) -> Stats:
    st = Stats()
    rows = _read_csv_rows(path)
    out_rows: list[dict] = []
    for r in rows:
        st.rows += 1
        name = (r.get(name_col) or "").strip()
        raw_team = (r.get(team_col) or "").strip()
        if not raw_team:
            st.blank_before += 1
        canon_existing = _norm_team(raw_team)
        resolved = _resolve_team(date_str, name, pred, cross, espn_cache) if name else ""
        final = resolved or canon_existing

        if raw_team and canon_existing and raw_team.upper() != canon_existing:
            st.normalized += 1
        if not raw_team and final:
            st.filled += 1
        elif raw_team and final and canon_existing and final != canon_existing:
            st.corrected += 1

        if not final:
            st.blank_after += 1
            if name:
                st.unresolved += 1

        r[team_col] = final
        out_rows.append(r)

    if not dry_run:
        _write_csv_rows(path, fieldnames, out_rows)
    return st


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dates", type=str, default="", help="Comma-separated YYYY-MM-DD list (new ingestions only)")
    args = ap.parse_args()
    date_filter = _parse_dates_arg(args.dates)

    pred = _prediction_maps(date_filter if date_filter else None)
    cross = _cross_dataset_maps(date_filter if date_filter else None)
    espn_cache: dict[str, dict[str, str]] = {}

    total = Stats()

    # top_performers.csv
    tp = DATA / "top_performers.csv"
    if tp.is_file():
        tp_rows = _read_csv_rows(tp)
        out_rows = []
        st = Stats()
        fields = [
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
        for r in tp_rows:
            st.rows += 1
            date_str = (r.get("date") or "").strip()
            if date_filter and date_str not in date_filter:
                out_rows.append(r)
                continue
            name = (r.get("player_name") or "").strip()
            raw_team = (r.get("team") or "").strip()
            if not raw_team:
                st.blank_before += 1
            canon_existing = _norm_team(raw_team)
            resolved = _resolve_team(date_str, name, pred, cross, espn_cache) if date_str and name else ""
            final = resolved or canon_existing
            if raw_team and canon_existing and raw_team.upper() != canon_existing:
                st.normalized += 1
            if not raw_team and final:
                st.filled += 1
            elif raw_team and final and canon_existing and final != canon_existing:
                st.corrected += 1
            if not final:
                st.blank_after += 1
                if name:
                    st.unresolved += 1
            r["team"] = final
            out_rows.append(r)
        if not args.dry_run:
            _write_csv_rows(tp, fields, out_rows)
        print(
            f"[top_performers] rows={st.rows} blank_before={st.blank_before} blank_after={st.blank_after} "
            f"filled={st.filled} normalized={st.normalized} corrected={st.corrected} unresolved={st.unresolved}"
        )
        total.rows += st.rows
        total.blank_before += st.blank_before
        total.blank_after += st.blank_after
        total.filled += st.filled
        total.normalized += st.normalized
        total.corrected += st.corrected
        total.unresolved += st.unresolved

    # Per-date datasets
    def run_dir(subdir: str, name_col: str, fields: list[str]):
        nonlocal total
        agg = Stats()
        for p in sorted((DATA / subdir).glob("*.csv")):
            if date_filter and p.stem not in date_filter:
                continue
            s = _process_file(
                p,
                p.stem,
                name_col,
                "team",
                fields,
                pred,
                cross,
                espn_cache,
                args.dry_run,
            )
            agg.rows += s.rows
            agg.blank_before += s.blank_before
            agg.blank_after += s.blank_after
            agg.filled += s.filled
            agg.normalized += s.normalized
            agg.corrected += s.corrected
            agg.unresolved += s.unresolved
        print(
            f"[{subdir}] rows={agg.rows} blank_before={agg.blank_before} blank_after={agg.blank_after} "
            f"filled={agg.filled} normalized={agg.normalized} corrected={agg.corrected} unresolved={agg.unresolved}"
        )
        total.rows += agg.rows
        total.blank_before += agg.blank_before
        total.blank_after += agg.blank_after
        total.filled += agg.filled
        total.normalized += agg.normalized
        total.corrected += agg.corrected
        total.unresolved += agg.unresolved

    run_dir(
        "actuals",
        "player_name",
        ["player_name", "team", "actual_rs", "actual_card_boost", "drafts", "avg_finish", "total_value", "source"],
    )
    run_dir(
        "most_popular",
        "player",
        ["player", "team", "draft_count", "actual_rs", "actual_card_boost", "avg_finish", "rank", "saved_at"],
    )
    run_dir(
        "most_drafted_3x",
        "player",
        ["player", "team", "draft_count", "actual_rs", "actual_card_boost", "avg_finish", "rank", "saved_at"],
    )
    run_dir(
        "winning_drafts",
        "player_name",
        ["winner_rank", "drafter_label", "total_score", "slot_index", "player_name", "team", "actual_rs", "slot_mult", "card_boost", "saved_at"],
    )

    print(
        f"[total] rows={total.rows} blank_before={total.blank_before} blank_after={total.blank_after} "
        f"filled={total.filled} normalized={total.normalized} corrected={total.corrected} unresolved={total.unresolved}"
    )
    if args.dry_run:
        print("[mode] dry-run (no files written)")
    if date_filter:
        print(f"[scope] processed dates={sorted(date_filter)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
