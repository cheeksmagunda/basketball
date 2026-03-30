#!/usr/bin/env python3
"""Second-pass team consistency cleanup after bulk audit.

Actions:
1) Remove blank-player rows from most_popular / most_drafted_3x (unusable rows).
2) Fill winning_drafts blank teams via date-aware initial+surname matching.
3) Fill residual known full-name blanks in actuals/top_performers.
"""

from __future__ import annotations

import csv
import json
import re
import urllib.request
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

try:
    from scripts.team_utils import normalize_team
except ImportError:
    from team_utils import normalize_team

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

KNOWN_PLAYER_TEAMS = {
    "alexandre sarr": "WSH",
    "gg jackson ii": "MEM",
    "nicolas claxton": "BKN",
    "ron holland ii": "DET",
}

KNOWN_ABBREV_TEAMS = {
    "l doncic": "LAL",
    "ldoncic": "LAL",
    "j green": "HOU",
    "j smith jr": "HOU",
}

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_team(t: str) -> str:
    return normalize_team(t)


def _parse_dates_arg(raw: str) -> Set[str]:
    out: Set[str] = set()
    for part in (raw or "").split(","):
        d = part.strip()
        if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
            out.add(d)
    return out


def name_key(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").strip())
    s = s.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "", s)


def initial_surname(s: str) -> tuple[str, str]:
    toks = [t for t in re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).split() if t]
    if not toks:
        return "", ""
    first = toks[0][0]
    surname = ""
    for t in reversed(toks):
        if t not in SUFFIXES:
            surname = t
            break
    return first, surname


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "team-pass2/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _espn_indices_for_date(
    date_str: str,
    cache: Dict[str, Tuple[Dict[str, str], Dict[Tuple[str, str], Set[str]]]],
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], Set[str]]]:
    if date_str in cache:
        return cache[date_str]
    full: Dict[str, str] = {}
    initial_idx: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    ymd = date_str.replace("-", "")
    try:
        board = _fetch_json(f"{ESPN}/scoreboard?dates={ymd}")
    except Exception:
        cache[date_str] = (full, initial_idx)
        return full, initial_idx
    for ev in board.get("events") or []:
        gid = str(ev.get("id") or "")
        if not gid:
            continue
        try:
            summ = _fetch_json(f"{ESPN}/summary?event={gid}")
        except Exception:
            continue
        for team_block in (summ.get("boxscore") or {}).get("players", []):
            tm = norm_team((team_block.get("team") or {}).get("abbreviation", ""))
            if not tm:
                continue
            for stat_block in team_block.get("statistics") or []:
                for ath in stat_block.get("athletes") or []:
                    nm = ((ath.get("athlete") or {}).get("displayName") or "").strip()
                    if not nm:
                        continue
                    full[name_key(nm)] = tm
                    initial_idx[initial_surname(nm)].add(tm)
    cache[date_str] = (full, initial_idx)
    return full, initial_idx


def clean_blank_player_rows(date_filter: Set[str]) -> tuple[int, int]:
    removed_mp = 0
    removed_3x = 0
    spec = [
        ("most_popular", "player", ["player", "team", "draft_count", "actual_rs", "actual_card_boost", "avg_finish", "rank", "saved_at"]),
        ("most_drafted_3x", "player", ["player", "team", "draft_count", "actual_rs", "actual_card_boost", "avg_finish", "rank", "saved_at"]),
    ]
    for sub, name_col, fields in spec:
        for p in sorted((DATA / sub).glob("*.csv")):
            if date_filter and p.stem not in date_filter:
                continue
            rows = read_rows(p)
            out = [r for r in rows if (r.get(name_col) or "").strip()]
            for r in out:
                r["team"] = norm_team(r.get("team", ""))
            write_rows(p, fields, out)
            removed = len(rows) - len(out)
            if sub == "most_popular":
                removed_mp += removed
            else:
                removed_3x += removed
    return removed_mp, removed_3x


def build_date_maps(date_filter: Set[str]) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[Tuple[str, str], Set[str]]]]:
    full_map: Dict[str, Dict[str, str]] = defaultdict(dict)
    init_map: Dict[str, Dict[Tuple[str, str], Set[str]]] = defaultdict(lambda: defaultdict(set))

    # predictions (strongest per-date source)
    for p in sorted((DATA / "predictions").glob("*.csv")):
        d = p.stem
        if date_filter and d not in date_filter:
            continue
        for r in read_rows(p):
            nm = (r.get("player_name") or "").strip()
            tm = norm_team(r.get("team", ""))
            if nm and tm:
                full_map[d][name_key(nm)] = tm
                init_map[d][initial_surname(nm)].add(tm)

    # actuals as additional source
    for p in sorted((DATA / "actuals").glob("*.csv")):
        d = p.stem
        if date_filter and d not in date_filter:
            continue
        for r in read_rows(p):
            nm = (r.get("player_name") or "").strip()
            tm = norm_team(r.get("team", ""))
            if nm and tm:
                full_map[d][name_key(nm)] = tm
                init_map[d][initial_surname(nm)].add(tm)

    return full_map, init_map


def fill_winning_drafts(full_map: Dict[str, Dict[str, str]], init_map: Dict[str, Dict[Tuple[str, str], Set[str]]], date_filter: Set[str]) -> int:
    changed = 0
    espn_cache: Dict[str, Tuple[Dict[str, str], Dict[Tuple[str, str], Set[str]]]] = {}
    fields = ["winner_rank", "drafter_label", "total_score", "slot_index", "player_name", "team", "actual_rs", "slot_mult", "card_boost", "saved_at"]
    for p in sorted((DATA / "winning_drafts").glob("*.csv")):
        d = p.stem
        if date_filter and d not in date_filter:
            continue
        rows = read_rows(p)
        for r in rows:
            team = norm_team(r.get("team", ""))
            if team:
                r["team"] = team
                continue
            nm = (r.get("player_name") or "").strip()
            if not nm:
                continue
            resolved = full_map.get(d, {}).get(name_key(nm), "")
            if not resolved:
                teams = init_map.get(d, {}).get(initial_surname(nm), set())
                if len(teams) == 1:
                    resolved = next(iter(teams))
            if not resolved:
                efull, einit = _espn_indices_for_date(d, espn_cache)
                resolved = efull.get(name_key(nm), "")
                if not resolved:
                    et = einit.get(initial_surname(nm), set())
                    if len(et) == 1:
                        resolved = next(iter(et))
            if not resolved:
                raw_key = re.sub(r"[^a-z0-9]+", " ", nm.lower()).strip()
                resolved = KNOWN_ABBREV_TEAMS.get(raw_key, "") or KNOWN_ABBREV_TEAMS.get(name_key(nm), "")
            if resolved:
                r["team"] = resolved
                changed += 1
        write_rows(p, fields, rows)
    return changed


def fill_residual_known_names(date_filter: Set[str]) -> int:
    changed = 0
    # actuals
    act_fields = ["player_name", "team", "actual_rs", "actual_card_boost", "drafts", "avg_finish", "total_value", "source"]
    for p in sorted((DATA / "actuals").glob("*.csv")):
        if date_filter and p.stem not in date_filter:
            continue
        rows = read_rows(p)
        for r in rows:
            if (r.get("team") or "").strip():
                continue
            nm = (r.get("player_name") or "").strip().lower()
            tm = KNOWN_PLAYER_TEAMS.get(nm, "")
            if tm:
                r["team"] = tm
                changed += 1
        write_rows(p, act_fields, rows)

    # top_performers
    tp = DATA / "top_performers.csv"
    if tp.is_file():
        tp_fields = ["date", "player_name", "team", "actual_rs", "actual_card_boost", "drafts", "avg_finish", "total_value", "source"]
        rows = read_rows(tp)
        for r in rows:
            if date_filter and (r.get("date") or "").strip() not in date_filter:
                continue
            if (r.get("team") or "").strip():
                r["team"] = norm_team(r.get("team", ""))
                continue
            nm = (r.get("player_name") or "").strip().lower()
            tm = KNOWN_PLAYER_TEAMS.get(nm, "")
            if tm:
                r["team"] = tm
                changed += 1
        write_rows(tp, tp_fields, rows)
    return changed


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", type=str, default="", help="Comma-separated YYYY-MM-DD list (new ingestions only)")
    args = ap.parse_args()
    date_filter = _parse_dates_arg(args.dates)

    rm_mp, rm_3x = clean_blank_player_rows(date_filter)
    full_map, init_map = build_date_maps(date_filter)
    wd_changed = fill_winning_drafts(full_map, init_map, date_filter)
    known_changed = fill_residual_known_names(date_filter)
    print(
        f"[pass2] removed_blank_rows most_popular={rm_mp} most_drafted_3x={rm_3x} "
        f"winning_drafts_filled={wd_changed} residual_known_filled={known_changed}"
    )
    if date_filter:
        print(f"[scope] processed dates={sorted(date_filter)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
