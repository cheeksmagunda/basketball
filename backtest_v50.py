"""
Backtest v50: scoring-dominant DFS weights + star anchor constraint.
Uses player stats from prediction CSVs, re-runs _build_lineups() with current model.
Mar 5–19 backtest — shows what the updated model would have picked each day.
"""
import csv
import sys
import json
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import api.index as idx

DATES = [
    "2026-03-05", "2026-03-06", "2026-03-07", "2026-03-08", "2026-03-09",
    "2026-03-10", "2026-03-11", "2026-03-12", "2026-03-13", "2026-03-14",
    "2026-03-15", "2026-03-16", "2026-03-17", "2026-03-18", "2026-03-19",
]

SLOT_ORDER = {"2.0x": 1, "1.8x": 2, "1.6x": 3, "1.4x": 4, "1.2x": 5}


def load_projections_from_csv(date_str):
    """Load player projections from prediction CSV. Returns list of player dicts."""
    csv_path = Path(f"data/predictions/{date_str}.csv")
    if not csv_path.exists():
        return []

    seen = set()
    players = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            name = row.get("player_name", "")
            if not name or name in seen:
                continue
            seen.add(name)

            pts = float(row.get("pts", 0) or 0)
            reb = float(row.get("reb", 0) or 0)
            ast = float(row.get("ast", 0) or 0)
            stl = float(row.get("stl", 0) or 0)
            blk = float(row.get("blk", 0) or 0)
            pred_min = float(row.get("pred_min", 0) or 0)
            boost = float(row.get("est_card_boost", 0) or 0)
            orig_rs = float(row.get("predicted_rs", 0) or 0)

            # Re-compute RS with new DFS weights using the live formula
            dfs = idx._dfs_score(pts, reb, ast, stl, blk, 0)
            comp_div = idx._cfg("real_score.compression_divisor", 4.5)
            comp_pow = idx._cfg("real_score.compression_power", 0.78)
            rs_cap   = idx._cfg("real_score.rs_cap", 20.0)
            rs = min((dfs / comp_div) ** comp_pow, rs_cap)

            # Season proxies: use projected stats as season averages (backtest approximation)
            season_pts = pts
            season_min = pred_min * 0.9
            recent_min = pred_min * 0.95

            players.append({
                "name":         name,
                "id":           row.get("player_id", name),
                "team":         row.get("team", ""),
                "pos":          row.get("pos", ""),
                "rating":       round(rs, 3),
                "est_mult":     boost,
                "predMin":      pred_min,
                "pts":          pts,
                "reb":          reb,
                "ast":          ast,
                "stl":          stl,
                "blk":          blk,
                "season_pts":   season_pts,
                "season_min":   season_min,
                "recent_min":   recent_min,
                "player_variance": 0.3,
                "injury_status": "",
                "opp":          "",
                "_cascade_bonus": 0,
                "chalk_ev":     round(rs * (1.6 + boost), 2),
            })

    return players


def fmt_lineup(lineup):
    return sorted(lineup, key=lambda p: SLOT_ORDER.get(p.get("slot", "1.0x"), 9))


# ── Main backtest loop ──────────────────────────────────────────────────────
config = idx._load_config()
print(f"\n{'='*72}")
print(f"  V50 BACKTEST — Mar 5–19, 2026 (scoring-dominant weights + star anchor)")
print(f"  Config v{config.get('version','?')}  |  DFS pts={idx._cfg('real_score.dfs_weights',{}).get('pts','?')}")
print(f"{'='*72}\n")

for date_str in DATES:
    target = datetime.date.fromisoformat(date_str)
    orig_et = idx._et_date
    idx._et_date = lambda d=target: d

    projections = load_projections_from_csv(date_str)
    if not projections:
        print(f"  {date_str}: no CSV data\n")
        idx._et_date = orig_et
        continue

    try:
        chalk, upside, _pool = idx._build_lineups(projections)
    except Exception as e:
        print(f"  {date_str}: ERROR — {e}\n")
        idx._et_date = orig_et
        continue

    idx._et_date = orig_et

    print(f"{'─'*72}")
    print(f"  {date_str}  ({len(projections)} players)")
    print()

    print("  STARTING 5:")
    if chalk:
        for p in fmt_lineup(chalk):
            star = " ★" if p.get("_is_star_anchor") else ""
            print(f"    {p.get('slot','?'):5s}  {p['name']:<26s} {p.get('team',''):4s}  "
                  f"RS:{p.get('rating',0):4.1f}  Pts:{p.get('pts',0):5.1f}  "
                  f"Boost:+{p.get('est_mult',0):.1f}x{star}")
    else:
        print("    (none)")

    print()
    print("  MOONSHOT:")
    if upside:
        for p in fmt_lineup(upside):
            star = " ★" if p.get("_is_star_anchor") else ""
            print(f"    {p.get('slot','?'):5s}  {p['name']:<26s} {p.get('team',''):4s}  "
                  f"RS:{p.get('rating',0):4.1f}  Pts:{p.get('pts',0):5.1f}  "
                  f"Boost:+{p.get('est_mult',0):.1f}x{star}")
    else:
        print("    (none)")
    print()

print(f"{'='*72}\n")
