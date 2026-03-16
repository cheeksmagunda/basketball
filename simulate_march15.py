"""
Simulate slate-wide Starting 5 + Moonshot for Sunday, March 15, 2026.

Approach: Load the cached ESPN game projections from data/slate/2026-03-15_games.json
(player projections already computed: LightGBM + Monte Carlo RS + card boost),
then re-run _build_lineups() with the CURRENT model-config and code.

This gives you: "what would today's model pick, given March 15 games" — no ESPN
calls, no log/actuals touched, nothing written or committed.
"""
import sys
import os
import json
import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

TARGET_DATE = datetime.date(2026, 3, 15)

# Patch _et_date before anything runs so config/cache keys use March 15
import api.index as idx
idx._et_date = lambda: TARGET_DATE

# Load current model config (GitHub fallback → defaults if no token)
config = idx._load_config()

# ── Load cached March 15 game projections ─────────────────────────────────────
games_file = Path("data/slate/2026-03-15_games.json")
slate_file = Path("data/slate/2026-03-15_slate.json")

if not games_file.exists():
    print("ERROR: data/slate/2026-03-15_games.json not found.")
    sys.exit(1)

games_data = json.loads(games_file.read_text())   # {gameId: [players...]}
slate_meta = json.loads(slate_file.read_text()) if slate_file.exists() else {}

# Flatten all projections across all games
all_proj = []
for game_id, players in games_data.items():
    for p in players:
        # Re-derive card boost fields if missing (est_mult is already stored)
        all_proj.append(p)

# Pull game metadata for display
games_meta = {g["gameId"]: g for g in slate_meta.get("games", [])}

print(f"\n{'='*66}")
print(f"  MARCH 15, 2026 — SLATE SIMULATION (current model + config)")
print(f"{'='*66}")
print(f"  Games: {len(games_data)}    Players projected: {len(all_proj)}")
print(f"  Config version: {config.get('version', '?')}")
print()

# Show the slate games
print("  GAMES ON SLATE:")
for gid, players in games_data.items():
    meta = games_meta.get(gid, {})
    label = meta.get("label", f"Game {gid}")
    spread = meta.get("spread")
    total  = meta.get("total")
    spread_str = f"  spread={spread}" if spread is not None else ""
    total_str  = f"  total={total}"   if total  is not None else ""
    print(f"    {label:22s}{spread_str}{total_str}")
print()

# ── Re-run _build_lineups with current model ──────────────────────────────────
chalk, upside = idx._build_lineups(all_proj)

SLOT_ORDER = {"2.0x": 1, "1.8x": 2, "1.6x": 3, "1.4x": 4, "1.2x": 5}

def fmt_lineup(lineup):
    return sorted(lineup, key=lambda p: SLOT_ORDER.get(p.get("slot","1.0x"), 9))

# ── Starting 5 ────────────────────────────────────────────────────────────────
print(f"{'─'*66}")
print(f"  STARTING 5  (Chalk — MILP optimized, capped card boost)")
print(f"{'─'*66}")
if chalk:
    chalk_sorted = fmt_lineup(chalk)
    for p in chalk_sorted:
        slot     = p.get("slot", "?")
        name     = p["name"]
        team     = p.get("team", "")
        rs       = p.get("rating", 0)
        mins     = p.get("predMin", 0)
        boost    = p.get("est_mult", 0)
        pts      = p.get("pts", 0)
        reb      = p.get("reb", 0)
        ast      = p.get("ast", 0)
        s_min    = p.get("season_min", 0)
        chalk_ev = p.get("chalk_ev", 0)
        inj      = f"  [{p['injury_status']}]" if p.get("injury_status") else ""
        print(f"  {slot:5s}  {name:<26s} {team:4s}  "
              f"RS:{rs:4.1f}  Min:{mins:4.1f}  Boost:{boost:4.2f}  "
              f"Proj: {pts:.1f}pts/{reb:.1f}reb/{ast:.1f}ast{inj}")
    # Score range
    sb = slate_meta.get("score_bounds", {})
    if sb.get("chalk"):
        lo = sb["chalk"].get("low_total", "?")
        hi = sb["chalk"].get("high_total", "?")
        print(f"\n  Score range: {lo} – {hi} RS")
else:
    print("  (no chalk lineup generated)")

# ── Moonshot ──────────────────────────────────────────────────────────────────
print(f"\n{'─'*66}")
print(f"  MOONSHOT  (Contrarian — min card boost ≥1.5, low-ownership plays)")
print(f"{'─'*66}")
if upside:
    upside_sorted = fmt_lineup(upside)
    for p in upside_sorted:
        slot     = p.get("slot", "?")
        name     = p["name"]
        team     = p.get("team", "")
        rs       = p.get("rating", 0)
        mins     = p.get("predMin", 0)
        boost    = p.get("est_mult", 0)
        pts      = p.get("pts", 0)
        reb      = p.get("reb", 0)
        ast      = p.get("ast", 0)
        s_min    = p.get("season_min", 0)
        inj      = f"  [{p['injury_status']}]" if p.get("injury_status") else ""
        print(f"  {slot:5s}  {name:<26s} {team:4s}  "
              f"RS:{rs:4.1f}  Min:{mins:4.1f}  Boost:{boost:4.2f}  "
              f"Proj: {pts:.1f}pts/{reb:.1f}reb/{ast:.1f}ast{inj}")
    sb = slate_meta.get("score_bounds", {})
    if sb.get("upside"):
        lo = sb["upside"].get("low_total", "?")
        hi = sb["upside"].get("high_total", "?")
        print(f"\n  Score range: {lo} – {hi} RS")
else:
    print("  (no moonshot lineup generated)")

# ── Full eligible pool (top 25 by chalk EV) ──────────────────────────────────
print(f"\n{'─'*66}")
print(f"  FULL PROJECTION POOL  (top 25 by chalk EV — all games)")
print(f"{'─'*66}")
pool_sorted = sorted(all_proj, key=lambda p: p.get("chalk_ev", 0), reverse=True)[:25]
chalk_names  = {p["name"] for p in chalk}
upside_names = {p["name"] for p in upside}

for p in pool_sorted:
    name    = p["name"]
    team    = p.get("team", "")
    rs      = p.get("rating", 0)
    mins    = p.get("predMin", 0)
    boost   = p.get("est_mult", 0)
    chalk_ev = p.get("chalk_ev", 0)
    s_min   = p.get("season_min", 0)
    r_min   = p.get("recent_min", 0)
    tag = " ◀ S5" if name in chalk_names else (" ◀ MS" if name in upside_names else "")
    inj  = f" [{p['injury_status']}]" if p.get("injury_status") else ""
    print(f"  {name:<28s} {team:4s}  RS:{rs:4.1f}  "
          f"Min:{mins:4.1f}(s{s_min:.0f}/r{r_min:.0f})  "
          f"Boost:{boost:4.2f}  cEV:{chalk_ev:5.2f}{inj}{tag}")

print(f"\n{'='*66}\n")
