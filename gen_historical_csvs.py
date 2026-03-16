"""
gen_historical_csvs.py

Generate prediction CSVs for dates before March 5, 2026 using nba_api.
Run this LOCALLY where you have internet access to stats.nba.com.

Usage:
    python gen_historical_csvs.py                  # all dates Feb 7 – Mar 4
    python gen_historical_csvs.py 2026-02-07       # single date
    python gen_historical_csvs.py 2026-02-07 2026-02-11 2026-02-19  # specific dates

Output: data/predictions/{date}.csv — same format as live pipeline CSVs.
Then run: python simulate_v20.py
"""

import os
import sys
import csv
import math
import time
import pickle
import numpy as np
from datetime import datetime, date, timedelta
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# nba_api imports (pip install nba_api)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from nba_api.stats.endpoints import (
        scoreboardv3,
        leaguedashplayerstats,
        playergamelogs,
    )
    from nba_api.stats.static import teams as nba_teams_static
except ImportError:
    print("[ERROR] nba_api not installed. Run: pip install nba_api")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — mirrors simulate_v20.py / api/index.py log-formula values
# ─────────────────────────────────────────────────────────────────────────────
LOG_A = 4.2
LOG_B = 1.1
OWNERSHIP_SCALAR = 80.0
FAME_PTS_BASE = 14.0
FAME_EXPONENT = 2.5
BIG_MARKET_TEAMS = {"LAL", "GS", "GSW", "BOS", "NY", "NYK", "PHI", "MIA", "DEN", "LAC", "CHI"}
BOOST_CEILING = 3.0
BOOST_FLOOR = 0.2

USAGE_TREND_MIN, USAGE_TREND_MAX = 0.90, 1.50
MIN_MINUTES_GATE = 8.0   # players averaging fewer minutes are skipped
MIN_SEASON_GAMES = 3     # skip players with < 3 games (too small a sample)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data", "predictions")

# nba_api team abbreviations → our system's abbreviations
# (nba_api is the canonical source; our system has some quirks inherited from ESPN)
NBA_API_ABBR_MAP = {
    "NOP": "NO",    # Pelicans: nba_api uses NOP, our system uses NO
    "UTA": "UTAH",  # Jazz: nba_api uses UTA, our system uses UTAH
    "SAS": "SA",    # Spurs: nba_api uses SAS, our system uses SA
    "NYK": "NY",    # Knicks: nba_api uses NYK, our system uses NY
    # All others match: GSW, LAL, BOS, DEN, MEM, SAC, PHI, OKC, etc.
}

# Position normalization: nba_api returns G, F, C, G-F, F-C, F-G, etc.
# We want: G, F, C, PF, SF — keep primary position only
def normalize_pos(raw_pos: str) -> str:
    if not raw_pos:
        return "G"
    p = raw_pos.strip().upper()
    # Forward-center combos → C (rebounders, bigs)
    if p in ("C", "C-F", "F-C"):
        return "C"
    # Power forward → PF (treated as F in optimizer)
    if p in ("PF", "PF-C", "C-PF"):
        return "PF"
    # Small forward → SF (treated as F)
    if p in ("SF", "SF-PF", "PF-SF", "F", "F-G", "G-F", "SF-SG"):
        return "F"
    # Guard combos → G
    return "G"


def abbr(nba_api_abbr: str) -> str:
    """Convert nba_api team abbreviation to our system's abbreviation."""
    return NBA_API_ABBR_MAP.get(nba_api_abbr, nba_api_abbr)


def card_boost(pts: float, pred_min: float, team: str) -> float:
    """Log-formula card boost — mirrors simulate_v20.py."""
    fame_mult = max(1.0, (pts / FAME_PTS_BASE) ** FAME_EXPONENT)
    big_market = 1.3 if team in BIG_MARKET_TEAMS else 1.0
    drafts = OWNERSHIP_SCALAR * (pts / 10) ** 2 * (pred_min / 30) ** 0.5 * fame_mult * big_market
    drafts = max(drafts, 1.0)
    raw = LOG_A - LOG_B * math.log10(drafts)
    return round(max(BOOST_FLOOR, min(BOOST_CEILING, raw)), 1)


def load_lgbm():
    """Load lgbm_model.pkl from repo root. Returns (model, feature_names) or (None, None)."""
    pkl_path = os.path.join(os.path.dirname(__file__), "lgbm_model.pkl")
    if not os.path.exists(pkl_path):
        print("  [WARN] lgbm_model.pkl not found — will use heuristic RS formula")
        return None, None
    try:
        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        features = bundle["features"]
        print(f"  Loaded LightGBM model ({len(features)} features: {features})")
        return model, features
    except Exception as e:
        print(f"  [WARN] Could not load lgbm_model.pkl: {e} — using heuristic")
        return None, None


def heuristic_rs(pts, reb, ast, stl, blk, tov, avg_min) -> float:
    """Fallback RS estimate when LightGBM unavailable."""
    dfs = pts * 1.0 + reb * 1.0 + ast * 1.5 + stl * 4.5 + blk * 4.0 - tov * 1.2
    pts_per_min = pts / max(avg_min, 1)
    ast_per_min = ast / max(avg_min, 1)
    def_per_min = (stl + blk) / max(avg_min, 1)
    clutch = 1.0 + min(pts_per_min * 0.10 + ast_per_min * 0.18 + def_per_min * 0.25, 0.40)
    return max(0.5, round(dfs * clutch * 0.18, 2))


def lgbm_rs(model, features, avg_min, avg_pts, avg_reb, avg_ast, avg_stl, avg_blk,
            home_away, recent_vs_season, games_played) -> float:
    """Run LightGBM inference — mirrors api/index.py project_player()."""
    usage = float(np.clip(avg_pts / max(avg_min, 1) * 0.8, USAGE_TREND_MIN, USAGE_TREND_MAX))
    opp_def_rating = 112.0  # neutral (no spread data for historical)
    ast_rate = avg_ast / max(avg_min, 1)
    def_rate = (avg_stl + avg_blk) / max(avg_min, 1)
    pts_per_min = avg_pts / max(avg_min, 1)
    rest_days = 2.0  # typical schedule
    rvs = float(np.clip(recent_vs_season, 0.5, 2.0))
    reb_per_min = float(np.clip(avg_reb / max(avg_min, 1), 0.0, 1.5))

    feat_vec = [avg_min, avg_pts, usage, opp_def_rating, home_away,
                ast_rate, def_rate, pts_per_min, rest_days, rvs,
                games_played, reb_per_min]

    # Verify feature vector length vs model expectation
    if features and len(feat_vec) != len(features):
        print(f"  [WARN] Feature length mismatch: got {len(feat_vec)}, model expects {len(features)}")
        # Pad or trim to match
        while len(feat_vec) < len(features):
            feat_vec.append(1.0)
        feat_vec = feat_vec[:len(features)]

    X = np.array(feat_vec, dtype=float).reshape(1, -1)
    pred = float(model.predict(X)[0])
    return max(0.5, round(pred, 2))


def get_games_on_date(date_str: str) -> list[dict]:
    """
    Return list of {home: abbr, away: abbr, home_id: int, away_id: int}
    for all NBA regular-season games on date_str (YYYY-MM-DD).
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    fmt = dt.strftime("%m/%d/%Y")
    try:
        sb = scoreboardv3.ScoreboardV3(game_date=fmt)
        df = sb.get_data_frames()[0]  # GamesHeader
    except Exception as e:
        print(f"  [ERROR] ScoreboardV3 failed for {date_str}: {e}")
        return []

    # Build team_id → abbr lookup
    all_teams = nba_teams_static.get_teams()
    id_to_abbr = {t["id"]: t["abbreviation"] for t in all_teams}

    games = []
    for _, row in df.iterrows():
        home_id = int(row["HOME_TEAM_ID"])
        away_id = int(row["VISITOR_TEAM_ID"])
        home_abbr = abbr(id_to_abbr.get(home_id, str(home_id)))
        away_abbr = abbr(id_to_abbr.get(away_id, str(away_id)))
        games.append({
            "home": home_abbr,
            "away": away_abbr,
            "home_id": home_id,
            "away_id": away_id,
        })
    return games


def get_season_stats_as_of(date_str: str) -> dict:
    """
    Pull NBA season stats per game BEFORE the given date.
    date_str is the GAME DATE (YYYY-MM-DD).
    We use date_to = one day before to exclude same-day games.

    Returns: {player_id_str: {name, team, pos, gp, avg_min, avg_pts, avg_reb,
                               avg_ast, avg_stl, avg_blk, avg_tov}}
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    date_to = dt.strftime("%m/%d/%Y")

    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season="2025-26",
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
            date_to_nullable=date_to,
        )
        df = stats.get_data_frames()[0]
    except Exception as e:
        print(f"  [ERROR] LeagueDashPlayerStats failed for {date_str}: {e}")
        return {}

    result = {}
    for _, row in df.iterrows():
        pid = str(int(row["PLAYER_ID"]))
        gp = int(row.get("GP", 0))
        if gp < MIN_SEASON_GAMES:
            continue
        team_abbr = abbr(str(row.get("TEAM_ABBREVIATION", "")))
        # PLAYER_POSITION not always in this endpoint — will be enriched below
        result[pid] = {
            "name": str(row["PLAYER_NAME"]),
            "team": team_abbr,
            "pos": str(row.get("PLAYER_POSITION", "")),  # may be empty
            "gp": gp,
            "avg_min": float(row.get("MIN", 0)),
            "avg_pts": float(row.get("PTS", 0)),
            "avg_reb": float(row.get("REB", 0)),
            "avg_ast": float(row.get("AST", 0)),
            "avg_stl": float(row.get("STL", 0)),
            "avg_blk": float(row.get("BLK", 0)),
            "avg_tov": float(row.get("TOV", 0)),
        }
    return result


def get_recent_scoring(date_str: str, player_ids: set) -> dict:
    """
    Get rolling 5-game average PTS for players to compute recent_vs_season.
    Uses PlayerGameLogs filtered to last 5 games before date_str.

    Returns: {player_id_str: recent_5g_avg_pts}
    """
    dt = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    date_to = dt.strftime("%m/%d/%Y")
    # Start from 30 days back to get ~5 games
    date_from_dt = dt - timedelta(days=30)
    date_from = date_from_dt.strftime("%m/%d/%Y")

    try:
        logs = playergamelogs.PlayerGameLogs(
            season_nullable="2025-26",
            date_from_nullable=date_from,
            date_to_nullable=date_to,
            per_mode_simple="PerGame",
        )
        df = logs.get_data_frames()[0]
    except Exception as e:
        print(f"  [WARN] PlayerGameLogs failed (recent scoring): {e}")
        return {}

    result = {}
    for pid_int, group in df.groupby("PLAYER_ID"):
        pid = str(int(pid_int))
        if pid not in player_ids:
            continue
        last5 = group.sort_values("GAME_DATE", ascending=False).head(5)
        if len(last5) > 0:
            result[pid] = float(last5["PTS"].mean())
    return result


def get_positions(date_str: str, player_ids: set) -> dict:
    """
    Get player positions via PlayeGameLogs (has PLAYER_POSITION in newer nba_api versions).
    Falls back to empty string if unavailable.

    Returns: {player_id_str: position_str}
    """
    # Try getting from the last 30 days of game logs (which includes position in some API versions)
    dt = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    date_to = dt.strftime("%m/%d/%Y")
    date_from = (dt - timedelta(days=30)).strftime("%m/%d/%Y")

    try:
        logs = playergamelogs.PlayerGameLogs(
            season_nullable="2025-26",
            date_from_nullable=date_from,
            date_to_nullable=date_to,
        )
        df = logs.get_data_frames()[0]
        if "POSITION" in df.columns:
            return {str(int(row["PLAYER_ID"])): str(row["POSITION"])
                    for _, row in df.drop_duplicates("PLAYER_ID").iterrows()
                    if str(int(row["PLAYER_ID"])) in player_ids}
    except Exception as e:
        print(f"  [WARN] Position fetch failed: {e}")

    return {}


def generate_csv_for_date(date_str: str, model, features, dry_run: bool = False) -> bool:
    """
    Generate a prediction CSV for date_str.
    Returns True on success, False if no games found.
    """
    out_path = os.path.join(OUTPUT_DIR, f"{date_str}.csv")
    if os.path.exists(out_path):
        print(f"  [SKIP] {date_str}.csv already exists — delete it to regenerate")
        return True

    print(f"\n{'─'*60}")
    print(f"  Generating {date_str}...")

    # Step 1: Get games on this date
    games = get_games_on_date(date_str)
    time.sleep(0.8)  # rate limit
    if not games:
        print(f"  [SKIP] No games found for {date_str} (All-Star break or off day)")
        return False

    teams_playing = set()
    team_side = {}  # abbr -> 'home' | 'away'
    for g in games:
        teams_playing.add(g["home"])
        teams_playing.add(g["away"])
        team_side[g["home"]] = "home"
        team_side[g["away"]] = "away"

    print(f"  Games ({len(games)}): " +
          " | ".join(f"{g['away']}@{g['home']}" for g in games))

    # Step 2: Season stats as of day before
    stats_map = get_season_stats_as_of(date_str)
    time.sleep(0.8)

    if not stats_map:
        print(f"  [ERROR] No stats returned for {date_str}")
        return False

    # Filter to players on teams playing today
    eligible = {pid: s for pid, s in stats_map.items()
                if s["team"] in teams_playing and s["avg_min"] >= MIN_MINUTES_GATE}
    print(f"  Eligible players on playing teams: {len(eligible)}")

    if not eligible:
        print(f"  [ERROR] No eligible players for {date_str}")
        return False

    # Step 3: Recent scoring for recent_vs_season feature
    player_ids = set(eligible.keys())
    recent_pts = get_recent_scoring(date_str, player_ids)
    time.sleep(0.8)

    # Step 4: Positions (if not already in stats_map)
    missing_pos = {pid for pid, s in eligible.items() if not s["pos"]}
    if missing_pos:
        pos_map = get_positions(date_str, missing_pos)
        time.sleep(0.8)
        for pid, pos in pos_map.items():
            if pid in eligible:
                eligible[pid]["pos"] = pos

    # Step 5: Build rows
    rows = []
    for pid, s in eligible.items():
        avg_min = s["avg_min"]
        avg_pts = s["avg_pts"]
        avg_reb = s["avg_reb"]
        avg_ast = s["avg_ast"]
        avg_stl = s["avg_stl"]
        avg_blk = s["avg_blk"]
        avg_tov = s["avg_tov"]
        team = s["team"]
        pos_raw = s.get("pos", "")
        pos = normalize_pos(pos_raw) if pos_raw else "G"
        gp = s.get("gp", 40)
        side = team_side.get(team, "home")
        home_away = 1.0 if side == "home" else 0.0
        games_played = float(min(gp, 82))

        # recent_vs_season: how player's last 5 games compare to season avg
        rec = recent_pts.get(pid, avg_pts)
        recent_vs_season = rec / max(avg_pts, 1.0)

        # Predicted RS
        if model is not None:
            pred_rs = lgbm_rs(model, features, avg_min, avg_pts, avg_reb, avg_ast,
                              avg_stl, avg_blk, home_away, recent_vs_season, games_played)
        else:
            pred_rs = heuristic_rs(avg_pts, avg_reb, avg_ast, avg_stl, avg_blk, avg_tov, avg_min)

        boost = card_boost(avg_pts, avg_min, team)

        rows.append({
            "scope": "slate",
            "lineup_type": "chalk",
            "slot": "2.0x",
            "player_name": s["name"],
            "player_id": pid,
            "team": team,
            "pos": pos,
            "predicted_rs": pred_rs,
            "est_card_boost": boost,
            "pred_min": round(avg_min, 1),
            "pts": round(avg_pts, 1),
            "reb": round(avg_reb, 1),
            "ast": round(avg_ast, 1),
            "stl": round(avg_stl, 2),
            "blk": round(avg_blk, 2),
        })

    if not rows:
        print(f"  [ERROR] No rows produced for {date_str}")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would write {len(rows)} players to {out_path}")
        # Print top 10 by predicted_rs
        rows.sort(key=lambda r: r["predicted_rs"], reverse=True)
        print(f"  Top 10 by predicted RS:")
        for r in rows[:10]:
            print(f"    {r['player_name']:<24} {r['team']:<5} {r['pos']:<3} "
                  f"RS={r['predicted_rs']:.1f}  boost={r['est_card_boost']:.1f}x "
                  f"min={r['pred_min']:.0f}  pts={r['pts']:.1f}")
        return True

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fieldnames = ["scope", "lineup_type", "slot", "player_name", "player_id",
                  "team", "pos", "predicted_rs", "est_card_boost", "pred_min",
                  "pts", "reb", "ast", "stl", "blk"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    rows.sort(key=lambda r: r["predicted_rs"], reverse=True)
    print(f"  Wrote {len(rows)} players → {out_path}")
    print(f"  Top 5 RS: " + ", ".join(
        f"{r['player_name'].split()[-1]}({r['predicted_rs']:.1f})" for r in rows[:5]))
    return True


def date_range(start: str, end: str) -> list[str]:
    """Generate list of YYYY-MM-DD strings from start to end inclusive."""
    d = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    result = []
    while d <= e:
        result.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return result


if __name__ == "__main__":
    # Default: all dates from Feb 7 to Mar 4 (day before our first real CSV)
    DEFAULT_START = "2026-02-07"
    DEFAULT_END = "2026-03-04"

    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]

    if args:
        target_dates = args
    else:
        target_dates = date_range(DEFAULT_START, DEFAULT_END)

    print(f"\nHistorical CSV Generator")
    print(f"{'='*60}")
    print(f"Target dates: {len(target_dates)} ({target_dates[0]} → {target_dates[-1]})")
    print(f"Output dir:   {OUTPUT_DIR}")
    if dry_run:
        print(f"Mode:         DRY RUN (no files written)")
    print()

    model, features = load_lgbm()

    success = 0
    skipped = 0
    failed = 0

    for d in target_dates:
        try:
            result = generate_csv_for_date(d, model, features, dry_run=dry_run)
            if result:
                success += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"  [ERROR] {d} failed: {e}")
            import traceback; traceback.print_exc()
            failed += 1
        # Rate limit pause between dates
        time.sleep(1.0)

    print(f"\n{'='*60}")
    print(f"Done. Success: {success}  Skipped/no games: {skipped}  Failed: {failed}")
    if success > 0 and not dry_run:
        print(f"\nNext: run  python simulate_v20.py  to compare vs leaderboard actuals")

    # Print which dates now exist vs are missing
    print(f"\nCSV status for {DEFAULT_START} → {DEFAULT_END}:")
    all_dates = date_range(DEFAULT_START, DEFAULT_END)
    for d in all_dates:
        p = os.path.join(OUTPUT_DIR, f"{d}.csv")
        status = "✓" if os.path.exists(p) else "✗"
        print(f"  {status} {d}")
