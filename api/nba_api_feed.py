"""Runtime nba_api data feed — enriches LightGBM features with training-aligned signals.

Fetches league-wide game logs from NBA.com (same source as train_lgbm.py) and computes
per-player features that were previously hardcoded or approximated at inference:
  - usage_share: player PPG / team avg PPG (was 0.0)
  - team_pace_proxy: team avg total pts per game (was game_total/2)
  - opp_pts_allowed: actual opponent defensive rating (was spread-derived heuristic)
  - recent_3g_pts: rolling 3-game avg points (was approximation in l3_vs_l5)
  - min_volatility: rolling 5-game std of minutes (was |recent-season| proxy)
  - games_played: actual cumulative count (was estimate)

Cached per ET date (4-hour TTL). Graceful fallback: callers get empty dict on any failure
and continue using ESPN-based features.

grep: NBA API FEED
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from nba_api.stats.endpoints import playergamelogs
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


CACHE_DIR = Path(os.environ.get("NBA_CACHE_DIR", "/tmp/nba_cache_v19"))
_CACHE_TTL = 14400  # 4 hours


def _normalize_name(name: str) -> str:
    n = str(name).lower().strip()
    n = re.sub(r"['\.\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"nba_api_{key}.json"


def _read_cache(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if time.time() - data.get("_ts", 0) > _CACHE_TTL:
            return None
        return data
    except Exception:
        return None


def _write_cache(key: str, data: dict):
    try:
        data["_ts"] = time.time()
        _cache_path(key).write_text(json.dumps(data, default=str))
    except Exception:
        pass


def _current_season() -> str:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    year, month = now.year, now.month
    if month >= 10:
        return f"{year}-{str(year + 1)[-2:]}"
    return f"{year - 1}-{str(year)[-2:]}"


def _fetch_season_logs():
    """Fetch full season game logs via nba_api. Returns a pandas DataFrame or None."""
    if not HAS_NBA_API or not HAS_PANDAS:
        return None
    season = _current_season()
    try:
        logs = playergamelogs.PlayerGameLogs(season_nullable=season, timeout=90)
        df = logs.get_data_frames()[0]
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        print(f"[nba-api-feed] PlayerGameLogs failed: {e}")
        return None


def prefetch_enrichment(date_str: str = None) -> dict:
    """Pre-fetch and cache nba_api enrichment for the current slate.

    Call once before parallel _run_game calls. Returns the full enrichment
    dict {norm_name: features} so callers can also use it directly.
    """
    if not HAS_NBA_API or not HAS_PANDAS:
        return {}

    ck = f"enrichment_{date_str or 'today'}"
    cached = _read_cache(ck)
    if cached and "players" in cached:
        print(f"[nba-api-feed] cache hit: {len(cached['players'])} players, {len(cached.get('teams', {}))} teams")
        return cached

    df = _fetch_season_logs()
    if df is None:
        return {}

    try:
        players, teams = _compute_all_features(df)
        result = {"players": players, "teams": teams}
        _write_cache(ck, result)
        print(f"[nba-api-feed] computed features: {len(players)} players, {len(teams)} teams")
        return result
    except Exception as e:
        print(f"[nba-api-feed] feature computation failed: {e}")
        return {}


def get_player_enrichment(date_str: str = None) -> dict:
    """Get cached per-player enrichment. Returns {norm_name: feature_dict}."""
    ck = f"enrichment_{date_str or 'today'}"
    cached = _read_cache(ck)
    if cached and "players" in cached:
        return cached["players"]
    return {}


def get_team_enrichment(date_str: str = None) -> dict:
    """Get cached per-team enrichment. Returns {team_abbr: feature_dict}."""
    ck = f"enrichment_{date_str or 'today'}"
    cached = _read_cache(ck)
    if cached and "teams" in cached:
        return cached["teams"]
    return {}


def enrich_stats_map(stats_map: dict, player_names: dict, date_str: str = None) -> int:
    """Merge nba_api features into an existing stats_map in-place.

    Args:
        stats_map: {player_id: stats_dict} from ESPN
        player_names: {player_id: player_name} for name-based lookup
        date_str: ET date string for cache key

    Returns: count of players enriched.
    """
    player_data = get_player_enrichment(date_str)
    team_data = get_team_enrichment(date_str)
    if not player_data:
        return 0

    enriched = 0
    for pid, stats in stats_map.items():
        if pid == "_cached_ts":
            continue
        name = player_names.get(str(pid), "")
        norm = _normalize_name(name) if name else ""
        pf = player_data.get(norm)
        if pf:
            stats["_nba_api_usage_share"] = pf.get("usage_share", 0.0)
            stats["_nba_api_team_pace"] = pf.get("team_pace_proxy", 0.0)
            stats["_nba_api_recent_3g_pts"] = pf.get("recent_3g_pts", 0.0)
            stats["_nba_api_min_volatility"] = pf.get("min_volatility", 0.0)
            stats["_nba_api_games_played"] = pf.get("games_played", 0)
            enriched += 1

        team_abbr = stats.get("_nba_api_team") or ""
        if not team_abbr:
            nba_team = pf.get("nba_api_team", "") if pf else ""
            if nba_team:
                stats["_nba_api_team"] = nba_team
                team_abbr = nba_team

        tf = team_data.get(team_abbr, {})
        if tf:
            stats["_nba_api_opp_pts_allowed"] = tf.get("pts_allowed", 0.0)
            stats["_nba_api_team_avg_pts"] = tf.get("avg_pts", 0.0)

    return enriched


def _compute_all_features(df) -> tuple:
    """Compute per-player and per-team features from season game logs.

    Aligns with train_lgbm.py feature engineering:
    - Causal (shift not needed — we compute features from all past games,
      and the target game hasn't happened yet at inference time)
    - Same rolling windows (5-game for min/pts, 3-game for l3_vs_l5)
    - Same formulas via numpy
    """
    df = df.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"].astype(str).str[:10])
    df = df.sort_values(by=["PLAYER_ID", "GAME_DATE"])
    df["norm_name"] = df["PLAYER_NAME"].apply(_normalize_name)

    g = df.groupby("PLAYER_ID")

    df["avg_min"] = g["MIN"].transform(lambda x: x.expanding().mean())
    df["avg_pts"] = g["PTS"].transform(lambda x: x.expanding().mean())
    df["recent_3g_pts"] = g["PTS"].transform(lambda x: x.rolling(3, min_periods=1).mean())
    df["min_roll5_std"] = g["MIN"].transform(lambda x: x.rolling(5, min_periods=2).std())
    df["games_played"] = g.cumcount() + 1

    df["min_volatility"] = np.where(
        df["avg_min"] > 0,
        df["min_roll5_std"].fillna(0) / (df["avg_min"] + 0.5),
        0.0,
    ).clip(0.0, 1.2)

    # --- Team-level metrics ---
    # Team game total points (per game)
    _tg = (
        df.groupby(["TEAM_ABBREVIATION", "GAME_DATE"])["PTS"]
        .sum()
        .reset_index(name="team_pts_game")
    )
    team_avg_pts = _tg.groupby("TEAM_ABBREVIATION")["team_pts_game"].mean().to_dict()

    # Usage share: player expanding avg pts / team expanding avg pts
    team_expanding = (
        _tg.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
        .groupby("TEAM_ABBREVIATION")["team_pts_game"]
        .transform(lambda x: x.expanding().mean())
    )
    _tg["team_expanding_avg"] = team_expanding
    team_game_avg = _tg.set_index(["TEAM_ABBREVIATION", "GAME_DATE"])["team_expanding_avg"].to_dict()

    df["_team_game_key"] = list(zip(df["TEAM_ABBREVIATION"], df["GAME_DATE"]))
    df["_team_exp_avg"] = df["_team_game_key"].map(team_game_avg).fillna(110.0)
    df["usage_share"] = np.where(df["_team_exp_avg"] > 0, df["avg_pts"] / df["_team_exp_avg"], 0.0)

    # Opponent points allowed
    if "MATCHUP" in df.columns and "TEAM_ABBREVIATION" in df.columns:
        def _parse_opp(mu, team_abbr):
            mu = str(mu)
            if " vs. " in mu:
                parts = mu.split(" vs. ", 1)
                return parts[1].strip() if parts[0].strip() == team_abbr else parts[0].strip()
            if " @ " in mu:
                parts = mu.split(" @ ", 1)
                return parts[1].strip() if parts[0].strip() == team_abbr else parts[0].strip()
            return ""

        df["OPP_TEAM"] = [
            _parse_opp(m, t) for m, t in zip(df["MATCHUP"], df["TEAM_ABBREVIATION"])
        ]
    else:
        df["OPP_TEAM"] = ""

    # Opponent points allowed = avg points scored by teams playing AGAINST this team
    _opp_game = (
        df.groupby(["OPP_TEAM", "GAME_DATE"])["PTS"]
        .sum()
        .reset_index(name="opp_pts_scored")
    )
    opp_pts_allowed = _opp_game.groupby("OPP_TEAM")["opp_pts_scored"].mean().to_dict()

    # Take most recent row per player for feature snapshot
    latest = df.groupby("PLAYER_ID").tail(1).copy()

    players = {}
    for _, row in latest.iterrows():
        name = row["norm_name"]
        team = row["TEAM_ABBREVIATION"]
        players[name] = {
            "usage_share": round(float(row.get("usage_share", 0)), 4),
            "team_pace_proxy": round(float(team_avg_pts.get(team, 110.0)), 1),
            "recent_3g_pts": round(float(row.get("recent_3g_pts", 0)), 1),
            "min_volatility": round(float(row.get("min_volatility", 0)), 4),
            "games_played": int(row.get("games_played", 0)),
            "nba_api_team": team,
        }

    teams = {}
    all_teams = set(team_avg_pts.keys()) | set(opp_pts_allowed.keys())
    for team in all_teams:
        teams[team] = {
            "avg_pts": round(team_avg_pts.get(team, 110.0), 1),
            "pace_proxy": round(team_avg_pts.get(team, 110.0), 1),
            "pts_allowed": round(opp_pts_allowed.get(team, 110.0), 1),
        }

    return players, teams
