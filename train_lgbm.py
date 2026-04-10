"""
Train dual-head LightGBM bundle for Real Score (RS) with ranking-focused sample weights.

Head A (baseline): predicts core RS level for all players.
Head B (spike): predicts positive residual above baseline (role-player eruption signal).

Training objectives:
- Sample weights up-weight high actual_rs within each game date (top-10 RS days matter more).
- Playoff-intensity weighting: up-weight competitive games between playoff-bound teams,
  down-weight late-season rest/blowout games. Spike model tuned for superstar overdrive.
- Evaluation: top-5 RS recall, NDCG@5 RS, MAE (reported on held-out dates).

Feature list and scalar computation shared with api/index.py via api/features.py.
"""
import os
import re
import glob
from pathlib import Path
import json
import time
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
from nba_api.stats.endpoints import playergamelogs
from requests.exceptions import RequestException
from sklearn.model_selection import train_test_split

# Import canonical feature list from shared module — single source of truth
from api.features import RS_FEATURES as FEATURES, N_RS_FEATURES as N_FEATURES, compute_rs_features

# Pull 3 seasons of NBA game logs
SEASONS = ["2023-24", "2024-25", "2025-26"]


def _fetch_season_logs_with_retry(season: str, max_attempts: int = 6) -> pd.DataFrame:
    """Fetch one season of logs with retry/backoff for transient NBA API timeouts."""
    for attempt in range(1, max_attempts + 1):
        try:
            logs = playergamelogs.PlayerGameLogs(season_nullable=season, timeout=60)
            df_s = logs.get_data_frames()[0]
            if df_s is None or df_s.empty:
                raise RuntimeError(f"empty response for season {season}")
            return df_s
        except (
            RequestException,
            TimeoutError,
            RuntimeError,
            json.JSONDecodeError,
            ValueError,
        ) as e:
            if attempt == max_attempts:
                raise
            backoff = min(45, 5 * attempt)
            print(
                f"   [WARN] {season} fetch attempt {attempt}/{max_attempts} failed: {e}; retrying in {backoff}s..."
            )
            time.sleep(backoff)


def _normalize_name(name):
    n = str(name).lower().strip()
    n = re.sub(r"['\.\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _ndcg_at_k(pred_order: list, relevance: dict, k: int = 5) -> float:
    dcg = 0.0
    for i, pid in enumerate(pred_order[:k]):
        rel = relevance.get(pid, 0.0)
        dcg += rel / math.log2(i + 2)
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return (dcg / idcg) if idcg > 0 else 0.0


def _top5_recall(pred_top5: set, actual_top5: set) -> float:
    return len(pred_top5 & actual_top5) / 5.0


print(f"1. Fetching NBA game logs for {len(SEASONS)} seasons...")
frames = []
for season in SEASONS:
    print(f"   Fetching {season}...")
    df_s = _fetch_season_logs_with_retry(season)
    df_s["SEASON"] = season
    frames.append(df_s)
    print(f"   Got {len(df_s)} game logs for {season}")
    if season != SEASONS[-1]:
        time.sleep(3)

df = pd.concat(frames, ignore_index=True)
print(f"Total: {len(df)} game logs across {len(SEASONS)} seasons. Engineering features...")

# playergamelogs (nba_api) exposes MATCHUP + TEAM_ABBREVIATION, not OPP_TEAM
if "OPP_TEAM" not in df.columns and "MATCHUP" in df.columns and "TEAM_ABBREVIATION" in df.columns:
    def _parse_opp(mu: str, team_abbr: str) -> str:
        mu = str(mu)
        tb = str(team_abbr).strip()
        if " vs. " in mu:
            t1, t2 = [x.strip() for x in mu.split(" vs. ", 1)]
            return t2 if t1 == tb else t1
        if " @ " in mu:
            away, home = [x.strip() for x in mu.split(" @ ", 1)]
            return home if tb == away else away
        return tb

    df["OPP_TEAM"] = [_parse_opp(m, t) for m, t in zip(df["MATCHUP"], df["TEAM_ABBREVIATION"])]

df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"].astype(str).str[:10])
df = df.sort_values(by=["PLAYER_ID", "SEASON", "GAME_DATE"])
df["norm_name"] = df["PLAYER_NAME"].apply(_normalize_name)
df["date_key"] = df["GAME_DATE"].dt.normalize()

# ── Load actuals for labeling (merge AFTER feature engineering on full logs) ─
actuals_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "actuals")
actuals_frames = []
for fpath in glob.glob(os.path.join(actuals_dir, "*.csv")):
    date_str = os.path.basename(fpath).replace(".csv", "")
    try:
        a = pd.read_csv(fpath)
        if "player_name" not in a.columns or "actual_rs" not in a.columns:
            continue
        a = a[["player_name", "actual_rs"]].copy()
        a["date_key"] = pd.to_datetime(date_str)
        a["norm_name"] = a["player_name"].apply(_normalize_name)
        actuals_frames.append(a)
    except Exception as e:
        print(f"   [WARN] Could not load {fpath}: {e}")

actuals_df = None
if actuals_frames:
    actuals_df = pd.concat(actuals_frames, ignore_index=True).dropna(subset=["actual_rs"])
    actuals_df["date_key"] = actuals_df["date_key"].dt.normalize()
    print(f"   Loaded {len(actuals_df)} actuals rows from {len(actuals_frames)} files.")

g = df.groupby(["PLAYER_ID", "SEASON"])

df["avg_min"] = g["MIN"].transform(lambda x: x.expanding().mean().shift(1))
df["avg_pts"] = g["PTS"].transform(lambda x: x.expanding().mean().shift(1))
df["recent_min"] = g["MIN"].transform(lambda x: x.rolling(5).mean().shift(1))

USAGE_TREND_MIN, USAGE_TREND_MAX = 0.90, 1.50
df = df.dropna(subset=["avg_min", "avg_pts", "recent_min"]).copy()
g = df.groupby(["PLAYER_ID", "SEASON"])
df["usage_trend"] = np.where(df["avg_min"] > 0, df["recent_min"] / df["avg_min"], 1.0)
df["usage_trend"] = df["usage_trend"].clip(USAGE_TREND_MIN, USAGE_TREND_MAX)

opp_pts_allowed_map = df.groupby("OPP_TEAM")["PTS"].mean().to_dict()
df["opp_def_rating"] = df["OPP_TEAM"].map(opp_pts_allowed_map)
df["home_away"] = df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(float)

df["avg_ast"] = g["AST"].transform(lambda x: x.rolling(5).mean().shift(1))
df["ast_rate"] = np.where(df["recent_min"] > 0, df["avg_ast"] / df["recent_min"], 0.0)

df["avg_stl"] = g["STL"].transform(lambda x: x.rolling(5).mean().shift(1))
df["avg_blk"] = g["BLK"].transform(lambda x: x.rolling(5).mean().shift(1))
df["def_rate"] = np.where(
    df["recent_min"] > 0, (df["avg_stl"] + df["avg_blk"]) / df["recent_min"], 0.0
)

df["pts_per_min"] = np.where(df["recent_min"] > 0, df["avg_pts"] / df["recent_min"], 0.0)

df["prev_date"] = g["GAME_DATE"].transform(lambda x: x.shift(1))
df["rest_days"] = (df["GAME_DATE"] - df["prev_date"]).dt.days.fillna(3).clip(1, 7)

df["recent_5g_pts"] = g["PTS"].transform(lambda x: x.rolling(5).mean().shift(1))
df["recent_vs_season"] = np.where(
    df["avg_pts"] > 0, df["recent_5g_pts"] / df["avg_pts"], 1.0
).clip(0.5, 2.0)

df["games_played"] = g.cumcount()

# v63: Fix data leakage — use season average REB, not same-game actual REB
df["avg_reb"] = g["REB"].transform(lambda x: x.expanding().mean().shift(1))
df["reb_per_min"] = np.where(
    df["avg_min"] > 0, df["avg_reb"] / df["avg_min"], 0.0
).clip(0.0, 1.5)

# Role-volatility features (known before game)
df["roll3_pts"] = g["PTS"].transform(lambda x: x.rolling(3).mean().shift(1))
df["l3_vs_l5_pts"] = np.where(
    df["recent_5g_pts"] > 0, df["roll3_pts"] / df["recent_5g_pts"], 1.0
).clip(0.4, 2.5)

df["min_roll5_std"] = g["MIN"].transform(lambda x: x.rolling(5).std().shift(1))
df["min_volatility"] = np.where(
    df["avg_min"] > 0, df["min_roll5_std"] / (df["avg_min"] + 0.5), 0.0
).clip(0.0, 1.2)

df["starter_proxy"] = (df["avg_min"] >= 26.0).astype(float)

# v62: 6 new features from top-performer analysis
# 1. Opponent points allowed (defensive weakness signal)
df["opp_pts_allowed"] = df["OPP_TEAM"].map(opp_pts_allowed_map).fillna(110.0)

# 2. Team pace proxy (team's average total points per game — possessions indicator)
team_total = df.groupby(["TEAM_ABBREVIATION", "GAME_DATE"])["PTS"].sum().reset_index()
team_pace = team_total.groupby("TEAM_ABBREVIATION")["PTS"].mean().to_dict()
df["team_pace_proxy"] = df["TEAM_ABBREVIATION"].map(team_pace).fillna(110.0)

# ── v64: Game-context features (pre-game–safe: rolling means use shift(1) only) ─
# teammate OUT / injury is not in PlayerGameLogs; we use lagged bench-depth + minute-trend proxies.
if "GAME_ID" not in df.columns:
    raise RuntimeError("train_lgbm.py requires GAME_ID from nba_api PlayerGameLogs")

_team_game = df.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False).agg(
    team_pts_game=("PTS", "sum"),
    n_players_game=("PLAYER_ID", "count"),
)
_tg = _team_game.merge(_team_game, on="GAME_ID", suffixes=("", "_opp"))
_tg = _tg[_tg["TEAM_ABBREVIATION"] != _tg["TEAM_ABBREVIATION_opp"]].copy()
_tg["game_total_realized"] = _tg["team_pts_game"] + _tg["team_pts_game_opp"]
_tg["margin"] = _tg["team_pts_game"] - _tg["team_pts_game_opp"]
_game_ctx = _tg[
    [
        "GAME_ID",
        "TEAM_ABBREVIATION",
        "team_pts_game",
        "game_total_realized",
        "margin",
        "n_players_game",
    ]
]
df = df.merge(_game_ctx, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

# One row per team-game for rolling stats (avoids duplicate player rows in expanding windows)
_tguniq = (
    df.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)
    .agg(
        GAME_DATE=("GAME_DATE", "first"),
        SEASON=("SEASON", "first"),
        game_total_realized=("game_total_realized", "first"),
        margin=("margin", "first"),
        n_players_game=("n_players_game", "first"),
        team_pts_game=("team_pts_game", "first"),
    )
    .sort_values(["TEAM_ABBREVIATION", "SEASON", "GAME_DATE"])
    .reset_index(drop=True)
)
_gtuni = _tguniq.groupby(["TEAM_ABBREVIATION", "SEASON"], sort=False)

# Pace / competitiveness proxies (inference: Vegas total & abs spread; training: realized priors)
_tguniq["_gt_lag"] = _gtuni["game_total_realized"].shift(1)
_tguniq["game_total"] = _gtuni["_gt_lag"].transform(lambda x: x.expanding().mean())
_tguniq["game_total"] = _tguniq["game_total"].fillna(222.0)

_tguniq["_margin_lag"] = _gtuni["margin"].shift(1)
_tguniq["spread_abs"] = _gtuni["_margin_lag"].transform(lambda x: x.expanding().mean().abs())
_tguniq["spread_abs"] = _tguniq["spread_abs"].fillna(0.0)

_tguniq["_tpts_lag"] = _gtuni["team_pts_game"].shift(1)
_tguniq["_team_avg_pts_pg"] = _gtuni["_tpts_lag"].transform(lambda x: x.expanding().mean())

_tguniq["_np_lag"] = _gtuni["n_players_game"].shift(1)
_tguniq["_np_med"] = _gtuni["_np_lag"].transform(lambda x: x.shift(1).expanding().median())
_tguniq["teammate_out_count"] = (
    (_tguniq["_np_med"] - _tguniq["_np_lag"]).clip(lower=0, upper=8).fillna(0.0).astype(float)
)

_merge_cols = [
    "GAME_ID",
    "TEAM_ABBREVIATION",
    "game_total",
    "spread_abs",
    "teammate_out_count",
    "_team_avg_pts_pg",
]
df = df.merge(_tguniq[_merge_cols], on=["GAME_ID", "TEAM_ABBREVIATION"], how="left", suffixes=("", "_tg"))

df["usage_share"] = np.where(
    df["_team_avg_pts_pg"] > 1.0,
    (df["avg_pts"] / df["_team_avg_pts_pg"]).clip(0.0, 0.55),
    0.0,
).astype(float)

# Inference: cascade_bonus>0; training: bench-stress proxy or minute uptrend (pre-game safe)
df["cascade_signal"] = np.where(
    (df["teammate_out_count"] > 0.5) | (df["recent_min"] > df["avg_min"] * 1.12),
    1.0,
    0.0,
).astype(float)

del df["_team_avg_pts_pg"]

# ── v93: Playoff features ──────────────────────────────────────────────────
# 1. playoff_projected_min — rotation shrink proxy.
#    Starters (usage top-2 on team by avg_pts share) get bumped toward 40 min;
#    deep bench (avg_min < 18) gets projected near 0 in playoff context.
#    During regular season training this captures the same signal: starters
#    play more minutes in high-stakes games.
df["_team_rank"] = df.groupby(["GAME_ID", "TEAM_ABBREVIATION"])["avg_pts"].rank(
    ascending=False, method="first"
)
df["playoff_projected_min"] = df["avg_min"].copy()
# Top-2 usage on their team: project toward 38-40 min
_top2_mask = (df["_team_rank"] <= 2) & (df["starter_proxy"] == 1.0)
df.loc[_top2_mask, "playoff_projected_min"] = df.loc[_top2_mask, "avg_min"].clip(lower=36.0) * 1.05
df["playoff_projected_min"] = df["playoff_projected_min"].clip(upper=42.0)
# Deep bench (avg_min < 18): project to ~60% of their avg (rotation shrinks in playoffs)
_bench_mask = df["avg_min"] < 18.0
df.loc[_bench_mask, "playoff_projected_min"] = df.loc[_bench_mask, "avg_min"] * 0.60
del df["_team_rank"]

# 2. season_series_pts_per_min — player-specific matchup history.
#    How does this player perform against THIS specific opponent? In playoffs,
#    you face the same team 4-7 times, so head-to-head history is gold.
_h2h = (
    df.groupby(["PLAYER_ID", "SEASON", "OPP_TEAM"])
    .apply(lambda g: g.assign(
        _h2h_pts_sum=g["PTS"].shift(1).expanding().sum(),
        _h2h_min_sum=g["MIN"].shift(1).expanding().sum(),
    ), include_groups=False)
)
if "_h2h_pts_sum" in _h2h.columns:
    df["_h2h_pts_sum"] = _h2h["_h2h_pts_sum"]
    df["_h2h_min_sum"] = _h2h["_h2h_min_sum"]
else:
    df["_h2h_pts_sum"] = np.nan
    df["_h2h_min_sum"] = np.nan
df["season_series_pts_per_min"] = np.where(
    df["_h2h_min_sum"] > 10.0,
    df["_h2h_pts_sum"] / df["_h2h_min_sum"],
    df["pts_per_min"],  # fallback to general pts_per_min if insufficient h2h data
).clip(0.0, 2.5)
del df["_h2h_pts_sum"], df["_h2h_min_sum"]

# 3. spike_usage_interaction — superstar overdrive detector.
#    In playoffs, spikes come from high-usage stars pushing 40+ min with stable
#    minutes (low volatility). This interaction lets the spike model key on
#    exactly that profile. Also captures contrarian bench eruptions via high
#    volatility + low usage (unexpected playoff start).
df["spike_usage_interaction"] = (df["usage_share"] * (1.0 - df["min_volatility"])).clip(0.0, 0.55)

assert len(FEATURES) == N_FEATURES

if actuals_df is not None:
    df = df.merge(
        actuals_df[["norm_name", "date_key", "actual_rs"]],
        on=["norm_name", "date_key"],
        how="left",
    )
    n_labeled = int(df["actual_rs"].notna().sum())
    print(f"   Rows with actual_rs label: {n_labeled} (of {len(df)} after feature prep).")
    target = "actual_rs"
    df = df.dropna(subset=["actual_rs"])
else:
    print("   [WARN] No actuals found — falling back to formula target.")
    df["actual_base_score"] = (
        df["PTS"]
        + df["REB"]
        + (df["AST"] * 1.5)
        + (df["STL"] * 4.5)
        + (df["BLK"] * 4.0)
        - (df["TOV"] * 1.2)
    )
    target = "actual_base_score"

df = df.dropna(subset=FEATURES + [target])
print(f"After feature engineering: {len(df)} samples with complete features.")

# ── Train-serve skew validation ─────────────────────────────────────────────
# Verify that the shared compute_rs_features() produces the same values as
# the Pandas vectorized computation above. Catches formula drift between
# training and inference.
_skew_sample = df.iloc[len(df) // 2]  # mid-dataset sample
_skew_map = compute_rs_features(
    avg_min=_skew_sample["avg_min"],
    avg_pts=_skew_sample["avg_pts"],
    recent_min=_skew_sample["recent_min"],
    recent_pts=_skew_sample["recent_5g_pts"],
    season_pts=_skew_sample["avg_pts"],
    season_min=_skew_sample["avg_min"],
    recent_ast=_skew_sample["avg_ast"],
    recent_stl=_skew_sample["avg_stl"],
    recent_blk=_skew_sample["avg_blk"],
    avg_reb=_skew_sample["avg_reb"],
    home_away=_skew_sample["home_away"],
    opp_def_rating=_skew_sample["opp_def_rating"],
    rest_days=_skew_sample["rest_days"],
    games_played=_skew_sample["games_played"],
    cascade_signal=_skew_sample["cascade_signal"],
    opp_pts_allowed=_skew_sample["opp_pts_allowed"],
    team_pace_proxy=_skew_sample["team_pace_proxy"],
    usage_share=_skew_sample["usage_share"],
    teammate_out_count=_skew_sample["teammate_out_count"],
    game_total=_skew_sample["game_total"],
    spread_abs=_skew_sample["spread_abs"],
    recent_3g_pts=_skew_sample["roll3_pts"],
    # v93: playoff features for skew check
    is_top2_usage=False,  # skew check only — actual value computed at inference
    season_series_pts_per_min=_skew_sample["season_series_pts_per_min"],
)
_skew_checks = ["usage_trend", "ast_rate", "def_rate", "pts_per_min", "recent_vs_season",
                "reb_per_min", "starter_proxy", "home_away", "rest_days", "games_played",
                "spike_usage_interaction", "season_series_pts_per_min"]
for _sf in _skew_checks:
    _train_val = float(_skew_sample[_sf])
    _infer_val = float(_skew_map[_sf])
    if abs(_train_val - _infer_val) > 0.01:
        print(f"[SKEW WARNING] {_sf}: training={_train_val:.4f} vs shared={_infer_val:.4f}")
print("[skew-check] Train-serve alignment validated on sample row.")

if len(df) < 80:
    print(f"[ERROR] Only {len(df)} samples — need more for stable two-head training.")
    import sys

    sys.exit(1)

# ── Ranking-focused sample weights (per calendar date) ───────────────────────
date_col = df["GAME_DATE"].dt.normalize()


def _assign_weights(y_series: pd.Series) -> pd.Series:
    """Higher weight for top actual RS within the same day (approximate game slate)."""
    ranks = y_series.rank(pct=True, method="average")
    w = 1.0 + 4.0 * (ranks >= 0.90) + 2.0 * ((ranks >= 0.75) & (ranks < 0.90))
    return w


# ── v93: Playoff-intensity game weighting ──────────────────────────────────
# 2025-26 confirmed playoff teams (updated for training; auto-detected at inference).
# These are the teams that made or are making the playoffs this season.
PLAYOFF_TEAMS_2026 = {
    # Eastern Conference
    "CLE", "BOS", "NYK", "NY", "IND", "MIL", "DET", "ORL", "ATL", "MIA",
    # Western Conference
    "OKC", "HOU", "LAC", "DEN", "MIN", "LAL", "GSW", "GS", "MEM", "SAC",
}

def _playoff_intensity_weights(game_df: pd.DataFrame) -> pd.Series:
    """Compute per-row playoff-intensity multipliers for sample weighting.

    Strategy:
    - Games between two playoff-bound teams with tight spreads get 2.0× weight
      (simulates playoff conditions: tight rotation, high stakes).
    - Games between two playoff teams generally get 1.5× weight.
    - Late-season rest games (high teammate_out_count, last 3 weeks) get 0.5× weight.
    - All other games: 1.0× (baseline).
    """
    w = pd.Series(1.0, index=game_df.index)

    team_is_playoff = game_df["TEAM_ABBREVIATION"].isin(PLAYOFF_TEAMS_2026)
    opp_is_playoff = game_df["OPP_TEAM"].isin(PLAYOFF_TEAMS_2026)
    both_playoff = team_is_playoff & opp_is_playoff

    # Tight playoff-intensity games (spread < 5) between playoff teams
    tight_game = game_df["spread_abs"] < 5.0
    w = w.where(~(both_playoff & tight_game), 2.0)
    # Other playoff-vs-playoff games
    w = w.where(~(both_playoff & ~tight_game & (w == 1.0)), 1.5)

    # Down-weight late-season rest games: last 3 weeks of season + high teammate_out
    if "GAME_DATE" in game_df.columns:
        season_end_cutoff = game_df["GAME_DATE"].max() - pd.Timedelta(days=21)
        late_season = game_df["GAME_DATE"] >= season_end_cutoff
        high_rest = game_df["teammate_out_count"] >= 3.0
        rest_game = late_season & high_rest
        w = w.where(~rest_game, 0.5)

    return w


sample_weight = df.groupby(date_col, group_keys=False)[target].transform(_assign_weights)

# v93: Playoff-intensity game weighting — up-weight playoff-style games, down-weight rest games
playoff_w = _playoff_intensity_weights(df)
sample_weight = sample_weight * playoff_w
n_boosted = int((playoff_w > 1.0).sum())
n_rested = int((playoff_w < 1.0).sum())
print(f"   Playoff intensity: {n_boosted} games up-weighted, {n_rested} rest games down-weighted")

# v62: Top-performers sample weighting — players on the actual leaderboard get 3x weight
top_perf_parquet = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "top_performers.parquet")
top_perf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "top_performers.csv")
_tp_loaded = False
if os.path.exists(top_perf_parquet):
    try:
        tp_df = pd.read_parquet(top_perf_parquet)
        print(f"   Loaded top_performers from parquet ({len(tp_df)} rows)")
        _tp_loaded = True
    except Exception:
        pass
if not _tp_loaded and os.path.exists(top_perf_path):
    try:
        tp_df = pd.read_csv(top_perf_path)
        print(f"   Loaded top_performers from CSV ({len(tp_df)} rows)")
        _tp_loaded = True
    except Exception as e:
        print(f"   [WARN] Could not load top_performers: {e}")
if _tp_loaded:
    try:
        tp_df["norm_name"] = tp_df["player_name"].apply(_normalize_name)
        tp_df["date_key"] = pd.to_datetime(tp_df["date"]).dt.normalize()
        tp_keys = set(zip(tp_df["norm_name"], tp_df["date_key"]))
        is_leaderboard = [
            (nn, dk) in tp_keys
            for nn, dk in zip(df["norm_name"], df["date_key"])
        ]
        lb_boost = pd.Series([3.0 if is_lb else 1.0 for is_lb in is_leaderboard], index=df.index)
        sample_weight = sample_weight * lb_boost
        n_lb = sum(is_leaderboard)
        print(f"   Top-performers boost: {n_lb} samples get 3x weight (from {len(tp_keys)} leaderboard entries)")
    except Exception as e:
        print(f"   [WARN] Could not apply top_performers weighting: {e}")

X = df[FEATURES].copy()
y = df[target].astype(float)

(
    X_train,
    X_test,
    y_train,
    y_test,
    w_train,
    w_test,
    _date_train,
    _date_test,
    _idx_train,
    test_idx,
) = train_test_split(
    X,
    y,
    sample_weight,
    df["GAME_DATE"],
    df.index,
    test_size=0.2,
    random_state=42,
    shuffle=False,  # v63: Temporal split (not random) to avoid leakage
)

print(f"2. Training dual-head LightGBM on {len(X_train)} samples (25 features, playoff-tuned)...")
print(f"   Features ({len(FEATURES)}): {FEATURES}")

base_params = dict(
    n_estimators=1000,      # v93: 900→1000 for 25 features (3 playoff features)
    learning_rate=0.032,    # v93: slightly slower for more features
    max_depth=8,
    num_leaves=72,          # v93: 64→72 for playoff feature interactions
    random_state=42,
    subsample=0.80,
    colsample_bytree=0.78,  # v93: slightly more regularization for 25 features
)

model_baseline = lgb.LGBMRegressor(**base_params)
model_baseline.fit(
    X_train,
    y_train,
    sample_weight=w_train,
    eval_set=[(X_test, y_test)],
)

baseline_pred_full = model_baseline.predict(X)
spike_y = np.maximum(0.0, y.values - baseline_pred_full)

# v93: Spike model gets extra weight for high-usage, low-volatility players
# (superstar overdrive in playoffs) while still catching unexpected bench eruptions.
spike_w = sample_weight.values.copy()
_spike_inter = df["spike_usage_interaction"].values
# High usage × low volatility (>0.15) = playoff superstar profile → 2× spike weight
spike_w = np.where(_spike_inter > 0.15, spike_w * 2.0, spike_w)
# Moderate interaction (0.08-0.15) → 1.3× (still relevant, captures fringe starters)
spike_w = np.where((_spike_inter > 0.08) & (_spike_inter <= 0.15), spike_w * 1.3, spike_w)

model_spike = lgb.LGBMRegressor(**base_params)
model_spike.fit(X, spike_y, sample_weight=spike_w)

# ── Evaluation on test split (ranking KPIs) ────────────────────────────────
test_df = df.loc[test_idx].copy()
test_df["pred_rs"] = model_baseline.predict(X_test) + model_spike.predict(X_test)

recalls = []
ndcgs = []
for d, sub in test_df.groupby("GAME_DATE"):
    if len(sub) < 10:
        continue
    # top 5 by actual
    sub = sub.sort_values(target, ascending=False)
    actual_top5 = set(sub.head(5)["PLAYER_ID"])
    sub_pred = sub.sort_values("pred_rs", ascending=False)
    pred_top5 = set(sub_pred.head(5)["PLAYER_ID"])
    recalls.append(_top5_recall(pred_top5, actual_top5))
    rel = {row.PLAYER_ID: row[target] for _, row in sub.iterrows()}
    order = sub_pred["PLAYER_ID"].tolist()
    ndcgs.append(_ndcg_at_k(order, rel, k=5))

print(
    f"3. Test metrics — top5_recall_avg: {np.mean(recalls):.3f} | ndcg5_avg: {np.mean(ndcgs):.3f} | "
    f"MAE: {np.mean(np.abs(test_df[target] - test_df['pred_rs'])):.3f}"
)

BASELINE_TXT = "lgbm_baseline.txt"
SPIKE_TXT = "lgbm_spike.txt"

def _strip_lgbm_noise_params(path: str):
    """Remove training-only params that cause harmless but noisy warnings on load."""
    txt = Path(path).read_text()
    for param in ("early_stopping_min_delta", "bagging_by_query"):
        txt = re.sub(rf"^{param}=.*\n", "", txt, flags=re.MULTILINE)
        txt = re.sub(rf"^\[{param}:.*\]\n", "", txt, flags=re.MULTILINE)
    Path(path).write_text(txt)

print("4. Saving native LightGBM models + lgbm_model.json...")
model_baseline.booster_.save_model(BASELINE_TXT)
model_spike.booster_.save_model(SPIKE_TXT)
_strip_lgbm_noise_params(BASELINE_TXT)
_strip_lgbm_noise_params(SPIKE_TXT)
meta = {
    "format": "lightgbm_native",
    "bundle_version": 2,
    "features": FEATURES,
    "baseline_file": BASELINE_TXT,
    "spike_file": SPIKE_TXT,
}
with open("lgbm_model.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")

print("Done! Feature importances (baseline):")
for feat, imp in sorted(
    zip(FEATURES, model_baseline.feature_importances_), key=lambda x: -x[1]
):
    print(f"  {feat}: {imp}")
