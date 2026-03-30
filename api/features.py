# ─────────────────────────────────────────────────────────────────────────────
# SHARED FEATURE ENGINEERING — Single Source of Truth
# grep: SHARED FEATURES
#
# Eliminates train-serve skew by centralizing:
#   1. RS model feature list + scalar computation (used by inference, validated by training)
#   2. Boost/drafts model helpers (TEAM_MARKET_SCORES, pos_bucket, ppg_tier_bucket)
#
# Imported by: api/index.py (inference), train_lgbm.py, train_boost_lgbm.py, train_drafts_lgbm.py
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np

# ── Real Score LightGBM: canonical 22-feature list ─────────────────────────
# Order must match lgbm_model.json and train_lgbm.py DataFrame column order.
RS_FEATURES = [
    "avg_min",
    "avg_pts",
    "usage_trend",
    "opp_def_rating",
    "home_away",
    "ast_rate",
    "def_rate",
    "pts_per_min",
    "rest_days",
    "recent_vs_season",
    "games_played",
    "reb_per_min",
    "l3_vs_l5_pts",
    "min_volatility",
    "starter_proxy",
    "cascade_signal",
    "opp_pts_allowed",
    "team_pace_proxy",
    "usage_share",
    "teammate_out_count",
    "game_total",
    "spread_abs",
]

N_RS_FEATURES = len(RS_FEATURES)  # 22

# Clipping constants — shared between training (Pandas) and inference (scalar)
USAGE_TREND_MIN = 0.90
USAGE_TREND_MAX = 1.50
RECENT_VS_SEASON_MIN = 0.5
RECENT_VS_SEASON_MAX = 2.0
REB_PER_MIN_MIN = 0.0
REB_PER_MIN_MAX = 1.5
L3_VS_L5_MIN = 0.4
L3_VS_L5_MAX = 2.5
MIN_VOLATILITY_MIN = 0.0
MIN_VOLATILITY_MAX = 1.2
STARTER_PROXY_THRESHOLD = 26.0
REST_DAYS_DEFAULT = 2.0
GAMES_PLAYED_DEFAULT = 40.0
OPP_PTS_ALLOWED_DEFAULT = 110.0
TEAM_PACE_DEFAULT = 110.0
GAME_TOTAL_DEFAULT = 222.0


def compute_rs_features(
    *,
    avg_min: float,
    avg_pts: float,
    recent_min: float,
    recent_pts: float,
    season_pts: float,
    season_min: float,
    recent_ast: float,
    recent_stl: float,
    recent_blk: float,
    avg_reb: float,
    home_away: float,
    opp_def_rating: float,
    rest_days: float = REST_DAYS_DEFAULT,
    games_played: float = GAMES_PLAYED_DEFAULT,
    cascade_signal: float = 0.0,
    opp_pts_allowed: float = OPP_PTS_ALLOWED_DEFAULT,
    team_pace_proxy: float = TEAM_PACE_DEFAULT,
    usage_share: float = 0.0,
    teammate_out_count: float = 0.0,
    game_total: float = GAME_TOTAL_DEFAULT,
    spread_abs: float = 0.0,
    recent_3g_pts: float | None = None,
) -> dict[str, float]:
    """Compute all 22 RS features from scalar inputs.

    Formulas match train_lgbm.py Pandas operations exactly.
    Called by api/index.py at inference time.
    Training script validates its Pandas output against this function.
    """
    # 1. usage_trend — recent_min / avg_min, clipped
    usage_trend = float(np.clip(
        recent_min / max(avg_min, 1.0), USAGE_TREND_MIN, USAGE_TREND_MAX
    ))

    # 2. ast_rate — rolling(5) AST / recent_min  (training: avg_ast / recent_min)
    #    At inference, recent_ast ≈ ESPN L5/L10 average AST
    ast_rate = recent_ast / max(recent_min, 1.0)

    # 3. def_rate — rolling(5)(STL+BLK) / recent_min
    def_rate = (recent_stl + recent_blk) / max(recent_min, 1.0)

    # 4. pts_per_min — avg_pts / recent_min  (training uses season avg pts / recent rolling min)
    pts_per_min = avg_pts / max(recent_min, 1.0)

    # 5. recent_vs_season — recent_pts / avg_pts, clipped
    recent_vs_season = float(np.clip(
        recent_pts / max(avg_pts, 1.0), RECENT_VS_SEASON_MIN, RECENT_VS_SEASON_MAX
    ))

    # 6. reb_per_min — avg_reb / avg_min, clipped
    reb_per_min = float(np.clip(
        avg_reb / max(avg_min, 1.0), REB_PER_MIN_MIN, REB_PER_MIN_MAX
    ))

    # 7. l3_vs_l5_pts — roll3_pts / recent_5g_pts (training)
    #    At inference: use recent_3g_pts if available, else approximate
    if recent_3g_pts is not None:
        l3_vs_l5 = recent_3g_pts / max(recent_pts, 0.1)
    else:
        # Approximation: weighted blend as proxy for L3/L5 ratio
        l3_vs_l5 = (0.55 * recent_pts + 0.45 * season_pts) / max(recent_pts, 0.1)
    l3_vs_l5_pts = float(np.clip(l3_vs_l5, L3_VS_L5_MIN, L3_VS_L5_MAX))

    # 8. min_volatility — rolling(5).std() / (avg_min + 0.5) (training)
    #    At inference: |recent_min - season_min| / (avg_min + 0.5) as proxy
    #    Uses avg_min + 0.5 denominator to match training (not season_min)
    min_volatility = float(np.clip(
        abs(recent_min - season_min) / max(avg_min + 0.5, 1.0),
        MIN_VOLATILITY_MIN, MIN_VOLATILITY_MAX,
    ))

    # 9. starter_proxy — binary: 1.0 if avg_min >= 26
    starter_proxy = 1.0 if avg_min >= STARTER_PROXY_THRESHOLD else 0.0

    return {
        "avg_min": float(avg_min),
        "avg_pts": float(avg_pts),
        "usage_trend": usage_trend,
        "opp_def_rating": float(opp_def_rating),
        "home_away": float(home_away),
        "ast_rate": float(ast_rate),
        "def_rate": float(def_rate),
        "pts_per_min": float(pts_per_min),
        "rest_days": float(rest_days),
        "recent_vs_season": recent_vs_season,
        "games_played": float(games_played),
        "reb_per_min": reb_per_min,
        "l3_vs_l5_pts": l3_vs_l5_pts,
        "min_volatility": min_volatility,
        "starter_proxy": starter_proxy,
        "cascade_signal": float(cascade_signal),
        "opp_pts_allowed": float(opp_pts_allowed),
        "team_pace_proxy": float(team_pace_proxy),
        "usage_share": float(usage_share),
        "teammate_out_count": float(teammate_out_count),
        "game_total": float(game_total),
        "spread_abs": float(spread_abs),
    }


def rs_feature_vector(feature_map: dict[str, float], feature_order: list[str]) -> list[float]:
    """Extract ordered feature vector from a feature map.

    Args:
        feature_map: dict from compute_rs_features()
        feature_order: list of feature names (from model bundle or RS_FEATURES)

    Returns:
        List of float values in the requested order.

    Raises:
        ValueError if feature_order contains unknown feature names.
    """
    missing = [f for f in feature_order if f not in feature_map]
    if missing:
        raise ValueError(f"Model requires unknown features: {missing}")
    return [feature_map[f] for f in feature_order]


# ── Boost / Drafts model shared helpers ─────────────────────────────────────

# Continuous team market score (0.0–1.0) for all 30 NBA teams.
# Higher = larger fanbase = player more likely to be drafted = lower expected card boost.
TEAM_MARKET_SCORES: dict[str, float] = {
    "LAL": 1.00, "GSW": 0.95, "GS": 0.95, "BOS": 0.90, "NYK": 0.90, "NY": 0.90,
    "PHI": 0.75, "MIA": 0.75, "LAC": 0.70, "CHI": 0.70,
    "BKN": 0.65, "DEN": 0.65, "MIL": 0.60, "DAL": 0.60,
    "HOU": 0.55, "PHX": 0.55, "ATL": 0.50, "TOR": 0.50,
    "CLE": 0.45, "IND": 0.40, "ORL": 0.35, "POR": 0.35,
    "DET": 0.30, "MIN": 0.30, "OKC": 0.25, "UTA": 0.25,
    "SAS": 0.25, "NOP": 0.20, "NO": 0.20, "MEM": 0.20,
    "CHA": 0.15, "SAC": 0.15, "WSH": 0.10,
}


def get_team_market_score(team: str) -> float:
    """Look up continuous market score for an NBA team abbreviation."""
    return TEAM_MARKET_SCORES.get((team or "").strip().upper(), 0.3)


def pos_bucket(pos: str) -> int:
    """Coarse position bucket for ML models.

    Returns: 0=guard, 1=forward, 2=center, 3=unknown
    """
    p = (pos or "").strip().upper()
    if not p:
        return 3
    if p.startswith("C"):
        return 2
    if p[0] in ("G", "P") or p.startswith("PG") or p.startswith("SG"):
        return 0
    return 1


def ppg_tier_bucket(season_pts: float) -> int:
    """Coarse PPG tier (0-4) for player recognition.

    0=bench/fringe, 1=role player, 2=secondary scorer, 3=main option, 4=star/franchise
    """
    if season_pts < 8:
        return 0
    if season_pts < 13:
        return 1
    if season_pts < 18:
        return 2
    if season_pts < 24:
        return 3
    return 4
