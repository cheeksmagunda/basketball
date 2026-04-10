# ─────────────────────────────────────────────────────────────────────────────
# REAL SCORE ENGINE — Context-Aware Monte Carlo Game Simulator
#
# The Real Sports App's "Real Score" algorithm fundamentally rejects linear,
# volume-based DFS scoring. Instead, player value is nonlinearly amplified by:
#
#   1. Game Closeness (C_c) — actions in tight games worth exponentially more
#   2. Clutch Factor (C_k) — late-game, lead-changing plays get massive boosts
#   3. Momentum Bonus (M_m) — streaky/high-variance players score more
#   4. Condition Coefficient (C_cond) — DFS ownership × card boost meta-game layer
#
# This module uses vectorized numpy Monte Carlo simulation to derive these
# coefficients from game spread and total, then applies them to the baseline
# statistical projection to produce a Real Score estimate.
#
# E(RealScore) = S_base × C_c × C_k × M_m × C_cond
#
# grep: REAL SCORE ENGINE — real_score_projection, closeness_coefficient, clutch_factor
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from datetime import datetime, timezone, timedelta
import hashlib
from api.shared import et_date as _shared_et_date


def _et_today():
    """Current date in Eastern Time as ISO string.
    Delegates to shared.et_date() for single source of truth.
    """
    return _shared_et_date().isoformat()


def _make_rng(spread, total, seed_date=None, game_id=None):
    """Deterministic RNG seeded by game parameters + ET date for cache stability.
    Uses ET date (not UTC) so the seed stays consistent for the full NBA evening."""
    d = seed_date or _et_today()
    seed_key = f"{d}|{round(spread, 1)}|{round(total, 1)}|{game_id or ''}"
    seed = int(hashlib.sha256(seed_key.encode("utf-8")).hexdigest()[:8], 16)
    return np.random.default_rng(seed)


def closeness_coefficient(spread, total, rng=None, n_sims=2000, game_id=None):
    """Simulate final score differential to estimate probability of a close game.

    Models the final margin as Normal(|spread|, sigma) where sigma scales with
    the square root of the total (higher-scoring games have more variance).

    Returns C_c in [1.0, 2.0]:
        - Pick'em (spread ~0): C_c ≈ 1.65–1.80
        - Moderate favorite (spread ~5): C_c ≈ 1.30–1.50
        - Heavy favorite (spread ~12+): C_c ≈ 1.05–1.15
    """
    if rng is None:
        rng = _make_rng(spread, total, game_id=game_id)

    abs_spread = abs(spread or 0)
    t = total or 222

    # Standard deviation: higher-scoring games have more variance.
    # NOTE: For very low totals (~180), sigma grows with sqrt(t) but the spread
    # stays constant, producing wide distributions where p_close becomes
    # counter-intuitively high (more sims land within 5 pts). This is
    # mathematically correct — low-scoring games ARE closer on average — but
    # users should be aware when interpreting closeness coefficients for
    # unusually low totals.
    sigma = 0.45 * np.sqrt(t)

    # Simulate final score differentials
    sims = rng.normal(loc=abs_spread, scale=sigma, size=n_sims)

    # P(close) = fraction of sims where final margin is within one possession (5 pts)
    p_close = np.mean(np.abs(sims) <= 5.0)

    # C_c ranges from 1.0 (guaranteed blowout) to 2.0 (guaranteed close)
    c_c = 1.0 + p_close
    return round(float(c_c), 3)


def clutch_coefficient(spread, total, usage_rate, player_variance,
                       rng=None, n_sims=2000, game_id=None):
    """Simulate 4th quarter scoring trajectory to estimate clutch opportunity.

    Models Q4 as a random walk over ~12 possessions per team. Lead changes
    indicate high-leverage moments where clutch plays occur. Players with
    higher usage rates and variance are more likely to be involved.

    Returns C_k in [0.9, 1.8]:
        - Blowout game + low-usage player: C_k ≈ 0.90–1.00
        - Close game + high-usage star: C_k ≈ 1.40–1.70
        - Overtime-bound thriller + streaky scorer: C_k ≈ 1.60–1.80
    """
    if rng is None:
        rng = _make_rng(spread, total, game_id=game_id)

    abs_spread = abs(spread or 0)
    t = total or 222

    # Q4 scoring: each team scores ~25-30 pts in Q4, modeled as random walk
    q4_pts_per_team = t / 8.0  # ~27.5 for a 220 total
    possessions = 12  # approximate Q4 possessions per team
    pts_per_poss = q4_pts_per_team / possessions

    # Simulate Q4 score differential evolution
    # Start with entering-Q4 margin (scaled from spread)
    entering_margin = rng.normal(loc=abs_spread * 0.75, scale=3.0, size=n_sims)

    # Simulate possession-by-possession scoring as random walk
    # Each possession: delta ~ Normal(0, pts_per_poss)
    walk = rng.normal(loc=0, scale=pts_per_poss, size=(n_sims, possessions))
    cumulative = np.cumsum(walk, axis=1)

    # Track margin through Q4
    margins = entering_margin[:, np.newaxis] + cumulative

    # Count lead changes: margin crosses zero (ignore ties — np.sign(0)=0)
    # Only count transitions between positive and negative, skipping zeros
    signs = np.sign(margins)
    # Forward-fill zeros with last nonzero sign so ties don't create false crossings
    for col in range(1, signs.shape[1]):
        mask = signs[:, col] == 0
        signs[mask, col] = signs[mask, col - 1]
    sign_changes = np.sum(np.abs(np.diff(signs, axis=1)) > 0, axis=1)

    # P(clutch) = fraction of sims with 2+ lead changes in Q4
    p_clutch = np.mean(sign_changes >= 2)

    # Weight by player's usage rate and variance
    # High-usage players are more likely to be the ones making clutch plays
    usage_weight = np.clip(usage_rate, 0.5, 2.0)
    variance_weight = 1.0 + np.clip(player_variance, 0, 0.5) * 0.6

    # C_k: base from game context, amplified by player profile
    c_k = 0.9 + (p_clutch * 0.6 * usage_weight * variance_weight)

    return round(float(np.clip(c_k, 0.9, 1.8)), 3)


def momentum_bonus(player_variance):
    """Calculate momentum bonus for streaky/high-variance players.

    The Real Score algorithm rewards clustered production over steady output.
    A player who scores 15 points in a 3-minute burst generates more Real Score
    than one who scores 15 points evenly across the game.

    Returns M_m in [1.0, 1.25]:
        - Steady, consistent player (low variance): M_m ≈ 1.00–1.05
        - Streaky scorer (high variance): M_m ≈ 1.15–1.25
    """
    v = np.clip(player_variance, 0, 0.5)
    m_m = 1.0 + v * 0.5
    return round(float(m_m), 3)


# ─────────────────────────────────────────────────────────────────────────────
# META-GAME CONDITION ENGINE (Ported from Baseball Condition Classifier)
# ─────────────────────────────────────────────────────────────────────────────
#
# The (boost_tier × ownership_tier) matrix is the core edge. Historical HV
# rates show ghost+elite-boost dominates across formats. This is format-agnostic
# — it exploits how DFS value is calculated and how the crowd drafts, not
# anything sport-specific.
#
# Basketball-specific notes:
#   - Ownership clusters tighter in playoffs (smaller player pool)
#   - Boost leverage is HIGHER because RS floors are higher (less variance)
#   - "Ghost hunting" in playoffs = role players in high-usage situations
#     (backup PG who becomes starter due to injury, stretch-4 getting extra
#     minutes in a specific matchup, 3-and-D wing getting hot in high-pace game)
# ─────────────────────────────────────────────────────────────────────────────

# Historical HV (Highest Value) rate by (ownership_tier, boost_tier)
CONDITION_MATRIX = {
    "ghost": {
        "no_boost": 0.35, "low_boost": 0.55, "mid_boost": 0.75,
        "elite_boost": 0.88, "max_boost": 1.00,
    },
    "low": {
        "no_boost": 0.28, "low_boost": 0.42, "mid_boost": 0.55,
        "elite_boost": 0.62, "max_boost": 0.70,
    },
    "medium": {
        "no_boost": 0.22, "low_boost": 0.35, "mid_boost": 0.38,
        "elite_boost": 0.28, "max_boost": 0.23,
    },
    "chalk": {
        "no_boost": 0.18, "low_boost": 0.25, "mid_boost": 0.28,
        "elite_boost": 0.25, "max_boost": 0.23,
    },
    "mega_chalk": {
        "no_boost": 0.15, "low_boost": 0.18, "mid_boost": 0.20,
        "elite_boost": 0.15, "max_boost": 0.12,
    },
}

# Dead capital: Trap plays where the crowd is too heavy for positive EV.
# If a player hits one of these conditions, their projection is zeroed out
# so the lineup optimizer automatically skips them.
DEAD_CAPITAL_CONDITIONS = {
    ("chalk", "elite_boost"),
    ("chalk", "max_boost"),
    ("chalk", "low_boost"),
    ("mega_chalk", "low_boost"),
    ("mega_chalk", "no_boost"),
}


def condition_coefficient(drafts, card_boost):
    """
    Evaluates DFS ownership and boost tier to return a meta-game multiplier.

    This is the format exploit layer. It doesn't care about basketball —
    it cares about how value is calculated and how the crowd drafts.

    Returns:
        float: 0.0 for DEAD_CAPITAL (auto-filtered), otherwise the
               historical HV rate as an EV multiplier (0.12 – 1.00).
    """
    # 1. Determine Ownership Tier
    if drafts is None:
        ownership_tier = "medium"  # Conservative default
    elif drafts < 100:
        ownership_tier = "ghost"
    elif drafts < 500:
        ownership_tier = "low"
    elif drafts < 1000:
        ownership_tier = "medium"
    elif drafts < 2000:
        ownership_tier = "chalk"
    else:
        ownership_tier = "mega_chalk"

    # 2. Determine Boost Tier
    boost = card_boost or 1.0
    if boost < 1.2:
        boost_tier = "no_boost"
    elif boost < 1.6:
        boost_tier = "low_boost"
    elif boost < 1.8:
        boost_tier = "mid_boost"
    elif boost <= 2.0:
        boost_tier = "elite_boost"
    else:
        boost_tier = "max_boost"

    # 3. Guard against trap plays (Dead Capital)
    if (ownership_tier, boost_tier) in DEAD_CAPITAL_CONDITIONS:
        return 0.0

    # 4. Return the historical HV rate as EV multiplier
    c_cond = CONDITION_MATRIX[ownership_tier][boost_tier]
    return round(float(c_cond), 3)


def real_score_projection(s_base, spread, total, usage_rate, player_variance,
                          drafts=None, card_boost=1.0, rng=None):
    """Master Real Score projection combining all contextual coefficients.

    On-court layer:  C_c (closeness), C_k (clutch), M_m (momentum)
    Meta-game layer: C_cond (ownership × boost condition matrix)

    Args:
        s_base: Baseline statistical projection (from LightGBM/heuristic blend)
        spread: Game point spread (positive = home favored)
        total: Over/under total
        usage_rate: Player's usage rate proxy (pts / minutes * scaling)
        player_variance: |recent_performance - season_average| / season_average
        drafts: Number of times this player has been drafted (ownership proxy)
        card_boost: Player's current card boost multiplier
        rng: Optional numpy RNG for deterministic results

    Returns:
        (real_score, metadata) where metadata contains all coefficients
    """
    if rng is None:
        rng = _make_rng(spread, total)

    # On-Court Game Mechanics
    c_c = closeness_coefficient(spread, total, rng)
    c_k = clutch_coefficient(spread, total, usage_rate, player_variance, rng)
    m_m = momentum_bonus(player_variance)

    # Meta-Game Mechanics (DFS Ownership/Boost conditions)
    c_cond = condition_coefficient(drafts, card_boost)

    # Master Equation
    real_score = s_base * c_c * c_k * m_m * c_cond

    metadata = {
        "c_closeness": c_c,
        "c_clutch": c_k,
        "m_momentum": m_m,
        "c_condition": c_cond,
        "composite_mult": round(c_c * c_k * m_m * c_cond, 3),
        "s_base": round(s_base, 3),
        "is_dead_capital": c_cond == 0.0,
    }

    return real_score, metadata
