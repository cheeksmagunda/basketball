# ─────────────────────────────────────────────────────────────────────────────
# REAL SCORE ENGINE — Context-Aware Monte Carlo Game Simulator
#
# The Real Sports App's "Real Score" algorithm fundamentally rejects linear,
# volume-based DFS scoring. Instead, player value is nonlinearly amplified by:
#
#   1. Game Closeness (C_c) — actions in tight games worth exponentially more
#   2. Clutch Factor (C_k) — late-game, lead-changing plays get massive boosts
#   3. Momentum Bonus (M_m) — streaky/high-variance players score more
#
# This module uses vectorized numpy Monte Carlo simulation to derive these
# coefficients from game spread and total, then applies them to the baseline
# statistical projection to produce a Real Score estimate.
#
# E(RealScore) = S_base × C_c × C_k × M_m
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from datetime import date


def _make_rng(spread, total, seed_date=None):
    """Deterministic RNG seeded by game parameters + date for cache stability."""
    d = seed_date or date.today().isoformat()
    seed = hash((d, round(spread, 1), round(total, 1))) % (2**31)
    return np.random.default_rng(seed)


def closeness_coefficient(spread, total, rng=None, n_sims=2000):
    """Simulate final score differential to estimate probability of a close game.

    Models the final margin as Normal(|spread|, sigma) where sigma scales with
    the square root of the total (higher-scoring games have more variance).

    Returns C_c in [1.0, 2.0]:
        - Pick'em (spread ~0): C_c ≈ 1.65–1.80
        - Moderate favorite (spread ~5): C_c ≈ 1.30–1.50
        - Heavy favorite (spread ~12+): C_c ≈ 1.05–1.15
    """
    if rng is None:
        rng = _make_rng(spread, total)

    abs_spread = abs(spread or 0)
    t = total or 222

    # Standard deviation: higher-scoring games have more variance
    sigma = 0.45 * np.sqrt(t)

    # Simulate final score differentials
    sims = rng.normal(loc=abs_spread, scale=sigma, size=n_sims)

    # P(close) = fraction of sims where final margin is within one possession (5 pts)
    p_close = np.mean(np.abs(sims) <= 5.0)

    # C_c ranges from 1.0 (guaranteed blowout) to 2.0 (guaranteed close)
    c_c = 1.0 + p_close
    return round(float(c_c), 3)


def clutch_coefficient(spread, total, usage_rate, player_variance,
                       rng=None, n_sims=2000):
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
        rng = _make_rng(spread, total)

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

    # Count lead changes: margin crosses zero
    signs = np.sign(margins)
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


def real_score_projection(s_base, spread, total, usage_rate, player_variance,
                          rng=None):
    """Master Real Score projection combining all contextual coefficients.

    Args:
        s_base: Baseline statistical projection (from LightGBM/heuristic blend)
        spread: Game point spread (positive = home favored)
        total: Over/under total
        usage_rate: Player's usage rate proxy (pts / minutes * scaling)
        player_variance: |recent_performance - season_average| / season_average
        rng: Optional numpy RNG for deterministic results

    Returns:
        (real_score, metadata) where metadata contains all coefficients
    """
    if rng is None:
        rng = _make_rng(spread, total)

    c_c = closeness_coefficient(spread, total, rng)
    c_k = clutch_coefficient(spread, total, usage_rate, player_variance, rng)
    m_m = momentum_bonus(player_variance)

    real_score = s_base * c_c * c_k * m_m

    metadata = {
        "c_closeness": c_c,
        "c_clutch": c_k,
        "m_momentum": m_m,
        "composite_mult": round(c_c * c_k * m_m, 3),
        "s_base": round(s_base, 3),
    }

    return real_score, metadata
