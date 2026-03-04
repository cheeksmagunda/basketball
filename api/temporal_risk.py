# ─────────────────────────────────────────────────────────────────────────────
# TEMPORAL RISK-ADJUSTED VALUE (TRAV) MODULE
#
# The Real Sports App uses a strict time-based tiebreaker: identical lineups
# are ranked by submission time (earlier wins). This creates a game theory
# dilemma — lock early for tiebreaker equity vs. wait for injury certainty.
#
# This module:
#   1. Estimates player ownership via minutes-based proxy
#   2. Calculates lineup duplication probability
#   3. Computes optimal lock time balancing tiebreaker vs. injury risk
#   4. Produces TRAV-adjusted scores for lineup ranking
# ─────────────────────────────────────────────────────────────────────────────

import math
import numpy as np
from datetime import datetime


def estimate_ownership(player, pool_size, game_spread=0):
    """Estimate a player's ownership percentage using minutes-based proxy.

    Minutes tiers map to expected ownership rates. Adjusted by:
    - Spread: underdog players get lower ownership (field chases favorites)
    - Cascade: injury-cascade picks have lower awareness
    - Pool size: larger pools dilute individual ownership

    Args:
        player: Player dict with predMin, team, is_cascade_pick, etc.
        pool_size: Number of viable players in the draft pool
        game_spread: Point spread (positive = home favored)

    Returns:
        Estimated ownership fraction (0.0 to 1.0)
    """
    proj_min = player.get("predMin", 20)

    # Base ownership from minutes tier
    if proj_min >= 33:
        base_own = 0.60    # Stars — everyone drafts them
    elif proj_min >= 28:
        base_own = 0.35    # Starters — popular picks
    elif proj_min >= 22:
        base_own = 0.15    # Role players — moderate ownership
    elif proj_min >= 15:
        base_own = 0.05    # Bench — low ownership
    else:
        base_own = 0.02    # Deep bench — very low

    # Underdog discount: players on heavily unfavored teams get less ownership
    # (field gravitates toward favorites)
    abs_spread = abs(game_spread or 0)
    if abs_spread > 5:
        # Determine if player is on underdog side
        # We don't have side info here, so use a general discount
        base_own *= 0.85

    # Cascade discount: injury-boost players are less known to the field
    if player.get("is_cascade_pick"):
        base_own *= 0.60

    # Pool size normalization: larger pools dilute ownership
    if pool_size > 0:
        pool_factor = min(30.0 / max(pool_size, 10), 1.5)
        base_own *= pool_factor

    return round(float(np.clip(base_own, 0.01, 0.85)), 4)


def lineup_duplication_probability(lineup, pool_size, estimated_entrants=1000):
    """Calculate probability that at least one other entrant has the exact same lineup.

    Uses the product of individual ownership probabilities as a proxy for
    the probability of any single entrant matching, then scales by field size.

    P(dup) = 1 - (1 - Π(ownership_i))^(entrants - 1)

    Args:
        lineup: List of player dicts
        pool_size: Total viable players in the pool
        estimated_entrants: Estimated number of draft entrants

    Returns:
        Probability of lineup duplication (0.0 to 1.0)
    """
    if not lineup:
        return 0.0

    # Product of individual ownership probabilities
    joint_prob = 1.0
    for p in lineup:
        own = estimate_ownership(p, pool_size)
        joint_prob *= own

    # P(at least one duplicate among entrants)
    p_no_dup = (1.0 - joint_prob) ** max(estimated_entrants - 1, 1)
    p_dup = 1.0 - p_no_dup

    return round(float(np.clip(p_dup, 0.0, 1.0)), 4)


def optimal_lock_time(game_start_iso, lineup, pool_size,
                      injury_risk_per_hour=0.02, estimated_entrants=1000):
    """Calculate the optimal lineup submission time.

    Sweeps from 5 to 180 minutes before tipoff, evaluating:
    - Tiebreaker equity: exponential — most entrants lock late, so early = advantage
    - Injury risk: probability that a rostered player gets scratched before tipoff

    Net EV is asymmetric: injury cost >> tiebreaker benefit, so the optimal
    time balances meaningful tiebreaker edge against tolerable injury risk.

    Args:
        game_start_iso: ISO timestamp of game start
        lineup: List of player dicts
        pool_size: Total viable players
        injury_risk_per_hour: Per-player-per-hour probability of late scratch
        estimated_entrants: Estimated field size

    Returns:
        Dict with optimal_minutes_before, tiebreaker_equity, injury_risk,
        lock_recommendation (human-readable)
    """
    n_players = len(lineup) if lineup else 5
    dup_prob = lineup_duplication_probability(lineup, pool_size, estimated_entrants)

    best_ev = -float("inf")
    best_minutes = 30  # default

    sweep_results = []

    for minutes_before in range(5, 181, 5):
        hours_before = minutes_before / 60.0

        # Tiebreaker equity: most people lock 5-15 min before
        # Locking earlier means beating more entrants on tiebreaker
        # Modeled as exponential: tiebreaker_eq = 1 - exp(-0.05 * minutes)
        tiebreaker_eq = 1.0 - math.exp(-0.05 * minutes_before)

        # Injury risk: P(at least 1 of n_players scratched in next hours_before)
        p_all_healthy = (1.0 - injury_risk_per_hour) ** (hours_before * n_players)
        injury_risk = 1.0 - p_all_healthy

        # Net EV: tiebreaker matters more when lineup is highly duplicated
        # Injury cost is weighted much higher (losing a player = catastrophic)
        tiebreaker_value = tiebreaker_eq * dup_prob * 0.3
        injury_cost = injury_risk * 0.7

        net_ev = tiebreaker_value - injury_cost

        sweep_results.append({
            "minutes_before": minutes_before,
            "tiebreaker_equity": round(tiebreaker_eq, 4),
            "injury_risk": round(injury_risk, 4),
            "net_ev": round(net_ev, 4),
        })

        if net_ev > best_ev:
            best_ev = net_ev
            best_minutes = minutes_before

    # Generate human-readable recommendation
    if best_minutes >= 120:
        recommendation = f"Lock lineup ~{best_minutes // 60}+ hours before tipoff — high duplication risk"
    elif best_minutes >= 60:
        recommendation = f"Lock lineup ~{best_minutes} min before tipoff — moderate duplication risk"
    elif best_minutes >= 30:
        recommendation = f"Lock lineup ~{best_minutes} min before tipoff — standard timing"
    else:
        recommendation = f"Safe to wait until ~{best_minutes} min before tipoff — low duplication"

    return {
        "optimal_minutes_before": best_minutes,
        "duplication_probability": dup_prob,
        "lock_recommendation": recommendation,
        "tiebreaker_equity_at_optimal": round(
            1.0 - math.exp(-0.05 * best_minutes), 4
        ),
        "injury_risk_at_optimal": round(
            1.0 - (1.0 - injury_risk_per_hour) ** (best_minutes / 60.0 * n_players), 4
        ),
    }


def trav_adjusted_score(raw_score, dup_prob, tiebreaker_equity):
    """Apply Temporal Risk-Adjusted Value to a player's raw score.

    Penalizes scores in highly-duplicated lineups (less leverage against field)
    and rewards early-lock tiebreaker equity.

    TRAV = raw_score × (1 - dup_prob × 0.3) × (1 + tiebreaker_eq × 0.1)

    Args:
        raw_score: Player's projected raw score
        dup_prob: Lineup duplication probability (0-1)
        tiebreaker_equity: Tiebreaker equity at current lock time (0-1)

    Returns:
        TRAV-adjusted score
    """
    dup_penalty = 1.0 - (dup_prob * 0.3)
    tiebreaker_bonus = 1.0 + (tiebreaker_equity * 0.1)
    return round(raw_score * dup_penalty * tiebreaker_bonus, 2)
