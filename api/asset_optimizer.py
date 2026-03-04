# ─────────────────────────────────────────────────────────────────────────────
# MILP SLOT OPTIMIZER — Intelligent Draft Multiplier Assignment
#
# The Real Sports App assigns escalating multipliers to draft slots:
#   MVP (2.0x) > Star (1.5x) > Pro (1.2x) > Utility (1.0x, 1.0x)
#
# Simple sort-and-assign (current approach) doesn't truly optimize the
# compound product of player_score × slot_multiplier across all combinations.
# This module uses Mixed-Integer Linear Programming (PuLP/CBC) to find the
# mathematically optimal player-to-slot assignment.
#
# The solver maximizes: Σ E(RealScore_i) × SlotMult_j × X[i,j]
# Subject to: each player in ≤1 slot, each slot exactly 1 player,
#             optional team balance constraints.
# ─────────────────────────────────────────────────────────────────────────────

import copy

try:
    from pulp import (
        LpProblem, LpMaximize, LpVariable, LpBinary, lpSum, LpStatus,
        PULP_CBC_CMD,
    )
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

# Slot multipliers: Real Sports App draft slot values
SLOT_MULTIPLIERS = [2.0, 1.5, 1.2, 1.0, 1.0]
SLOT_LABELS = ["2.0x", "1.5x", "1.2x", "1.0x", "1.0x"]


def optimize_lineup(projections, n=5, min_per_team=0, max_per_team=0,
                    sort_key="chalk_ev", rating_key="rating",
                    card_boost_key="est_mult", time_limit=5):
    """Find the optimal player-to-slot assignment using MILP.

    Uses the ADDITIVE Real Sports formula:
      Value = RawScore × (SlotMult + CardBoost)

    The solver maximizes: Σ rating_i × (slot_mult_j + card_boost_i) × X[i,j]
    This correctly assigns the highest RAW SCORE player to the 2.0x slot,
    since the marginal slot benefit is proportional to raw score.

    Args:
        projections: List of player dicts with rating and team fields
        n: Number of lineup slots (default 5)
        min_per_team: Minimum players per team (0 = no constraint)
        max_per_team: Maximum players per team (0 = no constraint)
        sort_key: Key to sort by in fallback mode
        rating_key: Key containing raw score (for slot assignment)
        card_boost_key: Key containing additive card boost
        time_limit: Solver time limit in seconds

    Returns:
        List of n player dicts with slot assignments applied
    """
    if not projections or len(projections) < n:
        # Not enough players; return what we have
        result = sorted(projections, key=lambda x: x.get(sort_key, 0), reverse=True)[:n]
        for i, p in enumerate(result):
            p["slot"] = SLOT_LABELS[i] if i < len(SLOT_LABELS) else "1.0x"
        return result

    if not PULP_AVAILABLE:
        return _fallback_sort(projections, n, sort_key)

    try:
        return _solve_milp(projections, n, min_per_team, max_per_team,
                           rating_key, card_boost_key, time_limit)
    except Exception:
        return _fallback_sort(projections, n, sort_key)


def _solve_milp(projections, n, min_per_team, max_per_team, rating_key,
                card_boost_key, time_limit):
    """Run the MILP solver to optimize player-slot assignments.

    Uses the ADDITIVE Real Sports formula:
      Value_ij = rating_i × (slot_mult_j + card_boost_i)
    """
    players = list(range(len(projections)))
    slots = list(range(n))
    slot_mults = SLOT_MULTIPLIERS[:n]

    prob = LpProblem("DraftSlotOptimizer", LpMaximize)

    # Decision variables: X[i][j] = 1 if player i assigned to slot j
    x = {
        (i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary)
        for i in players for j in slots
    }

    # Objective: maximize Σ rating_i × (slot_mult_j + card_boost_i) × X[i,j]
    # This is the ADDITIVE Real Sports formula — slot and card boost add together
    prob += lpSum(
        projections[i].get(rating_key, 0) *
        (slot_mults[j] + projections[i].get(card_boost_key, 0)) * x[i, j]
        for i in players for j in slots
    )

    # Constraint: each slot gets exactly one player
    for j in slots:
        prob += lpSum(x[i, j] for i in players) == 1

    # Constraint: each player in at most one slot
    for i in players:
        prob += lpSum(x[i, j] for j in slots) <= 1

    # Team constraints — build team index once
    teams = {}
    for i, p in enumerate(projections):
        t = p.get("team", "")
        teams.setdefault(t, []).append(i)

    # Optional: minimum per team (for per-game drafts — ensures both teams represented)
    if min_per_team > 0:
        for t, player_indices in teams.items():
            if len(teams) >= 2:
                prob += lpSum(
                    x[i, j] for i in player_indices for j in slots
                ) >= min(min_per_team, len(player_indices))

    # Optional: maximum per team (for full slate — forces game diversification)
    if max_per_team > 0:
        for t, player_indices in teams.items():
            prob += lpSum(
                x[i, j] for i in player_indices for j in slots
            ) <= max_per_team

    # Solve with CBC (bundled with PuLP, no external binary needed)
    solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    if LpStatus[prob.status] != "Optimal":
        # Solver didn't find optimal; fallback
        return _fallback_sort(projections, n, "chalk_ev")

    # Extract solution
    result = []
    for j in slots:
        for i in players:
            if x[i, j].varValue and x[i, j].varValue > 0.5:
                p = copy.deepcopy(projections[i])
                p["slot"] = SLOT_LABELS[j] if j < len(SLOT_LABELS) else "1.0x"
                result.append(p)
                break

    # Sort result by slot multiplier (highest first) for display consistency
    slot_order = {label: idx for idx, label in enumerate(SLOT_LABELS)}
    result.sort(key=lambda p: slot_order.get(p.get("slot", "1.0x"), 99))

    return result


def _fallback_sort(projections, n, sort_key):
    """Simple sort-and-assign fallback when MILP is unavailable."""
    result = sorted(projections, key=lambda x: x.get(sort_key, 0), reverse=True)[:n]
    for i, p in enumerate(result):
        p["slot"] = SLOT_LABELS[i] if i < len(SLOT_LABELS) else "1.0x"
    return result


def contrarian_score(player, spread=0):
    """Calculate moonshot/contrarian value for the Real Sports App.

    Moonshot targets players with high CARD-ADJUSTED ceiling:
    - Card advantage (est_card_boost from player tier)
    - Close-game environments (high Real Score ceiling)
    - High-variance/streaky players (momentum bonus potential)
    - Underdog side in competitive games

    Args:
        player: Player dict with rating, _real_meta, est_mult, etc.
        spread: Game spread (used for closeness/underdog assessment)

    Returns:
        Moonshot-adjusted score for ranking
    """
    base_rating = player.get("rating", 0)
    card_boost = player.get("est_mult", 0.5)

    # Game closeness boost — derived from Real Score metadata
    meta = player.get("_real_meta", {})
    closeness = meta.get("c_closeness", 1.3)
    # Normalize closeness (1.0-2.0 range) to a scoring boost (0.7-1.3)
    closeness_boost = 0.7 + (closeness - 1.0) * 0.6

    # Momentum/variance — streaky players have higher Real Score ceiling
    momentum = meta.get("m_momentum", 1.0)

    # Underdog side — underdogs in close games generate huge Real Scores
    underdog_bonus = 1.0
    game_spread = abs(player.get("_spread", spread) or 0)
    if 2 < game_spread <= 7:
        underdog_bonus = 1.1  # Competitive game with clear underdog

    # Additive formula: rating × (avg_slot + card_boost) × context multipliers
    avg_slot = 1.34
    c_score = base_rating * (avg_slot + card_boost) * closeness_boost * momentum * underdog_bonus
    return round(c_score, 2)
