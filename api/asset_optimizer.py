# ─────────────────────────────────────────────────────────────────────────────
# MILP SLOT OPTIMIZER — Intelligent Draft Multiplier Assignment
#
# The Real Sports App assigns escalating multipliers to draft slots:
#   Slot 1 (2.0x) > Slot 2 (1.8x) > Slot 3 (1.6x) > Slot 4 (1.4x) > Slot 5 (1.2x)
#
# Uses Mixed-Integer Linear Programming (PuLP/CBC) to find the optimal
# player-to-slot assignment, maximizing:
#
#   Σ E(RealScore_i) × (SlotMult_j + CardBoost_i) × X[i,j]
#
# This is the ADDITIVE Real Sports formula — slot and card boost add together.
# Assigning the highest raw-score player to 2.0x is optimal because marginal
# slot benefit scales with raw score.
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

# Position group map — same buckets as index.py POS_GROUPS
_POS_GROUPS = {
    "PG": "G", "SG": "G", "G": "G",
    "SF": "F", "PF": "F", "F": "F",
    "C": "C",
}

def _pos_group(pos):
    return _POS_GROUPS.get(pos, "F")

# Slot multipliers: Real Sports App draft slot values
SLOT_MULTIPLIERS = [2.0, 1.8, 1.6, 1.4, 1.2]
SLOT_LABELS = ["2.0x", "1.8x", "1.6x", "1.4x", "1.2x"]


def optimize_lineup(projections, n=5, min_per_team=0, max_per_team=0,
                    sort_key="chalk_ev", rating_key="rating",
                    card_boost_key="est_mult", time_limit=5):
    """Find the optimal player-to-slot assignment using MILP.

    Maximizes: Σ rating_i × (slot_mult_j + card_boost_i) × X[i,j]

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
    """Run the MILP solver to find optimal player-slot assignments."""
    players = list(range(len(projections)))
    slots = list(range(n))
    slot_mults = SLOT_MULTIPLIERS[:n]

    prob = LpProblem("DraftSlotOptimizer", LpMaximize)

    x = {
        (i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary)
        for i in players for j in slots
    }

    prob += lpSum(
        projections[i].get(rating_key, 0) *
        (slot_mults[j] + projections[i].get(card_boost_key, 0)) * x[i, j]
        for i in players for j in slots
    )

    for j in slots:
        prob += lpSum(x[i, j] for i in players) == 1

    for i in players:
        prob += lpSum(x[i, j] for j in slots) <= 1

    teams = {}
    for i, p in enumerate(projections):
        t = p.get("team", "")
        teams.setdefault(t, []).append(i)

    if min_per_team > 0:
        for t, player_indices in teams.items():
            if len(teams) >= 2:
                prob += lpSum(
                    x[i, j] for i in player_indices for j in slots
                ) >= min(min_per_team, len(player_indices))

    if max_per_team > 0:
        for t, player_indices in teams.items():
            prob += lpSum(
                x[i, j] for i in player_indices for j in slots
            ) <= max_per_team

    # Position-per-team constraint: at most 1 player per (team, pos_group) in lineup.
    # Prevents two centers (or two guards, two forwards) from the same team both
    # appearing — they share a real-world role and the pick looks redundant.
    pos_team_groups = {}
    for i, p in enumerate(projections):
        key = (p.get("team", ""), _pos_group(p.get("pos", "")))
        pos_team_groups.setdefault(key, []).append(i)
    for player_indices in pos_team_groups.values():
        if len(player_indices) >= 2:
            prob += lpSum(
                x[i, j] for i in player_indices for j in slots
            ) <= 1

    solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    if LpStatus[prob.status] != "Optimal":
        return _fallback_sort(projections, n, "chalk_ev")

    result = []
    for j in slots:
        for i in players:
            if x[i, j].varValue and x[i, j].varValue > 0.5:
                p = copy.deepcopy(projections[i])
                p["slot"] = SLOT_LABELS[j] if j < len(SLOT_LABELS) else "1.0x"
                result.append(p)
                break

    slot_order = {label: idx for idx, label in enumerate(SLOT_LABELS)}
    result.sort(key=lambda p: slot_order.get(p.get("slot", "1.0x"), 99))

    return result


def _fallback_sort(projections, n, sort_key):
    """Simple sort-and-assign fallback when MILP is unavailable."""
    result = sorted(projections, key=lambda x: x.get(sort_key, 0), reverse=True)[:n]
    for i, p in enumerate(result):
        p["slot"] = SLOT_LABELS[i] if i < len(SLOT_LABELS) else "1.0x"
    return result
