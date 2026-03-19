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
#
# TWO-PHASE OPTIMIZATION (moonshot):
#   Phase 1: Select 5 players using shaped ratings (boost leverage, variance)
#   Phase 2: Re-assign slots using pure raw RS — because boost is a player-level
#            constant, only raw RS determines optimal slot placement.
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
SLOT_MULTIPLIERS = [2.0, 1.8, 1.6, 1.4, 1.2]
SLOT_LABELS = ["2.0x", "1.8x", "1.6x", "1.4x", "1.2x"]


def optimize_lineup(projections, n=5, min_per_team=0, max_per_team=0,
                    sort_key="chalk_ev", rating_key="rating",
                    card_boost_key="est_mult", time_limit=5,
                    max_low_boost=0, low_boost_threshold=0.0,
                    objective_mode=None,
                    variance_penalty=0.0, variance_uplift=0.0,
                    boost_leverage_extra_power=0.0,
                    overlap_player_ids=None, overlap_cap=0,
                    overlap_id_key="id",
                    two_phase=False, raw_rating_key=None):
    """Find the optimal player-to-slot assignment using MILP.

    Maximizes: Σ rating_i × (slot_mult_j + card_boost_i) × X[i,j]

    Args:
        projections: List of player dicts with rating and team fields
        n: Number of lineup slots (default 5)
        min_per_team: Minimum players per team (0 = no constraint)
        max_per_team: Maximum players per team (0 = no constraint)
        sort_key: Key to sort by in fallback mode
        rating_key: Key containing score for player selection
        card_boost_key: Key containing additive card boost
        time_limit: Solver time limit in seconds
        max_low_boost: Max low-boost (star) players allowed (0 = no limit)
        low_boost_threshold: Boost below this = "low boost" player
        objective_mode: Optional shaping ("chalk"|"moonshot") applied to the
            MILP objective coefficients using player_variance.
        variance_penalty: Used when objective_mode="chalk" to downweight
            high-variance players.
        variance_uplift: Used when objective_mode="moonshot" to upweight
            high-variance players.
        boost_leverage_extra_power: When objective_mode="moonshot", further
            scales rating by est_mult^(extra_power) to emphasize boost.
        overlap_player_ids: Optional list of player ids to overlap-limit
            against. Only applied when overlap_player_ids is non-empty and
            overlap_cap > 0.
        overlap_cap: Maximum allowed overlapped players in the returned lineup.
            Interpreted as a count of players (not slots).
        overlap_id_key: Projection dict key containing the player id.
        two_phase: When True, run Phase 1 (player selection with shaped
            ratings) then Phase 2 (slot assignment with raw RS). This
            decouples selection from slotting so moonshot shaping doesn't
            corrupt slot placement.
        raw_rating_key: Key for raw (unaltered) RS used in Phase 2 slotting.
            Required when two_phase=True. Falls back to rating_key if unset.

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
        selected = _solve_milp(projections, n, min_per_team, max_per_team,
                               rating_key, card_boost_key, time_limit,
                               max_low_boost, low_boost_threshold,
                               objective_mode,
                               variance_penalty, variance_uplift,
                               boost_leverage_extra_power,
                               overlap_player_ids, overlap_cap, overlap_id_key)

        if two_phase and len(selected) == n:
            # Phase 2: re-assign slots using raw RS for optimal placement.
            # In Score = RS × (Slot + Boost), boost is player-constant so
            # highest raw RS must go in highest slot.
            rr_key = raw_rating_key or rating_key
            return _solve_milp(selected, n, 0, 0,
                               rr_key, card_boost_key, time_limit,
                               0, 0.0, None, 0.0, 0.0, 0.0,
                               None, 0, overlap_id_key)

        return selected
    except Exception:
        return _fallback_sort(projections, n, sort_key)


def _solve_milp(projections, n, min_per_team, max_per_team, rating_key,
                card_boost_key, time_limit, max_low_boost=0, low_boost_threshold=0.0,
                objective_mode=None,
                variance_penalty=0.0, variance_uplift=0.0,
                boost_leverage_extra_power=0.0,
                overlap_player_ids=None, overlap_cap=0, overlap_id_key="id"):
    """Run the MILP solver to find optimal player-slot assignments."""
    players = list(range(len(projections)))
    slots = list(range(n))
    slot_mults = SLOT_MULTIPLIERS[:n]

    prob = LpProblem("DraftSlotOptimizer", LpMaximize)

    x = {
        (i, j): LpVariable(f"x_{i}_{j}", cat=LpBinary)
        for i in players for j in slots
    }

    # Precompute objective coefficients per player to keep the MILP linear:
    # objective = Σ (effective_rating_i) × (slot_mult_j + card_boost_i) × x[i,j]
    eff_rating = {}
    for i in players:
        base_rating = projections[i].get(rating_key, 0) or 0
        v = projections[i].get("player_variance", 0.0) or 0.0
        card_boost = projections[i].get(card_boost_key, 0) or 0.0

        if objective_mode == "chalk":
            # Median-ish: downweight high variance.
            base_rating = base_rating * max(0.0, 1.0 - float(variance_penalty) * float(v))
        elif objective_mode == "moonshot":
            # High-end-ish: upweight high variance.
            base_rating = base_rating * (1.0 + float(variance_uplift) * float(v))
            if boost_leverage_extra_power and boost_leverage_extra_power > 0:
                # Extra emphasis on boost signal.
                base_rating = base_rating * (max(float(card_boost), 0.0) ** float(boost_leverage_extra_power))

        eff_rating[i] = base_rating

    prob += lpSum(
        eff_rating[i] * (slot_mults[j] + projections[i].get(card_boost_key, 0)) * x[i, j]
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

    # No position-per-team constraint — Real Sports has no position requirements.
    # The solver freely selects any player combination regardless of position overlap.

    # Low-boost star cap: prevents 3+ superstars (low card boost) crowding out
    # higher-EV role players. Only active when both params are set (chalk mode).
    if max_low_boost > 0 and low_boost_threshold > 0:
        low_boost_players = [
            i for i, p in enumerate(projections)
            if p.get(card_boost_key, 0) < low_boost_threshold
        ]
        if len(low_boost_players) > max_low_boost:
            prob += lpSum(
                x[i, j] for i in low_boost_players for j in slots
            ) <= max_low_boost

    # Overlap constraint: allow at most overlap_cap players from overlap_player_ids.
    # This is applied only when feasible based on candidate pool size.
    if overlap_player_ids and overlap_cap and overlap_cap > 0:
        overlap_set = set(overlap_player_ids)
        overlap_indices = [i for i, p in enumerate(projections) if p.get(overlap_id_key) in overlap_set]
        non_overlap_indices = [i for i in players if i not in overlap_indices]

        # Only enforce when we can still fill the lineup with enough non-overlap players.
        required_non_overlap = max(0, n - int(overlap_cap))
        if len(non_overlap_indices) >= required_non_overlap and overlap_indices:
            prob += lpSum(
                x[i, j] for i in overlap_indices for j in slots
            ) <= min(int(overlap_cap), n)

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
