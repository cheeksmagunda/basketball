# ─────────────────────────────────────────────────────────────────────────────
# MILP SLOT OPTIMIZER — Data-Driven Barbell Strategy
#
# Redesigned from 85-entry leaderboard analysis (Mar 5–22, 2026):
#
#   ARCHETYPE DISTRIBUTION OF TOP-5 DAILY WINNERS:
#     Elite Hybrid (48.2%): RS ≥ 4.0, Boost ≥ 1.5  → avg value 21.1
#     Star Anchor  (23.5%): RS ≥ 5.5, Boost < 1.5  → avg value 19.8
#     Boost Leverage (20%): RS 2.5–4.0, Boost ≥ 2.5 → avg value 17.2
#     Pure Boost    (0.0%): RS < 2.5, Boost ≥ 3.0   → NEVER wins
#
# The Real Sports App assigns escalating multipliers to draft slots:
#   Slot 1 (2.0x) > Slot 2 (1.8x) > Slot 3 (1.6x) > Slot 4 (1.4x) > Slot 5 (1.2x)
#
# ADDITIVE formula: Value = RS × (SlotMult + CardBoost)
#
# CHALK (Starting 5): Maximize Total Value directly. Allow stars with 0.0x
#   boost if RS ≥ 5.5 — they win 23.5% of the time through sheer production.
#   No artificial boost floors that kill generational RS performances.
#
# MOONSHOT: Maximize Leverage Value = RS^α × Boost^β × (Slot + Boost).
#   The β exponent on boost captures the 48.2% Elite Hybrid archetype
#   (RS 4+ AND boost 2+) that dominates leaderboards. RS floor at 3.0
#   prevents the Pure Boost trap (0% historical win rate for RS < 2.5).
#
# TWO-PHASE OPTIMIZATION (moonshot):
#   Phase 1: Select 5 players using leverage-shaped ratings
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

from api.shared import SLOT_MULTIPLIERS, SLOT_LABELS

# One-shot diagnostics when PuLP/MILP cannot run (production normally has both).
_PULP_MISSING_LOGGED = False
_MILP_FAILURE_LOGGED = False

# Slot multipliers: Real Sports App draft slot values


def optimize_lineup(projections, n=5, min_per_team=0, max_per_team=0,
                    sort_key="chalk_ev", rating_key="rating",
                    card_boost_key="est_mult", time_limit=5,
                    max_low_boost=0, low_boost_threshold=0.0,
                    objective_mode=None,
                    variance_penalty=0.0, variance_uplift=0.0,
                    boost_leverage_extra_power=0.0,
                    overlap_player_ids=None, overlap_cap=0,
                    overlap_id_key="id",
                    two_phase=False, raw_rating_key=None,
                    star_indices=None, min_star_count=0, max_star_count=0,
                    max_per_game=0, player_games=None,
                    min_high_boost_count=0, high_boost_threshold=2.0,
                    min_big_boost_count=0, big_boost_threshold=2.8,
                    min_scorer_count=0, scorer_pts_threshold=0.0,
                    max_double_teams=0):
    """Find the optimal player-to-slot assignment using MILP.

    Data-driven objective functions based on 85-entry leaderboard analysis:

    CHALK mode: Maximize Σ RS_i × (SlotMult_j + Boost_i) directly.
      - No variance penalty — winners have ALL variance profiles
      - Stars with 0.0x boost allowed — they win 23.5% through RS ≥ 5.5
      - Elite Hybrids (RS ≥ 4.0, Boost ≥ 1.5) naturally float to top

    MOONSHOT mode: Maximize Σ (RS_i^α × Boost_i^β) × (SlotMult_j + Boost_i)
      - α = 1.0 (RS is the foundation — 98.8% of winners have RS ≥ 3.0)
      - β = 0.8 (boost leverage — Elite Hybrids at 48.2% dominate)
      - Variance uplift rewards streaky players (momentum bonus)
      - Pure Boost trap prevented by RS ≥ 3.0 floor from pool gating

    Returns:
        List of n player dicts with slot assignments applied
    """
    if not projections or len(projections) < n:
        result = sorted(projections, key=lambda x: x.get(sort_key, 0), reverse=True)[:n]
        for i, p in enumerate(result):
            p["slot"] = SLOT_LABELS[i] if i < len(SLOT_LABELS) else "1.0x"
        return result

    if not PULP_AVAILABLE:
        global _PULP_MISSING_LOGGED
        if not _PULP_MISSING_LOGGED:
            print(
                "[WARN] PuLP not installed — lineup optimizer uses greedy sort fallback",
                flush=True,
            )
            _PULP_MISSING_LOGGED = True
        return _fallback_sort(projections, n, sort_key, max_per_game, player_games)

    try:
        selected = _solve_milp(projections, n, min_per_team, max_per_team,
                               rating_key, card_boost_key, time_limit,
                               max_low_boost, low_boost_threshold,
                               objective_mode,
                               variance_penalty, variance_uplift,
                               boost_leverage_extra_power,
                               overlap_player_ids, overlap_cap, overlap_id_key,
                               star_indices=star_indices,
                               min_star_count=min_star_count,
                               max_star_count=max_star_count,
                               max_per_game=max_per_game,
                               player_games=player_games,
                               min_high_boost_count=min_high_boost_count,
                               high_boost_threshold=high_boost_threshold,
                               min_big_boost_count=min_big_boost_count,
                               big_boost_threshold=big_boost_threshold,
                               min_scorer_count=min_scorer_count,
                               scorer_pts_threshold=scorer_pts_threshold,
                               max_double_teams=max_double_teams)

        if two_phase and len(selected) == n:
            # Phase 2: re-assign slots using raw RS for optimal placement.
            # In Score = RS × (Slot + Boost), boost is player-constant so
            # highest raw RS must go in highest slot.
            rr_key = raw_rating_key or rating_key
            return _solve_milp(selected, n, 0, 0,
                               rr_key, card_boost_key, time_limit,
                               0, 0.0, None, 0.0, 0.0, 0.0,
                               None, 0, overlap_id_key,
                               star_indices=None, min_star_count=0,
                               max_star_count=0)

        return selected
    except Exception as e:
        global _MILP_FAILURE_LOGGED
        if not _MILP_FAILURE_LOGGED:
            print(
                f"[WARN] MILP lineup solver failed ({e!r}); using greedy sort fallback",
                flush=True,
            )
            _MILP_FAILURE_LOGGED = True
        return _fallback_sort(projections, n, sort_key, max_per_game, player_games)


def _solve_milp(projections, n, min_per_team, max_per_team, rating_key,
                card_boost_key, time_limit, max_low_boost=0, low_boost_threshold=0.0,
                objective_mode=None,
                variance_penalty=0.0, variance_uplift=0.0,
                boost_leverage_extra_power=0.0,
                overlap_player_ids=None, overlap_cap=0, overlap_id_key="id",
                star_indices=None, min_star_count=0, max_star_count=0,
                max_per_game=0, player_games=None,
                min_high_boost_count=0, high_boost_threshold=2.0,
                min_big_boost_count=0, big_boost_threshold=2.8,
                min_scorer_count=0, scorer_pts_threshold=0.0,
                max_double_teams=0):
    """Run the MILP solver to find optimal player-slot assignments.

    Objective shaping (data-driven from 85-entry archetype analysis):

    CHALK: Pure Total Value — RS × (Slot + Boost). No artificial shaping.
      The math naturally produces the barbell: stars get high RS × slot benefit,
      role players get high boost × slot benefit. The solver finds the optimal
      mix without us artificially constraining it.

    MOONSHOT: Leverage-shaped — (RS × boost^β × (1 + variance×uplift)) × (Slot + Boost).
      β = boost_leverage_extra_power (default 0.8) creates a nonlinear preference
      for the Elite Hybrid archetype (RS 4+ / Boost 2+) that wins 48.2% of slates.
      Variance uplift rewards hot-streak players (+momentum).
    """
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
            # CHALK: Pure RS drives the objective. No variance penalty.
            # Data shows winners span ALL variance profiles — penalizing variance
            # killed Star Anchors (23.5% of winners) who are high-variance by nature.
            # The only shaping is a mild consistency bonus for low-variance players
            # to break ties between similar-RS candidates.
            consistency_bonus = max(0.0, 0.02 * (1.0 - min(v, 1.0)))
            base_rating = base_rating * (1.0 + consistency_bonus)
        elif objective_mode == "moonshot":
            # MOONSHOT: Boost leverage is the primary differentiator.
            # RS^1.0 × Boost^β × (1 + variance×uplift).
            # β = 0.8: creates strong preference for Elite Hybrids (RS 4+ / Boost 2+)
            # while still allowing Star Anchors (RS 5.5+ / Boost 0.5) through pure RS.
            # Variance uplift: hot-streak players get moonshot ceiling boost.
            base_rating = base_rating * (1.0 + float(variance_uplift) * float(v))
            if boost_leverage_extra_power and boost_leverage_extra_power > 0:
                # Boost leverage: +3.0x boost player gets 3.0^0.8 = 2.41x rating mult
                # +1.0x boost player gets 1.0^0.8 = 1.0x (neutral)
                # +0.3x boost star gets 0.3^0.8 = 0.37x (heavily penalized in selection)
                # This naturally produces the barbell without hardcoded thresholds.
                base_rating = base_rating * (max(float(card_boost), 0.1) ** float(boost_leverage_extra_power))

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
        if max_double_teams > 0 and max_per_team >= 2:
            # "At most max_double_teams teams may contribute max_per_team players;
            # all other teams contribute at most (max_per_team - 1) players."
            #
            # Formulation: introduce binary z_t per team.
            #   n_t ≤ (max_per_team - 1) + z_t        [z_t unlocks the extra slot]
            #   Σ z_t ≤ max_double_teams               [budget for double-team slots]
            #
            # In maximisation the solver sets z_t=1 only when it actually uses
            # the extra slot, so no tightening penalty for teams with 1 player.
            z = {}
            for t, player_indices in teams.items():
                z[t] = LpVariable(f"z_{t}", cat=LpBinary)
                n_t = lpSum(x[i, j] for i in player_indices for j in slots)
                prob += n_t <= (max_per_team - 1) + z[t], f"max_team_{t}"
            prob += lpSum(z[t] for t in teams) <= max_double_teams, "max_double_teams"
        else:
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

    # Star anchor constraint: at least min_star_count players from star_indices
    # must appear in the lineup. Forces inclusion of a genuine scorer (season_pts >= 20)
    # alongside high-boost role players. Only applied when feasible.
    if star_indices and min_star_count > 0:
        valid_stars = [i for i in star_indices if i in players]
        if len(valid_stars) >= min_star_count:
            prob += lpSum(
                x[i, j] for i in valid_stars for j in slots
            ) >= min_star_count

    # Cap how many designated star anchors can appear (when config sets max_count).
    if star_indices and max_star_count and max_star_count > 0:
        valid_stars_cap = [i for i in star_indices if i in players]
        if valid_stars_cap:
            prob += lpSum(
                x[i, j] for i in valid_stars_cap for j in slots
            ) <= min(int(max_star_count), n)

    # Max players per game (matchup) — prevents over-concentration in one game.
    if max_per_game > 0 and player_games and len(player_games) == len(projections):
        game_groups = {}
        for i, game_id in enumerate(player_games):
            game_groups.setdefault(game_id, []).append(i)
        for game_id, game_idxs in game_groups.items():
            valid_idxs = [i for i in game_idxs if i in players]
            if len(valid_idxs) > max_per_game:
                prob += lpSum(
                    x[i, j] for i in valid_idxs for j in slots
                ) <= max_per_game, f"max_per_game_{game_id}"

    # Minimum high-boost players — ensures contrarian plays are present.
    if min_high_boost_count > 0:
        hb_indices = [
            i for i, p in enumerate(projections)
            if p.get("est_mult", 0) >= high_boost_threshold
        ]
        if len(hb_indices) >= min_high_boost_count:
            prob += lpSum(
                x[i, j] for i in hb_indices for j in slots
            ) >= min_high_boost_count, "min_high_boost"

    # Minimum big-boost players — ensures at least one high-leverage play.
    if min_big_boost_count > 0:
        bb_indices = [
            i for i, p in enumerate(projections)
            if p.get("est_mult", 0) >= big_boost_threshold
        ]
        if len(bb_indices) >= min_big_boost_count:
            prob += lpSum(
                x[i, j] for i in bb_indices for j in slots
            ) >= min_big_boost_count, "min_big_boost"

    # Production anchor constraint — ensures at least min_scorer_count players
    # with projected pts >= scorer_pts_threshold are in the lineup. This guarantees
    # guaranteed scoring production (historical winners always had 1+ big scorer).
    if min_scorer_count > 0 and scorer_pts_threshold > 0:
        scorer_indices = [
            i for i, p in enumerate(projections)
            if (p.get("pts", 0) or 0) >= scorer_pts_threshold
        ]
        if len(scorer_indices) >= min_scorer_count:
            prob += lpSum(
                x[i, j] for i in scorer_indices for j in slots
            ) >= min_scorer_count, "min_scorer"

    solver = PULP_CBC_CMD(msg=0, timeLimit=time_limit)
    prob.solve(solver)

    if LpStatus[prob.status] != "Optimal":
        return _fallback_sort(projections, n, "chalk_ev",
                              max_per_game=max_per_game, player_games=player_games)

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


def _fallback_sort(projections, n, sort_key, max_per_game=0, player_games=None):
    """Sort-and-assign fallback when MILP is unavailable.

    When max_per_game and player_games are provided, uses a greedy selection
    that respects the per-game cap so fallback doesn't stack 3+ players from
    the same matchup (the most common infeasibility symptom).
    """
    if max_per_game > 0 and player_games and len(player_games) == len(projections):
        sorted_pairs = sorted(
            enumerate(projections),
            key=lambda x: x[1].get(sort_key, 0),
            reverse=True,
        )
        result = []
        game_counts = {}
        for i, p in sorted_pairs:
            gid = player_games[i]
            if game_counts.get(gid, 0) < max_per_game:
                result.append(p)
                game_counts[gid] = game_counts.get(gid, 0) + 1
            if len(result) == n:
                break
        # If game-aware greedy can't fill n slots (very small slate), fall back to plain sort
        if len(result) < n:
            result = sorted(projections, key=lambda x: x.get(sort_key, 0), reverse=True)[:n]
    else:
        result = sorted(projections, key=lambda x: x.get(sort_key, 0), reverse=True)[:n]

    for i, p in enumerate(result):
        p["slot"] = SLOT_LABELS[i] if i < len(SLOT_LABELS) else "1.0x"
    return result
