# ─────────────────────────────────────────────────────────────────────────────
# PARLAY ENGINE — Safest 3-Leg Player Prop Parlay Optimizer
#
# Optimizes for CERTAINTY (floor), not edge or upside.
# Pipeline:
#   1. Build candidate legs from projections + odds (Z-score hit probability)
#   2. Anti-fragility filters (blowout, volatility, GTD risk)
#   3. Market alignment (model + Vegas blended confidence)
#   4. Strategic correlation scoring (same-game synergy, negative veto)
#   5. Return the single highest-scoring 3-leg combination + narrative
# ─────────────────────────────────────────────────────────────────────────────

import math
import itertools
from datetime import datetime, timezone

# ── Statistical helpers ──────────────────────────────────────────────────────

_STAT_LABEL = {"points": "PTS", "rebounds": "REB", "assists": "AST"}

def _american_to_implied(american_odds):
    """Convert American odds (e.g. -140, +120) to implied probability [0, 1]."""
    if american_odds is None:
        return None
    try:
        odds = float(american_odds)
    except (TypeError, ValueError):
        return None
    if odds == 0:
        return None
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 100.0 / (odds + 100.0)


def _z_to_probability(z):
    """Approximate cumulative normal distribution (CDF) for a Z-score.
    Returns P(X < z) — the probability of being below z.
    Uses Abramowitz & Stegun approximation (accurate to ~1e-5)."""
    if z > 6:
        return 1.0
    if z < -6:
        return 0.0
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p_const = 0.3275911
    sign = 1 if z >= 0 else -1
    x = abs(z) / math.sqrt(2)
    t = 1.0 / (1.0 + p_const * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def _compute_hit_probability(projection, line, std_dev, direction):
    """Compute hit probability for a prop bet using Z-score.

    For OVER: P(actual > line) = 1 - CDF((line - projection) / sigma)
    For UNDER: P(actual < line) = CDF((line - projection) / sigma)
    """
    if std_dev is None or std_dev <= 0:
        return None
    z = (line - projection) / std_dev
    if direction == "over":
        return 1.0 - _z_to_probability(z)
    else:
        return _z_to_probability(z)


def _compute_blended_confidence(model_prob, vegas_prob, model_weight=0.55):
    """Blend model hit probability with Vegas implied probability.
    Model gets slightly more weight (0.55) since it has player-specific context."""
    if model_prob is None and vegas_prob is None:
        return None
    if model_prob is None:
        return vegas_prob
    if vegas_prob is None:
        return model_prob
    return model_prob * model_weight + vegas_prob * (1.0 - model_weight)


# ── Anti-Fragility Filters ──────────────────────────────────────────────────

def _is_blowout_game(spread, max_spread=8.5):
    """Discard players in blowout games (spread > threshold)."""
    if spread is None:
        return False
    return abs(float(spread)) > max_spread


def _minutes_cv(gamelog_minutes):
    """Compute coefficient of variation for minutes.
    CV = std_dev / mean. High CV = volatile minutes = bad for parlays."""
    if not gamelog_minutes or len(gamelog_minutes) < 5:
        return 999.0  # Insufficient data → treat as volatile
    avg = sum(gamelog_minutes) / len(gamelog_minutes)
    if avg <= 0:
        return 999.0
    variance = sum((m - avg) ** 2 for m in gamelog_minutes) / len(gamelog_minutes)
    return math.sqrt(variance) / avg


def _has_gtd_star_teammate(player_team, rw_statuses, player_name_lower):
    """Check if a star teammate (starter) is questionable/GTD.
    If so, the team's usage tree is unpredictable → discard all props."""
    if not rw_statuses:
        return False
    for name_key, info in rw_statuses.items():
        if not isinstance(info, dict):
            continue
        team = (info.get("team") or "").upper()
        status = (info.get("status") or "").lower()
        is_starter = info.get("is_starter", False)
        if team == player_team.upper() and name_key != player_name_lower:
            if is_starter and status == "questionable":
                return True
    return False


# ── Candidate Leg Builder ────────────────────────────────────────────────────

def build_candidate_legs(projections, games, player_odds_map, gamelogs,
                         rw_statuses, parlay_config=None):
    """Build all valid candidate parlay legs from projections + odds.

    Args:
        projections: List of player projection dicts from the pipeline.
        games: List of game dicts (with spread, total, teams).
        player_odds_map: {(player_name_lower, stat_type): {line, odds_over, odds_under, books_consensus}}
        gamelogs: {player_id: {stat_type: [last N values], "minutes": [last N values]}}
        rw_statuses: RotoWire statuses dict.
        parlay_config: Optional config overrides.

    Returns:
        List of candidate leg dicts, each with hit_prob, blended_conf, etc.
    """
    cfg = parlay_config or {}
    max_spread = cfg.get("max_spread", 8.5)
    max_minutes_cv = cfg.get("max_minutes_cv", 0.25)
    min_blended_conf = cfg.get("min_blended_conf", 0.55)
    min_season_minutes = cfg.get("min_season_minutes", 24.0)
    min_games_played = cfg.get("min_games_played", 15)
    juice_threshold = cfg.get("juice_threshold", -125)  # Vegas must juice at least this much

    # Build game lookup: team_abbr → game info
    game_lookup = {}
    for g in games:
        home_abbr = g["home"]["abbr"]
        away_abbr = g["away"]["abbr"]
        game_id = g.get("gameId", "")
        spread = g.get("spread")
        total = g.get("total", 222)
        start_time = g.get("startTime", "")
        for abbr, opp in [(home_abbr, away_abbr), (away_abbr, home_abbr)]:
            game_lookup[abbr] = {
                "gameId": game_id, "opponent": opp, "spread": spread,
                "total": total, "startTime": start_time,
                "home": home_abbr, "away": away_abbr,
            }

    stat_types = ["points", "rebounds", "assists"]
    candidates = []

    for p in projections:
        pid = str(p.get("id", ""))
        name = p.get("name", "")
        team = p.get("team", "")
        name_lower = name.lower()

        if not pid or not name or not team:
            continue

        # Season minutes floor
        season_min = float(p.get("season_min") or p.get("predMin") or 0)
        if season_min < min_season_minutes:
            continue

        # Injury filter — skip OUT or questionable players
        injury = (p.get("injury_status") or "").lower()
        if injury in ("out", "questionable"):
            continue

        # Game context
        gctx = game_lookup.get(team)
        if not gctx:
            continue

        # Blowout filter
        if _is_blowout_game(gctx.get("spread"), max_spread):
            continue

        # GTD star teammate filter
        if _has_gtd_star_teammate(team, rw_statuses, name_lower):
            continue

        # Minutes volatility filter
        player_log = gamelogs.get(pid, {})
        log_minutes = player_log.get("minutes", [])
        cv = _minutes_cv(log_minutes)
        if cv > max_minutes_cv:
            continue

        # Games played filter (need enough data for reliable std dev)
        games_in_log = len(log_minutes)
        if games_in_log < min_games_played:
            continue

        # Build legs for each stat type × direction
        for stat in stat_types:
            stat_key = {"points": "pts", "rebounds": "reb", "assists": "ast"}.get(stat, stat)
            proj_val = float(p.get(stat_key) or 0)
            if proj_val <= 0:
                continue

            # Get sportsbook line
            odds_data = None
            if player_odds_map:
                odds_data = player_odds_map.get((name_lower, stat))
                if not odds_data:
                    # Substring match fallback
                    for (oname, ostat), odata in player_odds_map.items():
                        if ostat == stat and (name_lower in oname or oname in name_lower):
                            odds_data = odata
                            break
            if not odds_data:
                continue  # No sportsbook line → can't build a parlay leg

            book_line = float(odds_data.get("line", 0))
            if book_line <= 0:
                continue

            # Determine direction: pick the side where model + Vegas agree
            for direction in ("over", "under"):
                odds_key = f"odds_{direction}"
                american_odds = odds_data.get(odds_key)
                vegas_prob = _american_to_implied(american_odds)

                # Juice filter: Vegas must be pricing this direction at juice_threshold or more negative
                # (more juiced = Vegas thinks this side is more likely)
                if american_odds is not None:
                    try:
                        if float(american_odds) > juice_threshold:
                            continue  # Not juiced enough — Vegas doesn't favor this side
                    except (TypeError, ValueError):
                        continue

                # Compute model hit probability via Z-score
                log_values = player_log.get(stat, [])
                if len(log_values) < 5:
                    continue
                std_dev = _std_dev(log_values)
                model_prob = _compute_hit_probability(proj_val, book_line, std_dev, direction)
                if model_prob is None:
                    continue

                # Blended confidence
                blended = _compute_blended_confidence(model_prob, vegas_prob)
                if blended is None or blended < min_blended_conf:
                    continue

                # Raw edge
                edge = proj_val - book_line if direction == "over" else book_line - proj_val

                candidates.append({
                    "player_name": name,
                    "player_id": pid,
                    "team": team,
                    "opponent": gctx.get("opponent", ""),
                    "gameId": gctx.get("gameId", ""),
                    "stat_type": stat,
                    "direction": direction,
                    "line": book_line,
                    "projection": round(proj_val, 1),
                    "edge": round(edge, 1),
                    "std_dev": round(std_dev, 2),
                    "z_score": round((proj_val - book_line) / std_dev if std_dev > 0 else 0, 2),
                    "model_hit_prob": round(model_prob, 4),
                    "vegas_implied_prob": round(vegas_prob, 4) if vegas_prob else None,
                    "blended_confidence": round(blended, 4),
                    "american_odds": american_odds,
                    "minutes_cv": round(cv, 3),
                    "season_avg": round(float(p.get(f"season_{stat_key}") or proj_val), 1),
                    "game_spread": gctx.get("spread"),
                    "game_total": gctx.get("total"),
                    "game_time": gctx.get("startTime", ""),
                    "home_team": gctx.get("home", ""),
                    "away_team": gctx.get("away", ""),
                })

    return candidates


def _std_dev(values):
    """Population standard deviation of a list of numbers."""
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    avg = sum(values) / n
    variance = sum((v - avg) ** 2 for v in values) / n
    return math.sqrt(variance)


# ── Correlation Scoring ──────────────────────────────────────────────────────

def _correlation_modifier(legs, parlay_config=None):
    """Score a 3-leg combination for strategic correlation.

    Returns (multiplier, reasons) where multiplier >= 0.
    multiplier = 0 means VETO (negative correlation detected).

    Positive correlation examples:
      - PG Assists Over + teammate Points Over (pick-and-roll synergy)
      - Opposing primary scorers both Points Over in tight game (shootout)

    Negative correlation (VETO):
      - Two players on same team, both Rebounds Over (zero-sum boards)
    """
    cfg = parlay_config or {}
    positive_boost = cfg.get("positive_correlation_boost", 1.08)
    shootout_boost = cfg.get("shootout_correlation_boost", 1.05)

    multiplier = 1.0
    reasons = []

    # Index legs by team and game
    by_team = {}
    by_game = {}
    for leg in legs:
        t = leg["team"]
        gid = leg["gameId"]
        by_team.setdefault(t, []).append(leg)
        by_game.setdefault(gid, []).append(leg)

    # ── VETO: Same team, both Rebounds Over ──
    for team, team_legs in by_team.items():
        reb_overs = [l for l in team_legs if l["stat_type"] == "rebounds" and l["direction"] == "over"]
        if len(reb_overs) >= 2:
            return 0.0, ["VETO: Two players on same team competing for rebounds"]

    # ── VETO: Same team, both Assists Over (only one ball) ──
    for team, team_legs in by_team.items():
        ast_overs = [l for l in team_legs if l["stat_type"] == "assists" and l["direction"] == "over"]
        if len(ast_overs) >= 2:
            return 0.0, ["VETO: Two players on same team competing for assists"]

    # ── POSITIVE: PG Assists Over + teammate Points Over (pick-and-roll synergy) ──
    for team, team_legs in by_team.items():
        ast_overs = [l for l in team_legs if l["stat_type"] == "assists" and l["direction"] == "over"]
        pts_overs = [l for l in team_legs if l["stat_type"] == "points" and l["direction"] == "over"]
        if ast_overs and pts_overs:
            multiplier *= positive_boost
            a_name = ast_overs[0]["player_name"]
            p_name = pts_overs[0]["player_name"]
            reasons.append(f"Positive correlation: {a_name} assists feed {p_name} scoring (same team)")

    # ── POSITIVE: Opposing primary scorers, Points Over, tight game (shootout) ──
    for gid, game_legs in by_game.items():
        if len(game_legs) < 2:
            continue
        pts_overs_by_team = {}
        game_spread = None
        for l in game_legs:
            if l["stat_type"] == "points" and l["direction"] == "over":
                pts_overs_by_team.setdefault(l["team"], []).append(l)
            if game_spread is None:
                game_spread = l.get("game_spread")
        teams_with_pts = list(pts_overs_by_team.keys())
        if len(teams_with_pts) >= 2:
            spread = abs(float(game_spread)) if game_spread is not None else 999
            if spread <= 5.0:
                multiplier *= shootout_boost
                reasons.append(f"Shootout correlation: opposing scorers in tight game (spread {spread})")

    # ── POSITIVE: Same game, different useful stats (game-script alignment) ──
    for gid, game_legs in by_game.items():
        if len(game_legs) >= 2:
            stats_in_game = set(l["stat_type"] for l in game_legs)
            if len(stats_in_game) >= 2 and "points" in stats_in_game:
                total = game_legs[0].get("game_total")
                if total is not None and float(total) >= 228:
                    multiplier *= 1.03
                    reasons.append(f"High-total game script alignment ({total} O/U)")

    return round(multiplier, 4), reasons


# ── Narrative Builder ────────────────────────────────────────────────────────

def _build_parlay_narrative(legs, correlation_reasons, combined_prob):
    """Build a natural-language narrative explaining the parlay selection."""
    parts = []

    # Opening with combined probability
    pct = round(combined_prob * 100, 1)
    parts.append(f"This 3-leg parlay has a blended hit probability of {pct}%.")

    # Describe each leg's strength
    for i, leg in enumerate(legs):
        name = leg["player_name"]
        stat_label = _STAT_LABEL.get(leg["stat_type"], leg["stat_type"].upper())
        direction = leg["direction"].upper()
        line = leg["line"]
        prob = round(leg["blended_confidence"] * 100, 1)
        edge = leg["edge"]
        edge_sign = "+" if edge > 0 else ""
        cv = leg.get("minutes_cv", 0)
        parts.append(
            f"{name} {direction} {line} {stat_label} ({prob}% confidence, "
            f"{edge_sign}{edge} edge, {round(cv * 100, 1)}% min volatility)."
        )

    # Correlation reasons
    if correlation_reasons:
        parts.append(" ".join(correlation_reasons))

    return " ".join(parts)


# ── Main Engine ──────────────────────────────────────────────────────────────

def run_parlay_engine(projections, games, player_odds_map, gamelogs,
                      rw_statuses=None, parlay_config=None):
    """Find the optimal 3-leg parlay from today's slate.

    Args:
        projections: Player projections from the pipeline.
        games: Today's games.
        player_odds_map: Odds API data.
        gamelogs: {player_id: {"points": [...], "rebounds": [...], "assists": [...], "minutes": [...]}}
        rw_statuses: RotoWire availability statuses.
        parlay_config: Config overrides.

    Returns:
        {
            "legs": [leg1, leg2, leg3],
            "combined_probability": float,
            "correlation_multiplier": float,
            "correlation_reasons": [str, ...],
            "parlay_score": float,
            "narrative": str,
            "candidates_evaluated": int,
            "combinations_scored": int,
            "timestamp": str,
        } or None if no valid parlay found.
    """
    cfg = parlay_config or {}

    # Step 1: Build all valid candidate legs
    candidates = build_candidate_legs(
        projections, games, player_odds_map, gamelogs,
        rw_statuses, cfg,
    )

    if len(candidates) < 3:
        return None

    # Deduplicate: one leg per player per stat (keep highest blended confidence)
    seen = {}
    for c in candidates:
        key = (c["player_id"], c["stat_type"])
        if key not in seen or c["blended_confidence"] > seen[key]["blended_confidence"]:
            seen[key] = c
    deduped = list(seen.values())

    if len(deduped) < 3:
        return None

    # Sort by blended confidence descending; limit to top N for combinatorial sanity
    max_candidates = cfg.get("max_candidates_for_combinations", 25)
    deduped.sort(key=lambda x: x["blended_confidence"], reverse=True)
    pool = deduped[:max_candidates]

    # Step 2: Evaluate all 3-leg combinations
    best_score = -1
    best_combo = None
    best_corr_mult = 1.0
    best_corr_reasons = []
    combos_scored = 0

    for combo in itertools.combinations(pool, 3):
        legs = list(combo)

        # No duplicate players in the parlay
        player_ids = set(l["player_id"] for l in legs)
        if len(player_ids) < 3:
            continue

        # Correlation check (may VETO)
        corr_mult, corr_reasons = _correlation_modifier(legs, cfg)
        if corr_mult <= 0:
            continue  # Vetoed

        # Combined probability = product of individual blended confidences × correlation
        raw_prob = 1.0
        for l in legs:
            raw_prob *= l["blended_confidence"]
        score = raw_prob * corr_mult

        combos_scored += 1

        if score > best_score:
            best_score = score
            best_combo = legs
            best_corr_mult = corr_mult
            best_corr_reasons = corr_reasons

    if best_combo is None:
        return None

    # Sort legs by blended confidence descending for display
    best_combo.sort(key=lambda x: x["blended_confidence"], reverse=True)

    combined_prob = 1.0
    for l in best_combo:
        combined_prob *= l["blended_confidence"]

    narrative = _build_parlay_narrative(best_combo, best_corr_reasons, combined_prob * best_corr_mult)

    return {
        "legs": best_combo,
        "combined_probability": round(combined_prob, 4),
        "correlation_multiplier": best_corr_mult,
        "correlation_reasons": best_corr_reasons,
        "parlay_score": round(best_score, 6),
        "narrative": narrative,
        "candidates_evaluated": len(candidates),
        "combinations_scored": combos_scored,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
