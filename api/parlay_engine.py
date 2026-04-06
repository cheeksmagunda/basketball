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
# grep: PARLAY ENGINE MODULE — run_parlay_engine, Z-score legs (see also api/index grep: PARLAY ENGINE)
# ─────────────────────────────────────────────────────────────────────────────

import math
import itertools
from datetime import datetime, timezone
from api.shared import STAT_ABBR
from api.odds_math import american_to_implied as _american_to_implied

# ── Statistical helpers ──────────────────────────────────────────────────────

_STAT_LABEL = STAT_ABBR


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
    n = len(gamelog_minutes)
    variance = sum((m - avg) ** 2 for m in gamelog_minutes) / (n - 1)  # Bessel's correction
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


def select_parlay_gamelog_player_ids(projections, games, player_odds_map, rw_statuses,
                                    parlay_config, projection_only):
    """Choose which ESPN player ids need gamelog fetches before build_candidate_legs.

    When projection_only is False, only players with at least one sportsbook line in
    player_odds_map can become legs — others need no gamelog. When True (synthetic
    lines for everyone), cap by projected RS (rating) to limit ESPN fan-out.

    Pool size scales with max_candidates_for_combinations (same spirit as combo cap).
    """
    cfg = parlay_config or {}
    max_spread = cfg.get("max_spread", 8.5)
    min_season_minutes = float(cfg.get("min_season_minutes", 20.0))
    max_cand = int(cfg.get("max_candidates_for_combinations", 25))
    pool_cap = min(max(max_cand * 5, 40), int(cfg.get("parlay_gamelog_pool_cap", 100)))

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

    scored = []
    for p in projections:
        pid = str(p.get("id", ""))
        name = p.get("name", "")
        team = p.get("team", "")
        name_lower = name.lower()
        if not pid or not name or not team:
            continue
        season_min = float(p.get("season_min") or p.get("predMin") or 0)
        if season_min < min_season_minutes:
            continue
        injury = (p.get("injury_status") or "").lower()
        if injury in ("out", "questionable"):
            continue
        gctx = game_lookup.get(team)
        if not gctx:
            continue
        if _is_blowout_game(gctx.get("spread"), max_spread):
            continue
        if _has_gtd_star_teammate(team, rw_statuses, name_lower):
            continue
        if not projection_only:
            has_odds = False
            for st in ("points", "rebounds", "assists"):
                if player_odds_map and player_odds_map.get((name_lower, st)):
                    has_odds = True
                    break
            if not has_odds and player_odds_map:
                for (oname, ostat), _ in player_odds_map.items():
                    if ostat in ("points", "rebounds", "assists") and (name_lower in oname or oname in name_lower):
                        has_odds = True
                        break
            if not has_odds:
                continue
        rating = float(p.get("rating") or 0)
        scored.append((rating, pid))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [pid for _, pid in scored[:pool_cap]]


# ── Candidate Leg Builder ────────────────────────────────────────────────────

def build_candidate_legs(projections, games, player_odds_map, gamelogs,
                         rw_statuses, parlay_config=None, gamelog_player_ids=None, dvp_data=None,
                         fair_value_data=None):
    """Build all valid candidate parlay legs from projections + odds.

    Args:
        projections: List of player projection dicts from the pipeline.
        games: List of game dicts (with spread, total, teams).
        player_odds_map: {(player_name_lower, stat_type): {line, odds_over, odds_under, books_consensus}}
        gamelogs: {player_id: {stat_type: [last N values], "minutes": [last N values]}}
        rw_statuses: RotoWire statuses dict.
        parlay_config: Optional config overrides.
        gamelog_player_ids: Optional set of ESPN player ids that have gamelog rows fetched
            (from ``select_parlay_gamelog_player_ids``). When None, every projection is
            evaluated (legacy). When set, other ids are skipped to match scoped fetches.

    Returns:
        List of candidate leg dicts, each with hit_prob, blended_conf, etc.
    """
    cfg = parlay_config or {}
    max_spread = cfg.get("max_spread", 8.5)
    max_minutes_cv = cfg.get("max_minutes_cv", 0.30)
    min_blended_conf = cfg.get("min_blended_conf", 0.52)
    min_season_minutes = cfg.get("min_season_minutes", 20.0)
    min_games_played = cfg.get("min_games_played", 10)
    juice_threshold = cfg.get("juice_threshold", -105)  # Vegas must juice at least this much

    # Minimum sportsbook line floors — filter out trivially easy props (0.5 ast, 1.5 reb)
    min_line_floors = cfg.get("min_line_floors", {})
    _line_floor_pts = float(min_line_floors.get("points", 10.5))
    _line_floor_reb = float(min_line_floors.get("rebounds", 3.5))
    _line_floor_ast = float(min_line_floors.get("assists", 2.5))
    _LINE_FLOORS = {"points": _line_floor_pts, "rebounds": _line_floor_reb, "assists": _line_floor_ast}

    # Build game lookup: team_abbr → game info
    game_lookup = {}
    for g in games:
        home_abbr = g["home"]["abbr"]
        away_abbr = g["away"]["abbr"]
        game_id = g.get("gameId", "")
        spread = g.get("spread")
        total = g.get("total", 222)
        start_time = g.get("startTime", "")
        home_b2b = g.get("home_b2b", False)
        away_b2b = g.get("away_b2b", False)
        for abbr, opp in [(home_abbr, away_abbr), (away_abbr, home_abbr)]:
            is_home = (abbr == home_abbr)
            game_lookup[abbr] = {
                "gameId": game_id, "opponent": opp, "spread": spread,
                "total": total, "startTime": start_time,
                "home": home_abbr, "away": away_abbr,
                "is_b2b": home_b2b if is_home else away_b2b,
                "opp_b2b": away_b2b if is_home else home_b2b,
            }

    stat_types = ["points", "rebounds", "assists"]
    candidates = []
    # Diagnostic counters for pipeline visibility
    _f = {"total": 0, "no_id": 0, "low_min": 0, "injury": 0, "no_game": 0,
          "blowout": 0, "gtd": 0, "skipped_pool": 0, "high_cv": 0, "low_games": 0,
          "no_odds": 0, "low_line": 0, "bad_line": 0, "no_juice": 0, "no_log": 0, "low_conf": 0,
          "switch_heavy": 0, "fake_juice": 0, "accepted": 0}
    _pool = set(gamelog_player_ids) if gamelog_player_ids is not None else None

    # Auto-fade config — use `or {}` so a null JSON value doesn't crash the subsequent .get() calls
    auto_fade_cfg = cfg.get("auto_fade") or {}
    _switch_heavy = set(t.upper() for t in auto_fade_cfg.get("switch_heavy_teams", []))
    _fake_juice_recent = float(auto_fade_cfg.get("fake_juice_recent_threshold", 0.80))
    _fake_juice_season = float(auto_fade_cfg.get("fake_juice_season_ceiling", 0.55))

    for p in projections:
        pid = str(p.get("id", ""))
        name = p.get("name", "")
        team = p.get("team", "")
        name_lower = name.lower()
        _f["total"] += 1

        if not pid or not name or not team:
            _f["no_id"] += 1
            continue

        # Season minutes floor
        season_min = float(p.get("season_min") or p.get("predMin") or 0)
        if season_min < min_season_minutes:
            _f["low_min"] += 1
            continue

        # Injury filter — skip OUT or questionable players
        injury = (p.get("injury_status") or "").lower()
        if injury in ("out", "questionable"):
            _f["injury"] += 1
            continue

        # Game context
        gctx = game_lookup.get(team)
        if not gctx:
            _f["no_game"] += 1
            continue

        # Blowout filter
        if _is_blowout_game(gctx.get("spread"), max_spread):
            _f["blowout"] += 1
            continue

        # GTD star teammate filter
        if _has_gtd_star_teammate(team, rw_statuses, name_lower):
            _f["gtd"] += 1
            continue

        if _pool is not None and pid not in _pool:
            _f["skipped_pool"] += 1
            continue

        # Player position (used by auto-fade filters)
        _pos_raw = (p.get("position") or p.get("pos") or "").upper()

        # Minutes volatility filter
        player_log = gamelogs.get(pid, {})
        log_minutes = player_log.get("minutes", [])
        cv = _minutes_cv(log_minutes)
        if cv > max_minutes_cv:
            _f["high_cv"] += 1
            continue

        # Games played filter (need enough data for reliable std dev)
        games_in_log = len(log_minutes)
        if games_in_log < min_games_played:
            _f["low_games"] += 1
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
                # Diagnostic: first time we see a player without odds, log it
                if not odds_data and _f.get("no_odds", 0) < 2:
                    available_for_stat = [k for k, v in player_odds_map.items() if k[1] == stat]
                    print(f"[parlay] debug: {name_lower} ({stat}) not found. Available keys for {stat}: {len(available_for_stat)}")
            if not odds_data:
                _f["no_odds"] += 1
                continue  # No sportsbook line → can't build a parlay leg

            # Ensure float preservation throughout pipeline (6.5, 21.5, 8.5 must not become 6, 21, 8)
            # Guard against null line values from the Odds API ({"line": null} is valid and common
            # when a sportsbook pulls a prop off the board mid-session).
            raw_line = odds_data.get("line")
            if raw_line is None or float(raw_line) <= 0:
                _f["no_odds"] += 1
                continue
            book_line = float(round(float(raw_line) * 2) / 2)  # Snap to nearest 0.5

            # Minimum line floor — filter trivially easy props (e.g. 0.5 ast, 1.5 reb)
            line_floor = _LINE_FLOORS.get(stat, 0)
            if book_line < line_floor:
                _f["low_line"] = _f.get("low_line", 0) + 1
                continue

            # Sanity check: reject lines that deviate >40% from projection.
            # A line far below projection (e.g. 12.5 pts when projecting 21) is almost
            # certainly bad odds data (alternate line, early placeholder, mismatched market).
            # These produce near-100% model confidence — inflating ranking artificially.
            if proj_val > 0 and book_line < proj_val * 0.60:
                _f["bad_line"] += 1
                continue
            if proj_val > 0 and book_line > proj_val * 1.80:
                _f["bad_line"] += 1
                continue

            # Determine direction: pick the side where model + Vegas agree
            for direction in ("over", "under"):
                # Auto-fade: centers rebounds over vs switch-heavy defenses
                if (stat == "rebounds" and direction == "over"
                        and _pos_raw in ("C", "PF", "C-PF", "PF-C")
                        and gctx.get("opponent", "").upper() in _switch_heavy):
                    _f["switch_heavy"] += 1
                    continue

                odds_key = f"odds_{direction}"
                american_odds = odds_data.get(odds_key)
                vegas_prob = _american_to_implied(american_odds)

                # Juice filter: Vegas must be pricing this direction at juice_threshold or more negative
                # (more juiced = Vegas thinks this side is more likely)
                if american_odds is not None:
                    try:
                        if float(american_odds) > juice_threshold:
                            _f["no_juice"] += 1
                            continue  # Not juiced enough — Vegas doesn't favor this side
                    except (TypeError, ValueError):
                        _f["no_juice"] += 1
                        continue

                # Compute model hit probability via Z-score
                log_values = player_log.get(stat, [])
                if len(log_values) < 5:
                    _f["no_log"] += 1
                    continue
                std_dev = _std_dev(log_values)
                model_prob = _compute_hit_probability(proj_val, book_line, std_dev, direction)
                if fair_value_data:
                    fp = fair_value_data.get(pid) or fair_value_data.get(str(pid))
                    if fp and isinstance(fp, dict):
                        hp = (fp.get("_fv_hit_probs") or {}).get(stat) or {}
                        try:
                            if direction == "over" and hp.get("over") is not None:
                                model_prob = float(hp["over"])
                            elif direction == "under" and hp.get("under") is not None:
                                model_prob = float(hp["under"])
                        except (TypeError, ValueError):
                            pass
                if model_prob is None:
                    _f["no_log"] += 1
                    continue

                # Blended confidence
                blended = _compute_blended_confidence(model_prob, vegas_prob)
                if blended is None:
                    _f["low_conf"] += 1
                    continue

                # DvP confidence nudge: position-specific defensive quality
                if dvp_data:
                    _opp = gctx.get("opponent", "")
                    _raw_pos = (p.get("pos") or p.get("position") or "").upper()
                    _pos_group = "G" if _raw_pos in ("PG", "SG", "G") else ("C" if _raw_pos == "C" else "F")
                    _DVP_AVG = {"G": 26.0, "F": 23.0, "C": 20.0}
                    if _opp and _opp in dvp_data and _pos_group in dvp_data[_opp]:
                        _dvp_pts = dvp_data[_opp][_pos_group]
                        _dvp_avg = _DVP_AVG.get(_pos_group, 23.0)
                        # Weak defense (+10% above avg) → +0.02 confidence for overs, -0.02 for unders
                        # Elite defense (-10% below avg) → reverse
                        _dvp_delta = (_dvp_pts - _dvp_avg) / _dvp_avg  # fraction above/below avg
                        _nudge = round(min(0.03, max(-0.03, _dvp_delta * 0.15)), 3)
                        if direction == "over":
                            blended = round(blended + _nudge, 4)
                        else:
                            blended = round(blended - _nudge, 4)

                if blended < min_blended_conf:
                    _f["low_conf"] += 1
                    continue

                # Auto-fade: fake juice trap — high recent hit rate but model says regression likely
                if direction == "over" and len(log_values) >= 5:
                    _recent_hits = sum(1 for v in log_values[-10:] if v > book_line)
                    _recent_total = min(len(log_values), 10)
                    _recent_hit_rate = _recent_hits / _recent_total if _recent_total > 0 else 0
                    if _recent_hit_rate >= _fake_juice_recent and model_prob < _fake_juice_season:
                        _f["fake_juice"] += 1
                        continue

                # Raw edge
                edge = proj_val - book_line if direction == "over" else book_line - proj_val

                _f["accepted"] += 1
                candidates.append({
                    "player_name": name,
                    "player_id": pid,
                    "team": team,
                    "opponent": gctx.get("opponent", ""),
                    "gameId": gctx.get("gameId", ""),
                    "stat_type": stat,
                    "direction": direction,
                    "line": float(book_line),  # Explicit float to prevent JSON truncation (6.5 not 6)
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
                    "recent_values": [round(v, 1) for v in log_values[-5:]],
                    "avg_min": round(float(p.get("season_min") or p.get("predMin") or 0), 1),
                    "season_pts": float(p.get("season_pts") or p.get("pts") or 0),
                    "season_ast": float(p.get("season_ast") or p.get("ast") or 0),
                    "position": (p.get("position") or "").upper(),
                    "game_spread": gctx.get("spread"),
                    "game_total": gctx.get("total"),
                    "game_time": gctx.get("startTime", ""),
                    "home_team": gctx.get("home", ""),
                    "away_team": gctx.get("away", ""),
                    "is_b2b": bool(gctx.get("is_b2b")),
                    "opp_b2b": bool(gctx.get("opp_b2b")),
                    "season_reb": float(p.get("season_reb") or p.get("reb") or 0),
                })

    print(f"[parlay] filter funnel: {_f}")
    # Diagnostic: show which players made it through and why some didn't
    if candidates:
        sample_cands = candidates[:2]
        print(f"[parlay] sample candidates: {[(c.get('player_name'), c.get('stat_type'), c.get('direction')) for c in sample_cands]}")
    return candidates, _f


def _std_dev(values):
    """Sample standard deviation of a list of numbers (Bessel's correction, divides by n-1).
    Using sample std dev because we're working with small slices (L5-L15 games), not a full
    population. Population std dev (/ n) artificially lowers dispersion on small samples,
    inflating hit probability estimates."""
    if not values or len(values) < 2:
        return 0.0
    n = len(values)
    avg = sum(values) / n
    variance = sum((v - avg) ** 2 for v in values) / (n - 1)  # Bessel's correction
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
    pnr_rim_boost = cfg.get("pnr_rim_boost", 1.20)
    pace_boost_threshold = cfg.get("pace_boost_total_threshold", 232.0)
    pace_boost = cfg.get("pace_boost", 1.06)
    rest_adv_boost = cfg.get("rest_advantage_boost", 1.08)
    auto_fade_cfg = cfg.get("auto_fade") or {}
    perimeter_reb_floor = float(auto_fade_cfg.get("perimeter_scorer_reb_floor", 4.0))

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

    # ── POSITIVE: PG Assists Over + teammate Points Over ──
    # Tiered: PnR-to-rim (interior finisher C/PF with 7+ reb) gets stronger boost
    for team, team_legs in by_team.items():
        ast_overs = [l for l in team_legs if l["stat_type"] == "assists" and l["direction"] == "over"]
        pts_overs = [l for l in team_legs if l["stat_type"] == "points" and l["direction"] == "over"]
        if ast_overs and pts_overs:
            a_name = ast_overs[0]["player_name"]
            p_leg = pts_overs[0]
            p_name = p_leg["player_name"]
            p_pos = p_leg.get("position", "")
            p_reb = float(p_leg.get("season_reb", 0))
            # Interior finisher: C/PF with 7+ rebounds → PnR-to-rim synergy (most stable)
            is_interior = p_pos in ("C", "PF", "C-PF", "PF-C") and p_reb >= 7.0
            # Perimeter-only scorer: low rebounds → fragile 3PT-dependent correlation
            is_perimeter = p_reb < perimeter_reb_floor and p_pos in ("SG", "SF", "G", "GF", "SG-SF")
            if is_interior:
                multiplier *= pnr_rim_boost
                reasons.append(f"PnR-to-rim synergy: {a_name} assists feed {p_name} interior finishing ({pnr_rim_boost}x)")
            elif is_perimeter:
                multiplier *= 0.95
                reasons.append(f"Risk: perimeter-to-perimeter — {p_name} scoring depends on 3PT variance")
            else:
                multiplier *= positive_boost
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

    # ── POSITIVE: Pace asymmetry — high-total games inflate counting stats ──
    _pace_applied = set()
    for gid, game_legs in by_game.items():
        if gid in _pace_applied:
            continue
        total = game_legs[0].get("game_total")
        if total is not None and float(total) >= pace_boost_threshold:
            multiplier *= pace_boost
            _pace_applied.add(gid)
            reasons.append(f"Pace boost: high-total game ({total} O/U ≥ {pace_boost_threshold})")

    # ── POSITIVE: Rest advantage — team rested vs opponent on B2B ──
    for team, team_legs in by_team.items():
        # Check if this team has a rest advantage (not on B2B, opponent is)
        sample = team_legs[0]
        if not sample.get("is_b2b") and sample.get("opp_b2b"):
            multiplier *= rest_adv_boost
            reasons.append(f"Rest advantage: {team} rested vs opponent on back-to-back ({rest_adv_boost}x)")

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


def _build_structured_narrative(legs, correlation_reasons, combined_prob):
    """Build a narrative for a prescriptive structured parlay (Market Match + Correlated Pair)."""
    parts = []
    pct = round(combined_prob * 100, 1)
    parts.append(f"This structured 3-leg parlay has a blended hit probability of {pct}%.")

    if len(legs) >= 1:
        mm = legs[0]
        stat_label = _STAT_LABEL.get(mm["stat_type"], mm["stat_type"].upper())
        juice = mm.get("american_odds", "")
        juice_str = f" (juiced to {juice})" if juice else ""
        parts.append(
            f"The Market Match: {mm['player_name']} {mm['direction'].upper()} "
            f"{mm['line']} {stat_label}{juice_str} — "
            f"{round(mm['blended_confidence'] * 100, 1)}% confidence from a matchup advantage."
        )

    if len(legs) >= 3:
        ast_leg = legs[1]
        pts_leg = legs[2]
        spread = abs(float(ast_leg.get("game_spread") or 0))
        parts.append(
            f"The Correlated Pair: {ast_leg['player_name']} Assists OVER {ast_leg['line']} "
            f"feeds {pts_leg['player_name']} Points OVER {pts_leg['line']} "
            f"in a tight {spread:.1f}-pt spread game."
        )

    if correlation_reasons:
        parts.append(" ".join(correlation_reasons))

    return " ".join(parts)


# ── Main Engine ──────────────────────────────────────────────────────────────

def _is_playmaker(leg):
    """Identify if a player is a playmaker (point guard or high-assist player).
    Used to find the 'correlated pair' anchor — the PG whose assists feed a teammate."""
    pos = leg.get("position", "")
    season_ast = leg.get("season_ast", 0)
    # PG position or averages 4+ assists per game (playmaker)
    return ("PG" in pos) or ("G" in pos and season_ast >= 4.0) or (season_ast >= 5.0)


def _score_structure(legs, parlay_config=None):
    """Score a 3-leg combination for how well it matches the ideal parlay structure.

    Ideal structure:
      Leg 1 (Market Match): Role player stat OVER with high confidence + Vegas juice
      Leg 2 (Correlated Pair A): Playmaker assists OVER in tight-spread game
      Leg 3 (Correlated Pair B): That playmaker's teammate points OVER

    Returns (structure_bonus, structure_reasons) where bonus is a multiplier >= 1.0.
    """
    cfg = parlay_config or {}
    reasons = []
    bonus = 1.0

    # Check for a correlated pair: same-team assists_over + points_over
    by_team = {}
    for leg in legs:
        by_team.setdefault(leg["team"], []).append(leg)

    has_correlated_pair = False
    for team, team_legs in by_team.items():
        ast_overs = [l for l in team_legs if l["stat_type"] == "assists" and l["direction"] == "over"]
        pts_overs = [l for l in team_legs if l["stat_type"] == "points" and l["direction"] == "over"]
        if ast_overs and pts_overs:
            has_correlated_pair = True
            a_name = ast_overs[0]["player_name"]
            p_name = pts_overs[0]["player_name"]
            # Stronger bonus if the assist player is a playmaker
            if _is_playmaker(ast_overs[0]):
                bonus *= 1.50
                reasons.append(f"Ideal structure: {a_name} assists feed {p_name} scoring")
            else:
                bonus *= 1.30
                reasons.append(f"Correlated pair: {a_name} assists + {p_name} scoring (same team)")

    # Penalize duplicate stat types (even cross-team) — diversification matters
    stat_counts = {}
    for l in legs:
        stat_counts[l["stat_type"]] = stat_counts.get(l["stat_type"], 0) + 1
    has_duplicate_stat = any(v >= 2 for v in stat_counts.values())
    if has_duplicate_stat:
        bonus *= 0.65
        reasons.append("Penalty: duplicate stat types reduce diversification")

    # Reward stat diversity — ideal parlay covers different stat types
    stat_types = set(l["stat_type"] for l in legs)
    if len(stat_types) >= 3:
        bonus *= 1.10
        reasons.append("Full stat diversification (PTS + REB + AST)")
    elif len(stat_types) >= 2:
        bonus *= 1.05
        reasons.append("Stat diversification across 2 categories")

    # Reward including a "market match" leg (Vegas heavily juiced, high confidence, low CV)
    mm_max_cv = cfg.get("market_match_max_cv", 0.25)
    has_market_match = any(
        l.get("american_odds") is not None
        and float(l.get("american_odds", 0)) <= -140
        and l["blended_confidence"] >= 0.60
        and l.get("minutes_cv", 0) <= mm_max_cv  # CV gate: volatile ≠ reliable
        for l in legs
    )
    if has_market_match:
        bonus *= 1.12
        reasons.append("Market match: strong Vegas alignment + stable rotation on at least one leg")

    # Penalize assists overs in wide-spread games (PG gets benched in blowouts)
    pair_spread_threshold = cfg.get("correlated_pair_max_spread", 5.0)
    for l in legs:
        if l["stat_type"] == "assists" and l["direction"] == "over":
            spread = abs(float(l.get("game_spread") or 0))
            if spread > pair_spread_threshold:
                bonus *= 0.88
                reasons.append(f"Risk: {l['player_name']} assists over in {spread}-pt spread game")

    # B2B penalty on correlated pairs — fatigue breaks assists→points chain
    b2b_penalty = float((cfg.get("auto_fade") or {}).get("b2b_correlated_pair_penalty", 0.75))
    for team, team_legs in by_team.items():
        ast_overs = [l for l in team_legs if l["stat_type"] == "assists" and l["direction"] == "over"]
        pts_overs = [l for l in team_legs if l["stat_type"] == "points" and l["direction"] == "over"]
        if ast_overs and pts_overs:
            if ast_overs[0].get("is_b2b"):
                bonus *= b2b_penalty
                reasons.append(f"Risk: correlated pair on B2B — fatigue disrupts assists+points correlation")

    # Penalize correlated pairs in low-total games (insufficient possessions)
    min_total = cfg.get("min_game_total", 225.5)
    for team, team_legs in by_team.items():
        ast_overs = [l for l in team_legs if l["stat_type"] == "assists" and l["direction"] == "over"]
        pts_overs = [l for l in team_legs if l["stat_type"] == "points" and l["direction"] == "over"]
        if ast_overs and pts_overs:
            total = ast_overs[0].get("game_total")
            if total is not None and float(total) < min_total:
                bonus *= 0.85
                reasons.append(f"Risk: low game total ({total} < {min_total}) — fewer possessions for pair")

    # Mild bonus for all-over combos (correlated with game pace)
    all_overs = all(l["direction"] == "over" for l in legs)
    if all_overs:
        bonus *= 1.06
        reasons.append("All overs: correlated with game pace")

    # Reward spread diversity — legs from different games reduce correlated risk
    game_ids = set(l["gameId"] for l in legs)
    if len(game_ids) >= 2:
        bonus *= 1.05

    # Penalize all legs from the same team (too correlated)
    teams = set(l["team"] for l in legs)
    if len(teams) == 1:
        bonus *= 0.80
        reasons.append("Warning: all legs from same team")

    return round(bonus, 4), reasons


def _find_best_correlated_pair(pool, cfg=None):
    """Find the best PG-assists-over + teammate-points-over correlated pair.

    Returns (assists_leg, points_leg, score) or None if no valid pair exists.
    """
    cfg = cfg or {}
    max_spread = cfg.get("correlated_pair_max_spread", 5.0)
    min_total = cfg.get("min_game_total", 225.5)
    b2b_penalty = float((cfg.get("auto_fade") or {}).get("b2b_correlated_pair_penalty", 0.75))

    # Collect all assists-over legs from playmakers in non-blowout, tight-spread, high-total games
    ast_overs = [
        l for l in pool
        if l["stat_type"] == "assists"
        and l["direction"] == "over"
        and _is_playmaker(l)
        and (l.get("game_spread") is None or abs(float(l.get("game_spread", 0))) <= max_spread)
        and (l.get("game_total") is None or float(l.get("game_total", 222)) >= min_total)
    ]
    if not ast_overs:
        return None

    best = None
    best_score = -1
    for ast_leg in ast_overs:
        # Find same-team points-over legs (potential lob targets / teammates)
        pts_overs = [
            l for l in pool
            if l["team"] == ast_leg["team"]
            and l["stat_type"] == "points"
            and l["direction"] == "over"
            and l["player_id"] != ast_leg["player_id"]
        ]
        for pts_leg in pts_overs:
            spread = abs(float(ast_leg.get("game_spread") or 0))
            # Tighter spread = more game time = better for both legs
            spread_bonus = 1.15 if spread <= 3.0 else (1.08 if spread <= 5.0 else 1.0)
            score = ast_leg["blended_confidence"] * pts_leg["blended_confidence"] * spread_bonus
            # B2B penalty: fatigue disrupts assists→points chain
            if ast_leg.get("is_b2b"):
                score *= b2b_penalty
            if score > best_score:
                best_score = score
                best = (ast_leg, pts_leg, score)

    return best


def _find_best_market_match(pool, pair, cfg=None):
    """Find the best market-match leg from a different game than the correlated pair.

    Market match = role player stat OVER with heavy Vegas juice and high model confidence.
    CV gate: rejects volatile players regardless of juice (prevents "fake juice" trap).
    Dynamic substitution: rebounds deprioritized vs elite defensive teams.
    Returns leg dict or None.
    """
    cfg = cfg or {}
    juice_threshold = cfg.get("market_match_juice_threshold", -140)
    min_conf = cfg.get("market_match_min_conf", 0.58)
    max_cv = cfg.get("market_match_max_cv", 0.25)
    auto_fade_cfg = cfg.get("auto_fade") or {}
    rebound_fade = set(t.upper() for t in auto_fade_cfg.get("rebound_fade_teams", []))

    pair_game_id = pair[0]["gameId"]
    pair_player_ids = {pair[0]["player_id"], pair[1]["player_id"]}

    def _mm_eligible(l, juice_thresh):
        if l["gameId"] == pair_game_id:
            return False
        if l["player_id"] in pair_player_ids:
            return False
        if l["direction"] != "over":
            return False
        if l["blended_confidence"] < min_conf:
            return False
        # CV gate: volatile players are unreliable for "sure thing" Market Match
        if l.get("minutes_cv", 0) > max_cv:
            return False
        odds = l.get("american_odds")
        if odds is not None:
            try:
                if float(odds) > juice_thresh:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    # Filter: different game, different players, over direction, meets juice + confidence + CV
    eligible = [l for l in pool if _mm_eligible(l, juice_threshold)]

    if not eligible:
        # Relax juice threshold to -120 if nothing at -140 (keep CV gate)
        relaxed_juice = cfg.get("market_match_juice_relaxed", -120)
        eligible = [l for l in pool if _mm_eligible(l, relaxed_juice)]

    if not eligible:
        # Final fallback: any over leg from a different game with decent confidence (relax CV)
        for l in pool:
            if l["gameId"] == pair_game_id:
                continue
            if l["player_id"] in pair_player_ids:
                continue
            if l["direction"] != "over":
                continue
            if l["blended_confidence"] >= min_conf:
                eligible.append(l)

    if not eligible:
        return None

    # Dynamic Leg 1 substitution: deprioritize rebounds vs elite defensive teams
    def _mm_sort(l):
        # Rebounds preferred generally, but faded vs top defensive teams
        if l["stat_type"] == "rebounds" and l.get("opponent", "").upper() in rebound_fade:
            stat_pref = 2  # worst priority — elite defense suppresses boards
        elif l["stat_type"] == "rebounds":
            stat_pref = 0  # preferred (matchup-driven)
        else:
            stat_pref = 1
        return (stat_pref, -l["blended_confidence"])

    eligible.sort(key=_mm_sort)
    return eligible[0]


def run_parlay_engine(projections, games, player_odds_map, gamelogs,
                      rw_statuses=None, parlay_config=None, gamelog_player_ids=None, dvp_data=None,
                      fair_value_data=None):
    """Find the optimal 3-leg parlay from today's slate.

    Uses a PRESCRIPTIVE builder that targets the ideal 3-leg structure:
      Leg 1 (Market Match): Role player stat OVER with Vegas juice + high model confidence
      Leg 2 (Correlated Pair A): Playmaker assists OVER in tight-spread game
      Leg 3 (Correlated Pair B): That playmaker's teammate's points OVER

    Falls back to combinatorial scoring if the ideal structure can't be built.

    Args:
        projections: Player projections from the pipeline.
        games: Today's games.
        player_odds_map: Odds API data.
        gamelogs: {player_id: {"points": [...], "rebounds": [...], "assists": [...], "minutes": [...]}}
        rw_statuses: RotoWire availability statuses.
        parlay_config: Config overrides.

    Returns:
        dict with legs, probabilities, correlation, structure info, or None.
    """
    cfg = parlay_config or {}

    # Step 1: Build all valid candidate legs
    candidates, filter_funnel = build_candidate_legs(
        projections, games, player_odds_map, gamelogs,
        rw_statuses, cfg, gamelog_player_ids=gamelog_player_ids, dvp_data=dvp_data,
        fair_value_data=fair_value_data,
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

    # Sort by blended confidence descending; limit to top N
    max_candidates = cfg.get("max_candidates_for_combinations", 25)
    deduped.sort(key=lambda x: x["blended_confidence"], reverse=True)
    pool = deduped[:max_candidates]

    # ── Step 2: TRY PRESCRIPTIVE STRUCTURE (Market Match + Correlated Pair) ──
    structured_result = None
    pair_data = _find_best_correlated_pair(pool, cfg)
    if pair_data:
        ast_leg, pts_leg, pair_score = pair_data
        market_match = _find_best_market_match(pool, (ast_leg, pts_leg), cfg)
        if market_match:
            combo = [market_match, ast_leg, pts_leg]
            # Verify no VETO from correlation check
            corr_mult, corr_reasons = _correlation_modifier(combo, cfg)
            if corr_mult > 0:
                struct_bonus, struct_reasons = _score_structure(combo, cfg)
                combined_prob = 1.0
                for l in combo:
                    combined_prob *= l["blended_confidence"]
                score = combined_prob * corr_mult * struct_bonus
                all_reasons = corr_reasons + struct_reasons
                narrative = _build_structured_narrative(combo, all_reasons, combined_prob * corr_mult * struct_bonus)
                structured_result = {
                    "legs": combo,
                    "combined_probability": round(combined_prob, 4),
                    "correlation_multiplier": round(corr_mult * struct_bonus, 4),
                    "correlation_reasons": all_reasons,
                    "parlay_score": round(score, 6),
                    "narrative": narrative,
                    "candidates_evaluated": len(candidates),
                    "combinations_scored": 1,
                    "structured": True,
                    "filter_funnel": filter_funnel,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                print(f"[parlay] structured build: market={market_match['player_name']} ({market_match['stat_type']}), "
                      f"pair={ast_leg['player_name']} AST + {pts_leg['player_name']} PTS, "
                      f"score={score:.4f}")

    if structured_result:
        return structured_result

    print("[parlay] no ideal structure found — falling back to combinatorial scoring")

    # ── Step 3: FALLBACK — combinatorial scoring (all 3-leg combos) ──
    best_score = -1
    best_combo = None
    best_corr_mult = 1.0
    best_corr_reasons = []
    best_struct_bonus = 1.0
    best_struct_reasons = []
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

        # Structure bonus — rewards the ideal 3-leg profile
        struct_bonus, struct_reasons = _score_structure(legs, cfg)

        # Combined probability = product of individual blended confidences
        raw_prob = 1.0
        for l in legs:
            raw_prob *= l["blended_confidence"]

        # Final score: probability × correlation × structure
        score = raw_prob * corr_mult * struct_bonus

        combos_scored += 1

        if score > best_score:
            best_score = score
            best_combo = legs
            best_corr_mult = corr_mult
            best_corr_reasons = corr_reasons
            best_struct_bonus = struct_bonus
            best_struct_reasons = struct_reasons

    if best_combo is None:
        return None

    # Sort legs for display: correlated pair together, market match first
    _team_count = {}
    for l in best_combo:
        _team_count[l["team"]] = _team_count.get(l["team"], 0) + 1

    def _leg_sort_key(l):
        pair_team = _team_count.get(l["team"], 0) > 1
        return (pair_team, -l["blended_confidence"])

    best_combo.sort(key=_leg_sort_key)

    combined_prob = 1.0
    for l in best_combo:
        combined_prob *= l["blended_confidence"]

    all_reasons = best_corr_reasons + best_struct_reasons
    narrative = _build_parlay_narrative(best_combo, all_reasons, combined_prob * best_corr_mult * best_struct_bonus)

    return {
        "legs": best_combo,
        "combined_probability": round(combined_prob, 4),
        "correlation_multiplier": round(best_corr_mult * best_struct_bonus, 4),
        "correlation_reasons": all_reasons,
        "parlay_score": round(best_score, 6),
        "narrative": narrative,
        "candidates_evaluated": len(candidates),
        "combinations_scored": combos_scored,
        "structured": False,
        "filter_funnel": filter_funnel,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
