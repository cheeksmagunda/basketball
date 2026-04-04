"""MLB Draft Strategy — "Filter, Not Forecast" Pipeline.

# grep: MLB STRATEGY

An external-variables framework for daily MLB draft optimization.
Instead of predicting Real Score (RS), we identify CONDITIONS under which
high RS is most likely to emerge, then select from that filtered pool.

Philosophy: We do not predict RS. We identify the conditions under which
high RS is most likely to emerge, then select from that filtered pool.
This is analogous to a venture capital model: we don't know which startup
will succeed, but we know the market conditions, team profiles, and timing
patterns that correlate with breakout success.

The Five Filters (Applied Sequentially):
  1. Slate Architecture — What type of day is this?
  2. Environmental Advantage — Who has the conditions?
  3. Ownership Leverage — Who is the crowd ignoring?
  4. Boost Optimization — How to allocate boosts?
  5. Lineup Construction — Slot sequencing for max value.

Core formula (additive, NOT multiplicative):
  total_value = RS × (slot_multiplier + card_boost)

Key insight: The additive formula compresses the slot penalty for boosted
players. An unboosted player loses 40% of their value moving from Slot 1
(2.0x) to Slot 5 (1.2x). A 3.0x-boosted player only loses 16%.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

try:
    from api.mlb_data import (
        has_platoon_advantage, get_park_factor, is_pitcher_park,
        is_hitter_park, canonicalize_team,
    )
except ImportError:
    from .mlb_data import (
        has_platoon_advantage, get_park_factor, is_pitcher_park,
        is_hitter_park, canonicalize_team,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG DEFAULTS — All tunable via model-config.json "mlb_strategy" section
# ─────────────────────────────────────────────────────────────────────────────

MLB_STRATEGY_DEFAULTS: dict[str, Any] = {
    "slot_multipliers": [2.0, 1.8, 1.6, 1.4, 1.2],
    "slot_labels": ["2.0x", "1.8x", "1.6x", "1.4x", "1.2x"],
    "min_environment_score": 20,
    "pitcher_day_threshold": 5,
    "hitter_day_ou_threshold": 9.0,
    "hitter_day_game_count": 5,
    "tiny_slate_max_games": 3,
    "max_same_team": 3,
    "min_games_represented": 2,
    "ghost_drafts_threshold": 100,
    "chalk_drafts_threshold": 2000,
    "ace_era_threshold": 3.50,
    "ace_k9_threshold": 8.0,
    "weak_starter_era": 4.50,
    "high_total_threshold": 8.5,
    "elite_total_threshold": 9.5,
    "boost_trap_env_threshold": 30,
    "boost_leveraged_min_boost": 2.5,
    "boost_leveraged_min_env": 50,
    "ownership_leverage_weights": {
        "ghost": 0.50,
        "low": 0.30,
        "contrarian": 0.15,
        "neutral": 0.0,
        "chalk": -0.30,
    },
    "composition": {
        "tiny":        {"min_pitchers": 1, "max_pitchers": 1},
        "pitcher_day": {"min_pitchers": 4, "max_pitchers": 5},
        "hitter_day":  {"min_pitchers": 0, "max_pitchers": 1},
        "standard":    {"min_pitchers": 2, "max_pitchers": 3},
    },
}

# Day-type strategy labels and descriptions
_STRATEGY_META = {
    "tiny":        {"label": "Stack Day",      "description": "Heavy favorite team stack. 1P + 4 hitters."},
    "pitcher_day": {"label": "Ace Day",        "description": "4-5 pitchers. Best ace vs weakest offense in Slot 1."},
    "hitter_day":  {"label": "Bash Day",       "description": "4-5 hitters from highest-total games. Team stack priority."},
    "standard":    {"label": "Standard Build",  "description": "2-3 pitchers + 2-3 boosted hitters from favorable environments."},
}


def _cfg_val(config: dict, key: str, default: Any = None) -> Any:
    """Read from passed config with MLB_STRATEGY_DEFAULTS fallback."""
    if config and key in config:
        return config[key]
    return MLB_STRATEGY_DEFAULTS.get(key, default)


def _sf(v, default=0.0) -> float:
    """Safe float conversion."""
    if v is None:
        return default
    try:
        return float(v)
    except (ValueError, TypeError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# FILTER 1: SLATE CLASSIFICATION
# grep: SLATE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_slate(games: list[dict], team_stats: dict | None = None,
                   config: dict | None = None) -> str:
    """Classify the day's slate to determine optimal composition.

    Returns: 'tiny', 'pitcher_day', 'hitter_day', or 'standard'.

    Decision tree:
      - 1-3 games → tiny (limited pool, team-stack from favorite)
      - 5+ quality SP matchups → pitcher_day (aces dominate)
      - 5+ games with O/U >= 9.0 → hitter_day (run environment)
      - else → standard (mixed 2-3P + 2-3H)
    """
    cfg = config or {}
    n_games = len(games)

    if n_games <= _cfg_val(cfg, "tiny_slate_max_games", 3):
        print(f"[mlb_strategy] Slate classified: tiny ({n_games} games)")
        return "tiny"

    # Count quality SP matchups: ace (ERA < 3.50, K/9 > 8.0) in favorable spot
    ace_era = _cfg_val(cfg, "ace_era_threshold", 3.50)
    ace_k9 = _cfg_val(cfg, "ace_k9_threshold", 8.0)
    quality_sp_count = 0
    for g in games:
        for side in ("home_probable_pitcher", "away_probable_pitcher"):
            pitcher = g.get(side)
            if not pitcher or not isinstance(pitcher, dict):
                continue
            era = _sf(pitcher.get("era", 99))
            k9 = _sf(pitcher.get("k9", 0))
            if era < ace_era and k9 > ace_k9:
                quality_sp_count += 1

    if quality_sp_count >= _cfg_val(cfg, "pitcher_day_threshold", 5):
        print(f"[mlb_strategy] Slate classified: pitcher_day ({quality_sp_count} quality SP matchups)")
        return "pitcher_day"

    # Count high-total games for hitter day detection
    ou_thresh = _cfg_val(cfg, "hitter_day_ou_threshold", 9.0)
    high_total_games = sum(1 for g in games if _sf(g.get("total", 0)) >= ou_thresh)

    if high_total_games >= _cfg_val(cfg, "hitter_day_game_count", 5):
        print(f"[mlb_strategy] Slate classified: hitter_day ({high_total_games} high-total games)")
        return "hitter_day"

    print(f"[mlb_strategy] Slate classified: standard ({n_games} games, {quality_sp_count} aces, {high_total_games} high-total)")
    return "standard"


# ─────────────────────────────────────────────────────────────────────────────
# FILTER 2: ENVIRONMENTAL SCORING
# grep: ENVIRONMENTAL SCORING
# ─────────────────────────────────────────────────────────────────────────────

def score_pitcher_environment(
    pitcher_stats: dict,
    opp_team_stats: dict | None,
    game: dict,
    park_factor: int = 100,
    is_home: bool = False,
    config: dict | None = None,
) -> float:
    """Score a pitcher's environmental advantage (0-100).

    Factors:
      - Facing weak offense (bottom-10 K% or wOBA): +25
      - Pitcher-friendly park (factor <= 96): +15
      - Pitching at home: +10
      - High K/9 (>9.0: +20, >8.0: +10)
      - Confirmed probable starter: +10
      - Low ERA (<3.00: +15, <3.50: +10)
    """
    score = 0.0

    # Opponent weakness
    if opp_team_stats:
        opp_k_rate = _sf(opp_team_stats.get("team_k_rate", 0))
        opp_woba = _sf(opp_team_stats.get("team_woba", 0))
        # High K rate or low wOBA = weak offense
        if opp_k_rate >= 0.25 or (opp_woba > 0 and opp_woba <= 0.300):
            score += 25

    # Park factor
    if park_factor <= 96:
        score += 15

    # Home advantage
    if is_home:
        score += 10

    # K rate
    k9 = _sf(pitcher_stats.get("k9", 0))
    if k9 > 9.0:
        score += 20
    elif k9 > 8.0:
        score += 10

    # Confirmed starter
    if pitcher_stats.get("is_probable_starter") or pitcher_stats.get("is_starter"):
        score += 10

    # ERA quality
    era = _sf(pitcher_stats.get("era", 99))
    if era < 3.00:
        score += 15
    elif era < 3.50:
        score += 10

    return round(min(score, 100.0), 2)


def score_hitter_environment(
    hitter_stats: dict,
    opp_pitcher: dict | None,
    game: dict,
    park_factor: int = 100,
    batting_order: int = 0,
    config: dict | None = None,
) -> float:
    """Score a hitter's environmental advantage (0-100).

    Factors:
      - High Vegas total (>=9.5: +25, >=8.5: +15)
      - Weak opposing starter (ERA >4.50: +20, >4.00: +10)
      - Platoon advantage (opposite hand): +15
      - Top of lineup (1-4: +20, 5-6: +10)
      - Hitter-friendly park (factor >= 104): +15
      - Team favored (moneyline < -150): +10
    """
    cfg = config or {}
    score = 0.0

    # Vegas total
    total = _sf(game.get("total", 0))
    elite_total = _cfg_val(cfg, "elite_total_threshold", 9.5)
    high_total = _cfg_val(cfg, "high_total_threshold", 8.5)
    if total >= elite_total:
        score += 25
    elif total >= high_total:
        score += 15

    # Opposing starter quality
    if opp_pitcher and isinstance(opp_pitcher, dict):
        opp_era = _sf(opp_pitcher.get("era", 0))
        weak_era = _cfg_val(cfg, "weak_starter_era", 4.50)
        if opp_era > weak_era:
            score += 20
        elif opp_era > 4.00:
            score += 10

        # Platoon advantage
        batter_bats = hitter_stats.get("bats", "")
        pitcher_throws = opp_pitcher.get("hand", opp_pitcher.get("throws", ""))
        if has_platoon_advantage(batter_bats, pitcher_throws):
            score += 15

    # Batting order
    if 1 <= batting_order <= 4:
        score += 20
    elif 5 <= batting_order <= 6:
        score += 10

    # Park factor
    if park_factor >= 104:
        score += 15

    # Team favored
    # Determine team's moneyline from game data
    team = canonicalize_team(hitter_stats.get("team", ""))
    home = canonicalize_team(game.get("home", ""))
    if team == home:
        ml = _sf(game.get("home_moneyline", 0))
    else:
        ml = _sf(game.get("away_moneyline", 0))
    if ml < -150:
        score += 10

    return round(min(score, 100.0), 2)


# ─────────────────────────────────────────────────────────────────────────────
# FILTER 3: OWNERSHIP LEVERAGE
# grep: OWNERSHIP LEVERAGE
# ─────────────────────────────────────────────────────────────────────────────

def compute_ownership_leverage(
    drafts: float,
    environment_score: float = 0,
    config: dict | None = None,
) -> float:
    """Compute ownership leverage multiplier.

    The most consistent edge comes from players who are barely drafted
    but deliver elite RS. These are invisible to the field.

    Tiers:
      - drafts > 2000: -0.30 (chalk penalty — crowd chases names)
      - drafts 500-2000: 0.0 (neutral)
      - drafts 100-500: +0.15 (contrarian edge)
      - drafts < 100: +0.30 (ghost player edge)
      - drafts < 20 AND env_score > 50: +0.50 (maximum ghost edge)

    Returns: additive leverage multiplier.
    """
    cfg = config or {}
    weights = _cfg_val(cfg, "ownership_leverage_weights",
                       MLB_STRATEGY_DEFAULTS["ownership_leverage_weights"])
    ghost_thresh = _cfg_val(cfg, "ghost_drafts_threshold", 100)
    chalk_thresh = _cfg_val(cfg, "chalk_drafts_threshold", 2000)

    d = _sf(drafts, 0)

    if d < 20 and environment_score > 50:
        return float(weights.get("ghost", 0.50))
    elif d < ghost_thresh:
        return float(weights.get("low", 0.30))
    elif d < 500:
        return float(weights.get("contrarian", 0.15))
    elif d < chalk_thresh:
        return float(weights.get("neutral", 0.0))
    else:
        return float(weights.get("chalk", -0.30))


# ─────────────────────────────────────────────────────────────────────────────
# FILTER 4: BOOST OPTIMIZATION
# grep: BOOST OPTIMIZATION
# ─────────────────────────────────────────────────────────────────────────────

def optimize_boost_allocation(
    candidates: list[dict],
    config: dict | None = None,
) -> list[dict]:
    """Optimize boost allocation by flagging traps and leveraged plays.

    Rules:
      - boost_trap: boost >= 2.0 AND environment_score < 30 (amplifies downside)
      - boost_leveraged: boost >= 2.5 AND environment_score >= 50 (sweet spot)
      - boost_ev: boost * (env_score / 100.0) to rank boost efficiency

    The boost is a multiplier on an unknown outcome, not a signal of that outcome.
    A 3.0 boost on a player who scores -3.0 RS produces -15.0 in Slot 1.
    Never chase a boost without environment.
    """
    cfg = config or {}
    trap_thresh = _cfg_val(cfg, "boost_trap_env_threshold", 30)
    lev_min_boost = _cfg_val(cfg, "boost_leveraged_min_boost", 2.5)
    lev_min_env = _cfg_val(cfg, "boost_leveraged_min_env", 50)

    for c in candidates:
        boost = _sf(c.get("boost", c.get("est_mult", 0)))
        env = _sf(c.get("environment_score", 0))

        c["boost_trap"] = boost >= 2.0 and env < trap_thresh
        c["boost_leveraged"] = boost >= lev_min_boost and env >= lev_min_env
        c["boost_ev"] = round(boost * (env / 100.0), 2) if env > 0 else 0.0

    # Sort by boost_ev descending, then environment_score, then boost
    candidates.sort(key=lambda x: (
        -x.get("boost_ev", 0),
        -_sf(x.get("environment_score", 0)),
        -_sf(x.get("boost", x.get("est_mult", 0))),
    ))

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# FILTER 5: LINEUP CONSTRUCTION
# grep: LINEUP CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def build_optimal_lineup(
    candidates: list[dict],
    slate_type: str,
    config: dict | None = None,
    exclude_names: set | None = None,
    prefer_contrarian: bool = False,
) -> list[dict]:
    """Build optimal 5-player lineup using filter-based selection.

    Slot sequencing:
      Slot 1 (2.0x): Highest conviction — unboosted ace or best-env 3.0-boosted
      Slot 2 (1.8x): Second-highest conviction
      Slots 3-5: Fill with remaining, prefer boosted (slot penalty is minimal)

    Constraints:
      - At least 2 different games represented
      - No more than 3 players from same team
      - Composition follows day-type rules (pitcher/hitter mix)

    Expected value formula (since we don't predict RS):
      ev = environment_score * (slot_mult + card_boost) * (1 + ownership_leverage)
    """
    cfg = config or {}
    _excl = exclude_names or set()
    slot_mults = _cfg_val(cfg, "slot_multipliers", [2.0, 1.8, 1.6, 1.4, 1.2])
    slot_labels = _cfg_val(cfg, "slot_labels", ["2.0x", "1.8x", "1.6x", "1.4x", "1.2x"])
    max_same_team = _cfg_val(cfg, "max_same_team", 3)
    min_games = _cfg_val(cfg, "min_games_represented", 2)
    min_env = _cfg_val(cfg, "min_environment_score", 20)

    comp = _cfg_val(cfg, "composition", MLB_STRATEGY_DEFAULTS["composition"])
    day_comp = comp.get(slate_type, comp.get("standard", {}))
    min_pitchers = day_comp.get("min_pitchers", 0)
    max_pitchers = day_comp.get("max_pitchers", 5)

    # Filter available candidates
    available = [
        c for c in candidates
        if c.get("name") not in _excl
        and _sf(c.get("environment_score", 0)) >= min_env
        and not c.get("boost_trap", False)
        and not c.get("is_out", False)
    ]

    if prefer_contrarian:
        # For moonshot: prefer low-ownership, high-variance plays
        available.sort(key=lambda x: (
            -_sf(x.get("ownership_leverage", 0)),
            -_sf(x.get("boost_ev", 0)),
            -_sf(x.get("environment_score", 0)),
        ))
    else:
        # Default: sort by expected value
        available.sort(key=lambda x: (
            -_sf(x.get("expected_value", 0)),
            -_sf(x.get("environment_score", 0)),
            -_sf(x.get("boost_ev", 0)),
        ))

    # Greedy selection with constraints
    lineup = []
    team_counts: dict[str, int] = {}
    game_ids: set[str] = set()
    pitcher_count = 0

    def _can_add(c: dict) -> bool:
        nonlocal pitcher_count
        team = canonicalize_team(c.get("team", ""))
        gid = c.get("gameId", c.get("game_id", ""))
        is_p = c.get("is_pitcher", False)

        # Team cap
        if team and team_counts.get(team, 0) >= max_same_team:
            return False

        # Pitcher composition constraints
        if is_p and pitcher_count >= max_pitchers:
            return False
        if not is_p and (5 - len(lineup)) <= (min_pitchers - pitcher_count) and pitcher_count < min_pitchers:
            # Need to reserve remaining slots for pitchers
            return False

        return True

    def _add(c: dict):
        nonlocal pitcher_count
        lineup.append(c)
        team = canonicalize_team(c.get("team", ""))
        if team:
            team_counts[team] = team_counts.get(team, 0) + 1
        gid = c.get("gameId", c.get("game_id", ""))
        if gid:
            game_ids.add(gid)
        if c.get("is_pitcher", False):
            pitcher_count += 1

    # Select 5 players
    for c in available:
        if len(lineup) >= 5:
            break
        if _can_add(c):
            _add(c)

    # Ensure minimum games constraint — swap last player if needed
    if len(game_ids) < min_games and len(lineup) >= 5 and len(available) > 5:
        for swap_candidate in available:
            if swap_candidate.get("name") in {p.get("name") for p in lineup}:
                continue
            swap_gid = swap_candidate.get("gameId", swap_candidate.get("game_id", ""))
            if swap_gid and swap_gid not in game_ids:
                # Replace last player with this one to diversify games
                lineup[-1] = swap_candidate
                game_ids.add(swap_gid)
                break

    # Assign slots: highest expected value → Slot 1 (2.0x), etc.
    # Key insight from strategy doc: unboosted players MUST go in top slots.
    # Boosted players are slot-flexible (16% loss vs 40% for unboosted).
    # Sort by: unboosted first to top slots, then by expected value
    def _slot_priority(p):
        boost = _sf(p.get("boost", p.get("est_mult", 0)))
        ev = _sf(p.get("expected_value", 0))
        env = _sf(p.get("environment_score", 0))
        # Unboosted high-env players go to Slot 1 (RS dominates there)
        if boost < 0.5:
            return (-env, -ev)
        # Boosted players can go anywhere — sort by EV
        return (0, -ev)

    lineup.sort(key=_slot_priority)

    for i, p in enumerate(lineup):
        if i < len(slot_mults):
            p["slot"] = slot_labels[i]
            p["slot_multiplier"] = slot_mults[i]
        else:
            p["slot"] = "1.0x"
            p["slot_multiplier"] = 1.0

    print(f"[mlb_strategy] Built lineup: {len(lineup)} players, "
          f"{pitcher_count}P/{len(lineup)-pitcher_count}H, "
          f"{len(game_ids)} games, slate_type={slate_type}")

    return lineup


# ─────────────────────────────────────────────────────────────────────────────
# MASTER PIPELINE
# grep: FILTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_filter_pipeline(
    games: list[dict],
    all_candidates: list[dict],
    team_stats: dict | None = None,
    config: dict | None = None,
) -> dict:
    """Orchestrate the 5-filter pipeline end-to-end.

    Step 1: Classify slate type
    Step 2: Score all candidates (environment)
    Step 3: Compute ownership leverage
    Step 4: Optimize boost allocation
    Step 5: Build Starting 5 (safe) and Moonshot (contrarian)

    Returns:
      {
        "starting_5": [player_dicts with slots],
        "moonshot": [player_dicts with slots],
        "slate_type": str,
        "strategy_label": str,
        "strategy_description": str,
        "candidates_scored": int,
        "candidates_filtered": int,
      }
    """
    cfg = config or {}
    ts = team_stats or {}
    slot_mults = _cfg_val(cfg, "slot_multipliers", [2.0, 1.8, 1.6, 1.4, 1.2])

    # ── Step 1: Classify slate ─────────────────────────────────────────────
    slate_type = classify_slate(games, ts, cfg)
    meta = _STRATEGY_META.get(slate_type, _STRATEGY_META["standard"])

    # ── Step 2: Score all candidates ───────────────────────────────────────
    # Build game lookup for fast access
    game_lookup = {}
    for g in games:
        game_lookup[g.get("gameId", "")] = g

    scored = 0
    for c in all_candidates:
        gid = c.get("gameId", c.get("game_id", ""))
        game = game_lookup.get(gid, {})
        pk = get_park_factor(game.get("home", ""))
        team = canonicalize_team(c.get("team", ""))
        is_home = team == canonicalize_team(game.get("home", ""))

        if c.get("is_pitcher", False):
            # Get opposing team stats
            opp_team = game.get("away" if is_home else "home", "")
            opp_stats = ts.get(opp_team, {})
            c["environment_score"] = score_pitcher_environment(
                c, opp_stats, game, pk, is_home, cfg
            )
        else:
            # Get opposing pitcher
            if is_home:
                opp_pitcher = game.get("away_probable_pitcher")
            else:
                opp_pitcher = game.get("home_probable_pitcher")
            batting_order = int(_sf(c.get("batting_order", c.get("order", 0))))
            c["environment_score"] = score_hitter_environment(
                c, opp_pitcher, game, pk, batting_order, cfg
            )
        scored += 1

    print(f"[mlb_strategy] Scored {scored} candidates")

    # ── Step 3: Ownership leverage ─────────────────────────────────────────
    for c in all_candidates:
        drafts = _sf(c.get("drafts", 0))
        env = _sf(c.get("environment_score", 0))
        c["ownership_leverage"] = compute_ownership_leverage(drafts, env, cfg)

    # ── Step 4: Boost optimization ─────────────────────────────────────────
    all_candidates = optimize_boost_allocation(all_candidates, cfg)

    # ── Compute expected value for each candidate ──────────────────────────
    # Since we don't predict RS, use environment as proxy:
    # ev = env_score * (avg_slot_mult + boost) * (1 + ownership_leverage)
    avg_slot = sum(slot_mults) / len(slot_mults)  # 1.6
    for c in all_candidates:
        env = _sf(c.get("environment_score", 0))
        boost = _sf(c.get("boost", c.get("est_mult", 0)))
        leverage = _sf(c.get("ownership_leverage", 0))
        c["expected_value"] = round(
            env * (avg_slot + boost) * (1 + leverage), 2
        )

    # Filter below minimum environment
    min_env = _cfg_val(cfg, "min_environment_score", 20)
    filtered = [c for c in all_candidates if _sf(c.get("environment_score", 0)) >= min_env]
    filtered_count = len(all_candidates) - len(filtered)

    print(f"[mlb_strategy] {len(filtered)} candidates passed env filter "
          f"({filtered_count} filtered out)")

    # ── Step 5: Build lineups ──────────────────────────────────────────────
    starting_5 = build_optimal_lineup(filtered, slate_type, cfg)
    s5_names = {p.get("name") for p in starting_5}

    # Moonshot: exclude Starting 5, prefer contrarian plays
    moonshot = build_optimal_lineup(
        filtered, slate_type, cfg,
        exclude_names=s5_names,
        prefer_contrarian=True,
    )

    return {
        "starting_5": starting_5,
        "moonshot": moonshot,
        "slate_type": slate_type,
        "strategy_label": meta["label"],
        "strategy_description": meta["description"],
        "candidates_scored": scored,
        "candidates_filtered": filtered_count,
    }
