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
    # Filter 1: Slate classification
    "pitcher_day_threshold": 5,
    "hitter_day_ou_threshold": 9.0,
    "hitter_day_game_count": 5,
    "tiny_slate_max_games": 3,
    # Filter 2: Environmental scoring
    "ace_era_threshold": 3.50,
    "ace_k9_threshold": 8.0,
    "weak_starter_era": 4.50,
    "high_total_threshold": 8.5,
    "elite_total_threshold": 9.5,
    "debut_return_env_bonus": 10,
    "debut_games_threshold": 5,
    # Filter 3: Ownership leverage
    "ghost_drafts_threshold": 100,
    "chalk_drafts_threshold": 2000,
    "ownership_leverage_weights": {
        "ghost": 0.50,
        "low": 0.30,
        "contrarian": 0.15,
        "neutral": 0.0,
        "chalk": -0.30,
    },
    "big_market_teams_mlb": ["NYY", "LAD", "NYM", "BOS", "CHC", "PHI", "HOU", "ATL", "SF", "SD"],
    # Filter 4: Boost optimization
    "boost_trap_env_threshold": 30,
    "boost_leveraged_min_boost": 2.5,
    "boost_leveraged_min_env": 50,
    # Filter 5: Lineup construction
    "max_same_team": 3,
    "max_same_team_tiny": 4,
    "min_games_represented": 2,
    "team_stack_bonus": 1.20,
    "team_stack_min_ml": -150,
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

    Strategy doc Section 2.1 — pitcher RS correlates with:
      - Facing weak offense (bottom-10 K% or wOBA): +25
      - Pitcher-friendly park (factor <= 96): +15
      - Pitching at home: +10
      - High K/9 (>9.0: +20, >8.0: +10) — K upside is PRIMARY RS driver
      - Confirmed probable starter: +10
      - Low ERA (<3.00: +15, <3.50: +10)
      - Team favored (moneyline < -150): +10 (Condition A: run support)
      - Debut/return premium: +10 (Condition C)
      - Weather factor: +5/-5 (pitcher-friendly cold/wind-in)
    """
    score = 0.0

    # ── Opponent weakness (top signal for pitcher RS) ─────────────────────
    if opp_team_stats:
        opp_k_rate = _sf(opp_team_stats.get("team_k_rate", 0))
        opp_woba = _sf(opp_team_stats.get("team_woba", 0))
        if opp_k_rate >= 0.25 or (opp_woba > 0 and opp_woba <= 0.300):
            score += 25

    # ── Park factor ───────────────────────────────────────────────────────
    if park_factor <= 96:
        score += 15
    elif park_factor <= 98:
        score += 5  # Mild pitcher-friendly bonus

    # ── Home advantage ────────────────────────────────────────────────────
    if is_home:
        score += 10

    # ── K/9 upside — "strikeouts are the primary RS driver for pitchers" ──
    k9 = _sf(pitcher_stats.get("k9", 0))
    if k9 > 10.0:
        score += 25  # Elite K upside
    elif k9 > 9.0:
        score += 20
    elif k9 > 8.0:
        score += 10

    # ── Confirmed probable starter ────────────────────────────────────────
    if pitcher_stats.get("is_probable_starter") or pitcher_stats.get("is_starter"):
        score += 10

    # ── ERA quality ───────────────────────────────────────────────────────
    era = _sf(pitcher_stats.get("era", 99))
    if era < 2.50:
        score += 20  # Elite ace
    elif era < 3.00:
        score += 15
    elif era < 3.50:
        score += 10

    # ── WHIP quality (additional signal) ──────────────────────────────────
    whip = _sf(pitcher_stats.get("whip", 99))
    if whip < 1.00:
        score += 5

    # ── Team favored — Condition A: "Starting Pitcher in a Winning Team" ──
    # Every pitcher with RS >= 5.0 was on a team that won. Run support matters.
    team = canonicalize_team(pitcher_stats.get("team", ""))
    home_team = canonicalize_team(game.get("home", ""))
    if team == home_team:
        ml = _sf(game.get("home_moneyline", 0))
    else:
        ml = _sf(game.get("away_moneyline", 0))
    if ml < -150:
        score += 10
    elif ml < -120:
        score += 5

    # ── Debut/return premium — Condition C ────────────────────────────────
    if pitcher_stats.get("is_debut_or_return", False):
        score += 10

    # ── Weather factor (pitcher-friendly conditions boost) ────────────────
    weather_factor = _sf(game.get("weather_factor", 1.0))
    if weather_factor < 0.98:
        score += 5  # Cold/wind-in favors pitchers

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

    Strategy doc Section 2.2 + Conditions B-E:
      - High Vegas total (>=9.5: +25, >=8.5: +15)
      - Weak opposing starter (ERA >4.50: +20, >4.00: +10)
      - Platoon advantage (opposite hand): +15
      - Top of lineup (1-4: +20, 5-6: +10)
      - Hitter-friendly park (factor >= 104): +15
      - Team favored (moneyline < -150): +10
      - Debut/return premium: +10 (Condition C)
      - Weather factor: +5 (warm/wind-out)
      - Opposing pitcher high HR/9: +5
    """
    cfg = config or {}
    score = 0.0

    # ── Vegas total — Condition B: "Hitters in High-Run Games" ────────────
    # "Hitters virtually never achieve RS > 5.0 in games with < 6 total runs"
    total = _sf(game.get("total", 0))
    elite_total = _cfg_val(cfg, "elite_total_threshold", 9.5)
    high_total = _cfg_val(cfg, "high_total_threshold", 8.5)
    if total >= elite_total:
        score += 25
    elif total >= high_total:
        score += 15
    elif total > 0 and total < 7.0:
        score -= 10  # Low-total games are hitter deserts

    # ── Opposing starter quality ──────────────────────────────────────────
    if opp_pitcher and isinstance(opp_pitcher, dict):
        opp_era = _sf(opp_pitcher.get("era", 0))
        weak_era = _cfg_val(cfg, "weak_starter_era", 4.50)
        if opp_era > 5.00:
            score += 25  # Very weak starter
        elif opp_era > weak_era:
            score += 20
        elif opp_era > 4.00:
            score += 10

        # Platoon advantage — Condition D
        batter_bats = hitter_stats.get("bats", "")
        pitcher_throws = opp_pitcher.get("hand", opp_pitcher.get("throws", ""))
        if has_platoon_advantage(batter_bats, pitcher_throws):
            score += 15

        # Opposing pitcher HR vulnerability
        opp_hr9 = _sf(opp_pitcher.get("hr9", 0))
        if opp_hr9 > 1.5:
            score += 5

    # ── Batting order — more PA = more RS opportunity ─────────────────────
    if 1 <= batting_order <= 4:
        score += 20
    elif 5 <= batting_order <= 6:
        score += 10
    elif 7 <= batting_order <= 9:
        score += 3  # Still in lineup, small bonus

    # ── Park factor ───────────────────────────────────────────────────────
    if park_factor >= 110:
        score += 20  # Coors-level
    elif park_factor >= 104:
        score += 15
    elif park_factor >= 101:
        score += 5

    # ── Team favored ──────────────────────────────────────────────────────
    team = canonicalize_team(hitter_stats.get("team", ""))
    home = canonicalize_team(game.get("home", ""))
    if team == home:
        ml = _sf(game.get("home_moneyline", 0))
    else:
        ml = _sf(game.get("away_moneyline", 0))
    if ml < -200:
        score += 15  # Heavy favorite — Condition E: team stack potential
    elif ml < -150:
        score += 10

    # ── Debut/return premium — Condition C ────────────────────────────────
    # "Debut players have near-zero draft ownership... yet they often perform
    # at elite levels due to adrenaline, preparation, and opponent unfamiliarity."
    if hitter_stats.get("is_debut_or_return", False):
        score += 10

    # ── Weather factor (hitter-friendly conditions) ───────────────────────
    weather_factor = _sf(game.get("weather_factor", 1.0))
    if weather_factor > 1.02:
        score += 5  # Warm/wind-out favors hitters

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

def _detect_two_way(candidate: dict) -> bool:
    """Detect two-way players (Ohtani playbook).

    Strategy doc "Two-Way Day": Ohtani is almost always the highest-EV
    Slot 1 or 2 when pitching and hitting. Two-way players get pitcher
    environmental scoring PLUS hitter batting order bonus.
    """
    is_p = candidate.get("is_pitcher", False)
    # Two-way: pitcher who also bats (non-zero batting stats)
    if is_p:
        hr = _sf(candidate.get("hr", 0))
        rbi = _sf(candidate.get("rbi", 0))
        avg = _sf(candidate.get("avg", 0))
        if hr >= 3 or rbi >= 10 or avg >= 0.200:
            return True
    return False


def build_optimal_lineup(
    candidates: list[dict],
    slate_type: str,
    config: dict | None = None,
    exclude_names: set | None = None,
    prefer_contrarian: bool = False,
) -> list[dict]:
    """Build optimal 5-player lineup using the "Filter, Not Forecast" strategy.

    # grep: LINEUP CONSTRUCTION

    Strategy doc Section 4.2 — The Decision Tree + 10 Commandments:

    Slot sequencing (Section 3.4):
      Slot 1 (2.0x): Highest-conviction play.
        - Unboosted ace vs weakest offense, OR
        - 3.0-boosted player in best environment
      Slot 2 (1.8x): Second-highest conviction (backup ace or boosted)
      Slots 3-5: Boosted players (slot penalty is minimal: only 16% for 3.0x boost)

    Constraints (Commandments 2, 7, 10):
      - At least 2 different games represented (Commandment 10)
      - Boost diversification: boosted players across 2-3 games (Commandment 10)
      - No more than 3 players from same team (relaxed to 4 for tiny/stack days)
      - Composition follows day-type rules (pitcher/hitter mix)
      - Never draft boost without environment (Commandment 6)

    Team stacking (Condition E):
      For tiny/stack days, actively seek 3-4 from the heavy favorite.

    Two-way players (Ohtani playbook):
      Two-way players combine pitcher + hitter environmental advantages.
    """
    cfg = config or {}
    _excl = exclude_names or set()
    slot_mults = _cfg_val(cfg, "slot_multipliers", [2.0, 1.8, 1.6, 1.4, 1.2])
    slot_labels = _cfg_val(cfg, "slot_labels", ["2.0x", "1.8x", "1.6x", "1.4x", "1.2x"])
    min_games = _cfg_val(cfg, "min_games_represented", 2)
    min_env = _cfg_val(cfg, "min_environment_score", 20)

    # Team cap: relaxed for tiny/stack days (Condition E: team stack correlation)
    max_same_team = _cfg_val(cfg, "max_same_team", 3)
    if slate_type == "tiny":
        max_same_team = max(max_same_team, 4)  # Allow 4-1 team split on tiny slates

    comp = _cfg_val(cfg, "composition", MLB_STRATEGY_DEFAULTS["composition"])
    day_comp = comp.get(slate_type, comp.get("standard", {}))
    min_pitchers = day_comp.get("min_pitchers", 0)
    max_pitchers = day_comp.get("max_pitchers", 5)

    # ── Tag two-way players ───────────────────────────────────────────────
    for c in candidates:
        c["is_two_way"] = _detect_two_way(c)

    # ── Filter available candidates ───────────────────────────────────────
    available = [
        c for c in candidates
        if c.get("name") not in _excl
        and _sf(c.get("environment_score", 0)) >= min_env
        and not c.get("boost_trap", False)
        and not c.get("is_out", False)
    ]

    if not available:
        print(f"[mlb_strategy] WARNING: No candidates pass environment filter (min_env={min_env})")
        # Fallback: relax to min_env/2
        available = [
            c for c in candidates
            if c.get("name") not in _excl
            and not c.get("is_out", False)
        ]

    # ── Sort candidates ───────────────────────────────────────────────────
    if prefer_contrarian:
        # Moonshot: prefer ghost players with environmental support
        # Strategy doc: "The contrarian principle: If a player has the environmental
        # conditions for a big day but low ownership, they are the highest-EV pick"
        available.sort(key=lambda x: (
            -_sf(x.get("ownership_leverage", 0)),
            -_sf(x.get("boost_ev", 0)),
            -_sf(x.get("environment_score", 0)),
            -_sf(x.get("is_debut_or_return", False)),  # Debut edge
        ))
    else:
        # Starting 5: sort by expected value (environment-weighted)
        available.sort(key=lambda x: (
            -_sf(x.get("expected_value", 0)),
            -_sf(x.get("environment_score", 0)),
            -_sf(x.get("boost_ev", 0)),
        ))

    # ── Team stacking for tiny/stack days (Condition E) ───────────────────
    # "When one hitter on a team has a big game, teammates often do too"
    # For tiny slates, identify the heavy favorite and boost their candidates
    if slate_type == "tiny" and not prefer_contrarian:
        _apply_team_stack_bonus(available, candidates, cfg)

    # ── Greedy selection with constraints ─────────────────────────────────
    lineup: list[dict] = []
    team_counts: dict[str, int] = {}
    game_ids: set[str] = set()
    boost_game_ids: set[str] = set()  # Track which games have boosted players
    pitcher_count = 0

    def _can_add(c: dict) -> bool:
        nonlocal pitcher_count
        team = canonicalize_team(c.get("team", ""))
        gid = c.get("gameId", c.get("game_id", ""))
        is_p = c.get("is_pitcher", False)
        boost = _sf(c.get("boost", c.get("est_mult", 0)))

        # Team cap
        if team and team_counts.get(team, 0) >= max_same_team:
            return False

        # Pitcher composition constraints
        if is_p and pitcher_count >= max_pitchers:
            return False
        if not is_p and (5 - len(lineup)) <= (min_pitchers - pitcher_count) and pitcher_count < min_pitchers:
            return False

        # Boost diversification: don't stack all boosted players in one game
        # Commandment 10: "Diversify across games. A 2-1 pitcher's duel kills
        # concentrated lineups."
        if boost >= 2.0 and len(boost_game_ids) >= 1 and gid in boost_game_ids:
            # Already have a boosted player from this game
            # Allow if we have <2 games represented yet, or if no better option
            if len(game_ids) >= 2:
                return False  # Force diversification

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
        if _sf(c.get("boost", c.get("est_mult", 0))) >= 2.0:
            boost_game_ids.add(gid)
        if c.get("is_pitcher", False):
            pitcher_count += 1

    # Select 5 players
    for c in available:
        if len(lineup) >= 5:
            break
        if _can_add(c):
            _add(c)

    # If we couldn't fill 5 due to strict constraints, relax boost diversification
    if len(lineup) < 5:
        for c in available:
            if len(lineup) >= 5:
                break
            if c.get("name") in {p.get("name") for p in lineup}:
                continue
            team = canonicalize_team(c.get("team", ""))
            if team and team_counts.get(team, 0) >= max_same_team:
                continue
            _add(c)

    # ── Ensure minimum games constraint ───────────────────────────────────
    # Commandment 10: "Never put all 5 players in the same game"
    if len(game_ids) < min_games and len(lineup) >= 5 and len(available) > 5:
        for swap_candidate in available:
            if swap_candidate.get("name") in {p.get("name") for p in lineup}:
                continue
            swap_gid = swap_candidate.get("gameId", swap_candidate.get("game_id", ""))
            if swap_gid and swap_gid not in game_ids:
                lineup[-1] = swap_candidate
                game_ids.add(swap_gid)
                break

    # ── SLOT ASSIGNMENT (Strategy Doc Section 3.3-3.4) ────────────────────
    # "Place your highest-conviction unboosted player in Slot 1."
    # "Distribute boosted players across remaining slots freely."
    #
    # The optimal construction rule:
    #   1. Unboosted players MUST go in top slots (67% value loss Slot 1→5)
    #   2. Boosted players are slot-flexible (only 16% loss for 3.0x boost)
    #   3. Two-way players get Slot 1 or 2 (combined pitcher+hitter advantage)
    #
    # Sort priority: two-way → unboosted by env → boosted by EV
    def _slot_priority(p):
        boost = _sf(p.get("boost", p.get("est_mult", 0)))
        ev = _sf(p.get("expected_value", 0))
        env = _sf(p.get("environment_score", 0))
        is_two_way = p.get("is_two_way", False)

        # Two-way players (Ohtani) → highest slot priority
        if is_two_way:
            return (0, -env, -ev)
        # Unboosted players → high slot priority (RS dominates slot value)
        if boost < 1.0:
            return (1, -env, -ev)
        # Medium boost (1.0-2.0) → middle slots
        if boost < 2.0:
            return (2, -ev, -env)
        # High boost (2.0+) → flexible, sort by EV
        return (3, -ev, -env)

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
          f"{len(game_ids)} games, slate_type={slate_type}"
          f"{' (contrarian)' if prefer_contrarian else ''}")

    return lineup


def _apply_team_stack_bonus(available: list[dict], all_candidates: list[dict],
                            config: dict) -> None:
    """Boost EV for players on the heaviest favorite team.

    # grep: TEAM STACK BONUS

    Strategy doc Condition E — "Team-Stack Correlation":
      When one hitter on a team has a big game, teammates often do too
      (because runs require baserunners). Winning lineups exploit this.

    For tiny slates (1-3 games), identify the heaviest moneyline favorite
    and boost their candidates' expected_value by +20% to encourage stacking.
    """
    # Find heaviest favorite team
    best_team = ""
    best_ml = 0
    for c in all_candidates:
        team = canonicalize_team(c.get("team", ""))
        # Check moneyline from the game data
        ml = _sf(c.get("_team_moneyline", 0))
        if ml < best_ml:
            best_ml = ml
            best_team = team

    if best_team and best_ml < -150:
        stack_bonus = 1.20  # 20% EV bonus for stacking
        for c in available:
            if canonicalize_team(c.get("team", "")) == best_team:
                c["expected_value"] = round(_sf(c.get("expected_value", 0)) * stack_bonus, 2)
                c["_team_stack"] = True
        print(f"[mlb_strategy] Team stack bonus: {best_team} (ML={best_ml})")


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

        # Inject team moneyline for team stack bonus (Condition E)
        if is_home:
            c["_team_moneyline"] = _sf(game.get("home_moneyline", 0))
        else:
            c["_team_moneyline"] = _sf(game.get("away_moneyline", 0))

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
