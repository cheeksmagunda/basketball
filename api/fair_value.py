# ─────────────────────────────────────────────────────────────────────────────
# grep: FAIR VALUE ENGINE — project_player_fv, compute_fv_hit_probs
# FAIR VALUE ENGINE — Pure deterministic projections + EV vs books
# No I/O, no HTTP. Callers pass gamelog arrays, DvP dicts, odds slices, config.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from api.odds_math import american_to_implied

# Default league anchors (align with _compute_matchup_factor in index.py)
_DVP_LEAGUE_AVG = {"G": 26.0, "F": 23.0, "C": 20.0}
_DEFAULT_TOTAL = 222.0

_STAT_KEYS = (
    "points",
    "rebounds",
    "assists",
    "steals",
    "blocks",
    "threes",
    "field_goals_attempted",
    "turnovers",
    "minutes",
)

# ── Game Script Tiers ─────────────────────────────────────────────────────
# Callers MUST pass gs_config from model-config.json (via _cfg("game_script")).
# No hardcoded defaults here — single source of truth is _CONFIG_DEFAULTS
# in index.py / data/model-config.json.  If gs_config is None the function
# uses minimal safe fallbacks so pure-unit-test paths don't crash.
_GAME_SCRIPT_FALLBACK = {
    "defensive_grind_ceiling": 220,
    "balanced_ceiling": 235,
    "fast_paced_ceiling": 245,
    "blowout_spread_threshold": 8,
    "blowout_pts_penalty": 0.90,
    "blowout_ast_penalty": 0.90,
    "blowout_reb_penalty": 0.94,
    "defensive_grind": {"pts": 0.85, "reb": 0.90, "ast": 0.85, "stl": 1.40, "blk": 1.35, "tov": 1.15},
    "balanced":        {"pts": 1.0,  "reb": 1.0,  "ast": 1.0,  "stl": 1.05, "blk": 1.05, "tov": 1.0},
    "fast_paced":      {"pts": 1.15, "reb": 1.10, "ast": 1.15, "stl": 0.95, "blk": 0.95, "tov": 0.90},
    "track_meet":      {"pts": 1.25, "reb": 1.05, "ast": 1.20, "stl": 0.90, "blk": 0.90, "tov": 0.85},
}


def game_script_weights(
    total: float,
    spread: float,
    gs_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Per-stat multipliers based on game total/spread (game script tier).

    Pure function version of _game_script_weights in index.py.
    Returns dict with keys: pts, reb, ast, stl, blk, tov.
    """
    gs = gs_config or _GAME_SCRIPT_FALLBACK
    t = total or _DEFAULT_TOTAL

    dg_ceil = float(gs.get("defensive_grind_ceiling", 220))
    bal_ceil = float(gs.get("balanced_ceiling", 235))
    fp_ceil = float(gs.get("fast_paced_ceiling", 245))
    blow_thr = float(gs.get("blowout_spread_threshold", 8))

    if t < dg_ceil:
        tier = "defensive_grind"
    elif t <= bal_ceil:
        tier = "balanced"
    elif t <= fp_ceil:
        tier = "fast_paced"
    else:
        tier = "track_meet"

    defaults = _GAME_SCRIPT_FALLBACK[tier]
    tier_cfg = gs.get(tier, defaults)
    w = {k: float(tier_cfg.get(k, defaults.get(k, 1.0))) for k in ["pts", "reb", "ast", "stl", "blk", "tov"]}

    if tier == "track_meet" and abs(spread or 0) > blow_thr:
        w["pts"] *= float(gs.get("blowout_pts_penalty", 0.90))
        w["ast"] *= float(gs.get("blowout_ast_penalty", 0.90))
        w["reb"] *= float(gs.get("blowout_reb_penalty", 0.94))

    return w


def spread_adjustment(
    spread: float,
    total: float,
    pts: float,
    avg_min: float,
    bench_pts_threshold: float = 14.0,
    bench_min_threshold: float = 30.0,
    starter_blowout_floor: float = 0.70,
) -> float:
    """RS-clutch-aligned spread adjustment (mirrors heuristic path in index.py).

    RS algorithm heavily weights game closeness: tight games → every play matters.
    Stars get big bonus in tight games, penalty in blowouts.
    Bench players get bonus in blowouts (garbage time minutes).
    Returns multiplier typically in [starter_blowout_floor, 1.35].
    starter_blowout_floor: minimum multiplier for starters in heavy-spread games (default 0.70).
    """
    abs_spread = abs(spread or 0)
    is_bench = pts <= bench_pts_threshold and avg_min <= bench_min_threshold

    if is_bench:
        if abs_spread <= 4:
            s_adj = 1.0
        else:
            s_adj = min(1.15, 1.0 + (abs_spread - 4) * 0.02)
    else:
        if abs_spread <= 3:
            s_adj = 1.25 - (abs_spread * 0.03)
        elif abs_spread <= 7:
            s_adj = 1.16 - ((abs_spread - 3) * 0.04)
        else:
            s_adj = max(starter_blowout_floor, 1.0 - (abs_spread - 7) * 0.09)

    # Total interaction: high total + tight spread = shootout bonus
    _total = total or _DEFAULT_TOTAL
    if not is_bench and _total >= 230 and abs_spread <= 5:
        s_adj *= 1.0 + min(0.10, (_total - 230) * 0.005)

    return s_adj


def momentum_adjustment(
    rolling_stats: Dict[str, Any],
    strength: float = 0.15,
) -> float:
    """Scoring momentum from L3 vs L10 trend.

    Players on a hot streak (L3 >> L10) get a mild RS uplift.
    Players cooling off (L3 << L10) get a mild downgrade.
    Returns multiplier in [0.92, 1.12].
    """
    pts_d = rolling_stats.get("points") or {}
    momentum = float(pts_d.get("L3_vs_L10_momentum", 1.0))
    # Clamp momentum ratio to reasonable range
    momentum = max(0.5, min(2.0, momentum))
    # Convert to mild adjustment: (1.2 - 1.0) * 0.15 = +0.03 for 20% hot streak
    adj = 1.0 + (momentum - 1.0) * strength
    return max(0.92, min(1.12, adj))


def _pos_binary_bucket(pos: str) -> str:
    """Map roster position to guards vs forwards bucket (binary DvP split)."""
    p = (pos or "").strip().upper()
    if p in ("PG", "SG", "G") or p.startswith("G"):
        return "guards"
    return "forwards"


def _std_norm_cdf(z: float) -> float:
    """Standard normal CDF via erf."""
    if z > 8.0:
        return 1.0
    if z < -8.0:
        return 0.0
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_cdf(x: float, mu: float, sigma: float) -> float:
    """CDF of Normal(mu, sigma^2) at x."""
    if sigma is None or sigma < 1e-9:
        return 0.5 if x >= mu else 0.0
    return _std_norm_cdf((x - mu) / sigma)


def compute_rolling_stats(
    gamelog: Dict[str, List[float]],
    window: int = 15,
    short_window: int = 10,
) -> Dict[str, Any]:
    """Rolling window statistics per stat type from ESPN-style gamelog dict.

    gamelog keys: points, rebounds, assists, minutes, (+ optional steals, blocks, ...).
    """
    out: Dict[str, Any] = {}
    minutes = gamelog.get("minutes") or []
    for stat in _STAT_KEYS:
        if stat == "minutes":
            vals = minutes
        else:
            vals = gamelog.get(stat) or []
        if not vals:
            out[stat] = {
                "mean_L10": 0.0,
                "mean_L15": 0.0,
                "std_L10": 0.0,
                "p20": 0.0,
                "p80": 0.0,
                "recent_min": 0.0,
                "recent_max": 0.0,
                "per_minute_rate": 0.0,
                "L3_vs_L10_momentum": 1.0,
            }
            continue
        w = min(window, len(vals))
        sw = min(short_window, len(vals))
        arr = np.array(vals[-w:], dtype=float)
        sarr = np.array(vals[-sw:], dtype=float)
        mean_L15 = float(np.mean(arr))
        mean_L10 = float(np.mean(sarr))
        std_L10 = float(np.std(sarr, ddof=0)) if len(sarr) > 1 else 0.0
        p20 = float(np.percentile(arr, 20))
        p80 = float(np.percentile(arr, 80))
        recent_min = float(np.min(arr))
        recent_max = float(np.max(arr))
        # Per-minute: per-game stat/minutes for games in window, then mean
        pm_rates = []
        if stat != "minutes":
            m_tail = minutes[-w:] if len(minutes) >= w else minutes[-len(vals) :]
            for i, sv in enumerate(vals[-w:]):
                mi = m_tail[i] if i < len(m_tail) else 0.0
                if mi and mi > 0:
                    pm_rates.append(float(sv) / float(mi))
            per_minute_rate = float(np.mean(pm_rates)) if pm_rates else 0.0
        else:
            per_minute_rate = 1.0
        l3 = vals[-3:] if len(vals) >= 3 else vals
        m_l3 = float(np.mean(l3)) if l3 else 0.0
        momentum = (m_l3 / mean_L10) if mean_L10 > 1e-6 else 1.0
        out[stat] = {
            "mean_L10": mean_L10,
            "mean_L15": mean_L15,
            "std_L10": std_L10,
            "p20": p20,
            "p80": p80,
            "recent_min": recent_min,
            "recent_max": recent_max,
            "per_minute_rate": per_minute_rate,
            "L3_vs_L10_momentum": float(min(2.0, max(0.5, momentum))),
        }
    # Minutes CV across window
    m_arr = np.array(minutes[-window:] if minutes else [], dtype=float)
    if len(m_arr) > 0 and float(np.mean(m_arr)) > 1e-6:
        minutes_cv = float(np.std(m_arr, ddof=0) / np.mean(m_arr))
    else:
        minutes_cv = 0.0
    out["_minutes_cv"] = minutes_cv
    out["_mean_minutes_L15"] = float(np.mean(m_arr)) if len(m_arr) else 0.0
    return out


def dvp_binary_from_nba_com(
    dvp_row: Optional[Dict[str, float]],
    league_avg: Optional[Dict[str, float]] = None,
) -> Tuple[float, float]:
    """Build pts_allowed_guards / pts_allowed_forwards from {G,F,C} DvP row.

    forwards ← average of F and C; guards ← G.
    """
    if not dvp_row:
        return 115.0, 115.0
    la = league_avg or _DVP_LEAGUE_AVG
    g = float(dvp_row.get("G", la["G"]))
    f = float(dvp_row.get("F", la["F"]))
    c = float(dvp_row.get("C", la["C"]))
    forwards = (f + c) / 2.0
    return g, forwards


def adjust_for_opponent(
    rolling_stats: Dict[str, Any],
    opp_def: Dict[str, Any],
    position: str,
) -> Dict[str, Any]:
    """Scale per-stat mean_L15 by opponent defensive quality (binary guards vs forwards).

    opp_def may contain:
      pts_allowed_guards, pts_allowed_forwards, league_avg_guards, league_avg_forwards
    Or caller passes dvp_g, dvp_f, league_g, league_f explicitly.
    """
    bucket = _pos_binary_bucket(position)
    pa_g = float(
        opp_def.get("pts_allowed_guards")
        or opp_def.get("dvp_guards")
        or 115.0
    )
    pa_f = float(
        opp_def.get("pts_allowed_forwards")
        or opp_def.get("dvp_forwards")
        or 115.0
    )
    lg_g = float(opp_def.get("league_avg_guards") or _DVP_LEAGUE_AVG["G"])
    lg_f = float(opp_def.get("league_avg_forwards") or (_DVP_LEAGUE_AVG["F"] + _DVP_LEAGUE_AVG["C"]) / 2.0)
    if bucket == "guards":
        ratio = pa_g / max(lg_g, 1e-6)
    else:
        ratio = pa_f / max(lg_f, 1e-6)
    ratio = float(min(1.25, max(0.80, ratio)))

    adjusted = dict(rolling_stats)
    for stat in _STAT_KEYS:
        if stat == "minutes":
            continue
        if stat not in adjusted or not isinstance(adjusted[stat], dict):
            continue
        d = dict(adjusted[stat])
        # Points/rebounds/assists get full DvP ratio; peripheral stats get milder nudge
        if stat == "points":
            m = ratio
        elif stat in ("rebounds", "assists"):
            m = 1.0 + (ratio - 1.0) * 0.85
        else:
            m = 1.0 + (ratio - 1.0) * 0.5
        for k in ("mean_L10", "mean_L15", "p20", "p80"):
            if k in d:
                d[k] = float(d[k]) * m
        adjusted[stat] = d
    return adjusted


def closeness_factor(
    spread: float,
    total: float,
    closeness_max: float = 1.5,
    default_total: float = _DEFAULT_TOTAL,
) -> float:
    """Closed-form P(|margin|<=5) with Normal(|spread|, 0.45*sqrt(total)); maps to [1.0, closeness_max]."""
    abs_spread = abs(spread or 0.0)
    t = float(total or default_total)
    sigma = 0.45 * math.sqrt(max(t, 1.0))
    mu = abs_spread
    # P(-5 <= M <= 5) for M ~ N(mu, sigma^2)
    p_close = _norm_cdf(5.0, mu, sigma) - _norm_cdf(-5.0, mu, sigma)
    p_close = float(min(1.0, max(0.0, p_close)))
    return 1.0 + p_close * (float(closeness_max) - 1.0)


def ats_regression_factor(
    team_recent_ats: Optional[float],
    window: int = 15,
    strength: float = 0.5,
) -> float:
    """Stub: no team ATS feed yet — neutral multiplier. Interface reserved."""
    _ = (team_recent_ats, window, strength)
    return 1.0


def should_cascade(
    out_player_name: str,
    out_player_ppg: float,
    config: Optional[Dict[str, Any]] = None,
) -> bool:
    """Elite-only vacancy inflation gate."""
    cfg = config or {}
    policy = (cfg.get("cascade_policy") or "elite_only").lower()
    if policy == "disabled":
        return False
    if policy == "all":
        return True
    threshold = float(cfg.get("elite_cascade_ppg", 27.0))
    elites = set(cfg.get("elite_players") or ())
    name = (out_player_name or "").strip()
    if name in elites:
        return True
    return float(out_player_ppg or 0.0) >= threshold


def compute_fair_value(
    projection: float,
    book_line: float,
    odds_over: Optional[int],
    odds_under: Optional[int],
    std_dev: float,
    pinnacle_line: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fair value, edge, hit probabilities, EV, edge classification."""
    cfg = config or {}
    thresholds = cfg.get("edge_thresholds") or {}
    min_edge_pct = float(thresholds.get("min_edge_pct", 5.0))
    min_ev = float(thresholds.get("min_ev", 0.03))
    sharp_bonus = float(thresholds.get("sharp_aligned_bonus", 1.15))

    line = float(book_line)
    proj = float(projection)
    edge_points = proj - line
    edge_pct = (edge_points / line * 100.0) if abs(line) > 1e-9 else 0.0
    sig = max(float(std_dev), 1e-6)
    # Hit probs: X ~ N(proj, sig^2)
    hit_over = 1.0 - _norm_cdf(line, proj, sig)
    hit_under = _norm_cdf(line, proj, sig)
    io = american_to_implied(odds_over if odds_over is not None else -110)
    iu = american_to_implied(odds_under if odds_under is not None else -110)
    if io is None or io <= 0:
        io = 0.5
    if iu is None or iu <= 0:
        iu = 0.5
    ev_over = (hit_over / io - 1.0) if io else 0.0
    ev_under = (hit_under / iu - 1.0) if iu else 0.0

    # Edge class
    edge_class = "no_edge"
    if abs(edge_pct) >= min_edge_pct or max(abs(ev_over), abs(ev_under)) >= min_ev:
        if pinnacle_line is not None:
            try:
                pl = float(pinnacle_line)
                if abs(proj - pl) <= abs(proj - line) * 0.5:
                    edge_class = "sharp_aligned"
                elif max(abs(ev_over), abs(ev_under)) >= min_ev * sharp_bonus:
                    edge_class = "model_only"
                else:
                    edge_class = "model_only"
            except (TypeError, ValueError):
                edge_class = "model_only"
        else:
            edge_class = "model_only"
    if edge_class == "no_edge" and max(abs(ev_over), abs(ev_under)) >= min_ev * 1.2:
        edge_class = "public_fade"

    return {
        "fair_value": proj,
        "book_line": line,
        "edge_pct": edge_pct,
        "edge_points": edge_points,
        "hit_prob_over": hit_over,
        "hit_prob_under": hit_under,
        "implied_prob_over": io,
        "implied_prob_under": iu,
        "ev_over": ev_over,
        "ev_under": ev_under,
        "edge_class": edge_class,
    }


def _dfs_from_stats(
    pts: float,
    reb: float,
    ast: float,
    stl: float,
    blk: float,
    tov: float,
    dfs_weights: Optional[Dict[str, float]] = None,
) -> float:
    w = dfs_weights or {
        "pts": 1.5,
        "reb": 0.5,
        "ast": 1.0,
        "stl": 3.0,
        "blk": 1.5,
        "tov": -1.0,
    }
    return (
        pts * w.get("pts", 1.5)
        + reb * w.get("reb", 0.5)
        + ast * w.get("ast", 1.0)
        + stl * w.get("stl", 3.0)
        + blk * w.get("blk", 1.5)
        + tov * w.get("tov", -1.0)
    )


def _stat_to_book_key(stat: str) -> str:
    m = {
        "points": "points",
        "rebounds": "rebounds",
        "assists": "assists",
        "steals": "steals",
        "blocks": "blocks",
        "threes": "threes",
        "pra": "points_rebounds_assists",
    }
    return m.get(stat, stat)


def project_player_fv(
    gamelog: Dict[str, List[float]],
    athlete_stats: Dict[str, Any],
    position: str,
    opp_def: Dict[str, Any],
    spread: float,
    total: float,
    side: str,
    book_lines: Optional[Dict[str, Dict[str, Any]]],
    config: Dict[str, Any],
    dfs_weights: Optional[Dict[str, float]] = None,
    gs_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Unified projection: rolling stats → DvP → game script → spread adj → momentum → RS rating + per-stat FV."""
    cfg = config or {}
    primary_window = int(cfg.get("primary_window", 15))
    short_w = int(cfg.get("short_window", 10))
    default_total = float(cfg.get("default_total", _DEFAULT_TOTAL))

    roll = compute_rolling_stats(gamelog, window=primary_window, short_window=short_w)
    roll_adj = adjust_for_opponent(roll, opp_def, position)
    home_adj = 1.02 if (side or "").lower() == "home" else 1.0

    # ── Game script weights (pace-tier stat multipliers) ──────────────────
    gs_w = game_script_weights(total, spread, gs_config=gs_config)

    # Base per-stat projections from mean_L15 (DvP-adjusted)
    def _gv(key: str) -> float:
        d = roll_adj.get(key) or {}
        return float(d.get("mean_L15", 0.0))

    pts = _gv("points")
    reb = _gv("rebounds")
    ast = _gv("assists")
    stl = _gv("steals")
    blk = _gv("blocks")
    threes = _gv("threes")
    tov = _gv("turnovers")

    # Minutes projection: athlete season/recent blend if present
    avg_min = float(athlete_stats.get("min") or roll.get("_mean_minutes_L15") or 0.0)
    if avg_min <= 0:
        avg_min = float(roll.get("_mean_minutes_L15") or 22.0)

    # Apply game script per-stat weights + home adjustment
    pts *= gs_w.get("pts", 1.0) * home_adj
    reb *= gs_w.get("reb", 1.0) * home_adj
    ast *= gs_w.get("ast", 1.0) * home_adj
    stl *= gs_w.get("stl", 1.0) * home_adj
    blk *= gs_w.get("blk", 1.0) * home_adj
    threes *= home_adj  # threes not in game_script weights
    tov *= gs_w.get("tov", 1.0)

    # ── Spread adjustment (RS-clutch-aligned, bench vs starter) ───────────
    s_adj = spread_adjustment(
        spread, total, pts, avg_min,
        bench_pts_threshold=float(cfg.get("bench_pts_threshold", 14.0)),
        bench_min_threshold=float(cfg.get("bench_min_threshold", 30.0)),
        starter_blowout_floor=float((cfg.get("game_script") or {}).get("starter_blowout_floor", 0.70)),
    )

    # ── Pace adjustment ───────────────────────────────────────────────────
    _total = total or default_total
    pace_adj = 1.0 + (0.06 * (_total - default_total) / 20.0)

    # ── Momentum (L3 vs L10 scoring trend) ────────────────────────────────
    momentum_str = float(cfg.get("momentum_strength", 0.15))
    mom_adj = momentum_adjustment(roll_adj, strength=momentum_str)

    # Combined game-context multiplier
    game_mult = s_adj * pace_adj * mom_adj

    # Apply game-context multiplier to stat projections
    pts *= game_mult
    reb *= game_mult
    ast *= game_mult
    stl *= game_mult
    blk *= game_mult
    threes *= game_mult

    s_base = _dfs_from_stats(pts, reb, ast, stl, blk, tov, dfs_weights=dfs_weights)
    rs_cfg = cfg.get("compression") or {}
    comp_div = float(rs_cfg.get("compression_divisor", 5.5))
    comp_pow = float(rs_cfg.get("compression_power", 0.72))
    rs_cap = float(rs_cfg.get("rs_cap", 20.0))
    raw_linear = s_base / max(comp_div, 1e-6)
    rating = min(float(raw_linear**comp_pow), rs_cap)

    minutes_cv = float(roll_adj.get("_minutes_cv") or 0.0)
    conf = max(0.0, min(1.0, 1.0 - min(0.5, minutes_cv)))

    stat_payload: Dict[str, Any] = {}
    edge_map: Dict[str, Any] = {}

    stat_list = list(cfg.get("stat_types") or ["points", "rebounds", "assists"])
    bl = book_lines or {}
    ev_cfg = {**cfg, "edge_thresholds": cfg.get("edge_thresholds") or {}}

    def _roll_key(st: str) -> str:
        if st == "pra":
            return "points"
        if st in ("threes", "points", "rebounds", "assists", "steals", "blocks"):
            return st
        return "points"

    for st in stat_list:
        rk = _roll_key(st)
        if st == "pra":
            mean = pts + reb + ast
            sp = float((roll_adj.get("points") or {}).get("std_L10", 2.0))
            sr = float((roll_adj.get("rebounds") or {}).get("std_L10", 2.0))
            sa = float((roll_adj.get("assists") or {}).get("std_L10", 2.0))
            std = math.sqrt(max(sp * sp + sr * sr + sa * sa, 1.0))
        else:
            d0 = roll_adj.get(rk) or {}
            mean = float(d0.get("mean_L15", 0.0)) * game_mult * home_adj
            std = float(d0.get("std_L10") or 2.0)

        droll = roll_adj.get(rk) or {}
        p20 = float(droll.get("p20", mean * 0.85))
        p80 = float(droll.get("p80", mean * 1.15))
        floor = float(droll.get("recent_min", p20))
        ceiling = float(droll.get("recent_max", p80))

        bk_key = _stat_to_book_key(st)
        book = bl.get(bk_key) or bl.get(st) or {}
        line = float(book.get("line") or mean)
        oo = book.get("odds_over")
        ou = book.get("odds_under")
        pin = book.get("pinnacle_line")

        fv = compute_fair_value(mean, line, oo, ou, std, pinnacle_line=pin, config=ev_cfg)
        row = {
            "mean": mean,
            "std": std,
            "p20": p20,
            "p80": p80,
            "floor": floor,
            "ceiling": ceiling,
            **fv,
        }
        stat_payload[st] = row
        edge_map[st] = {
            "direction": "over" if fv["ev_over"] >= fv["ev_under"] else "under",
            "edge_pct": fv["edge_pct"],
            "ev": max(fv["ev_over"], fv["ev_under"]),
            "hit_prob": fv["hit_prob_over"] if fv["ev_over"] >= fv["ev_under"] else fv["hit_prob_under"],
            "edge_class": fv["edge_class"],
            # Line-engine Claude prompt: both sides needed when force_direction is over vs under
            "fair_median": round(float(mean), 2),
            "hit_prob_over": float(fv["hit_prob_over"]),
            "hit_prob_under": float(fv["hit_prob_under"]),
            "ev_over": float(fv["ev_over"]),
            "ev_under": float(fv["ev_under"]),
        }

    mroll = roll_adj.get("minutes") or {}
    return {
        "pts": stat_payload.get("points"),
        "reb": stat_payload.get("rebounds"),
        "ast": stat_payload.get("assists"),
        "stl": stat_payload.get("steals"),
        "blk": stat_payload.get("blocks"),
        "threes": stat_payload.get("threes"),
        "pra": stat_payload.get("pra"),
        "min": {
            "mean": avg_min,
            "std": float(mroll.get("std_L10", 0.0)),
            "cv": minutes_cv,
        },
        "rating": round(rating, 3),
        "edge_map": edge_map,
        "confidence": conf,
        "_rolling": roll_adj,
    }
