"""Unit tests for api/fair_value.py (pure deterministic engine)."""

import math

import pytest

from api.fair_value import (
    adjust_for_opponent,
    ats_regression_factor,
    closeness_factor,
    compute_fair_value,
    compute_rolling_stats,
    dvp_binary_from_nba_com,
    game_script_weights,
    momentum_adjustment,
    project_player_fv,
    should_cascade,
    spread_adjustment,
)


def test_compute_rolling_stats_basic():
    gl = {
        "points": [10, 12, 14, 16, 18, 20, 10, 12, 14, 16, 18, 20, 22, 24, 26],
        "rebounds": [5] * 15,
        "assists": [3] * 15,
        "minutes": [30.0] * 15,
    }
    r = compute_rolling_stats(gl, window=15, short_window=10)
    assert r["points"]["mean_L15"] == pytest.approx(16.8, rel=0.01)
    assert r["points"]["std_L10"] >= 0
    assert r["_minutes_cv"] == pytest.approx(0.0, abs=0.01)


def test_closeness_factor_bounds():
    c0 = closeness_factor(0.0, 220.0, closeness_max=1.5)
    assert 1.0 <= c0 <= 1.5
    c_blow = closeness_factor(15.0, 220.0, closeness_max=1.5)
    assert c_blow >= 1.0
    assert c_blow <= c0 + 1e-6 or c_blow < c0  # blowout typically lower closeness bonus


def test_dvp_binary_from_nba_com():
    g, f = dvp_binary_from_nba_com({"G": 28.0, "F": 24.0, "C": 22.0})
    assert g == 28.0
    assert f == pytest.approx(23.0)


def test_adjust_for_opponent_scales_points():
    roll = {
        "points": {"mean_L15": 20.0, "mean_L10": 20.0, "std_L10": 2.0, "p20": 15, "p80": 25, "recent_min": 10, "recent_max": 30, "per_minute_rate": 0.5, "L3_vs_L10_momentum": 1.0},
        "_minutes_cv": 0.1,
        "_mean_minutes_L15": 30.0,
    }
    opp = {"pts_allowed_guards": 120.0, "pts_allowed_forwards": 110.0, "league_avg_guards": 26.0, "league_avg_forwards": 21.5}
    adj = adjust_for_opponent(roll, opp, "PG")
    assert adj["points"]["mean_L15"] > roll["points"]["mean_L15"]


def test_compute_fair_value_ev():
    fv = compute_fair_value(
        projection=22.0,
        book_line=20.0,
        odds_over=-110,
        odds_under=-110,
        std_dev=5.0,
        config={"edge_thresholds": {"min_edge_pct": 1.0, "min_ev": 0.0}},
    )
    assert fv["edge_points"] == pytest.approx(2.0)
    assert 0.0 <= fv["hit_prob_over"] <= 1.0
    assert 0.0 <= fv["hit_prob_under"] <= 1.0
    assert fv["implied_prob_over"] is not None


def test_should_cascade():
    assert should_cascade("Jane Doe", 10.0, {"cascade_policy": "disabled"}) is False
    assert should_cascade("Jane Doe", 10.0, {"cascade_policy": "all"}) is True
    assert should_cascade("Nikola Jokic", 20.0, {"cascade_policy": "elite_only", "elite_players": {"Nikola Jokic"}}) is True
    assert should_cascade("Role Player", 28.0, {"cascade_policy": "elite_only", "elite_cascade_ppg": 27.0}) is True
    assert should_cascade("Role Player", 20.0, {"cascade_policy": "elite_only", "elite_cascade_ppg": 27.0}) is False


def test_ats_stub():
    assert ats_regression_factor(0.7) == 1.0


def test_game_script_weights_tiers():
    """Game script produces different stat weights by pace tier."""
    dg = game_script_weights(215.0, 3.0)  # defensive grind
    assert dg["pts"] < 1.0  # points suppressed in defensive grind
    assert dg["stl"] > 1.0  # steals boosted

    tm = game_script_weights(250.0, 3.0)  # track meet
    assert tm["pts"] > 1.0  # points boosted
    assert tm["stl"] < 1.0  # steals suppressed

    bal = game_script_weights(228.0, 3.0)  # balanced
    assert bal["pts"] == pytest.approx(1.0)


def test_game_script_blowout_penalty():
    """Track meet + blowout spread applies penalties."""
    normal = game_script_weights(250.0, 3.0)
    blowout = game_script_weights(250.0, 12.0)
    assert blowout["pts"] < normal["pts"]
    assert blowout["ast"] < normal["ast"]


def test_spread_adjustment_tight_game_starter():
    """Starters get bonus in tight games."""
    adj = spread_adjustment(spread=1.0, total=225.0, pts=20.0, avg_min=32.0)
    assert adj > 1.1  # meaningful bonus for tight game


def test_spread_adjustment_blowout_starter():
    """Starters get penalized in blowouts."""
    adj = spread_adjustment(spread=12.0, total=225.0, pts=20.0, avg_min=32.0)
    assert adj < 0.8  # steep penalty


def test_spread_adjustment_blowout_bench():
    """Bench players get bonus in blowouts (garbage time)."""
    adj = spread_adjustment(spread=12.0, total=225.0, pts=8.0, avg_min=18.0)
    assert adj > 1.0  # bonus for garbage time


def test_spread_adjustment_shootout_bonus():
    """High total + tight spread = shootout bonus for starters."""
    normal = spread_adjustment(spread=2.0, total=225.0, pts=20.0, avg_min=32.0)
    shootout = spread_adjustment(spread=2.0, total=240.0, pts=20.0, avg_min=32.0)
    assert shootout > normal


def test_momentum_adjustment_hot():
    """Hot streak (L3 > L10) produces positive multiplier."""
    roll = {"points": {"L3_vs_L10_momentum": 1.25}}
    adj = momentum_adjustment(roll, strength=0.15)
    assert adj > 1.0
    assert adj <= 1.12


def test_momentum_adjustment_cold():
    """Cold streak (L3 < L10) produces negative multiplier."""
    roll = {"points": {"L3_vs_L10_momentum": 0.80}}
    adj = momentum_adjustment(roll, strength=0.15)
    assert adj < 1.0
    assert adj >= 0.92


def test_momentum_adjustment_neutral():
    """Neutral momentum returns ~1.0."""
    roll = {"points": {"L3_vs_L10_momentum": 1.0}}
    adj = momentum_adjustment(roll, strength=0.15)
    assert adj == pytest.approx(1.0, abs=0.001)


def test_project_player_fv_smoke():
    gl = {
        "points": [15 + i * 0.5 for i in range(15)],
        "rebounds": [6.0] * 15,
        "assists": [4.0] * 15,
        "steals": [1.0] * 15,
        "blocks": [0.5] * 15,
        "threes": [2.0] * 15,
        "turnovers": [2.0] * 15,
        "minutes": [32.0] * 15,
    }
    athlete = {"min": 32.0}
    opp = {"pts_allowed_guards": 118.0, "pts_allowed_forwards": 114.0, "league_avg_guards": 26.0, "league_avg_forwards": 21.5}
    book = {
        "points": {"line": 18.0, "odds_over": -110, "odds_under": -110},
        "rebounds": {"line": 5.5, "odds_over": -110, "odds_under": -110},
        "assists": {"line": 3.5, "odds_over": -110, "odds_under": -110},
    }
    cfg = {
        "primary_window": 15,
        "stat_types": ["points", "rebounds", "assists"],
        "compression": {"compression_divisor": 5.5, "compression_power": 0.72, "rs_cap": 20.0},
        "edge_thresholds": {"min_edge_pct": 1.0, "min_ev": 0.0},
    }
    out = project_player_fv(gl, athlete, "SG", opp, 3.0, 225.0, "home", book, cfg)
    assert out["rating"] > 0
    assert "points" in out["edge_map"]
    assert out["confidence"] >= 0.0


def test_project_player_fv_game_script_affects_rating():
    """Verify game_script config changes the RS rating."""
    gl = {
        "points": [20.0] * 15,
        "rebounds": [6.0] * 15,
        "assists": [5.0] * 15,
        "steals": [1.5] * 15,
        "blocks": [0.5] * 15,
        "threes": [2.0] * 15,
        "turnovers": [2.0] * 15,
        "minutes": [32.0] * 15,
    }
    athlete = {"min": 32.0}
    opp = {"pts_allowed_guards": 26.0, "pts_allowed_forwards": 21.5,
           "league_avg_guards": 26.0, "league_avg_forwards": 21.5}
    cfg = {
        "primary_window": 15,
        "stat_types": ["points"],
        "compression": {"compression_divisor": 4.5, "compression_power": 0.78, "rs_cap": 20.0},
    }
    # Defensive grind (total=210) should give lower rating than track meet (total=250)
    out_dg = project_player_fv(gl, athlete, "SG", opp, 3.0, 210.0, "home", None, cfg)
    out_tm = project_player_fv(gl, athlete, "SG", opp, 3.0, 250.0, "home", None, cfg)
    assert out_tm["rating"] > out_dg["rating"], "Track meet should produce higher RS than defensive grind"


def test_project_player_fv_spread_adj_tight_vs_blowout():
    """Tight spread should give higher rating than blowout for a starter."""
    gl = {
        "points": [22.0] * 15,
        "rebounds": [7.0] * 15,
        "assists": [5.0] * 15,
        "steals": [1.0] * 15,
        "blocks": [0.5] * 15,
        "turnovers": [2.0] * 15,
        "minutes": [34.0] * 15,
    }
    athlete = {"min": 34.0}
    opp = {"pts_allowed_guards": 26.0, "pts_allowed_forwards": 21.5,
           "league_avg_guards": 26.0, "league_avg_forwards": 21.5}
    cfg = {
        "primary_window": 15,
        "stat_types": ["points"],
        "compression": {"compression_divisor": 4.5, "compression_power": 0.78, "rs_cap": 20.0},
    }
    out_tight = project_player_fv(gl, athlete, "SG", opp, 1.0, 225.0, "home", None, cfg)
    out_blow = project_player_fv(gl, athlete, "SG", opp, 14.0, 225.0, "home", None, cfg)
    assert out_tight["rating"] > out_blow["rating"], "Tight game should produce higher RS for starters"

