"""
Tests for the Parlay Engine — Z-score hit probability, anti-fragility filters,
correlation scoring, and end-to-end optimal parlay selection.

Run: pytest tests/test_parlay.py -v
"""
import pytest
import math

# ── Parlay engine imports ────────────────────────────────────────────────────

from api.parlay_engine import (
    _american_to_implied,
    _z_to_probability,
    _compute_hit_probability,
    _compute_blended_confidence,
    _is_blowout_game,
    _minutes_cv,
    _has_gtd_star_teammate,
    _correlation_modifier,
    _std_dev,
    _build_parlay_narrative,
    build_candidate_legs,
    run_parlay_engine,
)


# ── Z-Score & Probability Tests ─────────────────────────────────────────────

class TestAmericanToImplied:
    def test_negative_odds(self):
        """Standard favorite: -140 ≈ 58.3%"""
        prob = _american_to_implied(-140)
        assert abs(prob - 0.583) < 0.01

    def test_positive_odds(self):
        """+150 ≈ 40%"""
        prob = _american_to_implied(150)
        assert abs(prob - 0.40) < 0.01

    def test_even_money(self):
        """-100 ≈ 50%"""
        prob = _american_to_implied(-100)
        assert abs(prob - 0.50) < 0.01

    def test_heavy_favorite(self):
        """-300 ≈ 75%"""
        prob = _american_to_implied(-300)
        assert abs(prob - 0.75) < 0.01

    def test_none_returns_none(self):
        assert _american_to_implied(None) is None

    def test_zero_returns_none(self):
        assert _american_to_implied(0) is None


class TestZToProbability:
    def test_z_zero(self):
        """Z=0 → 50%"""
        assert abs(_z_to_probability(0) - 0.5) < 0.001

    def test_z_positive(self):
        """Z=1.0 → ~84.1%"""
        assert abs(_z_to_probability(1.0) - 0.8413) < 0.01

    def test_z_negative(self):
        """Z=-1.0 → ~15.9%"""
        assert abs(_z_to_probability(-1.0) - 0.1587) < 0.01

    def test_extreme_positive(self):
        assert _z_to_probability(7.0) == 1.0

    def test_extreme_negative(self):
        assert _z_to_probability(-7.0) == 0.0


class TestComputeHitProbability:
    def test_over_high_proj(self):
        """Projection well above line with low variance → high probability."""
        prob = _compute_hit_probability(25.0, 20.5, 3.0, "over")
        assert prob is not None
        assert prob > 0.85  # Z = (20.5 - 25) / 3 = -1.5 → P(X > 20.5) > 93%

    def test_under_low_proj(self):
        """Projection well below line → high under probability."""
        prob = _compute_hit_probability(6.0, 8.5, 2.0, "under")
        assert prob is not None
        assert prob > 0.85

    def test_zero_std_dev_returns_none(self):
        prob = _compute_hit_probability(25.0, 20.5, 0, "over")
        assert prob is None

    def test_negative_std_dev_returns_none(self):
        prob = _compute_hit_probability(25.0, 20.5, -1.0, "over")
        assert prob is None

    def test_even_split(self):
        """Projection == line → ~50% regardless of direction."""
        prob_over = _compute_hit_probability(20.5, 20.5, 3.0, "over")
        prob_under = _compute_hit_probability(20.5, 20.5, 3.0, "under")
        assert abs(prob_over - 0.5) < 0.01
        assert abs(prob_under - 0.5) < 0.01


class TestBlendedConfidence:
    def test_both_present(self):
        """55/45 blend"""
        result = _compute_blended_confidence(0.65, 0.58)
        expected = 0.65 * 0.55 + 0.58 * 0.45
        assert abs(result - expected) < 0.001

    def test_model_only(self):
        result = _compute_blended_confidence(0.65, None)
        assert result == 0.65

    def test_vegas_only(self):
        result = _compute_blended_confidence(None, 0.58)
        assert result == 0.58

    def test_both_none(self):
        assert _compute_blended_confidence(None, None) is None


# ── Anti-Fragility Filter Tests ──────────────────────────────────────────────

class TestBlowoutFilter:
    def test_large_spread_is_blowout(self):
        assert _is_blowout_game(10.0) is True

    def test_moderate_spread_ok(self):
        assert _is_blowout_game(5.0) is False

    def test_exact_threshold(self):
        assert _is_blowout_game(8.5) is False
        assert _is_blowout_game(8.6) is True

    def test_none_spread(self):
        assert _is_blowout_game(None) is False

    def test_negative_spread(self):
        """Negative spread (underdog) with large magnitude is still a blowout."""
        assert _is_blowout_game(-10.0) is True


class TestMinutesCV:
    def test_low_variance(self):
        """Consistent minutes → low CV."""
        cv = _minutes_cv([32, 33, 31, 32, 34, 33, 32, 31, 33, 32])
        assert cv < 0.05

    def test_high_variance(self):
        """Wildly inconsistent minutes → high CV."""
        cv = _minutes_cv([35, 12, 30, 8, 28, 15, 33, 10, 25, 5])
        assert cv > 0.30

    def test_insufficient_data(self):
        cv = _minutes_cv([30, 31])
        assert cv == 999.0

    def test_empty_list(self):
        assert _minutes_cv([]) == 999.0

    def test_zero_average(self):
        assert _minutes_cv([0, 0, 0, 0, 0]) == 999.0


class TestGTDStarTeammate:
    def test_questionable_starter_triggers(self):
        rw = {"lebron james": {"team": "LAL", "status": "questionable", "is_starter": True}}
        assert _has_gtd_star_teammate("LAL", rw, "anthony davis") is True

    def test_out_starter_does_not_trigger(self):
        """OUT is already settled — not a GTD risk."""
        rw = {"lebron james": {"team": "LAL", "status": "out", "is_starter": True}}
        assert _has_gtd_star_teammate("LAL", rw, "anthony davis") is False

    def test_questionable_bench_ignored(self):
        rw = {"bench player": {"team": "LAL", "status": "questionable", "is_starter": False}}
        assert _has_gtd_star_teammate("LAL", rw, "anthony davis") is False

    def test_different_team_ignored(self):
        rw = {"jayson tatum": {"team": "BOS", "status": "questionable", "is_starter": True}}
        assert _has_gtd_star_teammate("LAL", rw, "anthony davis") is False

    def test_empty_statuses(self):
        assert _has_gtd_star_teammate("LAL", {}, "anthony davis") is False

    def test_none_statuses(self):
        assert _has_gtd_star_teammate("LAL", None, "anthony davis") is False


# ── Correlation Scoring Tests ────────────────────────────────────────────────

def _make_leg(player_name, team, stat_type, direction, gameId="g1", **kw):
    """Helper to build a leg dict for testing."""
    return {
        "player_name": player_name, "team": team, "stat_type": stat_type,
        "direction": direction, "gameId": gameId,
        "blended_confidence": 0.60, "edge": 2.0,
        "game_spread": kw.get("spread", 3.0), "game_total": kw.get("total", 225),
        "minutes_cv": 0.10, **kw,
    }


class TestCorrelationModifier:
    def test_veto_same_team_rebounds_over(self):
        legs = [
            _make_leg("A", "LAL", "rebounds", "over"),
            _make_leg("B", "LAL", "rebounds", "over"),
            _make_leg("C", "BOS", "points", "over", gameId="g2"),
        ]
        mult, reasons = _correlation_modifier(legs)
        assert mult == 0.0
        assert any("rebounds" in r.lower() for r in reasons)

    def test_veto_same_team_assists_over(self):
        legs = [
            _make_leg("A", "LAL", "assists", "over"),
            _make_leg("B", "LAL", "assists", "over"),
            _make_leg("C", "BOS", "points", "over", gameId="g2"),
        ]
        mult, reasons = _correlation_modifier(legs)
        assert mult == 0.0

    def test_positive_assist_points_same_team(self):
        legs = [
            _make_leg("PG", "LAL", "assists", "over"),
            _make_leg("Center", "LAL", "points", "over"),
            _make_leg("Star", "BOS", "points", "over", gameId="g2"),
        ]
        mult, reasons = _correlation_modifier(legs)
        assert mult >= 1.08
        assert any("assists feed" in r.lower() or "correlation" in r.lower() for r in reasons)

    def test_shootout_correlation(self):
        legs = [
            _make_leg("Star1", "LAL", "points", "over", spread=2.0),
            _make_leg("Star2", "BOS", "points", "over", spread=2.0),
            _make_leg("Other", "MIA", "rebounds", "over", gameId="g2"),
        ]
        mult, reasons = _correlation_modifier(legs)
        assert mult >= 1.05
        assert any("shootout" in r.lower() for r in reasons)

    def test_no_special_correlation(self):
        legs = [
            _make_leg("A", "LAL", "points", "over", gameId="g1"),
            _make_leg("B", "BOS", "rebounds", "over", gameId="g2"),
            _make_leg("C", "MIA", "assists", "over", gameId="g3"),
        ]
        mult, reasons = _correlation_modifier(legs)
        assert mult == 1.0
        assert reasons == []


# ── Standard Deviation Test ──────────────────────────────────────────────────

class TestStdDev:
    def test_known_values(self):
        # [2, 4, 4, 4, 5, 5, 7, 9] → mean=5, σ=2
        vals = [2, 4, 4, 4, 5, 5, 7, 9]
        sd = _std_dev(vals)
        assert abs(sd - 2.0) < 0.01

    def test_single_value(self):
        assert _std_dev([5]) == 0.0

    def test_empty(self):
        assert _std_dev([]) == 0.0


# ── Narrative Builder Test ───────────────────────────────────────────────────

class TestNarrativeBuilder:
    def test_contains_probability(self):
        legs = [
            {**_make_leg("A", "LAL", "points", "over"), "blended_confidence": 0.62, "edge": 2.5, "line": 20.5, "minutes_cv": 0.08},
            {**_make_leg("B", "BOS", "rebounds", "over"), "blended_confidence": 0.60, "edge": 1.5, "line": 8.5, "minutes_cv": 0.12},
            {**_make_leg("C", "MIA", "assists", "over"), "blended_confidence": 0.58, "edge": 1.0, "line": 6.5, "minutes_cv": 0.10},
        ]
        combined = 0.62 * 0.60 * 0.58
        narrative = _build_parlay_narrative(legs, ["Positive correlation: A feeds B"], combined)
        assert "%" in narrative
        assert "A" in narrative
        assert "Positive correlation" in narrative


# ── End-to-End Engine Test ───────────────────────────────────────────────────

class TestRunParlayEngine:
    def _make_proj(self, pid, name, team, pts=18, reb=6, ast=4, season_min=30, injury_status=""):
        return {
            "id": pid, "name": name, "team": team,
            "pts": pts, "reb": reb, "ast": ast,
            "season_min": season_min, "predMin": season_min,
            "season_pts": pts, "season_reb": reb, "season_ast": ast,
            "injury_status": injury_status,
        }

    def _make_game(self, gid, home, away, spread=3.0, total=225):
        return {
            "gameId": gid,
            "home": {"abbr": home, "id": f"{home}_id", "name": home},
            "away": {"abbr": away, "id": f"{away}_id", "name": away},
            "spread": spread, "total": total, "startTime": "2026-03-21T23:00:00Z",
        }

    def _make_gamelogs(self, pid, pts_vals, reb_vals, ast_vals, min_vals):
        return {pid: {"points": pts_vals, "rebounds": reb_vals, "assists": ast_vals, "minutes": min_vals}}

    def test_returns_none_with_too_few_candidates(self):
        result = run_parlay_engine([], [], {}, {})
        assert result is None

    def test_full_pipeline_with_valid_data(self):
        """End-to-end: 4 players, 2 games, sufficient odds → should find a parlay."""
        games = [
            self._make_game("g1", "LAL", "BOS", spread=3.0, total=228),
            self._make_game("g2", "MIA", "GSW", spread=2.0, total=220),
        ]
        projs = [
            self._make_proj("p1", "Player A", "LAL", pts=22, reb=5, ast=3),
            self._make_proj("p2", "Player B", "BOS", pts=20, reb=8, ast=5),
            self._make_proj("p3", "Player C", "MIA", pts=18, reb=6, ast=7),
            self._make_proj("p4", "Player D", "GSW", pts=25, reb=4, ast=6),
        ]
        odds = {
            ("player a", "points"): {"line": 20.5, "odds_over": -140, "odds_under": 120, "books_consensus": 3},
            ("player b", "rebounds"): {"line": 7.5, "odds_over": -135, "odds_under": 115, "books_consensus": 3},
            ("player c", "assists"): {"line": 6.5, "odds_over": -130, "odds_under": 110, "books_consensus": 2},
            ("player d", "points"): {"line": 23.5, "odds_over": -145, "odds_under": 125, "books_consensus": 3},
        }
        # Consistent gamelogs (low variance)
        gamelogs = {}
        for pid, base_pts, base_reb, base_ast in [
            ("p1", 22, 5, 3), ("p2", 20, 8, 5),
            ("p3", 18, 6, 7), ("p4", 25, 4, 6),
        ]:
            gamelogs[pid] = {
                "points": [base_pts + (i % 3 - 1) for i in range(15)],
                "rebounds": [base_reb + (i % 3 - 1) for i in range(15)],
                "assists": [base_ast + (i % 3 - 1) for i in range(15)],
                "minutes": [32 + (i % 3 - 1) for i in range(15)],
            }

        result = run_parlay_engine(projs, games, odds, gamelogs, {}, {
            "min_blended_conf": 0.50,
            "juice_threshold": -120,
            "min_games_played": 10,
        })

        assert result is not None
        assert len(result["legs"]) == 3
        assert result["combined_probability"] > 0
        assert result["parlay_score"] > 0
        assert result["narrative"]
        assert result["candidates_evaluated"] > 0

    def test_structured_builder_with_ideal_data(self):
        """When PG assists + teammate points + market match exist → structured path."""
        games = [
            self._make_game("g1", "LAL", "BOS", spread=3.0, total=228),
            self._make_game("g2", "MIA", "GSW", spread=2.0, total=220),
        ]
        projs = [
            self._make_proj("p1", "PG Star", "LAL", pts=22, reb=5, ast=8, season_min=34),
            self._make_proj("p2", "Teammate", "LAL", pts=20, reb=6, ast=3, season_min=32),
            self._make_proj("p3", "Role Player", "MIA", pts=14, reb=8, ast=2, season_min=28),
            self._make_proj("p4", "Other", "GSW", pts=25, reb=4, ast=6, season_min=33),
        ]
        # p1 is a PG/playmaker (ast=8), p2 is teammate (same team LAL)
        # p3 is a role player on different game (MIA) → market match
        projs[0]["position"] = "PG"
        projs[1]["position"] = "SF"
        projs[2]["position"] = "PF"
        projs[3]["position"] = "SG"
        odds = {
            ("pg star", "assists"): {"line": 7.5, "odds_over": -145, "odds_under": 125, "books_consensus": 3},
            ("teammate", "points"): {"line": 18.5, "odds_over": -140, "odds_under": 120, "books_consensus": 3},
            ("role player", "rebounds"): {"line": 7.5, "odds_over": -150, "odds_under": 130, "books_consensus": 3},
            ("other", "points"): {"line": 23.5, "odds_over": -135, "odds_under": 115, "books_consensus": 2},
        }
        gamelogs = {}
        for pid, base_pts, base_reb, base_ast in [
            ("p1", 22, 5, 8), ("p2", 20, 6, 3),
            ("p3", 14, 8, 2), ("p4", 25, 4, 6),
        ]:
            gamelogs[pid] = {
                "points": [base_pts + (i % 3 - 1) for i in range(15)],
                "rebounds": [base_reb + (i % 3 - 1) for i in range(15)],
                "assists": [base_ast + (i % 3 - 1) for i in range(15)],
                "minutes": [32 + (i % 3 - 1) for i in range(15)],
            }
        result = run_parlay_engine(projs, games, odds, gamelogs, {}, {
            "min_blended_conf": 0.50,
            "juice_threshold": -120,
            "min_games_played": 10,
            "market_match_juice_threshold": -140,
            "market_match_min_conf": 0.50,
            "correlated_pair_max_spread": 6.5,
        })
        assert result is not None
        assert len(result["legs"]) == 3
        assert result.get("structured") is True
        # Verify leg structure: market match first, then assists, then points
        legs = result["legs"]
        assert legs[0]["gameId"] != legs[1]["gameId"]  # Market match from different game
        assert legs[1]["stat_type"] == "assists"
        assert legs[1]["direction"] == "over"
        assert legs[2]["stat_type"] == "points"
        assert legs[2]["direction"] == "over"
        assert legs[1]["team"] == legs[2]["team"]  # Correlated pair on same team

    def test_lines_snap_to_half(self):
        """All lines in parlay output should be on 0.5 increments."""
        games = [
            self._make_game("g1", "LAL", "BOS", spread=3.0, total=228),
            self._make_game("g2", "MIA", "GSW", spread=2.0, total=220),
        ]
        projs = [
            self._make_proj("p1", "Player A", "LAL", pts=22, reb=5, ast=3),
            self._make_proj("p2", "Player B", "BOS", pts=20, reb=8, ast=5),
            self._make_proj("p3", "Player C", "MIA", pts=18, reb=6, ast=7),
            self._make_proj("p4", "Player D", "GSW", pts=25, reb=4, ast=6),
        ]
        # Use lines that are NOT on 0.5 — engine should snap them
        odds = {
            ("player a", "points"): {"line": 20.3, "odds_over": -140, "odds_under": 120, "books_consensus": 3},
            ("player b", "rebounds"): {"line": 7.7, "odds_over": -135, "odds_under": 115, "books_consensus": 3},
            ("player c", "assists"): {"line": 6.9, "odds_over": -130, "odds_under": 110, "books_consensus": 2},
            ("player d", "points"): {"line": 23.1, "odds_over": -145, "odds_under": 125, "books_consensus": 3},
        }
        gamelogs = {}
        for pid, base_pts, base_reb, base_ast in [
            ("p1", 22, 5, 3), ("p2", 20, 8, 5),
            ("p3", 18, 6, 7), ("p4", 25, 4, 6),
        ]:
            gamelogs[pid] = {
                "points": [base_pts + (i % 3 - 1) for i in range(15)],
                "rebounds": [base_reb + (i % 3 - 1) for i in range(15)],
                "assists": [base_ast + (i % 3 - 1) for i in range(15)],
                "minutes": [32 + (i % 3 - 1) for i in range(15)],
            }
        result = run_parlay_engine(projs, games, odds, gamelogs, {}, {
            "min_blended_conf": 0.50,
            "juice_threshold": -120,
            "min_games_played": 10,
        })
        assert result is not None
        for leg in result["legs"]:
            # Every line must be on 0.5: line * 2 should be an integer
            assert leg["line"] * 2 == int(leg["line"] * 2), \
                f"Line {leg['line']} for {leg['player_name']} is not on 0.5 increment"

    def test_fallback_to_combinatorial_when_no_pair(self):
        """When no PG assists + teammate points pair exists, falls back to combinatorial."""
        games = [
            self._make_game("g1", "LAL", "BOS", spread=3.0, total=228),
            self._make_game("g2", "MIA", "GSW", spread=2.0, total=220),
        ]
        # All players on different teams, no playmakers → no correlated pair
        projs = [
            self._make_proj("p1", "Player A", "LAL", pts=22, reb=5, ast=1),
            self._make_proj("p2", "Player B", "BOS", pts=20, reb=8, ast=1),
            self._make_proj("p3", "Player C", "MIA", pts=18, reb=6, ast=1),
            self._make_proj("p4", "Player D", "GSW", pts=25, reb=4, ast=1),
        ]
        for p in projs:
            p["position"] = "C"  # Centers, not playmakers
        odds = {
            ("player a", "points"): {"line": 20.5, "odds_over": -140, "odds_under": 120, "books_consensus": 3},
            ("player b", "rebounds"): {"line": 7.5, "odds_over": -135, "odds_under": 115, "books_consensus": 3},
            ("player c", "points"): {"line": 16.5, "odds_over": -130, "odds_under": 110, "books_consensus": 2},
            ("player d", "points"): {"line": 23.5, "odds_over": -145, "odds_under": 125, "books_consensus": 3},
        }
        gamelogs = {}
        for pid, base_pts, base_reb, base_ast in [
            ("p1", 22, 5, 1), ("p2", 20, 8, 1),
            ("p3", 18, 6, 1), ("p4", 25, 4, 1),
        ]:
            gamelogs[pid] = {
                "points": [base_pts + (i % 3 - 1) for i in range(15)],
                "rebounds": [base_reb + (i % 3 - 1) for i in range(15)],
                "assists": [base_ast + (i % 3 - 1) for i in range(15)],
                "minutes": [32 + (i % 3 - 1) for i in range(15)],
            }
        result = run_parlay_engine(projs, games, odds, gamelogs, {}, {
            "min_blended_conf": 0.50,
            "juice_threshold": -120,
            "min_games_played": 10,
        })
        assert result is not None
        assert result.get("structured") is False
        assert len(result["legs"]) == 3

    def test_blowout_filter_removes_all(self):
        """All games are blowouts → no valid parlay."""
        games = [self._make_game("g1", "LAL", "BOS", spread=12.0)]
        projs = [
            self._make_proj("p1", "Player A", "LAL"),
            self._make_proj("p2", "Player B", "BOS"),
            self._make_proj("p3", "Player C", "LAL"),
        ]
        result = run_parlay_engine(projs, games, {}, {})
        assert result is None


# ── Backend Integration Tests ────────────────────────────────────────────────

class TestParlayConfigDefaults:
    """Verify parlay config exists in _CONFIG_DEFAULTS."""

    def test_parlay_section_in_defaults(self):
        from api.index import _CONFIG_DEFAULTS
        assert "parlay" in _CONFIG_DEFAULTS

    def test_parlay_max_spread(self):
        from api.index import _CONFIG_DEFAULTS
        assert _CONFIG_DEFAULTS["parlay"]["max_spread"] == 8.5

    def test_parlay_min_blended_conf(self):
        from api.index import _CONFIG_DEFAULTS
        assert _CONFIG_DEFAULTS["parlay"]["min_blended_conf"] == 0.52

    def test_parlay_max_minutes_cv(self):
        from api.index import _CONFIG_DEFAULTS
        assert _CONFIG_DEFAULTS["parlay"]["max_minutes_cv"] == 0.30

    def test_parlay_structured_config_keys(self):
        from api.index import _CONFIG_DEFAULTS
        p = _CONFIG_DEFAULTS["parlay"]
        assert p["market_match_juice_threshold"] == -140
        assert p["market_match_juice_relaxed"] == -120
        assert p["market_match_min_conf"] == 0.58
        assert p["correlated_pair_max_spread"] == 6.5
        assert p["parlay_gamelog_pool_cap"] == 100


class TestParlayRateLimit:
    """Verify parlay is rate-limited."""

    def test_parlay_in_rate_limits(self):
        from api.index import _RATE_LIMITS
        assert "parlay" in _RATE_LIMITS
        assert _RATE_LIMITS["parlay"] == 10


class TestFetchGamelog:
    """Verify _fetch_gamelog exists and has caching."""

    def test_function_exists(self):
        from api.index import _fetch_gamelog
        assert callable(_fetch_gamelog)

    def test_returns_dict_on_invalid_pid(self):
        """Non-existent player ID → empty dict (no crash)."""
        from api.index import _fetch_gamelog
        result = _fetch_gamelog("nonexistent_999999")
        assert isinstance(result, dict)


class TestParlayEndpointExists:
    """Verify the /api/parlay endpoint is registered."""

    def test_endpoint_registered(self):
        from api.index import app
        routes = [r.path for r in app.routes]
        assert "/api/parlay" in routes


# ── Frontend JS Integration Tests ────────────────────────────────────────────

class TestParlayFrontend:
    """Verify frontend HTML/JS for the Parlay tab."""

    def test_tab_parlay_exists(self):
        html = open("index.html").read()
        assert 'id="tab-parlay"' in html

    def test_parlay_nav_button(self):
        html = open("index.html").read()
        assert 'data-tab="parlay"' in html
        assert "switchTab('parlay')" in html

    def test_parlay_accent_in_tab_accent(self):
        html = open("index.html").read()
        assert "parlay:" in html
        # Parlay uses teal accent (unified color system)
        assert "20,184,166" in html

    def test_parlay_css_variable(self):
        html = open("index.html").read()
        assert "--parlay:" in html

    def test_parlay_state_init(self):
        html = open("index.html").read()
        assert "PARLAY_STATE" in html
        assert "initParlayPage" in html
        assert "fetchParlay" in html

    def test_parlay_ticket_container(self):
        html = open("index.html").read()
        assert 'id="parlayTicket"' in html
        assert 'id="parlayEmpty"' in html
        assert 'id="parlayLoading"' in html

    def test_parlay_tab_glow(self):
        html = open("index.html").read()
        # Parlay tab glow uses teal (unified color system)
        assert "parlay: 'rgba(20,184,166" in html

    def test_render_parlay_leg_function(self):
        html = open("index.html").read()
        assert "function renderParlayLeg(" in html

    def test_render_parlay_ticket_function(self):
        html = open("index.html").read()
        assert "function renderParlayTicket(" in html

    def test_parlay_history_wrap_exists(self):
        html = open("index.html").read()
        assert 'id="parlayHistoryWrap"' in html

    def test_parlay_history_list_exists(self):
        html = open("index.html").read()
        assert 'id="parlayHistoryList"' in html

    def test_parlay_history_stats_exists(self):
        html = open("index.html").read()
        assert 'id="parlayHistoryStats"' in html

    def test_parlay_history_message_exists(self):
        html = open("index.html").read()
        assert 'id="parlayHistoryMessage"' in html

    def test_render_parlay_history_error_exists(self):
        html = open("index.html").read()
        assert "function renderParlayHistoryError(" in html

    def test_parlay_modal_exists(self):
        html = open("index.html").read()
        assert 'id="parlayModal"' in html
        assert 'id="parlayModalContent"' in html

    def test_fetch_parlay_history_function(self):
        html = open("index.html").read()
        assert "function fetchParlayHistory(" in html

    def test_render_parlay_history_function(self):
        html = open("index.html").read()
        assert "function renderParlayHistory(" in html

    def test_open_parlay_modal_function(self):
        html = open("index.html").read()
        assert "function openParlayModal(" in html

    def test_close_parlay_modal_function(self):
        html = open("index.html").read()
        assert "function closeParlayModal(" in html

    def test_parlay_hist_data_global(self):
        html = open("index.html").read()
        assert "PARLAY_HIST_DATA" in html
        assert "PARLAY_HIST_ERROR" in html

    def test_parlay_modal_escape_key(self):
        html = open("index.html").read()
        assert "closeParlayModal()" in html

    def test_parlay_modal_aria_attributes(self):
        html = open("index.html").read()
        assert 'aria-modal="true"' in html
        assert 'aria-label="Parlay detail"' in html

    def test_parlay_tab_aria_label(self):
        html = open("index.html").read()
        assert 'aria-label="Parlay tab"' in html

    def test_parlay_history_aria_region(self):
        html = open("index.html").read()
        assert 'aria-label="Parlay history"' in html

    def test_parlay_content_aria_live(self):
        html = open("index.html").read()
        assert 'id="parlayContent" aria-live="polite"' in html


class TestParlayHistoryEndpoint:
    """Verify the /api/parlay-history endpoint is registered."""

    def test_endpoint_registered(self):
        from api.index import app
        routes = [r.path for r in app.routes]
        assert "/api/parlay-history" in routes
        assert "/api/parlay-live-stream" in routes

    def test_parlay_auto_save_fields(self):
        """Verify the /api/parlay endpoint adds result and actual_stat fields."""
        # Check that the code path for auto-save exists
        import inspect
        from api.index import get_parlay
        source = inspect.getsource(get_parlay)
        assert "result" in source
        assert "data/parlays/" in source
        assert "_github_write_file" in source
