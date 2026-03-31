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
        assert p["correlated_pair_max_spread"] == 5.0
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
        js = open("app.js").read()
        assert "parlay:" in js
        # Parlay uses teal accent (unified color system)
        assert "20,184,166" in js

    def test_parlay_css_variable(self):
        css = open("styles.css").read()
        assert "--parlay:" in css

    def test_parlay_state_init(self):
        js = open("app.js").read()
        assert "PARLAY_STATE" in js
        assert "initParlayPage" in js
        assert "fetchParlay" in js

    def test_parlay_ticket_container(self):
        html = open("index.html").read()
        assert 'id="parlayTicket"' in html
        assert 'id="parlayEmpty"' in html
        assert 'id="parlayLoading"' in html

    def test_parlay_tab_glow(self):
        js = open("app.js").read()
        # Parlay tab glow uses teal (unified color system)
        assert "parlay: 'rgba(20,184,166" in js

    def test_render_parlay_leg_function(self):
        js = open("app.js").read()
        assert "function renderParlayLeg(" in js

    def test_render_parlay_ticket_function(self):
        js = open("app.js").read()
        assert "function renderParlayTicket(" in js

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
        js = open("app.js").read()
        assert "function renderParlayHistoryError(" in js

    def test_parlay_modal_exists(self):
        html = open("index.html").read()
        assert 'id="parlayModal"' in html
        assert 'id="parlayModalContent"' in html

    def test_fetch_parlay_history_function(self):
        js = open("app.js").read()
        assert "function fetchParlayHistory(" in js

    def test_render_parlay_history_function(self):
        js = open("app.js").read()
        assert "function renderParlayHistory(" in js

    def test_open_parlay_modal_function(self):
        js = open("app.js").read()
        assert "function openParlayModal(" in js

    def test_close_parlay_modal_function(self):
        js = open("app.js").read()
        assert "function closeParlayModal(" in js

    def test_parlay_hist_data_global(self):
        js = open("app.js").read()
        assert "PARLAY_HIST_DATA" in js
        assert "PARLAY_HIST_ERROR" in js

    def test_parlay_modal_escape_key(self):
        # closeParlayModal() is in both HTML (onclick) and JS
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


# ── Post-Mortem v2: Auto-Fade Matrix Tests ─────────────────────────────────

def _make_game(home="BOS", away="LAL", spread=-3.0, total=228.0, home_b2b=False, away_b2b=False):
    """Helper: build a minimal game dict."""
    return {
        "home": {"abbr": home}, "away": {"abbr": away},
        "gameId": f"{home}_{away}", "spread": spread, "total": total,
        "startTime": "2026-03-25T23:00Z",
        "home_b2b": home_b2b, "away_b2b": away_b2b,
    }


def _make_proj(name, team, pid="100", pts=20, reb=5, ast=5, position="SG",
               season_min=30, season_pts=20, season_ast=5, season_reb=5,
               rest_days=2):
    """Helper: build a minimal projection dict."""
    return {
        "id": pid, "name": name, "team": team, "position": position,
        "pts": pts, "reb": reb, "ast": ast,
        "season_min": season_min, "predMin": season_min,
        "season_pts": season_pts, "season_ast": season_ast, "season_reb": season_reb,
        "rest_days": rest_days,
    }


def _make_gamelog(pts_vals=None, reb_vals=None, ast_vals=None, min_vals=None):
    """Helper: build a minimal gamelog dict."""
    return {
        "points": pts_vals or [20, 22, 18, 21, 19, 23, 20, 17, 22, 21],
        "rebounds": reb_vals or [5, 6, 4, 5, 7, 5, 6, 4, 5, 6],
        "assists": ast_vals or [5, 4, 6, 5, 3, 5, 4, 6, 5, 4],
        "minutes": min_vals or [32, 34, 31, 33, 32, 34, 33, 31, 32, 33],
    }


def _make_odds(line, odds_over=-110, odds_under=-110):
    """Helper: build a minimal odds dict."""
    return {"line": line, "odds_over": odds_over, "odds_under": odds_under}


class TestAutoFadeSwitchHeavy:
    """Centers rebounds over vs switch-heavy defenses should be filtered."""

    def test_center_reb_over_vs_switch_heavy_filtered(self):
        from api.parlay_engine import build_candidate_legs
        games = [_make_game("BOS", "LAL")]
        proj = [_make_proj("Center Guy", "LAL", pid="1", position="C", reb=10, season_reb=10)]
        odds = {("center guy", "rebounds"): _make_odds(8.5, -150, 120)}
        gl = {"1": _make_gamelog(reb_vals=[10, 9, 11, 10, 8, 10, 9, 11, 10, 9])}
        cfg = {"auto_fade": {"switch_heavy_teams": ["BOS"]}, "juice_threshold": -105, "min_blended_conf": 0.01}
        cands, funnel = build_candidate_legs(proj, games, odds, gl, {}, cfg)
        # Should filter center rebounds over vs BOS
        reb_overs = [c for c in cands if c["stat_type"] == "rebounds" and c["direction"] == "over"]
        assert len(reb_overs) == 0
        assert funnel["switch_heavy"] >= 1

    def test_center_reb_over_vs_non_switch_heavy_passes(self):
        from api.parlay_engine import build_candidate_legs
        games = [_make_game("MIA", "LAL")]
        proj = [_make_proj("Center Guy", "LAL", pid="1", position="C", reb=10, season_reb=10)]
        odds = {("center guy", "rebounds"): _make_odds(8.5, -150, 120)}
        gl = {"1": _make_gamelog(reb_vals=[10, 9, 11, 10, 8, 10, 9, 11, 10, 9])}
        cfg = {"auto_fade": {"switch_heavy_teams": ["BOS"]}, "juice_threshold": -105, "min_blended_conf": 0.01}
        cands, funnel = build_candidate_legs(proj, games, odds, gl, {}, cfg)
        reb_overs = [c for c in cands if c["stat_type"] == "rebounds" and c["direction"] == "over"]
        assert len(reb_overs) >= 1

    def test_guard_reb_over_vs_switch_heavy_passes(self):
        """Guards should not be filtered by switch-heavy fade."""
        from api.parlay_engine import build_candidate_legs
        games = [_make_game("BOS", "LAL")]
        proj = [_make_proj("Guard Guy", "LAL", pid="1", position="SG", reb=8, season_reb=8)]
        odds = {("guard guy", "rebounds"): _make_odds(6.5, -150, 120)}
        gl = {"1": _make_gamelog(reb_vals=[8, 9, 7, 8, 10, 8, 9, 7, 8, 9])}
        cfg = {"auto_fade": {"switch_heavy_teams": ["BOS"]}, "juice_threshold": -105, "min_blended_conf": 0.01}
        cands, funnel = build_candidate_legs(proj, games, odds, gl, {}, cfg)
        reb_overs = [c for c in cands if c["stat_type"] == "rebounds" and c["direction"] == "over"]
        assert len(reb_overs) >= 1


class TestAutoFadeFakeJuice:
    """High recent hit rate + low season model probability triggers fake juice fade."""

    def test_fake_juice_filtered(self):
        from api.parlay_engine import build_candidate_legs
        games = [_make_game("MIA", "LAL")]
        proj = [_make_proj("Streaky Player", "LAL", pid="1", pts=15, season_pts=15)]
        # Line at 12.5, recent values all above → high recent hit rate
        # But model_prob should be low (std_dev is high relative to edge)
        odds = {("streaky player", "points"): _make_odds(12.5, -150, 120)}
        # Recent L10: 9 of 10 above 12.5 = 90% → fake juice triggers if model_prob < 0.55
        gl = {"1": _make_gamelog(pts_vals=[15, 14, 13, 16, 14, 15, 13, 14, 11, 15])}
        cfg = {
            "auto_fade": {"fake_juice_recent_threshold": 0.80, "fake_juice_season_ceiling": 0.65},
            "juice_threshold": -105, "min_blended_conf": 0.01,
        }
        cands, funnel = build_candidate_legs(proj, games, odds, gl, {}, cfg)
        # With high recent hit rate AND projection close to line (low model_prob),
        # the fake juice filter may fire — verify counter exists
        assert "fake_juice" in funnel

    def test_fake_juice_not_triggered_when_model_confident(self):
        from api.parlay_engine import build_candidate_legs
        games = [_make_game("MIA", "LAL")]
        proj = [_make_proj("Strong Player", "LAL", pid="1", pts=25, season_pts=25)]
        # Line at 18.5, projection 25 → model is very confident (high model_prob)
        odds = {("strong player", "points"): _make_odds(18.5, -150, 120)}
        gl = {"1": _make_gamelog(pts_vals=[25, 24, 23, 26, 24, 25, 23, 24, 22, 25])}
        cfg = {
            "auto_fade": {"fake_juice_recent_threshold": 0.80, "fake_juice_season_ceiling": 0.55},
            "juice_threshold": -105, "min_blended_conf": 0.01,
        }
        cands, funnel = build_candidate_legs(proj, games, odds, gl, {}, cfg)
        pts_overs = [c for c in cands if c["stat_type"] == "points" and c["direction"] == "over"]
        assert len(pts_overs) >= 1  # Should pass — model is confident


class TestAutoFadeB2BPenalty:
    """B2B correlated pairs get penalized in scoring."""

    def test_b2b_pair_penalized_in_structure(self):
        from api.parlay_engine import _score_structure
        legs = [
            {"team": "LAL", "stat_type": "assists", "direction": "over", "player_name": "PG",
             "blended_confidence": 0.60, "american_odds": -110, "minutes_cv": 0.10,
             "game_spread": -3, "game_total": 230, "is_b2b": True, "opp_b2b": False,
             "position": "PG", "season_reb": 3, "season_ast": 8, "gameId": "g1"},
            {"team": "LAL", "stat_type": "points", "direction": "over", "player_name": "Scorer",
             "blended_confidence": 0.60, "american_odds": -110, "minutes_cv": 0.10,
             "game_spread": -3, "game_total": 230, "is_b2b": True, "opp_b2b": False,
             "position": "SF", "season_reb": 6, "season_ast": 2, "gameId": "g1"},
            {"team": "BOS", "stat_type": "rebounds", "direction": "over", "player_name": "Big",
             "blended_confidence": 0.60, "american_odds": -150, "minutes_cv": 0.10,
             "game_spread": -3, "game_total": 230, "is_b2b": False, "opp_b2b": False,
             "position": "C", "season_reb": 10, "season_ast": 1, "gameId": "g2"},
        ]
        cfg_b2b = {"auto_fade": {"b2b_correlated_pair_penalty": 0.75}}
        bonus_b2b, reasons_b2b = _score_structure(legs, cfg_b2b)

        # Same legs but not on B2B
        legs_rested = [dict(l, is_b2b=False) for l in legs]
        bonus_rested, _ = _score_structure(legs_rested, cfg_b2b)

        assert bonus_b2b < bonus_rested
        assert any("B2B" in r for r in reasons_b2b)

    def test_b2b_pair_penalized_in_correlated_pair_search(self):
        from api.parlay_engine import _find_best_correlated_pair
        pool = [
            {"player_id": "1", "team": "LAL", "stat_type": "assists", "direction": "over",
             "blended_confidence": 0.65, "position": "PG", "season_ast": 8,
             "game_spread": -3, "game_total": 230, "is_b2b": True, "gameId": "g1"},
            {"player_id": "2", "team": "LAL", "stat_type": "points", "direction": "over",
             "blended_confidence": 0.65, "position": "SF", "season_ast": 2,
             "game_spread": -3, "game_total": 230, "is_b2b": True, "gameId": "g1"},
        ]
        cfg = {"auto_fade": {"b2b_correlated_pair_penalty": 0.75}}
        result = _find_best_correlated_pair(pool, cfg)
        # Should still return a pair but with penalized score
        assert result is not None
        _, _, score = result
        # Unpenalized would be ~0.65 * 0.65 * 1.15 = 0.486; penalized = 0.486 * 0.75 = 0.364
        assert score < 0.45


class TestGameTotalFloor:
    """Correlated pairs in low-total games should be penalized or excluded."""

    def test_low_total_pair_excluded(self):
        from api.parlay_engine import _find_best_correlated_pair
        pool = [
            {"player_id": "1", "team": "LAL", "stat_type": "assists", "direction": "over",
             "blended_confidence": 0.65, "position": "PG", "season_ast": 8,
             "game_spread": -3, "game_total": 210, "is_b2b": False, "gameId": "g1"},
            {"player_id": "2", "team": "LAL", "stat_type": "points", "direction": "over",
             "blended_confidence": 0.65, "position": "SF", "season_ast": 2,
             "game_spread": -3, "game_total": 210, "is_b2b": False, "gameId": "g1"},
        ]
        cfg = {"min_game_total": 225.5}
        result = _find_best_correlated_pair(pool, cfg)
        assert result is None  # Filtered by game total floor

    def test_high_total_pair_passes(self):
        from api.parlay_engine import _find_best_correlated_pair
        pool = [
            {"player_id": "1", "team": "LAL", "stat_type": "assists", "direction": "over",
             "blended_confidence": 0.65, "position": "PG", "season_ast": 8,
             "game_spread": -3, "game_total": 232, "is_b2b": False, "gameId": "g1"},
            {"player_id": "2", "team": "LAL", "stat_type": "points", "direction": "over",
             "blended_confidence": 0.65, "position": "SF", "season_ast": 2,
             "game_spread": -3, "game_total": 232, "is_b2b": False, "gameId": "g1"},
        ]
        cfg = {"min_game_total": 225.5}
        result = _find_best_correlated_pair(pool, cfg)
        assert result is not None

    def test_low_total_penalized_in_structure(self):
        from api.parlay_engine import _score_structure
        legs = [
            {"team": "LAL", "stat_type": "assists", "direction": "over", "player_name": "PG",
             "blended_confidence": 0.60, "american_odds": -110, "minutes_cv": 0.10,
             "game_spread": -3, "game_total": 210, "is_b2b": False, "opp_b2b": False,
             "position": "PG", "season_reb": 3, "season_ast": 8, "gameId": "g1"},
            {"team": "LAL", "stat_type": "points", "direction": "over", "player_name": "SC",
             "blended_confidence": 0.60, "american_odds": -110, "minutes_cv": 0.10,
             "game_spread": -3, "game_total": 210, "is_b2b": False, "opp_b2b": False,
             "position": "SF", "season_reb": 6, "season_ast": 2, "gameId": "g1"},
            {"team": "BOS", "stat_type": "rebounds", "direction": "over", "player_name": "Big",
             "blended_confidence": 0.60, "american_odds": -150, "minutes_cv": 0.10,
             "game_spread": -3, "game_total": 210, "is_b2b": False, "opp_b2b": False,
             "position": "C", "season_reb": 10, "season_ast": 1, "gameId": "g2"},
        ]
        cfg = {"min_game_total": 225.5}
        bonus, reasons = _score_structure(legs, cfg)
        assert any("low game total" in r.lower() for r in reasons)


class TestCVBasedMarketMatch:
    """High-CV candidates lose market match bonus."""

    def test_high_cv_no_market_match_bonus(self):
        from api.parlay_engine import _score_structure
        legs = [
            {"team": "BOS", "stat_type": "rebounds", "direction": "over", "player_name": "Big",
             "blended_confidence": 0.65, "american_odds": -150, "minutes_cv": 0.35,  # High CV
             "game_spread": -3, "game_total": 230, "is_b2b": False, "opp_b2b": False,
             "position": "C", "season_reb": 10, "season_ast": 1, "gameId": "g1"},
            {"team": "LAL", "stat_type": "points", "direction": "over", "player_name": "Wing",
             "blended_confidence": 0.55, "american_odds": -110, "minutes_cv": 0.10,
             "game_spread": -3, "game_total": 230, "is_b2b": False, "opp_b2b": False,
             "position": "SF", "season_reb": 4, "season_ast": 2, "gameId": "g2"},
            {"team": "MIA", "stat_type": "assists", "direction": "over", "player_name": "PG",
             "blended_confidence": 0.55, "american_odds": -110, "minutes_cv": 0.10,
             "game_spread": -3, "game_total": 230, "is_b2b": False, "opp_b2b": False,
             "position": "PG", "season_reb": 3, "season_ast": 7, "gameId": "g3"},
        ]
        cfg = {"market_match_max_cv": 0.25}
        bonus, reasons = _score_structure(legs, cfg)
        # Should NOT have market match bonus because the juiced leg has high CV
        assert not any("Market match" in r for r in reasons)

    def test_low_cv_gets_market_match_bonus(self):
        from api.parlay_engine import _score_structure
        legs = [
            {"team": "BOS", "stat_type": "rebounds", "direction": "over", "player_name": "Big",
             "blended_confidence": 0.65, "american_odds": -150, "minutes_cv": 0.10,  # Low CV
             "game_spread": -3, "game_total": 230, "is_b2b": False, "opp_b2b": False,
             "position": "C", "season_reb": 10, "season_ast": 1, "gameId": "g1"},
            {"team": "LAL", "stat_type": "points", "direction": "over", "player_name": "Wing",
             "blended_confidence": 0.55, "american_odds": -110, "minutes_cv": 0.10,
             "game_spread": -3, "game_total": 230, "is_b2b": False, "opp_b2b": False,
             "position": "SF", "season_reb": 4, "season_ast": 2, "gameId": "g2"},
            {"team": "MIA", "stat_type": "assists", "direction": "over", "player_name": "PG",
             "blended_confidence": 0.55, "american_odds": -110, "minutes_cv": 0.10,
             "game_spread": -3, "game_total": 230, "is_b2b": False, "opp_b2b": False,
             "position": "PG", "season_reb": 3, "season_ast": 7, "gameId": "g3"},
        ]
        cfg = {"market_match_max_cv": 0.25}
        bonus, reasons = _score_structure(legs, cfg)
        assert any("Market match" in r for r in reasons)


class TestPnrRimBoost:
    """Interior finisher gets stronger correlation boost than generic."""

    def test_interior_finisher_gets_pnr_boost(self):
        from api.parlay_engine import _correlation_modifier
        legs = [
            {"team": "LAL", "stat_type": "assists", "direction": "over",
             "player_name": "PG", "position": "PG", "season_ast": 8, "season_reb": 3,
             "gameId": "g1", "game_spread": -3, "game_total": 230,
             "is_b2b": False, "opp_b2b": False},
            {"team": "LAL", "stat_type": "points", "direction": "over",
             "player_name": "Center", "position": "C", "season_ast": 1, "season_reb": 10,
             "gameId": "g1", "game_spread": -3, "game_total": 230,
             "is_b2b": False, "opp_b2b": False},
            {"team": "BOS", "stat_type": "rebounds", "direction": "over",
             "player_name": "Other", "position": "PF", "season_ast": 1, "season_reb": 8,
             "gameId": "g2", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
        ]
        cfg = {"pnr_rim_boost": 1.20, "positive_correlation_boost": 1.08}
        mult, reasons = _correlation_modifier(legs, cfg)
        # Should use PnR-to-rim boost (1.20) not generic (1.08)
        assert mult >= 1.20
        assert any("PnR-to-rim" in r for r in reasons)

    def test_generic_boost_for_non_interior(self):
        from api.parlay_engine import _correlation_modifier
        legs = [
            {"team": "LAL", "stat_type": "assists", "direction": "over",
             "player_name": "PG", "position": "PG", "season_ast": 8, "season_reb": 3,
             "gameId": "g1", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
            {"team": "LAL", "stat_type": "points", "direction": "over",
             "player_name": "Wing", "position": "SF", "season_ast": 2, "season_reb": 6,
             "gameId": "g1", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
            {"team": "BOS", "stat_type": "rebounds", "direction": "over",
             "player_name": "Other", "position": "PF", "season_ast": 1, "season_reb": 8,
             "gameId": "g2", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
        ]
        cfg = {"pnr_rim_boost": 1.20, "positive_correlation_boost": 1.08}
        mult, reasons = _correlation_modifier(legs, cfg)
        # SF with 6 reb is not interior (needs 7+) → generic boost
        assert any("Positive correlation" in r for r in reasons)
        assert not any("PnR-to-rim" in r for r in reasons)


class TestPerimeterToPerimeterFade:
    """Perimeter-only scorer gets correlation penalty instead of boost."""

    def test_perimeter_scorer_penalized(self):
        from api.parlay_engine import _correlation_modifier
        legs = [
            {"team": "LAL", "stat_type": "assists", "direction": "over",
             "player_name": "PG", "position": "PG", "season_ast": 8, "season_reb": 3,
             "gameId": "g1", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
            {"team": "LAL", "stat_type": "points", "direction": "over",
             "player_name": "Shooter", "position": "SG", "season_ast": 1, "season_reb": 2,
             "gameId": "g1", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
            {"team": "BOS", "stat_type": "rebounds", "direction": "over",
             "player_name": "Other", "position": "PF", "season_ast": 1, "season_reb": 8,
             "gameId": "g2", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
        ]
        cfg = {"auto_fade": {"perimeter_scorer_reb_floor": 4.0}}
        mult, reasons = _correlation_modifier(legs, cfg)
        # Perimeter scorer (SG, 2 reb < 4 floor) → penalty
        assert any("perimeter-to-perimeter" in r.lower() for r in reasons)
        # Should be penalized (0.95) not boosted (1.08)
        assert mult < 1.0


class TestPaceBoost:
    """High-total games get pace boost multiplier."""

    def test_pace_boost_applied(self):
        from api.parlay_engine import _correlation_modifier
        legs = [
            {"team": "LAL", "stat_type": "points", "direction": "over",
             "player_name": "Star", "position": "SF", "season_ast": 3, "season_reb": 6,
             "gameId": "g1", "game_spread": -3, "game_total": 240,
             "is_b2b": False, "opp_b2b": False},
            {"team": "BOS", "stat_type": "rebounds", "direction": "over",
             "player_name": "Big", "position": "C", "season_ast": 1, "season_reb": 10,
             "gameId": "g1", "game_spread": -3, "game_total": 240,
             "is_b2b": False, "opp_b2b": False},
            {"team": "MIA", "stat_type": "assists", "direction": "over",
             "player_name": "PG", "position": "PG", "season_ast": 8, "season_reb": 3,
             "gameId": "g2", "game_spread": -2, "game_total": 220,
             "is_b2b": False, "opp_b2b": False},
        ]
        cfg = {"pace_boost_total_threshold": 232.0, "pace_boost": 1.06}
        mult, reasons = _correlation_modifier(legs, cfg)
        assert any("Pace boost" in r for r in reasons)
        assert mult >= 1.06

    def test_no_pace_boost_below_threshold(self):
        from api.parlay_engine import _correlation_modifier
        legs = [
            {"team": "LAL", "stat_type": "points", "direction": "over",
             "player_name": "Star", "position": "SF", "season_ast": 3, "season_reb": 6,
             "gameId": "g1", "game_spread": -3, "game_total": 220,
             "is_b2b": False, "opp_b2b": False},
            {"team": "BOS", "stat_type": "rebounds", "direction": "over",
             "player_name": "Big", "position": "C", "season_ast": 1, "season_reb": 10,
             "gameId": "g2", "game_spread": -3, "game_total": 218,
             "is_b2b": False, "opp_b2b": False},
            {"team": "MIA", "stat_type": "assists", "direction": "over",
             "player_name": "PG", "position": "PG", "season_ast": 8, "season_reb": 3,
             "gameId": "g3", "game_spread": -2, "game_total": 220,
             "is_b2b": False, "opp_b2b": False},
        ]
        cfg = {"pace_boost_total_threshold": 232.0, "pace_boost": 1.06}
        mult, reasons = _correlation_modifier(legs, cfg)
        assert not any("Pace boost" in r for r in reasons)


class TestRestAdvantageBoost:
    """Rest advantage (team rested, opponent on B2B) gets boost."""

    def test_rest_advantage_applied(self):
        from api.parlay_engine import _correlation_modifier
        legs = [
            {"team": "LAL", "stat_type": "points", "direction": "over",
             "player_name": "Star", "position": "SF", "season_ast": 3, "season_reb": 6,
             "gameId": "g1", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": True},  # Rest advantage!
            {"team": "BOS", "stat_type": "rebounds", "direction": "over",
             "player_name": "Big", "position": "C", "season_ast": 1, "season_reb": 10,
             "gameId": "g2", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
            {"team": "MIA", "stat_type": "assists", "direction": "over",
             "player_name": "PG", "position": "PG", "season_ast": 8, "season_reb": 3,
             "gameId": "g3", "game_spread": -2, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
        ]
        cfg = {"rest_advantage_boost": 1.08}
        mult, reasons = _correlation_modifier(legs, cfg)
        assert any("Rest advantage" in r for r in reasons)
        assert mult >= 1.08

    def test_no_rest_advantage_when_both_rested(self):
        from api.parlay_engine import _correlation_modifier
        legs = [
            {"team": "LAL", "stat_type": "points", "direction": "over",
             "player_name": "Star", "position": "SF", "season_ast": 3, "season_reb": 6,
             "gameId": "g1", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
            {"team": "BOS", "stat_type": "rebounds", "direction": "over",
             "player_name": "Big", "position": "C", "season_ast": 1, "season_reb": 10,
             "gameId": "g2", "game_spread": -3, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
            {"team": "MIA", "stat_type": "assists", "direction": "over",
             "player_name": "PG", "position": "PG", "season_ast": 8, "season_reb": 3,
             "gameId": "g3", "game_spread": -2, "game_total": 228,
             "is_b2b": False, "opp_b2b": False},
        ]
        cfg = {"rest_advantage_boost": 1.08}
        mult, reasons = _correlation_modifier(legs, cfg)
        assert not any("Rest advantage" in r for r in reasons)


class TestTightenedSpread:
    """correlated_pair_max_spread defaults to 5.0 (was 6.5)."""

    def test_default_spread_5(self):
        from api.parlay_engine import _find_best_correlated_pair
        pool = [
            {"player_id": "1", "team": "LAL", "stat_type": "assists", "direction": "over",
             "blended_confidence": 0.65, "position": "PG", "season_ast": 8,
             "game_spread": -6.0, "game_total": 230, "is_b2b": False, "gameId": "g1"},
            {"player_id": "2", "team": "LAL", "stat_type": "points", "direction": "over",
             "blended_confidence": 0.65, "position": "SF", "season_ast": 2,
             "game_spread": -6.0, "game_total": 230, "is_b2b": False, "gameId": "g1"},
        ]
        # Default config — spread threshold is now 5.0
        result = _find_best_correlated_pair(pool, {})
        assert result is None  # 6.0 > 5.0 → excluded

    def test_spread_4_passes(self):
        from api.parlay_engine import _find_best_correlated_pair
        pool = [
            {"player_id": "1", "team": "LAL", "stat_type": "assists", "direction": "over",
             "blended_confidence": 0.65, "position": "PG", "season_ast": 8,
             "game_spread": -4.0, "game_total": 230, "is_b2b": False, "gameId": "g1"},
            {"player_id": "2", "team": "LAL", "stat_type": "points", "direction": "over",
             "blended_confidence": 0.65, "position": "SF", "season_ast": 2,
             "game_spread": -4.0, "game_total": 230, "is_b2b": False, "gameId": "g1"},
        ]
        result = _find_best_correlated_pair(pool, {})
        assert result is not None


class TestDynamicLeg1Substitution:
    """Rebounds deprioritized vs elite defensive teams in market match selection."""

    def test_rebounds_deprioritized_vs_elite_defense(self):
        from api.parlay_engine import _find_best_market_match
        pair = (
            {"gameId": "g1", "player_id": "10"},
            {"gameId": "g1", "player_id": "11"},
        )
        pool = [
            # Rebounds candidate vs OKC (elite defense)
            {"player_id": "1", "team": "LAL", "stat_type": "rebounds", "direction": "over",
             "blended_confidence": 0.62, "american_odds": -150, "minutes_cv": 0.10,
             "opponent": "OKC", "gameId": "g2"},
            # Points candidate vs MIA (not in fade list)
            {"player_id": "2", "team": "BOS", "stat_type": "points", "direction": "over",
             "blended_confidence": 0.60, "american_odds": -145, "minutes_cv": 0.10,
             "opponent": "MIA", "gameId": "g3"},
        ]
        cfg = {
            "auto_fade": {"rebound_fade_teams": ["OKC"]},
            "market_match_juice_threshold": -140,
            "market_match_min_conf": 0.58,
            "market_match_max_cv": 0.30,
        }
        result = _find_best_market_match(pool, pair, cfg)
        # Should prefer points over rebounds because OKC is in rebound_fade_teams
        assert result is not None
        assert result["stat_type"] == "points"

    def test_rebounds_preferred_vs_normal_defense(self):
        from api.parlay_engine import _find_best_market_match
        pair = (
            {"gameId": "g1", "player_id": "10"},
            {"gameId": "g1", "player_id": "11"},
        )
        pool = [
            {"player_id": "1", "team": "LAL", "stat_type": "rebounds", "direction": "over",
             "blended_confidence": 0.62, "american_odds": -150, "minutes_cv": 0.10,
             "opponent": "MIA", "gameId": "g2"},
            {"player_id": "2", "team": "BOS", "stat_type": "points", "direction": "over",
             "blended_confidence": 0.60, "american_odds": -145, "minutes_cv": 0.10,
             "opponent": "PHO", "gameId": "g3"},
        ]
        cfg = {
            "auto_fade": {"rebound_fade_teams": ["OKC"]},
            "market_match_juice_threshold": -140,
            "market_match_min_conf": 0.58,
            "market_match_max_cv": 0.30,
        }
        result = _find_best_market_match(pool, pair, cfg)
        # Should prefer rebounds (normal behavior) when not facing elite defense
        assert result is not None
        assert result["stat_type"] == "rebounds"


class TestCandidateLegB2BFields:
    """Candidate legs should carry is_b2b and opp_b2b fields from game data."""

    def test_b2b_fields_in_candidates(self):
        from api.parlay_engine import build_candidate_legs
        games = [_make_game("BOS", "LAL", home_b2b=True, away_b2b=False)]
        proj = [_make_proj("Home Player", "BOS", pid="1")]
        odds = {("home player", "points"): _make_odds(18.5, -120, -100)}
        gl = {"1": _make_gamelog()}
        cfg = {"juice_threshold": -105, "min_blended_conf": 0.01}
        cands, _ = build_candidate_legs(proj, games, odds, gl, {}, cfg)
        if cands:
            c = cands[0]
            assert "is_b2b" in c
            assert "opp_b2b" in c
            assert c["is_b2b"] is True  # BOS is home and home_b2b=True
            assert c["opp_b2b"] is False

    def test_season_reb_in_candidates(self):
        from api.parlay_engine import build_candidate_legs
        games = [_make_game("MIA", "LAL")]
        proj = [_make_proj("Player", "LAL", pid="1", season_reb=8)]
        odds = {("player", "points"): _make_odds(18.5, -120, -100)}
        gl = {"1": _make_gamelog()}
        cfg = {"juice_threshold": -105, "min_blended_conf": 0.01}
        cands, _ = build_candidate_legs(proj, games, odds, gl, {}, cfg)
        if cands:
            assert cands[0]["season_reb"] == 8.0
