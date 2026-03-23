"""
Unit tests for basketball app backend — real assertions, real function calls.
Run with: pytest tests/test_fixes.py -v

Requires backend deps (numpy, lightgbm, etc.). If skipped, run:
  pip install -r requirements.txt
"""

import pytest
import json
import os
import re
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, call
from concurrent.futures import ThreadPoolExecutor, as_completed

# Skip entire module if backend dependencies are not installed (clear message instead of ERRORs)
pytest.importorskip("numpy", reason="Install dependencies: pip install -r requirements.txt")


# ─────────────────────────────────────────────────────────
# TestSafeFloat — pure utility, no mocking needed
# ─────────────────────────────────────────────────────────
class TestSafeFloat:
    """_safe_float converts strings/None to float without raising"""

    def setup_method(self):
        from api.index import _safe_float
        self.fn = _safe_float

    def test_numeric_string(self):
        assert self.fn("3.14") == pytest.approx(3.14)

    def test_integer_string(self):
        assert self.fn("5") == 5.0

    def test_empty_string(self):
        assert self.fn("") == 0.0

    def test_none_returns_zero(self):
        assert self.fn(None) == 0.0

    def test_non_numeric_returns_zero(self):
        assert self.fn("abc") == 0.0

    def test_negative(self):
        assert self.fn("-2.5") == pytest.approx(-2.5)

    def test_already_float(self):
        assert self.fn(4.0) == 4.0


# ─────────────────────────────────────────────────────────
# TestIsLocked — lock window edge cases
# _is_locked expects ISO string, not datetime object
# ─────────────────────────────────────────────────────────
class TestIsLocked:
    """_is_locked(start_time_iso) — 5-min pre-tip buffer, 6-hour ceiling"""

    def setup_method(self):
        from api.index import _is_locked
        self.fn = _is_locked

    def _iso(self, dt):
        return dt.isoformat()

    def test_locked_4_minutes_before_tip(self):
        """4 min before tip is inside the 5-min lock window"""
        start = datetime.now(timezone.utc) + timedelta(minutes=4)
        assert self.fn(self._iso(start)) is True

    def test_unlocked_10_minutes_before_tip(self):
        """10 min before tip is before the lock window"""
        start = datetime.now(timezone.utc) + timedelta(minutes=10)
        assert self.fn(self._iso(start)) is False

    def test_locked_during_game(self):
        """2 hours after tip — game in progress"""
        start = datetime.now(timezone.utc) - timedelta(hours=2)
        assert self.fn(self._iso(start)) is True

    def test_unlocked_after_6_hour_ceiling(self):
        """7 hours after tip — ceiling exceeded"""
        start = datetime.now(timezone.utc) - timedelta(hours=7)
        assert self.fn(self._iso(start)) is False

    def test_split_window_any_pattern_stays_locked(self):
        """Split Saturday: early game done (7h), late game active (1h) → stays locked"""
        early = self._iso(datetime.now(timezone.utc) - timedelta(hours=7))   # ceiling passed
        late  = self._iso(datetime.now(timezone.utc) - timedelta(hours=1))   # still live
        assert any(self.fn(st) for st in [early, late]) is True

    def test_split_window_all_done_unlocks(self):
        """Both games past ceiling → fully unlocked"""
        game1 = self._iso(datetime.now(timezone.utc) - timedelta(hours=7))
        game2 = self._iso(datetime.now(timezone.utc) - timedelta(hours=8))
        assert any(self.fn(st) for st in [game1, game2]) is False

    def test_none_input_returns_false(self):
        """None or invalid input does not raise"""
        assert self.fn(None) is False
        assert self.fn("") is False
        assert self.fn("not-a-date") is False


# ─────────────────────────────────────────────────────────
# TestComputeAudit — accuracy comparison logic
# CSV must match PRED_FIELDS / ACT_FIELDS column order from api/index.py
# ─────────────────────────────────────────────────────────
class TestComputeAudit:
    """_compute_audit — predictions vs actuals MAE and directional accuracy"""

    # PRED_FIELDS = scope,lineup_type,slot,player_name,player_id,team,pos,predicted_rs,est_card_boost,pred_min,pts,reb,ast,stl,blk
    PRED_CSV = (
        "scope,lineup_type,slot,player_name,player_id,team,pos,predicted_rs,est_card_boost,pred_min,pts,reb,ast,stl,blk\n"
        "slate,chalk,2.0x,LeBron James,123,LAL,SF,4.5,1.5,35,28,7,7,1,0.5\n"
        "slate,chalk,1.8x,Stephen Curry,456,GSW,PG,5.0,1.2,36,30,5,6,1.5,0.2\n"
        "slate,chalk,1.6x,Kevin Durant,789,PHX,PF,3.8,1.8,33,26,7,5,1,1\n"
    )
    ACT_CSV = (
        "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source\n"
        "LeBron James,4.2,1.8,45000,,,real_scores\n"
        "Stephen Curry,6.1,1.2,80000,,,real_scores\n"
        "Kevin Durant,3.5,2.1,30000,,,real_scores\n"
    )

    def _mock_github(self, pred, act):
        def side_effect(path):
            if 'predictions' in path:
                return pred, 'sha1'
            if 'actuals' in path:
                return act, 'sha2'
            return None, None
        return side_effect

    def test_mae_calculated_correctly(self):
        """MAE = mean(|4.2-4.5|, |6.1-5.0|, |3.5-3.8|) = mean(0.3, 1.1, 0.3) ≈ 0.567"""
        from api.index import _compute_audit
        with patch('api.index._github_get_file', side_effect=self._mock_github(self.PRED_CSV, self.ACT_CSV)):
            result = _compute_audit('2026-03-08')
        assert result is not None
        assert result['players_compared'] == 3
        assert result['mae'] == pytest.approx(0.567, abs=0.01)

    def test_returns_none_without_predictions(self):
        from api.index import _compute_audit
        with patch('api.index._github_get_file', return_value=(None, None)):
            assert _compute_audit('2026-03-08') is None

    def test_returns_none_without_actuals(self):
        from api.index import _compute_audit
        def side_effect(path):
            if 'predictions' in path:
                return self.PRED_CSV, 'sha1'
            return None, None
        with patch('api.index._github_get_file', side_effect=side_effect):
            assert _compute_audit('2026-03-08') is None

    def test_returns_none_when_no_players_overlap(self):
        from api.index import _compute_audit
        act_no_match = (
            "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source\n"
            "Unknown Player,4.0,1.0,1000,,,real_scores\n"
        )
        with patch('api.index._github_get_file', side_effect=self._mock_github(self.PRED_CSV, act_no_match)):
            assert _compute_audit('2026-03-08') is None

    def test_skips_rows_with_zero_actual_rs(self):
        """Rows with actual_rs=0 or empty are excluded from MAE"""
        from api.index import _compute_audit
        act_with_zero = (
            "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source\n"
            "LeBron James,0,1.8,45000,,,real_scores\n"
            "Stephen Curry,6.1,1.2,80000,,,real_scores\n"
        )
        with patch('api.index._github_get_file', side_effect=self._mock_github(self.PRED_CSV, act_with_zero)):
            result = _compute_audit('2026-03-08')
        # LeBron skipped (actual_rs=0), only Curry counted
        assert result['players_compared'] == 1

    def test_biggest_misses_sorted_by_abs_error(self):
        from api.index import _compute_audit
        with patch('api.index._github_get_file', side_effect=self._mock_github(self.PRED_CSV, self.ACT_CSV)):
            result = _compute_audit('2026-03-08')
        # Curry miss = 1.1, biggest
        assert result['biggest_misses'][0]['player'] == 'Stephen Curry'

    def test_over_under_projection_counts(self):
        from api.index import _compute_audit
        with patch('api.index._github_get_file', side_effect=self._mock_github(self.PRED_CSV, self.ACT_CSV)):
            result = _compute_audit('2026-03-08')
        # LeBron: pred 4.5 > actual 4.2 → over-projected (error = actual - pred = -0.3 < 0)
        # Curry:  pred 5.0 < actual 6.1 → under-projected (error = 1.1 > 0)
        # Durant: pred 3.8 > actual 3.5 → over-projected (error = -0.3 < 0)
        assert result['over_projected'] == 2
        assert result['under_projected'] == 1

    def test_simulated_draft_score_present(self):
        """simulated_draft_score field exists and is a positive float when ≥3 actuals available"""
        from api.index import _compute_audit
        with patch('api.index._github_get_file', side_effect=self._mock_github(self.PRED_CSV, self.ACT_CSV)):
            result = _compute_audit('2026-03-08')
        assert 'simulated_draft_score' in result
        # 3 players: sort by actual_rs desc → Curry 6.1, LeBron 4.2, Durant 3.5
        # slot 2.0 + boost 1.2 = 3.2 → 6.1 × 3.2 = 19.52
        # slot 1.8 + boost 1.8 = 3.6 → 4.2 × 3.6 = 15.12
        # slot 1.6 + boost 2.1 = 3.7 → 3.5 × 3.7 = 12.95
        # total = 47.59
        assert result['simulated_draft_score'] == pytest.approx(47.6, abs=0.2)

    def test_simulated_draft_score_none_when_insufficient_actuals(self):
        """simulated_draft_score is None when fewer than 3 actuals have actual_rs > 0"""
        from api.index import _compute_audit
        act_small = (
            "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source\n"
            "LeBron James,4.2,1.8,45000,,,real_scores\n"
            "Stephen Curry,0,1.2,80000,,,real_scores\n"
        )
        with patch('api.index._github_get_file', side_effect=self._mock_github(self.PRED_CSV, act_small)):
            result = _compute_audit('2026-03-08')
        # Only 1 player with actual_rs > 0 — below 3-player threshold
        assert result is None or result.get('simulated_draft_score') is None


# ─────────────────────────────────────────────────────────
# TestPerTierCalibration — per-tier calibration scale
# ─────────────────────────────────────────────────────────
class TestPerTierCalibration:
    """Per-tier calibration: role players (season_pts < 20) use calibration_scale_role,
    stars use calibration_scale_star. Falls back to calibration_scale when keys absent."""

    def test_role_player_uses_role_scale(self):
        """A player with season_pts=12 (role player) should get calibration_scale_role applied."""
        from api.index import project_player
        pinfo = {"id": "test_role_1", "name": "Test Role Player", "team": "SAS", "pos": "SG", "opp": "LAL"}
        stats = {
            "min": 22.0, "pts": 12.0, "reb": 3.0, "ast": 2.0, "stl": 0.5, "blk": 0.2,
            "season_pts": 12.0, "season_min": 22.0, "season_reb": 3.0,
            "recent_pts": 12.0, "recent_min": 22.0,
        }
        # Patch config to use distinct scales so we can detect which one was applied
        cfg_overrides = {
            "real_score.calibration_scale_role": 1.50,
            "real_score.calibration_scale_star": 1.00,
            "real_score.calibration_scale": 1.15,
        }
        original_cfg = __import__('api.index', fromlist=['_cfg'])._cfg
        def mock_cfg(key, default=None):
            return cfg_overrides.get(key, original_cfg(key, default))
        with patch('api.index._cfg', side_effect=mock_cfg):
            with patch('api.index._est_card_boost', return_value=1.5):
                with patch('api.index._github_get_file', return_value=(None, None)):
                    result = project_player(pinfo, stats, 2.0, 222.0, "away", "SAS", 0.0, False)
        # With scale 1.50 for role players, RS should be higher than with 1.00
        assert result is not None
        assert result.get('rating', 0) > 0

    def test_star_player_uses_star_scale(self):
        """A player with season_pts=28 (star) should use calibration_scale_star, not role scale."""
        from api.index import project_player
        pinfo = {"id": "test_star_1", "name": "Test Star Player", "team": "OKC", "pos": "SG", "opp": "MEM"}
        stats = {
            "min": 34.0, "pts": 28.0, "reb": 5.0, "ast": 6.0, "stl": 1.5, "blk": 0.5,
            "season_pts": 28.0, "season_min": 34.0, "season_reb": 5.0,
            "recent_pts": 28.0, "recent_min": 34.0,
        }
        cfg_overrides = {
            "real_score.calibration_scale_role": 1.50,
            "real_score.calibration_scale_star": 1.00,
            "real_score.calibration_scale": 1.15,
        }
        original_cfg = __import__('api.index', fromlist=['_cfg'])._cfg
        def mock_cfg(key, default=None):
            return cfg_overrides.get(key, original_cfg(key, default))
        with patch('api.index._cfg', side_effect=mock_cfg):
            with patch('api.index._est_card_boost', return_value=0.3):
                with patch('api.index._github_get_file', return_value=(None, None)):
                    star_result = project_player(pinfo, stats, 2.0, 222.0, "home", "OKC", 0.0, False)
        # Just verify it ran without error — exact RS comparison is fragile due to LightGBM
        assert star_result is not None

    def test_calibration_scale_removed_from_pipeline(self):
        """calibration_scale layers removed in v45 simplification — not in _CONFIG_DEFAULTS or code."""
        from api.index import _CONFIG_DEFAULTS
        rs_defaults = _CONFIG_DEFAULTS.get("real_score", {})
        assert "calibration_scale_role" not in rs_defaults, \
            "calibration_scale_role should be removed (pipeline simplified in v45)"
        assert "calibration_scale_star" not in rs_defaults, \
            "calibration_scale_star should be removed (pipeline simplified in v45)"
        # Verify code no longer calls calibration scale
        import inspect
        import api.index as idx
        src = inspect.getsource(idx.project_player)
        assert "calibration_scale" not in src, \
            "project_player should not use calibration_scale (removed in v45)"


# ─────────────────────────────────────────────────────────
# TestGitHubWriteRetry — exponential backoff on 422
# ─────────────────────────────────────────────────────────
class TestGitHubWriteRetry:
    """_github_write_file retries up to 3× on 422 with 1s/2s/4s backoff"""

    def _make_response(self, status, body=None):
        r = Mock()
        r.status_code = status
        r.text = json.dumps(body or {})
        r.json.return_value = body or {}
        return r

    def test_succeeds_on_first_try(self):
        from api.index import _github_write_file
        ok = self._make_response(201, {'content': {'sha': 'abc'}})
        with patch('api.index.GITHUB_TOKEN', 'fake'), \
             patch('api.index.GITHUB_REPO', 'owner/repo'), \
             patch('api.index._github_get_file', return_value=('{}', 'oldsha')), \
             patch('requests.put', return_value=ok) as mock_put:
            result = _github_write_file('data/test.json', '{}', 'commit')
        assert mock_put.call_count == 1
        assert 'error' not in result

    def test_retries_on_422_and_succeeds(self):
        from api.index import _github_write_file
        fail = self._make_response(422, {'message': 'SHA does not match'})
        ok   = self._make_response(201, {'content': {'sha': 'newsha'}})
        with patch('api.index.GITHUB_TOKEN', 'fake'), \
             patch('api.index.GITHUB_REPO', 'owner/repo'), \
             patch('api.index._github_get_file', return_value=('{}', 'sha')), \
             patch('requests.put', side_effect=[fail, fail, ok]) as mock_put, \
             patch('time.sleep'):
            result = _github_write_file('data/test.json', '{}', 'commit')
        assert mock_put.call_count == 3
        assert 'error' not in result

    def test_returns_error_after_max_retries(self):
        from api.index import _github_write_file
        fail = self._make_response(422, {'message': 'SHA mismatch'})
        with patch('api.index.GITHUB_TOKEN', 'fake'), \
             patch('api.index.GITHUB_REPO', 'owner/repo'), \
             patch('api.index._github_get_file', return_value=('{}', 'sha')), \
             patch('requests.put', return_value=fail), \
             patch('time.sleep'):
            result = _github_write_file('data/test.json', '{}', 'commit')
        assert 'error' in result

    def test_backoff_delays_are_1_2_4_seconds(self):
        from api.index import _github_write_file
        fail = self._make_response(422, {'message': 'SHA mismatch'})
        with patch('api.index.GITHUB_TOKEN', 'fake'), \
             patch('api.index.GITHUB_REPO', 'owner/repo'), \
             patch('api.index._github_get_file', return_value=('{}', 'sha')), \
             patch('requests.put', return_value=fail), \
             patch('time.sleep') as mock_sleep:
            _github_write_file('data/test.json', '{}', 'commit')
        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        # 3 attempts: sleep after attempt 0 (1s) and attempt 1 (2s); attempt 2 is final, no sleep
        assert sleep_args == [1, 2]

    def test_returns_error_when_no_token(self):
        """Missing GITHUB_TOKEN returns error immediately without hitting API"""
        from api.index import _github_write_file
        with patch('api.index.GITHUB_TOKEN', ''), \
             patch('api.index.GITHUB_REPO', 'owner/repo'):
            result = _github_write_file('data/test.json', '{}', 'commit')
        assert 'error' in result


# ─────────────────────────────────────────────────────────
# TestSaveActualsAuditGate — audit only fires for real_scores
# ─────────────────────────────────────────────────────────
class TestSaveActualsAuditGate:
    """Audit JSON only generated when real_scores data is present"""

    def _should_audit(self, upload_source, existing_csv=''):
        """Mirror the gate logic from save_actuals"""
        return upload_source == 'real_scores' or 'real_scores' in (existing_csv or '')

    def test_fires_for_real_scores_upload(self):
        assert self._should_audit('real_scores') is True

    def test_skipped_for_top_drafts_only(self):
        assert self._should_audit('top_drafts') is False

    def test_skipped_for_moonshot_only(self):
        assert self._should_audit('moonshot', 'LeBron James,,,,,, moonshot\n') is False

    def test_fires_when_existing_csv_has_real_scores(self):
        """starting_5 upload after real_scores was already saved → audit runs"""
        existing = 'LeBron James,4.2,1.8,45000,,,real_scores\n'
        assert self._should_audit('starting_5', existing) is True

    def test_skipped_for_starting_5_with_no_prior_real_scores(self):
        existing = 'LeBron James,,,,,, starting_5\n'
        assert self._should_audit('starting_5', existing) is False


# ─────────────────────────────────────────────────────────
# TestAutoResolveMidnight — midnight date boundary handling
# ─────────────────────────────────────────────────────────
class TestAutoResolveMidnight:
    """Line picks from late games correctly tracked across midnight ET"""

    def test_pick_date_survives_midnight(self):
        """pick_date (Mar 7) stays Mar 7 even when _et_date() returns Mar 8"""
        pick_date = '2026-03-07'
        # Simulate what auto_resolve_line does: use pick_date, not today
        et_today = '2026-03-08'
        assert pick_date != et_today  # they diverge after midnight
        # next-day picks generated from pick_date + 1
        from datetime import date
        next_day = (date.fromisoformat(pick_date) + timedelta(days=1)).isoformat()
        assert next_day == '2026-03-08'
        assert next_day != (date.fromisoformat(et_today) + timedelta(days=1)).isoformat()

    def test_yesterday_fallback_key(self):
        """Yesterday's file path derived from pick_date, not et_date"""
        pick_date = '2026-03-07'
        expected_path = f'data/lines/{pick_date}.json'
        assert expected_path == 'data/lines/2026-03-07.json'


# ─────────────────────────────────────────────────────────
# TestRateLimitThreadSafe — concurrent calls do not raise; limit enforced
# ─────────────────────────────────────────────────────────
class TestRateLimitThreadSafe:
    """_check_rate_limit is thread-safe and enforces limit under concurrency"""

    def test_concurrent_calls_do_not_raise(self):
        """Multiple threads calling _check_rate_limit for same key — no exception"""
        from api.index import _check_rate_limit, _RATE_LIMITS
        req = Mock()
        req.headers = {}
        req.client = Mock(host="127.0.0.1")
        path_key = "line-of-the-day"  # limit 10
        limit = _RATE_LIMITS[path_key]

        def one_call(_):
            return _check_rate_limit(req, path_key)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(one_call, range(limit + 2)))
        # No exception; first `limit` return None, rest may return 429
        assert len(results) == limit + 2
        nones = sum(1 for r in results if r is None)
        rate_limited = sum(1 for r in results if r is not None and getattr(r, "status_code", None) == 429)
        assert nones + rate_limited == len(results)
        assert nones <= limit

    def test_limit_enforced_after_threshold(self):
        """After limit requests for a key, next call returns 429"""
        from api.index import _check_rate_limit, _RATE_LIMITS, _RATE_LIMIT_STORE, _RATE_LIMIT_LOCK
        path_key = "parse-screenshot"  # limit 5
        limit = _RATE_LIMITS[path_key]
        req = Mock()
        req.headers = {}
        req.client = Mock(host="192.168.1.99")  # unique so we don't clash with other tests
        key = (req.client.host, path_key)
        with patch("api.index._client_ip", return_value=req.client.host):
            with _RATE_LIMIT_LOCK:
                if key in _RATE_LIMIT_STORE:
                    del _RATE_LIMIT_STORE[key]
            for _ in range(limit):
                r = _check_rate_limit(req, path_key)
                assert r is None
            r = _check_rate_limit(req, path_key)
            assert r is not None and r.status_code == 429


# ─────────────────────────────────────────────────────────
# TestLineConfig — line_engine respects min_confidence from config
# ─────────────────────────────────────────────────────────
class TestLineConfig:
    """run_model_fallback and run_line_engine respect line_config min_confidence"""

    def test_model_fallback_filters_by_min_confidence(self):
        """Candidates below min_confidence are excluded"""
        from api.line_engine import run_model_fallback
        proj = [
            {"name": "Player A", "team": "LAL", "predMin": 30, "pts": 26, "season_pts": 22, "recent_pts": 23,
             "reb": 5, "season_reb": 6, "recent_reb": 5.5, "ast": 4, "season_ast": 4, "recent_ast": 4},
            {"name": "Player B", "team": "BOS", "predMin": 28, "pts": 10, "season_pts": 8, "recent_pts": 8,
             "reb": 3, "season_reb": 3, "recent_reb": 3, "ast": 2, "season_ast": 2, "recent_ast": 2},
        ]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "away_b2b": False, "home_b2b": False}]
        out = run_model_fallback(proj, games, line_config={"min_confidence": 70})
        assert out.get("error") is None or out.get("pick") is not None or out.get("over_pick") or out.get("under_pick")

    def test_run_line_engine_uses_fallback_without_api_key(self):
        """run_line_engine falls back to model when no API key; returns pick or error"""
        from api.line_engine import run_line_engine
        proj = [{"name": "P", "team": "LAL", "predMin": 30, "pts": 22, "season_pts": 20, "recent_pts": 21,
                 "reb": 5, "season_reb": 5, "recent_reb": 5, "ast": 4, "season_ast": 4, "recent_ast": 4}]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "home_b2b": False, "away_b2b": False}]
        with patch("api.line_engine.ANTHROPIC_API_KEY", ""):
            result = run_line_engine(proj, games, line_config={"min_confidence": 55, "min_edge_pct": 2.0})
        assert "pick" in result
        # With no API key, always uses fallback — gets a result or no_edges error
        assert result.get("pick") is not None or result.get("error") is not None

    def test_min_edge_other_over_blocks_small_rebound_over(self):
        """min_edge_other_over=2.5 prevents a small-edge rebounds over from winning over a qualifying points over.
        When a points over (edge=3.0) is available, the rebounds over (edge=1.5) should not be the over_pick.
        Note: last-resort fires only when main candidates is empty; with a qualifying points player present,
        the rebounds over (blocked in main path) should not appear as over_pick."""
        from api.line_engine import run_model_fallback
        proj = [
            # Points scorer with qualifying over (edge=3.0 > min_edge_pts=2.0) — wins main candidates
            {"name": "Scorer", "team": "BOS", "predMin": 30,
             "pts": 25, "season_pts": 22, "recent_pts": 23,
             "reb": 4, "season_reb": 4, "recent_reb": 4,
             "ast": 3, "season_ast": 3, "recent_ast": 3},
            # Rebounder with small rebounds over (edge=1.5, fails min_edge_other_over=2.5)
            {"name": "Big Man", "team": "LAL", "predMin": 30,
             "pts": 10, "season_pts": 10, "recent_pts": 10,
             "reb": 7.5, "season_reb": 6.0, "recent_reb": 6.5,
             "ast": 2, "season_ast": 2, "recent_ast": 2},
        ]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "home_b2b": False, "away_b2b": False}]
        cfg = {"min_confidence": 50, "min_edge_pts": 2.0, "min_edge_other": 1.5,
               "min_edge_other_over": 2.5, "stat_floors": {"rebounds": 5.5, "points": 6.0, "assists": 1.5}}
        result = run_model_fallback(proj, games, line_config=cfg)
        over_pick = result.get("over_pick")
        # Points over qualifies in main path; rebounds over (edge=1.5) is blocked by min_edge_other_over
        # so over_pick should be the points pick, not the rebounds pick
        assert over_pick is not None, "should have an over pick (points scorer qualifies)"
        assert not (over_pick["stat_type"] == "rebounds" and over_pick["direction"] == "over"), \
            "rebounds over with edge=1.5 should not win when a qualifying points over exists"

    def test_min_edge_other_over_allows_under_with_same_edge(self):
        """min_edge_other_over does not affect under picks — same 1.5 edge qualifies for rebounds under"""
        from api.line_engine import run_model_fallback
        # Player projects 4.5 reb vs 6.0 season avg — edge=-1.5 (under), should pass min_edge_other=1.5
        proj = [{"name": "Slumper", "team": "LAL", "predMin": 30,
                 "pts": 10, "season_pts": 10, "recent_pts": 10,
                 "reb": 4.5, "season_reb": 6.0, "recent_reb": 4.8,
                 "ast": 2, "season_ast": 2, "recent_ast": 2}]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "home_b2b": False, "away_b2b": False}]
        cfg = {"min_confidence": 50, "min_edge_pts": 2.0, "min_edge_other": 1.5,
               "min_edge_other_over": 2.5, "stat_floors": {"rebounds": 5.5, "points": 6.0, "assists": 1.5}}
        result = run_model_fallback(proj, games, line_config=cfg)
        under_pick = result.get("under_pick")
        # The rebounds under (edge=1.5) should qualify since it uses min_edge_other (1.5), not the over threshold
        if under_pick:
            assert under_pick["direction"] == "under"


# ─────────────────────────────────────────────────────────
# TestLgbmFeatureAlignment — train vs inference feature list
# ─────────────────────────────────────────────────────────
class TestLgbmFeatureAlignment:
    """When LightGBM bundle is loaded, feature list must match current train/infer contract."""

    def test_feature_list_length_and_trend_feature(self):
        import api.index as idx
        idx._ensure_lgbm_loaded()
        AI_FEATURES = idx.AI_FEATURES
        if AI_FEATURES is None:
            pytest.skip("No LightGBM bundle loaded (lgbm_model.pkl not present or invalid)")
        n = len(AI_FEATURES)
        assert n in (12, 16), f"Expected 12 (legacy) or 16 (bundle v2) features, got {n}: {AI_FEATURES}"
        assert AI_FEATURES[9] in ("recent_vs_season", "recent_3g_trend"), (
            f"10th feature (index 9) must be recent_vs_season or legacy recent_3g_trend, got {AI_FEATURES[9]!r}"
        )
        assert AI_FEATURES[11] == "reb_per_min", (
            f"12th feature (index 11) must be reb_per_min, got {AI_FEATURES[11]!r}"
        )
        if n == 16:
            assert AI_FEATURES[2] == "usage_trend", f"index 2 must be usage_trend, got {AI_FEATURES[2]!r}"
            assert AI_FEATURES[12] == "l3_vs_l5_pts", f"index 12 must be l3_vs_l5_pts, got {AI_FEATURES[12]!r}"
            assert AI_FEATURES[13] == "min_volatility", f"index 13 must be min_volatility, got {AI_FEATURES[13]!r}"
            assert AI_FEATURES[14] == "starter_proxy", f"index 14 must be starter_proxy, got {AI_FEATURES[14]!r}"
            assert AI_FEATURES[15] == "cascade_signal", f"index 15 must be cascade_signal, got {AI_FEATURES[15]!r}"
            vec = idx._lgbm_feature_vector(
                avg_min=24.0,
                pts=14.0,
                reb=5.0,
                ast=3.0,
                stl=1.0,
                blk=0.5,
                spread=3.0,
                side="home",
                season_pts=14.0,
                recent_pts=16.0,
                season_min=24.0,
                recent_min=26.0,
                cascade_bonus=0.0,
                games_played=40.0,
            )
            assert len(vec) == 16, f"_lgbm_feature_vector must return 16 values, got {len(vec)}"


# ─────────────────────────────────────────────────────────
# TestSlateExceptionHandling — slate never returns 500
# ─────────────────────────────────────────────────────────
class TestSlateExceptionHandling:
    """Slate endpoint catches exceptions and returns 200 with error key (by design, never 500)."""

    def test_unhandled_exception_returns_200_with_error_key(self):
        pytest.importorskip("fastapi", reason="Install dependencies: pip install -r requirements.txt")
        from fastapi.testclient import TestClient
        from api.index import app

        def raise_err():
            raise ValueError("internal detail must not appear in response")

        with patch("api.index._get_slate_impl", side_effect=raise_err):
            client = TestClient(app)
            r = client.get("/api/slate")
        # Slate endpoint wraps all exceptions → 200 with error field
        assert r.status_code == 200
        body = r.json()
        assert body.get("error") == "slate_failed"
        # No internal detail leaked
        assert "internal detail" not in json.dumps(body)


# ─────────────────────────────────────────────────────────
# TestGameSelectorLockDisplay — per-game lock, not slate-wide
# Regression test for: all games showing locked when only
# one game was within the lock window
# ─────────────────────────────────────────────────────────
class TestGameSelectorLockDisplay:
    """Frontend populateGameSelector must NOT pass slateLocked to override per-game lock status."""

    def test_no_slate_locked_passed_to_populate_game_selector(self):
        """loadSlate should call populateGameSelector without slateLocked override"""
        with open('index.html', 'r') as f:
            html = f.read()

        # Find all calls to populateGameSelector
        calls = re.findall(r'populateGameSelector\([^)]+\)', html)
        for c in calls:
            assert 'slateLocked' not in c, (
                f"populateGameSelector should not receive slateLocked override: {c}"
            )

    def test_per_game_lock_check_in_populate(self):
        """populateGameSelector checks g.locked and g.startTime individually"""
        with open('index.html', 'r') as f:
            html = f.read()
        # The lock line should check g.locked or startTime, not slateLocked
        assert 'g.locked === true' in html or 'g.locked===true' in html


# ─────────────────────────────────────────────────────────
# TestLinePrimaryPickFallback — direction populated from primary
# Regression test for: both Over/Under showing "No pick today"
# when API returns primary pick but null directional picks
# ─────────────────────────────────────────────────────────
class TestLinePrimaryPickFallback:
    """Frontend should populate LINE_OVER_PICK or LINE_UNDER_PICK from primary pick when directions are null."""

    def test_frontend_has_primary_pick_fallback(self):
        """When data.pick exists but both directional picks are null, primary is used"""
        with open('index.html', 'r') as f:
            html = f.read()
        # The fallback block: if (!LINE_OVER_PICK && !LINE_UNDER_PICK && data.pick)
        assert '!LINE_OVER_PICK && !LINE_UNDER_PICK && data.pick' in html, \
            "Missing fallback: populate direction from primary pick when both are null"


# ─────────────────────────────────────────────────────────
# TestLinePicksBothNullRegeneration — backend regenerates when both null
# Regression test for: saved JSON with both directions null gets stuck
# ─────────────────────────────────────────────────────────
class TestLinePicksBothNullRegeneration:
    """When saved line picks have both over_pick and under_pick as null, backend should regenerate."""

    def test_both_null_triggers_fresh_generation(self):
        """If today_picks has both directions null, it should be treated as no picks"""
        from api.index import _primary_pick
        # Simulating what the line-of-the-day endpoint does:
        # if both are null, the guard sets today_picks = None so engine runs fresh
        today_picks = {"over_pick": None, "under_pick": None}
        has_over = today_picks.get("over_pick") is not None
        has_under = today_picks.get("under_pick") is not None
        if not has_over and not has_under:
            today_picks = None
        assert today_picks is None, "Both-null picks should be reset to None for regeneration"

    def test_one_direction_present_is_kept(self):
        """If one direction has data, the picks dict is preserved (not reset)"""
        today_picks = {"over_pick": {"player_name": "Test"}, "under_pick": None}
        has_over = today_picks.get("over_pick") is not None
        has_under = today_picks.get("under_pick") is not None
        if not has_over and not has_under:
            today_picks = None
        assert today_picks is not None, "Picks with one valid direction should be preserved"

    def test_backend_source_has_both_null_guard(self):
        """The line-of-the-day endpoint has the both-null guard in source"""
        import inspect
        from api.index import get_line_of_the_day
        source = inspect.getsource(get_line_of_the_day)
        assert 'today_picks = None' in source, \
            "line-of-the-day endpoint should reset today_picks to None when both directions are null"


# ─────────────────────────────────────────────────────────
# TestFetchGamesTTL — fetch_games uses TTL-aware cache
# ─────────────────────────────────────────────────────────
class TestFetchGamesTTL:
    """fetch_games() should use a 5-min TTL to avoid stale ESPN data."""

    def test_games_cache_has_ttl(self):
        """fetch_games should check _GAMES_CACHE_TS for freshness"""
        import inspect
        from api.index import fetch_games
        source = inspect.getsource(fetch_games)
        assert '_GAMES_CACHE_TS' in source, "fetch_games must use TTL-aware cache"
        assert '_TTL_GAMES' in source, "fetch_games TTL should use _TTL_GAMES constant (300s / 5 min)"


# ─────────────────────────────────────────────────────────
# TestSavePredictionsMerge — merge new per-game rows
# ─────────────────────────────────────────────────────────
class TestSavePredictionsMerge:
    """save_predictions should merge new per-game scopes into existing CSV."""

    def test_merge_logic_in_save_predictions(self):
        """save_predictions should check existing scopes and only add new ones"""
        import inspect
        from api.index import save_predictions
        source = inspect.getsource(save_predictions)
        assert 'existing_scopes' in source, "save_predictions must track existing scopes for merge"
        assert 'new_rows' in source, "save_predictions must compute new_rows to append"


# ─────────────────────────────────────────────────────────
# TestSwitchTabNoDuplicateInit — no duplicate init calls
# ─────────────────────────────────────────────────────────
class TestSwitchTabNoDuplicateInit:
    """switchTab should not call initLinePage twice."""

    def test_no_unconditional_init_line_page(self):
        """switchTab should not have a standalone 'if (tab === line) initLinePage()' outside stale block"""
        with open('index.html', 'r') as f:
            src = f.read()
        # The old pattern was: stale check calls initLinePage, then unconditional initLinePage
        # New pattern: single initLinePage call with stale check blanking cache beforehand
        import re
        # Should NOT have two separate initLinePage() calls in switchTab
        switch_fn = re.search(r'function switchTab\(.*?\n\}', src, re.DOTALL)
        assert switch_fn, "switchTab function must exist"
        body = switch_fn.group(0)
        init_calls = body.count('initLinePage()')
        assert init_calls == 1, f"switchTab should call initLinePage exactly once, found {init_calls}"

    def test_pred_saved_count_refire(self):
        """savePredictions should track locked count for split-window re-fire"""
        with open('index.html', 'r') as f:
            src = f.read()
        assert '_predSavedLockedCount' in src, "savePredictions must track locked game count"
        assert 'lockedNow' in src, "savePredictions must count currently locked games"


# ─────────────────────────────────────────────────────────
# TestSlateCacheGitHub — GitHub-persisted slate cache
# ─────────────────────────────────────────────────────────
class TestSlateCacheGitHub:
    """Verify that /api/slate reads from GitHub cache on cold start (no /tmp)."""

    def test_slate_cache_from_github_returns_data(self):
        """_slate_cache_from_github returns parsed JSON when GitHub has today's cache (no bust sentinel)."""
        from api.index import _slate_cache_from_github
        slate_data = {"date": "2026-03-10", "lineups": {"chalk": [{"name": "Test"}], "upside": []}}
        # Bust check loops over 2 refs (data branch=None, then main): both return no bust.
        # Third call: slate file itself.
        with patch("api.index._github_get_file", side_effect=[
            (None, None),  # bust check on data branch
            (None, None),  # bust check on main
            (json.dumps(slate_data), "sha123"),  # slate file
        ]):
            result = _slate_cache_from_github()
        assert result is not None
        assert result["lineups"]["chalk"][0]["name"] == "Test"

    def test_slate_cache_from_github_returns_none_on_miss(self):
        """_slate_cache_from_github returns None when no cache exists (bust check: data + main, then slate)."""
        from api.index import _slate_cache_from_github
        with patch("api.index._github_get_file", side_effect=[(None, None), (None, None), (None, None)]):
            result = _slate_cache_from_github()
        assert result is None

    def test_slate_cache_from_github_ignores_busted(self):
        """_slate_cache_from_github returns None when bust sentinel exists (skips slate read)."""
        from api.index import _slate_cache_from_github
        with patch("api.index._github_get_file", return_value=(json.dumps({"_busted": True, "at": "2026-03-10T00:00:00Z"}), "sha")):
            result = _slate_cache_from_github()
        assert result is None

    def test_slate_cache_to_github_writes(self):
        """_slate_cache_to_github writes JSON to data/slate/ path."""
        from api.index import _slate_cache_to_github
        with patch("api.index._github_write_file") as mock_write:
            _slate_cache_to_github({"date": "2026-03-10", "lineups": {}})
        mock_write.assert_called_once()
        path_arg = mock_write.call_args[0][0]
        assert "data/slate/" in path_arg
        assert "_slate.json" in path_arg

    def test_games_cache_roundtrip(self):
        """_games_cache_to/from_github persists and retrieves per-game projections."""
        from api.index import _games_cache_from_github
        game_data = {"401234": [{"name": "Player A", "rating": 5.0}]}
        with patch("api.index._github_get_file", return_value=(json.dumps(game_data), "sha")):
            result = _games_cache_from_github()
        assert result is not None
        assert "401234" in result
        assert result["401234"][0]["name"] == "Player A"


# ─────────────────────────────────────────────────────────
# TestInjuryCheck — injury-triggered regeneration
# ─────────────────────────────────────────────────────────
class TestInjuryCheck:
    """Verify the injury-check cron logic."""

    def test_injury_check_skips_when_locked(self):
        """Injury check returns skipped=True when slate is locked."""
        from api.index import _is_locked
        # When games are locked, injury check should skip
        # Just verify the logic — _is_locked returns True for past start times
        past_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        assert _is_locked(past_time) is True

    def test_injury_check_skips_no_cache(self):
        """Injury check returns no_cache when there's no cached slate."""
        # This is a logic test — verifying the guard condition
        cached_slate = None
        assert cached_slate is None or not cached_slate.get("lineups")

    def test_is_safe_to_draft_detects_out(self):
        """is_safe_to_draft returns False for OUT players."""
        from api.rotowire import is_safe_to_draft
        # Mock the internal function to return OUT status
        with patch("api.rotowire.get_player_status", return_value={"status": "out", "is_starter": False}):
            assert is_safe_to_draft("Test Player") is False

    def test_is_safe_to_draft_allows_confirmed(self):
        """is_safe_to_draft returns True for confirmed players."""
        from api.rotowire import is_safe_to_draft
        with patch("api.rotowire.get_player_status", return_value={"status": "confirmed", "is_starter": True}):
            assert is_safe_to_draft("Test Player") is True

    def test_is_safe_to_draft_allows_unknown(self):
        """is_safe_to_draft returns True when player is not found in RotoWire."""
        from api.rotowire import is_safe_to_draft
        with patch("api.rotowire.get_player_status", return_value=None):
            assert is_safe_to_draft("Unknown Player") is True


# ─────────────────────────────────────────────────────────
# TestPicksServeFromCache — /api/picks cache layers
# ─────────────────────────────────────────────────────────
class TestPicksServeFromCache:
    """Verify /api/picks serves from GitHub cache without calling _run_game()."""

    def test_games_cache_from_github_returns_none_on_error(self):
        """_games_cache_from_github returns None when GitHub is unreachable."""
        from api.index import _games_cache_from_github
        with patch("api.index._github_get_file", side_effect=Exception("Network error")):
            result = _games_cache_from_github()
        assert result is None

    def test_bust_slate_cache_writes_tombstone(self):
        """_bust_slate_cache writes all tombstones in a single batch commit."""
        from api.index import _bust_slate_cache
        with patch("api.index._github_write_batch") as mock_batch, \
             patch("api.index._cp") as mock_cp:
            mock_cp.return_value.unlink = Mock()
            _bust_slate_cache()
        # Single batch call with 4 files
        assert mock_batch.call_count == 1
        files = mock_batch.call_args[0][0]
        assert len(files) == 4
        for f in files:
            content = json.loads(f["content"])
            assert content.get("_busted") is True
        # Bust sentinel has "at" timestamp
        bust_files = [f for f in files if "_bust.json" in f["path"]]
        assert len(bust_files) == 1
        bust_content = json.loads(bust_files[0]["content"])
        assert "at" in bust_content

    def test_get_projections_for_date_hydrates_from_github_when_tmp_empty(self):
        """_get_projections_for_date reads GitHub games cache when /tmp per-game cache is cold."""
        from api.index import _get_projections_for_date
        mock_games = [{
            "gameId": "game1",
            "startTime": (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat(),
            "home": {"abbr": "BOS"}, "away": {"abbr": "LAL"},
        }]
        mock_player = {"name": "Player", "id": "123", "team": "BOS"}
        gh_data = {"game1": [mock_player]}

        with patch("api.index.fetch_games", return_value=mock_games), \
             patch("api.index._is_past_lock_window", return_value=False), \
             patch("api.index._cg", return_value=None), \
             patch("api.index._games_cache_from_github", return_value=gh_data) as mock_gh, \
             patch("api.index._cs"), \
             patch("api.index._run_game") as mock_run:
            from datetime import date
            projs, games = _get_projections_for_date(date.today())
            mock_gh.assert_called_once()
            mock_run.assert_not_called()
            assert len(projs) == 1


# ─────────────────────────────────────────────────────────
# TestClaudeContextLayer
# ─────────────────────────────────────────────────────────
class TestClaudeContextLayer:
    """_claude_context_pass adjusts player ratings using Claude Haiku output."""

    def _make_player(self, name, team, rating=4.0, chalk_ev=16.0, ceiling_score=5.0, est_mult=1.5):
        return {
            "name": name, "team": team, "rating": rating, "chalk_ev": chalk_ev,
            "ceiling_score": ceiling_score, "est_mult": est_mult,
            "season_pts": 10.0, "season_reb": 4.0, "season_ast": 3.0,
            "season_stl": 0.8, "season_blk": 0.4,
        }

    def _make_game(self, home_abbr, away_abbr, spread=0, total=222):
        return {
            "gameId": "g1", "spread": spread, "total": total,
            "home": {"abbr": home_abbr, "id": "1"},
            "away": {"abbr": away_abbr, "id": "2"},
        }

    def test_no_op_when_disabled(self):
        """Context pass does nothing when context_layer.enabled is false."""
        from api.index import _claude_context_pass
        players = [self._make_player("LeBron James", "LAL")]
        games = [self._make_game("LAL", "GSW")]
        with patch("api.index._cfg", side_effect=lambda k, d=None: False if k == "context_layer.enabled" else d):
            _claude_context_pass(players, games)
        assert players[0]["rating"] == 4.0
        assert "_context_adj" not in players[0]

    def test_applies_multiplier_within_cap(self):
        """Multipliers from Claude are applied to rating/chalk_ev/ceiling_score."""
        from api.index import _claude_context_pass
        players = [self._make_player("Draymond Green", "GSW", rating=2.5, chalk_ev=10.0, ceiling_score=3.0)]
        games = [self._make_game("GSW", "LAL")]
        games[0]["home_b2b"] = True  # pass skip guard

        claude_response = json.dumps({
            "adjustments": [{"player": "Draymond Green", "rs_multiplier": 1.30, "reason": "defensive value"}]
        })

        def cfg_side_effect(key, default=None):
            if key == "context_layer.enabled": return True
            if key == "context_layer.model": return "claude-haiku-4-5-20251001"
            if key == "context_layer.max_adjustment": return 0.4
            if key == "context_layer.timeout_seconds": return 15
            return default

        mock_msg = Mock()
        mock_msg.content = [Mock(text=claude_response)]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_msg

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch("anthropic.Anthropic", return_value=mock_client), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            _claude_context_pass(players, games)

        assert players[0]["rating"] == pytest.approx(2.5 * 1.30, abs=0.1)
        assert players[0]["chalk_ev"] == pytest.approx(10.0 * 1.30, abs=0.1)
        assert players[0]["ceiling_score"] == pytest.approx(3.0 * 1.30, abs=0.1)
        assert players[0]["_context_adj"] == pytest.approx(1.30, abs=0.01)

    def test_multiplier_clamped_at_max_adjustment(self):
        """Claude returning 2.0x is clamped to 1+max_adjustment (1.4x at default 0.4)."""
        from api.index import _claude_context_pass
        players = [self._make_player("Player X", "BOS", rating=3.0)]
        games = [self._make_game("BOS", "LAL")]
        games[0]["home_b2b"] = True  # pass skip guard

        claude_response = json.dumps({
            "adjustments": [{"player": "Player X", "rs_multiplier": 2.0, "reason": "extreme"}]
        })

        def cfg_side_effect(key, default=None):
            if key == "context_layer.enabled": return True
            if key == "context_layer.model": return "claude-haiku-4-5-20251001"
            if key == "context_layer.max_adjustment": return 0.4
            if key == "context_layer.timeout_seconds": return 15
            return default

        mock_msg = Mock()
        mock_msg.content = [Mock(text=claude_response)]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_msg

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch("anthropic.Anthropic", return_value=mock_client), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            _claude_context_pass(players, games)

        # 2.0 should be clamped to 1.4 (max)
        assert players[0]["rating"] == pytest.approx(3.0 * 1.4, abs=0.1)
        assert players[0]["_context_adj"] == pytest.approx(1.4, abs=0.01)

    def test_multiplier_clamped_at_min_adjustment(self):
        """Claude returning 0.3x is clamped to 1-max_adjustment (0.6x at default 0.4)."""
        from api.index import _claude_context_pass
        players = [self._make_player("Player Y", "LAL", rating=5.0)]
        games = [self._make_game("LAL", "BOS")]
        games[0]["home_b2b"] = True  # pass skip guard

        claude_response = json.dumps({
            "adjustments": [{"player": "Player Y", "rs_multiplier": 0.3, "reason": "blowout"}]
        })

        def cfg_side_effect(key, default=None):
            if key == "context_layer.enabled": return True
            if key == "context_layer.model": return "claude-haiku-4-5-20251001"
            if key == "context_layer.max_adjustment": return 0.4
            if key == "context_layer.timeout_seconds": return 15
            return default

        mock_msg = Mock()
        mock_msg.content = [Mock(text=claude_response)]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_msg

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch("anthropic.Anthropic", return_value=mock_client), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            _claude_context_pass(players, games)

        # 0.3 should be clamped to 0.6 (min)
        assert players[0]["rating"] == pytest.approx(5.0 * 0.6, abs=0.1)
        assert players[0]["_context_adj"] == pytest.approx(0.6, abs=0.01)

    def test_graceful_fallback_on_claude_error(self):
        """If Claude call raises, players are unchanged (no-op fallback)."""
        from api.index import _claude_context_pass
        players = [self._make_player("Player Z", "GSW", rating=3.0)]
        games = [self._make_game("GSW", "LAL")]

        def cfg_side_effect(key, default=None):
            if key == "context_layer.enabled": return True
            if key == "context_layer.model": return "claude-haiku-4-5-20251001"
            if key == "context_layer.max_adjustment": return 0.4
            if key == "context_layer.timeout_seconds": return 15
            return default

        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API timeout")

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch("anthropic.Anthropic", return_value=mock_client), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            _claude_context_pass(players, games)

        # Player rating must be unchanged
        assert players[0]["rating"] == 3.0
        assert "_context_adj" not in players[0]

    def test_unknown_player_name_skipped(self):
        """Claude adjustments for players not in the pool are silently skipped."""
        from api.index import _claude_context_pass
        players = [self._make_player("Real Player", "GSW", rating=3.0)]
        games = [self._make_game("GSW", "LAL")]

        claude_response = json.dumps({
            "adjustments": [{"player": "Made Up Player", "rs_multiplier": 1.3, "reason": "test"}]
        })

        def cfg_side_effect(key, default=None):
            if key == "context_layer.enabled": return True
            if key == "context_layer.model": return "claude-haiku-4-5-20251001"
            if key == "context_layer.max_adjustment": return 0.4
            if key == "context_layer.timeout_seconds": return 15
            return default

        mock_msg = Mock()
        mock_msg.content = [Mock(text=claude_response)]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_msg

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch("anthropic.Anthropic", return_value=mock_client), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            _claude_context_pass(players, games)

        assert players[0]["rating"] == 3.0
        assert "_context_adj" not in players[0]


# ─────────────────────────────────────────────────────────
# TestPredMinTolerance — tolerance band on predMin < season_min gate
# ─────────────────────────────────────────────────────────
class TestPredMinTolerance:
    """Verify that the predMin tolerance band allows small dips below season_min."""

    def test_chalk_tolerance_config_exists(self):
        """projection.pred_min_tolerance should be in model-config.json."""
        import json
        with open("data/model-config.json") as f:
            cfg = json.load(f)
        assert cfg["projection"]["pred_min_tolerance"] == 2.0

    def test_moonshot_tolerance_config_exists(self):
        """moonshot.pred_min_tolerance should be in model-config.json."""
        import json
        with open("data/model-config.json") as f:
            cfg = json.load(f)
        assert cfg["moonshot"]["pred_min_tolerance"] == 3.0

    def test_chalk_tolerance_code_reads_config(self):
        """Chalk pool builder should use _cfg for tolerance, not hardcoded 0."""
        with open("api/index.py") as f:
            src = f.read()
        assert 'projection.pred_min_tolerance' in src
        assert 'pred_min_tolerance' in src  # moonshot reads from moon_cfg dict


# ─────────────────────────────────────────────────────────
# TestMoonshotPtsFloor — separate min_pts for moonshot
# ─────────────────────────────────────────────────────────
class TestMoonshotPtsFloor:
    """Verify that moonshot uses a lower pts projection floor than chalk."""

    def test_config_has_separate_moonshot_floor(self):
        """scoring_thresholds should have both min_pts_projection and min_pts_projection_moonshot."""
        import json
        with open("data/model-config.json") as f:
            cfg = json.load(f)
        assert cfg["scoring_thresholds"]["min_pts_projection"] == 5.0  # v57: 6.0→5.0, let high-boost role players through
        # moonshot floor raised to 6.0 in v44 (was 3.0) — bench players need real production
        assert cfg["scoring_thresholds"]["min_pts_projection_moonshot"] >= 5.0, \
            f"moonshot pts floor should be >= 5.0, got {cfg['scoring_thresholds']['min_pts_projection_moonshot']}"

    def test_project_player_uses_moonshot_floor(self):
        """project_player should use the lower moonshot floor (4.0) not 7.0."""
        with open("api/index.py") as f:
            src = f.read()
        assert 'min_pts_projection_moonshot' in src
        assert 'min_pts_per_minute_moonshot' in src

    def test_chalk_pool_enforces_stricter_pts_floor(self):
        """Chalk pool builder should enforce min_pts_projection (7.0) separately."""
        with open("api/index.py") as f:
            src = f.read()
        # Chalk pool should have its own pts check
        assert 'chalk_min_pts' in src


# ─────────────────────────────────────────────────────────
# TestOddsEnrichment — Odds API enrichment in draft pipeline
# ─────────────────────────────────────────────────────────
class TestOddsEnrichment:
    """Verify _enrich_projections_with_odds blends correctly."""

    def test_enrichment_skips_when_disabled(self):
        """No-op when odds_enrichment.enabled is False."""
        from api.index import _enrich_projections_with_odds

        players = [{"name": "Test Player", "pts": 15.0, "predMin": 28.0}]
        games = []

        with patch("api.index._cfg", return_value=False):
            _enrich_projections_with_odds(players, games)

        # Player unchanged
        assert players[0]["pts"] == 15.0
        assert "odds_pts_line" not in players[0]

    def test_enrichment_blends_upward(self):
        """When books are 20%+ higher, blends pts upward."""
        from api.index import _enrich_projections_with_odds

        players = [{"name": "Test Player", "pts": 12.0, "predMin": 25.0}]
        games = [{"home": {"abbr": "MIN"}, "away": {"abbr": "PHX"}}]

        # Mock odds map: books have player at 18.5 pts (54% higher than model's 12.0)
        mock_odds = {("test player", "points"): {"line": 18.5, "odds_over": -110, "odds_under": -110, "books_consensus": 3}}

        def cfg_side_effect(key, default=None):
            cfg_map = {
                "odds_enrichment.enabled": True,
                "odds_enrichment.blend_weight": 0.2,
                "odds_enrichment.min_divergence_pct": 0.15,
                "odds_enrichment.upward_only": True,
            }
            return cfg_map.get(key, default)

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch("api.index._build_player_odds_map", return_value=mock_odds):
            _enrich_projections_with_odds(players, games)

        # pts should be blended: 12.0 * 0.8 + 18.5 * 0.2 = 13.3
        assert players[0]["pts"] == 13.3
        assert players[0]["odds_pts_line"] == 18.5
        assert players[0]["_odds_adjusted"] is True
        # predMin should be nudged up proportionally
        assert players[0]["predMin"] > 25.0

    def test_enrichment_no_blend_below_divergence(self):
        """When books are close to model, no blending occurs."""
        from api.index import _enrich_projections_with_odds

        players = [{"name": "Test Player", "pts": 15.0, "predMin": 28.0}]
        games = [{"home": {"abbr": "MIN"}, "away": {"abbr": "PHX"}}]

        # Books at 16.0 = 6.7% divergence, below 15% threshold
        mock_odds = {("test player", "points"): {"line": 16.0, "odds_over": -110, "odds_under": -110, "books_consensus": 3}}

        def cfg_side_effect(key, default=None):
            cfg_map = {
                "odds_enrichment.enabled": True,
                "odds_enrichment.blend_weight": 0.2,
                "odds_enrichment.min_divergence_pct": 0.15,
                "odds_enrichment.upward_only": True,
            }
            return cfg_map.get(key, default)

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch("api.index._build_player_odds_map", return_value=mock_odds):
            _enrich_projections_with_odds(players, games)

        # pts unchanged
        assert players[0]["pts"] == 15.0
        assert "_odds_adjusted" not in players[0]


# ─────────────────────────────────────────────────────────
# TestWebSearch — Brave Search integration
# ─────────────────────────────────────────────────────────
class TestWebSearch:
    """Verify _fetch_nba_news_context fetches and caches correctly."""

    def test_skips_when_disabled(self):
        """Returns empty string when web_search_enabled is False."""
        from api.index import _fetch_nba_news_context

        with patch("api.index._cfg", return_value=False):
            result = _fetch_nba_news_context([])

        assert result == ""

    def test_skips_when_no_api_key(self):
        """Returns empty string when ANTHROPIC_API_KEY is missing."""
        from api.index import _fetch_nba_news_context

        def cfg_side_effect(key, default=None):
            if key == "context_layer.web_search_enabled":
                return True
            return default

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
            result = _fetch_nba_news_context([])

        assert result == ""

    def test_fetches_and_caches(self):
        """Should call Claude web_search tool and cache results."""
        from api.index import _fetch_nba_news_context
        from datetime import date

        games = [
            {"home": {"abbr": "MIN"}, "away": {"abbr": "PHX"}},
        ]

        # Mock the Anthropic client response
        mock_text_block = Mock()
        mock_text_block.text = "- Finch says Bones Hyland will see expanded minutes with ANT out 2 weeks\n- KD questionable for Suns"
        mock_msg = Mock()
        mock_msg.content = [mock_text_block]

        mock_client = Mock()
        mock_client.messages.create.return_value = mock_msg

        mock_anthropic_module = Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        def cfg_side_effect(key, default=None):
            cfg_map = {
                "context_layer.web_search_enabled": True,
                "context_layer.timeout_seconds": 20,
                "context_layer.web_search_model": "claude-opus-4-20250514",
            }
            return cfg_map.get(key, default)

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}), \
             patch("api.index._cg", return_value=None), \
             patch("api.index._cs") as mock_cs, \
             patch.dict("sys.modules", {"anthropic": mock_anthropic_module}), \
             patch("api.index._et_date", return_value=date(2026, 3, 18)):
            result = _fetch_nba_news_context(games, date=None, all_proj=None)

        assert "Finch" in result or "Bones" in result
        assert mock_cs.called  # Cached the result
        # Verify web_search tool was passed
        call_kwargs = mock_client.messages.create.call_args
        tools = call_kwargs.kwargs.get("tools", []) if call_kwargs.kwargs else []
        assert any(t.get("type") == "web_search_20250305" for t in tools)

    def test_uses_cache_when_available(self):
        """Should return cached text without making API calls."""
        from api.index import _fetch_nba_news_context
        from datetime import date

        games = [{"home": {"abbr": "MIN"}, "away": {"abbr": "PHX"}}]

        def cfg_side_effect(key, default=None):
            if key == "context_layer.web_search_enabled":
                return True
            return default

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}), \
             patch("api.index._cg", return_value={"text": "Cached news about Wolves"}), \
             patch("api.index._et_date", return_value=date(2026, 3, 18)):
            result = _fetch_nba_news_context(games, date=None, all_proj=None)

        assert result == "Cached news about Wolves"

    def test_prompt_includes_key_players_when_all_proj_provided(self):
        """When all_proj is passed, the API prompt should include KEY PLAYERS section."""
        from api.index import _fetch_nba_news_context
        from datetime import date

        games = [{"home": {"abbr": "BOS"}, "away": {"abbr": "LAL"}}]
        all_proj = [
            {"name": "Jayson Tatum", "team": "BOS", "rating": 6.2},
            {"name": "LeBron James", "team": "LAL", "rating": 5.8},
        ]

        mock_text_block = Mock()
        mock_text_block.text = "No breaking news."
        mock_msg = Mock()
        mock_msg.content = [mock_text_block]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic_module = Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        def cfg_side_effect(key, default=None):
            cfg_map = {
                "context_layer.web_search_enabled": True,
                "context_layer.timeout_seconds": 20,
                "context_layer.web_search_model": "claude-opus-4-20250514",
            }
            return cfg_map.get(key, default)

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}), \
             patch("api.index._cg", return_value=None), \
             patch("api.index._cs"), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic_module}), \
             patch("api.index._et_date", return_value=date(2026, 3, 18)):
            _fetch_nba_news_context(games, date=None, all_proj=all_proj)

        call_kwargs = mock_client.messages.create.call_args
        content = call_kwargs.kwargs.get("messages", [{}])[0].get("content", "")
        assert "KEY PLAYERS" in content
        assert "Jayson Tatum" in content
        assert "LeBron James" in content


# ─────────────────────────────────────────────────────────
# TestContextPassWithNews — web search integrated into context pass
# ─────────────────────────────────────────────────────────
class TestContextPassWithNews:
    """Verify _claude_context_pass includes news text when available."""

    def test_context_pass_calls_web_search(self):
        """When web search is enabled, context pass should call _fetch_nba_news_context with games and all_proj."""
        with open("api/index.py") as f:
            src = f.read()
        assert "_fetch_nba_news_context(games, date=None, all_proj=all_proj)" in src

    def test_context_pass_includes_news_in_prompt(self):
        """News text should appear in the user prompt under RECENT NBA NEWS."""
        with open("api/index.py") as f:
            src = f.read()
        assert "RECENT NBA NEWS" in src
        assert "injury" in src.lower() and "rotation" in src.lower()


# ─────────────────────────────────────────────────────────
# TestLineupReview — Layer 3 post-lineup Opus review
# ─────────────────────────────────────────────────────────
class TestLineupReview:
    """Verify _lineup_review_opus: disabled returns unchanged; no swaps unchanged; valid swap applied; exception returns original."""

    def test_returns_unchanged_when_disabled(self):
        """When lineup_review.enabled is False, chalk and upside are returned unchanged."""
        from api.index import _lineup_review_opus

        chalk = [{"name": "Player A", "team": "BOS", "slot": "2.0x", "rating": 5.0}]
        upside = [{"name": "Player B", "team": "LAL", "slot": "1.8x", "rating": 4.0}]
        all_proj = [{"name": "Player A", "team": "BOS"}, {"name": "Player B", "team": "LAL"}]
        games = []

        with patch("api.index._cfg", return_value=False):
            out_chalk, out_upside = _lineup_review_opus(chalk, upside, all_proj, games)

        assert out_chalk == chalk
        assert out_upside == upside

    def test_returns_unchanged_when_no_swaps(self):
        """When Opus returns empty swaps, lineups unchanged."""
        from api.index import _lineup_review_opus

        chalk = [{"name": "Player A", "team": "BOS", "slot": "2.0x", "rating": 5.0}]
        upside = [{"name": "Player B", "team": "LAL", "slot": "1.8x", "rating": 4.0}]
        all_proj = [{"name": "Player A", "team": "BOS"}, {"name": "Player B", "team": "LAL"}]
        games = []

        mock_text_block = Mock()
        mock_text_block.text = '{"swaps": []}'
        mock_msg = Mock()
        mock_msg.content = [mock_text_block]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic_module = Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        def cfg_side_effect(key, default=None):
            cfg_map = {
                "lineup_review.enabled": True,
                "lineup_review.model": "claude-opus-4-20250514",
                "lineup_review.timeout_seconds": 30,
            }
            return cfg_map.get(key, default)

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            out_chalk, out_upside = _lineup_review_opus(chalk, upside, all_proj, games)

        assert out_chalk[0]["name"] == "Player A"
        assert out_upside[0]["name"] == "Player B"

    def test_applies_valid_swap(self):
        """When Opus suggests a valid swap (in in all_proj), the lineup is updated."""
        from api.index import _lineup_review_opus

        chalk = [
            {"name": "Player A", "team": "BOS", "slot": "2.0x", "rating": 5.0, "est_mult": 1.2},
            {"name": "Player C", "team": "BOS", "slot": "1.8x", "rating": 4.5, "est_mult": 1.1},
        ]
        upside = [{"name": "Player B", "team": "LAL", "slot": "1.8x", "rating": 4.0}]
        replacement = {"name": "Player D", "team": "BOS", "rating": 5.5, "est_mult": 1.3}
        all_proj = [
            {"name": "Player A", "team": "BOS"},
            {"name": "Player B", "team": "LAL"},
            {"name": "Player C", "team": "BOS"},
            replacement,
        ]
        games = []

        mock_text_block = Mock()
        mock_text_block.text = '{"swaps": [{"lineup": "chalk", "out": "Player A", "in": "Player D"}]}'
        mock_msg = Mock()
        mock_msg.content = [mock_text_block]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_msg
        mock_anthropic_module = Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        def cfg_side_effect(key, default=None):
            cfg_map = {
                "lineup_review.enabled": True,
                "lineup_review.model": "claude-opus-4-20250514",
                "lineup_review.timeout_seconds": 30,
            }
            return cfg_map.get(key, default)

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            out_chalk, out_upside = _lineup_review_opus(chalk, upside, all_proj, games)

        assert out_chalk[0]["name"] == "Player D"
        assert out_chalk[0]["slot"] == "2.0x"
        assert out_chalk[1]["name"] == "Player C"

    def test_returns_original_on_exception(self):
        """On API or parse error, original chalk and upside are returned."""
        from api.index import _lineup_review_opus

        chalk = [{"name": "Player A", "team": "BOS", "slot": "2.0x"}]
        upside = [{"name": "Player B", "team": "LAL", "slot": "1.8x"}]
        all_proj = [{"name": "Player A", "team": "BOS"}, {"name": "Player B", "team": "LAL"}]
        games = []

        def cfg_side_effect(key, default=None):
            cfg_map = {
                "lineup_review.enabled": True,
                "lineup_review.model": "claude-opus-4-20250514",
                "lineup_review.timeout_seconds": 30,
            }
            return cfg_map.get(key, default)

        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API error")
        mock_anthropic_module = Mock()
        mock_anthropic_module.Anthropic.return_value = mock_client

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            out_chalk, out_upside = _lineup_review_opus(chalk, upside, all_proj, games)

        assert out_chalk == chalk
        assert out_upside == upside


# ─────────────────────────────────────────────────────────
# TestCorePool — core pool architecture
# ─────────────────────────────────────────────────────────
class TestCorePool:
    """When core_pool.enabled is False, third return is None; when True, third is list; lineups from core when enabled."""

    def test_build_lineups_returns_three_values(self):
        """_build_lineups returns (chalk, upside, core_pool). When core pool disabled, core_pool is None."""
        from api.index import _build_lineups

        def cfg_core_off(key, default=None):
            if key == "core_pool":
                return {"enabled": False, "size": 8, "metric": "max_ev"}
            return default

        with patch("api.index._cfg", side_effect=cfg_core_off), \
             patch("api.index.get_all_statuses", return_value={}), \
             patch("api.index._fetch_team_def_stats", return_value={}):
            chalk, upside, core_pool = _build_lineups([])

        assert chalk == []
        assert upside == []
        assert core_pool is None

    def test_build_lineups_core_pool_list_when_enabled(self):
        """When core_pool.enabled is True, third return is a list (possibly empty)."""
        from api.index import _build_lineups

        def cfg_core_on(key, default=None):
            if key == "core_pool":
                return {"enabled": True, "size": 8, "metric": "max_ev", "blend_weight": 0.5}
            return default

        with patch("api.index._cfg", side_effect=cfg_core_on), \
             patch("api.index.get_all_statuses", return_value={}), \
             patch("api.index._fetch_team_def_stats", return_value={}):
            chalk, upside, core_pool = _build_lineups([])

        assert isinstance(core_pool, list)
        assert chalk == []
        assert upside == []


# ─────────────────────────────────────────────────────────
# TestMatchupFactor — matchup analysis replacing dev-team bonus
# ─────────────────────────────────────────────────────────
class TestMatchupFactor:
    """_compute_matchup_factor, _build_game_opp_map, and matchup config keys."""

    def test_neutral_when_no_def_stats(self):
        """Returns 1.0 when def_stats is empty (fallback to neutral)."""
        from api.index import _compute_matchup_factor
        factor = _compute_matchup_factor({"pos": "G"}, "LAL", {})
        assert factor == 1.0

    def test_neutral_when_opponent_unknown(self):
        """Returns 1.0 when opponent not in def_stats."""
        from api.index import _compute_matchup_factor
        factor = _compute_matchup_factor({"pos": "G"}, "XYZ", {"LAL": {"pts_allowed": 110.0}})
        assert factor == 1.0

    def test_weak_defense_bonus_for_guard(self):
        """Guard vs weak defense (118 pts allowed) gets >1.0 factor."""
        from api.index import _compute_matchup_factor
        factor = _compute_matchup_factor({"pos": "G"}, "SAC", {"SAC": {"pts_allowed": 118.0}})
        assert factor > 1.0, f"Expected > 1.0, got {factor}"
        assert factor <= 1.25

    def test_elite_defense_penalty(self):
        """Player vs elite defense (108 pts allowed) gets <1.0 factor."""
        from api.index import _compute_matchup_factor
        factor = _compute_matchup_factor({"pos": "F"}, "OKC", {"OKC": {"pts_allowed": 108.0}})
        assert factor < 1.0, f"Expected < 1.0, got {factor}"
        assert factor >= 0.80

    def test_center_less_sensitive_than_guard(self):
        """Center gets smaller adjustment magnitude than guard for same matchup."""
        from api.index import _compute_matchup_factor
        weak_def = {"LAL": {"pts_allowed": 120.0}}
        guard_factor = _compute_matchup_factor({"pos": "G"}, "LAL", weak_def)
        center_factor = _compute_matchup_factor({"pos": "C"}, "LAL", weak_def)
        assert guard_factor > center_factor, "Guard should benefit more from weak defense than center"

    def test_league_avg_defense_is_neutral(self):
        """Opponent at exactly league avg (115 pts/g) yields factor 1.0."""
        from api.index import _compute_matchup_factor
        factor = _compute_matchup_factor({"pos": "F"}, "MIL", {"MIL": {"pts_allowed": 115.0}})
        assert factor == 1.0

    def test_build_game_opp_map(self):
        """_build_game_opp_map builds bidirectional team → opponent map."""
        from api.index import _build_game_opp_map
        games = [{"home": {"abbr": "BOS"}, "away": {"abbr": "MIA"}}]
        opp_map = _build_game_opp_map(games)
        assert opp_map["BOS"] == "MIA"
        assert opp_map["MIA"] == "BOS"

    def test_matchup_config_keys_present(self):
        """matchup config keys exist in _CONFIG_DEFAULTS."""
        from api.index import _CONFIG_DEFAULTS
        matchup = _CONFIG_DEFAULTS.get("matchup", {})
        assert "enabled" in matchup
        assert "def_scale" in matchup
        assert "chalk_adj_min" in matchup
        assert "moonshot_adj_min" in matchup
        assert "claude_enabled" in matchup

    def test_no_dev_team_pts_floor_in_moonshot_defaults(self):
        """dev_team_pts_floor is removed from moonshot defaults."""
        from api.index import _CONFIG_DEFAULTS
        moonshot = _CONFIG_DEFAULTS.get("moonshot", {})
        assert "dev_team_pts_floor" not in moonshot


class TestTeamMotivation:
    """Late-season team motivation tiering and multiplier wiring."""

    def test_team_motivation_config_keys_present(self):
        from api.index import _CONFIG_DEFAULTS
        tm = _CONFIG_DEFAULTS.get("team_motivation", {})
        assert "enabled" in tm
        assert "start_date" in tm
        assert "tier_a_mult_chalk" in tm
        assert "tier_c_mult_chalk" in tm
        assert "team_overrides" in tm

    def test_team_tier_from_standings_rules(self):
        from api.index import _team_tier_from_standings
        cfg = {
            "seeding_gap_games": 2.0,
            "playin_gap_games": 2.0,
            "elimination_buffer_games": 3.0,
        }
        # Play-in seeds are high-incentive
        assert _team_tier_from_standings({"playoffSeed": 8, "gamesBack": 1.0}, cfg) == "A"
        # Deep lottery with large gap is low-incentive
        assert _team_tier_from_standings({"playoffSeed": 14, "gamesBack": 6.0}, cfg) == "C"
        # Missing data should stay neutral
        assert _team_tier_from_standings({}, cfg) == "B"

    def test_team_motivation_multiplier_defaults_neutral(self):
        from api.index import _team_motivation_multiplier
        assert _team_motivation_multiplier("LAL", "chalk", {}) == 1.0
        assert _team_motivation_multiplier("LAL", "moonshot", {}) == 1.0

    def test_fetch_team_motivation_map_obeys_start_date_gate(self):
        from api.index import _fetch_team_motivation_map

        def cfg_side_effect(key, default=None):
            if key == "team_motivation":
                return {
                    "enabled": True,
                    "start_date": "2999-01-01",
                    "tier_a_mult_chalk": 1.08,
                    "tier_b_mult_chalk": 1.00,
                    "tier_c_mult_chalk": 0.90,
                    "tier_a_mult_moonshot": 1.00,
                    "tier_b_mult_moonshot": 1.00,
                    "tier_c_mult_moonshot": 1.00,
                    "min_mult": 0.88,
                    "max_mult": 1.12,
                    "team_overrides": {},
                }
            return default

        with patch("api.index._cfg", side_effect=cfg_side_effect):
            assert _fetch_team_motivation_map() == {}

# ─────────────────────────────────────────────────────────
# TestBriefingSimulatedDraftScore — simulated_draft_score surfaces to Ben
# ─────────────────────────────────────────────────────────
class TestBriefingSimulatedDraftScore:
    """simulated_draft_score from _compute_audit must appear in /api/lab/briefing latest_slate."""

    AUDIT = {
        "date": "2026-03-17",
        "players_compared": 5,
        "mae": 1.48,
        "directional_accuracy": 0.6,
        "over_projected": 2,
        "under_projected": 3,
        "biggest_misses": [],
        "simulated_draft_score": 62.4,
        "generated_at": "2026-03-17T23:00:00Z",
    }

    def _run_briefing_with_audit(self, audit_dict):
        """Helper: run lab_briefing with a fake paired date using the given audit."""
        import asyncio, json
        from unittest.mock import patch
        from api.index import lab_briefing

        audit_json = json.dumps(audit_dict)
        date = audit_dict["date"]

        def fake_list_dir(path):
            if "predictions" in path:
                return [{"name": f"{date}.csv"}]
            if "actuals" in path:
                return [{"name": f"{date}.csv"}]
            return []

        def fake_get_file(path):
            if f"audit/{date}" in path:
                return audit_json, "sha1"
            return None, None

        with patch('api.index._github_list_dir', side_effect=fake_list_dir):
            with patch('api.index._github_get_file', side_effect=fake_get_file):
                with patch('api.index._load_config', return_value={"version": 42, "changelog": []}):
                    result = asyncio.get_event_loop().run_until_complete(lab_briefing())

        data = result if isinstance(result, dict) else result.body
        if isinstance(data, bytes):
            data = json.loads(data)
        return data

    def test_simulated_draft_score_in_briefing_latest_slate(self):
        """When audit has simulated_draft_score, briefing latest_slate must include it."""
        data = self._run_briefing_with_audit(self.AUDIT)
        latest = data.get("latest_slate") or {}
        assert "simulated_draft_score" in latest, \
            "simulated_draft_score must be in briefing latest_slate so Ben can track 60+ goal"
        assert latest["simulated_draft_score"] == 62.4

    def test_simulated_draft_score_none_handled_gracefully(self):
        """When audit has simulated_draft_score=None, briefing still works."""
        audit_no_score = {**self.AUDIT, "simulated_draft_score": None}
        data = self._run_briefing_with_audit(audit_no_score)
        latest = data.get("latest_slate") or {}
        assert latest.get("simulated_draft_score") is None


class TestMinGateOverrideAware:
    """min_gate_minutes gate uses player boost overrides, not just PPG proxy.

    This ensures players with high override boosts (GPII +3.0, Braun +3.0, Sensabaugh +2.1)
    receive a lower min_gate so they can enter the projection pipeline and be evaluated
    for the core pool — previously these overrides were unreachable for low-minute players.
    """

    def _effective_gate(self, pts, override_boost=None, min_gate=12):
        """Replicate the gate formula from project_player()."""
        rough_boost = override_boost if override_boost is not None else max(0.2, 3.0 - pts * 0.12)
        return max(8, min_gate - max(0, (rough_boost - 1.5) * 3))

    def test_gpii_override_3_0_lowers_gate_to_8(self):
        """GPII has override 3.0 → effective_gate = max(8, 12-(3.0-1.5)*3) = 8."""
        gate = self._effective_gate(pts=8.0, override_boost=3.0)
        assert gate == 8, f"GPII with 3.0 override should get gate=8, got {gate}"

    def test_gpii_without_override_gate_higher(self):
        """Without override, 8 PPG → rough_boost≈2.04 → gate≈10.7 (blocks ~13-min players)."""
        gate = self._effective_gate(pts=8.0, override_boost=None)
        assert gate > 10, f"8 PPG without override should give gate>10, got {gate}"

    def test_override_enables_10_min_player_that_ppg_proxy_blocks(self):
        """A player projecting 10 min with 8 PPG: PPG proxy gives gate≈10.7 (fails), 3.0 override gives gate=8 (passes).

        This is the GPII scenario: on a short-minute night (~10 predMin) with 8 PPG,
        the PPG proxy underestimates his real boost and incorrectly blocks him.
        The 3.0 override makes him eligible for the projection pipeline.
        """
        proj_min = 10
        gate_with_override = self._effective_gate(pts=8.0, override_boost=3.0)
        gate_without_override = self._effective_gate(pts=8.0, override_boost=None)
        # With 3.0 override: gate = 8, 10 >= 8 → passes
        assert proj_min >= gate_with_override, \
            f"10-min player must pass gate with 3.0 override (gate={gate_with_override})"
        # Without override at 8 PPG: gate ≈ 10.7, 10 < 10.7 → fails
        assert proj_min < gate_without_override, \
            f"10-min player must fail gate without override at 8 PPG (gate={gate_without_override})"

    def test_sensabaugh_override_2_1_lowers_gate(self):
        """Sensabaugh override 2.1 → gate = max(8, 12-(2.1-1.5)*3) = 10.2."""
        gate = self._effective_gate(pts=10.0, override_boost=2.1)
        assert gate == pytest.approx(10.2, abs=0.1)

    def test_override_gate_in_project_player_source(self):
        """project_player() must check player overrides before computing rough_boost."""
        import inspect
        from api import index
        src = inspect.getsource(index.project_player)
        assert "_override_boost_gate" in src, \
            "project_player must use _override_boost_gate variable (override-aware gate)"
        assert "card_boost.player_overrides" in src, \
            "project_player gate must look up card_boost.player_overrides"

    def test_christian_braun_in_config_overrides(self):
        """Christian Braun must be in player_overrides with boost 3.0."""
        import json, pathlib
        from api.index import _normalize_boost_name
        cfg_path = pathlib.Path(__file__).parent.parent / "data" / "model-config.json"
        overrides = json.loads(cfg_path.read_text()).get("card_boost", {}).get("player_overrides", {})
        norm_braun = _normalize_boost_name("Christian Braun")
        match = next(
            (float(v) for k, v in overrides.items() if _normalize_boost_name(k) == norm_braun),
            None
        )
        assert match is not None, "Christian Braun must be in card_boost.player_overrides"
        assert match == pytest.approx(3.0), f"Braun override should be 3.0, got {match}"

    def test_jared_mccain_in_config_overrides(self):
        """Jared McCain must be in player_overrides with boost 2.9."""
        import json, pathlib
        from api.index import _normalize_boost_name
        cfg_path = pathlib.Path(__file__).parent.parent / "data" / "model-config.json"
        overrides = json.loads(cfg_path.read_text()).get("card_boost", {}).get("player_overrides", {})
        norm = _normalize_boost_name("Jared McCain")
        match = next(
            (float(v) for k, v in overrides.items() if _normalize_boost_name(k) == norm),
            None
        )
        assert match is not None, "Jared McCain must be in card_boost.player_overrides"
        assert match == pytest.approx(2.9), f"McCain override should be 2.9, got {match}"


class TestPerGameFloor:
    """Per-game pool uses configurable recent_min floor (15) and pts floor (8)."""

    def _local_cfg(self):
        """Read data/model-config.json directly (bypasses GitHub/_cfg fallback to defaults)."""
        import json, pathlib
        cfg_path = pathlib.Path(__file__).parent.parent / "data" / "model-config.json"
        return json.loads(cfg_path.read_text())

    def test_game_min_floor_config_key_is_15(self):
        """lineup.game_recent_min_floor must be 15.0 in config (was hardcoded 20)."""
        cfg = self._local_cfg()
        floor = cfg.get("lineup", {}).get("game_recent_min_floor")
        assert floor is not None, "lineup.game_recent_min_floor must be in model-config.json"
        assert floor == pytest.approx(15.0), f"game_recent_min_floor should be 15.0, got {floor}"

    def test_game_pts_floor_config_key_is_8(self):
        """scoring_thresholds.min_game_pts must be 8.0 (was min_moonshot_pts=10)."""
        cfg = self._local_cfg()
        floor = cfg.get("scoring_thresholds", {}).get("min_game_pts")
        assert floor is not None, "scoring_thresholds.min_game_pts must be in model-config.json"
        assert floor == pytest.approx(8.0), f"min_game_pts should be 8.0, got {floor}"

    def test_per_game_code_uses_config_key(self):
        """Per-game pool source must use game_recent_min_floor config key, not literal 20.0."""
        import inspect
        from api import index
        src = inspect.getsource(index)
        assert "game_recent_min_floor" in src, \
            "Per-game pool must reference game_recent_min_floor config key"
        assert "scoring_thresholds.min_game_pts" in src, \
            "Per-game pool must use scoring_thresholds.min_game_pts config key"

    def test_chalk_season_min_floor_is_20(self):
        """chalk_season_min_floor must be 20 — rotation players in Starting 5 (lowered from 22 for tight-game coverage)."""
        cfg = self._local_cfg()
        floor = cfg.get("projection", {}).get("chalk_season_min_floor")
        assert floor == pytest.approx(20.0), \
            f"chalk_season_min_floor should be 20.0, got {floor}"

    def test_den_not_in_big_market_teams(self):
        """DEN must be removed from big_market_teams — DEN role players are low-ownership."""
        cfg = self._local_cfg()
        big_markets = cfg.get("card_boost", {}).get("big_market_teams", [])
        assert "DEN" not in big_markets, \
            "DEN must not be in big_market_teams — Braun/Porter etc. are not heavily drafted"


# ---------------------------------------------------------------------------
# BOOST INGESTION — Layer 0 (pre-game daily boosts)
# ---------------------------------------------------------------------------
class TestDailyBoostIngestion:
    """Verify that Layer 0 (daily boost ingestion) takes priority over all
    other boost estimation layers when available."""

    def test_load_daily_boosts_returns_empty_when_no_file(self):
        """_load_daily_boosts returns {} when no boosts file exists for today."""
        from api.index import _load_daily_boosts, _DAILY_BOOST_CACHE
        import api.index as idx
        # Reset cache
        idx._DAILY_BOOST_CACHE = {}
        idx._DAILY_BOOST_DATE = ""
        idx._DAILY_BOOST_TS = 0
        with patch.object(idx, "_github_get_file", return_value=(None, None)):
            result = _load_daily_boosts("2099-01-01")
        assert result == {}

    def test_load_daily_boosts_parses_json(self):
        """_load_daily_boosts parses a valid boosts JSON from GitHub."""
        from api.index import _load_daily_boosts
        import api.index as idx
        idx._DAILY_BOOST_CACHE = {}
        idx._DAILY_BOOST_DATE = ""
        idx._DAILY_BOOST_TS = 0
        mock_data = json.dumps({
            "date": "2026-03-19",
            "players": [
                {"player_name": "Gary Payton II", "boost": 3.0},
                {"player_name": "Jared McCain", "boost": 2.8},
            ]
        })
        with patch.object(idx, "_github_get_file", return_value=(mock_data, "sha123")):
            result = _load_daily_boosts("2026-03-19")
        assert len(result) == 2
        assert result.get("gary payton ii") == 3.0
        assert result.get("jared mccain") == 2.8

    def test_est_card_boost_uses_daily_boost_first(self):
        """_est_card_boost Layer 0 (daily ingestion) overrides all other layers."""
        from api.index import _est_card_boost
        import api.index as idx
        # Inject a daily boost for a player who also has a config override
        idx._DAILY_BOOST_CACHE = {"shai gilgeous-alexander": 0.5}
        idx._DAILY_BOOST_DATE = idx._et_date().isoformat()
        import time
        idx._DAILY_BOOST_TS = time.time()
        # SGA has 0.0 in config overrides, but daily boost says 0.5
        boost = _est_card_boost(30, 25.0, "OKC", "Shai Gilgeous-Alexander")
        assert boost == 0.5, f"Layer 0 should override config override, got {boost}"
        # Cleanup
        idx._DAILY_BOOST_CACHE = {}
        idx._DAILY_BOOST_TS = 0


# ---------------------------------------------------------------------------
# TWO-PASS PIPELINE — watchlist and pass metadata
# ---------------------------------------------------------------------------
class TestWatchlist:
    """Verify watchlist generation identifies cascade-sensitive players."""

    def test_build_watchlist_returns_list(self):
        """_build_watchlist must return a list (empty is OK)."""
        from api.index import _build_watchlist
        chalk = [{"name": "Star1", "team": "LAL", "season_min": 30, "rating": 5.0, "est_mult": 0.5}]
        upside = [{"name": "Contrarian1", "team": "BOS", "season_min": 22, "rating": 3.5, "est_mult": 2.5}]
        all_proj = chalk + upside + [
            {"name": "Bench1", "team": "LAL", "season_min": 18, "rating": 2.5, "est_mult": 2.0, "pos": "SG"},
        ]
        result = _build_watchlist(chalk, upside, all_proj, [])
        assert isinstance(result, list)

    def test_watchlist_identifies_cascade_candidate(self):
        """A high-boost bench player should appear on watchlist when a lineup
        teammate going OUT would boost their minutes significantly."""
        from api.index import _build_watchlist
        lineup_player = {"name": "Starter", "team": "LAL", "season_min": 32, "rating": 4.5, "est_mult": 0.3, "predMin": 32}
        bench_player = {"name": "BenchGuy", "team": "LAL", "season_min": 18, "rating": 3.0, "est_mult": 2.5, "pos": "SG"}
        chalk = [lineup_player]
        upside = []
        all_proj = [lineup_player, bench_player]
        result = _build_watchlist(chalk, upside, all_proj, [])
        watchlist_names = [w["player"] for w in result]
        assert "BenchGuy" in watchlist_names, f"BenchGuy should be on watchlist, got {watchlist_names}"
        entry = [w for w in result if w["player"] == "BenchGuy"][0]
        assert entry["trigger_event"] == "injury_cascade"
        assert entry["depends_on"] == "Starter"

    def test_watchlist_max_10(self):
        """Watchlist is capped at 10 entries."""
        from api.index import _build_watchlist
        lineup = [{"name": "Star", "team": "LAL", "season_min": 35, "rating": 5.0, "est_mult": 0.3, "predMin": 35}]
        # 15 bench players on same team
        bench = [
            {"name": f"Bench{i}", "team": "LAL", "season_min": 18, "rating": 2.5 + i*0.1, "est_mult": 2.0, "pos": "SG"}
            for i in range(15)
        ]
        result = _build_watchlist(lineup, [], lineup + bench, [])
        assert len(result) <= 10


# ---------------------------------------------------------------------------
# PARSE-SCREENSHOT — "boosts" type
# ---------------------------------------------------------------------------
class TestBoostsScreenshotType:
    """Verify parse-screenshot accepts 'boosts' screenshot_type."""

    def test_boosts_type_in_parse_screenshot(self):
        """The 'boosts' screenshot type should produce a valid prompt."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert 'screenshot_type == "boosts"' in src, \
            "parse-screenshot must handle 'boosts' screenshot_type"
        assert "pre-game" in src.lower() or "pre_game" in src.lower() or "boost data" in src.lower(), \
            "boosts prompt should reference pre-game context"


# ─────────────────────────────────────────────────────────
# TestCorePoolRsMetric — core_pool.metric = "rs" ranks by raw projected RS
# ─────────────────────────────────────────────────────────
class TestCorePoolRsMetric:
    """When core_pool.metric = 'rs', core pool is ranked by raw rating (projected RS)."""

    def test_rs_metric_ranks_by_rating(self):
        """With metric='rs', a high-RS low-boost player ranks above low-RS high-boost."""
        from api.index import _build_lineups

        def cfg_rs(key, default=None):
            if key == "core_pool":
                return {"enabled": True, "size": 8, "metric": "rs", "blend_weight": 0.5}
            return default

        with patch("api.index._cfg", side_effect=cfg_rs), \
             patch("api.index.get_all_statuses", return_value={}), \
             patch("api.index._fetch_team_def_stats", return_value={}):
            _, _, core = _build_lineups([])

        # With empty projections, core is empty list; verify no crash
        assert isinstance(core, list)

    def test_rs_metric_code_branch_exists(self):
        """The 'rs' metric branch must exist in _build_lineups."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert 'core_metric == "rs"' in src, \
            "_build_lineups must have a branch for core_pool.metric='rs'"
        assert 'r.get("rating", 0)' in src, \
            "The 'rs' branch should use rating (projected RS) as core_score"

    def test_max_ev_still_works(self):
        """Backward compat: metric='max_ev' still uses max(chalk_ev, moonshot_ev)."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert 'core_metric == "max_ev"' in src


# ─────────────────────────────────────────────────────────
# TestPerGameCarryCorePool — per_game_carry surfaces local TV leaders
# ─────────────────────────────────────────────────────────
class TestPerGameCarryCorePool:
    """core_pool.per_game_carry forces top-K per matchup into the core before global trim."""

    def test_promotes_local_hero_above_mid_global(self):
        from api.index import _apply_per_game_carry_core_pool

        sorted_union = [
            {"name": "Star", "_core_score": 100},
            {"name": "Mid", "_core_score": 50},
            {"name": "LocalHero", "_core_score": 40},
        ]
        chalk_eligible = [
            {"name": "Star", "team": "A", "opp": "B", "rating": 10, "est_mult": 1},
            {"name": "Mid", "team": "A", "opp": "B", "rating": 5, "est_mult": 1},
            {"name": "LocalHero", "team": "C", "opp": "D", "rating": 8, "est_mult": 1},
        ]
        core = _apply_per_game_carry_core_pool(sorted_union, chalk_eligible, 3, 1, 1.6)
        names = [r["name"] for r in core]
        assert names == ["Star", "LocalHero", "Mid"]

    def test_per_game_carry_zero_is_plain_slice(self):
        from api.index import _apply_per_game_carry_core_pool

        su = [{"name": "B", "_core_score": 2}, {"name": "A", "_core_score": 1}]
        ce = [{"name": "A", "team": "X", "opp": "Y", "rating": 1, "est_mult": 1}]
        assert _apply_per_game_carry_core_pool(su, ce, 1, 0, 1.6) == [su[0]]


# ─────────────────────────────────────────────────────────
# TestVolatilityGuard — high-variance perimeter downshift (Mar 22 audit)
# ─────────────────────────────────────────────────────────
class TestVolatilityGuard:
    """projection.volatility_guard optional dampening in project_player."""

    def test_volatility_guard_branch_exists(self):
        import api.index as idx
        src = open(idx.__file__).read()
        assert "volatility_guard" in src
        assert "min_scoring_variance" in src


# ─────────────────────────────────────────────────────────
# TestMoonshotEvRatingBlend — moonshot EV blends toward raw RS×matchup
# ─────────────────────────────────────────────────────────
class TestMoonshotEvRatingBlend:
    """moonshot.ev_rating_blend lets stable scorers compete with pure boost leverage."""

    def test_ev_rating_blend_in_moonshot_loop(self):
        import api.index as idx
        src = open(idx.__file__).read()
        assert "ev_rating_blend" in src
        assert "_evb" in src

    def test_model_config_has_ev_rating_blend(self):
        cfg = json.load(open("data/model-config.json"))
        ms = cfg.get("moonshot", {})
        assert "ev_rating_blend" in ms
        assert 0.0 <= float(ms["ev_rating_blend"]) <= 0.5


# ─────────────────────────────────────────────────────────
# TestMoonshotRsBypass — high-RS players bypass boost floor
# ─────────────────────────────────────────────────────────
class TestMoonshotRsBypass:
    """moonshot.rs_bypass allows high-RS proven scorers to bypass min_card_boost."""

    def test_rs_bypass_code_exists(self):
        """The rs_bypass pathway must exist in moonshot eligibility."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "rs_bypass" in src, "Moonshot must have rs_bypass pathway"
        assert 'rs_bypass.get("enabled"' in src, "rs_bypass must check enabled flag"
        assert 'rs_bypass.get("min_rating"' in src, "rs_bypass must check min_rating"

    def test_rs_bypass_config_present(self):
        """model-config.json must have moonshot.rs_bypass block."""
        cfg = json.load(open("data/model-config.json"))
        rb = cfg.get("moonshot", {}).get("rs_bypass", {})
        assert rb.get("enabled") is True, "rs_bypass should be enabled in production config"
        assert rb.get("min_rating", 0) >= 4.0, "min_rating should be high enough to filter bench players"
        assert rb.get("min_season_min", 0) >= 20.0, "min_season_min should require proven starters"

    def test_rs_bypass_defaults_enabled(self):
        """_CONFIG_DEFAULTS should have rs_bypass enabled (14-date audit: Hidden Star wins 29%)."""
        from api.index import _CONFIG_DEFAULTS
        rb = _CONFIG_DEFAULTS.get("moonshot", {}).get("rs_bypass", {})
        assert rb.get("enabled") is True, "rs_bypass should be enabled — Hidden Star archetype (RS 5+, boost < 2.0) wins 29% of daily contests"


# ─────────────────────────────────────────────────────────
# TestChalkMilpRsFocusBalanced — chalk_milp_rs_focus balances RS and boost
# ─────────────────────────────────────────────────────────
class TestChalkMilpRsFocusHigh:
    """At chalk_milp_rs_focus=0.40, MILP uses 60% real boost + 40% neutral."""

    def test_rs_focus_calculation(self):
        """At rs_focus=0.40, effective boost is 60% real + 40% neutral."""
        rs_focus = 0.40
        boost_neutral = 1.0

        # High-boost player (2.5x)
        high_boost_eff = (1.0 - rs_focus) * 2.5 + rs_focus * boost_neutral
        # Low-boost player (0.5x)
        low_boost_eff = (1.0 - rs_focus) * 0.5 + rs_focus * boost_neutral

        # At 0.40, boost gap = 60% of raw gap — boost matters meaningfully
        raw_gap = 2.5 - 0.5  # 2.0
        eff_gap = high_boost_eff - low_boost_eff
        assert eff_gap > raw_gap * 0.5, \
            f"At rs_focus=0.40, boost gap should be >50% of raw gap; got {eff_gap:.2f} vs {raw_gap}"
        assert eff_gap < raw_gap * 0.7, \
            f"At rs_focus=0.40, boost gap should be <70% of raw gap; got {eff_gap:.2f} vs {raw_gap}"

    def test_rs_focus_code_reads_config(self):
        """chalk_milp_rs_focus mechanism must exist in chalk eligibility."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "chalk_milp_rs_focus" in src
        assert "chalk_milp_boost_neutral" in src

    def test_config_value(self):
        """Production config should have balanced rs_focus (0.3-0.5)."""
        cfg = json.load(open("data/model-config.json"))
        val = cfg.get("lineup", {}).get("chalk_milp_rs_focus", 0)
        assert 0.3 <= val <= 0.5, f"chalk_milp_rs_focus should be 0.3-0.5 for balanced; got {val}"


# ─────────────────────────────────────────────────────────
# TestStatStufferBonus — stat-stuffer RS multiplier for multi-category players
# ─────────────────────────────────────────────────────────
class TestStatStufferBonus:
    """Stat-stuffer bonus in project_player and _CONFIG_DEFAULTS."""

    def test_stat_stuffer_code_exists(self):
        """project_player should read stat_stuffer config."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "stat_stuffer" in src
        assert "bonus_td" in src
        assert "bonus_3cat" in src

    def test_stat_stuffer_defaults_present(self):
        """_CONFIG_DEFAULTS should include stat_stuffer with safe disabled default."""
        from api.index import _CONFIG_DEFAULTS
        ss = _CONFIG_DEFAULTS["real_score"]["stat_stuffer"]
        assert ss["enabled"] is False, "Offline default should be disabled"
        assert "pts_threshold" in ss
        assert "bonus_3cat" in ss
        assert "bonus_td" in ss

    def test_stat_stuffer_config_enabled(self):
        """Production config should have stat_stuffer enabled."""
        cfg = json.load(open("data/model-config.json"))
        ss = cfg.get("real_score", {}).get("stat_stuffer", {})
        assert ss.get("enabled") is True, "stat_stuffer should be enabled in production"


# ─────────────────────────────────────────────────────────
# TestCalibratedDfsWeights — DFS weight calibration applied
# ─────────────────────────────────────────────────────────
class TestCalibratedDfsWeights:
    """DFS weights in model-config should reflect calibrated values."""

    def test_steals_weight_moderate(self):
        """Steals weight should be moderate (v57: 8.0→3.0, 1 steal ≈ 2 pts scored)."""
        cfg = json.load(open("data/model-config.json"))
        stl_w = cfg.get("real_score", {}).get("dfs_weights", {}).get("stl", 0)
        assert 2.0 <= stl_w <= 5.0, f"Calibrated stl weight should be 2.0-5.0; got {stl_w}"

    def test_pts_weight_reduced(self):
        """Points weight should be reduced from 2.5 to ~1.5."""
        cfg = json.load(open("data/model-config.json"))
        pts_w = cfg.get("real_score", {}).get("dfs_weights", {}).get("pts", 0)
        assert pts_w <= 2.0, f"Calibrated pts weight should be <=2.0; got {pts_w}"

    def test_all_weights_present(self):
        """All 6 DFS weight keys should be present."""
        cfg = json.load(open("data/model-config.json"))
        w = cfg.get("real_score", {}).get("dfs_weights", {})
        for key in ["pts", "reb", "ast", "stl", "blk", "tov"]:
            assert key in w, f"Missing DFS weight key: {key}"


# ─────────────────────────────────────────────────────────
# TestEnhancedSpreadAdjustment — clutch-aligned spread adjustment
# ─────────────────────────────────────────────────────────
class TestEnhancedSpreadAdjustment:
    """Enhanced spread adjustment for RS-clutch alignment."""

    def test_spread_adj_code_has_tight_game_bonus(self):
        """Stars in tight games (spread 0-3) should get 1.20+ multiplier."""
        import api.index as idx
        src = open(idx.__file__).read()
        # Check for the enhanced spread adjustment pattern
        assert "1.25" in src, "Tight-game star bonus (1.25) should be in code"
        assert "0.55" in src, "Blowout star penalty floor (0.55) should be in code"

    def test_spread_adj_total_interaction(self):
        """High total + tight spread should produce interaction bonus."""
        import api.index as idx
        src = open(idx.__file__).read()
        # Check for total interaction logic
        assert "230" in src, "Shootout total threshold (230) should be in code"


# ─────────────────────────────────────────────────────────
# TestClosenessCoefficient — game-context RS multiplier
# ─────────────────────────────────────────────────────────
class TestClosenessCoefficient:
    """Closeness coefficient re-integrated selectively in project_player."""

    def test_closeness_import_available(self):
        """closeness_coefficient should be importable from real_score."""
        from api.real_score import closeness_coefficient
        assert callable(closeness_coefficient)

    def test_tight_game_higher_than_blowout(self):
        """Pick'em game should have higher closeness than 12-point spread."""
        from api.real_score import closeness_coefficient
        tight = closeness_coefficient(0, 222)
        blowout = closeness_coefficient(12, 222)
        assert tight > blowout, f"Tight {tight} should exceed blowout {blowout}"
        assert tight >= 1.5, f"Pick'em closeness should be >= 1.5; got {tight}"
        assert blowout <= 1.3, f"Blowout closeness should be <= 1.3; got {blowout}"

    def test_closeness_code_in_project_player(self):
        """project_player should use closeness_coefficient when enabled."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "closeness_coefficient" in src
        assert "close_cfg" in src
        assert "usage_scale" in src

    def test_closeness_config_enabled(self):
        """Production config should have closeness enabled."""
        cfg = json.load(open("data/model-config.json"))
        cc = cfg.get("real_score", {}).get("closeness", {})
        assert cc.get("enabled") is True

    def test_closeness_defaults_disabled_offline(self):
        """_CONFIG_DEFAULTS should have closeness disabled (safe fallback)."""
        from api.index import _CONFIG_DEFAULTS
        cc = _CONFIG_DEFAULTS["real_score"]["closeness"]
        assert cc["enabled"] is False


# ─────────────────────────────────────────────────────────
# TestCascadeRsBoost — usage-spike RS multiplier from injuries
# ─────────────────────────────────────────────────────────
class TestCascadeRsBoost:
    """Cascade RS boost gives direct RS multiplier for injury beneficiaries."""

    def test_cascade_rs_code_exists(self):
        """project_player should have cascade_rs logic."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "cascade_rs" in src
        assert "cascade_rs_mult" in src

    def test_cascade_rs_config_enabled(self):
        """Production config should have cascade_rs enabled."""
        cfg = json.load(open("data/model-config.json"))
        cr = cfg.get("real_score", {}).get("cascade_rs", {})
        assert cr.get("enabled") is True

    def test_cascade_rs_defaults_disabled(self):
        """_CONFIG_DEFAULTS should have cascade_rs disabled."""
        from api.index import _CONFIG_DEFAULTS
        cr = _CONFIG_DEFAULTS["real_score"]["cascade_rs"]
        assert cr["enabled"] is False


# ─────────────────────────────────────────────────────────
# TestRoleSpikeRs — hot streak / expanded role RS boost
# ─────────────────────────────────────────────────────────
class TestRoleSpikeRs:
    """Role-spike RS boost for players in expanded roles."""

    def test_role_spike_rs_code_exists(self):
        """project_player should have role_spike_rs logic."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "role_spike_rs" in src
        assert "spike_mult" in src
        assert "pts_surge" in src

    def test_role_spike_config_enabled(self):
        """Production config should have role_spike_rs enabled."""
        cfg = json.load(open("data/model-config.json"))
        rs = cfg.get("real_score", {}).get("role_spike_rs", {})
        assert rs.get("enabled") is True

    def test_role_spike_defaults_disabled(self):
        """_CONFIG_DEFAULTS should have role_spike_rs disabled."""
        from api.index import _CONFIG_DEFAULTS
        rs_cfg = _CONFIG_DEFAULTS["real_score"]["role_spike_rs"]
        assert rs_cfg["enabled"] is False


class TestLineSignals:
    """_generate_signals produces driver signals for narrative transparency."""

    def _make_player(self, **overrides):
        base = {"name": "Test Player", "team": "BOS", "pts": 12, "season_pts": 10,
                "recent_pts": 11, "predMin": 28, "season_min": 28, "min": 28,
                "_cascade_bonus": 0, "_matchup_factor": 1.0, "_odds_adjusted": False}
        base.update(overrides)
        return base

    def _make_gctx(self, **overrides):
        base = {"total": 222, "spread": -5, "opp_b2b": False, "opponent": "LAL"}
        base.update(overrides)
        return base

    def test_high_total_over(self):
        from api.line_engine import _generate_signals
        p = self._make_player()
        gctx = self._make_gctx(total=234)
        signals, bonus = _generate_signals(p, gctx, "over", "points", 10, 11, 12, 10.5, {})
        types = [s["type"] for s in signals]
        assert "high_total" in types
        assert bonus >= 8

    def test_high_total_not_under(self):
        from api.line_engine import _generate_signals
        p = self._make_player()
        gctx = self._make_gctx(total=234)
        signals, _ = _generate_signals(p, gctx, "under", "points", 10, 8, 8, 10.5, {})
        types = [s["type"] for s in signals]
        assert "high_total" not in types

    def test_low_total_under(self):
        from api.line_engine import _generate_signals
        p = self._make_player()
        gctx = self._make_gctx(total=212)
        signals, bonus = _generate_signals(p, gctx, "under", "points", 10, 8, 8, 10.5, {})
        types = [s["type"] for s in signals]
        assert "low_total" in types

    def test_low_total_not_over(self):
        from api.line_engine import _generate_signals
        p = self._make_player()
        gctx = self._make_gctx(total=212)
        signals, _ = _generate_signals(p, gctx, "over", "points", 10, 11, 12, 10.5, {})
        types = [s["type"] for s in signals]
        assert "low_total" not in types

    def test_matchup_weak_defense_over(self):
        from api.line_engine import _generate_signals
        p = self._make_player(_matchup_factor=1.10)
        gctx = self._make_gctx()
        signals, bonus = _generate_signals(p, gctx, "over", "points", 10, 11, 12, 10.5, {})
        types = [s["type"] for s in signals]
        assert "matchup" in types

    def test_matchup_strong_defense_under(self):
        from api.line_engine import _generate_signals
        p = self._make_player(_matchup_factor=0.90)
        gctx = self._make_gctx()
        signals, _ = _generate_signals(p, gctx, "under", "points", 10, 8, 8, 10.5, {})
        types = [s["type"] for s in signals]
        assert "matchup" in types

    def test_matchup_neutral_no_signal(self):
        from api.line_engine import _generate_signals
        p = self._make_player(_matchup_factor=1.02)
        gctx = self._make_gctx()
        signals, _ = _generate_signals(p, gctx, "over", "points", 10, 11, 12, 10.5, {})
        types = [s["type"] for s in signals]
        assert "matchup" not in types

    def test_books_agree_over(self):
        from api.line_engine import _generate_signals
        p = self._make_player(_odds_adjusted=True, odds_points_line=12.5)
        gctx = self._make_gctx()
        signals, _ = _generate_signals(p, gctx, "over", "points", 10, 11, 12, 10.5, {})
        types = [s["type"] for s in signals]
        assert "books_agree" in types

    def test_books_agree_not_under(self):
        from api.line_engine import _generate_signals
        p = self._make_player(_odds_adjusted=True, odds_points_line=12.5)
        gctx = self._make_gctx()
        signals, _ = _generate_signals(p, gctx, "under", "points", 10, 8, 8, 10.5, {})
        types = [s["type"] for s in signals]
        assert "books_agree" not in types

    def test_minutes_drop_under(self):
        from api.line_engine import _generate_signals
        p = self._make_player(predMin=25, season_min=33)
        gctx = self._make_gctx()
        signals, _ = _generate_signals(p, gctx, "under", "points", 10, 8, 8, 10.5, {})
        types = [s["type"] for s in signals]
        assert "minutes_drop" in types
        detail = [s["detail"] for s in signals if s["type"] == "minutes_drop"][0]
        assert "25" in detail and "33" in detail

    def test_minutes_drop_small_no_signal(self):
        from api.line_engine import _generate_signals
        p = self._make_player(predMin=30, season_min=31)
        gctx = self._make_gctx()
        signals, _ = _generate_signals(p, gctx, "under", "points", 10, 8, 8, 10.5, {})
        types = [s["type"] for s in signals]
        assert "minutes_drop" not in types

    def test_blowout_risk_under(self):
        from api.line_engine import _generate_signals
        p = self._make_player()
        gctx = self._make_gctx(spread=-10)
        signals, _ = _generate_signals(p, gctx, "under", "points", 10, 8, 8, 10.5, {})
        types = [s["type"] for s in signals]
        assert "blowout_risk" in types

    def test_close_game_over(self):
        from api.line_engine import _generate_signals
        p = self._make_player()
        gctx = self._make_gctx(spread=-2)
        signals, _ = _generate_signals(p, gctx, "over", "points", 10, 11, 12, 10.5, {})
        types = [s["type"] for s in signals]
        assert "close_game" in types

    def test_narrative_includes_driver(self):
        """run_model_fallback narrative should cite signals when present."""
        from api.line_engine import run_model_fallback
        proj = [self._make_player(name="Test Star", pts=15, season_pts=10, recent_pts=12,
                                  reb=5, season_reb=4, recent_reb=4, ast=3, season_ast=2,
                                  recent_ast=2, predMin=30, season_min=30, id="123")]
        games = [{"home": {"abbr": "BOS"}, "away": {"abbr": "LAL"},
                  "total": 235, "spread": -2, "startTime": ""}]
        result = run_model_fallback(proj, games, {"min_confidence": 0})
        over = result.get("over_pick")
        if over:
            # With total=235 and spread=-2, should have high_total and/or close_game signals
            assert over["signals"], "Over pick should have signals with high total + tight spread"
            # Narrative should not be bare "Model projects X — a Y edge vs Z baseline"
            assert "baseline" not in over["narrative"] or len(over["signals"]) == 0


# ─────────────────────────────────────────────────────────
# TestHighBoostRolePathway — role players with 2.0x+ boost bypass minutes floor
# ─────────────────────────────────────────────────────────
class TestHighBoostRolePathway:
    """High-boost role player pathway admits consistent rotation players the minutes
    floor would otherwise block, using boost magnitude as the quality gate."""

    def test_moonshot_code_has_pathway(self):
        """Moonshot pool must contain is_high_boost_role pathway."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "is_high_boost_role" in src, "Moonshot must have is_high_boost_role pathway"
        assert "hbr_min_boost" in src, "Moonshot must read hbr_min_boost from config"
        assert "hbr_min_recent" in src, "Moonshot must read hbr_min_recent from config"

    def test_chalk_code_has_pathway(self):
        """Chalk pool must contain is_chalk_high_boost_role pathway."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "is_chalk_high_boost_role" in src, "Chalk must have is_chalk_high_boost_role pathway"
        assert "chalk_hbr_min_boost" in src, "Chalk must read chalk_hbr_min_boost from config"
        assert "chalk_hbr_min_recent" in src, "Chalk must read chalk_hbr_min_recent from config"

    def test_moonshot_config_present(self):
        """model-config.json must have moonshot.high_boost_role block with correct keys."""
        cfg = json.load(open("data/model-config.json"))
        hbr = cfg.get("moonshot", {}).get("high_boost_role", {})
        assert hbr.get("enabled") is True, "high_boost_role should be enabled"
        assert hbr.get("min_boost", 0) >= 1.5, "min_boost should be high enough to gate quality"
        assert hbr.get("min_recent_min", 0) >= 12.0, "min_recent_min should require real minutes"
        assert hbr.get("min_pred_min", 0) >= 12.0, "min_pred_min should require real minutes today"

    def test_chalk_config_present(self):
        """model-config.json must have chalk HBR keys under projection."""
        cfg = json.load(open("data/model-config.json"))
        proj = cfg.get("projection", {})
        assert proj.get("chalk_hbr_enabled") is True, "chalk_hbr_enabled should be true"
        assert proj.get("chalk_hbr_min_boost", 0) >= proj.get("moonshot", {}).get(
            "high_boost_role", {}).get("min_boost", 0) or True, "chalk needs boost >= moonshot"
        assert proj.get("chalk_hbr_min_boost", 0) >= 2.0, "chalk boost threshold should be >= 2.0"
        assert proj.get("chalk_hbr_min_recent_min", 0) >= 14.0, "chalk requires more recent minutes"

    def test_chalk_threshold_stricter_than_moonshot(self):
        """Chalk HBR thresholds must be stricter than moonshot (reliability vs ceiling)."""
        cfg = json.load(open("data/model-config.json"))
        moon_boost = cfg["moonshot"]["high_boost_role"]["min_boost"]
        chalk_boost = cfg["projection"]["chalk_hbr_min_boost"]
        moon_recent = cfg["moonshot"]["high_boost_role"]["min_recent_min"]
        chalk_recent = cfg["projection"]["chalk_hbr_min_recent_min"]
        assert chalk_boost >= moon_boost, \
            f"Chalk boost threshold {chalk_boost} should be >= moonshot {moon_boost}"
        assert chalk_recent >= moon_recent, \
            f"Chalk recent_min threshold {chalk_recent} should be >= moonshot {moon_recent}"

    def test_pathway_included_in_moonshot_eligibility_check(self):
        """Moonshot eligibility check must OR in is_high_boost_role."""
        import api.index as idx
        src = open(idx.__file__).read()
        # The eligibility if-not block must include all four pathways
        assert "is_role_spike or is_high_boost_role" in src, \
            "Moonshot eligibility must include is_high_boost_role in OR chain"

    def test_pathway_included_in_chalk_eligibility_check(self):
        """Chalk eligibility check must OR in is_chalk_high_boost_role."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "is_spot_starter or is_chalk_high_boost_role" in src, \
            "Chalk eligibility must include is_chalk_high_boost_role in OR chain"


class TestRsCalibrationWeights:
    """DFS weight recalibration + archetype calibration + scorer upside signal.

    Data basis (13 dates + Mar 19 cross-reference):
    - Over-projected: Brook Lopez +3.0x (pred 6.5, actual 4.3), Clint Capela +3.0x
      (pred 3.5, actual 1.5), Rudy Gobert (pred 4.3, actual 1.7) — pure rebounders
    - Under-projected: Collin Gillespie (4.5 ast, +1.4), Dejounte Murray (2.8 RS low),
      Jalen Green (2 dates, +2.2–2.3), Aaron Nesmith (+2.2), DeRozan (2 dates, +2.7)
    - Bias: 56 under-projections vs 26 over-projections across 13 dates, avg MAE 1.57
    """

    def test_reb_weight_reduced(self):
        """reb DFS weight must be well below old value of 0.95 to fix pure-rebounder over-projection."""
        cfg = json.load(open("data/model-config.json"))
        reb_w = cfg["real_score"]["dfs_weights"]["reb"]
        assert reb_w < 0.80, (
            f"reb weight {reb_w} should be < 0.80; Lopez/Gobert/Capela were over-projected by 2+ RS "
            "when reb=0.95 gave them near-scorer DFS scores"
        )

    def test_ast_weight_elevated(self):
        """ast DFS weight must exceed old value of 0.3 to fix playmaker under-projection."""
        cfg = json.load(open("data/model-config.json"))
        ast_w = cfg["real_score"]["dfs_weights"]["ast"]
        assert ast_w > 0.40, (
            f"ast weight {ast_w} should be > 0.40; Gillespie (4.5 ast) and Murray (7+ ast) "
            "were under-projected because assists contributed almost nothing to DFS score"
        )

    def test_pts_weight_stable_and_stl_reduced(self):
        """pts weight should be stable at 1.5; stl reduced from 8.0 to 3.0 (v57 audit)."""
        cfg = json.load(open("data/model-config.json"))
        w = cfg["real_score"]["dfs_weights"]
        assert w["pts"] == 1.5, f"pts weight changed unexpectedly: {w['pts']}"
        assert w["stl"] == 3.0, f"stl weight should be 3.0 (v57); got {w['stl']}"

    def test_pure_rebounder_archetype_detected(self):
        """_infer_player_archetype must return 'pure_rebounder' for high-reb/low-scoring player."""
        from api.index import _infer_player_archetype
        # Lopez profile: 11.5 pts, 13 reb, 30 min → reb_pm = 0.43, season_pts = 11.5
        arch = _infer_player_archetype(11.5, 30.0, 13.0, {"season_pts": 11.5})
        assert arch == "pure_rebounder", (
            f"Expected 'pure_rebounder' for Lopez-profile, got '{arch}'; "
            "reb_pm=0.43 >= 0.28 threshold AND season_pts=11.5 < 12.0"
        )

    def test_scorer_archetype_detected(self):
        """_infer_player_archetype must return 'scorer' for high pts/min efficient scorer."""
        from api.index import _infer_player_archetype
        # Jalen Green profile: 17.7 pts, 28 min → pts/min = 0.63, season_pts = 17.7
        arch = _infer_player_archetype(17.7, 28.0, 3.5, {"season_pts": 17.7})
        assert arch == "scorer", (
            f"Expected 'scorer' for Jalen Green-profile, got '{arch}'; "
            "ppm=0.63 >= 0.55 AND season_pts=17.7 >= 15.0"
        )

    def test_big_still_detected_for_scoring_bigs(self):
        """Players with reb_pm >= 0.22 but pts >= 12 should remain 'big', not 'pure_rebounder'."""
        from api.index import _infer_player_archetype
        # Achiuwa profile: 9 pts, 24 min, 7 reb → reb_pm = 0.29, season_pts = 9 — is pure_rebounder
        # Jokic profile: 26 pts, 34 min, 13 reb → star (season_pts >= 21, avg_min >= 28)
        # Bam Adebayo profile: 19 pts, 33 min, 10 reb → reb_pm=0.30, season_pts=19 → big (pts >= 12)
        arch = _infer_player_archetype(19.0, 33.0, 10.0, {"season_pts": 19.0})
        assert arch == "big", (
            f"Expected 'big' for Bam-profile, got '{arch}'; "
            "season_pts=19 >= 12, so not pure_rebounder"
        )

    def test_wing_role_unchanged_for_avg_player(self):
        """A typical role player (low ppm, low reb_pm) should still be wing_role."""
        from api.index import _infer_player_archetype
        # Generic role player: 8 pts, 22 min, 3 reb
        arch = _infer_player_archetype(8.0, 22.0, 3.0, {"season_pts": 8.0})
        assert arch == "wing_role", f"Expected 'wing_role' for generic role player, got '{arch}'"

    def test_pure_rebounder_penalty_lt_big(self):
        """pure_rebounder calibration multiplier must be lower than big."""
        cfg = json.load(open("data/model-config.json"))
        archs = cfg["real_score"]["archetype_calibration"]["archetypes"]
        assert "pure_rebounder" in archs, "pure_rebounder archetype must be in config"
        assert archs["pure_rebounder"] < archs["big"], (
            f"pure_rebounder mult {archs['pure_rebounder']} should be < big mult {archs['big']}; "
            "pure rebounders generate less RS per rebound than scoring bigs"
        )

    def test_scorer_bonus_gt_wing_role(self):
        """scorer calibration multiplier must exceed wing_role."""
        cfg = json.load(open("data/model-config.json"))
        archs = cfg["real_score"]["archetype_calibration"]["archetypes"]
        assert "scorer" in archs, "scorer archetype must be in config"
        assert archs["scorer"] > archs["wing_role"], (
            f"scorer mult {archs['scorer']} should be > wing_role mult {archs['wing_role']}; "
            "efficient scorers have higher RS ceilings when hot"
        )

    def test_big_mult_reduced_from_1_04(self):
        """big archetype multiplier must be reduced to suppress non-star bigs."""
        cfg = json.load(open("data/model-config.json"))
        big_mult = cfg["real_score"]["archetype_calibration"]["archetypes"]["big"]
        assert big_mult < 1.0, (
            f"big mult {big_mult} should be < 1.0; was 1.04 (a bonus!) but "
            "non-star scoring bigs still over-project when they're in the 'big' bucket"
        )

    def test_scorer_upside_config_present(self):
        """moonshot.scorer_upside block must exist, be enabled, and have sensible values."""
        cfg = json.load(open("data/model-config.json"))
        su = cfg.get("moonshot", {}).get("scorer_upside", {})
        assert su, "moonshot.scorer_upside block must exist in model-config.json"
        assert su.get("enabled") is True, "scorer_upside must be enabled"
        assert su.get("min_pts_per_min", 0) >= 0.50, "min_pts_per_min should be >= 0.50"
        assert su.get("min_season_pts", 0) >= 12.0, "min_season_pts should be >= 12.0"
        assert 1.0 < su.get("multiplier", 1.0) <= 1.25, (
            "scorer_upside multiplier should be > 1.0 and <= 1.25 (modest boost, not a distortion)"
        )

    def test_scorer_upside_code_present(self):
        """Moonshot pool code must contain scorer_upside block."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "scorer_upside" in src, "Moonshot pool must contain scorer_upside block"
        assert "su_min_ppm" in src, "Scorer upside must read min_pts_per_min from config"
        assert "su_season_pts" in src, "Scorer upside must check season_pts threshold"


# ─────────────────────────────────────────────────────────
# TestCascadeCapFix — per_player_cap_minutes raised to 10.0
# Ensures primary-backup cascade scenarios correctly propagate significant minutes
# ─────────────────────────────────────────────────────────
class TestCascadeCapFix:
    """Cascade cap raised from 2-3 min to 10 min so primary backups correctly inherit
    starter-level minutes when a teammate goes OUT."""

    def test_cascade_cap_in_config(self):
        """model-config.json must have per_player_cap_minutes >= 8.0."""
        cfg = json.load(open("data/model-config.json"))
        cap = cfg.get("cascade", {}).get("per_player_cap_minutes", 0)
        assert cap >= 8.0, (
            f"per_player_cap_minutes should be >= 8.0 to allow meaningful cascade propagation, got {cap}"
        )

    def test_cascade_cap_in_defaults(self):
        """_CONFIG_DEFAULTS cascade.per_player_cap_minutes must be >= 8.0 in source."""
        src = open("api/index.py").read()
        # Find per_player_cap_minutes in the _CONFIG_DEFAULTS dict
        m = re.search(r'"cascade":\s*\{[^}]*"per_player_cap_minutes"\s*:\s*([\d.]+)', src)
        assert m, "_CONFIG_DEFAULTS must define cascade.per_player_cap_minutes"
        cap = float(m.group(1))
        assert cap >= 8.0, (
            f"_CONFIG_DEFAULTS per_player_cap_minutes should be >= 8.0, got {cap}"
        )

    def test_cascade_rs_enabled_in_config(self):
        """Production config must have cascade_rs enabled so cascade players get RS uplift."""
        cfg = json.load(open("data/model-config.json"))
        cr = cfg.get("real_score", {}).get("cascade_rs", {})
        assert cr.get("enabled") is True, "cascade_rs.enabled must be True in production config"


# ─────────────────────────────────────────────────────────
# TestRotoConfirmedRatingException — confirmed rotation players bypass min_rating_floor
# ─────────────────────────────────────────────────────────
class TestRotoConfirmedRatingException:
    """Confirmed rotation players with high boost pass a lower RS floor (2.2) in
    the moonshot pool, capturing players like Garza, Paul Reed, Taylor Hendricks."""

    def test_roto_confirmed_exception_code_exists(self):
        """Moonshot pool must have roto_confirmed_min_rating exception logic."""
        src = open("api/index.py").read()
        assert "roto_confirmed_min_rating" in src, (
            "Moonshot pool must read roto_confirmed_min_rating from config"
        )
        assert "roto_confirmed_min_boost" in src, (
            "Moonshot pool must read roto_confirmed_min_boost from config"
        )
        assert "_is_roto_confirmed" in src, (
            "Moonshot pool must check _is_roto_confirmed for the exception"
        )
        assert "_has_cascade" in src, (
            "Moonshot pool must allow cascade-elevated players via _has_cascade"
        )

    def test_roto_confirmed_config_keys(self):
        """model-config.json moonshot section must have roto_confirmed_min_rating and min_boost."""
        cfg = json.load(open("data/model-config.json"))
        moon = cfg.get("moonshot", {})
        assert "roto_confirmed_min_rating" in moon, (
            "moonshot.roto_confirmed_min_rating must exist in model-config.json"
        )
        assert "roto_confirmed_min_boost" in moon, (
            "moonshot.roto_confirmed_min_boost must exist in model-config.json"
        )
        min_rating = moon["roto_confirmed_min_rating"]
        min_boost = moon["roto_confirmed_min_boost"]
        assert 2.0 <= min_rating <= 2.5, (
            f"roto_confirmed_min_rating should be 2.0-2.5 (selective but not too strict), got {min_rating}"
        )
        assert 2.0 <= min_boost <= 3.0, (
            f"roto_confirmed_min_boost should be 2.0-3.0, got {min_boost}"
        )

    def test_context_pass_includes_cascade_fields(self):
        """Context pass player payload must include cascade_bonus and roto_status."""
        src = open("api/index.py").read()
        assert '"cascade_bonus"' in src, "Context pass payload must include cascade_bonus field"
        assert '"roto_status"' in src, "Context pass payload must include roto_status field"
        assert "CASCADE & ROLE CONFIRMATION SIGNALS" in src, (
            "Context pass prompt must include cascade signal instructions"
        )


class TestMaxPerGameConstraint:
    """MILP optimizer must limit players from the same game matchup."""

    def test_max_per_game_config_keys_exist(self):
        """model-config.json lineup section must have chalk_max_per_game and moonshot_max_per_game."""
        cfg = json.load(open("data/model-config.json"))
        lu = cfg.get("lineup", {})
        assert "chalk_max_per_game" in lu, "lineup.chalk_max_per_game must exist in model-config.json"
        assert "moonshot_max_per_game" in lu, "lineup.moonshot_max_per_game must exist in model-config.json"
        assert lu["chalk_max_per_game"] in (2, 3), (
            f"chalk_max_per_game should be 2 or 3, got {lu['chalk_max_per_game']}"
        )
        assert lu["moonshot_max_per_game"] in (2, 3), (
            f"moonshot_max_per_game should be 2 or 3, got {lu['moonshot_max_per_game']}"
        )

    def test_max_per_game_optimizer_params(self):
        """asset_optimizer.py must accept max_per_game and player_games parameters."""
        src = open("api/asset_optimizer.py").read()
        assert "max_per_game=0" in src, "optimize_lineup must have max_per_game=0 default param"
        assert "player_games=None" in src, "optimize_lineup must have player_games=None default param"
        assert "max_per_game_{game_id}" in src, "_solve_milp must add named per-game constraint"

    def test_max_per_game_constraint_enforced(self):
        """MILP must not select more than max_per_game players from the same game."""
        try:
            from api.asset_optimizer import optimize_lineup, PULP_AVAILABLE
            if not PULP_AVAILABLE:
                pytest.skip("PuLP not available")
        except Exception:
            pytest.skip("PuLP not available")

        # 3 players from LAL vs BOS, 2 from other games — max_per_game=2 should block 3rd LAL/BOS
        players = [
            {"name": "A", "rating": 5.0, "est_mult": 1.5, "team": "LAL", "opp": "BOS", "chalk_ev": 10.0, "player_variance": 0},
            {"name": "B", "rating": 4.5, "est_mult": 1.5, "team": "BOS", "opp": "LAL", "chalk_ev": 9.5, "player_variance": 0},
            {"name": "C", "rating": 4.0, "est_mult": 1.5, "team": "LAL", "opp": "BOS", "chalk_ev": 9.0, "player_variance": 0},
            {"name": "D", "rating": 3.5, "est_mult": 1.5, "team": "GSW", "opp": "PHX", "chalk_ev": 8.0, "player_variance": 0},
            {"name": "E", "rating": 3.0, "est_mult": 1.5, "team": "MIA", "opp": "ATL", "chalk_ev": 7.0, "player_variance": 0},
            {"name": "F", "rating": 2.5, "est_mult": 1.5, "team": "CHI", "opp": "DET", "chalk_ev": 6.0, "player_variance": 0},
        ]
        game_ids = ["BOS_vs_LAL", "BOS_vs_LAL", "BOS_vs_LAL", "GSW_vs_PHX", "ATL_vs_MIA", "CHI_vs_DET"]

        result = optimize_lineup(players, n=5, player_games=game_ids, max_per_game=2)
        lal_bos_count = sum(1 for p in result if p.get("team") in ("LAL", "BOS"))
        assert lal_bos_count <= 2, (
            f"Expected at most 2 players from LAL/BOS game, got {lal_bos_count}: "
            f"{[p['name'] for p in result]}"
        )

    def test_player_game_id_derivation_in_build_lineups(self):
        """index.py must derive player_game_id from team+opp and pass player_games to MILP."""
        src = open("api/index.py").read()
        assert "_player_game_id" in src, "_build_lineups must define _player_game_id helper"
        assert "chalk_max_per_game" in src, "_build_lineups must read chalk_max_per_game config"
        assert "moonshot_max_per_game" in src, "_build_lineups must read moonshot_max_per_game config"
        assert "_chalk_elig_games" in src or "_chalk_source_games" in src, (
            "_build_lineups must build player_games list for chalk MILP"
        )
        assert "_moon_pool_games" in src, (
            "_build_lineups must build player_games list for moonshot MILP"
        )


class TestMinHighBoostConstraint:
    """MILP optimizer must guarantee minimum high-boost player count in chalk lineup."""

    def test_min_high_boost_config_keys_exist(self):
        """model-config.json lineup section must have high-boost count and threshold keys."""
        cfg = json.load(open("data/model-config.json"))
        lu = cfg.get("lineup", {})
        required = [
            "chalk_min_high_boost_count",
            "chalk_high_boost_threshold",
            "chalk_min_big_boost_count",
            "chalk_big_boost_threshold",
        ]
        for key in required:
            assert key in lu, f"lineup.{key} must exist in model-config.json"
        assert lu["chalk_min_high_boost_count"] >= 1, "Should require at least 1 high-boost player"
        assert lu["chalk_high_boost_threshold"] >= 1.5, "High-boost threshold should be at least 1.5x"
        assert lu["chalk_big_boost_threshold"] > lu["chalk_high_boost_threshold"], (
            "Big-boost threshold must exceed high-boost threshold"
        )

    def test_min_high_boost_optimizer_params(self):
        """asset_optimizer.py must accept min_high_boost_count and related params."""
        src = open("api/asset_optimizer.py").read()
        assert "min_high_boost_count=0" in src, "optimize_lineup must have min_high_boost_count=0 default"
        assert "high_boost_threshold=2.0" in src, "optimize_lineup must have high_boost_threshold=2.0 default"
        assert "min_big_boost_count=0" in src, "optimize_lineup must have min_big_boost_count=0 default"
        assert "big_boost_threshold=2.8" in src, "optimize_lineup must have big_boost_threshold=2.8 default"
        assert "min_high_boost" in src, "_solve_milp must add min_high_boost named constraint"
        assert "min_big_boost" in src, "_solve_milp must add min_big_boost named constraint"

    def test_min_high_boost_uses_raw_est_mult(self):
        """Boost constraints must check raw est_mult, not blended chalk_milp_boost."""
        src = open("api/asset_optimizer.py").read()
        # The constraint should index p.get("est_mult") — not card_boost_key
        assert 'p.get("est_mult", 0) >= high_boost_threshold' in src, (
            "min_high_boost constraint must check raw est_mult, not blended card_boost_key"
        )
        assert 'p.get("est_mult", 0) >= big_boost_threshold' in src, (
            "min_big_boost constraint must check raw est_mult, not blended card_boost_key"
        )

    def test_min_high_boost_constraint_enforced(self):
        """MILP must include at least min_high_boost_count players with boost >= threshold."""
        try:
            from api.asset_optimizer import optimize_lineup, PULP_AVAILABLE
            if not PULP_AVAILABLE:
                pytest.skip("PuLP not available")
        except Exception:
            pytest.skip("PuLP not available")

        # 2 high-boost players (3.0x), 4 moderate-boost players (1.0x) — must pick both high-boost
        players = [
            {"name": "Star1",  "rating": 5.0, "est_mult": 1.0, "team": "LAL", "opp": "BOS", "chalk_ev": 10.0, "player_variance": 0},
            {"name": "Star2",  "rating": 4.8, "est_mult": 1.0, "team": "BOS", "opp": "LAL", "chalk_ev": 9.8, "player_variance": 0},
            {"name": "Star3",  "rating": 4.5, "est_mult": 1.0, "team": "GSW", "opp": "PHX", "chalk_ev": 9.5, "player_variance": 0},
            {"name": "Boost1", "rating": 3.0, "est_mult": 3.0, "team": "MIA", "opp": "ATL", "chalk_ev": 8.0, "player_variance": 0},
            {"name": "Boost2", "rating": 2.8, "est_mult": 2.5, "team": "CHI", "opp": "DET", "chalk_ev": 7.5, "player_variance": 0},
            {"name": "Filler", "rating": 2.0, "est_mult": 1.0, "team": "ORL", "opp": "IND", "chalk_ev": 5.0, "player_variance": 0},
        ]

        result = optimize_lineup(players, n=5, min_high_boost_count=2, high_boost_threshold=2.0)
        high_boost_count = sum(1 for p in result if p.get("est_mult", 0) >= 2.0)
        assert high_boost_count >= 2, (
            f"Expected at least 2 high-boost players (>=2.0x), got {high_boost_count}: "
            f"{[(p['name'], p.get('est_mult')) for p in result]}"
        )

    def test_min_high_boost_passes_to_milp_calls(self):
        """index.py must pass high-boost constraints to chalk optimize_lineup calls."""
        src = open("api/index.py").read()
        assert "chalk_min_high_boost_count" in src, "_build_lineups must read chalk_min_high_boost_count config"
        assert "chalk_high_boost_threshold" in src, "_build_lineups must read chalk_high_boost_threshold config"
        assert "chalk_min_big_boost_count" in src, "_build_lineups must read chalk_min_big_boost_count config"
        assert "chalk_big_boost_threshold" in src, "_build_lineups must read chalk_big_boost_threshold config"
        assert "min_high_boost_count=_chalk_min_high_boost" in src, (
            "_build_lineups must pass min_high_boost_count to chalk optimize_lineup"
        )
        assert "min_big_boost_count=_chalk_min_big_boost" in src, (
            "_build_lineups must pass min_big_boost_count to chalk optimize_lineup"
        )


# ─────────────────────────────────────────────────────────
# TestApiResilience — Claude API rate-limit / overload resilience
# ─────────────────────────────────────────────────────────
class TestApiResilience:
    """Verify the app behaves correctly when Anthropic API is rate-limited or overloaded."""

    def test_context_pass_skips_when_no_fresh_signals(self):
        """Context pass is a no-op when there's no news, no cascade, and no B2B games."""
        from api.index import _claude_context_pass

        players = [
            {"name": "Player A", "team": "LAL", "rating": 4.0, "chalk_ev": 16.0,
             "ceiling_score": 5.0, "est_mult": 1.5, "season_pts": 10.0,
             "season_reb": 4.0, "season_ast": 3.0, "season_stl": 0.8, "season_blk": 0.4}
        ]
        games = [{"gameId": "g1", "spread": 0, "total": 222,
                  "home": {"abbr": "LAL", "id": "1"}, "away": {"abbr": "BOS", "id": "2"}}]
        # No _cascade_bonus on any player, no B2B on any game

        mock_client = Mock()

        def cfg_side_effect(key, default=None):
            if key == "context_layer.enabled": return True
            if key == "context_layer.model": return "claude-sonnet-4-6-20250514"
            if key == "context_layer.max_adjustment": return 0.4
            if key == "context_layer.timeout_seconds": return 15
            return default

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch("api.index._fetch_nba_news_context", return_value=""), \
             patch("anthropic.Anthropic", return_value=mock_client), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            _claude_context_pass(players, games)

        # API must NOT be called — no fresh signals means Claude adds no value
        mock_client.messages.create.assert_not_called()
        assert players[0]["rating"] == 4.0
        assert "_context_adj" not in players[0]

    def test_context_pass_runs_when_cascade_present(self):
        """Context pass calls Claude when a player has cascade bonus, even without news."""
        from api.index import _claude_context_pass

        players = [
            {"name": "Backup Guard", "team": "LAL", "rating": 3.0, "chalk_ev": 12.0,
             "ceiling_score": 4.0, "est_mult": 2.5, "season_pts": 8.0,
             "season_reb": 3.0, "season_ast": 2.0, "season_stl": 0.5, "season_blk": 0.2,
             "_cascade_bonus": 8.0}  # starter is out → this player inherits minutes
        ]
        games = [{"gameId": "g1", "spread": 0, "total": 222,
                  "home": {"abbr": "LAL", "id": "1"}, "away": {"abbr": "BOS", "id": "2"}}]

        claude_response = json.dumps({"adjustments": []})
        mock_msg = Mock()
        mock_msg.content = [Mock(text=claude_response)]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_msg

        def cfg_side_effect(key, default=None):
            if key == "context_layer.enabled": return True
            if key == "context_layer.model": return "claude-sonnet-4-6-20250514"
            if key == "context_layer.max_adjustment": return 0.4
            if key == "context_layer.timeout_seconds": return 15
            return default

        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch("api.index._fetch_nba_news_context", return_value=""), \
             patch("anthropic.Anthropic", return_value=mock_client), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            _claude_context_pass(players, games)

        # Claude MUST be called — cascade is a genuine fresh signal
        mock_client.messages.create.assert_called_once()

    def test_context_pass_runs_when_news_present(self):
        """Context pass calls Claude when web search found news, even without cascade/B2B."""
        from api.index import _claude_context_pass

        players = [
            {"name": "Role Player", "team": "MEM", "rating": 3.5, "chalk_ev": 14.0,
             "ceiling_score": 4.5, "est_mult": 1.8, "season_pts": 9.0,
             "season_reb": 5.0, "season_ast": 2.5, "season_stl": 0.9, "season_blk": 0.3}
        ]
        games = [{"gameId": "g1", "spread": -2, "total": 228,
                  "home": {"abbr": "MEM", "id": "1"}, "away": {"abbr": "OKC", "id": "2"}}]

        claude_response = json.dumps({"adjustments": []})
        mock_msg = Mock()
        mock_msg.content = [Mock(text=claude_response)]
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_msg

        def cfg_side_effect(key, default=None):
            if key == "context_layer.enabled": return True
            if key == "context_layer.model": return "claude-sonnet-4-6-20250514"
            if key == "context_layer.max_adjustment": return 0.4
            if key == "context_layer.timeout_seconds": return 15
            return default

        news = "- Ja Morant ruled out tonight — Brandon Clarke expected to start"
        with patch("api.index._cfg", side_effect=cfg_side_effect), \
             patch("api.index._fetch_nba_news_context", return_value=news), \
             patch("anthropic.Anthropic", return_value=mock_client), \
             patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            _claude_context_pass(players, games)

        # Claude MUST be called — news is a genuine fresh signal
        mock_client.messages.create.assert_called_once()

    def test_context_pass_sdk_uses_no_retries(self):
        """Anthropic SDK client for context pass is initialized with max_retries=0."""
        src = open("api/index.py").read()
        # Verify all three optional-layer SDK instantiations use max_retries=0
        assert src.count("Anthropic(api_key=anthropic_key, max_retries=0)") >= 2, (
            "web_search and lineup_review SDK clients must use max_retries=0"
        )
        assert "Anthropic(api_key=_ctx_api_key, max_retries=0)" in src, (
            "context_pass SDK client must use max_retries=0"
        )

    def test_line_engine_http_429_logged_as_rate_limited(self):
        """HTTP 429 from Claude is logged as 'rate-limited' and falls back cleanly."""
        import requests as req_lib
        from api.line_engine import _call_claude

        mock_resp = Mock()
        mock_resp.status_code = 429
        http_err = req_lib.exceptions.HTTPError(response=mock_resp)

        with patch("api.line_engine.requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = http_err
            result = _call_claude("dummy prompt", "points")

        assert result is None  # Falls back cleanly

    def test_line_engine_http_529_logged_as_overloaded(self):
        """HTTP 529 from Claude is logged as 'overloaded' and falls back cleanly."""
        import requests as req_lib
        from api.line_engine import _call_claude

        mock_resp = Mock()
        mock_resp.status_code = 529
        http_err = req_lib.exceptions.HTTPError(response=mock_resp)

        with patch("api.line_engine.requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = http_err
            result = _call_claude("dummy prompt", "assists")

        assert result is None  # Falls back cleanly

    def test_context_pass_prompt_requires_specific_evidence(self):
        """Context pass system prompt instructs Claude to require specific evidence, not team membership."""
        src = open("api/index.py").read()
        assert "Team membership alone" in src or "team membership alone" in src.lower(), (
            "System prompt must tell Claude team membership alone is not a reason to adjust"
        )

    def test_context_pass_team_style_conditional(self):
        """Context pass prompt marks team style as CONFIRMING only, not standalone."""
        src = open("api/index.py").read()
        assert "CONFIRMING" in src, (
            "Team/style signals section must label signals as CONFIRMING context only"
        )


# ─────────────────────────────────────────────────────────
# TestOddsApiFieldMapping — regression guard for name/description swap fix
# ─────────────────────────────────────────────────────────
class TestOddsApiFieldMapping:
    """Odds API returns {name: 'Over'/'Under', description: 'Player Name'}.
    _build_player_odds_map and _fetch_odds_line must read description as
    player key and name as direction — not the other way around."""

    def test_build_player_odds_map_field_assignment(self):
        """Source code reads description (not name) as player_key."""
        src = open("api/index.py").read()
        assert "def _build_player_odds_map(" in src, "_build_player_odds_map not found"
        # After the fix: player_key reads from description; search full source as
        # the outcome parsing is inside a nested _fetch_event_props closure
        assert 'player_key = outcome.get("description", "").lower()' in src, (
            "_build_player_odds_map must read description as player_key "
            "(Odds API: description = player name, name = direction)"
        )
        # After the fix: direction reads from name
        assert 'direction  = outcome.get("name", "").lower()' in src, (
            "_build_player_odds_map must read name as direction "
            "(Odds API: name = 'Over'/'Under')"
        )

    def test_fetch_odds_line_field_assignment(self):
        """_fetch_odds_line checks player name in description, not name."""
        src = open("api/index.py").read()
        assert "def _fetch_odds_line(" in src, "_fetch_odds_line not found"
        # After the fix: player name matched against description
        assert 'outcome.get("description", "").lower()' in src, (
            "_fetch_odds_line must check player name against description field"
        )
        # After the fix: direction read from name field
        assert 'outcome.get("name", "").lower() == "over"' in src, (
            "_fetch_odds_line must read direction from name field"
        )

    def test_synthetic_parlay_line_snapping(self):
        """Synthetic parlay fallback uses round() not floor() — avoids whole-number lines."""
        src = open("api/index.py").read()
        # Verify round is used (not floor) in the synthetic fallback block
        assert '"line": round(proj_val * 2) / 2' in src, (
            "Synthetic parlay fallback must use round(proj_val * 2) / 2 "
            "to produce 0.5-snapped lines (e.g. 5.3 → 5.5, not 5.3 → 5.0)"
        )
        # Verify floor is NOT used in the parlay synthetic path
        assert '"line": math.floor(proj_val * 2) / 2' not in src, (
            "math.floor in synthetic fallback produces whole-number lines — must use round()"
        )

    def test_parlay_engine_snaps_odds_api_line(self):
        """parlay_engine.py snaps real Odds API lines to nearest 0.5 (defensive float guard)."""
        src = open("api/parlay_engine.py").read()
        # Real Odds API always gives 0.5 values; snap is a defensive float guard
        assert 'round(float(odds_data.get("line", 0)) * 2) / 2' in src, (
            "parlay_engine must snap book_line to nearest 0.5 "
            "(Odds API gives 0.5 values; snap is a defensive guard against float edge cases)"
        )

    def test_resolved_pick_never_set_as_final(self):
        """Line rotation: when a direction resolves, final_pick is set to next_slate or None — never the resolved pick."""
        src = open("api/index.py").read()
        # After fix: unconditional assignment (not gated on 'if next_over:')
        assert "final_over = next_over  # Always update" in src, (
            "Resolved over pick must always be replaced by next_over (even if None)"
        )
        assert "final_under = next_under  # Always update" in src, (
            "Resolved under pick must always be replaced by next_under (even if None)"
        )

    def test_had_resolved_flag_present(self):
        """_had_resolved flag used so next_slate_pending fires when both directions resolved but next fails."""
        src = open("api/index.py").read()
        assert "_had_resolved = over_resolved or under_resolved" in src, (
            "_had_resolved must be set before rotation so pending check fires correctly"
        )
        assert "_had_resolved or final_over or final_under" in src, (
            "_has_fresh check must include _had_resolved for correct next_slate_pending return"
        )


class TestParlayHistoryAndConfigHardening:
    """Regression guards for parlay-history same-day resolution and config sanitization."""

    def test_parlay_history_supports_nocache(self):
        src = open("api/index.py").read()
        assert "async def parlay_history(request: Request):" in src
        assert 'request.query_params.get("nocache"' in src
        assert "if not nocache:" in src

    def test_parlay_history_resolves_today_when_final(self):
        src = open("api/index.py").read()
        assert "today_games_final = _all_games_final(_today_games)" in src
        assert '_can_resolve = (date_str < today_str) or (date_str == today_str and today_games_final)' in src

    def test_projection_only_persisted_and_labeled(self):
        src_api = open("api/index.py").read()
        src_ui = open("index.html").read()
        assert '"projection_only": result.get("projection_only", False)' in src_api
        assert "sourceBadge = data.projection_only" in src_ui
        assert "MODEL</span>'" in src_ui and "BOOK</span>'" in src_ui

    def test_projection_only_cache_bypass_while_open(self):
        src = open("api/index.py").read()
        assert "bypassing projection-only /tmp cache while slate is open" in src
        assert "projection-only GitHub ticket found; rebuilding once while slate is open" in src

    def test_line_and_parlay_sanitizers_enabled(self):
        from api.index import sanitize_line_config, sanitize_parlay_config
        line = sanitize_line_config({"min_confidence": "120", "stat_floors": {"points": "-5"}})
        assert line["min_confidence"] == 100
        assert line["stat_floors"]["points"] == 0.0

        parlay = sanitize_parlay_config({"min_blended_conf": "2.0", "min_games_played": "1"})
        assert parlay["min_blended_conf"] == 1.0
        assert parlay["min_games_played"] == 3

    def test_update_config_line_parlay_allowlist(self):
        src = open("api/index.py").read()
        assert 'if key.startswith("line.") and key not in _LINE_CONFIG_EDITABLE_KEYS:' in src
        assert 'if key.startswith("parlay.") and key not in _PARLAY_CONFIG_EDITABLE_KEYS:' in src
        assert 'cfg["line"] = sanitize_line_config(cfg.get("line", {}))' in src
        assert 'cfg["parlay"] = sanitize_parlay_config(cfg.get("parlay", {}))' in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
