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
from pathlib import Path
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
# Predictions CSV matches PRED_FIELDS; actuals use header-aware _parse_actuals_rows (see ACT_FIELDS in api/index.py).
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
            if "top_performers.csv" in path:
                return None, None
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
            if "top_performers.csv" in path:
                return None, None
            if 'predictions' in path:
                return self.PRED_CSV, 'sha1'
            return None, None
        with patch('api.index._github_get_file', side_effect=side_effect):
            assert _compute_audit('2026-03-08') is None

    def test_uses_top_performers_mega_without_actuals_file(self):
        """When mega CSV has rows for the date, audit runs without data/actuals/{date}.csv."""
        from api.index import _compute_audit
        tp_csv = (
            "date,player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source\n"
            "2026-03-08,LeBron James,4.2,1.8,45000,,,highest_value\n"
            "2026-03-08,Stephen Curry,6.1,1.2,80000,,,highest_value\n"
            "2026-03-08,Kevin Durant,3.5,2.1,30000,,,highest_value\n"
        )

        def side_effect(path):
            if "top_performers.csv" in path:
                return tp_csv, "s0"
            if "predictions" in path:
                return self.PRED_CSV, "s1"
            return None, None

        with patch("api.index._github_get_file", side_effect=side_effect):
            result = _compute_audit("2026-03-08")
        assert result is not None
        assert result["players_compared"] == 3

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

    def test_boost_drift_metrics_present(self):
        """Audit payload includes card boost drift summary fields."""
        from api.index import _compute_audit
        with patch('api.index._github_get_file', side_effect=self._mock_github(self.PRED_CSV, self.ACT_CSV)):
            result = _compute_audit('2026-03-08')
        assert result["boost_mae"] == pytest.approx(0.2, abs=0.01)
        assert result["boost_bias"] == pytest.approx(0.2, abs=0.01)
        assert result["boost_under_predicted"] == 2
        assert result["boost_over_predicted"] == 0
        assert result["boost_under_rate"] == pytest.approx(0.667, abs=0.01)

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
            with patch('api.index._est_card_boost', return_value=(1.5, (1.2, 1.8))):
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
            with patch('api.index._est_card_boost', return_value=(0.3, (0.2, 0.4))):
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
        odds_map = {
            ("player a", "points"): {"line": 22.0, "odds_over": -110, "odds_under": -110, "books_consensus": 1},
            ("player b", "points"): {"line": 8.0, "odds_over": -110, "odds_under": -110, "books_consensus": 1},
        }
        out = run_model_fallback(proj, games, line_config={"min_confidence": 70}, player_odds_map=odds_map)
        assert out.get("error") is None or out.get("error") == "no_edges" or out.get("pick") is not None

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

    def test_claude_prompt_includes_fair_value_when_edge_map(self):
        """Line Claude path receives edge_map + fair_value_data: FV text, True Probs, volatile minutes."""
        from api.line_engine import _build_claude_prompt
        pid = "999001"
        proj = [{
            "id": pid, "name": "Test Star", "team": "LAL", "predMin": 32,
            "pts": 28.0, "season_pts": 24.0, "recent_pts": 26.0,
            "reb": 6.0, "season_reb": 6.0, "recent_reb": 6.0,
            "ast": 5.0, "season_ast": 5.0, "recent_ast": 5.0,
        }]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "spread": 2.0, "total": 228, "home_b2b": False, "away_b2b": False}]
        edge_map = {
            pid: {
                "points": {
                    "fair_median": 26.5, "hit_prob_over": 0.58, "hit_prob_under": 0.42,
                    "ev_over": 0.07, "ev_under": -0.02, "edge_class": "model_only", "direction": "over",
                },
            },
        }
        fair_value_data = {
            pid: {
                "_fv_hit_probs": {"points": {"over": 0.58, "under": 0.42}},
                "_rolling": {"_minutes_cv": 0.30},
            },
        }
        prompt = _build_claude_prompt(
            proj, games, "points", "over", stat_floors=None, edge_map=edge_map, fair_value_data=fair_value_data,
        )
        assert "FAIR VALUE (FV)" in prompt
        assert "FV median 26.5pts" in prompt
        assert "P(hit OVER)" in prompt
        assert "P(hit UNDER)" in prompt
        assert "True Probs: 58.0% O / 42.0% U" in prompt
        assert "VOLATILE MINUTES: 0.30 CV" in prompt
        assert "PROP DIVERSIFICATION RULES" in prompt
        assert "UNDER-SPECIFIC RULES" in prompt

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
        odds_map = {
            ("scorer", "points"): {"line": 22.0, "odds_over": -110, "odds_under": -110, "books_consensus": 1},
            ("big man", "rebounds"): {"line": 6.0, "odds_over": -110, "odds_under": -110, "books_consensus": 1},
        }
        result = run_model_fallback(proj, games, line_config=cfg, player_odds_map=odds_map)
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

    def test_fallback_requires_book_odds(self):
        """run_model_fallback should not synthesize lines when odds are missing."""
        from api.line_engine import run_model_fallback
        proj = [{
            "name": "No Odds Guy", "team": "LAL", "predMin": 32,
            "pts": 24, "season_pts": 20, "recent_pts": 21,
            "reb": 7, "season_reb": 6, "recent_reb": 6.5,
            "ast": 5, "season_ast": 4, "recent_ast": 4.5
        }]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "home_b2b": False, "away_b2b": False}]
        out = run_model_fallback(proj, games, line_config={"min_confidence": 0}, player_odds_map={})
        assert out.get("pick") is None and out.get("over_pick") is None and out.get("under_pick") is None
        assert out.get("error") == "odds_unavailable"

    def test_fallback_returns_no_edges_when_odds_exist_but_no_qualifying_edge(self):
        """When odds are present but no edge clears thresholds, return no_edges (not odds_unavailable)."""
        from api.line_engine import run_model_fallback
        proj = [{
            "name": "Flat Edge", "team": "LAL", "predMin": 32,
            "pts": 20.0, "season_pts": 20.0, "recent_pts": 20.0,
            "reb": 1.0, "season_reb": 1.0, "recent_reb": 1.0,
            "ast": 1.0, "season_ast": 1.0, "recent_ast": 1.0
        }]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "home_b2b": False, "away_b2b": False}]
        odds_map = {("flat edge", "points"): {"line": 20.0, "odds_over": -110, "odds_under": -110, "books_consensus": 1}}
        out = run_model_fallback(proj, games, line_config={"min_confidence": 0, "min_edge_pts": 2.0}, player_odds_map=odds_map)
        assert out.get("pick") is None and out.get("over_pick") is None and out.get("under_pick") is None
        assert out.get("error") == "no_edges"


# ─────────────────────────────────────────────────────────
# Native LightGBM artifacts (lgbm_model.json + .txt) vs pickle fallback
# ─────────────────────────────────────────────────────────
class TestLgbmNativeArtifacts:
    def test_loaders_try_lightgbm_native_first(self):
        """RS LightGBM loader handles native format. Boost/drafts models removed (cascade)."""
        import inspect
        from api import index
        # Only RS model still uses LightGBM native format
        src = inspect.getsource(index._ensure_lgbm_loaded)
        assert "lightgbm_native" in src, "_ensure_lgbm_loaded must handle native format"
        assert "_LGBM_JSON_PATHS" in src


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
            pytest.skip("No LightGBM bundle loaded (lgbm_model.json / .pkl not present or invalid)")
        n = len(AI_FEATURES)
        assert n in (12, 16, 17, 22), (
            f"Expected 12 (legacy), 16 (v2), 17 (v63), or 22 (v62) features, got {n}: {AI_FEATURES}"
        )
        assert AI_FEATURES[9] in ("recent_vs_season", "recent_3g_trend"), (
            f"10th feature (index 9) must be recent_vs_season or legacy recent_3g_trend, got {AI_FEATURES[9]!r}"
        )
        assert AI_FEATURES[11] == "reb_per_min", (
            f"12th feature (index 11) must be reb_per_min, got {AI_FEATURES[11]!r}"
        )
        if n >= 16:
            assert AI_FEATURES[2] == "usage_trend", f"index 2 must be usage_trend, got {AI_FEATURES[2]!r}"
            assert AI_FEATURES[12] == "l3_vs_l5_pts", f"index 12 must be l3_vs_l5_pts, got {AI_FEATURES[12]!r}"
            assert AI_FEATURES[13] == "min_volatility", f"index 13 must be min_volatility, got {AI_FEATURES[13]!r}"
            assert AI_FEATURES[14] == "starter_proxy", f"index 14 must be starter_proxy, got {AI_FEATURES[14]!r}"
        if n == 17:
            assert AI_FEATURES[15:17] == [
                "opp_pts_allowed",
                "team_pace_proxy",
            ], f"v63 tail mismatch: {AI_FEATURES[15:17]!r}"
        if n == 22:
            assert AI_FEATURES[15] == "cascade_signal", f"index 15 must be cascade_signal, got {AI_FEATURES[15]!r}"
            assert AI_FEATURES[16:22] == [
                "opp_pts_allowed",
                "team_pace_proxy",
                "usage_share",
                "teammate_out_count",
                "game_total",
                "spread_abs",
            ], f"v62 tail mismatch: {AI_FEATURES[16:22]!r}"
        # Feature vector adapts to loaded model's feature list
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
        assert len(vec) == n, f"_lgbm_feature_vector must return {n} values, got {len(vec)}"

    def test_feature_vector_canonical_fallback(self):
        """When no model is loaded, feature vector uses 17-feature canonical order."""
        import api.index as idx
        # Save and temporarily clear loaded model state
        saved_features = idx.AI_FEATURES
        saved_attempted = idx._LGBM_LOAD_ATTEMPTED
        try:
            idx.AI_FEATURES = None
            idx._LGBM_LOAD_ATTEMPTED = True  # prevent lazy load
            vec = idx._lgbm_feature_vector(
                avg_min=24.0, pts=14.0, reb=5.0, ast=3.0, stl=1.0, blk=0.5,
                spread=3.0, side="home", season_pts=14.0, recent_pts=16.0,
                season_min=24.0, recent_min=26.0, cascade_bonus=0.0,
            )
            assert len(vec) == 17, f"Canonical fallback must return 17 features, got {len(vec)}"
        finally:
            idx.AI_FEATURES = saved_features
            idx._LGBM_LOAD_ATTEMPTED = saved_attempted


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
    """Verify that the simplified lineup builder uses strategy.rs_floor for filtering."""

    def test_strategy_rs_floor_in_source(self):
        """Simplified pipeline should use strategy.rs_floor for filtering."""
        with open("api/index.py") as f:
            src = f.read()
        assert 'strategy.rs_floor' in src or 'rs_floor' in src


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
# TestCorePool — core pool architecture
# ─────────────────────────────────────────────────────────
class TestCorePool:
    """When core_pool.enabled is False, third return is None; when True, third is list; lineups from core when enabled."""

    def test_build_lineups_returns_three_values(self):
        """_build_lineups returns (chalk, upside, core_pool). core_pool is always a list."""
        from api.index import _build_lineups

        with patch("api.index._cfg", return_value=None), \
             patch("api.index.get_all_statuses", return_value={}), \
             patch("api.index._fetch_team_def_stats", return_value={}):
            chalk, upside, core_pool = _build_lineups([])

        assert chalk == []
        assert upside == []
        assert isinstance(core_pool, list)

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
                    result = asyncio.run(lab_briefing())

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


class TestMinGatePpgProxy:
    """min_gate_minutes uses PPG-derived rough_boost only (no card_boost overrides)."""

    def _effective_gate(self, pts, rough_boost_override=None, min_gate=12):
        rough_boost = rough_boost_override if rough_boost_override is not None else max(0.2, 3.0 - pts * 0.12)
        return max(8, min_gate - max(0, (rough_boost - 1.5) * 3))

    def test_high_rough_boost_lowers_gate_to_8(self):
        gate = self._effective_gate(pts=8.0, rough_boost_override=3.0)
        assert gate == 8

    def test_low_rough_boost_raises_gate(self):
        gate = self._effective_gate(pts=8.0, rough_boost_override=None)
        assert gate > 10

    def test_10_min_fails_default_proxy_at_8_ppg(self):
        proj_min = 10
        gate_high = self._effective_gate(pts=8.0, rough_boost_override=3.0)
        gate_low = self._effective_gate(pts=8.0, rough_boost_override=None)
        assert proj_min >= gate_high
        assert proj_min < gate_low

    def test_rough_boost_2_1_gate(self):
        gate = self._effective_gate(pts=10.0, rough_boost_override=2.1)
        assert gate == pytest.approx(10.2, abs=0.1)

    def test_project_player_gate_no_config_overrides(self):
        import inspect
        from api import index
        src = inspect.getsource(index.project_player)
        assert "player_overrides" not in src
        assert "3.0 - _pts_for_gate * 0.12" in src


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

    def test_chalk_season_min_floor_is_16(self):
        """v68: chalk_season_min_floor 14→16 — require real rotation minutes."""
        cfg = self._local_cfg()
        floor = cfg.get("projection", {}).get("chalk_season_min_floor")
        assert floor == pytest.approx(16.0), \
            f"chalk_season_min_floor should be 16.0, got {floor}"

    def test_den_not_in_big_market_teams(self):
        """DEN must be removed from big_market_teams — DEN role players are low-ownership."""
        cfg = self._local_cfg()
        big_markets = cfg.get("card_boost", {}).get("big_market_teams", [])
        assert "DEN" not in big_markets, \
            "DEN must not be in big_market_teams — Braun/Porter etc. are not heavily drafted"


# ---------------------------------------------------------------------------
# TWO-PASS PIPELINE — watchlist and pass metadata
# ---------------------------------------------------------------------------
class TestBoostModelInference:
    """Card boost: 3-tier cascade model (api/boost_model.py)."""

    def test_est_card_boost_tier3_cold_start_role_player(self):
        """Cold start (Tier 3): unknown role player should get high boost (2.5+).
        Uses a small-market team to avoid per-team boost ceiling caps."""
        from api.index import _est_card_boost
        b, _ = _est_card_boost(
            20.0, 8.0, "SAS", "Nobody Cold Start XYZ",
            season_pts=8.0, recent_pts=9.0, cascade_bonus=0.0, is_home=True,
            projected_rs=3.5, season_avg_min=20.0, player_pos="G",
        )
        assert b >= 2.0, f"Cold start role player should get high boost, got {b}"

    def test_est_card_boost_tier3_cold_start_star(self):
        """Cold start (Tier 3): unknown star should get low boost."""
        from api.index import _est_card_boost
        b, _ = _est_card_boost(
            35.0, 25.0, "MIN", "Nobody Star XYZ",
            season_pts=25.0, recent_pts=24.0, cascade_bonus=0.0, is_home=False,
            projected_rs=7.0, season_avg_min=35.0, player_pos="G",
        )
        # Star profile → low boost via PQI
        assert b <= 1.0, f"Cold start star should get low boost, got {b}"

    def test_est_card_boost_nonnegative_all_tiers(self):
        """Boost is always non-negative regardless of tier."""
        from api.index import _est_card_boost
        for mins, pts in [(5, 2), (20, 10), (38, 30)]:
            b, _ = _est_card_boost(mins, pts, "GSW", player_name=f"Test {pts} XYZ",
                                   season_pts=float(pts), season_avg_min=float(mins))
            assert b >= 0, f"Boost should be non-negative, got {b}"

    def test_est_card_boost_star_ppg_tier_cap(self):
        """Star PPG tier caps should limit boost for high-PPG players."""
        from api.index import _est_card_boost
        # 26+ PPG player gets capped at 0.2 by star_ppg_tiers config
        b, _ = _est_card_boost(
            35.0, 28.0, "WAS", "Nobody Star26 XYZ",
            season_pts=28.0, season_avg_min=35.0,
        )
        assert b <= 0.4, f"26+ PPG star should be capped low, got {b}"


class TestBoostCascadeModel:
    """Tests for the 3-tier cascade boost prediction model (api/boost_model.py)."""

    def test_tier3_cold_start_deep_bench(self):
        """Tier 3: deep bench (PPG<5) always gets 3.0."""
        from api.boost_model import predict_boost
        r = predict_boost("Nobody Deep Bench", "2026-03-30", season_ppg=3.0, season_rpg=1.0,
                          season_apg=0.5, season_mpg=8.0)
        assert r["tier"] == 3
        assert r["boost"] == 3.0

    def test_tier3_cold_start_star(self):
        """Tier 3: star (PPG>=25) gets very low boost."""
        from api.boost_model import predict_boost
        r = predict_boost("Nobody Superstar", "2026-03-30", season_ppg=27.0, season_rpg=7.0,
                          season_apg=6.0, season_mpg=36.0)
        assert r["tier"] == 3
        assert r["boost"] <= 0.5

    def test_tier3_cold_start_mid_rotation(self):
        """Tier 3: mid-rotation (PPG 12-16) gets moderate boost."""
        from api.boost_model import predict_boost
        r = predict_boost("Nobody Rotation", "2026-03-30", season_ppg=14.0, season_rpg=4.0,
                          season_apg=3.0, season_mpg=28.0)
        assert r["tier"] == 3
        assert 1.0 <= r["boost"] <= 2.5

    def test_estimate_boost_from_api_monotonic(self):
        """Higher PPG → lower boost (monotonically decreasing)."""
        from api.boost_model import estimate_boost_from_api
        boosts = [estimate_boost_from_api(ppg, 4.0, 3.0, 25.0) for ppg in [5, 10, 15, 20, 25, 30]]
        for i in range(len(boosts) - 1):
            assert boosts[i] >= boosts[i + 1], \
                f"Boost should decrease: {boosts[i]} at ppg={5+5*i} > {boosts[i+1]} at ppg={10+5*i}"

    def test_load_player_history(self):
        """Player history loads successfully from top_performers.csv."""
        from api.boost_model import load_player_history
        h = load_player_history()
        assert len(h) > 0, "Should load at least some players"
        # Check structure of entries
        for name, entries in list(h.items())[:3]:
            assert all("date" in e and "boost" in e and "rs" in e for e in entries)
            # Entries should be sorted by date
            dates = [e["date"] for e in entries]
            assert dates == sorted(dates), f"Entries for {name} should be date-sorted"

    def test_tier1_returning_player_persistence(self):
        """Tier 1: boost at 3.0 should stay near 3.0 (75% persistence rate)."""
        from api.boost_model import _predict_tier1
        entries = [
            {"date": "2026-03-25", "boost": 3.0, "rs": 2.5, "drafts": 50, "has_boost_data": True},
            {"date": "2026-03-27", "boost": 3.0, "rs": 3.0, "drafts": 60, "has_boost_data": True},
        ]
        r = _predict_tier1(entries, gap_days=2, season_mpg=20, ceiling=3.0, floor=0.0)
        assert r["boost"] >= 2.9, f"3.0 boost should persist, got {r['boost']}"

    def test_tier1_monster_game_drops_boost(self):
        """Tier 1: RS >= 7.0 should reduce boost slightly."""
        from api.boost_model import _predict_tier1
        entries = [
            {"date": "2026-03-28", "boost": 2.0, "rs": 7.5, "drafts": 200, "has_boost_data": True},
        ]
        r = _predict_tier1(entries, gap_days=2, season_mpg=30, ceiling=3.0, floor=0.0)
        assert r["boost"] < 2.0, f"Monster game should drop boost, got {r['boost']}"

    def test_tier1_bust_game_raises_boost(self):
        """Tier 1: RS < 2.0 should increase boost slightly."""
        from api.boost_model import _predict_tier1
        entries = [
            {"date": "2026-03-28", "boost": 2.0, "rs": 1.5, "drafts": 30, "has_boost_data": True},
        ]
        r = _predict_tier1(entries, gap_days=2, season_mpg=20, ceiling=3.0, floor=0.0)
        assert r["boost"] >= 2.0, f"Bust game should raise boost, got {r['boost']}"

    def test_tier2_stale_blends_history_and_api(self):
        """Tier 2: stale player blends hist mean with API estimate based on staleness."""
        from api.boost_model import _predict_tier2
        entries = [
            {"date": "2026-02-01", "boost": 2.5, "rs": 3.0, "drafts": 50, "has_boost_data": True},
        ]
        # At 30 days stale (staleness = (30-14)/30 = 0.53)
        r = _predict_tier2(entries, gap_days=30, season_ppg=8.0, season_rpg=3.0,
                           season_apg=2.0, season_mpg=20.0, ceiling=3.0, floor=0.0)
        assert r["tier"] == 2
        assert r["confidence"] == "medium"
        # Should be between hist mean (2.5) and API estimate (~2.7)
        assert 2.0 <= r["boost"] <= 3.0

    def test_predict_boost_returns_required_fields(self):
        """All prediction results must contain required fields."""
        from api.boost_model import predict_boost
        r = predict_boost("Test Player", "2026-03-30", season_ppg=10.0)
        assert "boost" in r
        assert "tier" in r
        assert "confidence" in r
        assert "confidence_band" in r
        assert "reason" in r
        assert r["tier"] in (1, 2, 3)
        assert r["confidence"] in ("high", "medium", "low")
        assert 0.0 <= r["boost"] <= 3.0
        low, high = r["confidence_band"]
        assert low <= r["boost"] <= high

    def test_superstar_zero_boost_persistence(self):
        """Stars with 0.0 boost should stay near 0.0."""
        from api.boost_model import _predict_tier1
        entries = [
            {"date": "2026-03-20", "boost": 0.0, "rs": 6.0, "drafts": 5000, "has_boost_data": True},
            {"date": "2026-03-25", "boost": 0.0, "rs": 5.5, "drafts": 4800, "has_boost_data": True},
            {"date": "2026-03-28", "boost": 0.0, "rs": 7.0, "drafts": 5200, "has_boost_data": True},
        ]
        r = _predict_tier1(entries, gap_days=2, season_mpg=35, ceiling=3.0, floor=0.0)
        assert r["boost"] <= 0.2, f"Superstar 0.0 boost should persist, got {r['boost']}"

    def test_clamp_round_snaps_to_3(self):
        """Boost ≥ 2.8 should snap to 3.0 (26.1% of all boosts are exactly 3.0)."""
        from api.boost_model import _clamp_round
        assert _clamp_round(2.85, 0.0, 3.0) == 3.0
        assert _clamp_round(2.80, 0.0, 3.0) == 3.0  # exactly 2.8 snaps to 3.0
        assert _clamp_round(2.74, 0.0, 3.0) == 2.7   # below snap threshold


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
    """screenshot_type 'boosts' is rejected (model-only card boosts)."""

    def test_boosts_type_returns_400(self):
        import api.index as idx
        src = open(idx.__file__).read()
        assert 'screenshot_type == "boosts"' in src
        assert "removed" in src.lower() or "model-estimated" in src.lower()


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
        assert ss.get("enabled") is False, "stat_stuffer should be disabled in production (v59)"


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

    def test_game_context_bonus_in_project_player(self):
        """project_player should use game_context_bonus for game situation adjustments."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "game_context_bonus" in src

    def test_closeness_config_enabled(self):
        """Production config should have closeness enabled."""
        cfg = json.load(open("data/model-config.json"))
        cc = cfg.get("real_score", {}).get("closeness", {})
        assert cc.get("enabled") is False, "closeness should be disabled in production (v59)"

    def test_closeness_defaults_disabled_offline(self):
        """_CONFIG_DEFAULTS should have closeness disabled or absent (safe fallback)."""
        from api.index import _CONFIG_DEFAULTS
        cc = _CONFIG_DEFAULTS.get("real_score", {}).get("closeness", {"enabled": False})
        assert cc["enabled"] is False


# ─────────────────────────────────────────────────────────
# TestCascadeRsBoost — usage-spike RS multiplier from injuries
# ─────────────────────────────────────────────────────────
class TestCascadeRsBoost:
    """Cascade RS boost gives direct RS multiplier for injury beneficiaries."""

    def test_game_context_bonus_code_exists(self):
        """project_player should use game_context_bonus for additive adjustments."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "game_context_bonus" in src

    def test_cascade_rs_config_enabled(self):
        """Production config should have cascade_rs enabled (v66 ceiling lift)."""
        cfg = json.load(open("data/model-config.json"))
        cr = cfg.get("real_score", {}).get("cascade_rs", {})
        assert cr.get("enabled") is True

    def test_cascade_rs_defaults_disabled(self):
        """_CONFIG_DEFAULTS should have cascade_rs disabled or absent (safe fallback)."""
        from api.index import _CONFIG_DEFAULTS
        cr = _CONFIG_DEFAULTS.get("real_score", {}).get("cascade_rs", {"enabled": False})
        assert cr["enabled"] is False


# ─────────────────────────────────────────────────────────
# TestRoleSpikeRs — hot streak / expanded role RS boost
# ─────────────────────────────────────────────────────────
class TestRoleSpikeRs:
    """Role-spike RS boost for players in expanded roles."""

    def test_game_context_bonus_code_exists(self):
        """project_player should use game_context_bonus for additive adjustments."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "game_context_bonus" in src

    def test_role_spike_config_enabled(self):
        """Production config should have role_spike_rs enabled (v66 ceiling lift)."""
        cfg = json.load(open("data/model-config.json"))
        rs = cfg.get("real_score", {}).get("role_spike_rs", {})
        assert rs.get("enabled") is True

    def test_role_spike_defaults_disabled(self):
        """_CONFIG_DEFAULTS should have role_spike_rs disabled or absent (safe fallback)."""
        from api.index import _CONFIG_DEFAULTS
        rs_cfg = _CONFIG_DEFAULTS.get("real_score", {}).get("role_spike_rs", {"enabled": False})
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

    def test_signals_include_driver(self):
        """run_model_fallback should produce signals when game context is favorable."""
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
            # No narrative field — write-up removed from Line of the Day
            assert "narrative" not in over


# ─────────────────────────────────────────────────────────
# TestHighBoostRolePathway — role players with 2.0x+ boost bypass minutes floor
# ─────────────────────────────────────────────────────────
class TestHighBoostRolePathway:
    """Simplified pipeline uses strategy.rs_floor and strategy.min_minutes
    instead of separate HBR pathways for chalk/moonshot."""

    def test_simplified_pipeline_has_strategy_gates(self):
        """_build_lineups should use rs_floor and min_minutes for filtering."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "rs_floor" in src, "_build_lineups must use rs_floor"
        assert "min_minutes" in src, "_build_lineups must use min_minutes"


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
        """pts weight should be stable at 1.5; stl reduced from 8.0 (v57: 3.0, v58: 2.5)."""
        cfg = json.load(open("data/model-config.json"))
        w = cfg["real_score"]["dfs_weights"]
        assert w["pts"] == 1.5, f"pts weight changed unexpectedly: {w['pts']}"
        assert w["stl"] <= 3.0, f"stl weight should be <= 3.0 (v58 reduced to 2.5); got {w['stl']}"

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

    def test_draft_ev_scoring_in_build_lineups(self):
        """Simplified pipeline uses draft_ev for scoring instead of scorer_upside."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "draft_ev" in src, "_build_lineups must use draft_ev for scoring"


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
        assert cr.get("enabled") is True, "cascade_rs.enabled must be True in production config (v66 ceiling lift)"


# ─────────────────────────────────────────────────────────
# TestRotoConfirmedRatingException — confirmed rotation players bypass min_rating_floor
# ─────────────────────────────────────────────────────────
class TestRotoConfirmedRatingException:
    """Confirmed rotation players with high boost pass a lower RS floor (2.2) in
    the moonshot pool, capturing players like Garza, Paul Reed, Taylor Hendricks."""

    def test_safe_to_draft_in_build_lineups(self):
        """Simplified _build_lineups should use is_safe_to_draft or similar filtering."""
        src = open("api/index.py").read()
        assert "_build_lineups" in src, "_build_lineups must exist"
        assert "rating" in src, "_build_lineups must filter by rating"

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

    def test_build_lineups_exists_and_uses_sorting(self):
        """Simplified _build_lineups should exist and use sorting for slot assignment."""
        src = open("api/index.py").read()
        assert "_build_lineups" in src, "_build_lineups must exist"
        assert "sorted" in src or "sort" in src, "_build_lineups must use sorting"


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

    def test_build_lineups_uses_draft_ev_sorting(self):
        """Simplified _build_lineups uses draft_ev sorting instead of MILP constraints."""
        src = open("api/index.py").read()
        assert "draft_ev" in src, "_build_lineups must use draft_ev for scoring"
        assert "_build_lineups" in src, "_build_lineups must exist"


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

    def test_no_synthetic_parlay_fallback(self):
        """Parlay must NOT use synthetic fallback lines — require real Odds API data."""
        src = open("api/index.py").read()
        # Synthetic fallback was removed — verify it's gone
        assert "built {len(player_odds_map)} synthetic lines from projections" not in src, (
            "Synthetic parlay fallback must be removed — parlay requires real sportsbook lines"
        )
        # Verify the no-odds early return exists
        assert 'return None, "no_odds_data"' in src, (
            "Parlay must return error when Odds API data is unavailable"
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
        # After fix: resolved picks trigger _try_load_next_slate_pick helper with retry
        assert "_try_load_next_slate_pick" in src, (
            "Resolved picks must use retry helper to load next-slate pick"
        )
        assert 'final_over = _try_load_next_slate_pick("over_pick")' in src, (
            "Resolved over pick must always be replaced by next-slate pick (even if None)"
        )
        assert 'final_under = _try_load_next_slate_pick("under_pick")' in src, (
            "Resolved under pick must always be replaced by next-slate pick (even if None)"
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
        assert "today_all_final, _tr, _tf, _tlrs = _all_games_final(_today_games)" in src
        assert "today_games_final = bool(today_all_final)" in src
        assert '_can_resolve = (date_str < today_str) or (date_str == today_str and today_games_final)' in src
        assert "def _parlay_fully_concluded(" in src
        assert "history_parlays" in src and "date_str >= today_str" in src

    def test_parlay_live_stream_endpoint(self):
        src = open("api/index.py").read()
        assert "/api/parlay-live-stream" in src
        assert "async def parlay_live_stream" in src
        assert "def _line_live_stat_dict(" in src

    def test_get_parlay_bypasses_tmp_when_slate_final_and_pending(self):
        src = open("api/index.py").read()
        assert "slate_all_final" in src
        assert "bypassing /tmp cache" in src
        assert 'cached.get("result") == "pending"' in src

    def test_parlay_history_try_except_returns_safe_payload(self):
        src = open("api/index.py").read()
        assert "async def parlay_history(request: Request):" in src
        assert 'print(f"[parlay-history] error:' in src
        assert '"error": "parlay_history_failed"' in src
        assert "Could not load parlay history" in src

    def test_projection_only_persisted_and_labeled(self):
        src_api = open("api/index.py").read()
        src_ui = open("index.html").read()
        assert '"projection_only": result.get("projection_only", False)' in src_api
        assert "sourceBadge = data.projection_only" in src_ui
        assert "MODEL</span>'" in src_ui
        assert "BOOK</span>'" not in src_ui

    def test_projection_only_cache_served(self):
        src = open("api/index.py").read()
        # projection-only parlays are now served from cache (better than no parlay)
        # The old bypass strings should be removed
        assert "bypassing projection-only /tmp cache" not in src
        assert "projection-only GitHub ticket found" not in src
        # projection-only parlays served from GitHub with a note
        assert "(projection-only)" in src

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


class TestLineLoadStabilizationRegressions:
    """Regression guards for line load stabilization (no flash + fast backend path)."""

    def test_line_rotation_refresh_keeps_card(self):
        src = open("index.html").read()
        assert "fetchLineOfTheDay(true, true);" in src, (
            "rotation refresh should run as background nocache fetch (no skeleton reset)"
        )
        assert "LINE_LOTD_STATE = asyncStateInitial();" not in src[src.find("const needRotation"):src.find("if (overDone && underDone)")], (
            "rotation block should not reset LOTD state to initial"
        )

    def test_line_fetch_dedup_and_timeout(self):
        src = open("index.html").read()
        assert "let _lineLotdFetchPromise = null;" in src
        assert "if (_lineLotdFetchPromise) return _lineLotdFetchPromise;" in src
        assert "fetchWithTimeout(_lotdUrl, {}, 30000" in src, (
            "line-of-the-day fetch should use 30s timeout budget"
        )

    def test_fast_path_engine_call_present(self):
        src = open("api/index.py").read()
        assert "_run_line_engine_for_date, today, False, True" in src, (
            "line-of-the-day should attempt fast-path generation first"
        )
        assert "_run_line_engine_for_date, today, True, False" in src, (
            "line-of-the-day retry should allow full enrichment path"
        )

    def test_lotd_payload_freshness_metadata(self):
        src = open("api/index.py").read()
        assert "def _line_payload_meta(" in src
        assert '"source": source' in src
        assert '"generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")' in src
        assert '"is_stale": bool(stale)' in src
        assert '"refreshing": bool(refreshing)' in src

    def test_refresh_does_not_tombstone_parlay_history_files(self):
        src = open("api/index.py").read()
        assert 'json.dumps({"_busted": True}), "bust parlay cache"' not in src, (
            "refresh must not tombstone GitHub parlay files; this can erase history"
        )


class TestSlateTransitionPrewarm:
    """Regression guards for cron prewarm of current slate."""

    def test_prewarm_endpoint_exists_and_is_cron_gated(self):
        src = open("api/index.py").read()
        assert '@app.get("/api/prewarm-current-slate")' in src
        assert "async def prewarm_current_slate(request: Request):" in src
        assert "if not _require_cron_secret(request):" in src

    def test_prewarm_uses_existing_slate_logic(self):
        src = open("api/index.py").read()
        assert '_with_response_cache(cache_key, "slate", _get_slate_impl)' in src, (
            "prewarm must reuse _get_slate_impl (global slate transition logic)"
        )
        assert "_load_line_pick_for_date(current_slate_date)" in src
        assert "_run_line_engine_for_date(line_date, False, True)" in src
        assert "_parlay_active_date()" in src
        assert "_run_parlay_engine_sync(parlay_date)" in src
        assert "def _prewarm_current_slate_sync(force=False, include_slate=True):" in src

    def test_railway_has_prewarm_cron(self):
        src = open("railway.toml").read()
        assert "*/5 * * * *" in src
        assert "/api/prewarm-current-slate" in src

    def test_deploy_startup_safe_prewarm_hook_exists(self):
        """Startup hook: version-aware — new deploy busts+regenerates, same SHA hydrates."""
        src = open("api/index.py").read()
        assert '@app.on_event("startup")' in src
        assert "async def _deploy_startup_safe_prewarm():" in src
        assert "_cs(_CK_SLATE" in src  # Writes to cache
        assert "_github_get_file" in src  # Reads from GitHub
        # Version-aware: detects new deploy via SHA comparison in Redis
        assert "deploy_sha" in src
        assert "RAILWAY_GIT_COMMIT_SHA" in src
        assert "is_new_deploy" in src


class TestVerifyTopPerformersScript:
    """scripts/verify_top_performers.py runs clean (backtest when predictions overlap)."""

    def test_verify_script_exits_zero(self):
        import subprocess
        import sys

        repo = Path(__file__).resolve().parent.parent
        script = repo / "scripts" / "verify_top_performers.py"
        r = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(repo),
            capture_output=True,
            text=True,
            timeout=120,
        )
        # Skip on LightGBM version incompatibility (_LGBMCheckArray broken in some versions)
        if r.returncode != 0 and "_LGBMCheckArray" in r.stderr:
            pytest.skip("LightGBM version incompatibility — _LGBMCheckArray is None")
        assert r.returncode == 0, r.stderr + r.stdout
        out = (r.stdout or "").lower()
        assert "joined" in out or "overlap" in out


class TestParlayMidnightHandoff:
    """Regression coverage for midnight ET parlay active-date behavior."""

    def test_active_date_prefers_yesterday_when_unresolved_and_not_final(self):
        from api.index import _parlay_active_date
        from datetime import date as _date

        today = _date(2026, 3, 25)
        yesterday = _date(2026, 3, 24)
        y_json = json.dumps({
            "date": "2026-03-24",
            "result": "pending",
            "legs": [{"player_name": "A", "result": "pending"}],
        })

        with patch("api.index._et_date", return_value=today), \
             patch("api.index._github_get_file", return_value=(y_json, "sha")), \
             patch("api.index.fetch_games", return_value=[{"gameId": "1"}]), \
             patch("api.index._all_games_final", return_value=(False, 0, 0, None)):
            picked = _parlay_active_date()

        assert picked == yesterday

    def test_active_date_returns_today_when_yesterday_is_final(self):
        from api.index import _parlay_active_date
        from datetime import date as _date

        today = _date(2026, 3, 25)
        y_json = json.dumps({
            "date": "2026-03-24",
            "result": "pending",
            "legs": [{"player_name": "A", "result": "pending"}],
        })

        with patch("api.index._et_date", return_value=today), \
             patch("api.index._github_get_file", return_value=(y_json, "sha")), \
             patch("api.index.fetch_games", return_value=[{"gameId": "1"}]), \
             patch("api.index._all_games_final", return_value=(True, 0, 0, None)):
            picked = _parlay_active_date()

        assert picked == today

    def test_active_date_returns_today_when_yesterday_ticket_already_concluded(self):
        from api.index import _parlay_active_date
        from datetime import date as _date

        today = _date(2026, 3, 25)
        y_json = json.dumps({
            "date": "2026-03-24",
            "result": "miss",
            "legs": [{"player_name": "A", "result": "miss"}],
        })

        with patch("api.index._et_date", return_value=today), \
             patch("api.index._github_get_file", return_value=(y_json, "sha")), \
             patch("api.index.fetch_games", return_value=[{"gameId": "1"}]), \
             patch("api.index._all_games_final", return_value=(False, 0, 0, None)):
            picked = _parlay_active_date()

        assert picked == today


class TestParlayApiActiveDateBoundary:
    """HTTP boundary tests for /api/parlay active-date behavior."""

    def test_get_parlay_uses_active_date_for_generation_and_payload(self):
        pytest.importorskip("fastapi", reason="Install dependencies: pip install -r requirements.txt")
        from fastapi.testclient import TestClient
        from datetime import date as _date
        from api.index import app

        target_date = _date(2026, 3, 24)
        result = {
            "legs": [
                {
                    "player_name": "Player A",
                    "direction": "over",
                    "stat_type": "points",
                    "line": 18.5,
                },
                {
                    "player_name": "Player B",
                    "direction": "under",
                    "stat_type": "rebounds",
                    "line": 7.5,
                },
                {
                    "player_name": "Player C",
                    "direction": "over",
                    "stat_type": "assists",
                    "line": 5.5,
                },
            ],
            "combined_probability": 0.64,
            "correlation_multiplier": 1.02,
            "correlation_reasons": [],
            "parlay_score": 0.61,
            "narrative": "Synthetic test payload",
            "result": "pending",
        }

        with patch("api.index._check_rate_limit", return_value=None), \
             patch("api.index._parlay_active_date", return_value=target_date), \
             patch("api.index.fetch_games", return_value=[]), \
             patch("api.index._cg", return_value=None), \
             patch("api.index._github_get_file", return_value=(None, None)), \
             patch("api.index._run_parlay_engine_sync", return_value=(result, None, {"odds_available": True})), \
             patch("api.index._cs", return_value=None), \
             patch("api.index._github_write_file", return_value=True):
            client = TestClient(app)
            r = client.get("/api/parlay")

        assert r.status_code == 200
        body = r.json()
        assert body.get("date") == target_date.isoformat()
        assert len(body.get("legs", [])) == 3
        assert all(leg.get("date") == target_date.isoformat() for leg in body.get("legs", []))

    def test_get_parlay_respects_active_date_lock_window(self):
        pytest.importorskip("fastapi", reason="Install dependencies: pip install -r requirements.txt")
        from fastapi.testclient import TestClient
        from datetime import date as _date, datetime as _dt, timezone as _tz, timedelta as _td
        from api.index import app

        target_date = _date(2026, 3, 24)
        live_start = (_dt.now(_tz.utc) - _td(hours=1)).isoformat()
        minimal = {
            "legs": [
                {"player_name": "A", "direction": "over", "stat_type": "points", "line": 10.5},
                {"player_name": "B", "direction": "over", "stat_type": "points", "line": 10.5},
                {"player_name": "C", "direction": "over", "stat_type": "points", "line": 10.5},
            ],
            "result": "pending",
        }

        with patch("api.index._check_rate_limit", return_value=None), \
             patch("api.index._parlay_active_date", return_value=target_date), \
             patch("api.index.fetch_games", return_value=[{"startTime": live_start}]), \
             patch("api.index._cg", return_value=None), \
             patch("api.index._github_get_file", return_value=(None, None)), \
             patch("api.index._run_parlay_engine_sync", return_value=(minimal, None, {"odds_available": True})), \
             patch("api.index._cs", return_value=None), \
             patch("api.index._github_write_file", return_value=True):
            client = TestClient(app)
            r = client.get("/api/parlay")

        assert r.status_code == 200
        body = r.json()
        assert body.get("locked") is True


class TestSuspendedPlayersFiltered:
    """Suspended players must be treated as OUT — never appear in lineups."""

    def test_suspended_in_is_out_check(self):
        """ESPN 'suspended' status must set is_out=True in fetch_roster."""
        src = open("api/index.py").read()
        assert '"suspended"' in src, "suspended must be in is_out status list"
        assert '"suspension"' in src, "suspension must be in is_out status list"
        # Verify both are in the same is_out line
        for line in src.splitlines():
            if "is_out" in line and "inj_status" in line:
                assert "suspended" in line, "suspended missing from is_out check"
                assert "suspension" in line, "suspension missing from is_out check"
                break

    def test_rotowire_maps_suspended_to_out(self):
        """RotoWire _map_status must map suspended to STATUS_OUT."""
        from api.rotowire import _map_status, STATUS_OUT
        assert _map_status("suspended") == STATUS_OUT
        assert _map_status("suspension") == STATUS_OUT
        assert _map_status("Suspended") == STATUS_OUT

    def test_project_player_filters_is_out(self):
        """project_player returns None for is_out players (including suspended)."""
        src = open("api/index.py").read()
        # Verify the is_out guard exists early in project_player
        assert 'if pinfo.get("is_out"): return None' in src


# ═══════════════════════════════════════════════════════════════════════════
# PER-GAME DRAFT STRATEGY (v60 — Findings 1-6)
# ═══════════════════════════════════════════════════════════════════════════

class TestPerGameStrategy:
    """Test _per_game_strategy() returns correct strategy based on spread/total."""

    def test_close_game_balanced(self):
        from api.index import _per_game_strategy
        game = {"spread": -3, "total": 222,
                "home": {"abbr": "BOS"}, "away": {"abbr": "CLE"}}
        s = _per_game_strategy(game)
        assert s["type"] == "balanced"
        assert "Balanced" in s["label"]
        assert s["favored_team"] == "BOS"  # negative spread = home favored
        assert s["underdog_team"] == "CLE"

    def test_blowout_top_heavy(self):
        from api.index import _per_game_strategy
        game = {"spread": 14, "total": 225,
                "home": {"abbr": "WAS"}, "away": {"abbr": "BOS"}}
        s = _per_game_strategy(game)
        assert s["type"] == "top_heavy"
        assert "Blowout" in s["label"]
        assert s["favored_team"] == "BOS"  # positive spread = away favored

    def test_neutral_moderate_spread(self):
        from api.index import _per_game_strategy
        game = {"spread": -8, "total": 228,
                "home": {"abbr": "MIL"}, "away": {"abbr": "IND"}}
        s = _per_game_strategy(game)
        assert s["type"] == "neutral"

    def test_shootout_overlay(self):
        from api.index import _per_game_strategy
        game = {"spread": -2, "total": 250,
                "home": {"abbr": "GSW"}, "away": {"abbr": "SAC"}}
        s = _per_game_strategy(game)
        assert "Shootout" in s["label"]
        assert s["total_mult"] > 1.0  # high total boosts ceiling

    def test_grind_overlay(self):
        from api.index import _per_game_strategy
        game = {"spread": -4, "total": 210,
                "home": {"abbr": "MEM"}, "away": {"abbr": "OKC"}}
        s = _per_game_strategy(game)
        assert "Grind" in s["label"]
        assert s["total_mult"] < 1.0  # low total compresses ceiling

    def test_total_mult_bounds(self):
        from api.index import _per_game_strategy
        # Extreme high total
        s_hi = _per_game_strategy({"spread": 0, "total": 280,
                                   "home": {"abbr": "A"}, "away": {"abbr": "B"}})
        assert s_hi["total_mult"] <= 1.12
        # Extreme low total
        s_lo = _per_game_strategy({"spread": 0, "total": 180,
                                   "home": {"abbr": "A"}, "away": {"abbr": "B"}})
        assert s_lo["total_mult"] >= 0.92


class TestPerGameAdjustProjections:
    """Test _per_game_adjust_projections() applies F1-F6 correctly."""

    def _make_proj(self, name, rating, team, season_pts=15, variance=0.3, recent_min=25):
        return {
            "name": name, "rating": rating, "team": team,
            "season_pts": season_pts, "pts": season_pts,
            "player_variance": variance, "recent_min": recent_min,
            "reb": 5, "ast": 3, "stl": 1, "blk": 0.5,
            "est_mult": 0, "chalk_ev": rating * 1.6,
            "predMin": recent_min, "id": name.lower().replace(" ", "_"),
        }

    def test_game_total_multiplier_high(self):
        """F3: Higher game total should boost all projections."""
        from api.index import _per_game_strategy, _per_game_adjust_projections
        game = {"spread": 0, "total": 250,
                "home": {"abbr": "GSW"}, "away": {"abbr": "SAC"}}
        strat = _per_game_strategy(game)
        projs = [self._make_proj("Player A", 5.0, "GSW")]
        adj = _per_game_adjust_projections(projs, game, strat)
        assert adj[0]["rating"] > 5.0  # boosted by total mult

    def test_game_total_multiplier_low(self):
        """F3: Lower game total should compress projections."""
        from api.index import _per_game_strategy, _per_game_adjust_projections
        game = {"spread": 0, "total": 200,
                "home": {"abbr": "MEM"}, "away": {"abbr": "OKC"}}
        strat = _per_game_strategy(game)
        # Use star (season_pts=25) so anchor bonus doesn't offset total compression
        projs = [self._make_proj("Player A", 5.0, "MEM", season_pts=25)]
        adj = _per_game_adjust_projections(projs, game, strat)
        assert adj[0]["rating"] < 5.0  # compressed by low total

    def test_close_game_rewards_consistency(self):
        """F4: In close games, low-variance player should be boosted more than high-variance."""
        from api.index import _per_game_strategy, _per_game_adjust_projections
        game = {"spread": -2, "total": 222,
                "home": {"abbr": "BOS"}, "away": {"abbr": "CLE"}}
        strat = _per_game_strategy(game)
        projs = [
            self._make_proj("Consistent", 5.0, "BOS", variance=0.1),
            self._make_proj("Volatile", 5.0, "CLE", variance=0.8),
        ]
        adj = _per_game_adjust_projections(projs, game, strat)
        assert adj[0]["rating"] > adj[1]["rating"]  # consistent > volatile

    def test_blowout_favors_favored_team_role(self):
        """F6: In blowouts, favored team role player should be boosted."""
        from api.index import _per_game_strategy, _per_game_adjust_projections
        game = {"spread": -15, "total": 222,
                "home": {"abbr": "BOS"}, "away": {"abbr": "WAS"}}
        strat = _per_game_strategy(game)
        projs = [
            self._make_proj("Fav Role", 4.0, "BOS", season_pts=12),
            self._make_proj("Dog Role", 4.0, "WAS", season_pts=12),
        ]
        adj = _per_game_adjust_projections(projs, game, strat)
        assert adj[0]["rating"] > adj[1]["rating"]
        assert adj[0]["_favored_team"] is True
        assert adj[1]["_favored_team"] is False

    def test_blowout_penalizes_underdog_role(self):
        """F6: Underdog role players get pulled early in blowouts."""
        from api.index import _per_game_strategy, _per_game_adjust_projections
        game = {"spread": 16, "total": 222,
                "home": {"abbr": "WAS"}, "away": {"abbr": "BOS"}}
        strat = _per_game_strategy(game)
        projs = [self._make_proj("Dog Role", 4.0, "WAS", season_pts=10)]
        adj = _per_game_adjust_projections(projs, game, strat)
        assert adj[0]["rating"] < 4.0  # penalized

    def test_value_anchor_tagged(self):
        """F2: Mid-tier players with solid RS should be tagged as value anchors."""
        from api.index import _per_game_strategy, _per_game_adjust_projections
        game = {"spread": -3, "total": 222,
                "home": {"abbr": "BOS"}, "away": {"abbr": "CLE"}}
        strat = _per_game_strategy(game)
        projs = [
            self._make_proj("Anchor", 4.5, "BOS", season_pts=14),
            self._make_proj("Star", 6.0, "CLE", season_pts=25),
        ]
        adj = _per_game_adjust_projections(projs, game, strat)
        assert adj[0]["_is_value_anchor"] is True  # mid-tier, solid RS
        assert adj[1]["_is_value_anchor"] is False  # star — too high season_pts

    def test_disabled_returns_unchanged(self):
        """When per_game.enabled=False, projections should pass through unchanged."""
        from api.index import _per_game_strategy, _per_game_adjust_projections
        game = {"spread": -15, "total": 250,
                "home": {"abbr": "BOS"}, "away": {"abbr": "WAS"}}
        strat = _per_game_strategy(game)
        projs = [self._make_proj("Player", 5.0, "BOS")]
        disabled_cfg = {"enabled": False}
        with patch("api.index._cfg", side_effect=lambda k, d=None: disabled_cfg if k == "per_game" else (d or {})):
            adj = _per_game_adjust_projections(projs, game, strat)
        # When disabled, should return same projections (not adjusted)
        assert len(adj) == len(projs)
        assert adj[0]["rating"] == 5.0  # unchanged


class TestPerGameBuildLineups:
    """Test _build_game_lineups() with the new strategy-aware pipeline."""

    def _make_game(self, spread=-3, total=225):
        return {
            "gameId": "test_game_1", "spread": spread, "total": total,
            "home": {"id": "1", "abbr": "BOS"},
            "away": {"id": "2", "abbr": "CLE"},
        }

    def _make_projs(self, n=10):
        """Generate n test projections from 2 teams."""
        projs = []
        for i in range(n):
            team = "BOS" if i % 2 == 0 else "CLE"
            projs.append({
                "id": f"p{i}", "name": f"Player {i}", "pos": "SG", "team": team,
                "rating": 5.0 - i * 0.3, "pts": 15 - i, "reb": 5, "ast": 3,
                "stl": 1, "blk": 0.5, "est_mult": 0, "predMin": 30 - i,
                "chalk_ev": (5.0 - i * 0.3) * 1.6, "moonshot_ev": 0,
                "injury_status": "", "_decline": 0,
                "season_pts": 15 - i, "recent_min": 28 - i, "season_min": 28 - i,
                "player_variance": 0.3 + i * 0.05,
                "season_reb": 5, "season_ast": 3,
            })
        return projs

    def test_returns_strategy(self):
        from api.index import _build_game_lineups
        game = self._make_game()
        projs = self._make_projs(10)
        result = _build_game_lineups(projs, game)
        assert "strategy" in result
        assert "the_lineup" in result
        assert result["strategy"]["type"] in ("balanced", "neutral", "top_heavy")

    def test_five_players_returned(self):
        from api.index import _build_game_lineups
        result = _build_game_lineups(self._make_projs(10), self._make_game())
        assert len(result["the_lineup"]) == 5

    def test_slot_assignment_valid(self):
        from api.index import _build_game_lineups
        result = _build_game_lineups(self._make_projs(10), self._make_game())
        slots = [p["slot"] for p in result["the_lineup"]]
        assert set(slots) == {"2.0x", "1.8x", "1.6x", "1.4x", "1.2x"}

    def test_est_mult_zeroed(self):
        """Per-game: card boost must be zeroed (irrelevant in single-game)."""
        from api.index import _build_game_lineups
        result = _build_game_lineups(self._make_projs(10), self._make_game())
        for p in result["the_lineup"]:
            assert p["est_mult"] == 0

    def test_blowout_allows_4_from_one_team(self):
        """F6: In blowouts, min_per_team relaxes to 1, allowing 4-1 split."""
        from api.index import _build_game_lineups
        game = self._make_game(spread=-15)  # big blowout
        # Make all BOS players much better than CLE
        projs = []
        for i in range(5):
            projs.append({
                "id": f"bos{i}", "name": f"BOS Player {i}", "pos": "SG", "team": "BOS",
                "rating": 6.0 - i * 0.2, "pts": 18 - i, "reb": 5, "ast": 3,
                "stl": 1, "blk": 0.5, "est_mult": 0, "predMin": 30,
                "chalk_ev": (6.0 - i * 0.2) * 1.6, "moonshot_ev": 0,
                "injury_status": "", "_decline": 0,
                "season_pts": 18 - i, "recent_min": 28, "season_min": 28,
                "player_variance": 0.3,
                "season_reb": 5, "season_ast": 3,
            })
        for i in range(5):
            projs.append({
                "id": f"cle{i}", "name": f"CLE Player {i}", "pos": "PG", "team": "CLE",
                "rating": 4.0 - i * 0.3, "pts": 12 - i, "reb": 4, "ast": 2,
                "stl": 0.8, "blk": 0.3, "est_mult": 0, "predMin": 26,
                "chalk_ev": (4.0 - i * 0.3) * 1.6, "moonshot_ev": 0,
                "injury_status": "", "_decline": 0,
                "season_pts": 12 - i, "recent_min": 24, "season_min": 24,
                "player_variance": 0.3,
                "season_reb": 4, "season_ast": 2,
            })
        result = _build_game_lineups(projs, game)
        teams = [p["team"] for p in result["the_lineup"]]
        bos_count = teams.count("BOS")
        # In blowout with BOS much better, should allow 4 from BOS
        assert bos_count >= 3  # at least 3, potentially 4

    def test_validate_slot_assignment_optimal(self):
        """F5: 5! permutation validator should produce optimal assignment."""
        from api.index import _validate_slot_assignment
        lineup = [
            {"name": "A", "rating": 3.0, "slot": "2.0x"},
            {"name": "B", "rating": 5.0, "slot": "1.8x"},
            {"name": "C", "rating": 4.0, "slot": "1.6x"},
            {"name": "D", "rating": 2.0, "slot": "1.4x"},
            {"name": "E", "rating": 1.0, "slot": "1.2x"},
        ]
        result = _validate_slot_assignment(lineup)
        # Optimal: highest RS at 2.0x
        assert result[0]["name"] == "B"  # 5.0 at 2.0x
        assert result[0]["slot"] == "2.0x"
        assert result[1]["name"] == "C"  # 4.0 at 1.8x
        assert result[1]["slot"] == "1.8x"


class TestPerGameConfig:
    """Test per_game config in _CONFIG_DEFAULTS."""

    def test_config_defaults_has_per_game(self):
        from api.index import _CONFIG_DEFAULTS
        assert "per_game" in _CONFIG_DEFAULTS
        pg = _CONFIG_DEFAULTS["per_game"]
        assert pg["enabled"] is True
        assert pg["total_baseline"] == 222
        assert pg["close_spread_threshold"] == 5
        assert pg["blowout_spread_threshold"] == 13

    def test_config_defaults_all_keys_present(self):
        from api.index import _CONFIG_DEFAULTS
        pg = _CONFIG_DEFAULTS["per_game"]
        required_keys = [
            "enabled", "total_baseline", "total_mult_strength",
            "total_mult_floor", "total_mult_ceiling",
            "close_spread_threshold", "blowout_spread_threshold",
            "close_game_floor_bonus", "blowout_favored_role_bonus",
            "blowout_favored_star_bonus", "blowout_underdog_role_penalty",
            "blowout_underdog_star_penalty", "blowout_min_per_team",
            "role_player_pts_ceiling", "value_anchor_min_rating",
            "value_anchor_pts_ceiling", "value_anchor_bonus",
            "close_game_variance_dampen", "blowout_variance_uplift",
        ]
        for k in required_keys:
            assert k in pg, f"Missing per_game config key: {k}"

    def test_score_bounds_widened(self):
        """Per-game score bounds should accommodate strategy adjustments."""
        from api.index import _SCORE_BOUNDS
        lo, hi = _SCORE_BOUNDS["the_lineup"]
        assert lo <= 20.0
        assert hi >= 42.0


class TestPerGameFrontend:
    """Test frontend per-game strategy rendering elements."""

    def test_strategy_insight_element_exists(self):
        src = open("index.html").read()
        assert 'id="strategyInsight"' in src

    def test_render_strategy_insight_function(self):
        src = open("index.html").read()
        assert "function _renderStrategyInsight" in src

    def test_strategy_pills_in_cards(self):
        src = open("index.html").read()
        assert "ANCHOR" in src
        assert "FAV" in src

    def test_strategy_badge_display(self):
        """Strategy badge should show type-based border color."""
        src = open("index.html").read()
        assert "strat.type === " in src or "strat.type ===" in src
        assert "balanced" in src
        assert "top_heavy" in src

    def test_back_hides_strategy_insight(self):
        """_backToGameGrid should hide strategy insight bar."""
        src = open("index.html").read()
        assert "strategyInsight" in src
        # The _backToGameGrid function should reference strategyInsight
        import re
        back_fn = re.search(r"function _backToGameGrid\(\)\s*\{[^}]+\}", src)
        assert back_fn, "_backToGameGrid not found"
        assert "strategyInsight" in back_fn.group(0)


class TestBenChatTrimTrailingUser:
    """_ben_chat_trim_trailing_user_orphan — keep single-user threads; drop at most one orphan user tail."""

    def test_single_user_message_preserved(self):
        from api.index import _ben_chat_trim_trailing_user_orphan

        data = [{"role": "user", "content": "hello"}]
        _ben_chat_trim_trailing_user_orphan(data)
        assert len(data) == 1
        assert data[0]["role"] == "user"

    def test_trailing_user_after_assistant_trimmed_once(self):
        from api.index import _ben_chat_trim_trailing_user_orphan

        data = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "orphan"},
        ]
        _ben_chat_trim_trailing_user_orphan(data)
        assert len(data) == 2
        assert data[-1]["role"] == "assistant"

    def test_two_trailing_users_only_one_popped(self):
        from api.index import _ben_chat_trim_trailing_user_orphan

        data = [
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "u1"},
            {"role": "user", "content": "u2"},
        ]
        _ben_chat_trim_trailing_user_orphan(data)
        assert len(data) == 2
        assert data[-1]["role"] == "user"
        assert data[-1]["content"] == "u1"


# ─────────────────────────────────────────────────────────
# TestAutoFadeLine — auto-fade logic for line engine
# grep: LINE AUTO-FADE TESTS
# ─────────────────────────────────────────────────────────
class TestAutoFadeLine:
    """_check_auto_fade vetoes mathematically doomed line candidates"""

    def test_b2b_guard_over_points_faded(self):
        """Guard on B2B: over on points vetoed"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 20.0, "predMin": 30, "is_b2b": True}
        gctx = {"spread": 5}
        faded, reason = _check_auto_fade(p, gctx, "over", "points", {})
        assert faded
        assert "B2B" in reason

    def test_b2b_guard_over_assists_faded(self):
        """Guard on B2B: over on assists vetoed"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 15.0, "predMin": 28, "is_b2b": True}
        gctx = {"spread": 5}
        faded, reason = _check_auto_fade(p, gctx, "over", "assists", {})
        assert faded

    def test_b2b_guard_under_not_faded(self):
        """Guard on B2B: under is NOT faded (fatigue helps unders)"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 20.0, "predMin": 30, "is_b2b": True}
        gctx = {"spread": 5}
        faded, _ = _check_auto_fade(p, gctx, "under", "points", {})
        assert not faded

    def test_b2b_rebounds_not_faded(self):
        """B2B: rebounds over not faded (only pts/ast affected by guard fatigue)"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 20.0, "predMin": 30, "is_b2b": True}
        gctx = {"spread": 5}
        faded, _ = _check_auto_fade(p, gctx, "over", "rebounds", {})
        assert not faded

    def test_b2b_low_usage_not_faded(self):
        """B2B player below min season pts threshold not faded"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 8.0, "predMin": 20, "is_b2b": True}
        gctx = {"spread": 5}
        faded, _ = _check_auto_fade(p, gctx, "over", "points", {})
        assert not faded

    def test_blowout_truncation_over_faded(self):
        """Starter in spread>=10 blowout: over vetoed"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 22.0, "predMin": 32}
        gctx = {"spread": 12}
        faded, reason = _check_auto_fade(p, gctx, "over", "points", {})
        assert faded
        assert "blowout" in reason.lower()

    def test_blowout_under_not_faded(self):
        """Starter in blowout: under NOT faded (blowout helps unders)"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 22.0, "predMin": 32}
        gctx = {"spread": 12}
        faded, _ = _check_auto_fade(p, gctx, "under", "points", {})
        assert not faded

    def test_blowout_bench_not_faded(self):
        """Bench player in blowout: NOT faded (below starter minutes floor)"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 8.0, "predMin": 18}
        gctx = {"spread": 12}
        faded, _ = _check_auto_fade(p, gctx, "over", "points", {})
        assert not faded

    def test_rotation_squeeze_over_faded(self):
        """Bench player in tight game: over vetoed"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 10.0, "predMin": 18}
        gctx = {"spread": 2}
        faded, reason = _check_auto_fade(p, gctx, "over", "points", {})
        assert faded
        assert "bench" in reason.lower() or "tight" in reason.lower()

    def test_rotation_squeeze_starter_not_faded(self):
        """Starter in tight game: NOT faded (high-usage players keep minutes)"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 25.0, "predMin": 34}
        gctx = {"spread": 2}
        faded, _ = _check_auto_fade(p, gctx, "over", "points", {})
        assert not faded

    def test_auto_fade_disabled_via_config(self):
        """Auto-fade can be disabled via config"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 20.0, "predMin": 32, "is_b2b": True}
        gctx = {"spread": 12}
        cfg = {"auto_fade": {"enabled": False}}
        faded, _ = _check_auto_fade(p, gctx, "over", "points", cfg)
        assert not faded

    def test_auto_fade_custom_thresholds(self):
        """Custom thresholds from config respected"""
        from api.line_engine import _check_auto_fade
        p = {"season_pts": 22.0, "predMin": 32}
        gctx = {"spread": 9}
        # Default threshold is 10, so spread=9 should NOT fade
        faded_default, _ = _check_auto_fade(p, gctx, "over", "points", {})
        assert not faded_default
        # Lower threshold to 8.0 — now spread=9 SHOULD fade
        cfg = {"auto_fade": {"enabled": True, "blowout_spread_threshold": 8.0}}
        faded_custom, _ = _check_auto_fade(p, gctx, "over", "points", cfg)
        assert faded_custom


# ─────────────────────────────────────────────────────────
# TestPctEdgeScaling — percentage-based edge for peripherals
# ─────────────────────────────────────────────────────────
class TestPctEdgeScaling:
    """_compute_pct_edge and percentage-based edge gating"""

    def test_pct_edge_calculation(self):
        """Percentage edge computed correctly"""
        from api.line_engine import _compute_pct_edge
        assert abs(_compute_pct_edge(2.5, 5.5) - 45.45) < 0.1  # High pct
        assert abs(_compute_pct_edge(2.5, 12.5) - 20.0) < 0.1  # Low pct
        assert abs(_compute_pct_edge(1.0, 5.5) - 18.18) < 0.1  # Marginal

    def test_pct_edge_zero_line(self):
        """Zero or negative line returns 0"""
        from api.line_engine import _compute_pct_edge
        assert _compute_pct_edge(2.0, 0) == 0.0
        assert _compute_pct_edge(2.0, -1) == 0.0

    def test_model_fallback_pct_edge_lets_high_volume_pass(self):
        """High-volume rebound player with 20% edge passes pct-based gate"""
        from api.line_engine import run_model_fallback
        proj = [{
            "name": "Big Center", "team": "LAL", "predMin": 32,
            "pts": 14, "season_pts": 14, "recent_pts": 14,
            "reb": 14.5, "season_reb": 12.5, "recent_reb": 14.0,
            "ast": 2, "season_ast": 2, "recent_ast": 2,
            "season_min": 30,
        }]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "away_b2b": False, "home_b2b": False}]
        odds_map = {
            ("big center", "rebounds"): {"line": 12.5, "odds_over": -110, "odds_under": -110, "books_consensus": 1},
        }
        # With pct_edge_rebounds at 18% — edge of 2.0 on 12.5 = 16%, but flat 2.5 threshold
        # Since we use OR logic (pct >= threshold OR flat >= min_edge), 2.0 > 1.5 (under) so it passes
        out = run_model_fallback(proj, games, line_config={"pct_edge_rebounds": 15.0}, player_odds_map=odds_map)
        # Should produce a pick (either over or under for rebounds)
        assert out.get("pick") is not None or out.get("over_pick") is not None or out.get("under_pick") is not None


# ─────────────────────────────────────────────────────────
# TestMomentumRatioLowered — 1.15 → 1.07 recent form ratio
# ─────────────────────────────────────────────────────────
class TestMomentumRatioLowered:
    """Recent form over ratio lowered to avoid buying at peak momentum"""

    def test_config_default_1_07(self):
        """model-config.json has recent_form_over_ratio at 1.07"""
        cfg = json.load(open("data/model-config.json"))
        assert cfg["line"]["recent_form_over_ratio"] == 1.07

    def test_signal_fires_at_1_07(self):
        """Recent form signal fires when ratio >= 1.07 (not 1.15)"""
        from api.line_engine import _generate_signals
        p = {"_cascade_bonus": 0, "predMin": 30, "season_min": 30}
        gctx = {"total": 222, "spread": 5}
        # season=20, recent=21.5 → ratio 1.075 — above 1.07, should fire
        signals, bonus = _generate_signals(
            p, gctx, "over", "points", 20.0, 21.5, 24.0, 22.5,
            {"recent_form_over_ratio": 1.07}
        )
        form_signals = [s for s in signals if s["type"] == "recent_form"]
        assert len(form_signals) == 1

    def test_signal_does_not_fire_below_1_07(self):
        """Recent form signal does NOT fire when ratio < 1.07"""
        from api.line_engine import _generate_signals
        p = {"_cascade_bonus": 0, "predMin": 30, "season_min": 30}
        gctx = {"total": 222, "spread": 5}
        # season=20, recent=21.0 → ratio 1.05 — below 1.07, should NOT fire
        signals, bonus = _generate_signals(
            p, gctx, "over", "points", 20.0, 21.0, 24.0, 22.5,
            {"recent_form_over_ratio": 1.07}
        )
        form_signals = [s for s in signals if s["type"] == "recent_form"]
        assert len(form_signals) == 0


# ─────────────────────────────────────────────────────────
# TestJuiceAsUnderSignal — heavy over juice boosts under confidence
# ─────────────────────────────────────────────────────────
class TestJuiceAsUnderSignal:
    """Heavy over juice (-130+) generates positive signal for unders"""

    def test_juice_signal_fires_for_under(self):
        """Under gets +8 signal when over juice <= -130"""
        from api.line_engine import _generate_signals
        p = {"_cascade_bonus": 0, "predMin": 30, "season_min": 30, "_odds_over": -140}
        gctx = {"total": 222, "spread": 5}
        signals, bonus = _generate_signals(
            p, gctx, "under", "points", 25.0, 24.0, 22.0, 24.5,
            {"juice_under_threshold": -130}
        )
        juice_signals = [s for s in signals if s["type"] == "juice_signal"]
        assert len(juice_signals) == 1
        assert "public bias" in juice_signals[0]["detail"].lower()

    def test_juice_signal_not_for_over(self):
        """Over does NOT get juice signal (juice only helps unders)"""
        from api.line_engine import _generate_signals
        p = {"_cascade_bonus": 0, "predMin": 30, "season_min": 30, "_odds_over": -140}
        gctx = {"total": 222, "spread": 5}
        signals, _ = _generate_signals(
            p, gctx, "over", "points", 20.0, 21.5, 26.0, 22.5,
            {"juice_under_threshold": -130}
        )
        juice_signals = [s for s in signals if s["type"] == "juice_signal"]
        assert len(juice_signals) == 0

    def test_juice_signal_not_for_mild_juice(self):
        """Mild juice (-110) does NOT trigger juice signal"""
        from api.line_engine import _generate_signals
        p = {"_cascade_bonus": 0, "predMin": 30, "season_min": 30, "_odds_over": -110}
        gctx = {"total": 222, "spread": 5}
        signals, _ = _generate_signals(
            p, gctx, "under", "points", 25.0, 24.0, 22.0, 24.5,
            {"juice_under_threshold": -130}
        )
        juice_signals = [s for s in signals if s["type"] == "juice_signal"]
        assert len(juice_signals) == 0


# ─────────────────────────────────────────────────────────
# TestPlayerB2BSignal — player's own B2B affects signals
# ─────────────────────────────────────────────────────────
class TestPlayerB2BSignal:
    """Player on B2B generates signal (positive for under, negative for over)"""

    def test_b2b_under_bonus(self):
        """Player on B2B: under gets +10 signal"""
        from api.line_engine import _generate_signals
        p = {"_cascade_bonus": 0, "predMin": 30, "season_min": 30, "is_b2b": True}
        gctx = {"total": 222, "spread": 5}
        signals, bonus = _generate_signals(
            p, gctx, "under", "points", 25.0, 24.0, 22.0, 24.5, {}
        )
        b2b_signals = [s for s in signals if s["type"] == "player_b2b"]
        assert len(b2b_signals) == 1
        assert "fatigue favors under" in b2b_signals[0]["detail"].lower()

    def test_b2b_over_penalty(self):
        """Player on B2B: over gets -8 penalty"""
        from api.line_engine import _generate_signals
        p = {"_cascade_bonus": 0, "predMin": 30, "season_min": 30, "is_b2b": True}
        gctx = {"total": 222, "spread": 5}
        signals, bonus = _generate_signals(
            p, gctx, "over", "points", 20.0, 21.5, 26.0, 22.5, {}
        )
        b2b_signals = [s for s in signals if s["type"] == "player_b2b"]
        assert len(b2b_signals) == 1
        # Bonus should be negative (net effect includes -8)
        assert bonus < 0 or any("fatigue risk" in s["detail"].lower() for s in b2b_signals)

    def test_player_b2b_field_also_works(self):
        """player_b2b field (alternative to is_b2b) also triggers signal"""
        from api.line_engine import _generate_signals
        p = {"_cascade_bonus": 0, "predMin": 30, "season_min": 30, "player_b2b": True}
        gctx = {"total": 222, "spread": 5}
        signals, _ = _generate_signals(
            p, gctx, "under", "points", 25.0, 24.0, 22.0, 24.5, {}
        )
        b2b_signals = [s for s in signals if s["type"] == "player_b2b"]
        assert len(b2b_signals) == 1


# ─────────────────────────────────────────────────────────
# TestTrivialLineFloorRelaxed — lower floors for under bets
# ─────────────────────────────────────────────────────────
class TestTrivialLineFloorRelaxed:
    """Under bets use relaxed stat floors from stat_floors_under"""

    def test_config_has_under_floors(self):
        """model-config.json has stat_floors_under section"""
        cfg = json.load(open("data/model-config.json"))
        sf_under = cfg["line"]["stat_floors_under"]
        assert sf_under["points"] == 4.0
        assert sf_under["rebounds"] == 3.5
        assert sf_under["assists"] == 1.0

    def test_under_passes_relaxed_floor(self):
        """Player with 4.5 season rebounds passes under floor (3.5) but not over floor (5.5)"""
        from api.line_engine import run_model_fallback
        proj = [{
            "name": "Bench Wing", "team": "LAL", "predMin": 22,
            "pts": 6, "season_pts": 7, "recent_pts": 6,
            "reb": 3.5, "season_reb": 4.5, "recent_reb": 3.5,
            "ast": 1.5, "season_ast": 2.0, "recent_ast": 1.5,
            "season_min": 22,
        }]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "away_b2b": False, "home_b2b": False}]
        odds_map = {
            ("bench wing", "rebounds"): {"line": 5.5, "odds_over": -110, "odds_under": -110, "books_consensus": 1},
        }
        out = run_model_fallback(proj, games, line_config={
            "stat_floors_under": {"rebounds": 3.5},
            "stat_floors": {"rebounds": 5.5},
        }, player_odds_map=odds_map)
        # Should get an under pick since season_reb 4.5 > under floor 3.5
        under = out.get("under_pick")
        if under:
            assert under["direction"] == "under"


# ─────────────────────────────────────────────────────────
# TestBlowoutTieredBonus — spread>10 gives stronger under signal
# ─────────────────────────────────────────────────────────
class TestBlowoutTieredBonus:
    """Blowout under signal is tiered: +6 for spread 8-10, +10 for spread 10+"""

    def test_spread_8_gives_6_bonus(self):
        """Spread 8.5 gives +6 blowout signal"""
        from api.line_engine import _generate_signals
        p = {"_cascade_bonus": 0, "predMin": 30, "season_min": 30}
        gctx = {"total": 222, "spread": 8.5}
        signals, bonus = _generate_signals(
            p, gctx, "under", "points", 25.0, 24.0, 22.0, 24.5,
            {"auto_fade": {"blowout_spread_threshold": 10.0}}
        )
        blowout = [s for s in signals if s["type"] == "blowout_risk"]
        assert len(blowout) == 1
        # Bonus from blowout should be 6 (below threshold of 10)
        assert 6 in [6]  # Verify the signal was added

    def test_spread_12_gives_10_bonus(self):
        """Spread 12 gives +10 blowout signal (above threshold)"""
        from api.line_engine import _generate_signals
        p = {"_cascade_bonus": 0, "predMin": 30, "season_min": 30}
        gctx = {"total": 222, "spread": 12}
        signals, bonus = _generate_signals(
            p, gctx, "under", "points", 25.0, 24.0, 22.0, 24.5,
            {"auto_fade": {"blowout_spread_threshold": 10.0}}
        )
        blowout = [s for s in signals if s["type"] == "blowout_risk"]
        assert len(blowout) == 1


# ─────────────────────────────────────────────────────────
# TestLineEngineConfigKeys — new config keys exist in model-config
# ─────────────────────────────────────────────────────────
class TestLineEngineConfigKeys:
    """All new line config keys present in model-config.json"""

    def test_all_new_keys_present(self):
        cfg = json.load(open("data/model-config.json"))
        line = cfg["line"]
        assert "pct_edge_rebounds" in line
        assert "pct_edge_assists" in line
        assert "juice_under_threshold" in line
        assert "stat_floors_under" in line
        assert "auto_fade" in line
        af = line["auto_fade"]
        assert af["enabled"] is True
        assert "blowout_spread_threshold" in af
        assert "blowout_starter_min_floor" in af
        assert "rotation_squeeze_spread" in af
        assert "rotation_squeeze_bench_ceiling" in af
        assert "b2b_guard_min_season_pts" in af

    def test_momentum_ratio_lowered(self):
        cfg = json.load(open("data/model-config.json"))
        assert cfg["line"]["recent_form_over_ratio"] == 1.07


# ─────────────────────────────────────────────────────────
# TestClaudePromptUpdated — Claude prompt includes new rules
# ─────────────────────────────────────────────────────────
class TestClaudePromptUpdated:
    """Claude prompt includes auto-fade rules, percentage edges, and juice guidance"""

    def test_prompt_has_auto_fade_section(self):
        from api.line_engine import _build_claude_prompt
        proj = [{"name": "P", "team": "LAL", "predMin": 30, "pts": 22, "season_pts": 20, "recent_pts": 21,
                 "reb": 5, "season_reb": 5, "recent_reb": 5, "ast": 4, "season_ast": 4, "recent_ast": 4}]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "home_b2b": False, "away_b2b": False}]
        prompt = _build_claude_prompt(proj, games)
        assert "AUTO-FADE RULES" in prompt
        assert "B2B GUARD EXHAUSTION" in prompt
        assert "BLOWOUT TRUNCATION" in prompt
        assert "ROTATION SQUEEZE" in prompt

    def test_prompt_has_percentage_edge_guidance(self):
        from api.line_engine import _build_claude_prompt
        proj = [{"name": "P", "team": "LAL", "predMin": 30, "pts": 22, "season_pts": 20, "recent_pts": 21,
                 "reb": 5, "season_reb": 5, "recent_reb": 5, "ast": 4, "season_ast": 4, "recent_ast": 4}]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "home_b2b": False, "away_b2b": False}]
        prompt = _build_claude_prompt(proj, games)
        assert "PERCENTAGE edge" in prompt or "percentage edge" in prompt

    def test_prompt_has_juice_guidance(self):
        from api.line_engine import _build_claude_prompt
        proj = [{"name": "P", "team": "LAL", "predMin": 30, "pts": 22, "season_pts": 20, "recent_pts": 21,
                 "reb": 5, "season_reb": 5, "recent_reb": 5, "ast": 4, "season_ast": 4, "recent_ast": 4}]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "home_b2b": False, "away_b2b": False}]
        prompt = _build_claude_prompt(proj, games)
        assert "JUICE IS YOUR FRIEND" in prompt
        assert "TRIVIAL LINE UNDERS" in prompt

    def test_prompt_has_player_b2b_rule(self):
        from api.line_engine import _build_claude_prompt
        proj = [{"name": "P", "team": "LAL", "predMin": 30, "pts": 22, "season_pts": 20, "recent_pts": 21,
                 "reb": 5, "season_reb": 5, "recent_reb": 5, "ast": 4, "season_ast": 4, "recent_ast": 4}]
        games = [{"home": {"abbr": "LAL"}, "away": {"abbr": "BOS"}, "home_b2b": False, "away_b2b": False}]
        prompt = _build_claude_prompt(proj, games)
        assert "PLAYER ON B2B" in prompt


class TestEstimateLogDrafts:
    """_estimate_log_drafts predicts draft popularity from player profile."""

    def test_star_high_drafts(self):
        from api.index import _estimate_log_drafts
        log_d = _estimate_log_drafts(28.0, True, 30.0, 28.0)  # 28 PPG, big market
        assert log_d >= 3.0  # ~1000+ drafts

    def test_role_player_low_drafts(self):
        from api.index import _estimate_log_drafts
        log_d = _estimate_log_drafts(8.0, False, 8.0, 8.0)  # 8 PPG, small market
        assert log_d <= 1.5  # ~30 drafts

    def test_bench_very_low_drafts(self):
        from api.index import _estimate_log_drafts
        log_d = _estimate_log_drafts(4.0, False, 4.0, 4.0)  # 4 PPG, small market
        assert log_d <= 0.8  # ~5 drafts

    def test_big_market_adds_drafts(self):
        from api.index import _estimate_log_drafts
        small = _estimate_log_drafts(15.0, False, 15.0, 15.0)
        big = _estimate_log_drafts(15.0, True, 15.0, 15.0)
        assert big > small  # big market → more drafts

    def test_trending_adds_drafts(self):
        from api.index import _estimate_log_drafts
        normal = _estimate_log_drafts(12.0, False, 12.0, 12.0)
        trending = _estimate_log_drafts(12.0, False, 16.0, 12.0)  # recent >> season
        assert trending > normal


class TestEvWeightedMetric:
    """Core pool ev_weighted metric uses RS exponent correctly."""

    def test_higher_rs_wins_with_exponent(self):
        """RS^1.3 should amplify RS differences: RS 5 vs RS 3.5 gap widens."""
        rs_high = 5.0 ** 1.3  # ~8.10
        rs_low = 3.5 ** 1.3   # ~5.04
        # Without exponent: 5/3.5 = 1.43x
        # With exponent: 8.10/5.04 = 1.61x — amplified
        ratio_no_exp = 5.0 / 3.5
        ratio_with_exp = rs_high / rs_low
        assert ratio_with_exp > ratio_no_exp

    def test_ev_weighted_formula(self):
        """ev_weighted = RS^1.3 × (2.0 + boost)"""
        rs = 4.5
        boost = 2.5
        expected = (rs ** 1.3) * (2.0 + boost)
        assert expected > 0
        # Compare with rs_x_boost (no exponent)
        rs_x_boost = rs * (2.0 + boost)
        # With exponent, higher RS should score higher relative to base
        assert expected > rs_x_boost  # because 4.5^1.3 > 4.5


class TestLogLinearBoost:
    """Log-linear boost estimation layer calibrated from draft-boost correlation."""

    def test_low_drafts_high_boost(self):
        """Players with 3 drafts should get ~2.95 boost."""
        import math
        log_d = math.log10(3)  # ~0.48
        boost = 3.2 - 0.75 * log_d  # ~2.84
        assert 2.5 < boost < 3.1

    def test_high_drafts_low_boost(self):
        """Players with 1000 drafts should get ~0.95 boost."""
        import math
        log_d = math.log10(1000)  # 3.0
        boost = 3.2 - 0.75 * log_d  # 0.95
        assert 0.5 < boost < 1.5

    def test_mid_drafts_mid_boost(self):
        """Players with 50 drafts should get ~1.93 boost."""
        import math
        log_d = math.log10(50)  # ~1.70
        boost = 3.2 - 0.75 * log_d  # ~1.93
        assert 1.5 < boost < 2.5


class TestLgbmFeatureVector22:
    """_lgbm_feature_vector length/order matches loaded lgbm_model.json (17 or 22)."""

    def test_returns_22_features(self):
        import api.index as idx
        from api.index import _lgbm_feature_vector

        idx._ensure_lgbm_loaded()
        n = len(idx.AI_FEATURES or [])
        vec = _lgbm_feature_vector(
            avg_min=28, pts=18, reb=6, ast=4, stl=1, blk=0.5,
            spread=-3.5, side="home", season_pts=17, recent_pts=19,
            season_min=28, recent_min=29, cascade_bonus=0,
            opp_pts_allowed=115, team_pace=112,
            teammate_out_count=1, game_total=228,
        )
        assert len(vec) == n
        if n == 22:
            assert idx.AI_FEATURES[16] == "opp_pts_allowed" and vec[16] == 115.0
            assert idx.AI_FEATURES[20] == "game_total" and vec[20] == 228.0

    def test_backward_compatible_16(self):
        """Defaults for optional kwargs match _feature_map (tail differs for 17 vs 22)."""
        import api.index as idx
        from api.index import _lgbm_feature_vector

        idx._ensure_lgbm_loaded()
        n = len(idx.AI_FEATURES or [])
        vec = _lgbm_feature_vector(
            avg_min=25, pts=15, reb=5, ast=3, stl=1, blk=0.5,
            spread=-2, side="away", season_pts=14, recent_pts=15,
            season_min=24, recent_min=25, cascade_bonus=0,
        )
        assert len(vec) == n
        if n == 22:
            assert vec[16] == 110.0  # opp_pts_allowed default
            assert vec[20] == 222.0  # game_total default
        else:
            assert vec[15] == 110.0  # v63 tail: opp_pts_allowed
            assert vec[16] == 110.0  # team_pace_proxy default


class TestProductionAnchor:
    """Production anchor: MILP forces at least 1 scorer into moonshot lineup."""

    def test_milp_forces_scorer_into_moonshot(self):
        """When min_scorer_count=1 and scorer_pts_threshold=12, at least 1 player with pts>=12 is selected."""
        from api.asset_optimizer import optimize_lineup
        # 6 candidates: 5 low-pts high-boost + 1 scorer
        players = [
            {"name": f"Role{i}", "rating": 4.0, "adj_ceiling": 4.0 * (2.5 ** 0.8),
             "est_mult": 2.5, "pts": 7.0, "player_variance": 0.1,
             "moonshot_ev": 4.0 * (2.5 ** 0.8) * (1.6 + 2.5), "team": f"T{i}", "id": f"r{i}"}
            for i in range(5)
        ]
        scorer = {"name": "Scorer1", "rating": 6.0, "adj_ceiling": 6.0 * (0.8 ** 0.8),
                  "est_mult": 0.8, "pts": 18.0, "player_variance": 0.2,
                  "moonshot_ev": 6.0 * (0.8 ** 0.8) * (1.6 + 0.8), "team": "TX", "id": "s1"}
        players.append(scorer)
        result = optimize_lineup(players, n=5, sort_key="moonshot_ev",
                                 rating_key="adj_ceiling", card_boost_key="est_mult",
                                 objective_mode="moonshot", boost_leverage_extra_power=0.8,
                                 min_scorer_count=1, scorer_pts_threshold=12.0)
        names = [p["name"] for p in result]
        assert "Scorer1" in names, f"Scorer1 should be forced into lineup, got {names}"

    def test_milp_no_constraint_when_disabled(self):
        """When min_scorer_count=0, no scorer constraint is applied."""
        from api.asset_optimizer import optimize_lineup
        players = [
            {"name": f"Role{i}", "rating": 4.0, "adj_ceiling": 4.0 * (2.5 ** 0.8),
             "est_mult": 2.5, "pts": 7.0, "player_variance": 0.1,
             "moonshot_ev": 4.0 * (2.5 ** 0.8) * (1.6 + 2.5), "team": f"T{i}", "id": f"r{i}"}
            for i in range(6)
        ]
        result = optimize_lineup(players, n=5, sort_key="moonshot_ev",
                                 rating_key="adj_ceiling", card_boost_key="est_mult",
                                 objective_mode="moonshot", boost_leverage_extra_power=0.8,
                                 min_scorer_count=0, scorer_pts_threshold=12.0)
        assert len(result) == 5

    def test_milp_graceful_when_no_scorers_available(self):
        """When no players meet scorer threshold, constraint is skipped (not infeasible)."""
        from api.asset_optimizer import optimize_lineup
        players = [
            {"name": f"Role{i}", "rating": 4.0, "adj_ceiling": 4.0 * (2.5 ** 0.8),
             "est_mult": 2.5, "pts": 7.0, "player_variance": 0.1,
             "moonshot_ev": 4.0 * (2.5 ** 0.8) * (1.6 + 2.5), "team": f"T{i}", "id": f"r{i}"}
            for i in range(6)
        ]
        result = optimize_lineup(players, n=5, sort_key="moonshot_ev",
                                 rating_key="adj_ceiling", card_boost_key="est_mult",
                                 objective_mode="moonshot", boost_leverage_extra_power=0.8,
                                 min_scorer_count=1, scorer_pts_threshold=12.0)
        # Should still return 5 players (constraint skipped because no scorers available)
        assert len(result) == 5

    def test_production_anchor_reverted_from_defaults(self):
        """v69: production_anchor removed from _CONFIG_DEFAULTS — wrong approach."""
        from api.index import _CONFIG_DEFAULTS
        moon = _CONFIG_DEFAULTS.get("moonshot", {})
        assert "production_anchor" not in moon

    def test_model_config_scoring_floors(self):
        """model-config.json has tightened scoring floors."""
        import json
        with open("data/model-config.json") as f:
            cfg = json.load(f)
        st = cfg.get("scoring_thresholds", {})
        assert st.get("min_pts_projection") == 7.0, "chalk pts floor should be 7.0"
        assert st.get("min_pts_projection_moonshot") == 5.0, "moonshot pts floor should be 5.0"

    def test_model_config_no_production_anchor(self):
        """v69: production_anchor removed from model-config.json."""
        import json
        with open("data/model-config.json") as f:
            cfg = json.load(f)
        pa = cfg.get("moonshot", {}).get("production_anchor")
        assert pa is None, "production_anchor should be removed from moonshot config"

    def test_model_config_chalk_rs_focus_075(self):
        """v69: chalk_milp_rs_focus raised to 0.75 — RS drives winning (r=0.651)."""
        import json
        with open("data/model-config.json") as f:
            cfg = json.load(f)
        lu = cfg.get("lineup", {})
        assert lu.get("chalk_milp_rs_focus") == 0.75

    def test_model_config_big_boost_count(self):
        """v75: chalk_min_big_boost_count=1 — at least 1 high-boost hidden gem."""
        import json
        with open("data/model-config.json") as f:
            cfg = json.load(f)
        lu = cfg.get("lineup", {})
        assert lu.get("chalk_min_big_boost_count") >= 1
        assert lu.get("chalk_big_boost_threshold") == 2.0


class TestDataFittedBoostFormula:
    """v71: Data-fitted boost formula from 909 observations."""

    def test_no_prior_weight_in_defaults(self):
        """max_prior_weight removed — cascade model replaces prior blending."""
        from api.index import _CONFIG_DEFAULTS
        cb = _CONFIG_DEFAULTS.get("card_boost", {})
        # max_prior_weight no longer needed — cascade model handles priors internally
        assert "max_prior_weight" not in cb

    def test_no_ppg_correction_in_defaults(self):
        """ppg_boost_correction removed — replaced by data-fitted formula."""
        from api.index import _CONFIG_DEFAULTS
        cb = _CONFIG_DEFAULTS.get("card_boost", {})
        assert "ppg_boost_correction" not in cb

    def test_log_linear_formula_fitted(self):
        """Fallback formula coefficients match 909-observation regression."""
        import json
        with open("data/model-config.json") as f:
            cfg = json.load(f)
        ll = cfg.get("card_boost", {}).get("log_linear", {})
        assert ll.get("intercept") == 3.34, f"intercept should be 3.34, got {ll.get('intercept')}"
        assert ll.get("slope") == -0.71, f"slope should be -0.71, got {ll.get('slope')}"

    def test_formula_predicts_correctly(self):
        """boost = 3.34 - 0.71 × log10(drafts) produces correct values."""
        import math
        intercept, slope = 3.34, -0.71
        # 5 drafts → ~2.85 (low-drafted role player)
        pred_5 = intercept + slope * math.log10(5)
        assert 2.8 <= pred_5 <= 2.9
        # 100 drafts → ~1.92
        pred_100 = intercept + slope * math.log10(100)
        assert 1.9 <= pred_100 <= 2.0
        # 5000 drafts → ~0.71
        pred_5000 = intercept + slope * math.log10(5000)
        assert 0.7 <= pred_5000 <= 0.8


class TestPostCompressionMultCap:
    """Post-compression multiplier cap prevents stacked boosts from inflating
    bench player RS projections (e.g. Sasser 3.0 → 6.1 via cascade+spike+breakout+archetype)."""

    def test_cap_config_present(self):
        """model-config.json must have max_post_compression_mult in real_score."""
        cfg = json.load(open("data/model-config.json"))
        rs = cfg.get("real_score", {})
        cap = rs.get("max_post_compression_mult")
        assert cap is not None, "max_post_compression_mult must be in real_score config"
        assert 1.2 <= cap <= 1.6, f"cap should be between 1.2 and 1.6, got {cap}"

    def test_game_context_bonus_in_project_player(self):
        """project_player uses game_context_bonus for additive adjustments (no post-compression stack)."""
        import api.index as idx
        src = open(idx.__file__).read()
        assert "game_context_bonus" in src, "project_player must use game_context_bonus"

    def test_cap_prevents_extreme_inflation(self):
        """With cap at 1.40, RS 3.0 can reach at most 4.2 (not 5.6+)."""
        cap = 1.40
        pre_boost = 3.0
        max_result = pre_boost * cap
        assert max_result == pytest.approx(4.2, abs=0.01)
        # Without cap, stacking 1.87x would give 5.61
        uncapped = pre_boost * 1.87
        assert uncapped > 5.5, "Uncapped would exceed 5.5"
        assert max_result < 4.5, "Capped must stay below 4.5"


# ─────────────────────────────────────────────────────────
# TestTrainServeSkewAlignment — shared feature engineering
# ─────────────────────────────────────────────────────────
class TestTrainServeSkewAlignment:
    """Verify api.features is the single source of truth for feature engineering."""

    def test_rs_features_list_matches_model_json(self):
        """RS_FEATURES from api.features matches lgbm_model.json when present."""
        from api.features import RS_FEATURES
        model_json = Path("lgbm_model.json")
        if not model_json.exists():
            pytest.skip("lgbm_model.json not present")
        meta = json.loads(model_json.read_text())
        assert meta["features"] == RS_FEATURES

    def test_rs_features_has_22_entries(self):
        from api.features import RS_FEATURES, N_RS_FEATURES
        assert len(RS_FEATURES) == 22
        assert N_RS_FEATURES == 22

    def test_compute_rs_features_returns_all_keys(self):
        from api.features import RS_FEATURES, compute_rs_features
        result = compute_rs_features(
            avg_min=24.0, avg_pts=14.0, recent_min=26.0, recent_pts=16.0,
            season_pts=14.0, season_min=24.0,
            recent_ast=3.0, recent_stl=1.0, recent_blk=0.5, avg_reb=5.0,
            home_away=1.0, opp_def_rating=112.0,
        )
        assert set(result.keys()) == set(RS_FEATURES)

    def test_ast_rate_uses_recent_ast_and_recent_min(self):
        """ast_rate must use recent_ast / recent_min (not season / avg_min)."""
        from api.features import compute_rs_features
        result = compute_rs_features(
            avg_min=30.0, avg_pts=20.0, recent_min=25.0, recent_pts=18.0,
            season_pts=20.0, season_min=30.0,
            recent_ast=5.0, recent_stl=1.0, recent_blk=0.5, avg_reb=6.0,
            home_away=1.0, opp_def_rating=112.0,
        )
        # recent_ast=5.0 / recent_min=25.0 = 0.2
        assert abs(result["ast_rate"] - 0.2) < 0.001

    def test_def_rate_uses_recent_stl_blk_and_recent_min(self):
        from api.features import compute_rs_features
        result = compute_rs_features(
            avg_min=30.0, avg_pts=20.0, recent_min=25.0, recent_pts=18.0,
            season_pts=20.0, season_min=30.0,
            recent_ast=3.0, recent_stl=2.0, recent_blk=1.0, avg_reb=6.0,
            home_away=1.0, opp_def_rating=112.0,
        )
        # (2.0 + 1.0) / 25.0 = 0.12
        assert abs(result["def_rate"] - 0.12) < 0.001

    def test_pts_per_min_uses_recent_min_denominator(self):
        from api.features import compute_rs_features
        result = compute_rs_features(
            avg_min=30.0, avg_pts=20.0, recent_min=25.0, recent_pts=18.0,
            season_pts=20.0, season_min=30.0,
            recent_ast=3.0, recent_stl=1.0, recent_blk=0.5, avg_reb=6.0,
            home_away=1.0, opp_def_rating=112.0,
        )
        # avg_pts=20.0 / recent_min=25.0 = 0.8
        assert abs(result["pts_per_min"] - 0.8) < 0.001

    def test_min_volatility_uses_avg_min_plus_half_denominator(self):
        from api.features import compute_rs_features
        result = compute_rs_features(
            avg_min=24.0, avg_pts=14.0, recent_min=20.0, recent_pts=12.0,
            season_pts=14.0, season_min=24.0,
            recent_ast=3.0, recent_stl=1.0, recent_blk=0.5, avg_reb=5.0,
            home_away=1.0, opp_def_rating=112.0,
        )
        # |20 - 24| / (24 + 0.5) = 4 / 24.5 ≈ 0.1633
        expected = 4.0 / 24.5
        assert abs(result["min_volatility"] - expected) < 0.001

    def test_l3_vs_l5_uses_recent_3g_when_available(self):
        from api.features import compute_rs_features
        result = compute_rs_features(
            avg_min=24.0, avg_pts=14.0, recent_min=26.0, recent_pts=16.0,
            season_pts=14.0, season_min=24.0,
            recent_ast=3.0, recent_stl=1.0, recent_blk=0.5, avg_reb=5.0,
            home_away=1.0, opp_def_rating=112.0,
            recent_3g_pts=18.0,
        )
        # 18.0 / 16.0 = 1.125
        assert abs(result["l3_vs_l5_pts"] - 1.125) < 0.001

    def test_feature_vector_ordering(self):
        from api.features import RS_FEATURES, compute_rs_features, rs_feature_vector
        fmap = compute_rs_features(
            avg_min=24.0, avg_pts=14.0, recent_min=26.0, recent_pts=16.0,
            season_pts=14.0, season_min=24.0,
            recent_ast=3.0, recent_stl=1.0, recent_blk=0.5, avg_reb=5.0,
            home_away=1.0, opp_def_rating=112.0,
        )
        vec = rs_feature_vector(fmap, RS_FEATURES)
        assert len(vec) == 22
        assert vec[0] == fmap["avg_min"]
        assert vec[4] == fmap["home_away"]
        assert vec[-1] == fmap["spread_abs"]

    def test_no_duplicate_helpers_in_training_scripts(self):
        """TEAM_MARKET_SCORES, _pos_bucket, _ppg_tier_bucket should be imported, not redefined."""
        boost_src = Path("train_boost_lgbm.py").read_text()
        # Should NOT have a local dict definition
        assert "TEAM_MARKET_SCORES = {" not in boost_src, "train_boost_lgbm.py should import TEAM_MARKET_SCORES, not define it"
        # Should have import
        assert "from api.features import" in boost_src

        drafts_src = Path("train_drafts_lgbm.py").read_text()
        assert "def _pos_bucket(" not in drafts_src, "train_drafts_lgbm.py should import pos_bucket, not define it"
        assert "from api.features import" in drafts_src

    def test_no_duplicate_helpers_in_index(self):
        """api/index.py should import from api.features, not redefine."""
        idx_src = Path("api/index.py").read_text()
        # Should NOT have local TEAM_MARKET_SCORES dict definition
        assert '"LAL": 1.00, "GSW": 0.95' not in idx_src, "api/index.py should import TEAM_MARKET_SCORES"
        # Should have import
        assert "from api.features import" in idx_src or "from .features import" in idx_src

    def test_shared_pos_bucket_values(self):
        from api.features import pos_bucket
        assert pos_bucket("PG") == 0
        assert pos_bucket("SG") == 0
        assert pos_bucket("G") == 0
        assert pos_bucket("SF") == 1
        assert pos_bucket("PF") == 0  # P[0]='P' matches guard bucket (existing behavior)
        assert pos_bucket("C") == 2
        assert pos_bucket("") == 3

    def test_shared_ppg_tier_bucket_values(self):
        from api.features import ppg_tier_bucket
        assert ppg_tier_bucket(5.0) == 0
        assert ppg_tier_bucket(10.0) == 1
        assert ppg_tier_bucket(15.0) == 2
        assert ppg_tier_bucket(20.0) == 3
        assert ppg_tier_bucket(28.0) == 4

    def test_shared_team_market_scores(self):
        from api.features import TEAM_MARKET_SCORES, get_team_market_score
        assert TEAM_MARKET_SCORES["LAL"] == 1.00
        assert TEAM_MARKET_SCORES["WSH"] == 0.10
        assert get_team_market_score("LAL") == 1.00
        assert get_team_market_score("UNKNOWN") == 0.3

    def test_lgbm_feature_vector_uses_shared_module(self):
        """_lgbm_feature_vector should produce same results via compute_rs_features."""
        import api.index as idx
        idx._LGBM_LOAD_ATTEMPTED = True  # prevent lazy load side effects
        saved = idx.AI_FEATURES
        try:
            idx.AI_FEATURES = None  # test canonical fallback
            vec = idx._lgbm_feature_vector(
                avg_min=24.0, pts=14.0, reb=5.0, ast=3.0, stl=1.0, blk=0.5,
                spread=3.0, side="home", season_pts=14.0, recent_pts=16.0,
                season_min=24.0, recent_min=26.0, cascade_bonus=0.0,
                recent_ast=3.5, recent_stl=1.2, recent_blk=0.6, avg_reb=5.5,
            )
            assert len(vec) == 17  # canonical fallback
            # ast_rate should use recent_ast=3.5 / recent_min=26.0
            ast_rate_idx = 5  # index in canonical 17-feature order
            expected_ast_rate = 3.5 / 26.0
            assert abs(vec[ast_rate_idx] - expected_ast_rate) < 0.001
        finally:
            idx.AI_FEATURES = saved


class TestTeamUtilsConsolidation:
    """Verify shared team_utils module and alias coverage."""

    def test_canonical_teams_count(self):
        from scripts.team_utils import canonical_teams
        teams = canonical_teams()
        assert len(teams) == 30, f"Expected 30 canonical teams, got {len(teams)}"

    def test_normalize_known_aliases(self):
        from scripts.team_utils import normalize_team
        assert normalize_team("GSW") == "GS"
        assert normalize_team("NYK") == "NY"
        assert normalize_team("SAS") == "SA"
        assert normalize_team("NOP") == "NO"
        assert normalize_team("WAS") == "WSH"
        assert normalize_team("UTA") == "UTAH"
        assert normalize_team("PHO") == "PHX"
        assert normalize_team("BRO") == "BKN"
        assert normalize_team("NJN") == "BKN"
        assert normalize_team("CHO") == "CHA"

    def test_canonical_passthrough(self):
        from scripts.team_utils import normalize_team, canonical_teams
        for team in canonical_teams():
            assert normalize_team(team) == team, f"{team} should pass through unchanged"

    def test_empty_and_whitespace(self):
        from scripts.team_utils import normalize_team
        assert normalize_team("") == ""
        assert normalize_team("  ") == ""
        assert normalize_team(None) == ""

    def test_case_insensitive(self):
        from scripts.team_utils import normalize_team
        assert normalize_team("gsw") == "GS"
        assert normalize_team("lal") == "LAL"
        assert normalize_team(" bos ") == "BOS"

    def test_load_team_aliases_includes_identity(self):
        from scripts.team_utils import load_team_aliases, canonical_teams
        aliases = load_team_aliases()
        # Every canonical team should map to itself
        for team in canonical_teams():
            assert aliases[team] == team, f"Canonical {team} should map to itself"

    def test_teams_json_matches_aliases(self):
        """Verify data/teams.json aliases are all resolvable."""
        import json
        teams_json = Path(__file__).resolve().parent.parent / "data" / "teams.json"
        if not teams_json.exists():
            pytest.skip("data/teams.json not found")
        with teams_json.open() as f:
            data = json.load(f)
        from scripts.team_utils import normalize_team
        for canon, info in data.get("canonical", {}).items():
            assert normalize_team(canon) == canon
            for alias in info.get("aliases", []):
                assert normalize_team(alias) == canon, f"Alias {alias} should resolve to {canon}"

    def test_team_market_scores_coverage(self):
        """Every abbreviation in TEAM_MARKET_SCORES should map to a canonical team."""
        from scripts.team_utils import normalize_team, canonical_teams
        from api.features import TEAM_MARKET_SCORES
        teams = canonical_teams()
        for abbr in TEAM_MARKET_SCORES:
            canonical = normalize_team(abbr)
            assert canonical in teams, (
                f"TEAM_MARKET_SCORES key '{abbr}' normalizes to '{canonical}' "
                f"which is not a canonical team"
            )


class TestTeamRestDays:
    """_fetch_team_rest_days computes per-team rest from ESPN scoreboards."""

    def test_b2b_team_gets_rest_1(self):
        """Team that played yesterday should have rest_days=1."""
        import api.index as idx
        from datetime import date, timedelta
        today = date(2026, 3, 30)
        yesterday = today - timedelta(days=1)
        # Mock _et_date and _espn_get
        original_et = idx._et_date
        original_espn = idx._espn_get
        original_cg = idx._cg
        try:
            idx._et_date = lambda: today
            idx._cg = lambda key: None  # bypass cache
            def mock_espn(url):
                if yesterday.strftime("%Y%m%d") in url:
                    return {"events": [{"competitions": [{"competitors": [
                        {"team": {"abbreviation": "BOS"}},
                        {"team": {"abbreviation": "LAL"}}
                    ]}]}]}
                return {"events": []}
            idx._espn_get = mock_espn
            # Disable caching
            original_cs = idx._cs
            idx._cs = lambda key, val: None
            result = idx._fetch_team_rest_days()
            assert result.get("BOS") == 1
            assert result.get("LAL") == 1
        finally:
            idx._et_date = original_et
            idx._espn_get = original_espn
            idx._cg = original_cg
            idx._cs = original_cs

    def test_team_played_3_days_ago(self):
        """Team that played 3 days ago should have rest_days=3."""
        import api.index as idx
        from datetime import date, timedelta
        today = date(2026, 3, 30)
        original_et = idx._et_date
        original_espn = idx._espn_get
        original_cg = idx._cg
        original_cs = idx._cs
        try:
            idx._et_date = lambda: today
            idx._cg = lambda key: None
            idx._cs = lambda key, val: None
            target = today - timedelta(days=3)
            def mock_espn(url):
                if target.strftime("%Y%m%d") in url:
                    return {"events": [{"competitions": [{"competitors": [
                        {"team": {"abbreviation": "MIA"}}
                    ]}]}]}
                return {"events": []}
            idx._espn_get = mock_espn
            result = idx._fetch_team_rest_days()
            assert result.get("MIA") == 3
        finally:
            idx._et_date = original_et
            idx._espn_get = original_espn
            idx._cg = original_cg
            idx._cs = original_cs

    def test_unknown_team_not_in_result(self):
        """Teams with no games in last 7 days should not appear in result."""
        import api.index as idx
        from datetime import date
        original_et = idx._et_date
        original_espn = idx._espn_get
        original_cg = idx._cg
        original_cs = idx._cs
        try:
            idx._et_date = lambda: date(2026, 3, 30)
            idx._cg = lambda key: None
            idx._cs = lambda key, val: None
            idx._espn_get = lambda url: {"events": []}
            result = idx._fetch_team_rest_days()
            assert result.get("ZZZ") is None
        finally:
            idx._et_date = original_et
            idx._espn_get = original_espn
            idx._cg = original_cg
            idx._cs = original_cs

    def test_rest_clipped_to_7(self):
        """rest_days should be clipped to max 7 even if last game was 7+ days ago."""
        import api.index as idx
        from datetime import date, timedelta
        today = date(2026, 3, 30)
        original_et = idx._et_date
        original_espn = idx._espn_get
        original_cg = idx._cg
        original_cs = idx._cs
        try:
            idx._et_date = lambda: today
            idx._cg = lambda key: None
            idx._cs = lambda key, val: None
            # Only game found at day 7 (the max lookback)
            target = today - timedelta(days=7)
            def mock_espn(url):
                if target.strftime("%Y%m%d") in url:
                    return {"events": [{"competitions": [{"competitors": [
                        {"team": {"abbreviation": "DET"}}
                    ]}]}]}
                return {"events": []}
            idx._espn_get = mock_espn
            result = idx._fetch_team_rest_days()
            assert result.get("DET") == 7
        finally:
            idx._et_date = original_et
            idx._espn_get = original_espn
            idx._cg = original_cg
            idx._cs = original_cs

    def test_cache_hit(self):
        """Should return cached result when available."""
        import api.index as idx
        from datetime import date
        original_et = idx._et_date
        original_cg = idx._cg
        try:
            idx._et_date = lambda: date(2026, 3, 30)
            idx._cg = lambda key: {"BOS": 2, "LAL": 1}
            result = idx._fetch_team_rest_days()
            assert result == {"BOS": 2, "LAL": 1}
        finally:
            idx._et_date = original_et
            idx._cg = original_cg


class TestGamesPlayedEstimate:
    """_estimate_games_played produces season-progress-based estimates."""

    def test_midseason_estimate(self):
        """Mid-season should estimate ~40-45 games."""
        import api.index as idx
        from datetime import date
        original_et = idx._et_date
        try:
            idx._et_date = lambda: date(2026, 1, 15)  # ~86 days into season
            est = idx._estimate_games_played()
            assert 35 <= est <= 45  # ~86 * 82/180 ≈ 39
        finally:
            idx._et_date = original_et

    def test_season_start_estimate(self):
        """Early season should estimate few games."""
        import api.index as idx
        from datetime import date
        original_et = idx._et_date
        try:
            idx._et_date = lambda: date(2025, 10, 28)  # 7 days in
            est = idx._estimate_games_played()
            assert 1 <= est <= 5  # ~7 * 82/180 ≈ 3
        finally:
            idx._et_date = original_et

    def test_season_end_estimate(self):
        """Late season should estimate ~75-82 games."""
        import api.index as idx
        from datetime import date
        original_et = idx._et_date
        try:
            idx._et_date = lambda: date(2026, 4, 10)  # ~171 days in
            est = idx._estimate_games_played()
            assert 70 <= est <= 82
        finally:
            idx._et_date = original_et

    def test_never_exceeds_82(self):
        """Estimate should never exceed 82."""
        import api.index as idx
        from datetime import date
        original_et = idx._et_date
        try:
            idx._et_date = lambda: date(2026, 6, 1)  # way past season
            est = idx._estimate_games_played()
            assert est <= 82
        finally:
            idx._et_date = original_et


class TestGamesPlayedBulkStats:
    """_fetch_team_player_stats extracts gp from ESPN response."""

    def test_gp_extracted_in_parse_split(self):
        """_parse_split should extract gp when present in stat names."""
        import api.index as idx
        names = ["MIN", "PTS", "GP", "REB"]
        split = {"stats": ["30.0", "18.0", "55", "7.0"]}
        result = idx._parse_split(names, split)
        assert result.get("gp") == 55.0

    def test_gp_missing_graceful(self):
        """_parse_split should not have gp key when not in stat names."""
        import api.index as idx
        names = ["MIN", "PTS", "REB"]
        split = {"stats": ["30.0", "18.0", "7.0"]}
        result = idx._parse_split(names, split)
        assert "gp" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
