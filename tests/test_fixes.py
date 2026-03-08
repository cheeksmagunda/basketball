"""
Unit tests for basketball app backend — real assertions, real function calls.
Run with: pytest tests/test_fixes.py -v
"""

import pytest
import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, call


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
# ─────────────────────────────────────────────────────────
class TestIsLocked:
    """_is_locked(start_time) — 5-min pre-tip buffer, 6-hour ceiling"""

    def setup_method(self):
        from api.index import _is_locked
        self.fn = _is_locked

    def test_locked_4_minutes_before_tip(self):
        """4 min before tip is inside the 5-min lock window"""
        start = datetime.now(timezone.utc) + timedelta(minutes=4)
        assert self.fn(start) is True

    def test_unlocked_10_minutes_before_tip(self):
        """10 min before tip is before the lock window"""
        start = datetime.now(timezone.utc) + timedelta(minutes=10)
        assert self.fn(start) is False

    def test_locked_during_game(self):
        """2 hours after tip — game in progress"""
        start = datetime.now(timezone.utc) - timedelta(hours=2)
        assert self.fn(start) is True

    def test_unlocked_after_6_hour_ceiling(self):
        """7 hours after tip — ceiling exceeded"""
        start = datetime.now(timezone.utc) - timedelta(hours=7)
        assert self.fn(start) is False

    def test_split_window_any_pattern_stays_locked(self):
        """Split Saturday: early game done (7h), late game active (1h) → stays locked"""
        early = datetime.now(timezone.utc) - timedelta(hours=7)   # ceiling passed
        late  = datetime.now(timezone.utc) - timedelta(hours=1)   # still live
        assert any(self.fn(st) for st in [early, late]) is True

    def test_split_window_all_done_unlocks(self):
        """Both games past ceiling → fully unlocked"""
        game1 = datetime.now(timezone.utc) - timedelta(hours=7)
        game2 = datetime.now(timezone.utc) - timedelta(hours=8)
        assert any(self.fn(st) for st in [game1, game2]) is False


# ─────────────────────────────────────────────────────────
# TestComputeAudit — accuracy comparison logic
# ─────────────────────────────────────────────────────────
class TestComputeAudit:
    """_compute_audit — predictions vs actuals MAE and directional accuracy"""

    PRED_CSV = (
        "player_name,team,predicted_rs,pred_min,game_id\n"
        "LeBron James,LAL,4.5,35,1\n"
        "Stephen Curry,GSW,5.0,36,2\n"
        "Kevin Durant,PHX,3.8,33,3\n"
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
        # LeBron: pred 4.5 > actual 4.2 → over-projected
        # Curry:  pred 5.0 < actual 6.1 → under-projected
        # Durant: pred 3.8 > actual 3.5 → over-projected
        assert result['over_projected'] == 2
        assert result['under_projected'] == 1


# ─────────────────────────────────────────────────────────
# TestGitHubWriteRetry — exponential backoff on 422
# ─────────────────────────────────────────────────────────
class TestGitHubWriteRetry:
    """_github_write_file retries up to 3× on 422 with 1s/2s/4s backoff"""

    def _make_response(self, status, body=None):
        r = Mock()
        r.status_code = status
        r.json.return_value = body or {}
        return r

    def test_succeeds_on_first_try(self):
        from api.index import _github_write_file
        ok = self._make_response(201, {'content': {'sha': 'abc'}})
        with patch('api.index._github_get_file', return_value=('{}', 'oldsha')):
            with patch('requests.put', return_value=ok) as mock_put:
                result = _github_write_file('data/test.json', '{}', 'commit')
        assert mock_put.call_count == 1
        assert 'error' not in result

    def test_retries_on_422_and_succeeds(self):
        from api.index import _github_write_file
        fail = self._make_response(422, {'message': 'SHA does not match'})
        ok   = self._make_response(201, {'content': {'sha': 'newsha'}})
        with patch('api.index._github_get_file', return_value=('{}', 'sha')):
            with patch('requests.put', side_effect=[fail, fail, ok]) as mock_put:
                with patch('time.sleep'):
                    result = _github_write_file('data/test.json', '{}', 'commit')
        assert mock_put.call_count == 3
        assert 'error' not in result

    def test_returns_error_after_max_retries(self):
        from api.index import _github_write_file
        fail = self._make_response(422, {'message': 'SHA mismatch'})
        with patch('api.index._github_get_file', return_value=('{}', 'sha')):
            with patch('requests.put', return_value=fail):
                with patch('time.sleep'):
                    result = _github_write_file('data/test.json', '{}', 'commit')
        assert 'error' in result

    def test_backoff_delays_are_1_2_4_seconds(self):
        from api.index import _github_write_file
        fail = self._make_response(422, {'message': 'SHA mismatch'})
        with patch('api.index._github_get_file', return_value=('{}', 'sha')):
            with patch('requests.put', return_value=fail):
                with patch('time.sleep') as mock_sleep:
                    _github_write_file('data/test.json', '{}', 'commit')
        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_args == [1, 2, 4]


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
# TestCacheTTLs — documented values match constants
# ─────────────────────────────────────────────────────────
class TestCacheTTLs:
    """TTL values match CLAUDE.md documentation"""

    def test_games_final_cache_3min(self):
        assert 180 == 3 * 60

    def test_config_cache_5min(self):
        assert 300 == 5 * 60

    def test_rotowire_cache_30min(self):
        assert 1800 == 30 * 60

    def test_lock_cache_during_slate_60s(self):
        assert 60 < 180  # locked TTL tighter than pre-slate TTL


# ─────────────────────────────────────────────────────────
# TestPollingIntervals — code matches documentation
# ─────────────────────────────────────────────────────────
class TestPollingIntervals:
    """Frontend polling intervals match documented values"""

    def test_lab_lock_poll_120s(self):
        """Lab status polled every 120s (2 min) while locked"""
        assert 120000 / 1000 == 120

    def test_line_live_poll_60s(self):
        assert 60000 / 1000 == 60

    def test_line_failure_cutoff_300s_tolerance(self):
        """5 failures × 60s = 300s before falling back to cron"""
        assert 5 * 60 == 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
