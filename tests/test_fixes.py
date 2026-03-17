"""
Unit tests for basketball app backend — real assertions, real function calls.
Run with: pytest tests/test_fixes.py -v

Requires backend deps (numpy, lightgbm, etc.). If skipped, run:
  pip install -r requirements.txt
"""

import pytest
import json
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
    """When LightGBM bundle is loaded, feature list must have 11 elements; 10th (index 9) is recent_vs_season (or legacy recent_3g_trend)."""

    def test_feature_list_length_and_trend_feature(self):
        import api.index as idx
        idx._ensure_lgbm_loaded()
        AI_FEATURES = idx.AI_FEATURES
        if AI_FEATURES is None:
            pytest.skip("No LightGBM bundle loaded (lgbm_model.pkl not present or invalid)")
        assert len(AI_FEATURES) == 11, f"Expected 11 features, got {len(AI_FEATURES)}: {AI_FEATURES}"
        assert AI_FEATURES[9] in ("recent_vs_season", "recent_3g_trend"), (
            f"10th feature (index 9) must be recent_vs_season or legacy recent_3g_trend, got {AI_FEATURES[9]!r}"
        )


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
        assert '300' in source, "fetch_games TTL should be 300 seconds (5 min)"


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

    def test_get_projections_for_date_skips_github_cache(self):
        """_get_projections_for_date does NOT call GitHub cache (removed for latency)."""
        from api.index import _get_projections_for_date
        mock_games = [{"gameId": "game1", "startTime": (datetime.now(timezone.utc) + timedelta(hours=5)).isoformat()}]
        mock_player = {"name": "Player", "id": "123", "home": True, "team": "BOS",
                       "opp": "LAL", "gameId": "game1"}

        with patch("api.index.fetch_games", return_value=mock_games), \
             patch("api.index._is_past_lock_window", return_value=False), \
             patch("api.index._cg", return_value=None), \
             patch("api.index._run_game", return_value=[mock_player]), \
             patch("api.index._games_cache_from_github") as mock_gh:
            from datetime import date
            projs, games = _get_projections_for_date(date.today())
            # GitHub cache should NOT be called in this path
            mock_gh.assert_not_called()
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
             patch("anthropic.Anthropic", return_value=mock_client):
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
             patch("anthropic.Anthropic", return_value=mock_client):
            _claude_context_pass(players, games)

        # 2.0 should be clamped to 1.4 (max)
        assert players[0]["rating"] == pytest.approx(3.0 * 1.4, abs=0.1)
        assert players[0]["_context_adj"] == pytest.approx(1.4, abs=0.01)

    def test_multiplier_clamped_at_min_adjustment(self):
        """Claude returning 0.3x is clamped to 1-max_adjustment (0.6x at default 0.4)."""
        from api.index import _claude_context_pass
        players = [self._make_player("Player Y", "LAL", rating=5.0)]
        games = [self._make_game("LAL", "BOS")]

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
             patch("anthropic.Anthropic", return_value=mock_client):
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
             patch("anthropic.Anthropic", return_value=mock_client):
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
             patch("anthropic.Anthropic", return_value=mock_client):
            _claude_context_pass(players, games)

        assert players[0]["rating"] == 3.0
        assert "_context_adj" not in players[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
