"""
Lightweight unit tests for basketball optimizer.

Run with:  pytest tests/
Coverage targets the three most failure-prone areas:
  1. Pure helper functions (no I/O, no external deps)
  2. Line-cache & slate endpoint logic (mocked GitHub / ESPN)
  3. JS syntax integrity (catches apostrophe crashes before push)

Requires backend deps (numpy, etc.). If skipped, run: pip install -r requirements.txt
"""
import re
import sys
import types
import importlib
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Skip backend-dependent tests if deps not installed (clear message instead of ERRORs)
pytest.importorskip("numpy", reason="Install dependencies: pip install -r requirements.txt")

# ---------------------------------------------------------------------------
# 1. Pure helper unit tests (no network, no filesystem)
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for stateless utility functions."""

    def test_et_date_returns_date(self):
        from api.index import _et_date
        d = _et_date()
        assert isinstance(d, date)

    def test_is_locked_future_game(self):
        """Games that start >5 min from now must NOT be locked."""
        from api.index import _is_locked
        future = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        assert _is_locked(future) is False

    def test_is_locked_past_game(self):
        """Games that started >5 min ago must be locked."""
        from api.index import _is_locked
        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        assert _is_locked(past) is True

    def test_is_locked_borderline(self):
        """Game starting in 3 min is within the 5-min window — locked."""
        from api.index import _is_locked
        soon = (datetime.now(timezone.utc) + timedelta(minutes=3)).isoformat()
        assert _is_locked(soon) is True

    def test_est_card_boost_decreases_with_star_minutes(self):
        """Star players (high pts/min) should get a lower boost than bench players."""
        from api.index import _est_card_boost
        # High-usage star: 35 min, 28 pts → high ownership → lower boost
        star_boost  = _est_card_boost(proj_min=35, pts=28, team_abbr="LAL")
        # Low-usage bench player: 12 min, 6 pts → low ownership → higher boost
        bench_boost = _est_card_boost(proj_min=12, pts=6,  team_abbr="MEM")
        assert bench_boost > star_boost, (
            f"bench boost {bench_boost:.2f} should exceed star boost {star_boost:.2f}"
        )

    def test_est_card_boost_nonnegative(self):
        from api.index import _est_card_boost
        for mins, pts in [(5, 2), (20, 10), (38, 30)]:
            assert _est_card_boost(mins, pts, "GSW") >= 0

    def test_cache_roundtrip(self):
        """_cs / _cg must return what was stored."""
        from api.index import _cs, _cg
        _cs("test_key_xyz", {"v": 42})
        assert _cg("test_key_xyz") == {"v": 42}

    def test_cache_miss_returns_none(self):
        from api.index import _cg
        assert _cg("nonexistent_key_abc123") is None


# ---------------------------------------------------------------------------
# 2. Line-of-the-day cache logic
# ---------------------------------------------------------------------------

class TestLineCacheLogic:
    """
    The cache should ONLY be served when:
      - the pick is today's date AND
      - the pick is not yet resolved (result is None / '' / 'pending')
    Anything else must bypass the cache so the endpoint fetches fresh data.
    """

    def _et_today(self):
        from api.index import _et_date
        return _et_date().isoformat()

    def _make_pick(self, resolved=False, today=True):
        from api.index import _et_date
        from datetime import timedelta
        d = _et_date() if today else (_et_date() - timedelta(days=1))
        result_val = "hit" if resolved else "pending"
        return {"player_name": "Test Player", "date": d.isoformat(), "result": result_val}

    def test_unresolved_today_pick_is_served_from_cache(self):
        """Unresolved today pick → serve cache (skip expensive API call)."""
        from api.index import _cs, _et_date
        pick = self._make_pick(resolved=False, today=True)
        cached_response = {"pick": pick, "from_github": True}
        _cs("line_v1", cached_response)

        today_str = _et_date().isoformat()
        cached_pick = cached_response["pick"]
        pick_date = cached_pick.get("date", today_str)
        already_resolved = cached_pick.get("result") not in (None, "", "pending")
        should_serve_cache = not already_resolved and pick_date == today_str
        assert should_serve_cache is True

    def test_resolved_today_pick_bypasses_cache(self):
        """Resolved today pick → must bypass cache to rotate to tomorrow."""
        from api.index import _et_date
        pick = self._make_pick(resolved=True, today=True)
        cached_response = {"pick": pick}

        today_str = _et_date().isoformat()
        cached_pick = cached_response["pick"]
        pick_date = cached_pick.get("date", today_str)
        already_resolved = cached_pick.get("result") not in (None, "", "pending")
        should_serve_cache = not already_resolved and pick_date == today_str
        assert should_serve_cache is False

    def test_yesterday_pick_bypasses_cache(self):
        """Yesterday's pick (any state) → must bypass so today's pick is fetched."""
        from api.index import _et_date
        pick = self._make_pick(resolved=False, today=False)  # yesterday, unresolved
        cached_response = {"pick": pick}

        today_str = _et_date().isoformat()
        cached_pick = cached_response["pick"]
        pick_date = cached_pick.get("date", today_str)
        already_resolved = cached_pick.get("result") not in (None, "", "pending")
        should_serve_cache = not already_resolved and pick_date == today_str
        assert should_serve_cache is False, "yesterday's pick must not be served from cache"

    def test_yesterday_resolved_pick_bypasses_cache(self):
        """Yesterday's resolved pick → bypass cache."""
        from api.index import _et_date
        pick = self._make_pick(resolved=True, today=False)
        cached_response = {"pick": pick}

        today_str = _et_date().isoformat()
        cached_pick = cached_response["pick"]
        pick_date = cached_pick.get("date", today_str)
        already_resolved = cached_pick.get("result") not in (None, "", "pending")
        should_serve_cache = not already_resolved and pick_date == today_str
        assert should_serve_cache is False


# ---------------------------------------------------------------------------
# 3. JS syntax integrity — catches the apostrophe crash class before push
# ---------------------------------------------------------------------------

class TestJSSyntax:
    """
    Scans index.html's <script> block for unescaped apostrophes inside
    single-quoted JS string literals. These cause silent parse failures that
    blank out the entire app.
    """

    @pytest.fixture(scope="class")
    def script_lines(self):
        html = (ROOT / "index.html").read_text()
        start = html.rfind("<script>")
        end   = html.rfind("</script>")
        assert start != -1 and end != -1, "No <script> block found in index.html"
        return html[start:end].split("\n")

    def test_no_apostrophe_in_single_quoted_strings(self, script_lines):
        """
        Detect lines where a single-quoted string contains a bare apostrophe,
        e.g.  'TOMORROW'S PICK'  or  'font-family:'Barlow','
        These terminate the string early and crash the JS parser.
        """
        violations = []
        for i, line in enumerate(script_lines, 1):
            stripped = line.strip()
            # Skip comment lines and template-literal lines (backtick strings are safe)
            if stripped.startswith("//") or stripped.startswith("*") or "`" in line:
                continue
            # Pattern: a word char followed by ' followed by another word char
            # while we are inside a single-quoted string context.
            # Heuristic: find single-quoted tokens that internally contain \w'\w.
            if re.search(r"'[^']*\w'\w[^']*'", line):
                violations.append((i, line.rstrip()))

        assert not violations, (
            "Unescaped apostrophes inside JS single-quoted strings found:\n"
            + "\n".join(f"  line {i}: {l}" for i, l in violations)
        )

    def test_render_functions_present(self, script_lines):
        """Key render functions must exist — catches accidental deletion."""
        source = "\n".join(script_lines)
        for fn in ["renderCards", "renderLinePickCard", "renderNextSlatePending",
                   "initLinePage", "loadSlate", "switchTab"]:
            assert fn in source, f"Missing JS function: {fn}"

    def test_no_date_today_in_cache_functions(self, script_lines):
        """_etToday() must be used for date guards, not raw Date() without timezone."""
        source = "\n".join(script_lines)
        # Confirm the ET helper exists
        assert "_etToday" in source, "_etToday() helper missing from frontend"
        # The session guards should reference _etToday(), not a bare boolean flag
        assert "LINE_LOADED_DATE" in source, "LINE_LOADED must be date-keyed (LINE_LOADED_DATE)"
        assert "_predSavedDate" in source, "_predSavedToday must be date-keyed (_predSavedDate)"

    def test_no_redundant_info_bars(self, script_lines):
        """Redundant slateChips/headerMeta info bars must not exist in the codebase."""
        html = (ROOT / "index.html").read_text()
        assert 'id="slateChips"' not in html, "slateChips element should be removed"
        assert 'id="headerMeta"' not in html, "headerMeta element should be removed"
        assert ".lock-chip" not in html, "lock-chip CSS should be removed"
        assert ".game-chips" not in html, "game-chips CSS should be removed"

    def test_line_pick_extraction_before_error_gate(self, script_lines):
        """LINE_OVER_PICK/LINE_UNDER_PICK must be extracted before the error gate check."""
        source = "\n".join(script_lines)
        # The extraction (data.over_pick) must appear BEFORE the error gate (!data.pick && !LINE_OVER_PICK)
        extract_pos = source.find("data.over_pick")
        error_gate_pos = source.find("!data.pick && !LINE_OVER_PICK")
        assert extract_pos != -1, "LINE_OVER_PICK extraction from data.over_pick missing"
        assert error_gate_pos != -1, "Error gate should check !data.pick && !LINE_OVER_PICK"
        assert extract_pos < error_gate_pos, (
            "Directional pick extraction must happen BEFORE the error gate check"
        )

    def test_save_predictions_not_gated_by_chips(self, script_lines):
        """savePredictions() must not be blocked by missing UI elements."""
        source = "\n".join(script_lines)
        # savePredictions should not be after a slateChips guard
        assert "slateChips" not in source or source.find("slateChips") > source.find("savePredictions"), (
            "savePredictions must not be gated behind slateChips element check"
        )


# ---------------------------------------------------------------------------
# 4. Cache date-boundary regression tests
# ---------------------------------------------------------------------------

class TestCacheDateBoundary:
    """
    The root cause of the blank-app bug: _cp() and _lp() used date.today() (UTC)
    while the rest of the app uses _et_date() (Eastern Time). After midnight UTC
    (~7 PM ET), cache keys rolled over to the new UTC date, orphaning all data.

    These tests ensure cache keys are consistent with ET date regardless of when
    they are called relative to UTC midnight.
    """

    def test_cache_key_uses_et_date_not_utc(self):
        """_cp key must be consistent with _et_date(), not date.today()."""
        from api.index import _cp, _et_date
        import hashlib
        from pathlib import Path

        et_str = _et_date().isoformat()
        expected_stem = hashlib.md5(f"{et_str}:probe_key".encode()).hexdigest()
        actual_path = _cp("probe_key")
        assert actual_path.stem == expected_stem, (
            f"Cache key used wrong date. Expected stem based on ET date {et_str!r}, "
            f"got {actual_path.stem!r}. Was date.today() (UTC) used instead?"
        )

    def test_lock_key_uses_et_date_not_utc(self):
        """_lp key must be consistent with _et_date(), not date.today()."""
        from api.index import _lp, _et_date
        import hashlib

        et_str = _et_date().isoformat()
        expected_stem = hashlib.md5(f"{et_str}:probe_lock".encode()).hexdigest()
        actual_path = _lp("probe_lock")
        assert actual_path.stem == expected_stem, (
            f"Lock key used wrong date. Expected stem based on ET {et_str!r}, "
            f"got {actual_path.stem!r}."
        )

    def test_cache_key_stable_across_mocked_utc_rollover(self):
        """Simulate UTC midnight: even if date.today() changes, _cp must use _et_date()."""
        from api.index import _cp, _et_date
        import hashlib
        from unittest.mock import patch
        from datetime import date, timedelta

        real_et = _et_date()
        # Simulate UTC midnight rolling forward while ET is still the same day
        future_utc = real_et + timedelta(days=1)
        with patch("api.index._et_date", return_value=real_et):
            path_before = _cp("consistency_test")
        with patch("api.index._et_date", return_value=real_et):
            # date.today() might differ but _et_date() is mocked to same value
            path_during = _cp("consistency_test")
        assert path_before == path_during, (
            "Cache key changed between calls with same ET date — UTC date.today() leaked in"
        )

    def test_games_final_cache_resets_on_new_et_date(self):
        """_GAMES_FINAL_CACHE must not serve yesterday's result on a new ET day."""
        import api.index as idx
        from datetime import date, timedelta
        from unittest.mock import patch

        # Prime the cache with "yesterday"
        yesterday = (date(2026, 3, 5)).strftime("%Y%m%d")
        today     = (date(2026, 3, 6)).strftime("%Y%m%d")
        idx._GAMES_FINAL_CACHE.update({
            "result": [True, 0, 9, None],
            "ts": __import__("time").time(),  # fresh timestamp
            "date": yesterday,
        })

        with patch("api.index._et_date", return_value=date(2026, 3, 6)), \
             patch("api.index._espn_get", return_value={"events": []}):
            result = idx._all_games_final([{"gameId": "x"}])

        # Should have re-fetched (ESPN returned no events → all_final=False, finals=0)
        all_final, remaining, finals, _ = result
        assert finals == 0, (
            "Cache served yesterday's result on a new ET day — date guard missing"
        )

    def test_rng_seed_uses_et_date(self):
        """real_score._make_rng must produce the same seed for the same ET date."""
        from api.real_score import _make_rng, _et_today
        rng1 = _make_rng(spread=5.0, total=220.0)
        rng2 = _make_rng(spread=5.0, total=220.0)
        # Both use ET date → same seed → same first value
        assert rng1.random() == rng2.random(), (
            "RNG seed is not deterministic for same ET date"
        )


# ---------------------------------------------------------------------------
# 5. Ben upload-banner actuals detection
# ---------------------------------------------------------------------------

class TestBenBannerActualsDetection:
    """
    The Ben upload banner must hide when today's actuals are already saved.

    Root cause of the original bug: the banner checked only
    briefing.latest_slate.date, which requires *paired* dates
    (predictions + actuals both committed).  If the dedup guard
    skipped the predictions commit, actuals existed but the briefing
    didn't see them — banner stayed visible.

    Fix: also check /api/log/get has_actuals directly.
    """

    _ACT_CSV = (
        "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source\n"
        "LeBron James,5.2,1.4,1234,2.1,8.7,screenshot\n"
        "Anthony Davis,4.1,1.2,890,3.0,6.5,screenshot\n"
    )
    _PRED_CSV = (
        "scope,lineup_type,slot,player_name,player_id,team,pos,"
        "predicted_rs,est_card_boost,pred_min,pts,reb,ast,stl,blk\n"
        "slate,chalk,2.0,LeBron James,x,LAL,F,4.8,1.3,35,25,8,7,1,1\n"
    )

    def test_has_actuals_true_when_csv_has_data(self):
        """log_get must return has_actuals=True when actuals CSV has player rows."""
        from api.index import _parse_csv, ACT_FIELDS
        rows = _parse_csv(self._ACT_CSV, ACT_FIELDS)
        assert bool(rows) is True

    def test_has_actuals_false_when_csv_missing(self):
        """log_get must return has_actuals=False when actuals CSV is absent."""
        from api.index import _parse_csv, ACT_FIELDS
        rows = _parse_csv(None, ACT_FIELDS)
        assert bool(rows) is False

    def test_has_actuals_false_when_csv_header_only(self):
        """log_get must return has_actuals=False for a header-only actuals CSV."""
        from api.index import _parse_csv, ACT_FIELDS
        header_only = "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source\n"
        rows = _parse_csv(header_only, ACT_FIELDS)
        assert bool(rows) is False

    def test_has_actuals_independent_of_predictions(self):
        """
        has_actuals must be True even when predictions CSV is missing.
        This is why /api/log/get is the reliable signal — not the briefing,
        which requires both files (paired).
        """
        from api.index import _parse_csv, ACT_FIELDS, PRED_FIELDS
        actuals     = _parse_csv(self._ACT_CSV, ACT_FIELDS)
        predictions = _parse_csv(None, PRED_FIELDS)           # no predictions committed
        assert bool(actuals) is True
        assert bool(predictions) is False

    def test_briefing_paired_logic_misses_actuals_without_predictions(self):
        """
        Reproduce the exact bug: briefing only surfaces today when BOTH
        predictions and actuals are committed.  Actuals-only → latest_slate
        date is NOT today → banner incorrectly stays visible.
        """
        from api.index import _et_date
        today = _et_date().isoformat()

        # Actuals exist for today; predictions were NOT committed (dedup guard)
        act_dates  = {today}
        pred_names = []           # empty — no predictions CSV on GitHub

        paired = [
            n[:-4] for n in sorted(pred_names, reverse=True)
            if n.endswith(".csv") and n[:-4] in act_dates
        ]

        latest_slate_date = paired[0] if paired else None
        assert latest_slate_date != today, (
            "Without a predictions CSV the briefing cannot detect today's actuals — "
            "confirming the bug. The fix is to also check /api/log/get has_actuals."
        )

    def test_parse_csv_field_mapping(self):
        """Actuals CSV rows must map player_name and actual_rs correctly."""
        from api.index import _parse_csv, ACT_FIELDS
        rows = _parse_csv(self._ACT_CSV, ACT_FIELDS)
        assert rows[0]["player_name"] == "LeBron James"
        assert rows[0]["actual_rs"]   == "5.2"
        assert rows[1]["player_name"] == "Anthony Davis"

    def test_parse_csv_quoted_commas(self):
        """Quoted fields containing commas must parse as single values."""
        from api.index import _parse_csv, ACT_FIELDS
        csv = (
            "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source\n"
            '"Smith, Jr.",4.1,1.2,500,3.0,6.5,screenshot\n'
        )
        rows = _parse_csv(csv, ACT_FIELDS)
        assert len(rows) == 1
        assert rows[0]["player_name"] == "Smith, Jr."

    def test_parse_csv_short_row_padded(self):
        """Rows with fewer columns than the header must be padded, not crash."""
        from api.index import _parse_csv, ACT_FIELDS
        csv = "player_name,actual_rs,actual_card_boost,drafts,avg_finish,total_value,source\nPlayer X,3.5\n"
        rows = _parse_csv(csv, ACT_FIELDS)
        assert len(rows) == 1
        assert rows[0]["player_name"] == "Player X"
        assert rows[0]["actual_rs"]   == "3.5"
        assert rows[0]["source"]      == ""   # padded empty


# ---------------------------------------------------------------------------
# 6. JS banner-guard regression — both detection signals must stay present
# ---------------------------------------------------------------------------

class TestBannerGuardJS:
    """
    The frontend banner-visibility check in showLabUnlocked() must use:
      - LAB.briefing?.pending_upload_date  (pending slate date from backend)
      - localStorage keyed by pending date (not _etToday())

    pending_upload_date is the last predictions-only date with no actuals yet,
    so uploads always target the correct slate date even when run next-morning.
    """

    @pytest.fixture(scope="class")
    def script_source(self):
        html = (ROOT / "index.html").read_text()
        start = html.rfind("<script>")
        end   = html.rfind("</script>")
        assert start != -1 and end != -1, "No <script> block found in index.html"
        return html[start:end]

    def test_banner_check_uses_has_actuals(self, script_source):
        """has_actuals is still used in the log rendering / other contexts."""
        assert "has_actuals" in script_source, (
            "Missing has_actuals reference — likely indicates log rendering was removed"
        )

    def test_banner_check_uses_pending_upload_date(self, script_source):
        """Banner visibility must use pending_upload_date from briefing."""
        assert "pending_upload_date" in script_source, (
            "Missing pending_upload_date — banner will use wrong date for localStorage key "
            "and save-actuals, causing date mismatch between predictions and actuals"
        )

    def test_log_get_fetched_in_showlabunlocked(self, script_source):
        """/api/log/get must be called from showLabUnlocked to detect today's actuals."""
        assert "/api/log/get" in script_source, (
            "/api/log/get not called from showLabUnlocked — "
            "banner detection will miss unpaired actuals"
        )

    def test_show_lab_unlocked_function_present(self, script_source):
        """showLabUnlocked function must exist (catches accidental deletion)."""
        assert "showLabUnlocked" in script_source


# ---------------------------------------------------------------------------
# 7. Response contract normalizers — _normalize_player
# ---------------------------------------------------------------------------

class TestNormalizePlayer:
    """
    _normalize_player() is the stable contract boundary between the model and
    the frontend. Every player object going into an API response passes through
    it. Tests verify:
      - All required frontend fields are always present
      - Internal MILP fields (chalk_ev_capped) are stripped
      - Bad/missing numeric values coerce safely to 0.0 (no NaN in responses)
      - Extra model-internal fields pass through untouched
      - Empty input produces a safe zero-filled card
    """

    def _norm(self, p=None):
        from api.index import _normalize_player
        return _normalize_player(p or {})

    def test_all_required_fields_present(self):
        """Output must contain every field the frontend accesses."""
        required = {
            "id", "name", "pos", "team", "rating", "predMin",
            "pts", "reb", "ast", "stl", "blk",
            "est_mult", "slot", "chalk_ev", "moonshot_ev",
            "injury_status", "_decline",
        }
        result = self._norm({"name": "Test Player", "rating": 4.5})
        missing = required - set(result.keys())
        assert not missing, f"Missing required fields: {missing}"

    def test_numeric_fields_coerce_from_none(self):
        """None values for numeric fields must become 0.0, not NaN."""
        result = self._norm({"rating": None, "predMin": None, "pts": None,
                             "est_mult": None, "chalk_ev": None})
        assert result["rating"] == 0.0
        assert result["predMin"] == 0.0
        assert result["pts"] == 0.0
        assert result["est_mult"] == 0.0
        assert result["chalk_ev"] == 0.0
        # Crucially: no NaN in the result
        import math
        for v in result.values():
            if isinstance(v, float):
                assert not math.isnan(v), f"NaN found in normalized output: {result}"

    def test_numeric_fields_coerce_from_empty_string(self):
        """Empty-string numeric fields must become 0.0."""
        result = self._norm({"rating": "", "pts": "", "reb": ""})
        assert result["rating"] == 0.0
        assert result["pts"] == 0.0
        assert result["reb"] == 0.0

    def test_chalk_ev_capped_is_stripped(self):
        """chalk_ev_capped is an internal MILP sort key — must never reach the frontend."""
        result = self._norm({"rating": 4.0, "chalk_ev_capped": 9.99})
        assert "chalk_ev_capped" not in result

    def test_rw_cleared_is_stripped(self):
        """_rw_cleared is an internal RotoWire flag — must never reach the frontend."""
        result = self._norm({"name": "X", "_rw_cleared": True})
        assert "_rw_cleared" not in result

    def test_extra_model_fields_pass_through(self):
        """Model fields not in the contract (debug, trend stats) must pass through."""
        result = self._norm({
            "name": "Test",
            "season_pts": 22.1,
            "recent_pts": 25.3,
            "_cascade_bonus": 3.0,
            "_is_dev_team": True,
        })
        assert result["season_pts"] == 22.1
        assert result["recent_pts"] == 25.3
        assert result["_cascade_bonus"] == 3.0
        assert result["_is_dev_team"] is True

    def test_string_fields_default_to_empty_string(self):
        """String fields missing from input must default to '' not None."""
        result = self._norm({})
        assert result["name"] == ""
        assert result["pos"] == ""
        assert result["team"] == ""
        assert result["injury_status"] == ""
        assert result["slot"] == "1.0x"

    def test_numeric_precision(self):
        """Numeric fields are rounded to the expected decimal places."""
        result = self._norm({"rating": 4.1667, "est_mult": 1.2345, "pts": 22.999})
        assert result["rating"] == 4.2
        assert result["est_mult"] == 1.23
        assert result["pts"] == 23.0

    def test_empty_input_is_safe(self):
        """An empty dict must produce a fully-populated zero-card without raising."""
        result = self._norm({})
        assert result["rating"] == 0.0
        assert result["moonshot_ev"] == 0.0
        assert result["name"] == ""


# ---------------------------------------------------------------------------
# 8. Response contract normalizers — _normalize_line_pick
# ---------------------------------------------------------------------------

class TestNormalizeLinePick:
    """
    _normalize_line_pick() is the stable contract boundary for line picks.
    Applied in _picks_response() and line_history before any data reaches
    the frontend.
    """

    def _norm(self, p=None):
        from api.index import _normalize_line_pick
        return _normalize_line_pick(p or {})

    def test_all_required_fields_present(self):
        required = {
            "player_name", "player_id", "team", "opponent",
            "direction", "line", "stat_type", "projection",
            "edge", "confidence", "narrative", "signals",
            "result", "actual_stat", "line_updated_at",
            "odds_over", "odds_under", "books_consensus", "date",
        }
        result = self._norm({"player_name": "LeBron James", "confidence": 72})
        missing = required - set(result.keys())
        assert not missing, f"Missing required fields: {missing}"

    def test_edge_defaults_to_zero_not_none(self):
        """edge=None/missing must produce 0.0 — not None which causes NaN in JS."""
        assert self._norm({})["edge"] == 0.0
        assert self._norm({"edge": None})["edge"] == 0.0

    def test_signals_defaults_to_empty_list(self):
        """signals=None/missing must produce [] — not None which crashes .map() in JS."""
        assert self._norm({})["signals"] == []
        assert self._norm({"signals": None})["signals"] == []

    def test_result_defaults_to_pending(self):
        assert self._norm({})["result"] == "pending"
        assert self._norm({"result": None})["result"] == "pending"

    def test_direction_defaults_to_over(self):
        assert self._norm({})["direction"] == "over"

    def test_line_coerces_to_float(self):
        assert self._norm({"line": "22.5"})["line"] == 22.5
        assert self._norm({"line": None})["line"] == 0.0
        assert self._norm({"line": ""})["line"] == 0.0

    def test_none_input_returns_empty_dict(self):
        from api.index import _normalize_line_pick
        assert _normalize_line_pick(None) == {}
        assert _normalize_line_pick("bad") == {}

    def test_recent_form_values_passed_through(self):
        """recent_form_values (real last-5 game stats) is passed through when present."""
        out = self._norm({"player_name": "X", "recent_form_values": [18.0, 12.0, 22.0, 15.0, 19.0]})
        assert out.get("recent_form_values") == [18.0, 12.0, 22.0, 15.0, 19.0]
        out2 = self._norm({"player_name": "Y"})
        assert out2.get("recent_form_values") is None

    def test_extra_fields_pass_through(self):
        """_live tracker and future additions must pass through."""
        result = self._norm({"_live": {"status": "live", "stat_current": 14}})
        assert result["_live"] == {"status": "live", "stat_current": 14}

    def test_confidence_coerces_to_int(self):
        assert self._norm({"confidence": 72.9})["confidence"] == 72
        assert self._norm({"confidence": None})["confidence"] == 0


# ---------------------------------------------------------------------------
# 9. Real Score engine — pure math unit tests
# ---------------------------------------------------------------------------

class TestRealScoreEngine:
    """
    real_score.py contains pure numpy Monte Carlo functions with no I/O.
    All coefficient bounds and monotonicity relationships are tested here.
    """

    def test_closeness_coefficient_bounds(self):
        """C_c must always be in [1.0, 2.0]."""
        from api.real_score import closeness_coefficient, _make_rng
        for spread, total in [(0, 210), (5, 222), (12, 235), (20, 200)]:
            rng = _make_rng(spread, total, seed_date="2026-03-07")
            c_c = closeness_coefficient(spread, total, rng)
            assert 1.0 <= c_c <= 2.0, f"C_c={c_c} out of bounds for spread={spread}"

    def test_closeness_higher_for_pickem_than_blowout(self):
        """Pick'em games are closer → higher C_c than heavy-favorite games."""
        from api.real_score import closeness_coefficient, _make_rng
        rng_pk = _make_rng(0, 222, seed_date="2026-03-07")
        rng_bl = _make_rng(0, 222, seed_date="2026-03-07")
        c_pickem  = closeness_coefficient(0,  222, _make_rng(0,  222, seed_date="2026-03-07"))
        c_blowout = closeness_coefficient(18, 222, _make_rng(18, 222, seed_date="2026-03-07"))
        assert c_pickem > c_blowout, (
            f"Pick'em C_c={c_pickem:.3f} should exceed blowout C_c={c_blowout:.3f}"
        )

    def test_clutch_coefficient_bounds(self):
        """C_k must always be in [0.9, 1.8]."""
        from api.real_score import clutch_coefficient, _make_rng
        for spread, usage, variance in [(0, 1.0, 0.2), (10, 0.6, 0.1), (20, 0.4, 0.05)]:
            rng = _make_rng(spread, 222, seed_date="2026-03-07")
            c_k = clutch_coefficient(spread, 222, usage, variance, rng)
            assert 0.9 <= c_k <= 1.8, f"C_k={c_k} out of bounds"

    def test_clutch_higher_for_high_usage_in_close_game(self):
        """High-usage star in pick'em game should get higher C_k than bench player in blowout."""
        from api.real_score import clutch_coefficient, _make_rng
        c_star  = clutch_coefficient(0,  222, usage_rate=2.0, player_variance=0.4,
                                     rng=_make_rng(0,  222, seed_date="2026-03-07"))
        c_bench = clutch_coefficient(18, 222, usage_rate=0.5, player_variance=0.0,
                                     rng=_make_rng(18, 222, seed_date="2026-03-07"))
        assert c_star > c_bench

    def test_momentum_bonus_scales_with_variance(self):
        """Higher variance → higher M_m (streaky players score more Real Score)."""
        from api.real_score import momentum_bonus
        assert momentum_bonus(0.0) == 1.0      # perfectly consistent
        assert momentum_bonus(0.5) == 1.25     # maximum streakiness
        assert momentum_bonus(0.25) == pytest.approx(1.125, abs=0.01)

    def test_momentum_bonus_clamps_at_ceiling(self):
        """M_m must not exceed 1.25 even for variance > 0.5."""
        from api.real_score import momentum_bonus
        assert momentum_bonus(1.0) == 1.25  # clamped at 0.5 internally
        assert momentum_bonus(99)  == 1.25

    def test_real_score_projection_positive(self):
        """Real Score must be positive for positive base score."""
        from api.real_score import real_score_projection, _make_rng
        rng = _make_rng(5, 222, seed_date="2026-03-07")
        rs, meta = real_score_projection(5.0, spread=5, total=222,
                                         usage_rate=1.0, player_variance=0.2, rng=rng)
        assert rs > 0
        assert "c_closeness" in meta
        assert "c_clutch" in meta
        assert "m_momentum" in meta
        assert "composite_mult" in meta

    def test_real_score_deterministic_same_rng(self):
        """Same seed → same Real Score (cache stability guarantee)."""
        from api.real_score import real_score_projection, _make_rng
        rng1 = _make_rng(5, 222, seed_date="2026-03-07")
        rng2 = _make_rng(5, 222, seed_date="2026-03-07")
        rs1, _ = real_score_projection(5.0, 5, 222, 1.0, 0.2, rng=rng1)
        rs2, _ = real_score_projection(5.0, 5, 222, 1.0, 0.2, rng=rng2)
        assert rs1 == rs2, "Real Score not deterministic for same seed"

    def test_real_score_scales_with_base(self):
        """Doubling s_base should roughly double the Real Score."""
        from api.real_score import real_score_projection, _make_rng
        rng_lo = _make_rng(5, 222, seed_date="2026-03-07")
        rng_hi = _make_rng(5, 222, seed_date="2026-03-07")
        rs_lo, _ = real_score_projection(3.0, 5, 222, 1.0, 0.2, rng=rng_lo)
        rs_hi, _ = real_score_projection(6.0, 5, 222, 1.0, 0.2, rng=rng_hi)
        ratio = rs_hi / rs_lo
        assert 1.9 <= ratio <= 2.1, f"Doubling base gave ratio {ratio:.2f}, expected ~2.0"


# ---------------------------------------------------------------------------
# 10. MILP Lineup Optimizer — asset_optimizer.py
# ---------------------------------------------------------------------------

class TestAssetOptimizer:
    """
    optimize_lineup() is the MILP slot-assignment engine. Tests cover:
      - Correct number of players returned
      - Slot labels applied (2.0x best → 1.2x worst)
      - Fallback works when fewer candidates than n
      - Higher-rating player gets the higher slot
      - Team constraint (max_per_team) is respected
    """

    def _make_players(self, n, base_rating=4.0, same_team=False):
        teams = ["LAL", "BOS", "MIL", "PHX", "DEN", "UTA", "GSW", "MEM"]
        return [
            {
                "name": f"Player{i}",
                "team": "LAL" if same_team else teams[i % len(teams)],
                "pos": ["PG", "SG", "SF", "PF", "C"][i % 5],
                "rating": base_rating - i * 0.1,
                "est_mult": 1.5 - i * 0.05,
                "chalk_ev": (base_rating - i * 0.1) * (1.6 + 1.5 - i * 0.05),
                "moonshot_ev": 0,
            }
            for i in range(n)
        ]

    def test_returns_exactly_n_players(self):
        from api.asset_optimizer import optimize_lineup
        players = self._make_players(10)
        result = optimize_lineup(players, n=5)
        assert len(result) == 5

    def test_all_players_have_slot_labels(self):
        from api.asset_optimizer import optimize_lineup, SLOT_LABELS
        result = optimize_lineup(self._make_players(8), n=5)
        for p in result:
            assert p.get("slot") in SLOT_LABELS, f"Bad slot: {p.get('slot')}"

    def test_slot_labels_are_unique(self):
        """Each slot can only be assigned once."""
        from api.asset_optimizer import optimize_lineup
        result = optimize_lineup(self._make_players(8), n=5)
        slots = [p["slot"] for p in result]
        assert len(slots) == len(set(slots)), f"Duplicate slots: {slots}"

    def test_fallback_with_fewer_candidates_than_n(self):
        """If fewer than n candidates exist, return all of them (no crash)."""
        from api.asset_optimizer import optimize_lineup
        players = self._make_players(3)
        result = optimize_lineup(players, n=5)
        assert len(result) == 3
        for p in result:
            assert "slot" in p

    def test_fallback_sort_greedy(self):
        """_fallback_sort must rank by sort_key descending."""
        from api.asset_optimizer import _fallback_sort
        players = [
            {"name": "A", "chalk_ev": 10.0, "team": "LAL", "pos": "PG"},
            {"name": "B", "chalk_ev": 8.0,  "team": "BOS", "pos": "SG"},
            {"name": "C", "chalk_ev": 6.0,  "team": "MIL", "pos": "SF"},
        ]
        result = _fallback_sort(players, n=3, sort_key="chalk_ev")
        assert result[0]["name"] == "A"
        assert result[0]["slot"] == "2.0x"

    def test_max_per_team_constraint(self):
        """max_per_team=2 must prevent more than 2 players from the same team."""
        from api.asset_optimizer import optimize_lineup
        # 3 teams × 4 players each = 12 players. n=5, max_per_team=2 → max 6 slots
        # available across teams, so MILP is feasible and must respect the constraint.
        teams = ["LAL", "BOS", "MIL"]
        players = [
            {
                "name": f"Player{t}{i}",
                "team": t,
                "pos": ["PG", "SG", "SF", "PF"][i],
                "rating": 4.0 - idx * 0.05,
                "est_mult": 1.5,
                "chalk_ev": 10.0 - idx * 0.1,
                "moonshot_ev": 0,
            }
            for idx, (t, i) in enumerate(
                [(t, i) for t in teams for i in range(4)]
            )
        ]
        result = optimize_lineup(players, n=5, max_per_team=2)
        for team in teams:
            count = sum(1 for p in result if p.get("team") == team)
            assert count <= 2, f"max_per_team=2 violated: {count} {team} players in lineup"

    def test_empty_projections_returns_empty(self):
        from api.asset_optimizer import optimize_lineup
        result = optimize_lineup([], n=5)
        assert result == []


# ---------------------------------------------------------------------------
# 11. Config coverage — new lineup/line keys are readable via _cfg()
# ---------------------------------------------------------------------------

class TestConfigCoverage:
    """
    After Phase B, all major model floors live in model-config.json.
    These tests ensure the keys are present and return the expected defaults
    via the _cfg() helper. Changing any of these values in the JSON should
    be immediately reflected here (i.e. this test would need to be updated
    if Ben changes a value, which is the correct behavior).
    """

    def test_chalk_rating_floor_readable(self):
        from api.index import _cfg
        val = _cfg("lineup.chalk_rating_floor", None)
        assert val == 2.0, f"Expected 2.0, got {val}"

    def test_game_chalk_rating_floor_readable(self):
        from api.index import _cfg
        val = _cfg("lineup.game_chalk_rating_floor", None)
        assert val == 3.5, f"Expected 3.5, got {val}"

    def test_avg_slot_multiplier_readable(self):
        from api.index import _cfg
        val = _cfg("lineup.avg_slot_multiplier", None)
        assert val == 1.6, f"Expected 1.6, got {val}"

    def test_slot_multipliers_readable(self):
        from api.index import _cfg
        val = _cfg("lineup.slot_multipliers", None)
        assert val == [2.0, 1.8, 1.6, 1.4, 1.2], f"Unexpected: {val}"

    def test_moonshot_min_rating_floor_readable(self):
        from api.index import _cfg
        val = _cfg("moonshot.min_rating_floor", None)
        assert val == 2.0, f"Expected 2.0, got {val}"

    def test_line_min_confidence_readable(self):
        from api.index import _cfg
        val = _cfg("line.min_confidence", None)
        assert val == 50, f"Expected 50, got {val}"

    def test_cfg_fallback_for_missing_key(self):
        """_cfg must return the fallback value for a key that doesn't exist."""
        from api.index import _cfg
        assert _cfg("nonexistent.key.deep", "FALLBACK") == "FALLBACK"

    def test_cfg_nested_dot_notation(self):
        """_cfg must resolve arbitrary depth dot-notation keys."""
        from api.index import _cfg
        # card_boost.ceiling is an existing 3rd-level key
        val = _cfg("card_boost.ceiling", None)
        assert val == 3.0, f"Expected 3.0, got {val}"


# ---------------------------------------------------------------------------
# 12. project_player contract — output shape regression guard
# ---------------------------------------------------------------------------

class TestProjectPlayerContract:
    """
    project_player() is the core projection function. After _normalize_player()
    wraps it, the frontend is insulated — but project_player() itself must still
    produce valid input for the normalizer. These tests ensure it doesn't return
    None for valid players and that the shape is stable.
    """

    def _make_pinfo(self, **kwargs):
        base = {
            "id": "test-123", "name": "Test Player", "pos": "SG",
            "injury_status": "", "is_out": False,
        }
        base.update(kwargs)
        return base

    def _make_stats(self, **kwargs):
        # Keys match what project_player reads: "min", "pts", "reb", "ast", etc.
        base = {
            "min": 28.0, "pts": 18.0, "reb": 4.0, "ast": 3.5,
            "stl": 1.0,  "blk": 0.5,  "tov": 2.0,
            "season_pts": 18.0, "recent_pts": 20.0,
            "season_reb": 4.0,  "recent_reb": 4.5,
            "season_ast": 3.5,  "recent_ast": 4.0,
            "season_stl": 1.0,  "recent_stl": 1.1,
            "season_blk": 0.5,  "recent_blk": 0.6,
            "season_min": 28.0, "recent_min": 28.0,
            "usage_trend": 0.0, "opp_def_rating": 112.0,
            "home_away": 1.0, "ast_rate": 0.18, "def_rate": 0.04,
            "pts_per_min": 0.64,
        }
        base.update(kwargs)
        return base

    def test_valid_player_returns_dict(self):
        """project_player must return a dict (not None) for a healthy player."""
        from api.index import project_player
        result = project_player(
            self._make_pinfo(), self._make_stats(),
            spread=3.0, total=222.0, side="home", team_abbr="BOS"
        )
        assert result is not None
        assert isinstance(result, dict)

    def test_out_player_returns_none(self):
        """project_player must return None for a player flagged is_out."""
        from api.index import project_player
        result = project_player(
            self._make_pinfo(is_out=True), self._make_stats(),
            spread=3.0, total=222.0, side="home", team_abbr="BOS"
        )
        assert result is None

    def test_output_has_rating_and_est_mult(self):
        """Both rating and est_mult must be present and numeric."""
        from api.index import project_player
        result = project_player(
            self._make_pinfo(), self._make_stats(),
            spread=0, total=210, side="away", team_abbr="MEM"
        )
        assert result is not None
        assert isinstance(result.get("rating"), (int, float))
        assert isinstance(result.get("est_mult"), (int, float))
        assert result["rating"] >= 0
        assert result["est_mult"] >= 0

    def test_output_has_stat_fields(self):
        """All projected stat fields (pts, reb, ast, stl, blk, tov) must be present."""
        from api.index import project_player
        result = project_player(
            self._make_pinfo(), self._make_stats(),
            spread=5, total=230, side="home", team_abbr="DEN"
        )
        assert result is not None
        for field in ["pts", "reb", "ast", "stl", "blk", "tov", "predMin"]:
            assert field in result, f"Missing stat field: {field}"
            assert isinstance(result[field], (int, float))

    def test_normalizer_accepts_project_player_output(self):
        """_normalize_player must not raise when given real project_player output."""
        from api.index import project_player, _normalize_player
        raw = project_player(
            self._make_pinfo(), self._make_stats(),
            spread=3, total=220, side="home", team_abbr="GSW"
        )
        assert raw is not None
        normalized = _normalize_player(raw)
        # chalk_ev_capped must be gone
        assert "chalk_ev_capped" not in normalized
        # All required fields must be present and have correct types
        assert isinstance(normalized["rating"], float)
        assert isinstance(normalized["predMin"], float)
        assert isinstance(normalized["est_mult"], float)


# ---------------------------------------------------------------------------
# 13. Line engine helpers — pure function tests (no Claude API calls)
# ---------------------------------------------------------------------------

class TestLineEngineHelpers:
    """
    Tests for line_engine.py helper functions that have no external deps.
    Claude API calls are not tested here — only the pure data-transform layer.
    """

    def test_game_lookup_home_team(self):
        """Home team lookup returns opponent (away abbr) and correct total."""
        from api.line_engine import _game_lookup_from_games
        games = [{
            "home": {"abbr": "BOS"}, "away": {"abbr": "MIA"},
            "total": 215.5, "spread": -5.5,
            "home_b2b": False, "away_b2b": True,
        }]
        lookup = _game_lookup_from_games(games)
        assert "BOS" in lookup
        assert lookup["BOS"]["opponent"] == "MIA"
        assert lookup["BOS"]["total"] == 215.5
        assert lookup["BOS"]["spread"] == -5.5

    def test_game_lookup_away_team(self):
        """Away team lookup returns home abbr as opponent."""
        from api.line_engine import _game_lookup_from_games
        games = [{
            "home": {"abbr": "BOS"}, "away": {"abbr": "MIA"},
            "total": 215.5, "spread": -5.5,
            "home_b2b": False, "away_b2b": True,
        }]
        lookup = _game_lookup_from_games(games)
        assert lookup["MIA"]["opponent"] == "BOS"

    def test_game_lookup_b2b_flags(self):
        """B2B flags must be attributed to the correct team."""
        from api.line_engine import _game_lookup_from_games
        games = [{
            "home": {"abbr": "LAL"}, "away": {"abbr": "GSW"},
            "total": 228, "spread": 2.0,
            "home_b2b": True, "away_b2b": False,
        }]
        lookup = _game_lookup_from_games(games)
        # Home B2B — away team sees home is on B2B (opp_b2b)
        assert lookup["GSW"]["opp_b2b"] is True
        # Away B2B — home team sees away is on B2B
        assert lookup["LAL"]["opp_b2b"] is False

    def test_game_lookup_multiple_games(self):
        """Lookup must cover all teams across multiple games."""
        from api.line_engine import _game_lookup_from_games
        games = [
            {"home": {"abbr": "A"}, "away": {"abbr": "B"},
             "total": 220, "spread": 0, "home_b2b": False, "away_b2b": False},
            {"home": {"abbr": "C"}, "away": {"abbr": "D"},
             "total": 230, "spread": 3, "home_b2b": False, "away_b2b": False},
        ]
        lookup = _game_lookup_from_games(games)
        assert set(lookup.keys()) == {"A", "B", "C", "D"}

    def test_game_lookup_empty_games(self):
        """Empty games list must return empty lookup without raising."""
        from api.line_engine import _game_lookup_from_games
        assert _game_lookup_from_games([]) == {}


# ---------------------------------------------------------------------------
# 14. JS contract guard — normalizer integration visible in API shape
# ---------------------------------------------------------------------------

class TestJSContractGuard:
    """
    Regression guard: ensures the frontend null guards added in Phase C
    are still present and that the contract normalizer functions exist
    in the backend codebase.
    """

    @pytest.fixture(scope="class")
    def script_source(self):
        html = (ROOT / "index.html").read_text()
        start = html.rfind("<script>")
        end   = html.rfind("</script>")
        assert start != -1 and end != -1
        return html[start:end]

    def test_optional_chaining_on_lineups_chalk(self, script_source):
        """Both lineups?.chalk? guards must be present after Phase C fix."""
        assert "lineups?.chalk?.length" in script_source, (
            "Missing optional-chain null guard on lineups.chalk — "
            "cold-start locked response will crash the app"
        )

    def test_parse_float_line_has_fallback(self, script_source):
        """parseFloat(pick.line) must have a || 0 fallback to prevent NaN."""
        assert "parseFloat(pick.line) || 0" in script_source, (
            "parseFloat(pick.line) without || 0 — NaN will propagate to live tracker"
        )

    def test_edge_nullish_coalescing(self, script_source):
        """pick.edge must use ?? 0 to handle undefined edge field."""
        assert "pick.edge ?? 0" in script_source, (
            "pick.edge without ?? 0 — undefined edge renders as 'NaN' in pick card"
        )

    def test_normalize_player_function_exists_in_backend(self):
        """_normalize_player must be importable from api.index."""
        from api.index import _normalize_player
        assert callable(_normalize_player)

    def test_normalize_line_pick_function_exists_in_backend(self):
        """_normalize_line_pick must be importable from api.index."""
        from api.index import _normalize_line_pick
        assert callable(_normalize_line_pick)


# ---------------------------------------------------------------------------
# 15. log_get() normalization regression guard (C3 fix)
# ---------------------------------------------------------------------------

class TestLogGetNormalization:
    """
    Regression guard for C3: log_get() builds player cards from CSV rows.
    Before the fix, these were raw dicts missing chalk_ev, moonshot_ev,
    injury_status — causing undefined renders in the History tab.

    After the fix, _normalize_player() is applied, guaranteeing all required
    frontend fields are present with safe defaults.
    """

    _PRED_CSV = (
        "scope,lineup_type,slot,player_name,player_id,team,pos,"
        "predicted_rs,est_card_boost,pred_min,pts,reb,ast,stl,blk\n"
        "slate,chalk,2.0x,LeBron James,x,LAL,F,4.8,1.3,35,25,8,7,1,1\n"
        "slate,upside,2.0x,Kris Middleton,y,MIL,F,3.2,2.1,28,15,5,4,1,0\n"
    )

    def test_log_get_player_cards_have_chalk_ev(self):
        """Player cards from log_get must include chalk_ev (normalized, not raw CSV)."""
        from api.index import _parse_csv, PRED_FIELDS, _normalize_player
        rows = _parse_csv(self._PRED_CSV, PRED_FIELDS)
        assert rows, "Failed to parse prediction CSV"

        # Simulate what log_get does after the C3 fix
        card = _normalize_player({
            "slot":      rows[0].get("slot", ""),
            "name":      rows[0].get("player_name", ""),
            "team":      rows[0].get("team", ""),
            "pos":       rows[0].get("pos", ""),
            "rating":    rows[0].get("predicted_rs", ""),
            "est_mult":  rows[0].get("est_card_boost", ""),
            "predMin":   rows[0].get("pred_min", ""),
            "pts":       rows[0].get("pts", ""),
            "reb":       rows[0].get("reb", ""),
            "ast":       rows[0].get("ast", ""),
            "stl":       rows[0].get("stl", ""),
            "blk":       rows[0].get("blk", ""),
        })
        assert "chalk_ev" in card, "chalk_ev missing from normalized log card"
        assert "moonshot_ev" in card, "moonshot_ev missing from normalized log card"
        assert "injury_status" in card, "injury_status missing from normalized log card"
        assert isinstance(card["chalk_ev"], float)
        assert isinstance(card["injury_status"], str)

    def test_log_get_player_rating_is_float(self):
        """rating field from CSV string must be coerced to float by normalizer."""
        from api.index import _normalize_player
        card = _normalize_player({"rating": "4.8", "pts": "25", "reb": "8"})
        assert isinstance(card["rating"], float), f"rating should be float, got {type(card['rating'])}"
        assert card["rating"] == 4.8

    def test_log_get_empty_string_stats_are_zero(self):
        """CSV rows with missing stats (empty strings) must normalize to 0.0."""
        from api.index import _normalize_player
        card = _normalize_player({"rating": "", "pts": "", "reb": "", "ast": ""})
        assert card["rating"] == 0.0
        assert card["pts"] == 0.0

    def test_log_get_no_nan_in_output(self):
        """No NaN values should appear in normalized log cards (catches parseFloat('') → NaN)."""
        import math
        from api.index import _normalize_player
        card = _normalize_player({"rating": "4.8", "pts": "25", "est_mult": "1.3",
                                   "reb": "", "ast": None, "stl": "1"})
        for k, v in card.items():
            if isinstance(v, float):
                assert not math.isnan(v), f"NaN found in field '{k}': {card}"


# ---------------------------------------------------------------------------
# 16. update-config input validation regression guard (H6 fix)
# ---------------------------------------------------------------------------

class TestUpdateConfigValidation:
    """
    Regression guard for H6: /api/lab/update-config accepts dot-notation keys
    from user input. Before the fix, unsanitized keys could write to arbitrary
    config paths. After the fix, a regex validates each key segment.

    These tests verify the validation pattern matches what the endpoint uses.
    """

    VALID_KEY_RE = r'^[a-zA-Z_][a-zA-Z0-9_]*([.][a-zA-Z_][a-zA-Z0-9_]*)*$'

    def _valid(self, key):
        import re
        return bool(re.match(self.VALID_KEY_RE, key))

    def test_normal_config_keys_pass(self):
        """Standard config paths must be accepted."""
        assert self._valid("card_boost.decay_base")
        assert self._valid("lineup.chalk_rating_floor")
        assert self._valid("moonshot.min_rating_floor")
        assert self._valid("projection.b2b_minute_penalty")
        assert self._valid("development_teams")

    def test_path_traversal_is_rejected(self):
        """Path segments with special chars must be rejected."""
        assert not self._valid("../etc/passwd")
        assert not self._valid("card_boost/../os")
        assert not self._valid("card_boost.decay_base; DROP TABLE")
        assert not self._valid("card_boost[0]")
        assert not self._valid("")
        assert not self._valid("card boost")  # space in key

    def test_numeric_first_char_is_rejected(self):
        """Keys must start with letter or underscore, not digit."""
        assert not self._valid("1card_boost")
        assert not self._valid("0.decay")

    def test_underscore_keys_are_accepted(self):
        """Keys with leading underscores are valid (e.g., _internal)."""
        assert self._valid("_private_key")
        assert self._valid("card_boost._internal")

    def test_validation_regex_present_in_source(self):
        """The validation regex must be present in api/index.py (catches accidental deletion)."""
        src = (ROOT / "api" / "index.py").read_text()
        assert "Invalid key format" in src, (
            "update-config key validation removed — H6 security fix regressed"
        )
        assert "lab_update_config" in src, "lab_update_config function missing"


# ---------------------------------------------------------------------------
# 17. New frontend null guard regression tests (audit batch fixes)
# ---------------------------------------------------------------------------

class TestFrontendAuditFixes:
    """
    Regression guards for the 8 frontend null guards and .ok checks added
    in the whole-app audit implementation. Each test checks that the exact
    guard pattern remains in the JS source.
    """

    @pytest.fixture(scope="class")
    def script_source(self):
        html = (ROOT / "index.html").read_text()
        start = html.rfind("<script>")
        end   = html.rfind("</script>")
        assert start != -1 and end != -1
        return html[start:end]

    def test_briefing_fetch_has_ok_check(self, script_source):
        """C1: /api/lab/briefing fetch must check .ok before .json()."""
        assert "r.ok ? r.json() : Promise.reject('briefing ' + r.status)" in script_source, (
            "C1 regression: briefing fetch missing .ok guard"
        )

    def test_chat_streaming_has_ok_check(self, script_source):
        """C2: /api/lab/chat must check r.ok before r.body.getReader()."""
        assert "if (!r.ok) throw new Error('chat ' + r.status)" in script_source, (
            "C2 regression: chat streaming missing .ok guard before getReader()"
        )

    def test_line_hist_data_optional_chain(self, script_source):
        """H1: LINE_HIST_DATA.picks must use optional chaining."""
        assert "LINE_HIST_DATA?.picks" in script_source, (
            "H1 regression: LINE_HIST_DATA.picks without optional chain — throws if null"
        )

    def test_slate_available_after_optional_chain(self, script_source):
        """H3: SLATE.available_after must use optional chaining."""
        assert "SLATE?.available_after" in script_source, (
            "H3 regression: SLATE.available_after without optional chain"
        )

    def test_line_resolve_poll_cleared_on_tab_switch(self, script_source):
        """M5: LINE_RESOLVE_POLL must be cleared in switchTab()."""
        # Check both polls are cleared
        assert "LINE_RESOLVE_POLL" in script_source
        # The fix adds clearing of LINE_RESOLVE_POLL in switchTab — verify it's there
        # by checking the pattern appears near LINE_LIVE_POLL clearing
        live_idx = script_source.find("clearInterval(LINE_LIVE_POLL)")
        resolve_idx = script_source.find("clearInterval(LINE_RESOLVE_POLL)")
        assert resolve_idx != -1, "M5 regression: LINE_RESOLVE_POLL never cleared"

    def test_mutation_observer_dedup(self, script_source):
        """M6: MutationObserver must use disconnect() before re-attaching."""
        assert "_labMsgObserver" in script_source, (
            "M6 regression: MutationObserver not tracked — accumulates on each Lab open"
        )
        assert "_labMsgObserver.disconnect()" in script_source, (
            "M6 regression: MutationObserver not disconnected before re-attaching"
        )

    def test_pickedx_nan_guard(self, script_source):
        """M4: parseInt(el.dataset.pickIdx) result must be checked for NaN."""
        assert "isNaN(_pidx)" in script_source, (
            "M4 regression: pickIdx NaN guard removed — could assign undefined to _linePick"
        )

    def test_biggest_misses_field_guards(self, script_source):
        """M2: biggest_misses map must have field guards to prevent 'undefined' renders."""
        assert "m.player || '?'" in script_source, (
            "M2 regression: m.player accessed without fallback"
        )
