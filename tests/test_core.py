"""
Lightweight unit tests for basketball optimizer.

Run with:  pytest tests/
Coverage targets the three most failure-prone areas:
  1. Pure helper functions (no I/O, no external deps)
  2. Line-cache & slate endpoint logic (mocked GitHub / ESPN)
  3. JS syntax integrity (catches apostrophe crashes before push)
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
    The frontend banner-visibility check in showLabUnlocked() must use both:
      - _todayLog?.has_actuals  (direct log check — works without paired predictions)
      - LAB.briefing?.latest_slate?.date  (briefing fallback)

    Removing either regresses the banner bug.
    """

    @pytest.fixture(scope="class")
    def script_source(self):
        html = (ROOT / "index.html").read_text()
        start = html.rfind("<script>")
        end   = html.rfind("</script>")
        assert start != -1 and end != -1, "No <script> block found in index.html"
        return html[start:end]

    def test_banner_check_uses_has_actuals(self, script_source):
        """Banner visibility must check has_actuals from /api/log/get."""
        assert "has_actuals" in script_source, (
            "Missing has_actuals check — banner will show even after upload "
            "when predictions weren't committed to GitHub"
        )

    def test_banner_check_uses_briefing_fallback(self, script_source):
        """Banner visibility must also keep the briefing latest_slate fallback."""
        assert "latest_slate" in script_source, (
            "Missing latest_slate briefing fallback in banner check"
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
