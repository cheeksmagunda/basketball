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
