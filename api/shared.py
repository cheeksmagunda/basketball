# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS & UTILITIES
#
# Single source of truth for values duplicated across backend modules.
# Import from here instead of redefining locally.
# ─────────────────────────────────────────────────────────────────────────────

from datetime import datetime, timezone, timedelta

# ── Slot multipliers (Real Sports App draft slot values) ─────────────────────
SLOT_MULTIPLIERS = [2.0, 1.8, 1.6, 1.4, 1.2]
SLOT_LABELS = ["2.0x", "1.8x", "1.6x", "1.4x", "1.2x"]

# ── Stat type mappings ───────────────────────────────────────────────────────
STAT_TYPES = ["points", "rebounds", "assists"]
STAT_ABBR = {"points": "PTS", "rebounds": "REB", "assists": "AST"}
STAT_FIELDS = {"points": "pts", "rebounds": "reb", "assists": "ast"}


def et_date():
    """Current date in Eastern Time (handles EST/EDT).

    Returns a datetime.date object. Used for cache keys, file paths,
    and lock window calculations across the app.
    """
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date()
    except ImportError:
        # Fallback: EST=UTC-5 (Nov–Mar), EDT=UTC-4 (Mar–Nov)
        now_utc = datetime.now(timezone.utc)
        offset = timedelta(hours=-4 if 3 < now_utc.month < 11 else -5)
        return (now_utc + offset).date()
