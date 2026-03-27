# ─────────────────────────────────────────────────────────────────────────────
# SHARED CONSTANTS & UTILITIES
#
# Single source of truth for values duplicated across backend modules.
# Import from here instead of redefining locally.
# ─────────────────────────────────────────────────────────────────────────────

import re
import unicodedata
from datetime import datetime, timezone, timedelta

# ── Slot multipliers (Real Sports App draft slot values) ─────────────────────
SLOT_MULTIPLIERS = [2.0, 1.8, 1.6, 1.4, 1.2]
SLOT_LABELS = ["2.0x", "1.8x", "1.6x", "1.4x", "1.2x"]

# ── Stat type mappings ───────────────────────────────────────────────────────
STAT_TYPES = ["points", "rebounds", "assists"]
STAT_ABBR = {"points": "PTS", "rebounds": "REB", "assists": "AST"}
STAT_FIELDS = {"points": "pts", "rebounds": "reb", "assists": "ast"}


# ── ESPN scoreboard URL helper ──────────────────────────────────────────────
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

def espn_scoreboard_url(date_str_ymd: str) -> str:
    """Build ESPN scoreboard URL. date_str_ymd should be YYYYMMDD format."""
    return f"{ESPN_BASE}/scoreboard?dates={date_str_ymd}"


# ── Player name normalization (DRY — single source of truth) ────────────────
def normalize_player_name(name: str) -> str:
    """Normalize player name for consistent joining across datasets.

    Strips accents/diacritics, periods, common suffixes (Jr., Sr., III, II, IV),
    lowercases. Works for ESPN, RotoWire, CSV pipelines.

    Examples:
        'Nikola Jokić' → 'nikola jokic'
        'P.J. Washington' → 'pj washington'
        'Marcus Morris Sr.' → 'marcus morris'
    """
    n = unicodedata.normalize("NFKD", name or "").encode("ASCII", "ignore").decode("ASCII")
    n = n.replace(".", "")
    n = re.sub(r'\s+(jr\.?|sr\.?|iii|ii|iv)\s*$', '', n, flags=re.IGNORECASE)
    return n.strip().lower()


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
