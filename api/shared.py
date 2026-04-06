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

    Strips accents/diacritics, periods, hyphens, apostrophes, common suffixes
    (Jr., Sr., III, II, IV), lowercases, collapses whitespace.
    Works for ESPN, RotoWire, Odds API, CSV pipelines.

    Examples:
        'Nikola Jokić' → 'nikola jokic'
        'P.J. Washington' → 'pj washington'
        'Marcus Morris Sr.' → 'marcus morris'
        'Shai Gilgeous-Alexander' → 'shai gilgeousalexander'
        "De'Aaron Fox" → 'deaaron fox'
    """
    n = unicodedata.normalize("NFKD", name or "").encode("ASCII", "ignore").decode("ASCII")
    n = re.sub(r"['\.\-]", "", n)
    n = re.sub(r'\s+(jr\.?|sr\.?|iii|ii|iv)\s*$', '', n, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', n).strip().lower()


def et_date():
    """Current date in Eastern Time (handles EST/EDT).

    Returns a datetime.date object. Used for cache keys, file paths,
    and lock window calculations across the app.
    """
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).date()
    except ImportError:
        # Fallback approximation when tzdata is unavailable.
        # US DST starts 2nd Sunday of March (~Mar 8–14) and ends 1st Sunday of Nov (~Nov 1–7).
        # Day-based bounds are more accurate than month-boundary-only checks, which would
        # produce the wrong offset for 1–2 weeks in March and November every year.
        now_utc = datetime.now(timezone.utc)
        m, d = now_utc.month, now_utc.day
        is_dst = (m == 3 and d >= 8) or (4 <= m <= 10) or (m == 11 and d < 8)
        offset = timedelta(hours=-4 if is_dst else -5)
        return (now_utc + offset).date()
