"""3-Tier Cascade Boost Prediction Model.

# grep: BOOST CASCADE MODEL

Predicts Real Sports card boost (0.0–3.0) using a data-driven cascade:

  Tier 1 — Returning player (appeared on a slate within last 14 days)
           Uses prev_boost + adjustment factors (RS, drafts, trend, gap, mean reversion).
           Expected MAE: ~0.10–0.15.

  Tier 2 — Known player, stale data (appeared but >14 days ago)
           Blends historical boost mean with API-derived estimate, weighted by staleness.

  Tier 3 — Cold start (never seen on Real Sports)
           Player Quality Index (PQI) from season stats → boost mapping.
           Default high (3.0) since most unknown players are role players.

Architecture replaces the prior LightGBM boost_model.pkl / drafts_model.pkl chain
with a deterministic heuristic calibrated from 2,234 player-date records (148 dates).

Key insight: prev_boost correlates +0.957 with actual boost — the single strongest
signal in the entire feature space. 46% of day-over-day changes are exactly 0.0,
88.2% are within ±0.3. The cascade exploits this persistence.
"""

from __future__ import annotations

import csv
import math
import threading
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Player history index — loaded once from data/top_performers.csv
# ---------------------------------------------------------------------------

_PLAYER_HISTORY: dict[str, list[dict]] = {}
_HISTORY_LOADED = False
_HISTORY_LOCK = threading.Lock()


def _normalize_name(name: str) -> str:
    """Lightweight name normalizer (lowercase, strip suffixes, collapse whitespace)."""
    n = (name or "").strip().lower()
    for suffix in (" jr.", " jr", " sr.", " sr", " ii", " iii", " iv"):
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
            break
    return " ".join(n.split())


def load_player_history(force: bool = False) -> dict[str, list[dict]]:
    """Load and index top_performers.csv into per-player sorted appearance lists.

    Each entry: {date, boost, rs, drafts} sorted ascending by date so
    most recent is last.  Thread-safe; loads once.
    """
    global _PLAYER_HISTORY, _HISTORY_LOADED
    if _HISTORY_LOADED and not force:
        return _PLAYER_HISTORY
    with _HISTORY_LOCK:
        if _HISTORY_LOADED and not force:
            return _PLAYER_HISTORY

        history: dict[str, list[dict]] = defaultdict(list)
        tp = Path("data/top_performers.csv")
        if not tp.exists():
            # Try relative to this file's parent (api/ → repo root)
            tp = Path(__file__).parent.parent / "data" / "top_performers.csv"

        # First pass: detect all-zero dates (season openers where boosts aren't set yet)
        _all_zero_dates: set[str] = set()
        if tp.exists():
            try:
                _date_boosts: dict[str, list[float]] = defaultdict(list)
                with tp.open("r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        dt = row.get("date", "").strip()
                        if dt:
                            _date_boosts[dt].append(_safe(row.get("actual_card_boost")))
                _all_zero_dates = {
                    d for d, boosts in _date_boosts.items()
                    if boosts and all(b == 0 for b in boosts)
                }
                if _all_zero_dates:
                    print(f"[boost_cascade] Skipping {len(_all_zero_dates)} all-zero date(s): "
                          f"{sorted(_all_zero_dates)[:3]}")
            except Exception as e:
                print(f"[boost_cascade] Pre-scan failed: {e}")

        if tp.exists():
            try:
                with tp.open("r", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        name = _normalize_name(row.get("player_name", ""))
                        if not name:
                            continue
                        dt = row.get("date", "").strip()
                        if not dt or dt in _all_zero_dates:
                            continue
                        boost = _safe(row.get("actual_card_boost"))
                        rs = _safe(row.get("actual_rs"))
                        drafts = _safe(row.get("drafts"))
                        # boost=0.0 IS valid for superstars (3.1% of entries on real dates)
                        # We mark it so the cascade knows this is real data
                        entry = {"date": dt, "boost": boost, "rs": rs, "drafts": drafts,
                                 "has_boost_data": True}
                        history[name].append(entry)
            except Exception as e:
                print(f"[boost_cascade] Failed to load top_performers.csv: {e}")

        # Also load from data/actuals/ for broader coverage
        for data_dir in ("data/actuals",):
            pdir = Path(data_dir)
            if not pdir.exists():
                pdir = Path(__file__).parent.parent / data_dir
            if not pdir.exists():
                continue
            try:
                for csvfile in pdir.glob("*.csv"):
                    dt_str = csvfile.stem  # e.g. "2025-03-15"
                    if dt_str in _all_zero_dates:
                        continue
                    with csvfile.open("r", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            name = _normalize_name(
                                row.get("player_name") or row.get("player") or ""
                            )
                            if not name:
                                continue
                            boost = _safe(row.get("actual_card_boost"))
                            rs = _safe(row.get("actual_rs"))
                            drafts = _safe(row.get("drafts"))
                            entry = {"date": dt_str, "boost": boost, "rs": rs, "drafts": drafts,
                                     "has_boost_data": True}
                            history[name].append(entry)
            except Exception as e:
                print(f"[boost_cascade] Failed to load {data_dir}: {e}")

        # De-duplicate by (name, date) keeping the row with higher boost
        for name in history:
            seen: dict[str, dict] = {}
            for entry in history[name]:
                dt = entry["date"]
                if dt not in seen or entry["boost"] > seen[dt]["boost"]:
                    seen[dt] = entry
            history[name] = sorted(seen.values(), key=lambda e: e["date"])

        _PLAYER_HISTORY = dict(history)
        _HISTORY_LOADED = True
        print(f"[boost_cascade] Loaded history: {len(_PLAYER_HISTORY)} players, "
              f"{sum(len(v) for v in _PLAYER_HISTORY.values())} total appearances")
        return _PLAYER_HISTORY


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        return float(v) if v is not None and str(v).strip() != "" else default
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Tier baselines — calibrated from 2,234 player-date records
# ---------------------------------------------------------------------------

# Player tier (by avg RS) → expected boost
_TIER_BASELINES = [
    # (max_rs, expected_boost)
    (2.5, 2.92),   # Bench
    (3.5, 2.80),   # Rotation
    (4.5, 2.12),   # Starter
    (5.5, 1.06),   # Star
    (7.0, 0.64),   # Superstar
    (999, 0.05),   # Elite
]


def get_tier_baseline(hist_rs_mean: float) -> float:
    """Map player quality (historical avg RS) to expected boost tier."""
    for max_rs, baseline in _TIER_BASELINES:
        if hist_rs_mean < max_rs:
            return baseline
    return 0.05


def estimate_boost_from_api(
    season_ppg: float = 0,
    season_rpg: float = 0,
    season_apg: float = 0,
    season_mpg: float = 0,
) -> float:
    """Tier 3: estimate boost for a player never seen on Real Sports.

    Uses a Player Quality Index (PQI) that combines stats weighted
    by their correlation with player fame/quality.
    """
    pqi = season_ppg * 1.0 + season_rpg * 0.4 + season_apg * 0.6 + season_mpg * 0.2

    if pqi < 8:
        return 3.0
    elif pqi < 14:
        return round(3.0 - (pqi - 8) * 0.08, 1)   # 2.5–3.0
    elif pqi < 22:
        return round(2.5 - (pqi - 14) * 0.12, 1)   # 1.5–2.5
    elif pqi < 30:
        return round(1.5 - (pqi - 22) * 0.12, 1)   # 0.5–1.5
    elif pqi < 38:
        return round(0.5 - (pqi - 30) * 0.06, 1)   # 0.0–0.5
    else:
        return 0.0


def estimate_draft_popularity(
    season_ppg: float = 0,
    team: str = "",
    is_national_tv: bool = False,
    n_games: int = 6,
    recent_ppg: float = 0,
) -> float:
    """Estimate how many users will draft this player (used for boost movement)."""
    base_score = season_ppg * 100

    # Star name recognition multiplier
    if season_ppg >= 25:
        base_score *= 2.0
    elif season_ppg >= 20:
        base_score *= 1.5

    # Big market team multiplier
    _big_markets = {"LAL", "GSW", "GS", "BOS", "NYK", "NY", "CHI", "PHI", "MIA"}
    if (team or "").upper() in _big_markets:
        base_score *= 1.3

    # National TV game multiplier
    if is_national_tv:
        base_score *= 1.2

    # Slate size (fewer games → more concentrated drafts)
    if n_games <= 3:
        base_score *= 1.4
    elif n_games >= 10:
        base_score *= 0.8

    # Recent hot streak → more drafts
    if season_ppg > 0 and recent_ppg > season_ppg * 1.2:
        base_score *= 1.15

    return base_score


# ---------------------------------------------------------------------------
# Main prediction: 3-tier cascade
# ---------------------------------------------------------------------------

# Stale threshold in days — above this, switch from Tier 1 to Tier 2
_TIER1_MAX_GAP_DAYS = 14


def predict_boost(
    player_name: str,
    today_str: str | None = None,
    *,
    # API stats (for Tier 2/3 fallback and Tier 1 minutes-change signals)
    season_ppg: float = 0,
    season_rpg: float = 0,
    season_apg: float = 0,
    season_mpg: float = 0,
    recent_ppg: float = 0,
    team: str = "",
    # Config overrides
    ceiling: float = 3.0,
    floor: float = 0.0,
) -> dict:
    """Predict card boost using 3-tier cascade.

    Returns dict with:
      - boost: predicted boost (rounded to 0.1, clamped to [floor, ceiling])
      - tier: 1, 2, or 3
      - confidence: "high", "medium", or "low"
      - confidence_band: (low, high) range
      - reason: human-readable explanation
    """
    history = load_player_history()
    key = _normalize_name(player_name)

    # Parse today's date
    if today_str:
        try:
            today = datetime.strptime(today_str, "%Y-%m-%d").date()
        except ValueError:
            today = date.today()
    else:
        today = date.today()

    player_entries = history.get(key, [])
    # All entries from non-all-zero dates have valid boost data (including 0.0 for superstars)
    entries_with_boost = [e for e in player_entries if e.get("has_boost_data", False)]

    if not entries_with_boost:
        # Tier 3: Cold start
        return _predict_tier3(
            season_ppg=season_ppg,
            season_rpg=season_rpg,
            season_apg=season_apg,
            season_mpg=season_mpg,
            ceiling=ceiling,
            floor=floor,
        )

    # Determine gap days since most recent appearance
    most_recent = entries_with_boost[-1]
    try:
        last_date = datetime.strptime(most_recent["date"], "%Y-%m-%d").date()
        gap_days = (today - last_date).days
    except ValueError:
        gap_days = 999

    if gap_days <= _TIER1_MAX_GAP_DAYS and len(entries_with_boost) >= 1:
        return _predict_tier1(
            entries_with_boost,
            gap_days=gap_days,
            season_mpg=season_mpg,
            ceiling=ceiling,
            floor=floor,
        )
    else:
        return _predict_tier2(
            entries_with_boost,
            gap_days=gap_days,
            season_ppg=season_ppg,
            season_rpg=season_rpg,
            season_apg=season_apg,
            season_mpg=season_mpg,
            ceiling=ceiling,
            floor=floor,
        )


def _predict_tier1(
    entries: list[dict],
    gap_days: int,
    season_mpg: float,
    ceiling: float,
    floor: float,
) -> dict:
    """Tier 1: Returning player — high accuracy from prev_boost + adjustments."""
    prev = entries[-1]
    base = prev["boost"]
    prev_rs = prev["rs"]
    prev_drafts = prev["drafts"]

    # Compute historical stats (0.0 is a valid boost for superstars)
    all_boosts = [e["boost"] for e in entries]
    all_rs = [e["rs"] for e in entries if e["rs"] > 0]
    hist_boost_mean = sum(all_boosts) / len(all_boosts) if all_boosts else base
    hist_rs_mean = sum(all_rs) / len(all_rs) if all_rs else prev_rs
    tier_baseline = get_tier_baseline(hist_rs_mean)

    adjustments = []

    # Factor 1: RS performance decay — high RS → boost drops, low RS → boost rises
    if prev_rs >= 7.0:
        base -= 0.1
        adjustments.append("monster game last out (-0.1)")
    elif prev_rs >= 5.0:
        base -= 0.05
        adjustments.append("strong game last out (-0.05)")
    elif prev_rs < 2.0 and prev_rs > 0:
        base += 0.1
        adjustments.append("bust game last out (+0.1)")

    # Factor 2: Draft popularity effect
    if prev_drafts > 1000:
        base -= 0.05
        adjustments.append(f"heavily drafted ({int(prev_drafts)}) (-0.05)")
    elif prev_drafts < 10 and prev_drafts > 0:
        base += 0.05
        adjustments.append(f"ignored ({int(prev_drafts)} drafts) (+0.05)")

    # Factor 3: Mean reversion toward tier baseline (5% pull)
    reversion = (tier_baseline - base) * 0.05
    if abs(reversion) >= 0.01:
        base += reversion
        adjustments.append(f"mean reversion toward {tier_baseline:.1f} ({reversion:+.2f})")

    # Factor 4: Trend continuation (if 2+ data points)
    if len(entries) >= 2:
        prev2 = entries[-2]
        if prev2["boost"] > 0:
            trend = prev["boost"] - prev2["boost"]
            trend_adj = trend * 0.15  # 15% trend continuation
            if abs(trend_adj) >= 0.01:
                base += trend_adj
                adjustments.append(f"trend {trend:+.1f} continuation ({trend_adj:+.2f})")

    # Factor 5: Gap days (returning from absence)
    if gap_days > 5:
        gap_weight = min(gap_days / 30, 0.3)
        old_base = base
        base = base * (1 - gap_weight) + tier_baseline * gap_weight
        diff = base - old_base
        if abs(diff) >= 0.01:
            adjustments.append(f"{gap_days}d gap, blend toward baseline ({diff:+.2f})")

    # Factor 6: Minutes change signal (API data vs history)
    if season_mpg > 0 and len(entries) >= 3:
        # Rough historical minutes proxy from drafts (higher drafts ≈ more minutes/fame)
        recent_drafts_avg = sum(e["drafts"] for e in entries[-3:]) / 3
        if recent_drafts_avg > 500 and prev_drafts > recent_drafts_avg * 1.3:
            base -= 0.05  # Getting more popular → boost likely drops
            adjustments.append("rising popularity (-0.05)")

    # Special cases: boundary persistence
    if prev["boost"] >= 3.0 and base >= 2.7:
        # 3.0 → 3.0 in ~75% of cases — strong persistence at ceiling
        base = max(base, 2.9)
    if prev["boost"] <= 0.1 and hist_rs_mean >= 5.5:
        # Stars with 0.0 boost almost never gain boost
        base = min(base, 0.2)

    predicted = _clamp_round(base, floor, ceiling)

    # Confidence band — Tier 1 is tight
    band_width = 0.15
    if gap_days > 7:
        band_width = 0.25

    return {
        "boost": predicted,
        "tier": 1,
        "confidence": "high",
        "confidence_band": (
            _clamp_round(predicted - band_width, floor, ceiling),
            _clamp_round(predicted + band_width, floor, ceiling),
        ),
        "reason": f"Tier 1 (returning, {gap_days}d gap): prev {prev['boost']:.1f} → {predicted:.1f}"
                  + (f" [{'; '.join(adjustments)}]" if adjustments else ""),
        "prev_boost": prev["boost"],
        "hist_boost_mean": round(hist_boost_mean, 2),
    }


def _predict_tier2(
    entries: list[dict],
    gap_days: int,
    season_ppg: float,
    season_rpg: float,
    season_apg: float,
    season_mpg: float,
    ceiling: float,
    floor: float,
) -> dict:
    """Tier 2: Known player, stale data — blend history with API estimate."""
    all_boosts = [e["boost"] for e in entries]
    hist_boost_mean = sum(all_boosts) / len(all_boosts) if all_boosts else 1.5

    # API-based estimate (same as Tier 3)
    api_estimate = estimate_boost_from_api(season_ppg, season_rpg, season_apg, season_mpg)

    # Staleness weight: 0 at 14 days, 1.0 at 44+ days
    staleness = min((gap_days - _TIER1_MAX_GAP_DAYS) / 30, 1.0)

    # As data gets staler, rely more on API stats
    predicted_raw = hist_boost_mean * (1 - staleness * 0.5) + api_estimate * (staleness * 0.5)
    predicted = _clamp_round(predicted_raw, floor, ceiling)

    # Wider confidence band
    band_width = 0.3 + staleness * 0.2

    return {
        "boost": predicted,
        "tier": 2,
        "confidence": "medium",
        "confidence_band": (
            _clamp_round(predicted - band_width, floor, ceiling),
            _clamp_round(predicted + band_width, floor, ceiling),
        ),
        "reason": f"Tier 2 (stale, {gap_days}d): hist_mean {hist_boost_mean:.1f}, "
                  f"api_est {api_estimate:.1f}, staleness {staleness:.0%} → {predicted:.1f}",
        "hist_boost_mean": round(hist_boost_mean, 2),
    }


def _predict_tier3(
    season_ppg: float,
    season_rpg: float,
    season_apg: float,
    season_mpg: float,
    ceiling: float,
    floor: float,
) -> dict:
    """Tier 3: Cold start — never seen on Real Sports."""
    api_estimate = estimate_boost_from_api(season_ppg, season_rpg, season_apg, season_mpg)
    predicted = _clamp_round(api_estimate, floor, ceiling)

    # Widest confidence band
    band_width = 0.5

    return {
        "boost": predicted,
        "tier": 3,
        "confidence": "low",
        "confidence_band": (
            _clamp_round(predicted - band_width, floor, ceiling),
            _clamp_round(predicted + band_width, floor, ceiling),
        ),
        "reason": f"Tier 3 (cold start): PQI from stats (ppg={season_ppg:.1f}) → {predicted:.1f}",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_round(x: float, floor_val: float, ceiling: float) -> float:
    """Clamp to [floor, ceiling] and round to nearest 0.1 (Real Sports precision)."""
    # Special case: if prediction is ≥2.8, snap to 3.0 (26.1% of all boosts are 3.0)
    if x >= 2.8:
        x = 3.0
    return round(min(max(float(x), floor_val), ceiling), 1)


def get_player_history_summary(player_name: str) -> dict | None:
    """Return summary stats for a player's boost history (for debugging/UI)."""
    history = load_player_history()
    key = _normalize_name(player_name)
    entries = history.get(key)
    if not entries:
        return None

    entries_with_boost = [e for e in entries if e.get("has_boost_data", False)]
    if not entries_with_boost:
        return None

    boosts = [e["boost"] for e in entries_with_boost]
    rs_vals = [e["rs"] for e in entries_with_boost if e["rs"] > 0]
    drafts_vals = [e["drafts"] for e in entries_with_boost if e["drafts"] > 0]

    return {
        "appearances": len(entries_with_boost),
        "boost_mean": round(sum(boosts) / len(boosts), 2),
        "boost_min": min(boosts),
        "boost_max": max(boosts),
        "boost_last": entries_with_boost[-1]["boost"],
        "rs_mean": round(sum(rs_vals) / len(rs_vals), 2) if rs_vals else 0,
        "drafts_mean": round(sum(drafts_vals) / len(drafts_vals), 0) if drafts_vals else 0,
        "last_date": entries_with_boost[-1]["date"],
        "first_date": entries_with_boost[0]["date"],
    }
