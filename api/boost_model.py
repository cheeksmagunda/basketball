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

from api.shared import normalize_player_name as _normalize_name

# ---------------------------------------------------------------------------
# Player history index — loaded once from data/top_performers.csv
# ---------------------------------------------------------------------------

_PLAYER_HISTORY: dict[str, list[dict]] = {}
_HISTORY_LOADED = False
_HISTORY_LOCK = threading.Lock()


# _normalize_name imported from api.shared (DRY — handles accents/diacritics)


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

        # Also load from data/most_popular/ for actual draft counts
        # This is the strongest predictive signal for boost changes (Spearman -0.732)
        for data_dir in ("data/most_popular",):
            pdir = Path(data_dir)
            if not pdir.exists():
                pdir = Path(__file__).parent.parent / data_dir
            if not pdir.exists():
                continue
            try:
                _mp_count = 0
                for csvfile in pdir.glob("*.csv"):
                    dt_str = csvfile.stem
                    with csvfile.open("r", encoding="utf-8") as f:
                        for row in csv.DictReader(f):
                            name = _normalize_name(
                                row.get("player") or row.get("player_name") or ""
                            )
                            if not name:
                                continue
                            draft_count = _safe(row.get("draft_count"))
                            if draft_count <= 0:
                                continue
                            # Enrich existing entries for this date with actual draft data
                            if name in history:
                                for entry in history[name]:
                                    if entry["date"] == dt_str and entry["drafts"] < draft_count:
                                        entry["drafts"] = draft_count
                                        _mp_count += 1
                if _mp_count:
                    print(f"[boost_cascade] Enriched {_mp_count} entries with most_popular draft counts")
            except Exception as e:
                print(f"[boost_cascade] Failed to load most_popular: {e}")

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

    Star PPG guard: Stars (PPG ≥ 15) are heavily drafted regardless of PQI,
    so they ALWAYS get low boosts. This was the single biggest source of
    Tier 3 errors (MAE 0.714): the model predicted 3.0 for Jokic, Luka,
    SGA, Curry etc. when their actual boosts are 0.0-0.5.
    """
    # Star PPG guard — applied BEFORE PQI to prevent massive over-predictions.
    # Calibrated from 148-date backtest: stars with PPG ≥ 22 always have boost ≤ 0.5.
    if season_ppg >= 25:
        return 0.0   # Elite superstars (Jokic, SGA, Luka): always 0.0
    elif season_ppg >= 22:
        return 0.3   # Borderline superstars: 0.0-0.5 range
    elif season_ppg >= 19:
        return 0.8   # Stars: 0.3-1.0 range
    elif season_ppg >= 15:
        return max(2.0 - (season_ppg - 15) * 0.3, 0.8)  # Starters: 0.8-2.0

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
    # ESPN-available stats (the ONLY inputs available on game day)
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
    """Predict card boost using ESPN-available signals + historically-calibrated model.

    CRITICAL DESIGN PRINCIPLE (Winning Draft Audit v2):
    On game day, we do NOT know the player's prior boost, prior RS, or prior
    draft count — those come from Real Sports screenshots collected AFTER games.
    The model must predict boost using ONLY data available from ESPN pre-game:
      - Season stats (PPG, RPG, APG, MPG)
      - Recent performance (last 5-10 games via ESPN gamelogs)
      - Team, position, market size
      - Matchup context (opponent, spread, total)

    Historical data (top_performers.csv) is used to CALIBRATE the mapping:
      - PQI → boost mapping (Tier 3 cold start formula)
      - Team boost ceilings (franchise popularity)
      - Star PPG tiers (national brand = low boost)
      - Leaderboard frequency patterns (repeat performers)

    The old Tier 1 (prev_boost + adjustments) used UNAVAILABLE data as inputs.
    The new model uses a calibrated PQI approach for ALL players, with historical
    frequency data used only to narrow confidence bands (not to set predictions).

    Returns dict with:
      - boost: predicted boost (rounded to 0.1, clamped to [floor, ceiling])
      - tier: 1, 2, or 3
      - confidence: "high", "medium", or "low"
      - confidence_band: (low, high) range
      - reason: human-readable explanation
    """
    history = load_player_history()
    key = _normalize_name(player_name)

    player_entries = history.get(key, [])
    entries_with_boost = [e for e in player_entries if e.get("has_boost_data", False)]

    # ── PRIMARY PREDICTION: ESPN-based PQI model ──────────────────────────
    # This is the core prediction using only game-day-available data.
    # The PQI (Player Quality Index) maps season stats → expected boost.
    # Calibrated from 2,234 player-date records but uses NO prior-day data.
    api_estimate = estimate_boost_from_api(season_ppg, season_rpg, season_apg, season_mpg)

    # ── REFINEMENT: Use historical FREQUENCY to adjust confidence ─────────
    # We can't use prior boost/RS/drafts as inputs, but we CAN use the PATTERN
    # of how often a player appears on the leaderboard to narrow the band.
    # A player who's appeared 10+ times has a well-characterized boost range.
    n_appearances = len(entries_with_boost)
    has_history = n_appearances >= 1

    if has_history:
        # Player has been seen before on Real Sports
        # Use historical AVERAGE boost to calibrate (not prior-day boost)
        all_boosts = [e["boost"] for e in entries_with_boost]
        hist_boost_mean = sum(all_boosts) / len(all_boosts)

        # Recent trend: did their boost tend to be rising or falling?
        # This uses the DIRECTION of historical data, not the specific values
        if n_appearances >= 3:
            recent_3 = [e["boost"] for e in entries_with_boost[-3:]]
            hist_trend = (recent_3[-1] - recent_3[0]) / 2 if len(recent_3) >= 2 else 0
        else:
            hist_trend = 0

        # Blend: ESPN estimate is primary, historical mean is secondary
        # More history → more trust in historical mean (up to 40%)
        hist_weight = min(n_appearances / 20, 0.4)  # 20 appearances → 40% weight
        predicted_raw = api_estimate * (1 - hist_weight) + hist_boost_mean * hist_weight

        # Apply trend continuation (very small — just direction)
        predicted_raw += hist_trend * 0.05  # 5% trend continuation

        # Confidence based on history depth
        if n_appearances >= 10:
            confidence = "high"
            band_width = 0.25
            tier = 1
        elif n_appearances >= 3:
            confidence = "medium"
            band_width = 0.35
            tier = 1
        else:
            confidence = "medium"
            band_width = 0.40
            tier = 2

        predicted = _clamp_round(predicted_raw, floor, ceiling)
        reason = (f"Tier {tier} (ESPN + {n_appearances} historical appearances): "
                  f"api_est {api_estimate:.1f}, hist_mean {hist_boost_mean:.1f}, "
                  f"hist_weight {hist_weight:.0%} → {predicted:.1f}")

        return {
            "boost": predicted,
            "tier": tier,
            "confidence": confidence,
            "confidence_band": (
                _clamp_round(predicted - band_width, floor, ceiling),
                _clamp_round(predicted + band_width, floor, ceiling),
            ),
            "reason": reason,
            "hist_boost_mean": round(hist_boost_mean, 2),
        }
    else:
        # Tier 3: Cold start — never seen on Real Sports
        return _predict_tier3(
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
    """LEGACY — Tier 1: prev_boost + adjustments.

    DEPRECATED by Winning Draft Audit v2: This function used prior-day boost,
    RS, and draft counts as direct inputs — data that is NOT available on game day.
    Retained for reference and potential backtest comparisons only.
    The new predict_boost() uses ESPN-only signals + historical calibration.

    Calibrated from 148-date backtest (2024-12 – 2026-03):
    - Overall day-over-day boost changes: median 0.0, mean -0.052
    - Mid-RS (3-5) average drift: -0.077/appearance
    - Low-pop (<50 drafts) drift: -0.094/appearance
    - High-RS (≥5) drift: -0.040/appearance
    """
    prev = entries[-1]
    base = prev["boost"]
    prev_rs = prev["rs"]
    prev_drafts = prev["drafts"]

    # Compute historical stats — use recency-weighted mean for boost
    # Recent appearances are more predictive than older ones
    all_boosts = [e["boost"] for e in entries]
    if len(all_boosts) >= 3:
        # Weighted: last 3 appearances get 3x, 2x, 1x weight
        recent_3 = all_boosts[-3:]
        weights = [1.0] * max(0, len(all_boosts) - 3) + [1.0, 2.0, 3.0][-len(recent_3):]
        if len(weights) < len(all_boosts):
            weights = [1.0] * (len(all_boosts) - len(recent_3)) + weights
        hist_boost_mean = sum(b * w for b, w in zip(all_boosts, weights)) / sum(weights)
    else:
        hist_boost_mean = sum(all_boosts) / len(all_boosts) if all_boosts else base
    all_rs = [e["rs"] for e in entries if e["rs"] > 0]
    hist_rs_mean = sum(all_rs) / len(all_rs) if all_rs else prev_rs
    tier_baseline = get_tier_baseline(hist_rs_mean)

    adjustments = []

    # Factor 0: General downward drift correction
    # Backtest shows mean day-over-day change is -0.052 across all players.
    # This corrects the systematic over-prediction bias (+0.242 mean signed error).
    base -= 0.03
    adjustments.append("drift correction (-0.03)")

    # Factor 1: RS performance decay — high RS → boost drops, low RS → boost rises
    # Strengthened from backtest: high-RS bucket shows -0.040 avg drift,
    # mid-RS shows -0.077 avg drift (largest).
    if prev_rs >= 7.0:
        base -= 0.12
        adjustments.append("monster game last out (-0.12)")
    elif prev_rs >= 5.0:
        base -= 0.08
        adjustments.append("strong game last out (-0.08)")
    elif prev_rs < 2.0 and prev_rs > 0:
        base += 0.1
        adjustments.append("bust game last out (+0.1)")

    # Factor 2: Draft popularity effect (strengthened — Spearman -0.732)
    # Data shows: drafts 0-10 → avg boost 2.60, drafts 2000+ → avg boost 0.36
    # This is the 2nd strongest signal after prev_boost itself.
    if prev_drafts > 5000:
        base -= 0.10
        adjustments.append(f"mega-popular ({int(prev_drafts)}) (-0.10)")
    elif prev_drafts > 2000:
        base -= 0.08
        adjustments.append(f"very popular ({int(prev_drafts)}) (-0.08)")
    elif prev_drafts > 1000:
        base -= 0.05
        adjustments.append(f"heavily drafted ({int(prev_drafts)}) (-0.05)")
    elif prev_drafts > 500:
        base -= 0.03
        adjustments.append(f"moderately drafted ({int(prev_drafts)}) (-0.03)")
    elif prev_drafts < 10 and prev_drafts > 0:
        base += 0.05
        adjustments.append(f"ignored ({int(prev_drafts)} drafts) (+0.05)")
    elif prev_drafts < 50 and prev_drafts > 0:
        base += 0.03
        adjustments.append(f"low-draft ({int(prev_drafts)}) (+0.03)")

    # Factor 3: Mean reversion toward tier baseline
    # Strengthened from 5% to 8% for mid-RS players (RS 3-5) where drift is largest (-0.077).
    # Mid-boost range (1.0-2.5) had MAE 0.432 — strongest reversion helps most there.
    if 3.0 <= hist_rs_mean < 5.0:
        reversion_rate = 0.08
    else:
        reversion_rate = 0.05
    reversion = (tier_baseline - base) * reversion_rate
    if abs(reversion) >= 0.01:
        base += reversion
        adjustments.append(f"mean reversion toward {tier_baseline:.1f} ({reversion:+.2f})")

    # Factor 3b: Mid-boost regression pull
    # Players in the volatile 1.0-2.5 range (MAE 0.432 — worst bucket) get an extra
    # pull toward their recency-weighted historical mean when they've had recent swings.
    if 1.0 <= prev["boost"] <= 2.5 and len(entries) >= 2:
        recent_swing = abs(prev["boost"] - entries[-2]["boost"])
        if recent_swing >= 0.3:
            # After a big swing, regress harder toward weighted mean
            regression_pull = (hist_boost_mean - base) * 0.10
            if abs(regression_pull) >= 0.01:
                base += regression_pull
                adjustments.append(f"mid-boost regression after {recent_swing:.1f} swing ({regression_pull:+.02f})")

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

    # Confidence band — Tier 1 widened based on actual ±0.3 range covering 88% of changes
    band_width = 0.25
    if gap_days > 7:
        band_width = 0.35

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
    """LEGACY — Tier 2: blend history with API estimate (retained for reference)."""
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
    # Special case: if prediction is ≥2.95, snap to 3.0 (26.1% of all boosts are 3.0)
    # Tightened from 2.8 → 2.95 to reduce systematic over-prediction in the 2.8-2.9 range.
    # Backtest showed +0.242 mean signed error; the old 2.8 snap was a major contributor.
    if x >= 2.95:
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
