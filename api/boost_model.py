"""Boost Prediction Model — 3-Tier Historical Cascade.

# grep: BOOST CASCADE MODEL

Predicts Real Sports card boost (0.0–3.0) using a 3-tier cascade that
prioritizes historical boost data from top_performers.csv.

Key insight: prev_boost → actual_boost correlation is +0.955 (1,846 pairs).
88.2% of day-over-day changes are within ±0.3. MAE of naive prev_boost
predictor is 0.140 vs 0.996 for a static 2.5 predictor (7x better).

  predict_boost() — Primary entry point. Routes to tier based on history.
  _predict_tier1() — Returning player (≤14d gap): prev_boost + adjustments.
  _predict_tier2() — Stale player (>14d gap): blend hist mean with PQI.
  _predict_tier3() — Cold start (never seen): PQI from ESPN season stats.
  estimate_boost_from_api() — PQI formula (used by Tier 2/3 only).

Tier routing:
  Tier 1 — Last appearance ≤14 days ago     (prev_boost + 6 adjustment factors)
  Tier 2 — Last appearance >14 days ago     (hist mean blended with PQI estimate)
  Tier 3 — Never seen on Real Sports        (PQI from season stats)
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
# Tier baselines — kept for reference, used by deprecated _predict_tier1 only
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
    recent_ppg: float = 0,
) -> float:
    """Predict boost from ESPN-available signals only.

    Uses two complementary signals:
      1. MPG-based role detection (primary) — playing time directly reflects
         roster role, which is the strongest driver of draft popularity.
         Heavy minutes → starter/star role → low boost (heavily drafted).
         Light minutes → bench role → high boost (ignored by casual drafters).

      2. PPG-based quality index (secondary) — scoring + peripherals capture
         player prominence beyond just minutes.

    Plus a recent form modifier: hot-streak players attract more drafts next
    slate, so Real Sports tends to lower their boost. Cold-streak players
    attract fewer drafts, so their boost tends to be higher.

    All inputs are from ESPN pre-game data — no prior boost values are used.

    Calibrated from 2,234 player-date records (151 dates).
    """
    # ── Star PPG guard (unchanged, correct) ──────────────────────────────
    # Calibrated from 148-date backtest: stars with PPG ≥ 22 always have boost ≤ 0.5.
    # Applied BEFORE role detection to prevent over-prediction for superstars.
    if season_ppg >= 26:
        return 0.0    # Elite superstars (Jokic, SGA, Luka, Curry)
    elif season_ppg >= 19:
        # Linear ramp: 19 PPG → 1.2, 26 PPG → 0.0 (smooth gradient, no hard steps)
        return round(max(0.0, 1.2 - (season_ppg - 19) * (1.2 / 7.0)), 1)

    # ── MPG-based role detection (primary signal) ─────────────────────────
    # Minutes per game reflects rotation role more reliably than scoring.
    # Continuous mapping to avoid hard threshold artifacts.
    if season_mpg > 0:
        if season_mpg >= 30:
            # Heavy minutes → rotation anchor / feature starter
            mpg_boost = 0.6 + (30 - season_mpg) * 0.05  # 30 MPG→0.6, capped
            mpg_boost = max(mpg_boost, 0.3)
        elif season_mpg >= 24:
            # Starter minutes: 24 MPG → 2.0, 30 MPG → 0.6
            mpg_boost = 2.0 - (season_mpg - 24) / 6.0 * 1.4
        elif season_mpg >= 16:
            # Rotation minutes: 16 MPG → 2.8, 24 MPG → 2.0
            mpg_boost = 2.8 - (season_mpg - 16) / 8.0 * 0.8
        else:
            # Bench minutes: shallow decay toward 3.0
            mpg_boost = 3.0 - season_mpg * 0.013
        mpg_boost = float(max(0.0, mpg_boost))
    else:
        mpg_boost = None  # No MPG data — fall back to PPG-only

    # ── PPG quality index (secondary signal) ─────────────────────────────
    # Combined PQI from scoring + peripherals.
    pqi = season_ppg * 1.0 + season_rpg * 0.4 + season_apg * 0.6 + season_mpg * 0.15
    if pqi < 8:
        ppg_boost = 3.0
    elif pqi < 14:
        ppg_boost = 3.0 - (pqi - 8) * 0.083    # 2.5–3.0
    elif pqi < 22:
        ppg_boost = 2.5 - (pqi - 14) * 0.125   # 1.5–2.5
    elif pqi < 30:
        ppg_boost = 1.5 - (pqi - 22) * 0.125   # 0.5–1.5
    elif pqi < 38:
        ppg_boost = max(0.5 - (pqi - 30) * 0.063, 0.0)
    else:
        ppg_boost = 0.0

    # ── Blend: MPG role (55%) + PPG quality (45%) ────────────────────────
    if mpg_boost is not None:
        predicted = mpg_boost * 0.55 + ppg_boost * 0.45
    else:
        predicted = ppg_boost

    # ── Recent form modifier ──────────────────────────────────────────────
    # Hot streak (recent > season × 1.25): player becomes popular → boost
    # likely lower as Real Sports suppresses heavily-drafted players.
    # Cold streak (recent < season × 0.75): player ignored → boost likely
    # higher as RS tries to attract drafts.
    # This uses only ESPN L5 rolling data — no prior boost needed.
    if season_ppg > 0 and recent_ppg > 0:
        ratio = recent_ppg / season_ppg
        if ratio > 1.25:
            # Demand spike: small negative adjustment (max −0.1)
            adj = -0.1 * min((ratio - 1.25) / 0.5, 1.0)
            predicted += adj
        elif ratio < 0.75:
            # Demand drought: small positive adjustment (max +0.1)
            adj = 0.1 * min((0.75 - ratio) / 0.25, 1.0)
            predicted += adj

    return max(0.0, predicted)


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
    # ESPN-available stats
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
    """Predict card boost via 3-tier historical cascade.

    Routing logic:
      1. Load player's appearance history from top_performers.csv
      2. If player has appearances AND most recent is ≤14 days ago → Tier 1
         (prev_boost + adjustment factors; MAE ~0.14)
      3. If player has appearances but most recent is >14 days ago → Tier 2
         (blend historical boost mean with PQI estimate)
      4. If player has no appearances → Tier 3
         (PQI from ESPN season stats; reasonable for unknowns)

    Historical boost data (prev_boost, prev_rs, prev_drafts) is known post-game
    data from PRIOR slates — it is NOT leaked future data. Real Sports publishes
    leaderboards after each slate, making this information available.

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
    n_appearances = len(entries_with_boost)

    # ── Route to appropriate tier ─────────────────────────────────────────
    today = _parse_date(today_str) if today_str else date.today()

    # Only use entries from BEFORE today (can't use today's boost to predict today)
    if today:
        prior_entries = [e for e in entries_with_boost
                         if (_parse_date(e["date"]) or date.min) < today]
    else:
        prior_entries = entries_with_boost

    if prior_entries:
        last_date = _parse_date(prior_entries[-1]["date"])
        gap_days = (today - last_date).days if last_date and today else 999

        if gap_days <= _TIER1_MAX_GAP_DAYS:
            # Tier 1: Returning player — use prev_boost + adjustments
            return _predict_tier1(
                prior_entries, gap_days, season_mpg, ceiling, floor,
            )
        else:
            # Tier 2: Stale player — blend historical mean with PQI
            return _predict_tier2(
                prior_entries, gap_days,
                season_ppg, season_rpg, season_apg, season_mpg,
                ceiling, floor,
            )
    else:
        # Tier 3: Cold start — PQI from ESPN season stats
        return _predict_tier3(
            season_ppg, season_rpg, season_apg, season_mpg,
            ceiling, floor,
        )


def _predict_tier1(
    entries: list[dict],
    gap_days: int,
    season_mpg: float,
    ceiling: float,
    floor: float,
) -> dict:
    """Tier 1: Returning player (last appearance ≤14 days ago).

    Uses prev_boost as base, then applies 6 adjustment factors calibrated
    from 1,846 prev→actual pairs across 151 dates:
      - Factor 0: General downward drift (-0.03, corrects +0.242 bias)
      - Factor 1: RS performance decay (high RS → drop, bust → rise)
      - Factor 2: Draft popularity (Spearman -0.732 with boost)
      - Factor 3: Mean reversion toward tier baseline (5-8%)
      - Factor 4: Trend continuation (15% of last direction)
      - Factor 5: Gap-day blend toward baseline
      - Factor 6: Rising popularity signal

    Boundary persistence: 3.0 stays 3.0 in ~75% of cases; stars at 0.0
    almost never gain boost.

    Expected MAE: ~0.14 (vs 0.996 for static 2.5 predictor).
    """
    prev = entries[-1]
    base = prev["boost"]
    prev_rs = prev["rs"]
    prev_drafts = prev["drafts"]

    # Compute historical stats — use recency-weighted mean for boost
    # Recent appearances are more predictive than older ones
    all_boosts = [e["boost"] for e in entries]
    if len(all_boosts) >= 3:
        # Weighted: last 3 appearances get 3x, 2x, 1x weight; older entries get 1x
        recent_3 = all_boosts[-3:]
        weights = [1.0] * max(0, len(all_boosts) - 3) + [1.0, 2.0, 3.0][-len(recent_3):]
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
    elif prev_drafts >= 200:
        base -= 0.01
        adjustments.append(f"mid-draft ({int(prev_drafts)}) (-0.01)")
    elif prev_drafts >= 50:
        base += 0.01
        adjustments.append(f"low-mid draft ({int(prev_drafts)}) (+0.01)")
    elif prev_drafts < 50 and prev_drafts >= 10:
        base += 0.03
        adjustments.append(f"low-draft ({int(prev_drafts)}) (+0.03)")
    elif prev_drafts < 10 and prev_drafts > 0:
        base += 0.05
        adjustments.append(f"ignored ({int(prev_drafts)} drafts) (+0.05)")

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
    """Tier 2: Known player, stale (last appearance >14 days ago).

    Blends historical boost mean with PQI estimate, weighted by staleness.
    At 14 days: 100% historical mean. At 44+ days: 50/50 blend.
    """
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
    """Tier 3: Cold start (never seen on Real Sports).

    Uses PQI from ESPN season stats. Stars (25+ PPG) get low boost;
    bench players (low PPG/MPG) get high boost. Reasonable default
    when no historical data exists.
    """
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

def _parse_date(s: str | None) -> date | None:
    """Parse YYYY-MM-DD string to date, or None on failure."""
    if not s:
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


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
