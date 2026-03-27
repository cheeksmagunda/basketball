#!/usr/bin/env python3
"""
Train LightGBM card boost model from historical outcomes + prediction features.

Direct 8-feature model:
  [projected_rs, season_pts, recent_pts, season_min, pred_min,
   team_market_score, pos_bucket, ppg_tier]

Replaces the old binary is_big_market flag with a continuous team_market_score
(0.0–1.0) covering all 30 teams, plus a ppg_tier feature (0–4) that segments
player recognition tiers without a hard PPG threshold cliff.

Labels:
  actual_card_boost from top_performers + actuals + most_popular (merged, deduped)

Sample weighting:
  gold   = 3.0 (same-date prediction match exists)
  silver = 1.5 (player has prediction history, but no same-date match)
"""

import csv
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np

TOP_PERFORMERS = Path("data/top_performers.csv")
ACTUALS_DIR = Path("data/actuals")
MOST_POPULAR_DIR = Path("data/most_popular")
PREDICTIONS_DIR = Path("data/predictions")
MODEL_CONFIG = Path("data/model-config.json")
MODEL_JSON = Path("boost_model.json")
MODEL_TXT = Path("boost_model.txt")

FEATURES = [
    "projected_rs",
    "season_pts",
    "recent_pts",
    "season_min",
    "pred_min",
    "team_market_score",
    "pos_bucket",
    "ppg_tier",
]

# Continuous team market score (0.0–1.0) covering all 30 teams.
# Reflects fanbase size + franchise recognition + playoff relevance.
# Higher score = more fans drafting from that team = lower expected card boost.
# Must stay in sync with TEAM_MARKET_SCORES in api/index.py.
TEAM_MARKET_SCORES = {
    "LAL": 1.00, "GSW": 0.95, "GS": 0.95, "BOS": 0.90, "NYK": 0.90, "NY": 0.90,
    "PHI": 0.75, "MIA": 0.75, "LAC": 0.70, "CHI": 0.70,
    "BKN": 0.65, "DEN": 0.65, "MIL": 0.60, "DAL": 0.60,
    "HOU": 0.55, "PHX": 0.55, "ATL": 0.50, "TOR": 0.50,
    "CLE": 0.45, "IND": 0.40, "ORL": 0.35, "POR": 0.35,
    "DET": 0.30, "MIN": 0.30, "OKC": 0.25, "UTA": 0.25,
    "SAS": 0.25, "NOP": 0.20, "NO": 0.20, "MEM": 0.20,
    "CHA": 0.15, "SAC": 0.15, "WSH": 0.10,
}
LOOKBACK_DAYS = 14
SILVER_SINGLE_MIN_RS = 4.0


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_date(s: str) -> Optional[datetime]:
    try:
        return datetime.strptime((s or "").strip(), "%Y-%m-%d")
    except Exception:
        return None


def _pos_bucket(pos: str) -> int:
    """Must match _draft_pos_bucket() logic in api/index.py."""
    p = (pos or "").strip().upper()
    if not p:
        return 3
    if p.startswith("C"):
        return 2
    if p[0] in ("G", "P") or p.startswith("PG") or p.startswith("SG"):
        return 0
    return 1


def _get_team_market_score(team: str) -> float:
    """Return continuous market score (0.0–1.0) for a team abbreviation.

    Higher = larger/more engaged fanbase = player more likely to be drafted
    regardless of their stats = lower expected card boost.
    Defaults to 0.3 for unknown teams (small-market neutral).
    """
    return TEAM_MARKET_SCORES.get((team or "").strip().upper(), 0.3)


def _ppg_tier_bucket(season_pts: float) -> int:
    """Coarse PPG tier (0–4) for player recognition.

    Segments players into popularity tiers without a hard threshold cliff.
    Must stay in sync with _ppg_tier_bucket() in api/index.py.
    """
    if season_pts < 8:
        return 0   # bench/fringe — truly obscure
    if season_pts < 13:
        return 1   # role player — recognizable but not star
    if season_pts < 18:
        return 2   # secondary scorer
    if season_pts < 24:
        return 3   # main option
    return 4       # star/franchise — nationally known, heavily drafted


def _load_prediction_index() -> tuple[dict, dict]:
    """
    Build prediction indices from data/predictions/*.csv.

    Returns:
      by_player_date[(player, date)] = row features
      by_player[player] = sorted list of row features across dates
    """
    by_player_date: dict[tuple[str, str], dict] = {}
    by_player: dict[str, list[dict]] = defaultdict(list)
    if not PREDICTIONS_DIR.exists():
        return by_player_date, by_player

    for csv_path in sorted(PREDICTIONS_DIR.glob("*.csv")):
        date_str = csv_path.stem
        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player = (row.get("player_name") or "").strip()
                if not player:
                    continue
                scope = (row.get("scope") or "").strip().lower()
                if scope and scope != "slate":
                    continue

                projected_rs = None
                for col in ("predicted_rs", "proj_rs", "rating", "projected_rs"):
                    if row.get(col):
                        projected_rs = _safe_float(row.get(col), 0.0)
                        if projected_rs > 0:
                            break
                if not projected_rs or projected_rs <= 0:
                    continue

                pts = _safe_float(row.get("pts"), 0.0)
                pred_min = _safe_float(row.get("pred_min") or row.get("proj_min"), 0.0)
                team = (row.get("team") or row.get("team_abbr") or "").strip().upper()
                pos = (row.get("pos") or row.get("player_pos") or "").strip().upper()

                item = {
                    "date": date_str,
                    "projected_rs": projected_rs,
                    "pts": pts,
                    "pred_min": pred_min,
                    "team": team,
                    "pos": pos,
                    "dt": _parse_date(date_str),
                }
                by_player_date[(player, date_str)] = item
                by_player[player].append(item)

    for player in by_player:
        by_player[player].sort(key=lambda r: r["date"])
    return by_player_date, by_player


def _load_top_performers() -> list[dict]:
    if not TOP_PERFORMERS.exists():
        print(f"[ERROR] Missing {TOP_PERFORMERS}")
        return []
    rows = []
    with TOP_PERFORMERS.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            boost = _safe_float(r.get("actual_card_boost"), 0.0)
            rs = _safe_float(r.get("actual_rs"), 0.0)
            name = (r.get("player_name") or "").strip()
            date_str = (r.get("date") or "").strip()
            if boost > 0 and rs > 0 and name and date_str:
                rows.append(
                    {
                        "date": date_str,
                        "player_name": name,
                        "actual_card_boost": boost,
                    }
                )
    return rows


def _load_actuals() -> list[dict]:
    if not ACTUALS_DIR.exists():
        return []
    rows = []
    for csv_path in sorted(ACTUALS_DIR.glob("*.csv")):
        date_str = csv_path.stem
        with csv_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                boost = _safe_float(r.get("actual_card_boost"), 0.0)
                rs = _safe_float(r.get("actual_rs"), 0.0)
                name = (r.get("player_name") or "").strip()
                if boost > 0 and rs > 0 and name:
                    rows.append(
                        {
                            "date": date_str,
                            "player_name": name,
                            "actual_card_boost": boost,
                        }
                    )
    return rows


def _load_most_popular() -> list[dict]:
    if not MOST_POPULAR_DIR.is_dir():
        return []
    rows: list[dict] = []
    for csv_path in sorted(MOST_POPULAR_DIR.glob("*.csv")):
        date_str = csv_path.stem
        with csv_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                boost = _safe_float(r.get("actual_card_boost"), 0.0)
                rs = _safe_float(r.get("actual_rs"), 0.0)
                name = (r.get("player") or r.get("player_name") or "").strip()
                if boost > 0 and rs > 0 and name:
                    rows.append(
                        {
                            "date": date_str,
                            "player_name": name,
                            "actual_card_boost": boost,
                        }
                    )
    return rows


def _player_agg(history: list[dict]) -> dict:
    pts_vals = [r["pts"] for r in history if r["pts"] > 0]
    min_vals = [r["pred_min"] for r in history if r["pred_min"] > 0]
    rs_vals = [r["projected_rs"] for r in history if r["projected_rs"] > 0]
    season_pts = float(np.mean(pts_vals)) if pts_vals else 0.0
    season_min = float(np.mean(min_vals)) if min_vals else 0.0
    mean_rs = float(np.mean(rs_vals)) if rs_vals else 0.0
    team = next((r["team"] for r in history if r["team"]), "")
    pos = next((r["pos"] for r in history if r["pos"]), "")
    return {
        "season_pts": season_pts,
        "season_min": season_min,
        "mean_rs": mean_rs,
        "team": team,
        "pos": pos,
    }


def _recent_pts_14d(history: list[dict], current_date: str, fallback: float) -> float:
    cur_dt = _parse_date(current_date)
    if not cur_dt:
        return fallback
    vals = []
    for row in history:
        dt = row.get("dt")
        if not dt or dt >= cur_dt:
            continue
        delta = (cur_dt - dt).days
        if 0 < delta <= LOOKBACK_DAYS and row["pts"] > 0:
            vals.append(row["pts"])
    if vals:
        return float(np.mean(vals))
    return fallback


def build_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_player_date, by_player = _load_prediction_index()
    all_rows = _load_top_performers() + _load_actuals() + _load_most_popular()

    seen: set[tuple[str, str]] = set()
    labels: list[dict] = []
    for r in all_rows:
        key = (r["player_name"], r["date"])
        if key not in seen:
            seen.add(key)
            labels.append(r)

    if not labels:
        return np.array([]), np.array([]), np.array([])

    X_list: list[list[float]] = []
    y_list: list[float] = []
    w_list: list[float] = []
    gold = 0
    silver = 0
    skipped = 0

    for row in labels:
        name = row["player_name"]
        date_str = row["date"]
        cur_dt = _parse_date(date_str)
        if not cur_dt:
            skipped += 1
            continue
        same_day = by_player_date.get((name, date_str))
        history = by_player.get(name, [])
        if not history:
            skipped += 1
            continue

        history_prior = [h for h in history if h.get("dt") and h["dt"] < cur_dt]
        # Use history_prior for aggregates to avoid temporal leakage —
        # season_pts/season_min must only reflect games BEFORE the label date.
        agg = _player_agg(history_prior) if history_prior else _player_agg(history)
        season_pts = agg["season_pts"]
        season_min = agg["season_min"]
        recent_pts = _recent_pts_14d(history_prior, date_str, season_pts)

        if same_day:
            projected_rs = float(same_day["projected_rs"])
            pred_min = float(same_day["pred_min"]) if same_day["pred_min"] > 0 else season_min
            team = same_day["team"] or agg["team"]
            pos = same_day["pos"] or agg["pos"]
            weight = 3.0
            gold += 1
        else:
            projected_rs = agg["mean_rs"]
            if len(history) < 2 and projected_rs < SILVER_SINGLE_MIN_RS:
                skipped += 1
                continue
            pred_min = season_min
            team = agg["team"]
            pos = agg["pos"]
            weight = 1.5
            silver += 1

        if projected_rs <= 0:
            skipped += 1
            continue

        feat_vec = [
            projected_rs,
            season_pts,
            recent_pts,
            season_min,
            pred_min if pred_min > 0 else season_min,
            _get_team_market_score(team),
            float(_pos_bucket(pos)),
            float(_ppg_tier_bucket(season_pts)),
        ]

        X_list.append(feat_vec)
        y_list.append(float(row["actual_card_boost"]))
        w_list.append(weight)

    print(f"Combined labels: {len(labels)} unique (player, date)")
    print(f"Training rows: {len(X_list)} (skipped: {skipped})")
    print(f"Sample weighting: gold={gold} silver={silver}")
    return np.array(X_list), np.array(y_list), np.array(w_list)


def _print_validation_grid(model: lgb.LGBMRegressor) -> None:
    print("\nValidation grid (ppg tier x projected RS x market):")
    print("Expected direction: higher season_pts / higher market -> lower boost at fixed RS")
    # [projected_rs, season_pts, recent_pts, season_min, pred_min, team_market_score, pos_bucket, ppg_tier]
    test_cases = [
        (8.0, 20.0, 0.10, 1),   # role player, small market (e.g. UTA bench)
        (9.0, 22.0, 0.40, 1),   # role player, mid market (e.g. IND/CLE role player — TJ McConnell)
        (24.0, 35.0, 0.45, 4),  # star, mid market (e.g. Harden on CLE)
        (24.0, 35.0, 1.00, 4),  # star, top market (e.g. LeBron on LAL)
    ]
    rs_levels = [3.0, 5.0, 7.0, 10.0]
    labels = ["bench/sm-mkt", "role/mid-mkt (McConnell)", "star/mid-mkt (Harden)", "star/top-mkt (LeBron)"]
    for (ppg, smin, mkt, tier), label in zip(test_cases, labels):
        preds = []
        for rs in rs_levels:
            vec = [rs, ppg, ppg, smin, smin, mkt, 0.0, float(tier)]
            pred = float(model.predict([vec])[0])
            preds.append(f"RS{rs:.0f}:{pred:.2f}")
        print(f"  {label:30s} -> " + " | ".join(preds))


def train_model() -> None:
    print("Building training data from top_performers + actuals + most_popular...")
    X, y, weights = build_training_data()
    if len(X) == 0:
        print("[ERROR] No valid rows. Aborting.")
        return

    print(f"Training boost model on {len(X)} samples")
    print(f"Features: {FEATURES}")
    print(f"Target range: {y.min():.2f} - {y.max():.2f}")

    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=3,
        num_leaves=10,
        min_child_samples=10,
        reg_lambda=1.0,
        subsample=0.9,
        colsample_bytree=1.0,
        objective="regression",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y, sample_weight=weights)

    model.booster_.save_model(str(MODEL_TXT))
    # Strip training-only params that cause harmless but noisy warnings on load
    _txt = MODEL_TXT.read_text()
    for _param in ("early_stopping_min_delta", "bagging_by_query"):
        _txt = re.sub(rf"^{_param}=.*\n", "", _txt, flags=re.MULTILINE)
        _txt = re.sub(rf"^\[{_param}:.*\]\n", "", _txt, flags=re.MULTILINE)
    MODEL_TXT.write_text(_txt)
    meta = {"format": "lightgbm_native", "features": FEATURES, "model_file": MODEL_TXT.name}
    with MODEL_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    print(f"Saved {MODEL_TXT} + {MODEL_JSON}")
    _print_validation_grid(model)


if __name__ == "__main__":
    train_model()
