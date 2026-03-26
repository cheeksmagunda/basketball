#!/usr/bin/env python3
"""
Train LightGBM card boost model from top_performers.csv.

Uses two features to predict card boost:
1. perf_score: 14-day trailing average RS (performance signal)
2. min_proxy: 12 + 5*log1p(drafts) (rotation level signal)

Target: actual_card_boost (0.0 to 3.0)

At inference (api/index.py):
  - perf_score = projected_rs from the RS projection pipeline
  - min_proxy = 12 + 5 * log1p(drafts_est) where drafts_est comes from drafts_model.pkl
    when present (trained on top_performers + predictions — stable role + market + pos),
    else legacy mapping from projected minutes.

RS (game performance) and draft popularity (name/market/role) are separated: popularity
is not assumed to track tonight's RS spike.

Writes boost_model.pkl as {"model": ..., "features": [...]}.

Optional calibration pass: after drafts_model.pkl has been retrained on broad
prediction/top_performers overlap and deployed, re-run this script so the saved
boost weights reflect any residual error. Training labels use `drafts` from
`top_performers.csv`, `data/actuals/`, and `data/most_popular/` (merged, de-duped).
"""

import csv
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np

TOP_PERFORMERS = Path("data/top_performers.csv")
ACTUALS_DIR = Path("data/actuals")
MOST_POPULAR_DIR = Path("data/most_popular")
PREDICTIONS_DIR = Path("data/predictions")
MODEL_OUT = Path("boost_model.pkl")

FEATURES = ["perf_score", "min_proxy"]
LOOKBACK_DAYS = 14


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _load_predictions() -> dict:
    """Load predicted RS values from data/predictions/*.csv.

    Returns dict of (player_name, date) → projected_rs.
    Looks for columns: predicted_rs, proj_rs, rating.
    Falls back to empty dict if predictions dir doesn't exist.

    TRAIN/INFERENCE ALIGNMENT FIX:
    At inference (api/index.py), perf_score uses projected_rs from the RS projection pipeline.
    At training time, we now use the same projected_rs from predictions CSVs (when available)
    instead of actual_rs. This ensures the model learns the relationship between pre-game
    estimates (what we actually know at draft time) and card boost, not post-game actuals.
    """
    if not PREDICTIONS_DIR.exists():
        return {}

    lookup = {}
    for csv_path in sorted(PREDICTIONS_DIR.glob("*.csv")):
        date_str = csv_path.stem  # e.g., "2026-03-20"
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    player = (row.get("player_name") or "").strip()
                    if not player:
                        continue

                    # Try multiple column names for projected RS
                    proj_rs = None
                    for col in ["predicted_rs", "proj_rs", "rating", "projected_rs"]:
                        if col in row and row[col]:
                            try:
                                proj_rs = float(row[col])
                                break
                            except (ValueError, TypeError):
                                continue

                    if proj_rs is not None and proj_rs > 0:
                        lookup[(player, date_str)] = proj_rs
        except Exception as e:
            print(f"[WARN] Could not read predictions from {csv_path}: {e}")

    return lookup


def _load_top_performers() -> list[dict]:
    """Load labeled examples from top_performers.csv."""
    if not TOP_PERFORMERS.exists():
        print(f"[ERROR] Missing {TOP_PERFORMERS}")
        return []
    rows = []
    with TOP_PERFORMERS.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            boost = _safe_float(r.get("actual_card_boost"), 0.0)
            rs = _safe_float(r.get("actual_rs"), 0.0)
            drafts = _safe_float(r.get("drafts"), 0.0)
            if boost > 0 and rs > 0:
                rows.append({
                    "date": r.get("date", ""),
                    "player_name": (r.get("player_name") or "").strip(),
                    "actual_rs": rs,
                    "actual_card_boost": boost,
                    "drafts": drafts,
                })
    return rows


def _load_actuals() -> list[dict]:
    """Load additional labeled examples from data/actuals/*.csv."""
    if not ACTUALS_DIR.exists():
        return []
    rows = []
    for csv_path in sorted(ACTUALS_DIR.glob("*.csv")):
        date_str = csv_path.stem
        with csv_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                boost = _safe_float(r.get("actual_card_boost"), 0.0)
                rs = _safe_float(r.get("actual_rs"), 0.0)
                drafts = _safe_float(r.get("drafts"), 0.0)
                name = (r.get("player_name") or "").strip()
                if boost > 0 and rs > 0 and name:
                    rows.append({
                        "date": date_str,
                        "player_name": name,
                        "actual_rs": rs,
                        "actual_card_boost": boost,
                        "drafts": drafts,
                    })
    return rows


def _load_most_popular() -> list[dict]:
    """Labeled rows from data/most_popular/{date}.csv (player, draft_count, boosts)."""
    if not MOST_POPULAR_DIR.is_dir():
        return []
    rows: list[dict] = []
    for csv_path in sorted(MOST_POPULAR_DIR.glob("*.csv")):
        date_str = csv_path.stem
        with csv_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                boost = _safe_float(r.get("actual_card_boost"), 0.0)
                rs = _safe_float(r.get("actual_rs"), 0.0)
                drafts = _safe_float(r.get("draft_count") or r.get("drafts"), 0.0)
                name = (r.get("player") or r.get("player_name") or "").strip()
                if boost > 0 and rs > 0 and name:
                    rows.append({
                        "date": date_str,
                        "player_name": name,
                        "actual_rs": rs,
                        "actual_card_boost": boost,
                        "drafts": drafts,
                    })
    return rows


def build_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, weights) from top_performers + actuals with 14-day lookback.

    Features:
      1. perf_score: 14-day trailing avg projected RS (or actual RS if predictions unavailable)
      2. min_proxy: 12 + 5*log1p(avg_drafts) where avg_drafts is 14-day trailing avg

    TRAIN/INFERENCE ALIGNMENT:
    Uses projected RS from data/predictions/ when available to match inference behavior.
    Falls back to actual RS for dates without predictions.
    """
    predictions_lookup = _load_predictions()
    all_rows = _load_top_performers() + _load_actuals() + _load_most_popular()
    seen: set[tuple[str, str]] = set()
    unique = []
    for r in all_rows:
        key = (r["player_name"], r["date"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    if not unique:
        return np.array([]), np.array([]), np.array([])

    print(f"Combined data: {len(unique)} unique (player, date) entries")
    pred_available = sum(1 for (_, d) in predictions_lookup.keys() if d)
    pred_used = 0
    print(f"Predictions available: {pred_available} entries in lookup")

    by_player: dict[str, list[dict]] = defaultdict(list)
    for r in unique:
        by_player[r["player_name"]].append(r)

    X_list, y_list, w_list = [], [], []
    for name, entries in by_player.items():
        entries.sort(key=lambda x: x["date"])
        for i, entry in enumerate(entries):
            try:
                dt = datetime.strptime(entry["date"], "%Y-%m-%d")
            except ValueError:
                continue

            prior_14d = [
                e for e in entries[:i]
                if (dt - datetime.strptime(e["date"], "%Y-%m-%d")).days <= LOOKBACK_DAYS
            ]
            prior_all = entries[:i]

            if prior_14d:
                rs_values = []
                for e in prior_14d:
                    # Use predicted RS if available, fall back to actual RS
                    key = (e["player_name"], e["date"])
                    if key in predictions_lookup:
                        rs_values.append(predictions_lookup[key])
                        pred_used += 1
                    else:
                        rs_values.append(e["actual_rs"])
                perf_score = float(np.mean(rs_values))
                avg_drafts = float(np.mean([e["drafts"] for e in prior_14d]))
                weight = 3.0
            elif prior_all:
                rs_values = []
                for e in prior_all:
                    # Use predicted RS if available, fall back to actual RS
                    key = (e["player_name"], e["date"])
                    if key in predictions_lookup:
                        rs_values.append(predictions_lookup[key])
                        pred_used += 1
                    else:
                        rs_values.append(e["actual_rs"])
                perf_score = float(np.mean(rs_values))
                avg_drafts = float(np.mean([e["drafts"] for e in prior_all]))
                weight = 2.0
            else:
                # Single observation: try to use prediction, fall back to actual
                key = (entry["player_name"], entry["date"])
                if key in predictions_lookup:
                    perf_score = predictions_lookup[key]
                    pred_used += 1
                else:
                    perf_score = entry["actual_rs"]
                avg_drafts = entry["drafts"]
                weight = 1.0

            # min_proxy estimates minutes from draft popularity
            # 1-20 drafts → ~17-22 min (low rotation)
            # 20-100 drafts → ~22-30 min (established role player)
            # 100+ drafts → ~30+ min (star/starter)
            min_proxy = 12.0 + 5.0 * np.log1p(avg_drafts)

            X_list.append([perf_score, min_proxy])
            y_list.append(entry["actual_card_boost"])
            w_list.append(weight)

    print(f"Used {pred_used} predicted RS values (vs actual RS fallback)")
    return np.array(X_list), np.array(y_list), np.array(w_list)


def train_model() -> None:
    print("Building training data from top_performers.csv + actuals...")
    X, y, weights = build_training_data()
    if len(X) == 0:
        print("[ERROR] No valid rows. Aborting.")
        return

    print(f"Training boost model on {len(X)} samples (features: {FEATURES})")
    print(f"  Target range: {y.min():.1f} - {y.max():.1f}")
    print(f"  Weighted: {int(sum(weights >= 3.0))} gold (14d), "
          f"{int(sum((weights >= 2.0) & (weights < 3.0)))} silver, "
          f"{int(sum(weights < 2.0))} bronze")

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.02,
        max_depth=4,
        num_leaves=15,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=1.0,
        objective="regression",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y, sample_weight=weights)

    bundle = {"model": model, "features": FEATURES}
    with MODEL_OUT.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {MODEL_OUT}")

    # Print learned RS × Minutes → Boost mapping
    print("\nLearned RS × Minutes → Boost mapping:")
    print("(Simulating different rotation levels at fixed RS values)")
    for rs in [3.0, 4.0, 5.0, 6.0]:
        print(f"\n  RS {rs:.1f}:")
        for drafts in [5, 25, 100, 410]:
            min_proxy = 12.0 + 5.0 * np.log1p(drafts)
            pred = model.predict([[rs, min_proxy]])[0]
            print(f"    {drafts:3d} drafts (min_proxy {min_proxy:.1f}) → boost {pred:.2f}")


if __name__ == "__main__":
    train_model()
