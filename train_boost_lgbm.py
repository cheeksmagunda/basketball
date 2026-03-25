#!/usr/bin/env python3
"""
Train LightGBM card boost model from top_performers.csv.

Uses projected RS as the primary performance signal. At inference, this is
blended with a PPG-based sigmoid in _est_card_boost() to capture both
game-level RS dynamics and stable player-tier recognition.

Feature: perf_score (14-day trailing avg RS from top_performers.csv)
Target:  actual_card_boost (0.0 to 3.0)

At inference: perf_score = projected_rs from the RS projection model.
The sigmoid blend uses season_pts (PPG) from ESPN for the recognition signal.

Writes boost_model.pkl as {"model": ..., "features": [...]}.
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
MODEL_OUT = Path("boost_model.pkl")

FEATURES = ["perf_score"]
LOOKBACK_DAYS = 14


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


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
            if boost > 0 and rs > 0:
                rows.append({
                    "date": r.get("date", ""),
                    "player_name": (r.get("player_name") or "").strip(),
                    "actual_rs": rs,
                    "actual_card_boost": boost,
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
                name = (r.get("player_name") or "").strip()
                if boost > 0 and rs > 0 and name:
                    rows.append({
                        "date": date_str,
                        "player_name": name,
                        "actual_rs": rs,
                        "actual_card_boost": boost,
                    })
    return rows


def build_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, weights) from top_performers + actuals with 14-day lookback."""
    all_rows = _load_top_performers() + _load_actuals()
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
                perf_score = float(np.mean([e["actual_rs"] for e in prior_14d]))
                weight = 3.0
            elif prior_all:
                perf_score = float(np.mean([e["actual_rs"] for e in prior_all]))
                weight = 2.0
            else:
                perf_score = entry["actual_rs"]
                weight = 1.0

            X_list.append([perf_score])
            y_list.append(entry["actual_card_boost"])
            w_list.append(weight)

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

    # Print learned RS → Boost mapping
    print("\nLearned RS → Boost mapping:")
    for rs in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 8.0]:
        pred = model.predict([[rs]])[0]
        print(f"  RS {rs:.1f} → boost {pred:.2f}")


if __name__ == "__main__":
    train_model()
