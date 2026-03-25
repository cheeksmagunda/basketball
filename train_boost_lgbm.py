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

FEATURES = ["perf_score", "min_proxy"]
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


def build_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, weights) from top_performers + actuals with 14-day lookback.

    Features:
      1. perf_score: 14-day trailing avg RS
      2. min_proxy: 12 + 5*log1p(avg_drafts) where avg_drafts is 14-day trailing avg
    """
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
                avg_drafts = float(np.mean([e["drafts"] for e in prior_14d]))
                weight = 3.0
            elif prior_all:
                perf_score = float(np.mean([e["actual_rs"] for e in prior_all]))
                avg_drafts = float(np.mean([e["drafts"] for e in prior_all]))
                weight = 2.0
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
