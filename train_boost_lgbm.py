#!/usr/bin/env python3
"""
Train LightGBM card boost model from historical actuals.

This script builds an 8-feature training set from data/actuals/*.csv
and pre-game NBA logs (date_to = target_date - 1 day), then writes
boost_model.pkl as {"model": ..., "features": [...]}.
"""

import csv
import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from nba_api.stats.endpoints import playergamelogs
from nba_api.stats.static import players

ACTUALS_DIR = Path("data/actuals")
MODEL_OUT = Path("boost_model.pkl")
BIG_MARKETS = {"LAL", "GSW", "BOS", "NYK", "PHI", "MIA", "DEN", "LAC", "CHI"}

FEATURES = [
    "season_pts",
    "recent_pts",
    "season_min",
    "recent_min",
    "pts_ratio",
    "is_big_market",
    "role_change_min",
    "is_home",
]


def _normalize_name(name: str) -> str:
    s = (name or "").strip().lower()
    for tok in (" jr.", " sr.", " ii", " iii", " iv", ".", "'"):
        s = s.replace(tok, "")
    return " ".join(s.split())


def _player_index() -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for p in players.get_players():
        full = (p.get("full_name") or "").strip()
        pid = p.get("id")
        if full and pid:
            idx[_normalize_name(full)] = int(pid)
    return idx


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _guess_season(target_date: datetime) -> str:
    # NBA season rolls in October. Jan-Sep belong to prior year start.
    y = target_date.year if target_date.month >= 10 else target_date.year - 1
    return f"{y}-{str((y + 1) % 100).zfill(2)}"


def _extract_market_and_home(matchup: str) -> tuple[float, float]:
    text = (matchup or "").strip()
    if not text:
        return 0.0, 0.0
    team_abbr = text.split(" ")[0].strip().upper()
    is_big_market = 1.0 if team_abbr in BIG_MARKETS else 0.0
    is_home = 0.0 if "@" in text else 1.0
    return is_big_market, is_home


def _fetch_logs_before_date(player_id: int, target_date: datetime) -> Optional[pd.DataFrame]:
    season_str = _guess_season(target_date)
    date_to = (target_date - timedelta(days=1)).strftime("%m/%d/%Y")
    logs = playergamelogs.PlayerGameLogs(
        player_id_nullable=player_id,
        season_nullable=season_str,
        date_to_nullable=date_to,
        timeout=45,
    ).get_data_frames()[0]
    if logs is None or logs.empty:
        return None
    return logs


def build_training_data() -> pd.DataFrame:
    if not ACTUALS_DIR.exists():
        print(f"[ERROR] Missing directory: {ACTUALS_DIR}")
        return pd.DataFrame()

    pidx = _player_index()
    rows = []

    for csv_path in sorted(ACTUALS_DIR.glob("*.csv")):
        date_str = csv_path.stem
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            continue

        with csv_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                player_name = (row.get("player_name") or "").strip()
                if not player_name:
                    continue
                actual_boost = _safe_float(row.get("actual_card_boost"), 0.0)
                if actual_boost <= 0:
                    continue

                pid = pidx.get(_normalize_name(player_name))
                if not pid:
                    continue

                try:
                    logs = _fetch_logs_before_date(pid, target_date)
                    # Conservative client-side rate-limiting.
                    time.sleep(0.6)
                except Exception as e:
                    print(f"[WARN] log fetch failed for {player_name} ({date_str}): {e}")
                    time.sleep(1.0)
                    continue

                if logs is None or len(logs) < 3:
                    continue

                season_pts = _safe_float(logs["PTS"].mean(), 0.0)
                season_min = _safe_float(logs["MIN"].mean(), 0.0)
                recent_logs = logs.head(5)
                recent_pts = _safe_float(recent_logs["PTS"].mean(), 0.0)
                recent_min = _safe_float(recent_logs["MIN"].mean(), 0.0)
                pts_ratio = recent_pts / max(season_pts, 1.0)
                role_change_min = recent_min - season_min

                latest = logs.iloc[0]
                is_big_market, is_home = _extract_market_and_home(str(latest.get("MATCHUP", "")))

                rows.append(
                    {
                        "player_name": player_name,
                        "season_pts": season_pts,
                        "recent_pts": recent_pts,
                        "season_min": season_min,
                        "recent_min": recent_min,
                        "pts_ratio": pts_ratio,
                        "is_big_market": is_big_market,
                        "role_change_min": role_change_min,
                        "is_home": is_home,
                        "actual_card_boost": actual_boost,
                    }
                )

    return pd.DataFrame(rows)


def train_model() -> None:
    print("Building training data from actuals...")
    df = build_training_data()
    if df.empty:
        print("[ERROR] No valid rows. Aborting.")
        return

    X = df[FEATURES]
    y = df["actual_card_boost"]
    weights = np.where(y >= 2.0, 3.0, np.where(y >= 1.0, 1.5, 1.0))

    print(f"Training boost model on {len(df)} samples...")
    model = lgb.LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=31,
        objective="regression",
        random_state=42,
    )
    model.fit(X, y, sample_weight=weights)

    bundle = {"model": model, "features": FEATURES}
    with MODEL_OUT.open("wb") as f:
        pickle.dump(bundle, f)
    print(f"Saved {MODEL_OUT}")


if __name__ == "__main__":
    train_model()
