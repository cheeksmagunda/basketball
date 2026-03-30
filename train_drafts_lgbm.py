#!/usr/bin/env python3
"""
Train LightGBM to estimate draft counts (popularity) for Real Sports players.

Uses top_performers.csv + actuals (drafts column) joined to data/predictions/{date}.csv.
Features emphasize **stable** signals — role scoring, rotation minutes, market, position —
not a single hot night's RS (Cam Spencer / narrative + GSW persists across games).

Target: log1p(actual drafts). At inference, min_proxy = 12 + 5 * model_output
would be wrong — the boost pipeline uses min_proxy = 12 + 5*log1p(drafts).
So the model predicts **z = log1p(drafts)**; runtime converts to drafts via expm1 and
min_proxy = 12 + 5 * z  (since z == log1p(drafts) → min_proxy formula matches).

Actually: if model predicts `log1p(drafts)` directly, then `min_proxy = 12 + 5 * prediction`
because min_proxy = 12 + 5*log1p(drafts). Yes.

Writes drafts_model.json + drafts_model.txt (native LightGBM; portable across sklearn versions).
"""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

# Import shared helpers from api.features — single source of truth
from api.features import pos_bucket as _pos_bucket

REPO = Path(__file__).resolve().parent
TOP_PERFORMERS = REPO / "data" / "top_performers.csv"
ACTUALS_DIR = REPO / "data" / "actuals"
MOST_POPULAR_DIR = REPO / "data" / "most_popular"
PRED_DIR = REPO / "data" / "predictions"
MODEL_CFG = REPO / "data" / "model-config.json"
MODEL_JSON = REPO / "drafts_model.json"
MODEL_TXT = REPO / "drafts_model.txt"

FEATURES = ["role_pts", "role_avg_min", "big_market", "pos_bucket"]


def _load_big_market_teams() -> set[str]:
    try:
        cfg = json.loads(MODEL_CFG.read_text(encoding="utf-8"))
        teams = cfg.get("card_boost", {}).get("big_market_teams", [])
        if teams:
            return {str(t).upper() for t in teams}
    except Exception:
        pass
    return {
        "LAL", "GS", "GSW", "BOS", "NY", "NYK", "PHI", "MIA", "LAC", "CHI",
    }


def _normalize_name(name: str) -> str:
    n = unicodedata.normalize("NFKD", name or "").encode("ASCII", "ignore").decode("ASCII")
    return n.strip().lower()


def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_prediction_index() -> dict[str, dict[str, dict]]:
    """date_str -> normalized_name -> first slate row (chalk/upside share same proj)."""
    big_m = _load_big_market_teams()
    out: dict[str, dict[str, dict]] = defaultdict(dict)
    if not PRED_DIR.exists():
        return out
    for csv_path in sorted(PRED_DIR.glob("*.csv")):
        d = csv_path.stem
        with csv_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if (r.get("scope") or "") != "slate":
                    continue
                nm = _normalize_name(r.get("player_name", ""))
                if not nm or nm in out[d]:
                    continue
                team = (r.get("team") or "").strip().upper()
                out[d][nm] = {
                    "pts": _safe_float(r.get("pts")),
                    "pred_min": _safe_float(r.get("pred_min")),
                    "predicted_rs": _safe_float(r.get("predicted_rs")),
                    "team": team,
                    "big_market": 1.0 if team in big_m else 0.0,
                    "pos_bucket": float(_pos_bucket(r.get("pos", ""))),
                    "pos": (r.get("pos") or "").strip(),
                }
    return dict(out)


def _player_role_aggregates(pred_index: dict[str, dict[str, dict]]) -> dict[str, dict[str, float]]:
    """Per player: mean pts and pred_min across all slates (stable role proxies)."""
    pts_sum: dict[str, float] = defaultdict(float)
    pm_sum: dict[str, float] = defaultdict(float)
    ct: dict[str, int] = defaultdict(int)
    for game_map in pred_index.values():
        for nm, row in game_map.items():
            pts_sum[nm] += row["pts"]
            pm_sum[nm] += row["pred_min"]
            ct[nm] += 1
    out = {}
    for nm in pts_sum:
        n = max(1, ct[nm])
        out[nm] = {"role_pts": pts_sum[nm] / n, "role_avg_min": pm_sum[nm] / n}
    return out


def _labeled_rows() -> list[dict]:
    rows: list[dict] = []
    if TOP_PERFORMERS.exists():
        with TOP_PERFORMERS.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                drafts = _safe_float(r.get("drafts"), 0.0)
                if drafts <= 0:
                    continue
                rows.append(
                    {
                        "date": r.get("date", ""),
                        "player_name": (r.get("player_name") or "").strip(),
                        "drafts": drafts,
                        "actual_boost": _safe_float(r.get("actual_card_boost"), -1.0),
                    }
                )
    if ACTUALS_DIR.exists():
        for csv_path in sorted(ACTUALS_DIR.glob("*.csv")):
            d = csv_path.stem
            with csv_path.open("r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    drafts = _safe_float(r.get("drafts"), 0.0)
                    name = (r.get("player_name") or "").strip()
                    if drafts <= 0 or not name:
                        continue
                    rows.append(
                        {
                            "date": d,
                            "player_name": name,
                            "drafts": drafts,
                            "actual_boost": _safe_float(r.get("actual_card_boost"), -1.0),
                        }
                    )
    if MOST_POPULAR_DIR.is_dir():
        for csv_path in sorted(MOST_POPULAR_DIR.glob("*.csv")):
            d = csv_path.stem
            with csv_path.open("r", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    drafts = _safe_float(r.get("draft_count") or r.get("drafts"), 0.0)
                    name = (r.get("player") or r.get("player_name") or "").strip()
                    if drafts <= 0 or not name:
                        continue
                    rows.append(
                        {
                            "date": d,
                            "player_name": name,
                            "drafts": drafts,
                            "actual_boost": _safe_float(r.get("actual_card_boost"), -1.0),
                        }
                    )
    # De-dupe (player, date)
    seen: set[tuple[str, str]] = set()
    uniq = []
    for row in rows:
        k = (_normalize_name(row["player_name"]), row["date"])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(row)
    return uniq


def build_training_matrix(
    pred_index: dict[str, dict[str, dict]],
    role_agg: dict[str, dict[str, float]],
    labeled: list[dict] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, w = [], [], []
    if labeled is None:
        labeled = _labeled_rows()
    for row in labeled:
        d, name = row["date"], row["player_name"]
        nm = _normalize_name(name)
        game = pred_index.get(d, {}).get(nm)
        if not game:
            continue
        role = role_agg.get(nm)
        if not role:
            continue
        drafts = row["drafts"]
        y.append(np.log1p(drafts))
        X.append(
            [
                role["role_pts"],
                role["role_avg_min"],
                game["big_market"],
                game["pos_bucket"],
            ]
        )
        # Up-weight rows with extreme draft counts (signal for stars vs deep bench)
        w.append(1.0 + min(2.0, np.log1p(drafts) / 5.0))
    return np.array(X), np.array(y), np.array(w)


def train() -> None:
    pred_index = _load_prediction_index()
    if not pred_index:
        print("[drafts_model] No prediction CSVs — cannot train.")
        return
    tp_dates: set[str] = set()
    if TOP_PERFORMERS.exists():
        with TOP_PERFORMERS.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = (r.get("date") or "").strip()
                if d:
                    tp_dates.add(d)
    actuals_dates = set()
    if ACTUALS_DIR.exists():
        for p in ACTUALS_DIR.glob("*.csv"):
            actuals_dates.add(p.stem)
    if MOST_POPULAR_DIR.is_dir():
        for p in MOST_POPULAR_DIR.glob("*.csv"):
            actuals_dates.add(p.stem)
    labeled = _labeled_rows()
    label_dates = {r["date"] for r in labeled if (r.get("date") or "").strip()}
    pred_dates = set(pred_index.keys())
    ovl = label_dates & pred_dates
    print(
        f"[drafts_model] top_performers file dates: {len(tp_dates)} | "
        f"actuals CSV dates: {len(actuals_dates)} | "
        f"labeled dates (drafts>0, deduped): {len(label_dates)} | "
        f"prediction CSV dates: {len(pred_dates)} | overlap: {len(ovl)}"
    )
    role_agg = _player_role_aggregates(pred_index)
    X, y, weights = build_training_matrix(pred_index, role_agg, labeled=labeled)
    if len(X) < 10:
        print(
            f"[drafts_model] Only {len(X)} joined rows — need top_performers/actuals dates "
            "that have a matching data/predictions/{{date}}.csv (min 10)."
        )
        return

    model = lgb.LGBMRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=20,
        min_child_samples=8,
        subsample=0.85,
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
    print(f"[drafts_model] Saved {MODEL_TXT.name} + {MODEL_JSON.name} ({len(X)} samples)")
    # Sanity check: high minute + big market → higher log1p(drafts)
    for label, vec in [
        ("bench_small", [7.0, 14.0, 0.0, 1.0]),
        ("rotation_gsw", [12.0, 28.0, 1.0, 0.0]),
        ("star", [24.0, 34.0, 1.0, 0.0]),
    ]:
        pred = float(model.predict([vec])[0])
        est_d = np.expm1(pred)  # if pred is log1p(drafts)
        print(f"  {label}: log1p(drafts)≈{pred:.2f} → drafts≈{est_d:.0f}")


if __name__ == "__main__":
    train()
