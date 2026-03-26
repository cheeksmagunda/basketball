#!/usr/bin/env python3
"""
Backtest drafts + card boost on leaderboard-style labels where we have a matching
slate row in data/predictions/{date}.csv.

Labels use the same union as train_drafts_lgbm:
  - data/top_performers.csv (multi-date file)
  - data/actuals/{date}.csv
  - data/most_popular/{date}.csv (draft_count → drafts)

Usage (repo root):
  python scripts/verify_top_performers.py
  python scripts/verify_top_performers.py --strict --min-rows 15 --min-spearman 0.12

Exit codes: 0 = ok (or inconclusive with small n); 2 = --strict thresholds not met.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3 or b.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a.astype(float)))
    rb = np.argsort(np.argsort(b.astype(float)))
    c = np.corrcoef(ra.astype(float), rb.astype(float))[0, 1]
    return float(c) if np.isfinite(c) else float("nan")


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def main() -> int:
    import train_drafts_lgbm as td

    ap = argparse.ArgumentParser(description="Verify drafts/boost vs top_performers + predictions")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 if thresholds fail (for CI when overlap is large enough)",
    )
    ap.add_argument("--min-rows", type=int, default=5, help="Minimum rows for strict metrics")
    ap.add_argument(
        "--min-spearman-log-drafts",
        type=float,
        default=0.08,
        help="Minimum Spearman on log1p(drafts) vs predicted log1p (strict mode)",
    )
    ap.add_argument(
        "--max-mae-boost",
        type=float,
        default=0.55,
        help="Maximum mean |error| on card boost vs actual (strict mode)",
    )
    args = ap.parse_args()

    tp_path = REPO / "data" / "top_performers.csv"
    pred_index = td._load_prediction_index()
    role_agg = td._player_role_aggregates(pred_index)

    tp_dates: set[str] = set()
    if tp_path.exists():
        with tp_path.open("r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                d = (r.get("date") or "").strip()
                if d:
                    tp_dates.add(d)
    actuals_dir = REPO / "data" / "actuals"
    actuals_dates = (
        {p.stem for p in actuals_dir.glob("*.csv")} if actuals_dir.is_dir() else set()
    )

    labeled = td._labeled_rows()
    label_dates = {r["date"] for r in labeled if (r.get("date") or "").strip()}
    pred_dates = set(pred_index.keys())
    overlap_dates = label_dates & pred_dates

    rows: list[dict] = []
    for row in labeled:
        date = (row.get("date") or "").strip()
        name = (row.get("player_name") or "").strip()
        drafts = float(row.get("drafts") or 0.0)
        if drafts <= 0 or not date or not name:
            continue
        nm = td._normalize_name(name)
        game = pred_index.get(date, {}).get(nm)
        role = role_agg.get(nm)
        if not game or not role:
            continue
        boost_act = float(row.get("actual_boost", -1.0))
        rows.append(
            {
                "date": date,
                "name": name,
                "drafts": drafts,
                "actual_boost": boost_act,
                "role_pts": role["role_pts"],
                "role_avg_min": role["role_avg_min"],
                "game": game,
            }
        )

    n = len(rows)
    print(
        f"[verify] top_performers.csv dates: {len(tp_dates)} | "
        f"data/actuals/*.csv dates: {len(actuals_dates)} | "
        f"labeled dates (drafts>0, deduped): {len(label_dates)}"
    )
    print(f"[verify] prediction CSV dates: {len(pred_dates)} | overlap w/ labeled: {len(overlap_dates)}")
    print(f"[verify] joined rows (label + slate projection): {n}")
    if overlap_dates and n < 50:
        print(
            "[verify] note: predictions CSVs usually only include scope=slate lineup picks "
            "(~10 players/day), not the full slate — most leaderboard names won't match a row."
        )

    model_path = REPO / "drafts_model.pkl"
    if not model_path.exists():
        print("[verify] drafts_model.pkl missing — skip drafts metrics")
        draft_mae = float("nan")
        sp = float("nan")
        y_log = np.array([])
        pred_log = np.array([])
    elif n == 0:
        print(
            "[verify] No joined rows — need data/predictions/{{date}}.csv for dates that have "
            "labels in top_performers.csv and/or data/actuals/{{date}}.csv."
        )
        draft_mae = float("nan")
        sp = float("nan")
        y_log = np.array([])
        pred_log = np.array([])
    else:
        import pickle

        with model_path.open("rb") as f:
            bundle = pickle.load(f)
        model = bundle["model"]
        feats = bundle.get("features") or []
        if list(feats) != list(td.FEATURES):
            print(f"[verify] WARN: model features {feats!r} != train_drafts_lgbm.FEATURES {td.FEATURES!r}")

        X = np.array(
            [
                [
                    row["role_pts"],
                    row["role_avg_min"],
                    row["game"]["big_market"],
                    row["game"]["pos_bucket"],
                ]
                for row in rows
            ],
            dtype=float,
        )
        y_log = np.array([np.log1p(row["drafts"]) for row in rows], dtype=float)
        pred_log = model.predict(X).astype(float)
        draft_mae = _mae(pred_log, y_log)
        sp = _spearman(pred_log, y_log)
        print(f"[verify] drafts (log1p): MAE={draft_mae:.4f} | Spearman(pred, actual)={sp:.4f}")

    # Boost vs actual (production path)
    boost_errs: list[float] = []
    if n > 0:
        import api.index as idx

        idx._ensure_boost_model_loaded()
        if idx.BOOST_MODEL is None:
            print("[verify] boost_model.pkl not loaded — skip boost MAE")
        for row in rows:
            g = row["game"]
            rs = float(g.get("predicted_rs") or 0.0)
            if rs <= 0:
                continue
            est = idx._est_card_boost(
                g["pred_min"],
                g["pts"],
                str(g.get("team") or "").strip().upper(),
                player_name=row["name"],
                season_pts=row["role_pts"],
                recent_pts=g["pts"],
                cascade_bonus=0.0,
                is_home=False,
                projected_rs=rs,
                season_avg_min=row["role_avg_min"],
                player_pos=str(g.get("pos") or ""),
            )
            act = row["actual_boost"]
            if act >= 0 and est is not None:
                boost_errs.append(float(est) - float(act))
        if boost_errs:
            boost_mae = float(np.mean(np.abs(np.array(boost_errs))))
            print(f"[verify] card boost: MAE vs actual_card_boost={boost_mae:.3f} (n={len(boost_errs)})")
        else:
            boost_mae = float("nan")
            print("[verify] card boost: no comparable rows (need predicted_rs > 0 and actual boost)")
    else:
        boost_mae = float("nan")

    if not args.strict:
        return 0

    ok = True
    if n < args.min_rows:
        print(
            f"[verify] strict: only {n} rows (need {args.min_rows}) — "
            "not failing; add overlapping predictions first."
        )
        return 0

    if model_path.exists() and n > 0 and np.isfinite(sp) and sp < args.min_spearman_log_drafts:
        print(
            f"[verify] strict FAIL: Spearman log-drafts {sp:.4f} < {args.min_spearman_log_drafts}"
        )
        ok = False
    if np.isfinite(boost_mae) and boost_mae > args.max_mae_boost:
        print(f"[verify] strict FAIL: boost MAE {boost_mae:.3f} > {args.max_mae_boost}")
        ok = False

    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
