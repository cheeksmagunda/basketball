"""
Calibrate DFS weights by regressing actual RS against predicted stat lines.

Uses stored predictions (pts/reb/ast/stl/blk) matched against actual RS from data/top_performers.csv
(per date) with fallback to data/actuals/{date}.csv when the mega file has no rows for that date.
Outputs optimal DFS weights that minimize RS prediction error.

Usage:
    python scripts/calibrate_dfs_weights.py
"""
import csv
import numpy as np
from pathlib import Path

PRED_DIR = Path(__file__).parent.parent / "data" / "predictions"
ACT_DIR = Path(__file__).parent.parent / "data" / "actuals"
TOP_PERFORMERS = Path(__file__).parent.parent / "data" / "top_performers.csv"


def sf(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _actual_rs_map_for_date(date_str: str) -> dict:
    """player_name.lower() -> actual_rs for one slate date."""
    act_map = {}
    if TOP_PERFORMERS.is_file():
        with open(TOP_PERFORMERS, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (row.get("date") or "").strip() != date_str:
                    continue
                name = (row.get("player_name") or "").strip()
                if not name:
                    continue
                act_map[name.lower()] = sf(row.get("actual_rs", 0))
    if act_map:
        return act_map
    per_day = ACT_DIR / f"{date_str}.csv"
    if not per_day.is_file():
        return {}
    with open(per_day, encoding="utf-8") as f:
        acts = list(csv.DictReader(f))
    for a in acts:
        nm = (a.get("player_name") or "").strip().lower()
        if nm:
            act_map[nm] = sf(a.get("actual_rs", 0))
    return act_map


def load_matched_data():
    """Load matched prediction stats + actual RS for all prediction dates that have labels."""
    pred_dates = sorted(f.stem for f in PRED_DIR.glob("*.csv"))
    dates = [d for d in pred_dates if _actual_rs_map_for_date(d)]

    rows = []
    for d in dates:
        with open(PRED_DIR / f"{d}.csv") as f:
            preds = list(csv.DictReader(f))

        act_map = _actual_rs_map_for_date(d)

        # Deduplicate: keep highest predicted RS per player per date
        seen = {}
        for p in preds:
            name = p["player_name"]
            key = name.lower()
            pts = sf(p.get("pts", 0))
            reb = sf(p.get("reb", 0))
            ast = sf(p.get("ast", 0))
            stl = sf(p.get("stl", 0))
            blk = sf(p.get("blk", 0))
            pred_rs = sf(p.get("predicted_rs", 0))

            if key in act_map and pts > 0 and act_map[key] > 0:
                if key not in seen or pred_rs > seen[key]["pred_rs"]:
                    seen[key] = {
                        "date": d,
                        "name": name,
                        "pts": pts,
                        "reb": reb,
                        "ast": ast,
                        "stl": stl,
                        "blk": blk,
                        "pred_rs": pred_rs,
                        "actual_rs": act_map[key],
                    }

        rows.extend(seen.values())

    return rows


def calibrate():
    """Run linear regression to find optimal DFS weights."""
    data = load_matched_data()
    if len(data) < 10:
        print(f"Only {len(data)} matched samples — need more data.")
        return

    print(f"Loaded {len(data)} matched player-date samples.\n")

    # Build feature matrix: [pts, reb, ast, stl, blk]
    X = np.array([[r["pts"], r["reb"], r["ast"], r["stl"], r["blk"]] for r in data])
    y = np.array([r["actual_rs"] for r in data])

    # Linear regression: actual_rs = a*pts + b*reb + c*ast + d*stl + e*blk + intercept
    # Add intercept column
    X_int = np.column_stack([X, np.ones(len(X))])
    # Least squares
    coeffs, residuals, rank, sv = np.linalg.lstsq(X_int, y, rcond=None)

    pts_w, reb_w, ast_w, stl_w, blk_w, intercept = coeffs

    print("=" * 60)
    print("OPTIMAL DFS WEIGHTS (linear regression)")
    print("=" * 60)
    print(f"  pts: {pts_w:.4f}")
    print(f"  reb: {reb_w:.4f}")
    print(f"  ast: {ast_w:.4f}")
    print(f"  stl: {stl_w:.4f}")
    print(f"  blk: {blk_w:.4f}")
    print(f"  intercept: {intercept:.4f}")

    # Predict using the optimal weights
    y_pred_new = X_int @ coeffs
    mae_new = np.mean(np.abs(y - y_pred_new))

    # Compare with current DFS weights
    current_weights = {"pts": 2.5, "reb": 0.5, "ast": 1.0, "stl": 2.0, "blk": 1.5}
    dfs_current = np.array([
        r["pts"] * current_weights["pts"] +
        r["reb"] * current_weights["reb"] +
        r["ast"] * current_weights["ast"] +
        r["stl"] * current_weights["stl"] +
        r["blk"] * current_weights["blk"]
        for r in data
    ])
    # Apply same compression as the pipeline
    comp_div, comp_pow = 4.5, 0.78
    rs_current = np.power(np.maximum(dfs_current / comp_div, 0.01), comp_pow)
    mae_current = np.mean(np.abs(y - rs_current))

    print(f"\n  Current DFS→RS MAE: {mae_current:.3f}")
    print(f"  Optimal linear MAE: {mae_new:.3f}")
    print(f"  Improvement: {(mae_current - mae_new)/mae_current*100:.1f}%")

    # Now compute what DFS weights would look like scaled to the compression pipeline
    # We want: (w_pts*pts + w_reb*reb + ... ) / comp_div ) ^ comp_pow ≈ actual_rs
    # Inverse: actual_rs^(1/comp_pow) * comp_div ≈ w_pts*pts + ...
    y_uncompressed = np.power(np.maximum(y, 0.01), 1.0/comp_pow) * comp_div
    X_uc = np.column_stack([X, np.ones(len(X))])
    coeffs_uc, _, _, _ = np.linalg.lstsq(X_uc, y_uncompressed, rcond=None)
    pts_uc, reb_uc, ast_uc, stl_uc, blk_uc, int_uc = coeffs_uc

    print(f"\n{'='*60}")
    print("DFS WEIGHTS (compression-aware, for model-config.json)")
    print(f"{'='*60}")
    print(f"  pts: {pts_uc:.3f}")
    print(f"  reb: {reb_uc:.3f}")
    print(f"  ast: {ast_uc:.3f}")
    print(f"  stl: {stl_uc:.3f}")
    print(f"  blk: {blk_uc:.3f}")
    print(f"  (intercept offset: {int_uc:.3f})")

    # Verify with compression pipeline
    dfs_new = X @ coeffs_uc[:5] + int_uc
    rs_new = np.power(np.maximum(dfs_new / comp_div, 0.01), comp_pow)
    mae_calibrated = np.mean(np.abs(y - rs_new))
    print(f"\n  Calibrated DFS→RS MAE: {mae_calibrated:.3f}")

    # Ranking analysis: how well do these weights rank players?
    print(f"\n{'='*60}")
    print("RANKING ANALYSIS (per-date)")
    print(f"{'='*60}")

    dates = sorted(set(r["date"] for r in data))
    for d in dates:
        day_data = [r for r in data if r["date"] == d]
        if len(day_data) < 5:
            continue

        # Sort by new predicted RS
        for r in day_data:
            r["new_pred_rs"] = (max((r["pts"]*pts_uc + r["reb"]*reb_uc +
                                     r["ast"]*ast_uc + r["stl"]*stl_uc +
                                     r["blk"]*blk_uc + int_uc) / comp_div, 0.01)
                                ) ** comp_pow

        by_new = sorted(day_data, key=lambda x: -x["new_pred_rs"])
        by_actual = sorted(day_data, key=lambda x: -x["actual_rs"])

        new_top5 = set(r["name"].lower() for r in by_new[:5])
        actual_top5 = set(r["name"].lower() for r in by_actual[:5])
        hits = len(new_top5 & actual_top5)
        print(f"  {d}: {hits}/5 top-5 overlap | "
              f"predicted: {[r['name'] for r in by_new[:3]]} | "
              f"actual: {[r['name'] for r in by_actual[:3]]}")

    return {
        "compression_aware_weights": {
            "pts": round(pts_uc, 3),
            "reb": round(reb_uc, 3),
            "ast": round(ast_uc, 3),
            "stl": round(stl_uc, 3),
            "blk": round(blk_uc, 3),
        },
        "intercept": round(int_uc, 3),
        "mae_current": round(mae_current, 3),
        "mae_calibrated": round(mae_calibrated, 3),
    }


if __name__ == "__main__":
    calibrate()
