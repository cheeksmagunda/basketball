"""
Offline backtest: measure how well predicted RS rankings match actual top RS scorers.

Usage:
    python scripts/backtest_rs.py                    # baseline with current predictions
    python scripts/backtest_rs.py --recalc           # recalculate RS from stored stats using current config
    python scripts/backtest_rs.py --weights pts=3.0 reb=0.8 ast=1.5  # test DFS weight overrides

Metrics:
    - Top-7 recall: of our top-15 predicted RS players, how many hit the actual top-7?
    - Top-10 recall: of our top-15, how many hit the actual top-10?
    - MAE: mean absolute error for matched players
"""
import csv
import sys
import os
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

PRED_DIR = Path(__file__).parent.parent / "data" / "predictions"
ACT_DIR = Path(__file__).parent.parent / "data" / "actuals"


def sf(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def _dfs_score(pts, reb, ast, stl, blk, tov, weights=None):
    """DFS score with configurable weights."""
    w = weights or {"pts": 2.5, "reb": 0.5, "ast": 1.0, "stl": 2.0, "blk": 1.5, "tov": -1.5}
    return (pts * w.get("pts", 2.5) + reb * w.get("reb", 0.5) +
            ast * w.get("ast", 1.0) + stl * w.get("stl", 2.0) +
            blk * w.get("blk", 1.5) + tov * w.get("tov", -1.5))


def _recalc_rs(row, weights=None, comp_div=4.5, comp_pow=0.78, rs_cap=20.0,
               stat_stuffer=None, spread=None):
    """Recalculate RS from stored ESPN stats using given parameters."""
    pts = sf(row.get("pts", 0))
    reb = sf(row.get("reb", 0))
    ast = sf(row.get("ast", 0))
    stl = sf(row.get("stl", 0))
    blk = sf(row.get("blk", 0))
    tov = 0  # not stored in predictions CSV

    dfs = _dfs_score(pts, reb, ast, stl, blk, tov, weights)
    if dfs <= 0:
        return 0.0

    raw_linear = dfs / comp_div
    rs = raw_linear ** comp_pow
    rs = min(rs, rs_cap)

    # Stat-stuffer bonus
    if stat_stuffer and stat_stuffer.get("enabled", False):
        cats = 0
        if pts >= stat_stuffer.get("pts_threshold", 12):
            cats += 1
        if reb >= stat_stuffer.get("reb_threshold", 6):
            cats += 1
        if ast >= stat_stuffer.get("ast_threshold", 4):
            cats += 1
        if stl >= stat_stuffer.get("stl_threshold", 1.5):
            cats += 1
        if blk >= stat_stuffer.get("blk_threshold", 1.0):
            cats += 1

        if cats >= 4:  # near triple-double territory
            rs *= stat_stuffer.get("bonus_td", 1.20)
        elif cats >= 3:
            rs *= stat_stuffer.get("bonus_3cat", 1.10)

    return round(min(rs, rs_cap), 2)


def run_backtest(recalc=False, weights=None, comp_div=4.5, comp_pow=0.78,
                 stat_stuffer=None, verbose=True):
    """Run backtest across all dates with both predictions and actuals."""
    dates = sorted(
        set(f.stem for f in PRED_DIR.glob("*.csv"))
        & set(f.stem for f in ACT_DIR.glob("*.csv"))
    )

    if not dates:
        print("No matching dates found.")
        return

    total_recall_top5 = []
    total_recall_top7 = []
    total_recall_top10 = []
    total_mae = []

    for d in dates:
        with open(PRED_DIR / f"{d}.csv") as f:
            preds = list(csv.DictReader(f))
        with open(ACT_DIR / f"{d}.csv") as f:
            acts = list(csv.DictReader(f))

        # Get all unique players with their best predicted RS
        all_pred = {}
        for p in preds:
            name = p["player_name"]
            if recalc:
                rs = _recalc_rs(p, weights=weights, comp_div=comp_div,
                                comp_pow=comp_pow, stat_stuffer=stat_stuffer)
            else:
                rs = sf(p.get("predicted_rs", 0))
            if name not in all_pred or rs > all_pred[name]:
                all_pred[name] = rs

        # Sort by predicted RS
        pred_sorted = sorted(all_pred.items(), key=lambda x: -x[1])
        our_top10 = set(n.lower() for n, _ in pred_sorted[:10])
        our_top15 = set(n.lower() for n, _ in pred_sorted[:15])

        # Actual top RS
        act_sorted = sorted(acts, key=lambda x: sf(x.get("actual_rs", 0)), reverse=True)
        actual_top5 = set(a["player_name"].lower() for a in act_sorted[:5])
        actual_top7 = set(a["player_name"].lower() for a in act_sorted[:7])
        actual_top10 = set(a["player_name"].lower() for a in act_sorted[:10])

        hits5 = len(our_top15 & actual_top5)
        hits7 = len(our_top15 & actual_top7)
        hits10 = len(our_top15 & actual_top10)

        total_recall_top5.append(hits5)
        total_recall_top7.append(hits7)
        total_recall_top10.append(hits10)

        # MAE for matched players
        act_map = {a["player_name"].lower(): sf(a.get("actual_rs", 0)) for a in acts}
        errs = []
        for name, pred_rs in all_pred.items():
            if name.lower() in act_map and pred_rs > 0:
                errs.append(abs(pred_rs - act_map[name.lower()]))
        if errs:
            total_mae.append(sum(errs) / len(errs))

        if verbose:
            our_top5_names = [n for n, _ in pred_sorted[:5]]
            act_top5_names = [a["player_name"] for a in act_sorted[:5]]
            print(f"{d}: top15→top7={hits7}/7 | top15→top10={hits10}/10 | "
                  f"MAE={sum(errs)/len(errs):.2f}" if errs else f"{d}: no matches")
            if verbose and hits7 < 2:
                print(f"  OURS: {our_top5_names}")
                print(f"  REAL: {act_top5_names}")

    n = len(dates)
    print(f"\n{'='*60}")
    print(f"BACKTEST SUMMARY ({n} dates)")
    print(f"{'='*60}")
    print(f"Avg top-5 recall (our top-15 → actual top-5):  {sum(total_recall_top5)/n:.1f}/5 ({sum(total_recall_top5)/n/5*100:.0f}%)")
    print(f"Avg top-7 recall (our top-15 → actual top-7):  {sum(total_recall_top7)/n:.1f}/7 ({sum(total_recall_top7)/n/7*100:.0f}%)")
    print(f"Avg top-10 recall (our top-15 → actual top-10): {sum(total_recall_top10)/n:.1f}/10 ({sum(total_recall_top10)/n/10*100:.0f}%)")
    if total_mae:
        print(f"Avg MAE (matched players): {sum(total_mae)/len(total_mae):.2f}")

    return {
        "dates": n,
        "avg_top5_recall": sum(total_recall_top5) / n,
        "avg_top7_recall": sum(total_recall_top7) / n,
        "avg_top10_recall": sum(total_recall_top10) / n,
        "avg_mae": sum(total_mae) / len(total_mae) if total_mae else None,
    }


if __name__ == "__main__":
    recalc = "--recalc" in sys.argv
    weights = None

    # Parse --weights flag
    if "--weights" in sys.argv:
        idx = sys.argv.index("--weights")
        weights = {}
        for arg in sys.argv[idx + 1:]:
            if "=" in arg:
                k, v = arg.split("=", 1)
                weights[k] = float(v)

    stat_stuffer = None
    if "--stat-stuffer" in sys.argv:
        stat_stuffer = {
            "enabled": True,
            "pts_threshold": 12, "reb_threshold": 6, "ast_threshold": 4,
            "stl_threshold": 1.5, "blk_threshold": 1.0,
            "bonus_3cat": 1.10, "bonus_td": 1.20,
        }

    print("=" * 60)
    if recalc:
        w_str = str(weights) if weights else "default"
        print(f"RECALC mode — DFS weights: {w_str}")
        if stat_stuffer:
            print(f"  Stat-stuffer: ENABLED")
    else:
        print("STORED PREDICTIONS mode (no recalculation)")
    print("=" * 60)
    print()

    run_backtest(recalc=recalc, weights=weights, stat_stuffer=stat_stuffer)
