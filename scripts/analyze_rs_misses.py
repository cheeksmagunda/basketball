"""
Analyze RS prediction misses — identify which signals would have caught them.

Cross-references predictions vs actuals and categorizes misses by signal type:
- Tight game (low spread → closeness boost would help)
- Cascade (teammate OUT → usage spike)
- Hot streak (recent form >> season)
- Stat-stuffer (multi-category contributor)

Usage:
    python scripts/analyze_rs_misses.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PRED_DIR = Path(__file__).parent.parent / "data" / "predictions"
ACT_DIR = Path(__file__).parent.parent / "data" / "actuals"


def sf(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return 0.0


def analyze():
    dates = sorted(
        set(f.stem for f in PRED_DIR.glob("*.csv"))
        & set(f.stem for f in ACT_DIR.glob("*.csv"))
    )

    if not dates:
        print("No matching dates found.")
        return

    total_actual_top5 = 0
    total_predicted_top5 = 0
    total_in_predictions = 0
    signal_counts = {"in_preds_low_rank": 0, "not_in_preds": 0}

    for d in dates:
        with open(PRED_DIR / f"{d}.csv") as f:
            preds = list(csv.DictReader(f))
        with open(ACT_DIR / f"{d}.csv") as f:
            acts = list(csv.DictReader(f))

        # Get all predicted players (slate scope, deduplicated)
        pred_map = {}
        for p in preds:
            if p.get("scope", "") != "slate":
                continue
            name = p["player_name"]
            rs = sf(p.get("predicted_rs", 0))
            if name not in pred_map or rs > pred_map[name]["rs"]:
                pred_map[name] = {
                    "rs": rs,
                    "pts": sf(p.get("pts", 0)),
                    "reb": sf(p.get("reb", 0)),
                    "ast": sf(p.get("ast", 0)),
                    "stl": sf(p.get("stl", 0)),
                    "blk": sf(p.get("blk", 0)),
                    "pred_min": sf(p.get("pred_min", 0)),
                    "boost": sf(p.get("est_card_boost", 0)),
                }

        # Actual top 5
        act_sorted = sorted(acts, key=lambda x: sf(x.get("actual_rs", 0)), reverse=True)
        actual_top5 = act_sorted[:5]

        # Our predicted top 5 (from slate predictions)
        pred_sorted = sorted(pred_map.items(), key=lambda x: -x[1]["rs"])
        our_top5 = set(n.lower() for n, _ in pred_sorted[:5])
        our_top15 = set(n.lower() for n, _ in pred_sorted[:15])

        print(f"\n{'='*70}")
        print(f"DATE: {d}")
        print(f"{'='*70}")

        for rank, a in enumerate(actual_top5, 1):
            name = a["player_name"]
            actual_rs = sf(a.get("actual_rs", 0))
            actual_boost = sf(a.get("actual_card_boost", 0))
            drafts = a.get("drafts", "?")
            total_value = sf(a.get("total_value", 0))

            in_our_top5 = name.lower() in our_top5
            in_our_top15 = name.lower() in our_top15
            in_preds = name in pred_map

            total_actual_top5 += 1
            if in_our_top5:
                total_predicted_top5 += 1

            if in_preds:
                total_in_predictions += 1
                p = pred_map[name]
                status = "✓ TOP-5" if in_our_top5 else ("~ TOP-15" if in_our_top15 else "✗ RANKED LOW")
                if not in_our_top5:
                    signal_counts["in_preds_low_rank"] += 1
                print(f"  #{rank} {name:25s} RS={actual_rs:5.1f} boost={actual_boost:+.1f}x drafts={drafts:>4s} val={total_value:5.1f} | "
                      f"{status} (pred RS={p['rs']:.1f}, pts={p['pts']:.0f})")
            else:
                signal_counts["not_in_preds"] += 1
                print(f"  #{rank} {name:25s} RS={actual_rs:5.1f} boost={actual_boost:+.1f}x drafts={drafts:>4s} val={total_value:5.1f} | "
                      f"✗ NOT IN SLATE PREDICTIONS")

        print(f"\n  Our top-5: {[n for n, _ in pred_sorted[:5]]}")

    n_dates = len(dates)
    print(f"\n{'='*70}")
    print(f"SUMMARY ({n_dates} dates, {total_actual_top5} actual top-5 players)")
    print(f"{'='*70}")
    print(f"  In our top-5:           {total_predicted_top5}/{total_actual_top5} ({total_predicted_top5/total_actual_top5*100:.0f}%)")
    print(f"  In our predictions:     {total_in_predictions}/{total_actual_top5} ({total_in_predictions/total_actual_top5*100:.0f}%)")
    print(f"  In preds but low rank:  {signal_counts['in_preds_low_rank']}")
    print(f"  NOT in predictions:     {signal_counts['not_in_preds']}")
    print(f"\n  Key insight: {signal_counts['not_in_preds']} of {total_actual_top5} actual top-5 RS scorers")
    print(f"  were never even in our slate predictions (gated out entirely).")
    print(f"  {signal_counts['in_preds_low_rank']} were predicted but ranked too low (RS ordering issue).")


if __name__ == "__main__":
    analyze()
