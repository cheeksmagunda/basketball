#!/usr/bin/env python3
"""Post-mortem: compare saved slate lineups to actual RS + quantify costly slots.

Reads local data/predictions/{date}.csv and data/actuals/{date}.csv (no GitHub).
Usage:
  python scripts/postmortem_slate.py 2026-03-22
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = ROOT / "data" / "predictions"
ACT_DIR = ROOT / "data" / "actuals"


def _slot_key(slot: str) -> float:
    s = str(slot).strip().lower().replace("x", "")
    try:
        return -float(s)
    except ValueError:
        return 0.0


def _sf(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_actuals(path: Path) -> dict[str, dict]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {r["player_name"].strip().lower(): r for r in rows if r.get("player_name")}


def lineup_rows(rows: list[dict], lineup_type: str) -> list[dict]:
    out = [r for r in rows if r.get("scope") == "slate" and r.get("lineup_type") == lineup_type]
    out.sort(key=lambda r: _slot_key(r.get("slot", "")))
    return out


def contribution(row: dict, act: dict | None, pred_boost_fallback: float) -> tuple[float, float, str]:
    """Returns (value, actual_rs, note)."""
    name = row.get("player_name", "")
    if act:
        ar = _sf(act.get("actual_rs"))
        ab = _sf(act.get("actual_card_boost"))
        note = "actuals"
    else:
        ar = 0.0
        ab = pred_boost_fallback
        note = "missing_actuals"
    sm = _sf(str(row.get("slot", "1.6x")).replace("x", ""))
    if ar <= 0 and note == "missing_actuals":
        return 0.0, 0.0, note
    val = ar * (sm + ab) if ar > 0 else 0.0
    return val, ar, note


def lineup_total(lineup: list[dict], act_map: dict[str, dict]) -> tuple[float, list[dict]]:
    details = []
    total = 0.0
    for row in lineup:
        key = row.get("player_name", "").strip().lower()
        act = act_map.get(key)
        pred_b = _sf(row.get("est_card_boost"))
        v, ar, note = contribution(row, act, pred_b)
        total += v
        details.append({
            "player": row.get("player_name"),
            "slot": row.get("slot"),
            "pred_rs": _sf(row.get("predicted_rs")),
            "actual_rs": ar,
            "value": round(v, 2),
            "data": note,
        })
    return total, details


def hindsight_optimal(players: list[dict], slot_mults: list[float]) -> tuple[float, list[dict]]:
    """Greedy: assign highest RS players to highest slots (same as audit helper)."""
    usable = [p for p in players if _sf(p.get("actual_rs")) > 0]
    usable.sort(key=lambda p: _sf(p.get("actual_rs")), reverse=True)
    top = usable[:5]
    out = []
    total = 0.0
    for i, p in enumerate(top):
        sm = slot_mults[i] if i < len(slot_mults) else 1.2
        ab = _sf(p.get("actual_card_boost"))
        ar = _sf(p.get("actual_rs"))
        v = ar * (sm + ab)
        total += v
        out.append({"player": p.get("player_name"), "slot": f"{sm}x", "actual_rs": ar, "value": round(v, 2)})
    return round(total, 2), out


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/postmortem_slate.py YYYY-MM-DD", file=sys.stderr)
        return 1
    date = sys.argv[1]
    pred_path = PRED_DIR / f"{date}.csv"
    act_path = ACT_DIR / f"{date}.csv"
    if not pred_path.is_file():
        print(f"Missing {pred_path}", file=sys.stderr)
        return 1
    if not act_path.is_file():
        print(f"Missing {act_path}", file=sys.stderr)
        return 1

    preds = load_predictions(pred_path)
    act_map = load_actuals(act_path)

    slot_mults = [2.0, 1.8, 1.6, 1.4, 1.2]
    pool = [
        {
            "player_name": r.get("player_name", ""),
            "actual_rs": r.get("actual_rs", ""),
            "actual_card_boost": r.get("actual_card_boost", ""),
        }
        for r in act_map.values()
    ]
    opt_total, opt_lineup = hindsight_optimal(pool, slot_mults)

    for label, lt in (("Starting 5 (chalk)", "chalk"), ("Moonshot (upside)", "upside")):
        lu = lineup_rows(preds, lt)
        if len(lu) != 5:
            print(f"\n{label}: expected 5 rows, got {len(lu)}")
            continue
        tot, det = lineup_total(lu, act_map)
        det.sort(key=lambda x: x["value"])
        print(f"\n=== {label} ===")
        print(f"Realized draft value (actual RS × slot + actual boost): {round(tot, 2)}")
        print("Lineup (lowest value first — replacement candidates):")
        for d in det:
            print(
                f"  {d['player']:<22} slot={d['slot']:<5} pred_rs={d['pred_rs']:.1f} "
                f"actual_rs={d['actual_rs']:.1f} value={d['value']:.2f} [{d['data']}]"
            )
        gap = round(opt_total - tot, 2)
        print(f"Hindsight ceiling (from actuals ∪ predicted names, greedy top-5 RS): {opt_total}  (gap +{gap})")

    print("\n=== Optimal greedy lineup (same pool) ===")
    for o in opt_lineup:
        print(f"  {o['player']:<22} {o['slot']:<5} actual_rs={o['actual_rs']:.1f} value={o['value']:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
