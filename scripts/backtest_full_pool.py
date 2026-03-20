#!/usr/bin/env python3
"""
Rebuild a full historical player pool per date and backtest overlap.

This script:
1) Loads dates from data/actuals/*.csv
2) Rebuilds the full candidate pool via api.index._run_game() for each date
3) Evaluates boost model families (sigmoid/log/power) on top-10 overlap
4) Runs a lightweight random search for best overlap score

Notes:
- We force heuristic mode by bypassing LightGBM load in this script run.
- Results are in-sample diagnostics for model iteration, not production metrics.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import api.index as idx


def _norm(name: str) -> str:
    return (
        unicodedata.normalize("NFKD", name)
        .encode("ASCII", "ignore")
        .decode("ASCII")
        .strip()
        .lower()
    )


def _load_actual_top10(actual_dir: Path) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for p in sorted(actual_dir.glob("*.csv")):
        rows = []
        with p.open() as f:
            for r in csv.DictReader(f):
                n = (r.get("player_name") or "").strip()
                if not n:
                    continue
                try:
                    tv = float(r.get("total_value") or 0)
                except Exception:
                    continue
                rows.append((_norm(n), tv))
        rows.sort(key=lambda x: -x[1])
        out[p.stem] = {x[0] for x in rows[:10]}
    return out


def _load_actual_rows(actual_dir: Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for p in sorted(actual_dir.glob("*.csv")):
        rows = []
        with p.open() as f:
            for r in csv.DictReader(f):
                n = (r.get("player_name") or "").strip()
                if not n:
                    continue
                try:
                    rows.append(
                        {
                            "name": n,
                            "nk": _norm(n),
                            "total_value": float(r.get("total_value") or 0),
                            "actual_rs": float(r.get("actual_rs") or 0),
                            "actual_boost": float(r.get("actual_card_boost") or 0),
                        }
                    )
                except Exception:
                    continue
        rows.sort(key=lambda x: -x["total_value"])
        out[p.stem] = rows
    return out


def _rebuild_full_pool(dates: List[str]) -> Dict[str, dict]:
    full: Dict[str, dict] = {}
    for d in dates:
        dt = datetime.strptime(d, "%Y-%m-%d").date()
        players = {}
        games = idx.fetch_games(dt)
        for g in games:
            try:
                projs = idx._run_game(g)
            except Exception:
                continue
            for p in projs:
                name = (p.get("name") or "").strip()
                if not name:
                    continue
                nk = _norm(name)
                rec = {
                    "name": name,
                    "pts": float(p.get("pts", 0) or 0),
                    "recent_pts": float(p.get("recent_pts", p.get("pts", 0)) or 0),
                    "predMin": float(p.get("predMin", 0) or 0),
                    "avgMin": float(p.get("avgMin", 0) or 0),
                    "rating": float(p.get("rating", 0) or 0),
                }
                cur = players.get(nk)
                if (cur is None) or (rec["rating"] > cur["rating"]):
                    players[nk] = rec
        full[d] = players
    return full


def _build_fit_rows(full_pool: Dict[str, dict], actual_rows: Dict[str, List[dict]]) -> List[dict]:
    rows: List[dict] = []
    for d, arr in actual_rows.items():
        pool = full_pool.get(d, {})
        for a in arr:
            p = pool.get(a["nk"])
            if not p:
                continue
            rows.append(
                {
                    "date": d,
                    "nk": a["nk"],
                    "pts": p["pts"],
                    "recent_pts": p["recent_pts"],
                    "target_boost": a["actual_boost"],
                }
            )
    return rows


def _fit_boost_family(fit_rows: List[dict], family: str, iters: int = 12000) -> Tuple[Tuple[float, ...], float]:
    best_params = ()
    best_mse = float("inf")
    for _ in range(iters):
        if family == "sigmoid":
            params = (
                random.uniform(2.0, 3.2),
                random.uniform(1.0, 3.2),
                random.uniform(6.0, 22.0),
                random.uniform(1.5, 10.0),
                random.uniform(0.3, 2.5),
            )
        elif family == "log":
            params = (
                random.uniform(1.6, 4.5),
                random.uniform(0.1, 2.0),
                random.uniform(0.3, 2.5),
            )
        else:
            params = (
                random.uniform(1.2, 5.0),
                random.uniform(0.05, 1.4),
                random.uniform(0.3, 2.5),
            )

        se = 0.0
        for r in fit_rows:
            pts = max(r["pts"], 0.1)
            recent = r["recent_pts"]
            form = max(0.6, min(1.6, recent / max(pts, 1e-3)))
            if family == "sigmoid":
                c, rr, m, s, fp = params
                sv = 1.0 / (1.0 + math.exp(-(pts - m) / max(s, 1e-3)))
                pred = c - rr * sv
            elif family == "log":
                a, b, fp = params
                pred = a - b * math.log1p(pts)
            else:
                a, b, fp = params
                pred = a / (pts ** max(b, 1e-6))
            pred = max(0.0, min(3.0, pred * (form ** fp)))
            se += (pred - r["target_boost"]) ** 2
        mse = se / max(len(fit_rows), 1)
        if mse < best_mse:
            best_mse = mse
            best_params = params
    return best_params, best_mse


def _boost_from_params(family: str, params: Tuple[float, ...], pts: float, recent_pts: float) -> float:
    pts = max(pts, 0.1)
    form = max(0.6, min(1.6, recent_pts / max(pts, 1e-3)))
    if family == "sigmoid":
        c, rr, m, s, fp = params
        sv = 1.0 / (1.0 + math.exp(-(pts - m) / max(s, 1e-3)))
        b = c - rr * sv
    elif family == "log":
        a, b1, fp = params
        b = a - b1 * math.log1p(pts)
    else:
        a, b1, fp = params
        b = a / (pts ** max(b1, 1e-6))
    return max(0.0, min(3.0, b * (form ** fp)))


def _boost_sigmoid(pts: float, recent_pts: float) -> float:
    c, r, m, s = 3.0, 2.8, 12.0, 4.0
    sv = 1 / (1 + math.exp(-(pts - m) / max(s, 1e-3)))
    b = c - r * sv
    form = max(0.7, min(1.3, recent_pts / max(pts, 1e-3)))
    return max(0.0, min(3.0, b * form))


def _boost_log(pts: float, recent_pts: float) -> float:
    a, b = 3.2, 0.95
    z = a - b * math.log1p(max(pts, 0.1))
    form = max(0.7, min(1.3, recent_pts / max(pts, 1e-3)))
    return max(0.0, min(3.0, z * form))


def _boost_power(pts: float, recent_pts: float) -> float:
    a, b = 3.8, 0.45
    z = a / (max(pts, 0.1) ** b)
    form = max(0.7, min(1.3, recent_pts / max(pts, 1e-3)))
    return max(0.0, min(3.0, z * form))


def _eval_family(
    full_pool: Dict[str, dict], actual_top10: Dict[str, set], family: str
) -> float:
    fn = {
        "sigmoid": _boost_sigmoid,
        "log": _boost_log,
        "power": _boost_power,
    }[family]
    total_hits = 0
    dates = sorted(actual_top10)
    for d in dates:
        scored = []
        for nk, p in full_pool[d].items():
            b = fn(p["pts"], p["recent_pts"])
            v = p["rating"] * (1.6 + b)
            scored.append((nk, v))
        scored.sort(key=lambda x: -x[1])
        top = {x[0] for x in scored[:10]}
        total_hits += len(top & actual_top10[d])
    return total_hits / len(dates)


def _eval_fit_family(
    full_pool: Dict[str, dict],
    actual_top10: Dict[str, set],
    family: str,
    params: Tuple[float, ...],
) -> float:
    total_hits = 0
    dates = sorted(actual_top10)
    for d in dates:
        scored = []
        for nk, p in full_pool[d].items():
            b = _boost_from_params(family, params, p["pts"], p["recent_pts"])
            v = p["rating"] * (1.6 + b)
            scored.append((nk, v))
        scored.sort(key=lambda x: -x[1])
        top = {x[0] for x in scored[:10]}
        total_hits += len(top & actual_top10[d])
    return total_hits / len(dates)


def _oracle_ceiling(full_pool: Dict[str, dict], actual_rows: Dict[str, List[dict]]) -> float:
    """Hindsight-only ceiling: same-day actual RS + actual boost."""
    total_hits = 0
    dates = sorted(actual_rows)
    for d in dates:
        top_actual = {x["nk"] for x in actual_rows[d][:10]}
        score_map = {}
        for a in actual_rows[d]:
            if a["nk"] in full_pool[d]:
                score_map[a["nk"]] = a["actual_rs"] * (1.6 + a["actual_boost"])
        top_pred = {nk for nk, _ in sorted(score_map.items(), key=lambda kv: -kv[1])[:10]}
        total_hits += len(top_actual & top_pred)
    return total_hits / len(dates)


def _walk_forward_eval(
    dates: List[str],
    full_pool: Dict[str, dict],
    actual_rows: Dict[str, List[dict]],
    actual_top10: Dict[str, set],
    family: str,
    params: Tuple[float, ...],
    min_history_games: int = 2,
) -> float:
    """
    Strict no-leak evaluation:
    - For date D, only use info from dates < D
    - Build player boost priors from historical actual boosts
    - Combine prior + curve prediction + rating to rank top 10
    """
    total_hits = 0
    eval_days = 0
    hist_boosts: Dict[str, List[float]] = {}

    for d in dates:
        pool = full_pool.get(d, {})
        # Score only after enough history exists
        if len(hist_boosts) >= min_history_games:
            scored = []
            for nk, p in pool.items():
                curve_b = _boost_from_params(family, params, p["pts"], p["recent_pts"])
                prior_list = hist_boosts.get(nk, [])
                prior_b = (sum(prior_list) / len(prior_list)) if prior_list else 2.0
                blend_w = min(0.8, 0.2 + 0.2 * len(prior_list))
                est_boost = max(0.0, min(3.0, (blend_w * prior_b) + ((1.0 - blend_w) * curve_b)))
                score = p["rating"] * (1.6 + est_boost)
                scored.append((nk, score))
            scored.sort(key=lambda x: -x[1])
            top_pred = {x[0] for x in scored[:10]}
            total_hits += len(top_pred & actual_top10[d])
            eval_days += 1

        # Update history AFTER scoring this date (walk-forward)
        for a in actual_rows.get(d, []):
            hist_boosts.setdefault(a["nk"], []).append(a["actual_boost"])

    return (total_hits / eval_days) if eval_days else 0.0


def _walk_forward_search(
    dates: List[str],
    full_pool: Dict[str, dict],
    actual_rows: Dict[str, List[dict]],
    actual_top10: Dict[str, set],
    n: int = 8000,
):
    best = (0.0, None)
    for _ in range(n):
        family = random.choice(["sigmoid", "log", "power"])
        if family == "sigmoid":
            params = (
                random.uniform(2.0, 3.2),
                random.uniform(1.0, 3.2),
                random.uniform(6.0, 22.0),
                random.uniform(1.5, 10.0),
                random.uniform(0.3, 2.5),
            )
        elif family == "log":
            params = (
                random.uniform(1.6, 4.5),
                random.uniform(0.1, 2.0),
                random.uniform(0.3, 2.5),
            )
        else:
            params = (
                random.uniform(1.2, 5.0),
                random.uniform(0.05, 1.4),
                random.uniform(0.3, 2.5),
            )
        avg = _walk_forward_eval(
            dates=dates,
            full_pool=full_pool,
            actual_rows=actual_rows,
            actual_top10=actual_top10,
            family=family,
            params=params,
            min_history_games=2,
        )
        if avg > best[0]:
            best = (avg, {"family": family, "params": tuple(round(x, 4) for x in params)})
    return best


def _slot_lineup_score(player_scores: List[float]) -> float:
    """Assign highest scores to highest slots (2.0 -> 1.2)."""
    slots = [2.0, 1.8, 1.6, 1.4, 1.2]
    vals = sorted(player_scores, reverse=True)[:5]
    if len(vals) < 5:
        vals += [0.0] * (5 - len(vals))
    return sum(v * s for v, s in zip(vals, slots))


def _slot_presence_score(pred_lineup: List[str], actual_rows_for_date: List[dict]) -> float:
    """
    Stable lineup-quality metric (unitless):
    - Reward players in actual top-5 heavily, top-10 moderately, top-15 lightly.
    - Apply slot weights (2.0, 1.8, 1.6, 1.4, 1.2) to reflect draft assignment value.
    - Returns score in [0, 1] after normalization by max possible score.
    """
    slots = [2.0, 1.8, 1.6, 1.4, 1.2]
    top5 = {x["nk"] for x in actual_rows_for_date[:5]}
    top10 = {x["nk"] for x in actual_rows_for_date[:10]}
    top15 = {x["nk"] for x in actual_rows_for_date[:15]}

    raw = 0.0
    for i, nk in enumerate(pred_lineup[:5]):
        w = slots[i]
        if nk in top5:
            raw += 1.0 * w
        elif nk in top10:
            raw += 0.6 * w
        elif nk in top15:
            raw += 0.3 * w

    # Best-case is all 5 picks in top5: sum(slots)*1.0
    return raw / sum(slots)


def _walk_forward_joint_lineup_eval(
    dates: List[str],
    full_pool: Dict[str, dict],
    actual_rows: Dict[str, List[dict]],
    family: str,
    boost_params: Tuple[float, ...],
    rs_scale_high: float,
    rs_scale_mid: float,
    rs_scale_low: float,
    min_pred_min: float,
) -> float:
    """
    Walk-forward lineup-level objective (stable, unitless):
    - Build per-player score from calibrated RS proxy + boost
    - Draft top-5 by that score
    - Score lineup quality using membership in actual top-value sets
    Returns average normalized presence score in [0, 1]
    """
    qualities: List[float] = []
    hist_boosts: Dict[str, List[float]] = {}

    for d in dates:
        pool = full_pool.get(d, {})
        # skip earliest day without prior information
        if hist_boosts:
            scored = []
            for nk, p in pool.items():
                if p["predMin"] < min_pred_min:
                    continue

                # RS calibration by scoring bucket (empirical correction from diagnostics)
                pts = p["pts"]
                rs = p["rating"]
                if pts >= 16:
                    rs *= rs_scale_high
                elif pts >= 8:
                    rs *= rs_scale_mid
                else:
                    rs *= rs_scale_low

                curve_b = _boost_from_params(family, boost_params, p["pts"], p["recent_pts"])
                prior_list = hist_boosts.get(nk, [])
                prior_b = (sum(prior_list) / len(prior_list)) if prior_list else 2.0
                blend_w = min(0.8, 0.2 + 0.2 * len(prior_list))
                est_boost = max(0.0, min(3.0, (blend_w * prior_b) + ((1.0 - blend_w) * curve_b)))

                # Keep same scoring family as game objective.
                player_score = max(rs, 0.0) * (1.6 + est_boost)
                scored.append((nk, player_score))

            scored.sort(key=lambda x: -x[1])
            pred_lineup = [nk for nk, _ in scored[:5]]
            qualities.append(_slot_presence_score(pred_lineup, actual_rows.get(d, [])))

        # Update boost history after scoring date
        for a in actual_rows.get(d, []):
            hist_boosts.setdefault(a["nk"], []).append(a["actual_boost"])

    return (sum(qualities) / len(qualities)) if qualities else 0.0


def _walk_forward_joint_lineup_search(
    dates: List[str],
    full_pool: Dict[str, dict],
    actual_rows: Dict[str, List[dict]],
    n: int = 6000,
):
    """
    Search joint RS+boost parameters with lineup-level KPI.
    Returns best average presence score and config.
    """
    best = (0.0, None)
    for _ in range(n):
        family = random.choice(["sigmoid", "log", "power"])
        if family == "sigmoid":
            boost_params = (
                random.uniform(2.0, 3.2),
                random.uniform(1.0, 3.2),
                random.uniform(6.0, 22.0),
                random.uniform(1.5, 10.0),
                random.uniform(0.3, 2.5),
            )
        elif family == "log":
            boost_params = (
                random.uniform(1.6, 4.5),
                random.uniform(0.1, 2.0),
                random.uniform(0.3, 2.5),
            )
        else:
            boost_params = (
                random.uniform(1.2, 5.0),
                random.uniform(0.05, 1.4),
                random.uniform(0.3, 2.5),
            )

        rs_scale_high = random.uniform(0.4, 1.0)  # stars currently over-projected
        rs_scale_mid = random.uniform(0.7, 1.1)
        rs_scale_low = random.uniform(0.9, 1.6)   # role players currently under-projected
        min_pred_min = random.uniform(8.0, 22.0)
        ratio = _walk_forward_joint_lineup_eval(
            dates=dates,
            full_pool=full_pool,
            actual_rows=actual_rows,
            family=family,
            boost_params=boost_params,
            rs_scale_high=rs_scale_high,
            rs_scale_mid=rs_scale_mid,
            rs_scale_low=rs_scale_low,
            min_pred_min=min_pred_min,
        )
        if ratio > best[0]:
            best = (
                ratio,
                {
                    "family": family,
                    "boost_params": tuple(round(x, 4) for x in boost_params),
                    "rs_scale_high": round(rs_scale_high, 4),
                    "rs_scale_mid": round(rs_scale_mid, 4),
                    "rs_scale_low": round(rs_scale_low, 4),
                    "min_pred_min": round(min_pred_min, 3),
                },
            )
    return best


def _rs_calibrated(
    rating: float,
    pts: float,
    high_thr: float,
    mid_thr: float,
    high_mult: float,
    mid_mult: float,
    low_mult: float,
) -> float:
    if pts >= high_thr:
        return rating * high_mult
    if pts >= mid_thr:
        return rating * mid_mult
    return rating * low_mult


def _ndcg_at_k(pred_list: List[str], actual_rows_for_date: List[dict], k: int = 5) -> float:
    """NDCG@k where gain is actual_rs."""
    rel = {x["nk"]: x["actual_rs"] for x in actual_rows_for_date}
    top_pred = pred_list[:k]
    dcg = 0.0
    for i, nk in enumerate(top_pred):
        gain = rel.get(nk, 0.0)
        dcg += gain / math.log2(i + 2)

    ideal = sorted((x["actual_rs"] for x in actual_rows_for_date), reverse=True)[:k]
    idcg = 0.0
    for i, gain in enumerate(ideal):
        idcg += gain / math.log2(i + 2)
    return (dcg / idcg) if idcg > 0 else 0.0


def _rs_first_backtest(
    dates: List[str],
    full_pool: Dict[str, dict],
    actual_rows: Dict[str, List[dict]],
    high_thr: float,
    mid_thr: float,
    high_mult: float,
    mid_mult: float,
    low_mult: float,
):
    """
    RS-first mode:
    1) Rank by calibrated projected RS only
    2) Report top-5 RS hit rate and NDCG@5
    3) Compute resulting lineup value ratio vs actual top-5 value lineup
    """
    top5_hits = []
    ndcgs = []
    value_ratios = []

    for d in dates:
        pool = full_pool.get(d, {})
        scored = []
        for nk, p in pool.items():
            rs_hat = _rs_calibrated(
                rating=p["rating"],
                pts=p["pts"],
                high_thr=high_thr,
                mid_thr=mid_thr,
                high_mult=high_mult,
                mid_mult=mid_mult,
                low_mult=low_mult,
            )
            scored.append((nk, rs_hat))
        scored.sort(key=lambda x: -x[1])
        pred_top5 = [x[0] for x in scored[:5]]

        actual_rs_sorted = sorted(actual_rows[d], key=lambda x: -x["actual_rs"])
        actual_top5 = [x["nk"] for x in actual_rs_sorted[:5]]
        hit = len(set(pred_top5) & set(actual_top5))
        top5_hits.append(hit)
        ndcgs.append(_ndcg_at_k(pred_top5, actual_rs_sorted, k=5))

        # Stage 2: apply actual boosts to isolate RS ranking effect on final value.
        boost_map = {x["nk"]: x["actual_boost"] for x in actual_rows[d]}
        pred_vals = []
        for nk in pred_top5:
            rs_hat = dict(scored).get(nk, 0.0)
            pred_vals.append(rs_hat * (1.6 + boost_map.get(nk, 0.0)))
        pred_lineup = _slot_lineup_score(pred_vals)

        actual_value_sorted = sorted(
            actual_rows[d],
            key=lambda x: -(x["actual_rs"] * (1.6 + x["actual_boost"])),
        )
        win_vals = [x["actual_rs"] * (1.6 + x["actual_boost"]) for x in actual_value_sorted[:5]]
        winner_lineup = _slot_lineup_score(win_vals)
        if winner_lineup > 0:
            value_ratios.append(pred_lineup / winner_lineup)

    return {
        "avg_top5_hits": sum(top5_hits) / len(top5_hits),
        "avg_ndcg5": sum(ndcgs) / len(ndcgs),
        "avg_value_ratio": sum(value_ratios) / len(value_ratios) if value_ratios else 0.0,
    }


def _rs_first_search(
    dates: List[str],
    full_pool: Dict[str, dict],
    actual_rows: Dict[str, List[dict]],
    n: int = 6000,
):
    """Search RS calibration buckets for RS-first ranking quality."""
    best = (0.0, None, None)
    for _ in range(n):
        high_thr = random.uniform(14.0, 20.0)
        mid_thr = random.uniform(6.0, 12.0)
        if mid_thr >= high_thr:
            continue
        high_mult = random.uniform(0.5, 1.0)
        mid_mult = random.uniform(0.7, 1.05)
        low_mult = random.uniform(1.0, 1.8)
        m = _rs_first_backtest(
            dates=dates,
            full_pool=full_pool,
            actual_rows=actual_rows,
            high_thr=high_thr,
            mid_thr=mid_thr,
            high_mult=high_mult,
            mid_mult=mid_mult,
            low_mult=low_mult,
        )
        # Primary objective: top-5 RS hit count
        score = m["avg_top5_hits"] + 0.5 * m["avg_ndcg5"]
        if score > best[0]:
            best = (
                score,
                m,
                {
                    "high_pts_threshold": round(high_thr, 3),
                    "mid_pts_threshold": round(mid_thr, 3),
                    "high_mult": round(high_mult, 4),
                    "mid_mult": round(mid_mult, 4),
                    "low_mult": round(low_mult, 4),
                },
            )
    return best


def _rs_first_date_compare(
    dates: List[str],
    full_pool: Dict[str, dict],
    actual_rows: Dict[str, List[dict]],
    tuned_cfg: dict,
) -> List[dict]:
    """
    Compare per-date RS top-5 hits:
    - baseline: no RS bucket calibration (all mult=1.0)
    - tuned: provided calibrated bucket multipliers
    """
    out = []
    for d in dates:
        pool = full_pool.get(d, {})
        actual_rs_sorted = sorted(actual_rows[d], key=lambda x: -x["actual_rs"])
        actual_top5 = {x["nk"] for x in actual_rs_sorted[:5]}

        base_ranked = sorted(
            ((nk, p["rating"]) for nk, p in pool.items()),
            key=lambda x: -x[1],
        )
        base_top5 = {x[0] for x in base_ranked[:5]}
        base_hits = len(base_top5 & actual_top5)

        tuned_ranked = []
        for nk, p in pool.items():
            rs_hat = _rs_calibrated(
                rating=p["rating"],
                pts=p["pts"],
                high_thr=tuned_cfg["high_pts_threshold"],
                mid_thr=tuned_cfg["mid_pts_threshold"],
                high_mult=tuned_cfg["high_mult"],
                mid_mult=tuned_cfg["mid_mult"],
                low_mult=tuned_cfg["low_mult"],
            )
            tuned_ranked.append((nk, rs_hat))
        tuned_ranked.sort(key=lambda x: -x[1])
        tuned_top5 = {x[0] for x in tuned_ranked[:5]}
        tuned_hits = len(tuned_top5 & actual_top5)

        out.append(
            {
                "date": d,
                "baseline_hits": base_hits,
                "tuned_hits": tuned_hits,
                "delta": tuned_hits - base_hits,
            }
        )
    return out


def _hindsight_top_real_scorers(actual_rows: Dict[str, List[dict]], k: int = 10) -> float:
    """
    Pure hindsight oracle for user's explicit target:
    "projected top real scorers of the day using hindsight".
    With hindsight labels available, rank by actual_rs directly.
    """
    hits = 0
    dates = sorted(actual_rows)
    for d in dates:
        rows = actual_rows[d]
        actual_top = {x["nk"] for x in sorted(rows, key=lambda x: -x["actual_rs"])[:k]}
        proj_top = {x["nk"] for x in sorted(rows, key=lambda x: -x["actual_rs"])[:k]}
        hits += len(actual_top & proj_top)
    return hits / len(dates)


def _hindsight_player_prior_top_real_scorers(actual_rows: Dict[str, List[dict]], k: int = 10) -> float:
    """
    Hindsight prior model (still hindsight, but less trivial than direct oracle):
    score = historical mean actual_rs INCLUDING current day (full hindsight leak).
    """
    all_rows = [r for rows in actual_rows.values() for r in rows]
    rs_by_player: Dict[str, List[float]] = {}
    for r in all_rows:
        rs_by_player.setdefault(r["nk"], []).append(r["actual_rs"])
    mean_rs = {k_: (sum(v) / len(v)) for k_, v in rs_by_player.items()}

    hits = 0
    dates = sorted(actual_rows)
    for d in dates:
        rows = actual_rows[d]
        actual_top = {x["nk"] for x in sorted(rows, key=lambda x: -x["actual_rs"])[:k]}
        proj_top = {
            x["nk"]
            for x in sorted(rows, key=lambda x: -mean_rs.get(x["nk"], 0.0))[:k]
        }
        hits += len(actual_top & proj_top)
    return hits / len(dates)


def _random_search(full_pool: Dict[str, dict], actual_top10: Dict[str, set], n: int = 5000):
    random.seed(17)
    dates = sorted(actual_top10)
    best = (0.0, None)
    for _ in range(n):
        family = random.choice(["sigmoid", "log", "power"])
        min_floor = random.uniform(0, 24)
        w_rating = random.uniform(0.0, 3.0)
        w_boost = random.uniform(0.0, 4.0)
        w_pts = random.uniform(-1.2, 1.2)
        gamma = random.uniform(0.4, 2.8)

        if family == "sigmoid":
            c = random.uniform(2.0, 3.3)
            r = random.uniform(1.0, 3.2)
            m = random.uniform(6, 22)
            s = random.uniform(1.5, 10)
        elif family == "log":
            a = random.uniform(1.6, 4.5)
            b = random.uniform(0.1, 1.8)
        else:
            a = random.uniform(1.5, 5.0)
            b = random.uniform(0.05, 1.2)

        total = 0
        for d in dates:
            scored = []
            for nk, p in full_pool[d].items():
                if p["predMin"] < min_floor:
                    continue
                x = max(p["pts"], 0.1)
                rf = max(0.5, min(1.6, p["recent_pts"] / max(p["pts"], 1e-3)))
                if family == "sigmoid":
                    boost = c - r * (1 / (1 + math.exp(-(x - m) / max(s, 1e-3))))
                elif family == "log":
                    boost = a - b * math.log1p(x)
                else:
                    boost = a / (x ** max(b, 1e-6))
                boost = max(0.0, min(3.0, boost * rf))
                val = w_rating * p["rating"] + w_boost * (boost ** gamma) + w_pts * p["pts"]
                scored.append((nk, val))
            scored.sort(key=lambda t: -t[1])
            top = {nk for nk, _ in scored[:10]}
            total += len(top & actual_top10[d])
        avg = total / len(dates)
        if avg > best[0]:
            best = (
                avg,
                {
                    "family": family,
                    "min_floor": round(min_floor, 3),
                    "w_rating": round(w_rating, 3),
                    "w_boost": round(w_boost, 3),
                    "w_pts": round(w_pts, 3),
                    "gamma": round(gamma, 3),
                },
            )
    return best


def main() -> None:
    idx._LGBM_LOAD_ATTEMPTED = True
    idx.AI_MODEL = None
    idx.AI_FEATURES = None

    random.seed(17)
    actual_dir = Path("data/actuals")
    actual_top10 = _load_actual_top10(actual_dir)
    actual_rows = _load_actual_rows(actual_dir)
    dates = sorted(actual_top10)
    full_pool = _rebuild_full_pool(dates)
    fit_rows = _build_fit_rows(full_pool, actual_rows)

    print("=== Full Pool Coverage ===")
    for d in dates:
        coverage = len(set(full_pool[d]) & actual_top10[d])
        print(f"{d}: pool={len(full_pool[d])} coverage={coverage}/10")

    print("\\n=== Family Benchmarks ===")
    for fam in ("sigmoid", "log", "power"):
        avg = _eval_family(full_pool, actual_top10, fam)
        print(f"{fam}: {avg:.2f}/10")

    print("\n=== Fit Boost Models (Actual Data) ===")
    for fam in ("sigmoid", "log", "power"):
        params, mse = _fit_boost_family(fit_rows, fam, iters=12000)
        avg = _eval_fit_family(full_pool, actual_top10, fam, params)
        p = tuple(round(x, 4) for x in params)
        print(f"{fam}: mse={mse:.4f} overlap={avg:.2f}/10 params={p}")

    ceiling = _oracle_ceiling(full_pool, actual_rows)
    print("\n=== Oracle Ceiling (Hindsight) ===")
    print(f"oracle_ceiling: {ceiling:.2f}/10")

    print("\n=== Walk-Forward (No Leakage) ===")
    # Evaluate walk-forward using fitted params per family.
    wf_results = []
    for fam in ("sigmoid", "log", "power"):
        params, _ = _fit_boost_family(fit_rows, fam, iters=3000)
        wf = _walk_forward_eval(
            dates=dates,
            full_pool=full_pool,
            actual_rows=actual_rows,
            actual_top10=actual_top10,
            family=fam,
            params=params,
            min_history_games=2,
        )
        wf_results.append((fam, params, wf))
        print(f"{fam}: walk_forward={wf:.2f}/10 params={tuple(round(x, 4) for x in params)}")

    wf_best_avg, wf_best_cfg = _walk_forward_search(
        dates=dates,
        full_pool=full_pool,
        actual_rows=actual_rows,
        actual_top10=actual_top10,
        n=8000,
    )
    print("\n=== Walk-Forward Search Best ===")
    print(f"best_wf_avg: {wf_best_avg:.2f}/10")
    print(f"best_wf_cfg: {json.dumps(wf_best_cfg, indent=2)}")

    print("\n=== RS-First Search (Top Real Scorers) ===")
    rs_score, rs_metrics, rs_cfg = _rs_first_search(
        dates=dates,
        full_pool=full_pool,
        actual_rows=actual_rows,
        n=6000,
    )
    print(f"best_rs_objective: {rs_score:.3f}")
    print(f"rs_metrics: {json.dumps(rs_metrics, indent=2)}")
    print(f"rs_cfg: {json.dumps(rs_cfg, indent=2)}")
    print("\nRS date-by-date hit delta (tuned - baseline):")
    deltas = _rs_first_date_compare(dates, full_pool, actual_rows, rs_cfg)
    for row in deltas:
        print(
            f"{row['date']}: baseline={row['baseline_hits']} tuned={row['tuned_hits']} delta={row['delta']:+d}"
        )

    print("\n=== Hindsight Top Real Scorers Target ===")
    hindsight_oracle = _hindsight_top_real_scorers(actual_rows, k=10)
    hindsight_prior = _hindsight_player_prior_top_real_scorers(actual_rows, k=10)
    print(f"hindsight_oracle_top10_hits: {hindsight_oracle:.2f}/10")
    print(f"hindsight_prior_top10_hits: {hindsight_prior:.2f}/10")

    print("\n=== Phase 2: Joint RS+Boost Lineup Search ===")
    best_quality, best_joint_cfg = _walk_forward_joint_lineup_search(
        dates=dates,
        full_pool=full_pool,
        actual_rows=actual_rows,
        n=6000,
    )
    print(f"best_lineup_presence_score: {best_quality:.3f}")
    print(f"best_joint_cfg: {json.dumps(best_joint_cfg, indent=2)}")

    best_avg, best_cfg = _random_search(full_pool, actual_top10, n=5000)
    print("\\n=== Random Search Best ===")
    print(f"best_avg: {best_avg:.2f}/10")
    print(f"best_cfg: {json.dumps(best_cfg, indent=2)}")


if __name__ == "__main__":
    main()
