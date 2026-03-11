# ─────────────────────────────────────────────────────────────────────────────
# LINE ENGINE — Daily Pick powered by ESPN projections + Claude
#
# Pipeline:
#   1. Pull today's player projections from the Oracle's projection engine
#   2. Build structured context (stats, game script, injuries, B2B, spread)
#   3. Ask Claude Haiku to identify the single best player prop edge
#   4. Claude returns structured JSON pick + narrative reasoning
#   5. Falls back to algorithmic model pick if Claude API unavailable
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-haiku-4-5-20251001"


# ─────────────────────────────────────────────────────────────────────────────
# GAME CONTEXT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _format_game_time_et(iso_date_str):
    """Format ESPN ISO date to e.g. 1:00 PM ET (portable)."""
    if not iso_date_str:
        return ""
    try:
        dt = datetime.fromisoformat(iso_date_str.replace("Z", "+00:00"))
        from zoneinfo import ZoneInfo
        et = dt.astimezone(ZoneInfo("America/New_York"))
        hour_12 = et.hour % 12 or 12
        return f"{hour_12}:{et.minute:02d} {et.strftime('%p')} ET"
    except Exception:
        return ""


def _game_lookup_from_games(games):
    """Build team abbr -> {opponent, opp_b2b, total, spread, game_time} lookup."""
    lookup = {}
    for g in games:
        home_abbr = g["home"]["abbr"]
        away_abbr = g["away"]["abbr"]
        game_time = _format_game_time_et(g.get("startTime", ""))
        entry = {
            "opponent": away_abbr, "opp_b2b": g.get("away_b2b", False),
            "total": g.get("total", 222), "spread": g.get("spread", 0),
            "game_time": game_time,
        }
        lookup[home_abbr] = dict(entry)
        lookup[away_abbr] = {
            "opponent": home_abbr, "opp_b2b": g.get("home_b2b", False),
            "total": g.get("total", 222), "spread": g.get("spread", 0),
            "game_time": game_time,
        }
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE-POWERED ENGINE
# ─────────────────────────────────────────────────────────────────────────────

_STAT_META = {
    "points":   {"field": "pts",  "season_field": "season_pts",  "recent_field": "recent_pts",  "min_season": 6.0,  "label": "pts"},
    "rebounds": {"field": "reb",  "season_field": "season_reb",  "recent_field": "recent_reb",  "min_season": 2.0,  "label": "reb"},
    "assists":  {"field": "ast",  "season_field": "season_ast",  "recent_field": "recent_ast",  "min_season": 1.5,  "label": "ast"},
}


def _build_claude_prompt(projections, games, stat_focus="points", force_direction=None):
    """Format ESPN projection data into a structured prompt for Claude for a given stat type."""
    meta         = _STAT_META[stat_focus]
    game_ctx_map = _game_lookup_from_games(games)

    # Sort by biggest projected edge for this stat vs season baseline
    scored = []
    for p in projections:
        season_val = p.get(meta["season_field"], p.get(meta["field"], 0))
        proj_val   = p.get(meta["field"], 0)
        if season_val < meta["min_season"] or p.get("predMin", 0) < 15 or proj_val <= 0:
            continue
        scored.append((abs(proj_val - season_val), p))
    scored.sort(reverse=True)
    top = [p for _, p in scored[:25]]

    # Game slate
    game_lines = []
    for g in games:
        b2b = []
        if g.get("home_b2b"): b2b.append(f"{g['home']['abbr']} on B2B")
        if g.get("away_b2b"): b2b.append(f"{g['away']['abbr']} on B2B")
        b2b_note = f" [{', '.join(b2b)}]" if b2b else ""
        game_lines.append(
            f"  {g['away']['abbr']} @ {g['home']['abbr']}: "
            f"spread {g.get('spread', 0):+.1f}, O/U {g.get('total', 222)}{b2b_note}"
        )

    # Player projection rows
    lbl = meta["label"]
    player_lines = []
    for p in top:
        season_val = p.get(meta["season_field"], p.get(meta["field"], 0))
        recent_val = p.get(meta["recent_field"], season_val)
        proj_val   = p.get(meta["field"], 0)
        gctx       = game_ctx_map.get(p.get("team", ""), {})
        opp        = gctx.get("opponent", "?")
        edge       = round(proj_val - season_val, 1)
        flags = []
        cascade = p.get("_cascade_bonus", 0)
        if cascade > 0:          flags.append(f"cascade+{cascade:.1f}min")
        if gctx.get("opp_b2b"):  flags.append("opp-B2B")
        inj = p.get("injury_status", "")
        if inj:                  flags.append(inj)
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        player_lines.append(
            f"  {p['name']} ({p.get('team','')}) vs {opp}: "
            f"proj {p.get('pts',0):.1f}pts/{p.get('reb',0):.1f}reb/{p.get('ast',0):.1f}ast "
            f"in {p.get('predMin',0):.0f}min | "
            f"season {season_val:.1f}{lbl} / recent {recent_val:.1f}{lbl} | "
            f"edge {'+' if edge>=0 else ''}{edge:.1f}{flag_str}"
        )

    direction_instruction = (
        f"Your pick MUST be {force_direction.upper()}. Find only the best {force_direction.upper()} edge — do not suggest the other direction."
        if force_direction else
        "You may pick OVER or UNDER based on which direction has the strongest edge."
    )
    direction_value = force_direction or "over"

    return f"""You are The Oracle's line engine. Analyze today's NBA slate and identify the single best player prop bet for {stat_focus.upper()}.

TODAY'S GAMES:
{chr(10).join(game_lines)}

TOP PLAYER PROJECTIONS FOR {stat_focus.upper()} (sorted by edge vs season baseline):
{chr(10).join(player_lines)}

PICK CRITERIA (in priority order):
1. Cascade bonus — player getting extra minutes because a teammate is OUT
2. Opponent on B2B — fatigued defense, more easy buckets/boards/dimes
3. High game total (230+) — faster pace, more possessions, more stats
4. Big recent form spike or slump vs season average
5. Close spread — starters play full minutes in tight games

{direction_instruction}
AVOID: players on B2B themselves, blowout favorites (team spread >10), injury-doubtful

Set "line" to season average rounded to nearest 0.5 (what books typically set).
Confidence range: 60-85.

Respond with ONLY valid JSON, no markdown fences:
{{
  "player_name": "Full Name",
  "team": "ABBR",
  "opponent": "ABBR",
  "stat_type": "{stat_focus}",
  "direction": "{direction_value}",
  "line": 22.5,
  "projection": 26.8,
  "edge": 4.3,
  "confidence": 74,
  "narrative": "2-3 sentences explaining the pick with specific data points",
  "signals": ["Cascade: +4.2 projected minutes from teammate injury", "Opponent on B2B", "High total (234.5)"]
}}"""


def _call_claude(prompt, stat_focus="points"):
    """Call Claude Haiku and return parsed JSON pick, or None on failure."""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": CLAUDE_MODEL,
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["content"][0]["text"].strip()
        # Strip markdown fences if Claude adds them despite instructions
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()
        pick = json.loads(text)
        # Ensure stat_type is set correctly for this call
        if pick and not pick.get("stat_type"):
            pick["stat_type"] = stat_focus
        return pick
    except Exception as e:
        print(f"[LineEngine] Claude API error ({stat_focus}): {e}")
        return None


def _call_claude_for_stat(projections, games, stat_focus, force_direction=None):
    """Build prompt and call Claude for a single stat type + direction. Returns (confidence, pick) or None."""
    prompt = _build_claude_prompt(projections, games, stat_focus, force_direction)
    pick   = _call_claude(prompt, stat_focus)
    if pick and pick.get("player_name") and pick.get("confidence", 0) > 0:
        # Enforce direction if forced (Claude occasionally ignores instruction)
        if force_direction:
            pick["direction"] = force_direction
        return (pick.get("confidence", 0), pick)
    return None


def _run_parallel_claude(projections, games):
    """
    Run Claude in parallel for points/rebounds/assists × over/under (6 calls).
    Returns (best_over_pick, best_under_pick) independently.
    """
    stat_types = ["points", "rebounds", "assists"]
    directions = ["over", "under"]
    best_over, best_under = None, None
    best_over_conf, best_under_conf = 0, 0

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_call_claude_for_stat, projections, games, s, d): (s, d)
            for s in stat_types for d in directions
        }
        for future in as_completed(futures):
            stat, direction = futures[future]
            try:
                result = future.result()
                if result:
                    conf, pick = result
                    print(f"[LineEngine] {stat}/{direction}: conf={conf}, player={pick.get('player_name','?')}")
                    if direction == "over" and conf > best_over_conf:
                        best_over_conf, best_over = conf, pick
                    elif direction == "under" and conf > best_under_conf:
                        best_under_conf, best_under = conf, pick
            except Exception as e:
                print(f"[LineEngine] parallel call failed ({stat}/{direction}): {e}")

    return best_over, best_under


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHMIC FALLBACK — pure ESPN model, no external API calls
# ─────────────────────────────────────────────────────────────────────────────

def run_model_fallback(projections, games, line_config=None):
    """
    Algorithmic pick when Claude API is unavailable.
    Scores edges across points, rebounds, and assists; picks the highest-confidence one.
    line_config: optional dict with min_confidence, min_edge_pct (from model-config line section).
    """
    if not projections:
        return {"pick": None, "error": "no_projections"}

    game_ctx_map = _game_lookup_from_games(games)
    candidates = []
    cfg = line_config or {}
    min_confidence = cfg.get("min_confidence", 50)
    min_edge_pct = cfg.get("min_edge_pct", 0.0)
    recent_form_over_ratio = cfg.get("recent_form_over_ratio", 1.08)
    recent_form_under_ratio = cfg.get("recent_form_under_ratio", 0.92)
    min_edge_pts = cfg.get("min_edge_pts", 2.0)
    min_edge_other = cfg.get("min_edge_other", 1.5)

    stat_configs = [
        ("points",   "pts",  "season_pts",  "recent_pts",  8.0, 18),
        ("rebounds", "reb",  "season_reb",  "recent_reb",  2.5, 15),
        ("assists",  "ast",  "season_ast",  "recent_ast",  2.0, 15),
    ]

    for p in projections:
        pred_min  = p.get("predMin", 0)
        team_abbr = p.get("team", "")
        gctx      = game_ctx_map.get(team_abbr, {})
        opponent  = gctx.get("opponent", "")

        for stat_type, field, season_field, recent_field, min_season, min_min in stat_configs:
            season_val = p.get(season_field, p.get(field, 0))
            proj_val   = p.get(field, 0)
            if season_val < min_season or pred_min < min_min or proj_val <= 0:
                continue

            min_edge = min_edge_pts if stat_type == "points" else min_edge_other
            line      = round(round(season_val * 2) / 2, 1)
            edge      = round(proj_val - line, 1)
            direction = "over" if edge > 0 else "under"
            if abs(edge) < min_edge:
                continue

            signals, signal_bonus = [], 0
            cascade = p.get("_cascade_bonus", 0)
            if cascade > 0:
                signals.append({"type": "cascade", "detail": f"Injury upgrade — +{cascade:.1f} projected minutes"})
                signal_bonus += 15

            recent_val = p.get(recent_field, proj_val)
            if direction == "over" and recent_val > season_val * recent_form_over_ratio:
                signals.append({"type": "recent_form", "detail": f"Averaging {recent_val:.1f} {stat_type} recently vs {season_val:.1f} season"})
                signal_bonus += 12
            elif direction == "under" and recent_val < season_val * recent_form_under_ratio:
                signals.append({"type": "recent_form", "detail": f"Averaging {recent_val:.1f} {stat_type} recently vs {season_val:.1f} season"})
                signal_bonus += 12

            if gctx.get("opp_b2b"):
                signals.append({"type": "opp_b2b", "detail": "Opponent on second night of B2B"})
                signal_bonus += 10

            edge_score = min(abs(edge) / 5.0 * 40, 40)
            confidence = round(min(52 + edge_score + signal_bonus, 80))
            narrative  = (
                f"Model projects {proj_val:.1f} {stat_type} — a {abs(edge):.1f} edge "
                f"vs the {line:.1f} season baseline."
            )
            edge_pct = (abs(edge) / line * 100.0) if line and line > 0 else 0.0
            if confidence < min_confidence or (min_edge_pct > 0 and edge_pct < min_edge_pct):
                continue
            # L5 heuristic: 5 bars with ratio recent/season (0–1) for sparkline
            ratio = min(1.2, recent_val / max(season_val, 0.01)) if season_val else 1.0
            recent_form_bars = [round(ratio, 2)] * 5
            avg_min = p.get("season_min", p.get("min", 0))
            candidates.append({
                "player_name": p["name"], "player_id": p.get("id", ""),
                "team": team_abbr, "opponent": opponent,
                "stat_type": stat_type, "line": line, "direction": direction,
                "projection": proj_val, "edge": edge, "confidence": confidence,
                "odds_over": None, "odds_under": None, "books_consensus": 0,
                "model_only": True, "signals": signals, "narrative": narrative,
                "season_avg": round(season_val, 1),
                "proj_min": round(pred_min, 1),
                "avg_min": round(avg_min, 1) if isinstance(avg_min, (int, float)) else 0,
                "game_time": gctx.get("game_time", ""),
                "recent_form_bars": recent_form_bars,
            })

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    over_candidates  = [c for c in candidates if c["direction"] == "over"]
    under_candidates = [c for c in candidates if c["direction"] == "under"]
    over_pick  = over_candidates[0]  if over_candidates  else None
    under_pick = under_candidates[0] if under_candidates else None

    # Last-resort pass: if one direction is still empty, find the best available
    # pick for it by relaxing the edge threshold — guarantees both directions always
    # produce a pick as long as any player is projected above or below their line.
    if not over_pick or not under_pick:
        last_resort = []
        for p in projections:
            # Use season_min as fallback when predMin is 0 (common for future-date projections)
            pred_min  = p.get("predMin", 0) or p.get("season_min", p.get("min", 0))
            team_abbr = p.get("team", "")
            gctx      = game_ctx_map.get(team_abbr, {})
            for stat_type, field, season_field, recent_field, min_season, min_min in stat_configs:
                season_val = p.get(season_field, p.get(field, 0))
                proj_val   = p.get(field, 0)
                if season_val < min_season or pred_min < min_min or proj_val <= 0:
                    continue
                line      = round(round(season_val * 2) / 2, 1)
                edge      = round(proj_val - line, 1)
                if edge == 0:
                    continue
                direction = "over" if edge > 0 else "under"
                if direction == "over" and over_pick:
                    continue
                if direction == "under" and under_pick:
                    continue
                avg_min = p.get("season_min", p.get("min", 0))
                last_resort.append({
                    "player_name": p["name"], "player_id": p.get("id", ""),
                    "team": team_abbr, "opponent": gctx.get("opponent", ""),
                    "stat_type": stat_type, "line": line, "direction": direction,
                    "projection": proj_val, "edge": edge, "confidence": 52,
                    "odds_over": None, "odds_under": None, "books_consensus": 0,
                    "model_only": True, "signals": [],
                    "narrative": f"Model projects {proj_val:.1f} {stat_type} vs the {line:.1f} baseline.",
                    "season_avg": round(season_val, 1),
                    "proj_min": round(pred_min, 1),
                    "avg_min": round(avg_min, 1) if isinstance(avg_min, (int, float)) else 0,
                    "game_time": gctx.get("game_time", ""),
                    "recent_form_bars": [1.0] * 5,
                })
        if last_resort:
            last_resort.sort(key=lambda c: abs(c["edge"]), reverse=True)
            if not over_pick:
                lr_over = [c for c in last_resort if c["direction"] == "over"]
                if lr_over:
                    over_pick = lr_over[0]
            if not under_pick:
                lr_under = [c for c in last_resort if c["direction"] == "under"]
                if lr_under:
                    under_pick = lr_under[0]

    if not candidates and not over_pick and not under_pick:
        return {"pick": None, "over_pick": None, "under_pick": None, "error": "no_edges"}
    primary    = over_pick if (over_pick and (not under_pick or over_pick["confidence"] >= under_pick["confidence"])) else under_pick
    return {
        "pick": primary, "over_pick": over_pick, "under_pick": under_pick, "runner_up": None,
        "slate_summary": {
            "games_evaluated": len(games), "props_scanned": len(candidates),
            "edges_found": len(candidates), "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_only": True,
        },
    }


def _enrich_pick_from_projections(pick, projections, game_ctx_map):
    """Add season_avg, proj_min, avg_min, game_time, recent_form_bars from projections when missing."""
    if not pick or not projections:
        return
    name = pick.get("player_name", "")
    team = pick.get("team", "")
    stat_type = pick.get("stat_type", "points")
    meta = _STAT_META.get(stat_type, _STAT_META["points"])
    season_field = meta["season_field"]
    recent_field = meta["recent_field"]
    for p in projections:
        if p.get("name") == name and p.get("team") == team:
            if "season_avg" not in pick or pick.get("season_avg") is None:
                pick["season_avg"] = round(p.get(season_field, p.get(meta["field"], 0)), 1)
            if "proj_min" not in pick or pick.get("proj_min") is None:
                pick["proj_min"] = round(p.get("predMin", 0), 1)
            avg_min = p.get("season_min", p.get("min", 0))
            if "avg_min" not in pick or pick.get("avg_min") is None:
                pick["avg_min"] = round(avg_min, 1) if isinstance(avg_min, (int, float)) else 0
            if "game_time" not in pick or not pick.get("game_time"):
                pick["game_time"] = game_ctx_map.get(team, {}).get("game_time", "")
            if "recent_form_bars" not in pick or not pick.get("recent_form_bars"):
                season_val = p.get(season_field, p.get(meta["field"], 0)) or 0.01
                recent_val = p.get(recent_field, pick.get("projection", 0))
                ratio = min(1.2, recent_val / max(season_val, 0.01))
                pick["recent_form_bars"] = [round(ratio, 2)] * 5
            break


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_line_engine(projections, games, line_config=None):
    """
    Main entry point. Uses Claude Haiku to reason about ESPN projection data
    and pick today's best player prop edge. Falls back to algorithmic model
    if Claude API is unavailable.
    line_config: optional dict (e.g. from model-config "line" section) with min_confidence, min_edge_pct.
    """
    if not games:
        return {"pick": None, "error": "no_games"}
    if not projections:
        return {"pick": None, "error": "no_projections"}

    # Filter out players who don't average enough season minutes — prevents
    # fringe vets from qualifying on a single inflated projection day.
    min_season_min = (line_config or {}).get("min_season_minutes", 20.0)
    if min_season_min > 0:
        projections = [
            p for p in projections
            if (p.get("season_min") or p.get("min") or 0) >= min_season_min
        ]

    min_confidence = (line_config or {}).get("min_confidence", 50)

    if not ANTHROPIC_API_KEY:
        return run_model_fallback(projections, games, line_config)

    over_data, under_data = _run_parallel_claude(projections, games)

    if not over_data and not under_data:
        print("[LineEngine] Claude returned no picks — using algorithmic fallback")
        return run_model_fallback(projections, games, line_config)

    def _build_pick(pd):
        if not pd:
            return None
        signals = [
            {"type": s.lower().split(":")[0].strip().replace(" ", "_"), "detail": s}
            for s in pd.get("signals", [])
        ]
        return {
            "player_name":     pd["player_name"],
            "player_id":       "",
            "team":            pd.get("team", ""),
            "opponent":        pd.get("opponent", ""),
            "stat_type":       pd.get("stat_type", "points"),
            "line":            pd.get("line", 0),
            "direction":       pd.get("direction", "over"),
            "projection":      pd.get("projection", 0),
            "edge":            pd.get("edge", 0),
            "confidence":      pd.get("confidence", 70),
            "odds_over":       None,
            "odds_under":      None,
            "books_consensus": 0,
            "model_only":      True,
            "signals":         signals,
            "narrative":       pd.get("narrative", ""),
        }

    over_pick  = _build_pick(over_data)
    under_pick = _build_pick(under_data)

    # Enforce min_confidence: reject Claude picks below threshold
    if over_pick and over_pick.get("confidence", 0) < min_confidence:
        over_pick = None
    if under_pick and under_pick.get("confidence", 0) < min_confidence:
        under_pick = None

    # Fill missing direction from algorithmic fallback
    if not over_pick or not under_pick:
        fallback = run_model_fallback(projections, games, line_config)
        if not over_pick:
            over_pick = fallback.get("over_pick")
        if not under_pick:
            under_pick = fallback.get("under_pick")

    # Enrich Claude-built picks with season_avg, proj_min, avg_min, game_time, recent_form_bars
    game_ctx_map = _game_lookup_from_games(games)
    if over_pick:
        _enrich_pick_from_projections(over_pick, projections, game_ctx_map)
    if under_pick:
        _enrich_pick_from_projections(under_pick, projections, game_ctx_map)

    primary = over_pick if (over_pick and (not under_pick or over_pick.get("confidence", 0) >= under_pick.get("confidence", 0))) else under_pick

    return {
        "pick":       primary,
        "over_pick":  over_pick,
        "under_pick": under_pick,
        "runner_up":  None,
        "slate_summary": {
            "games_evaluated": len(games),
            "props_scanned":   len(projections),
            "edges_found":     (1 if over_pick else 0) + (1 if under_pick else 0),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "model_only":      True,
        },
    }
