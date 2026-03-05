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
from datetime import datetime, timezone

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-opus-4-6"


# ─────────────────────────────────────────────────────────────────────────────
# GAME CONTEXT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _game_lookup_from_games(games):
    """Build team abbr -> {opponent, opp_b2b, total, spread} lookup."""
    lookup = {}
    for g in games:
        home_abbr = g["home"]["abbr"]
        away_abbr = g["away"]["abbr"]
        lookup[home_abbr] = {
            "opponent": away_abbr, "opp_b2b": g.get("away_b2b", False),
            "total": g.get("total", 222), "spread": g.get("spread", 0),
        }
        lookup[away_abbr] = {
            "opponent": home_abbr, "opp_b2b": g.get("home_b2b", False),
            "total": g.get("total", 222), "spread": g.get("spread", 0),
        }
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE-POWERED ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _build_claude_prompt(projections, games):
    """Format ESPN projection data into a structured prompt for Claude."""
    game_ctx_map = _game_lookup_from_games(games)

    # Sort by biggest projected edge vs season baseline
    scored = []
    for p in projections:
        season_pts = p.get("season_pts", p.get("pts", 0))
        proj_pts   = p.get("pts", 0)
        if season_pts < 6 or p.get("predMin", 0) < 15 or proj_pts <= 0:
            continue
        scored.append((abs(proj_pts - season_pts), p))
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
    player_lines = []
    for p in top:
        season_pts = p.get("season_pts", p.get("pts", 0))
        recent_pts = p.get("recent_pts", season_pts)
        gctx       = game_ctx_map.get(p.get("team", ""), {})
        opp        = gctx.get("opponent", "?")
        edge       = round(p.get("pts", 0) - season_pts, 1)
        flags = []
        cascade = p.get("_cascade_bonus", 0)
        if cascade > 0:        flags.append(f"cascade+{cascade:.1f}min")
        if gctx.get("opp_b2b"): flags.append("opp-B2B")
        inj = p.get("injury_status", "")
        if inj:                flags.append(inj)
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        player_lines.append(
            f"  {p['name']} ({p.get('team','')}) vs {opp}: "
            f"proj {p.get('pts',0):.1f}pts/{p.get('reb',0):.1f}reb/{p.get('ast',0):.1f}ast "
            f"in {p.get('predMin',0):.0f}min | "
            f"season {season_pts:.1f}pts / recent {recent_pts:.1f}pts | "
            f"edge {'+' if edge>=0 else ''}{edge:.1f}{flag_str}"
        )

    return f"""You are The Oracle's line engine. Analyze today's NBA slate and identify the single best player prop bet.

TODAY'S GAMES:
{chr(10).join(game_lines)}

TOP PLAYER PROJECTIONS (sorted by edge vs season baseline):
{chr(10).join(player_lines)}

PICK CRITERIA (in priority order):
1. Cascade bonus — player getting extra minutes because a teammate is OUT
2. Opponent on B2B — fatigued defense, more easy buckets
3. High game total (230+) — faster pace, more possessions, more stats
4. Big recent form spike — player is running hot vs season average
5. Close spread — starters play full minutes in tight games

AVOID: players on B2B themselves, blowout favorites (team spread >10), injury-doubtful

Set "line" to season average rounded to nearest 0.5 (what books typically set).
Confidence range: 60-82. Only pick OVER bets where our projection exceeds the baseline.

Respond with ONLY valid JSON, no markdown fences:
{{
  "player_name": "Full Name",
  "team": "ABBR",
  "opponent": "ABBR",
  "stat_type": "points",
  "direction": "over",
  "line": 22.5,
  "projection": 26.8,
  "edge": 4.3,
  "confidence": 74,
  "narrative": "2-3 sentences explaining the pick with specific data points",
  "signals": ["Cascade: +4.2 projected minutes from teammate injury", "Opponent on B2B", "High total (234.5)"]
}}"""


def _call_claude(prompt):
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
        return json.loads(text)
    except Exception as e:
        print(f"[LineEngine] Claude API error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHMIC FALLBACK — pure ESPN model, no external API calls
# ─────────────────────────────────────────────────────────────────────────────

def run_model_fallback(projections, games):
    """
    Algorithmic pick when Claude API is unavailable.
    Ranks players by projected pts vs season baseline with signal bonuses.
    """
    if not projections:
        return {"pick": None, "error": "no_projections"}

    game_ctx_map = _game_lookup_from_games(games)
    candidates = []

    for p in projections:
        season_pts = p.get("season_pts", 0)
        proj_pts   = p.get("pts", 0)
        pred_min   = p.get("predMin", 0)
        if season_pts < 8.0 or pred_min < 18 or proj_pts <= 0:
            continue

        line      = round(round(season_pts * 2) / 2, 1)
        edge      = round(proj_pts - line, 1)
        direction = "over" if edge > 0 else "under"
        if abs(edge) < 2.0:
            continue

        signals, signal_bonus = [], 0
        cascade = p.get("_cascade_bonus", 0)
        if cascade > 0:
            signals.append({"type": "cascade", "detail": f"Injury upgrade — +{cascade:.1f} projected minutes"})
            signal_bonus += 15

        recent_pts = p.get("recent_pts", proj_pts)
        if direction == "over" and recent_pts > season_pts * 1.08:
            signals.append({"type": "recent_form", "detail": f"Averaging {recent_pts:.1f} pts recently vs {season_pts:.1f} season"})
            signal_bonus += 12
        elif direction == "under" and recent_pts < season_pts * 0.92:
            signals.append({"type": "recent_form", "detail": f"Averaging {recent_pts:.1f} pts recently vs {season_pts:.1f} season"})
            signal_bonus += 12

        team_abbr = p.get("team", "")
        gctx      = game_ctx_map.get(team_abbr, {})
        opponent  = gctx.get("opponent", "")
        if gctx.get("opp_b2b"):
            signals.append({"type": "opp_b2b", "detail": "Opponent on second night of B2B"})
            signal_bonus += 10

        edge_score = min(abs(edge) / 5.0 * 40, 40)
        confidence = round(min(52 + edge_score + signal_bonus, 80))
        narrative  = (
            f"Model projects {proj_pts:.1f} pts — a {abs(edge):.1f}-pt edge "
            f"vs the {line:.1f} season baseline at {confidence}% confidence."
        )
        candidates.append({
            "player_name": p["name"], "player_id": p.get("id", ""),
            "team": team_abbr, "opponent": opponent,
            "stat_type": "points", "line": line, "direction": direction,
            "projection": proj_pts, "edge": edge, "confidence": confidence,
            "odds_over": None, "odds_under": None, "books_consensus": 0,
            "model_only": True, "signals": signals, "narrative": narrative,
        })

    if not candidates:
        return {"pick": None, "error": "no_edges"}

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    pick      = candidates[0]
    runner_up = next((c for c in candidates[1:] if c["player_name"] != pick["player_name"]), None)
    return {
        "pick": pick, "runner_up": runner_up,
        "slate_summary": {
            "games_evaluated": len(games), "props_scanned": len(candidates),
            "edges_found": len(candidates), "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_only": True,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_line_engine(projections, games):
    """
    Main entry point. Uses Claude Haiku to reason about ESPN projection data
    and pick today's best player prop edge. Falls back to algorithmic model
    if Claude API is unavailable.
    """
    if not games:
        return {"pick": None, "error": "no_games"}
    if not projections:
        return {"pick": None, "error": "no_projections"}

    if not ANTHROPIC_API_KEY:
        return run_model_fallback(projections, games)

    prompt    = _build_claude_prompt(projections, games)
    pick_data = _call_claude(prompt)

    if not pick_data or not pick_data.get("player_name"):
        print("[LineEngine] Claude returned no pick — using algorithmic fallback")
        return run_model_fallback(projections, games)

    signals = [
        {"type": s.lower().split(":")[0].strip().replace(" ", "_"), "detail": s}
        for s in pick_data.get("signals", [])
    ]
    pick = {
        "player_name":     pick_data["player_name"],
        "player_id":       "",
        "team":            pick_data.get("team", ""),
        "opponent":        pick_data.get("opponent", ""),
        "stat_type":       pick_data.get("stat_type", "points"),
        "line":            pick_data.get("line", 0),
        "direction":       pick_data.get("direction", "over"),
        "projection":      pick_data.get("projection", 0),
        "edge":            pick_data.get("edge", 0),
        "confidence":      pick_data.get("confidence", 70),
        "odds_over":       None,
        "odds_under":      None,
        "books_consensus": 0,
        "model_only":      True,
        "signals":         signals,
        "narrative":       pick_data.get("narrative", ""),
    }

    return {
        "pick": pick,
        "runner_up": None,
        "slate_summary": {
            "games_evaluated": len(games),
            "props_scanned":   len(projections),
            "edges_found":     1,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "model_only":      True,
        },
    }
