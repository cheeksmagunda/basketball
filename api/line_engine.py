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


def _lookup_player_odds(player_odds_map, player_name, stat_type):
    """Return bookmaker odds data for a player+stat from the pre-fetched map.

    Uses exact match first, then substring match to handle minor name
    format differences (e.g. apostrophes, Jr/Sr suffixes).
    Returns {"line", "odds_over", "odds_under", "books_consensus"} or None.
    """
    if not player_odds_map:
        return None
    pname = player_name.lower()
    if (pname, stat_type) in player_odds_map:
        return player_odds_map[(pname, stat_type)]
    for (odds_name, odds_stat), data in player_odds_map.items():
        if odds_stat == stat_type and (pname in odds_name or odds_name in pname):
            return data
    return None


def _game_lookup_from_games(games):
    """Build team abbr -> {opponent, opp_b2b, total, spread, game_time, game_start_iso} lookup."""
    lookup = {}
    for g in games:
        home_abbr = g["home"]["abbr"]
        away_abbr = g["away"]["abbr"]
        game_time = _format_game_time_et(g.get("startTime", ""))
        start_iso = g.get("startTime", "")
        entry = {
            "opponent": away_abbr, "opp_b2b": g.get("away_b2b", False),
            "total": g.get("total", 222), "spread": g.get("spread", 0),
            "game_time": game_time, "game_start_iso": start_iso,
        }
        lookup[home_abbr] = dict(entry)
        lookup[away_abbr] = {
            "opponent": home_abbr, "opp_b2b": g.get("home_b2b", False),
            "total": g.get("total", 222), "spread": g.get("spread", 0),
            "game_time": game_time, "game_start_iso": start_iso,
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


def _build_claude_prompt(projections, games, stat_focus="points", force_direction=None, stat_floors=None, player_odds_map=None, news_context="", dvp_data=None):
    """Format ESPN projection data into a structured prompt for Claude for a given stat type."""
    meta = dict(_STAT_META[stat_focus])  # copy so we can override min_season without mutating module-level dict
    if stat_floors and stat_focus in stat_floors:
        meta["min_season"] = stat_floors[stat_focus]
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

    # Game slate with defensive context
    game_lines = []
    for g in games:
        b2b = []
        if g.get("home_b2b"): b2b.append(f"{g['home']['abbr']} on B2B")
        if g.get("away_b2b"): b2b.append(f"{g['away']['abbr']} on B2B")
        b2b_note = f" [{', '.join(b2b)}]" if b2b else ""
        # Include opponent defensive rating when available for matchup context
        home_def = g.get("home", {}).get("opp_def_rating", "")
        away_def = g.get("away", {}).get("opp_def_rating", "")
        def_note = ""
        if home_def or away_def:
            parts = []
            if away_def: parts.append(f"{g['away']['abbr']} defRtg {away_def}")
            if home_def: parts.append(f"{g['home']['abbr']} defRtg {home_def}")
            def_note = f" ({', '.join(parts)})"
        game_lines.append(
            f"  {g['away']['abbr']} @ {g['home']['abbr']}: "
            f"spread {g.get('spread', 0):+.1f}, O/U {g.get('total', 222)}{b2b_note}{def_note}"
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
        book_odds = _lookup_player_odds(player_odds_map, p["name"], stat_focus)
        book_str = f" [book: {book_odds['line']:.1f}{lbl}]" if book_odds else ""
        # Recent form trend indicator
        trend = ""
        if season_val > 0:
            ratio = recent_val / season_val
            if ratio >= 1.15:   trend = " ↑HOT"
            elif ratio >= 1.05: trend = " ↑warm"
            elif ratio <= 0.85: trend = " ↓COLD"
            elif ratio <= 0.95: trend = " ↓cool"
        player_lines.append(
            f"  {p['name']} ({p.get('team','')}) vs {opp}: "
            f"proj {p.get('pts',0):.1f}pts/{p.get('reb',0):.1f}reb/{p.get('ast',0):.1f}ast "
            f"in {p.get('predMin',0):.0f}min | "
            f"season {season_val:.1f}{lbl} / recent {recent_val:.1f}{lbl}{trend} | "
            f"edge {'+' if edge>=0 else ''}{edge:.1f}{flag_str}{book_str}"
        )

    direction_instruction = (
        f"Your pick MUST be {force_direction.upper()}. Find only the best {force_direction.upper()} edge — do not suggest the other direction."
        if force_direction else
        "You may pick OVER or UNDER based on which direction has the strongest edge."
    )
    direction_value = force_direction or "over"

    # DvP section — position-specific defensive weaknesses for today's matchups
    dvp_section = ""
    if dvp_data:
        dvp_lines = []
        for g in games:
            home = g.get("home", {}).get("abbr", "")
            away = g.get("away", {}).get("abbr", "")
            for team, opp in ((home, away), (away, home)):
                if opp and opp in dvp_data:
                    opp_dvp = dvp_data[opp]
                    # Show worst position (most PPG allowed) as the key exploit signal
                    if opp_dvp:
                        sorted_pos = sorted(opp_dvp.items(), key=lambda x: x[1], reverse=True)
                        worst_pos, worst_ppg = sorted_pos[0]
                        best_pos, best_ppg = sorted_pos[-1]
                        dvp_lines.append(
                            f"  {opp} defense: allows most to {worst_pos}s ({worst_ppg:.1f} PPG), "
                            f"least to {best_pos}s ({best_ppg:.1f} PPG) — "
                            f"target {team} {worst_pos}s"
                        )
        if dvp_lines:
            dvp_section = (
                "\n\nDEFENSE-VS-POSITION DATA (from NBA.com — use to refine pick):\n"
                + "\n".join(dvp_lines)
                + "\nFavor overs for players whose position faces a weak defense (high PPG allowed). "
                "Favor unders for players whose position faces an elite defense (low PPG allowed)."
            )

    # Web search news section — late-breaking injuries, rotation changes, rest decisions
    news_section = ""
    if news_context:
        news_section = (
            f"\n\nRECENT NBA NEWS (last 24-48 hours):\n{news_context}\n\n"
            "USE THIS NEWS to inform your pick. Injury upgrades (star OUT = role player minutes up) "
            "are HIGH-VALUE signals for overs. Load management or unexpected rest = unders. "
            "Coach rotation quotes are actionable — weight them heavily."
        )

    return f"""You are The Oracle's line engine. Analyze today's NBA slate and identify the single best player prop bet for {stat_focus.upper()}.

TODAY'S GAMES:
{chr(10).join(game_lines)}{dvp_section}

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
OVER-SPECIFIC RULES (overs have a 17% hit rate — be VERY selective):
- Points overs: ONLY pick when edge ≥ 3.0 AND at least one catalyst (cascade, opp-B2B, high total, or hot recent streak)
- Rebounds/assists overs: REQUIRE a strong catalyst (cascade, opp-B2B, or high total 230+) — do NOT pick on season-average form alone
- Prefer points OVER when edge quality is similar across stat types
- Weak opponent defense (high defRtg) is a CONFIRMING signal for overs, not standalone
- If news shows a teammate OUT, the cascade bonus is the strongest over signal available
UNDER-SPECIFIC: Unders are more reliable. Stars on heavy favorites (spread >8) sitting early = strong under signal. Player in a slump (recent < 0.85× season) = confirmed under.
{news_section}
Set "line" to the [book: X.X] value when shown — that is the real bookmaker number and is more accurate than the season average. Only fall back to season average rounded to nearest 0.5 if no [book: X.X] is present. Recalculate "edge" as projection minus the line you use.
Confidence range: 60-85.

NARRATIVE REQUIREMENT: Your narrative MUST explain WHY the projection differs from the baseline — not just restate the numbers. Cite at least one concrete driver: game pace/total, opponent defense quality, recent form trend, injury cascade, opponent B2B fatigue, spread/blowout risk, minutes projection change, or bookmaker line alignment. Bad: "Model projects 9.2 points vs 7.5 baseline." Good: "High-total game (235) and weak Wizards defense create extra scoring opportunities; Lopez has seen 28+ minutes in 3 of last 5."

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
  "narrative": "2-3 sentences explaining WHY with specific data points and drivers",
  "signals": ["High-total game (234.5) — more possessions expected", "Favorable matchup vs weak defense", "Opponent on B2B"]
}}"""


def _call_claude(prompt, stat_focus="points"):
    """Call Claude Haiku and return parsed JSON pick, or None on failure."""
    if not ANTHROPIC_API_KEY:
        print(f"[LineEngine] ANTHROPIC_API_KEY not set — using fallback for {stat_focus}")
        return None
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
    except requests.exceptions.HTTPError as http_err:
        status = http_err.response.status_code if http_err.response is not None else "?"
        label = {402: "credits-exhausted", 429: "rate-limited", 529: "overloaded"}.get(status, "HTTP error")
        print(f"[LineEngine] Claude {label} ({status}) for {stat_focus} — using fallback")
        return None
    except Exception as e:
        print(f"[LineEngine] Claude API error ({stat_focus}): {e}")
        return None


def _call_claude_for_stat(projections, games, stat_focus, force_direction=None, stat_floors=None, player_odds_map=None, news_context="", dvp_data=None):
    """Build prompt and call Claude for a single stat type + direction. Returns (confidence, pick) or None."""
    prompt = _build_claude_prompt(projections, games, stat_focus, force_direction, stat_floors, player_odds_map, news_context, dvp_data=dvp_data)
    pick   = _call_claude(prompt, stat_focus)
    if pick and pick.get("player_name") and pick.get("confidence", 0) > 0:
        # Enforce direction if forced (Claude occasionally ignores instruction)
        if force_direction:
            pick["direction"] = force_direction
        return (pick.get("confidence", 0), pick)
    return None


def _run_parallel_claude(projections, games, stat_floors=None, player_odds_map=None, news_context="", dvp_data=None):
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
            executor.submit(_call_claude_for_stat, projections, games, s, d, stat_floors, player_odds_map, news_context, dvp_data): (s, d)
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
# SIGNAL GENERATION — shared by both main candidate loop and last-resort path
# ─────────────────────────────────────────────────────────────────────────────

def _generate_signals(p, gctx, direction, stat_type, season_val, recent_val, proj_val, line, cfg):
    """Build driver signals and compute signal_bonus for a candidate pick.

    Returns (signals, signal_bonus) where signals is a list of
    {"type": str, "detail": str} dicts.
    """
    signals, signal_bonus = [], 0
    recent_form_over_ratio = cfg.get("recent_form_over_ratio", 1.08)
    recent_form_under_ratio = cfg.get("recent_form_under_ratio", 0.92)

    # 1. Cascade (injury minutes) — strongest signal
    cascade = p.get("_cascade_bonus", 0)
    if cascade > 0:
        signals.append({"type": "cascade", "detail": f"Injury upgrade — +{cascade:.1f} projected minutes"})
        signal_bonus += 15

    # 2. Recent form
    if direction == "over" and recent_val > season_val * recent_form_over_ratio:
        signals.append({"type": "recent_form", "detail": f"Averaging {recent_val:.1f} {stat_type} recently vs {season_val:.1f} season"})
        signal_bonus += 12
    elif direction == "under" and recent_val < season_val * recent_form_under_ratio:
        signals.append({"type": "recent_form", "detail": f"Averaging {recent_val:.1f} {stat_type} recently vs {season_val:.1f} season"})
        signal_bonus += 12

    # 3. Opponent on B2B
    if gctx.get("opp_b2b"):
        signals.append({"type": "opp_b2b", "detail": "Opponent on second night of B2B"})
        signal_bonus += 10

    # 4. Game total / pace
    game_total = gctx.get("total", 222)
    if game_total >= 230 and direction == "over":
        signals.append({"type": "high_total", "detail": f"High-total game ({game_total}) — more possessions expected"})
        signal_bonus += 8
    elif game_total <= 215 and direction == "under":
        signals.append({"type": "low_total", "detail": f"Low-total game ({game_total}) — fewer scoring opportunities"})
        signal_bonus += 8

    # 5. Matchup — opponent defensive quality
    matchup = p.get("_matchup_factor", 1.0)
    if matchup >= 1.06 and direction == "over":
        signals.append({"type": "matchup", "detail": "Favorable matchup vs weak defense"})
        signal_bonus += 6
    elif matchup <= 0.94 and direction == "under":
        signals.append({"type": "matchup", "detail": "Tough matchup vs strong defense"})
        signal_bonus += 6

    # 6. Books agreement (odds blend pushed projection up)
    if p.get("_odds_adjusted") and direction == "over":
        odds_field = f"odds_{stat_type.rstrip('s') if stat_type != 'assists' else 'ast'}_line"
        # Try common field patterns
        book_val = p.get(f"odds_{stat_type}_line", 0) or p.get(odds_field, 0)
        if book_val > 0:
            signals.append({"type": "books_agree", "detail": f"Sportsbooks project higher ({book_val:.1f} {stat_type})"})
            signal_bonus += 5

    # 7. Minutes drop (for unders)
    pred_min = p.get("predMin", 0) or p.get("season_min", p.get("min", 0))
    season_min_val = p.get("season_min", p.get("min", 0))
    if direction == "under" and pred_min > 0 and season_min_val > 0:
        min_drop = season_min_val - pred_min
        if min_drop >= 3.0:
            signals.append({"type": "minutes_drop", "detail": f"Projected {pred_min:.0f} min vs {season_min_val:.0f} season avg"})
            signal_bonus += 8

    # 8. Spread — blowout risk (starters sit) or close game (full minutes)
    spread = abs(gctx.get("spread", 0))
    if direction == "under" and spread >= 8:
        signals.append({"type": "blowout_risk", "detail": f"Heavy favorite (spread {spread:.1f}) — starters may sit early"})
        signal_bonus += 6
    elif direction == "over" and spread <= 3:
        signals.append({"type": "close_game", "detail": f"Tight spread ({spread:.1f}) — starters play full minutes"})
        signal_bonus += 4

    return signals, signal_bonus


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHMIC FALLBACK — pure ESPN model, no external API calls
# ─────────────────────────────────────────────────────────────────────────────

def run_model_fallback(projections, games, line_config=None, player_odds_map=None):
    """
    Algorithmic pick when Claude API is unavailable.
    Scores edges across points, rebounds, and assists; picks the highest-confidence one.
    line_config: optional dict with min_confidence, min_edge_pct (from model-config line section).
    player_odds_map: optional pre-fetched Odds API lines {(name_lower, stat_type): {...}}.
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
    # Over picks require a higher edge on non-points stats — rebounds/assists are
    # high-variance; a small edge isn't enough signal to bet on an over.
    min_edge_other_over = cfg.get("min_edge_other_over", min_edge_other)

    sf = cfg.get("stat_floors", {})
    stat_configs = [
        ("points",   "pts",  "season_pts",  "recent_pts",  sf.get("points",   8.0), 18),
        ("rebounds", "reb",  "season_reb",  "recent_reb",  sf.get("rebounds", 2.5), 15),
        ("assists",  "ast",  "season_ast",  "recent_ast",  sf.get("assists",  2.0), 15),
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

            book_odds = _lookup_player_odds(player_odds_map, p.get("name", ""), stat_type)
            line      = book_odds["line"] if book_odds else round(round(season_val * 2) / 2, 1)
            edge      = round(proj_val - line, 1)
            direction = "over" if edge > 0 else "under"
            if stat_type == "points":
                min_edge = min_edge_pts
            elif direction == "over":
                min_edge = min_edge_other_over
            else:
                min_edge = min_edge_other
            if abs(edge) < min_edge:
                continue

            recent_val = p.get(recent_field, proj_val)
            signals, signal_bonus = _generate_signals(
                p, gctx, direction, stat_type, season_val, recent_val, proj_val, line, cfg
            )

            edge_score = min(abs(edge) / 5.0 * 40, 40)
            # Penalty for overs without strong catalysts — overs hit only 17% historically.
            # Require at least one signal (cascade, recent form, opp-B2B) for full confidence.
            over_penalty = 0
            if direction == "over" and signal_bonus == 0:
                over_penalty = 12  # 12-point confidence penalty for unsupported overs
            confidence = round(min(52 + edge_score + signal_bonus - over_penalty, 80))
            # Driver-aware narrative: cite top signals when available
            if signals:
                driver_summary = "; ".join(s["detail"] for s in signals[:2])
                narrative = (
                    f"Model projects {proj_val:.1f} {stat_type} ({direction} {line:.1f}). "
                    f"{driver_summary}."
                )
            else:
                narrative = (
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
                "odds_over": book_odds["odds_over"] if book_odds else None,
                "odds_under": book_odds["odds_under"] if book_odds else None,
                "books_consensus": book_odds["books_consensus"] if book_odds else 0,
                "model_only": not bool(book_odds), "signals": signals, "narrative": narrative,
                "season_avg": round(season_val, 1),
                "proj_min": round(pred_min, 1),
                "avg_min": round(avg_min, 1) if isinstance(avg_min, (int, float)) else 0,
                "game_time": gctx.get("game_time", ""),
                "game_start_iso": gctx.get("game_start_iso", ""),
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
                lr_book   = _lookup_player_odds(player_odds_map, p.get("name", ""), stat_type)
                line      = lr_book["line"] if lr_book else round(round(season_val * 2) / 2, 1)
                edge      = round(proj_val - line, 1)
                if edge == 0:
                    continue
                direction = "over" if edge > 0 else "under"
                if direction == "over" and over_pick:
                    continue
                if direction == "under" and under_pick:
                    continue
                recent_val = p.get(recent_field, proj_val)
                lr_signals, _ = _generate_signals(
                    p, gctx, direction, stat_type, season_val, recent_val, proj_val, line, cfg
                )
                if lr_signals:
                    driver_summary = "; ".join(s["detail"] for s in lr_signals[:2])
                    lr_narrative = (
                        f"Model projects {proj_val:.1f} {stat_type} ({direction} {line:.1f}). "
                        f"{driver_summary}."
                    )
                else:
                    lr_narrative = f"Model projects {proj_val:.1f} {stat_type} vs the {line:.1f} baseline."
                avg_min = p.get("season_min", p.get("min", 0))
                last_resort.append({
                    "player_name": p["name"], "player_id": p.get("id", ""),
                    "team": team_abbr, "opponent": gctx.get("opponent", ""),
                    "stat_type": stat_type, "line": line, "direction": direction,
                    "projection": proj_val, "edge": edge, "confidence": 52,
                    "odds_over": lr_book["odds_over"] if lr_book else None,
                    "odds_under": lr_book["odds_under"] if lr_book else None,
                    "books_consensus": lr_book["books_consensus"] if lr_book else 0,
                    "model_only": not bool(lr_book), "signals": lr_signals,
                    "narrative": lr_narrative,
                    "season_avg": round(season_val, 1),
                    "proj_min": round(pred_min, 1),
                    "avg_min": round(avg_min, 1) if isinstance(avg_min, (int, float)) else 0,
                    "game_time": gctx.get("game_time", ""),
                    "game_start_iso": gctx.get("game_start_iso", ""),
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
        "pick": primary, "over_pick": over_pick, "under_pick": under_pick,
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
            if "game_start_iso" not in pick or not pick.get("game_start_iso"):
                pick["game_start_iso"] = game_ctx_map.get(team, {}).get("game_start_iso", "")
            if "recent_form_bars" not in pick or not pick.get("recent_form_bars"):
                season_val = p.get(season_field, p.get(meta["field"], 0)) or 0.01
                recent_val = p.get(recent_field, pick.get("projection", 0))
                ratio = min(1.2, recent_val / max(season_val, 0.01))
                pick["recent_form_bars"] = [round(ratio, 2)] * 5
            break


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_line_engine(projections, games, line_config=None, player_odds_map=None, news_context="", dvp_data=None):
    """
    Main entry point. Uses Claude Haiku to reason about ESPN projection data
    and pick today's best player prop edge. Falls back to algorithmic model
    if Claude API is unavailable.
    line_config: optional dict (e.g. from model-config "line" section) with min_confidence, min_edge_pct.
    player_odds_map: optional pre-fetched Odds API lines {(name_lower, stat_type): {...}}.
    news_context: optional string of recent NBA news from web search (Layer 1).
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
    stat_floors = (line_config or {}).get("stat_floors", {})

    if not ANTHROPIC_API_KEY:
        return run_model_fallback(projections, games, line_config, player_odds_map)

    over_data, under_data = _run_parallel_claude(projections, games, stat_floors, player_odds_map, news_context, dvp_data=dvp_data)

    if not over_data and not under_data:
        print("[LineEngine] Claude returned no picks — using algorithmic fallback")
        return run_model_fallback(projections, games, line_config, player_odds_map)

    def _build_pick(pd):
        if not pd:
            return None
        signals = [
            {"type": s.lower().split(":")[0].strip().replace(" ", "_"), "detail": s}
            for s in pd.get("signals", [])
        ]
        pick = {
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
        # Override line/edge with authoritative bookmaker data when available.
        # Claude is already instructed to use the [book: X.X] annotation, but
        # we enforce it here as a safety net in case Claude returns a different value.
        book = _lookup_player_odds(player_odds_map, pd.get("player_name", ""), pd.get("stat_type", "points"))
        if book:
            pick["line"]            = book["line"]
            pick["edge"]            = round(pd.get("projection", 0) - book["line"], 1)
            pick["odds_over"]       = book["odds_over"]
            pick["odds_under"]      = book["odds_under"]
            pick["books_consensus"] = book["books_consensus"]
            pick["model_only"]      = False
        return pick

    over_pick  = _build_pick(over_data)
    under_pick = _build_pick(under_data)

    # Enforce min_confidence: reject Claude picks below threshold
    if over_pick and over_pick.get("confidence", 0) < min_confidence:
        over_pick = None
    if under_pick and under_pick.get("confidence", 0) < min_confidence:
        under_pick = None

    # Fill missing direction from algorithmic fallback
    if not over_pick or not under_pick:
        fallback = run_model_fallback(projections, games, line_config, player_odds_map)
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
        "slate_summary": {
            "games_evaluated": len(games),
            "props_scanned":   len(projections),
            "edges_found":     (1 if over_pick else 0) + (1 if under_pick else 0),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "model_only":      True,
        },
    }
