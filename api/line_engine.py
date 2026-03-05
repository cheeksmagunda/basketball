# ─────────────────────────────────────────────────────────────────────────────
# LINE ENGINE — Daily Prop Edge Detection for The Oracle
#
# Pipeline:
#   1. Fetch player props from The Odds API (5 stat categories)
#   2. Match bookmaker names to ESPN player projections
#   3. Compute raw edge = projection - consensus_line
#   4. Score confidence: edge (40%) + signals (30%) + stability (20%) + sharpness (10%)
#   5. Filter invalid picks, rank by confidence
#   6. Build template narrative
#   7. Return pick + runner-up + slate summary
# ─────────────────────────────────────────────────────────────────────────────

import os
import re
import requests
from datetime import datetime, timezone

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"

# Stat categories to fetch from Odds API
PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_steals",
    "player_blocks",
]

# Human-readable stat labels
STAT_LABEL = {
    "player_points":   "points",
    "player_rebounds": "rebounds",
    "player_assists":  "assists",
    "player_steals":   "steals",
    "player_blocks":   "blocks",
}

# Minimum edge required to consider a prop (discard below)
MIN_EDGE = {
    "player_points":   2.5,
    "player_rebounds": 1.5,
    "player_assists":  1.5,
    "player_steals":   0.5,
    "player_blocks":   0.5,
}

# Normalization divisor for edge component of confidence score
EDGE_DIVISOR = {
    "player_points":   5.0,
    "player_rebounds": 3.0,
    "player_assists":  3.0,
    "player_steals":   1.5,
    "player_blocks":   1.5,
}

# Which player projection key each market maps to
PROJ_KEY = {
    "player_points":   "pts",
    "player_rebounds": "reb",
    "player_assists":  "ast",
    "player_steals":   "stl",
    "player_blocks":   "blk",
}

# ─────────────────────────────────────────────────────────────────────────────
# ODDS API HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _odds_get(url, params):
    """GET request to Odds API. Returns parsed JSON or None on failure."""
    try:
        r = requests.get(url, params={**params, "apiKey": ODDS_API_KEY}, timeout=15)
        if not r.ok:
            print(f"[LineEngine] Odds API error: HTTP {r.status_code} {r.reason} — {r.text[:300]}")
            return None
        return r.json()
    except Exception as e:
        print(f"[LineEngine] Odds API error: {e}")
        return None


def fetch_nba_events():
    """Get today's NBA game event IDs from The Odds API."""
    data = _odds_get(
        f"{ODDS_API_BASE}/odds/",
        {"regions": "us", "markets": "h2h", "oddsFormat": "american"},
    )
    if not data:
        return []
    return [
        {
            "event_id":    ev["id"],
            "home_team":   ev.get("home_team", ""),
            "away_team":   ev.get("away_team", ""),
            "commence_time": ev.get("commence_time", ""),
        }
        for ev in data
    ]


def fetch_event_props(event_id):
    """Fetch player props for a single NBA event. Returns raw bookmaker data."""
    markets = ",".join(PROP_MARKETS)
    data = _odds_get(
        f"{ODDS_API_BASE}/events/{event_id}/odds",
        {"regions": "us", "markets": markets, "oddsFormat": "american"},
    )
    return data or {}


def build_prop_map(events_with_props):
    """
    Aggregate bookmaker lines into consensus props.
    Returns: {(player_name_lower, market): {"line": float, "over_odds": int, "under_odds": int, "books": int}}
    """
    props = {}
    for event_data in events_with_props:
        for bookmaker in event_data.get("bookmakers", []):
            for market in bookmaker.get("markets", []):
                market_key = market.get("key", "")
                if market_key not in PROP_MARKETS:
                    continue
                for outcome in market.get("outcomes", []):
                    player = outcome.get("description", "").lower().strip()
                    side = outcome.get("name", "").lower()  # "over" or "under"
                    point = outcome.get("point")
                    price = outcome.get("price")
                    if not player or point is None:
                        continue
                    key = (player, market_key)
                    if key not in props:
                        props[key] = {"lines": [], "over_odds": [], "under_odds": [], "books": set()}
                    props[key]["lines"].append(float(point))
                    props[key]["books"].add(bookmaker["key"])
                    if "over" in side:
                        props[key]["over_odds"].append(price)
                    elif "under" in side:
                        props[key]["under_odds"].append(price)

    # Compute consensus (median line, most-common over/under odds)
    result = {}
    for (player, market), data in props.items():
        if not data["lines"]:
            continue
        lines = sorted(data["lines"])
        mid = len(lines) // 2
        consensus_line = lines[mid] if len(lines) % 2 else (lines[mid-1] + lines[mid]) / 2
        over_odds  = round(sum(data["over_odds"])  / len(data["over_odds"]))  if data["over_odds"]  else -110
        under_odds = round(sum(data["under_odds"]) / len(data["under_odds"])) if data["under_odds"] else -110
        result[(player, market)] = {
            "line":       round(consensus_line, 1),
            "over_odds":  over_odds,
            "under_odds": under_odds,
            "books":      len(data["books"]),
            "line_spread": max(lines) - min(lines),  # disagreement between books
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PLAYER NAME MATCHING
# ─────────────────────────────────────────────────────────────────────────────

_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b\.?", re.IGNORECASE)
_PUNCT = re.compile(r"[.\-']")

def _normalize_name(name):
    """Normalize a player name for fuzzy matching."""
    name = _PUNCT.sub("", name).lower().strip()
    name = _SUFFIXES.sub("", name).strip()
    return " ".join(name.split())

def _match_player(odds_name, espn_players):
    """Match a bookmaker player name to an ESPN player dict.
    Returns the matching player dict or None.
    espn_players: list of dicts with 'name' and 'team' keys.
    """
    odds_norm = _normalize_name(odds_name)
    odds_parts = odds_norm.split()
    odds_last  = odds_parts[-1] if odds_parts else ""
    odds_first_init = odds_parts[0][0] if odds_parts else ""

    for p in espn_players:
        espn_norm  = _normalize_name(p["name"])
        espn_parts = espn_norm.split()
        espn_last  = espn_parts[-1] if espn_parts else ""
        espn_first = espn_parts[0]  if espn_parts else ""

        # 1. Exact match
        if odds_norm == espn_norm:
            return p

        # 2. Last name + first initial
        if (odds_last == espn_last and odds_first_init and
                espn_first and odds_first_init == espn_first[0]):
            return p

        # 3. Last name only (riskier, only if single result — handled by caller)
        if odds_last == espn_last and len(odds_parts) == 1:
            return p  # best effort

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_signals(player_proj, game, market, prop_meta):
    """
    Detect positive signals for a prop pick. Returns (score 0-100, list of signal dicts).
    Signals:
      - cascade: player received injury cascade bonus (+25)
      - opp_b2b: opponent on 2nd night of B2B (+20)
      - game_script: favorable game script for this stat (+20)
      - recent_form: recent trend aligned with edge direction (+20)
      - spread_safety: close game, full minutes expected (+15)
    """
    signals = []
    score = 0
    stat_key = PROJ_KEY.get(market, "pts")
    direction = prop_meta.get("direction", "over")

    # Cascade bonus
    cascade_bonus = player_proj.get("_cascade_bonus", 0)
    if cascade_bonus > 0:
        signals.append({"type": "cascade", "detail": f"Injury upgrade — +{cascade_bonus:.1f} projected minutes"})
        score += 25

    # Opponent B2B
    opp_is_b2b = game.get("opp_b2b", False)
    if opp_is_b2b:
        signals.append({"type": "opp_b2b", "detail": f"Opponent on second night of a back-to-back"})
        score += 20

    # Game script — does the game environment favor this stat?
    total  = game.get("total", 222)
    spread = game.get("spread", 0)
    try:
        from api.index import _game_script_weights
    except ImportError:
        try:
            from .index import _game_script_weights
        except ImportError:
            _game_script_weights = None

    if _game_script_weights:
        gw = _game_script_weights(total, spread)
        stat_weight = gw.get(stat_key, 1.0)
        if (direction == "over" and stat_weight >= 1.10) or (direction == "under" and stat_weight <= 0.90):
            try:
                try:
                    from api.index import _game_script_label
                except ImportError:
                    from .index import _game_script_label
                label = _game_script_label(total)
            except Exception:
                label = "current environment"
            signals.append({"type": "game_script", "detail": f"{label} (O/U {total}) favors {stat_key} ({stat_weight:.2f}x)"})
            score += 20

    # Recent form alignment
    recent_stat  = player_proj.get(f"recent_{stat_key}", player_proj.get(stat_key, 0))
    season_stat  = player_proj.get(f"season_{stat_key}", player_proj.get(stat_key, 0))
    proj_stat    = player_proj.get(stat_key, 0)
    if season_stat and recent_stat:
        trending_up = recent_stat > season_stat
        if (direction == "over" and trending_up) or (direction == "under" and not trending_up):
            signals.append({"type": "recent_form", "detail": f"Trending {'up' if trending_up else 'down'}: {recent_stat:.1f} recent vs {season_stat:.1f} season"})
            score += 20

    # Spread safety — close game means starters play full minutes
    abs_spread = abs(spread or 0)
    if abs_spread < 7:
        signals.append({"type": "spread_safety", "detail": f"Close game expected (spread {spread:+.1f}) — full minutes likely"})
        score += 15

    return min(score, 100), signals


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE SCORING
# ─────────────────────────────────────────────────────────────────────────────

def _score_confidence(market, raw_edge, player_proj, game, prop_meta):
    """
    Composite confidence score 0-100.
    edge: 40% | signals: 30% | stability: 20% | sharpness: 10%
    """
    # 1. Normalized edge (40%)
    divisor = EDGE_DIVISOR.get(market, 3.0)
    edge_score = min(abs(raw_edge) / divisor, 1.0) * 100

    # 2. Signal stack (30%)
    signal_score, signals = _detect_signals(player_proj, game, market, prop_meta)

    # 3. Projection stability (20%)
    stat_key = PROJ_KEY.get(market, "pts")
    season_stat = player_proj.get(f"season_{stat_key}", player_proj.get(stat_key, 1))
    recent_stat = player_proj.get(f"recent_{stat_key}", player_proj.get(stat_key, 1))
    stability = 1.0 - abs(recent_stat - season_stat) / max(abs(season_stat), 1.0)
    stability_score = max(0.0, min(1.0, stability)) * 100

    # 4. Line sharpness (10%) — how much do books disagree?
    spread = prop_meta.get("line_spread", 0)
    books  = prop_meta.get("books", 1)
    if books <= 1:
        sharpness = 20
    elif spread == 0:
        sharpness = 30
    elif spread <= 1.0:
        sharpness = 60
    else:
        sharpness = 100

    confidence = (
        edge_score   * 0.40 +
        signal_score * 0.30 +
        stability_score * 0.20 +
        sharpness * 0.10
    )
    return round(min(confidence, 99), 1), signals


# ─────────────────────────────────────────────────────────────────────────────
# NARRATIVE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def _build_narrative(player_name, team, opponent, market, line, direction,
                     projection, edge, confidence, signals):
    """Template-based narrative. No LLM."""
    stat = STAT_LABEL.get(market, market)
    unit = "pt" if market == "player_points" else stat[:-1] if stat.endswith("s") else stat

    # Top signals (up to 3)
    signal_texts = []
    for s in signals[:3]:
        t = s["type"]
        d = s["detail"]
        signal_texts.append(d)

    signal_str = ". ".join(signal_texts) + ("." if signal_texts else "")
    direction_upper = direction.upper()

    narrative = (
        f"{direction_upper} {line} {stat} for {player_name} ({team} vs {opponent}). "
        f"{signal_str} "
        f"Model projects {projection:.1f} {stat} — a {abs(edge):.1f}-{unit} edge at {confidence:.0f}% confidence."
    ).strip()
    return narrative


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_line_engine(projections, games):
    """
    Main entry point. Accepts:
      projections: flat list of player projection dicts (from /api/slate pipeline)
      games: list of game dicts (from fetch_games())
    Returns: {pick, runner_up, slate_summary} or {error: "..."}
    """
    if not ODDS_API_KEY:
        return {"pick": None, "error": "no_api_key"}
    if not games:
        return {"pick": None, "error": "no_games"}

    # Build ESPN player lookup: lower(name) -> proj
    proj_by_name = {}
    for p in projections:
        proj_by_name[_normalize_name(p["name"])] = p

    # Build game lookup: team name -> game info
    game_by_team = {}
    for g in games:
        home_name = g["home"]["name"].lower()
        away_name = g["away"]["name"].lower()
        game_by_team[home_name] = {**g, "side": "home", "opp_b2b": g.get("away_b2b", False)}
        game_by_team[away_name] = {**g, "side": "away", "opp_b2b": g.get("home_b2b", False)}
        # Also add abbr mapping
        game_by_team[g["home"]["abbr"].lower()] = game_by_team[home_name]
        game_by_team[g["away"]["abbr"].lower()] = game_by_team[away_name]

    # Fetch events from Odds API
    odds_events = fetch_nba_events()
    if not odds_events:
        return {"pick": None, "error": "odds_unavailable"}

    # Fetch props for each event (rate-limit budget: 1 call per event)
    events_with_props = []
    props_scanned = 0
    for ev in odds_events:
        event_data = fetch_event_props(ev["event_id"])
        if event_data:
            event_data["_meta"] = ev
            events_with_props.append(event_data)

    if not events_with_props:
        return {"pick": None, "error": "odds_unavailable"}

    prop_map = build_prop_map(events_with_props)
    props_scanned = len(prop_map)

    # Score all candidate props
    candidates = []

    for (odds_name, market), prop_meta in prop_map.items():
        line = prop_meta["line"]
        stat_key = PROJ_KEY.get(market, "pts")

        # Match to ESPN projection
        player_proj = None
        if odds_name in proj_by_name:
            player_proj = proj_by_name[odds_name]
        else:
            # Try all ESPN players
            for p in projections:
                matched = _match_player(odds_name, [p])
                if matched:
                    player_proj = matched
                    break

        if not player_proj:
            continue

        # Filter: injuries and min gate
        inj = player_proj.get("injury_status", "")
        if inj in ("GTD", "DTD", "DOUBT"):
            continue
        if player_proj.get("predMin", 0) < 15:
            continue

        # Get projection for this stat
        proj_stat = player_proj.get(stat_key)
        if proj_stat is None:
            continue

        raw_edge = proj_stat - line
        direction = "over" if raw_edge > 0 else "under"
        min_edge_req = MIN_EDGE.get(market, 1.0)
        if abs(raw_edge) < min_edge_req:
            continue

        # Find game context
        team_abbr = player_proj.get("team", "").lower()
        game_ctx  = game_by_team.get(team_abbr) or {}

        # Filter: team favored by 10+
        spread = game_ctx.get("spread") or 0
        side   = game_ctx.get("side", "home")
        player_is_favorite = (side == "home" and spread < -10) or (side == "away" and spread > 10)
        if player_is_favorite:
            continue

        prop_meta_enriched = {**prop_meta, "direction": direction}
        confidence, signals = _score_confidence(market, raw_edge, player_proj, game_ctx, prop_meta_enriched)

        # Build opponent label
        if game_ctx:
            opp_team = game_ctx["away"]["abbr"] if side == "home" else game_ctx["home"]["abbr"]
        else:
            opp_team = "?"

        candidates.append({
            "player_name":     player_proj["name"],
            "player_id":       player_proj.get("id", ""),
            "team":            player_proj.get("team", ""),
            "opponent":        opp_team,
            "stat_type":       STAT_LABEL.get(market, market),
            "market":          market,
            "line":            line,
            "direction":       direction,
            "projection":      round(float(proj_stat), 1),
            "edge":            round(raw_edge, 2),
            "confidence":      confidence,
            "odds_over":       prop_meta["over_odds"],
            "odds_under":      prop_meta["under_odds"],
            "books_consensus": prop_meta["books"],
            "signals":         signals,
            "_game_ctx":       game_ctx,
            "_player_proj":    player_proj,
        })

    if not candidates:
        return {"pick": None, "error": "no_edges",
                "slate_summary": {"games_evaluated": len(odds_events), "props_scanned": props_scanned,
                                  "edges_found": 0, "timestamp": datetime.now(timezone.utc).isoformat()}}

    # Rank by confidence
    candidates.sort(key=lambda c: c["confidence"], reverse=True)

    def _finalize(c):
        """Strip internal keys and build narrative."""
        game_ctx    = c.pop("_game_ctx", {})
        player_proj = c.pop("_player_proj", {})
        c["narrative"] = _build_narrative(
            c["player_name"], c["team"], c["opponent"],
            c["market"], c["line"], c["direction"],
            c["projection"], c["edge"], c["confidence"], c["signals"],
        )
        c.pop("market", None)
        return c

    pick       = _finalize(candidates[0])
    # Runner-up: first candidate from a different player
    runner_up_raw = next((c for c in candidates[1:] if c["player_name"] != candidates[0]["player_name"]), None)
    runner_up  = _finalize(runner_up_raw) if runner_up_raw else None

    return {
        "pick":       pick,
        "runner_up":  runner_up,
        "slate_summary": {
            "games_evaluated": len(odds_events),
            "props_scanned":   props_scanned,
            "edges_found":     len(candidates),
            "timestamp":       datetime.now(timezone.utc).isoformat(),
        },
    }
