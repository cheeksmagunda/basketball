"""DFS salary CSV popularity proxy for card boost adjustment.

Ingests DraftKings/FanDuel daily salary CSVs to estimate Real Sports draft
popularity. Players with the highest DFS salaries + hype are most likely to be
"most-drafted" on Real Sports, meaning their card boosts will be depressed.

Pipeline:
  1. CSV saved via POST /api/save-dfs-salaries → data/dfs_salaries/{date}.csv
  2. At inference, load today's (or most recent) DFS salary snapshot
  3. Compute a 0-1 popularity_score per player:
     - Raw score from salary rank + salary percentile
     - Adjusted for slate size and multi-position eligibility
  4. Feed into _est_card_boost as a bounded anti-popularity penalty term

grep: DFS SALARY FEED
"""

import csv
import io
import json
import re
import time
from pathlib import Path
from typing import Optional


CACHE_DIR = Path("/tmp/nba_cache_v19")
DATA_DIR = Path("data/dfs_salaries")
_CACHE_TTL = 7200  # 2 hours


def _normalize_name(name: str) -> str:
    n = str(name).lower().strip()
    n = re.sub(r"['\.\-]", "", n)
    n = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", n)
    return re.sub(r"\s+", " ", n).strip()


def _cache_path(key: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"dfs_{key}.json"


def _read_cache(key: str) -> Optional[dict]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
        if time.time() - data.get("_ts", 0) > _CACHE_TTL:
            return None
        return data
    except Exception:
        return None


def _write_cache(key: str, data: dict):
    try:
        data["_ts"] = time.time()
        _cache_path(key).write_text(json.dumps(data, default=str))
    except Exception:
        pass


def save_dfs_salaries(date_str: str, csv_content: str, platform: str = "draftkings") -> dict:
    """Parse and save a DFS salary CSV. Returns parsed player count.

    Expected CSV formats:
      DraftKings: Name, Position, Salary, GameInfo, AvgPointsPerGame, ...
      FanDuel:    Nickname, Position, Salary, Game, FPPG, ...

    Saves to data/dfs_salaries/{date}.csv (GitHub-compatible flat format).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    players = _parse_dfs_csv(csv_content, platform)
    if not players:
        return {"error": "no_players_parsed", "count": 0}

    out_path = DATA_DIR / f"{date_str}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["player_name", "position", "salary", "avg_fpts", "platform"])
        w.writeheader()
        for p in players:
            w.writerow(p)

    # Bust cache
    _write_cache(f"salaries_{date_str}", {"players": players, "platform": platform})

    return {"status": "saved", "count": len(players), "path": str(out_path)}


def _parse_dfs_csv(csv_content: str, platform: str) -> list:
    """Parse DFS salary CSV into normalized player list."""
    players = []
    reader = csv.DictReader(io.StringIO(csv_content))

    for row in reader:
        if platform == "fanduel":
            name = row.get("Nickname") or row.get("Player") or row.get("Name") or ""
            salary = row.get("Salary") or "0"
            pos = row.get("Position") or ""
            fpts = row.get("FPPG") or row.get("Avg Points") or "0"
        else:  # draftkings default
            name = row.get("Name") or row.get("Player") or ""
            salary = row.get("Salary") or row.get("DK Salary") or "0"
            pos = row.get("Position") or row.get("Roster Position") or ""
            fpts = row.get("AvgPointsPerGame") or row.get("FPPG") or "0"

        name = name.strip()
        if not name:
            continue

        try:
            salary_val = int(str(salary).replace("$", "").replace(",", "").strip())
        except (ValueError, TypeError):
            salary_val = 0

        try:
            fpts_val = float(str(fpts).strip())
        except (ValueError, TypeError):
            fpts_val = 0.0

        players.append({
            "player_name": name,
            "position": pos.strip(),
            "salary": salary_val,
            "avg_fpts": round(fpts_val, 1),
            "platform": platform,
        })

    return players


def load_dfs_salaries(date_str: str = None) -> list:
    """Load DFS salary data for a date. Checks cache, then filesystem, then GitHub."""
    if date_str:
        cached = _read_cache(f"salaries_{date_str}")
        if cached and "players" in cached:
            return cached["players"]

    # Try loading from local file
    if date_str:
        paths_to_try = [
            DATA_DIR / f"{date_str}.csv",
            Path(__file__).parent.parent / "data" / "dfs_salaries" / f"{date_str}.csv",
        ]
        for p in paths_to_try:
            if p.exists():
                try:
                    with p.open("r", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        players = []
                        for row in reader:
                            players.append({
                                "player_name": row.get("player_name", ""),
                                "position": row.get("position", ""),
                                "salary": int(row.get("salary", 0)),
                                "avg_fpts": float(row.get("avg_fpts", 0)),
                                "platform": row.get("platform", ""),
                            })
                        if players:
                            _write_cache(f"salaries_{date_str}", {"players": players})
                            return players
                except Exception:
                    continue

    return []


def compute_popularity_scores(date_str: str = None) -> dict:
    """Compute per-player popularity scores from DFS salaries.

    Returns {normalized_name: {
        "popularity_score": float (0-1),  # 1 = most popular
        "salary": int,
        "salary_pct": float,  # percentile among all players
        "anti_pop_adjustment": float,  # negative boost adjustment for popular players
    }}
    """
    players = load_dfs_salaries(date_str)
    if not players:
        return {}

    cache_key = f"pop_scores_{date_str or 'today'}"
    cached = _read_cache(cache_key)
    if cached and "scores" in cached:
        return cached["scores"]

    salaries = [p["salary"] for p in players if p["salary"] > 0]
    if not salaries:
        return {}

    max_salary = max(salaries)
    min_salary = min(salaries)
    salary_range = max_salary - min_salary if max_salary > min_salary else 1

    sorted_salaries = sorted(salaries)
    n = len(sorted_salaries)

    scores = {}
    for p in players:
        name = p["player_name"]
        norm = _normalize_name(name)
        sal = p["salary"]
        if sal <= 0:
            continue

        # Salary percentile (0-1)
        rank = sum(1 for s in sorted_salaries if s <= sal)
        pct = rank / n

        # Normalized salary (0-1)
        sal_norm = (sal - min_salary) / salary_range

        # Combined popularity score: blend of percentile rank and raw salary position
        pop_score = 0.6 * pct + 0.4 * sal_norm

        # Anti-popularity adjustment: popular players get negative boost modifier
        # Scale: top 10% get -0.3 to -0.5, bottom 50% get +0.1 to +0.15
        if pop_score > 0.9:
            adjustment = -(pop_score - 0.5) * 0.8  # -0.32 to -0.40
        elif pop_score > 0.75:
            adjustment = -(pop_score - 0.5) * 0.5  # -0.13 to -0.20
        elif pop_score > 0.5:
            adjustment = 0.0  # neutral zone
        else:
            adjustment = (0.5 - pop_score) * 0.3    # +0.0 to +0.15

        scores[norm] = {
            "popularity_score": round(pop_score, 3),
            "salary": sal,
            "salary_pct": round(pct, 3),
            "anti_pop_adjustment": round(adjustment, 3),
        }

    if scores:
        _write_cache(cache_key, {"scores": scores})
    return scores


def get_anti_popularity_adjustment(
    player_name: str,
    date_str: str = None,
    max_penalty: float = 0.5,
    max_bonus: float = 0.2,
) -> float:
    """Get the DFS-salary-based anti-popularity boost adjustment for a player.

    Positive = boost up (unpopular, contrarian pick)
    Negative = boost down (popular, high-salary, likely most-drafted)

    Returns 0.0 when no DFS data is available (graceful fallback).
    """
    scores = compute_popularity_scores(date_str)
    if not scores:
        return 0.0

    norm = _normalize_name(player_name)
    info = scores.get(norm)
    if not info:
        return 0.0

    adj = info["anti_pop_adjustment"]
    return max(-max_penalty, min(max_bonus, adj))
