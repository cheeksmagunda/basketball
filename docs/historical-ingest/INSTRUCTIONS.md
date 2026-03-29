# Historical ingest — three steps

**No server.** You do not need `uvicorn`, Railway, or `parse-screenshot`. Rasterize the PDF, read the PNGs (yourself or in any chat tool), and write CSV/JSON straight into `data/`. The optional commands at the bottom only rebuild rollup files from what you already saved.

## Current coverage gap (priority)

Full coverage map: `docs/HISTORICAL_DATA.md` → Data Coverage table.

## 1. Read the PNGs

- Open each PNG and transcribe what’s on screen into structured data (JSON or CSV-ready).
- Figure out **slate date** (`YYYY-MM-DD`) from the UI (e.g. “Oct 24”, caption under the image).

**Include all of these (do not skip Games pages):**

| Screen | What to capture |
|--------|------------------|
| **Games / scoreboard** | Each **Final** game: both teams, final scores, winner. Use **3-letter NBA abbreviations** (ESPN-style: `GS`, `NY`, `NO`). Only include completed games (including `Final/OT` if shown). |
| Most popular | Full list + drafts / RS / boost / avg as shown |
| Most drafted 3x | Same style as most popular |
| Highest value | Top-performer rows |
| Winning lineups leaderboard | Flat rows per winner × 5 slots |

**Skip:** blank pages only.

Player / CSV column details: **`docs/HISTORICAL_DATA.md`**.

---

## 3. Write the data to the right place

Under repo `data/`:

| Kind of screen | Write to |
|----------------|----------|
| **Games (final scores)** | `data/slate_results/{YYYY-MM-DD}.json` |
| Most popular | `data/most_popular/{YYYY-MM-DD}.csv` |
| Most drafted 3x | `data/most_drafted_3x/{YYYY-MM-DD}.csv` |
| Highest value / top performers | `data/actuals/{YYYY-MM-DD}.csv` (`source` = `highest_value` where applicable) |
| Winning lineups | `data/winning_drafts/{YYYY-MM-DD}.csv` |

### `data/slate_results/{date}.json` (one file per calendar day)

Use the same structure as existing slate result files (regular season, finals only on that screen):

```json
{
  "date": "2025-10-21",
  "game_count": 2,
  "games": [
    {
      "home": "LAL",
      "away": "GS",
      "home_score": 109,
      "away_score": 119,
      "winner": "GS",
      "loser": "LAL",
      "winner_score": 119,
      "loser_score": 109
    }
  ],
  "season_stage": "regular-season",
  "source": "screenshot_ingest",
  "saved_at": "2026-03-27T12:00:00Z"
}
```

- `game_count` = length of `games`.
- Off days or no games on that screenshot: `"game_count": 0`, `"games": []`.

Then **commit and push** `data/` as you normally do.

### Optional: rebuild derived files (local Python only)

| Goal | Command (from repo root) |
|------|---------------------------|
| Flatten slate JSON → `regular_season_games_flat.csv` | `python scripts/fetch_slate_results_espn.py --manifest-only --start YYYY-MM-DD --end YYYY-MM-DD` (range must cover dates you edited; **no network**) |
| Merge `data/actuals/*.csv` → mega | `python scripts/rebuild_top_performers_mega.py` |

**Reference:** `docs/HISTORICAL_DATA.md` (column shapes); copy an existing `data/slate_results/*.json` if you want a template.
