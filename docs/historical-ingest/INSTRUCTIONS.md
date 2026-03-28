# Historical ingest — three steps

## 1. Rasterize the PDFs

- Put source PDFs in this folder (or keep them wherever you work).
- Export **one PNG per page** (~200 DPI is enough). Tools: macOS Preview (Export), `pdftoppm -png`, ImageMagick `convert`, etc.
- Use clear filenames if it helps (e.g. `pt1-page-01.png`).

---

## 2. Read the PNGs

- Open each PNG and transcribe what’s on screen into structured data (JSON or CSV-ready).
- Figure out **slate date** (`YYYY-MM-DD`) from the UI (e.g. “Oct 24”, caption under the image).

**Include all of these (do not skip Games pages):**

| Screen | What to capture |
|--------|------------------|
| **Games / scoreboard** | Each **Final** game: both teams, final scores, who won. Prefer **NBA 3-letter abbreviations** (ESPN-style: `GS`, `NY`, `NO`, etc.) even if the app shows nicknames. Note `Final/OT` or `Final/2OT` in a field if you add one, or only include truly final lines. |
| Most popular | Full list + drafts / RS / boost / avg as shown |
| Most drafted 3x | Same style as most popular |
| Highest value | Top-performer rows |
| Winning lineups leaderboard | Flat rows per winner × 5 slots |

**Skip:** blank pages only.

Player / CSV column details: **`docs/HISTORICAL_DATA.md`**.

Slate results JSON shape (must match other files in `data/slate_results/`): see step 3.

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
- After adding/updating JSON files, optionally rebuild the flat rollup:

```bash
python scripts/fetch_slate_results_espn.py --manifest-only --start 2025-10-21 --end 2026-03-27
```

(Adjust `--start` / `--end` to cover the dates you touched; this only merges existing JSON → `data/slate_results/regular_season_games_flat.csv`, no HTTP.)

### Other datasets

If you use the top-performers mega file:

```bash
python scripts/rebuild_top_performers_mega.py
```

Then commit and push `data/` as usual.

**Reference:** `docs/HISTORICAL_DATA.md` for ingest shapes and columns; existing `data/slate_results/*.json` for score file examples.
