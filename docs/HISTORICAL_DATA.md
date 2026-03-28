# Historical data (developer ingestion)

This season, **Log** and **audit** treat `data/top_performers.csv` as the **primary** source of per-player outcomes (filtered by `date`). Legacy `data/actuals/{date}.csv` is still read when a date has no rows in the mega file.

**File-only ingest (no API):** **`docs/historical-ingest/INSTRUCTIONS.md`** — where to put CSV/JSON, including **`data/slate_results/`** finals.

Six on-disk historical structures (mega rollup + per-date files):

| Dataset | Path | Role |
|--------|------|------|
| Top performers (mega) | `data/top_performers.csv` | Rolled-up rows: `date`, `player_name`, **`team`** (NBA abbr), `actual_rs`, `actual_card_boost`, `drafts`, `avg_finish`, `total_value`, `source` |
| Most popular | `data/most_popular/{date}.csv` | Full list: `player`, **`team`**, `draft_count`, `actual_rs`, `actual_card_boost`, … (same `team` idea; column is `player` not `player_name`) |
| Most drafted (high boost) | `data/most_drafted_3x/{date}.csv` | Same columns as most popular (`player`, `team`, …) |
| Winning drafts | `data/winning_drafts/{date}.csv` | Long format: `winner_rank`, `drafter_label`, `total_score`, `slot_index`, `player_name`, **`team`**, `actual_rs`, `slot_mult`, `card_boost`, `saved_at` |
| Per-day actuals (legacy) | `data/actuals/{date}.csv` | Same shape as mega minus `date`: `player_name`, **`team`**, `actual_rs`, … |
| Slate results | `data/slate_results/{date}.json` | **Regular-season** finals only: `game_count`, `games[]` each with `home`, `away`, `home_score`, `away_score`, `winner`, `loser`, `winner_score`, `loser_score`, plus `season_stage`, `source` (often `espn_scoreboard_api`; manual screenshot fills may use `screenshot_ingest`). One file **per calendar day**; off-days use `game_count: 0`. Flattened copy: `data/slate_results/regular_season_games_flat.csv`. **Abbreviations follow ESPN** (`NO`, `NY`, `GS`, …). ESPN fetch: `python scripts/fetch_slate_results_espn.py`; flatten JSON only (no HTTP): same script with `--manifest-only`. **Not wired into live `/api` yet** — analytics / future features. |

**Team column:** Use 3-letter NBA abbr when known. Parser prompts ask Haiku for `team` on `actuals` / `top_performers` / `winning_drafts`. Backfill: `python scripts/migrate_historical_add_team.py` (fills blanks from `data/predictions/{date}.csv` when player names match). Legacy CSVs without `team` still load via `_parse_actuals_rows` / `_parse_top_performers_mega_rows`.

### Team audit (new ingestions only)

After each new ingest date, run a **date-scoped** team audit (do not run full-history unless needed):

```bash
# Example for one new ingest date
python scripts/audit_backfill_teams.py --dates 2026-03-25
python scripts/fix_team_consistency_pass2.py --dates 2026-03-25

# Multiple new dates at once (comma-separated)
python scripts/audit_backfill_teams.py --dates 2026-03-25,2026-03-26
python scripts/fix_team_consistency_pass2.py --dates 2026-03-25,2026-03-26
```

What this does (per date):
- normalizes team aliases to canonical app abbreviations (`GS`, `NY`, `SA`, `NO`, `UTAH`, `WSH`)
- backfills missing teams from predictions, same-date historical rows, and ESPN boxscores
- handles mid-season trades correctly by resolving teams **by date**
- cleans blank-player ingestion artifacts in `most_popular` / `most_drafted_3x`

After updating `data/actuals/{date}.csv`, run `python scripts/rebuild_top_performers_mega.py` so `top_performers.csv` stays in sync.

Legacy `data/ownership/{date}.csv` is still readable for calibrate-boost; new writes use **`POST /api/save-most-popular`** or **`POST /api/save-ownership`** (alias → `data/most_popular/`).

## Optional auth

When `INGEST_SECRET` is set in the environment, these endpoints require header **`X-Ingest-Key: <secret>`** or **`Authorization: Bearer <secret>`**:

- `POST /api/save-most-popular`
- `POST /api/save-ownership` (alias)
- `POST /api/save-most-drafted-3x`
- `POST /api/save-winning-drafts`

`POST /api/parse-screenshot` stays usable without the secret (rate-limited) for local OCR; lock down Railway with network rules if needed.

## `screenshot_type` values (`POST /api/parse-screenshot`)

- `most_drafted` / `most_popular` — Most popular list
- `most_drafted_high_boost` — High-boost sub-leaderboard
- `top_performers` — Highest-value rows only
- `winning_drafts` — Up to four winning lineups (flat JSON)
- `actuals` (default) — Broad My Draft + Highest value + leaderboard heuristics

## Example: parse + save most popular

```bash
curl -sS -X POST "$BASE/api/parse-screenshot" \
  -F "file=@screenshot.jpg" \
  -F "screenshot_type=most_popular" | tee /tmp/parsed.json

# Then POST players array (adjust jq to your shape):
curl -sS -X POST "$BASE/api/save-most-popular" \
  -H "Content-Type: application/json" \
  -H "X-Ingest-Key: $INGEST_SECRET" \
  -d "{\"date\":\"2026-03-20\",\"players\":$(jq '.players' /tmp/parsed.json)}"
```

## Example: winning drafts

```bash
curl -sS -X POST "$BASE/api/save-winning-drafts" \
  -H "Content-Type: application/json" \
  -H "X-Ingest-Key: $INGEST_SECRET" \
  -d '{"date":"2026-03-20","rows":[
    {"winner_rank":1,"drafter_label":"user1","total_score":71.2,"slot_index":1,"player_name":"A","actual_rs":5.1,"slot_mult":2.0,"card_boost":1.2}
  ]}'
```

## Verification

```bash
python scripts/verify_historical_datasets.py
python scripts/verify_top_performers.py   # drafts model overlap vs predictions
```

## Training

- `train_drafts_lgbm.py` — labels from `top_performers.csv`, `data/actuals/`, and `data/most_popular/`.
- `train_boost_lgbm.py` — adds `data/most_popular/` to the top_performers + actuals union.

User-facing Ben **screenshot upload banner is removed** this season; ingestion is **script/curl only** until a future UI return.

**Related:** `POST /api/save-actuals` still merges into `data/actuals/{date}.csv` for admin repair; `POST /api/lab/skip-uploads` makes `save-actuals` a no-op for a date.
