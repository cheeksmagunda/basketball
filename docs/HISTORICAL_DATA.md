# Historical data (developer ingestion)

This season, **Log** and **audit** treat `data/top_performers.csv` as the **primary** source of per-player outcomes (filtered by `date`). Legacy `data/actuals/{date}.csv` is still read when a date has no rows in the mega file.

Six on-disk historical structures (mega rollup + per-date files):

| Dataset | Path | Role |
|--------|------|------|
| Top performers (mega) | `data/top_performers.csv` | Rolled-up rows: `date`, `player_name`, **`team`** (NBA abbr), `actual_rs`, `actual_card_boost`, `drafts`, `avg_finish`, `total_value`, `source` |
| Most popular | `data/most_popular/{date}.csv` | Full list: `player`, **`team`**, `draft_count`, `actual_rs`, `actual_card_boost`, … (same `team` idea; column is `player` not `player_name`) |
| Most drafted (high boost) | `data/most_drafted_3x/{date}.csv` | Same columns as most popular (`player`, `team`, …) |
| Winning drafts | `data/winning_drafts/{date}.csv` | Long format: `winner_rank`, `drafter_label`, `total_score`, `slot_index`, `player_name`, **`team`**, `actual_rs`, `slot_mult`, `card_boost`, `saved_at` |
| Per-day actuals (legacy) | `data/actuals/{date}.csv` | Same shape as mega minus `date`: `player_name`, **`team`**, `actual_rs`, … |
| Slate results | `data/slate_results/{date}.json` | That day’s NBA slate: `game_count`, `games[]` with `winner`/`loser` team abbr and final scores (manual or API backfill). **Not wired into live `/api` yet** — for analytics and future model features (pace, blowouts, rest). |

**Team column:** Use 3-letter NBA abbr when known. Parser prompts ask Haiku for `team` on `actuals` / `top_performers` / `winning_drafts`. Backfill: `python scripts/migrate_historical_add_team.py` (fills blanks from `data/predictions/{date}.csv` when player names match). Legacy CSVs without `team` still load via `_parse_actuals_rows` / `_parse_top_performers_mega_rows`.

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
