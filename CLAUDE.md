# Basketball — Real Sports Draft Optimizer

**Document Status:** Current Reference

## What This Is

A daily NBA draft optimizer for the **Real Sports** app. Uses a **Strategy-Report-Aligned Architecture**: LightGBM for RS projection, a 3-tier deterministic cascade for card boost prediction, and a sort-based lineup builder using `EV = RS × (2.0 + boost)`. The system follows findings from a 90-date historical analysis showing boost is 40% more valuable per unit than RS, slot assignment is a simple RS sort, and anti-popularity provides a 24% value edge. Deployed on **Railway** as a Dockerized Python (FastAPI) backend + single-page HTML frontend.

## How Real Sports Works

- Users draft 5 NBA players each day
- Each player earns a **Real Score** (RS) based on in-game impact (not just box score stats)
- Each player gets a **Card Boost** set by Real Sports based on their **recent performance** and **how many people drafted them in prior slates** — players who performed well and got heavily drafted see their boost drop; cold or ignored players see it rise
- **Total Value = Real Score × (Slot Multiplier + Card Boost)**
- Slot multipliers: 2.0x, 1.8x, 1.6x, 1.4x, 1.2x (user manually assigns their 5 picks to slots pre-game)
- The winning strategy is drafting **high-RS role players with huge card boosts**, not superstars

## Architecture

```
frontend/              — React + Vite + TypeScript SPA (replaces vanilla JS monolith)
  src/main.tsx         — React entry point (QueryClientProvider + global CSS)
  src/App.tsx          — Shell: Header + TabRouter + BottomNav
  src/types/           — TypeScript interfaces (slate, lab, common)
  src/api/             — React Query hooks: useSlate, useLabBriefing, …
  src/store/           — Zustand state: uiStore (activeTab, …), labStore (messages, system)
  src/hooks/           — useAbortOnTabSwitch, useEventSource, useEtDate, useKeyboardNavHide
  src/components/      — Predict | Ben tabs + shared (PlayerCard, OracleLoader, …)
  src/styles/          — tokens.css, global.css, animations.css (CSS Modules per component)
  vite.config.ts       — Vite: proxy /api → localhost:8080
  Dockerfile           — Multi-stage: Node build (npm run build) → Python runtime (COPY dist)
index.html             — Legacy vanilla JS frontend (fallback when frontend/dist/ absent)
api/index.py           — FastAPI backend (all endpoints, projection engine, Lab)
api/real_score.py      — DFS scoring weights used by draft pipeline; Monte Carlo simulation retained for reference only
api/asset_optimizer.py — MILP lineup optimizer (PuLP, used by per-game pipeline only)
api/rotowire.py        — RotoWire lineup scraper (free tier: availability + injury flags)
data/model-config.json — Runtime model config (Lab writes here; 5-min cache)
data/predictions/      — Git-tracked daily prediction CSVs (via GitHub API)
data/top_performers.csv — **Main historical dataset** (Real Sports leaderboard by date); primary for audit (`_load_player_actuals_for_date`)
data/actuals/          — Legacy per-day CSVs; fallback when a date is absent from the mega file
data/most_popular/     — Per-date most-drafted CSVs (`POST /api/save-most-popular` / `save-ownership` alias)
data/most_drafted_3x/  — Optional high-boost popular slices
data/winning_drafts/   — Optional long-format top-4 winner lineups
data/slate_results/    — Per-date JSON: game_count, final scores by matchup (training / analytics; no save-* API yet)
data/audit/            — Git-tracked daily audit JSONs (auto-generated on save-actuals)
data/slate/            — GitHub-persisted prediction cache ({date}_slate.json, {date}_games.json)
data/locks/            — Cold-start recovery: {date}_slate.json written at lock-promotion time
data/skipped-uploads.json — Dates where `save-actuals` no-ops (optional; `POST /api/lab/skip-uploads`)
lgbm_model.pkl         — LightGBM model bundle {model, features} for Real Score projections
api/boost_model.py     — 3-Tier Cascade Boost Prediction (replaces boost_model.pkl + drafts_model.pkl)
train_lgbm.py          — Training script (12 features, run locally or via GitHub Actions)
scripts/verify_top_performers.py — Backtest drafts + boost vs leaderboard labels + predictions overlap
railway.toml          — Railway config (crons, health check, watchPatterns for deploy)
vercel.json            — Legacy (unused in production; Railway replaced Vercel)
server.py              — Local dev server (uvicorn)
```

## Machine Learning + Cascade Architecture

The backend uses:
1. **`lgbm_model.pkl` (Real Score Projection)**: 12-feature LightGBM model predicting player RS. Blended with heuristic DFS score, compressed, then simple additive game context adjustments applied (+0.3 RS close games, +0.15/10pts pace). No Monte Carlo — removed in v82 strategy simplification. Trained nightly via GitHub Actions.
2. **`api/boost_model.py` (3-Tier Cascade Boost Prediction)**: Deterministic cascade that predicts card boost (0.0–3.0) from historical Real Sports data. **Tier 1** (returning player, ≤14 days): uses prev_boost + adjustments for RS, drafts, trend, gap, mean reversion (MAE ~0.10–0.15). **Tier 2** (stale, >14 days): blends historical boost mean with API-derived estimate. **Tier 3** (cold start): Player Quality Index from season stats. **Anti-popularity penalty** applied post-cascade: estimates draft popularity from season PPG + team market + hot streak; high-popularity players get boost depressed (Finding 4: -0.457 correlation).
3. **Draft Lineup Builder** (`_build_lineups` in `api/index.py`): Condition-Matrix-driven sort-based pipeline. Composite EV: `condition_coeff × RS × (2.0 + boost) × data_multipliers`. The **Condition Matrix** (`CONDITION_MATRIX` in `api/real_score.py`) maps ownership tier × boost tier → historical HV rate (0.0–1.0) and is the PRIMARY signal. Dead capital combos auto-filtered. No positional caps — composition driven by composite EV. Max 1 player per team per lineup. Two gates: RS ≥ 2.0, minutes ≥ 12. Safe lineup = top 5 by safe_ev (cb_low). Upside lineup = top 5 by upside_ev (cb_high). Slot assignment = sort by RS descending (provably optimal).

**Historical outcomes** for audit: **`data/top_performers.csv`** is primary (filter by `date`); **`data/actuals/{date}.csv`** remains a transition fallback. **Simplest ingest:** **`docs/historical-ingest/INSTRUCTIONS.md`** — rasterize PDF → transcribe PNGs → write `data/` (no server). Alternate: **`docs/HISTORICAL_DATA.md`** (API `parse-screenshot` + `save-*` POSTs if you prefer). `data/predictions/` supplies pre-game features for training joins.

**2025-26 data coverage:** Oct 21 – Nov 29 ✅ | Nov 30 – Jan 16 ✅ (gap closed) | Jan 17 – Feb 11 ✅ | Feb 12–18 All-Star break | Feb 19 – Apr 8 ✅. Dec 24 = no games. Full map: `docs/HISTORICAL_DATA.md`.

### Condition-Matrix-Driven Draft Pipeline (Starting 5 + Moonshot)
- **Pipeline**: ESPN → LightGBM + heuristic blend → compression → simple game context (+0.3 close, +pace) → Card Boost (cascade + anti-popularity) → **Condition Matrix** (ownership tier × boost tier → HV rate) → `Composite EV = condition_coeff × RS × (2.0 + boost) × data_multipliers` → Sort-based selection → RS-descending slot assignment
- **Why**: Historical analysis proves format mechanics (ownership × boost interaction) dominate basketball conditions. Ghost players (+max boost) have 100% HV rate; mega-chalk stars have 12%. The Condition Matrix (`api/real_score.py`) is the PRIMARY signal. Dead capital combos (e.g. chalk+low_boost) auto-filtered. No positional caps — composition driven by composite EV. Max 1 player per team per lineup (S5 and Moonshot each independent).
- **Endpoints**: `/api/slate`, `/api/picks`, `/api/force-regenerate`, `/api/injury-check`
- **Code**: `project_player()` and `_build_lineups()` in `api/index.py`, `api/boost_model.py`, `condition_coefficient()` in `api/real_score.py`

## UI Structure

2-tab segmented control navigation (Apple glassmorphism pill style): **Predict | Ben**

- **Predict**: Live slate optimizer (Starting 5 + Moonshot) and per-game analysis ("THE LINE UP" — single 5-player format, no card boost). "Slate-Wide | Game" sub-tabs inline at top of tab. Magic 8-ball loading animation.
- **Ben**: Plain chat with Claude (teal accent). Chat always available. **No in-app historical screenshot upload banner this season** — ingestion is script/curl only (`docs/HISTORICAL_DATA.md`).

### Sub-Nav Tabs (inline, not floating)
`predictSubNav` (Slate-Wide | Game) is an inline `div.predict-sub-nav` element positioned at the top of the Predict tab page. It uses the `.mode-tab` visual language — same height, padding, `border-radius:11px`, Barlow Condensed 800. Active state: predict = chalk blue.

## Codebase Navigation (grep tags)

All major sections in `api/index.py` and `index.html` are tagged for fast searching. In `api/index.py` search for `# grep:`; in `index.html` search for `grep:` (HTML/JS comments). **`server.py`** documents the dev entrypoint via `grep: DEV SERVER` in its docstring.

```
grep: PREDICT TAB              — index.html DOM: tab-predictions, slateList, oracleLoader (logic: SLATE + PER-GAME below)
grep: TEAM_COLORS              — team color hex map in index.html
grep: GLOBAL STATE             — SLATE, PICKS_DATA, LAB state objects
grep: TAB NAVIGATION           — switchTab, movePill, setPillAccent
grep: SLATE                    — loadSlate, /api/slate, Starting 5, Moonshot
grep: PER-GAME ANALYSIS        — runAnalysis, /api/picks
grep: CARD RENDERING           — renderCards, player-card, tcolor
grep: PREDICTION PERSISTENCE   — savePredictions, dedup guard
grep: LAB PAGE                 — initLabPage, LAB state, labCallClaude, buildLabSystemPrompt
grep: HISTORICAL DATA          — TOP_PERFORMERS_GH_PATH, _load_player_actuals_for_date, save-most-popular, winning_drafts, slate_results
grep: PDF INGEST PLAYBOOK      — docs/historical-ingest/INSTRUCTIONS.md (file-only); or rasterize + parse-screenshot + save-* + rebuild mega
grep: DEV SERVER               — server.py, uvicorn, PORT, SPA index catch-all
grep: DATA / TRAINING SCRIPTS  — train_lgbm; scripts/verify_top_performers, verify_historical_datasets, sync_actuals_from_top_performers, rebuild_top_performers_mega, migrate_historical_add_team, fetch_slate_results_espn
grep: github_storage           — _github_get_file, _github_write_file
grep: SLATE CACHE GITHUB       — _slate_cache_to_github, _games_cache_from_github, _bust_slate_cache
grep: CONSTANTS & CACHE        — _cp, _cg, _cs, _lp, _lg, ESPN, MIN_GATE
grep: ESPN DATA FETCHERS       — fetch_games, fetch_roster, _fetch_athlete
grep: INJURY CASCADE           — _cascade_minutes, _pos_group
grep: CASCADE TEAM DETECTOR    — star OUT detection in _run_game, RS mult + boost floor in project_player, relaxed gates in _build_lineups
grep: CARD BOOST               — _est_card_boost, _dfs_score (cascade in api/boost_model.py grep: BOOST CASCADE MODEL)
grep: CONDITION MATRIX         — _build_lineups() condition_coefficient integration, ownership × boost → HV rate (api/real_score.py)
grep: HISTORICAL RS DISCOUNT   — project_player() soft pull-back when predicted RS exceeds player's historical track record
grep: MOMENTUM CURVE           — _build_lineups() hype trap penalty + rising wave bonus detection
grep: MOMENTUM CURVE SCORING   — per-player mc_mult calculation in _build_lineups() scoring loop
grep: GAME SCRIPT              — _game_script_weights, _game_script_label
grep: PLAYER PROJECTION        — project_player, pinfo, rating, est_mult
grep: WEB INTELLIGENCE         — _fetch_nba_news_context, Claude web_search, news_text
grep: LINEUP REVIEW            — _lineup_review_opus, post-lineup Opus, lineup_review
grep: CORE POOL                — _build_lineups candidate pool, safe/upside lineups, draft_ev
grep: GAME RUNNER              — _run_game, _build_lineups, draft_ev
grep: PER-GAME                 — _build_game_lineups, _per_game_strategy, _per_game_adjust
grep: PER_GAME_CONFIG          — per_game config defaults in _CONFIG_DEFAULTS
grep: INJURY CHECK             — /api/injury-check, RotoWire re-check, affected game regeneration
grep: CORE API ENDPOINTS       — /api/games, /api/slate, /api/picks, /api/health, /api/version
grep: BEN / LAB ENGINE         — /api/lab/*, _all_games_final, lab lock
grep: FORCE REGENERATE         — /api/force-regenerate, _force_write_predictions, deploy SHA mismatch, late draft
grep: LOCK HELPERS             — _is_locked, _is_past_lock_window, _et_date
grep: ALL GAMES FINAL          — _all_games_final, ESPN scoreboard poll, midnight rollover, 4.5h fallback
grep: NEXT SLATE DATE          — _find_next_slate_date, multi-day gap, All-Star break
grep: FORCE REGENERATE SYNC    — _force_regenerate_sync, scope=full|remaining
grep: PRODUCTION CACHE         — _TTL_* constants, _CK_* keys, _cp/_cg/_cs, CACHE_DIR
```

## Key Endpoints

### Core
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check for monitoring (config + GitHub reachability) |
| `/api/version` | GET | Build identifier (RAILWAY_GIT_COMMIT_SHA) for deploy checks |
| `/api/slate` | GET | Full-slate predictions (all games). Never returns 5xx; on backend exception returns 200 with `error: "slate_failed"` and empty lineups so the frontend shows "Slate temporarily unavailable" and Retry. |
| `/api/picks?gameId=X` | GET | Per-game predictions |
| `/api/games` | GET | Today's games with lock status |
| `/api/save-predictions` | POST | Save cached predictions to GitHub CSV (deduped — skips commit if unchanged) |
| `/api/parse-screenshot` | POST | Upload Real Sports screenshot, Claude Haiku parses it |
| `/api/save-actuals` | POST | Save parsed actuals to GitHub CSV + auto-generates audit JSON |
| `/api/audit/get?date=X` | GET | Pre-computed accuracy audit for a date (MAE, directional acc, misses) |
| `/api/log/dates` | GET | List dates with stored prediction/actual data (backend only — no frontend Log tab) |
| `/api/log/get?date=X` | GET | Predictions + actuals for a given date, grouped by scope (backend only) |
| `/api/log/actuals-stats?date=X` | GET | ESPN box score stats (PTS, REB, AST, STL, BLK, MIN) for all players on a date's completed games (backend only) |
| `/api/hindsight` | POST | Optimal hindsight lineup from actual RS scores |
| `/api/cold-reset` | GET | Secured global cold reset for slate (CRON_SECRET required) |
| `/api/injury-check` | GET | Cron: check RotoWire for newly OUT/questionable players; regenerate affected games only (requires CRON_SECRET when set) |
| `/api/force-regenerate?scope=X` | GET | **Force-regenerate predictions mid-slate.** `scope=full`: all games (dev deploy/model refresh; CRON_SECRET-gated). `scope=remaining`: only unlocked games (late draft, user-facing). Updates `data/predictions/` CSV and all cache layers. |
| `/api/mae-drift-check` | GET | **Weekly cron** (Monday 6am UTC): compute 7-day rolling MAE, write backend flag if > 2.5 threshold. CRON_SECRET-gated. Returns `{status, computed_mae, triggered, per_date}` |

### Lab (Ben)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/lab/status` | GET | Lock status (locked during slate, unlocked after all games final) |
| `/api/lab/briefing` | GET | Prediction accuracy analysis (MAE, biggest misses, patterns) |
| `/api/lab/update-config` | POST | Apply dot-notation param changes, increment version |
| `/api/lab/config-history` | GET | Full config + changelog |
| `/api/lab/rollback` | POST | Note rollback to target version (new version number) |
| `/api/lab/backtest` | POST | Replay historical slates with proposed params, compare MAE |
| `/api/lab/auto-improve` | GET | **Cron** (daily 9am UTC): briefing → Haiku proposes change → backtest → auto-apply if ≥3% (requires CRON_SECRET when set) |
| `/api/lab/chat-history` | GET | Return persisted daily Ben chat history as array `[{role, content}, ...]` |
| `/api/lab/chat` | POST | Proxy to claude-opus-4-6 with Lab system prompt (keeps key server-side) |
| `/api/lab/skip-uploads` | POST | Append date to `data/skipped-uploads.json` (save-actuals no-op); no in-app UI |
| `/api/lab/calibrate-boost` | GET | Fit card boost params from `data/most_popular/` + legacy `data/ownership/`; requires ≥4 samples |
| `/api/save-most-popular` | POST | Save Most Popular CSV → `data/most_popular/{date}.csv` (`INGEST_SECRET` when set) |
| `/api/save-ownership` | POST | Alias: same as save-most-popular (writes `data/most_popular/`) |
| `/api/save-most-drafted-3x` | POST | High-boost list → `data/most_drafted_3x/{date}.csv` |
| `/api/save-winning-drafts` | POST | Winning lineups → `data/winning_drafts/{date}.csv` |

**Admin / optional (not used by main UI):** `POST /api/hindsight` — optimal hindsight lineup from actual RS (Ben-driven or future). `GET /api/version` — build identifier for deploy/monitoring.

## App init and tab data flow

- **Startup:** `loadSlate()` and `initGameSelector()` run in parallel: `GET /api/slate` (10s) and `GET /api/games` (10s). Predict tab is default; slate list and game dropdown populate.
- **Tab load (lazy):** Lab loads on first visit when `switchTab()` is called:
  - **Lab:** `initLabPage()` → chat UI immediately; `showLabUnlocked()` loads briefing + config-history + slate in the background. `/api/lab/status` is pre-warmed from Predict and used when the locked Lab view/poll path runs (rare).

## Environment Variables (Railway)

- `GITHUB_TOKEN` — GitHub PAT with repo scope (for CSV + config read/write via Contents API)
- `GITHUB_REPO` — e.g. `cheeksmagunda/basketball`
- `ANTHROPIC_API_KEY` — Claude Haiku (screenshot OCR) + claude-opus-4-6 (Ben/Lab chat)
- `INGEST_SECRET` — (optional) When set, `POST /api/save-most-popular`, `save-ownership`, `save-most-drafted-3x`, and `save-winning-drafts` require `X-Ingest-Key` or `Authorization: Bearer <INGEST_SECRET>`. See `docs/HISTORICAL_DATA.md`.
- `CRON_SECRET` — (optional) When set, protected endpoints (`/api/cold-reset`, `/api/lab/auto-improve`, `/api/injury-check`) require `Authorization: Bearer <CRON_SECRET>`. Railway injects this via cron commands in `railway.toml`.
- `DOCS_SECRET` — (optional) When set, `/docs`, `/redoc`, and `/openapi.json` require `?docs_key=<value>` or `X-Docs-Key` header so only people with the secret can browse/test the API.
- `REDIS_URL` — (recommended) Full `redis://` connection string. Auto-injected by Railway Redis plugin. When set, all cache reads/writes go through Redis (sub-ms latency) with `/tmp` file fallback. Without it, the system works but relies on ephemeral `/tmp` files that don't survive container restarts. Setup: Railway dashboard → "+ New" → "Database" → "Redis". Verify: `GET /api/health` returns `"redis": "ok"`.

**OpenAPI docs:** FastAPI serves `/docs` (Swagger UI) and `/redoc` in production. Use them to browse and try endpoints. When `DOCS_SECRET` is set, append `?docs_key=<DOCS_SECRET>` to the URL or send the header to access.

## Runtime Config System

Model parameters are stored in `data/model-config.json` on GitHub. The backend loads this
file at startup and caches it for 5 minutes. The Lab writes updates via the GitHub Contents API.

- **No redeploy needed** to tune parameters — changes take effect within 5 minutes
- **Fallback to defaults** if GitHub is unreachable — app never breaks
- Use `_cfg("dot.path", default)` helper anywhere in `api/index.py` to read config
- `/api/cold-reset` clears config/cache layers and regenerates slate

## 4-Layer Cache Architecture (Generate Once, Serve from Redis)

The cold pipeline runs from three triggers only: (1) deploy SHA change startup hook, (2) `/api/injury-check` when injuries affect cached lineups, (3) manual `/api/cold-reset`. All picks are slate-bound and regenerated together.

### Cache Layers (read order)
0. **Layer 0 — In-memory `ResponseCache`**: Thread-safe dict with TTL. Cleared on every request to `_bust_slate_cache()`.
1. **Layer 0.5 — Redis** (`api/cache.py`): Primary persistent cache. Sub-ms reads. Survives container restarts. All `_cg()`/`_cs()` calls go here first. Requires `REDIS_URL` env var (Railway Redis plugin). Graceful fallback to Layer 1 when unavailable.
2. **Layer 1 — `/tmp` files**: Fallback file cache on Railway container filesystem. Cleared on container restart. `_cg()` backfills Redis from `/tmp` hits.
3. **Layer 2 — GitHub persistent cache (`data/slate/`)**: `{date}_slate.json` and `{date}_games.json`. Survives everything. Cold-start recovery source.
4. **Layer 3 — Full pipeline**: ESPN → LightGBM + heuristic blend → compression + game context → card boost (cascade + anti-popularity) → sort-based lineup builder. Only runs when all layers miss.

### Cache Helpers (grep: SLATE CACHE GITHUB)
- `_cg(key)` / `_cs(key, value)` — Redis-first read/write with `/tmp` fallback and automatic backfill
- `_bust_slate_cache(_caller)` — **Bust all layers**: in-memory + Redis (`rflush()`) + `/tmp` + GitHub tombstones. **Must only be called from paths that subsequently regenerate** — orphan tombstones cause infinite loading. Logs caller for debugging.
- `_slate_cache_to_github(slate_data)` — writes today's slate to `data/slate/{date}_slate.json`
- `_slate_cache_from_github()` — reads today's slate; returns `None` if missing or busted

### Cache Invalidation

**CRITICAL: Cache only regenerates cold on deploys. Deploys only happen on 3 triggers:**
1. **Slate turnover** (cron/automated) — `_run_cold_pipeline("slate_turnover")`
2. **Afternoon pre-slate injury/news update** (cron/automated) — `/api/injury-check` → `_run_cold_pipeline("injury_check")`
3. **Manual dev pushes to main** — Railway deploy → startup SHA mismatch → `_run_cold_pipeline("deploy_sha_change")`

**NEVER write bust tombstones to GitHub `data/slate/` manually** (via MCP tools, GitHub API, or ad-hoc scripts). `_bust_slate_cache()` writes tombstones that tell the cache layer "skip me, regenerate from pipeline." If no pipeline runs after the bust, the tombstones persist forever → Layer 2 always returns None → Layer 3 runs the full pipeline on every request → infinite loading when concurrent requests collide. Only the cold pipeline (bust + regen as an atomic unit) should write these files.

**Invalidation paths:**
- **Config change** (`/api/lab/update-config`): calls `_bust_slate_cache()` → busts all layers → next `/api/slate` request regenerates via Layer 3
- **Manual cold reset** (`/api/cold-reset`): calls `_run_cold_pipeline()` (bust + regen atomic)
- **Injury check** (`/api/injury-check` cron): calls `_run_cold_pipeline()` when injuries affect cached lineups
- **New deploy**: startup hook detects SHA mismatch → `_run_cold_pipeline()` (bust + regen atomic)

`_bust_slate_cache()` logs its caller (`_caller` param) for debugging orphan busts. All 4 call sites are tagged: `cold_pipeline:{trigger}`, `force_regenerate`, `lab_update_config`, `lab_rollback`.

### Version-Aware Startup (grep: deploy_sha)
On container start, `_deploy_startup_safe_prewarm()` compares `RAILWAY_GIT_COMMIT_SHA` against a `deploy_sha` key in Redis:
- **SHA mismatch** (new deploy): bust all caches, write new SHA, background-regenerate full slate
- **SHA match** (container restart): check Redis (instant) → `/tmp` → GitHub for existing cache — no pipeline run

### Prediction model boundaries

- **Draft model:** Config in `data/model-config.json` (card_boost, game_script, real_score, cascade, projection, lineup, moonshot, development_teams); code in `api/index.py`, `api/real_score.py`, `api/asset_optimizer.py`; `lgbm_model.pkl` is trained separately (GitHub Actions). Ben can change draft behavior via Lab (update-config, backtest).

## Ben (Lab) Interface

Ben is a **pure chat interface** — no quick-action buttons. The user types naturally and Ben:
- Auto-loads the briefing and config context silently on open (hidden messages)
- Offers to run backtests, apply config changes, analyze accuracy — all via conversation
- Decision history and config changes are stored in `LAB.messages` and `data/model-config.json`
- The chat prompt includes full system context (briefing data, config state, backtest capability)

### Historical data (developer-only this season)
- No in-app Ben upload banner. **Default:** **`docs/historical-ingest/INSTRUCTIONS.md`** (write `data/` directly; includes `slate_results`). **Optional API path:** **`docs/HISTORICAL_DATA.md`** (`parse-screenshot` + `save-*`; optional **`INGEST_SECRET`**).
- `/api/lab/briefing` returns **`pending_upload_date`** / **`pending_historical_date`**: most recent prediction date (excluding today) with **no rows in `data/top_performers.csv` for that date** (primary signal for “missing historical outcomes”).
- `POST /api/save-actuals` remains for rare manual merges; audit still auto-writes when `real_scores` is present in the merged upload.
- `/api/lab/skip-uploads` kept for API compatibility.

### Assistant playbook: user uploads PDFs (screenshots inside)

**Prefer** **`docs/historical-ingest/INSTRUCTIONS.md`**: rasterize → transcribe → write `data/` (no backend required).

**API alternative** (when the user wants automated OCR through this repo’s backend): **`POST /api/parse-screenshot` accepts images only** — `image/png`, `image/jpeg`, `image/gif`, `image/webp` (max 10MB). **Do not POST the PDF** to parse-screenshot.

1. **Rasterize first** — Export each screenshot page to PNG or JPEG (one file per screen the model should read). Options: macOS Preview (File → Export), `pdftoppm -png file.pdf page`, ImageMagick `convert -density 200 file.pdf page-%02d.png`, or ask the user to save each page as an image. Crop so one Real Sports screen dominates the image if the PDF has margins.

2. **Slate date** — Get `YYYY-MM-DD` from the user, the PDF filename (e.g. `2026-03-20-leaderboards.pdf`), or explicit labels in the chat. Every save payload must use the same `date` for that slate.

3. **Per image: classify → `screenshot_type` → parse → save**

   | What the image shows | `screenshot_type` (Form field) | Next step: POST body |
   |----------------------|-------------------------------|----------------------|
   | Most popular / most drafted list | `most_popular` or `most_drafted` | `POST /api/save-most-popular` with `{"date":"…","players": <response.players>}` |
   | High-boost (e.g. 3x+) sub-leaderboard | `most_drafted_high_boost` | `POST /api/save-most-drafted-3x` with same shape + optional `"min_boost": 3.0` |
   | Up to four winning lineups (flat rows) | `winning_drafts` | `POST /api/save-winning-drafts` with `{"date":"…","rows": <response.players>}` (or `players`) |
   | Highest-value / top performers strip only | `top_performers` | `POST /api/save-actuals` with `{"date":"…","players": <response.players>}` |
   | My Draft + Highest value / mixed leaderboard | `actuals` (default) | `POST /api/save-actuals` with same shape |

   Parse (example): `curl -sS -X POST "$BASE/api/parse-screenshot" -F "file=@./page01.png" -F "screenshot_type=most_popular"`. Response is always `{"players":[...]}` — pass that array through to the save call (field name `players` for most endpoints; `winning_drafts` also accepts `rows`). Haiku is prompted to extract **`team`** (NBA abbr) for `actuals`, `top_performers`, and `winning_drafts`; `save-actuals` / `save-winning-drafts` persist it so mega + per-day CSVs stay joinable to predictions and `slate_results`.

4. **`INGEST_SECRET`** — When set on the server, `save-most-popular`, `save-ownership`, `save-most-drafted-3x`, and `save-winning-drafts` require header **`X-Ingest-Key: <secret>`** or **`Authorization: Bearer <secret>`**. `save-actuals` does not use this secret. Use production `BASE` (e.g. Railway app URL) or local `uvicorn` if the user is developing.

5. **Rate limit** — `parse-screenshot` is capped at **5 requests/minute** per IP. Space OCR calls (~12+ seconds apart) or the user gets HTTP 429.

6. **`data/top_performers.csv` (mega)** — `save-actuals` writes **`data/actuals/{date}.csv`** on GitHub. Audit prefers rows in **`data/top_performers.csv`** keyed by `date`. After ingesting via `save-actuals`, tell the user to run locally: **`python scripts/rebuild_top_performers_mega.py`**, then commit and push **`data/top_performers.csv`** (merges mega + all `data/actuals/*.csv`). Skip this only if they rely solely on per-day actuals for that date.

7. **Verify** — `python scripts/verify_historical_datasets.py` and, when predictions exist for those dates, `python scripts/verify_top_performers.py`.

8. **If the assistant cannot call the user’s API** — Output exact `curl` commands (with placeholder paths and `BASE`), the chosen `screenshot_type` per file, and the reminder about rasterizing PDFs and rebuild script.

Canonical reference: **`docs/HISTORICAL_DATA.md`**.

### Lab / Ben availability
- Ben (Lab) chat is always available from the frontend’s perspective — the Lab tab no longer shows a locked vs unlocked state.
- Backend lock helpers (`_is_locked`, `_all_games_final`, `/api/lab/status`) still exist for internal decisions (e.g. slate generation, cron behavior), but they no longer gate Ben’s UI.

### Keyboard / Nav Behavior (Ben tab)
- On **mobile**: focusing `#labInput` hides the bottom nav and expands `#tab-lab` to fill freed keyboard space via `lab-kb-open` CSS class. Blur restores everything.
- On **desktop**: keyboard handler is skipped entirely via `window.matchMedia('(hover: none) and (pointer: coarse)')` — no nav hiding.
- CSS class `#tab-lab.active` uses `height: calc(100dvh - 80px - 120px)` (leaves room for nav). `#tab-lab.active.lab-kb-open` expands to `calc(100dvh - 80px)` (nav hidden).
- `#labMessages` has `padding-right: 12px` so the scrollbar does not overlap right-aligned user message bubbles when scrolling.

## Loading Animation

A **Magic 8-ball** animation plays on app load and during API calls (slate fetch, game analysis).
- Dark floating sphere with "8" and a triangle window showing rotating oracle messages
- CSS keyframe animation: `ballFloat` (3s ease loop), `ballShake` on load
- Controlled by `showLoader()` / `hideLoader()` in JS
- Messages cycle: "READING THE GAME", "CONSULTING THE ORACLE", "CALCULATING EDGE", etc.

## Prediction Save Deduplication

`savePredictions()` fires at most **once per session** (frontend flag) and the backend compares
the new CSV content against what's already stored — skipping the GitHub commit if unchanged.
This prevents the commit → Railway redeploy cascade that was triggering 6+ redeploys per visit.

The `/api/save-predictions` endpoint also enforces a server-side lock guard — it returns HTTP 409
if called before the slate is locked, making it impossible to persist pre-lock projections regardless
of which call path invoked it (frontend, cron, or direct POST).

## Lock System (Event-Driven Unlock)

Predictions lock 5 minutes before the earliest game starts (lock window). Once locked, slate remains locked based on **game completion status**, not clock:

### Unlock Triggers (in priority order)
1. **ESPN Game Final** (Primary): If `_all_games_final()` returns True, unlock immediately (game completion event)
2. **Time-Based Fallback**: If latest game running 4.5+ hours, mark as complete (ESPN lag protection)
3. **6-Hour Ceiling**: If game_start + 6h has passed, stop polling (safety net)

### Lock Cache Behavior
- Lock cache (`/tmp/nba_locks_v1/`) survives within a running Railway container
- Cache TTL during locked slate: **60 seconds** (balanced for event-driven detection and Railway resource use)
- Cache TTL pre-slate: **180 seconds** (normal polling)
- On cache expiration, immediately checks ESPN to see if games are done
- On cold start with no cache, `data/locks/{date}_slate.json` on GitHub is the recovery source
- On cold start with no GitHub backup either, returns empty locked response (frontend preserves displayed data)

### `any()` Lock Pattern (split-window days)
All slate-level lock checks use `any(_is_locked(st) for st in start_times)` — NOT `_is_locked(min(...))`.

**Why `any()`?** On split-window Saturdays (e.g. 2 PM + 9 PM CT):
- 2 PM game's 6h ceiling expires at 8 PM
- 9 PM game's 6h ceiling expires at 3 AM (next day)
- System must stay locked while ANY game is live, not just the first one
- `min()` would incorrectly unlock at 8 PM when late games still active
- `any()` correctly stays locked until BOTH time window AND game completion

This applies to:
- `/api/slate`: `any(_is_locked(st))` before computing predictions
- `/api/save-predictions`: `any(_is_locked(...))` guard prevents pre-lock saves
- `/api/cold-reset`: secured trigger for global bust+regen; pre-save runs if slate is locked
- `/api/lab/status`: `any(_is_locked(st))` determines locked state

Per-game checks (e.g. `/api/picks`) correctly use single-game `_is_locked(game_start)`.

### Triple-Gated Save Pipeline
Predictions are saved to `data/predictions/` through exactly two code paths, both strictly post-lock:
1. **`/api/save-predictions`** — called by frontend `savePredictions()` and pre-save within the cold-reset pipeline
2. **Inline at lock-promotion** in `/api/slate` — first locked request promotes cache and writes CSV

Three independent gates prevent pre-lock saves:
- **Frontend guard**: `if (!SLATE || !SLATE.locked) return;` in `savePredictions()`
- **Backend guard**: `if not any(_is_locked(st) ...)` → HTTP 409 in `/api/save-predictions`
- **Cold-reset guard**: global cold pipeline pre-saves predictions when `any(_is_locked(...))`

## Player Eligibility & RS Projection (v82 — Strategy-Report-Aligned)

### Projection-level gates (project_player)

- **Universal RS floor: 2.0** (`strategy.min_pts_projection`) — Strategy report Finding 2: only 1.3% of winning players have RS < 2.0. Single gate replaces the prior ~15 separate thresholds.
- **Minimum 12 projected minutes** (`strategy.min_minutes`) — ensures rotation-level players only.
- **OUT players filtered** — RotoWire integration removes injured/OUT players before lineup selection.

### RS Projection Pipeline (simplified)

1. **Heuristic DFS score** — weighted sum of projected stats (`pts × 1.5 + ast × 1.2 + reb × 0.5 + stl × 3.0 + blk × 3.0 - tov × 1.0`) using `real_score.dfs_weights`
2. **Game context adjustments** (simple additive):
   - Close games (spread ≤ 5): +0.3 RS (`strategy.close_game_rs_bonus`)
   - High-pace games (total > 220): +0.15 RS per 10 pts above 220 (`strategy.pace_rs_bonus_per_10`)
   - Home court: 1.02× multiplier
3. **Compression**: `min((s_base / divisor)^power, rs_cap)` — divisor 5.5, power 0.72, cap 20.0
4. **AI/heuristic blend**: 35/65 (`ai_blend_weight: 0.35`)
5. **Game context bonus applied additively** after compression
6. **EV calculation**: `draft_ev = RS × (2.0 + boost)`

No Monte Carlo, no post-compression multiplier stack, no archetype calibration, no matchup factor.

### Per-game lineup gates (unchanged)

- **Minimum 10 projected pts** — single-game format has no card boost, so low-scoring players are ceiling liabilities

### Tunable via Ben

**Draft strategy** in `strategy`: `rs_floor` (2.0), `min_pts_projection` (2.0), `min_minutes` (12.0),
`close_game_rs_bonus` (0.3), `pace_rs_bonus_per_10` (0.15), `ev_swap_threshold` (2.0),
`max_upside_swaps` (2), `anti_popularity_enabled` (true), `anti_popularity_strength` (0.1).

**Historical RS discount** in `strategy.historical_rs_discount`: `enabled` (true), `min_appearances` (3),
`saturation_k` (8.0), `max_prior_strength` (0.5), `discount_scale` (0.4), `max_discount_frac` (0.6).

**Momentum curve** in `strategy.momentum_curve`: `enabled` (true), `min_history` (3),
`hype_trap_max_penalty` (0.20), `draft_growth_threshold` (0.5), `boost_decline_threshold` (0.4),
`rising_wave_max_bonus` (0.35), `rs_trend_min` (0.15), `wave_max_drafts` (300), `wave_min_boost` (1.5).

**Per-game**, **Context layer** config sections remain unchanged.

---

## Two Draft Strategies (Condition-Matrix-Driven Architecture)

Both lineups use the **Condition Matrix** as the PRIMARY signal: `composite_ev = condition_coeff × RS × (2.0 + boost) × data_multipliers`. The `condition_coeff` comes from `CONDITION_MATRIX[ownership_tier][boost_tier]` in `api/real_score.py` — a historical HV rate lookup (0.0–1.0). Format mechanics (who drafts whom × what boost they get) dominate basketball conditions (RS, matchups, pace). No positional caps — composition driven entirely by composite EV. Max 1 player per team per lineup.

**Pipeline** (7-step in `_build_lineups`):
1. **Filter**: RS ≥ 2.0 (`strategy.rs_floor`), minutes ≥ 12 (`strategy.min_minutes`), not OUT, not blacklisted
2. **Score**: `composite_ev = condition_coeff × RS × (2.0 + boost) × data_multipliers` for each candidate
   - `condition_coeff`: ownership tier (from historical draft counts or `estimate_draft_popularity()` fallback) × boost tier → HV rate
   - `data_multipliers`: minutes increase, leaderboard frequency, contrarian bonus, momentum curve
3. **Dead Capital Filter**: Remove players with `condition_coeff = 0.0` (trap plays: chalk+low_boost, mega_chalk+no_boost, etc.)
4. **Safe Lineup (Starting 5)**: Top 5 by `safe_ev` (cb_low). Ties broken by lower variance (reliable floor). Max 1 per team.
5. **Upside Lineup (Moonshot)**: Top 5 by `upside_ev` (cb_high) from remaining pool (excluding S5 players). Max 1 per team.
6. **Slot Assignment**: Both lineups sorted by RS descending → 2.0x, 1.8x, 1.6x, 1.4x, 1.2x. Provably optimal: `d(Value)/d(slot) = RS`.
7. **Core Pool**: Top 20 candidates for watchlist/review.

**Anti-popularity** (Finding 4): Draft popularity has -0.457 correlation with boost. The cascade in `boost_model.py` estimates popularity from season PPG + team market + hot streak, then penalizes popular players' boost predictions. The least-drafted 50% produce 24-26% more total value.

**Dead Capital** (`DEAD_CAPITAL_CONDITIONS` in `api/real_score.py`): Specific ownership × boost combos with zero historical HV rate. These are auto-filtered before lineup selection: chalk+elite_boost, chalk+max_boost, chalk+low_boost, mega_chalk+low_boost, mega_chalk+no_boost.

Config: All draft strategy parameters in `strategy.*` section of model-config defaults. `strategy.condition_matrix.enabled` (default True) controls the meta-game layer.

### Slate-Wide: Starting 5 (Safe)
Top 5 by `safe_ev` (using `cb_low` conservative boost floor). Tie-breaking prefers lower variance for floor reliability. No MILP, no boost floors, no star anchor pathway — the composite EV formula naturally selects the right mix. Max 1 player per team.

### Slate-Wide: Moonshot (Upside)
Top 5 by `upside_ev` (using `cb_high` optimistic boost ceiling) from the pool excluding S5 players. Sorted by upside_ev descending, tie-break on boost. Max 1 player per team. Divergence from S5 comes from `upside_ev` (cb_high) vs `safe_ev` (cb_low) ranking differences + team cap.

### Per-Game: THE LINE UP (v60 — Strategy-Aware Draft Model)
Redesigned from 18-game / 76-lineup empirical analysis (Jan 6 – Mar 23, 2026). Single 5-player format for single-game drafts. **No Starting 5 / Moonshot split** — both users draft from the same 2-team pool, making card boost irrelevant.

**Pipeline** (6-step):
1. **Game script re-scoring** — stat-weight tiers by game pace (existing)
2. **Per-game strategy adjustments** (NEW) — `_per_game_adjust_projections()` applies:
   - **F3: Game total multiplier** — scales all projections ±12% based on O/U vs 222 baseline (250+ → 32.1 avg winning score; <210 → 25.7)
   - **F4: Spread-based composition** — close games (≤5pt spread) reward balanced/consistent producers; blowouts (13+pt) lean toward favored team
   - **F6: Favored team role player tilt** — in blowouts, favored team's 3rd-5th options get +12% (extended garbage-time run); underdog role players get -12%
   - **F2: Value anchor bonus** — mid-tier players (RS ≥3.8, season_pts ≤16) get +8% rating boost for their floor-lifting effect at 1.2x-1.4x slots
   - **F1: Conviction slot variance shaping** — close games dampen variance (consistent players rise to 2.0x); blowouts uplift variance (high-ceiling players rise)
3. **Eligibility gating** — `recent_min ≥ 15`, `rating ≥ 3.5`, `pts ≥ 8`, not blacklisted
4. **MILP optimization** — RS × slot_mult (card boost zeroed). In blowouts, `min_per_team` relaxes from 2 to 1 (allows 4-1 team split)
5. **5! permutation validation** (F5) — brute-force 120 combos confirms optimal slot assignment (razor-thin margins: <1.5 pts in 8/18 games)
6. **Strategy metadata** — `strategy` object returned with type, label, description for frontend display

**Strategy Types:** Balanced Build (spread ≤5), Standard Build (6-12), Blowout Lean (13+); overlays: Shootout (total ≥245), Defensive Grind (total <215).

**Config:** `per_game.*` section in model-config.json (20 tunable parameters). All adjustable via Ben.

**Frontend:** Strategy badge (color-coded by type), strategy insight bar with description, per-player ANCHOR/FAV pills on cards.

### Game Context (Strategy Report Finding 7)
Simple additive RS adjustments replace the prior complex multiplicative matchup/spread system:
- **Close games** (spread ≤ 5): +0.3 RS (`strategy.close_game_rs_bonus`). Finding 7: close games produce avg RS 4.44.
- **High-pace games** (total > 220): +0.15 RS per 10 pts above 220 (`strategy.pace_rs_bonus_per_10`).
- **Home court**: 1.02× multiplier.
- `_compute_matchup_factor()` still exists in codebase (retained for potential reactivation) but is no longer called from the draft pipeline.

### RotoWire Integration (`api/rotowire.py`)
Free-tier scrape of RotoWire NBA lineups page. Runs ~30 min before first tip. Returns player availability (confirmed/expected/questionable/OUT). Draft pipeline filters OUT players in `_build_lineups()`. Cache TTL: 30 minutes.

## Model Improvements (deployed)

### LightGBM (12 features, `lgbm_model.pkl`)
Features: `avg_min, avg_pts, usage_trend, opp_def_rating, home_away, ast_rate, def_rate, pts_per_min, rest_days, recent_vs_season, games_played, reb_per_min`

- Model bundle format: `{"model": lgb.LGBMRegressor, "features": [...]}` — inference verifies feature vector length; bundle required.
- `rest_days` and `games_played` default to `2.0` / `40.0` at inference (not in ESPN splits). `recent_vs_season` = recent scoring vs season average (training: recent_5g_pts/avg_pts; inference: recent_pts/season_pts).
- Retrained nightly by GitHub Actions (`retrain-model.yml`). Retrain manually: `python train_lgbm.py`.

### Card Boost (`_est_card_boost`) — 3-Tier Cascade (api/boost_model.py)
No pre-game boost uploads or per-player overrides at inference. Pipeline uses a deterministic cascade calibrated from 2,234 player-date records across 148 dates:

- **Tier 1 — Returning Player** (appeared on a slate within 14 days): `predicted_boost = prev_boost + adjustments`. Adjustments: RS performance decay (high RS → -0.1, bust → +0.1), draft popularity (heavily drafted → -0.05, ignored → +0.05), 5% mean reversion toward tier baseline, 15% trend continuation, gap-day blend toward baseline, boundary persistence (3.0 stays 3.0 in ~75% of cases; 0.0 stars stay at 0.0). Expected MAE ~0.10–0.15.
- **Tier 2 — Known Player, Stale** (>14 days since last appearance): Blends historical boost mean with API-derived estimate, weighted by staleness (0% at 14 days, 50% at 44+ days).
- **Tier 3 — Cold Start** (never seen on Real Sports): Player Quality Index (PQI) from season stats (PPG×1.0 + RPG×0.4 + APG×0.6 + MPG×0.2) maps to boost. Default high (3.0) for unknown players.
- **Post-prediction caps**: Star PPG tier caps (26+ PPG → 0.2x, 22+ → 0.4x, 19+ → 0.8x), per-team boost ceilings (GS/CLE/MEM/OKC fans draft at higher rates).
- **Data source**: `data/top_performers.csv` + `data/actuals/*.csv`, loaded once at startup, indexed per-player by date.

`boost_model.pkl` and `drafts_model.pkl` are no longer used at inference. `train_boost_lgbm.py` and `train_drafts_lgbm.py` are retained for reference but no longer required for production.

Post-game **ownership CSVs** remain for `/api/lab/calibrate-boost` (proposed formula tweaks).

### Cascade Team Detector + Deep Rotation Sweet Spot (grep: CASCADE TEAM DETECTOR)
Based on analysis of 2,299 top performer entries across 151 dates:

- **Cascade Team Detector**: When a star (20+ PPG season avg) is marked OUT on a team, all active teammates are flagged with `_cascade_team=True`. Flagged players receive a boost floor (2.5) — no RS multiplier to avoid overfitting both lineups to injury plays. RS comes from actual projected stats; the cascade signal flows through boost (underdrafted → high card boost). Historical data: 192 mega-stack instances where 3+ same-team players hit the leaderboard with combined values of 50-80+. Detection is purely stat-based (PPG threshold) — no hardcoded player names.
- **Deep Rotation Sweet Spot**: Cascade team players get relaxed gates in both `project_player` (min_gate 12 min vs 25) and `_build_lineups` (RS floor 1.5 vs 2.0, minutes floor 12 vs 25, rotation-bubble filter bypassed). The 5-20 draft archetype historically produces the highest avg value (16.1) — higher than superstars (8.4) and starters (14.9).
- **Proportional Cascade Cap**: Cascade minute bonus capped at 40% of player's avg minutes (`max_cascade_pct`). Prevents bench players (16 avg min) from being projected at 26+.
- **Config**: `cascade.team_detector.*` section in model-config.json (`enabled`, `star_ppg_threshold`, `boost_floor`, `deep_rotation_rs_floor`, `deep_rotation_min_minutes`). All tunable via Ben.

### Removed Systems (v82 Strategy Simplification)
The following systems were removed or replaced in the strategy-report-aligned architecture:
- **Monte Carlo simulation** (`real_score_projection`, `closeness_coefficient`, `clutch_coefficient`) — replaced by simple additive game context. Module `api/real_score.py` retained for reference.
- **Post-compression multiplier stack** (archetype calibration, cascade RS, role spike RS, breakout probability, volatility guard) — removed entirely. RS compression is now `(s_base / divisor)^power + game_context_bonus`.
- **MILP optimizer for slate-wide** (`optimize_lineup` from `api/asset_optimizer.py`) — replaced by sort-based selection. MILP still used for per-game lineups.
- **Complex player gating** (star anchor, high-boost-role pathway, RS bypass, scorer upside, roto confirmed exceptions, min chalk rating, moonshot pts floor, etc.) — replaced by two gates: RS ≥ 2.0 and minutes ≥ 12.
- **Matchup factor** (`_compute_matchup_factor`) — no longer called; function retained for potential reactivation.
- **Separate chalk/moonshot formulas** — unified to `draft_ev = RS × (2.0 + boost)`.
- **Core pool ranking metrics** (`rs`, `max_ev`, `blend`, `ev_weighted`) — replaced by single `draft_ev` sort.

### Audit Pipeline
- `save-actuals` auto-writes `data/audit/{date}.json` with MAE, directional accuracy, over/under counts, top-8 misses.
- `GET /api/audit/get?date=X` returns pre-computed audit (falls back to live computation).
- `lab_briefing` uses cached audits when available; adds over-projection pattern detection.

## z-index Hierarchy (fixed elements)

| Element | z-index |
|---------|---------|
| `.bottom-nav` | 1000 |

`switchTab()` resets `document.body.style.overflow` on every tab switch to prevent scroll lock leaking between tabs.

Note: `predictSubNav` is an **inline element** (not fixed/floating) — no z-index needed.

## Global Design Tokens (index.html :root)

Single source of truth for UI consistency and future theming:

| Token | Purpose |
|-------|---------|
| `--color-success` | Neon green — HITs, positive edges, success states |
| `--color-danger` | Coral red — MISSes, negative edges, alerts |
| `--color-warning` | Gold/orange — mid-tier multipliers, warnings |
| `--color-text-primary` | White/off-white — primary text, buttons |
| `--color-text-muted` | Slate grey (#8A96A3) — metadata, labels |
| `--radius-pill` | 9999px — tags, pill buttons |
| `--radius-card` | 14px — main containers, cards, modals |
| `--tracking-caps` | 0.06em — letter-spacing for ALL CAPS labels |

Use these tokens instead of hardcoded hex or pixel radii across Predict and Ben.

## Cron Schedule (railway.toml)

Crons and frontend poll intervals are tuned to minimize Railway compute and ESPN API usage while preserving lock/unlock behavior.

| Schedule (UTC) | Endpoint | Purpose |
|----------------|----------|---------|
| `0 9 * * 0,3` | `/api/lab/auto-improve` | Auto-tune model if ≥3% MAE improvement (Wed + Sun only — 2×/week) |
| `0 18,22,2 * * *` | `/api/injury-check` | 3 key windows: pre-tip (1 PM ET), mid-evening (5 PM ET), late (9 PM ET) |
| `0 6 * * 1` | `/api/mae-drift-check` | Weekly MAE drift monitoring (Monday 6am UTC); CRON_SECRET-gated |

## Deployment Pipeline

Railway `watchPatterns` in `railway.toml` prevents rebuilds on data-only commits:
```
railway.toml watchPatterns already excludes `data/` and `.github/` — only code changes trigger a Docker rebuild.
```
This ensures GitHub API writes to `data/` and `.github/` workflow changes don't trigger unnecessary Docker rebuilds. Only code changes trigger deployments.

## Production Robustness Notes

All frontend API calls (`fetch(...)`) have `.ok` checks before calling `.json()`. Missing `.ok` checks were a common source of silent failures in prior versions.

Key patterns used throughout:
- Async functions: `if (!r.ok) throw new Error('HTTP ' + r.status)` before `.json()`
- Promise.allSettled chains: `fetch(...).then(r => r.ok ? r.json() : Promise.reject(...))`
- Polling loops: `.then(r => r.ok ? r.json() : Promise.reject())` with empty `.catch`
- `savePredictions`: resets `_predSavedDate` on non-OK responses so the next call can retry

### Resilience (error boundaries)
- **Backend:** A global `@app.exception_handler(Exception)` in `api/index.py` catches any unhandled exception: logs full traceback server-side and returns `JSONResponse({"error": "An unexpected error occurred"}, status_code=500)` with no stack trace or internal detail in the response.
- **Frontend:** All `JSON.parse(localStorage.getItem(...))` usages are wrapped via `_safeParseLocalStorage(key, fallback)` so corrupted or invalid JSON does not throw. Critical DOM access uses `_el(id)` with null checks (or early return) in Lab/Predict and Lab lock poll so missing elements do not throw. Lab lock poll logs status fetch failures with `console.warn('[lab] Lock poll status check failed:', ...)` instead of failing silently. Lab chat uses an `AbortController` with a 60s connection timeout for the initial `/api/lab/chat` request; on timeout the user sees "Request timed out. Please try again."

Hidden `appendLabMessage(..., hidden=true)` rows are excluded from the visible chat list and from the payload sent to `/api/lab/chat`.

**Health in deployment:** Use `GET /api/health` for uptime monitoring. Railway uses `healthcheckPath = "/api/health"` from railway.toml and shows health in the dashboard. Configure an external checker (e.g. UptimeRobot, Cronitor) for alerting on non-200.

## Event-Based Slate Transition

The system now uses **game completion events** instead of clock-based timeouts for slate unlocking. When the final game on a slate completes, the system:

1. **Immediately detects completion** — `_all_games_final()` checks ESPN scoreboard and fires the unlock event
2. **Unlocks Lab-facing flows** — slate no longer treated as in-progress for internal lock checks; next-day Predict data can load as games go final
3. **Enables next slate** — New games become draftable within seconds of previous slate completion

### Cache TTL Optimization (Adaptive)
- **Locked slate**: Cache TTL **60 seconds** (from 180s; balances responsiveness and Railway resource use)
- **Pre-slate**: Cache TTL remains 180 seconds
- During locked periods, the backend refreshes game status every 60s instead of waiting 3 minutes
- Unlock still detected within the next frontend poll (2 min Lab)

### Aggressive ESPN Fallback (4.5-Hour Rule)
If ESPN API delays updating game status to "Final":
- If latest game running 4.5+ hours: automatically mark all games as complete
- Ignores `finals > 0` requirement — fires even if ESPN completely lagged
- Prevents indefinite lock waits when ESPN slow during high-traffic windows (Saturday evenings)
- Prevents false unlocks: still requires `remaining == 0` (at least one game attempt started)

### Event-Driven Frontend Unlock
When game polling detects games finished (`status === 'final'`):
- Immediately triggers `/api/lab/status` check instead of waiting for next poll cycle (~1-2 min)

### Lock System Priority
The unlock logic prioritizes game completion over time windows:
1. **First**: Check if all games final via ESPN → unlock immediately
2. **Second**: Check if `any(_is_locked(st))` for remaining games → stay locked
3. **Fallback**: Use 4.5-hour timeout → unlock if game running too long

This prevents lock waits even when ESPN lags during high-traffic periods.

## Responsiveness & Reliability Improvements

### Fetch Timeout Protection
All frontend API calls use a `fetchWithTimeout()` wrapper function that enforces hard timeout limits via `Promise.race()` and `AbortController`:
- **Default timeout**: 10 seconds (most endpoints)
- **Screenshot parse timeout**: 30 seconds (longer due to Claude vision processing)
- Prevents indefinite hangs if backend becomes slow or unresponsive
- Returns HTTP error status if timeout occurs, triggering normal error handling

Affected endpoints: slate load, picks, games, save-predictions, screenshot parse, save-actuals, audit, lab-status, lab-briefing, lab-chat, lab-config-history, hindsight. Full timeout table and loading UX: **docs/LOADING_AUDIT.md**.

### Worker Pool Optimization
Backend uses Python `ThreadPoolExecutor` for parallel processing:
- **Standard pool: 8 workers** (game runner, slate processor, picks processor, audit runner)
- Handles 14-game Saturdays efficiently without bottlenecking

### Polling Interval Tuning
- **Lab lock polling**: 2 minutes (reduces API call frequency; see `initLabPage` in index.html, ~3135)
  - Unlock detected within ~2 min; user can tap Retry for immediate check

### GitHub API Retry Logic
`_github_write_file()` (api/index.py lines 75-110) implements exponential backoff for concurrent write conflicts:
- **Retries up to 3 times** on HTTP 422 (SHA mismatch)
- **Backoff delays**: 1s, 2s, 4s between retries
- **Fresh SHA fetch** on each retry (not cached)
- Protects against concurrent writes from cron + overlapping API writes (rare but possible edge case)
- Used for: predictions, actuals, config updates

### Cache TTL & Invalidation
Explicit TTLs protect against stale data while minimizing API calls:

| Cache | TTL | Purpose | Invalidation |
|-------|-----|---------|--------------|
| Game final status (`_all_games_final`) | 60s when locked, 180s pre-slate | Detects when ALL games reach Final status | global cold reset clears |
| Model config (`data/model-config.json`) | 5 min | Runtime tuning parameters | global cold reset clears; Lab writes bypass cache |
| RotoWire lineups | 30 min | Player availability (OUT, questionable, etc.) | 30 min expiration; manual refresh via app |
| Lock status per game | 6 hours | 5 min before tip to 6h after (ceiling) | Natural expiration |
| Slate cache (`data/slate/`) | 1 day | GitHub-persisted predictions (full slate + per-game) | `_bust_slate_cache()` via cold reset, config change, or injury check |
| Log dates (`log_dates_v1`) | 10 min | Dates with stored prediction/actual data (backend only) | global cold reset clears all `/tmp` caches |
| Log get (`log_get_{date}`) | 5 min | Per-date predictions + actuals from GitHub (backend only) | global cold reset clears all `/tmp` caches |
| ESPN scoreboard (`fetch_games`) | 5 min (`_TTL_GAMES`) | Schedule + spreads; shared `_GAMES_CACHE_TS` | New fetch resets TS |

### Production: DRY rules and cache inventory

**Do not duplicate TTLs** — All backend TTLs live in one block in [`api/index.py`](api/index.py) (`_TTL_CONFIG`, `_TTL_GAMES`, `_TTL_LOG`, `_TTL_LOCKED`, `_TTL_PRE_SLATE`). Endpoint-specific ages should stay aligned with these constants when changed.

**Cache helpers (single pattern)** — `CACHE_DIR` (`/tmp/nba_cache_v19`) + `_cp(key, date_str?)` → file path; `_cg` read JSON; `_cs` write JSON. Same helpers for slate, news, team stats. Avoid ad hoc `Path` writes outside this pattern.

**Named logical keys (`_CK_*`)** — `_CK_SLATE`, `_CK_SLATE_LOCKED`, `_CK_LOG_DATES`; per-game `game_proj_{gameId}` via `_ck_game_proj()`. Search `# grep: CONSTANTS & CACHE` in `api/index.py`.

**GitHub vs `/tmp`** — Slate: Layer 1 `/tmp` → Layer 2 `data/slate/*` → Layer 3 pipeline. `_hydrate_game_projs_from_github()` loads per-game projections from GitHub before `_run_game()` when `/tmp` is cold. Single read replaces N× ESPN/LGBM for the same ET day when a prior instance already wrote `data/slate/{date}_games.json`.

**Bust / reset** — `_bust_slate_cache(_caller)` tombstones GitHub slate + games + locks and clears local `/tmp` JSON. **NEVER call directly or write tombstones to GitHub manually** — orphan tombstones without a subsequent pipeline regen cause infinite loading. `GET /api/cold-reset` runs the global bust+regen pipeline (atomic) and reloads config/cache layers. Cache only regenerates cold on deploys (3 triggers: slate turnover cron, injury-check cron, dev push to main).

### ESPN API Fallback
`_all_games_final()` protects against ESPN outages:
- If game status not updated for 4+ hours, mark as final (assume game completed)
- Safety guard: `if finals == 0 and remaining == 0: all_final = False` — prevents false unlock on ESPN API down
- Requires `finals > 0` before returning true (at least one game must have reached Final status)
- Falls back to GitHub lock file recovery on cold start if ESPN unreachable

## Skip uploads (`/api/lab/skip-uploads`)

- **POST** `/api/lab/skip-uploads` with `{ "date": "YYYY-MM-DD" }` appends to `data/skipped-uploads.json` (GitHub write with retry). No in-app UI; optional for scripts.
- **`save_actuals()`** returns early when the date is listed — avoids clobbering data for slates you intentionally ignore.

`data/skipped-uploads.json` shape: `{ "skipped_dates": ["..."], "last_skipped_at": "ISO8601" }`.

## Other Files (Extended Audit)

| File | Role | Notes |
|------|------|------|
| **api/rotowire.py** | RotoWire lineup scraper | Free-tier scrape for availability (OUT, questionable). 30 min cache. Used by slate/Moonshot filtering. |
| **api/real_score.py** | DFS scoring weights + Monte Carlo (reference) | DFS stat weights (`dfs_weights`) used by `project_player()`. Monte Carlo simulation (closeness, clutch, momentum) retained for reference only — not called by draft pipeline. |
| **api/asset_optimizer.py** | MILP lineup optimizer | PuLP/CBC. **Used by per-game pipeline only** (`_build_game_lineups`). Slate-wide drafts use sort-based selection in `_build_lineups`. |
| **server.py** | Local dev server | Serves index.html at `/` and re-exports FastAPI app for `uvicorn server:app`. Production runs as a persistent Docker container on Railway. |
| **scripts/check-env.py** | Env verification | Validates REQUIRED (GITHUB_TOKEN, GITHUB_REPO, ANTHROPIC_API_KEY) and OPTIONAL vars. Run before local dev. |
| **scripts/sync_model_config.py** | Config sync | Syncs model-config from GitHub (used by workflows). |
| **scripts/bump_retrain_config.py** | Retrain config | Bumps retrain config for GitHub Actions. |
| **train_lgbm.py** | Model training | 12-feature LightGBM training; outputs lgbm_model.pkl. Run locally or via retrain-model.yml. |

No orphan entrypoints; all API surface is in api/index.py. Scripts are for local/CI use.

## Audits

- **docs/AUDIT-LIGHTWEIGHT.md** — Production, object/variable/reference, pipeline/caching, LightGBM (includes fix for recent_vs_season train/inference alignment).
- **docs/AUDIT-HEAVY.md** — Security, error handling, API consistency, contracts, timeouts, deployment, observability, tests/docs.
- **docs/LOADING_AUDIT.md** — Frontend loading: fetch timeouts, skeletons, async state.

## Unit Testing Framework

Two test modules; run both for full coverage:

**tests/test_fixes.py** — Backend behavior, mocked I/O:

```python
# Test classes (pytest):
TestSafeFloat               — numeric/None/empty string edge cases for _safe_float()
TestIsLocked                — 5-min pre-tip buffer, 6h ceiling, split-window any() pattern
TestComputeAudit            — MAE calculation, no-data guard, zero-RS skip, miss sorting
TestGitHubWriteRetry        — 422 SHA conflict, 1s/2s/4s backoff, max-retry error return
TestSaveActualsAuditGate    — audit only fires when real_scores data is present
TestCacheTTLs               — 3 min games, 5 min config, 30 min RotoWire, 60s locked TTL
TestPollingIntervals        — 120s lab lock
TestRateLimitThreadSafe     — _check_rate_limit is thread-safe under concurrent calls
TestLgbmFeatureAlignment    — when bundle loaded, 12 features with recent_vs_season at index 9 and reb_per_min at index 11
TestSlateExceptionHandling  — slate endpoint catches exceptions and returns 200 with error key (never 500)
TestGameSelectorLockDisplay — frontend populateGameSelector must NOT override per-game lock with slateLocked
TestFetchGamesTTL           — fetch_games() enforces 5-min TTL to avoid stale ESPN data
TestSavePredictionsMerge    — save_predictions merges new per-game scopes into existing CSV
TestSlateCacheGitHub        — GitHub slate cache read/write, tombstone/busted handling, games cache roundtrip
TestInjuryCheck             — injury-check lock guards, cache misses, RotoWire OUT/confirmed/unknown detection
TestPicksServeFromCache     — per-game cache loading, GitHub fallback for picks, bust tombstone writes
TestClaudeContextLayer      — context pass enable/disable, multiplier clamping, graceful fallback
TestPredMinTolerance        — chalk (2.0) and moonshot (3.0) tolerance band config and code presence
TestMoonshotPtsFloor        — separate moonshot pts floor (4.0) config and chalk enforcement (7.0)
TestDailyBoostIngestion     — Layer 0 daily boost load/parse, _est_card_boost priority over config overrides
TestBoostModelInference     — Layer 1 ML path + override fallback; shared JSON dict helper for boosts
TestWatchlist               — _build_watchlist cascade candidate detection, max-10 cap
TestBoostsScreenshotType    — parse-screenshot accepts 'boosts' screenshot_type
TestCorePoolRsMetric        — core_pool.metric="rs" ranks by raw projected RS; "max_ev" backward compat
TestMoonshotRsBypass        — moonshot.rs_bypass allows high-RS players to bypass boost floor; config validation; offline defaults
TestChalkMilpRsFocusHigh    — chalk_milp_rs_focus=0.85 nearly neutralizes boost in MILP; calculation verification
TestWebSearch               — Claude web_search skip when disabled/no key, fetch+cache, cache reuse
TestContextPassWithNews     — web search called from context pass, news text in prompt
TestHighBoostRolePathway    — high-boost role players bypass minutes floor in moonshot + chalk pools
TestRsCalibrationWeights    — DFS weight recalibration, archetype detection + calibration, scorer upside
TestCascadeCapFix           — per_player_cap_minutes raised to 10.0 for meaningful cascade propagation
TestCascadeTeamDetector     — cascade team config, flag propagation, RS multiplier, boost floor, relaxed gates, no hardcoded names, max_per_team (9 tests)
TestRotoConfirmedRatingException — confirmed rotation players with high boost bypass min_rating_floor
TestMaxPerGameConstraint    — MILP max_per_game limits players from same game matchup
TestMinHighBoostConstraint  — MILP min_big_boost ensures minimum high-boost players in lineup
TestPerGameStrategy         — _per_game_strategy() returns correct type/label based on spread+total (6 tests)
TestPerGameAdjustProjections — F1-F6 adjustments: total mult, close game consistency, blowout tilt, value anchors (8 tests)
TestPerGameBuildLineups     — _build_game_lineups() returns strategy, 5 players, valid slots, blowout 4-1 split (6 tests)
TestPerGameConfig           — per_game section in _CONFIG_DEFAULTS, all 20 keys present, score bounds widened (3 tests)
TestPerGameFrontend         — strategyInsight element, render function, ANCHOR/FAV pills, back hides insight (5 tests)
```

**tests/test_core.py** — Helpers, JS syntax, date-boundary regressions, and contract guards:

- **TestHelpers** — _et_date, _is_locked, _est_card_boost, cache roundtrip
- **TestJSSyntax** — unescaped apostrophes in single-quoted strings; presence of renderCards, loadSlate, switchTab; _etToday / _predSavedDate
- **TestCacheDateBoundary** — cache keys consistent with ET date
- **TestBenBannerActualsDetection** — ACT_FIELDS / actuals-shaped CSV parsing (audit inputs)
- **TestBannerGuardJS** — banner-visibility check in showLabUnlocked() uses correct localStorage keys
- **TestNormalizePlayer** — _normalize_player() contract: all required fields present with correct types
- **TestRealScoreEngine** — Monte Carlo closeness/clutch/momentum coefficients (pure numpy, no I/O)
- **TestAssetOptimizer** — optimize_lineup() MILP slot assignment: chalk vs moonshot modes, edge cases, RS-ordered slotting, two-phase moonshot, same-position-same-team allowed
- **TestConfigCoverage** — all major model floors read from model-config.json via _cfg()
- **TestProjectPlayerContract** — project_player() returns all required fields after _normalize_player()
- **TestLogGetNormalization** — log_get() builds player cards from CSV rows with correct field mapping (backend-only endpoint)
- **TestUpdateConfigValidation** — /api/lab/update-config accepts dot-notation keys, rejects invalid paths
- **TestFrontendAuditFixes** — regression guards for frontend null guards and .ok checks

Run all: `pytest tests/ -v`
Run fast subset only: `pytest tests/test_fixes.py -v`  
Note: Tests that import `api.index` require dependencies (e.g. numpy, lightgbm). Use `pip install -r requirements.txt` before running.

## Known Limitations

- Rate limiting uses an in-memory store with a lock (thread-safe); it does not persist across container restarts, so limits reset on redeploy.
- `/tmp` is cleared on container restart (redeploy or crash) — caches don't survive restarts. GitHub-persisted caches (`data/slate/`, `data/locks/`) provide cold-start recovery for both predictions and lock state.
- **Concurrent write conflicts (mitigated)**: `_github_write_file` implements exponential backoff (1s, 2s, 4s retries) to handle HTTP 422 SHA mismatches. Overlapping cron + API writes are handled; conflicts are rare.
- Historical slate/lock cache files can create operational noise. Keep `data/slate/` and `data/locks/` pruned to active-slate artifacts.
- Fetch timeouts: All frontend calls have hard limits (10s default, 30s screenshot). Exception: `/api/lab/chat` uses a raw streaming fetch (SSE) by design — no timeout on the stream body, only on connection.
- Upload screenshot type validation is client-side trust only — the system cannot verify that a "Real Scores" button upload actually contains a Real Scores screenshot. Wrong uploads produce skewed audit data for that date.

## Troubleshooting

If slate fails to load:
1. **Deployed URL** — Use the production URL (`https://the-oracle.up.railway.app`); avoid file:// or wrong origin.
2. **Health and version** — Call `GET /api/health` and `GET /api/version`; if they fail, the backend is unreachable or cold-starting.
3. **Railway logs** — Look for `[slate] error:`, `[games] error:` or "Task timed out" to identify the failing path.

## Robustness Fixes (this session)

| Fix | File | Detail |
|-----|------|--------|
| Stale response guard on game switching | `index.html` | `runAnalysis()` captures `gameId` at call time; discards response if selector changed mid-flight |
| `fetchWithTimeout` on `/api/picks` | `index.html` | Was raw `fetch`, now 15s timeout |
| `fetchWithTimeout` on `/api/audit/get` | `index.html` | Was raw `fetch`, now 10s timeout |
| `fetchWithTimeout` on `top_drafts` save | `index.html` | Was raw `fetch`, now 10s timeout |
| Upload banner: hide completed buttons | `index.html` | On reload, done buttons hide immediately; on new upload, flash green → hide after 1.5s |
| Upload banner: X/4 progress counter | `index.html` | Title updates live as each upload completes |
| `_checkBannerDone` uses localStorage | `index.html` | DOM disabled state unreliable after hide; localStorage is source of truth |
| Audit gate on `real_scores` | `api/index.py` | `save-actuals` only generates audit JSON when `real_scores` rows present |
| Dead code pruned | `index.html` | Removed empty `_renderBenEodPrompt()` function |
| Ben historical upload banner removed | `index.html` | Ingestion is developer-only (`docs/HISTORICAL_DATA.md`); `skip-uploads` API retained |
| Rate-limit thread-safety | `api/index.py` | `_RATE_LIMIT_LOCK` wraps read-modify-write of `_RATE_LIMIT_STORE` so concurrent requests are safe |
| Predict tab next-day transition | `index.html` | `loadSlate()` busts stale previous-day predictions; Predict tab no longer stuck on finished slate after midnight rollover |
| Ben briefing timeout precision | `index.html` | `initLabPage` error-fallback briefing and `showLabUnlocked` context load raised to 30s; auto-retry `/api/lab/status` standardized to 10s (was 15s). Context-load paths get 30s; user-triggered refresh actions stay at 10s for responsiveness. |
| `_CONFIG_DEFAULTS` sync | `api/index.py` | Fallback defaults match `data/model-config.json`: `compression_divisor` 5.5, `compression_power` 0.72, `rs_cap` 20.0, `ai_blend_weight` 0.35, `per_player_cap_minutes` 2.0, `big_market_teams` inline fallback removes MIL/DAL/PHX. Prevents silent model behavior change on GitHub outage. |
| `auto_improve_threshold_pct` externalized | `api/index.py`, `data/model-config.json` | `IMPROVEMENT_THRESHOLD` reads from `_cfg("lab.auto_improve_threshold_pct", 3.0)`. Tunable via Ben without code deploy. |
| 3-layer slate cache (generate once per day) | `api/index.py` | `/tmp` → GitHub `data/slate/` → full pipeline. First request generates and persists; all subsequent requests serve from cache. Reduces API calls from N per visit to ~6-8 per day |
| `/api/injury-check` cron endpoint | `api/index.py`, `railway.toml` | Every 2h: bust RotoWire cache, check cached players, regenerate only affected games. Lock-guarded, CRON_SECRET-protected |
| Force-regenerate endpoint | `api/index.py`, `index.html` | `GET /api/force-regenerate?scope=full\|remaining` — two scenarios: (1) dev deploys mid-slate → auto-detects SHA mismatch, regenerates all games in background; (2) user wakes up late → "Late Draft" banner on Predict tab regenerates picks for remaining games only. Both update `data/predictions/` CSV and all cache layers. CRON_SECRET-gated. |
| Deploy SHA tracking | `api/index.py` | `deploy_sha` stamped in slate cache at generation + GitHub write time. `/api/slate` locked path compares cached SHA vs current `RAILWAY_GIT_COMMIT_SHA`; on mismatch fires background `_force_regenerate_sync("full")`. |
| Late Draft UI | `index.html` | Banner with "Generate Late Draft" button shown on Predict tab when slate is locked but remaining games exist. Calls `/api/force-regenerate?scope=remaining`, updates SLATE, re-renders, and hides banner on success. |
| `var` → `let` modernisation | `index.html` | Converted `_slateAutoRefreshCount`, `_slateAutoRefreshTimer`, `_slateNextDayPoll`, `_predSavedLockedCount`, `_lateDraftTriggered` from `var` to `let` for block-scope consistency with the rest of the module. |
| `oracle-ball.svg` cache header | `railway.toml` | Added `Cache-Control: public, max-age=86400` entry for `/oracle-ball.svg` to match the existing `server.py` header and keep cache strategy consistent across Railway and local dev. |
| Grep tags for key helpers | `api/index.py` | Added `# grep:` tags for `LOCK HELPERS` (`_is_locked`, `_is_past_lock_window`, `_et_date`), `ALL GAMES FINAL`, `NEXT SLATE DATE` (`_find_next_slate_date`), and `FORCE REGENERATE SYNC`. Updated CLAUDE.md navigation table to match. |
| `force-regenerate` scope=remaining unprotected | `api/index.py` | `/api/force-regenerate?scope=remaining` no longer requires CRON_SECRET — it's user-triggered from the Late Draft button. `scope=full` stays CRON_SECRET-gated. |
| Vercel → Railway migration | `CLAUDE.md`, `api/index.py` | All Vercel references replaced with Railway equivalents (deployment model, env vars, cron schedule, URLs, watchPatterns). `VERCEL_GIT_COMMIT_SHA` → `RAILWAY_GIT_COMMIT_SHA` at all call sites. `vercel.json` noted as legacy/unused. |
| Per-game card boost pill removed | `api/index.py` | `_build_game_lineups` now zeroes `est_mult` in returned player data (not just MILP input) so the `+X.Xx card` pill never renders on per-game (THE LINE UP) cards where card boost is irrelevant. |
| `predMin` tolerance band | `api/index.py`, `data/model-config.json` | Chalk pool allows `predMin` up to 2.0 min below `season_min` (`projection.pred_min_tolerance`); moonshot allows 3.0 min (`moonshot.pred_min_tolerance`). Saved 4 missed players from Mar 17 (Carrington 1.5 gap, Jenkins 0.7, Riley 0.8, Champagnie 1.8). |
| Separate moonshot pts floor | `api/index.py`, `data/model-config.json` | Universal floor in `project_player()` lowered to 4.0 (`min_pts_projection_moonshot`); chalk enforces 7.0 separately. Oso Ighodaro (4 PPG, +2.9x, Value 16.4) now enters moonshot pool. `min_pts_per_minute_moonshot` = 0.20 (chalk keeps 0.28). |
| `min_chalk_rating` synced | `data/model-config.json` | Config value 4.0 → 3.5 to match code fallback and CLAUDE.md documentation. Mar 17 showed 7/9 missed players filtered by this gate. |
| Web intelligence (Layer 1) | `api/index.py`, `data/model-config.json` | `_fetch_nba_news_context()` — once-per-slate Sonnet call (`context_layer.web_search_model`, default `claude-sonnet-4-6-20250514`) with `web_search_20250305`. Downgraded from Opus — news gathering is search+summarize, doesn't need Opus reasoning. Player/RS-aware: when `all_proj` is passed, top 20–25 by rating are included so intel prioritizes likely draft picks. Results injected into context pass as "RECENT NBA NEWS". Config: `context_layer.web_search_enabled`, `context_layer.web_search_model`, `timeout_seconds`. |
| Context pass (Layer 2) | `api/index.py`, `data/model-config.json` | RS adjustment uses Sonnet (`context_layer.model`, default `claude-sonnet-4-6-20250514`). Downgraded from Opus — structured JSON task (RS multipliers) handled well by Sonnet at ~15x lower cost. Directive: map each RECENT NBA NEWS bullet to specific players and up/down adjustments. |
| Lineup review (Layer 3) | `api/index.py`, `data/model-config.json` | `_lineup_review_opus()` — after MILP, Sonnet + web_search reviews assembled Starting 5 and Moonshot; can suggest swaps for late-breaking news; auto-applies valid swaps. Config: `lineup_review.enabled` (default off), `lineup_review.model`, `lineup_review.timeout_seconds`. Non-fatal: on error returns original lineups. |
| Claude cost reduction | `api/index.py`, `data/model-config.json` | Layer 1 (news) and Layer 2 (context pass) downgraded from Opus → Sonnet. Layer 1.5 (Claude DvP matchup intel) disabled — ESPN def stats in `_compute_matchup_factor()` provide equivalent signal. ~95% cost reduction on pipeline Claude calls ($5-12/slate → $0.25-0.50/slate). All config-reversible. |
| Health check timeout + Vercel cleanup | `index.html` | Health pre-warm converted from raw `fetch()` to `fetchWithTimeout(..., 5000)`. Stale Vercel references in comments updated to Railway. All frontend fetches now use `fetchWithTimeout` (except lab/chat SSE which uses manual AbortController). |
| Core pool architecture | `api/index.py`, `data/model-config.json`, CLAUDE.md | Single up-to-15 player core pool; Starting 5 and Moonshot are two 5-of-core configurations (reliability vs ceiling). Config: `core_pool.enabled`, `core_pool.size`, `core_pool.metric` (`"rs"` ranks by raw projected RS). Layer 2/3 prompts and Layer 3 swap-in respect core pool. |
| RS-first strategy (v8) | `api/index.py`, `data/model-config.json` | Strategy shift: top RS scorers over everything. (1) `core_pool.metric="rs"` ranks core by raw projected RS, not EV. (2) `chalk_milp_rs_focus=0.85` nearly neutralizes boost in MILP — RS drives slot assignment. (3) `moonshot.rs_bypass` lets high-RS players (5.0+, 25min+) bypass boost floor. (4) `boost_leverage_power` 1.2→0.6 halves boost dominance. (5) Boost floors lowered: chalk 1.0→0.5, moonshot 1.0→0.5. (6) `star_anchor.max_count` 2→3, `min_boost` 0.8→0.3. (7) `ai_blend_weight` 0.4→0.5 for better RS ordering. All config-reversible. |
| 13-date accuracy audit fixes | `api/index.py`, `data/model-config.json` | RS calibration_scale=1.15 (34% under-projection), AI blend 0.5→0.35 (LightGBM compression), 3-layer card boost (config→170 ownership samples→sigmoid), moonshot gates widened (min_minutes 12, wildcard 6, pts 3.0). |
| Layer 1→3 news passthrough | `api/index.py` | `_lineup_review_opus()` now accepts `news_context` param. Main slate generation pre-fetches news and passes to Layer 3, reducing redundant web searches. Layer 3 focuses on truly late-breaking news (last 2-4h). |
| `.ok` check before `.json()` | `index.html` | `/api/lab/update-config` response: check `.ok` before parsing JSON to prevent misleading error handling. |
| Vercel→Railway comment cleanup | `api/index.py` | Replaced 7 stale Vercel references with Railway equivalents (watchPatterns, container instances, timeout limits). |
| MILP solver audit — 3 fixes | `api/asset_optimizer.py`, `api/index.py` | (1) Removed `leverage_top_slots` constraint — mathematically wrong for additive formula `RS × (Slot + Boost)` since boost is player-constant; solver naturally places highest RS in highest slot. (2) Two-phase moonshot optimization — Phase 1 selects players using shaped ratings (boost leverage, variance uplift); Phase 2 re-assigns slots using raw RS for optimal placement. Decouples selection from slotting. (3) Removed position-per-team constraint — Real Sports has no position requirements; artificial constraint blocked legitimate same-position stacks. |
| High-boost role pathway | `api/index.py`, `data/model-config.json` | Rotation players with 2.0x+ boost bypass minutes floor in both moonshot and chalk pools (`is_high_boost_role`, `is_chalk_high_boost_role`). Config: `moonshot.high_boost_role.*`, `projection.chalk_hbr_*`. |
| RS calibration weights | `api/index.py`, `data/model-config.json` | DFS weight recalibration from 13-date audit (reb 0.95→0.65, ast 0.3→0.55). `_infer_player_archetype()` detects star/scorer/big/pure_rebounder/wing_role. `archetype_calibration` applies per-archetype RS multipliers. `scorer_upside` gives efficient scorers moonshot bonus. |
| Cascade cap fix | `api/index.py`, `data/model-config.json` | `cascade.per_player_cap_minutes` raised from 2-3 to 10.0 so primary backups correctly inherit starter-level minutes. `cascade_rs` and `role_spike_rs` add RS uplift for cascade-elevated players. |
| Roto confirmed rating exception | `api/index.py`, `data/model-config.json` | Confirmed rotation players with high boost (2.5x+) bypass `min_rating_floor` in moonshot (use 2.2 floor instead). Context pass includes cascade_bonus and roto_status. |
| Max per game MILP constraint | `api/asset_optimizer.py`, `api/index.py`, `data/model-config.json` | `lineup.chalk_max_per_game=2`, `moonshot_max_per_game=2` — limits players from same game matchup. Prevents over-concentration in single games. |
| Min big boost MILP constraint | `api/asset_optimizer.py`, `api/index.py`, `data/model-config.json` | `lineup.chalk_min_big_boost_count=1`, `moonshot_min_big_boost_count=1` — ensures minimum high-boost players in lineup for card boost value. |
| MAE drift check cron | `api/index.py`, `railway.toml` | Weekly (Monday 6am UTC) MAE drift monitoring — computes 7-day rolling MAE, writes backend flag if > 2.5 threshold. CRON_SECRET-gated. |
| Ben chat history | `api/index.py` | `/api/lab/chat-history` — persisted daily chat history with thread-safe read via `_BEN_CHAT_HISTORY_LOCK`. |
| Per-game draft strategy redesign (v60) | `api/index.py`, `data/model-config.json`, `index.html` | 18-game / 76-lineup empirical analysis (Jan 6 – Mar 23). 6-step pipeline: game script → per-game strategy adjustments (F1-F6) → eligibility gating → MILP → 5! permutation validation → strategy metadata. New functions: `_per_game_strategy()`, `_per_game_adjust_projections()`, `_validate_slot_assignment()`. Config: `per_game.*` (20 params). Frontend: strategy insight bar, ANCHOR/FAV pills, color-coded strategy badge. Strategy types: Balanced Build / Standard Build / Blowout Lean + Shootout/Grind overlays. Score bounds widened to (20, 42). 38 new tests. |
| Mar 27 draft review — S5 coverage + RS inflation fix | `api/index.py`, `data/model-config.json`, `tests/test_fixes.py` | **Problem**: Mar 27 slate — moonshot hit 1/5 (Jenkins #1, but Sasser/McDermott/Alvarado/Carter all missed). S5 blocked top RS performers (Duren +0.6x, Knueppel +0.8x, DeRozan +1.0x) via 1.5x chalk boost floor. RS over-projection: 4 stacked post-compression multipliers (cascade_rs × role_spike × breakout × archetype = 1.87×) inflated Sasser from RS 3.0 to 6.1. **Fix 1**: `chalk_min_boost_floor` 1.5→0.3, star anchor widened (min_season_pts 20→12, min_rating 4.5→3.5, max_count 2→3) so top RS performers enter chalk pool and MILP (rs_focus=0.75) selects them. **Fix 2**: Post-compression multiplier cap (`max_post_compression_mult: 1.40`) — tracks `_pre_boost_rs` before archetype/cascade/spike/breakout boosts; clamps total inflation to 1.40×. Prevents bench players from inflating from RS 3 to 6+. **Fix 3**: HBR `min_rating: 2.5` (already in config+code from prior session). **Goal**: S5 catches top RS carries (Duren, Knueppel, DeRozan, Banchero), Moonshot catches one-slate-ahead contrarians (Jenkins, Plowden, Huerter); 2-3 "lock" players overlap in both lineups. |

| 3-Tier Cascade Boost Prediction | `api/boost_model.py` (new), `api/index.py`, `tests/test_fixes.py`, `tests/test_core.py` | **Architecture**: Replaced LightGBM `boost_model.pkl` + `drafts_model.pkl` with deterministic 3-tier cascade calibrated from 2,234 player-date records. **Tier 1** (returning, ≤14d): prev_boost + 6 adjustment factors (RS decay, draft popularity, mean reversion, trend, gap blend, boundary persistence). **Tier 2** (stale, >14d): staleness-weighted blend of historical mean and API-derived PQI estimate. **Tier 3** (cold start): Player Quality Index from season stats. Post-prediction star PPG caps and per-team ceilings. Key insight: prev_boost correlates +0.957 with actual boost; 88.2% of day-over-day changes are within ±0.3. Removed: `_ensure_boost_model_loaded()`, `_lgbm_predict_boost()`, `_ensure_drafts_model_loaded()`, `_lgbm_predict_log1p_drafts()`, `_ensure_boost_priors_loaded()`, `_get_boost_prior()`, `BOOST_MODEL/FEATURES` globals, `DRAFTS_MODEL/FEATURES` globals. Updated `_CONFIG_DEFAULTS.card_boost` (ceiling 3.5→3.0, floor 0.2→0.0, removed `ml_additive_correction`/`max_prior_weight`, added `star_ppg_tiers`/`team_boost_ceiling`). 17 new tests (13 cascade + 4 integration). 693 tests pass. |

| Cascade team detector + deep rotation | `api/index.py`, `data/model-config.json`, `tests/test_fixes.py` | Analysis of 2,299 top performer entries across 151 dates. **Cascade Team Detector**: star (20+ PPG) OUT → flag all teammates with `_cascade_team=True` → RS multiplier 1.3x + boost floor 2.5. **Deep Rotation Sweet Spot**: cascade team players get relaxed gates (RS floor 1.5 vs 2.0, min_gate 12 vs 25, rotation-bubble filter bypassed). The 5-20 draft archetype has the highest historical avg value (16.1). **Proportional cascade cap**: `max_cascade_pct=0.40` prevents bench players from inflating to 26+ projected minutes. Config: `cascade.team_detector.*`. 9 new tests. |
| Cascade config sync + proportional cap | `api/index.py`, `data/model-config.json`, `tests/test_fixes.py` | Fixed `data/model-config.json` missing cascade params (dtd_sit_probability, partial_cascade_cap_minutes, etc.) — `_cfg()` was falling through to old hardcoded fallbacks. Added `max_cascade_pct=0.40` proportional cap: cascade bonus cannot exceed 40% of player's avg minutes. 3 updated tests. |
| Historical RS confidence discount | `api/index.py` | **Bayesian RS regression** in `project_player()`: cross-references predicted RS against player's actual historical RS distribution from `top_performers.csv`. When predicted RS exceeds P75, applies soft pull-back weighted by history depth (more appearances = stronger prior). NOT a hard cap — players can still pop off, but extreme over-projections are tempered. Prior strength saturates at ~15 appearances via `n/(n+k)`. Example: Sensabaugh predicted 6.4 RS with median 3.3 → discount of ~0.5 RS. Config: `strategy.historical_rs_discount.*` (6 params). |
| Momentum curve detection | `api/index.py` | **Hype trap penalty + rising wave bonus** in `_build_lineups()` EV scoring. Loads player history from `boost_model.load_player_history()`. **Hype trap**: drafts exploding (200%+) AND boost declining (0.5+) = player peaked → up to 20% EV penalty. Catches Sensabaugh-type players (2→207 drafts, boost 3.0→2.0). **Rising wave**: RS trending up (0.3+) AND low drafts (<200) AND high boost (≥2.0) → up to 20% EV bonus. Catches Fears/Hawkins-type players coming up the curve. Config: `strategy.momentum_curve.*` (9 params). |

## Loading audit

**docs/LOADING_AUDIT.md** — Catalogs frontend loading states, fetch timeouts, skeletons, async state pattern. All blocking API calls use `fetchWithTimeout`; no critical gaps for production.

| Apr 8 post-mortem — EV formula fix + high-boost bypass + RS discount | `api/index.py`, `data/model-config.json`, `tests/test_fixes.py`, `tests/test_core.py` | **Problem**: Apr 8 slate — chalk 0/5, moonshot 2/5 (Clayton, Dieng). ALL winning players had 3.0x boost with <25 drafts. Bones Hyland predicted RS 5.0, actual 0.5 (catastrophic). Keon Ellis predicted RS 5.0, actual 0.9. Winning draft (73.96) = 5 players with 3.0x boost: Clayton, Dieng, Sims, Shannon Jr, Hendricks. **Fix 1**: `avg_slot_multiplier` 1.6→2.0 — EV formula was `RS × (1.6 + boost)` instead of documented `RS × (2.0 + boost)`, undervaluing boost by ~8%. **Fix 2**: High-boost bypass — players with predicted boost ≥2.5 AND RS ≥2.0 now bypass the min_minutes gate (uses 12 min like cascade teams). Deep bench 3.0x contrarians (Shannon, Bitadze, Sims, Hendricks, Bryant) were ALL filtered by 25-min gate. **Fix 3**: `min_minutes` 25→15 (global relaxation). **Fix 4**: Historical RS discount strengthened — `min_appearances` 3→2, `saturation_k` 8→6, `max_prior_strength` 0.5→0.65, `discount_scale` 0.4→0.6, `max_discount_frac` 0.6→0.8. Prevents Hyland-type 5.0→0.5 busts. **Fix 5**: `anti_popularity_enabled` true + `strength` 0.2 in model-config.json (was disabled). Config v91. |

| Apr 9 pipeline + picks quality overhaul | `api/index.py`, `data/model-config.json`, `tests/test_fixes.py` | **Problem**: Pipeline not running (empty `{}` slate cache, unprotected exception in `_run_cold_pipeline`), and 0/5 picks hitting on recent slates (Apr 5–8: ALL winners had boost 2.9–3.0 with <25 drafts; model predicted none of them). **6 fixes**: (1) **Pipeline crash fix**: wrapped unprotected `_prewarm_current_slate_sync()` call in `_run_cold_pipeline` with try/except — was silently crashing the entire cold pipeline, leaving empty tombstones. (2) **EV formula consistency**: `_build_lineups()` code default was 1.6 instead of config value 2.0 for `avg_slot_multiplier` — boost undervalued by ~8% in lineup selection relative to `project_player()`. Fixed default to 2.0. (3) **Anti-popularity 3x multiplier removed**: hidden `* 3.0` on line 3024 made `anti_popularity_strength: 0.2` effectively 0.6 — way too harsh, over-penalizing popular players' boost predictions. Now strength is directly applied. (4) **Rising wave tuning**: `rs_trend_min` 0.3→0.15 (slow-building players qualify), `rising_wave_max_bonus` 0.20→0.35, `wave_max_drafts` 200→300. (5) **Moonshot from full pool**: was excluding S5 players from moonshot pool, preventing best EV players from appearing in both lineups. Now moonshot selects independently — 2-3 overlap expected (matches winning draft patterns). (6) **ai_blend_weight config drift**: was 0.3 in config but v82 changelog says 1.0 (100% LightGBM). DFS heuristic was causing catastrophic RS over-projection for role players (Trent Jr. projected 5.7, actual 1.7). Fixed config + code default to 1.0. |

| Condition Matrix as primary signal | `api/index.py`, `api/real_score.py`, `data/model-config.json`, `tests/test_fixes.py` | **Problem**: Condition Matrix (ownership × boost → HV rate) existed in `api/real_score.py` but was disconnected from `_build_lineups()`. Pipeline used `EV = RS × (2.0 + boost)` — trait-based scoring with format bonus as afterthought. Historical data shows format mechanics dominate: ghost+max_boost=100% HV rate, mega_chalk+max_boost=12%. **Fix**: Wired `condition_coefficient()` into `_build_lineups()` as multiplicative EV factor. Draft counts estimated from player history (primary) or `estimate_draft_popularity()` (fallback). Dead capital combos (condition_coeff=0.0) auto-filtered from candidate pool. New formula: `composite_ev = condition_coeff × RS × (2.0 + boost) × data_multipliers`. No positional caps removed (already absent). Max 1 per team enforced for both S5 and Moonshot independently. Config: `strategy.condition_matrix.enabled` (default true). 14 new tests. |

## Production audit

Full audit: [docs/PRODUCTION_AUDIT.md](docs/PRODUCTION_AUDIT.md). Implemented: GitHub error sanitization (no leak to client), `GET /api/health`, `GET /api/version`, cron secret on protected endpoints (including `/api/cold-reset`), and `fetchWithTimeout` for lab/backtest and lab/update-config.

**Lock & routing audit:** [docs/LOCK_AND_ROUTING_AUDIT.md](docs/LOCK_AND_ROUTING_AUDIT.md). Covers all lock usage (slate, picks, save-predictions, lab status) and Railway/FastAPI routing. Fixes applied: `/api/lab/status` wrapped in try/except — on any exception returns 200 with `locked: true` and reason "Server temporarily unavailable — try again" so the frontend shows a retry instead of a generic fetch failure; ESPN-down GitHub lock check now uses `lock_content, _ = _github_get_file(...)` and `if lock_content:` (was incorrectly checking the tuple).

### Mar 17 Production & Model Audit

**Pipeline audit (179/179 tests pass):**
- Global exception handler active — no stack traces leak to clients
- Structured request logging (JSON with request_id, path, status, duration_ms)
- 39 `fetchWithTimeout` calls in frontend; 1 intentional raw `fetch()` (lab/chat SSE with manual AbortController)
- Thread pools: 8 workers (game/slate/picks/audit)
- Rate limiting: thread-safe with `_RATE_LIMIT_LOCK` (parse-screenshot 5/min, lab/chat 20/min)
- 35 endpoints total, all correctly routed with proper CRON_SECRET gating

**Caching audit (all TTLs verified):**
- 3-layer slate cache: `/tmp/nba_cache_v19/` → GitHub `data/slate/` → full pipeline
- Game final: 60s locked / 180s pre-slate; Model config: 5 min; ESPN games: 5 min; RotoWire: 30 min
- Cache bust tombstone pattern working correctly

**Model audit (Mar 17 leaderboard — "Highest Value" screenshot):**
- Role players dominate (13/14 top values RS 2.7-5.0) — validates v6 strategy
- Boost is dominant signal (+3.0x at RS 2.7 → Value 13.3+) — validates moonshot formula
- Stars don't win (Booker RS 5.5, +0.5x, 552 drafts = 13.7 value, ranked 11th)
- Player overrides systematically low by 0.1-0.2x vs actual (GP2: 2.8 override vs 3.0 actual; Riley: 2.8 vs 3.0; Santos: 2.6 vs 2.5 — this one is accurate)
- `min_chalk_rating` 4.0 correctly filters — all chalk winners RS ≥ 4.1
- `min_pts_per_minute` 0.28 well-calibrated — GP2 (~0.29) passes as intended
- Claude context layer correctly identified defensive role players (Draymond, GP2, Melton) as top performers

## Pre-deploy checklist (production finalization)

- **Env**: Required vars set in Railway (GITHUB_TOKEN, GITHUB_REPO, ANTHROPIC_API_KEY; optional CRON_SECRET, DOCS_SECRET).
- **Tests**: Run `python3 -m pytest tests/ -v` locally when changing backend contracts. `TestJSSyntax` is skipped (superseded by TypeScript compiler). For frontend: `cd frontend && npx tsc --noEmit`.
- **Docs**: CLAUDE.md and README.md reflect current endpoints, crons, lock/cache behavior, and core-pool architecture; docs/LOADING_AUDIT.md for loading and timeouts.
- **Health**: Use GET `/api/health` for uptime monitoring; alert on non-200.
- **Loading**: All blocking fetches use `fetchWithTimeout`.

## Development

```bash
# Local backend (Terminal 1)
pip install -r requirements.txt
python scripts/check-env.py        # verify required env vars (fail-fast)
uvicorn server:app --reload --port 8080

# Local frontend (Terminal 2)
cd frontend
npm install                         # first time only
npm run dev                         # Vite on :5173, proxies /api → :8080

# TypeScript check
cd frontend && npx tsc --noEmit

# Production build (what Railway runs)
cd frontend && npm run build        # outputs frontend/dist/

# Deploy — push to main → Railway detects watchPatterns change → rebuilds Docker container
git push -u origin <your-branch>
# Then open PR main ← your-branch and merge, or: git checkout main && git merge <your-branch> && git push origin main

# Verify on production
# https://the-oracle.up.railway.app
```

## Starting a New Claude Code Session

When starting fresh in a new chat, Claude Code automatically reads this file for context.
Provide the following to the new session to orient it quickly:

1. **Branch**: Work on a feature branch; merge to `main` via PR or local merge + push. Railway auto-deploys from `main` when watchPatterns match.
2. **Stack**: FastAPI backend (`api/index.py`) + React + Vite + TypeScript frontend (`frontend/`). Legacy `index.html`/`app.js`/`styles.css` remain as server.py fallback until React app is production-verified.
3. **Tests**: `pytest tests/ -v` (requires `pip install -r requirements.txt`). test_fixes.py covers lock/audit/cache; test_core.py covers helpers (`TestJSSyntax` is skipped — superseded by `cd frontend && npx tsc --noEmit`). Deploy triggers on push to main; verify on `the-oracle.up.railway.app`.
4. **Data layer**: All persistent state in GitHub via Contents API (`data/` directory). No database.
5. **Frontend state**: Zustand stores in `frontend/src/store/` (`uiStore`: activeTab, etc. | `labStore`: messages, system). React Query hooks in `frontend/src/api/` for all server state.
6. **Cache**: 3-layer: `/tmp` (ephemeral) → GitHub `data/slate/` (persistent) → full pipeline. Check `CACHE_DIR` in `api/index.py` for the current tmp path (versioned, e.g. `/tmp/nba_cache_v19/`). `/api/cold-reset` clears/regenerates caches + config. `_bust_slate_cache()` invalidates both layers.
7. **Config**: `data/model-config.json` on GitHub — Ben/Lab writes here, backend reads with 5-min TTL.
