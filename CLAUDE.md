# Basketball — Real Sports Draft Optimizer

**Document Status:** Current Reference

## What This Is

A daily NBA draft optimizer for the **Real Sports** app. Uses a **Strategy-Report-Aligned Architecture**: LightGBM for RS projection, a 3-tier deterministic cascade for card boost prediction, and a sort-based lineup builder using `EV = RS × (2.0 + boost)`. The system follows findings from a 90-date historical analysis showing boost is 40% more valuable per unit than RS, slot assignment is a simple RS sort, and anti-popularity provides a 24% value edge. Prop betting surfaces (Line of the Day + Parlay) are powered by a deterministic `fair_value` engine. Deployed on **Railway** as a Dockerized Python (FastAPI) backend + single-page HTML frontend.

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
  src/types/           — TypeScript interfaces (slate, line, parlay, lab, common)
  src/api/             — React Query hooks: useSlate, useLineOfTheDay, useParlay, useLabBriefing, …
  src/store/           — Zustand state: uiStore (activeTab, lineDir, …), labStore (messages, system)
  src/hooks/           — useAbortOnTabSwitch, useEventSource, useEtDate, useKeyboardNavHide
  src/components/      — Predict | Line | Parlay | Ben tabs + shared (PlayerCard, OracleLoader, …)
  src/styles/          — tokens.css, global.css, animations.css (CSS Modules per component)
  vite.config.ts       — Vite: proxy /api → localhost:8080
  Dockerfile           — Multi-stage: Node build (npm run build) → Python runtime (COPY dist)
index.html             — Legacy vanilla JS frontend (fallback when frontend/dist/ absent)
api/index.py           — FastAPI backend (all endpoints, projection engine, Lab/Line/Parlay)
api/fair_value.py      — Deterministic fair-value engine for prop betting (Line + Parlay only; pure functions)
api/odds_math.py       — Shared American-odds implied probability helpers
api/real_score.py      — Monte Carlo Real Score engine (retained for reference; not used by draft pipeline)
api/asset_optimizer.py — MILP lineup optimizer (PuLP, used by per-game pipeline only)
api/line_engine.py     — Prop edge detection pipeline (Odds API + confidence model)
api/parlay_engine.py   — Safest 3-leg parlay optimizer (Z-score + correlation scoring)
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
data/lines/            — Git-tracked daily Line of the Day picks (via GitHub API)
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
3. **Draft Lineup Builder** (`_build_lineups` in `api/index.py`): Sort-based pipeline replacing the prior MILP optimizer for slate-wide drafts. Single EV formula: `RS × (2.0 + boost)`. Two gates only: RS ≥ 2.0, minutes ≥ 12. Safe lineup = top 5 by EV (tie-break low variance). Upside lineup = 1-2 variance/boost swaps. Slot assignment = sort by RS descending (provably optimal per strategy report Finding 3).

**Historical outcomes** for audit: **`data/top_performers.csv`** is primary (filter by `date`); **`data/actuals/{date}.csv`** remains a transition fallback. **Simplest ingest:** **`docs/historical-ingest/INSTRUCTIONS.md`** — rasterize PDF → transcribe PNGs → write `data/` (no server). Alternate: **`docs/HISTORICAL_DATA.md`** (API `parse-screenshot` + `save-*` POSTs if you prefer). `data/predictions/` supplies pre-game features for training joins.

**2025-26 data coverage:** Oct 21 – Nov 29 ✅ | Nov 30 – Jan 16 ✅ (gap closed) | Jan 17 – Feb 11 ✅ | Feb 12–18 All-Star break | Feb 19 – Mar 25 ✅. Dec 24 = no games. Full map: `docs/HISTORICAL_DATA.md`.

The deterministic fair-value engine remains isolated to prop betting surfaces.

### Engine 1: Strategy-Report-Aligned Draft Pipeline (Starting 5 + Moonshot)
- **Pipeline**: ESPN → LightGBM + heuristic blend → compression → simple game context (+0.3 close, +pace) → Card Boost (cascade + anti-popularity) → `EV = RS × (2.0 + boost)` → Sort-based selection → RS-descending slot assignment
- **Why**: Strategy report (90 dates) proves boost is 40% more valuable per unit than RS. The game is structurally solvable with `RS × (2.0 + boost)`. Monte Carlo and MILP were over-engineering a problem that data shows is simple. Anti-popularity captures a 24% value edge from going contrarian.
- **Endpoints**: `/api/slate`, `/api/picks`, `/api/force-regenerate`, `/api/injury-check`
- **Code**: `project_player()` and `_build_lineups()` in `api/index.py`, `api/boost_model.py`

### Engine 2: Deterministic `fair_value` → Prop Betting (Line + Parlay)
- **Pipeline**: ESPN gamelogs (L10/L15 rolling windows) → DvP adjustment → Game script weights → Spread adj + momentum → Per-stat fair value + Z-score hit probabilities + EV classification
- **Why**: Prop bets reward median accuracy and floor stability. An over/under bet cares about the most likely stat line, not the ceiling. Rolling window medians with Normal CDF hit probs produce tighter Z-scores.
- **Endpoints**: `/api/line-of-the-day` (edge_map → fv_boost), `/api/parlay` (_fv_hit_probs → model_prob override)
- **Code**: `api/fair_value.py` (pure functions, no I/O), `_compute_betting_fair_value()` in `api/index.py`

### Isolation Boundary
- `_compute_betting_fair_value()` is the **sole entry point** for Engine 2. It is called only from `_run_line_engine_for_date()` and `_run_parlay_engine_sync()`.
- DFS draft paths (`/api/slate`, `/api/force-regenerate`, `/api/injury-check`) **never** invoke the fair value engine.
- Engine 1's `all_proj` feeds both surfaces; fair value adds per-stat edge maps and hit probabilities for prop betting on top.
- Config: `fair_value.*` section in `data/model-config.json` (primary_window, short_window, edge_thresholds, compression).

### grep tags
```
grep: FAIR VALUE BETTING        — _compute_betting_fair_value(), sole entry point for Engine 2
grep: FAIR VALUE ENGINE         — project_player_fv in api/fair_value.py
```

## UI Structure

4-tab segmented control navigation (Apple glassmorphism pill style): **Predict | Line | Parlay | Ben**

- **Predict**: Live slate optimizer (Starting 5 + Moonshot) and per-game analysis ("THE LINE UP" — single 5-player format, no card boost). "Slate-Wide | Game" sub-tabs inline at top of tab. Magic 8-ball loading animation.
- **Line**: Line of the Day — best player prop edge (gold accent). "Over | Under" sub-tabs inline at top of tab. Bookmaker odds sync on a **game-window cron** (see `railway.toml`); pick cards show "Odds · [time] CT". **Recent Picks** history section with streak + hit rate.
- **Parlay**: Safest 3-leg player prop parlay (electric purple accent `--parlay: #d946ef`). Optimizes for **certainty** (floor), not edge. Uses Z-score hit probabilities, Vegas market alignment, anti-fragility filters, and strategic correlation scoring. Displays a stacked "ticket" card with combined probability, correlation multiplier, and narrative explanation. **Recent Parlays** history section below the ticket with scrollable hit/miss rows; tapping a row opens a bottom-sheet modal with the full 3-leg ticket detail and resolution (actual stats, leg-by-leg HIT/MISS). Parlay data persisted to `data/parlays/{date}.json`; lazy resolution via ESPN box scores on `/api/parlay-history`.
- **Ben**: Plain chat with Claude (teal accent). Chat always available. **No in-app historical screenshot upload banner this season** — ingestion is script/curl only (`docs/HISTORICAL_DATA.md`).

### Sub-Nav Tabs (inline, not floating)
Both `predictSubNav` (Slate-Wide | Game) and `lineSubNav` (Over | Under) are inline `div.predict-sub-nav` elements positioned at the top of their respective tab pages. They match the `.mode-tab` visual language exactly — same height, padding, `border-radius:11px`, Barlow Condensed 800. Active states: predict = chalk blue, Over = gold (`--line`), Under = teal (`--lab`).

## Codebase Navigation (grep tags)

All major sections in `api/index.py` and `index.html` are tagged for fast searching. In `api/index.py` search for `# grep:`; in `index.html` search for `grep:` (HTML/JS comments) or section banners like `LINE PAGE`. **`api/line_engine.py`** / **`api/parlay_engine.py`** use `# grep: LINE|PARLAY ENGINE MODULE` at file top. **`server.py`** documents the dev entrypoint via `grep: DEV SERVER` in its docstring.

```
grep: PREDICT TAB              — index.html DOM: tab-predictions, slateList, oracleLoader (logic: SLATE + PER-GAME below)
grep: TEAM_COLORS              — team color hex map in index.html
grep: GLOBAL STATE             — SLATE, PICKS_DATA, LAB state objects
grep: TAB NAVIGATION           — switchTab, movePill, setPillAccent
grep: SLATE                    — loadSlate, /api/slate, Starting 5, Moonshot
grep: PER-GAME ANALYSIS        — runAnalysis, /api/picks
grep: CARD RENDERING           — renderCards, player-card, tcolor
grep: PREDICTION PERSISTENCE   — savePredictions, dedup guard
grep: LINE TAB DOM             — index.html tab-line, linePickModal (logic: LINE PAGE)
grep: LINE PAGE                — initLinePage, renderLinePickCard, switchLineDir, filterLineHistory, LINE_DIR
grep: LINE ENGINE MODULE       — api/line_engine.py (run_line_engine; HTTP in api/index grep: LINE OF THE DAY)
grep: PARLAY TAB DOM           — index.html tab-parlay, parlayModal (logic: PARLAY PAGE)
grep: PARLAY PAGE              — initParlayPage, fetchParlay, renderParlayTicket, PARLAY_STATE
grep: PARLAY LIVE SSE          — /api/parlay-live-stream, _parlay_live_tick_payload, EventSource
grep: PARLAY ENGINE MODULE     — api/parlay_engine.py (run_parlay_engine; HTTP grep: PARLAY ENGINE in index)
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
grep: CARD BOOST               — _est_card_boost, _dfs_score (cascade in api/boost_model.py grep: BOOST CASCADE MODEL)
grep: GAME SCRIPT              — _game_script_weights, _game_script_label
grep: PLAYER PROJECTION        — project_player, pinfo, rating, est_mult
grep: FAIR VALUE ENGINE        — project_player_fv in api/fair_value.py (pure functions, no I/O)
grep: FAIR VALUE BETTING       — _compute_betting_fair_value() in api/index.py (Line + Parlay only)
grep: ODDS ENRICHMENT          — _enrich_projections_with_odds, odds_map, blend_weight
grep: WEB INTELLIGENCE         — _fetch_nba_news_context, Claude web_search, news_text
grep: LINEUP REVIEW            — _lineup_review_opus, post-lineup Opus, lineup_review
grep: CORE POOL                — _build_lineups candidate pool, safe/upside lineups, draft_ev
grep: GAME RUNNER              — _run_game, _build_lineups, draft_ev
grep: PER-GAME                 — _build_game_lineups, _per_game_strategy, _per_game_adjust
grep: PER_GAME_CONFIG          — per_game config defaults in _CONFIG_DEFAULTS
grep: INJURY CHECK             — /api/injury-check, RotoWire re-check, affected game regeneration
grep: CORE API ENDPOINTS       — /api/games, /api/slate, /api/picks, /api/health, /api/version
grep: LINE OF THE DAY ENGINE   — /api/line-of-the-day, run_line_engine
grep: BEN / LAB ENGINE         — /api/lab/*, _all_games_final, lab lock
grep: FORCE REGENERATE         — /api/force-regenerate, _force_write_predictions, deploy SHA mismatch, late draft
grep: LOCK HELPERS             — _is_locked, _is_past_lock_window, _et_date
grep: ALL GAMES FINAL          — _all_games_final, ESPN scoreboard poll, midnight rollover, 4.5h fallback
grep: NEXT SLATE DATE          — _find_next_slate_date, multi-day gap, All-Star break
grep: FORCE REGENERATE SYNC    — _force_regenerate_sync, scope=full|remaining
grep: PARLAY ENGINE            — /api/parlay, _fetch_gamelog, _fetch_gamelogs_batch
grep: PARLAY HISTORY           — /api/parlay-history, parlay resolution, data/parlays/
grep: PRODUCTION CACHE         — _TTL_* constants, _CK_* keys, _cp/_cg/_cs, odds_fresh_map_v1, CACHE_DIR
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
| `/api/cold-reset` | GET | Secured global cold reset for slate + LOTD + parlay (CRON_SECRET required) |
| `/api/injury-check` | GET | Cron: check RotoWire for newly OUT/questionable players; regenerate affected games only (requires CRON_SECRET when set) |
| `/api/force-regenerate?scope=X` | GET | **Force-regenerate predictions mid-slate.** `scope=full`: all games (dev deploy/model refresh; CRON_SECRET-gated). `scope=remaining`: only unlocked games (late draft, user-facing). Updates `data/predictions/` CSV and all cache layers. |
| `/api/mae-drift-check` | GET | **Weekly cron** (Monday 6am UTC): compute 7-day rolling MAE, write backend flag if > 2.5 threshold. CRON_SECRET-gated. Returns `{status, computed_mae, triggered, per_date}` |

### Line of the Day
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/line-of-the-day` | GET | **Both** Over + Under picks, slate-bound until global cold reset; returns `{over_pick, under_pick, pick}` |
| `/api/refresh-line-odds` | GET | **Game-window cron** — bulk Odds API map + lookup; updates `line`, `odds_over`, `odds_under`, `books_consensus`, `line_updated_at` on today's pick JSON. No-op if slate is locked. Returns `{status, updated, timestamp}` |
| `/api/save-line` | POST | Save `{over_pick, under_pick}` JSON + primary pick to CSV; backward-compat with legacy single-pick |
| `/api/resolve-line` | POST | Mark pick hit/miss given actual stat |
| `/api/auto-resolve-line` | GET | **Cron** — resolves each pick when its game ends (requires CRON_SECRET when set) |
| `/api/line-live-stat` | GET | Fetch live in-game stat value for pick tracking (single-game lock check) |
| `/api/line-history` | GET | Recent picks with streak + hit rate (only resolved picks — never pending) |
| `/api/line-force-regenerate` | GET | Force-generate today's line picks (overwrites stale artifacts, busts cache) |

### Parlay
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/parlay` | GET | **Safest 3-leg player prop parlay** on today's slate. Optimizes for certainty via Z-score hit probabilities, Vegas market alignment, anti-fragility filters (blowout, minutes CV, GTD risk), and strategic correlation scoring. Auto-saves to `data/parlays/{date}.json` on GitHub. Returns `{legs[], combined_probability, correlation_multiplier, correlation_reasons[], parlay_score, narrative}`. 30-min cache. Rate-limited 10/min. Never returns 500. |
| `/api/parlay-history` | GET | Recent parlays with lazy resolution (hit/miss per leg via ESPN box scores). Resolves pending legs on read for historical dates and writes results back to GitHub. Returns `{parlays[], hit_rate, total, resolved, streak, streak_type}`. 10-min cache. |

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
- **Tab load (lazy):** Line, Parlay, and Lab load on first visit when `switchTab()` is called:
  - **Line:** `initLinePage()` → fire-and-forget `GET /api/auto-resolve-line`, then `GET /api/line-of-the-day`, `GET /api/line-history`; live stat poll and resolve poll when applicable.
  - **Parlay:** `initParlayPage()` → `GET /api/parlay` (90s timeout) + fire-and-forget `GET /api/parlay-history` (15s). Skeleton loading. Same-day cache guard (`PARLAY_LOADED_DATE`). Renders stacked 3-leg ticket card with combined probability + correlation multiplier + narrative. History section shows Recent Parlays with hit/miss pills; tapping a row opens bottom-sheet modal with full ticket detail.
  - **Lab:** `initLabPage()` → chat UI immediately; `showLabUnlocked()` loads briefing + config-history + line + slate + parlay in the background. `/api/lab/status` is pre-warmed from Predict and used when the locked Lab view/poll path runs (rare).

## Environment Variables (Railway)

- `GITHUB_TOKEN` — GitHub PAT with repo scope (for CSV + config read/write via Contents API)
- `GITHUB_REPO` — e.g. `cheeksmagunda/basketball`
- `ANTHROPIC_API_KEY` — Claude Haiku (screenshot OCR) + claude-opus-4-6 (Ben/Lab chat)
- `ODDS_API_KEY` — The Odds API for player prop lines (Line of the Day + draft pipeline enrichment)
- `INGEST_SECRET` — (optional) When set, `POST /api/save-most-popular`, `save-ownership`, `save-most-drafted-3x`, and `save-winning-drafts` require `X-Ingest-Key` or `Authorization: Bearer <INGEST_SECRET>`. See `docs/HISTORICAL_DATA.md`.
- `CRON_SECRET` — (optional) When set, protected endpoints (`/api/cold-reset`, `/api/auto-resolve-line`, `/api/lab/auto-improve`, `/api/injury-check`) require `Authorization: Bearer <CRON_SECRET>`. Railway injects this via cron commands in `railway.toml`.
- `DOCS_SECRET` — (optional) When set, `/docs`, `/redoc`, and `/openapi.json` require `?docs_key=<value>` or `X-Docs-Key` header so only people with the secret can browse/test the API.
- `REDIS_URL` — (recommended) Full `redis://` connection string. Auto-injected by Railway Redis plugin. When set, all cache reads/writes go through Redis (sub-ms latency) with `/tmp` file fallback. Without it, the system works but relies on ephemeral `/tmp` files that don't survive container restarts. Setup: Railway dashboard → "+ New" → "Database" → "Redis". Verify: `GET /api/health` returns `"redis": "ok"`.

**OpenAPI docs:** FastAPI serves `/docs` (Swagger UI) and `/redoc` in production. Use them to browse and try endpoints. When `DOCS_SECRET` is set, append `?docs_key=<DOCS_SECRET>` to the URL or send the header to access.

## Runtime Config System

Model parameters are stored in `data/model-config.json` on GitHub. The backend loads this
file at startup and caches it for 5 minutes. The Lab writes updates via the GitHub Contents API.

- **No redeploy needed** to tune parameters — changes take effect within 5 minutes
- **Fallback to defaults** if GitHub is unreachable — app never breaks
- Use `_cfg("dot.path", default)` helper anywhere in `api/index.py` to read config
- `/api/cold-reset` clears config/cache layers and regenerates slate + LOTD + parlay

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
- `_bust_slate_cache()` — **Atomic bust of all layers**: in-memory + Redis (`rflush()`) + `/tmp` + GitHub tombstones
- `_slate_cache_to_github(slate_data)` — writes today's slate to `data/slate/{date}_slate.json`
- `_slate_cache_from_github()` — reads today's slate; returns `None` if missing or busted

### Cache Invalidation
- **Config change** (`/api/lab/update-config`): calls `_bust_slate_cache()` → busts all layers → next request regenerates
- **Manual cold reset** (`/api/cold-reset`): calls global cold pipeline and regenerates slate + LOTD + parlay
- **Injury check** (`/api/injury-check` cron, afternoon): invokes the same global cold pipeline when injuries hit cached lineups
- **New deploy**: startup hook detects SHA mismatch → `_bust_slate_cache()` → background regeneration

### Version-Aware Startup (grep: deploy_sha)
On container start, `_deploy_startup_safe_prewarm()` compares `RAILWAY_GIT_COMMIT_SHA` against a `deploy_sha` key in Redis:
- **SHA mismatch** (new deploy): bust all caches, write new SHA, background-regenerate full slate
- **SHA match** (container restart): check Redis (instant) → `/tmp` → GitHub for existing cache — no pipeline run

### Line / Parlay projection hydration
`_run_line_engine_for_date()`, `_get_projections_for_date()`, and the parlay pipeline load per-game projections in order: **Layer 1** `/tmp` (`game_proj_{gameId}`), **Layer 2** GitHub `data/slate/{date}_games.json` (via `_hydrate_game_projs_from_github()`), **Layer 3** full `_run_game()` if still empty. One GitHub read on a cold instance avoids re-running ESPN + LightGBM for every tab.

### Prediction model boundaries (grep: LINE CONFIG, line_engine config)

- **Draft model:** Config in `data/model-config.json` (card_boost, game_script, real_score, cascade, projection, lineup, moonshot, development_teams); code in `api/index.py`, `api/real_score.py`, `api/asset_optimizer.py`; `lgbm_model.pkl` is trained separately (GitHub Actions). Ben can change draft behavior via Lab (update-config, backtest).
- **Line of the Day model:** `api/line_engine.py` receives projections and games from the draft pipeline plus an optional `line_config` dict passed from `api/index.py` (from the config `line` section). Recent-form sparklines use `recent_form_bars` from the engine (ratio vs season); `recent_form_values` may be present on stored picks when written at generation time. The line engine also receives **web search news context** (from `_fetch_nba_news_context()`, same Layer 1 cache as the draft model) — Claude Haiku sees injury updates, rotation changes, and rest decisions when making over/under picks. Line knobs in config: `min_confidence`, `min_edge_pct`, `recent_form_over_ratio` (lowered to 1.07 — avoids buying at peak momentum), `recent_form_under_ratio`, `min_edge_pts`, `min_edge_other`, `min_edge_other_over` (over-specific non-points edge floor; defaults to `min_edge_other`), `pct_edge_rebounds`, `pct_edge_assists` (percentage-based edge thresholds for peripherals — replaces flat 2.5 requirement with 18% dynamic scaling), `juice_under_threshold` (heavy over juice as positive under signal), `stat_floors`, `stat_floors_under` (relaxed floors for under bets — trivial lines are valid for unders), `auto_fade.enabled`, `auto_fade.blowout_spread_threshold`, `auto_fade.blowout_starter_min_floor`, `auto_fade.rotation_squeeze_spread`, `auto_fade.rotation_squeeze_bench_ceiling`, `auto_fade.b2b_guard_min_season_pts`. Ben can tune via the `line` section of model-config; no code in line_engine reads GitHub or `_cfg` — config is passed in by the caller to keep the engine self-contained.

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

Per-game checks (e.g. `/api/picks`, `/api/line-live-stat`) correctly use single-game `_is_locked(game_start)`.

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

**Parlay**, **Per-game**, **Line**, **Context layer**, **Odds enrichment** config sections remain unchanged.

---

## Two Draft Strategies (Strategy-Report-Aligned Architecture)

Both lineups use a **single EV formula**: `draft_ev = RS × (2.0 + boost)`. Strategy report (90 dates) proves boost is 40% more valuable per unit than RS, so both terms must be maximized. No separate chalk/moonshot pools or MILP optimizer — the problem is structurally solvable with a sort.

**Pipeline** (6-step in `_build_lineups`):
1. **Filter**: RS ≥ 2.0 (`strategy.rs_floor`), minutes ≥ 12 (`strategy.min_minutes`), not OUT, not blacklisted
2. **Score**: `draft_ev = RS × (2.0 + boost)` for each candidate
3. **Rank**: Candidates sorted by `draft_ev` descending
4. **Safe Lineup (Starting 5)**: Top 5 by EV. Ties broken by lower variance (reliable floor).
5. **Upside Lineup (Moonshot)**: Starts as copy of safe. Up to 2 swaps (`strategy.max_upside_swaps`) from remaining pool — prefers higher-variance or higher-boost players within a 2.0 EV threshold (`strategy.ev_swap_threshold`).
6. **Slot Assignment**: Both lineups sorted by RS descending → 2.0x, 1.8x, 1.6x, 1.4x, 1.2x. This is provably optimal: `d(Value)/d(slot) = RS`, so highest RS → highest slot.

**Anti-popularity** (Finding 4): Draft popularity has -0.457 correlation with boost. The cascade in `boost_model.py` estimates popularity from season PPG + team market + hot streak, then penalizes popular players' boost predictions. The least-drafted 50% produce 24-26% more total value.

Config: All draft strategy parameters in `strategy.*` section of model-config defaults (~15 params replacing ~120 prior scattered params).

### Slate-Wide: Starting 5 (Safe)
Top 5 by `draft_ev`. Tie-breaking prefers lower variance for floor reliability. No MILP, no boost floors, no star anchor pathway — the EV formula naturally selects the right mix.

### Slate-Wide: Moonshot (Upside)
Starts from safe lineup, swaps 1-2 players for higher-variance/higher-boost alternatives from the candidate pool. Swaps only happen when replacement EV is within `ev_swap_threshold` of the displaced player's EV — ensures upside without sacrificing expected value.

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

## Line Page — Direction Filter & Odds Refresh

The Over/Under inline sub-nav (`#lineSubNav`) and the inline All/Over/Under tabs in Recent Picks both call `filterLineHistory(dir)`. Selecting a direction also controls the **main pick card visibility**:

- `switchLineDir(dir)`: renders the appropriate pick (`LINE_OVER_PICK` or `LINE_UNDER_PICK`) via `renderLinePickCard()`
- **Slate-bound picks** — over and under stay tied to the active slate. Resolved picks remain visible until the global cold-reset pipeline advances to the next slate.
- Resolved picks appear only in Recent Picks history.
- Picks loaded from GitHub CSV lack `books_consensus/odds_over/odds_under` — render as `MODEL` label. Picks refreshed via `/api/refresh-line-odds` show actual book odds + count.
- Pick cards display `"Odds · [time] CT"` when `line_updated_at` is present (stamped by `/api/refresh-line-odds`)

### Line of the Day card
The main pick card (`renderLinePickCard`) uses a **zoned layout** and design tokens. Over and Under use the same component; only `dir` and content differ.

- **Zone 1 (Header):** Player name; subheader = matchup + game time (e.g. `CLE vs BOS · 1:00 PM ET`) when `game_time` is present, else `Team · vs Opponent`. Odds/timestamp in the top-right of the card (`position: relative` on the card, flex + `margin-left: auto` on the odds block).
- **Zone 2 (The Play):** One row: bet pill (OVER/UNDER) + target stat line (e.g. `13.0 pts`). Stat label is derived from `pick.stat_type` via a small map (points → PTS, rebounds → REB, assists → AST) — no hardcoded PTS/REB/AST in the UI.
- **Zone 3 (Data row):** A single full-width flex row (`.line-pick-data-row`, `justify-content: space-between`) with **5 columns:** (1) **Baseline** — sportsbook line + stat label; (2) **Edge** — mathematical edge, semantic colors `edge-plus` / `edge-minus`; (3) **Target stat** — stacked projection / season average; (4) **Minutes** — stacked `proj_min` / `avg_min`; (5) **L5** — text array of last-5 stat values (e.g. `12 • 14 • 9 • 16 • 11`) with hit/miss coloring vs baseline. When `recent_form_values` is present on the pick payload (saved at generation time), L5 uses those; otherwise it falls back to ratio-based values from `recent_form_bars`.
- **Conclusion (Oracle Insight):** Model reasoning is consolidated into a single narrative paragraph at the bottom of the card, not standalone pills. `_buildLineConclusion(pick)` merges `pick.narrative` with `pick.signals` (e.g. injury upgrade, B2B) into one natural-language sentence; key reasons are highlighted with `--color-text-primary`. The paragraph sits in `.line-pick-conclusion-wrap` (subtle background, 8px radius, padding). **This "Narrative Conclusion" pattern is the standard for model explanations app-wide** — use one prose block, not reasoning bubbles.
- **Pick payload fields** (from backend): Core fields plus `season_avg`, `proj_min`, `avg_min`, `game_time`, `recent_form_bars`, `recent_form_values`. `recent_form_bars` is set in `api/line_engine.py`. `recent_form_values` is optional and typically persisted in `data/lines/{date}_pick.json` when picks are first generated; load paths use stored JSON without re-fetching. All passed through `_normalize_line_pick`.
- **Design tokens:** Line card uses `--radius-card`, `--radius-pill`, `--font-size-micro`, `--color-success`, `--color-danger`, `--color-text-primary`, `--color-text-muted`, `--line`, `--lab`; no hardcoded hex for semantic colors in the card block.
- **Cache:** `index.html` is served with `Cache-Control: max-age=0, must-revalidate` (railway.toml static headers) so browsers and edge revalidate and users get the latest card (5-column + conclusion) after deploy. If an user still sees an old card, they should hard refresh (e.g. pull-to-refresh on mobile) or close and reopen the tab.

### Odds Refresh Pipeline
- **Cron** (source of truth: `railway.toml`): `55 19,20,21,22,23,0,1,2,3,4,5,6 * * *` UTC — odds sync during typical NBA game windows only (reduces cron load vs 24×/day).
- **Helpers**: `_abbr_matches(abbr, full_name)` maps ESPN abbrs → Odds API team name fragments; `_build_player_odds_map(games)` bulk-fetches props; `/api/refresh-line-odds` uses the bulk map + `_lookup_player_odds` with per-pick fallback to `_fetch_odds_line` when needed
- **Odds API outcome field structure**: The Odds API returns outcomes as `{name: "Over"|"Under", description: "Player Name", point: 20.5, price: -110}`. Note: `name` = direction, `description` = player name (counter-intuitive). Both `_build_player_odds_map` and `_fetch_odds_line` read `description` as player key and `name` as direction. This is not standard JSON naming; don't rely on field names — verify against live API responses when debugging.
- **Synthetic fallback** (parlay only): when `player_odds_map` is empty (Odds API unavailable or no matching games), `_run_parlay_engine_sync` builds model-only lines via `round(proj_val * 2) / 2` (nearest 0.5 snap). Parlay thresholds are loosened (`min_blended_conf` → 0.50, `max_minutes_cv` → 0.35) to compensate for lower Vegas signal quality. `projection_only: true` is stamped on the response.
- **Lock freeze**: `/api/refresh-line-odds` uses `any(_is_locked(...))` on start times — no-op if slate locked
- **REFRESH button**: calls `/api/refresh-line-odds` then reloads Line page data

## z-index Hierarchy (fixed elements)

| Element | z-index |
|---------|---------|
| `#linePickModal` (bottom sheet) | 1001 |
| `.bottom-nav` | 1000 |

`switchTab()` calls `closeLinePickModal()` + resets `document.body.style.overflow` on every tab switch to prevent scroll lock leaking between tabs.

Note: `predictSubNav` and `lineSubNav` are now **inline elements** (not fixed/floating) — no z-index needed.

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

Use these tokens instead of hardcoded hex or pixel radii across Predict, Line, Parlay, and Ben.

## Cron Schedule (railway.toml)

Crons and frontend poll intervals are tuned to minimize Railway compute and ESPN API usage while preserving lock/unlock, odds refresh, and line-resolve behavior.

| Schedule (UTC) | Endpoint | Purpose |
|----------------|----------|---------|
| `0 9 * * 0,3` | `/api/lab/auto-improve` | Auto-tune model if ≥3% MAE improvement (Wed + Sun only — 2×/week) |
| `55 19,20,21,22,23,0,1,2,3,4,5,6 * * *` | `/api/refresh-line-odds` | Bookmaker odds sync — game-window hours only |
| `0 20,21,22,23,0,1,2,3,4,5,6,7 * * *` | `/api/auto-resolve-line` | Resolve line picks — game-window hours only |
| `0 18,22,2 * * *` | `/api/injury-check` | 3 key windows: pre-tip (1 PM ET), mid-evening (5 PM ET), late (9 PM ET) |
| `0 6 * * 1` | `/api/mae-drift-check` | Weekly MAE drift monitoring (Monday 6am UTC); CRON_SECRET-gated |
| `0 17 * * *` | `/api/parlay-force-regenerate` | Pre-lock parlay generation → GitHub (before evening tips) |

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
- **Frontend:** All `JSON.parse(localStorage.getItem(...))` usages are wrapped via `_safeParseLocalStorage(key, fallback)` so corrupted or invalid JSON does not throw. Critical DOM access uses `_el(id)` with null checks (or early return) in Line/Lab/Predict and Lab lock poll so missing elements do not throw. Lab lock poll logs status fetch failures with `console.warn('[lab] Lock poll status check failed:', ...)` instead of failing silently. Lab chat uses an `AbortController` with a 60s connection timeout for the initial `/api/lab/chat` request; on timeout the user sees "Request timed out. Please try again."

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
- Unlock still detected within the next frontend poll (2 min Lab, 1 min Line when relevant)

### Aggressive ESPN Fallback (4.5-Hour Rule)
If ESPN API delays updating game status to "Final":
- If latest game running 4.5+ hours: automatically mark all games as complete
- Ignores `finals > 0` requirement — fires even if ESPN completely lagged
- Prevents indefinite lock waits when ESPN slow during high-traffic windows (Saturday evenings)
- Prevents false unlocks: still requires `remaining == 0` (at least one game attempt started)

### Event-Driven Frontend Unlock
When line polling detects games finished (`status === 'final'`):
- Immediately triggers `/api/lab/status` check instead of waiting for next poll cycle (~1-2 min)
- If Line poll runs while Lab is open, an extra `/api/lab/status` check can run (legacy paths); no upload UI
- Falls back to auto-resolve cron if client is not open

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

Affected endpoints: slate load, picks, games, save-predictions, screenshot parse, save-actuals, audit, line-of-the-day, refresh-line-odds, lab-status, lab-briefing, lab-chat, lab-config-history, line-history, hindsight. Full timeout table and loading UX: **docs/LOADING_AUDIT.md**.

### Worker Pool Optimization
Backend uses Python `ThreadPoolExecutor` for parallel processing:
- **Standard pool: 8 workers** (game runner, slate processor, picks processor, audit runner, line engine, Odds API per-event props)
- **Parlay gamelog pool: 10 workers** (ESPN athlete gamelog HTTP — I/O-bound; scoped batch via `select_parlay_gamelog_player_ids`)
- Handles 14-game Saturdays efficiently without bottlenecking

### Polling Interval Tuning
- **Lab lock polling**: 2 minutes (reduces API call frequency; see `initLabPage` in index.html, ~3135)
  - Unlock detected within ~2 min; user can tap Retry for immediate check
- **Line live stat polling**: 1 minute; max 5 consecutive failures (300s tolerance) before fallback to cron
  - Prevents indefinite polling on persistent network failures
  - Falls back to `/api/auto-resolve-line` cron (game-window hours; see `railway.toml`)

### GitHub API Retry Logic
`_github_write_file()` (api/index.py lines 75-110) implements exponential backoff for concurrent write conflicts:
- **Retries up to 3 times** on HTTP 422 (SHA mismatch)
- **Backoff delays**: 1s, 2s, 4s between retries
- **Fresh SHA fetch** on each retry (not cached)
- Protects against concurrent writes from cron + overlapping API writes (rare but possible edge case)
- Used for: predictions, actuals, line picks, config updates

### Cache TTL & Invalidation
Explicit TTLs protect against stale data while minimizing API calls:

| Cache | TTL | Purpose | Invalidation |
|-------|-----|---------|--------------|
| Game final status (`_all_games_final`) | 60s when locked, 180s pre-slate | Detects when ALL games reach Final status | global cold reset clears |
| Model config (`data/model-config.json`) | 5 min | Runtime tuning parameters | global cold reset clears; Lab writes bypass cache |
| RotoWire lineups | 30 min | Player availability (OUT, questionable, etc.) | 30 min expiration; manual refresh via app |
| Lock status per game | 6 hours | 5 min before tip to 6h after (ceiling) | Natural expiration |
| Line odds (`books_consensus`) | 1 hour | Bookmaker consensus line (refreshed by cron) | Game-window cron runs; slate-lock freeze |
| Parlay ESPN gamelog (`_fetch_gamelog`, `_TTL_L5`) | 30 min | Last games for volatility / Z-score (ESPN `site.api.espn.com` athlete gamelog) | Per-player cache key `gamelog_{pid}` in `/tmp` |
| Odds bulk map fresh cache (`odds_fresh_map_v1`) | 10 min | Reuse `_build_player_odds_map` across slate / line / parlay | global cold reset clears `/tmp` |
| Slate cache (`data/slate/`) | 1 day | GitHub-persisted predictions (full slate + per-game) | `_bust_slate_cache()` via cold reset, config change, or injury check |
| Log dates (`log_dates_v1`) | 10 min | Dates with stored prediction/actual data (backend only) | global cold reset clears all `/tmp` caches |
| Log get (`log_get_{date}`) | 5 min | Per-date predictions + actuals from GitHub (backend only) | global cold reset clears all `/tmp` caches |
| Parlay history (`parlay_history_v1`) | 10 min | Recent parlay results from `data/parlays/` | global cold reset clears all `/tmp` caches |
| Line `/tmp` (`line_v1`) | 30 min max age (inline check) | Fast path for `/api/line-of-the-day` | Rotation / odds refresh unlinks cache file |
| Parlay `/tmp` (`parlay_v1`) | 30 min | Today's parlay JSON | cold reset or next-slate regeneration |
| ESPN scoreboard (`fetch_games`) | 5 min (`_TTL_GAMES`) | Schedule + spreads; shared `_GAMES_CACHE_TS` | New fetch resets TS |
| Odds API degraded snapshot | 1 h | `odds_last_success_map_v1` when live bulk fetch fails | Superseded by next successful fetch |

### Production: DRY rules and cache inventory

**Do not duplicate TTLs** — All backend TTLs live in one block in [`api/index.py`](api/index.py) (`_TTL_CONFIG`, `_TTL_GAMES`, `_TTL_LOG`, `_TTL_LOCKED`, `_TTL_PRE_SLATE`, `_TTL_L5`, `_TTL_HOUR`, `_TTL_ODDS_FRESH`). Endpoint-specific ages (e.g. 30 min for line/parlay `/tmp`) should stay aligned with these constants when changed.

**Cache helpers (single pattern)** — `CACHE_DIR` (`/tmp/nba_cache_v19`) + `_cp(key, date_str?)` → file path; `_cg` read JSON; `_cs` write JSON. Same helpers for slate, line, parlay, news, team stats, gamelogs (`gamelog_{pid}`). Avoid ad hoc `Path` writes outside this pattern.

**Named logical keys (`_CK_*`)** — `_CK_SLATE`, `_CK_SLATE_LOCKED`, `_CK_LINE`, `_CK_LINE_HISTORY`, `_CK_LOG_DATES`, `_CK_PARLAY`, `_CK_PARLAY_HISTORY`; per-game `game_proj_{gameId}` via `_ck_game_proj()`. Search `# grep: CONSTANTS & CACHE` in `api/index.py`.

**Odds API (two-tier)** — `_build_player_odds_map`: (1) **Fresh reuse** — `odds_fresh_map_v1` keyed by slate fingerprint, `_TTL_ODDS_FRESH` (10 min). (2) **Failure fallback** — `odds_last_success_map_v1` up to 1 h when the live request fails. **`/api/refresh-line-odds`** uses bulk map + `_lookup_player_odds`, then `_fetch_odds_line` per pick only if needed.

**GitHub vs `/tmp`** — Slate: Layer 1 `/tmp` → Layer 2 `data/slate/*` → Layer 3 pipeline. Line/parlay enrichment: `_hydrate_game_projs_from_github()` before `_run_game()` when `/tmp` is cold. Single read replaces N× ESPN/LGBM for the same ET day when a prior instance already wrote `data/slate/{date}_games.json`.

**Bust / reset** — `_bust_slate_cache()` tombstones GitHub slate + games + locks and clears local `/tmp` JSON. `GET /api/cold-reset` runs the global bust+regen pipeline and reloads config/cache layers.

### Midnight Rollover Handling
`auto_resolve_line()` correctly handles games finishing after midnight ET:
- Tracks `pick_date` separately from `_et_date()` (which changes at midnight)
- Uses `pick_date` for both GitHub file lookups and ESPN API queries
- Falls back to yesterday's pick file if today's missing
- Computes next-day picks from `pick_date + 1`, not `_et_date() + 1`
- Prevents loss of line pick data on multi-day slates

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
| **api/fair_value.py** | Deterministic fair-value engine (Engine 2) | Rolling L10/L15 windows, DvP adjustment, game script weights, spread adj, momentum, per-stat fair value + Z-score hit probs. **Prop betting only** — called via `_compute_betting_fair_value()` in index.py for Line + Parlay. Pure functions, no I/O. |
| **api/line_engine.py** | Line of the Day engine | Claude Haiku prompts, _STAT_META (points/rebounds/assists), Odds API integration. Receives `edge_map` from fair_value for `fv_boost` on confidence. Called by api/index.py `/api/line-of-the-day`. No direct HTTP; all I/O via index. |
| **api/parlay_engine.py** | Safest 3-leg parlay optimizer | Z-score hit probability, anti-fragility filters (blowout/CV/GTD), market alignment, correlation scoring. Receives `fair_value_data` with `_fv_hit_probs` to override baseline model_prob. Called by api/index.py `/api/parlay`. Pure computation; no external I/O. |
| **api/rotowire.py** | RotoWire lineup scraper | Free-tier scrape for availability (OUT, questionable). 30 min cache. Used by slate/Moonshot filtering. |
| **api/real_score.py** | Monte Carlo Real Score (retained) | RS projection (closeness, clutch, momentum). **No longer used by draft pipeline** — retained for reference and potential reactivation. |
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
- **docs/LOADING_AUDIT.md** — Frontend loading: fetch timeouts, skeletons, async state, Line flash and first-load-after-hit fixes.

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
TestAutoResolveMidnight     — pick_date vs et_date divergence after midnight
TestCacheTTLs               — 3 min games, 5 min config, 30 min RotoWire, 60s locked TTL
TestPollingIntervals        — 120s lab lock, 60s line live, 300s failure cutoff
TestRateLimitThreadSafe     — _check_rate_limit is thread-safe under concurrent calls
TestLineConfig              — run_model_fallback and run_line_engine respect line_config min_confidence; min_edge_other_over asymmetry (over blocked, under passes with same edge)
TestLgbmFeatureAlignment    — when bundle loaded, 12 features with recent_vs_season at index 9 and reb_per_min at index 11
TestSlateExceptionHandling  — slate endpoint catches exceptions and returns 200 with error key (never 500)
TestGameSelectorLockDisplay — frontend populateGameSelector must NOT override per-game lock with slateLocked
TestLinePrimaryPickFallback — LINE_OVER_PICK/UNDER_PICK populated from primary pick when directions null
TestLinePicksBothNullRegeneration — backend regenerates picks when saved file has both directions null
TestFetchGamesTTL           — fetch_games() enforces 5-min TTL to avoid stale ESPN data
TestSavePredictionsMerge    — save_predictions merges new per-game scopes into existing CSV
TestSwitchTabNoDuplicateInit — switchTab does not call initLinePage twice
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
TestOddsEnrichment          — odds enrichment skip when disabled, upward blend at divergence, no-blend below threshold
TestWebSearch               — Claude web_search skip when disabled/no key, fetch+cache, cache reuse
TestContextPassWithNews     — web search called from context pass, news text in prompt
TestLineSignals             — _generate_signals produces driver signals for narrative transparency (8 signal types)
TestHighBoostRolePathway    — high-boost role players bypass minutes floor in moonshot + chalk pools
TestRsCalibrationWeights    — DFS weight recalibration, archetype detection + calibration, scorer upside
TestCascadeCapFix           — per_player_cap_minutes raised to 10.0 for meaningful cascade propagation
TestRotoConfirmedRatingException — confirmed rotation players with high boost bypass min_rating_floor
TestMaxPerGameConstraint    — MILP max_per_game limits players from same game matchup
TestMinHighBoostConstraint  — MILP min_big_boost ensures minimum high-boost players in lineup
TestPerGameStrategy         — _per_game_strategy() returns correct type/label based on spread+total (6 tests)
TestPerGameAdjustProjections — F1-F6 adjustments: total mult, close game consistency, blowout tilt, value anchors (8 tests)
TestPerGameBuildLineups     — _build_game_lineups() returns strategy, 5 players, valid slots, blowout 4-1 split (6 tests)
TestPerGameConfig           — per_game section in _CONFIG_DEFAULTS, all 20 keys present, score bounds widened (3 tests)
TestPerGameFrontend         — strategyInsight element, render function, ANCHOR/FAV pills, back hides insight (5 tests)
TestAutoFadeLine            — _check_auto_fade: B2B guard over veto (pts/ast), blowout truncation (spread>=10 starters), rotation squeeze (bench tight games), config disable, custom thresholds (12 tests)
TestPctEdgeScaling          — _compute_pct_edge calculation, zero line, percentage-based gate in model fallback (3 tests)
TestMomentumRatioLowered    — config default 1.07, signal fires at 1.07, no signal below 1.07 (3 tests)
TestJuiceAsUnderSignal      — juice signal fires for under at -130+, not for over, not for mild juice (3 tests)
TestPlayerB2BSignal         — B2B under bonus +10, B2B over penalty -8, player_b2b field alternative (3 tests)
TestTrivialLineFloorRelaxed — config stat_floors_under section, under passes relaxed floor (2 tests)
TestBlowoutTieredBonus      — spread 8 gives +6, spread 12 gives +10 tiered blowout bonus (2 tests)
TestLineEngineConfigKeys    — all new config keys present in model-config.json (2 tests)
TestClaudePromptUpdated     — prompt has auto-fade, percentage edge, juice, player B2B rules (4 tests)
```

**tests/test_core.py** — Helpers, line cache logic, JS syntax, date-boundary regressions, and contract guards:

- **TestHelpers** — _et_date, _is_locked, _est_card_boost, cache roundtrip
- **TestLineCacheLogic** — when line cache is served vs bypassed (today unresolved / resolved / yesterday)
- **TestJSSyntax** — unescaped apostrophes in single-quoted strings; presence of renderCards, renderLinePickCard, initLinePage, loadSlate, switchTab; _etToday / LINE_LOADED_DATE / _predSavedDate
- **TestCacheDateBoundary** — cache keys consistent with ET date
- **TestBenBannerActualsDetection** — ACT_FIELDS / actuals-shaped CSV parsing (audit inputs)
- **TestBannerGuardJS** — banner-visibility check in showLabUnlocked() uses correct localStorage keys
- **TestNormalizePlayer** — _normalize_player() contract: all required fields present with correct types
- **TestNormalizeLinePick** — _normalize_line_pick() contract: all required fields, result defaults to "pending"
- **TestRealScoreEngine** — Monte Carlo closeness/clutch/momentum coefficients (pure numpy, no I/O)
- **TestAssetOptimizer** — optimize_lineup() MILP slot assignment: chalk vs moonshot modes, edge cases, RS-ordered slotting, two-phase moonshot, same-position-same-team allowed
- **TestConfigCoverage** — all major model floors read from model-config.json via _cfg()
- **TestProjectPlayerContract** — project_player() returns all required fields after _normalize_player()
- **TestLineEngineHelpers** — line_engine.py helpers with no external deps (_abbr_matches, stat meta)
- **TestJSContractGuard** — frontend null guards for _normalize_line_pick fields added in Phase C
- **TestLogGetNormalization** — log_get() builds player cards from CSV rows with correct field mapping (backend-only endpoint)
- **TestUpdateConfigValidation** — /api/lab/update-config accepts dot-notation keys, rejects invalid paths
- **TestFrontendAuditFixes** — regression guards for frontend null guards and .ok checks

**tests/test_parlay.py** — Parlay engine: Z-score math, anti-fragility filters, correlation scoring, end-to-end:

- **TestAmericanToImplied** — American odds to implied probability conversion (-140, +150, edge cases)
- **TestZToProbability** — Cumulative normal Z-score approximation
- **TestComputeHitProbability** — Over/under hit probability from projection, line, and σ
- **TestBlendedConfidence** — Model + Vegas probability blending (55/45 split)
- **TestBlowoutFilter** — Spread > 8.5 filtering
- **TestMinutesCV** — Minutes coefficient of variation (rotation stability)
- **TestGTDStarTeammate** — Questionable starter RotoWire filter
- **TestCorrelationModifier** — VETO (same-team rebounds/assists), positive (PG assists + teammate points, shootout)
- **TestStdDev** — Population standard deviation helper
- **TestNarrativeBuilder** — Natural-language parlay narrative generation
- **TestRunParlayEngine** — End-to-end: valid data → 3-leg result, empty → None, blowout → None
- **TestParlayConfigDefaults** — parlay section in _CONFIG_DEFAULTS
- **TestParlayRateLimit** — parlay in _RATE_LIMITS
- **TestFetchGamelog** — _fetch_gamelog callable, graceful on invalid PID
- **TestParlayEndpointExists** — /api/parlay in app routes
- **TestParlayFrontend** — tab-parlay, nav button, CSS variable, state, ticket container, rendering functions, history section (wrap/list/stats), modal (parlayModal/content), accessibility (aria-modal, aria-live, aria-label), PARLAY_HIST_DATA global, escape key
- **TestParlayHistoryEndpoint** — /api/parlay-history in app routes, auto-save fields in /api/parlay
- **TestAutoFadeSwitchHeavy** — center reb over vs switch-heavy defense filtered; guards pass; non-switch-heavy passes
- **TestAutoFadeFakeJuice** — high recent hit rate + low season prob triggers fade; confident model passes
- **TestAutoFadeB2BPenalty** — B2B correlated pair penalized in structure scoring and pair search
- **TestGameTotalFloor** — low-total pair excluded from correlated pair search; penalized in structure
- **TestCVBasedMarketMatch** — high-CV candidate loses market match bonus; low-CV gets it
- **TestPnrRimBoost** — interior finisher (C/PF, 7+ reb) gets 1.20x; generic scorer gets 1.08x
- **TestPerimeterToPerimeterFade** — perimeter-only scorer (SG, <4 reb) gets correlation penalty
- **TestPaceBoost** — high-total game (≥232) gets pace multiplier; below threshold does not
- **TestRestAdvantageBoost** — team rested vs opponent on B2B gets 1.08x; both rested does not
- **TestTightenedSpread** — correlated_pair_max_spread defaults to 5.0 (6.0 excluded, 4.0 passes)
- **TestDynamicLeg1Substitution** — rebounds deprioritized vs elite defense; preferred vs normal
- **TestCandidateLegB2BFields** — candidate legs carry is_b2b, opp_b2b, season_reb from game data

Run all: `pytest tests/ -v`
Run fast subset only: `pytest tests/test_fixes.py -v`  
Note: Tests that import `api.index` require dependencies (e.g. numpy, lightgbm). Use `pip install -r requirements.txt` before running.

## Known Limitations

- Rate limiting uses an in-memory store with a lock (thread-safe); it does not persist across container restarts, so limits reset on redeploy.
- `/tmp` is cleared on container restart (redeploy or crash) — caches don't survive restarts. GitHub-persisted caches (`data/slate/`, `data/locks/`) provide cold-start recovery for both predictions and lock state.
- **Concurrent write conflicts (mitigated)**: `_github_write_file` implements exponential backoff (1s, 2s, 4s retries) to handle HTTP 422 SHA mismatches. Overlapping cron + API writes are handled; conflicts are rare.
- Odds API: when over_pick and under_pick are the same player, `/api/refresh-line-odds` fetches once and applies the result to both (deduped).
- Historical slate/lock cache files can create operational noise. Keep `data/slate/` and `data/locks/` pruned to active-slate artifacts.
- Fetch timeouts: All frontend calls have hard limits (10s default, 30s screenshot). Exception: `/api/lab/chat` uses a raw streaming fetch (SSE) by design — no timeout on the stream body, only on connection.
- Upload screenshot type validation is client-side trust only — the system cannot verify that a "Real Scores" button upload actually contains a Real Scores screenshot. Wrong uploads produce skewed audit data for that date.

## Troubleshooting

If slate and/or line fail to load:
1. **Deployed URL** — Use the production URL (`https://the-oracle.up.railway.app`); avoid file:// or wrong origin.
2. **Health and version** — Call `GET /api/health` and `GET /api/version`; if they fail, the backend is unreachable or cold-starting.
3. **Railway logs** — Look for `[slate] error:`, `[line-of-the-day] error:`, `[games] error:` or "Task timed out" to identify the failing path.
4. **Parlay SSE smoke** — Use `python3 scripts/parlay_sse_smoke.py --base-url <origin>` to validate `/api/parlay-live-stream` reconnect behavior and payload consistency against `/api/parlay`.

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
| Line config wired from model-config | `api/index.py`, `api/line_engine.py` | `run_line_engine(projections, games, line_config)`; `min_confidence`, `min_edge_pct`, `recent_form_over_ratio`, `recent_form_under_ratio`, `min_edge_pts`, `min_edge_other`, `min_season_minutes`; projections enriched with real L5 before engine run |
| Line min_season_minutes filter | `api/index.py`, `api/line_engine.py` | Filters out players whose season avg minutes fall below threshold before Claude or fallback runs. Default 20.0 min. Prevents fringe vets from qualifying on a single inflated projection day. Configurable via `line.min_season_minutes` in model-config. |
| Line tab auto-switch to available direction | `index.html` | `switchLineDir` auto-corrects to `under` if `over` has no pick and vice versa, instead of showing "No X pick today" |
| `next_slate_pending` handling for both null | `index.html` | `switchLineDir` shows pending card (not "No X pick today") when both picks are null; prevents false "missing direction" message |
| Predict tab next-day transition | `index.html` | `loadSlate()` busts stale previous-day predictions; Predict tab no longer stuck on finished slate after midnight rollover |
| `next_slate_pending` re-fetch fix | `index.html` | `_renderLineLOTDFromState()` resets `LINE_LOADED_DATE = ''` on `next_slate_pending` so next tab visit re-fetches; prevents stale "Tomorrow's pick coming soon" after picks become available |
| Retry button on pending card | `index.html` | `renderNextSlatePending()` now includes a "Check for picks" button that calls `fetchLineOfTheDay()` directly |
| Line-of-the-day load path: no L5 re-fetch | `api/index.py` | `recent_form_values` (L5) is fetched once at fresh-generation time and stored in the GitHub JSON. Load paths (fast-path today + next-slate) never re-fetch L5 — use whatever is in the file; card falls back to `recent_form_bars`. Eliminates 10-30s cold-start nba_api call on every load. |
| Line tab + Ben timeout bumps | `index.html` | Line tab `/api/line-of-the-day` timeout 60s→90s; Ben context load 10s→30s. Fixes "Couldn't reach the server" on first load and "Line data unavailable" in Ben. |
| Ben briefing timeout precision | `index.html` | `initLabPage` error-fallback briefing and `showLabUnlocked` context load raised to 30s; auto-retry `/api/lab/status` standardized to 10s (was 15s). Context-load paths get 30s; user-triggered refresh actions stay at 10s for responsiveness. |
| `_CONFIG_DEFAULTS` sync | `api/index.py` | Fallback defaults match `data/model-config.json`: `compression_divisor` 5.5, `compression_power` 0.72, `rs_cap` 20.0, `ai_blend_weight` 0.35, `per_player_cap_minutes` 2.0, `big_market_teams` inline fallback removes MIL/DAL/PHX. Prevents silent model behavior change on GitHub outage. |
| `auto_improve_threshold_pct` externalized | `api/index.py`, `data/model-config.json` | `IMPROVEMENT_THRESHOLD` reads from `_cfg("lab.auto_improve_threshold_pct", 3.0)`. Tunable via Ben without code deploy. |
| Line engine stat floors externalized | `api/line_engine.py`, `data/model-config.json` | `_STAT_META` and `stat_configs` min_season floors now read from `line_config.get("stat_floors", {})`. Tunable via `line.stat_floors` in model-config. No behavior change — defaults match prior hardcoded values. |
| Cron schedule restored | `railway.toml` | `/api/refresh-line-odds` cron fixed from `0 */3 * * *` (every 3h) to `55 * * * *` (hourly at :55). **Superseded:** now game-window-only — see **Cron Schedule** table. |
| `line_history` parallel fetch + 3-min cache | `api/index.py` | CSV + JSON files fetched in parallel via `ThreadPoolExecutor(8)`; 3-min result cache (`line_history_v1`) avoids repeated cold-start GitHub round-trips; cache cleared by global cold reset |
| 3-layer slate cache (generate once per day) | `api/index.py` | `/tmp` → GitHub `data/slate/` → full pipeline. First request generates and persists; all subsequent requests serve from cache. Reduces API calls from N per visit to ~6-8 per day |
| `/api/injury-check` cron endpoint | `api/index.py`, `railway.toml` | Every 2h: bust RotoWire cache, check cached players, regenerate only affected games. Lock-guarded, CRON_SECRET-protected |
| GitHub cache removed from line engine | `api/index.py` | `_games_cache_from_github()` removed from `_run_line_engine_for_date()` and `_get_projections_for_date()` — added latency without benefit for line paths |
| "Generating picks..." message removed | `index.html` | 12s setTimeout that showed misleading "Generating picks..." during Line page load removed; skeleton card provides sufficient loading feedback |
| Force-regenerate endpoint | `api/index.py`, `index.html` | `GET /api/force-regenerate?scope=full\|remaining` — two scenarios: (1) dev deploys mid-slate → auto-detects SHA mismatch, regenerates all games in background; (2) user wakes up late → "Late Draft" banner on Predict tab regenerates picks for remaining games only. Both update `data/predictions/` CSV and all cache layers. CRON_SECRET-gated. |
| Deploy SHA tracking | `api/index.py` | `deploy_sha` stamped in slate cache at generation + GitHub write time. `/api/slate` locked path compares cached SHA vs current `RAILWAY_GIT_COMMIT_SHA`; on mismatch fires background `_force_regenerate_sync("full")`. |
| Late Draft UI | `index.html` | Banner with "Generate Late Draft" button shown on Predict tab when slate is locked but remaining games exist. Calls `/api/force-regenerate?scope=remaining`, updates SLATE, re-renders, and hides banner on success. |
| `auto-resolve-line` explicit timeout | `index.html` | `fetchWithTimeout('/api/auto-resolve-line', {}, 15000)` — was using implicit 10s default; now explicitly documents the 15s intent for this endpoint. |
| `var` → `let` modernisation | `index.html` | Converted `_slateAutoRefreshCount`, `_slateAutoRefreshTimer`, `_slateNextDayPoll`, `_predSavedLockedCount`, `_lateDraftTriggered` from `var` to `let` for block-scope consistency with the rest of the module. |
| `LINE_HIST_DATA` declaration hoisted | `index.html` | Moved `let LINE_HIST_DATA = null` from inside the `renderLineHistory` section (line ~3410) to the Line globals block (line ~2901) alongside `LINE_RESOLVE_POLL` and `LINE_LIVE_POLL`. Eliminates forward-reference anti-pattern. |
| `oracle-ball.svg` cache header | `railway.toml` | Added `Cache-Control: public, max-age=86400` entry for `/oracle-ball.svg` to match the existing `server.py` header and keep cache strategy consistent across Railway and local dev. |
| Grep tags for key helpers | `api/index.py` | Added `# grep:` tags for `LOCK HELPERS` (`_is_locked`, `_is_past_lock_window`, `_et_date`), `ALL GAMES FINAL`, `NEXT SLATE DATE` (`_find_next_slate_date`), and `FORCE REGENERATE SYNC`. Updated CLAUDE.md navigation table to match. |
| Per-direction independent rotation | `api/index.py`, `index.html` | Over and Under picks now rotate independently — when one direction's game finishes, that direction shows the next-slate pick while the other stays live. Fixes: over picks gap in history (picks not generated for future days when only one direction existed), Shai-style stuck active pick (resolved game still showing as active pick card). |
| Auto-resolve next-day fill | `api/index.py` | `auto_resolve_line` now fills missing directions in existing next-day pick files (was skip-if-file-exists). Merges new engine output with existing picks. |
| Live poll per-direction rotation | `index.html` | `_startLineLivePoll` re-fetches `/api/line-of-the-day` when any game finishes (was waiting for both). `_lineRotationTriggered` gate prevents redundant re-fetches. |
| Cache bust both dates | `api/index.py` | `auto_resolve_line` busts line cache for both `pick_date` AND `today` (differ on midnight rollover). |
| `force-regenerate` scope=remaining unprotected | `api/index.py` | `/api/force-regenerate?scope=remaining` no longer requires CRON_SECRET — it's user-triggered from the Late Draft button. `scope=full` stays CRON_SECRET-gated. |
| Vercel → Railway migration | `CLAUDE.md`, `api/index.py` | All Vercel references replaced with Railway equivalents (deployment model, env vars, cron schedule, URLs, watchPatterns). `VERCEL_GIT_COMMIT_SHA` → `RAILWAY_GIT_COMMIT_SHA` at all call sites. `vercel.json` noted as legacy/unused. |
| Per-game card boost pill removed | `api/index.py` | `_build_game_lineups` now zeroes `est_mult` in returned player data (not just MILP input) so the `+X.Xx card` pill never renders on per-game (THE LINE UP) cards where card boost is irrelevant. |
| Over model tightening | `api/line_engine.py`, `api/index.py`, `data/model-config.json` | Four over-specific changes (under model untouched): (1) `stat_floors.rebounds` 2.0→5.5 — only legit rotation bigs qualify for rebounds picks. (2) `min_edge_other_over: 2.5` (new config key) — over picks for rebounds/assists need a 2.5+ edge; under picks keep 1.5. (3) `recent_form_over_ratio` 1.08→1.15 — require 15% recent spike to unlock +12 confidence bonus. (4) Claude AVOID clause updated — rebounds/assists overs require a catalyst (cascade, opp-B2B, or 230+ total). Tests added for `min_edge_other_over` asymmetry. |
| Odds cron schedule fix | `railway.toml` | Same as above (3h → hourly). **Superseded:** game-window-only schedule in current `railway.toml`. |
| `predMin` tolerance band | `api/index.py`, `data/model-config.json` | Chalk pool allows `predMin` up to 2.0 min below `season_min` (`projection.pred_min_tolerance`); moonshot allows 3.0 min (`moonshot.pred_min_tolerance`). Saved 4 missed players from Mar 17 (Carrington 1.5 gap, Jenkins 0.7, Riley 0.8, Champagnie 1.8). |
| Separate moonshot pts floor | `api/index.py`, `data/model-config.json` | Universal floor in `project_player()` lowered to 4.0 (`min_pts_projection_moonshot`); chalk enforces 7.0 separately. Oso Ighodaro (4 PPG, +2.9x, Value 16.4) now enters moonshot pool. `min_pts_per_minute_moonshot` = 0.20 (chalk keeps 0.28). |
| `min_chalk_rating` synced | `data/model-config.json` | Config value 4.0 → 3.5 to match code fallback and CLAUDE.md documentation. Mar 17 showed 7/9 missed players filtered by this gate. |
| Odds API draft enrichment | `api/index.py`, `data/model-config.json` | `_enrich_projections_with_odds()` blends sportsbook player props into projections. Upward-only 20% blend when books diverge 15%+ from model (`odds_enrichment.*` config). Also nudges `predMin` proportionally. Odds data passed to Claude context layer. |
| Web intelligence (Layer 1) | `api/index.py`, `data/model-config.json` | `_fetch_nba_news_context()` — once-per-slate Sonnet call (`context_layer.web_search_model`, default `claude-sonnet-4-6-20250514`) with `web_search_20250305`. Downgraded from Opus — news gathering is search+summarize, doesn't need Opus reasoning. Player/RS-aware: when `all_proj` is passed, top 20–25 by rating are included so intel prioritizes likely draft picks. Results injected into context pass as "RECENT NBA NEWS". Config: `context_layer.web_search_enabled`, `context_layer.web_search_model`, `timeout_seconds`. |
| Context pass (Layer 2) | `api/index.py`, `data/model-config.json` | RS adjustment uses Sonnet (`context_layer.model`, default `claude-sonnet-4-6-20250514`). Downgraded from Opus — structured JSON task (RS multipliers) handled well by Sonnet at ~15x lower cost. Directive: map each RECENT NBA NEWS bullet to specific players and up/down adjustments. |
| Lineup review (Layer 3) | `api/index.py`, `data/model-config.json` | `_lineup_review_opus()` — after MILP, Sonnet + web_search reviews assembled Starting 5 and Moonshot; can suggest swaps for late-breaking news; auto-applies valid swaps. Config: `lineup_review.enabled` (default off), `lineup_review.model`, `lineup_review.timeout_seconds`. Non-fatal: on error returns original lineups. |
| Claude cost reduction | `api/index.py`, `data/model-config.json` | Layer 1 (news) and Layer 2 (context pass) downgraded from Opus → Sonnet. Layer 1.5 (Claude DvP matchup intel) disabled — ESPN def stats in `_compute_matchup_factor()` provide equivalent signal. ~95% cost reduction on pipeline Claude calls ($5-12/slate → $0.25-0.50/slate). All config-reversible. |
| Health check timeout + Vercel cleanup | `index.html` | Health pre-warm converted from raw `fetch()` to `fetchWithTimeout(..., 5000)`. Stale Vercel references in comments updated to Railway. All frontend fetches now use `fetchWithTimeout` (except lab/chat SSE which uses manual AbortController). |
| Line card flash (live poll) | `index.html` | Live poll only updates card DOM when live snapshot key (`stat_current`, `clock`, `period`, `pace`) changes (`_lineLastLiveKey`); avoids full re-paint every 60s. |
| Line first load after hit | `index.html` | Same-day Line tab open runs background `fetchLineOfTheDay(true, true)` so rotated pick (post–resolution) appears without showing skeleton; failures leave cached card as-is. |
| Core pool architecture | `api/index.py`, `data/model-config.json`, CLAUDE.md | Single up-to-15 player core pool; Starting 5 and Moonshot are two 5-of-core configurations (reliability vs ceiling). Config: `core_pool.enabled`, `core_pool.size`, `core_pool.metric` (`"rs"` ranks by raw projected RS). Layer 2/3 prompts and Layer 3 swap-in respect core pool. |
| RS-first strategy (v8) | `api/index.py`, `data/model-config.json` | Strategy shift: top RS scorers over everything. (1) `core_pool.metric="rs"` ranks core by raw projected RS, not EV. (2) `chalk_milp_rs_focus=0.85` nearly neutralizes boost in MILP — RS drives slot assignment. (3) `moonshot.rs_bypass` lets high-RS players (5.0+, 25min+) bypass boost floor. (4) `boost_leverage_power` 1.2→0.6 halves boost dominance. (5) Boost floors lowered: chalk 1.0→0.5, moonshot 1.0→0.5. (6) `star_anchor.max_count` 2→3, `min_boost` 0.8→0.3. (7) `ai_blend_weight` 0.4→0.5 for better RS ordering. All config-reversible. |
| 13-date accuracy audit fixes | `api/index.py`, `data/model-config.json` | RS calibration_scale=1.15 (34% under-projection), AI blend 0.5→0.35 (LightGBM compression), 3-layer card boost (config→170 ownership samples→sigmoid), moonshot gates widened (min_minutes 12, wildcard 6, pts 3.0). |
| Web search in Line engine | `api/line_engine.py`, `api/index.py` | Line picks now receive `news_context` from Layer 1 (`_fetch_nba_news_context`). Claude Haiku sees injury updates, rotation changes, rest decisions when making over/under picks. Reuses same cache as draft model — no extra API calls. |
| Layer 1→3 news passthrough | `api/index.py` | `_lineup_review_opus()` now accepts `news_context` param. Main slate generation pre-fetches news and passes to Layer 3, reducing redundant web searches. Layer 3 focuses on truly late-breaking news (last 2-4h). |
| Line over model tightening | `api/line_engine.py` | Over picks penalized -12 confidence when no catalyst signals (cascade/form/B2B). Prompt expanded with explicit over-specific rules citing 17% historical hit rate. Recent form trend indicators (↑HOT/↓COLD) added to player context. |
| `.ok` check before `.json()` | `index.html` | `/api/lab/update-config` response: check `.ok` before parsing JSON to prevent misleading error handling. |
| Vercel→Railway comment cleanup | `api/index.py` | Replaced 7 stale Vercel references with Railway equivalents (watchPatterns, container instances, timeout limits). |
| MILP solver audit — 3 fixes | `api/asset_optimizer.py`, `api/index.py` | (1) Removed `leverage_top_slots` constraint — mathematically wrong for additive formula `RS × (Slot + Boost)` since boost is player-constant; solver naturally places highest RS in highest slot. (2) Two-phase moonshot optimization — Phase 1 selects players using shaped ratings (boost leverage, variance uplift); Phase 2 re-assigns slots using raw RS for optimal placement. Decouples selection from slotting. (3) Removed position-per-team constraint — Real Sports has no position requirements; artificial constraint blocked legitimate same-position stacks. |
| High-boost role pathway | `api/index.py`, `data/model-config.json` | Rotation players with 2.0x+ boost bypass minutes floor in both moonshot and chalk pools (`is_high_boost_role`, `is_chalk_high_boost_role`). Config: `moonshot.high_boost_role.*`, `projection.chalk_hbr_*`. |
| RS calibration weights | `api/index.py`, `data/model-config.json` | DFS weight recalibration from 13-date audit (reb 0.95→0.65, ast 0.3→0.55). `_infer_player_archetype()` detects star/scorer/big/pure_rebounder/wing_role. `archetype_calibration` applies per-archetype RS multipliers. `scorer_upside` gives efficient scorers moonshot bonus. |
| Cascade cap fix | `api/index.py`, `data/model-config.json` | `cascade.per_player_cap_minutes` raised from 2-3 to 10.0 so primary backups correctly inherit starter-level minutes. `cascade_rs` and `role_spike_rs` add RS uplift for cascade-elevated players. |
| Roto confirmed rating exception | `api/index.py`, `data/model-config.json` | Confirmed rotation players with high boost (2.5x+) bypass `min_rating_floor` in moonshot (use 2.2 floor instead). Context pass includes cascade_bonus and roto_status. |
| Max per game MILP constraint | `api/asset_optimizer.py`, `api/index.py`, `data/model-config.json` | `lineup.chalk_max_per_game=2`, `moonshot_max_per_game=2` — limits players from same game matchup. Prevents over-concentration in single games. |
| Min big boost MILP constraint | `api/asset_optimizer.py`, `api/index.py`, `data/model-config.json` | `lineup.chalk_min_big_boost_count=1`, `moonshot_min_big_boost_count=1` — ensures minimum high-boost players in lineup for card boost value. |
| Line narrative signals | `api/line_engine.py` | `_generate_signals()` produces 8 signal types (high_total, low_total, matchup, books_agree, minutes_drop, blowout_risk, close_game, cascade) so both over and under picks explain WHY, not just restate numbers. |
| MAE drift check cron | `api/index.py`, `railway.toml` | Weekly (Monday 6am UTC) MAE drift monitoring — computes 7-day rolling MAE, writes backend flag if > 2.5 threshold. CRON_SECRET-gated. |
| Line force regenerate | `api/index.py` | `/api/line-force-regenerate` — force-generate today's line picks, overwrite stale artifacts, bust history cache. |
| Ben chat history | `api/index.py` | `/api/lab/chat-history` — persisted daily chat history with thread-safe read via `_BEN_CHAT_HISTORY_LOCK`. |
| Explicit fetch timeouts (audit) | `index.html` | Added explicit 10s timeout to `/api/save-line` background save. All fetches now have documented timeouts. |
| Silent catch logging | `index.html` | Added `console.warn` to silent `.catch()` blocks on lab/status pre-warm and save-line background save for debugging visibility. |
| Odds API field swap fix | `api/index.py` | **Critical**: `_build_player_odds_map` and `_fetch_odds_line` had `name`/`description` swapped — Odds API returns `{name: "Over"/"Under", description: "Player Name"}` but code read them backwards. `result_map` was always empty so every parlay and line pick used synthetic/projection-based fallback lines instead of real Vegas lines despite a valid `ODDS_API_KEY`. Fixed: `description` → player_key, `name` → direction. |
| Resolved line pick rotation fix | `api/index.py` | When a direction's game resolved (e.g., Deni Avdija Under HIT) and next-slate pick generation returned `None`, the endpoint only updated `final_under` when `next_under` was non-None — leaving the resolved pick as the "active" pick card. Fixed: always set `final_under = next_under` (unconditional). Added `_had_resolved` flag so `next_slate_pending` is correctly returned when both directions resolved but next-slate fails. |
| Parlay synthetic line snapping | `api/index.py`, `api/parlay_engine.py` | Synthetic fallback used `math.floor(proj_val * 2) / 2` which produced whole-number lines (5.3 → 5.0, 21.3 → 21.0). Changed to `round(proj_val * 2) / 2` for nearest-0.5 snap (5.3 → 5.5, 21.3 → 21.5). Also removed unnecessary `round(*2)/2` snap on real Odds API values in `parlay_engine.py` — Odds API lines are already properly formatted. |
| Parlay engine + tab | `api/parlay_engine.py`, `api/index.py`, `index.html` | New "Parlay" tab (5th nav icon, electric purple accent `#d946ef`). Backend: `_fetch_gamelog()` ESPN gamelog helper (cached), `_fetch_gamelogs_batch()` parallel fetcher, `_run_parlay_engine_sync()` full pipeline, `GET /api/parlay` endpoint (30-min cache, rate-limited, auto-saves to `data/parlays/{date}.json`), `GET /api/parlay-history` (lazy ESPN resolution, 10-min cache). Engine: Z-score hit probability, American odds → implied probability, blended confidence (55% model / 45% Vegas), anti-fragility filters (blowout >8.5 spread, minutes CV >0.30, GTD star teammate, injury), correlation scoring (VETO same-team rebounds/assists over, boost PG assists + teammate points, shootout opposing scorers), diagnostic filter funnel logging. Frontend: `PARLAY_STATE`, `PARLAY_HIST_DATA`, `initParlayPage()`, `fetchParlay()` (90s timeout), stacked ticket card with 3 legs + combined probability + narrative. Recent Parlays history section (scrollable, hit/miss pills, hit rate + streak stats). Bottom-sheet `parlayModal` with full ticket detail + leg-by-leg resolution (actual stats, HIT/MISS coloring). ARIA accessibility (`aria-live`, `aria-modal`, `aria-label`, `role`). Config: `parlay.*` in model-config.json. Tests: 81 tests in `tests/test_parlay.py`. |

| Per-game draft strategy redesign (v60) | `api/index.py`, `data/model-config.json`, `index.html` | 18-game / 76-lineup empirical analysis (Jan 6 – Mar 23). 6-step pipeline: game script → per-game strategy adjustments (F1-F6) → eligibility gating → MILP → 5! permutation validation → strategy metadata. New functions: `_per_game_strategy()`, `_per_game_adjust_projections()`, `_validate_slot_assignment()`. Config: `per_game.*` (20 params). Frontend: strategy insight bar, ANCHOR/FAV pills, color-coded strategy badge. Strategy types: Balanced Build / Standard Build / Blowout Lean + Shootout/Grind overlays. Score bounds widened to (20, 42). 38 new tests. |
| Line engine stress test recalibration | `api/line_engine.py`, `data/model-config.json` | End-of-season analytical review of the 100-point confidence system. **Auto-fade matrix** (`_check_auto_fade`): (1) B2B guard exhaustion — guards/wings on own B2B vetoed for PTS/AST overs (glycogen depletion). (2) Blowout truncation — starters in spread>=10 games vetoed for ALL overs (4th-quarter benching). (3) Rotation squeeze — bench players in tight games (spread<=4) vetoed for overs (shortened rotations). **Percentage-based edge scaling**: Flat 2.5 edge threshold replaced with 18% dynamic scaling for rebounds/assists — `_compute_pct_edge()` makes edge proportional to line volume (2.5 on 5.5 line = 45% improbable; 2.5 on 12.5 line = 20% plausible). **Momentum ratio lowered**: `recent_form_over_ratio` 1.15→1.07 — stops buying at peak variance where sportsbooks have already adjusted. **Juice-as-under-signal**: Heavy over juice (-130+) now generates +8 under confidence bonus instead of vetoing unders — recognizes that juice reflects public bias management, not sharp money signal. **Player B2B signals**: Player's own B2B generates +10 under bonus / -8 over penalty (new signal types). **Relaxed trivial line floors**: `stat_floors_under` (pts 4.0, reb 3.5, ast 1.0) — counting stats accumulate in increments of 1 with hard floor of zero, making low-line unders highly predictable. **Tiered blowout bonus**: Spread 8-10 gives +6, spread 10+ gives +10. **Claude prompt updated** with auto-fade rules, percentage edge guidance, juice-as-friend rule, trivial line under validation, player B2B fatigue context. Config: `line.auto_fade.*`, `line.pct_edge_rebounds`, `line.pct_edge_assists`, `line.juice_under_threshold`, `line.stat_floors_under`. 34 new tests. |
| Parlay post-mortem v2 optimization | `api/parlay_engine.py`, `api/index.py`, `data/model-config.json` | Quantitative post-mortem of 2024-2026 NBA parlay architectures. **Auto-fade matrix**: (1) Centers rebounds over vs switch-heavy defenses (BOS/CLE/MIN/OKC) auto-faded — paint positioning neutralized by perimeter switching. (2) B2B correlated pair penalty (0.75x) — cognitive/physical fatigue breaks assists→points chain; 57% spread-failure rate on B2B. (3) Perimeter-to-perimeter correlation fade (0.95x penalty) — 3PT-dependent scoring chain is fragile. (4) Fake juice trap — high L5-L10 hit rate (>80%) + low season model_prob (<55%) = regression trap at peak valuation. **Threshold recalibrations**: Spread tightened 6.5→5.0 (blowout mirage — 51.4% of games decided by 10+ pts). Game total floor 225.5 (possession guarantee for correlated pairs). CV-based market match validation (volatile ≠ reliable). Dynamic Leg 1 substitution (rebounds→pts/ast vs elite defense). **Tiered correlation enhancers**: PnR-to-rim (1.20x for C/PF with 7+ reb — most stable conversion), pace asymmetry (1.06x for 232+ total games), rest advantage (1.08x team rested vs opponent B2B). Config: `parlay.auto_fade.*`, `parlay.min_game_total`, `parlay.market_match_max_cv`, `parlay.pnr_rim_boost`, `parlay.pace_boost*`, `parlay.rest_advantage_boost`. 25 new tests (101 total in test_parlay.py). |

| Mar 27 draft review — S5 coverage + RS inflation fix | `api/index.py`, `data/model-config.json`, `tests/test_fixes.py` | **Problem**: Mar 27 slate — moonshot hit 1/5 (Jenkins #1, but Sasser/McDermott/Alvarado/Carter all missed). S5 blocked top RS performers (Duren +0.6x, Knueppel +0.8x, DeRozan +1.0x) via 1.5x chalk boost floor. RS over-projection: 4 stacked post-compression multipliers (cascade_rs × role_spike × breakout × archetype = 1.87×) inflated Sasser from RS 3.0 to 6.1. **Fix 1**: `chalk_min_boost_floor` 1.5→0.3, star anchor widened (min_season_pts 20→12, min_rating 4.5→3.5, max_count 2→3) so top RS performers enter chalk pool and MILP (rs_focus=0.75) selects them. **Fix 2**: Post-compression multiplier cap (`max_post_compression_mult: 1.40`) — tracks `_pre_boost_rs` before archetype/cascade/spike/breakout boosts; clamps total inflation to 1.40×. Prevents bench players from inflating from RS 3 to 6+. **Fix 3**: HBR `min_rating: 2.5` (already in config+code from prior session). **Goal**: S5 catches top RS carries (Duren, Knueppel, DeRozan, Banchero), Moonshot catches one-slate-ahead contrarians (Jenkins, Plowden, Huerter); 2-3 "lock" players overlap in both lineups. |

| 3-Tier Cascade Boost Prediction | `api/boost_model.py` (new), `api/index.py`, `tests/test_fixes.py`, `tests/test_core.py` | **Architecture**: Replaced LightGBM `boost_model.pkl` + `drafts_model.pkl` with deterministic 3-tier cascade calibrated from 2,234 player-date records. **Tier 1** (returning, ≤14d): prev_boost + 6 adjustment factors (RS decay, draft popularity, mean reversion, trend, gap blend, boundary persistence). **Tier 2** (stale, >14d): staleness-weighted blend of historical mean and API-derived PQI estimate. **Tier 3** (cold start): Player Quality Index from season stats. Post-prediction star PPG caps and per-team ceilings. Key insight: prev_boost correlates +0.957 with actual boost; 88.2% of day-over-day changes are within ±0.3. Removed: `_ensure_boost_model_loaded()`, `_lgbm_predict_boost()`, `_ensure_drafts_model_loaded()`, `_lgbm_predict_log1p_drafts()`, `_ensure_boost_priors_loaded()`, `_get_boost_prior()`, `BOOST_MODEL/FEATURES` globals, `DRAFTS_MODEL/FEATURES` globals. Updated `_CONFIG_DEFAULTS.card_boost` (ceiling 3.5→3.0, floor 0.2→0.0, removed `ml_additive_correction`/`max_prior_weight`, added `star_ppg_tiers`/`team_boost_ceiling`). 17 new tests (13 cascade + 4 integration). 693 tests pass. |

## Loading audit

**docs/LOADING_AUDIT.md** — Catalogs frontend loading states, fetch timeouts, skeletons, async state pattern, and Line tab fixes (card flash, first-load-after-hit). All blocking API calls use `fetchWithTimeout`; no critical gaps for production.

## Production audit

Full audit: [docs/PRODUCTION_AUDIT.md](docs/PRODUCTION_AUDIT.md). Implemented: GitHub error sanitization (no leak to client), `GET /api/health`, `GET /api/version`, cron secret on protected endpoints (including `/api/cold-reset`), and `fetchWithTimeout` for lab/backtest and lab/update-config.

**Lock & routing audit:** [docs/LOCK_AND_ROUTING_AUDIT.md](docs/LOCK_AND_ROUTING_AUDIT.md). Covers all lock usage (slate, picks, save-predictions, lab status, line odds) and Railway/FastAPI routing. Fixes applied: `/api/lab/status` wrapped in try/except — on any exception returns 200 with `locked: true` and reason "Server temporarily unavailable — try again" so the frontend shows a retry instead of a generic fetch failure; ESPN-down GitHub lock check now uses `lock_content, _ = _github_get_file(...)` and `if lock_content:` (was incorrectly checking the tuple).

### Mar 23 Production Audit

**Critical fix (Odds API — Mar 23):** `_build_player_odds_map` and `_fetch_odds_line` had `name`/`description` fields swapped when parsing Odds API outcomes. The API returns `{name: "Over"/"Under", description: "Player Name"}` but code was reading `name` as the player key and `description` as direction. `result_map` was always empty → synthetic fallback triggered on every parlay and line pick. Fixed in commit `024b206`. Regression guard: `TestOddsApiFieldMapping` in `tests/test_fixes.py`.

### Mar 17 Production & Model Audit

**Pipeline audit (179/179 tests pass):**
- Global exception handler active — no stack traces leak to clients
- Structured request logging (JSON with request_id, path, status, duration_ms)
- 39 `fetchWithTimeout` calls in frontend; 1 intentional raw `fetch()` (lab/chat SSE with manual AbortController)
- Thread pools: 8 workers (game/slate/picks/audit/line/Odds); 10 workers for parlay ESPN gamelog batch
- Rate limiting: thread-safe with `_RATE_LIMIT_LOCK` (parse-screenshot 5/min, lab/chat 20/min, line-of-the-day 10/min)
- 35 endpoints total, all correctly routed with proper CRON_SECRET gating

**Caching audit (all TTLs verified):**
- 3-layer slate cache: `/tmp/nba_cache_v19/` → GitHub `data/slate/` → full pipeline
- Game final: 60s locked / 180s pre-slate; Model config: 5 min; ESPN games: 5 min; RotoWire: 30 min; Line history: 10 min; Parlay history: 10 min; Odds fresh map: 10 min; Parlay ESPN gamelog cache: 30 min (`_TTL_L5`)
- Cache bust tombstone pattern working correctly

**Odds cron evolution:** `railway.toml` `/api/refresh-line-odds` went from `0 */3 * * *` → hourly `55 * * * *` → **current:** game-window only (`55 19,20,21,22,23,0,1,2,3,4,5,6 * * *` UTC) to cut cron load; see **Cron Schedule** table above.

**Model audit (Mar 17 leaderboard — "Highest Value" screenshot):**
- Role players dominate (13/14 top values RS 2.7-5.0) — validates v6 strategy
- Boost is dominant signal (+3.0x at RS 2.7 → Value 13.3+) — validates moonshot formula
- Stars don't win (Booker RS 5.5, +0.5x, 552 drafts = 13.7 value, ranked 11th)
- Player overrides systematically low by 0.1-0.2x vs actual (GP2: 2.8 override vs 3.0 actual; Riley: 2.8 vs 3.0; Santos: 2.6 vs 2.5 — this one is accurate)
- `min_chalk_rating` 4.0 correctly filters — all chalk winners RS ≥ 4.1
- `min_pts_per_minute` 0.28 well-calibrated — GP2 (~0.29) passes as intended
- Claude context layer correctly identified defensive role players (Draymond, GP2, Melton) as top performers

## Pre-deploy checklist (production finalization)

- **Env**: Required vars set in Railway (GITHUB_TOKEN, GITHUB_REPO, ANTHROPIC_API_KEY; optional ODDS_API_KEY, CRON_SECRET, DOCS_SECRET).
- **Tests**: Run `python3 -m pytest tests/ -v` locally when changing backend contracts. `TestJSSyntax` is skipped (superseded by TypeScript compiler). For frontend: `cd frontend && npx tsc --noEmit`.
- **Docs**: CLAUDE.md and README.md reflect current endpoints, crons, lock/cache behavior, and core-pool architecture; docs/LOADING_AUDIT.md for loading and timeouts.
- **Health**: Use GET `/api/health` for uptime monitoring; alert on non-200.
- **Loading**: All blocking fetches use `fetchWithTimeout`; Line tab uses background re-fetch on same-day and live-card update only when data changes (see docs/LOADING_AUDIT.md).

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
3. **Tests**: `pytest tests/ -v` (requires `pip install -r requirements.txt`). test_fixes.py covers lock/audit/cache; test_core.py covers helpers and line cache (`TestJSSyntax` is skipped — superseded by `cd frontend && npx tsc --noEmit`). Deploy triggers on push to main; verify on `the-oracle.up.railway.app`.
4. **Data layer**: All persistent state in GitHub via Contents API (`data/` directory). No database.
5. **Frontend state**: Zustand stores in `frontend/src/store/` (`uiStore`: activeTab, lineDir, etc. | `labStore`: messages, system). React Query hooks in `frontend/src/api/` for all server state.
6. **Cache**: 3-layer: `/tmp` (ephemeral) → GitHub `data/slate/` (persistent) → full pipeline. Check `CACHE_DIR` in `api/index.py` for the current tmp path (versioned, e.g. `/tmp/nba_cache_v19/`). `/api/cold-reset` clears/regenerates caches + config. `_bust_slate_cache()` invalidates both layers.
7. **Config**: `data/model-config.json` on GitHub — Ben/Lab writes here, backend reads with 5-min TTL.
