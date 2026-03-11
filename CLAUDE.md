# Basketball — Real Sports Draft Optimizer

## What This Is

A daily NBA draft optimizer for the **Real Sports** app. It projects player Real Scores, estimates card boosts, and builds optimized 5-player lineups using MILP (mixed-integer linear programming). Deployed on **Vercel** as a serverless Python (FastAPI) backend + single-page HTML frontend.

## How Real Sports Works

- Users draft 5 NBA players each day
- Each player earns a **Real Score** (RS) based on in-game impact (not just box score stats)
- Each player gets a **Card Boost** inversely proportional to how many people drafted them (popular players get low boosts, obscure players get high boosts)
- **Total Value = Real Score × (Slot Multiplier + Card Boost)**
- Slot multipliers: 2.0x, 1.8x, 1.6x, 1.4x, 1.2x (user manually assigns their 5 picks to slots pre-game)
- The winning strategy is drafting **high-RS role players with huge card boosts**, not superstars

## Architecture

```
index.html             — 4-tab frontend (Predict | Line | Ben | Log, vanilla JS)
api/index.py           — FastAPI backend (all endpoints, projection engine, Lab/Line)
api/real_score.py      — Monte Carlo Real Score projection engine
api/asset_optimizer.py — MILP lineup optimizer (PuLP)
api/line_engine.py     — Prop edge detection pipeline (Odds API + confidence model)
api/rotowire.py        — RotoWire lineup scraper (free tier: availability + injury flags)
data/model-config.json — Runtime model config (Lab writes here; 5-min cache)
data/predictions/      — Git-tracked daily prediction CSVs (via GitHub API)
data/actuals/          — Git-tracked daily actual result CSVs (via GitHub API)
data/audit/            — Git-tracked daily audit JSONs (auto-generated on save-actuals)
data/lines/            — Git-tracked daily Line of the Day picks (via GitHub API)
data/slate/            — GitHub-persisted prediction cache ({date}_slate.json, {date}_games.json)
data/locks/            — Cold-start recovery: {date}_slate.json written at lock-promotion time
data/skipped-uploads.json — User-selected dates to skip uploading (persists skip decisions across sessions)
lgbm_model.pkl         — LightGBM model bundle {model, features} — retrained by retrain-model.yml
train_lgbm.py          — Training script (11 features, run locally or via GitHub Actions)
vercel.json            — Vercel config (routes, crons, 300s timeout on Pro plan)
server.py              — Local dev server (uvicorn)
```

## UI Structure

4-tab segmented control navigation (Apple glassmorphism pill style): **Predict | Line | Ben | Log**

- **Predict**: Live slate optimizer (Starting 5 + Moonshot), per-game analysis, Magic 8-ball loading animation. "Slate-Wide | Game" sub-tabs inline at top of tab.
- **Line**: Line of the Day — best player prop edge (gold accent). "Over | Under" sub-tabs inline at top of tab. Odds refresh hourly from Odds API; pick cards show "Odds · [time] CT".
- **Ben**: Plain chat interface with Claude (no quick-action buttons — user asks naturally). Teal accent. Locked during games, unlocked after final.
- **Log**: Historical drill-down — date strip, game grid, graded prediction cards vs actuals (no user input — upload happens through Ben). Cards have two states: **Pending** ("Waiting for results") and **Graded** (actual RS + ESPN box score stats overlaid on projections with hit/miss coloring).

### Sub-Nav Tabs (inline, not floating)
Both `predictSubNav` (Slate-Wide | Game) and `lineSubNav` (Over | Under) are inline `div.predict-sub-nav` elements positioned at the top of their respective tab pages. They match the `.mode-tab` visual language exactly — same height, padding, `border-radius:11px`, Barlow Condensed 800. Active states: predict = chalk blue, Over = gold (`--line`), Under = teal (`--lab`).

## Codebase Navigation (grep tags)

All major sections in `api/index.py` and `index.html` are tagged for fast searching. In `api/index.py` search for `# grep:`; in `index.html` grep for the section name (e.g. `LOG PAGE`) or function name (e.g. `initLogPage`).

```
grep: TEAM_COLORS              — team color hex map in index.html
grep: GLOBAL STATE             — SLATE, PICKS_DATA, LOG, LAB state objects
grep: TAB NAVIGATION           — switchTab, movePill, setPillAccent
grep: SLATE                    — loadSlate, /api/slate, Starting 5, Moonshot
grep: PER-GAME ANALYSIS        — runAnalysis, /api/picks
grep: CARD RENDERING           — renderCards, player-card, tcolor
grep: PREDICTION PERSISTENCE   — savePredictions, dedup guard
grep: LOG PAGE                 — initLogPage, selectLogDate, renderLogGrid, openLogDrilldown, drill-down
grep: LINE PAGE                — initLinePage, renderLinePickCard, switchLineDir, filterLineHistory, LINE_DIR
grep: LAB PAGE                 — initLabPage, LAB state, labCallClaude, buildLabSystemPrompt, _handleBenUpload
grep: github_storage           — _github_get_file, _github_write_file
grep: SLATE CACHE GITHUB       — _slate_cache_to_github, _games_cache_from_github, _bust_slate_cache
grep: CONSTANTS & CACHE        — _cp, _cg, _cs, _lp, _lg, ESPN, MIN_GATE
grep: ESPN DATA FETCHERS       — fetch_games, fetch_roster, _fetch_athlete
grep: INJURY CASCADE           — _cascade_minutes, _pos_group
grep: CARD BOOST               — _est_card_boost, _dfs_score
grep: GAME SCRIPT              — _game_script_weights, _game_script_label
grep: PLAYER PROJECTION        — project_player, pinfo, rating, est_mult
grep: GAME RUNNER              — _run_game, _build_lineups, chalk_ev
grep: INJURY CHECK             — /api/injury-check, RotoWire re-check, affected game regeneration
grep: CORE API ENDPOINTS       — /api/games, /api/slate, /api/picks, /api/health, /api/version
grep: LINE OF THE DAY ENGINE   — /api/line-of-the-day, run_line_engine
grep: BEN / LAB ENGINE         — /api/lab/*, _all_games_final, lab lock
```

## Key Endpoints

### Core
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Health check for monitoring (config + GitHub reachability) |
| `/api/version` | GET | Build identifier (VERCEL_GIT_COMMIT_SHA) for deploy checks |
| `/api/slate` | GET | Full-slate predictions (all games). Never returns 5xx; on backend exception returns 200 with `error: "slate_failed"` and empty lineups so the frontend shows "Slate temporarily unavailable" and Retry. |
| `/api/picks?gameId=X` | GET | Per-game predictions |
| `/api/games` | GET | Today's games with lock status |
| `/api/save-predictions` | POST | Save cached predictions to GitHub CSV (deduped — skips commit if unchanged) |
| `/api/parse-screenshot` | POST | Upload Real Sports screenshot, Claude Haiku parses it |
| `/api/save-actuals` | POST | Save parsed actuals to GitHub CSV + auto-generates audit JSON |
| `/api/audit/get?date=X` | GET | Pre-computed accuracy audit for a date (MAE, directional acc, misses) |
| `/api/log/dates` | GET | List dates with stored prediction/actual data |
| `/api/log/get?date=X` | GET | Predictions + actuals for a given date, grouped by scope |
| `/api/log/actuals-stats?date=X` | GET | ESPN box score stats (PTS, REB, AST, STL, BLK, MIN) for all players on a date's completed games |
| `/api/hindsight` | POST | Optimal hindsight lineup from actual RS scores |
| `/api/refresh` | GET | Clear cache + config cache (cron at 7pm UTC; requires CRON_SECRET when set) |
| `/api/injury-check` | GET | Cron: check RotoWire for newly OUT/questionable players; regenerate affected games only (requires CRON_SECRET when set) |

### Line of the Day
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/line-of-the-day` | GET | **Both** Over + Under picks (6 parallel Haiku calls: 3 stats × 2 dirs); returns `{over_pick, under_pick, pick}` |
| `/api/refresh-line-odds` | GET | **Hourly cron** — fetch current bookmaker line from Odds API and update `line`, `odds_over`, `odds_under`, `books_consensus`, `line_updated_at` on today's pick JSON. No-op if slate is locked. Returns `{status, updated, timestamp}` |
| `/api/save-line` | POST | Save `{over_pick, under_pick}` JSON + primary pick to CSV; backward-compat with legacy single-pick |
| `/api/resolve-line` | POST | Mark pick hit/miss given actual stat |
| `/api/auto-resolve-line` | GET | **Cron** — resolves each pick when its game ends; generates next-day picks when both resolve (requires CRON_SECRET when set) |
| `/api/line-live-stat` | GET | Fetch live in-game stat value for pick tracking (single-game lock check) |
| `/api/line-history` | GET | Recent picks with streak + hit rate (only resolved picks — never pending) |

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
| `/api/lab/chat` | POST | Proxy to claude-opus-4-6 with Lab system prompt (keeps key server-side) |
| `/api/lab/skip-uploads` | POST | Record dates the user skips uploading; persists to `data/skipped-uploads.json` |

**Admin / optional (not used by main UI):** `POST /api/reset-uploads` — deletes actuals + audit for a date (admin/debug). `POST /api/hindsight` — optimal hindsight lineup from actual RS (Ben-driven or future). `GET /api/version` — build identifier for deploy/monitoring.

## App init and tab data flow

- **Startup:** `loadSlate()` and `initGameSelector()` run in parallel: `GET /api/slate` (10s) and `GET /api/games` (10s). Predict tab is default; slate list and game dropdown populate.
- **Tab load (lazy):** Line, Log, and Lab load on first visit when `switchTab()` is called:
  - **Line:** `initLinePage()` → fire-and-forget `GET /api/auto-resolve-line`, then `GET /api/line-of-the-day`, `GET /api/line-history`; live stat poll and resolve poll when applicable.
  - **Log:** `initLogPage()` → `GET /api/log/dates`, then `buildLogDateStrip()` (60-day strip); selecting a date calls `GET /api/log/get?date=X`; drill-down may call `GET /api/log/actuals-stats?date=X` (cached in `LOG._statsCache`).
  - **Lab:** `initLabPage()` → pre-warm `GET /api/health`, then `GET /api/lab/status`; on unlock, loads briefing + config-history + line-of-the-day + slate + log for context; lock poll every 120s when locked.

## Environment Variables (Vercel)

- `GITHUB_TOKEN` — GitHub PAT with repo scope (for CSV + config read/write via Contents API)
- `GITHUB_REPO` — e.g. `cheeksmagunda/basketball`
- `ANTHROPIC_API_KEY` — Claude Haiku (screenshot OCR) + claude-opus-4-6 (Ben/Lab chat)
- `ODDS_API_KEY` — The Odds API for player prop lines (Line of the Day)
- `CRON_SECRET` — (optional) When set, cron-only endpoints (`/api/refresh`, `/api/auto-resolve-line`, `/api/lab/auto-improve`, `/api/injury-check`) require `Authorization: Bearer <CRON_SECRET>`. Vercel injects this when invoking crons.
- `DOCS_SECRET` — (optional) When set, `/docs`, `/redoc`, and `/openapi.json` require `?docs_key=<value>` or `X-Docs-Key` header so only people with the secret can browse/test the API.

**OpenAPI docs:** FastAPI serves `/docs` (Swagger UI) and `/redoc` in production. Use them to browse and try endpoints. When `DOCS_SECRET` is set, append `?docs_key=<DOCS_SECRET>` to the URL or send the header to access.

## Runtime Config System

Model parameters are stored in `data/model-config.json` on GitHub. The backend loads this
file at startup and caches it for 5 minutes. The Lab writes updates via the GitHub Contents API.

- **No redeploy needed** to tune parameters — changes take effect within 5 minutes
- **Fallback to defaults** if GitHub is unreachable — app never breaks
- Use `_cfg("dot.path", default)` helper anywhere in `api/index.py` to read config
- `/api/refresh` also clears the config cache for immediate effect

## 3-Layer Slate Cache (Generate Once, Serve Cached)

Predictions are generated **once per day** and cached. Subsequent requests serve from cache instead of re-running the full pipeline (ESPN + LightGBM + Monte Carlo + MILP). This reduces API calls from N per user visit to ~6-8 per day max.

### Cache Layers
1. **Layer 1 — `/tmp` (warm Vercel instance)**: In-memory file cache. Fastest, but ephemeral on cold start.
2. **Layer 2 — GitHub persistent cache (`data/slate/`)**: `{date}_slate.json` (full slate with lineups) and `{date}_games.json` (per-game projections keyed by gameId). Survives cold starts. Used by `/api/slate` and `/api/picks`.
3. **Layer 3 — Full pipeline**: ESPN → injury cascade → LightGBM → Monte Carlo RS → card boost → MILP optimizer. Only runs when both Layer 1 and Layer 2 are empty (true first run of the day).

### Cache Helpers (grep: SLATE CACHE GITHUB)
- `_slate_cache_to_github(slate_data)` — writes today's slate to `data/slate/{date}_slate.json`
- `_slate_cache_from_github()` — reads today's slate; returns `None` if missing or busted
- `_games_cache_to_github(game_proj_map)` — writes per-game projections to `data/slate/{date}_games.json`
- `_games_cache_from_github()` — reads per-game projections; returns `None` if missing or busted
- `_bust_slate_cache()` — clears both `/tmp` and GitHub caches using tombstone pattern (`{"_busted": true}`)

### Cache Invalidation
- **Config change** (`/api/lab/update-config`): calls `_bust_slate_cache()` → next request regenerates with new params
- **Manual refresh** (`/api/refresh`): calls `_bust_slate_cache()` → full cache clear
- **Injury check** (`/api/injury-check`): regenerates only affected games, updates both layers

### Where GitHub Cache is NOT Used
The Line of the Day engine paths (`_run_line_engine_for_date()` and `_get_projections_for_date()`) skip the GitHub cache layer entirely. The extra ~1-2s GitHub API round-trip adds latency without benefit — the line engine still needs to run Haiku calls regardless. These paths go directly from `/tmp` cache to the full pipeline.

### Prediction model boundaries (grep: LINE CONFIG, line_engine config)

- **Draft model:** Config in `data/model-config.json` (card_boost, game_script, real_score, cascade, projection, lineup, moonshot, development_teams); code in `api/index.py`, `api/real_score.py`, `api/asset_optimizer.py`; `lgbm_model.pkl` is trained separately (GitHub Actions). Ben can change draft behavior via Lab (update-config, backtest).
- **Line of the Day model:** `api/line_engine.py` receives projections and games from the draft pipeline plus an optional `line_config` dict passed from `api/index.py` (from the config `line` section). Before running the engine, `api/index.py` enriches projections with **real last-5 game stats** (nba_api) when available (`_enrich_projections_with_l5`), so Claude and the fallback see actual recent form. Line knobs in config: `min_confidence`, `min_edge_pct`, `recent_form_over_ratio`, `recent_form_under_ratio`, `min_edge_pts`, `min_edge_other`. Ben can tune via the `line` section of model-config; no code in line_engine reads GitHub or `_cfg` — config is passed in by the caller to keep the engine self-contained.

## Ben (Lab) Interface

Ben is a **pure chat interface** — no quick-action buttons. The user types naturally and Ben:
- Auto-loads the briefing and config context silently on open (hidden messages)
- Offers to run backtests, apply config changes, analyze accuracy — all via conversation
- Decision history and config changes are stored in `LAB.messages` and `data/model-config.json`
- The chat prompt includes full system context (briefing data, config state, backtest capability)

### End-of-Day Upload Flow (in Ben)
After all games are final and Ben unlocks:
- Upload banner with 5 buttons: **📸 Real Scores**, **🏆 Top Drafts**, **⚡ Starting 5**, **🚀 Moonshot**, **📊 Most Popular**
- Banner **hidden during lock** — only shows when Ben is unlocked
- Banner title shows pending date + live progress counter (e.g. "Log Results — Fri, Mar 6  2/5")
- `pending_upload_date` = most recent prediction date (excluding today) without actuals uploaded
- **Skip All** button in top-right of banner header (meta-action on the date, not a data action)
- `_handleBenUpload()` pipeline: `parse-screenshot → save-actuals → /api/audit/get → lab/briefing → labCallClaude()`
- On successful upload: button flashes green for 1.5s, then **hides** (user only sees remaining uploads)
- `_checkBannerDone()` uses **localStorage** as source of truth (not DOM disabled state, since buttons hide); fires on all 5 upload types including `most_drafted`
- On reload: already-done buttons hidden immediately (no stale green buttons cluttering the banner)
- Audit gate: `save-actuals` only generates `data/audit/{date}.json` when `real_scores` data is present
- Real Scores, Top Drafts, Starting 5, Moonshot merge into actuals CSV (dedup by player name); Most Popular saves to `data/ownership/{date}.csv`
- Log page is **read-only** — no upload UI there

### Lock System
- **Locked** 5 minutes before first game tip-off (slate is in progress)
- **Unlocked** when ALL games on today's slate reach "Final" status on ESPN (60s TTL when locked, 180s pre-slate)
- During lock: shows locked state with **total games remaining** (in-progress + scheduled) and estimated unlock based on **last game of the day** + 2.5h
- During unlock: full chat capabilities + upload banner (if pending date exists)
- **Break between game windows**: if early games finish but more are scheduled, Ben briefly unlocks with "Break — N games later today". Polling detects re-lock when next game window starts.
- **4 failure paths hardened**: (1) ESPN outage — `_all_games_final` requires `finals > 0`, (2) split-window — `any(_is_locked(st))` checks ALL games, (3) ESPN down — GitHub lock file fallback before unlocking, (4) frontend fetch failure — shows **connection error** ("Couldn't check status"), never "BEN IS LOCKED". (5) Backend exception path: when `/api/lab/status` catches an exception it returns 200 with `locked: true` and reason "Server temporarily unavailable — try again"; the frontend treats that as **status unknown** (`_isLabStatusErrorFallback`) and shows "Server couldn't determine status. Tap Retry." instead of "BEN IS LOCKED", so the locked banner only appears when the server actually knows the slate is in progress.

### Lab status, banner, and degraded mode
- **What "COULDN'T CHECK STATUS" indicates:** The request to `/api/lab/status` failed (timeout or network) before any response. Common on cold start when that endpoint is the first to hit the server (fetch_games + _all_games_final can exceed 30s).
- **What the upload banner depends on:** The banner (LOG RESULTS with 5 upload buttons) is shown only inside `showLabUnlocked()`, which runs after a successful `locked: false` from `/api/lab/status`. So if status never succeeds, we never call `showLabUnlocked()` and never run `_initUploadBanner()` — hence the banner doesn't load.
- **What else lab status gates:** Locked vs unlocked view, lock polling (2 min when locked), briefing/config/line/slate/log fetch (only when unlocked), chat input availability. The single status call is the gate for the whole Lab tab.
- **Fixes:** (1) **Backend pre-slate fast path:** When no game has started yet (now &lt; earliest tip − lock buffer), `lab_status` returns "Pre-slate window" without calling `_all_games_final()` (saves ESPN scoreboard calls and reduces timeouts). (2) **Frontend degraded mode:** When status fails (fetch throw or server error fallback), the frontend tries `/api/lab/briefing`; if it succeeds and has `pending_upload_date`, we call `showLabUnlocked()` so the banner and chat appear even when status didn't complete.

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
This prevents the commit → Vercel redeploy cascade that was triggering 6+ redeploys per visit.

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
- Lock cache (`/tmp/nba_locks_v1/`) survives within a warm Vercel instance
- Cache TTL during locked slate: **60 seconds** (balanced for event-driven detection and Vercel cost)
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
- `/api/slate` (line 1777): `any(_is_locked(st))` before computing predictions
- `/api/save-predictions` (line 1920): `any(_is_locked(...))` guard prevents pre-lock saves
- `/api/refresh` (line 2333): `any(_is_locked(...))` gate for auto-save
- `/api/lab/status` (line 3331): `any(_is_locked(st))` determines locked state

Per-game checks (e.g. `/api/picks`, `/api/line-live-stat`) correctly use single-game `_is_locked(game_start)`.

### Triple-Gated Save Pipeline
Predictions are saved to `data/predictions/` through exactly two code paths, both strictly post-lock:
1. **`/api/save-predictions`** — called by frontend `savePredictions()` + `/api/refresh` cron
2. **Inline at lock-promotion** in `/api/slate` — first locked request promotes cache and writes CSV

Three independent gates prevent pre-lock saves:
- **Frontend guard**: `if (!SLATE || !SLATE.locked) return;` in `savePredictions()`
- **Backend guard**: `if not any(_is_locked(st) ...)` → HTTP 409 in `/api/save-predictions`
- **Cron guard**: `/api/refresh` only calls `save_predictions()` if `any(_is_locked(...))`

## Two Lineup Types

- **Starting 5 (chalk)**: MILP-optimized for `chalk_ev = rating × (avg_slot + card_boost) × reliability`. Conservative, consistent.
- **Moonshot** (v2, tuned Mar 11): Options strategy. Hard floor of **15 projected minutes** (lowered from 20 to catch emergency starters) + RotoWire lineup clearance + minimum 2.0 rating. Ranked by `moonshot_ev = (predMin^min_weight) × (card_boost^2.5) × dev_team_bonus × rating × pos_efficiency`. Development/tanking team players get 1.25x boost. **Centers get `pos_efficiency=0.65`** — bigs generate far fewer RS events per minute (screens/rim protection don't accumulate RS) compared to guards/wings. Philosophy: ownership (boost^2.5) is the dominant signal; minutes just confirm the player will be on the court.

### Development Teams (configurable in model-config.json)
Current default: `UTA, IND, BKN, CHI, NOP, SAC, MEM, WAS, DAL` — teams effectively out of playoff contention whose role players get predictable developmental minutes and structurally lower ownership. **This list is a seasonal snapshot** — update via Ben or directly in `data/model-config.json` as the standings shift.

### RotoWire Integration (`api/rotowire.py`)
Free-tier scrape of RotoWire NBA lineups page. Runs ~30 min before first tip. Returns player availability (confirmed/expected/questionable/OUT). Moonshot hard-filters on this: any player flagged OUT or questionable is excluded. Cache TTL: 30 minutes.

## Model Improvements (deployed)

### LightGBM (11 features, `lgbm_model.pkl`)
Features: `avg_min, avg_pts, usage_trend, opp_def_rating, home_away, ast_rate, def_rate, pts_per_min, rest_days, recent_vs_season, games_played`

- Model bundle format: `{"model": lgb.LGBMRegressor, "features": [...]}` — inference verifies feature vector length; bundle required.
- `rest_days` and `games_played` default to `2.0` / `40.0` at inference (not in ESPN splits). `recent_vs_season` = recent scoring vs season average (training: recent_5g_pts/avg_pts; inference: recent_pts/season_pts).
- Retrained nightly by GitHub Actions (`retrain-model.yml`). Retrain manually: `python train_lgbm.py`.

### Card Boost (`_est_card_boost`)
- Default: exponential heuristic `scalar × decay_base^hype + base_offset`.
- Log-formula path (calibrated, off by default): `log_a - log_b × log10(predicted_drafts)`. Activate with `card_boost.log_formula_active: true` in config once 50+ actuals collected.
- Continuous fame multiplier: `fame_mult = max(1.0, (pts / fame_pts_base) ** fame_exponent)` — replaces the old `star_players` hardcoded list. Tunable via `card_boost.fame_pts_base` (default 8.0) and `card_boost.fame_exponent` (default 2.5).

### Moonshot Formula (tuned Mar 11)
`moonshot_ev = (predMin^min_weight) × (card_boost^2.5) × dev_team_bonus × rating × pos_efficiency`

Key knobs in `moonshot` section of model-config.json:
- `min_minutes_floor`: 15 (lowered from 20) — catches emergency starters projected ~15-18 min pre-game
- `card_boost_weight`: 2.5 (raised from 2.0) — ownership is the primary signal; boost^2.5 dominates
- `big_pos_efficiency`: 0.65 — centers penalized; screens/rim-protection generate far fewer RS events per minute than guards/wings

### Spread Adjustment (continuous, no cliff edges)
- Bench/role players (PPG ≤ 12, avg_min ≤ 26): neutral at spread ≤ 4, rises to +15% at large spreads (garbage-time minutes).
- Stars/starters: peak 1.15× at pick'em, continuous decay, floors at 0.70× for heavy favorites.

### Audit Pipeline
- `save-actuals` auto-writes `data/audit/{date}.json` with MAE, directional accuracy, over/under counts, top-8 misses.
- `GET /api/audit/get?date=X` returns pre-computed audit (falls back to live computation).
- `lab_briefing` uses cached audits when available; adds over-projection pattern detection.

## Line Page — Direction Filter & Odds Refresh

The Over/Under inline sub-nav (`#lineSubNav`) and the inline All/Over/Under tabs in Recent Picks both call `filterLineHistory(dir)`. Selecting a direction also controls the **main pick card visibility**:

- `switchLineDir(dir)`: renders the appropriate pick (`LINE_OVER_PICK` or `LINE_UNDER_PICK`) via `renderLinePickCard()`
- **No "yesterday's pick" banner** — resolved picks appear only in Recent Picks history. The main card always shows today's pick for the selected direction.
- Picks loaded from GitHub CSV lack `books_consensus/odds_over/odds_under` — render as `MODEL` label. Picks refreshed via `/api/refresh-line-odds` show actual book odds + count.
- Pick cards display `"Odds · [time] CT"` when `line_updated_at` is present (stamped by `/api/refresh-line-odds`)

### Line of the Day card
The main pick card (`renderLinePickCard`) uses a **zoned layout** and design tokens. Over and Under use the same component; only `dir` and content differ.

- **Zone 1 (Header):** Player name; subheader = matchup + game time (e.g. `CLE vs BOS · 1:00 PM ET`) when `game_time` is present, else `Team · vs Opponent`. Odds/timestamp in the top-right of the card (`position: relative` on the card, flex + `margin-left: auto` on the odds block).
- **Zone 2 (The Play):** One row: bet pill (OVER/UNDER) + target stat line (e.g. `13.0 pts`). Stat label is derived from `pick.stat_type` via a small map (points → PTS, rebounds → REB, assists → AST) — no hardcoded PTS/REB/AST in the UI.
- **Zone 3 (Data row):** A single full-width flex row (`.line-pick-data-row`, `justify-content: space-between`) with **5 columns:** (1) **Baseline** — sportsbook line + stat label; (2) **Edge** — mathematical edge, semantic colors `edge-plus` / `edge-minus`; (3) **Target stat** — stacked projection / season average; (4) **Minutes** — stacked `proj_min` / `avg_min`; (5) **L5** — text array of last-5 stat values (e.g. `12 • 14 • 9 • 16 • 11`) with hit/miss coloring vs baseline. When `recent_form_values` is present (real last-5 from nba_api), L5 uses those; otherwise it falls back to ratio-based values from `recent_form_bars`.
- **Conclusion (Oracle Insight):** Model reasoning is consolidated into a single narrative paragraph at the bottom of the card, not standalone pills. `_buildLineConclusion(pick)` merges `pick.narrative` with `pick.signals` (e.g. injury upgrade, B2B) into one natural-language sentence; key reasons are highlighted with `--color-text-primary`. The paragraph sits in `.line-pick-conclusion-wrap` (subtle background, 8px radius, padding). **This "Narrative Conclusion" pattern is the standard for model explanations app-wide** — use one prose block, not reasoning bubbles.
- **Pick payload fields** (from backend): Core fields plus `season_avg`, `proj_min`, `avg_min`, `game_time`, `recent_form_bars`, `recent_form_values`. `recent_form_bars` is set in `api/line_engine.py`; `recent_form_values` is filled in `api/index.py` via `_get_last5_game_stats()` (nba_api PlayerGameLog, 30-min cache). All passed through `_normalize_line_pick`.
- **Design tokens:** Line card uses `--radius-card`, `--radius-pill`, `--font-size-micro`, `--color-success`, `--color-danger`, `--color-text-primary`, `--color-text-muted`, `--line`, `--lab`; no hardcoded hex for semantic colors in the card block.
- **Cache:** `index.html` is served with `Cache-Control: max-age=0, must-revalidate` (vercel.json) so browsers and edge revalidate and users get the latest card (5-column + conclusion) after deploy. If an user still sees an old card, they should hard refresh (e.g. pull-to-refresh on mobile) or close and reopen the tab.

### Odds Refresh Pipeline
- **Cron**: `55 * * * *` (once per hour at :55; hits common 6:55 PM ET lock)
- **Helpers**: `_abbr_matches(abbr, full_name)` maps ESPN abbrs → Odds API team name fragments; `_fetch_odds_line(player, stat, team, opp)` makes 2-step Odds API call (events list → event player props)
- **Lock freeze**: `/api/refresh-line-odds` checks `_is_locked(earliest)` — no-op if locked
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

Use these tokens instead of hardcoded hex or pixel radii across Predict, Line, Ben, and Log.

## Cron Schedule (vercel.json)

Crons and frontend poll intervals are tuned to reduce Vercel invocations while preserving lock/unlock, odds refresh, and line-resolve behavior.

| Schedule (UTC) | Endpoint | Purpose |
|----------------|----------|---------|
| `0 19 * * *` | `/api/refresh` | Cache clear + auto-save locked predictions |
| `0 9 * * *` | `/api/lab/auto-improve` | Auto-tune model if ≥3% MAE improvement |
| `55 * * * *` | `/api/refresh-line-odds` | Hourly bookmaker odds sync (at :55) |
| `0,30 * * * *` | `/api/auto-resolve-line` | Resolve line picks as each game ends |
| `0 14,16,18,20,22,0 * * *` | `/api/injury-check` | Check RotoWire for injury changes; regenerate affected games only |

## Deployment Pipeline

Vercel `ignoreCommand` in `vercel.json` prevents builds on data-only commits:
```
git diff --quiet HEAD~1 HEAD -- . ':!data' ':!.github'
```
This ensures GitHub API writes to `data/` (predictions, actuals, line picks, config) and `.github/` workflow changes don't trigger unnecessary redeploys. Only code changes trigger builds.

## Production Robustness Notes

All frontend API calls (`fetch(...)`) have `.ok` checks before calling `.json()`. Missing `.ok` checks were a common source of silent failures in prior versions.

Key patterns used throughout:
- Async functions: `if (!r.ok) throw new Error('HTTP ' + r.status)` before `.json()`
- Promise.allSettled chains: `fetch(...).then(r => r.ok ? r.json() : Promise.reject(...))`
- Polling loops: `.then(r => r.ok ? r.json() : Promise.reject())` with empty `.catch`
- `savePredictions`: resets `_predSavedDate` on non-OK responses so the next call can retry

### Resilience (error boundaries)
- **Backend:** A global `@app.exception_handler(Exception)` in `api/index.py` catches any unhandled exception: logs full traceback server-side and returns `JSONResponse({"error": "An unexpected error occurred"}, status_code=500)` with no stack trace or internal detail in the response.
- **Frontend:** All `JSON.parse(localStorage.getItem(...))` usages are wrapped via `_safeParseLocalStorage(key, fallback)` so corrupted or invalid JSON does not throw. Critical DOM access uses `_el(id)` with null checks (or early return) in Line/Lab/Log/Predict and Lab lock poll so missing elements do not throw. Lab lock poll logs status fetch failures with `console.warn('[lab] Lock poll status check failed:', ...)` instead of failing silently. Lab chat uses an `AbortController` with a 60s connection timeout for the initial `/api/lab/chat` request; on timeout the user sees "Request timed out. Please try again."

EOD prompt check uses `LAB.messages.filter(m => !m.hidden).length === 0` — hidden context-loading messages don't suppress the upload prompt.

**Health in deployment:** Use `GET /api/health` for uptime monitoring. Configure an external checker (e.g. UptimeRobot, Cronitor) to hit the URL and alert on non-200; Vercel does not provide built-in health-check alerting.

## Event-Based Slate Transition

The system now uses **game completion events** instead of clock-based timeouts for slate unlocking. When the final game on a slate completes, the system:

1. **Immediately detects completion** — `_all_games_final()` checks ESPN scoreboard and fires the unlock event
2. **Triggers upload prompt** — Ben unlocks and shows the end-of-day upload banner
3. **Enables next slate** — New games become draftable within seconds of previous slate completion

### Cache TTL Optimization (Adaptive)
- **Locked slate**: Cache TTL **60 seconds** (from 180s; balances responsiveness and Vercel cost)
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
- If Lab is active and unlocks, shows upload banner instantly
- Falls back to auto-resolve cron if Ben not open

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

Affected endpoints: slate load, picks, games, save-predictions, screenshot parse, save-actuals, audit, log-dates, log-get, line-of-the-day, refresh-line-odds, lab-status, lab-briefing, lab-chat, lab-config-history, line-history, hindsight.

### Worker Pool Optimization
Backend uses Python `ThreadPoolExecutor` for parallel processing:
- **Standard pool: 8 workers** (game runner, slate processor, picks processor, audit runner, line engine)
- **L5 enrichment pool: 10 workers** (nba_api per-player stat fetches — more I/O-bound)
- Handles 14-game Saturdays efficiently without bottlenecking

### Polling Interval Tuning
- **Lab lock polling**: 2 minutes (reduces Vercel invocations; see `initLabPage` in index.html, ~3135)
  - Unlock detected within ~2 min; user can tap Retry for immediate check
- **Line live stat polling**: 1 minute; max 5 consecutive failures (300s tolerance) before fallback to cron
  - Prevents indefinite polling on persistent network failures
  - Falls back to `/api/auto-resolve-line` cron (0/30 min marks)

### GitHub API Retry Logic
`_github_write_file()` (api/index.py lines 75-110) implements exponential backoff for concurrent write conflicts:
- **Retries up to 3 times** on HTTP 422 (SHA mismatch)
- **Backoff delays**: 1s, 2s, 4s between retries
- **Fresh SHA fetch** on each retry (not cached)
- Protects against concurrent writes from cron + user uploads (rare but possible edge case)
- Used for: predictions, actuals, line picks, config updates

### Cache TTL & Invalidation
Explicit TTLs protect against stale data while minimizing API calls:

| Cache | TTL | Purpose | Invalidation |
|-------|-----|---------|--------------|
| Game final status (`_all_games_final`) | 60s when locked, 180s pre-slate | Detects when ALL games reach Final status | `/api/refresh` endpoint clears |
| Model config (`data/model-config.json`) | 5 min | Runtime tuning parameters | `/api/refresh` clears; Lab writes bypass cache |
| RotoWire lineups | 30 min | Player availability (OUT, questionable, etc.) | 30 min expiration; manual refresh via app |
| Lock status per game | 6 hours | 5 min before tip to 6h after (ceiling) | Natural expiration |
| Line odds (`books_consensus`) | 1 hour | Bookmaker consensus line (refreshed by cron) | Hourly cron runs; slate-lock freeze |
| L5 game stats (`_get_last5_game_stats`) | 30 min | Real last-5 PTS/REB/AST per player (nba_api) | Stored with ts in cache; TTL checked on read |
| Slate cache (`data/slate/`) | 1 day | GitHub-persisted predictions (full slate + per-game) | `_bust_slate_cache()` via refresh, config change, or injury check |

### Midnight Rollover Handling
`auto_resolve_line()` (api/index.py lines 2917-3040) correctly handles games finishing after midnight ET:
- Tracks `pick_date` separately from `_et_date()` (which changes at midnight)
- Uses `pick_date` for both GitHub file lookups and ESPN API queries
- Falls back to yesterday's pick file if today's missing
- Computes next-day picks from `pick_date + 1`, not `_et_date() + 1`
- Prevents loss of line pick data on multi-day slates

### ESPN API Fallback
`_all_games_final()` (api/index.py lines 3188-3258) protects against ESPN outages:
- If game status not updated for 4+ hours, mark as final (assume game completed)
- Safety guard: `if finals == 0 and remaining == 0: all_final = False` — prevents false unlock on ESPN API down
- Requires `finals > 0` before returning true (at least one game must have reached Final status)
- Falls back to GitHub lock file recovery on cold start if ESPN unreachable

## Skip Uploads Feature

Users can skip uploading results for specific slates without affecting model learning. This is useful for:
- Incomplete drafts or testing scenarios
- Days where Real Sports data is unreliable
- Preventing outliers from skewing model retraining

### UI
Ben upload banner includes a "Skip All" button (muted style, right-aligned):
- Only shown when Ben is unlocked (after all games final)
- Clicking hides the banner with fade animation
- Updates to green (✓) after successful skip

### Implementation
- Frontend `_benSkipAllUploads()` (index.html lines 2177-2213):
  - Stores skipped dates in `localStorage` (browser persistence)
  - Calls `/api/lab/skip-uploads` to record server-side
  - Hides banner immediately for better UX
- Backend `/api/lab/skip-uploads` POST endpoint (api/index.py lines 4027-4067):
  - Appends skipped date to `data/skipped-uploads.json`
  - Exponential backoff retry for concurrent writes
  - Returns success status
- `save_actuals()` checks for skipped date before processing screenshot
  - Silently skips processing if date is marked as skipped
  - Allows users to upload later if they change their mind

### Data Format
`data/skipped-uploads.json`:
```json
{
  "skipped_dates": ["2026-03-06", "2026-03-07"],
  "last_updated": "2026-03-08T18:30:00Z"
}
```

## Other Files (Extended Audit)

| File | Role | Notes |
|------|------|------|
| **api/line_engine.py** | Line of the Day engine | Claude Haiku prompts, _STAT_META (points/rebounds/assists), Odds API integration. Called by api/index.py `/api/line-of-the-day`. No direct HTTP; all I/O via index. |
| **api/rotowire.py** | RotoWire lineup scraper | Free-tier scrape for availability (OUT, questionable). 30 min cache. Used by slate/Moonshot filtering. |
| **api/real_score.py** | Monte Carlo Real Score | RS projection (closeness, clutch, momentum). Used by projection pipeline in index. |
| **api/asset_optimizer.py** | MILP lineup optimizer | PuLP/CBC for Starting 5 + Moonshot. Used by game runner in index. |
| **server.py** | Local dev server | Serves index.html at `/` and re-exports FastAPI app for `uvicorn server:app`. Production uses Vercel static + serverless. |
| **scripts/check-env.py** | Env verification | Validates REQUIRED (GITHUB_TOKEN, GITHUB_REPO, ANTHROPIC_API_KEY) and OPTIONAL vars. Run before local dev. |
| **scripts/sync_model_config.py** | Config sync | Syncs model-config from GitHub (used by workflows). |
| **scripts/bump_retrain_config.py** | Retrain config | Bumps retrain config for GitHub Actions. |
| **train_lgbm.py** | Model training | 11-feature LightGBM training; outputs lgbm_model.pkl. Run locally or via retrain-model.yml. |

No orphan entrypoints; all API surface is in api/index.py. Scripts are for local/CI use.

## Audits

- **docs/AUDIT-LIGHTWEIGHT.md** — Production, object/variable/reference, pipeline/caching, LightGBM (includes fix for recent_vs_season train/inference alignment).
- **docs/AUDIT-HEAVY.md** — Security, error handling, API consistency, contracts, timeouts, deployment, observability, tests/docs.

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
TestLineConfig              — run_model_fallback and run_line_engine respect line_config min_confidence
TestLgbmFeatureAlignment    — when bundle loaded, 11 features and 11th is recent_vs_season or recent_3g_trend
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
```

**tests/test_core.py** — Helpers, line cache logic, JS syntax, date-boundary regressions, and contract guards:

- **TestHelpers** — _et_date, _is_locked, _est_card_boost, cache roundtrip
- **TestLineCacheLogic** — when line cache is served vs bypassed (today unresolved / resolved / yesterday)
- **TestJSSyntax** — unescaped apostrophes in single-quoted strings; presence of renderCards, renderLinePickCard, initLinePage, loadSlate, switchTab; _etToday / LINE_LOADED_DATE / _predSavedDate
- **TestCacheDateBoundary** — cache keys consistent with ET date
- **TestBenBannerActualsDetection** — Ben upload banner hides when today's actuals already saved
- **TestBannerGuardJS** — banner-visibility check in showLabUnlocked() uses correct localStorage keys
- **TestNormalizePlayer** — _normalize_player() contract: all required fields present with correct types
- **TestNormalizeLinePick** — _normalize_line_pick() contract: all required fields, result defaults to "pending"
- **TestRealScoreEngine** — Monte Carlo closeness/clutch/momentum coefficients (pure numpy, no I/O)
- **TestAssetOptimizer** — optimize_lineup() MILP slot assignment: chalk vs moonshot modes, edge cases
- **TestConfigCoverage** — all major model floors read from model-config.json via _cfg()
- **TestProjectPlayerContract** — project_player() returns all required fields after _normalize_player()
- **TestLineEngineHelpers** — line_engine.py helpers with no external deps (_abbr_matches, stat meta)
- **TestJSContractGuard** — frontend null guards for _normalize_line_pick fields added in Phase C
- **TestLogGetNormalization** — log_get() builds player cards from CSV rows with correct field mapping
- **TestUpdateConfigValidation** — /api/lab/update-config accepts dot-notation keys, rejects invalid paths
- **TestFrontendAuditFixes** — regression guards for frontend null guards and .ok checks

Run all: `pytest tests/ -v`  
Run fast subset only: `pytest tests/test_fixes.py -v`  
Note: Tests that import `api.index` require dependencies (e.g. numpy, lightgbm). Use `pip install -r requirements.txt` before running.

## Known Limitations

- Rate limiting uses an in-memory store with a lock (thread-safe); it does not persist across Vercel instances, so limits are per-instance.
- `/tmp` is ephemeral on Vercel — caches don't survive cold starts. GitHub-persisted caches (`data/slate/`, `data/locks/`) provide cold-start recovery for both predictions and lock state.
- **Concurrent write conflicts (mitigated)**: `_github_write_file` implements exponential backoff (1s, 2s, 4s retries) to handle HTTP 422 SHA mismatches. The cron + user upload pattern is protected; conflicts are rare.
- Odds API: when over_pick and under_pick are the same player, `/api/refresh-line-odds` fetches once and applies the result to both (deduped).
- `data/locks/` accumulates one JSON per day with no automated cleanup. GitHub directory listings get marginally slower over a long season; manually prune if needed.
- Log tab shows a 60-day date strip (no "Load more"); dates with stored data are highlighted via `/api/log/dates`.
- Fetch timeouts: All frontend calls have hard limits (10s default, 30s screenshot). Exception: `/api/lab/chat` uses a raw streaming fetch (SSE) by design — no timeout on the stream body, only on connection.
- Upload screenshot type validation is client-side trust only — the system cannot verify that a "Real Scores" button upload actually contains a Real Scores screenshot. Wrong uploads produce skewed audit data for that date.

## Troubleshooting

If slate, line, and/or log all fail to load:
1. **Deployed URL** — Use the production URL (e.g. `https://basketball-chi-cyan.vercel.app`); avoid file:// or wrong origin.
2. **Health and version** — Call `GET /api/health` and `GET /api/version`; if they fail, the backend is unreachable or cold-starting.
3. **Vercel function logs** — Look for `[slate] error:`, `[line-of-the-day] error:`, `[games] error:`, `[log/dates] error:` or "Task timed out" to identify the failing path.

## Robustness Fixes (this session)

| Fix | File | Detail |
|-----|------|--------|
| Stale response guard on game switching | `index.html` | `runAnalysis()` captures `gameId` at call time; discards response if selector changed mid-flight |
| `fetchWithTimeout` on `/api/picks` | `index.html` | Was raw `fetch`, now 15s timeout |
| `fetchWithTimeout` on `/api/audit/get` | `index.html` | Was raw `fetch`, now 10s timeout |
| `fetchWithTimeout` on `top_drafts` save | `index.html` | Was raw `fetch`, now 10s timeout |
| Drill-down auto-close on Log tab return | `index.html` | `switchTab('log')` now calls `closeLogDrilldown()` before grid init |
| Upload banner: hide completed buttons | `index.html` | On reload, done buttons hide immediately; on new upload, flash green → hide after 1.5s |
| Upload banner: X/4 progress counter | `index.html` | Title updates live as each upload completes |
| `_checkBannerDone` uses localStorage | `index.html` | DOM disabled state unreliable after hide; localStorage is source of truth |
| Audit gate on `real_scores` | `api/index.py` | `save-actuals` only generates audit JSON when `real_scores` rows present |
| Dead code pruned | `index.html` | Removed empty `_renderBenEodPrompt()` function |
| Skip All button relocated | `index.html` | Moved from button row to title bar (top-right) — semantic meta-action placement |
| Rate-limit thread-safety | `api/index.py` | `_RATE_LIMIT_LOCK` wraps read-modify-write of `_RATE_LIMIT_STORE` so concurrent requests are safe |
| Line config wired from model-config | `api/index.py`, `api/line_engine.py` | `run_line_engine(projections, games, line_config)`; `min_confidence`, `min_edge_pct`, `recent_form_over_ratio`, `recent_form_under_ratio`, `min_edge_pts`, `min_edge_other`; projections enriched with real L5 before engine run |
| Line tab auto-switch to available direction | `index.html` | `switchLineDir` auto-corrects to `under` if `over` has no pick and vice versa, instead of showing "No X pick today" |
| `next_slate_pending` handling for both null | `index.html` | `switchLineDir` shows pending card (not "No X pick today") when both picks are null; prevents false "missing direction" message |
| Predict tab next-day transition | `index.html` | `loadSlate()` busts stale previous-day predictions; Predict tab no longer stuck on finished slate after midnight rollover |
| `next_slate_pending` re-fetch fix | `index.html` | `_renderLineLOTDFromState()` resets `LINE_LOADED_DATE = ''` on `next_slate_pending` so next tab visit re-fetches; prevents stale "Tomorrow's pick coming soon" after picks become available |
| Retry button on pending card | `index.html` | `renderNextSlatePending()` now includes a "Check for picks" button that calls `fetchLineOfTheDay()` directly |
| Line-of-the-day load path: no L5 re-fetch | `api/index.py` | `recent_form_values` (L5) is fetched once at fresh-generation time and stored in the GitHub JSON. Load paths (fast-path today + next-slate) never re-fetch L5 — use whatever is in the file; card falls back to `recent_form_bars`. Eliminates 10-30s cold-start nba_api call on every load. |
| Line tab + Ben timeout bumps | `index.html` | Line tab `/api/line-of-the-day` timeout 60s→90s; Ben context load 10s→30s. Fixes "Couldn't reach the server" on first load and "Line data unavailable" in Ben. |
| `line_history` parallel fetch + 3-min cache | `api/index.py` | CSV + JSON files fetched in parallel via `ThreadPoolExecutor(8)`; 3-min result cache (`line_history_v1`) avoids repeated cold-start GitHub round-trips; cache cleared by `/api/refresh` |
| 3-layer slate cache (generate once per day) | `api/index.py` | `/tmp` → GitHub `data/slate/` → full pipeline. First request generates and persists; all subsequent requests serve from cache. Reduces API calls from N per visit to ~6-8 per day |
| `/api/injury-check` cron endpoint | `api/index.py`, `vercel.json` | Every 2h: bust RotoWire cache, check cached players, regenerate only affected games. Lock-guarded, CRON_SECRET-protected |
| GitHub cache removed from line engine | `api/index.py` | `_games_cache_from_github()` removed from `_run_line_engine_for_date()` and `_get_projections_for_date()` — added latency without benefit for line paths |
| "Generating picks..." message removed | `index.html` | 12s setTimeout that showed misleading "Generating picks..." during Line page load removed; skeleton card provides sufficient loading feedback |

## Production audit

Full audit: [docs/PRODUCTION_AUDIT.md](docs/PRODUCTION_AUDIT.md). Implemented: GitHub error sanitization (no leak to client), `GET /api/health`, `GET /api/version`, cron secret on `/api/refresh`, `/api/auto-resolve-line`, `/api/lab/auto-improve`, and `fetchWithTimeout` for lab/backtest and lab/update-config.

**Lock & routing audit:** [docs/LOCK_AND_ROUTING_AUDIT.md](docs/LOCK_AND_ROUTING_AUDIT.md). Covers all lock usage (slate, picks, save-predictions, lab status, line odds) and Vercel/FastAPI routing. Fixes applied: `/api/lab/status` wrapped in try/except — on any exception returns 200 with `locked: true` and reason "Server temporarily unavailable — try again" so the frontend shows a retry instead of a generic fetch failure; ESPN-down GitHub lock check now uses `lock_content, _ = _github_get_file(...)` and `if lock_content:` (was incorrectly checking the tuple).

## Pre-deploy checklist (production finalization)

- **Env**: Required vars set in Vercel (GITHUB_TOKEN, GITHUB_REPO, ANTHROPIC_API_KEY; optional ODDS_API_KEY, CRON_SECRET, DOCS_SECRET).
- **Tests**: Run `pytest tests/ -v` locally when changing backend or frontend contract; test_core.py catches JS apostrophe crashes.
- **Docs**: CLAUDE.md and README.md reflect current endpoints, crons, and lock/cache behavior.
- **Health**: Use GET `/api/health` for uptime monitoring; alert on non-200.

## Development

```bash
# Local
pip install -r requirements.txt
python scripts/check-env.py   # verify required env vars (fail-fast)
uvicorn server:app --reload

# Deploy — push to your session branch; auto-merge-to-main.yml merges → main → Vercel
# Branch naming convention: claude/<session-id>  (e.g. claude/codebase-analysis-e3rsW)
git push -u origin <your-branch>

# Verify on production
# https://basketball-chi-cyan.vercel.app
```

## Starting a New Claude Code Session

When starting fresh in a new chat, Claude Code automatically reads this file for context.
Provide the following to the new session to orient it quickly:

1. **Branch**: Create a new `claude/<session-id>` branch (e.g. `claude/my-feature-xyz`). Push triggers auto-merge → main → Vercel. **Never push to main directly.**
2. **Stack**: FastAPI backend (`api/index.py`) + single-file vanilla JS frontend (`index.html`)
3. **Tests**: `pytest tests/ -v` (requires `pip install -r requirements.txt`). test_fixes.py covers lock/audit/cache; test_core.py covers helpers, line cache, JS syntax. Deploy still triggers on push; verify on `basketball-chi-cyan.vercel.app`.
4. **Data layer**: All persistent state in GitHub via Contents API (`data/` directory). No database.
5. **Key globals in frontend**: `SLATE`, `PICKS_DATA`, `LOG`, `LAB`, `LINE_DIR`, `LINE_OVER_PICK`, `LINE_UNDER_PICK`, `LINE_LOADED_DATE`
6. **Cache**: 3-layer: `/tmp` (ephemeral) → GitHub `data/slate/` (persistent) → full pipeline. Check `CACHE_DIR` in `api/index.py` for the current tmp path (versioned, e.g. `/tmp/nba_cache_v19/`). `/api/refresh` clears all caches + config. `_bust_slate_cache()` invalidates both layers.
7. **Config**: `data/model-config.json` on GitHub — Ben/Lab writes here, backend reads with 5-min TTL.
