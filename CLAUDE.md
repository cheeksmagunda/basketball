# Basketball — Real Sports Draft Optimizer

## What This Is

A daily NBA draft optimizer for the **Real Sports** app. It projects player Real Scores, estimates card boosts, and builds optimized 5-player lineups using MILP (mixed-integer linear programming). Deployed on **Railway** as a Dockerized Python (FastAPI) backend + single-page HTML frontend.

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
railway.toml          — Railway config (crons, health check, watchPatterns for deploy)
vercel.json            — Legacy (unused in production; Railway replaced Vercel)
server.py              — Local dev server (uvicorn)
```

## UI Structure

4-tab segmented control navigation (Apple glassmorphism pill style): **Predict | Line | Ben | Log**

- **Predict**: Live slate optimizer (Starting 5 + Moonshot) and per-game analysis ("THE LINE UP" — single 5-player format, no card boost). "Slate-Wide | Game" sub-tabs inline at top of tab. Magic 8-ball loading animation.
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
grep: FORCE REGENERATE         — /api/force-regenerate, _force_write_predictions, deploy SHA mismatch, late draft
grep: LOCK HELPERS             — _is_locked, _is_past_lock_window, _et_date
grep: ALL GAMES FINAL          — _all_games_final, ESPN scoreboard poll, midnight rollover, 4.5h fallback
grep: NEXT SLATE DATE          — _find_next_slate_date, multi-day gap, All-Star break
grep: FORCE REGENERATE SYNC    — _force_regenerate_sync, scope=full|remaining
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
| `/api/log/dates` | GET | List dates with stored prediction/actual data |
| `/api/log/get?date=X` | GET | Predictions + actuals for a given date, grouped by scope |
| `/api/log/actuals-stats?date=X` | GET | ESPN box score stats (PTS, REB, AST, STL, BLK, MIN) for all players on a date's completed games |
| `/api/hindsight` | POST | Optimal hindsight lineup from actual RS scores |
| `/api/refresh` | GET | Clear cache + config cache (cron at 7pm UTC; no auth required — non-destructive) |
| `/api/injury-check` | GET | Cron: check RotoWire for newly OUT/questionable players; regenerate affected games only (requires CRON_SECRET when set) |
| `/api/force-regenerate?scope=X` | GET | **Force-regenerate predictions mid-slate.** `scope=full`: all games (dev deploy). `scope=remaining`: only unlocked games (late draft). CRON_SECRET-gated. Updates `data/predictions/` CSV and all cache layers. |

### Line of the Day
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/line-of-the-day` | GET | **Both** Over + Under picks with **per-direction independent rotation** — each direction rotates to the next slate when its game finishes, without waiting for the other; returns `{over_pick, under_pick, pick}` |
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
| `/api/lab/calibrate-boost` | GET | Fit card boost log formula params (log_a, log_b) from real ownership data in `data/ownership/`; requires ≥4 samples |
| `/api/save-ownership` | POST | Save parsed Most Drafted data to `data/ownership/{date}.csv`; used as input for calibrate-boost |

**Admin / optional (not used by main UI):** `POST /api/reset-uploads` — deletes actuals + audit for a date (admin/debug). `POST /api/hindsight` — optimal hindsight lineup from actual RS (Ben-driven or future). `GET /api/version` — build identifier for deploy/monitoring.

## App init and tab data flow

- **Startup:** `loadSlate()` and `initGameSelector()` run in parallel: `GET /api/slate` (10s) and `GET /api/games` (10s). Predict tab is default; slate list and game dropdown populate.
- **Tab load (lazy):** Line, Log, and Lab load on first visit when `switchTab()` is called:
  - **Line:** `initLinePage()` → fire-and-forget `GET /api/auto-resolve-line`, then `GET /api/line-of-the-day`, `GET /api/line-history`; live stat poll and resolve poll when applicable.
  - **Log:** `initLogPage()` → `GET /api/log/dates`, then `buildLogDateStrip()` (60-day strip); selecting a date calls `GET /api/log/get?date=X`; drill-down may call `GET /api/log/actuals-stats?date=X` (cached in `LOG._statsCache`).
  - **Lab:** `initLabPage()` → pre-warm `GET /api/health`, then `GET /api/lab/status`; on unlock, loads briefing + config-history + line-of-the-day + slate + log for context; lock poll every 120s when locked.

## Environment Variables (Railway)

- `GITHUB_TOKEN` — GitHub PAT with repo scope (for CSV + config read/write via Contents API)
- `GITHUB_REPO` — e.g. `cheeksmagunda/basketball`
- `ANTHROPIC_API_KEY` — Claude Haiku (screenshot OCR) + claude-opus-4-6 (Ben/Lab chat)
- `ODDS_API_KEY` — The Odds API for player prop lines (Line of the Day)
- `CRON_SECRET` — (optional) When set, cron-only endpoints (`/api/auto-resolve-line`, `/api/lab/auto-improve`, `/api/injury-check`) require `Authorization: Bearer <CRON_SECRET>`. Railway injects this via the cron commands in railway.toml. `/api/refresh` is intentionally unprotected (non-destructive, user-facing).
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
1. **Layer 1 — `/tmp` (Railway container)**: In-memory file cache. Fastest, but cleared on container restart (deploy or crash restart).
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
- **Line of the Day model:** `api/line_engine.py` receives projections and games from the draft pipeline plus an optional `line_config` dict passed from `api/index.py` (from the config `line` section). Before running the engine, `api/index.py` enriches projections with **real last-5 game stats** (nba_api) when available (`_enrich_projections_with_l5`), so Claude and the fallback see actual recent form. Line knobs in config: `min_confidence`, `min_edge_pct`, `recent_form_over_ratio`, `recent_form_under_ratio`, `min_edge_pts`, `min_edge_other`, `min_edge_other_over` (over-specific non-points edge floor; defaults to `min_edge_other`), `stat_floors`. Ben can tune via the `line` section of model-config; no code in line_engine reads GitHub or `_cfg` — config is passed in by the caller to keep the engine self-contained.

## Ben (Lab) Interface

Ben is a **pure chat interface** — no quick-action buttons. The user types naturally and Ben:
- Auto-loads the briefing and config context silently on open (hidden messages)
- Offers to run backtests, apply config changes, analyze accuracy — all via conversation
- Decision history and config changes are stored in `LAB.messages` and `data/model-config.json`
- The chat prompt includes full system context (briefing data, config state, backtest capability)

### End-of-Day Upload Flow (in Ben)
After a slate has a pending upload date (regardless of current slate lock state):
- Upload banner with 5 buttons: **📸 Real Scores**, **🏆 Top Drafts**, **⚡ Starting 5**, **🚀 Moonshot**, **📊 Most Popular**
- Banner shows whenever there is a `pending_upload_date`; it is no longer gated on Lab lock state
- Banner title shows pending date + live progress counter (e.g. "Log Results — Fri, Mar 6  2/5")
- `pending_upload_date` = most recent prediction date (excluding today) without actuals uploaded
- **Skip All** button in top-right of banner header (meta-action on the date, not a data action)
- `_handleBenUpload()` pipeline (per-button): `parse-screenshot → save-actuals/save-ownership → localStorage flags → (optional) audit/briefing + Ben analysis`
- On successful upload: button flashes green for 1.5s, then **hides** (user only sees remaining uploads)
- `_checkBannerDone()` uses **localStorage** as source of truth (not DOM disabled state, since buttons hide); fires on all 5 upload types including `most_drafted`
- On reload: already-done buttons hidden immediately (no stale green buttons cluttering the banner)
- Audit gate: `save-actuals` only generates `data/audit/{date}.json` when `real_scores` data is present
- Real Scores, Top Drafts, Starting 5, Moonshot merge into actuals CSV (dedup by player name); Most Popular saves to `data/ownership/{date}.csv`
- Heavy Ben analysis (audit summary + refreshed briefing + `labCallClaude`) only runs once **all 5** uploads are complete for a date; partial uploads are persisted but do not trigger full analysis.
- Log page is **read-only** — no upload UI there

### Lab / Ben availability
- Ben (Lab) chat is always available from the frontend’s perspective — the Lab tab no longer shows a locked vs unlocked state.
- The upload banner depends only on data (`pending_upload_date` + localStorage flags), not on `/api/lab/status` or slate lock state.
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
- `/api/refresh`: `any(_is_locked(...))` gate for auto-save
- `/api/lab/status`: `any(_is_locked(st))` determines locked state

Per-game checks (e.g. `/api/picks`, `/api/line-live-stat`) correctly use single-game `_is_locked(game_start)`.

### Triple-Gated Save Pipeline
Predictions are saved to `data/predictions/` through exactly two code paths, both strictly post-lock:
1. **`/api/save-predictions`** — called by frontend `savePredictions()` + `/api/refresh` cron
2. **Inline at lock-promotion** in `/api/slate` — first locked request promotes cache and writes CSV

Three independent gates prevent pre-lock saves:
- **Frontend guard**: `if (!SLATE || !SLATE.locked) return;` in `savePredictions()`
- **Backend guard**: `if not any(_is_locked(st) ...)` → HTTP 409 in `/api/save-predictions`
- **Cron guard**: `/api/refresh` only calls `save_predictions()` if `any(_is_locked(...))`

## Scoring Upside Standards (v5)

The model enforces scoring floors at two levels to prevent clean-statline
role players and low-minute centers from dominating both lineup types.

### Projection-level gates (project_player)

- **Minimum 7 projected pts** — any player under this is excluded from all pools
- **Minimum 0.35 pts/min** — prevents low-minute players from passing on raw stats alone
- **Scoring bias multiplier** — players whose pts drive their base score (scorers)
  receive a mild upside boost (up to 1.15×) over balanced accumulators

### Chalk-specific gates

- **Minimum 3.5 rating** before card boost is applied — boost cannot rescue a weak base
- **Boost cap at 2.5** (configurable via `projection.chalk_boost_cap`) — allows Gobert/Sheppard-tier players
  (solid RS + meaningful boost) to rank above pure high-rating/no-boost picks

### Moonshot-specific gates (in addition to projection-level)

- **Minimum 3.0 rating** (raised from 2.0; wildcards exempt)
- **Minimum 10 projected pts** (non-wildcard only) — moonshot needs scoring upside, not just boost

### Per-game lineup gates

- **Minimum 10 projected pts** — single-game format has no card boost, so a low-scoring
  player projecting 8 pts is a ceiling liability, not a value play

### RS Distribution (asymmetric compression, v5)

The projection pipeline uses asymmetric compression to preserve floor accuracy
while widening the ceiling for high-upside players:

- **compression_divisor**: 5.5 (was 7.0) — less pre-compression dampening
- **compression_power**: 0.72 (was 0.62) — softer power law, lets stars separate from role players
- **rs_cap**: 20.0 (was 15.0, applied 4× in pipeline) — removes artificial ceiling that bunched everyone
- **AI blend**: 50/50 (was 70/30 AI-heavy) — LightGBM clusters outputs in 2.5-4.5; at 70% weight it
  killed the heuristic's wider distribution. 50/50 lets Monte Carlo upside shine through.

**Result**: RS distribution widens from ~3-4.5 to ~2-8. Stars can project RS 6-8,
role players stay at RS 2-4, and the gap between them drives better lineup selection.

### Design rationale

A player like Goga Bitadze (5.7 pts, +2.5x card) has a theoretical EV of
`5.7 × (slot + 2.5)` but in practice cannot win a lineup. The boost amplifies
a weak base — a 50% shooting night on 5 attempts still yields 7-8 pts.
Contrast with a player projecting 15 pts with +1.5x — a hot night goes to 20+.
The scoring floors enforce that the base must be real before boost matters.

### What gets displaced

Low-scoring centers, sub-20-minute role players, and specialists
whose RS score came primarily from rebounds/blocks/steals rather than scoring.
These are fine in certain roster construction contexts but should not be
auto-selected as Starting 5 or Moonshot anchors.

### Tunable via Ben

All floors live in `scoring_thresholds` in `data/model-config.json`:
`min_pts_projection`, `min_pts_per_minute`, `min_chalk_rating`, `min_moonshot_rating`,
`min_moonshot_pts`, `scoring_bias_base`, `scoring_bias_pts_weight`.

---

## Two Draft Strategies (Shared Player Pool)

Starting 5 and Moonshot represent two independent draft strategies — reliability vs ceiling — but **they can share players**. If a player is the best pick for both strategies, they appear in both lineups. Together they predict the two best possible drafts of the day. Target: **70+ total draft score** for both.

### Slate-Wide: Starting 5 (chalk)
MILP-optimized for `chalk_ev = rating × (avg_slot + card_boost) × reliability`. Conservative, consistent. **Requires 25-minute season avg minutes floor** (`season_min >= 25`). Configurable via `projection.chalk_season_min_floor`.

### Slate-Wide: Moonshot
Options strategy. Hard floor of **season avg minutes >= min_minutes_floor** (default 20) + RotoWire lineup clearance + minimum 2.0 rating. Ranked by `moonshot_ev = (predMin^min_weight) × (card_boost^2.5) × dev_team_bonus × rating × pos_efficiency`. Development/tanking team players get 1.25x boost. **Centers get `pos_efficiency=0.70`** — bigs generate fewer RS events per minute (screens/rim protection don't accumulate RS) compared to guards/wings. Philosophy: ownership (boost^2.5) is the dominant signal; minutes just confirm the player will be on the court.

### Per-Game: THE LINE UP
Single 5-player format for single-game drafts. **No Starting 5 / Moonshot split** — both users draft from the same 2-team pool, making card boost irrelevant. Optimized purely by projected Real Score × slot multiplier (`est_mult` zeroed out for MILP). **Requires 20-minute recent minutes floor.** Min 2 players per team enforced. Game script adjustments applied.

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
- `min_minutes_floor`: 20 (season avg) — minimum season avg floor; combined with `min_recent_minutes_floor: 25` for an AND gate
- `card_boost_weight`: 2.5 (raised from 2.0) — ownership is the primary signal; boost^2.5 dominates
- `big_pos_efficiency`: 0.70 — centers penalized; screens/rim-protection generate fewer RS events per minute than guards/wings

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
- **Per-direction independent rotation** — over and under rotate independently. When one direction's game finishes, that direction immediately shows the next-slate pick; the other direction stays live. The main card always shows the currently active (unresolved) pick for each direction.
- Resolved picks appear only in Recent Picks history.
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
- **Cache:** `index.html` is served with `Cache-Control: max-age=0, must-revalidate` (railway.toml static headers) so browsers and edge revalidate and users get the latest card (5-column + conclusion) after deploy. If an user still sees an old card, they should hard refresh (e.g. pull-to-refresh on mobile) or close and reopen the tab.

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

## Cron Schedule (railway.toml)

Crons and frontend poll intervals are tuned to minimize Railway compute and ESPN API usage while preserving lock/unlock, odds refresh, and line-resolve behavior.

| Schedule (UTC) | Endpoint | Purpose |
|----------------|----------|---------|
| `0 19 * * *` | `/api/refresh` | Cache clear + auto-save locked predictions |
| `0 9 * * *` | `/api/lab/auto-improve` | Auto-tune model if ≥3% MAE improvement |
| `55 * * * *` | `/api/refresh-line-odds` | Hourly bookmaker odds sync (at :55) |
| `0 * * * *` | `/api/auto-resolve-line` | Resolve line picks as each game ends (hourly at :00) |
| `0 14,16,18,20,22,0 * * *` | `/api/injury-check` | Check RotoWire for injury changes; regenerate affected games only |

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
- **Frontend:** All `JSON.parse(localStorage.getItem(...))` usages are wrapped via `_safeParseLocalStorage(key, fallback)` so corrupted or invalid JSON does not throw. Critical DOM access uses `_el(id)` with null checks (or early return) in Line/Lab/Log/Predict and Lab lock poll so missing elements do not throw. Lab lock poll logs status fetch failures with `console.warn('[lab] Lock poll status check failed:', ...)` instead of failing silently. Lab chat uses an `AbortController` with a 60s connection timeout for the initial `/api/lab/chat` request; on timeout the user sees "Request timed out. Please try again."

EOD prompt check uses `LAB.messages.filter(m => !m.hidden).length === 0` — hidden context-loading messages don't suppress the upload prompt.

**Health in deployment:** Use `GET /api/health` for uptime monitoring. Railway uses `healthcheckPath = "/api/health"` from railway.toml and shows health in the dashboard. Configure an external checker (e.g. UptimeRobot, Cronitor) for alerting on non-200.

## Event-Based Slate Transition

The system now uses **game completion events** instead of clock-based timeouts for slate unlocking. When the final game on a slate completes, the system:

1. **Immediately detects completion** — `_all_games_final()` checks ESPN scoreboard and fires the unlock event
2. **Triggers upload prompt** — Ben unlocks and shows the end-of-day upload banner
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
- **Lab lock polling**: 2 minutes (reduces API call frequency; see `initLabPage` in index.html, ~3135)
  - Unlock detected within ~2 min; user can tap Retry for immediate check
- **Line live stat polling**: 1 minute; max 5 consecutive failures (300s tolerance) before fallback to cron
  - Prevents indefinite polling on persistent network failures
  - Falls back to `/api/auto-resolve-line` cron (hourly at :00)

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
- Backend `/api/lab/skip-uploads` POST endpoint:
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
| **server.py** | Local dev server | Serves index.html at `/` and re-exports FastAPI app for `uvicorn server:app`. Production runs as a persistent Docker container on Railway. |
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
TestLineConfig              — run_model_fallback and run_line_engine respect line_config min_confidence; min_edge_other_over asymmetry (over blocked, under passes with same edge)
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

- Rate limiting uses an in-memory store with a lock (thread-safe); it does not persist across container restarts, so limits reset on redeploy.
- `/tmp` is cleared on container restart (redeploy or crash) — caches don't survive restarts. GitHub-persisted caches (`data/slate/`, `data/locks/`) provide cold-start recovery for both predictions and lock state.
- **Concurrent write conflicts (mitigated)**: `_github_write_file` implements exponential backoff (1s, 2s, 4s retries) to handle HTTP 422 SHA mismatches. The cron + user upload pattern is protected; conflicts are rare.
- Odds API: when over_pick and under_pick are the same player, `/api/refresh-line-odds` fetches once and applies the result to both (deduped).
- `data/locks/` accumulates one JSON per day with no automated cleanup. GitHub directory listings get marginally slower over a long season; manually prune if needed.
- Log tab shows a 60-day date strip (no "Load more"); dates with stored data are highlighted via `/api/log/dates`.
- Fetch timeouts: All frontend calls have hard limits (10s default, 30s screenshot). Exception: `/api/lab/chat` uses a raw streaming fetch (SSE) by design — no timeout on the stream body, only on connection.
- Upload screenshot type validation is client-side trust only — the system cannot verify that a "Real Scores" button upload actually contains a Real Scores screenshot. Wrong uploads produce skewed audit data for that date.

## Troubleshooting

If slate, line, and/or log all fail to load:
1. **Deployed URL** — Use the production URL (`https://the-oracle.up.railway.app`); avoid file:// or wrong origin.
2. **Health and version** — Call `GET /api/health` and `GET /api/version`; if they fail, the backend is unreachable or cold-starting.
3. **Railway logs** — Look for `[slate] error:`, `[line-of-the-day] error:`, `[games] error:`, `[log/dates] error:` or "Task timed out" to identify the failing path.

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
| `_CONFIG_DEFAULTS` sync | `api/index.py` | Fallback defaults match `data/model-config.json`: `compression_divisor` 5.5, `compression_power` 0.72, `rs_cap` 20.0, `ai_blend_weight` 0.5, `per_player_cap_minutes` 2.0, `big_market_teams` inline fallback removes MIL/DAL/PHX. Prevents silent model behavior change on GitHub outage. |
| `auto_improve_threshold_pct` externalized | `api/index.py`, `data/model-config.json` | `IMPROVEMENT_THRESHOLD` reads from `_cfg("lab.auto_improve_threshold_pct", 3.0)`. Tunable via Ben without code deploy. |
| Line engine stat floors externalized | `api/line_engine.py`, `data/model-config.json` | `_STAT_META` and `stat_configs` min_season floors now read from `line_config.get("stat_floors", {})`. Tunable via `line.stat_floors` in model-config. No behavior change — defaults match prior hardcoded values. |
| Cron schedule restored | `railway.toml` | `/api/refresh-line-odds` cron fixed from `0 */3 * * *` (every 3h) to `55 * * * *` (hourly at :55). Matches documentation and intended behavior. |
| `line_history` parallel fetch + 3-min cache | `api/index.py` | CSV + JSON files fetched in parallel via `ThreadPoolExecutor(8)`; 3-min result cache (`line_history_v1`) avoids repeated cold-start GitHub round-trips; cache cleared by `/api/refresh` |
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
| Log tab ESPN stats auto-fetch | `index.html` | Removed `has_actuals` gate — ESPN box score stats now auto-fetch for any past date without requiring RS screenshot upload. "Waiting for results" pill only shows when ESPN stats also unavailable. New partial-graded card state: shows actual game stats (PTS/REB/AST/STL/BLK) without hit/miss coloring until RS is uploaded. |
| Per-game card boost pill removed | `api/index.py` | `_build_game_lineups` now zeroes `est_mult` in returned player data (not just MILP input) so the `+X.Xx card` pill never renders on per-game (THE LINE UP) cards where card boost is irrelevant. |
| Over model tightening | `api/line_engine.py`, `api/index.py`, `data/model-config.json` | Four over-specific changes (under model untouched): (1) `stat_floors.rebounds` 2.0→5.5 — only legit rotation bigs qualify for rebounds picks. (2) `min_edge_other_over: 2.5` (new config key) — over picks for rebounds/assists need a 2.5+ edge; under picks keep 1.5. (3) `recent_form_over_ratio` 1.08→1.15 — require 15% recent spike to unlock +12 confidence bonus. (4) Claude AVOID clause updated — rebounds/assists overs require a catalyst (cascade, opp-B2B, or 230+ total). Tests added for `min_edge_other_over` asymmetry. |

## Production audit

Full audit: [docs/PRODUCTION_AUDIT.md](docs/PRODUCTION_AUDIT.md). Implemented: GitHub error sanitization (no leak to client), `GET /api/health`, `GET /api/version`, cron secret on `/api/refresh`, `/api/auto-resolve-line`, `/api/lab/auto-improve`, and `fetchWithTimeout` for lab/backtest and lab/update-config.

**Lock & routing audit:** [docs/LOCK_AND_ROUTING_AUDIT.md](docs/LOCK_AND_ROUTING_AUDIT.md). Covers all lock usage (slate, picks, save-predictions, lab status, line odds) and Railway/FastAPI routing. Fixes applied: `/api/lab/status` wrapped in try/except — on any exception returns 200 with `locked: true` and reason "Server temporarily unavailable — try again" so the frontend shows a retry instead of a generic fetch failure; ESPN-down GitHub lock check now uses `lock_content, _ = _github_get_file(...)` and `if lock_content:` (was incorrectly checking the tuple).

## Pre-deploy checklist (production finalization)

- **Env**: Required vars set in Railway (GITHUB_TOKEN, GITHUB_REPO, ANTHROPIC_API_KEY; optional ODDS_API_KEY, CRON_SECRET, DOCS_SECRET).
- **Tests**: Run `pytest tests/ -v` locally when changing backend or frontend contract; test_core.py catches JS apostrophe crashes.
- **Docs**: CLAUDE.md and README.md reflect current endpoints, crons, and lock/cache behavior.
- **Health**: Use GET `/api/health` for uptime monitoring; alert on non-200.

## Development

```bash
# Local
pip install -r requirements.txt
python scripts/check-env.py   # verify required env vars (fail-fast)
uvicorn server:app --reload

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
2. **Stack**: FastAPI backend (`api/index.py`) + single-file vanilla JS frontend (`index.html`)
3. **Tests**: `pytest tests/ -v` (requires `pip install -r requirements.txt`). test_fixes.py covers lock/audit/cache; test_core.py covers helpers, line cache, JS syntax. Deploy still triggers on push to main; verify on `the-oracle.up.railway.app`.
4. **Data layer**: All persistent state in GitHub via Contents API (`data/` directory). No database.
5. **Key globals in frontend**: `SLATE`, `PICKS_DATA`, `LOG`, `LAB`, `LINE_DIR`, `LINE_OVER_PICK`, `LINE_UNDER_PICK`, `LINE_LOADED_DATE`
6. **Cache**: 3-layer: `/tmp` (ephemeral) → GitHub `data/slate/` (persistent) → full pipeline. Check `CACHE_DIR` in `api/index.py` for the current tmp path (versioned, e.g. `/tmp/nba_cache_v19/`). `/api/refresh` clears all caches + config. `_bust_slate_cache()` invalidates both layers.
7. **Config**: `data/model-config.json` on GitHub — Ben/Lab writes here, backend reads with 5-min TTL.
