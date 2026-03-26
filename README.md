# The Oracle — NBA Draft Optimizer for Real Sports

**Document Status:** Current Reference

AI-powered daily NBA draft optimizer for the **Real Sports App**. Uses a **Dual-Engine Architecture**: DFS drafts are powered by a Monte Carlo `real_score` simulator (variance, clutch-factor, ceiling), while prop betting surfaces (Line of the Day, Parlay) are powered by a deterministic `fair_value` engine (rolling medians, Z-score probabilities, floor stability). Card boosts + MILP lineup optimization on the draft side. Deployed on **Railway** as a Dockerized Python (FastAPI) backend + single-page HTML frontend.

## What Real Sports Scores

| Factor | Traditional DFS | Real Sports App |
|--------|----------------|-----------------|
| Scoring | Static: PTS/REB/AST formula | **Real Score**: context-dependent, non-linear |
| Game Closeness | Irrelevant | **#1 factor** — tight games exponentially boost scores |
| Clutch Factor | Not modeled | 4th quarter, lead-changing plays get massive boosts |
| Optimal Strategy | Low-ownership stars | **High-RS role players with big card boosts** |

Total Value = Real Score × (Slot Multiplier + Card Boost). Slot multipliers: 2.0x, 1.8x, 1.6x, 1.4x, 1.2x.

## Architecture

```
index.html             — 5-tab frontend (Predict | Line | Parlay | Ben | Log, vanilla JS)
api/index.py           — FastAPI backend (all endpoints, projection engine, Lab/Line/Parlay)
api/fair_value.py      — Deterministic fair-value engine for prop betting (Line + Parlay only)
api/odds_math.py       — Shared American-odds implied-prob helpers
api/real_score.py      — Monte Carlo Real Score projection engine
api/asset_optimizer.py — MILP lineup optimizer (PuLP/CBC)
api/line_engine.py     — Line of the Day (Claude Haiku + Odds API)
api/parlay_engine.py   — Safest 3-leg parlay (Z-score + correlation scoring)
api/rotowire.py        — RotoWire lineup scraper (availability/injury flags)
lgbm_model.pkl         — LightGBM model bundle {model, features}
train_lgbm.py          — Training script (12 features)
data/model-config.json — Runtime model config (Ben/Lab writes here; 5-min cache)
data/predictions/      — Git-tracked daily prediction CSVs
data/top_performers.csv — **Main historical dataset** (leaderboard outcomes by date); primary source for Log + audit
data/actuals/          — Legacy per-day CSVs (same schema as rollup); optional fallback when a date is missing from the mega file
data/most_popular/     — Per-date most-drafted / popularity CSVs (developer ingest; `save-ownership` writes here)
data/most_drafted_3x/  — High-boost popular sub-list per date (optional)
data/winning_drafts/   — Long-format top-4 winner lineups per date (optional)
data/slate_results/    — Per-date JSON: slate game_count + final scores (manual; future model features)
data/audit/            — Git-tracked daily audit JSONs
data/lines/            — Git-tracked daily Line of the Day picks
data/parlays/          — Git-tracked daily parlay JSON (ticket + lazy resolution)
data/slate/            — GitHub-persisted prediction cache (current slate + games + bust marker)
data/locks/            — Cold-start lock recovery: {date}_slate.json at lock time (active slate only)
data/boosts/           — Pre-game boost uploads (Layer 0 ground-truth boosts, {date}.json)
data/skipped-uploads.json — Optional list of dates where save-actuals no-ops (`POST /api/lab/skip-uploads`)
railway.toml           — Railway config (crons, health check, watchPatterns)
vercel.json            — Legacy (unused in production; Railway replaced Vercel)
server.py              — Local dev server (uvicorn)
```

## Tabs

| Tab | Purpose |
|-----|---------|
| **Predict** | Live slate optimizer. Starting 5 (chalk) + Moonshot lineups. Sub-tabs: Slate-Wide / Game |
| **Line** | Line of the Day — best player prop edge. Over/Under sub-tabs. Odds refresh on a game-window cron. Resolved picks also in Recent Picks history |
| **Parlay** | Safest 3-leg player prop parlay (certainty-focused). Ticket + Recent Parlays history with modal resolution |
| **Ben** | Chat interface (Claude Opus). Always available. Historical screenshot ingestion is **developer-only** this season (see [docs/HISTORICAL_DATA.md](docs/HISTORICAL_DATA.md)) |
| **Log** | Historical drill-down — graded cards (actual RS from `top_performers` / legacy actuals + ESPN box scores vs projections) |

## Dual-Engine Architecture

The backend uses two mathematically distinct projection engines, each optimized for its use case:

### Engine 1: Monte Carlo `real_score` (DFS Drafts — Starting 5 + Moonshot)
```
ESPN API (games, rosters, injuries, spreads)
  → LightGBM 12-feature points projection
  → Monte Carlo Real Score simulator (closeness, clutch-factor, momentum coefficients)
  → Card Boost resolution (`drafts_model.pkl` popularity proxy → `boost_model.pkl` → sigmoid fallback)
  → MILP slot optimizer → Starting 5 + Moonshot lineups
```
**Why Monte Carlo?** DFS drafts reward high ceilings and variance. RS scoring is non-linear — tight games exponentially boost scores via closeness and clutch factors. Monte Carlo simulation captures these fat-tail distributions that deterministic medians miss.

### Engine 2: Deterministic `fair_value` (Prop Betting — Line + Parlay)
```
ESPN gamelogs (L10/L15 rolling windows per stat)
  → DvP opponent-quality adjustment (binary Guards vs Forwards mapping)
  → Game script weights (pace-tier stat multipliers)
  → Spread adjustment + momentum trend (L3 vs L10)
  → Per-stat fair value, Z-score hit probabilities, EV classification
  → Edge maps fed into Line engine (fv_boost on confidence) and Parlay engine (_fv_hit_probs override)
```
**Why deterministic?** Prop betting rewards median accuracy and floor stability. A points over/under bet cares about the most likely stat line, not the ceiling. Rolling window medians with Normal CDF hit probabilities produce tighter Z-scores for leg selection.

### Isolation Boundary
- `/api/slate`, `/api/picks`, `/api/force-regenerate`, `/api/injury-check` → **Engine 1 only** (Monte Carlo)
- `/api/line-of-the-day`, `/api/parlay` → **Engine 2** enriches Engine 1 projections (fair value edge maps + hit probs)
- `_compute_betting_fair_value()` is the sole entry point for Engine 2; it is never called from DFS draft paths

## 3-Layer Prediction Cache

Predictions are generated **once per day** and cached across three layers:

1. **`/tmp`** (Railway container) — fastest, ephemeral on cold start/redeploy
2. **GitHub `data/slate/`** — persistent `{date}_slate.json` + `{date}_games.json`, survives cold starts
3. **Full pipeline** — ESPN + LightGBM + Monte Carlo + MILP; only runs on true first request of the day

Two-pass usage: `/api/slate` is Pass 1 (morning baseline), and Pass 2 reruns are conditional via `/api/slate-check` + `/api/force-regenerate` only when material triggers are detected.

Subsequent visits serve from cache. Injury-triggered regeneration (`/api/injury-check` cron) only re-runs affected games. Config changes (`/api/lab/update-config`) bust the cache so the next request regenerates with new params.

## Two Lineup Types

When **core pool** is enabled (`core_pool.enabled` in model-config), both lineups are built from a single shared core (size from `core_pool.size`, often ~15; union of eligible players, ranked by `core_pool.metric`). **Starting 5** = best 5-of-core for reliability; **Moonshot** = best 5-of-core for ceiling. High overlap is intended. When disabled, legacy behavior: separate pools, each MILP from its own pool.

**Starting 5 (chalk)** — MILP-optimized for `chalk_ev = rating × (avg_slot + card_boost) × reliability`. Minutes floors are **configurable** (`projection.chalk_season_min_floor`, `chalk_recent_min_floor` in `data/model-config.json`). **Star anchor**: a star (e.g. season_pts ≥ `star_anchor_ppg`) can bypass the boost floor — `chalk_max_stars` limits exposure. Consistent, conservative.
Late-season (`team_motivation.start_date` onward), Starting 5 applies a soft team-incentive reliability tilt so playoff/play-in teams are favored and low-incentive teams are discounted (configurable A/B/C multipliers; no hard bans).

**Moonshot** — Contrarian EV strategy. Multiple eligibility pathways (regular, spot-starter, wildcard, role-spike, star/scoring anchors — see `moonshot` and `scoring_thresholds` in model-config). Ranked by `moonshot_ev` using `boost_leverage` from `est_mult` raised to **`moonshot.boost_leverage_power`** (configurable; not a fixed 1.6). MILP constraints like `max_low_boost` limit star exposure. Dev team bonus from live ESPN standings when enabled.

## Line of the Day

Daily Over + Under player prop picks from `api/line_engine.py` (Claude Haiku when the API key is set, plus an algorithmic fallback). Best over and best under are stored separately with per-direction rotation. The pick card uses a zoned layout: header (matchup, game time), play row, 5-column data row (Baseline, Edge, Target stat, Minutes, L5), and a **Conclusion** box at the bottom. Line behavior is tuned via the `line` section of `data/model-config.json` (`min_confidence`, `min_edge_pct`, etc.).

**Bookmaker odds sync**: Railway runs `/api/refresh-line-odds` on a **game-window schedule** (see [`railway.toml`](railway.toml)), not 24×/day. The endpoint uses a bulk Odds API map + per-pick lookup. Odds freeze once the slate is locked (5 min before first tip). Pick cards show "Odds · [time] CT" when `line_updated_at` is set.

## Ben (Lab) Interface

Plain chat powered by `claude-opus-4-6`. Context is auto-loaded in the background (briefing, config, slate, line, log, parlay when available). **Historical leaderboard CSVs are not uploaded from the app** this season — use [docs/HISTORICAL_DATA.md](docs/HISTORICAL_DATA.md) (curl/scripts). `POST /api/hindsight` remains available for optimal hindsight lineups when you have actual RS inputs.

**Config updates**: Ben can propose model parameter changes, run backtests, and apply changes to `data/model-config.json` via the GitHub Contents API — no redeploy needed.

## Key Endpoints

### Core
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check for monitoring (config + GitHub) |
| `/api/version` | GET | Build identifier (RAILWAY_GIT_COMMIT_SHA) |
| `/api/slate` | GET | Full-slate predictions (Starting 5 + Moonshot) |
| `/api/picks?gameId=X` | GET | Per-game predictions |
| `/api/games` | GET | Today's games with lock status |
| `/api/save-predictions` | POST | Save cached predictions to GitHub CSV (server-side lock guard — rejects pre-lock) |
| `/api/parse-screenshot` | POST | Claude Haiku vision OCR — `actuals` (default), `most_drafted` / `most_popular`, `most_drafted_high_boost`, `top_performers`, `winning_drafts` (see [HISTORICAL_DATA.md](docs/HISTORICAL_DATA.md); `boosts` rejected) |
| `/api/save-boosts` | POST | Persist pre-game boosts to `data/boosts/{date}.json` (Layer 0) and bust slate cache |
| `/api/save-actuals` | POST | Save parsed actuals to GitHub CSV + auto-generate audit JSON |
| `/api/audit/get?date=X` | GET | Pre-computed accuracy audit (MAE, directional acc, top misses) |
| `/api/log/dates` | GET | List dates with stored prediction/actual data |
| `/api/log/get?date=X` | GET | Predictions + actuals for a given date, grouped by scope |
| `/api/log/actuals-stats?date=X` | GET | ESPN box score stats (PTS, REB, AST, STL, BLK, MIN) per player for completed games |
| `/api/hindsight` | POST | Optimal hindsight lineup from actual RS scores |
| `/api/refresh` | GET | Clear `/tmp` caches + config cache + bust GitHub slate cache; optional auto-save if locked. **No auth required** (non-destructive; used by daily cron). |
| `/api/injury-check` | GET | Cron: check RotoWire for newly OUT/questionable players; regenerate affected games only |
| `/api/force-regenerate?scope=full\|remaining` | GET | Force-regenerate predictions mid-slate. `scope=full`: all games (deploy/model refresh; CRON_SECRET-gated). `scope=remaining`: unlocked games only (Late Draft banner; user-facing). |
| `/api/slate-check` | GET | Pass 2 trigger monitor: injury changes, watchlist activation, or Vegas total movement; returns `{changed, triggers, recommendation}` |

### Line of the Day
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/line-of-the-day` | GET | Both over + under picks (Claude Haiku + fallback) |
| `/api/refresh-line-odds` | GET | Bulk Odds API map + lookup; updates book lines on stored picks (game-window cron) |
| `/api/line-live-stat` | GET | Fetch live stat value for in-game pick tracking |
| `/api/save-line` | POST | Persist `{over_pick, under_pick}` + primary pick to GitHub |
| `/api/resolve-line` | POST | Mark pick hit/miss given actual stat |
| `/api/auto-resolve-line` | GET | Cron: auto-resolve picks when games end (requires CRON_SECRET when set) |
| `/api/line-history` | GET | Recent picks with streak + hit rate |

### Parlay
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/parlay` | GET | Safest 3-leg parlay (30 min `/tmp` cache; rate-limited). Auto-saves to `data/parlays/{date}.json` |
| `/api/parlay-history` | GET | Recent parlays + lazy ESPN resolution |
| `/api/parlay-force-regenerate` | GET | Cron: pre-lock regeneration (see `railway.toml`) |
| `/api/parlay-live-stream` | GET (SSE) | Live parlay leg progress stream for in-game updates and ticket state refresh |

### Ben (Lab)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/lab/status` | GET | Lock status + games-final count |
| `/api/lab/briefing` | GET | Prediction accuracy analysis (MAE, biggest misses, patterns) |
| `/api/lab/chat` | POST | Proxy to claude-opus-4-6 with full Lab system prompt |
| `/api/lab/update-config` | POST | Apply dot-notation model param changes, increment version |
| `/api/lab/config-history` | GET | Full config + changelog |
| `/api/lab/rollback` | POST | Note rollback to target version (new version number) |
| `/api/lab/backtest` | POST | Replay historical slates with proposed params, compare MAE |
| `/api/lab/auto-improve` | GET | Cron: auto-tune model (requires CRON_SECRET when set) |
| `/api/lab/skip-uploads` | POST | Record skipped dates (API compat; optional for scripts) |
| `/api/save-most-popular` | POST | Save Most Popular CSV → `data/most_popular/{date}.csv` (`INGEST_SECRET` when set) |
| `/api/save-ownership` | POST | Alias of save-most-popular (writes `data/most_popular/`, not `data/ownership/`) |
| `/api/save-most-drafted-3x` | POST | High-boost list → `data/most_drafted_3x/{date}.csv` |
| `/api/save-winning-drafts` | POST | Long-format winners → `data/winning_drafts/{date}.csv` |
| `/api/lab/calibrate-boost` | GET | Fit card boost params from `data/most_popular/` + legacy `data/ownership/` (≥4 samples) |

## Cron Schedule (UTC)

Source of truth: [`railway.toml`](railway.toml). When `CRON_SECRET` is set, protected crons send `Authorization: Bearer <CRON_SECRET>` (Railway injects via cron commands). `/api/refresh` is intentionally unauthenticated.

| Schedule | Endpoint | Purpose |
|----------|----------|---------|
| `0 19 * * *` | `/api/refresh` | Daily cache clear + auto-save locked predictions |
| `0 9 * * 0,3` | `/api/lab/auto-improve` | Auto-tune model (Wed + Sun only; requires CRON_SECRET when set) |
| `55 19,20,21,22,23,0,1,2,3,4,5,6 * * *` | `/api/refresh-line-odds` | Odds sync during game-window hours (UTC) |
| `0 20,21,22,23,0,1,2,3,4,5,6,7 * * *` | `/api/auto-resolve-line` | Resolve line picks (game-window hours; CRON_SECRET when set) |
| `0 18,22,2 * * *` | `/api/injury-check` | RotoWire injury check (3 windows/day; CRON_SECRET when set) |
| `0 6 * * 1` | `/api/mae-drift-check` | Weekly MAE drift (Monday; CRON_SECRET when set) |
| `0 17 * * *` | `/api/parlay-force-regenerate` | Pre-lock parlay generation → GitHub |

## Responsiveness & Reliability

**Error boundaries**: Backend uses a global exception handler — unhandled exceptions log server-side and return a generic 500 with no stack trace or internal detail. Frontend uses `_safeParseLocalStorage()` for all localStorage JSON and `_el()` with null checks in critical paths; Lab lock poll logs fetch failures; Lab chat has a 60s connection timeout.

**Fetch Timeout Protection**: All frontend API calls enforce hard timeouts (10s default, 30s screenshots) via `Promise.race()` and `AbortController`. Prevents indefinite UI hangs on slow backend.

**Worker pools**: 8 workers for game/slate/picks/audit/line/Odds per-event fetches; 10 workers for parlay ESPN gamelog batch (I/O-bound). See [CLAUDE.md](CLAUDE.md) (Production cache section).

**Polling Intervals**:
- Lab lock status: 2-minute checks (reduces invocations; Retry for immediate check)
- Line live stat: 1-minute checks; max 5 consecutive failures (300s tolerance) before fallback to cron

**GitHub Write Retry**: Exponential backoff (1s, 2s, 4s) on concurrent write conflicts (HTTP 422 SHA mismatch). Fresh SHA fetch on each retry.

**Cache TTLs**: Single source of truth for backend durations is [`api/index.py`](api/index.py) (`_TTL_*` constants) — see **Production: DRY rules and cache inventory** in [CLAUDE.md](CLAUDE.md). Includes: ESPN games list (5 min), model config (5 min), Odds bulk map fresh reuse (10 min) + 1 h degraded snapshot on failure, parlay/line `/tmp` (30 min), game-final polling (60s locked / 180s pre-slate), GitHub-persisted slate. Explicit invalidation via `GET /api/refresh` or `_bust_slate_cache()`.

**If caches look stale after a deploy or config change:** (1) **`/api/refresh` is unauthenticated** — it always runs when hit; it clears `/tmp`, lock files, config cache, busts GitHub slate (and today’s parlay file), and clears RotoWire cache. (2) **Multi-instance** — `/api/refresh` only clears `/tmp` on the container that receives the request; other Railway instances still have their own `/tmp` until they cycle or miss cache. GitHub tombstones from `_bust_slate_cache()` align all instances on the next slate read. (3) **Tombstones** — `data/slate/{date}_games.json` with `_busted` is treated as a miss so picks regenerate.

**Midnight Rollover Handling**: Auto-resolve line picks correctly track `pick_date` separately from ET date, preventing data loss on multi-day slates.

**ESPN API Fallback**: If game status not updated for 4+ hours, mark as final. Requires at least one game in Final status before unlocking (safety against outages).

## Skip uploads (`/api/lab/skip-uploads`)

Optional **API-only** hook: `POST /api/lab/skip-uploads` with `{ "date": "YYYY-MM-DD" }` appends to `data/skipped-uploads.json`. **`save-actuals`** still checks this list and no-ops for skipped dates (useful for scripts or manual merges). There is **no in-app button** for this anymore.

## Environment Variables

All secrets and config live in **environment variables only** — never hardcoded or committed. For local dev, use a `.env` file in the repo root (listed in `.gitignore`; do not commit). For production, set variables in the Railway project dashboard.

| Variable | Purpose |
|----------|---------|
| `GITHUB_TOKEN` | GitHub PAT (repo scope) — for CSV + config read/write |
| `GITHUB_REPO` | e.g. `cheeksmagunda/basketball` |
| `ANTHROPIC_API_KEY` | Claude (screenshot OCR + Ben chat) |
| `ODDS_API_KEY` | The Odds API — player prop lines for Line of the Day |
| `CRON_SECRET` | (optional) Secures cron-only endpoints; Railway injects via cron commands in railway.toml |
| `DOCS_SECRET` | (optional) When set, `/docs`, `/redoc`, and `/openapi.json` require `?docs_key=<value>` or `X-Docs-Key` header |

**Cache refresh:** `GET /api/refresh` requires **no secret** (non-destructive). For GitHub slate bust + full regen after model/config changes, use Lab `update-config` (auto-busts) or call refresh then wait for next slate request. Repository secrets like `CRON_SECRET` are for **protected** crons only (auto-improve, injury-check, auto-resolve-line, MAE drift), not for `/api/refresh`.

## Production readiness

- **Health**: `GET /api/health` — Railway `healthcheckPath`; alert on non-200.
- **Build**: `GET /api/version` — deploy SHA for verifying Railway picked up a push.
- **Tests**: `python3 -m pytest tests/ -v` before shipping (415+ tests; requires `pip install -r requirements.txt`).
- **TTLs / caches**: One backend block (`_TTL_*`, `_CK_*`, `_cp`/`_cg`/`_cs`) — document in [CLAUDE.md](CLAUDE.md); avoid duplicating magic numbers in new endpoints.
- **Deploy**: `watchPatterns` in `railway.toml` — `data/` commits do not rebuild the Docker image.
- **DRY**: Shared Odds bulk map (`_build_player_odds_map`), GitHub games hydration (`_hydrate_game_projs_from_github`), scoped parlay gamelogs (`select_parlay_gamelog_player_ids`).

## LightGBM Model

**12 features**: `avg_min, avg_pts, usage_trend, opp_def_rating, home_away, ast_rate, def_rate, pts_per_min, rest_days, recent_vs_season, games_played, reb_per_min`

Retrained nightly via GitHub Actions (`retrain-model.yml`). Manual retrain: `python train_lgbm.py`.

**Draft popularity + card boost:** `drafts_model.pkl` (`train_drafts_lgbm.py`) and `boost_model.pkl` (`train_boost_lgbm.py`) use labels from **`data/top_performers.csv`**, **`data/actuals/`**, and **`data/most_popular/`**, joined to `data/predictions/` for features—see **Primary historical dataset** below.

## Data Layer

All persistent data stored in the GitHub repo via Contents API:

### Primary historical dataset (top performers, ~2 months)

Real Sports **leaderboard outcomes** (who won the value contest, how many drafts, realized card boost, actual RS) live here. This corpus is the **main ground-truth history** for training/evaluating **draft-count** and **card-boost** models—not generic box scores alone.

| Artifact | Role |
|----------|------|
| **`data/top_performers.csv`** | **Canonical mega file**: one row per (date, player). **Log** and **`_compute_audit`** read this first for each date. Rebuild via `scripts/rebuild_top_performers_mega.py` / sync scripts as needed. |
| **`data/actuals/{date}.csv`** | Legacy per-day CSVs (same columns as mega minus `date`). Used when a date has **no** rows in the mega file, and for older tooling that still expects per-day files. |
| **`data/most_popular/{date}.csv`** | Most-drafted / popularity snapshots (`save-most-popular` / `save-ownership` alias). Training + `calibrate-boost`. |
| **`data/predictions/{date}.csv`** | Pre-game projections — feature source for drafts/boost training and `scripts/verify_top_performers.py`. |

### Other tracked data

Optional: **`data/most_drafted_3x/`**, **`data/winning_drafts/`**, **`data/slate_results/{date}.json`** — see [docs/HISTORICAL_DATA.md](docs/HISTORICAL_DATA.md). **`POST /api/save-actuals`** still merges into `data/actuals/` when used (e.g. manual admin).

- `data/audit/{date}.json` — pre-computed accuracy audit
- `data/lines/{date}.csv` — primary pick for result tracking/resolve
- `data/lines/{date}_pick.json` — dual-pick format `{over_pick, under_pick}` with odds fields
- `data/parlays/{date}.json` — daily parlay ticket + resolution fields
- `data/slate/{date}_slate.json` — GitHub-persisted full slate cache (Starting 5 + Moonshot lineups)
- `data/slate/{date}_games.json` — GitHub-persisted per-game projections (keyed by gameId); also hydrates cold `/tmp` for Line/Parlay
- `data/locks/{date}_slate.json` — cold-start lock recovery (written at lock-promotion time)
- `data/model-config.json` — runtime model parameters (5-min cache, fallback to defaults)
- `data/skipped-uploads.json` — dates where `save-actuals` should no-op (optional; usually set via `POST /api/lab/skip-uploads`)

## Local Development

```bash
pip install -r requirements.txt
python scripts/check-env.py   # verify required env vars (optional but recommended)
uvicorn server:app --reload
# open http://localhost:8000
```

**Historical ingest (curl/scripts):** [docs/HISTORICAL_DATA.md](docs/HISTORICAL_DATA.md) — `top_performers`, `most_popular`, `most_drafted_3x`, `winning_drafts`, `slate_results` JSON, optional `INGEST_SECRET`.

**Find code fast:** [CLAUDE.md — Codebase Navigation (grep tags)](CLAUDE.md#codebase-navigation-grep-tags) lists search anchors for Predict / Line / Parlay (DOM + JS + API), backend modules, dev server, and training scripts.

## Testing

Unit tests cover lock logic, audit computation, GitHub retries, cache TTLs, line cache behavior, and JS syntax (e.g. unescaped apostrophes). Run with:

```bash
python3 -m pytest tests/ -v
python3 scripts/verify_historical_datasets.py   # counts + prediction overlap (add --strict for CI)
```

- **tests/test_fixes.py** — Backend: `_safe_float`, `_is_locked`, audit, GitHub retries, slate/cache, line, parlay-related guards.
- **tests/test_core.py** — Helpers, line cache, JS contract checks, date boundaries.
- **tests/test_parlay.py** — Parlay engine and endpoint contracts.

SSE smoke test for reconnect/state consistency:

```bash
python3 scripts/parlay_sse_smoke.py --base-url http://127.0.0.1:8000 --cycles 5 --ticks 3
# or against prod:
python3 scripts/parlay_sse_smoke.py --base-url https://the-oracle.up.railway.app
```

Tests that import `api.index` require full dependencies (numpy, lightgbm, etc.). Use `pip install -r requirements.txt` first (e.g. in a virtual environment). If pytest reports tests *skipped* with reason "Install dependencies: pip install -r requirements.txt", install the requirements and re-run.

## Deployment

Push to `main` to deploy to production. Railway auto-deploys when `watchPatterns` in `railway.toml` match (code changes only — `data/` and `.github/` are excluded).

```bash
git push -u origin your-branch
# Then: GitHub → Pull requests → New PR (base: main, compare: your-branch) → Merge
# Or push main directly: git checkout main && git merge your-branch && git push origin main
```

## Monitoring / Health check

Use **GET `/api/health`** for uptime checks. It returns `200` with `{ "status": "ok", "config": "ok"|"error", "github": "ok"|"unreachable"|"skipped" }`. Railway uses `healthcheckPath = "/api/health"` from `railway.toml`. Configure an external monitor (e.g. [UptimeRobot](https://uptimerobot.com), [Cronitor](https://cronitor.io)) for alerting on non-200.

## Lock System (Event-Driven Unlock)

Predictions lock 5 minutes before the earliest game tip-off. **Slates unlock based on game completion events**, not clock timeouts.

### Unlock Priority
1. **ESPN Game Final** (primary): If all games marked Final on ESPN → unlock immediately
2. **Time Fallback** (4.5h): If latest game running 4.5+ hours → assume complete (handles ESPN lag)
3. **6h Ceiling**: Safety net if something hangs

### Technical Details
- Cache TTL during locked slate: **60 seconds** (balances responsiveness and Railway resource use)
- Cache TTL pre-slate: 180 seconds (normal polling)
- All slate-level checks use `any(_is_locked(st))` to handle split-window days (2 PM + 9 PM games)
- **Triple-gated prediction saves** — frontend `SLATE.locked` check, backend `any(_is_locked(...))` HTTP 409 guard, and cron guard
- Two write paths to `data/predictions/`: the `/api/save-predictions` endpoint and inline save at lock-promotion in `/api/slate`
- **Lab status resilience:** `/api/lab/status` is wrapped in try/except; on exception it returns 200 with `locked: true` and "Server temporarily unavailable — try again" so the Ben tab shows a retry instead of a failed fetch. See [docs/LOCK_AND_ROUTING_AUDIT.md](docs/LOCK_AND_ROUTING_AUDIT.md).

## Known Limitations

- `/tmp` is ephemeral on Railway — caches don't survive container restarts (deploy or crash). On cold start after lock, `data/locks/{date}_slate.json` on GitHub provides lock recovery.
- Line of the Day odds: picks loaded from the GitHub CSV lack `books_consensus`/`odds_over`/`odds_under` — rendered as `MODEL` label until odds are refreshed via `/api/refresh-line-odds`.
- When over and under picks are the same player, `/api/refresh-line-odds` fetches Odds API once and applies the result to both (deduped).
- RotoWire scraping is free-tier only (availability + injury flags, no projected minutes).
- Keep `data/slate/` and `data/locks/` lean. Old cache snapshots should be pruned so only active-slate artifacts remain.
- History tab shows a 60-day date strip; dates with stored data are highlighted (from `/api/log/dates`).
