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
index.html             — 4-tab frontend (Predict | Line | Ben | History, vanilla JS)
api/index.py           — FastAPI backend (all endpoints, projection engine, Lab/Line)
api/real_score.py      — Monte Carlo Real Score projection engine
api/asset_optimizer.py — MILP lineup optimizer (PuLP)
api/line_engine.py     — Prop edge detection pipeline (Odds API + confidence model)
api/rotowire.py        — RotoWire lineup scraper (free tier: availability + injury flags)
api/temporal_risk.py   — TRAV module (available, not active in picks)
data/model-config.json — Runtime model config (Lab writes here; 5-min cache)
data/predictions/      — Git-tracked daily prediction CSVs (via GitHub API)
data/actuals/          — Git-tracked daily actual result CSVs (via GitHub API)
data/audit/            — Git-tracked daily audit JSONs (auto-generated on save-actuals)
data/lines/            — Git-tracked daily Line of the Day picks (via GitHub API)
data/locks/            — Cold-start recovery: {date}_slate.json written at lock-promotion time
lgbm_model.pkl         — LightGBM model bundle {model, features} — retrained by retrain-model.yml
train_lgbm.py          — Training script (11 features, run locally or via GitHub Actions)
vercel.json            — Vercel config (routes, crons, 300s timeout on Pro plan)
server.py              — Local dev server (uvicorn)
```

## UI Structure

4-tab segmented control navigation (Apple glassmorphism pill style): **Predict | Line | Ben | History**

- **Predict**: Live slate optimizer (Starting 5 + Moonshot), per-game analysis, Magic 8-ball loading animation. "Slate-Wide | Game" sub-tabs inline at top of tab.
- **Line**: Line of the Day — best player prop edge (gold accent). "Over | Under" sub-tabs inline at top of tab. Odds refresh hourly from Odds API; pick cards show "Odds · [time] CT".
- **Ben**: Plain chat interface with Claude (no quick-action buttons — user asks naturally). Teal accent. Locked during games, unlocked after final.
- **History**: Historical drill-down — date strip, game grid, read-only prediction cards vs actuals (no user input — upload happens through Ben)

### Sub-Nav Tabs (inline, not floating)
Both `predictSubNav` (Slate-Wide | Game) and `lineSubNav` (Over | Under) are inline `div.predict-sub-nav` elements positioned at the top of their respective tab pages. They match the `.mode-tab` visual language exactly — same height, padding, `border-radius:11px`, Barlow Condensed 800. Active states: predict = chalk blue, Over = gold (`--line`), Under = teal (`--lab`).

## Codebase Navigation (grep tags)

All major sections in `api/index.py` and `index.html` are tagged with `# grep:` comments for fast searching:

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
grep: CONSTANTS & CACHE        — _cp, _cg, _cs, _lp, _lg, ESPN, MIN_GATE
grep: ESPN DATA FETCHERS       — fetch_games, fetch_roster, _fetch_athlete
grep: INJURY CASCADE           — _cascade_minutes, _pos_group
grep: CARD BOOST               — _est_card_boost, _dfs_score
grep: GAME SCRIPT              — _game_script_weights, _game_script_label
grep: PLAYER PROJECTION        — project_player, pinfo, rating, est_mult
grep: GAME RUNNER              — _run_game, _build_lineups, chalk_ev
grep: CORE API ENDPOINTS       — /api/games, /api/slate, /api/picks
grep: LINE OF THE DAY ENGINE   — /api/line-of-the-day, run_line_engine
grep: BEN / LAB ENGINE         — /api/lab/*, _all_games_final, lab lock
```

## Key Endpoints

### Core
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/slate` | GET | Full-slate predictions (all games) |
| `/api/picks?gameId=X` | GET | Per-game predictions |
| `/api/games` | GET | Today's games with lock status |
| `/api/save-predictions` | POST | Save cached predictions to GitHub CSV (deduped — skips commit if unchanged) |
| `/api/parse-screenshot` | POST | Upload Real Sports screenshot, Claude Haiku parses it |
| `/api/save-actuals` | POST | Save parsed actuals to GitHub CSV + auto-generates audit JSON |
| `/api/audit/get?date=X` | GET | Pre-computed accuracy audit for a date (MAE, directional acc, misses) |
| `/api/log/dates` | GET | List dates with stored prediction/actual data |
| `/api/log/get?date=X` | GET | Predictions + actuals for a given date, grouped by scope |
| `/api/hindsight` | POST | Optimal hindsight lineup from actual RS scores |
| `/api/refresh` | GET | Clear cache + config cache (also runs on cron at 7pm/8pm UTC) |

### Line of the Day
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/line-of-the-day` | GET | **Both** Over + Under picks (6 parallel Haiku calls: 3 stats × 2 dirs); returns `{over_pick, under_pick, pick}` |
| `/api/refresh-line-odds` | GET | **Hourly cron** — fetch current bookmaker line from Odds API and update `line`, `odds_over`, `odds_under`, `books_consensus`, `line_updated_at` on today's pick JSON. No-op if slate is locked. Returns `{status, updated, timestamp}` |
| `/api/save-line` | POST | Save `{over_pick, under_pick}` JSON + primary pick to CSV; backward-compat with legacy single-pick |
| `/api/resolve-line` | POST | Mark pick hit/miss given actual stat |
| `/api/line-history` | GET | Recent picks with streak + hit rate |

### Lab (Ben)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/lab/status` | GET | Lock status (locked during slate, unlocked after all games final) |
| `/api/lab/briefing` | GET | Prediction accuracy analysis (MAE, biggest misses, patterns) |
| `/api/lab/update-config` | POST | Apply dot-notation param changes, increment version |
| `/api/lab/config-history` | GET | Full config + changelog |
| `/api/lab/rollback` | POST | Note rollback to target version (new version number) |
| `/api/lab/backtest` | POST | Replay historical slates with proposed params, compare MAE |
| `/api/lab/auto-improve` | GET | **Cron endpoint** (daily 9am UTC): briefing → Haiku proposes change → backtest → auto-apply if ≥3% improvement |
| `/api/lab/chat` | POST | Proxy to claude-opus-4-6 with Lab system prompt (keeps key server-side) |

## Environment Variables (Vercel)

- `GITHUB_TOKEN` — GitHub PAT with repo scope (for CSV + config read/write via Contents API)
- `GITHUB_REPO` — e.g. `cheeksmagunda/basketball`
- `ANTHROPIC_API_KEY` — Claude Haiku (screenshot OCR) + claude-opus-4-6 (Ben/Lab chat)
- `ODDS_API_KEY` — The Odds API for player prop lines (Line of the Day)

## Runtime Config System

Model parameters are stored in `data/model-config.json` on GitHub. The backend loads this
file at startup and caches it for 5 minutes. The Lab writes updates via the GitHub Contents API.

- **No redeploy needed** to tune parameters — changes take effect within 5 minutes
- **Fallback to defaults** if GitHub is unreachable — app never breaks
- Use `_cfg("dot.path", default)` helper anywhere in `api/index.py` to read config
- `/api/refresh` also clears the config cache for immediate effect

## Ben (Lab) Interface

Ben is a **pure chat interface** — no quick-action buttons. The user types naturally and Ben:
- Auto-loads the briefing and config context silently on open (hidden messages)
- Offers to run backtests, apply config changes, analyze accuracy — all via conversation
- Decision history and config changes are stored in `LAB.messages` and `data/model-config.json`
- The chat prompt includes full system context (briefing data, config state, backtest capability)

### End-of-Day Upload Flow (in Ben)
After all games are final and Ben unlocks, if no messages yet:
- Ben auto-prompts with a message containing two upload buttons: **📸 Real Scores** and **🏆 Top Drafts**
- Tapping a button opens the device file picker (dynamic `createElement('input')` — no HTML needed)
- `_handleBenUpload()` runs the full pipeline: `parse-screenshot → save-actuals → hindsight → hidden message → labCallClaude()`
- Ben responds with analysis + hindsight lineup, buttons turn green (✓) on success
- History page is now **read-only** — no upload UI there

### Lock System
- **Locked** 5 minutes before first game tip-off (slate is in progress)
- **Unlocked** when ALL games on today's slate reach "Final" status on ESPN (3-min TTL cache)
- During lock: shows read-only locked state with estimated unlock time
- During unlock: full chat capabilities + end-of-day upload prompt (if first session open)

### Keyboard / Nav Behavior (Ben tab)
- On **mobile**: focusing `#labInput` hides the bottom nav and expands `#tab-lab` to fill freed keyboard space via `lab-kb-open` CSS class. Blur restores everything.
- On **desktop**: keyboard handler is skipped entirely via `window.matchMedia('(hover: none) and (pointer: coarse)')` — no nav hiding.
- CSS class `#tab-lab.active` uses `height: calc(100dvh - 80px - 120px)` (leaves room for nav). `#tab-lab.active.lab-kb-open` expands to `calc(100dvh - 80px)` (nav hidden).

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

## Lock System

Predictions lock 5 minutes before the earliest game starts. Once locked:
- **`/api/save-predictions`** rejects (HTTP 409) pre-lock calls — server-side guard regardless of caller
- Backend returns cached predictions (no recomputation with post-tipoff data)
- Lock cache (`/tmp/nba_locks_v1/`) survives within a warm Vercel instance
- On cold start with no cache, `data/locks/{date}_slate.json` on GitHub is the recovery source
- On cold start with no GitHub backup either, returns empty locked response (frontend preserves displayed data)

## Two Lineup Types

- **Starting 5 (chalk)**: MILP-optimized for `chalk_ev = rating × (avg_slot + card_boost) × reliability`. Conservative, consistent.
- **Moonshot** (v2): Options strategy. Hard floor of 20 projected minutes + RotoWire lineup clearance + minimum 2.0 rating. Ranked by `moonshot_ev = predMin × card_boost² × dev_team_bonus × rating`. Development/tanking team players get 1.25x boost. Philosophy: buy cheap lottery tickets (high minutes + low drafts), let positive variance do the work.

### Development Teams (configurable in model-config.json)
Current default: `UTA, IND, BKN, CHI, NOP, SAC, MEM, WAS, DAL` — teams effectively out of playoff contention whose role players get predictable developmental minutes and structurally lower ownership. **This list is a seasonal snapshot** — update via Ben or directly in `data/model-config.json` as the standings shift.

### RotoWire Integration (`api/rotowire.py`)
Free-tier scrape of RotoWire NBA lineups page. Runs ~30 min before first tip. Returns player availability (confirmed/expected/questionable/OUT). Moonshot hard-filters on this: any player flagged OUT or questionable is excluded. Cache TTL: 30 minutes.

## Model Improvements (deployed)

### LightGBM (11 features, `lgbm_model.pkl`)
Features: `avg_min, avg_pts, usage_trend, opp_def_rating, home_away, ast_rate, def_rate, pts_per_min, rest_days, recent_3g_trend, games_played`

- Model bundle format: `{"model": lgb.LGBMRegressor, "features": [...]}` — legacy bare-model pkl still supported.
- `rest_days` and `games_played` default to `2.0` / `40.0` at inference (not in ESPN splits).
- Retrained nightly by GitHub Actions (`retrain-model.yml`). Retrain manually: `python train_lgbm.py`.

### Card Boost (`_est_card_boost`)
- Default: exponential heuristic `scalar × decay_base^hype + base_offset`.
- Log-formula path (calibrated, off by default): `log_a - log_b × log10(predicted_drafts)`. Activate with `card_boost.log_formula_active: true` in config once 50+ actuals collected.
- Star player list in `card_boost.star_players` config (treated like big-market teams for ownership).

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
- Picks loaded from GitHub CSV lack `books_consensus/odds_over/odds_under` — render as `MODEL` label. Picks refreshed via `/api/refresh-line-odds` show actual book odds + count.
- Pick cards display `"Odds · [time] CT"` when `line_updated_at` is present (stamped by `/api/refresh-line-odds`)

### Odds Refresh Pipeline
- **Crons**: `0 * * * *` (hourly) + `55 * * * *` (every :55, hits common 6:55 PM ET lock)
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

## Cron Schedule (vercel.json)

| Schedule (UTC) | Endpoint | Purpose |
|----------------|----------|---------|
| `0 19 * * *` | `/api/refresh` | Cache clear + auto-save locked predictions |
| `0 20 * * *` | `/api/refresh` | Second cache clear pass |
| `0 9 * * *` | `/api/lab/auto-improve` | Auto-tune model if ≥3% MAE improvement |
| `0 * * * *` | `/api/refresh-line-odds` | Hourly bookmaker odds sync |
| `55 * * * *` | `/api/refresh-line-odds` | Pre-lock odds sync (hits 6:55 PM ET window) |

## Production Robustness Notes

All frontend API calls (`fetch(...)`) have `.ok` checks before calling `.json()`. Missing `.ok` checks were a common source of silent failures in prior versions.

Key patterns used throughout:
- Async functions: `if (!r.ok) throw new Error('HTTP ' + r.status)` before `.json()`
- Promise.allSettled chains: `fetch(...).then(r => r.ok ? r.json() : Promise.reject(...))`
- Polling loops: `.then(r => r.ok ? r.json() : Promise.reject())` with empty `.catch`
- `savePredictions`: resets `_predSavedDate` on non-OK responses so the next call can retry

EOD prompt check uses `LAB.messages.filter(m => !m.hidden).length === 0` — hidden context-loading messages don't suppress the upload prompt.

## Known Limitations

- `/tmp` is ephemeral on Vercel — caches don't survive cold starts. On cold start after lock, the frontend preserves the last displayed data client-side.
- `_github_write_file` does a GET + PUT internally (fetches current SHA before writing). On concurrent writes (rare), the second write may 422. The cron + user refresh pattern makes this extremely unlikely.
- Odds API odds refresh: if both over_pick and under_pick are for the same player, the function makes two identical Odds API calls (one per pick object). Functionally correct, marginally wasteful on quota.
- `data/locks/` accumulates one JSON per day with no automated cleanup. GitHub directory listings get marginally slower over a long season; manually prune if needed.
- History tab shows only the last 30 days. Predictions older than 30 days are stored in GitHub but not reachable from the UI date strip.

## Development

```bash
# Local
pip install -r requirements.txt
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
3. **No test suite to run** — deploy triggers automatically on push; verify on `basketball-chi-cyan.vercel.app`
4. **Data layer**: All persistent state in GitHub via Contents API (`data/` directory). No database.
5. **Key globals in frontend**: `SLATE`, `PICKS_DATA`, `LOG`, `LAB`, `LINE_DIR`, `LINE_OVER_PICK`, `LINE_UNDER_PICK`, `LINE_LOADED_DATE`
6. **Cache**: Check `CACHE_DIR` in `api/index.py` for the current tmp path (versioned, e.g. `/tmp/nba_cache_v19/`). `/api/refresh` clears all caches + config.
7. **Config**: `data/model-config.json` on GitHub — Ben/Lab writes here, backend reads with 5-min TTL.
