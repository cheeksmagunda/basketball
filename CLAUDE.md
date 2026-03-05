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
api/temporal_risk.py   — TRAV module (available, not active in picks)
data/model-config.json — Runtime model config (Lab writes here; 5-min cache)
data/predictions/      — Git-tracked daily prediction CSVs (via GitHub API)
data/actuals/          — Git-tracked daily actual result CSVs (via GitHub API)
data/audit/            — Git-tracked daily audit JSONs (auto-generated on save-actuals)
data/lines/            — Git-tracked daily Line of the Day picks (via GitHub API)
lgbm_model.pkl         — LightGBM model bundle {model, features} — retrained by retrain-model.yml
train_lgbm.py          — Training script (11 features, run locally or via GitHub Actions)
vercel.json            — Vercel config (routes, crons, 300s timeout on Pro plan)
server.py              — Local dev server (uvicorn)
```

## UI Structure

4-tab segmented control navigation (Apple glassmorphism pill style): **Predict | Line | Ben | History**

- **Predict**: Live slate optimizer (Starting 5 + Moonshot), per-game analysis, Magic 8-ball loading animation
- **Line**: Line of the Day — best player prop edge from Odds API (gold accent)
- **Ben**: Plain chat interface with Claude (no quick-action buttons — user asks naturally). Teal accent. Locked during games, unlocked after final.
- **History**: Historical drill-down — date strip, game grid, locked prediction cards, screenshot upload, winning drafts, hindsight optimal lineup

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
grep: LOG PAGE                 — initLogPage, selectLogDate, drill-down
grep: LINE PAGE                — initLinePage, renderLinePickCard
grep: LAB PAGE                 — initLabPage, LAB state, labCallClaude
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
| `/api/line-of-the-day` | GET | Best prop edge (Odds API → edge detection → confidence) |
| `/api/save-line` | POST | Save daily pick to data/lines/{date}.csv (once/day) |
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
| `/api/lab/chat` | POST | Proxy to claude-sonnet-4-6 with Lab system prompt (keeps key server-side) |

## Environment Variables (Vercel)

- `GITHUB_TOKEN` — GitHub PAT with repo scope (for CSV + config read/write via Contents API)
- `GITHUB_REPO` — e.g. `cheeksmagunda/basketball`
- `ANTHROPIC_API_KEY` — Claude Haiku (screenshot OCR) + claude-sonnet-4-6 (Ben/Lab chat)
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

### Lock System
- **Locked** 5 minutes before first game tip-off (slate is in progress)
- **Unlocked** when ALL games on today's slate reach "Final" status on ESPN (3-min TTL cache)
- During lock: shows read-only locked state with estimated unlock time
- During unlock: full chat capabilities

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

## Lock System

Predictions lock 5 minutes before the earliest game starts. Once locked:
- Backend returns cached predictions (no recomputation with post-tipoff data)
- Lock cache (`/tmp/nba_locks_v1/`) survives within a warm Vercel instance
- On cold start with no cache, returns empty locked response (frontend preserves displayed data)

## Two Lineup Types

- **Starting 5 (chalk)**: MILP-optimized for `chalk_ev = rating × (avg_slot + card_boost) × reliability`. Conservative, consistent.
- **Moonshot**: MILP-optimized for `moonshot_ev = rating × (avg_slot + card_boost × 1.5) × variance_bonus`. Weights card boost 1.5× and rewards inconsistent players who could boom. Excludes chalk picks. Always real projections — no garbage-time DNP traps.

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

## Known Limitation

`/tmp` is ephemeral on Vercel — caches don't survive cold starts. On cold start after lock, the frontend preserves the last displayed data client-side.

## Development

```bash
# Local
pip install -r requirements.txt
uvicorn server:app --reload

# Deploy — push to feature branch; auto-merge-to-main.yml merges → main → Vercel
git push -u origin claude/auto-merge-to-main-ZwBZw
```
