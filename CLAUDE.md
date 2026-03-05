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
data/lines/            — Git-tracked daily Line of the Day picks (via GitHub API)
vercel.json            — Vercel config (routes, crons, 60s timeout)
server.py              — Local dev server (uvicorn)
```

## UI Structure

4-tab segmented control navigation: **Predict | Line | Ben | History**

- **Predict**: Live slate optimizer (Starting 5 + Moonshot) and per-game analysis
- **Line**: Line of the Day — best player prop edge from Odds API (gold accent)
- **Ben**: Model Lab — Claude-powered model tuning, config management, backtesting (teal accent)
- **History**: Historical drill-down — date strip, game grid, locked prediction cards, screenshot upload, winning drafts, hindsight optimal lineup

## Key Endpoints

### Core
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/slate` | GET | Full-slate predictions (all games) |
| `/api/picks?gameId=X` | GET | Per-game predictions |
| `/api/games` | GET | Today's games with lock status |
| `/api/save-predictions` | POST | Save cached predictions to GitHub CSV |
| `/api/parse-screenshot` | POST | Upload Real Sports screenshot, Claude Haiku parses it |
| `/api/save-actuals` | POST | Save parsed actuals to GitHub CSV |
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

## Ben (Lab) Lock System

- **Locked** 5 minutes before first game tip-off (slate is in progress)
- **Unlocked** when ALL games on today's slate reach "Final" status on ESPN
- During lock: shows read-only config changelog, estimated unlock time
- During unlock: full chat + backtest + config update capabilities

## Lock System

Predictions lock 5 minutes before the earliest game starts. Once locked:
- Backend returns cached predictions (no recomputation with post-tipoff data)
- Lock cache (`/tmp/nba_locks_v1/`) survives within a warm Vercel instance
- On cold start with no cache, returns empty locked response (frontend preserves displayed data)

## Two Lineup Types

- **Starting 5 (chalk)**: MILP-optimized for expected value using `chalk_ev = rating × (avg_slot + card_boost)`. Conservative, consistent.
- **Moonshot**: The next 5 players by the same chalk_ev ranking (ranks 6-10). Same methodology as chalk — NOT a separate contrarian algorithm. This ensures moonshot picks are always players with real projected RS, avoiding DNP risks from extreme low-ownership targets.

## Known Limitation

`/tmp` is ephemeral on Vercel — caches don't survive cold starts. On cold start after lock, the frontend preserves the last displayed data client-side.

## Development

```bash
# Local
pip install -r requirements.txt
uvicorn server:app --reload

# Deploy
git push origin main  # Vercel auto-deploys from main
```
