# Basketball — Real Sports Draft Optimizer

## What This Is

A daily NBA draft optimizer for the **Real Sports** app. It projects player Real Scores, estimates card boosts, and builds optimized 5-player lineups using MILP (mixed-integer linear programming). Deployed on **Vercel** as a serverless Python (FastAPI) backend + single-page HTML frontend.

## How Real Sports Works

- Users draft 5 NBA players each day
- Each player earns a **Real Score** (RS) based on in-game impact (not just box score stats)
- Each player gets a **Card Boost** inversely proportional to how many people drafted them (popular players get low boosts, obscure players get high boosts)
- **Total Value = Real Score x (Slot Multiplier + Card Boost)**
- Slot multipliers: 2.0x, 1.8x, 1.6x, 1.4x, 1.2x (user manually assigns their 5 picks to slots pre-game)
- The winning strategy is drafting **high-RS role players with huge card boosts**, not superstars

## Architecture

```
index.html          — Single-page frontend (vanilla JS, no framework)
api/index.py        — FastAPI backend (all endpoints)
api/real_score.py   — Monte Carlo Real Score projection engine
api/asset_optimizer.py — MILP lineup optimizer (PuLP)
api/temporal_risk.py   — TRAV (Temporal Risk-Adjusted Value) system
vercel.json         — Vercel config (routes, crons, 60s timeout)
server.py           — Local dev server (uvicorn)
train_lgbm.py       — Offline script to retrain lgbm_model.pkl from NBA API data
lgbm_model.pkl      — Trained LightGBM model (blended 70/30 with heuristic)
data/predictions/   — Git-tracked daily prediction CSVs (via GitHub API)
data/actuals/       — Git-tracked daily actual result CSVs (via GitHub API)
```

## Key Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/slate` | GET | Full-slate predictions (all games) |
| `/api/picks?gameId=X` | GET | Per-game predictions |
| `/api/games` | GET | Today's games with lock status |
| `/api/save-predictions` | POST | Save cached predictions to GitHub CSV |
| `/api/parse-screenshot` | POST | Upload Real Sports screenshot, Claude Haiku parses it |
| `/api/save-actuals` | POST | Save parsed actuals to GitHub CSV |
| `/api/refresh` | GET | Clear cache (also runs on cron at 7pm/8pm UTC) |

## Environment Variables (Vercel)

- `GITHUB_TOKEN` — GitHub PAT with repo scope (for CSV read/write via Contents API)
- `GITHUB_REPO` — e.g. `cheeksmagunda/basketball`
- `ANTHROPIC_API_KEY` — For Claude Haiku screenshot parsing

## Lock System

Predictions lock 5 minutes before the earliest game starts. Once locked:
- Backend returns cached predictions (no recomputation with post-tipoff data)
- Lock cache (`/tmp/nba_locks_v1/`) survives within a warm Vercel instance
- On cold start with no cache, returns empty locked response (frontend preserves displayed data)

## Prediction Availability

The app always shows predictions for the current day's upcoming games. As soon as the prior slate ends (all games completed), the next request automatically fetches that day's upcoming games from ESPN using the current Eastern Time date. There is no dead period between slates.

The `no_games_yet` fallback (with first game start time) only triggers if ESPN returns stale data for the explicitly requested ET date — this should be rare.

## Known Limitations

- `/tmp` is ephemeral on Vercel — caches and lineup history don't survive cold starts
- The repetition penalty system (`_apply_repetition_penalty`) depends on `/tmp` history, so it's unreliable across instances
- `real-app-production.zip` in repo root is unused; can be deleted

## Two Lineup Types

- **Starting 5 (chalk)**: MILP-optimized for expected value. Conservative, consistent.
- **Moonshot (upside/contrarian)**: Different 5 players optimized for ceiling. High card boost leverage, opposite-team correlation to Starting 5.

## Development

```bash
# Local
pip install -r requirements.txt
uvicorn server:app --reload

# Deploy
git push origin main  # Vercel auto-deploys from main
```
