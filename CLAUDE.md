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
index.html             — Single-page frontend (vanilla JS, no framework)
api/index.py           — FastAPI backend (all endpoints, projection engine)
api/real_score.py      — Monte Carlo Real Score projection engine
api/asset_optimizer.py — MILP lineup optimizer (PuLP)
api/temporal_risk.py   — TRAV (Temporal Risk-Adjusted Value) system
vercel.json            — Vercel config (routes, crons, 60s timeout)
server.py              — Local dev server (uvicorn)
data/predictions/      — Git-tracked daily prediction CSVs (via GitHub API)
data/actuals/          — Git-tracked daily actual result CSVs (via GitHub API)
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
