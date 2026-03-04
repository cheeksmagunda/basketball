# The Oracle — NBA Draft Optimizer

AI-powered NBA draft optimizer built for the Real Sports App. Uses a closeness-aware scoring model where game context — not raw stats — drives player rankings.

## How It Works

### Closeness-Aware Scoring Model

The core insight: identical stats are worth far more in a close game than a blowout. The model applies a **closeness coefficient** (exponential decay based on pre-game spread) to all stat projections:

- **Pick'em (spread 0):** ~1.30x boost — every possession matters
- **Tight game (spread 3):** ~1.15x
- **Average (spread 7):** ~0.90x baseline
- **Blowout (spread 12+):** ~0.55x — starters sit, garbage time deflates value

This creates a ~2.2x ratio between the closest and widest-spread games on a slate. Defensive stats (STL/BLK) get an extra closeness bonus since they swing win probability in tight games.

### Full Slate (Starting 5 + Moonshot)

Analyzes all games on today's NBA slate and builds two 5-player lineups:

- **Starting 5:** Top 5 players by EV (Real Score × ownership multiplier). Bench players (15-22 min) get 3.0x, role players (22-28 min) get 2.5x, starters get 1.0x, and stars (33+ min) get 0.5x — matching actual draft slot data where low-owned players land in high-multiplier slots.
- **Moonshot:** 5 different players ranked by a ceiling score that blends production, game variance, recent hot streaks, and defensive upside. Targets role players in close, high-total games who can explode in crunch time or overtime.

### Per-Game Analysis

Single-game drafts with two additions:

1. **Team Balance:** Guarantees at least 2 players from each team.
2. **Game Script Engine:** Adjusts stat weights based on the game's over/under:

| O/U Range | Script | Strategy |
|-----------|--------|----------|
| < 220 | Defensive Grind | Boost STL/BLK, suppress volume stats |
| 220-235 | Balanced Pace | Neutral — lean on matchup and spread |
| 236-245 | Fast-Paced | Boost scorers, assists, rebounders |
| > 245 | Track Meet | Boost PTS+AST combos, blowout penalty if spread > 6 |

### Blowout Protection

Players projected 28+ minutes in high-spread games get penalized (up to 30% reduction). Universal blowout penalty kicks in at spread > 6 across all O/U tiers, suppressing PTS and AST projections.

### Injury Handling

Players ruled OUT are excluded from projections. Injury status badges (Questionable, Day-To-Day, Doubtful) display on player cards for user awareness — injury status does not factor into scoring.

### Calibration Feedback Loop

After entering actual results in the Lab, the backend calculates prediction bias and adjusts future projections via exponential moving average (alpha=0.3).

### Next-Slate Auto-Transition

When all games on the current slate have ended, the app automatically loads tomorrow's slate. Projections update hourly until games lock (5 min before tip).

## Architecture

- **Frontend:** Single-file `index.html` (vanilla JS, no framework). PWA-capable.
- **Backend:** FastAPI (`api/index.py`) deployed on Vercel serverless functions.
- **Data:** ESPN API (scoreboard, rosters, athlete stats). No API key required.
- **AI Model:** LightGBM (`lgbm_model.pkl`) blended at 15% with the heuristic scoring engine. The model was trained on traditional DFS targets and serves as a smoothing regularizer.
- **Storage:** LocalStorage for Lab history, `/tmp` for server-side caching.

## Local Development

```bash
pip install -r requirements.txt
uvicorn server:app --reload
```

## Deployment

Deployed on Vercel. Push to main triggers auto-deploy.

```bash
vercel --prod
```
