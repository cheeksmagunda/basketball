# The Oracle — NBA Draft Optimizer

AI-powered NBA DFS draft optimizer with injury cascade analysis, game script modeling, and calibration feedback.

## How It Works

### Full Slate (Starting 5 + Moonshot)
Analyzes all games on today's NBA slate and builds two 5-player lineups:

- **Starting 5:** Best expected value picks weighted by inverse ownership. Bench sweet-spot players (15-22 min) get the highest multiplier because fewer people draft them. Stars get penalized — everyone picks them, so they land in low-multiplier slots.
- **Moonshot:** 5 completely different players with a higher production floor (rating >= 6.0). Filters out low-production bench warmers and targets high-ceiling role players.

### Per-Game Analysis
Single-game drafts use a different model with two key differences:

1. **Team Balance:** Guarantees at least 2 players from each team (no all-NO lineups).
2. **Game Script Engine:** Adjusts stat weights based on the game's over/under:

| O/U Range | Script | Strategy |
|-----------|--------|----------|
| < 220 | Defensive Grind | Boost STL/BLK, suppress volume stats |
| 220-235 | Balanced Pace | Neutral — lean on matchup and spread |
| 236-245 | Fast-Paced | Boost scorers, assists, rebounders |
| > 245 | Track Meet | Boost PTS+AST combos, penalize if spread > 8 (blowout risk) |

### Injury Cascade Engine
When a player is ruled OUT, their projected minutes get redistributed to remaining teammates at the same position group. Bench players get proportionally more of the freed minutes. This is how the model finds value plays like expanded-role backups.

### Calibration Feedback Loop
After entering actual results in the Lab, the backend calculates prediction bias and adjusts future projections. The calibration is an exponential moving average (alpha=0.3) of prediction errors.

### DFS Scoring Formula
```
PTS + REB + AST*1.5 + STL*3.5 + BLK*3.0 - TOV*1.2
```

## Architecture

- **Frontend:** Single-file `index.html` (vanilla JS, no framework). PWA-capable with Add to Home Screen support.
- **Backend:** FastAPI (`api/index.py`) deployed on Vercel serverless functions.
- **Data:** ESPN API (scoreboard, rosters, athlete overview). No API key required.
- **AI Model:** LightGBM model (`lgbm_model.pkl`) blended 70/30 with heuristic scoring.
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
