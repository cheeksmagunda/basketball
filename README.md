# The Oracle — NBA Draft Optimizer for the Real Sports App

AI-powered NBA draft optimizer built specifically for the **Real Sports App** scoring system — not traditional DFS. Uses Monte Carlo game simulation, MILP slot optimization, and injury cascade analysis.

## Real Sports App vs Traditional DFS

The Real Sports App uses a fundamentally different scoring algorithm than traditional DFS platforms:

| Factor | Traditional DFS | Real Sports App |
|--------|----------------|-----------------|
| Scoring | Static: PTS + REB + AST×1.5 + ... | **Real Score**: context-dependent, non-linear |
| Game Closeness | Irrelevant | **#1 factor** — tight games = exponentially higher scores |
| Clutch Factor | Not modeled | 4th quarter, lead-changing plays get massive boosts |
| Momentum | Not modeled | Streaky/burst production scores higher than steady output |
| Optimal Strategy | Fade stars, target low-ownership bench | **Target players in close games with high clutch potential** |

## How It Works

### Scoring Pipeline
```
ESPN API (daily game data, spreads, totals, injuries)
  → 50/50 Season/Recent stat blending
  → Injury Cascade Engine (redistribute OUT player minutes, capped +3 min/player)
  → DFS Base Score (proxy for raw production)
  → LightGBM AI Model (70% weight, with spread-derived opponent quality)
  → Contextual Adjustments (pace, spread closeness, home/away)
  → Monte Carlo Real Score Engine (closeness C_c, clutch C_k, momentum M_m)
  → MILP Slot Optimizer (assigns players to 2.0x/1.8x/1.6x/1.4x/1.2x slots)
```

### Full Slate (Starting 5 + Moonshot)
- **Starting 5:** Highest projected Real Score picks. Near-neutral ownership weighting — Real Score engine already favors players in close games over stars in blowouts.
- **Moonshot:** 5 different players targeting high Real Score ceiling — close-game environments, high variance, momentum potential. NOT about low-minute bench players.

### Per-Game Analysis
Single-game drafts with team balance (min 2 per team) and game script adjustments:

| O/U Range | Script | Strategy |
|-----------|--------|----------|
| < 220 | Defensive Grind | Boost STL/BLK, suppress volume stats |
| 220-235 | Balanced Pace | Neutral — lean on matchup and spread |
| 236-245 | Fast-Paced | Boost scorers, assists, rebounders |
| > 245 | Track Meet | Boost PTS+AST combos, penalize if spread > 8 |

### Key Model Parameters
| Parameter | Value | Why |
|-----------|-------|-----|
| Ownership mult | 1.0 (neutral) | Real Score handles differentiation via game context |
| Stat blend | 50% season / 50% recent | Balances stability with matchup sensitivity |
| Cascade cap | +3 min/player max | Prevents unrealistic bench projections |
| Cascade badge threshold | +2.5 min | Only flags genuine role expansions |
| Decline trigger | Recent < 90% of season min | Catches declining usage earlier |
| Raw score cap | 15.0 | Power-compressed to match actual Real Score gaps |
| Spread adjustment | 4% range | Stronger boost for tight games |
| Opponent quality | Derived from spread | Not hardcoded (was 112.0 for all games) |

### Injury Cascade Engine
When a player is OUT, their minutes redistribute to teammates at same position group. Bench players get proportionally more. Capped at +3 min per player to prevent unrealistic projections.

### Calibration Feedback Loop
After entering actual results via the screenshot uploader, the data is saved to `data/actuals/` in the GitHub repo as a CSV. These CSVs can be diffed against `data/predictions/` to evaluate model accuracy over time.

## Architecture

- **Frontend:** Single-file `index.html` (vanilla JS). PWA-capable.
- **Backend:** FastAPI (`api/index.py`) on Vercel serverless.
- **Data:** ESPN API (scoreboard, rosters, athlete overview). No API key needed.
- **AI Model:** LightGBM (`lgbm_model.pkl`) blended 70/30 with heuristic scoring.
- **Real Score Engine:** Monte Carlo simulation (closeness, clutch, momentum coefficients).
- **Optimizer:** PuLP/CBC MILP solver for slot assignment.
- **Storage:** GitHub Contents API for persistent CSVs (`data/predictions/`, `data/actuals/`); `/tmp` for ephemeral server-side caching (lost on Vercel cold start).

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
