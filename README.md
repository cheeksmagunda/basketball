# The Oracle — NBA Draft Optimizer

AI-powered NBA DFS draft optimizer with injury cascade analysis, game script modeling, and calibration feedback.

## How It Works

### Full Slate (Starting 5 + Moonshot)
Analyzes all games on today's NBA slate and builds two 5-player lineups:

- **Starting 5:** Highest projected producers with a moderate ownership tilt (1.0–1.5x). Stars are NOT penalized below 1.0x — raw production is the primary signal. The raised rating cap (25.0 vs old 15.0) lets the optimizer differentiate between superstars and solid starters, producing mixed lineups of 2-3 stars + 2-3 role players instead of all-star or all-bench extremes.
- **Moonshot:** 5 completely different players with a production floor (rating >= 5.0). Contrarian scoring uses mild inverse-popularity (0.85–1.3x) to find under-owned producers without sacrificing quality for ownership.

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

### Ownership Philosophy (Production-First)
The previous ownership curve (0.9x stars – 2.8x bench) was a 3.1:1 ratio that caused the model to systematically pick low-production bench players over actual producers. Cross-analysis against real results showed 0/10 hit rate — the top performers were always mid-to-high-production players regardless of ownership tier.

**New curve (1.0x – 1.5x):**
| Minutes | Tier | Multiplier | Rationale |
|---------|------|-----------|-----------|
| 33+ | Stars | 1.0x | Neutral — high output carries them |
| 28-33 | Starters | 1.2x | Mild edge |
| 22-28 | Role players | 1.4x | Solid slot advantage |
| 15-22 | Bench | 1.5x | Best expected slot |
| <15 | Deep bench | Filtered | Below minutes gate |

**Rating cap raised** from 15.0 to 25.0 — the old cap squished all stars to identical 15.0 ratings, preventing the optimizer from differentiating between a superstar (18+) and a solid starter (12-14). Combined with the moderate ownership tilt, this naturally produces mixed lineups.

**INJ BOOST threshold raised** from 2.0 to 5.0 cascade minutes — the old threshold flagged nearly every teammate when multiple players were OUT. Now only genuinely significant minute redistributions (backup → starter role) trigger the badge.

**Lock logic fixed** — stale/completed games (started >3h ago) no longer trigger "PICKS LOCKED" on the new slate. This prevented yesterday's ESPN scoreboard data from locking tomorrow's games.

**Contrarian (Moonshot) scoring** similarly flattened from 0.5–2.0 to 0.85–1.3 inv_pop range. Cascade bonus reduced 1.4→1.2, underdog bonus 1.2→1.1. Total effective range went from 6.7:1 to ~1.7:1.

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
