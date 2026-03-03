# Daily Draft Optimizer (Real App)

A heuristic-based Daily Fantasy Sports (DFS) lineup optimizer specifically tailored for real-time NBA outperformance predictions. 

## The Prediction Logic
Unlike standard fantasy optimizers that project raw points, this application uses a customized weighting algorithm to predict **baseline outperformance**—hunting for players who will massively exceed their averages. 

The application utilizes two distinct scoring models depending on the slate:

1. **Full Slate Mode:** Enforces a strict minutes-floor penalty (26+ mins) to prevent fragile bench players from hijacking the top 2.0x slots, leaning heavily on high-volume starters with situational edges.
2. **Single Game Mode:** Highly aggressive. Forgiving of minute floors down to 18 minutes, specifically hunting for role players on hot streaks facing weak defenses to fill the critical 1.2x and 1.4x draft slots.

### Features
* **Live ESPN Integration:** Leverages raw ESPN API endpoints (scoreboard, rosters, common v3, and core v2) to bypass traditional NBA data rate limits.
* **Real-time Injury Parsing:** Automatically drops players who are ruled `Out` by the ESPN injury endpoints, preventing dead lineup slots.
* **Volume-Scaling (The "Scrub Trap" Fix):** A mathematical penalty applied to players projecting under 35 minutes, scaling down their situational boosts (Matchup/Form) to properly balance a 15-minute bench player with a perfect matchup vs a 36-minute star with an average matchup.

### Lineup Strategies
* **Chalk (Optimal):** Focuses heavily on high-floor, safe minutes anchored to baseline production `(rating^1.2 * boost^0.8)`. Always ensures top raw scorers are anchored in the lineup.
* **Upside (Tournament):** Blends differentiated form gems and injury-pivot plays `(usage^2.5 * matchup^2.0 * form^1.5)`. Hunts for expanding roles and the perfect defensive matchup.

## Local Development
Run the app locally with Uvicorn and FastAPI:

```bash
pip install -r requirements.txt
uvicorn server:app --reload
