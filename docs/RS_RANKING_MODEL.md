# RS ranking model (LightGBM v2)

## Goal

Draft quality is dominated by **ordering players by Real Score (RS)** before card boost and slot multipliers are applied. Training and offline eval prioritize **top‑5 RS recall** and **NDCG@5** on RS, not lineup overlap alone.

## Two-head bundle (`bundle_version: 2`)

Saved in `lgbm_model.pkl`:

| Head | Target |
|------|--------|
| **Baseline** | Core RS level |
| **Spike** | Positive residual above baseline (role-player eruption) |

**Inference:** `pred_rs = baseline + max(0, spike)` (see `api/index.py`: `_lgbm_predict_rs`).

Legacy bundles with a single `model` key remain supported.

## Features (16)

Aligned between `train_lgbm.py` and `api/index.py::_lgbm_feature_vector`:

`avg_min`, `avg_pts`, `usage_trend`, `opp_def_rating`, `home_away`, `ast_rate`, `def_rate`, `pts_per_min`, `rest_days`, `recent_vs_season`, `games_played`, `reb_per_min`, **`l3_vs_l5_pts`**, **`min_volatility`**, **`starter_proxy`**, **`cascade_signal`**.

Volatility features target minutes/usage shape; `cascade_signal` is non-zero at inference when the cascade engine assigns extra minutes.

## Training loss weighting

Per calendar date, sample weights up-weight high **actual RS** rows (approximate slate-wide top performers). See `train_lgbm.py::_assign_weights`.

## Post-lock RS nudge

`real_score.post_lock_calibration` (default **off**): after the context layer and before MILP, optionally re-tilts `rating` / `chalk_ev` / `ceiling_score` from recent scoring vs season and cascade bonus. Gated by `require_locked_slate` so morning Pass‑1 stays unchanged unless enabled.

## Archetype calibration

`real_score.archetype_calibration` (default **off**): coarse role buckets (`star`, `starter`, `wing_role`, `bench_microwave`, `big`) apply multipliers to RS after bucket PPG calibration. Tunable without redeploy via `data/model-config.json`.

## Offline KPIs

```bash
python scripts/eval_rs_ranking.py
python scripts/eval_rs_ranking.py --from 2026-03-05 --to 2026-03-19
```

Reports per date and means for:

- **top5_recall** — overlap between predicted and actual RS top 5 (intersection of that date’s prediction CSV — all scopes — and actuals file).
- **ndcg@5** — RS as relevance, predicted RS order.
- **rs_capture_ratio** — sum of actual RS in predicted top 5 vs oracle top 5.
- **winner_value_ratio** — hindsight total value (actual RS × (slot + actual card boost)) for predicted top 5 vs oracle top 5.

## Retrain

```bash
pip install -r requirements.txt
python train_lgbm.py
```

Requires `data/actuals/*.csv` for real RS labels; otherwise the script falls back to a formula target (see `train_lgbm.py`).
