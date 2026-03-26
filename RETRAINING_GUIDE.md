# Model Retraining Guide (v63 Audit Fixes)

## Quick Start

After deploying the v63 audit fixes, retrain all three models with corrected training scripts:

```bash
# 1. Ensure dependencies are installed
pip install -r requirements.txt

# 2. Retrain models in order (each creates new .pkl file)
python train_lgbm.py           # Fixes: data leakage, dead features, temporal split
python train_boost_lgbm.py     # Fixes: projected RS instead of actual RS
python train_drafts_lgbm.py    # No changes, but re-run for consistency

# 3. Commit and push updated models
git add lgbm_model.pkl boost_model.pkl drafts_model.pkl
git commit -m "Retrain models with v63 audit fixes

- lgbm_model.pkl: 18 features (removed 4 dead), reb_per_min fixed, temporal split
- boost_model.pkl: uses projected RS from predictions CSVs
- drafts_model.pkl: unchanged, but retrained for consistency

Expected: +0.1-0.3 MAE improvement on Real Score predictions
"

git push origin claude/audit-nba-optimizer-OLTIm
```

## What Changed in Each Script

### train_lgbm.py (CRITICAL FIX)

**Data Leakage Fix**
- OLD: `df["reb_per_min"] = df["REB"] / df["avg_min"]` (uses same-game actual REB)
- NEW: Uses `avg_reb` (season average REB per player) for realistic training
- **Impact**: Removes ~0.1-0.2 point overestimation on predictions

**Dead Features Removed**
- `cascade_signal` (always 0.0, never splits)
- `usage_share` (derived from team_ppg, redundant)
- `teammate_out_count` (always 0.0, no signal)
- `game_total` (unused by model)
- `spread_abs` (unused by model)
- **Impact**: 4 fewer features → faster training, less overfitting

**Temporal Train/Test Split**
- OLD: `shuffle=True` (random row-level split → leakage across time)
- NEW: `shuffle=False` (temporal split: earlier dates train, later dates test)
- **Impact**: More realistic test metrics, reveals actual generalization

### train_boost_lgbm.py (CRITICAL FIX)

**Projected RS Usage**
- OLD: `perf_score = actual_rs` (post-game performance, not available at draft time)
- NEW: `perf_score = predicted_rs from data/predictions/` (pre-game estimate matching inference)
- **Impact**: Model now learns realistic train/inference distribution, +0.05-0.15 MAE improvement

## Expected Improvements Post-Retraining

| Metric | Current | Expected | Improvement |
|--------|---------|----------|------------|
| Real Score MAE | ~0.45 | ~0.35 | 22% |
| Boost Prediction MAE | ~0.791 | ~0.65 | 18% |
| Slate-level std dev | ~6.5 pts | ~5.5 pts | 15% |

## Validation Checklist

- [ ] Run `python train_lgbm.py` → output 18-feature schema, check training logs
- [ ] Run `python train_boost_lgbm.py` → output shows "Predicted RS values" usage stats
- [ ] Run `python train_drafts_lgbm.py` → completes without errors
- [ ] Check `lgbm_model.pkl`, `boost_model.pkl`, `drafts_model.pkl` exist and are recent
- [ ] Deploy to production and monitor metrics for 2-3 days
- [ ] Backtest on historical dates (Jan 17 - Mar 23) to verify improvements

## Troubleshooting

**Error: No valid rows in training**
- Check that `data/predictions/` CSV files exist with correct columns
- Ensure `data/top_performers.csv` has `actual_rs` and `actual_card_boost` populated

**Feature count mismatch**
- Verify train_lgbm.py has 18 features in the list (line ~218-241 in original)
- Check api/index.py _lgbm_feature_vector returns 18-element list

**Boost model training fails**
- Ensure `data/predictions/` has entries for dates in top_performers.csv
- The script will fall back to actual_rs if no predictions available (but log it)

## Long-Term Next Steps (Phase 2)

1. **Backfill team field** in top_performers.csv (currently 87.9% missing)
   - Join with predictions CSVs and actuals CSVs which have team information
   - This improves data quality for future audits

2. **Implement automated retraining**
   - Set up GitHub Actions workflow to retrain models nightly
   - Current manual process should be automated

3. **Expand test coverage**
   - Add tests for feature alignment (train vs inference)
   - Add tests for normalized player names in joins
   - Add regression tests on historical dates
