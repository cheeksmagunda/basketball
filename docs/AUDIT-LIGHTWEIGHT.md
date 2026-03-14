# Lightweight Production & Pipeline Audit

**Date:** 2026-03-08  
**Scope:** Production config, object/variable/reference (design + data), pipeline/caching, LightGBM.

---

## 1. Production

- **vercel.json:** Routes and crons correct. Single `/api/refresh` at 19 UTC; `/api/lab/auto-improve` 09 UTC; `/api/refresh-line-odds` at :55; `/api/auto-resolve-line` at 0,30. `ignoreCommand` excludes data-only commits. `maxDuration: 300` on API.
- **Env:** `GITHUB_TOKEN`, `GITHUB_REPO`, `ANTHROPIC_API_KEY`, `ODDS_API_KEY`, cron secret used for refresh/auto-improve. No hardcoded secrets.
- **Health/version:** `/api/health` and `/api/version` present for pre-warm and debugging.

**Finding:** None. No changes.

---

## 2. Object / variable / reference

- **Design tokens:** `:root` in `index.html` defines semantic tokens (`--chalk`, `--line`, `--lab`, `--color-success`, `--radius-pill`, etc.). Line card and player cards use `var(--*)`; some intentional one-off rgba() remain (loader, pills). No broken references.
- **Frontend globals:** `SLATE`, `PICKS_DATA`, `LOG`, `LAB`, `LINE_DIR`, `LINE_OVER_PICK`, `LINE_UNDER_PICK`, `LINE_LOADED_DATE`, `pending_upload_date` used consistently. Tab navigation and data flow documented in CLAUDE.md.
- **API contracts:** Line pick contract includes `season_avg`, `proj_min`, `avg_min`, `game_time`, `recent_form_bars`. Slate returns `date`, `games`, `lineups`, `locked`, `draftable_count`, `lock_time`.

**Finding:** None. No changes.

---

## 3. Pipeline / caching

- **Cache helpers:** `_cp(k, date_str=None)`, `_lp(k, date_str=None)` key by `(date_str or _et_date(), k)`. Slate and picks use default date (today ET); lock restore uses `yesterday` where needed. Line uses `_cg("line_v1")` for "current" pick; `_cp("line_v1", date_str)` used for invalidation and midnight rollover.
- **Refresh:** `/api/refresh` clears all `CACHE_DIR` and `LOCK_DIR` files, plus config cache and RotoWire cache. Auto-save runs before clear when slate is locked.
- **Midnight:** Slate endpoint holds yesterday’s locked slate when no today games started and yesterday games still in progress; uses `_lg("slate_v5_locked", yesterday)` and GitHub `data/locks/{yesterday}_slate.json`.

**Finding:** None. No changes.

---

## 4. LightGBM

- **Training:** `train_lgbm.py` uses `nba_api` playergamelogs for 2023-24, 2024-25, 2025-26. Target = `actual_base_score` (matches `_dfs_score()`). Model bundle = `{model, features}`. Retrain: GitHub Actions `retrain-model.yml` at 6 AM UTC; `bump_retrain_config.py` stamps `model_retrained_at` in `data/model-config.json`.
- **Inference:** `api/index.py` loads bundle; uses `AI_FEATURES` to verify feature vector length. Usage trend clipping (0.90, 1.50) matches training.
- **Data flow:** Model does **not** use `data/actuals` or `data/predictions`; by design it trains on historical box scores only. App actuals feed audit/briefing and future log-formula calibration, not the pkl.

**Finding (fix applied):** Feature **recent_3g_trend** semantics differed between train and inference.
- **Train (before):** `recent_3g_trend` = (recent_3g_pts / recent_5g_pts).clip(0.5, 2.0) — 3-game vs 5-game rolling.
- **Inference:** Used `recent_pts / season_pts` (recent vs season) as the 11th feature.
- **Fix:** Training now uses **recent_vs_season** = (recent_5g_pts / avg_pts).clip(0.5, 2.0) so both train and inference use "recent scoring vs season average." Feature list in bundle is `recent_vs_season`; inference variable renamed to `recent_vs_season_` and comment added. Existing deployed pkl with `recent_3g_trend` in the list still works (same length and order).

---

## 5. Unit tests

- **test_fixes.py:** SafeFloat, IsLocked, ComputeAudit, GitHub retry, SaveActuals audit gate, AutoResolve midnight, Cache TTLs, Polling intervals.
- **test_core.py:** Helpers, cache roundtrip, et_date, is_locked, card boost, JS syntax.

**Finding:** None. No changes.

---

## Summary

| Area        | Status   | Action                          |
|------------|----------|----------------------------------|
| Production | OK       | None                             |
| Objects    | OK       | None                             |
| Pipeline   | OK       | None                             |
| LightGBM   | Fixed    | Align recent_vs_season train/inference |
| Tests      | OK       | None                             |
