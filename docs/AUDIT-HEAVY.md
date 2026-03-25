# Heavier Production Audit

**Document Status:** Historical Snapshot

**Date:** 2026-03-08  
**Scope:** Security, error handling, API consistency, frontend/backend contracts, timeouts, deployment, observability.
**Note:** This is a historical snapshot. For current deployment/runtime behavior, use `railway.toml`, `README.md`, and `CLAUDE.md`.

---

## 1. Security

- **Secrets:** No secrets in repo. `GITHUB_TOKEN`, `GITHUB_REPO`, `ANTHROPIC_API_KEY`, `ODDS_API_KEY`, `CRON_SECRET`, `DOCS_SECRET` from env. Startup warns on missing required vars; app degrades gracefully.
- **Cron protection:** `CRON_SECRET` gates cron-only endpoints (e.g. `/api/lab/auto-improve`, `/api/auto-resolve-line`, `/api/injury-check`, `/api/mae-drift-check`, `/api/force-regenerate?scope=full`) when set.
- **Docs protection:** Optional `DOCS_SECRET` gates `/docs`, `/redoc`, `/openapi.json` via middleware.
- **Rate limiting:** Thread-safe `_check_rate_limit()` applied to `parse-screenshot`, `line-of-the-day`, `lab/chat`. Limits and lock documented in code.
- **User input:** Screenshot upload type is client-side only (noted in CLAUDE.md as known limitation). No SQL/NoSQL; GitHub API and file paths are controlled.

**Finding:** No critical issues. No changes.

---

## 2. Error handling

- **Backend:** JSONResponse with status_code used for errors. Cron endpoints return 401 when unauthorized. Rate limit returns 429. Save-actuals returns 4xx on skip/validation. Exceptions in projection/lock logic fall back to safe defaults (e.g. locked=True on ESPN failure).
- **Frontend:** All critical fetches use `fetchWithTimeout`; responses checked with `r.ok` before `.json()`. Errors surface via throw or UI message. Save-predictions resets `_predSavedDate` on non-OK for retry.

**Finding:** No critical issues. No changes.

---

## 3. API consistency

- **REST:** GET for reads (slate, picks, games, log, audit, line, lab status/briefing/config). POST for mutations (save-predictions, save-actuals, save-line, parse-screenshot, lab/chat, lab/update-config, etc.). Query params for date/filters where appropriate.
- **Contracts:** Slate has `date`, `games`, `lineups`, `locked`, `draftable_count`, `lock_time`. Line pick contract includes normalized fields per `_LINE_PICK_CONTRACT_FIELDS`. Log/audit use date-scoped CSV/JSON.

**Finding:** No critical issues. No changes.

---

## 4. Frontend/backend contracts

- **Design/data:** Frontend globals (SLATE, PICKS_DATA, LOG, LAB, LINE_*) align with API responses. Line card uses backend fields (season_avg, proj_min, avg_min, game_time, recent_form_bars). Stat keys standardized (e.g. points vs pts) in _STAT_MARKET.

**Finding:** No critical issues. No changes.

---

## 5. Timeouts and external calls

- **Backend:** GitHub API 10–15s; ESPN 10s; Odds API 10s; Anthropic 30–45s for Lab/vision.
- **Frontend:** fetchWithTimeout 10s default; 15s for picks; 30s for screenshot parse; Lab chat uses streaming (no body timeout by design).

**Finding:** No critical issues. No changes.

---

## 6. Deployment and observability

- **Railway:** `watchPatterns` in `railway.toml` exclude data-only commits from rebuilds. Crons and health checks are configured in `railway.toml`.
- **Logging:** Request middleware logs request_id, path, method, status, duration_ms (NDJSON). No PII in logs. Print used for warnings (e.g. LightGBM fallback, health config error).

**Finding:** No critical issues. No changes.

---

## 7. Tests and docs

- **Tests:** test_fixes.py (SafeFloat, IsLocked, ComputeAudit, GitHub retry, SaveActuals gate, AutoResolve midnight, cache TTLs, polling, rate limit, Line engine fallback, LgbmFeatureAlignment). test_core.py (helpers, cache, line cache logic, config, project_player, normalize, parse_csv, etc.). Run: `pytest tests/ -v`.
- **Docs:** CLAUDE.md and README.md describe architecture, endpoints, crons, lock system, LightGBM, and data layer. Lightweight and heavy audit docs in docs/.

**Finding:** No critical issues. No changes.

---

## Summary

| Area           | Status | Action |
|----------------|--------|--------|
| Security       | OK     | None   |
| Error handling | OK     | None   |
| API consistency| OK     | None   |
| Contracts      | OK     | None   |
| Timeouts       | OK     | None   |
| Deployment     | OK     | None   |
| Observability  | OK     | None   |
| Tests/docs     | OK     | None   |

No code changes required from this audit. Lightweight audit fix (LightGBM recent_vs_season) already applied.
