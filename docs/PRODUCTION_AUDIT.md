# Production Audit — The Oracle (Basketball)

**Audit date:** 2026-03-08 (updated 2026-03-21)
**Scope:** Security, reliability, performance, deployment, observability.

---

## 1. Security

### 1.1 POST/PUT endpoints and user-controlled payloads

| Endpoint | Method | Payload / input | Writes to GitHub / external |
|----------|--------|------------------|-----------------------------|
| `/api/save-predictions` | POST | (none) | `data/predictions/{date}.csv`; slate backup |
| `/api/reset-uploads` | POST | `body.date` | Deletes `data/actuals/{date}.csv`, `data/audit/{date}.json` |
| `/api/parse-screenshot` | POST | `UploadFile` (image) | None (Claude API only) |
| `/api/save-actuals` | POST | `date`, `players[]` | `data/actuals/{date}.csv`, `data/audit/{date}.json` |
| `/api/hindsight` | POST | `players[]` | None |
| `/api/save-line` | POST | `pick`, `over_pick`, `under_pick` | `data/lines/{date}.csv`, `data/lines/{date}_pick.json` |
| `/api/resolve-line` | POST | `date`, `actual_stat` | `data/lines/{date}_pick.json`, CSV row |
| `/api/lab/update-config` | POST | `changes`, `change_description` | `data/model-config.json` |
| `/api/lab/rollback` | POST | `target_version` | `data/model-config.json` |
| `/api/lab/backtest` | POST | `proposed_changes`, `description` | None |
| `/api/lab/chat` | POST | `messages`, `system` | None (Anthropic API) |
| `/api/lab/skip-uploads` | POST | `date` | `data/skipped-uploads.json` |
| `/api/save-boosts` | POST | `date`, `players[]` | `data/boosts/{date}.json` |
| `/api/save-ownership` | POST | `date`, `players[]` | `data/ownership/{date}.csv` |

**Input validation:** Lab config keys are restricted to alphanumeric dot-paths (regex in `lab_update_config`). Ben system prompt restricts file writes to `data/model-config.json`. Rate limiting on `parse-screenshot` (5/min), `lab/chat` (20/min), `line-of-the-day` (10/min) via thread-safe `_check_rate_limit()`.

### 1.2 Secrets and sensitive data

- **Confirmed:** `GITHUB_TOKEN`, `GITHUB_REPO`, `ANTHROPIC_API_KEY`, `ODDS_API_KEY` are never echoed in JSON responses. They are used only in server-side requests.
- **Finding (addressed):** GitHub write failures now return generic "GitHub write failed" to the client and log full detail server-side only.
- **Startup:** Missing env vars are logged with names only (`[WARN] Missing env vars: [...]`), not values. Safe.

### 1.3 Security recommendations

1. **Cron authentication (implemented):** `CRON_SECRET` env is checked on `/api/refresh`, `/api/auto-resolve-line`, and `/api/lab/auto-improve`. Vercel sends it as `Authorization: Bearer <CRON_SECRET>`. `/api/refresh-line-odds` is not protected (also called by Line tab Refresh button).
2. **Request size limits:** Enforce max body size for `parse_screenshot` (e.g. 10MB already documented) and `lab_chat` (e.g. 100KB for JSON body) to avoid abuse.
3. **Keep in Known Limitations:** Upload screenshot type (Real Scores vs Top Drafts etc.) remains client-trust only unless server-side content checks are added.

---

## 2. Reliability

### 2.1 Frontend fetch timeouts

| Call site | Endpoint | Timeout | Notes |
|-----------|----------|---------|--------|
| loadSlate | `/api/slate` | 10s | OK |
| loadGames (initGameSelector) | `/api/games` | 10s | OK |
| runAnalysis | `/api/picks` | 15s | OK |
| savePredictions | `/api/save-predictions` | (default 10s) | OK |
| log dates/get | `/api/log/dates`, `/api/log/get` | 10s | OK |
| Ben skip-uploads | `/api/lab/skip-uploads` | default | OK |
| parse-screenshot | `/api/parse-screenshot` | 30s | OK |
| save-actuals, audit/get | `/api/save-actuals`, `/api/audit/get` | 10s | OK |
| initLinePage | `/api/auto-resolve-line` (fire-and-forget) | **none** | Background; no timeout by design. |
| initLinePage | `/api/line-of-the-day` | 10s | OK |
| Line refresh button | `/api/refresh-line-odds` | 10s | OK |
| save-line | `/api/save-line` | 10s | Fire-and-forget with console.warn on failure |
| line-history | `/api/line-history` | 25s | OK |
| log/get (all 5 sites) | `/api/log/get?date=X` | 15s | Explicit timeout (was 10s default) |
| lab/status, lab/briefing, config-history, slate, log | Various | 10s-30s | OK |
| **lab/chat** | `/api/lab/chat` | **60s AbortController** | Raw `fetch`; SSE streaming by design. |
| lab/backtest | `/api/lab/backtest` | 120s | fetchWithTimeout |
| lab/update-config | `/api/lab/update-config` | 15s | fetchWithTimeout |

**All fetches accounted for.** `lab/chat` is the only raw `fetch()` — intentional for SSE streaming; 60s connection timeout via manual AbortController.

### 2.2 Request lock and write guards

- **isFetching:** Guards `runAnalysis()` and `initLinePage()`; Analyze and Refresh buttons disabled while true. Prevents concurrent slate/line fetches and duplicate save-line calls.
- **save-predictions:** Guarded by `any(_is_locked(st))`; returns 409 if slate not locked. Confirmed.
- **save-actuals:** No lock guard (intentional: uploads after games final). Skip-uploads check and payload validation only.
- **save-line:** No lock required; idempotent merge to CSV/JSON.
- **lab/update-config, rollback, skip-uploads:** No lock; rely on GitHub write retries.

### 2.3 Reliability recommendation

- **Implemented:** `GET /api/health` returns 200 with `config` and `github` status for uptime monitoring.

---

## 3. Performance

### 3.1 Line-of-the-day cache usage

- **Confirmed:** `_run_line_engine_for_date()` uses `_cg(f"game_proj_{g['gameId']}")` for each draftable game and only runs `_run_game` when projections are missing. Slate computation populates `game_proj_*` when `/api/slate` or `/api/picks` run first, so a warm instance reuses them for `/api/line-of-the-day`. Cold start may recompute all games for line engine.
- **Cache key:** `line_v1` uses default `_et_date()` unless date_str is passed; line cache busting uses appropriate date (today_str, pick_date, date_str) in relevant endpoints.

### 3.2 Performance recommendations

1. **Warm-up:** If cold start latency is an issue, add a cron (e.g. 30 min before first tip) that hits `GET /api/slate` to warm cache and avoid first-user cold hit.
2. **Profiling:** For 10–14 game slates, measure cold vs warm `/api/slate` and `/api/picks?gameId=X` via Vercel logs or a one-off script; confirm RotoWire/ESPN are not unnecessarily on the critical path after cache is warm.

---

## 4. Deployment and infrastructure

### 4.1 Verdict

- **watchPatterns:** `railway.toml` watchPatterns excludes `data/` and `.github/` — only code changes trigger Docker rebuilds.
- **Crons (current, Railway):**
  - `/api/refresh` at 19:00 UTC — daily cache clear + auto-save.
  - `/api/lab/auto-improve` at 09:00 UTC — daily auto-tune.
  - `/api/refresh-line-odds` at :55 each hour — hourly odds sync.
  - `/api/auto-resolve-line` at :00 each hour — resolve line picks.
  - `/api/injury-check` at 14,16,18,20,22,0 UTC — RotoWire checks.
  - `/api/mae-drift-check` at 06:00 UTC Monday — weekly MAE drift.
- **Health check:** `/api/health` with 120s timeout in railway.toml.
- **Cron secret (implemented):** `CRON_SECRET` env var; when set, cron-only endpoints require `Authorization: Bearer <CRON_SECRET>`.

### 4.2 Deployment recommendations

1. **Cron secret:** Railway injects `CRON_SECRET` via cron commands. Protected endpoints: `/api/auto-resolve-line`, `/api/lab/auto-improve`, `/api/injury-check`, `/api/mae-drift-check`, `/api/force-regenerate?scope=full`.
2. **Docs:** `GITHUB_TOKEN`, `GITHUB_REPO`, `ANTHROPIC_API_KEY` required; `ODDS_API_KEY`, `CRON_SECRET`, `DOCS_SECRET` optional.
3. **Version:** `GET /api/version` returns `RAILWAY_GIT_COMMIT_SHA` for deploy checks.

---

## 5. Observability

### 5.1 print() map

| File | Line(s) | Classification | Content |
|------|---------|----------------|---------|
| api/index.py | 45 | Startup | Missing env var names (no values) |
| api/index.py | 150, 162 | Rare | Slate backup/restore errors (exception only) |
| api/index.py | 269, 277 | Config | Config cache/GitHub load failure (exception only) |
| api/index.py | 311 | Startup | LightGBM bundle missing/invalid |
| api/index.py | 617 | Per-request | Stat parse error (pid, exception) |
| api/index.py | 1222 | Per-request | LightGBM inference fallback (exception) |
| api/index.py | 1369 | Per-request | Player fetch error (name, exception) |
| api/index.py | 1482 | RotoWire | Fetch failed (exception) |
| api/index.py | 1819, 1827, 1829 | Slate lock | Lineup save warnings / errors |
| api/index.py | 1859, 1897, 2000 | Slate/picks | Slate or game pick errors (gameId/exception) |
| api/index.py | 2194 | save-actuals | Skip upload (date only) |
| api/index.py | 2398 | refresh | Auto-save skipped (exception) |
| api/index.py | 2588 | Line engine | Line proj err (exception) |
| api/index.py | 2921 | resolve-line | JSON update err (exception) |
| api/index.py | 3103 | auto-resolve | Next-day generation err (exception) |
| api/index.py | 3323 | ESPN fallback | 4.5h fallback (hours_since_start) |
| api/index.py | 4096 | skip-uploads | Error recording skip (date, exception) |
| api/rotowire.py | 123, 138 | RotoWire | Fetch/cache error (exception) |
| api/line_engine.py | 182, 225 | LineEngine | Claude API / parallel call error (stat, direction, exception) |
| api/line_engine.py | 219 | LineEngine | Fallback pick (stat, direction, conf, player_name) |
| api/line_engine.py | 342 | LineEngine | Claude no picks — algorithmic fallback |

**PII/secrets:** No tokens or repo paths printed. Player names and game IDs can appear in error/fallback logs (e.g. `player_name`, `gid`). Acceptable for server logs.

### 5.2 Observability recommendations

1. **Health endpoint:** Same as reliability — `GET /api/health` for monitoring.
2. **Structured logging (optional):** For critical paths (e.g. save-predictions, save-actuals, lab/update-config), log one JSON line per request with `level`, `endpoint`, `duration_ms`, `error` (if any) to improve log search in Vercel.
3. **Error tracking (optional):** If incident response is needed, add Sentry (or similar) with env-based DSN; capture only non-2xx or explicit catch blocks to avoid noise. Do not log request bodies that may contain PII.

---

## 6. Summary

| Area | Status | Top action |
|------|--------|------------|
| Security | No auth; secrets not echoed; GitHub error body can leak to client | Sanitize GitHub error in responses; add cron secret |
| Reliability | Timeouts and guards mostly in place; 3 fetches without timeout | Add fetchWithTimeout for lab/backtest and lab/update-config; add /api/health |
| Performance | Line engine uses game_proj cache when warm | Optional warm-up cron; profile cold vs warm |
| Deployment | Crons and ignoreCommand correct | Add cron secret; document Pro plan |
| Observability | print() only; no PII/secrets in logs | Add /api/health; optional structured logging and error tracking |

This audit is documentation only. Implementations (health endpoint, cron secret, timeout wrappers, error sanitization) are separate follow-up tasks.

**Doc sync (2026-03-08):** CLAUDE and LOCK_AND_ROUTING_AUDIT updated for TTL (60s when locked), cron schedules, lab poll 120s; History 60-day wording aligned with buildLogDateStrip; app init and tab data flow documented.
