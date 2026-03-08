# Lock Infrastructure & Routing Audit

**Date:** 2026-03-08

## 1. Lock infrastructure audit

### 1.1 Backend — definitions and cache

| Item | Location | Purpose |
|------|----------|---------|
| `_is_locked(start_time_iso)` | api/index.py ~391 | True if within lock_buffer of start or &lt;6h past start. Catches exceptions → False. |
| `_is_completed(start_time_iso)` | api/index.py ~406 | True if past lock window (lock or in progress). Used for draftable. |
| `_all_games_final(games)` | api/index.py ~3352 | ESPN scoreboard check; 60s TTL when locked, 180s pre-slate. Returns (all_final, remaining, finals, latest). |
| `LOCK_DIR` / `_lg` / _ls | api/index.py ~221, 348, 388 | In-memory lock cache; `_lp(k)` file path, `_lg` read, `_ls` write. |
| GitHub lock backup | api/index.py ~170–189 | `data/locks/{date}_slate.json` read/write for cold-start recovery. |

### 1.2 Backend — where lock is used

| Endpoint / flow | Guard / behavior |
|------------------|------------------|
| `/api/slate` | `any(_is_locked(st))` for full slate; midnight rollover uses yesterday cache/backup; promotes to lock cache and writes GitHub backup at lock time. |
| `/api/picks` | Per-game `_is_locked(start_time)`; serves lock cache when locked. |
| `/api/save-predictions` | 409 if `_start_times and not any(_is_locked(st))`. |
| `/api/refresh` (cron) | Calls save_predictions only when `any(_is_locked(st))`. |
| `/api/lab/status` | Uses `fetch_games()`, `any(_is_locked(st))`, `_all_games_final(games)`; returns locked/unlocked + reason. |
| `/api/refresh-line-odds` | No-op when `_is_locked(earliest)` (earliest game only). |

### 1.3 Frontend — lock usage

| Area | Behavior |
|------|----------|
| Predict tab | `SLATE.locked`, `SLATE.all_complete`; clientLocked fallback (5 min before first game); "Picks Locked" chip; savePredictions only when `SLATE.locked`. |
| Game analysis | `PICKS_DATA.locked`; badge "Picks locked — showing final predictions". |
| Ben tab | `initLabPage()` calls `/api/lab/status`; on failure shows "Unable to reach server — defaulting to locked" + Retry. Locked view hides upload banner; polling every 120s when locked. |
| Line tab | Odds refresh disabled when slate locked (backend). |

### 1.4 Issues found (and fixed)

1. **lab/status unhandled exceptions** — If `fetch_games()`, `_all_games_final()`, or `_load_config()` throw, the endpoint returns 500 and the frontend shows "Unable to reach server". **Fix:** Wrap handler in try/except; on exception return 200 with `locked: true` and reason "Server temporarily unavailable — try again".
2. **ESPN-down lock fallback** — When checking GitHub lock file, code used `if lock_data:` but `_github_get_file` returns `(content, sha)`; the tuple is always truthy. **Fix:** Use `lock_content, _ = _github_get_file(...)` and `if lock_content:`.

---

## 2. Routing audit

### 2.1 Vercel (vercel.json)

| Route | Dest | Notes |
|-------|------|--------|
| `/api/(.*)` | `/api/index.py` | All API traffic to FastAPI. |
| `/(.*)` | `/index.html` | SPA fallback; no server-side routes. |

Builds: `api/index.py` (Python, maxDuration 300s), `index.html` (static).  
Crons: refresh (19:00 UTC only), auto-improve (09:00), refresh-line-odds (hourly at :55), auto-resolve-line (0, 30 min).

### 2.2 FastAPI routes (api/index.py)

All under `/api/`. No path overlap; order does not affect matching. Health/version first; then games, slate, picks; then save/parse/save-actuals; log, audit, hindsight; refresh; line-of-the-day, save-line, refresh-line-odds, line-live-stat, resolve-line, auto-resolve-line, line-history; lab/status, lab/briefing, lab/update-config, lab/config-history, lab/rollback, lab/backtest, lab/auto-improve, lab/chat, lab/skip-uploads.

### 2.3 Frontend

Single-page app; tab state via `switchTab()` (predict, line, log, lab). No hash or path-based routing; no 404 route.

### 2.4 Issues found

None. Routing is consistent; API prefix and SPA catch-all are correct.

---

## 3. Fixes applied

- **api/index.py**
  - `lab_status()`: Wrapped in try/except; on exception return 200 with `locked: true`, `reason: "Server temporarily unavailable — try again"`.
  - ESPN-down branch: use `lock_content, _ = _github_get_file(...)` and `if lock_content:` instead of `if lock_data:`.
