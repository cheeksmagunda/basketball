# Loading Audit — Frontend & UX

Audit of loading states, timeouts, skeletons, and error handling across the app. Date: March 2026.

## 1. Fetch timeouts (frontend)

All blocking API calls use `fetchWithTimeout(url, options, timeoutMs)`. Default when omitted: **10s**.

| Call site | Endpoint / purpose | Timeout | Notes |
|-----------|--------------------|---------|--------|
| Startup | `/api/health` | 5s | Pre-warm; fire-and-forget |
| Predict | `/api/games` | 10s | Game selector |
| Predict | `/api/slate` | 30s | Main slate load (cold can be slow) |
| Predict | `/api/picks?gameId=X` | 15s | Per-game analysis |
| Predict | `/api/save-predictions` | 10s default | POST |
| Predict | `/api/force-regenerate?scope=remaining` | 60s | Late draft |
| Log | `/api/log/dates` | 15s | Date strip |
| Log | `/api/log/get?date=X` | 10s default | Grid data |
| Log | `/api/log/actuals-stats?date=X` | 15s | ESPN box scores |
| Log drill-down | `/api/audit/get?date=X` | 10s | MAE, misses |
| Line | `/api/line-of-the-day` | 90s | Main LOTD (cold can be slow) |
| Line (Ben context) | `/api/line-of-the-day` | 30s | When building Lab context |
| Line | `/api/line-history` | 25s | Recent picks |
| Line | `/api/line-live-stat` | 10s default | Live in-game stat |
| Line | `/api/save-line` | 10s default | POST; fire-and-forget |
| Lab | `/api/lab/status` | 10s | Lock poll + init |
| Lab | `/api/lab/briefing` | 10s (action) / 30s (context) | Context load 30s |
| Lab | `/api/lab/config-history` | 10s | |
| Lab | `/api/lab/update-config` | 15s | POST |
| Lab | `/api/lab/backtest` | 120s | POST; long-running |
| Lab | `/api/lab/skip-uploads` | 10s default | POST |
| Ben upload | `/api/parse-screenshot` | 30s | Claude vision |
| Ben upload | `/api/save-actuals`, `/api/save-ownership` | 10s default | POST |
| Event (game final) | `/api/lab/status` | 10s | After line poll detects final |
| Event (rotation) | `/api/auto-resolve-line` | 15s | When used from frontend |

**Intentional non-timeout:** Lab chat uses a raw `fetch()` for the SSE stream, with a 60s connection timeout via `AbortController`; the stream body is not time-limited by design.

## 2. Loading UI by tab

### Predict
- **Slate load:** Magic 8-ball overlay (`showLoader` / `hideLoader`) + skeleton cards in slate list (5 cards) when `!SLATE`. Background refresh does not show loader.
- **Per-game picks:** 8-ball + skeleton list (5) in picks container; `hideLoader()` on success/error.
- **Save predictions:** No dedicated loader; inline feedback.
- **Late draft:** 8-ball during force-regenerate; hide on success/error.
- **Error:** `SLATE_STATE` / `PICKS_STATE` → error state; Retry available. Slate endpoint never returns 5xx (returns 200 with `error: "slate_failed"`).

### Line
- **LOTD:** `LINE_LOTD_STATE` (initial → loading → success | error). Loading shows `renderLinePickCardSkeleton()` in main card area; no 8-ball.
- **Background re-fetch:** Same-day revisit runs `fetchLineOfTheDay(true, true)` — no skeleton, no loading state; card updates when response arrives.
- **Live poll:** Card DOM updated only when live snapshot key changes (`_lineLastLiveKey`) to avoid flash every 60s.
- **History:** `LINE_HISTORY_STATE`; loading shows 3 skeleton rows in list.
- **Error:** Tap Retry or "Check for picks" on pending card.

### Log
- **Init:** `LOG_STATE`; dates + get in parallel. Loading: date strip visible; grid shows loading or stale data.
- **Date change:** `LOG_STATE = loading`; fetch `/api/log/get?date=X`; grid updates on success.
- **Drill-down:** ESPN actuals-stats and audit fetched on open; loading handled inside drill-down UI.
- **Error:** Error state; user can reselect date or retry.

### Lab (Ben)
- **Lock poll:** 2 min interval; no visible loader.
- **Unlock context load:** Briefing, config-history, line-of-the-day, slate, log in parallel; 30s timeout for briefing and line/slate. On failure, fallback messaging; no blocking loader.
- **Chat:** SSE with 60s connection timeout; "Request timed out. Please try again." on timeout.
- **Upload:** Button shows "Uploading..."; 30s for parse-screenshot. Success: button flashes green then hides; failure: stays for retry.
- **Backtest:** Inline spinner "Running backtest..."; 120s timeout.

## 3. Async state pattern

Used for: slate, picks, log, line LOTD, line history.

- **Shape:** `{ status: 'initial' | 'loading' | 'success' | 'error', data: T | null, error: Error | null [, loadedAt ] }`
- **Helpers:** `asyncStateInitial()`, `asyncStateLoading(prev)`, `asyncStateSuccess(prev, data)`, `asyncStateError(prev, err)`
- **Render:** Each tab checks state and shows skeleton/empty/error or content accordingly.

## 4. Skeleton usage

- **Predict slate list:** `renderSkeletons('slateList', 5)` — only on first load when `!SLATE`.
- **Predict picks list:** `renderSkeletons('picksList', 5)` when loading picks.
- **Line card:** `renderLinePickCardSkeleton()` — matches card layout (header, direction row, 5-col data row, conclusion).
- **Line history:** 3 inline skeleton rows (skel-block) when history loading.

## 5. Backend timeouts (reference)

- **GitHub API:** 10s or 15s in `_github_get_file` / `_github_write_file`.
- **ESPN:** 10s in game/scoreboard fetches.
- **Context layer (Layer 2):** `context_layer.timeout_seconds` (default 20).
- **Lineup review (Layer 3):** `lineup_review.timeout_seconds` (default 30).
- **Line engine (Claude):** 30s in `line_engine.py`.
- **RotoWire:** 15s in `rotowire.py`.

## 6. Gaps and recommendations

| Area | Status | Recommendation |
|------|--------|----------------|
| Line card flash | Fixed | Live poll updates card only when `_lineLastLiveKey` changes. |
| First load after hit | Fixed | Same-day Line tab does background `fetchLineOfTheDay(true, true)` so rotated pick appears without skeleton. |
| save-predictions timeout | 10s default | OK for GitHub write; no change. |
| Log get (no explicit timeout) | 10s default | OK; consider 15s if log payload grows. |
| Lab chat stream | No body timeout | By design; 60s connection timeout only. |
| Retry surfaces | Present | Predict (slate/picks), Line (Retry / Check for picks), Log (date reselect), Lab (chat retry, upload retry). |

## 7. Cache and polling (loading impact)

- **Slate:** First request of the day can take 30s+ (full pipeline); subsequent from cache &lt; 1s. 30s timeout is appropriate.
- **Line-of-the-day:** Cold generation can be slow; 90s timeout. Background re-fetch on same-day avoids showing loader again.
- **Line live poll:** 60s interval; card only re-renders when live data changed (no flash).
- **Lab lock:** 2 min; no loader.

---

**Summary:** Loading is consistent across tabs (async state + skeletons or 8-ball where appropriate). All blocking fetches use `fetchWithTimeout`. Line tab fixes (no flash, first-load-after-hit background re-fetch) are in place. No critical gaps for production.
