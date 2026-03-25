# Parlay Synchronization & Data Integrity Fixes

**Document Status:** Historical Snapshot

## Summary of Issues Fixed (Session: March 24, 2026)

### 1. ✅ Parlay Lock Synchronization
**Problem**: Parlay ticket didn't follow global app slate logic. Locked/unlocked states weren't synchronized with Predict, Line, and Lab tabs.

**Solution**:
- Added lock polling to Parlay page (`_startParlayLockPoll()`) that monitors `/api/lab/status` every 2 minutes when slate is locked
- When games complete and slate unlocks, parlay data automatically re-fetches to resolve pending parlays
- Lock polling cleanup on tab switch prevents memory leaks
- Parlay endpoint already uses same `_is_locked()` lock logic as app (5-min pre-lock, 6-hour ceiling)

**Result**: Parlay now rolls over with global slate. Lock badge displays correctly when active.

---

### 2. ✅ Parlay "Ghosting" Issue (Christian Braun)
**Problem**: Resolved parlays (with MISS status) appeared in BOTH current ticket section AND Recent Parlays history, causing confusion. Christian Braun shown as both active and resolved.

**Solution**:
- Modified `renderParlayHistory()` to filter and show ONLY resolved parlays (result === "hit" || "miss")
- Pending parlays are now excluded from history display
- Current ticket always shows today's parlay in pending state
- Once a parlay resolves, it moves exclusively to Recent Parlays section

**Result**: Clean separation between current (pending) and historical (resolved) tickets.

---

### 3. ✅ Float Preservation in Line Values (6.5, 21.5, 8.5)
**Problem**: Line values like 6.5 (assists) potentially truncated to 6.0; 21.5 (points) to 21.0 in JSON serialization or JavaScript operations.

**Solution**:
- Explicit `float()` wrappers at three critical points:
  1. **Parlay engine** (`api/parlay_engine.py:320`): Round and preserve as float
  2. **Synthetic fallback** (`api/index.py:10055`): Explicit float on snapped lines
  3. **Leg normalization** (`api/index.py:10202`): Float conversion before JSON serialization
- Frontend already correct: `Number(leg.line).toFixed(1)` → "6.5" displays correctly

**Verification**: Line values stored as floats in JSON (`"line": 8.5` not `"line": 8`)

---

## OddsAPI Integration Status

### Current Flow
1. **Parlay endpoint** calls `_build_player_odds_map(target_games)` to fetch Odds API props
2. If odds map populated: use real sportsbook lines → `projection_only: false`, `SOURCE: BOOK`
3. If odds map empty (key missing, API fails, no props published): synthetic fallback → `projection_only: true`, `SOURCE: MODEL`

### Diagnostic Logging Added
```python
[parlay] projections={N} odds_entries={M} games={K}
[parlay] no Odds API data — built {N} synthetic lines from projections (ODDS_API_KEY={'set'|'not set'})
```

This logs:
- How many odds entries were successfully fetched
- Whether ODDS_API_KEY is configured
- When fallback to synthetic lines occurs

### How to Use OddsAPI (Required Setup)
1. **Set environment variable** on Railway:
   ```
   ODDS_API_KEY=<your-key>
   ```
2. **Verify in logs**: Look for `[odds_map] fetched N player+stat lines from Odds API`
3. **Check parlay response**: Should show `projection_only: false` and `SOURCE: BOOK` badge
4. **Line values**: Booker 6.5 (not 6.0), Gillespie 21.5 (not 21.0), Braun 8.5 (not 8.0)

### Why Synthetic Fallback Occurs
```python
if not player_odds_map and all_proj:
    projection_only = True
    # Build synthetic lines: round(projection * 2) / 2 → nearest 0.5
```

**Common reasons**:
- `ODDS_API_KEY` not set in Railway environment
- OddsAPI request timed out or returned error
- Game is past lock window (props no longer published)
- Player names don't match between ESPN roster and OddsAPI database

---

## Testing Checklist

- [ ] **Lock Sync**: Open Parlay tab while slate locked. Close app. Wait for games to finish. Re-open. Verify parlay re-fetched and shows resolved status.
- [ ] **Ghosting Fix**: If Christian Braun was resolved, verify he NO LONGER appears in current ticket. Should only appear in Recent Parlays with MISS status.
- [ ] **Float Values**: Check parlay JSON and frontend display. 6.5, 21.5, 8.5 should appear as decimals, not whole numbers.
- [ ] **OddsAPI**: Verify `ODDS_API_KEY` set in Railway. Check logs for `[odds_map]` and `[parlay]` lines. Confirm `projection_only: false` in response.
- [ ] **Lock Badge**: When slate locked, "LOCKED" badge appears on ticket. Disappears when games finish.

---

## Code Changes Summary

### Frontend (`index.html`)
- **Line polling**: `_startParlayLockPoll()` monitors lock status, refreshes on unlock
- **History filtering**: `renderParlayHistory()` shows only resolved parlays
- **Tab cleanup**: `switchTab()` stops lock polling when leaving Parlay tab

### Backend (`api/index.py`)
- **Float preservation**: Line values explicit float in leg normalization
- **Diagnostic logging**: ODDS_API_KEY status logged on synthetic fallback
- **Lock status**: Already correctly set from `_is_locked()` helper

### Parlay Engine (`api/parlay_engine.py`)
- **Float handling**: book_line and leg line values explicit float()
- **Line snapping**: Correct rounding to nearest 0.5 for both real and synthetic lines

---

## Known Limitations

1. **Midnight Rollover**: If parlay resolves after midnight ET, the lock poll may take up to 2 minutes to detect.
   - Workaround: User can manually refresh or switch tabs to trigger immediate check.

2. **OddsAPI Availability**: Props not published until ~1-2 hours before game tip.
   - If parlay generated >4 hours before games, may show synthetic lines initially.
   - Solution: Force-regenerate parlay closer to tip time when odds available.

3. **Player Name Matching**: OddsAPI player names must match ESPN roster (case-insensitive).
   - If a player not in Odds API: filtered out, can't be included in parlay.
   - Fallback to synthetic uses projection, not OddsAPI data.

---

## Next Steps (Optional Enhancements)

1. **Real-time Odds Refresh**: Add button to "Refresh Odds" mid-slate to pull latest lines.
2. **Parlay Leg Control**: Allow user to select specific directions (over/under) for each leg.
3. **Parlay History Download**: Export past parlays as CSV for tracking vs actual results.
