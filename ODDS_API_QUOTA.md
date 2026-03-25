# OddsAPI Quota Management (500 requests/day)

**Document Status:** Current Reference

## Overview
With a 500 request/day plan, we need to be efficient to avoid overage charges.

**Cost per full slate (14 games):**
- 1 events fetch
- 14 props fetches (1 per game)
- **Total: 15 API requests**

**Daily budget:**
- 500 ÷ 15 = **~33 full slate generations** before quota exhaustion

## Quota Conservation Strategies

### 1. ✅ Aggressive Caching (30 minutes)
**Setting:** `_TTL_ODDS_FRESH = 1800` (30 min cache)

**Effect:**
- All endpoints sharing odds within 30-min window reuse same fetch
- Draft pipeline (odds_enrichment + fair_value_prefetch) both hit cache
- Line page reuses draft pipeline's odds
- Parlay reuses draft pipeline's odds
- **Typical scenario:** 1 fetch per 30-min window, not per request

**Estimate:**
- Morning slate gen: 15 calls
- 30-min window reuse: 0 additional calls
- Noon parlay request: 0 calls (within cache window)
- Line page follow-up: 0 calls
- **Per slate:** ~15 calls (not 60+)

### 2. ✅ Slate Lock Guard
**Setting:** `/api/refresh-line-odds` checks lock status before fetching

**Effect:**
- During locked periods (5 min pre-tip to game final), props aren't changing
- Skip expensive odds fetch when slate is locked
- Save ~1-2 calls per hour during game windows

**Estimate:**
- 8-hour game window × hourly crons = 8 calls saved per day

### 3. ⚠️ Offline Fallback (Graceful Degradation)
**Behavior:**
- If ODDS_API_KEY invalid or quota exhausted: fall back to model-only lines
- Parlay generates with `projection_only: true` (synthetic lines)
- Line page shows `MODEL` badge instead of `BOOK`
- Draft pipeline enriches with projections instead of Vegas odds

**Not a solution, but prevents app breakage when quota exhausted**

## Daily Usage Pattern

```
Morning (7-8am ET):
  /api/slate              → 15 calls (all games unlocked, fresh generation)
  odds_enrichment         → cache hit (same instance, <1 min apart)
  fair_value_prefetch     → cache hit
  /api/line-of-the-day    → cache hit
  /api/parlay             → cache hit
  Subtotal: 15 calls

Mid-morning (8-9am):
  User visits Line tab     → cache hit (within 30 min)
  User visits Parlay tab   → cache hit
  Subtotal: 0 calls

Game window (1-11pm ET):
  /api/refresh-line-odds crons (8× hourly) → SKIPPED (slate locked)
  Subtotal: 0 calls (quota conservation)

Late night (next day):
  /api/auto-resolve-line  → cache expired, may refetch (0-15 calls depending on games)

Daily total: 15-30 calls (well under 500)
```

## Monitoring

**Check daily quota usage:**
```python
from pathlib import Path
import json
from datetime import date

cache_path = Path("/tmp/nba_cache_v19") / f"odds_quota_{date.today().isoformat()}.json"
if cache_path.exists():
    data = json.loads(cache_path.read_text())
    print(f"Estimated calls today: {data.get('calls', 0)}/500")
```

Or check Railway logs for:
```
[odds_map] estimated quota usage: +15 calls (total today: 15/500)
```

## Risks & Mitigation

| Risk | Mitigation | Status |
|------|-----------|--------|
| Parallel requests (multiple users) each triggering fresh fetch | 30-min cache deduplicates within single instance; cross-instance reuse via GitHub fallback | ✅ Implemented |
| Line cron refreshing every hour even when slate locked | Skip fetch during lock window (no prop changes anyway) | ✅ Implemented |
| Force-regenerate mid-slate burning quota | Only user-facing (`scope=remaining`), not cron; acceptable cost | ⚠️ Acceptable |
| Quota exhaustion mid-day | Fall back to projection-only lines; app doesn't break | ✅ Graceful fallback |
| Key rotation losing quota mid-month | Document new key, update Railway env var immediately | 📋 Manual process |

## What NOT to Do

❌ Don't call `_build_player_odds_map` independently (use cache via existing endpoints)
❌ Don't refresh-line-odds during game windows (locked period = props frozen)
❌ Don't force-regenerate multiple times in quick succession (each costs 15+ calls)
❌ Don't generate forecast/test parlays in dev without checking quota key

## If Quota Exhausted

1. Switch to fallback key (if available) or disable OddsAPI temporarily
2. All endpoints gracefully degrade to model-only mode
3. Parlay generates with synthetic lines (projection_only: true)
4. Line page shows MODEL badge instead of BOOK
5. Wait for quota reset or upgrade plan

## Future Optimizations

1. **Shared Redis cache** across Railway instances (currently 30-min TTL per instance)
2. **Hourly aggregate fetch** instead of per-request (8 calls total, vs 15 per slate)
3. **Upgrade to higher tier** if parlay becomes primary product
