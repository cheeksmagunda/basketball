# Real Sports Data Ingestion Automation

Automated Playwright-based pipeline that logs into your Real Sports Pro account,
navigates to each leaderboard screen, extracts structured data, and publishes
it to the basketball backend — replacing the manual screenshot → parse-screenshot → curl workflow.

---

## Architecture

```
scripts/real_ingest/
├── runner.py          — CLI entrypoint
├── session.py         — Playwright login + session persistence
├── discover.py        — Reconnaissance: map app screens → datasets
├── schemas.py         — TypedDicts + validation for all 5 dataset types
├── publish.py         — POST to existing backend API endpoints
├── verify.py          — Post-write verification
└── extractors/
    ├── base.py            — Shared nav + DOM + intercept utilities
    ├── most_popular.py    — Most Drafted leaderboard
    ├── most_drafted_3x.py — High-boost (3x+) sub-leaderboard
    ├── actuals.py         — Top Performers / Real Scores
    ├── winning_drafts.py  — Winning lineups (up to 4)
    └── boosts.py          — Pre-game player boost list
```

---

## Setup (one-time)

```bash
# 1. Install ingestion dependencies (separate from Railway requirements)
pip install -r requirements-ingest.txt
playwright install chromium

# 2. Copy and fill in credentials
cp .env.example .env
# Edit .env:
#   REAL_SPORTS_USERNAME=your_username
#   REAL_SPORTS_PASSWORD=your_password
#   BASKETBALL_API_BASE=http://localhost:8000  (or Railway URL)
#   INGEST_SECRET=...  (if set on your Railway instance)

# 3. Start the local backend (for local publishing)
uvicorn server:app --reload
```

---

## Step 1: Discovery (run once)

The discovery script logs in, clicks every visible nav element, and saves:
- Screenshots of each screen
- DOM snapshots
- Network request log (all XHR/fetch calls)
- `flow_map.json` — candidate screen → dataset mappings

```bash
cd /home/user/basketball
python -m scripts.real_ingest.discover
# Opens a browser window — let it run, don't interfere
```

Output: `scripts/real_ingest/discovery_report/`

### After discovery:

1. Open `discovery_report/screenshots/` and look at each PNG
2. Open `discovery_report/flow_map.json`
3. For each screen, set:
   ```json
   "confirmed_dataset": "most_popular",   // or: most_drafted_3x | actuals | winning_drafts | boosts | null
   "nav_selectors": [
     {"type": "click_text", "value": "Leaderboard"},
     {"type": "click_text", "value": "Most Drafted"}
   ]
   ```
4. Save `flow_map.json`

### Nav selector types:
| type | value | effect |
|------|-------|--------|
| `click_text` | `"Leaderboard"` | `page.get_by_text("Leaderboard").click()` |
| `click_css` | `".nav-item[data-tab='leaderboard']"` | `page.locator(css).click()` |
| `goto` | `"https://www.realsports.io/leaderboard"` | Direct URL navigation |

---

## Daily Workflow (post-game)

Run after NBA games complete (results visible in the app):

```bash
# Full ingestion — all datasets
python -m scripts.real_ingest.runner --date 2026-03-24

# Then rebuild the mega file
python scripts/rebuild_top_performers_mega.py

# Verify
python scripts/verify_historical_datasets.py
python scripts/verify_top_performers.py
```

### Pre-game (boosts only — before tip-off):

```bash
python -m scripts.real_ingest.runner --date 2026-03-25 --datasets boosts
```

### Dry run (safe — no writes):

```bash
python -m scripts.real_ingest.runner --date 2026-03-24 --dry-run
```

### Specific datasets:

```bash
python -m scripts.real_ingest.runner --date 2026-03-24 --datasets most_popular,actuals
```

### Headless (CI / cron):

```bash
python -m scripts.real_ingest.runner --date 2026-03-24 --headless
```

---

## Dataset Reference

| Dataset | When to collect | API endpoint | File |
|---------|----------------|--------------|------|
| `most_popular` | Post-game | `POST /api/save-most-popular` | `data/most_popular/{date}.csv` |
| `most_drafted_3x` | Post-game | `POST /api/save-most-drafted-3x` | `data/most_drafted_3x/{date}.csv` |
| `actuals` | Post-game | `POST /api/save-actuals` | `data/actuals/{date}.csv` |
| `winning_drafts` | Post-game | `POST /api/save-winning-drafts` | `data/winning_drafts/{date}.csv` |
| `boosts` | Pre-game (before tip-off) | `POST /api/save-boosts` | `data/boosts/{date}.json` |

After saving `actuals`, run `python scripts/rebuild_top_performers_mega.py` to merge
new rows into `data/top_performers.csv` (the primary training dataset).

---

## Troubleshooting

### Login fails
- Check `discovery_report/screenshots/login_failed.png` for the current page state
- Verify `REAL_SPORTS_USERNAME` / `REAL_SPORTS_PASSWORD` in `.env`
- Delete `/tmp/real_sports_session.json` to force a fresh login

### No data extracted from a screen
- The screen's `nav_selectors` in `flow_map.json` may be wrong or empty
- Re-run discovery: `python -m scripts.real_ingest.discover`
- Check `network_log.jsonl` for the API calls the app makes on that screen
  — look for JSON responses containing player arrays
- The app may require clicking a specific sub-tab or date picker before data loads

### Concurrent run blocked (lock file)
```bash
ls /tmp/real_ingest_*.lock
# If stale (> 1 hour old):
rm /tmp/real_ingest_2026-03-24.lock
```

### Kill switch
```bash
# Disable all automation
echo "INGEST_AUTOMATION_ENABLED=false" >> .env

# Re-enable
# Edit .env: INGEST_AUTOMATION_ENABLED=true
```

### Run completed but data not in Log tab
The backend caches log data. Run:
```bash
curl -X GET http://localhost:8000/api/refresh
```
Or wait for the 5-minute cache TTL to expire.

After saving actuals, run the rebuild:
```bash
python scripts/rebuild_top_performers_mega.py
git add data/top_performers.csv && git commit -m "rebuild top_performers after ingestion"
git push
```

---

## Security Notes

- Credentials live only in `.env` (never committed — `.gitignore` covers it)
- The automation only reads data — it never clicks Draft, Purchase, or any write action
- Network intercepts are passive (read-only response capture)
- Session state saved to `/tmp/real_sports_session.json` (ephemeral, cleared on reboot)
- Rate limiting: 1 run per date maximum (lock file), 2–3s wait between nav actions

---

## Credential Rotation

1. Update `REAL_SPORTS_USERNAME` / `REAL_SPORTS_PASSWORD` in `.env`
2. Delete the saved session: `rm /tmp/real_sports_session.json`
3. Run the next ingestion — it will re-authenticate with the new credentials

---

## Audit Log

Each run writes `/tmp/real_ingest_{date}.log` with:
- Row counts per dataset
- Publish results (HTTP status per endpoint)
- Verification checks (log endpoint spot-checks)
- Timestamps

---

## File Tree After Full Setup

```
scripts/real_ingest/
├── discovery_report/          (generated by discover.py — not committed)
│   ├── screenshots/
│   ├── dom_snapshots/
│   ├── network_log.jsonl
│   └── flow_map.json          ← edit this to confirm dataset mappings
├── __init__.py
├── runner.py
├── session.py
├── discover.py
├── schemas.py
├── publish.py
├── verify.py
└── extractors/
    ├── __init__.py
    ├── base.py
    ├── most_popular.py
    ├── most_drafted_3x.py
    ├── actuals.py
    ├── winning_drafts.py
    └── boosts.py
```
