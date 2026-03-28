# Historical ingest — three steps

## 1. Rasterize the PDFs

- Put source PDFs in this folder (or keep them wherever you work).
- Export **one PNG per page** (~200 DPI is enough). Tools: macOS Preview (Export), `pdftoppm -png`, ImageMagick `convert`, etc.
- Use clear filenames if it helps (e.g. `pt1-page-01.png`).

---

## 2. Read the PNGs

- Open each PNG and transcribe the tables into structured data (JSON or CSV-ready).
- Figure out **slate date** and **screen type** from what’s on the image (headers, captions, game date).
- **Types you care about:** Most popular, Most drafted 3x, Highest value (top performers), Winning lineups leaderboard.
- **Skip:** pure Games/scoreboard pages and blank pages.
- Column / field expectations: **`docs/HISTORICAL_DATA.md`**.

---

## 3. Write the data to the right place

Under repo `data/`:

| Kind of screen | Write to |
|----------------|----------|
| Most popular | `data/most_popular/{YYYY-MM-DD}.csv` |
| Most drafted 3x | `data/most_drafted_3x/{YYYY-MM-DD}.csv` |
| Highest value / top performers | `data/actuals/{YYYY-MM-DD}.csv` (`source` = `highest_value` where applicable) |
| Winning lineups | `data/winning_drafts/{YYYY-MM-DD}.csv` |

If you use the mega rollup:

```bash
python scripts/rebuild_top_performers_mega.py
```

Then commit and push `data/` as usual.

**Reference:** `docs/HISTORICAL_DATA.md` for ingest shapes and columns.
