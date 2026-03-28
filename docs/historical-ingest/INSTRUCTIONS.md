# Historical ingest — three steps

## 1. Rasterize the PDFs

- Input PDFs live in this folder: `historical-data-input-pt-1.pdf`, `historical-data-input-pt-2.pdf` (or your copies).
- Output folder: `rasterized/` (one PNG per page, ~200 DPI).

**From repo root:**

```bash
pip install pymupdf
python scripts/rasterize_historical_pdfs.py
```

Defaults read `docs/historical-ingest/*.pdf` and write `docs/historical-ingest/rasterized/*.png`.

**If you don’t use Python:** export each PDF page to PNG with Preview, `pdftoppm`, or ImageMagick — same naming pattern as above is fine as long as page order matches the PDFs.

---

## 2. Read the PNGs

- Open each PNG (or batch them in chat).
- For **what each file is** (slate date + screen type), use `batch_manifest.json`: each row has `image`, `date`, `screenshot_type`.
- Transcribe the on-screen tables into structured rows (JSON or CSV-ready). Field names should match what the app expects — see **`docs/HISTORICAL_DATA.md`** (ingest shapes for most_popular, most_drafted_3x, actuals / top performers, winning_drafts).

Skip pure **Games / scoreboard** pages and blank pages (nothing to transcribe for player ingest).

---

## 3. Write the data to the right place

Under repo `data/`:

| `screenshot_type` (manifest) | Write to |
|------------------------------|----------|
| `most_popular` | `data/most_popular/{date}.csv` |
| `most_drafted_high_boost` | `data/most_drafted_3x/{date}.csv` |
| `top_performers` | `data/actuals/{date}.csv` (same columns as other actuals rows; `source` = `highest_value`) |
| `winning_drafts` | `data/winning_drafts/{date}.csv` |

`{date}` is the ISO date in the manifest (e.g. `2025-10-24`).

After actuals / top_performers updates, run locally if you use the mega file:

```bash
python scripts/rebuild_top_performers_mega.py
```

Then commit and push `data/` as you normally do.

---

**Reference:** `batch_manifest.json` (which PNG → which date + type), `docs/HISTORICAL_DATA.md` (column lists), `scripts/rasterize_historical_pdfs.py` (step 1).
