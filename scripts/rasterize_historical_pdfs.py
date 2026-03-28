#!/usr/bin/env python3
"""Rasterize historical-ingest PDFs to PNG for POST /api/parse-screenshot.

Requires: pip install pymupdf

Default: docs/historical-ingest/*.pdf -> docs/historical-ingest/rasterized/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import fitz


def pdf_to_pngs(pdf_path: Path, out_dir: Path, dpi: int = 200) -> int:
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    stem = pdf_path.stem
    n = 0
    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        out = out_dir / f"{stem}-page-{i + 1:03d}.png"
        pix.save(out.as_posix())
        n += 1
    doc.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="PDF pages -> PNG for parse-screenshot")
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=Path("docs/historical-ingest"),
        help="Directory containing PDFs (default: docs/historical-ingest)",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/historical-ingest/rasterized"),
        help="Output directory for PNGs",
    )
    ap.add_argument("--dpi", type=int, default=200, help="Raster DPI (default: 200)")
    args = ap.parse_args()

    indir: Path = args.input_dir
    outdir: Path = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(indir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {indir}")

    total = 0
    for pdf in pdfs:
        n = pdf_to_pngs(pdf, outdir, dpi=args.dpi)
        print(f"{pdf.name}: {n} page(s) -> {outdir}")
        total += n
    print(f"Done. {total} PNG file(s) in {outdir}")


if __name__ == "__main__":
    main()
