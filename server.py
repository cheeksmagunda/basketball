"""
Production + local dev server.

Railway / any host: python server.py   (reads PORT env var automatically)
Local dev:          uvicorn server:app --reload
"""

import os
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, FileResponse

from api.index import app  # noqa: F401 – re-export for uvicorn

ROOT = Path(__file__).parent

# --- Static file routes (replaces Vercel's @vercel/static) ---

@app.get("/manifest.json")
async def serve_manifest():
    p = ROOT / "manifest.json"
    if p.exists():
        return FileResponse(p, media_type="application/json",
                            headers={"Cache-Control": "public, max-age=86400"})
    return HTMLResponse("Not found", status_code=404)


@app.get("/favicon.ico")
async def serve_favicon():
    p = ROOT / "favicon.ico"
    if p.exists():
        return FileResponse(p)
    return HTMLResponse("", status_code=204)


# Catch-all: serve index.html for any non-API route (SPA pattern)
# This MUST be last so /api/* routes in api/index.py take priority.
@app.get("/{full_path:path}")
async def serve_frontend(request: Request, full_path: str = ""):
    # Don't intercept /api/* or /docs or /redoc or /openapi.json
    if full_path.startswith("api/") or full_path in ("docs", "redoc", "openapi.json"):
        return HTMLResponse("Not found", status_code=404)
    html_path = ROOT / "index.html"
    return HTMLResponse(
        content=html_path.read_text(),
        status_code=200,
        headers={"Cache-Control": "max-age=0, must-revalidate"},
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, log_level="info")
