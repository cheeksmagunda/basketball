import sys, os
print(f"[boot] Python {sys.version} | PID={os.getpid()} | PORT={os.environ.get('PORT','NOT_SET')}", flush=True)

"""
Production + local dev server.

Railway / any host: python server.py   (reads PORT env var automatically)
Local dev:          uvicorn server:app --reload

grep: DEV SERVER — PORT, static routes (manifest/favicon/svg), SPA index.html catch-all

Dual-mode serving:
  - If frontend/dist/ exists: mount React build assets, serve dist/index.html for all SPA routes.
  - Otherwise: serve legacy index.html + app.js + styles.css from project root (fallback during migration).
"""
from pathlib import Path

from fastapi import Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

print("[boot] importing api.index...", flush=True)
from api.index import app  # noqa: F401 – re-export for uvicorn
print("[boot] api.index imported OK", flush=True)

ROOT = Path(__file__).parent
DIST = ROOT / "frontend" / "dist"
REACT_MODE = DIST.is_dir()

print(f"[boot] serving mode: {'React (frontend/dist/)' if REACT_MODE else 'legacy (root index.html)'}", flush=True)

# ── React mode: mount /assets static files ──────────────────────────────────

if REACT_MODE:
    assets_dir = DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

    # ── Immutable cache for Vite hashed assets ──
    @app.middleware("http")
    async def add_static_asset_cache_headers(request: Request, call_next):
        """Apply immutable 1-year cache to Vite hashed assets in /assets/."""
        response = await call_next(request)
        if request.url.path.startswith("/assets/") and response.status_code == 200:
            # Vite hashes filenames; they never change. Cache for 1 year (31536000 seconds).
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return response

    @app.get("/manifest.json")
    async def serve_manifest_react():
        p = DIST / "manifest.json"
        if p.exists():
            return FileResponse(p, media_type="application/json",
                                headers={"Cache-Control": "public, max-age=86400"})
        return HTMLResponse("Not found", status_code=404)

    @app.get("/favicon.ico")
    async def serve_favicon_react():
        p = DIST / "favicon.ico"
        if p.exists():
            return FileResponse(p)
        return HTMLResponse("", status_code=204)

    @app.get("/oracle-ball.svg")
    async def serve_oracle_ball_react():
        p = DIST / "oracle-ball.svg"
        if p.exists():
            return FileResponse(p, media_type="image/svg+xml",
                                headers={"Cache-Control": "public, max-age=86400"})
        return HTMLResponse("Not found", status_code=404)

    @app.get("/{full_path:path}")
    async def serve_frontend_react(request: Request, full_path: str = ""):
        if full_path.startswith("api/"):
            return JSONResponse({"error": "Endpoint not found"}, status_code=404)
        index = DIST / "index.html"
        return HTMLResponse(
            content=index.read_text(),
            status_code=200,
            headers={"Cache-Control": "max-age=0, must-revalidate"},
        )

# ── Legacy mode: serve root index.html + app.js + styles.css ────────────────

else:
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

    @app.get("/oracle-ball.svg")
    async def serve_oracle_ball():
        return FileResponse(ROOT / "oracle-ball.svg", media_type="image/svg+xml",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/styles.css")
    async def serve_styles():
        return FileResponse(ROOT / "styles.css", media_type="text/css",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/app.js")
    async def serve_app_js():
        return FileResponse(ROOT / "app.js", media_type="application/javascript",
                            headers={"Cache-Control": "public, max-age=86400"})

    @app.get("/{full_path:path}")
    async def serve_frontend(request: Request, full_path: str = ""):
        if full_path.startswith("api/"):
            return JSONResponse({"error": "Endpoint not found"}, status_code=404)
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
