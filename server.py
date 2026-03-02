"""Local development server. Run with: uvicorn server:app --reload"""

from pathlib import Path

from fastapi.responses import HTMLResponse

from api.index import app  # noqa: F401 – re-export for uvicorn


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)
