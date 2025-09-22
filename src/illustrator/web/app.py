"""FastAPI web application for Manuscript Illustrator."""

import os
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from illustrator.web.routes import manuscripts, chapters
from illustrator.web.models.web_models import ConnectionManager


# Get the web directory path
WEB_DIR = Path(__file__).parent
STATIC_DIR = WEB_DIR / "static"
TEMPLATES_DIR = WEB_DIR / "templates"

# Create FastAPI app
app = FastAPI(
    title="Manuscript Illustrator",
    description="AI-powered manuscript analysis and illustration generation",
    version="1.0.0",
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# WebSocket connection manager
connection_manager = ConnectionManager()

# Include routers
app.include_router(manuscripts.router, prefix="/api/manuscripts", tags=["manuscripts"])
app.include_router(chapters.router, prefix="/api/chapters", tags=["chapters"])


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "title": "Dashboard"}
    )


@app.get("/manuscript/new", response_class=HTMLResponse)
async def new_manuscript(request: Request):
    """New manuscript creation page."""
    return templates.TemplateResponse(
        "manuscript_form.html",
        {"request": request, "title": "New Manuscript", "manuscript": None}
    )


@app.get("/manuscript/{manuscript_id}", response_class=HTMLResponse)
async def manuscript_detail(request: Request, manuscript_id: str):
    """Manuscript detail and chapter management page."""
    return templates.TemplateResponse(
        "manuscript_detail.html",
        {"request": request, "title": "Manuscript", "manuscript_id": manuscript_id}
    )


@app.get("/manuscript/{manuscript_id}/edit", response_class=HTMLResponse)
async def edit_manuscript(request: Request, manuscript_id: str):
    """Edit manuscript page."""
    return templates.TemplateResponse(
        "manuscript_form.html",
        {"request": request, "title": "Edit Manuscript", "manuscript_id": manuscript_id}
    )


@app.get("/manuscript/{manuscript_id}/chapter/new", response_class=HTMLResponse)
async def new_chapter(request: Request, manuscript_id: str):
    """New chapter creation page."""
    return templates.TemplateResponse(
        "chapter_form.html",
        {"request": request, "title": "New Chapter", "manuscript_id": manuscript_id, "chapter": None}
    )


@app.get("/chapter/{chapter_id}/edit", response_class=HTMLResponse)
async def edit_chapter(request: Request, chapter_id: str):
    """Edit chapter page."""
    return templates.TemplateResponse(
        "chapter_form.html",
        {"request": request, "title": "Edit Chapter", "chapter_id": chapter_id}
    )


@app.get("/manuscript/{manuscript_id}/style", response_class=HTMLResponse)
async def style_config(request: Request, manuscript_id: str):
    """Style configuration page."""
    return templates.TemplateResponse(
        "style_config.html",
        {"request": request, "title": "Style Configuration", "manuscript_id": manuscript_id}
    )


@app.get("/manuscript/{manuscript_id}/process", response_class=HTMLResponse)
async def processing_page(request: Request, manuscript_id: str):
    """Processing and analysis page."""
    return templates.TemplateResponse(
        "processing.html",
        {"request": request, "title": "Processing", "manuscript_id": manuscript_id}
    )


@app.get("/chapter/{chapter_id}/headers", response_class=HTMLResponse)
async def chapter_headers_page(request: Request, chapter_id: str):
    """Chapter header options page."""
    return templates.TemplateResponse(
        "chapter_headers.html",
        {"request": request, "title": "Chapter Headers", "chapter_id": chapter_id}
    )


@app.get("/manuscript/{manuscript_id}/gallery", response_class=HTMLResponse)
async def gallery_page(request: Request, manuscript_id: str):
    """Image gallery page."""
    return templates.TemplateResponse(
        "gallery.html",
        {"request": request, "title": "Gallery", "manuscript_id": manuscript_id}
    )


@app.websocket("/ws/processing/{session_id}")
async def processing_websocket(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time processing updates."""
    await connection_manager.connect(websocket, session_id)
    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now - in practice this would trigger processing
            await connection_manager.send_personal_message(f"Echo: {data}", session_id)
    except WebSocketDisconnect:
        connection_manager.disconnect(session_id)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "manuscript-illustrator"}


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "illustrator.web.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run_server()