"""FastAPI web application for Manuscript Illustrator."""

import os
import json
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from rich.console import Console

from illustrator.web.routes import manuscripts, chapters
from illustrator.web.models.web_models import ConnectionManager

console = Console()

# Load environment variables from .env file in the project root
# Find the project root directory (where the .env file is located)
project_root = Path(__file__).parent.parent.parent.parent
env_file = project_root / '.env'
load_dotenv(env_file)


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

# Add the process endpoint directly here for simplicity
from illustrator.web.models.web_models import ProcessingRequest
from fastapi import BackgroundTasks
import uuid
from datetime import datetime
import asyncio

@app.post("/api/process")
async def start_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Start manuscript processing and illustration generation."""
    try:
        # Generate a session ID for tracking
        session_id = str(uuid.uuid4())

        # Start the actual processing in the background
        background_tasks.add_task(
            run_processing_workflow,
            session_id=session_id,
            manuscript_id=request.manuscript_id,
            style_config=request.style_config,
            max_emotional_moments=getattr(request, 'max_emotional_moments', 10)
        )

        return {
            "success": True,
            "session_id": session_id,
            "message": "Processing started successfully",
            "started_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting processing: {str(e)}"
        )

async def run_processing_workflow(
    session_id: str,
    manuscript_id: str,
    style_config: dict,
    max_emotional_moments: int = 10
):
    """Run the actual processing workflow with WebSocket updates."""
    try:
        # Import the processing logic
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from generate_scene_illustrations import ComprehensiveSceneAnalyzer, IllustrationGenerator
        from src.illustrator.database import DatabaseManager

        # Send initial status
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": "Starting manuscript processing..."
            }),
            session_id
        )

        # Initialize components
        db_manager = DatabaseManager()
        analyzer = ComprehensiveSceneAnalyzer()
        generator = IllustrationGenerator(
            image_provider=style_config.get("image_provider", "imagen4"),
            art_style=style_config.get("art_style", "digital painting")
        )

        # Load manuscript chapters
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "progress",
                "progress": 10,
                "message": "Loading manuscript chapters..."
            }),
            session_id
        )

        manuscript = db_manager.get_manuscript(manuscript_id)
        if not manuscript:
            raise Exception(f"Manuscript {manuscript_id} not found")

        chapters = db_manager.get_chapters_by_manuscript_id(manuscript_id)
        if not chapters:
            raise Exception(f"No chapters found for manuscript {manuscript_id}")

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "step",
                "step": 0,
                "status": "completed"
            }),
            session_id
        )

        # Process each chapter
        total_images = 0
        progress_per_chapter = 70 // len(chapters)

        for i, chapter in enumerate(chapters):
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "progress",
                    "progress": 20 + (i * progress_per_chapter),
                    "message": f"Analyzing Chapter {chapter.number}: {chapter.title}"
                }),
                session_id
            )

            # Analyze chapter for emotional moments
            emotional_moments = await analyzer.analyze_chapter_comprehensive(chapter)

            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "step",
                    "step": 1,
                    "status": "processing" if i == 0 else "completed"
                }),
                session_id
            )

            # Generate illustration prompts
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "progress",
                    "progress": 20 + (i * progress_per_chapter) + (progress_per_chapter // 3),
                    "message": f"Creating illustration prompts for Chapter {chapter.number}"
                }),
                session_id
            )

            prompts = []
            for moment in emotional_moments[:max_emotional_moments]:
                prompt_text = generator.create_detailed_prompt(
                    moment.description,
                    moment.tone,
                    chapter.title,
                    style_config.get("art_style", "digital painting")
                )
                prompts.append(prompt_text)

            if i == 0:
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "step",
                        "step": 1,
                        "status": "completed"
                    }),
                    session_id
                )
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "step",
                        "step": 2,
                        "status": "processing"
                    }),
                    session_id
                )

            # Generate images
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "progress",
                    "progress": 20 + (i * progress_per_chapter) + (2 * progress_per_chapter // 3),
                    "message": f"Generating {len(prompts)} images for Chapter {chapter.number}"
                }),
                session_id
            )

            results = await generator.generate_images(prompts, chapter)
            total_images += len(results)

            # Send image updates
            for result in results:
                if result.get("success") and result.get("file_path"):
                    # Convert file path to web-accessible URL
                    image_url = f"/static/generated/{Path(result['file_path']).name}"
                    await connection_manager.send_personal_message(
                        json.dumps({
                            "type": "image",
                            "image_url": image_url,
                            "prompt": result.get("prompt", "")
                        }),
                        session_id
                    )

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "step",
                "step": 2,
                "status": "completed"
            }),
            session_id
        )
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "step",
                "step": 3,
                "status": "processing"
            }),
            session_id
        )

        # Final completion
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "progress",
                "progress": 100,
                "message": "Processing completed successfully!"
            }),
            session_id
        )

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "step",
                "step": 3,
                "status": "completed"
            }),
            session_id
        )

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "complete",
                "images_count": total_images,
                "message": f"Successfully generated {total_images} illustrations!"
            }),
            session_id
        )

    except Exception as e:
        console.print(f"[red]Processing error: {e}[/red]")
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "error",
                "error": str(e)
            }),
            session_id
        )


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