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
from illustrator.models import EmotionalTone, IllustrationPrompt

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

# Mount generated images directory
GENERATED_IMAGES_DIR = Path("illustrator_output") / "generated_images"
GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/generated", StaticFiles(directory=str(GENERATED_IMAGES_DIR)), name="generated")

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

@app.get("/api/process/status/{manuscript_id}")
async def get_processing_status(manuscript_id: str):
    """Check if there's an active processing session for a manuscript."""
    try:
        # Look for active sessions for this manuscript
        active_session = None
        for session_id, session_data in connection_manager.sessions.items():
            if session_data.manuscript_id == manuscript_id:
                active_session = {
                    "session_id": session_id,
                    "status": session_data.status.dict() if hasattr(session_data, 'status') else None,
                    "is_connected": session_id in connection_manager.active_connections,
                    "logs": [log.dict() for log in session_data.logs],
                    "images": [image.dict() for image in session_data.images],
                    "start_time": session_data.start_time,
                    "step_status": session_data.step_status
                }
                break

        if active_session:
            return {
                "success": True,
                "active_session": active_session,
                "message": "Active session found"
            }
        else:
            return {
                "success": True,
                "active_session": None,
                "message": "No active session found"
            }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking processing status: {str(e)}"
        )

@app.post("/api/process")
async def start_processing(
    request: ProcessingRequest,
    background_tasks: BackgroundTasks
):
    """Start manuscript processing and illustration generation."""
    try:
        # Check if there's already an active session for this manuscript
        for session_id, session_data in connection_manager.sessions.items():
            if session_data.manuscript_id == request.manuscript_id:
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": "Reconnecting to existing processing session",
                    "started_at": datetime.now().isoformat(),
                    "is_existing": True
                }

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
            "started_at": datetime.now().isoformat(),
            "is_existing": False
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error starting processing: {str(e)}"
        )

@app.post("/api/process/{session_id}/pause")
async def pause_processing(session_id: str):
    """Pause an active processing session."""
    try:
        # Check if session exists
        if session_id not in connection_manager.sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Processing session {session_id} not found"
            )

        session_data = connection_manager.sessions[session_id]

        # Check if session is currently running
        if session_data.status.status not in ["started", "analyzing", "generating"]:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot pause session in '{session_data.status.status}' state"
            )

        # Set pause flag
        session_data.pause_requested = True

        # Send pause message to client
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": "Pause requested - processing will pause after current task completes..."
            }),
            session_id
        )

        return {
            "success": True,
            "message": "Pause requested successfully",
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error pausing processing: {str(e)}"
        )

@app.post("/api/process/{session_id}/resume")
async def resume_processing(session_id: str):
    """Resume a paused processing session."""
    try:
        # Check if session exists
        if session_id not in connection_manager.sessions:
            raise HTTPException(
                status_code=404,
                detail=f"Processing session {session_id} not found"
            )

        session_data = connection_manager.sessions[session_id]

        # Check if session is paused
        if session_data.status.status != "paused":
            raise HTTPException(
                status_code=400,
                detail=f"Cannot resume session in '{session_data.status.status}' state"
            )

        # Clear pause flag and resume
        session_data.pause_requested = False
        session_data.status.status = "generating"  # Resume with generation
        session_data.status.message = "Processing resumed..."

        # Send resume message to client
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "success",
                "message": "Processing resumed"
            }),
            session_id
        )

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "progress",
                "progress": session_data.status.progress,
                "message": "Processing resumed..."
            }),
            session_id
        )

        return {
            "success": True,
            "message": "Processing resumed successfully",
            "session_id": session_id
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resuming processing: {str(e)}"
        )

async def run_processing_workflow(
    session_id: str,
    manuscript_id: str,
    style_config: dict,
    max_emotional_moments: int = 10,
    resume_from_checkpoint: bool = False
):
    """Run the actual processing workflow with WebSocket updates and checkpoint support."""
    try:
        # Import the processing logic
        import sys
        import base64
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent.parent.parent))
        from generate_scene_illustrations import ComprehensiveSceneAnalyzer, IllustrationGenerator
        from illustrator.web.routes.manuscripts import get_saved_manuscripts
        from illustrator.web.models.web_models import ProcessingSessionData, ProcessingStatus
        from illustrator.services.checkpoint_manager import CheckpointManager, ProcessingStep
        from illustrator.services.session_persistence import SessionPersistenceService
        import uuid

        # Initialize persistence services
        persistence_service = SessionPersistenceService()
        checkpoint_manager = CheckpointManager(persistence_service)

        # Check if we're resuming from a checkpoint
        resume_info = None
        if resume_from_checkpoint:
            resume_info = checkpoint_manager.get_resume_info(session_id)
            if resume_info:
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"Resuming session from checkpoint: {resume_info['latest_checkpoint_type']}"
                    }),
                    session_id
                )

        # Create or restore session data
        if resume_info:
            initial_status = ProcessingStatus(
                session_id=session_id,
                manuscript_id=manuscript_id,
                status="resuming",
                progress=resume_info["progress_percent"],
                total_chapters=resume_info["total_chapters"],
                message=f"Resuming from {resume_info['latest_checkpoint_type']}...",
                current_chapter=resume_info.get("current_chapter"),
                error=None
            )
        else:
            initial_status = ProcessingStatus(
                session_id=session_id,
                manuscript_id=manuscript_id,
                status="started",
                progress=0,
                total_chapters=0,  # Will be updated when we load the manuscript
                message="Starting manuscript processing...",
                current_chapter=None,
                error=None
            )

        session_data = ProcessingSessionData(
            session_id=session_id,
            manuscript_id=manuscript_id,
            websocket=None,
            status=initial_status,
            start_time=datetime.now().isoformat(),
            step_status={0: "pending", 1: "pending", 2: "pending", 3: "pending"}
        )

        connection_manager.sessions[session_id] = session_data

        # Send initial status
        initial_message = "Resuming manuscript processing..." if resume_info else "Starting manuscript processing..."
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": initial_message
            }),
            session_id
        )

        # Find manuscript by ID (using the same logic as manuscripts.py)
        manuscripts = get_saved_manuscripts()
        manuscript = None
        for saved_manuscript in manuscripts:
            generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, saved_manuscript.file_path))
            if generated_id == manuscript_id:
                manuscript = saved_manuscript
                break

        if not manuscript:
            raise Exception(f"Manuscript {manuscript_id} not found")

        chapters = manuscript.chapters
        if not chapters:
            raise Exception(f"No chapters found for manuscript {manuscript_id}")

        # Create or update processing session in database
        if not resume_info:
            db_session = persistence_service.create_session(
                manuscript_id=manuscript_id,
                external_session_id=session_id,
                style_config=style_config,
                max_emotional_moments=max_emotional_moments,
                total_chapters=len(chapters)
            )

            # Create session start checkpoint
            checkpoint_manager.create_session_start_checkpoint(
                session_id=str(db_session.id),
                manuscript_id=manuscript_id,
                manuscript_title=manuscript.title,
                total_chapters=len(chapters),
                style_config=style_config,
                max_emotional_moments=max_emotional_moments
            )

        # Update total chapters in session status
        connection_manager.sessions[session_id].status.total_chapters = len(chapters)

        # Initialize components with WebSocket-enabled analyzer
        analyzer = WebSocketComprehensiveSceneAnalyzer(connection_manager, session_id)
        from illustrator.models import ImageProvider

        # Map string to ImageProvider enum
        provider_str = style_config.get("image_provider", "imagen4").lower()
        if provider_str == "dalle":
            provider = ImageProvider.DALLE
        elif provider_str == "imagen4":
            provider = ImageProvider.IMAGEN4
        elif provider_str == "flux":
            provider = ImageProvider.FLUX
        else:
            provider = ImageProvider.DALLE  # default fallback

        # Create output directory
        output_dir = Path("illustrator_output") / "generated_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = WebSocketIllustrationGenerator(connection_manager, session_id, provider, output_dir)

        # Load manuscript chapters
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "progress",
                "progress": 10,
                "message": "Loading manuscript chapters..."
            }),
            session_id
        )

        # Create manuscript loaded checkpoint if not resuming
        if not resume_info:
            chapters_info = [
                {
                    "number": chapter.number,
                    "title": chapter.title,
                    "word_count": len(chapter.content.split())
                }
                for chapter in chapters
            ]

            checkpoint_manager.create_manuscript_loaded_checkpoint(
                session_id=str(db_session.id),
                chapters_info=chapters_info
            )

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "step",
                "step": 0,
                "status": "completed"
            }),
            session_id
        )

        # Update step status in session
        if session_id in connection_manager.sessions:
            connection_manager.sessions[session_id].step_status[0] = "completed"

        # Process each chapter
        total_images = 0
        progress_per_chapter = 70 // len(chapters)

        # Determine starting point for resume
        start_chapter_index = 0
        if resume_info and resume_info.get("last_completed_chapter", 0) > 0:
            start_chapter_index = resume_info["last_completed_chapter"]
            total_images = resume_info.get("total_images_generated", 0)

        for i, chapter in enumerate(chapters[start_chapter_index:], start_chapter_index):
            # Check for pause request before processing each chapter
            if connection_manager.sessions[session_id].pause_requested:
                # Create pause checkpoint
                current_progress = connection_manager.sessions[session_id].status.progress
                checkpoint_manager.create_pause_checkpoint(
                    session_id=session_id,
                    chapter_number=chapter.number,
                    current_step=ProcessingStep.ANALYZING_CHAPTERS,
                    progress_percent=current_progress,
                    pause_reason="user_requested"
                )

                connection_manager.sessions[session_id].status.status = "paused"
                connection_manager.sessions[session_id].status.message = f"Processing paused at Chapter {chapter.number}"
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "warning",
                        "message": f"Processing paused at Chapter {chapter.number}"
                    }),
                    session_id
                )
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "progress",
                        "progress": connection_manager.sessions[session_id].status.progress,
                        "message": f"Processing paused at Chapter {chapter.number}"
                    }),
                    session_id
                )
                return  # Exit the processing function

            current_progress = 20 + (i * progress_per_chapter)

            # Create chapter start checkpoint
            checkpoint_manager.create_chapter_start_checkpoint(
                session_id=session_id,
                chapter_number=chapter.number,
                chapter_title=chapter.title,
                chapter_word_count=len(chapter.content.split()),
                progress_percent=current_progress
            )

            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "progress",
                    "progress": 20 + (i * progress_per_chapter),
                    "message": f"Analyzing Chapter {chapter.number}: {chapter.title}"
                }),
                session_id
            )

            # Analyze chapter for emotional moments
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "info",
                    "message": f"🔍 Deep analysis of Chapter {chapter.number}: {chapter.title}"
                }),
                session_id
            )

            emotional_moments = await analyzer.analyze_chapter_comprehensive(chapter)

            # Create chapter analyzed checkpoint
            analysis_results = {
                "chapter_analyzed": True,
                "emotional_moments_found": len(emotional_moments),
                "analysis_quality": "comprehensive"
            }

            checkpoint_manager.create_chapter_analyzed_checkpoint(
                session_id=session_id,
                chapter_number=chapter.number,
                emotional_moments=emotional_moments,
                analysis_results=analysis_results,
                progress_percent=current_progress + (progress_per_chapter // 3)
            )

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
            prompts_metadata = []
            for idx, moment in enumerate(emotional_moments[:max_emotional_moments], 1):
                primary_tone = moment.emotional_tones[0] if moment.emotional_tones else "neutral"

                # Use advanced AI-powered prompt engineering
                prompt_text = await generator.create_advanced_prompt(
                    emotional_moment=moment,
                    provider=provider,
                    style_config=style_config,
                    chapter=chapter
                )
                prompts.append(prompt_text)

                # Create metadata for the prompt
                prompt_metadata = {
                    "prompt_index": idx,
                    "emotional_tone": str(primary_tone),
                    "text_excerpt": moment.text_excerpt[:200],
                    "intensity_score": moment.intensity_score
                }
                prompts_metadata.append(prompt_metadata)

                # Log detailed prompt generation with AI analysis
                excerpt_preview = moment.text_excerpt
                prompt_preview = prompt_text
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"      🤖 AI-Enhanced Prompt {idx}: [{primary_tone}] \"{excerpt_preview}\""
                    }),
                    session_id
                )
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"         Generated: \"{prompt_preview}\""
                    }),
                    session_id
                )

            # Create prompts generated checkpoint
            checkpoint_manager.create_prompts_generated_checkpoint(
                session_id=session_id,
                chapter_number=chapter.number,
                generated_prompts=prompts,
                prompts_metadata=prompts_metadata,
                progress_percent=current_progress + (2 * progress_per_chapter // 3)
            )

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

                # Update step status in session
                if session_id in connection_manager.sessions:
                    connection_manager.sessions[session_id].step_status[1] = "completed"
                    connection_manager.sessions[session_id].step_status[2] = "processing"

            # Generate images
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "progress",
                    "progress": 20 + (i * progress_per_chapter) + (2 * progress_per_chapter // 3),
                    "message": f"Generating {len(prompts)} images for Chapter {chapter.number}"
                }),
                session_id
            )

            # Create checkpoint before starting image generation
            checkpoint_manager.create_images_generating_checkpoint(
                session_id=session_id,
                chapter_number=chapter.number,
                images_to_generate=len(prompts),
                current_image_index=0,
                progress_percent=current_progress + (2 * progress_per_chapter // 3)
            )

            results = await generator.generate_images(prompts, chapter)
            chapter_images_generated = len([r for r in results if r.get("success")])
            total_images += chapter_images_generated

            # Send image updates and add to session tracking
            for result in results:
                if result.get("success") and result.get("file_path"):
                    # Convert file path to web-accessible URL
                    image_url = f"/generated/{Path(result['file_path']).name}"

                    # Add image entry to session
                    connection_manager.add_image_entry(
                        session_id=session_id,
                        url=image_url,
                        prompt=result.get("prompt", ""),
                        chapter_number=chapter.number,
                        scene_number=results.index(result) + 1
                    )

                    await connection_manager.send_personal_message(
                        json.dumps({
                            "type": "image",
                            "image_url": image_url,
                            "prompt": result.get("prompt", "")
                        }),
                        session_id
                    )

            # Create chapter completed checkpoint
            checkpoint_manager.create_chapter_completed_checkpoint(
                session_id=session_id,
                chapter_number=chapter.number,
                images_generated=chapter_images_generated,
                total_images_so_far=total_images,
                progress_percent=20 + ((i + 1) * progress_per_chapter)
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

        # Update step status in session
        if session_id in connection_manager.sessions:
            connection_manager.sessions[session_id].step_status[2] = "completed"
            connection_manager.sessions[session_id].step_status[3] = "processing"

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

        # Update step status in session
        if session_id in connection_manager.sessions:
            connection_manager.sessions[session_id].step_status[3] = "completed"

        # Create session completion checkpoint
        checkpoint_manager.create_session_completed_checkpoint(
            session_id=session_id,
            total_images_generated=total_images,
            total_chapters_processed=len(chapters)
        )

        # Update session status in database
        persistence_service.update_session_status(
            session_id=session_id,
            status="completed",
            progress_percent=100,
            current_task="Session completed successfully"
        )

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "complete",
                "images_count": total_images,
                "message": f"Successfully generated {total_images} illustrations!"
            }),
            session_id
        )

        # Update session status to completed
        if session_id in connection_manager.sessions:
            connection_manager.sessions[session_id].status.status = "completed"
            connection_manager.sessions[session_id].status.progress = 100
            connection_manager.sessions[session_id].status.message = f"Successfully generated {total_images} illustrations!"

    except Exception as e:
        console.print(f"[red]Processing error: {e}[/red]")

        # Create error checkpoint if we have the checkpoint manager
        try:
            if 'checkpoint_manager' in locals():
                current_chapter = connection_manager.sessions[session_id].status.current_chapter if session_id in connection_manager.sessions else 0
                current_progress = connection_manager.sessions[session_id].status.progress if session_id in connection_manager.sessions else 0

                checkpoint_manager.create_error_checkpoint(
                    session_id=session_id,
                    chapter_number=current_chapter or 0,
                    current_step=ProcessingStep.ANALYZING_CHAPTERS,
                    error_message=str(e),
                    error_details={"error_type": type(e).__name__, "traceback": str(e)},
                    progress_percent=current_progress
                )

                # Update session status in database
                persistence_service.update_session_status(
                    session_id=session_id,
                    status="failed",
                    error_message=str(e)
                )
        except Exception as checkpoint_error:
            console.print(f"[yellow]Warning: Could not create error checkpoint: {checkpoint_error}[/yellow]")

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "error",
                "error": str(e)
            }),
            session_id
        )

        # Update session status to error
        if session_id in connection_manager.sessions:
            connection_manager.sessions[session_id].status.status = "error"
            connection_manager.sessions[session_id].status.error = str(e)

    finally:
        # Clean up resources
        try:
            if 'persistence_service' in locals():
                persistence_service.close()
            if 'checkpoint_manager' in locals():
                checkpoint_manager.close()
        except Exception as cleanup_error:
            console.print(f"[yellow]Warning: Error during cleanup: {cleanup_error}[/yellow]")


class WebSocketComprehensiveSceneAnalyzer:
    """Enhanced analyzer that sends progress updates via WebSocket."""

    def __init__(self, connection_manager, session_id, llm_model: str = "claude-sonnet-4-20250514"):
        # Import here to avoid circular imports
        from generate_scene_illustrations import ComprehensiveSceneAnalyzer
        self.analyzer = ComprehensiveSceneAnalyzer(llm_model)
        self.connection_manager = connection_manager
        self.session_id = session_id

    async def analyze_chapter_comprehensive(self, chapter):
        """Analyze chapter with WebSocket progress updates."""
        # Send initial analysis message
        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": f"📑 Created detailed text segments for analysis"
            }),
            self.session_id
        )

        # Create segments similar to the original analyzer
        segments = self.analyzer._create_detailed_segments(chapter.content, segment_size=300, overlap=50)

        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": f"   📑 Created {len(segments)} detailed text segments"
            }),
            self.session_id
        )

        # Process segments with progress updates
        all_scored_moments = []
        total_segments = len(segments)

        for i, segment in enumerate(segments):
            # Send progress update every few segments with details about high-scoring segments
            if i % max(1, total_segments // 20) == 0:
                progress = int((i / total_segments) * 100)
                await self.connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"⠋ Analyzing segments... {progress}% ({i}/{total_segments}) - Found {len(all_scored_moments)} promising moments so far"
                    }),
                    self.session_id
                )

            # Multi-criteria scoring
            emotional_score = await self.analyzer._score_emotional_intensity(segment)
            visual_score = await self.analyzer._score_visual_potential(segment)
            narrative_score = await self.analyzer._score_narrative_significance(segment)
            dialogue_score = await self.analyzer._score_dialogue_richness(segment)

            # Combined score with weights for illustration potential
            combined_score = (
                emotional_score * 0.3 +
                visual_score * 0.4 +
                narrative_score * 0.2 +
                dialogue_score * 0.1
            )

            if combined_score >= 0.4:  # Lower threshold for more candidates
                moment = await self.analyzer._create_detailed_moment(segment, combined_score, chapter)
                all_scored_moments.append((moment, combined_score))

                # Log high-scoring moments as they're discovered (only show very high scores to avoid spam)
                if combined_score >= 0.7:
                    segment_preview = segment
                    await self.connection_manager.send_personal_message(
                        json.dumps({
                            "type": "log",
                            "level": "info",
                            "message": f"      🌟 High-impact moment found (Score: {combined_score:.2f}): \"{segment_preview}\""
                        }),
                        self.session_id
                    )

        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": f"   ✅ Found {len(all_scored_moments)} high-potential illustration moments"
            }),
            self.session_id
        )

        # Log details about each high-potential moment
        for i, (moment, score) in enumerate(all_scored_moments[:10], 1):  # Show top 10
            excerpt_preview = moment.text_excerpt
            tones_str = ", ".join([tone.value for tone in moment.emotional_tones])
            await self.connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "info",
                    "message": f"      #{i} (Score: {score:.2f}) [{tones_str}]: \"{excerpt_preview}\""
                }),
                self.session_id
            )

        # Select diverse moments
        selected_moments = await self.analyzer._select_diverse_moments(all_scored_moments, target_count=10)

        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": f"   🎨 Selected {len(selected_moments)} diverse illustration scenes"
            }),
            self.session_id
        )

        # Log details about each selected moment
        for i, moment in enumerate(selected_moments, 1):
            excerpt_preview = moment.text_excerpt
            tones_str = ", ".join([tone.value for tone in moment.emotional_tones])
            await self.connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "info",
                    "message": f"      Selected #{i} [{tones_str}] (Intensity: {moment.intensity_score:.2f}): \"{excerpt_preview}\""
                }),
                self.session_id
            )

        return selected_moments


class WebSocketIllustrationGenerator:
    """Enhanced image generator that sends progress updates via WebSocket."""

    def __init__(self, connection_manager, session_id, provider, output_dir):
        # Import here to avoid circular imports
        from generate_scene_illustrations import IllustrationGenerator
        from illustrator.prompt_engineering import PromptEngineer
        from langchain.chat_models import init_chat_model

        self.generator = IllustrationGenerator(provider, output_dir)
        self.connection_manager = connection_manager
        self.session_id = session_id

        # Initialize the advanced prompt engineering system
        self.llm = init_chat_model(model="claude-sonnet-4-20250514", model_provider="anthropic")
        self.prompt_engineer = PromptEngineer(self.llm)

    async def create_advanced_prompt(self, emotional_moment, provider, style_config, chapter):
        """Create an advanced AI-analyzed prompt for image generation."""
        try:
            # Use the advanced prompt engineering system
            illustration_prompt = await self.prompt_engineer.engineer_prompt(
                emotional_moment=emotional_moment,
                provider=provider,
                style_preferences=style_config,
                chapter_context=chapter,
                previous_scenes=[]  # Could be enhanced to track previous scenes
            )

            await self.connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "info",
                    "message": f"🧠 AI Scene Analysis: Generated {len(illustration_prompt.prompt.split())}-word prompt with visual composition analysis"
                }),
                self.session_id
            )

            # Log additional AI analysis details
            if hasattr(illustration_prompt, 'style_modifiers') and illustration_prompt.style_modifiers:
                # Ensure all style modifiers are strings before joining
                style_modifiers_str = []
                for modifier in illustration_prompt.style_modifiers[:3]:
                    if isinstance(modifier, tuple):
                        # For tuples, join the tuple elements with spaces
                        style_modifiers_str.append(" ".join(str(elem) for elem in modifier))
                    else:
                        style_modifiers_str.append(str(modifier))

                await self.connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"   🎨 Style modifiers: {', '.join(style_modifiers_str)}..."
                    }),
                    self.session_id
                )

            return illustration_prompt.prompt

        except Exception as e:
            # Fallback to simple prompt generation
            await self.connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "warning",
                    "message": f"⚠️ AI prompt generation failed, using fallback: {str(e)}"
                }),
                self.session_id
            )

            return self._create_fallback_prompt(emotional_moment.text_excerpt,
                                              emotional_moment.emotional_tones[0] if emotional_moment.emotional_tones else EmotionalTone.NEUTRAL,
                                              chapter.title,
                                              style_config.get("art_style", "digital painting"))

    def _create_fallback_prompt(self, text_excerpt, emotional_tone, chapter_title, art_style):
        """Create an instructive, high-quality fallback prompt (string).

        This avoids vague phrasing (e.g., "depicting a scene") and
        instead gives concrete, directive guidance suitable for
        text-to-image models, tailored by tone and style.
        """
        # Normalize inputs
        style = (art_style or "digital painting").lower()
        tone_key = str(emotional_tone).split('.')[-1].upper() if emotional_tone else "NEUTRAL"
        excerpt = (text_excerpt or "").strip()

        # Media/style-specific lead-in
        if any(k in style for k in ["pencil", "shepard"]):
            lead = "A natural pencil sketch illustration in the classic E.H. Shepard style."
            technique = (
                "The sketch should balance subtle emotion and charm, with fine crosshatching, "
                "gentle graphite shading, and expressive, characterful linework."
            )
        elif "watercolor" in style:
            lead = "A delicate watercolor illustration with soft edges and layered washes."
            technique = "Use gentle color transitions, reserved whites, and restrained detail for a book-illustration feel."
        elif "oil" in style:
            lead = "A traditional oil painting with classic book-illustration sensibility."
            technique = "Employ controlled brushwork, clear focal hierarchy, and warm, cohesive tones."
        elif "digital" in style:
            lead = "A cinematic digital painting with classic book-illustration clarity."
            technique = "Use clean rendering, atmospheric lighting, and a readable focal point with restrained detailing."
        else:
            lead = f"A detailed {art_style} illustration in a classic book-illustration style."
            technique = "Maintain clear focal hierarchy, readable forms, and tasteful, story-forward detailing."

        # Tone-to-direction mapping (kept concise, model-friendly)
        tone_cues = {
            "JOY": "light, open posture; soft, warm lighting; relaxed expressions",
            "SADNESS": "slumped shoulders; downcast eyes; muted tones; gentle shadows",
            "FEAR": "stiff smile, wide eyes, tense shoulders; close, intimate framing; subtle unease",
            "ANGER": "tight jaw; furrowed brow; energetic angle; bold contrasts",
            "TENSION": "held breath; careful spacing between figures; controlled, taut poses",
            "MYSTERY": "soft shadows; partially obscured details; suggestive, not explicit cues",
            "ANTICIPATION": "forward lean; alert eyes; restrained motion; poised tension",
            "SUSPENSE": "stillness; withheld action; compressed spacing; shadow accents",
            "MELANCHOLY": "soft posture; thoughtful gaze; cool, quiet atmosphere",
            "PEACE": "relaxed stance; gentle light; uncluttered forms",
            "ROMANCE": "gentle posture; softened expressions; warm, intimate spacing",
            "NEUTRAL": "balanced posture; natural lighting; calm, observational framing"
        }
        tone_dir = tone_cues.get(tone_key, "balanced posture; natural lighting; calm framing")

        # Build the directive scene line
        if excerpt:
            # Keep excerpt content but in an instructive format
            scene_line = (
                f"Depict the specific moment described — \"{excerpt[:220]}\" — "
                f"as a clear, readable scene."
            )
        else:
            scene_line = (
                f"Depict a key moment from '{chapter_title}', focusing on a clear action and reaction."  # fallback when no excerpt
            )

        # Composition guidance by tone
        comp_by_tone = {
            "FEAR": "Use intimate, slightly off-center framing to heighten unease.",
            "TENSION": "Favor a medium, close, or three-quarter view that emphasizes body language.",
            "SADNESS": "Choose a quiet, balanced composition; leave gentle breathing space around the subject.",
            "ANGER": "Consider a dynamic angle with directional lines leading into the focal point.",
            "MYSTERY": "Let soft shadow shapes and partial occlusion guide the eye to the focal area.",
            "JOY": "Use an open composition with soft, welcoming shapes.",
            "ANTICIPATION": "Place the subject slightly off-center; leave room in the direction of attention.",
            "SUSPENSE": "Keep a controlled, symmetrical base with a small destabilizing element.",
            "MELANCHOLY": "Widen the framing slightly; allow negative space to carry feeling.",
            "PEACE": "Use an even, centered composition with gentle overlaps and clear separation.",
            "ROMANCE": "Favor a two-shot with soft triangulation and gentle overlap for intimacy.",
            "NEUTRAL": "Keep a balanced medium shot with clear focal hierarchy."
        }
        comp = comp_by_tone.get(tone_key, comp_by_tone["NEUTRAL"])

        # Optional environment hints based on common manuscript settings
        env_hints = []
        lower_all = f"{chapter_title} {excerpt}".lower()
        if any(k in lower_all for k in ["room", "hall", "hallway", "stair", "kitchen", "living room", "house", "home", "bedroom"]):
            env_hints.append("Interior domestic setting; suggest doorframes, window light, wall textures, and floor patterns.")
        if any(k in lower_all for k in ["phone", "call", "receiver", "hang up", "dial"]):
            env_hints.append("Include a phone or receiver as a subtle prop if appropriate.")
        if any(k in lower_all for k in ["night", "dark", "shadow", "dim", "lamp"]):
            env_hints.append("Low ambient light; soft lamp glow; readable shadow shapes.")

        # Assemble prompt in the instructive style
        parts = [
            lead,
            scene_line,
            f"Focus cues: {tone_dir}.",
            comp,
            "Include specific environmental cues suggested by the text (doors, windows, light sources, textures).",
            technique
        ]

        if env_hints:
            parts.insert(4, " ".join(env_hints))

        return " ".join(parts)

    async def generate_images(self, prompts, chapter):
        """Generate images with WebSocket progress updates."""
        generated_images = []
        total_prompts = len(prompts)

        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "info",
                "message": f"🎨 Starting image generation for {total_prompts} prompts..."
            }),
            self.session_id
        )

        for i, prompt in enumerate(prompts):
            try:
                # Send progress update with more detail
                progress = int(((i + 1) / total_prompts) * 100)
                # Extract scene description from prompt for logging
                prompt_preview = str(prompt)
                await self.connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "info",
                        "message": f"🖼️ Generating image {i+1}/{total_prompts} ({progress}%): {prompt_preview}"
                    }),
                    self.session_id
                )

                # Convert string prompt to IllustrationPrompt object if necessary
                if isinstance(prompt, str):
                    prompt_obj = IllustrationPrompt(
                        provider=self.generator.provider,
                        prompt=prompt,
                        style_modifiers=[],
                        negative_prompt="",
                        technical_params={}
                    )
                else:
                    prompt_obj = prompt

                result = await self.generator.image_provider.generate_image(prompt_obj)

                if result.get('success'):
                    # Save the image
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"chapter_{chapter.number}_scene_{i+1}_{timestamp}.png"
                    file_path = self.generator.output_dir / filename

                    # Ensure output directory exists
                    self.generator.output_dir.mkdir(parents=True, exist_ok=True)

                    # Save image based on result type
                    if 'base64_image' in result:
                        image_data = base64.b64decode(result['base64_image'])
                        with open(file_path, 'wb') as f:
                            f.write(image_data)
                    elif 'url' in result:
                        # Download the image from URL
                        try:
                            import aiohttp
                            async with aiohttp.ClientSession() as session:
                                async with session.get(result['url']) as response:
                                    if response.status == 200:
                                        image_data = await response.read()
                                        with open(file_path, 'wb') as f:
                                            f.write(image_data)
                                        await self.connection_manager.send_personal_message(
                                            json.dumps({
                                                "type": "log",
                                                "level": "info",
                                                "message": f"✅ Downloaded and saved image {i+1}"
                                            }),
                                            self.session_id
                                        )
                                    else:
                                        raise Exception(f"Failed to download image: HTTP {response.status}")
                        except Exception as download_error:
                            await self.connection_manager.send_personal_message(
                                json.dumps({
                                    "type": "log",
                                    "level": "warning",
                                    "message": f"⚠️ Failed to download image {i+1}: {download_error}. URL: {result['url']}"
                                }),
                                self.session_id
                            )
                            # Set file_path to None to indicate no file was saved
                            file_path = None

                    # Only add to results if we successfully saved a file
                    if file_path is not None:
                        result_info = {
                            'success': True,
                            'file_path': str(file_path),
                            'prompt': str(prompt),
                            'chapter_number': chapter.number,
                            'scene_number': i + 1,
                            'provider': str(self.generator.provider),
                            'generated_at': timestamp
                        }
                        generated_images.append(result_info)

                    await self.connection_manager.send_personal_message(
                        json.dumps({
                            "type": "log",
                            "level": "success",
                            "message": f"✅ Successfully generated image {i+1}/{total_prompts}"
                        }),
                        self.session_id
                    )
                else:
                    await self.connection_manager.send_personal_message(
                        json.dumps({
                            "type": "log",
                            "level": "warning",
                            "message": f"❌ Failed to generate image {i+1}/{total_prompts}: {result.get('error', 'Unknown error')}"
                        }),
                        self.session_id
                    )

            except Exception as e:
                await self.connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "error",
                        "message": f"❌ Error generating image {i+1}/{total_prompts}: {str(e)}"
                    }),
                    self.session_id
                )

        await self.connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "success",
                "message": f"🎉 Image generation complete! Generated {len(generated_images)} out of {total_prompts} images."
            }),
            self.session_id
        )

        return generated_images


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(
        request,
        "index.html",
        {"title": "Dashboard"}
    )


@app.get("/manuscript/new", response_class=HTMLResponse)
async def new_manuscript(request: Request):
    """New manuscript creation page."""
    return templates.TemplateResponse(
        request,
        "manuscript_form.html",
        {"title": "New Manuscript", "manuscript": None}
    )


@app.get("/manuscript/{manuscript_id}", response_class=HTMLResponse)
async def manuscript_detail(request: Request, manuscript_id: str):
    """Manuscript detail and chapter management page."""
    return templates.TemplateResponse(
        request,
        "manuscript_detail.html",
        {"title": "Manuscript", "manuscript_id": manuscript_id}
    )


@app.get("/manuscript/{manuscript_id}/edit", response_class=HTMLResponse)
async def edit_manuscript(request: Request, manuscript_id: str):
    """Edit manuscript page."""
    return templates.TemplateResponse(
        request,
        "manuscript_form.html",
        {"title": "Edit Manuscript", "manuscript_id": manuscript_id}
    )


@app.get("/manuscript/{manuscript_id}/chapter/new", response_class=HTMLResponse)
async def new_chapter(request: Request, manuscript_id: str):
    """New chapter creation page."""
    return templates.TemplateResponse(
        request,
        "chapter_form.html",
        {"title": "New Chapter", "manuscript_id": manuscript_id, "chapter": None}
    )


@app.get("/chapter/{chapter_id}/edit", response_class=HTMLResponse)
async def edit_chapter(request: Request, chapter_id: str):
    """Edit chapter page."""
    return templates.TemplateResponse(
        request,
        "chapter_form.html",
        {"title": "Edit Chapter", "chapter_id": chapter_id}
    )


@app.get("/manuscript/{manuscript_id}/style", response_class=HTMLResponse)
async def style_config(request: Request, manuscript_id: str):
    """Style configuration page."""
    return templates.TemplateResponse(
        request,
        "style_config.html",
        {"title": "Style Configuration", "manuscript_id": manuscript_id}
    )


@app.get("/manuscript/{manuscript_id}/process", response_class=HTMLResponse)
async def processing_page(request: Request, manuscript_id: str):
    """Processing and analysis page."""
    return templates.TemplateResponse(
        request,
        "processing.html",
        {"title": "Processing", "manuscript_id": manuscript_id}
    )


@app.get("/chapter/{chapter_id}/headers", response_class=HTMLResponse)
async def chapter_headers_page(request: Request, chapter_id: str):
    """Chapter header options page."""
    return templates.TemplateResponse(
        request,
        "chapter_headers.html",
        {"title": "Chapter Headers", "chapter_id": chapter_id}
    )


@app.get("/manuscript/{manuscript_id}/gallery", response_class=HTMLResponse)
async def gallery_page(request: Request, manuscript_id: str):
    """Image gallery page."""
    return templates.TemplateResponse(
        request,
        "gallery.html",
        {"title": "Gallery", "manuscript_id": manuscript_id}
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


@app.get("/api/manuscripts/{manuscript_id}/download/{filename}")
async def download_exported_file(manuscript_id: str, filename: str):
    """Download an exported manuscript file."""
    exports_dir = Path("illustrator_output") / "exports"
    file_path = exports_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Security: ensure the filename doesn't contain path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )


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
