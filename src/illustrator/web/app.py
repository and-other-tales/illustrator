"""FastAPI web application for Manuscript Illustrator."""

import os
import json
import inspect
import subprocess
import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List
from collections.abc import MutableMapping

import uvicorn
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, set_key, unset_key, dotenv_values
from rich.console import Console

from pydantic import BaseModel

from illustrator.web.routes import manuscripts, chapters
from illustrator.db_config import create_tables
from illustrator.web.models.web_models import ConnectionManager
from illustrator.models import EmotionalTone, IllustrationPrompt, ImageProvider

# Initialize logger
logger = logging.getLogger(__name__)

# Re-export commonly patched symbols for tests: make sure tests that patch
# `src.illustrator.web.app.get_saved_manuscripts` or `IllustrationGenerator`
# can find these names directly on this module.
try:
    # get_saved_manuscripts is defined in routes.manuscripts
    from illustrator.web.routes.manuscripts import get_saved_manuscripts, invalidate_manuscripts_cache
except Exception:
    # Provide harmless fallbacks if import fails during partial test setups
    def get_saved_manuscripts():
        return []

    def invalidate_manuscripts_cache():
        return None

try:
    # IllustrationGenerator is implemented in generate_scene_illustrations
    from illustrator.generate_scene_illustrations import IllustrationGenerator
except Exception:
    # Provide a functional fallback implementation
    class IllustrationGenerator:
        def __init__(self, context=None, **kwargs):
            self.context = context or {}
            
        async def generate_scene_illustrations(self, *args, **kwargs):
            """Fallback implementation returns empty results."""
            return {
                'generated_images': [],
                'analysis_results': [],
                'success': False,
                'error_message': 'IllustrationGenerator module not available in this environment'
            }
            
        def get_supported_providers(self):
            """Returns empty list when module unavailable."""
            return []

import websockets as websockets  # re-export name for tests that patch websockets.connect
import httpx as httpx  # re-export httpx for tests that patch httpx in this module

# Expose init_chat_model, PromptEngineer and ComprehensiveSceneAnalyzer at module level
# so tests can patch these symbols on this module (many tests patch e.g. 'src.illustrator.web.app.init_chat_model').
try:
    from langchain.chat_models import init_chat_model
except Exception:
    # Provide a functional fallback that returns a minimal chat model interface
    class FallbackChatModel:
        def __init__(self, **kwargs):
            self.model_kwargs = kwargs
            
        async def ainvoke(self, messages):
            """Fallback returns empty JSON response."""
            from langchain_core.messages import AIMessage
            return AIMessage(content='{}')
            
        def invoke(self, messages):
            """Synchronous fallback."""
            from langchain_core.messages import AIMessage
            return AIMessage(content='{}')
    
    def init_chat_model(*args, **kwargs):
        return FallbackChatModel(**kwargs)

try:
    from illustrator.prompt_engineering import PromptEngineer
except Exception:
    class PromptEngineer:
        def __init__(self, llm=None, **kwargs):
            self.llm = llm
            
        async def enhance_scene_description(self, text, *args, **kwargs):
            """Fallback returns original text unchanged."""
            return text
            
        async def generate_optimized_prompt(self, emotional_moment, *args, **kwargs):
            """Fallback returns basic prompt structure."""
            return {
                'prompt': f"Illustration of: {getattr(emotional_moment, 'text_excerpt', 'scene')}",
                'style_modifiers': [],
                'negative_prompt': None
            }
            
        def extract_visual_elements(self, text, *args, **kwargs):
            """Fallback returns empty visual elements."""
            return []

try:
    from illustrator.generate_scene_illustrations import ComprehensiveSceneAnalyzer
except Exception:
    class ComprehensiveSceneAnalyzer:
        def __init__(self, context=None, **kwargs):
            self.context = context
            
        async def analyze_and_score_scenes(self, chapter, *args, **kwargs):
            """Fallback returns minimal scene analysis."""
            return {
                'scenes': [],
                'analysis_summary': {
                    'total_scenes': 0,
                    'avg_emotional_intensity': 0.0,
                    'dominant_themes': []
                },
                'success': False,
                'error_message': 'ComprehensiveSceneAnalyzer module not available'
            }
            
        def get_analysis_config(self):
            """Returns default configuration."""
            return {'max_scenes': 10, 'min_intensity': 0.5}

console = Console()


def _get_valid_api_keys():
    """Get list of valid API keys from environment variables."""
    api_keys = []
    
    # Check for comma-separated API keys
    keys_env = os.getenv('ILLUSTRATOR_API_KEYS', '')
    if keys_env:
        api_keys.extend([key.strip() for key in keys_env.split(',') if key.strip()])
    
    # Check for single API key (backward compatibility)
    single_key = os.getenv('ILLUSTRATOR_API_KEY', '')
    if single_key and single_key not in api_keys:
        api_keys.append(single_key)
    
    return api_keys

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


CREDENTIAL_ENV_KEYS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "HUGGINGFACE_API_KEY",
    "HUGGINGFACE_ENDPOINT_URL",
    "HUGGINGFACE_USE_PIPELINE",
    "HUGGINGFACE_PIPELINE_TASK",
    "HUGGINGFACE_DEVICE",
    "HUGGINGFACE_MAX_NEW_TOKENS",
    "HUGGINGFACE_TEMPERATURE",
    "HUGGINGFACE_MODEL_KWARGS",
    "HUGGINGFACE_FLUX_ENDPOINT_URL",
    "FLUX_DEV_VERTEX_ENDPOINT_URL",
    "REPLICATE_API_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "GOOGLE_PROJECT_ID",
    "MONGODB_URI",
    "MONGO_URL",
    "MONGO_DB_NAME",
    "MONGO_USE_MOCK",
    "HUGGINGFACE_TIMEOUT",
    "LLM_PROVIDER",
    "DEFAULT_LLM_MODEL",
]


class CredentialUpdateRequest(BaseModel):
    """Payload for updating persistent API credentials."""

    credentials: Dict[str, str | None]


def _latest_commit_timestamp() -> str | None:
    """Return the latest git commit timestamp if available."""
    try:
        timestamp = subprocess.check_output(
            ["git", "log", "-1", "--format=%ct"],
            cwd=str(project_root),
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return timestamp or None
    except (subprocess.CalledProcessError, FileNotFoundError, PermissionError):
        return None


def _current_app_version() -> str:
    """Return the About dialog version string using git metadata when possible."""
    commit_ts = _latest_commit_timestamp()
    if commit_ts:
        return f"0.1.{commit_ts}"
    return f"0.1.{int(datetime.now().timestamp())}"


templates.env.globals["app_version"] = _current_app_version

# WebSocket connection manager
connection_manager = ConnectionManager()

async def _cleanup_completed_session_after_delay(session_id: str, delay_seconds: int):
    """Clean up a completed session after a delay to allow frontend to see completion."""
    await asyncio.sleep(delay_seconds)
    if session_id in connection_manager.sessions:
        session_status = connection_manager.sessions[session_id].status.status
        if session_status in ['completed', 'error']:
            console.log(f"Cleaning up completed session: {session_id}")
            connection_manager.cleanup_session(session_id)

# Ensure database tables exist on startup
@app.on_event("startup")
async def _ensure_tables():
    try:
        create_tables()
        console.log("Database tables ensured/created")
    except Exception as e:
        console.log(f"Failed to create tables: {e}")

# Include routers
app.include_router(manuscripts.router, prefix="/api/manuscripts", tags=["manuscripts"])
app.include_router(chapters.router, prefix="/api/chapters", tags=["chapters"])


def _ensure_env_file() -> None:
    """Ensure the .env file exists for credential persistence."""

    try:
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.touch(exist_ok=True)
    except Exception as exc:  # pragma: no cover - filesystem permission issues
        console.log(f"Unable to ensure .env file exists: {exc}")


def _current_env_snapshot() -> Dict[str, str]:
    """Capture current credential-related environment values."""

    env_values = dotenv_values(env_file) if env_file.exists() else {}
    snapshot: Dict[str, str] = {}

    for key in CREDENTIAL_ENV_KEYS:
        value = os.getenv(key)
        if value is None:
            value = env_values.get(key)
        snapshot[key] = value or ""

    # Legacy support: fall back to DEFAULT_LLM_PROVIDER when LLM_PROVIDER is unset
    if not snapshot.get("LLM_PROVIDER"):
        legacy_provider = os.getenv("DEFAULT_LLM_PROVIDER") or env_values.get("DEFAULT_LLM_PROVIDER")
        if legacy_provider:
            snapshot["LLM_PROVIDER"] = legacy_provider

    if not snapshot.get("MONGO_URL"):
        snapshot["MONGO_URL"] = "mongodb://localhost:27017"
    if not snapshot.get("MONGO_DB_NAME"):
        snapshot["MONGO_DB_NAME"] = "illustrator"
    if not snapshot.get("MONGODB_URI"):
        snapshot["MONGODB_URI"] = snapshot.get("MONGO_URL") or "mongodb://localhost:27017"

    return snapshot


@app.get("/api/config/credentials")
async def get_persisted_credentials() -> Dict[str, Any]:
    """Return persisted API credentials from environment configuration."""

    snapshot = _current_env_snapshot()
    return {
        "success": True,
        "credentials": snapshot,
    }


@app.post("/api/config/credentials")
async def update_persisted_credentials(payload: CredentialUpdateRequest) -> Dict[str, Any]:
    """Persist API credentials to the project's .env file and process environment."""

    try:
        _ensure_env_file()

        for key, raw_value in payload.credentials.items():
            if key not in CREDENTIAL_ENV_KEYS:
                continue

            value = (raw_value or "").strip()

            if value:
                set_key(str(env_file), key, value, quote_mode="never")
                os.environ[key] = value
            else:
                unset_key(str(env_file), key)
                os.environ.pop(key, None)

        if "LLM_PROVIDER" in payload.credentials:
            unset_key(str(env_file), "DEFAULT_LLM_PROVIDER")
            os.environ.pop("DEFAULT_LLM_PROVIDER", None)

        load_dotenv(env_file, override=True)
    except Exception as exc:  # pragma: no cover - defensive logging
        console.log(f"Failed to persist credentials: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to persist credentials: {exc}",
        ) from exc

    snapshot = _current_env_snapshot()
    return {
        "success": True,
        "credentials": snapshot,
    }

# Add the process endpoint directly here for simplicity
from illustrator.web.models.web_models import ProcessingRequest
from fastapi import BackgroundTasks
import uuid
from datetime import datetime
import asyncio
import base64

@app.get("/api/process/status/{manuscript_id}")
async def get_processing_status(manuscript_id: str):
    """Check if there's an active processing session for a manuscript."""
    try:
        logger.debug(f"Checking processing status for manuscript {manuscript_id}")
        logger.debug(f"Total sessions in connection manager: {len(connection_manager.sessions)}")
        
        # Log all sessions for debugging
        for sid, sdata in connection_manager.sessions.items():
            logger.debug(f"Session {sid}: manuscript_id={sdata.manuscript_id}, status={sdata.status.status}")
        
        # Look for sessions for this manuscript, keeping track of active and most recent terminal states
        active_session = None
        last_session = None
        last_session_started_at = None

        def _serialize_session(session_id: str, session_data) -> dict:
            return {
                "session_id": session_id,
                "status": session_data.status.dict() if hasattr(session_data, 'status') else None,
                "is_connected": session_id in connection_manager.active_connections,
                "logs": [log.dict() for log in session_data.logs],
                "images": [image.dict() for image in session_data.images],
                "start_time": getattr(session_data, 'start_time', None),
                "started_at": getattr(session_data, 'started_at', None),
                "step_status": dict(getattr(session_data, 'step_status', {}))
            }

        def _parse_session_time(session_data) -> datetime | None:
            raw_time = getattr(session_data, 'start_time', None) or getattr(session_data, 'started_at', None)
            if not raw_time:
                return None
            try:
                return datetime.fromisoformat(raw_time)
            except ValueError:
                # Gracefully handle timestamps without timezone info
                try:
                    return datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
                except Exception:
                    return None

        for session_id, session_data in connection_manager.sessions.items():
            if session_data.manuscript_id != manuscript_id:
                continue

            session_status = getattr(session_data.status, 'status', None)
            if session_status and session_status not in ['completed', 'error']:
                active_session = _serialize_session(session_id, session_data)
                logger.debug(f"Found active session {session_id} for manuscript {manuscript_id}")
                break

            session_started_at = _parse_session_time(session_data)
            if not last_session_started_at or (
                session_started_at and last_session_started_at and session_started_at > last_session_started_at
            ):
                last_session = _serialize_session(session_id, session_data)
                last_session_started_at = session_started_at

        response_payload = {
            "success": True,
            "active_session": active_session,
            "message": "Active session found" if active_session else "No active session found"
        }

        if last_session and not active_session:
            response_payload["last_session"] = last_session

        if not active_session and not last_session:
            logger.warning(f"No active session found for manuscript {manuscript_id}")

        return response_payload
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
        # Check if there's already an active session for this manuscript (exclude completed/error sessions)
        for session_id, session_data in connection_manager.sessions.items():
            if (session_data.manuscript_id == request.manuscript_id and 
                session_data.status.status not in ['completed', 'error']):
                return {
                    "success": True,
                    "session_id": session_id,
                    "message": "Reconnecting to existing processing session",
                    "started_at": datetime.now().isoformat(),
                    "is_existing": True
                }

        # Generate a session ID for tracking
        session_id = str(uuid.uuid4())

        # Create session data immediately to avoid race conditions
        from illustrator.web.models.web_models import ProcessingSessionData, ProcessingStatus
        
        initial_status = ProcessingStatus(
            session_id=session_id,
            manuscript_id=request.manuscript_id,
            status="initializing",
            progress=0,
            total_chapters=0,
            chapters_processed=0,
            images_generated=0,
            message="Initializing processing session...",
            current_chapter=None,
            error=None
        )

        session_data = ProcessingSessionData(
            session_id=session_id,
            manuscript_id=request.manuscript_id,
            websocket=None,
            status=initial_status,
            start_time=datetime.now().isoformat(),
            started_at=datetime.now().isoformat(),
            step_status={0: "pending", 1: "pending", 2: "pending", 3: "pending"}
        )

        # Register session immediately to avoid race conditions
        connection_manager.sessions[session_id] = session_data
        
        logger.info(f"Created processing session {session_id} for manuscript {request.manuscript_id}")

        # Enhance style config with LLM provider information from environment
        enhanced_style_config = dict(request.style_config)
        llm_provider = os.getenv('LLM_PROVIDER', '').lower()
        llm_model = os.getenv('DEFAULT_LLM_MODEL', '').strip()
        
        if llm_provider:
            enhanced_style_config['llm_provider'] = llm_provider
        if llm_model:
            enhanced_style_config['llm_model'] = llm_model

        # Start the actual processing in the background
        background_tasks.add_task(
            run_processing_workflow,
            session_id=session_id,
            manuscript_id=request.manuscript_id,
            style_config=enhanced_style_config,
            max_emotional_moments=getattr(request, 'max_emotional_moments', 10),
            resume_from_checkpoint=False
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

@app.post("/api/process/resume/{session_id}")
async def resume_processing_from_checkpoint(
    session_id: str,
    background_tasks: BackgroundTasks
):
    """Resume processing from the last checkpoint."""
    try:
        from illustrator.services.checkpoint_manager import CheckpointManager
        from illustrator.services.session_persistence import SessionPersistenceService

        # Initialize persistence services to check for resumable session
        persistence_service = SessionPersistenceService()
        checkpoint_manager = CheckpointManager(persistence_service)

        # Get resume information
        resume_info = checkpoint_manager.get_resume_info(session_id)
        if not resume_info:
            raise HTTPException(
                status_code=404,
                detail=f"No resumable session found with ID: {session_id}"
            )

        # Check if session is already active
        if session_id in connection_manager.sessions:
            return {
                "success": True,
                "session_id": session_id,
                "message": "Reconnecting to existing processing session",
                "resumed_at": datetime.now().isoformat(),
                "is_existing": True,
                "resume_info": resume_info
            }

        # Enhance style config with LLM provider information from environment
        enhanced_style_config = dict(resume_info["style_config"])
        llm_provider = os.getenv('LLM_PROVIDER', '').lower()
        llm_model = os.getenv('DEFAULT_LLM_MODEL', '').strip()
        
        if llm_provider:
            enhanced_style_config['llm_provider'] = llm_provider
        if llm_model:
            enhanced_style_config['llm_model'] = llm_model

        # Start processing with resume flag
        background_tasks.add_task(
            run_processing_workflow,
            session_id=session_id,
            manuscript_id=resume_info["manuscript_id"],
            style_config=enhanced_style_config,
            max_emotional_moments=resume_info["max_emotional_moments"],
            resume_from_checkpoint=True
        )

        persistence_service.close()
        checkpoint_manager.close()

        return {
            "success": True,
            "session_id": session_id,
            "message": "Processing resumed from checkpoint",
            "resumed_at": datetime.now().isoformat(),
            "is_existing": False,
            "resume_info": {
                "checkpoint_type": resume_info.get("latest_checkpoint_type"),
                "progress_percent": resume_info.get("progress_percent"),
                "total_chapters": resume_info.get("total_chapters"),
                "last_completed_chapter": resume_info.get("last_completed_chapter")
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error resuming processing: {str(e)}"
        )

@app.get("/api/process/resumable")
async def get_resumable_sessions():
    """Get all sessions that can be resumed."""
    try:
        from illustrator.services.session_persistence import SessionPersistenceService

        persistence_service = SessionPersistenceService()
        resumable_sessions = persistence_service.get_resumable_sessions()

        session_info = []
        for session in resumable_sessions:
            started_at = session.get("started_at")
            paused_at = session.get("paused_at")

            session_info.append({
                "session_id": session["id"],
                "manuscript_id": session.get("manuscript_id"),
                "external_session_id": session.get("external_session_id"),
                "status": session.get("status"),
                "progress_percent": session.get("progress_percent"),
                "total_chapters": session.get("total_chapters"),
                "last_completed_chapter": session.get("last_completed_chapter"),
                "total_images_generated": session.get("total_images_generated"),
                "started_at": started_at.isoformat() if isinstance(started_at, datetime) else started_at,
                "paused_at": paused_at.isoformat() if isinstance(paused_at, datetime) else paused_at,
                "error_message": session.get("error_message"),
            })

        persistence_service.close()

        return {
            "success": True,
            "resumable_sessions": session_info,
            "count": len(session_info)
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting resumable sessions: {str(e)}"
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
                "message": "Processing resumed...",
                "current_chapter": session_data.status.current_chapter,
                "chapters_processed": getattr(session_data.status, 'chapters_processed', 0),
                "images_generated": getattr(session_data.status, 'images_generated', 0),
                "total_chapters": getattr(session_data.status, 'total_chapters', 0)
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
        try:
            from illustrator import generate_scene_illustrations as _scene_tools
        except ModuleNotFoundError:
            import generate_scene_illustrations as _scene_tools  # type: ignore
        # Prefer a module-level patched ComprehensiveSceneAnalyzer when tests patch this module.
        ComprehensiveSceneAnalyzer = globals().get('ComprehensiveSceneAnalyzer') or getattr(_scene_tools, 'ComprehensiveSceneAnalyzer', None)
        IllustrationGenerator = _scene_tools.IllustrationGenerator
        from illustrator.web.routes.manuscripts import get_saved_manuscripts
        from illustrator.web.models.web_models import ProcessingSessionData, ProcessingStatus
        from illustrator.services.checkpoint_manager import CheckpointManager, ProcessingStep
        from illustrator.services.session_persistence import SessionPersistenceService
        import uuid

        # Initialize persistence services (resilient to missing database)
        persistence_service: SessionPersistenceService | None = None
        checkpoint_manager: CheckpointManager | None = None
        try:
            persistence_service = SessionPersistenceService()
            checkpoint_manager = CheckpointManager(persistence_service)
        except Exception as persistence_error:  # noqa: BLE001
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "warning",
                    "message": f"Persistence service unavailable, continuing without DB logging: {persistence_error}"
                }),
                session_id
            )

        # Check if we're resuming from a checkpoint
        resume_info = None
        if resume_from_checkpoint and checkpoint_manager:
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

        # Ensure connection manager has a mutable session store (supports tests with mocks)
        sessions = getattr(connection_manager, "sessions", None)
        if not isinstance(sessions, MutableMapping):
            connection_manager.sessions = {}

        # Get or update the existing session data (it should already exist from start_processing)
        if session_id in connection_manager.sessions:
            session_data = connection_manager.sessions[session_id]
            # Update status to indicate we're starting actual processing
            session_data.status.status = "started" if not resume_info else "resuming"
            session_data.status.message = "Starting manuscript processing..." if not resume_info else f"Resuming from {resume_info['latest_checkpoint_type']}..."
            if resume_info:
                session_data.status.progress = resume_info["progress_percent"]
                session_data.status.current_chapter = resume_info.get("current_chapter")
                session_data.status.chapters_processed = resume_info.get("last_completed_chapter", 0)
                session_data.status.images_generated = resume_info.get("total_images_generated", 0)
        else:
            # Fallback: create session if it doesn't exist (shouldn't happen with the new flow)
            logger.warning(f"Session {session_id} not found in connection manager, creating new one")
            if resume_info:
                initial_status = ProcessingStatus(
                    session_id=session_id,
                    manuscript_id=manuscript_id,
                    status="resuming",
                    progress=resume_info["progress_percent"],
                    total_chapters=resume_info["total_chapters"],
                    chapters_processed=resume_info.get("last_completed_chapter", 0),
                    images_generated=resume_info.get("total_images_generated", 0),
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
                    chapters_processed=0,
                    images_generated=0,
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
                started_at=datetime.now().isoformat(),
                step_status={0: "pending", 1: "pending", 2: "pending", 3: "pending"}
            )

            connection_manager.sessions[session_id] = session_data

        def progress_payload(progress_value, message, *, current_chapter=None):
            status = session_data.status
            status.progress = progress_value
            status.message = message
            if current_chapter is not None:
                status.current_chapter = current_chapter
            payload = {
                "type": "progress",
                "progress": progress_value,
                "message": message,
                "total_chapters": getattr(status, "total_chapters", 0),
                "chapters_processed": getattr(status, "chapters_processed", 0),
                "images_generated": getattr(status, "images_generated", 0),
            }
            chapter_value = status.current_chapter
            if chapter_value is not None:
                payload["current_chapter"] = chapter_value
            return payload

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

        # Ensure the manuscript and chapters exist in the database before creating a session
        # This prevents FK violations when inserting into processing_sessions.
        try:
            from illustrator.db_config import get_db
            from illustrator.db_models import CHAPTERS_COLLECTION, MANUSCRIPTS_COLLECTION

            db = get_db()
            manuscripts_col = db[MANUSCRIPTS_COLLECTION]
            chapters_col = db[CHAPTERS_COLLECTION]

            now = datetime.utcnow()
            manuscript_payload = {
                "title": getattr(manuscript, "title", None) or manuscript.metadata.title,
                "author": getattr(manuscript, "author", None) or getattr(manuscript.metadata, "author", None),
                "genre": getattr(manuscript, "genre", None) or getattr(manuscript.metadata, "genre", None),
                "total_chapters": len(chapters),
                "updated_at": now,
            }
            manuscripts_col.update_one(
                {"_id": manuscript_id},
                {
                    "$set": manuscript_payload,
                    "$setOnInsert": {"created_at": now},
                },
                upsert=True,
            )

            for ch in chapters:
                try:
                    chapter_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{ch.number}_{ch.title}"))
                    chapter_payload = {
                        "manuscript_id": manuscript_id,
                        "number": ch.number,
                        "title": ch.title,
                        "content": ch.content,
                        "word_count": getattr(ch, "word_count", 0) or 0,
                        "updated_at": now,
                    }
                    chapters_col.update_one(
                        {"_id": chapter_uuid},
                        {
                            "$set": chapter_payload,
                            "$setOnInsert": {"created_at": now},
                        },
                        upsert=True,
                    )
                except Exception:
                    pass
        except Exception as ensure_db_err:
            # Log but continue; create_session below will still raise if manuscript truly missing
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "log",
                    "level": "warning",
                    "message": f"Database ensure step encountered an issue (continuing): {str(ensure_db_err)}"
                }),
                session_id
            )

        # Track the DB session id for persistence/checkpoints
        db_session_id_for_persistence = None

        # Create or update processing session in database
        if not resume_info:
            if persistence_service and checkpoint_manager:
                try:
                    db_session = persistence_service.create_session(
                        manuscript_id=manuscript_id,
                        external_session_id=session_id,
                        style_config=style_config,
                        max_emotional_moments=max_emotional_moments,
                        total_chapters=len(chapters)
                    )
                    db_session_id_for_persistence = db_session["id"]
                    checkpoint_manager.create_session_start_checkpoint(
                        session_id=db_session_id_for_persistence,
                        manuscript_id=manuscript_id,
                        manuscript_title=manuscript.title if hasattr(manuscript, 'title') else manuscript.metadata.title,
                        total_chapters=len(chapters),
                        style_config=style_config,
                        max_emotional_moments=max_emotional_moments
                    )
                except Exception as create_session_error:  # noqa: BLE001
                    await connection_manager.send_personal_message(
                        json.dumps({
                            "type": "log",
                            "level": "warning",
                            "message": f"Unable to persist session metadata (continuing): {create_session_error}"
                        }),
                        session_id
                    )
                    persistence_service = None
                    checkpoint_manager = None
            persistence_enabled = bool(db_session_id_for_persistence and checkpoint_manager)
        else:
            # In resume flow, the incoming session_id is the DB session id
            db_session_id_for_persistence = session_id
            persistence_enabled = bool(checkpoint_manager)

        # Update total chapters in session status
        connection_manager.sessions[session_id].status.total_chapters = len(chapters)

        # Initialize components with WebSocket-enabled analyzer
        # Pass LLM model and style config for proper provider detection
        llm_model = style_config.get("llm_model")
        analyzer = WebSocketComprehensiveSceneAnalyzer(connection_manager, session_id, llm_model, style_config)
        from illustrator.models import ImageProvider

        # Map string to ImageProvider enum
        provider_str = style_config.get("image_provider", "imagen4").lower()
        if provider_str == "dalle":
            provider = ImageProvider.DALLE
        elif provider_str == "imagen4":
            provider = ImageProvider.IMAGEN4
        elif provider_str == "flux":
            provider = ImageProvider.FLUX
        elif provider_str == "flux_dev_vertex":
            provider = ImageProvider.FLUX_DEV_VERTEX
        elif provider_str == "huggingface":
            provider = ImageProvider.HUGGINGFACE
        elif provider_str in {"seedream", "seedream4"}:
            provider = ImageProvider.SEEDREAM
        else:
            provider = ImageProvider.DALLE  # default fallback

        # Validate provider-specific configuration before attempting generation
        validation_errors: list[str] = []
        resolved_flux_endpoint: str | None = None
        if provider == ImageProvider.FLUX_DEV_VERTEX:
            resolved_flux_endpoint = (
                style_config.get("flux_dev_vertex_endpoint_url")
                or os.getenv("FLUX_DEV_VERTEX_ENDPOINT_URL")
            )
            resolved_gcp_project = (
                style_config.get("gcp_project_id")
                or style_config.get("google_project_id")
                or os.getenv("GOOGLE_PROJECT_ID")
                or os.getenv("GCP_PROJECT_ID")
            )

            if not resolved_flux_endpoint:
                validation_errors.append("Flux Dev Vertex endpoint URL")
            if not resolved_gcp_project:
                validation_errors.append("Google Cloud project ID")

            if validation_errors:
                missing_items = ", ".join(validation_errors)
                guidance_message = (
                    "Flux Dev Vertex provider is missing required configuration: "
                    f"{missing_items}. Configure these values via the API Settings (gear icon) "
                    "or include them in your style configuration before retrying."
                )

                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "error",
                        "message": guidance_message
                    }),
                    session_id
                )

                raise RuntimeError(guidance_message)

            # Ensure downstream components receive the resolved endpoint
            if not style_config.get("flux_dev_vertex_endpoint_url"):
                style_config["flux_dev_vertex_endpoint_url"] = resolved_flux_endpoint

        # Create output directory
        output_dir = Path("illustrator_output") / "generated_images"
        output_dir.mkdir(parents=True, exist_ok=True)

        generator = WebSocketIllustrationGenerator(connection_manager, session_id, provider, output_dir, style_config)

        total_images = 0
        progress_per_chapter = 70 // len(chapters) if len(chapters) > 0 else 0
        start_chapter_index = 0
        if resume_info and resume_info.get("last_completed_chapter", 0) > 0:
            start_chapter_index = resume_info["last_completed_chapter"]
            total_images = resume_info.get("total_images_generated", 0)

        session_data.status.total_chapters = len(chapters)
        session_data.status.chapters_processed = start_chapter_index
        session_data.status.images_generated = total_images
        session_data.status.current_chapter = resume_info.get("current_chapter") if resume_info else None

        # Load manuscript chapters
        await connection_manager.send_personal_message(
            json.dumps(progress_payload(10, "Loading manuscript chapters...")),
            session_id
        )

        # Create manuscript loaded checkpoint if not resuming
        if persistence_enabled:
            chapters_info = [
                {
                    "number": chapter.number,
                    "title": chapter.title,
                    "word_count": len(chapter.content.split())
                }
                for chapter in chapters
            ]

            checkpoint_manager.create_manuscript_loaded_checkpoint(
                session_id=db_session_id_for_persistence,
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

        for i, chapter in enumerate(chapters[start_chapter_index:], start_chapter_index):
            session_data.status.current_chapter = chapter.number
            session_data.status.chapters_processed = i

            # Check for pause request before processing each chapter
            if connection_manager.sessions[session_id].pause_requested:
                # Create pause checkpoint
                current_progress = connection_manager.sessions[session_id].status.progress
                if checkpoint_manager:
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
                    json.dumps(progress_payload(session_data.status.progress, f"Processing paused at Chapter {chapter.number}")),
                    session_id
                )
                return  # Exit the processing function

            current_progress = 20 + (i * progress_per_chapter)

            # Create chapter start checkpoint
            if checkpoint_manager:
                checkpoint_manager.create_chapter_start_checkpoint(
                    session_id=session_id,
                    chapter_number=chapter.number,
                    chapter_title=chapter.title,
                    chapter_word_count=len(chapter.content.split()),
                    progress_percent=current_progress
                )

            await connection_manager.send_personal_message(
                json.dumps(progress_payload(
                    20 + (i * progress_per_chapter),
                    f"Analyzing Chapter {chapter.number}: {chapter.title}"
                )),
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

            if checkpoint_manager:
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
                json.dumps(progress_payload(
                    20 + (i * progress_per_chapter) + (progress_per_chapter // 3),
                    f"Creating illustration prompts for Chapter {chapter.number}"
                )),
                session_id
            )

            prompts = []
            prompts_metadata = []
            for idx, moment in enumerate(emotional_moments[:max_emotional_moments], 1):
                primary_tone = moment.emotional_tones[0] if moment.emotional_tones else "neutral"

                # Use advanced AI-powered prompt engineering
                prompt_candidate = generator.create_advanced_prompt(
                    emotional_moment=moment,
                    provider=provider,
                    style_config=style_config,
                    chapter=chapter
                )
                prompt_text = await prompt_candidate if inspect.isawaitable(prompt_candidate) else prompt_candidate
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
            if checkpoint_manager:
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
                json.dumps(progress_payload(
                    20 + (i * progress_per_chapter) + (2 * progress_per_chapter // 3),
                    f"Generating {len(prompts)} images for Chapter {chapter.number}"
                )),
                session_id
            )

            # Notify gallery of progress
            await connection_manager.notify_gallery_update(
                manuscript_id=manuscript_id,
                update_type="processing_progress",
                data={
                    "progress": 20 + (i * progress_per_chapter) + (2 * progress_per_chapter // 3),
                    "message": f"Generating images for Chapter {chapter.number}",
                    "chapter_number": chapter.number,
                    "images_generated": total_images
                }
            )

            # Create checkpoint before starting image generation
            if checkpoint_manager:
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
            session_data.status.images_generated = total_images

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

                    # Notify gallery of new image
                    await connection_manager.notify_gallery_update(
                        manuscript_id=manuscript_id,
                        update_type="new_image",
                        data={
                            "image_url": image_url,
                            "prompt": result.get("prompt", ""),
                            "chapter_number": chapter.number
                        }
                    )

            # Create chapter completed checkpoint
            if checkpoint_manager and db_session_id_for_persistence:
                checkpoint_manager.create_chapter_completed_checkpoint(
                    session_id=db_session_id_for_persistence,
                    chapter_number=chapter.number,
                    images_generated=chapter_images_generated,
                    total_images_so_far=total_images,
                    progress_percent=20 + ((i + 1) * progress_per_chapter)
                )

            session_data.status.chapters_processed = i + 1

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
        session_data.status.current_chapter = None
        session_data.status.chapters_processed = len(chapters)
        session_data.status.images_generated = total_images

        await connection_manager.send_personal_message(
            json.dumps(progress_payload(100, "Processing completed successfully!")),
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

        # Create session completion checkpoint if persistence is available
        if checkpoint_manager and db_session_id_for_persistence:
            checkpoint_manager.create_session_completed_checkpoint(
                session_id=db_session_id_for_persistence,
                total_images_generated=total_images,
                total_chapters_processed=len(chapters)
            )

        # Update session status in database when persistence is configured
        if persistence_service and db_session_id_for_persistence:
            persistence_service.update_session_status(
                session_id=db_session_id_for_persistence,
                status="completed",
                progress_percent=100,
                current_task="Session completed successfully"
            )

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "complete",
                "images_count": total_images,
                "chapters_processed": len(chapters),
                "total_chapters": len(chapters),
                "message": f"Successfully generated {total_images} illustrations!"
            }),
            session_id
        )

        # Notify gallery of processing completion
        await connection_manager.notify_gallery_update(
            manuscript_id=manuscript_id,
            update_type="processing_complete",
            data={
                "images_count": total_images,
                "message": f"Processing completed - {total_images} illustrations generated!"
            }
        )

        # Update session status to completed
        if session_id in connection_manager.sessions:
            connection_manager.sessions[session_id].status.status = "completed"
            connection_manager.sessions[session_id].status.progress = 100
            connection_manager.sessions[session_id].status.message = f"Successfully generated {total_images} illustrations!"
            
            # Schedule cleanup of completed session after 30 seconds to allow frontend to see completion
            asyncio.create_task(_cleanup_completed_session_after_delay(session_id, 30))

    except Exception as e:
        console.print(f"[red]Processing error: {e}[/red]")

        # Create error checkpoint if we have the checkpoint manager
        try:
            if checkpoint_manager and db_session_id_for_persistence:
                current_chapter = connection_manager.sessions[session_id].status.current_chapter if session_id in connection_manager.sessions else 0
                current_progress = connection_manager.sessions[session_id].status.progress if session_id in connection_manager.sessions else 0

                checkpoint_manager.create_error_checkpoint(
                    session_id=db_session_id_for_persistence,
                    chapter_number=current_chapter or 0,
                    current_step=ProcessingStep.ANALYZING_CHAPTERS,
                    error_message=str(e),
                    error_details={"error_type": type(e).__name__, "traceback": str(e)},
                    progress_percent=current_progress
                )

                if persistence_service:
                    persistence_service.update_session_status(
                        session_id=db_session_id_for_persistence,
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

        await connection_manager.send_personal_message(
            json.dumps({
                "type": "log",
                "level": "error",
                "message": str(e)
            }),
            session_id
        )

        # Update session status to error
        if session_id in connection_manager.sessions:
            connection_manager.sessions[session_id].status.status = "error"
            connection_manager.sessions[session_id].status.error = str(e)
            
            # Schedule cleanup of error session after 30 seconds to allow frontend to see error
            asyncio.create_task(_cleanup_completed_session_after_delay(session_id, 30))

    finally:
        # Clean up resources
        try:
            if persistence_service:
                persistence_service.close()
            if checkpoint_manager:
                checkpoint_manager.close()
        except Exception as cleanup_error:
            console.print(f"[yellow]Warning: Error during cleanup: {cleanup_error}[/yellow]")


class WebSocketComprehensiveSceneAnalyzer:
    """Enhanced analyzer that sends progress updates via WebSocket."""

    def __init__(
        self,
        connection_manager,
        session_id,
        llm_model: str | None = "claude-sonnet-4-20250514",
        style_config: dict | None = None,
    ):
        # Import here to avoid circular imports
        try:
            from illustrator import generate_scene_illustrations as _scene_tools
        except ModuleNotFoundError:
            import generate_scene_illustrations as _scene_tools  # type: ignore

        from illustrator.context import get_default_context, ManuscriptContext
        from illustrator.models import LLMProvider

        ComprehensiveSceneAnalyzer = _scene_tools.ComprehensiveSceneAnalyzer

        context: ManuscriptContext = get_default_context()

        # Apply LLM provider from style config if available
        if style_config and 'llm_provider' in style_config:
            provider_str = style_config['llm_provider'].lower()
            if provider_str == 'anthropic':
                context.llm_provider = LLMProvider.ANTHROPIC
            elif provider_str == 'anthropic_vertex':
                context.llm_provider = LLMProvider.ANTHROPIC_VERTEX
            elif provider_str == 'huggingface':
                context.llm_provider = LLMProvider.HUGGINGFACE

        # Ensure GCP project ID is available for Anthropic Vertex
        if context.llm_provider == LLMProvider.ANTHROPIC_VERTEX:
            import os
            gcp_project_id = os.getenv('GOOGLE_PROJECT_ID') or os.getenv('GCP_PROJECT_ID')
            if gcp_project_id:
                context.gcp_project_id = gcp_project_id

        if llm_model:
            try:
                context.model = llm_model
            except Exception:
                # context may be a SimpleNamespace without attributes; ignore
                pass
            # Check if model has provider prefix (e.g., "anthropic/claude-3-5-sonnet")
            if "/" in llm_model:
                provider_prefix = llm_model.split("/", 1)[0]
                try:
                    context.llm_provider = LLMProvider(provider_prefix)
                    # Remove provider prefix from model name for actual usage
                    context.model = llm_model.split("/", 1)[1]
                except Exception:
                    pass

        # Initialize analyzer instance (prefer module-level patched symbol)
        self.analyzer = None
        try:
            if ComprehensiveSceneAnalyzer is not None:
                self.analyzer = ComprehensiveSceneAnalyzer(context=context)
        except Exception:
            # If patched analyzer or its constructor fails, leave as None; tests will mock methods as needed
            self.analyzer = None

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
            try:
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

                # Add debug logging for the current segment
                await self.connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "debug",
                        "message": f"      Processing segment {i+1}/{total_segments}: {segment[:100]}..."
                    }),
                    self.session_id
                )

                # Multi-criteria scoring with timeout protection
                try:
                    emotional_score = await asyncio.wait_for(
                        self.analyzer._score_emotional_intensity(segment), 
                        timeout=30.0
                    )
                    visual_score = await asyncio.wait_for(
                        self.analyzer._score_visual_potential(segment), 
                        timeout=30.0
                    )
                    narrative_score = await asyncio.wait_for(
                        self.analyzer._score_narrative_significance(segment), 
                        timeout=30.0
                    )
                    dialogue_score = await asyncio.wait_for(
                        self.analyzer._score_dialogue_richness(segment), 
                        timeout=30.0
                    )
                except asyncio.TimeoutError:
                    await self.connection_manager.send_personal_message(
                        json.dumps({
                            "type": "log",
                            "level": "warning",
                            "message": f"      ⚠️ Scoring timeout for segment {i+1}, using fallback scores"
                        }),
                        self.session_id
                    )
                    # Use fallback scores
                    emotional_score = 0.2
                    visual_score = 0.2
                    narrative_score = 0.2
                    dialogue_score = 0.2

                # Combined score with weights for illustration potential
                combined_score = (
                    emotional_score * 0.3 +
                    visual_score * 0.4 +
                    narrative_score * 0.2 +
                    dialogue_score * 0.1
                )

                if combined_score >= 0.4:  # Lower threshold for more candidates
                    try:
                        moment = await asyncio.wait_for(
                            self.analyzer._create_detailed_moment(segment, combined_score, chapter),
                            timeout=15.0
                        )
                        all_scored_moments.append((moment, combined_score))

                        # Log high-scoring moments as they're discovered (only show very high scores to avoid spam)
                        if combined_score >= 0.7:
                            segment_preview = segment[:100] + "..." if len(segment) > 100 else segment
                            await self.connection_manager.send_personal_message(
                                json.dumps({
                                    "type": "log",
                                    "level": "info",
                                    "message": f"      🌟 High-impact moment found (Score: {combined_score:.2f}): \"{segment_preview}\""
                                }),
                                self.session_id
                            )
                    except asyncio.TimeoutError:
                        await self.connection_manager.send_personal_message(
                            json.dumps({
                                "type": "log",
                                "level": "warning",
                                "message": f"      ⚠️ Moment creation timeout for segment {i+1}, skipping"
                            }),
                            self.session_id
                        )
                    except Exception as e:
                        await self.connection_manager.send_personal_message(
                            json.dumps({
                                "type": "log",
                                "level": "error",
                                "message": f"      ❌ Error creating moment for segment {i+1}: {str(e)}"
                            }),
                            self.session_id
                        )

            except Exception as e:
                await self.connection_manager.send_personal_message(
                    json.dumps({
                        "type": "log",
                        "level": "error",
                        "message": f"      ❌ Error processing segment {i+1}: {str(e)}"
                    }),
                    self.session_id
                )
                continue

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

    def __init__(self, connection_manager, session_id, provider, output_dir, style_config=None):
        # Import here to avoid circular imports
        try:
            from illustrator import generate_scene_illustrations as _scene_tools
        except ModuleNotFoundError:
            import generate_scene_illustrations as _scene_tools  # type: ignore

        # Prefer module-level patched classes when available (tests often patch these names on this module)
        IllustrationGenerator = globals().get('IllustrationGenerator') or getattr(_scene_tools, 'IllustrationGenerator', None)
        PromptEngineerCls = globals().get('PromptEngineer')
        try:
            from illustrator.context import get_default_context, ManuscriptContext
            from illustrator.models import LLMProvider
            from illustrator.llm_factory import create_chat_model_from_context
        except Exception:
            # In some test environments these imports may fail; leave as None and allow tests to patch behavior
            get_default_context = lambda: None
            ManuscriptContext = None
            LLMProvider = None
            create_chat_model_from_context = None

        context = get_default_context()
        # Provide a lightweight fallback if context is None
        if context is None:
            from types import SimpleNamespace
            context = SimpleNamespace()
            context.model = None
            context.llm_provider = None
            context.huggingface_endpoint_url = None
            context.image_provider = None

        context.image_provider = provider
        
        # Apply LLM provider and model from style config if provided
        if style_config:
            llm_provider_str = style_config.get("llm_provider")
            if llm_provider_str:
                try:
                    context.llm_provider = LLMProvider(llm_provider_str)
                except ValueError:
                    pass  # Keep default if invalid
            
            llm_model = style_config.get("llm_model")
            if llm_model:
                context.model = llm_model
                # If model has provider prefix, extract provider
                if "/" in llm_model:
                    provider_prefix = llm_model.split("/", 1)[0]
                    try:
                        context.llm_provider = LLMProvider(provider_prefix)
                        context.model = llm_model.split("/", 1)[1]
                    except ValueError:
                        pass  # Keep original model if provider prefix is invalid

        # Set Flux Dev Vertex endpoint URL from style config if provided
        if style_config and provider.value == "flux_dev_vertex":
            flux_endpoint_url = style_config.get("flux_dev_vertex_endpoint_url")
            if flux_endpoint_url:
                # Set as environment variable so the provider can access it
                os.environ["FLUX_DEV_VERTEX_ENDPOINT_URL"] = flux_endpoint_url
                # Also set on context if it has the attribute
                if hasattr(context, 'flux_dev_vertex_endpoint_url'):
                    context.flux_dev_vertex_endpoint_url = flux_endpoint_url

        if not getattr(context, 'model', None):
            context.model = "gpt-oss-120b"

        try:
            if getattr(context, 'llm_provider', None) == getattr(LLMProvider, 'HUGGINGFACE', None) and not getattr(context, 'huggingface_endpoint_url', None):
                context.huggingface_endpoint_url = f"https://api-inference.huggingface.co/models/{context.model}"
        except Exception:
            # If LLMProvider is None or comparison fails, ignore
            pass

        self.generator = IllustrationGenerator(provider, output_dir, context) if IllustrationGenerator is not None else None
        self.connection_manager = connection_manager
        self.session_id = session_id

        # Initialize the advanced prompt engineering system
        try:
            if create_chat_model_from_context:
                self.llm = create_chat_model_from_context(context, session_id=session_id)
            else:
                self.llm = None
        except Exception as exc:  # pragma: no cover - surfaced in UI logs
            raise RuntimeError(f"Failed to initialize prompt engineering model: {exc}") from exc

        # Use a patched PromptEngineer class on the module when tests set it, otherwise import the real one
        if PromptEngineerCls is None:
            try:
                from illustrator.prompt_engineering import PromptEngineer as _PromptEngineer
                PromptEngineerCls = _PromptEngineer
            except Exception:
                PromptEngineerCls = None

        self.prompt_engineer = PromptEngineerCls(self.llm) if PromptEngineerCls is not None else None

    async def create_advanced_prompt(self, emotional_moment, provider, style_config, chapter):
        """Create an advanced AI-analyzed prompt for image generation."""
        illustration_prompt = None

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
                style_modifiers = illustration_prompt.style_modifiers or []
                for modifier in style_modifiers[:3]:
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

            prompt_text = self._add_flux_style_tags(illustration_prompt.prompt, provider)
            return prompt_text

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

            fallback_prompt = self._create_fallback_prompt(
                emotional_moment.text_excerpt,
                emotional_moment.emotional_tones[0] if emotional_moment.emotional_tones else EmotionalTone.NEUTRAL,
                chapter.title,
                style_config.get("art_style", "digital painting")
            )
            return self._add_flux_style_tags(fallback_prompt, provider)

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

    def _add_flux_style_tags(self, prompt_text: str | None, provider: ImageProvider) -> str | None:
        """Boost Flux prompts with explicit line-art tags for stronger style anchoring."""
        if not prompt_text or provider not in (ImageProvider.FLUX, ImageProvider.FLUX_DEV_VERTEX):
            return prompt_text

        updated = prompt_text.strip()

        prefix = "Line drawing, pen-and-ink illustration in the style of E.H. Shepard."
        if prefix.lower() not in updated.lower():
            updated = f"{prefix} {updated}" if updated else prefix

        suffix = "Detailed line art, vintage children's book illustration style, monochrome."
        if suffix.lower() not in updated.lower():
            if updated and not updated.strip().endswith(tuple(".!?")):
                updated = updated.rstrip() + "."
            updated = f"{updated} {suffix}" if updated else suffix

        return updated

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
                    elif 'image_data' in result:
                        try:
                            image_data = result['image_data']
                            if isinstance(image_data, str):
                                image_bytes = base64.b64decode(image_data)
                            else:
                                image_bytes = image_data
                            with open(file_path, 'wb') as f:
                                f.write(image_bytes)
                        except Exception as decode_error:
                            await self.connection_manager.send_personal_message(
                                json.dumps({
                                    "type": "log",
                                    "level": "warning",
                                    "message": f"⚠️ Failed to decode Imagen data for image {i+1}: {decode_error}"
                                }),
                                self.session_id
                            )
                            file_path = None
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


@app.websocket("/ws/gallery/{manuscript_id}")
async def gallery_websocket(websocket: WebSocket, manuscript_id: str):
    """WebSocket endpoint for real-time gallery updates."""
    await connection_manager.connect(websocket, f"gallery_{manuscript_id}")
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Handle gallery-specific messages if needed
            if data == "ping":
                await connection_manager.send_personal_message("pong", f"gallery_{manuscript_id}")
    except WebSocketDisconnect:
        connection_manager.disconnect(f"gallery_{manuscript_id}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "manuscript-illustrator"}


@app.get("/api/sessions/active")
async def get_active_sessions():
    """Get all active processing sessions."""
    active_sessions = []
    for session_id, session_data in connection_manager.sessions.items():
        if session_data.status.status not in ['completed', 'error']:
            active_sessions.append({
                "session_id": session_id,
                "manuscript_id": session_data.manuscript_id,
                "status": session_data.status.status,
                "progress": session_data.status.progress,
                "message": session_data.status.message,
                "started_at": getattr(session_data, 'started_at', None)
            })
    return active_sessions


@app.get("/api/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get detailed status for a specific session."""
    if session_id not in connection_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = connection_manager.sessions[session_id]
    return {
        "session_id": session_id,
        "manuscript_id": session_data.manuscript_id,
        "status": session_data.status.status,
        "progress": session_data.status.progress,
        "message": session_data.status.message,
        "current_chapter": session_data.status.current_chapter,
        "total_chapters": session_data.status.total_chapters,
        "step_status": getattr(session_data, 'step_status', {}),
        "logs": [{
            "level": log.level,
            "message": log.message,
            "timestamp": log.timestamp
        } for log in session_data.logs[-20:]],  # Last 20 logs
        "images": [{
            "url": img.url,
            "prompt": img.prompt,
            "chapter_number": img.chapter_number,
            "scene_number": img.scene_number,
            "timestamp": img.timestamp
        } for img in session_data.images],
        "pause_requested": getattr(session_data, 'pause_requested', False)
    }


@app.post("/api/sessions/{session_id}/attach")
async def attach_to_session(session_id: str):
    """Attach to an existing session for monitoring."""
    if session_id not in connection_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_data = connection_manager.sessions[session_id]
    return {
        "success": True,
        "session_id": session_id,
        "manuscript_id": session_data.manuscript_id,
        "message": "Successfully attached to session"
    }


@app.delete("/api/sessions/{session_id}")
async def terminate_session(session_id: str):
    """Terminate a processing session."""
    if session_id not in connection_manager.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Mark session for termination
    session_data = connection_manager.sessions[session_id]
    session_data.pause_requested = True
    session_data.status.status = "terminated"
    session_data.status.message = "Session terminated by user"
    
    # Send termination message
    await connection_manager.send_personal_message(
        json.dumps({
            "type": "log",
            "level": "warning",
            "message": "Session terminated by user request"
        }),
        session_id
    )
    
    # Clean up after a delay to allow message delivery
    async def cleanup_session():
        await asyncio.sleep(2)
        connection_manager.cleanup_session(session_id)
    
    asyncio.create_task(cleanup_session())
    
    return {"success": True, "message": "Session terminated"}


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


def create_api_only_app() -> FastAPI:
    """Create API-only FastAPI application (no web interface)."""
    from fastapi.openapi.docs import get_swagger_ui_html
    from fastapi.openapi.utils import get_openapi

    # Create a new FastAPI app for API only
    api_app = FastAPI(
        title="Manuscript Illustrator API",
        description="AI-powered manuscript analysis and illustration generation API",
        version="1.0.0",
    )

    # Ensure database tables exist on startup for API-only mode
    @api_app.on_event("startup")
    async def _ensure_tables_api_only():
        try:
            create_tables()
            console.log("[api-only] Database tables ensured/created")
        except Exception as e:
            console.log(f"[api-only] Failed to create tables: {e}")

    # Add CORS middleware
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add API key authentication middleware if configured
    api_keys = _get_valid_api_keys()
    if api_keys:
        @api_app.middleware("http")
        async def authenticate_api_key(request: Request, call_next):
            # Skip authentication for health check and docs
            if request.url.path in ["/health", "/docs", "/openapi.json"]:
                response = await call_next(request)
                return response

            # Check for API key in header
            provided_key = request.headers.get("X-API-Key") or request.headers.get("Authorization", "").replace("Bearer ", "")

            if not provided_key or provided_key not in api_keys:
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"}
                )

            response = await call_next(request)
            return response

    # Include API routes only
    api_app.include_router(manuscripts.router, prefix="/api/manuscripts", tags=["manuscripts"])
    api_app.include_router(chapters.router, prefix="/api/chapters", tags=["chapters"])

    # Add processing endpoints
    api_app.get("/api/process/status/{manuscript_id}")(get_processing_status)
    api_app.post("/api/process")(start_processing)
    api_app.post("/api/process/resume/{session_id}")(resume_processing_from_checkpoint)
    api_app.get("/api/process/resumable")(get_resumable_sessions)
    api_app.post("/api/process/{session_id}/pause")(pause_processing)
    api_app.post("/api/process/{session_id}/resume")(resume_processing)

    # Health check
    @api_app.get("/health")
    async def health_check():
        return {"status": "healthy", "service": "manuscript-illustrator-api", "mode": "api-only"}

    return api_app


def create_web_client_app() -> FastAPI:
    """Create web client FastAPI application (connects to remote API)."""
    import httpx
    from fastapi.responses import HTMLResponse, RedirectResponse

    # Create a new FastAPI app for web client
    web_app = FastAPI(
        title="Manuscript Illustrator Web Client",
        description="Web interface for Manuscript Illustrator (connects to remote API)",
        version="1.0.0",
    )

    # Add CORS middleware
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Get remote API configuration
    remote_api_url = os.getenv('ILLUSTRATOR_REMOTE_API_URL', 'http://127.0.0.1:8000')
    api_key = os.getenv('ILLUSTRATOR_API_KEY')
    mongo_url = os.getenv('MONGODB_URI') or os.getenv('MONGO_URL', 'mongodb://localhost:27017')
    mongo_db_name = os.getenv('MONGO_DB_NAME', 'illustrator')
    mongo_use_mock = os.getenv('MONGO_USE_MOCK', '').lower() in {'1', 'true', 'yes', 'on'}

    # Mount static files (same as main app)
    web_app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Templates
    web_templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    # Add global functions to templates
    web_templates.env.globals["app_version"] = _current_app_version

    # Create HTTP client for API requests
    async def get_api_client():
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        return httpx.AsyncClient(base_url=remote_api_url, headers=headers, timeout=30.0)

    # Client setup endpoint for saving configuration
    @web_app.post("/api/client-setup")
    async def save_client_setup(request: Request):
        nonlocal remote_api_url, api_key, mongo_url, mongo_db_name, mongo_use_mock
        try:
            data = await request.json()
            server_url = (data.get('server_url') or '').strip()
            new_api_key = (data.get('api_key') or '').strip()
            mongo_url_value = (data.get('mongo_url') or '').strip()
            mongo_db_name_value = (data.get('mongo_db_name') or '').strip()
            mongo_use_mock_raw = data.get('mongo_use_mock')
            if isinstance(mongo_use_mock_raw, bool):
                mongo_use_mock_value = mongo_use_mock_raw
            elif isinstance(mongo_use_mock_raw, str):
                mongo_use_mock_value = mongo_use_mock_raw.lower() in {'1', 'true', 'yes', 'on'}
            else:
                mongo_use_mock_value = False
            
            if not server_url:
                raise HTTPException(status_code=400, detail="Server URL is required")
            
            if not server_url.startswith(('http://', 'https://')):
                raise HTTPException(status_code=400, detail="Server URL must start with http:// or https://")
            
            # Find or create .env file
            from dotenv import find_dotenv, set_key, unset_key
            env_file = find_dotenv()
            if not env_file:
                env_file = Path.cwd() / '.env'
            
            # Update configuration
            set_key(env_file, 'ILLUSTRATOR_REMOTE_API_URL', server_url)
            os.environ['ILLUSTRATOR_REMOTE_API_URL'] = server_url
            remote_api_url = server_url
            if new_api_key:
                set_key(env_file, 'ILLUSTRATOR_API_KEY', new_api_key)
                os.environ['ILLUSTRATOR_API_KEY'] = new_api_key
                api_key = new_api_key
            else:
                unset_key(env_file, 'ILLUSTRATOR_API_KEY')
                os.environ.pop('ILLUSTRATOR_API_KEY', None)
                api_key = None

            if mongo_url_value:
                set_key(env_file, 'MONGO_URL', mongo_url_value)
                set_key(env_file, 'MONGODB_URI', mongo_url_value)
                os.environ['MONGO_URL'] = mongo_url_value
                os.environ['MONGODB_URI'] = mongo_url_value
                mongo_url = mongo_url_value
            else:
                unset_key(env_file, 'MONGO_URL')
                unset_key(env_file, 'MONGODB_URI')
                os.environ.pop('MONGO_URL', None)
                os.environ.pop('MONGODB_URI', None)
                mongo_url = 'mongodb://localhost:27017'

            if mongo_db_name_value:
                set_key(env_file, 'MONGO_DB_NAME', mongo_db_name_value)
                os.environ['MONGO_DB_NAME'] = mongo_db_name_value
                mongo_db_name = mongo_db_name_value
            else:
                unset_key(env_file, 'MONGO_DB_NAME')
                os.environ.pop('MONGO_DB_NAME', None)
                mongo_db_name = 'illustrator'

            if mongo_use_mock_value:
                set_key(env_file, 'MONGO_USE_MOCK', 'true')
                os.environ['MONGO_USE_MOCK'] = 'true'
                mongo_use_mock = True
            else:
                unset_key(env_file, 'MONGO_USE_MOCK')
                os.environ.pop('MONGO_USE_MOCK', None)
                mongo_use_mock = False
            
            return {"success": True, "message": "Configuration saved successfully"}
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Check if client needs setup
    def needs_client_setup():
        # Check if no remote API URL is configured or it's the default localhost
        if not remote_api_url or remote_api_url in ['http://127.0.0.1:8000', 'http://localhost:8000']:
            return True
        
        # Additional check: try to reach the remote API quickly
        try:
            import httpx
            with httpx.Client(timeout=2.0) as client:
                response = client.get(f"{remote_api_url}/health")
                return response.status_code != 200
        except Exception:
            # If we can't reach it, assume setup is needed
            return True

    # Proxy API requests to remote server
    @web_app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy_api_request(request: Request, path: str):
        async with await get_api_client() as client:
            try:
                # Forward the request to the remote API
                url = f"/api/{path}"
                query_params = str(request.query_params)
                if query_params:
                    url += f"?{query_params}"

                # Get request body if present
                body = None
                if request.method in ["POST", "PUT", "PATCH"]:
                    body = await request.body()

                # Make the request to remote API
                response = await client.request(
                    method=request.method,
                    url=url,
                    content=body,
                    headers={k: v for k, v in request.headers.items()
                            if k.lower() not in ['host', 'content-length']},
                )

                # Return the response
                return JSONResponse(
                    content=response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                    status_code=response.status_code,
                    headers=dict(response.headers)
                )
            except httpx.RequestError as e:
                return JSONResponse(
                    status_code=503,
                    content={"detail": f"Remote API unavailable: {str(e)}"}
                )
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"detail": f"Proxy error: {str(e)}"}
                )

    # WebSocket proxy for real-time updates
    @web_app.websocket("/ws/processing/{session_id}")
    async def websocket_proxy(websocket: WebSocket, session_id: str):
        await websocket.accept()

        try:
            # Try to connect to remote WebSocket
            import websockets
            remote_ws_url = remote_api_url.replace('http://', 'ws://').replace('https://', 'wss://')
            remote_ws_url += f"/ws/processing/{session_id}"

            async with websockets.connect(remote_ws_url) as remote_ws:
                # Forward messages between client and remote server
                async def forward_from_remote():
                    try:
                        async for message in remote_ws:
                            await websocket.send_text(message)
                    except websockets.exceptions.ConnectionClosed:
                        pass

                async def forward_to_remote():
                    try:
                        while True:
                            message = await websocket.receive_text()
                            await remote_ws.send(message)
                    except Exception:
                        pass

                # Run both forwarding tasks concurrently
                import asyncio
                await asyncio.gather(
                    forward_from_remote(),
                    forward_to_remote(),
                    return_exceptions=True
                )
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": f"WebSocket connection to remote server failed: {str(e)}"
            }))
        finally:
            await websocket.close()

    # WebSocket proxy for gallery updates
    @web_app.websocket("/ws/gallery/{manuscript_id}")
    async def gallery_websocket_proxy(websocket: WebSocket, manuscript_id: str):
        await websocket.accept()

        try:
            # Try to connect to remote WebSocket
            import websockets
            remote_ws_url = remote_api_url.replace('http://', 'ws://').replace('https://', 'wss://')
            remote_ws_url += f"/ws/gallery/{manuscript_id}"

            async with websockets.connect(remote_ws_url) as remote_ws:
                # Forward messages between client and remote server
                async def forward_from_remote():
                    try:
                        async for message in remote_ws:
                            await websocket.send_text(message)
                    except websockets.exceptions.ConnectionClosed:
                        pass

                async def forward_to_remote():
                    try:
                        while True:
                            message = await websocket.receive_text()
                            await remote_ws.send(message)
                    except Exception:
                        pass

                # Run both forwarding tasks concurrently
                import asyncio
                await asyncio.gather(
                    forward_from_remote(),
                    forward_to_remote(),
                    return_exceptions=True
                )
        except Exception as e:
            await websocket.send_text(json.dumps({
                "type": "error",
                "error": f"WebSocket connection to remote server failed: {str(e)}"
            }))
        finally:
            await websocket.close()

    # Serve HTML pages (same as main app)
    @web_app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        return web_templates.TemplateResponse(
            request,
            "index.html",
            {
                "title": "Dashboard", 
                "remote_mode": True, 
                "remote_api_url": remote_api_url,
                "needs_setup": needs_client_setup(),
                "mongo_url": mongo_url,
                "mongo_db_name": mongo_db_name,
                "mongo_use_mock": mongo_use_mock,
            }
        )

    @web_app.get("/manuscript/new", response_class=HTMLResponse)
    async def new_manuscript(request: Request):
        return web_templates.TemplateResponse(
            request,
            "manuscript_form.html",
            {"title": "New Manuscript", "manuscript": None, "remote_mode": True}
        )

    @web_app.get("/manuscript/{manuscript_id}", response_class=HTMLResponse)
    async def manuscript_detail(request: Request, manuscript_id: str):
        return web_templates.TemplateResponse(
            request,
            "manuscript_detail.html",
            {"title": "Manuscript", "manuscript_id": manuscript_id, "remote_mode": True}
        )

    @web_app.get("/manuscript/{manuscript_id}/edit", response_class=HTMLResponse)
    async def edit_manuscript(request: Request, manuscript_id: str):
        return web_templates.TemplateResponse(
            request,
            "manuscript_form.html",
            {"title": "Edit Manuscript", "manuscript_id": manuscript_id, "remote_mode": True}
        )

    @web_app.get("/manuscript/{manuscript_id}/chapter/new", response_class=HTMLResponse)
    async def new_chapter(request: Request, manuscript_id: str):
        return web_templates.TemplateResponse(
            request,
            "chapter_form.html",
            {"title": "New Chapter", "manuscript_id": manuscript_id, "chapter": None, "remote_mode": True}
        )

    @web_app.get("/chapter/{chapter_id}/edit", response_class=HTMLResponse)
    async def edit_chapter(request: Request, chapter_id: str):
        return web_templates.TemplateResponse(
            request,
            "chapter_form.html",
            {"title": "Edit Chapter", "chapter_id": chapter_id, "remote_mode": True}
        )

    @web_app.get("/manuscript/{manuscript_id}/style", response_class=HTMLResponse)
    async def style_config(request: Request, manuscript_id: str):
        return web_templates.TemplateResponse(
            request,
            "style_config.html",
            {"title": "Style Configuration", "manuscript_id": manuscript_id, "remote_mode": True}
        )

    @web_app.get("/manuscript/{manuscript_id}/process", response_class=HTMLResponse)
    async def processing_page(request: Request, manuscript_id: str):
        return web_templates.TemplateResponse(
            request,
            "processing.html",
            {"title": "Processing", "manuscript_id": manuscript_id, "remote_mode": True}
        )

    @web_app.get("/chapter/{chapter_id}/headers", response_class=HTMLResponse)
    async def chapter_headers_page(request: Request, chapter_id: str):
        return web_templates.TemplateResponse(
            request,
            "chapter_headers.html",
            {"title": "Chapter Headers", "chapter_id": chapter_id, "remote_mode": True}
        )

    @web_app.get("/manuscript/{manuscript_id}/gallery", response_class=HTMLResponse)
    async def gallery_page(request: Request, manuscript_id: str):
        return web_templates.TemplateResponse(
            request,
            "gallery.html",
            {"title": "Gallery", "manuscript_id": manuscript_id, "remote_mode": True}
        )

    # Health check
    @web_app.get("/health")
    async def health_check():
        # Also check remote API health
        try:
            async with await get_api_client() as client:
                response = await client.get("/health")
                remote_healthy = response.status_code == 200
        except Exception:
            remote_healthy = False

        return {
            "status": "healthy" if remote_healthy else "degraded",
            "service": "manuscript-illustrator-web-client",
            "mode": "web-client",
            "remote_api_url": remote_api_url,
            "remote_api_healthy": remote_healthy
        }

    return web_app


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
