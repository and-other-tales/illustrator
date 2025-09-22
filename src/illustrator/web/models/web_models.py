"""Web-specific Pydantic models for the FastAPI application."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import WebSocket

from illustrator.models import (
    Chapter,
    ChapterAnalysis,
    ManuscriptMetadata,
    SavedManuscript,
    ImageProvider,
    EmotionalTone,
    IllustrationPrompt
)


class ManuscriptCreateRequest(BaseModel):
    """Request model for creating a new manuscript."""
    title: str = Field(min_length=1, max_length=200)
    author: Optional[str] = Field(default=None, max_length=100)
    genre: Optional[str] = Field(default=None, max_length=50)


class ManuscriptResponse(BaseModel):
    """Response model for manuscript data."""
    id: str
    metadata: ManuscriptMetadata
    chapters: List[Chapter]
    total_images: int = 0
    processing_status: str = "draft"  # draft, processing, completed, error
    created_at: str
    updated_at: str


class ChapterCreateRequest(BaseModel):
    """Request model for creating a new chapter."""
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=10)
    manuscript_id: str


class ChapterResponse(BaseModel):
    """Response model for chapter data."""
    id: str
    chapter: Chapter
    analysis: Optional[ChapterAnalysis] = None
    images_generated: int = 0
    processing_status: str = "draft"


class StyleConfigRequest(BaseModel):
    """Request model for style configuration."""
    image_provider: ImageProvider
    art_style: str = "digital painting"
    color_palette: Optional[str] = None
    artistic_influences: Optional[str] = None
    style_config_path: Optional[str] = None


class ProcessingRequest(BaseModel):
    """Request model to start processing a manuscript."""
    manuscript_id: str
    style_config: StyleConfigRequest
    max_emotional_moments: int = Field(default=10, ge=1, le=20)


class ProcessingStatus(BaseModel):
    """Model for processing status updates."""
    session_id: str
    manuscript_id: str
    status: str  # started, analyzing, generating, completed, error
    progress: int  # 0-100
    current_chapter: Optional[int] = None
    total_chapters: int
    message: str
    error: Optional[str] = None


class ImageResponse(BaseModel):
    """Response model for generated images."""
    id: str
    chapter_number: int
    scene_number: int
    image_path: str
    thumbnail_path: Optional[str] = None
    prompt: str
    emotional_moment: str
    quality_scores: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any]
    generated_at: str


class ChapterHeaderOptionResponse(BaseModel):
    """Response model for chapter header option."""
    option_number: int
    title: str
    description: str
    visual_focus: str
    artistic_style: str
    composition_notes: str
    prompt: IllustrationPrompt


class ChapterHeaderResponse(BaseModel):
    """Response model for chapter header options."""
    chapter_id: str
    chapter_title: str
    header_options: List[ChapterHeaderOptionResponse]


class GalleryResponse(BaseModel):
    """Response model for image gallery."""
    manuscript_id: str
    manuscript_title: str
    total_images: int
    images_by_chapter: Dict[str, List[ImageResponse]]


class ProcessingSessionData(BaseModel):
    """Data structure for tracking processing sessions."""
    session_id: str
    manuscript_id: str
    websocket: Optional[WebSocket] = None
    status: ProcessingStatus

    class Config:
        arbitrary_types_allowed = True


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.sessions: Dict[str, ProcessingSessionData] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[session_id] = websocket

    def disconnect(self, session_id: str):
        """Remove a WebSocket connection."""
        if session_id in self.active_connections:
            del self.active_connections[session_id]
        if session_id in self.sessions:
            del self.sessions[session_id]

    async def send_personal_message(self, message: str, session_id: str):
        """Send a message to a specific connection."""
        if session_id in self.active_connections:
            websocket = self.active_connections[session_id]
            await websocket.send_text(message)

    async def send_status_update(self, status: ProcessingStatus):
        """Send a status update to the appropriate connection."""
        if status.session_id in self.active_connections:
            websocket = self.active_connections[status.session_id]
            await websocket.send_json(status.dict())

    async def broadcast_message(self, message: str):
        """Send a message to all active connections."""
        for connection in self.active_connections.values():
            await connection.send_text(message)


class DashboardStats(BaseModel):
    """Dashboard statistics model."""
    total_manuscripts: int
    total_chapters: int
    total_images: int
    recent_manuscripts: List[ManuscriptResponse]
    processing_count: int


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None


class SuccessResponse(BaseModel):
    """Standard success response model."""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None