"""State management for the manuscript illustration workflow."""

from typing import Annotated, Any, Dict, List

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from illustrator.models import (
    Chapter,
    ChapterAnalysis,
    ImageProvider,
    ManuscriptMetadata,
)


class ManuscriptState(TypedDict):
    """State for the manuscript illustration workflow."""

    # Core conversation messages
    messages: Annotated[List[BaseMessage], add_messages]

    # Manuscript data
    manuscript_metadata: ManuscriptMetadata | None
    current_chapter: Chapter | None
    chapters_completed: List[ChapterAnalysis]

    # Processing state
    awaiting_chapter_input: bool
    processing_complete: bool
    illustrations_generated: bool

    # Configuration
    image_provider: ImageProvider
    style_preferences: Dict[str, Any]
    analysis_depth: str  # "basic", "detailed", "comprehensive"

    # Current analysis and generated content
    current_analysis: ChapterAnalysis | None
    generated_images: List[Dict[str, Any]]

    # Error handling
    error_message: str | None
    retry_count: int