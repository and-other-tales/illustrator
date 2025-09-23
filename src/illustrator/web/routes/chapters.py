"""API routes for chapter management."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from illustrator.models import Chapter, SavedManuscript
from illustrator.web.models.web_models import (
    ChapterCreateRequest,
    ChapterResponse,
    ChapterHeaderResponse,
    ChapterHeaderOptionResponse,
    SuccessResponse,
    ErrorResponse
)

router = APIRouter()

# Storage paths
SAVED_MANUSCRIPTS_DIR = Path("saved_manuscripts")


def load_manuscript_by_id(manuscript_id: str) -> tuple[SavedManuscript, Path]:
    """Load a manuscript by its ID."""
    if not SAVED_MANUSCRIPTS_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No manuscripts found"
        )

    for file_path in SAVED_MANUSCRIPTS_DIR.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            manuscript = SavedManuscript(**data)

            # Check if this matches the requested ID
            generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path)))
            if generated_id == manuscript_id:
                return manuscript, file_path
        except Exception as e:
            print(f"Error loading manuscript {file_path}: {e}")
            continue

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Manuscript not found"
    )


def save_manuscript(manuscript: SavedManuscript, file_path: Path):
    """Save manuscript to disk."""
    manuscript.saved_at = datetime.now().isoformat()
    manuscript.file_path = str(file_path)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(manuscript.model_dump(), f, indent=2, ensure_ascii=False)


@router.get("/{manuscript_id}")
async def get_manuscript_chapters(manuscript_id: str) -> List[ChapterResponse]:
    """Get all chapters for a manuscript."""
    manuscript, _ = load_manuscript_by_id(manuscript_id)

    chapters = []
    for i, chapter in enumerate(manuscript.chapters):
        chapters.append(ChapterResponse(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chapter.number}")),
            chapter=chapter,
            analysis=None,  # TODO: Load analysis if available
            images_generated=0,  # TODO: Count generated images
            processing_status="draft"
        ))

    return chapters


@router.get("/detail/{chapter_id}")
async def get_chapter(chapter_id: str) -> ChapterResponse:
    """Get a specific chapter by ID."""
    # Parse chapter ID to get manuscript ID and chapter number
    # Format: uuid5(manuscript_id_chapter_number)

    # For now, we'll search through all manuscripts
    # In production, you'd want a more efficient lookup system
    if not SAVED_MANUSCRIPTS_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chapter not found"
        )

    for file_path in SAVED_MANUSCRIPTS_DIR.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            manuscript = SavedManuscript(**data)
            manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path)))

            for chapter in manuscript.chapters:
                generated_chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chapter.number}"))
                if generated_chapter_id == chapter_id:
                    return ChapterResponse(
                        id=chapter_id,
                        chapter=chapter,
                        analysis=None,  # TODO: Load analysis if available
                        images_generated=0,  # TODO: Count generated images
                        processing_status="draft"
                    )
        except Exception as e:
            print(f"Error loading manuscript {file_path}: {e}")
            continue

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Chapter not found"
    )


@router.post("/{manuscript_id}")
async def add_chapter(
    manuscript_id: str,
    request: ChapterCreateRequest
) -> ChapterResponse:
    """Add a new chapter to a manuscript."""
    manuscript, file_path = load_manuscript_by_id(manuscript_id)

    # Determine chapter number
    chapter_number = len(manuscript.chapters) + 1

    # Create new chapter
    new_chapter = Chapter(
        title=request.title,
        content=request.content,
        number=chapter_number,
        word_count=len(request.content.split())
    )

    # Add to manuscript
    manuscript.chapters.append(new_chapter)
    manuscript.metadata.total_chapters = len(manuscript.chapters)

    # Save manuscript
    save_manuscript(manuscript, file_path)

    # Generate chapter ID
    chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chapter_number}"))

    return ChapterResponse(
        id=chapter_id,
        chapter=new_chapter,
        analysis=None,
        images_generated=0,
        processing_status="draft"
    )


@router.put("/{chapter_id}")
async def update_chapter(
    chapter_id: str,
    request: ChapterCreateRequest
) -> ChapterResponse:
    """Update an existing chapter."""
    # Find the manuscript and chapter
    if not SAVED_MANUSCRIPTS_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chapter not found"
        )

    for file_path in SAVED_MANUSCRIPTS_DIR.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            manuscript = SavedManuscript(**data)
            manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path)))

            for i, chapter in enumerate(manuscript.chapters):
                generated_chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chapter.number}"))
                if generated_chapter_id == chapter_id:
                    # Update chapter
                    manuscript.chapters[i].title = request.title
                    manuscript.chapters[i].content = request.content
                    manuscript.chapters[i].word_count = len(request.content.split())

                    # Save manuscript
                    save_manuscript(manuscript, file_path)

                    return ChapterResponse(
                        id=chapter_id,
                        chapter=manuscript.chapters[i],
                        analysis=None,
                        images_generated=0,
                        processing_status="draft"
                    )
        except Exception as e:
            print(f"Error loading manuscript {file_path}: {e}")
            continue

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Chapter not found"
    )


@router.delete("/{chapter_id}")
async def delete_chapter(chapter_id: str) -> SuccessResponse:
    """Delete a chapter."""
    # Find the manuscript and chapter
    if not SAVED_MANUSCRIPTS_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chapter not found"
        )

    for file_path in SAVED_MANUSCRIPTS_DIR.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            manuscript = SavedManuscript(**data)
            manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path)))

            for i, chapter in enumerate(manuscript.chapters):
                generated_chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chapter.number}"))
                if generated_chapter_id == chapter_id:
                    # Remove chapter
                    del manuscript.chapters[i]

                    # Renumber remaining chapters
                    for j, remaining_chapter in enumerate(manuscript.chapters[i:], start=i):
                        remaining_chapter.number = j + 1

                    # Update total chapters count
                    manuscript.metadata.total_chapters = len(manuscript.chapters)

                    # Save manuscript
                    save_manuscript(manuscript, file_path)

                    return SuccessResponse(
                        message="Chapter deleted successfully"
                    )
        except Exception as e:
            print(f"Error loading manuscript {file_path}: {e}")
            continue

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Chapter not found"
    )


@router.post("/{manuscript_id}/reorder")
async def reorder_chapters(
    manuscript_id: str,
    chapter_order: List[str]
) -> SuccessResponse:
    """Reorder chapters in a manuscript."""
    manuscript, file_path = load_manuscript_by_id(manuscript_id)

    # Create a mapping of chapter IDs to chapters
    chapter_map = {}
    for chapter in manuscript.chapters:
        chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chapter.number}"))
        chapter_map[chapter_id] = chapter

    # Reorder chapters based on the provided order
    reordered_chapters = []
    for i, chapter_id in enumerate(chapter_order):
        if chapter_id in chapter_map:
            chapter = chapter_map[chapter_id]
            chapter.number = i + 1  # Renumber
            reordered_chapters.append(chapter)

    # Update manuscript
    manuscript.chapters = reordered_chapters
    manuscript.metadata.total_chapters = len(reordered_chapters)

    # Save manuscript
    save_manuscript(manuscript, file_path)

    return SuccessResponse(
        message="Chapters reordered successfully"
    )


@router.post("/{chapter_id}/headers")
async def generate_chapter_headers(
    chapter_id: str,
    style_config: dict = None
) -> ChapterHeaderResponse:
    """Generate 4 header illustration options for a chapter."""

    # Find the chapter
    if not SAVED_MANUSCRIPTS_DIR.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chapter not found"
        )

    chapter = None
    chapter_title = ""

    for file_path in SAVED_MANUSCRIPTS_DIR.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            manuscript = SavedManuscript(**data)
            manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path)))

            for chap in manuscript.chapters:
                generated_chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chap.number}"))
                if generated_chapter_id == chapter_id:
                    chapter = chap
                    chapter_title = chap.title
                    break

            if chapter:
                break

        except Exception as e:
            print(f"Error loading manuscript {file_path}: {e}")
            continue

    if not chapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chapter not found"
        )

    try:
        # Initialize the advanced PromptEngineer system for chapter header generation
        import os
        from langchain.chat_models import init_chat_model
        from illustrator.prompt_engineering import PromptEngineer
        from illustrator.models import ImageProvider

        # Check for required API key
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not anthropic_api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Anthropic API key is required for advanced prompt engineering"
            )

        # Initialize LLM and PromptEngineer
        llm = init_chat_model(
            model="anthropic/claude-3-5-sonnet-20241022",
            api_key=anthropic_api_key
        )
        prompt_engineer = PromptEngineer(llm)

        # Set default style preferences if not provided
        if not style_config:
            style_config = {
                'art_style': 'digital illustration',
                'color_palette': 'harmonious',
                'artistic_influences': None
            }

        # Generate sophisticated chapter headers using full content analysis
        provider = ImageProvider.DALLE  # Default provider for web interface
        chapter_header_options = await prompt_engineer.generate_chapter_header_options(
            chapter=chapter,
            style_preferences=style_config,
            provider=provider
        )

        # Convert to response format
        header_options = []
        for option in chapter_header_options:
            option_response = ChapterHeaderOptionResponse(
                option_number=option.option_number,
                title=option.title,
                description=option.description,
                visual_focus=option.visual_focus,
                artistic_style=option.artistic_style,
                composition_notes=option.composition_notes,
                prompt=option.prompt
            )
            header_options.append(option_response)

        return ChapterHeaderResponse(
            chapter_id=chapter_id,
            chapter_title=chapter_title,
            header_options=header_options
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate chapter headers: {str(e)}"
        )