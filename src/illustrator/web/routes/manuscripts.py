"""API routes for manuscript management."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from illustrator.models import ManuscriptMetadata, SavedManuscript, Chapter
from illustrator.web.models.web_models import (
    ManuscriptCreateRequest,
    ManuscriptResponse,
    DashboardStats,
    ErrorResponse,
    SuccessResponse,
    StyleConfigSaveRequest
)

router = APIRouter()

# Storage paths
SAVED_MANUSCRIPTS_DIR = Path("saved_manuscripts")
ILLUSTRATOR_OUTPUT_DIR = Path("illustrator_output")

# Ensure directories exist
SAVED_MANUSCRIPTS_DIR.mkdir(exist_ok=True)
ILLUSTRATOR_OUTPUT_DIR.mkdir(exist_ok=True)


def generate_manuscript_id() -> str:
    """Generate a unique manuscript ID."""
    return str(uuid.uuid4())


def get_saved_manuscripts() -> List[SavedManuscript]:
    """Load all saved manuscripts from disk."""
    manuscripts = []

    if not SAVED_MANUSCRIPTS_DIR.exists():
        return manuscripts

    for file_path in SAVED_MANUSCRIPTS_DIR.glob("*.json"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            manuscript = SavedManuscript(**data)
            manuscripts.append(manuscript)
        except Exception as e:
            print(f"Error loading manuscript {file_path}: {e}")
            continue

    # Sort by saved date (newest first)
    manuscripts.sort(key=lambda m: m.saved_at, reverse=True)
    return manuscripts


def save_manuscript_to_disk(manuscript: SavedManuscript) -> Path:
    """Save a manuscript to disk."""
    # Generate safe filename
    safe_title = "".join(c for c in manuscript.metadata.title if c.isalnum() or c in (' ', '-', '_')).strip()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{safe_title}_{timestamp}.json"
    file_path = SAVED_MANUSCRIPTS_DIR / filename

    # Update file path in manuscript
    manuscript.file_path = str(file_path)

    # Save to file
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(manuscript.model_dump(), f, indent=2, ensure_ascii=False)

    return file_path


def count_generated_images(manuscript_title: str) -> int:
    """Count generated images for a manuscript."""
    safe_title = manuscript_title.replace(" ", "_")
    images_dir = ILLUSTRATOR_OUTPUT_DIR / safe_title / "generated_images"

    if not images_dir.exists():
        return 0

    image_count = 0
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}

    for chapter_dir in images_dir.iterdir():
        if chapter_dir.is_dir():
            image_count += len([f for f in chapter_dir.iterdir()
                               if f.suffix.lower() in image_extensions])

    return image_count


@router.get("/stats")
async def get_dashboard_stats() -> DashboardStats:
    """Get dashboard statistics."""
    manuscripts = get_saved_manuscripts()

    total_chapters = sum(len(m.chapters) for m in manuscripts)
    total_images = sum(count_generated_images(m.metadata.title) for m in manuscripts)
    processing_count = 0  # TODO: Track active processing sessions

    # Get recent manuscripts (last 10)
    recent_manuscripts = []
    for manuscript in manuscripts[:10]:
        recent_manuscripts.append(ManuscriptResponse(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript.file_path)),
            metadata=manuscript.metadata,
            chapters=manuscript.chapters,
            total_images=count_generated_images(manuscript.metadata.title),
            processing_status="draft",  # TODO: Determine actual status
            created_at=manuscript.metadata.created_at,
            updated_at=manuscript.saved_at
        ))

    return DashboardStats(
        total_manuscripts=len(manuscripts),
        total_chapters=total_chapters,
        total_images=total_images,
        recent_manuscripts=recent_manuscripts,
        processing_count=processing_count
    )


@router.get("/")
async def list_manuscripts() -> List[ManuscriptResponse]:
    """List all manuscripts."""
    manuscripts = get_saved_manuscripts()

    response = []
    for manuscript in manuscripts:
        response.append(ManuscriptResponse(
            id=str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript.file_path)),
            metadata=manuscript.metadata,
            chapters=manuscript.chapters,
            total_images=count_generated_images(manuscript.metadata.title),
            processing_status="draft",  # TODO: Determine actual status
            created_at=manuscript.metadata.created_at,
            updated_at=manuscript.saved_at
        ))

    return response


@router.get("/{manuscript_id}")
async def get_manuscript(manuscript_id: str) -> ManuscriptResponse:
    """Get a specific manuscript by ID."""
    manuscripts = get_saved_manuscripts()

    # Find manuscript by generated ID
    for manuscript in manuscripts:
        generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript.file_path))
        if generated_id == manuscript_id:
            return ManuscriptResponse(
                id=manuscript_id,
                metadata=manuscript.metadata,
                chapters=manuscript.chapters,
                total_images=count_generated_images(manuscript.metadata.title),
                processing_status="draft",  # TODO: Determine actual status
                created_at=manuscript.metadata.created_at,
                updated_at=manuscript.saved_at
            )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Manuscript not found"
    )


@router.post("/")
async def create_manuscript(request: ManuscriptCreateRequest) -> ManuscriptResponse:
    """Create a new manuscript."""
    # Create manuscript metadata
    metadata = ManuscriptMetadata(
        title=request.title,
        author=request.author,
        genre=request.genre,
        total_chapters=0,
        created_at=datetime.now().isoformat()
    )

    # Create saved manuscript
    saved_manuscript = SavedManuscript(
        metadata=metadata,
        chapters=[],
        saved_at=datetime.now().isoformat(),
        file_path=""  # Will be set when saving
    )

    # Save to disk
    file_path = save_manuscript_to_disk(saved_manuscript)
    manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(file_path)))

    return ManuscriptResponse(
        id=manuscript_id,
        metadata=metadata,
        chapters=[],
        total_images=0,
        processing_status="draft",
        created_at=metadata.created_at,
        updated_at=saved_manuscript.saved_at
    )


@router.put("/{manuscript_id}")
async def update_manuscript(
    manuscript_id: str,
    request: ManuscriptCreateRequest
) -> ManuscriptResponse:
    """Update an existing manuscript."""
    manuscripts = get_saved_manuscripts()

    # Find and update manuscript
    for manuscript in manuscripts:
        generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript.file_path))
        if generated_id == manuscript_id:
            # Update metadata
            manuscript.metadata.title = request.title
            manuscript.metadata.author = request.author
            manuscript.metadata.genre = request.genre
            manuscript.saved_at = datetime.now().isoformat()

            # Save updated manuscript
            save_manuscript_to_disk(manuscript)

            return ManuscriptResponse(
                id=manuscript_id,
                metadata=manuscript.metadata,
                chapters=manuscript.chapters,
                total_images=count_generated_images(manuscript.metadata.title),
                processing_status="draft",
                created_at=manuscript.metadata.created_at,
                updated_at=manuscript.saved_at
            )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Manuscript not found"
    )


@router.post("/{manuscript_id}/style")
async def save_style_config(
    manuscript_id: str,
    request: StyleConfigSaveRequest
) -> SuccessResponse:
    """Save style configuration for a manuscript."""
    # Verify the manuscript exists first
    manuscripts = get_saved_manuscripts()

    manuscript_found = None
    for manuscript in manuscripts:
        generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript.file_path))
        if generated_id == manuscript_id:
            manuscript_found = manuscript
            break

    if not manuscript_found:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Manuscript not found"
        )

    try:
        # Create a style config directory for this manuscript if it doesn't exist
        style_config_dir = SAVED_MANUSCRIPTS_DIR / "style_configs"
        style_config_dir.mkdir(exist_ok=True)

        # Save the style configuration
        style_config_file = style_config_dir / f"{manuscript_id}_style.json"

        with open(style_config_file, 'w', encoding='utf-8') as f:
            json.dump(request.style_config.model_dump(), f, indent=2, ensure_ascii=False)

        return SuccessResponse(
            message="Style configuration saved successfully",
            data={"style_config_path": str(style_config_file)}
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving style configuration: {str(e)}"
        )


@router.get("/{manuscript_id}/style")
async def get_style_config(manuscript_id: str) -> Dict[str, Any]:
    """Get saved style configuration for a manuscript."""
    # Verify the manuscript exists first
    manuscripts = get_saved_manuscripts()

    for manuscript in manuscripts:
        generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript.file_path))
        if generated_id == manuscript_id:
            break
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Manuscript not found"
        )

    try:
        # Check if style configuration exists
        style_config_dir = SAVED_MANUSCRIPTS_DIR / "style_configs"
        style_config_file = style_config_dir / f"{manuscript_id}_style.json"

        if not style_config_file.exists():
            return {"style_config": None}

        with open(style_config_file, 'r', encoding='utf-8') as f:
            style_config_data = json.load(f)

        return {"style_config": style_config_data}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading style configuration: {str(e)}"
        )


@router.delete("/{manuscript_id}")
async def delete_manuscript(manuscript_id: str) -> SuccessResponse:
    """Delete a manuscript."""
    manuscripts = get_saved_manuscripts()

    # Find and delete manuscript
    for manuscript in manuscripts:
        generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript.file_path))
        if generated_id == manuscript_id:
            try:
                # Delete the file
                Path(manuscript.file_path).unlink(missing_ok=True)

                # Also delete generated images directory if it exists
                safe_title = manuscript.metadata.title.replace(" ", "_")
                images_dir = ILLUSTRATOR_OUTPUT_DIR / safe_title
                if images_dir.exists():
                    import shutil
                    shutil.rmtree(images_dir)

                return SuccessResponse(
                    message="Manuscript deleted successfully"
                )
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Error deleting manuscript: {str(e)}"
                )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Manuscript not found"
    )