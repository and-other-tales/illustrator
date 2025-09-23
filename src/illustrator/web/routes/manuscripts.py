"""API routes for manuscript management."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

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


@router.post("/{manuscript_id}/style/preview")
async def preview_style_image(
    manuscript_id: str,
    request: StyleConfigSaveRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Generate a preview image using the style configuration."""
    from illustrator.providers import get_image_provider
    from illustrator.models import StyleConfig

    # Verify the manuscript exists
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
        # Create style config from request
        style_config = StyleConfig(**request.style_config.model_dump())

        # Get the image provider
        provider = get_image_provider(style_config.image_provider)

        # Generate a simple preview prompt
        preview_prompt_text = f"A sample illustration in {style_config.art_style} style"
        if style_config.color_palette:
            preview_prompt_text += f" with {style_config.color_palette} colors"
        if style_config.artistic_influences:
            preview_prompt_text += f" inspired by {style_config.artistic_influences}"
        preview_prompt_text += ", high quality, detailed"

        # Create IllustrationPrompt object
        from illustrator.models import IllustrationPrompt
        illustration_prompt = IllustrationPrompt(
            provider=style_config.image_provider,
            prompt=preview_prompt_text,
            style_modifiers=[],
            technical_params={}
        )

        # Generate the image
        result = await provider.generate_image(illustration_prompt)

        # Extract image URL from result
        if result.get('success'):
            if result.get('image_data'):
                # Convert base64 to data URL
                image_url = f"data:image/png;base64,{result['image_data']}"
            else:
                # Use direct URL if available
                image_url = result.get('image_url', '')
        else:
            raise Exception(result.get('error', 'Image generation failed'))

        return {
            "image_url": image_url,
            "preview_prompt": preview_prompt_text,
            "style_summary": {
                "provider": style_config.image_provider,
                "art_style": style_config.art_style,
                "color_palette": style_config.color_palette,
                "artistic_influences": style_config.artistic_influences
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating preview image: {str(e)}"
        )


@router.get("/{manuscript_id}/images")
async def list_manuscript_images(manuscript_id: str) -> Dict[str, Any]:
    """List all generated images for a manuscript from the database."""
    try:
        from ..services.illustration_service import IllustrationService

        # Initialize illustration service
        illustration_service = IllustrationService()

        try:
            # Get illustrations from database
            illustrations = illustration_service.get_illustrations_by_manuscript(manuscript_id)

            # Convert to API format
            images = []
            for illustration in illustrations:
                # Get chapter information
                chapter_info = f"Chapter {illustration.chapter.number}" if illustration.chapter else "Unknown Chapter"

                image_data = {
                    "id": str(illustration.id),
                    "url": illustration.web_url,
                    "filename": illustration.filename,
                    "title": illustration.title or f"{chapter_info} - Scene {illustration.scene_number}",
                    "description": illustration.description or f"Generated illustration for {chapter_info}, Scene {illustration.scene_number}",
                    "chapter": chapter_info,
                    "scene": f"Scene {illustration.scene_number}",
                    "style": illustration.image_provider,
                    "emotional_tones": illustration.emotional_tones.split(",") if illustration.emotional_tones else [],
                    "intensity_score": illustration.intensity_score,
                    "prompt": illustration.prompt,
                    "text_excerpt": illustration.text_excerpt,
                    "size": illustration.file_size,
                    "width": illustration.width,
                    "height": illustration.height,
                    "created_at": illustration.created_at.isoformat() if illustration.created_at else None,
                    "generation_status": illustration.generation_status
                }
                images.append(image_data)

            # Get manuscript title (fallback to file-based system for now)
            manuscript_title = "Unknown Manuscript"
            try:
                manuscripts = get_saved_manuscripts()
                for manuscript in manuscripts:
                    generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript.file_path))
                    if generated_id == manuscript_id:
                        manuscript_title = manuscript.metadata.title
                        break
            except Exception:
                pass

            return {
                "images": images,
                "total_count": len(images),
                "manuscript_id": manuscript_id,
                "manuscript_title": manuscript_title
            }

        finally:
            illustration_service.close()

    except Exception as e:
        # Fallback to filesystem scanning if database fails
        print(f"Database query failed, falling back to filesystem: {e}")

        manuscripts = get_saved_manuscripts()

        # Find the manuscript
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

        # Fallback to filesystem scanning
        generated_images_dir = Path("illustrator_output") / "generated_images"
        images = []

        if generated_images_dir.exists():
            image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}

            for image_file in generated_images_dir.iterdir():
                if image_file.is_file() and image_file.suffix.lower() in image_extensions:
                    filename = image_file.name
                    chapter_info = "Unknown"
                    scene_info = "Unknown"

                    if "chapter_" in filename and "scene_" in filename:
                        try:
                            parts = filename.split("_")
                            chapter_idx = next(i for i, part in enumerate(parts) if part == "chapter")
                            scene_idx = next(i for i, part in enumerate(parts) if part == "scene")

                            if chapter_idx + 1 < len(parts) and scene_idx + 1 < len(parts):
                                chapter_num = parts[chapter_idx + 1]
                                scene_num = parts[scene_idx + 1].split(".")[0]
                                chapter_info = f"Chapter {int(chapter_num)}"
                                scene_info = f"Scene {int(scene_num)}"
                        except (ValueError, IndexError, StopIteration):
                            pass

                    image_data = {
                        "url": f"/generated/{image_file.name}",
                        "filename": image_file.name,
                        "title": f"{chapter_info} - {scene_info}",
                        "description": f"Generated illustration for {chapter_info}, {scene_info}",
                        "chapter": chapter_info,
                        "scene": scene_info,
                        "style": "Generated illustration",
                        "size": image_file.stat().st_size,
                        "created_at": datetime.fromtimestamp(image_file.stat().st_mtime).isoformat()
                    }
                    images.append(image_data)

        # Sort images by chapter and scene
        def sort_key(img):
            try:
                filename = img["filename"]
                if "chapter_" in filename and "scene_" in filename:
                    parts = filename.split("_")
                    chapter_idx = next(i for i, part in enumerate(parts) if part == "chapter")
                    scene_idx = next(i for i, part in enumerate(parts) if part == "scene")

                    if chapter_idx + 1 < len(parts) and scene_idx + 1 < len(parts):
                        chapter_num = int(parts[chapter_idx + 1])
                        scene_num = int(parts[scene_idx + 1].split(".")[0])
                        return (chapter_num, scene_num)
            except (ValueError, IndexError, StopIteration):
                pass
            return (999, 999)

        images.sort(key=sort_key)

        return {
            "images": images,
            "total_count": len(images),
            "manuscript_id": manuscript_id,
            "manuscript_title": manuscript_found.metadata.title
        }


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