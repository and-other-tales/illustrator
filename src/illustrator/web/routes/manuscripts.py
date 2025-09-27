"""API routes for manuscript management."""

import base64
import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse

from illustrator.models import ManuscriptMetadata, SavedManuscript, Chapter, ImageProvider
from illustrator.web.models.web_models import (
    ManuscriptCreateRequest,
    ManuscriptResponse,
    DashboardStats,
    ErrorResponse,
    SuccessResponse,
    StyleConfigSaveRequest
)
from illustrator.prompt_engineering import (
    StyleTranslator,
    SceneComposition,
    CompositionType,
    LightingMood,
)

router = APIRouter()

logger = logging.getLogger(__name__)

# Storage paths
SAVED_MANUSCRIPTS_DIR = Path("saved_manuscripts")
ILLUSTRATOR_OUTPUT_DIR = Path("illustrator_output")
GENERATED_IMAGES_DIR = ILLUSTRATOR_OUTPUT_DIR / "generated_images"
PREVIEW_IMAGES_DIR = GENERATED_IMAGES_DIR / "previews"

# Ensure directories exist
SAVED_MANUSCRIPTS_DIR.mkdir(exist_ok=True)
ILLUSTRATOR_OUTPUT_DIR.mkdir(exist_ok=True)
GENERATED_IMAGES_DIR.mkdir(exist_ok=True)
PREVIEW_IMAGES_DIR.mkdir(exist_ok=True)


def generate_manuscript_id() -> str:
    """Generate a unique manuscript ID."""
    return str(uuid.uuid4())


def get_manuscript_processing_status(manuscript_id: str) -> str:
    """Get the current processing status of a manuscript."""
    try:
        from illustrator.web.app import connection_manager
        if hasattr(connection_manager, 'sessions'):
            for session_id, session_data in connection_manager.sessions.items():
                if session_data.manuscript_id == manuscript_id:
                    if hasattr(session_data, 'status') and session_data.status:
                        return session_data.status.status
                    return "processing"
        return "draft"
    except:
        return "draft"


# Cache for manuscripts to avoid repeated file system scans
_manuscripts_cache = {}
_cache_timestamp = None

# Shared style translator instance for preview generation
_style_translator = StyleTranslator()

_ROUTES_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _ROUTES_DIR.parents[4]


def _normalize_style_modifiers(modifiers: Iterable[Any]) -> List[str]:
    """Convert style modifiers into a list of readable strings."""
    normalized: List[str] = []
    for modifier in modifiers or []:
        if isinstance(modifier, (list, tuple)):
            normalized.append(" ".join(str(part) for part in modifier if part))
        else:
            normalized.append(str(modifier))
    return [m for m in normalized if m]


def _resolve_style_config_path(style_config_path: str) -> Optional[Path]:
    """Resolve possible locations for a rich style configuration file."""
    candidate_paths = []
    raw_path = Path(style_config_path)

    if raw_path.is_absolute():
        candidate_paths.append(raw_path)
    else:
        candidate_paths.extend([
            raw_path,
            _PROJECT_ROOT / raw_path,
            SAVED_MANUSCRIPTS_DIR / raw_path,
            SAVED_MANUSCRIPTS_DIR / "style_configs" / raw_path,
        ])

    seen = set()
    for candidate in candidate_paths:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    return None


def _load_rich_style_config(style_config_path: str) -> Optional[Dict[str, Any]]:
    """Load a detailed style configuration file if available."""
    resolved_path = _resolve_style_config_path(style_config_path)
    if not resolved_path:
        logger.debug("Rich style config not found for path %s", style_config_path)
        return None

    try:
        with open(resolved_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001 - we want to log and continue
        logger.warning("Failed to load rich style config %s: %s", resolved_path, exc)
        return None

def invalidate_manuscripts_cache():
    """Invalidate the manuscripts cache to force a refresh."""
    global _manuscripts_cache, _cache_timestamp
    _manuscripts_cache.clear()
    _cache_timestamp = None

def get_saved_manuscripts() -> List[SavedManuscript]:
    """Load all saved manuscripts from disk with caching."""
    global _manuscripts_cache, _cache_timestamp

    if not SAVED_MANUSCRIPTS_DIR.exists():
        return []

    # Check if cache is still valid (cache for 30 seconds)
    import time
    current_time = time.time()

    if _cache_timestamp and current_time - _cache_timestamp < 30:
        return [manuscript for manuscript, _ in _manuscripts_cache.values()]

    manuscripts = []
    new_cache = {}

    for file_path in SAVED_MANUSCRIPTS_DIR.glob("*.json"):
        try:
            # Check file modification time to see if we need to reload
            file_mtime = file_path.stat().st_mtime
            cache_key = str(file_path)

            # Use cached version if file hasn't changed
            if cache_key in _manuscripts_cache:
                cached_manuscript, cached_mtime = _manuscripts_cache[cache_key]
                if file_mtime <= cached_mtime:
                    new_cache[cache_key] = (cached_manuscript, cached_mtime)
                    manuscripts.append(cached_manuscript)
                    continue

            # Load from file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            manuscript = SavedManuscript(**data)
            new_cache[cache_key] = (manuscript, file_mtime)
            manuscripts.append(manuscript)

        except Exception as e:
            print(f"Error loading manuscript {file_path}: {e}")
            continue

    # Update cache
    _manuscripts_cache = new_cache
    _cache_timestamp = current_time

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
    # Images are stored directly in illustrator_output/generated_images/
    images_dir = ILLUSTRATOR_OUTPUT_DIR / "generated_images"

    if not images_dir.exists():
        return 0

    image_count = 0
    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}

    # Count all image files in the generated_images directory
    for image_file in images_dir.iterdir():
        if image_file.is_file() and image_file.suffix.lower() in image_extensions:
            image_count += 1

    return image_count


@router.get("/stats")
async def get_dashboard_stats() -> DashboardStats:
    """Get dashboard statistics."""
    manuscripts = get_saved_manuscripts()

    total_chapters = sum(len(m.chapters) for m in manuscripts)
    total_images = sum(count_generated_images(m.metadata.title) for m in manuscripts)
    # Track active processing sessions by checking connection manager
    from illustrator.web.models.web_models import ConnectionManager
    try:
        # Check for active processing sessions
        processing_count = 0
        from illustrator.web.app import connection_manager
        if hasattr(connection_manager, 'sessions'):
            processing_count = len(connection_manager.sessions)
    except:
        processing_count = 0

    # Get recent manuscripts (last 10)
    recent_manuscripts = []
    for manuscript in manuscripts[:10]:
        generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript.file_path))
        recent_manuscripts.append(ManuscriptResponse(
            id=generated_id,
            metadata=manuscript.metadata,
            chapters=manuscript.chapters,
            total_images=count_generated_images(manuscript.metadata.title),
            processing_status=get_manuscript_processing_status(generated_id),
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
        generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript.file_path))
        response.append(ManuscriptResponse(
            id=generated_id,
            metadata=manuscript.metadata,
            chapters=manuscript.chapters,
            total_images=count_generated_images(manuscript.metadata.title),
            processing_status=get_manuscript_processing_status(generated_id),
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
                processing_status=get_manuscript_processing_status(generated_id),
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

    # Invalidate cache to ensure new manuscript appears in the dashboard
    invalidate_manuscripts_cache()

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

            # Invalidate cache to ensure updated manuscript appears in the dashboard
            invalidate_manuscripts_cache()

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

        # Merge base config with any linked rich configuration for accurate previews
        rich_config = None
        if style_config.style_config_path:
            rich_config = _load_rich_style_config(style_config.style_config_path)

        style_preferences = style_config.model_dump()
        style_preferences["image_provider"] = style_config.image_provider.value
        if rich_config:
            style_preferences = {**rich_config, **style_preferences}

        # Create a representative scene composition to translate style preferences
        scene_composition = SceneComposition(
            composition_type=CompositionType.MEDIUM_SHOT,
            focal_point="primary narrative subjects",
            background_elements=["supporting environment"],
            foreground_elements=["key characters"],
            lighting_mood=LightingMood.NATURAL,
            atmosphere="balanced preview for configured illustration style",
            color_palette_suggestion=style_config.color_palette or "cohesive tones",
            emotional_weight=0.45,
        )

        style_translation = _style_translator.translate_style_config(
            style_preferences,
            style_config.image_provider,
            scene_composition
        )

        style_modifiers = _normalize_style_modifiers(style_translation.get("style_modifiers", []))
        is_flux_family = style_config.image_provider in {
            ImageProvider.FLUX,
            ImageProvider.SEEDREAM,
        }

        if is_flux_family and len(style_modifiers) > 6:
            style_modifiers = style_modifiers[:6]
        provider_opts = style_translation.get("provider_optimizations", {}) or {}

        # Flux prompts are sensitive to length; trim verbose quality modifiers
        if is_flux_family and provider_opts.get("quality_modifiers"):
            provider_opts = {
                **provider_opts,
                "quality_modifiers": provider_opts["quality_modifiers"][:2],
            }

        technical_params = dict(style_translation.get("technical_params", {}) or {})
        technical_params.update(provider_opts.get("technical_adjustments", {}) or {})

        negative_prompt_items = [str(item) for item in style_translation.get("negative_prompt", []) or []]
        negative_prompt = ", ".join(negative_prompt_items) if negative_prompt_items else None

        # Build a preview prompt that reflects the configured style settings
        prompt_sections: List[str] = []
        base_description = f"Concept illustration in {style_config.art_style}"
        if style_config.color_palette:
            base_description += f" using the {style_config.color_palette} palette"
        if style_config.artistic_influences:
            base_description += f" inspired by {style_config.artistic_influences}"
        prompt_sections.append(base_description)

        if style_modifiers:
            prompt_sections.append("featuring " + ", ".join(style_modifiers))
        if provider_opts.get("style_emphasis"):
            prompt_sections.append(provider_opts["style_emphasis"])
        if (
            provider_opts.get("quality_modifiers")
            and not is_flux_family
        ):
            prompt_sections.append(
                "quality focus: " + ", ".join(provider_opts["quality_modifiers"])
            )

        preview_prompt_text = ". ".join(section for section in prompt_sections if section)
        if not preview_prompt_text:
            preview_prompt_text = "Concept illustration showcasing the configured style settings"
        if not preview_prompt_text.endswith('.'):
            preview_prompt_text += '.'

        from illustrator.models import IllustrationPrompt
        illustration_prompt = IllustrationPrompt(
            provider=style_config.image_provider,
            prompt=preview_prompt_text,
            style_modifiers=style_modifiers,
            negative_prompt=negative_prompt,
            technical_params=technical_params
        )

        # Generate the image
        result = await provider.generate_image(illustration_prompt)

        # Extract image URL from result
        metadata: Dict[str, Any] = {}
        if result.get('success'):
            if result.get('image_data'):
                # Persist preview image to disk so the browser can load it reliably
                image_bytes = base64.b64decode(result['image_data'])
                preview_filename = f"{manuscript_id}_style_preview_{uuid.uuid4().hex}.png"
                preview_path = PREVIEW_IMAGES_DIR / preview_filename

                with open(preview_path, 'wb') as preview_file:
                    preview_file.write(image_bytes)

                image_url = f"/generated/previews/{preview_filename}"
            else:
                # Use direct URL if available
                image_url = result.get('image_url', '')

            metadata = result.get('metadata', {}) or {}
        else:
            status_code = result.get('status_code') or status.HTTP_502_BAD_GATEWAY
            error_message = result.get('error', 'Image generation failed')
            raise HTTPException(
                status_code=status_code,
                detail=f"Flux preview failed: {error_message}"
            )

        style_summary: Dict[str, Any] = {
            "provider": style_config.image_provider.value,
            "art_style": style_config.art_style,
            "color_palette": style_config.color_palette,
            "artistic_influences": style_config.artistic_influences,
            "style_config_path": style_config.style_config_path,
            "style_modifiers": style_modifiers,
        }

        if rich_config and rich_config.get("style_name"):
            style_summary["style_name"] = rich_config["style_name"]
        if provider_opts.get("style_emphasis"):
            style_summary["style_emphasis"] = provider_opts["style_emphasis"]
        if provider_opts.get("quality_modifiers") and not is_flux_family:
            style_summary["quality_modifiers"] = provider_opts["quality_modifiers"]
        if is_flux_family:
            effective_prompt = metadata.get('prompt') or preview_prompt_text
            style_summary["effective_prompt"] = effective_prompt
            if metadata.get('prompt_truncated'):
                style_summary["prompt_truncated"] = True
                preview_prompt_text = effective_prompt

        return {
            "image_url": image_url,
            "preview_prompt": preview_prompt_text,
            "style_summary": style_summary
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
        from illustrator.services.illustration_service import IllustrationService

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


@router.delete("/{manuscript_id}/images/{image_id}")
async def delete_manuscript_image(manuscript_id: str, image_id: str) -> SuccessResponse:
    """Delete a specific image for a manuscript."""
    try:
        from illustrator.services.illustration_service import IllustrationService

        # Initialize illustration service
        illustration_service = IllustrationService()

        try:
            # Try to delete from database first
            illustration = illustration_service.get_illustration_by_id(image_id)

            if not illustration:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Image not found"
                )

            # Verify the image belongs to the correct manuscript
            if str(illustration.manuscript_id) != manuscript_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Image does not belong to this manuscript"
                )

            # Delete the physical file
            if illustration.file_path and Path(illustration.file_path).exists():
                Path(illustration.file_path).unlink()

            # Delete from database
            illustration_service.delete_illustration(image_id)

            return SuccessResponse(
                message="Image deleted successfully"
            )

        finally:
            illustration_service.close()

    except HTTPException:
        raise
    except Exception as e:
        # Fallback to filesystem deletion if database fails
        print(f"Database delete failed, falling back to filesystem: {e}")

        # Parse image_id as filename for filesystem fallback
        filename = image_id
        if not filename.endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            filename = f"{image_id}.png"  # Assume PNG if no extension

        generated_images_dir = Path("illustrator_output") / "generated_images"
        image_path = generated_images_dir / filename

        if not image_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found"
            )

        try:
            image_path.unlink()
            return SuccessResponse(
                message="Image deleted successfully"
            )
        except Exception as delete_error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error deleting image: {str(delete_error)}"
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

                # Invalidate cache to ensure the deleted manuscript doesn't appear in the dashboard
                invalidate_manuscripts_cache()

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


@router.post("/{manuscript_id}/export")
async def export_manuscript(
    manuscript_id: str,
    export_format: str = "pdf"
) -> Dict[str, Any]:
    """Export manuscript in various formats (PDF, DOCX, HTML, JSON)."""

    # Find the manuscript
    manuscripts = get_saved_manuscripts()
    manuscript = None

    for saved_manuscript in manuscripts:
        generated_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, saved_manuscript.file_path))
        if generated_id == manuscript_id:
            manuscript = saved_manuscript
            break

    if not manuscript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Manuscript not found"
        )

    # Validate export format
    supported_formats = ["pdf", "docx", "html", "json"]
    if export_format.lower() not in supported_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported export format. Supported formats: {', '.join(supported_formats)}"
        )

    try:
        # Create exports directory
        exports_dir = Path("illustrator_output") / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename
        safe_title = "".join(c for c in manuscript.metadata.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        export_format_lower = export_format.lower()

        if export_format_lower == "json":
            # JSON Export - raw data
            filename = f"{safe_title}_{timestamp}.json"
            file_path = exports_dir / filename

            export_data = {
                "manuscript_id": manuscript_id,
                "metadata": manuscript.metadata.model_dump(),
                "chapters": [chapter.model_dump() for chapter in manuscript.chapters],
                "export_info": {
                    "format": "json",
                    "exported_at": datetime.now().isoformat(),
                    "total_chapters": len(manuscript.chapters),
                    "total_words": sum(len(chapter.content.split()) for chapter in manuscript.chapters)
                }
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

        elif export_format_lower == "html":
            # HTML Export
            filename = f"{safe_title}_{timestamp}.html"
            file_path = exports_dir / filename

            html_content = generate_html_export(manuscript)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        elif export_format_lower == "pdf":
            # PDF Export
            filename = f"{safe_title}_{timestamp}.pdf"
            file_path = exports_dir / filename

            await generate_pdf_export(manuscript, file_path)

        elif export_format_lower == "docx":
            # DOCX Export
            filename = f"{safe_title}_{timestamp}.docx"
            file_path = exports_dir / filename

            generate_docx_export(manuscript, file_path)

        # Return download info
        return {
            "success": True,
            "export_format": export_format.upper(),
            "filename": filename,
            "download_url": f"/api/manuscripts/{manuscript_id}/download/{filename}",
            "file_size": file_path.stat().st_size,
            "exported_at": datetime.now().isoformat(),
            "manuscript_title": manuscript.metadata.title
        }

    except Exception as e:
        import traceback
        print(f"Export error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export manuscript: {str(e)}"
        )


def generate_html_export(manuscript: SavedManuscript) -> str:
    """Generate HTML export of the manuscript."""
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title} by {author}</title>
        <style>
            body {{
                font-family: 'Times New Roman', serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 2rem;
                background-color: #f8f9fa;
            }}
            .manuscript-header {{
                text-align: center;
                margin-bottom: 3rem;
                padding: 2rem;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .manuscript-title {{
                font-size: 2.5rem;
                margin-bottom: 0.5rem;
                color: #2c3e50;
            }}
            .manuscript-author {{
                font-size: 1.2rem;
                color: #6c757d;
                margin-bottom: 1rem;
            }}
            .manuscript-meta {{
                font-size: 0.9rem;
                color: #868e96;
            }}
            .chapter {{
                background: white;
                margin: 2rem 0;
                padding: 2rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .chapter-title {{
                font-size: 1.8rem;
                margin-bottom: 1.5rem;
                color: #2c3e50;
                border-bottom: 2px solid #3498db;
                padding-bottom: 0.5rem;
            }}
            .chapter-content {{
                text-align: justify;
                text-indent: 2rem;
            }}
            .chapter-content p {{
                margin-bottom: 1rem;
            }}
            @media print {{
                body {{
                    background-color: white;
                    max-width: none;
                }}
                .manuscript-header, .chapter {{
                    box-shadow: none;
                    border: 1px solid #ddd;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="manuscript-header">
            <h1 class="manuscript-title">{title}</h1>
            <p class="manuscript-author">by {author}</p>
            <div class="manuscript-meta">
                <p>Genre: {genre} | Chapters: {chapter_count} | Words: {word_count:,}</p>
                <p>Exported: {export_date}</p>
            </div>
        </div>

        {chapters_html}
    </body>
    </html>
    """

    # Generate chapters HTML
    chapters_html = ""
    for chapter in manuscript.chapters:
        # Convert line breaks to paragraphs
        content_paragraphs = [p.strip() for p in chapter.content.split('\n\n') if p.strip()]
        content_html = '\n'.join([f'<p>{p}</p>' for p in content_paragraphs])

        chapters_html += f"""
        <div class="chapter">
            <h2 class="chapter-title">Chapter {chapter.number}: {chapter.title}</h2>
            <div class="chapter-content">
                {content_html}
            </div>
        </div>
        """

    return html_template.format(
        title=manuscript.metadata.title,
        author=manuscript.metadata.author or "Unknown Author",
        genre=manuscript.metadata.genre or "Fiction",
        chapter_count=len(manuscript.chapters),
        word_count=sum(len(chapter.content.split()) for chapter in manuscript.chapters),
        export_date=datetime.now().strftime("%B %d, %Y"),
        chapters_html=chapters_html
    )


async def generate_pdf_export(manuscript: SavedManuscript, file_path: Path):
    """Generate PDF export using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
    except ImportError:
        # Fallback: generate HTML and convert to PDF using weasyprint if available
        try:
            import weasyprint
            html_content = generate_html_export(manuscript)
            weasyprint.HTML(string=html_content).write_pdf(str(file_path))
            return
        except ImportError:
            raise Exception("PDF generation requires 'reportlab' or 'weasyprint' package. Install with: pip install reportlab")

    # Create PDF document
    doc = SimpleDocTemplate(str(file_path), pagesize=letter, topMargin=1*inch)
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    )

    author_style = ParagraphStyle(
        'CustomAuthor',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=20,
        alignment=TA_CENTER
    )

    chapter_title_style = ParagraphStyle(
        'ChapterTitle',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=20,
        spaceBefore=30
    )

    content_style = ParagraphStyle(
        'Content',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12,
        alignment=TA_JUSTIFY,
        firstLineIndent=20
    )

    # Title page
    story.append(Paragraph(manuscript.metadata.title, title_style))
    story.append(Paragraph(f"by {manuscript.metadata.author or 'Unknown Author'}", author_style))
    story.append(Spacer(1, 0.5*inch))

    # Metadata
    metadata_text = f"""
    Genre: {manuscript.metadata.genre or 'Fiction'}<br/>
    Chapters: {len(manuscript.chapters)}<br/>
    Words: {sum(len(chapter.content.split()) for chapter in manuscript.chapters):,}<br/>
    Exported: {datetime.now().strftime('%B %d, %Y')}
    """
    story.append(Paragraph(metadata_text, styles['Normal']))
    story.append(PageBreak())

    # Chapters
    for i, chapter in enumerate(manuscript.chapters):
        if i > 0:
            story.append(PageBreak())

        # Chapter title
        story.append(Paragraph(f"Chapter {chapter.number}: {chapter.title}", chapter_title_style))

        # Chapter content
        paragraphs = [p.strip() for p in chapter.content.split('\n\n') if p.strip()]
        for paragraph in paragraphs:
            story.append(Paragraph(paragraph, content_style))

    # Build PDF
    doc.build(story)


def generate_docx_export(manuscript: SavedManuscript, file_path: Path):
    """Generate DOCX export using python-docx."""
    try:
        from docx import Document
        from docx.shared import Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        raise Exception("DOCX generation requires 'python-docx' package. Install with: pip install python-docx")

    # Create document
    doc = Document()

    # Title page
    title_para = doc.add_paragraph()
    title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_para.add_run(manuscript.metadata.title)
    title_run.bold = True
    title_run.font.size = doc.styles['Title'].font.size

    author_para = doc.add_paragraph()
    author_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author_run = author_para.add_run(f"by {manuscript.metadata.author or 'Unknown Author'}")
    author_run.italic = True

    doc.add_paragraph()  # Spacer

    # Metadata
    metadata_para = doc.add_paragraph()
    metadata_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    metadata_text = f"""Genre: {manuscript.metadata.genre or 'Fiction'}
Chapters: {len(manuscript.chapters)}
Words: {sum(len(chapter.content.split()) for chapter in manuscript.chapters):,}
Exported: {datetime.now().strftime('%B %d, %Y')}"""
    metadata_para.add_run(metadata_text)

    doc.add_page_break()

    # Chapters
    for chapter in manuscript.chapters:
        # Chapter title
        chapter_heading = doc.add_heading(f"Chapter {chapter.number}: {chapter.title}", level=1)

        # Chapter content
        paragraphs = [p.strip() for p in chapter.content.split('\n\n') if p.strip()]
        for paragraph in paragraphs:
            para = doc.add_paragraph(paragraph)
            para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        doc.add_page_break()

    # Remove last page break
    if doc.paragraphs:
        last_para = doc.paragraphs[-1]
        if last_para._p.getparent() is not None:
            last_para._p.getparent().remove(last_para._p)

    # Save document
    doc.save(str(file_path))
