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
from illustrator.services.illustration_service import IllustrationService
from langchain.chat_models import init_chat_model  # expose for test patching
from generate_scene_illustrations import ComprehensiveSceneAnalyzer  # expose for test patching

# Lazy-import friendly placeholders for patchability in tests
EmotionalAnalyzer = None  # type: ignore
LiterarySceneDetector = None  # type: ignore
NarrativeAnalyzer = None  # type: ignore

router = APIRouter()

# Storage paths
SAVED_MANUSCRIPTS_DIR = Path("saved_manuscripts")
SCENE_ILLUSTRATIONS_DIR = Path("scene_illustrations")
ILLUSTRATOR_OUTPUT_DIR = Path("illustrator_output")


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


def load_chapter_analysis(chapter_id: str) -> Optional[dict]:
    """Load saved analysis for a chapter if available."""
    try:
        # Check if analysis file exists
        analysis_dir = Path("illustrator_output") / "analysis"
        analysis_file = analysis_dir / f"chapter_{chapter_id}_analysis.json"

        if analysis_file.exists():
            with open(analysis_file, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error loading chapter analysis: {e}")

    return None


def count_chapter_images(manuscript_id: str, chapter_number: int) -> int:
    """Count generated images for a specific chapter."""
    try:
        # Try database approach first
        illustration_service = IllustrationService()
        try:
            illustrations = illustration_service.get_illustrations_by_manuscript(manuscript_id)
            count = sum(1 for ill in illustrations if ill.chapter and ill.chapter.number == chapter_number)
            illustration_service.close()
            return count
        finally:
            illustration_service.close()

    except Exception:
        # Fallback to filesystem counting
        try:
            generated_images_dir = Path("illustrator_output/generated_images")
            if not generated_images_dir.exists():
                return 0

            count = 0
            pattern = f"chapter_{chapter_number}_"

            for image_file in generated_images_dir.iterdir():
                if image_file.is_file() and pattern in image_file.name:
                    count += 1

            return count
        except Exception:
            return 0


@router.get("/{manuscript_id}")
async def get_manuscript_chapters(manuscript_id: str) -> List[ChapterResponse]:
    """Get all chapters for a manuscript."""
    manuscript, _ = load_manuscript_by_id(manuscript_id)

    chapters = []
    for i, chapter in enumerate(manuscript.chapters):
        chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chapter.number}"))
        chapters.append(ChapterResponse(
            id=chapter_id,
            chapter=chapter,
            analysis=load_chapter_analysis(chapter_id),
            images_generated=count_chapter_images(manuscript_id, chapter.number),
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
                    # Load analysis and count images for this chapter
                    analysis = load_chapter_analysis(generated_chapter_id)
                    images_count = count_chapter_images(manuscript_id, chapter.number)

                    return ChapterResponse(
                        id=chapter_id,
                        chapter=chapter,
                        analysis=analysis,
                        images_generated=images_count,
                        processing_status="draft" if images_count == 0 else "completed"
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


@router.post("/{chapter_id}/analyze")
async def analyze_chapter(chapter_id: str) -> dict:
    """Perform comprehensive analysis of a chapter including emotional moments, scenes, and visual potential."""

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
        # Initialize the analysis systems
        import os

        # Check for required API key
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        if not anthropic_api_key:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Anthropic API key is required for chapter analysis"
            )

        # Initialize LLM and analyzers
        llm = init_chat_model(
            model="claude-sonnet-4-20250514",
            model_provider="anthropic",
            api_key=anthropic_api_key
        )

        # Resolve analyzers (use patched module-level names if set; otherwise import lazily)
        from illustrator import analysis as _analysis_mod
        from illustrator import scene_detection as _scene_mod
        from illustrator import narrative_analysis as _narrative_mod

        EA = EmotionalAnalyzer or getattr(_analysis_mod, 'EmotionalAnalyzer')
        SD = LiterarySceneDetector or getattr(_scene_mod, 'LiterarySceneDetector')
        NA = NarrativeAnalyzer or getattr(_narrative_mod, 'NarrativeAnalyzer')

        emotional_analyzer = EA(llm)
        scene_detector = SD(llm)
        narrative_analyzer = NA(llm)
        comprehensive_analyzer = ComprehensiveSceneAnalyzer()

        # Perform comprehensive analysis
        analysis_results = {}

        # 1. Emotional Analysis with scenes
        emotional_moments = await emotional_analyzer.analyze_chapter_with_scenes(
            chapter=chapter,
            max_moments=10,
            min_intensity=0.5,
            scene_awareness=True
        )

        analysis_results["emotional_analysis"] = {
            "total_moments": len(emotional_moments),
            "moments": [
                {
                    "text_excerpt": moment.text_excerpt,
                    "emotional_tones": [tone.value for tone in moment.emotional_tones],
                    "intensity_score": moment.intensity_score,
                    "visual_potential": moment.visual_potential,
                    "context": moment.context,
                    "start_position": moment.start_position,
                    "end_position": moment.end_position
                } for moment in emotional_moments
            ]
        }

        # 2. Scene Detection
        scenes = await scene_detector.extract_scenes(chapter.content)
        analysis_results["scene_analysis"] = {
            "total_scenes": len(scenes),
            "scenes": [
                {
                    "scene_type": scene.scene_type,
                    "primary_characters": scene.primary_characters,
                    "location": (
                        ", ".join(getattr(scene, 'location_indicators', []) or getattr(scene, 'setting_indicators', []))
                        or getattr(scene, 'location', None) or ""
                    ),
                    "time_context": (
                        ", ".join(getattr(scene, 'time_indicators', []))
                        or getattr(scene, 'time_context', None) or ""
                    ),
                    "emotional_tone": (
                        "high" if getattr(scene, 'emotional_intensity', 0.0) >= 0.7
                        else "medium" if getattr(scene, 'emotional_intensity', 0.0) >= 0.4
                        else "low"
                    ),
                    "text_preview": scene.text[:200] + "..." if len(scene.text) > 200 else scene.text,
                    "word_count": len(scene.text.split()),
                    "start_position": scene.start_position,
                    "end_position": scene.end_position
                } for scene in scenes
            ]
        }

        # 3. Narrative Structure Analysis
        narrative_structure = await narrative_analyzer.analyze_structure(chapter.content)
        analysis_results["narrative_analysis"] = {
            "structure_type": narrative_structure.structure_type,
            "pacing": narrative_structure.pacing,
            "tension_points": [
                {
                    "position": point.position,
                    "intensity": point.intensity,
                    "description": point.description,
                    "tension_type": point.tension_type
                } for point in narrative_structure.tension_points
            ],
            "character_arcs": [
                {
                    "character_name": arc.character_name,
                    "arc_type": arc.arc_type,
                    "development_stage": arc.development_stage,
                    "key_moments": arc.key_moments
                } for arc in narrative_structure.character_arcs
            ],
            "themes": narrative_structure.themes,
            "narrative_devices": narrative_structure.narrative_devices
        }

        # 4. Visual Illustration Potential Analysis
        illustration_moments = await comprehensive_analyzer.analyze_chapter_comprehensive(chapter)
        analysis_results["illustration_potential"] = {
            "total_illustration_moments": len(illustration_moments),
            "top_moments": [
                {
                    "text_excerpt": moment.text_excerpt,
                    "visual_potential": moment.visual_potential,
                    "emotional_tones": [tone.value for tone in moment.emotional_tones],
                    "intensity_score": moment.intensity_score,
                    "illustration_description": f"Visual scene with {', '.join([tone.value for tone in moment.emotional_tones[:2]])} tones"
                } for moment in illustration_moments[:5]  # Top 5 for display
            ]
        }

        # 5. Chapter Statistics
        analysis_results["statistics"] = {
            "word_count": chapter.word_count or len(chapter.content.split()),
            "character_count": len(chapter.content),
            "paragraph_count": len([p for p in chapter.content.split('\n\n') if p.strip()]),
            "sentence_count": len([s for s in chapter.content.split('.') if s.strip()]),
            "average_sentence_length": round(len(chapter.content.split()) / max(len([s for s in chapter.content.split('.') if s.strip()]), 1), 2),
            "emotional_density": round(len(emotional_moments) / max(len(chapter.content.split()) / 100, 1), 2),  # moments per 100 words
            "scene_density": round(len(scenes) / max(len(chapter.content.split()) / 1000, 1), 2)  # scenes per 1000 words
        }

        # 6. Analysis Summary
        analysis_results["summary"] = {
            "dominant_emotional_tones": list(set([tone for moment in emotional_moments for tone in [t.value for t in moment.emotional_tones]])),
            "key_scenes": [scene.scene_type for scene in scenes[:3]],
            "narrative_complexity": "High" if len(narrative_structure.tension_points) > 3 else "Medium" if len(narrative_structure.tension_points) > 1 else "Low",
            "illustration_readiness": "Excellent" if len(illustration_moments) >= 8 else "Good" if len(illustration_moments) >= 5 else "Moderate",
            "analysis_timestamp": datetime.now().isoformat()
        }

        # Save analysis results for future loading
        try:
            analysis_dir = Path("illustrator_output") / "analysis"
            analysis_dir.mkdir(parents=True, exist_ok=True)
            analysis_file = analysis_dir / f"chapter_{chapter_id}_analysis.json"

            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        except Exception as save_error:
            print(f"Warning: Could not save analysis results: {save_error}")

        return {
            "success": True,
            "chapter_id": chapter_id,
            "chapter_title": chapter_title,
            "analysis": analysis_results
        }

    except Exception as e:
        import traceback
        print(f"Error analyzing chapter: {traceback.format_exc()}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze chapter: {str(e)}"
        )
