"""Service layer for managing illustrations in the database."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..db_config import get_db
from ..db_models import Illustration, Manuscript, Chapter, ProcessingSession
from ..models import EmotionalMoment, ImageProvider


class IllustrationService:
    """Service for managing illustrations in the database."""

    def __init__(self, db: Optional[Session] = None):
        """Initialize the service with an optional database session."""
        self.db = db or get_db()

    def save_illustration(
        self,
        manuscript_id: str,
        chapter_id: str,
        scene_number: int,
        filename: str,
        file_path: str,
        prompt: str,
        image_provider: ImageProvider,
        emotional_moment: Optional[EmotionalMoment] = None,
        style_config: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        file_size: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Illustration:
        """Save an illustration to the database.

        Args:
            manuscript_id: UUID of the manuscript
            chapter_id: UUID of the chapter
            scene_number: Scene number within the chapter
            filename: Name of the image file
            file_path: Full path to the image file
            prompt: The prompt used to generate the image
            image_provider: The AI provider used to generate the image
            emotional_moment: The emotional moment this image represents
            style_config: Style configuration used for generation
            title: Optional title for the illustration
            description: Optional description
            file_size: File size in bytes
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            The created Illustration object
        """
        # Generate web URL
        web_url = f"/generated/{filename}"

        # Prepare emotional tones
        emotional_tones = None
        intensity_score = None
        text_excerpt = None

        if emotional_moment:
            emotional_tones = ",".join([tone.value for tone in emotional_moment.emotional_tones])
            intensity_score = emotional_moment.intensity_score
            text_excerpt = emotional_moment.text_excerpt

            # Generate title and description from emotional moment if not provided
            if not title:
                title = f"Chapter Scene {scene_number}"
            if not description:
                primary_tone = emotional_moment.emotional_tones[0].value if emotional_moment.emotional_tones else "neutral"
                description = f"Illustration depicting: {emotional_moment.text_excerpt[:100]}... (Tone: {primary_tone})"

        # Create illustration record
        illustration = Illustration(
            manuscript_id=UUID(manuscript_id),
            chapter_id=UUID(chapter_id),
            filename=filename,
            file_path=file_path,
            web_url=web_url,
            scene_number=scene_number,
            title=title or f"Scene {scene_number}",
            description=description,
            prompt=prompt,
            style_config=json.dumps(style_config) if style_config else None,
            image_provider=image_provider.value,
            emotional_tones=emotional_tones,
            intensity_score=intensity_score,
            text_excerpt=text_excerpt,
            file_size=file_size,
            width=width,
            height=height,
            generation_status="completed"
        )

        self.db.add(illustration)
        self.db.commit()
        self.db.refresh(illustration)

        return illustration

    def get_illustrations_by_manuscript(self, manuscript_id: str) -> List[Illustration]:
        """Get all illustrations for a manuscript.

        Args:
            manuscript_id: UUID of the manuscript

        Returns:
            List of Illustration objects ordered by chapter number and scene number
        """
        return (
            self.db.query(Illustration)
            .join(Chapter)
            .filter(Illustration.manuscript_id == UUID(manuscript_id))
            .order_by(Chapter.number, Illustration.scene_number)
            .all()
        )

    def get_illustrations_by_chapter(self, chapter_id: str) -> List[Illustration]:
        """Get all illustrations for a chapter.

        Args:
            chapter_id: UUID of the chapter

        Returns:
            List of Illustration objects ordered by scene number
        """
        return (
            self.db.query(Illustration)
            .filter(Illustration.chapter_id == UUID(chapter_id))
            .order_by(Illustration.scene_number)
            .all()
        )

    def get_illustration_by_id(self, illustration_id: str) -> Optional[Illustration]:
        """Get an illustration by its ID.

        Args:
            illustration_id: UUID of the illustration

        Returns:
            Illustration object or None if not found
        """
        return (
            self.db.query(Illustration)
            .filter(Illustration.id == UUID(illustration_id))
            .first()
        )

    def close(self):
        """Close the database session."""
        if self.db:
            self.db.close()