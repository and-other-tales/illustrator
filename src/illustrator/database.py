"""Database manager for storing and retrieving manuscript and chapter data."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .models import Chapter, ManuscriptMetadata, SavedManuscript
from .web.models.web_models import ManuscriptResponse, ChapterResponse


class DatabaseManager:
    """Simple file-based database manager for manuscript storage."""

    def __init__(self, data_dir: str = "data"):
        """Initialize the database manager.

        Args:
            data_dir: Directory to store database files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.manuscripts_dir = self.data_dir / "manuscripts"
        self.chapters_dir = self.data_dir / "chapters"

        self.manuscripts_dir.mkdir(exist_ok=True)
        self.chapters_dir.mkdir(exist_ok=True)

        # Index files for quick lookups
        self.manuscripts_index_file = self.data_dir / "manuscripts_index.json"
        self.chapters_index_file = self.data_dir / "chapters_index.json"

        self._load_indexes()

    def _load_indexes(self):
        """Load the index files."""
        if self.manuscripts_index_file.exists():
            with open(self.manuscripts_index_file) as f:
                self.manuscripts_index = json.load(f)
        else:
            self.manuscripts_index = {}

        if self.chapters_index_file.exists():
            with open(self.chapters_index_file) as f:
                self.chapters_index = json.load(f)
        else:
            self.chapters_index = {}

    def _save_indexes(self):
        """Save the index files."""
        with open(self.manuscripts_index_file, 'w') as f:
            json.dump(self.manuscripts_index, f, indent=2)

        with open(self.chapters_index_file, 'w') as f:
            json.dump(self.chapters_index, f, indent=2)

    def get_manuscript(self, manuscript_id: str) -> Optional[ManuscriptResponse]:
        """Get a manuscript by ID.

        Args:
            manuscript_id: The manuscript ID

        Returns:
            ManuscriptResponse if found, None otherwise
        """
        if manuscript_id not in self.manuscripts_index:
            return None

        manuscript_file = self.manuscripts_dir / f"{manuscript_id}.json"
        if not manuscript_file.exists():
            return None

        with open(manuscript_file) as f:
            data = json.load(f)

        return ManuscriptResponse(**data)

    def get_chapters_by_manuscript_id(self, manuscript_id: str) -> List[ChapterResponse]:
        """Get all chapters for a manuscript.

        Args:
            manuscript_id: The manuscript ID

        Returns:
            List of ChapterResponse objects
        """
        chapters = []

        # Get chapter IDs for this manuscript
        manuscript_chapters = [
            chapter_id for chapter_id, chapter_data in self.chapters_index.items()
            if chapter_data.get("manuscript_id") == manuscript_id
        ]

        for chapter_id in manuscript_chapters:
            chapter_file = self.chapters_dir / f"{chapter_id}.json"
            if chapter_file.exists():
                with open(chapter_file) as f:
                    data = json.load(f)
                chapters.append(ChapterResponse(**data))

        # Sort by chapter number
        chapters.sort(key=lambda c: c.chapter.number)
        return chapters

    def save_manuscript(self, manuscript: ManuscriptResponse) -> str:
        """Save a manuscript to the database.

        Args:
            manuscript: The manuscript to save

        Returns:
            The manuscript ID
        """
        manuscript_id = manuscript.id

        # Save manuscript data
        manuscript_file = self.manuscripts_dir / f"{manuscript_id}.json"
        with open(manuscript_file, 'w') as f:
            json.dump(manuscript.dict(), f, indent=2)

        # Update index
        self.manuscripts_index[manuscript_id] = {
            "title": manuscript.metadata.title,
            "author": manuscript.metadata.author,
            "created_at": manuscript.created_at,
            "updated_at": manuscript.updated_at,
            "total_chapters": len(manuscript.chapters)
        }

        self._save_indexes()
        return manuscript_id

    def save_chapter(self, chapter: ChapterResponse, manuscript_id: str) -> str:
        """Save a chapter to the database.

        Args:
            chapter: The chapter to save
            manuscript_id: The ID of the manuscript this chapter belongs to

        Returns:
            The chapter ID
        """
        chapter_id = chapter.id

        # Save chapter data
        chapter_file = self.chapters_dir / f"{chapter_id}.json"
        with open(chapter_file, 'w') as f:
            json.dump(chapter.dict(), f, indent=2)

        # Update index
        self.chapters_index[chapter_id] = {
            "manuscript_id": manuscript_id,
            "title": chapter.chapter.title,
            "number": chapter.chapter.number,
            "word_count": chapter.chapter.word_count
        }

        self._save_indexes()
        return chapter_id

    def create_manuscript(self, title: str, author: Optional[str] = None,
                         genre: Optional[str] = None) -> ManuscriptResponse:
        """Create a new manuscript.

        Args:
            title: Manuscript title
            author: Author name
            genre: Literary genre

        Returns:
            The created manuscript
        """
        manuscript_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        metadata = ManuscriptMetadata(
            title=title,
            author=author,
            genre=genre,
            total_chapters=0,
            created_at=now
        )

        manuscript = ManuscriptResponse(
            id=manuscript_id,
            metadata=metadata,
            chapters=[],
            total_images=0,
            processing_status="draft",
            created_at=now,
            updated_at=now
        )

        self.save_manuscript(manuscript)
        return manuscript

    def create_chapter(self, title: str, content: str, manuscript_id: str,
                      number: int = None) -> ChapterResponse:
        """Create a new chapter.

        Args:
            title: Chapter title
            content: Chapter content
            manuscript_id: ID of the manuscript this chapter belongs to
            number: Chapter number (auto-assigned if None)

        Returns:
            The created chapter
        """
        chapter_id = str(uuid.uuid4())

        # Auto-assign chapter number if not provided
        if number is None:
            existing_chapters = self.get_chapters_by_manuscript_id(manuscript_id)
            number = len(existing_chapters) + 1

        chapter_data = Chapter(
            title=title,
            content=content,
            number=number,
            word_count=len(content.split())
        )

        chapter = ChapterResponse(
            id=chapter_id,
            chapter=chapter_data,
            analysis=None,
            images_generated=0,
            processing_status="draft"
        )

        self.save_chapter(chapter, manuscript_id)
        return chapter

    def list_manuscripts(self) -> List[ManuscriptResponse]:
        """List all manuscripts.

        Returns:
            List of all manuscripts
        """
        manuscripts = []

        for manuscript_id in self.manuscripts_index.keys():
            manuscript = self.get_manuscript(manuscript_id)
            if manuscript:
                manuscripts.append(manuscript)

        # Sort by creation date (newest first)
        manuscripts.sort(key=lambda m: m.created_at, reverse=True)
        return manuscripts