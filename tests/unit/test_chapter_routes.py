"""Unit tests for chapter routes."""

import json
import tempfile
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi import HTTPException

from illustrator.models import SavedManuscript, ManuscriptMetadata, Chapter, IllustrationPrompt, ImageProvider
from illustrator.web.routes.chapters import (
    load_manuscript_by_id,
    save_manuscript,
    SAVED_MANUSCRIPTS_DIR
)
from illustrator.web.models.web_models import ChapterHeaderResponse


class TestChapterRoutes:
    """Test class for chapter route functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_manuscripts_dir = self.temp_dir / "manuscripts"
        self.test_manuscripts_dir.mkdir(parents=True, exist_ok=True)

        # Create test manuscript with chapters
        self.test_manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test Novel",
                author="Test Author",
                genre="Fantasy",
                total_chapters=2,
                created_at=datetime.now().isoformat()
            ),
            chapters=[
                Chapter(
                    title="The Beginning",
                    content="This is the first chapter of our epic tale.",
                    number=1,
                    word_count=9
                ),
                Chapter(
                    title="The Journey Continues",
                    content="The adventure deepens as our heroes face new challenges.",
                    number=2,
                    word_count=10
                )
            ],
            saved_at=datetime.now().isoformat(),
            file_path=str(self.test_manuscripts_dir / "test_novel.json")
        )

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_test_manuscript_file(self, manuscript: SavedManuscript = None) -> Path:
        """Create a test manuscript file."""
        if manuscript is None:
            manuscript = self.test_manuscript

        file_path = Path(manuscript.file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(manuscript.model_dump(), f, indent=2)
        return file_path

    def get_chapter_id(self, manuscript_path: Path, chapter_number: int) -> str:
        """Generate chapter ID for testing."""
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(manuscript_path)))
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chapter_number}"))

    @patch('illustrator.web.routes.chapters.SAVED_MANUSCRIPTS_DIR')
    def test_load_manuscript_by_id_chapters(self, mock_manuscripts_dir):
        """Test loading manuscript specifically for chapter operations."""
        manuscript_file = self.create_test_manuscript_file()
        mock_manuscripts_dir.exists.return_value = True
        mock_manuscripts_dir.glob.return_value = [manuscript_file]

        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(manuscript_file)))
        manuscript, file_path = load_manuscript_by_id(manuscript_id)

        assert len(manuscript.chapters) == 2
        assert manuscript.chapters[0].title == "The Beginning"
        assert manuscript.chapters[1].title == "The Journey Continues"

    def test_chapter_model_creation(self):
        """Test creating chapter models."""
        content = "This is test content for the chapter"
        actual_word_count = len(content.split())

        chapter = Chapter(
            title="Test Chapter",
            content=content,
            number=1,
            word_count=actual_word_count
        )

        assert chapter.title == "Test Chapter"
        assert chapter.number == 1
        assert chapter.word_count == actual_word_count
        assert len(chapter.content.split()) == actual_word_count

    def test_chapter_model_validation(self):
        """Test chapter model validation."""
        # Test that word count must be calculated correctly
        chapter = Chapter(
            title="Test",
            content="One two three four five",
            number=1,
            word_count=5
        )

        assert chapter.word_count == 5

    def test_illustration_prompt_creation(self):
        """Test creating illustration prompts for chapters."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Fantasy chapter header showing a mystical forest",
            style_modifiers=["watercolor", "ethereal", "mystical"],
            negative_prompt="low quality, blurry",
            technical_params={
                "aspect_ratio": "16:9",
                "style": "artistic"
            }
        )

        assert prompt.provider == ImageProvider.DALLE
        assert "fantasy" in prompt.prompt.lower()
        assert len(prompt.style_modifiers) == 3
        assert prompt.negative_prompt == "low quality, blurry"

    def test_chapter_header_response_model(self):
        """Test chapter header response model creation."""
        from illustrator.web.models.web_models import ChapterHeaderOptionResponse

        # Create a sample illustration prompt
        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Symbolic chapter header for 'The Beginning', watercolor style",
            style_modifiers=["watercolor", "symbolic", "chapter header"],
            negative_prompt="text, words, low quality",
            technical_params={"aspect_ratio": "16:9"}
        )

        # Create header option
        option = ChapterHeaderOptionResponse(
            option_number=1,
            title="Symbolic Header",
            description="A symbolic representation focusing on the chapter's core themes",
            visual_focus="symbolic elements from the chapter",
            artistic_style="watercolor painting",
            composition_notes="Horizontal header layout with balanced composition",
            prompt=prompt
        )

        # Create header response
        response = ChapterHeaderResponse(
            chapter_id="test-chapter-id",
            chapter_title="The Beginning",
            header_options=[option]
        )

        assert response.chapter_title == "The Beginning"
        assert len(response.header_options) == 1
        assert response.header_options[0].option_number == 1
        assert response.header_options[0].artistic_style == "watercolor painting"

    def test_chapter_id_generation(self):
        """Test chapter ID generation logic."""
        manuscript_file = self.create_test_manuscript_file()
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(manuscript_file)))

        # Test chapter ID generation
        chapter_1_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_1"))
        chapter_2_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_2"))

        # IDs should be deterministic
        assert chapter_1_id == str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_1"))
        assert chapter_2_id == str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_2"))

        # IDs should be different
        assert chapter_1_id != chapter_2_id

    def test_save_manuscript_with_chapters(self):
        """Test saving manuscript with updated chapters."""
        manuscript = self.test_manuscript

        # Add a new chapter
        new_chapter = Chapter(
            title="New Chapter",
            content="This is a new chapter added to the manuscript.",
            number=3,
            word_count=10
        )
        manuscript.chapters.append(new_chapter)
        manuscript.metadata.total_chapters = 3

        file_path = self.test_manuscripts_dir / "updated_manuscript.json"
        save_manuscript(manuscript, file_path)

        # Verify the file was saved correctly
        assert file_path.exists()
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        assert saved_data["metadata"]["total_chapters"] == 3
        assert len(saved_data["chapters"]) == 3
        assert saved_data["chapters"][2]["title"] == "New Chapter"


class TestChapterHeaderGeneration:
    """Test chapter header generation functionality."""

    def test_header_option_types(self):
        """Test that different header option types are generated correctly."""
        option_types = ["Symbolic", "Character-focused", "Environmental", "Action"]
        styles = ["watercolor painting", "digital art", "pencil sketch", "oil painting"]

        for i, (option_type, style) in enumerate(zip(option_types, styles)):
            prompt = IllustrationPrompt(
                provider=ImageProvider.DALLE,
                prompt=f"{option_type} chapter header for 'Test Chapter', {style} style",
                style_modifiers=[style, "chapter header", "horizontal composition"],
                negative_prompt="text, words, letters, low quality",
                technical_params={
                    "aspect_ratio": "16:9",
                    "style": "artistic",
                    "quality": "high"
                }
            )

            from illustrator.web.models.web_models import ChapterHeaderOptionResponse
            option = ChapterHeaderOptionResponse(
                option_number=i + 1,
                title=f"{option_type} Header",
                description=f"A {option_type.lower()} representation focusing on the chapter's core themes",
                visual_focus=f"{option_type.lower()} elements from the chapter",
                artistic_style=style,
                composition_notes="Horizontal header layout with balanced composition",
                prompt=prompt
            )

            assert option.option_number == i + 1
            assert option.title == f"{option_type} Header"
            assert option.artistic_style == style
            assert option_type.lower() in option.visual_focus

    def test_technical_parameters_validation(self):
        """Test that technical parameters are properly structured."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Test prompt",
            style_modifiers=["test"],
            technical_params={
                "aspect_ratio": "16:9",
                "style": "artistic",
                "quality": "high",
                "seed": 12345
            }
        )

        assert prompt.technical_params["aspect_ratio"] == "16:9"
        assert prompt.technical_params["quality"] == "high"
        assert prompt.technical_params["seed"] == 12345

    def test_negative_prompt_handling(self):
        """Test negative prompt string handling."""
        # Test with None
        prompt1 = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Test prompt",
            style_modifiers=["test"],
            negative_prompt=None
        )
        assert prompt1.negative_prompt is None

        # Test with string
        prompt2 = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Test prompt",
            style_modifiers=["test"],
            negative_prompt="text, words, low quality"
        )
        assert prompt2.negative_prompt == "text, words, low quality"


class TestChapterOperations:
    """Test chapter CRUD operations."""

    def test_chapter_addition(self):
        """Test adding a new chapter to a manuscript."""
        manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test",
                total_chapters=0,
                created_at=datetime.now().isoformat()
            ),
            chapters=[],
            saved_at=datetime.now().isoformat(),
            file_path="/test/path.json"
        )

        new_chapter = Chapter(
            title="First Chapter",
            content="Content of the first chapter",
            number=1,
            word_count=5
        )

        manuscript.chapters.append(new_chapter)
        manuscript.metadata.total_chapters = 1

        assert len(manuscript.chapters) == 1
        assert manuscript.chapters[0].title == "First Chapter"
        assert manuscript.metadata.total_chapters == 1

    def test_chapter_deletion(self):
        """Test removing a chapter from a manuscript."""
        manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test",
                total_chapters=2,
                created_at=datetime.now().isoformat()
            ),
            chapters=[
                Chapter(title="Ch1", content="Content1", number=1, word_count=1),
                Chapter(title="Ch2", content="Content2", number=2, word_count=1)
            ],
            saved_at=datetime.now().isoformat(),
            file_path="/test/path.json"
        )

        # Remove first chapter
        manuscript.chapters.pop(0)
        # Renumber remaining chapters
        for i, chapter in enumerate(manuscript.chapters):
            chapter.number = i + 1
        manuscript.metadata.total_chapters = len(manuscript.chapters)

        assert len(manuscript.chapters) == 1
        assert manuscript.chapters[0].title == "Ch2"
        assert manuscript.chapters[0].number == 1  # Renumbered
        assert manuscript.metadata.total_chapters == 1

    def test_chapter_reordering(self):
        """Test reordering chapters in a manuscript."""
        chapters = [
            Chapter(title="Ch1", content="Content1", number=1, word_count=1),
            Chapter(title="Ch2", content="Content2", number=2, word_count=1),
            Chapter(title="Ch3", content="Content3", number=3, word_count=1)
        ]

        # Swap chapters 1 and 3
        chapters[0], chapters[2] = chapters[2], chapters[0]

        # Renumber
        for i, chapter in enumerate(chapters):
            chapter.number = i + 1

        assert chapters[0].title == "Ch3"
        assert chapters[0].number == 1
        assert chapters[2].title == "Ch1"
        assert chapters[2].number == 3