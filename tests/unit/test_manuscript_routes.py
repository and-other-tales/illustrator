"""Unit tests for manuscript routes."""

import json
import tempfile
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi import HTTPException

from illustrator.models import SavedManuscript, ManuscriptMetadata, Chapter
from illustrator.web.routes.manuscripts import (
    get_saved_manuscripts,
    save_manuscript_to_disk,
    count_generated_images,
    SAVED_MANUSCRIPTS_DIR,
    ILLUSTRATOR_OUTPUT_DIR
)


class TestManuscriptRoutes:
    """Test class for manuscript route functions."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_manuscripts_dir = self.temp_dir / "manuscripts"
        self.test_manuscripts_dir.mkdir(parents=True, exist_ok=True)
        self.test_output_dir = self.temp_dir / "output"
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        # Create test manuscript data
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
                    title="Chapter 1",
                    content="This is test content for chapter one.",
                    number=1,
                    word_count=8
                ),
                Chapter(
                    title="Chapter 2",
                    content="This is test content for chapter two with more words.",
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

    @patch('illustrator.web.routes.manuscripts.SAVED_MANUSCRIPTS_DIR')
    def test_get_saved_manuscripts_empty_directory(self, mock_manuscripts_dir):
        """Test get_saved_manuscripts with empty directory."""
        mock_manuscripts_dir.__bool__ = MagicMock(return_value=False)
        mock_manuscripts_dir.exists.return_value = False

        result = get_saved_manuscripts()
        assert result == []

    @patch('illustrator.web.routes.manuscripts.SAVED_MANUSCRIPTS_DIR')
    def test_get_saved_manuscripts_with_files(self, mock_manuscripts_dir):
        """Test get_saved_manuscripts with valid manuscript files."""
        mock_manuscripts_dir.exists.return_value = True
        mock_manuscripts_dir.glob.return_value = [self.create_test_manuscript_file()]

        result = get_saved_manuscripts()
        assert len(result) == 1
        assert result[0].metadata.title == "Test Novel"
        assert result[0].metadata.author == "Test Author"
        assert len(result[0].chapters) == 2

    @patch('illustrator.web.routes.manuscripts.SAVED_MANUSCRIPTS_DIR')
    def test_get_saved_manuscripts_with_invalid_file(self, mock_manuscripts_dir):
        """Test get_saved_manuscripts with invalid JSON file."""
        invalid_file = self.test_manuscripts_dir / "invalid.json"
        invalid_file.write_text("invalid json content")

        mock_manuscripts_dir.exists.return_value = True
        mock_manuscripts_dir.glob.return_value = [invalid_file]

        result = get_saved_manuscripts()
        assert result == []  # Invalid files should be skipped


    def test_save_manuscript(self):
        """Test save_manuscript function."""
        manuscript = self.test_manuscript

        file_path = save_manuscript_to_disk(manuscript)

        assert file_path.exists()
        with open(file_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)

        assert saved_data["metadata"]["title"] == "Test Novel"
        assert saved_data["file_path"] == str(file_path)
        assert "saved_at" in saved_data

    @patch('illustrator.web.routes.manuscripts.ILLUSTRATOR_OUTPUT_DIR')
    def test_count_generated_images_no_directory(self, mock_output_dir):
        """Test count_generated_images when output directory doesn't exist."""
        mock_output_dir.__truediv__.return_value.exists.return_value = False

        result = count_generated_images("Test Novel")
        assert result == 0

    @patch('illustrator.web.routes.manuscripts.ILLUSTRATOR_OUTPUT_DIR')
    def test_count_generated_images_with_files(self, mock_output_dir):
        """Test count_generated_images with image files."""
        # Create test image files
        mock_png = MagicMock()
        mock_png.is_file.return_value = True
        mock_png.suffix = '.png'

        mock_jpg = MagicMock()
        mock_jpg.is_file.return_value = True
        mock_jpg.suffix = '.jpg'

        mock_jpeg = MagicMock()
        mock_jpeg.is_file.return_value = True
        mock_jpeg.suffix = '.jpeg'

        mock_txt = MagicMock()
        mock_txt.is_file.return_value = True
        mock_txt.suffix = '.txt'

        mock_images_dir = MagicMock()
        mock_images_dir.exists.return_value = True
        mock_images_dir.iterdir.return_value = [mock_png, mock_jpg, mock_jpeg, mock_txt]

        mock_output_dir.__truediv__.return_value = mock_images_dir

        result = count_generated_images("Test Novel")
        assert result == 3  # Only image files should be counted


class TestManuscriptUtilities:
    """Test utility functions for manuscript handling."""

    def test_manuscript_model_validation(self):
        """Test that manuscript models validate correctly."""
        # Test valid manuscript
        manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test",
                author="Author",
                genre="Fiction",
                total_chapters=1,
                created_at="2024-01-01T00:00:00"
            ),
            chapters=[],
            saved_at="2024-01-01T00:00:00",
            file_path="/test/path.json"
        )

        assert manuscript.metadata.title == "Test"
        assert manuscript.metadata.author == "Author"
        assert len(manuscript.chapters) == 0

    def test_manuscript_serialization(self):
        """Test that manuscript can be serialized and deserialized."""
        original = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test Novel",
                author="Test Author",
                genre="Fantasy",
                total_chapters=1,
                created_at="2024-01-01T00:00:00"
            ),
            chapters=[
                Chapter(
                    title="Chapter 1",
                    content="Test content",
                    number=1,
                    word_count=2
                )
            ],
            saved_at="2024-01-01T00:00:00",
            file_path="/test/path.json"
        )

        # Serialize to dict
        data = original.model_dump()

        # Deserialize back to model
        restored = SavedManuscript(**data)

        assert restored.metadata.title == original.metadata.title
        assert len(restored.chapters) == len(original.chapters)
        assert restored.chapters[0].title == original.chapters[0].title