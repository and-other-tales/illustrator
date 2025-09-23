"""Comprehensive unit tests for the database module."""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, mock_open
from uuid import uuid4

from illustrator.database import DatabaseManager
from illustrator.models import Chapter, ManuscriptMetadata
from illustrator.web.models.web_models import ManuscriptResponse, ChapterResponse


class TestDatabaseManager:
    """Test cases for the DatabaseManager class."""

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create a temporary directory for testing."""
        return tmp_path / "test_db"

    @pytest.fixture
    def db_manager(self, temp_dir):
        """Create a DatabaseManager instance for testing."""
        return DatabaseManager(data_dir=str(temp_dir))

    @pytest.fixture
    def sample_manuscript_metadata(self):
        """Sample manuscript metadata for testing."""
        return ManuscriptMetadata(
            title="Test Novel",
            author="Test Author",
            genre="Fantasy",
            total_chapters=2,
            created_at=datetime.now().isoformat()
        )

    @pytest.fixture
    def sample_chapter(self):
        """Sample chapter for testing."""
        return Chapter(
            title="Chapter 1",
            content="This is test chapter content.",
            number=1,
            word_count=5
        )

    @pytest.fixture
    def sample_manuscript_response(self, sample_manuscript_metadata):
        """Sample manuscript response for testing."""
        return ManuscriptResponse(
            id=str(uuid4()),
            metadata=sample_manuscript_metadata,
            chapters=[],
            total_images=0,
            processing_status="draft",
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )

    @pytest.fixture
    def sample_chapter_response(self, sample_chapter):
        """Sample chapter response for testing."""
        return ChapterResponse(
            id=str(uuid4()),
            chapter=sample_chapter,
            analysis=None,
            images_generated=0,
            processing_status="draft"
        )

    def test_database_manager_initialization(self, temp_dir):
        """Test DatabaseManager initialization."""
        db_manager = DatabaseManager(data_dir=str(temp_dir))

        assert db_manager.data_dir == temp_dir
        assert db_manager.manuscripts_dir == temp_dir / "manuscripts"
        assert db_manager.chapters_dir == temp_dir / "chapters"
        assert db_manager.manuscripts_index_file == temp_dir / "manuscripts_index.json"
        assert db_manager.chapters_index_file == temp_dir / "chapters_index.json"

        # Check directories were created
        assert db_manager.data_dir.exists()
        assert db_manager.manuscripts_dir.exists()
        assert db_manager.chapters_dir.exists()

    def test_database_manager_default_data_dir(self):
        """Test DatabaseManager with default data directory."""
        with patch('pathlib.Path.mkdir'):
            db_manager = DatabaseManager()
            assert str(db_manager.data_dir) == "data"

    def test_load_indexes_empty(self, db_manager):
        """Test loading indexes when files don't exist."""
        assert db_manager.manuscripts_index == {}
        assert db_manager.chapters_index == {}

    def test_load_indexes_existing_files(self, temp_dir):
        """Test loading existing index files."""
        manuscripts_index = {"manuscript1": {"title": "Test"}}
        chapters_index = {"chapter1": {"title": "Chapter 1"}}

        # Create index files
        manuscripts_index_file = temp_dir / "manuscripts_index.json"
        chapters_index_file = temp_dir / "chapters_index.json"

        manuscripts_index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(manuscripts_index_file, 'w') as f:
            json.dump(manuscripts_index, f)

        chapters_index_file.parent.mkdir(parents=True, exist_ok=True)
        with open(chapters_index_file, 'w') as f:
            json.dump(chapters_index, f)

        # Initialize DatabaseManager
        db_manager = DatabaseManager(data_dir=str(temp_dir))

        assert db_manager.manuscripts_index == manuscripts_index
        assert db_manager.chapters_index == chapters_index

    def test_save_indexes(self, db_manager):
        """Test saving index files."""
        db_manager.manuscripts_index = {"manuscript1": {"title": "Test"}}
        db_manager.chapters_index = {"chapter1": {"title": "Chapter 1"}}

        db_manager._save_indexes()

        # Verify files were created and contain correct data
        assert db_manager.manuscripts_index_file.exists()
        assert db_manager.chapters_index_file.exists()

        with open(db_manager.manuscripts_index_file) as f:
            saved_manuscripts = json.load(f)
        with open(db_manager.chapters_index_file) as f:
            saved_chapters = json.load(f)

        assert saved_manuscripts == {"manuscript1": {"title": "Test"}}
        assert saved_chapters == {"chapter1": {"title": "Chapter 1"}}

    def test_get_manuscript_not_found(self, db_manager):
        """Test getting a manuscript that doesn't exist."""
        result = db_manager.get_manuscript("nonexistent_id")
        assert result is None

    def test_get_manuscript_file_not_found(self, db_manager):
        """Test getting a manuscript when index exists but file doesn't."""
        db_manager.manuscripts_index = {"manuscript1": {"title": "Test"}}
        result = db_manager.get_manuscript("manuscript1")
        assert result is None

    def test_get_manuscript_success(self, db_manager, sample_manuscript_response):
        """Test successfully getting a manuscript."""
        manuscript_id = sample_manuscript_response.id

        # Add to index
        db_manager.manuscripts_index = {
            manuscript_id: {
                "title": sample_manuscript_response.metadata.title,
                "author": sample_manuscript_response.metadata.author,
                "created_at": sample_manuscript_response.created_at,
                "updated_at": sample_manuscript_response.updated_at,
                "total_chapters": 0
            }
        }

        # Create manuscript file
        manuscript_file = db_manager.manuscripts_dir / f"{manuscript_id}.json"
        with open(manuscript_file, 'w') as f:
            json.dump(sample_manuscript_response.dict(), f)

        result = db_manager.get_manuscript(manuscript_id)

        assert result is not None
        assert result.id == manuscript_id
        assert result.metadata.title == sample_manuscript_response.metadata.title

    def test_get_chapters_by_manuscript_id_empty(self, db_manager):
        """Test getting chapters for a manuscript with no chapters."""
        result = db_manager.get_chapters_by_manuscript_id("nonexistent_manuscript")
        assert result == []

    def test_get_chapters_by_manuscript_id_success(self, db_manager, sample_chapter_response):
        """Test getting chapters for a manuscript."""
        manuscript_id = "test_manuscript"
        chapter_id = sample_chapter_response.id

        # Add to chapters index
        db_manager.chapters_index = {
            chapter_id: {
                "manuscript_id": manuscript_id,
                "title": sample_chapter_response.chapter.title,
                "number": sample_chapter_response.chapter.number,
                "word_count": sample_chapter_response.chapter.word_count
            }
        }

        # Create chapter file
        chapter_file = db_manager.chapters_dir / f"{chapter_id}.json"
        with open(chapter_file, 'w') as f:
            json.dump(sample_chapter_response.dict(), f)

        result = db_manager.get_chapters_by_manuscript_id(manuscript_id)

        assert len(result) == 1
        assert result[0].id == chapter_id
        assert result[0].chapter.title == sample_chapter_response.chapter.title

    def test_save_manuscript_success(self, db_manager, sample_manuscript_response):
        """Test saving a manuscript."""
        manuscript_id = db_manager.save_manuscript(sample_manuscript_response)

        assert manuscript_id == sample_manuscript_response.id

        # Check file was created
        manuscript_file = db_manager.manuscripts_dir / f"{manuscript_id}.json"
        assert manuscript_file.exists()

        # Check index was updated
        assert manuscript_id in db_manager.manuscripts_index
        index_entry = db_manager.manuscripts_index[manuscript_id]
        assert index_entry["title"] == sample_manuscript_response.metadata.title
        assert index_entry["author"] == sample_manuscript_response.metadata.author

    def test_save_chapter_success(self, db_manager, sample_chapter_response):
        """Test saving a chapter."""
        manuscript_id = "test_manuscript"
        chapter_id = db_manager.save_chapter(sample_chapter_response, manuscript_id)

        assert chapter_id == sample_chapter_response.id

        # Check file was created
        chapter_file = db_manager.chapters_dir / f"{chapter_id}.json"
        assert chapter_file.exists()

        # Check index was updated
        assert chapter_id in db_manager.chapters_index
        index_entry = db_manager.chapters_index[chapter_id]
        assert index_entry["manuscript_id"] == manuscript_id
        assert index_entry["title"] == sample_chapter_response.chapter.title

    def test_create_manuscript_success(self, db_manager):
        """Test creating a new manuscript."""
        manuscript = db_manager.create_manuscript(
            title="New Test Novel",
            author="New Author",
            genre="Science Fiction"
        )

        assert manuscript.metadata.title == "New Test Novel"
        assert manuscript.metadata.author == "New Author"
        assert manuscript.metadata.genre == "Science Fiction"
        assert manuscript.metadata.total_chapters == 0
        assert manuscript.processing_status == "draft"
        assert len(manuscript.chapters) == 0

        # Check it was saved
        assert manuscript.id in db_manager.manuscripts_index

    def test_create_manuscript_minimal(self, db_manager):
        """Test creating a manuscript with minimal parameters."""
        manuscript = db_manager.create_manuscript(title="Minimal Novel")

        assert manuscript.metadata.title == "Minimal Novel"
        assert manuscript.metadata.author is None
        assert manuscript.metadata.genre is None

    def test_create_chapter_success(self, db_manager):
        """Test creating a new chapter."""
        manuscript_id = "test_manuscript"

        chapter = db_manager.create_chapter(
            title="Test Chapter",
            content="This is test content for the chapter.",
            manuscript_id=manuscript_id,
            number=1
        )

        assert chapter.chapter.title == "Test Chapter"
        assert chapter.chapter.content == "This is test content for the chapter."
        assert chapter.chapter.number == 1
        assert chapter.chapter.word_count == 8  # Number of words in content
        assert chapter.processing_status == "draft"
        assert chapter.analysis is None

        # Check it was saved
        assert chapter.id in db_manager.chapters_index

    def test_create_chapter_auto_number(self, db_manager, sample_chapter_response):
        """Test creating a chapter with auto-assigned number."""
        manuscript_id = "test_manuscript"

        # Save an existing chapter first
        db_manager.save_chapter(sample_chapter_response, manuscript_id)

        # Create new chapter without specifying number
        chapter = db_manager.create_chapter(
            title="Second Chapter",
            content="Second chapter content.",
            manuscript_id=manuscript_id
        )

        assert chapter.chapter.number == 2  # Should be auto-assigned as 2

    def test_list_manuscripts_empty(self, db_manager):
        """Test listing manuscripts when database is empty."""
        result = db_manager.list_manuscripts()
        assert result == []

    def test_list_manuscripts_success(self, db_manager, sample_manuscript_response):
        """Test listing manuscripts."""
        # Save a manuscript
        db_manager.save_manuscript(sample_manuscript_response)

        result = db_manager.list_manuscripts()

        assert len(result) == 1
        assert result[0].id == sample_manuscript_response.id
        assert result[0].metadata.title == sample_manuscript_response.metadata.title

    def test_list_manuscripts_multiple_sorted(self, db_manager):
        """Test listing multiple manuscripts sorted by creation date."""
        # Create manuscripts with different timestamps
        from datetime import datetime, timedelta

        base_time = datetime.now()
        older_manuscript = ManuscriptResponse(
            id=str(uuid4()),
            metadata=ManuscriptMetadata(
                title="Older Novel",
                author="Author",
                genre="Fantasy",
                total_chapters=0,
                created_at=(base_time - timedelta(days=1)).isoformat()
            ),
            chapters=[],
            total_images=0,
            processing_status="draft",
            created_at=(base_time - timedelta(days=1)).isoformat(),
            updated_at=(base_time - timedelta(days=1)).isoformat()
        )

        newer_manuscript = ManuscriptResponse(
            id=str(uuid4()),
            metadata=ManuscriptMetadata(
                title="Newer Novel",
                author="Author",
                genre="Fantasy",
                total_chapters=0,
                created_at=base_time.isoformat()
            ),
            chapters=[],
            total_images=0,
            processing_status="draft",
            created_at=base_time.isoformat(),
            updated_at=base_time.isoformat()
        )

        # Save both manuscripts
        db_manager.save_manuscript(older_manuscript)
        db_manager.save_manuscript(newer_manuscript)

        result = db_manager.list_manuscripts()

        assert len(result) == 2
        # Should be sorted by creation date, newest first
        assert result[0].metadata.title == "Newer Novel"
        assert result[1].metadata.title == "Older Novel"

    def test_list_manuscripts_with_missing_files(self, db_manager):
        """Test listing manuscripts when some files are missing."""
        # Add manuscript to index but don't create file
        db_manager.manuscripts_index = {
            "missing_manuscript": {
                "title": "Missing Novel",
                "author": "Author",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "total_chapters": 0
            }
        }
        db_manager._save_indexes()

        result = db_manager.list_manuscripts()

        # Should return empty list since file doesn't exist
        assert result == []


class TestDatabaseManagerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager instance for testing."""
        return DatabaseManager(data_dir=str(tmp_path / "test_db"))

    def test_get_manuscript_invalid_json(self, db_manager):
        """Test getting a manuscript with invalid JSON file."""
        manuscript_id = "invalid_manuscript"

        # Add to index
        db_manager.manuscripts_index = {manuscript_id: {"title": "Test"}}

        # Create file with invalid JSON
        manuscript_file = db_manager.manuscripts_dir / f"{manuscript_id}.json"
        with open(manuscript_file, 'w') as f:
            f.write("invalid json content")

        # Should raise an exception or handle gracefully
        with pytest.raises((json.JSONDecodeError, Exception)):
            db_manager.get_manuscript(manuscript_id)

    def test_get_chapters_missing_files(self, db_manager):
        """Test getting chapters when some files are missing."""
        manuscript_id = "test_manuscript"

        # Add chapter to index but don't create file
        db_manager.chapters_index = {
            "missing_chapter": {
                "manuscript_id": manuscript_id,
                "title": "Missing Chapter",
                "number": 1,
                "word_count": 100
            }
        }

        result = db_manager.get_chapters_by_manuscript_id(manuscript_id)

        # Should return empty list since file doesn't exist
        assert result == []

    def test_create_chapter_word_count_calculation(self, db_manager):
        """Test chapter creation with various content for word count calculation."""
        test_cases = [
            ("", 0),  # Empty content
            ("word", 1),  # Single word
            ("hello world", 2),  # Two words
            ("  hello   world  ", 2),  # Extra whitespace
            ("word\nword\tword", 3),  # Different whitespace types
        ]

        for content, expected_count in test_cases:
            chapter = db_manager.create_chapter(
                title="Test Chapter",
                content=content,
                manuscript_id="test_manuscript"
            )
            assert chapter.chapter.word_count == expected_count

    def test_save_operations_create_directories(self, tmp_path):
        """Test that save operations create necessary directories."""
        # Initialize with non-existent directory
        data_dir = tmp_path / "nested" / "test_db"
        db_manager = DatabaseManager(data_dir=str(data_dir))

        # Directories should be created during initialization
        assert db_manager.data_dir.exists()
        assert db_manager.manuscripts_dir.exists()
        assert db_manager.chapters_dir.exists()


class TestDatabaseManagerIntegration:
    """Integration tests for DatabaseManager operations."""

    @pytest.fixture
    def db_manager(self, tmp_path):
        """Create a DatabaseManager instance for testing."""
        return DatabaseManager(data_dir=str(tmp_path / "integration_db"))

    def test_full_manuscript_workflow(self, db_manager):
        """Test complete workflow: create manuscript, add chapters, retrieve all."""
        # Create manuscript
        manuscript = db_manager.create_manuscript(
            title="Integration Test Novel",
            author="Test Author",
            genre="Testing"
        )

        # Add chapters
        chapter1 = db_manager.create_chapter(
            title="First Chapter",
            content="Content of the first chapter with multiple words.",
            manuscript_id=manuscript.id,
            number=1
        )

        chapter2 = db_manager.create_chapter(
            title="Second Chapter",
            content="Content of the second chapter.",
            manuscript_id=manuscript.id,
            number=2
        )

        # Retrieve and verify
        retrieved_manuscript = db_manager.get_manuscript(manuscript.id)
        assert retrieved_manuscript is not None
        assert retrieved_manuscript.metadata.title == "Integration Test Novel"

        chapters = db_manager.get_chapters_by_manuscript_id(manuscript.id)
        assert len(chapters) == 2
        assert chapters[0].chapter.number == 1
        assert chapters[1].chapter.number == 2

        # List all manuscripts
        all_manuscripts = db_manager.list_manuscripts()
        assert len(all_manuscripts) == 1
        assert all_manuscripts[0].id == manuscript.id

    def test_multiple_manuscripts_chapters_separation(self, db_manager):
        """Test that chapters are correctly separated by manuscript."""
        # Create two manuscripts
        manuscript1 = db_manager.create_manuscript(title="Novel 1")
        manuscript2 = db_manager.create_manuscript(title="Novel 2")

        # Add chapters to each
        chapter1_m1 = db_manager.create_chapter(
            title="Chapter 1 of Novel 1",
            content="Content 1",
            manuscript_id=manuscript1.id
        )

        chapter1_m2 = db_manager.create_chapter(
            title="Chapter 1 of Novel 2",
            content="Content 2",
            manuscript_id=manuscript2.id
        )

        chapter2_m1 = db_manager.create_chapter(
            title="Chapter 2 of Novel 1",
            content="More content",
            manuscript_id=manuscript1.id
        )

        # Retrieve chapters for each manuscript separately
        chapters_m1 = db_manager.get_chapters_by_manuscript_id(manuscript1.id)
        chapters_m2 = db_manager.get_chapters_by_manuscript_id(manuscript2.id)

        assert len(chapters_m1) == 2
        assert len(chapters_m2) == 1

        assert chapters_m1[0].chapter.title == "Chapter 1 of Novel 1"
        assert chapters_m1[1].chapter.title == "Chapter 2 of Novel 1"
        assert chapters_m2[0].chapter.title == "Chapter 1 of Novel 2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])