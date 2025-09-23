"""Comprehensive unit tests for the db_models module."""

import pytest
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from illustrator.db_models import Base, Manuscript, Chapter, Illustration, ProcessingSession


@pytest.fixture
def engine():
    """Create in-memory SQLite engine for testing."""
    engine = create_engine("sqlite:///:memory:")

    # Enable foreign key constraints in SQLite
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine

@pytest.fixture
def session(engine):
    """Create database session for testing."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()

@pytest.fixture
def sample_manuscript(session):
    """Create a sample manuscript for testing."""
    manuscript = Manuscript(
        title="Test Novel",
        author="Test Author",
        genre="Fantasy",
        total_chapters=2
    )
    session.add(manuscript)
    session.commit()
    return manuscript

@pytest.fixture
def sample_chapter(session, sample_manuscript):
    """Create a sample chapter for testing."""
    chapter = Chapter(
        manuscript_id=sample_manuscript.id,
        title="Chapter 1",
        content="This is test chapter content.",
        number=1,
        word_count=5
    )
    session.add(chapter)
    session.commit()
    return chapter


class TestDatabaseModels:
    """Test database model definitions and relationships."""


class TestManuscriptModel:
    """Test the Manuscript model."""

    def test_manuscript_creation(self, session):
        """Test creating a manuscript."""
        manuscript = Manuscript(
            title="Test Novel",
            author="Test Author",
            genre="Fantasy",
            total_chapters=1
        )

        session.add(manuscript)
        session.commit()

        assert manuscript.id is not None
        assert isinstance(manuscript.id, UUID)
        assert manuscript.title == "Test Novel"
        assert manuscript.author == "Test Author"
        assert manuscript.genre == "Fantasy"
        assert manuscript.total_chapters == 1
        assert manuscript.created_at is not None
        assert manuscript.updated_at is not None

    def test_manuscript_required_fields(self, session):
        """Test manuscript with required fields only."""
        manuscript = Manuscript(title="Minimal Novel")

        session.add(manuscript)
        session.commit()

        assert manuscript.title == "Minimal Novel"
        assert manuscript.author is None
        assert manuscript.genre is None
        assert manuscript.total_chapters == 0

    def test_manuscript_auto_uuid(self, session):
        """Test that manuscript ID is automatically generated."""
        manuscript1 = Manuscript(title="Novel 1")
        manuscript2 = Manuscript(title="Novel 2")

        session.add_all([manuscript1, manuscript2])
        session.commit()

        assert manuscript1.id != manuscript2.id
        assert isinstance(manuscript1.id, UUID)
        assert isinstance(manuscript2.id, UUID)

    def test_manuscript_timestamps(self, session):
        """Test manuscript timestamp handling."""
        manuscript = Manuscript(title="Test Novel")

        # Before saving
        assert manuscript.created_at is None
        assert manuscript.updated_at is None

        session.add(manuscript)
        session.commit()

        # After saving
        assert manuscript.created_at is not None
        assert manuscript.updated_at is not None

        original_updated = manuscript.updated_at

        # Update and check timestamp change
        import time
        time.sleep(0.01)  # Small delay to ensure timestamp difference
        manuscript.title = "Updated Novel"
        session.commit()

        assert manuscript.updated_at >= original_updated

    def test_manuscript_string_length_limits(self, session):
        """Test manuscript string field length limits."""
        # Test title length (500 chars)
        long_title = "x" * 500
        manuscript = Manuscript(title=long_title)
        session.add(manuscript)
        session.commit()
        assert manuscript.title == long_title

        # Test author length (255 chars)
        long_author = "x" * 255
        manuscript2 = Manuscript(title="Test", author=long_author)
        session.add(manuscript2)
        session.commit()
        assert manuscript2.author == long_author

        # Test genre length (100 chars)
        long_genre = "x" * 100
        manuscript3 = Manuscript(title="Test", genre=long_genre)
        session.add(manuscript3)
        session.commit()
        assert manuscript3.genre == long_genre

    def test_manuscript_relationships(self, session, sample_manuscript):
        """Test manuscript relationships."""
        # Initially no relationships
        assert len(sample_manuscript.chapters) == 0
        assert len(sample_manuscript.illustrations) == 0

        # Add chapter
        chapter = Chapter(
            manuscript_id=sample_manuscript.id,
            title="Test Chapter",
            content="Content",
            number=1
        )
        session.add(chapter)
        session.commit()

        # Refresh and check relationship
        session.refresh(sample_manuscript)
        assert len(sample_manuscript.chapters) == 1
        assert sample_manuscript.chapters[0].title == "Test Chapter"


class TestChapterModel:
    """Test the Chapter model."""

    def test_chapter_creation(self, session, sample_manuscript):
        """Test creating a chapter."""
        chapter = Chapter(
            manuscript_id=sample_manuscript.id,
            title="Test Chapter",
            content="This is test content for the chapter.",
            number=1,
            word_count=8
        )

        session.add(chapter)
        session.commit()

        assert chapter.id is not None
        assert isinstance(chapter.id, UUID)
        assert chapter.manuscript_id == sample_manuscript.id
        assert chapter.title == "Test Chapter"
        assert chapter.content == "This is test content for the chapter."
        assert chapter.number == 1
        assert chapter.word_count == 8
        assert chapter.created_at is not None
        assert chapter.updated_at is not None

    def test_chapter_required_fields(self, session, sample_manuscript):
        """Test chapter with required fields only."""
        chapter = Chapter(
            manuscript_id=sample_manuscript.id,
            title="Minimal Chapter",
            content="Minimal content",
            number=1
        )

        session.add(chapter)
        session.commit()

        assert chapter.word_count == 0  # Default value
        assert chapter.created_at is not None

    def test_chapter_manuscript_relationship(self, session, sample_manuscript):
        """Test chapter-manuscript relationship."""
        chapter = Chapter(
            manuscript_id=sample_manuscript.id,
            title="Test Chapter",
            content="Content",
            number=1
        )

        session.add(chapter)
        session.commit()

        assert chapter.manuscript == sample_manuscript

    def test_chapter_unique_constraint(self, session, sample_manuscript):
        """Test unique constraint on manuscript_id and number."""
        chapter1 = Chapter(
            manuscript_id=sample_manuscript.id,
            title="Chapter 1",
            content="Content 1",
            number=1
        )

        chapter2 = Chapter(
            manuscript_id=sample_manuscript.id,
            title="Chapter 1 Again",
            content="Content 2",
            number=1  # Same number as chapter1
        )

        session.add(chapter1)
        session.commit()

        session.add(chapter2)
        with pytest.raises(IntegrityError):
            session.commit()

    def test_chapter_different_manuscripts_same_number(self, session):
        """Test that same chapter numbers are allowed across different manuscripts."""
        manuscript1 = Manuscript(title="Novel 1")
        manuscript2 = Manuscript(title="Novel 2")
        session.add_all([manuscript1, manuscript2])
        session.commit()

        chapter1 = Chapter(
            manuscript_id=manuscript1.id,
            title="Chapter 1",
            content="Content 1",
            number=1
        )

        chapter2 = Chapter(
            manuscript_id=manuscript2.id,
            title="Chapter 1",
            content="Content 2",
            number=1  # Same number but different manuscript
        )

        session.add_all([chapter1, chapter2])
        session.commit()  # Should not raise an error

        assert chapter1.number == chapter2.number == 1
        assert chapter1.manuscript_id != chapter2.manuscript_id


class TestIllustrationModel:
    """Test the Illustration model."""

    def test_illustration_creation(self, session, sample_manuscript, sample_chapter):
        """Test creating an illustration."""
        illustration = Illustration(
            manuscript_id=sample_manuscript.id,
            chapter_id=sample_chapter.id,
            filename="test_image.png",
            file_path="/path/to/test_image.png",
            web_url="/generated/test_image.png",
            scene_number=1,
            title="Test Illustration",
            description="A test illustration",
            prompt="Generate a test image",
            image_provider="dalle",
            emotional_tones="joy,excitement",
            intensity_score=0.8,
            text_excerpt="Excerpt from the chapter",
            file_size=1024,
            width=512,
            height=512
        )

        session.add(illustration)
        session.commit()

        assert illustration.id is not None
        assert isinstance(illustration.id, UUID)
        assert illustration.manuscript_id == sample_manuscript.id
        assert illustration.chapter_id == sample_chapter.id
        assert illustration.filename == "test_image.png"
        assert illustration.file_path == "/path/to/test_image.png"
        assert illustration.web_url == "/generated/test_image.png"
        assert illustration.scene_number == 1
        assert illustration.title == "Test Illustration"
        assert illustration.description == "A test illustration"
        assert illustration.prompt == "Generate a test image"
        assert illustration.image_provider == "dalle"
        assert illustration.emotional_tones == "joy,excitement"
        assert illustration.intensity_score == 0.8
        assert illustration.text_excerpt == "Excerpt from the chapter"
        assert illustration.file_size == 1024
        assert illustration.width == 512
        assert illustration.height == 512
        assert illustration.generation_status == "completed"

    def test_illustration_required_fields(self, session, sample_manuscript, sample_chapter):
        """Test illustration with required fields only."""
        illustration = Illustration(
            manuscript_id=sample_manuscript.id,
            chapter_id=sample_chapter.id,
            filename="minimal.png",
            file_path="/path/to/minimal.png",
            web_url="/generated/minimal.png",
            scene_number=1,
            prompt="Minimal prompt",
            image_provider="dalle"
        )

        session.add(illustration)
        session.commit()

        assert illustration.title is None
        assert illustration.description is None
        assert illustration.generation_status == "completed"

    def test_illustration_relationships(self, session, sample_manuscript, sample_chapter):
        """Test illustration relationships."""
        illustration = Illustration(
            manuscript_id=sample_manuscript.id,
            chapter_id=sample_chapter.id,
            filename="test.png",
            file_path="/path/to/test.png",
            web_url="/generated/test.png",
            scene_number=1,
            prompt="Test prompt",
            image_provider="dalle"
        )

        session.add(illustration)
        session.commit()

        assert illustration.manuscript == sample_manuscript
        assert illustration.chapter == sample_chapter

    def test_illustration_unique_constraint(self, session, sample_manuscript, sample_chapter):
        """Test unique constraint on chapter_id and scene_number."""
        illustration1 = Illustration(
            manuscript_id=sample_manuscript.id,
            chapter_id=sample_chapter.id,
            filename="test1.png",
            file_path="/path/to/test1.png",
            web_url="/generated/test1.png",
            scene_number=1,
            prompt="Prompt 1",
            image_provider="dalle"
        )

        illustration2 = Illustration(
            manuscript_id=sample_manuscript.id,
            chapter_id=sample_chapter.id,
            filename="test2.png",
            file_path="/path/to/test2.png",
            web_url="/generated/test2.png",
            scene_number=1,  # Same scene number
            prompt="Prompt 2",
            image_provider="dalle"
        )

        session.add(illustration1)
        session.commit()

        session.add(illustration2)
        with pytest.raises(IntegrityError):
            session.commit()

    def test_illustration_float_intensity_score(self, session, sample_manuscript, sample_chapter):
        """Test illustration with float intensity score."""
        test_scores = [0.0, 0.5, 0.75, 1.0, -0.5, 1.5]

        for i, score in enumerate(test_scores, 1):
            illustration = Illustration(
                manuscript_id=sample_manuscript.id,
                chapter_id=sample_chapter.id,
                filename=f"test_{score}.png",
                file_path=f"/path/to/test_{score}.png",
                web_url=f"/generated/test_{score}.png",
                scene_number=i,  # Use incrementing scene numbers
                prompt="Test prompt",
                image_provider="dalle",
                intensity_score=score
            )

            session.add(illustration)

        session.commit()

        # Verify all were saved
        illustrations = session.query(Illustration).all()
        saved_scores = [ill.intensity_score for ill in illustrations if ill.intensity_score is not None]
        assert len(saved_scores) == len(test_scores)


class TestProcessingSessionModel:
    """Test the ProcessingSession model."""

    def test_processing_session_creation(self, session, sample_manuscript):
        """Test creating a processing session."""
        processing_session = ProcessingSession(
            manuscript_id=sample_manuscript.id,
            session_type="illustration_generation",
            status="running",
            progress_percent=50,
            current_task="Processing chapter 2",
            style_config='{"style": "digital painting"}',
            max_emotional_moments=5,
            total_images_generated=3,
            total_chapters_processed=2
        )

        session.add(processing_session)
        session.commit()

        assert processing_session.id is not None
        assert isinstance(processing_session.id, UUID)
        assert processing_session.manuscript_id == sample_manuscript.id
        assert processing_session.session_type == "illustration_generation"
        assert processing_session.status == "running"
        assert processing_session.progress_percent == 50
        assert processing_session.current_task == "Processing chapter 2"
        assert processing_session.style_config == '{"style": "digital painting"}'
        assert processing_session.max_emotional_moments == 5
        assert processing_session.total_images_generated == 3
        assert processing_session.total_chapters_processed == 2
        assert processing_session.started_at is not None
        assert processing_session.created_at is not None
        assert processing_session.updated_at is not None

    def test_processing_session_defaults(self, session, sample_manuscript):
        """Test processing session with default values."""
        processing_session = ProcessingSession(manuscript_id=sample_manuscript.id)

        session.add(processing_session)
        session.commit()

        assert processing_session.session_type == "illustration_generation"
        assert processing_session.status == "pending"
        assert processing_session.progress_percent == 0
        assert processing_session.max_emotional_moments == 10
        assert processing_session.total_images_generated == 0
        assert processing_session.total_chapters_processed == 0

    def test_processing_session_completion(self, session, sample_manuscript):
        """Test processing session completion flow."""
        processing_session = ProcessingSession(
            manuscript_id=sample_manuscript.id,
            status="running"
        )

        session.add(processing_session)
        session.commit()

        # Complete the session
        processing_session.status = "completed"
        processing_session.progress_percent = 100
        processing_session.completed_at = datetime.utcnow()
        session.commit()

        assert processing_session.status == "completed"
        assert processing_session.progress_percent == 100
        assert processing_session.completed_at is not None

    def test_processing_session_error_handling(self, session, sample_manuscript):
        """Test processing session error handling."""
        processing_session = ProcessingSession(
            manuscript_id=sample_manuscript.id,
            status="failed",
            error_message="Test error occurred during processing"
        )

        session.add(processing_session)
        session.commit()

        assert processing_session.status == "failed"
        assert processing_session.error_message == "Test error occurred during processing"

    def test_processing_session_relationship(self, session, sample_manuscript):
        """Test processing session manuscript relationship."""
        processing_session = ProcessingSession(manuscript_id=sample_manuscript.id)

        session.add(processing_session)
        session.commit()

        assert processing_session.manuscript == sample_manuscript


class TestModelRelationships:
    """Test relationships between models."""

    def test_manuscript_chapter_relationship(self, session, sample_manuscript):
        """Test manuscript-chapter relationship."""
        chapters = [
            Chapter(
                manuscript_id=sample_manuscript.id,
                title=f"Chapter {i}",
                content=f"Content {i}",
                number=i
            )
            for i in range(1, 4)
        ]

        session.add_all(chapters)
        session.commit()

        session.refresh(sample_manuscript)
        assert len(sample_manuscript.chapters) == 3
        assert all(ch.manuscript_id == sample_manuscript.id for ch in sample_manuscript.chapters)

    def test_manuscript_illustration_relationship(self, session, sample_manuscript, sample_chapter):
        """Test manuscript-illustration relationship."""
        illustrations = [
            Illustration(
                manuscript_id=sample_manuscript.id,
                chapter_id=sample_chapter.id,
                filename=f"image_{i}.png",
                file_path=f"/path/to/image_{i}.png",
                web_url=f"/generated/image_{i}.png",
                scene_number=i,
                prompt=f"Prompt {i}",
                image_provider="dalle"
            )
            for i in range(1, 4)
        ]

        session.add_all(illustrations)
        session.commit()

        session.refresh(sample_manuscript)
        assert len(sample_manuscript.illustrations) == 3
        assert all(ill.manuscript_id == sample_manuscript.id for ill in sample_manuscript.illustrations)

    def test_chapter_illustration_relationship(self, session, sample_manuscript, sample_chapter):
        """Test chapter-illustration relationship."""
        illustrations = [
            Illustration(
                manuscript_id=sample_manuscript.id,
                chapter_id=sample_chapter.id,
                filename=f"ch_image_{i}.png",
                file_path=f"/path/to/ch_image_{i}.png",
                web_url=f"/generated/ch_image_{i}.png",
                scene_number=i,
                prompt=f"Chapter prompt {i}",
                image_provider="dalle"
            )
            for i in range(1, 3)
        ]

        session.add_all(illustrations)
        session.commit()

        session.refresh(sample_chapter)
        assert len(sample_chapter.illustrations) == 2
        assert all(ill.chapter_id == sample_chapter.id for ill in sample_chapter.illustrations)


class TestModelValidation:
    """Test model validation and constraints."""

    def test_manuscript_title_not_null(self, session):
        """Test that manuscript title cannot be null."""
        manuscript = Manuscript(author="Author")

        session.add(manuscript)
        with pytest.raises(IntegrityError):
            session.commit()

    def test_chapter_foreign_key_constraint(self, session):
        """Test chapter foreign key constraint."""
        chapter = Chapter(
            manuscript_id=uuid4(),  # Non-existent manuscript
            title="Test Chapter",
            content="Content",
            number=1
        )

        session.add(chapter)
        with pytest.raises(IntegrityError):
            session.commit()

    def test_illustration_foreign_key_constraints(self, session):
        """Test illustration foreign key constraints."""
        # Test with non-existent manuscript
        illustration1 = Illustration(
            manuscript_id=uuid4(),  # Non-existent
            chapter_id=uuid4(),     # Non-existent
            filename="test.png",
            file_path="/path/test.png",
            web_url="/generated/test.png",
            scene_number=1,
            prompt="Test",
            image_provider="dalle"
        )

        session.add(illustration1)
        with pytest.raises(IntegrityError):
            session.commit()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])