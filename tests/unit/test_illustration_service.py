"""Comprehensive unit tests for the illustration_service module."""

import json
import pytest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch
from uuid import UUID, uuid4

from illustrator.services.illustration_service import IllustrationService
from illustrator.models import EmotionalMoment, EmotionalTone, ImageProvider
from illustrator.db_models import Illustration


class TestIllustrationService:
    """Test cases for the IllustrationService class."""

    @pytest.fixture
    def mock_db_session(self):
        """Create a mock database session."""
        return MagicMock()

    @pytest.fixture
    def service_with_mock_db(self, mock_db_session):
        """Create service instance with mock database."""
        return IllustrationService(db=mock_db_session)

    @pytest.fixture
    def sample_emotional_moment(self):
        """Sample emotional moment for testing."""
        return EmotionalMoment(
            text_excerpt="This is a test emotional moment from the chapter.",
            start_position=10,
            end_position=56,
            emotional_tones=[EmotionalTone.JOY, EmotionalTone.EXCITEMENT],
            intensity_score=0.8,
            context="This is a test emotional moment from the chapter."
        )

    @pytest.fixture
    def sample_manuscript_id(self):
        """Sample manuscript ID."""
        return str(uuid4())

    @pytest.fixture
    def sample_chapter_id(self):
        """Sample chapter ID."""
        return str(uuid4())

    def test_service_initialization_with_db(self, mock_db_session):
        """Test service initialization with provided database session."""
        service = IllustrationService(db=mock_db_session)
        assert service.db is mock_db_session

    @patch('illustrator.services.illustration_service.get_db')
    def test_service_initialization_without_db(self, mock_get_db):
        """Test service initialization without provided database session."""
        mock_session = MagicMock()
        mock_get_db.return_value = mock_session

        service = IllustrationService()
        assert service.db is mock_session
        mock_get_db.assert_called_once()

    def test_save_illustration_basic(self, service_with_mock_db, sample_manuscript_id, sample_chapter_id):
        """Test saving a basic illustration."""
        result = service_with_mock_db.save_illustration(
            manuscript_id=sample_manuscript_id,
            chapter_id=sample_chapter_id,
            scene_number=1,
            filename="test_image.png",
            file_path="/path/to/test_image.png",
            prompt="Generate a test image",
            image_provider=ImageProvider.DALLE
        )

        # Verify database operations
        service_with_mock_db.db.add.assert_called_once()
        service_with_mock_db.db.commit.assert_called_once()
        service_with_mock_db.db.refresh.assert_called_once()

        # Verify the illustration object passed to db.add
        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        assert isinstance(illustration, Illustration)
        assert illustration.manuscript_id == UUID(sample_manuscript_id)
        assert illustration.chapter_id == UUID(sample_chapter_id)
        assert illustration.scene_number == 1
        assert illustration.filename == "test_image.png"
        assert illustration.file_path == "/path/to/test_image.png"
        assert illustration.web_url == "/generated/test_image.png"
        assert illustration.prompt == "Generate a test image"
        assert illustration.image_provider == ImageProvider.DALLE.value
        assert illustration.generation_status == "completed"

    def test_save_illustration_with_emotional_moment(
        self, service_with_mock_db, sample_manuscript_id, sample_chapter_id, sample_emotional_moment
    ):
        """Test saving an illustration with emotional moment."""
        result = service_with_mock_db.save_illustration(
            manuscript_id=sample_manuscript_id,
            chapter_id=sample_chapter_id,
            scene_number=1,
            filename="emotional_image.png",
            file_path="/path/to/emotional_image.png",
            prompt="Generate an emotional scene",
            image_provider=ImageProvider.IMAGEN4,
            emotional_moment=sample_emotional_moment
        )

        # Verify the illustration object
        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        assert illustration.emotional_tones == "joy,excitement"
        assert illustration.intensity_score == 0.8
        assert illustration.text_excerpt == sample_emotional_moment.text_excerpt
        assert illustration.title == "Chapter Scene 1"
        assert "joy" in illustration.description.lower()

    def test_save_illustration_with_all_parameters(
        self, service_with_mock_db, sample_manuscript_id, sample_chapter_id, sample_emotional_moment
    ):
        """Test saving an illustration with all parameters."""
        style_config = {
            "art_style": "watercolor",
            "color_palette": "warm",
            "artistic_influences": "Monet"
        }

        result = service_with_mock_db.save_illustration(
            manuscript_id=sample_manuscript_id,
            chapter_id=sample_chapter_id,
            scene_number=2,
            filename="full_params.jpg",
            file_path="/full/path/to/full_params.jpg",
            prompt="Detailed watercolor scene",
            image_provider=ImageProvider.FLUX,
            emotional_moment=sample_emotional_moment,
            style_config=style_config,
            title="Custom Title",
            description="Custom description",
            file_size=2048,
            width=1024,
            height=768
        )

        # Verify all parameters
        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        assert illustration.scene_number == 2
        assert illustration.filename == "full_params.jpg"
        assert illustration.file_path == "/full/path/to/full_params.jpg"
        assert illustration.title == "Custom Title"
        assert illustration.description == "Custom description"
        assert illustration.style_config == json.dumps(style_config)
        assert illustration.image_provider == ImageProvider.FLUX.value
        assert illustration.file_size == 2048
        assert illustration.width == 1024
        assert illustration.height == 768

    def test_save_illustration_without_emotional_moment(
        self, service_with_mock_db, sample_manuscript_id, sample_chapter_id
    ):
        """Test saving an illustration without emotional moment."""
        result = service_with_mock_db.save_illustration(
            manuscript_id=sample_manuscript_id,
            chapter_id=sample_chapter_id,
            scene_number=1,
            filename="no_emotion.png",
            file_path="/path/to/no_emotion.png",
            prompt="Simple scene",
            image_provider=ImageProvider.DALLE
        )

        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        assert illustration.emotional_tones is None
        assert illustration.intensity_score is None
        assert illustration.text_excerpt is None
        assert illustration.title == "Scene 1"  # Default title
        assert illustration.description is None

    def test_save_illustration_web_url_generation(
        self, service_with_mock_db, sample_manuscript_id, sample_chapter_id
    ):
        """Test web URL generation for different filenames."""
        test_cases = [
            ("simple.png", "/generated/simple.png"),
            ("complex_name.jpg", "/generated/complex_name.jpg"),
            ("image with spaces.gif", "/generated/image with spaces.gif"),
            ("special-chars_123.webp", "/generated/special-chars_123.webp")
        ]

        for filename, expected_url in test_cases:
            service_with_mock_db.db.reset_mock()

            service_with_mock_db.save_illustration(
                manuscript_id=sample_manuscript_id,
                chapter_id=sample_chapter_id,
                scene_number=1,
                filename=filename,
                file_path=f"/path/to/{filename}",
                prompt="Test prompt",
                image_provider=ImageProvider.DALLE
            )

            call_args = service_with_mock_db.db.add.call_args[0]
            illustration = call_args[0]
            assert illustration.web_url == expected_url

    def test_get_illustrations_by_manuscript(self, service_with_mock_db, sample_manuscript_id):
        """Test getting illustrations by manuscript ID."""
        # Mock query results
        mock_illustrations = [MagicMock(), MagicMock()]
        mock_query = service_with_mock_db.db.query.return_value
        mock_query.join.return_value.filter.return_value.order_by.return_value.all.return_value = mock_illustrations

        result = service_with_mock_db.get_illustrations_by_manuscript(sample_manuscript_id)

        assert result == mock_illustrations
        service_with_mock_db.db.query.assert_called_once_with(Illustration)
        mock_query.join.assert_called_once()
        mock_query.join.return_value.filter.assert_called_once()
        mock_query.join.return_value.filter.return_value.order_by.assert_called_once()

    def test_get_illustrations_by_chapter(self, service_with_mock_db, sample_chapter_id):
        """Test getting illustrations by chapter ID."""
        # Mock query results
        mock_illustrations = [MagicMock(), MagicMock(), MagicMock()]
        mock_query = service_with_mock_db.db.query.return_value
        mock_query.filter.return_value.order_by.return_value.all.return_value = mock_illustrations

        result = service_with_mock_db.get_illustrations_by_chapter(sample_chapter_id)

        assert result == mock_illustrations
        service_with_mock_db.db.query.assert_called_once_with(Illustration)
        mock_query.filter.assert_called_once()
        mock_query.filter.return_value.order_by.assert_called_once()

    def test_get_illustration_by_id_found(self, service_with_mock_db):
        """Test getting illustration by ID when found."""
        illustration_id = str(uuid4())
        mock_illustration = MagicMock()

        mock_query = service_with_mock_db.db.query.return_value
        mock_query.filter.return_value.first.return_value = mock_illustration

        result = service_with_mock_db.get_illustration_by_id(illustration_id)

        assert result == mock_illustration
        service_with_mock_db.db.query.assert_called_once_with(Illustration)
        mock_query.filter.assert_called_once()

    def test_get_illustration_by_id_not_found(self, service_with_mock_db):
        """Test getting illustration by ID when not found."""
        illustration_id = str(uuid4())

        mock_query = service_with_mock_db.db.query.return_value
        mock_query.filter.return_value.first.return_value = None

        result = service_with_mock_db.get_illustration_by_id(illustration_id)

        assert result is None

    def test_close_session(self, service_with_mock_db):
        """Test closing the database session."""
        service_with_mock_db.close()
        service_with_mock_db.db.close.assert_called_once()

    def test_close_session_with_none_db(self):
        """Test closing when db is None."""
        service = IllustrationService(db=None)
        service.close()  # Should not raise an exception

    def test_emotional_moment_without_tones(self, service_with_mock_db, sample_manuscript_id, sample_chapter_id):
        """Test saving illustration with emotional moment that has no tones."""
        emotional_moment = EmotionalMoment(
            text_excerpt="Neutral moment",
            start_position=0,
            end_position=13,
            emotional_tones=[],  # Empty tones
            intensity_score=0.5,
            context="Neutral moment"
        )

        service_with_mock_db.save_illustration(
            manuscript_id=sample_manuscript_id,
            chapter_id=sample_chapter_id,
            scene_number=1,
            filename="neutral.png",
            file_path="/path/to/neutral.png",
            prompt="Neutral scene",
            image_provider=ImageProvider.DALLE,
            emotional_moment=emotional_moment
        )

        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        assert illustration.emotional_tones == ""
        assert illustration.intensity_score == 0.5


class TestIllustrationServiceEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def service_with_mock_db(self):
        """Create service instance with mock database."""
        return IllustrationService(db=MagicMock())

    def test_save_illustration_invalid_uuids(self, service_with_mock_db):
        """Test saving illustration with invalid UUID strings."""
        with pytest.raises(ValueError):
            service_with_mock_db.save_illustration(
                manuscript_id="invalid-uuid",
                chapter_id="also-invalid-uuid",
                scene_number=1,
                filename="test.png",
                file_path="/path/test.png",
                prompt="Test",
                image_provider=ImageProvider.DALLE
            )

    def test_save_illustration_empty_strings(self, service_with_mock_db):
        """Test saving illustration with empty string parameters."""
        manuscript_id = str(uuid4())
        chapter_id = str(uuid4())

        service_with_mock_db.save_illustration(
            manuscript_id=manuscript_id,
            chapter_id=chapter_id,
            scene_number=1,
            filename="",  # Empty filename
            file_path="",  # Empty path
            prompt="",  # Empty prompt
            image_provider=ImageProvider.DALLE
        )

        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        assert illustration.filename == ""
        assert illustration.file_path == ""
        assert illustration.prompt == ""
        assert illustration.web_url == "/generated/"

    def test_save_illustration_zero_scene_number(self, service_with_mock_db):
        """Test saving illustration with zero scene number."""
        manuscript_id = str(uuid4())
        chapter_id = str(uuid4())

        service_with_mock_db.save_illustration(
            manuscript_id=manuscript_id,
            chapter_id=chapter_id,
            scene_number=0,
            filename="zero_scene.png",
            file_path="/path/zero_scene.png",
            prompt="Zero scene",
            image_provider=ImageProvider.DALLE
        )

        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]
        assert illustration.scene_number == 0

    def test_save_illustration_negative_values(self, service_with_mock_db):
        """Test saving illustration with negative values."""
        manuscript_id = str(uuid4())
        chapter_id = str(uuid4())

        service_with_mock_db.save_illustration(
            manuscript_id=manuscript_id,
            chapter_id=chapter_id,
            scene_number=1,
            filename="negative.png",
            file_path="/path/negative.png",
            prompt="Negative values test",
            image_provider=ImageProvider.DALLE,
            file_size=-100,  # Negative file size
            width=-512,      # Negative width
            height=-768      # Negative height
        )

        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        assert illustration.file_size == -100
        assert illustration.width == -512
        assert illustration.height == -768

    def test_save_illustration_large_values(self, service_with_mock_db):
        """Test saving illustration with very large values."""
        manuscript_id = str(uuid4())
        chapter_id = str(uuid4())

        large_text = "x" * 10000  # Very long text

        service_with_mock_db.save_illustration(
            manuscript_id=manuscript_id,
            chapter_id=chapter_id,
            scene_number=999999,
            filename="large_values.png",
            file_path="/path/large_values.png",
            prompt=large_text,
            image_provider=ImageProvider.DALLE,
            title=large_text,
            description=large_text,
            file_size=999999999,
            width=99999,
            height=99999
        )

        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        assert illustration.scene_number == 999999
        assert illustration.prompt == large_text
        assert illustration.title == large_text
        assert illustration.description == large_text
        assert illustration.file_size == 999999999

    def test_save_illustration_unicode_content(self, service_with_mock_db):
        """Test saving illustration with Unicode content."""
        manuscript_id = str(uuid4())
        chapter_id = str(uuid4())

        unicode_content = "测试内容 🎨 émotions café naïve résumé"

        service_with_mock_db.save_illustration(
            manuscript_id=manuscript_id,
            chapter_id=chapter_id,
            scene_number=1,
            filename="unicode_测试.png",
            file_path="/path/to/unicode_测试.png",
            prompt=f"Generate {unicode_content}",
            image_provider=ImageProvider.DALLE,
            title=f"Title with {unicode_content}",
            description=f"Description: {unicode_content}"
        )

        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        # The filename is "unicode_测试.png", which contains Unicode characters
        assert "unicode_测试.png" == illustration.filename
        assert unicode_content in illustration.prompt
        assert unicode_content in illustration.title
        assert unicode_content in illustration.description

    def test_style_config_serialization(self, service_with_mock_db):
        """Test style config JSON serialization."""
        manuscript_id = str(uuid4())
        chapter_id = str(uuid4())

        complex_style_config = {
            "art_style": "watercolor",
            "nested": {
                "colors": ["red", "blue", "green"],
                "settings": {
                    "opacity": 0.8,
                    "texture": True
                }
            },
            "list_values": [1, 2, 3],
            "unicode": "café résumé"
        }

        service_with_mock_db.save_illustration(
            manuscript_id=manuscript_id,
            chapter_id=chapter_id,
            scene_number=1,
            filename="complex_config.png",
            file_path="/path/complex_config.png",
            prompt="Complex style test",
            image_provider=ImageProvider.DALLE,
            style_config=complex_style_config
        )

        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        # Verify JSON serialization
        stored_config = json.loads(illustration.style_config)
        assert stored_config == complex_style_config

    def test_get_methods_with_invalid_uuids(self, service_with_mock_db):
        """Test get methods with invalid UUID strings."""
        # These should raise ValueError when trying to convert to UUID
        with pytest.raises(ValueError):
            service_with_mock_db.get_illustrations_by_manuscript("invalid-uuid")

        with pytest.raises(ValueError):
            service_with_mock_db.get_illustrations_by_chapter("invalid-uuid")

        with pytest.raises(ValueError):
            service_with_mock_db.get_illustration_by_id("invalid-uuid")

    def test_database_error_handling(self, service_with_mock_db):
        """Test handling of database errors."""
        service_with_mock_db.db.commit.side_effect = Exception("Database error")

        service = service_with_mock_db

        with pytest.raises(Exception, match="Database error"):
            service.save_illustration(
                manuscript_id=str(uuid4()),
                chapter_id=str(uuid4()),
                scene_number=1,
                filename="error_test.png",
                file_path="/path/error_test.png",
                prompt="Error test",
                image_provider=ImageProvider.DALLE
            )

    def test_emotional_moment_edge_cases(self, service_with_mock_db):
        """Test emotional moment with edge case values."""
        manuscript_id = str(uuid4())
        chapter_id = str(uuid4())

        # Test with extreme values
        emotional_moment = EmotionalMoment(
            text_excerpt="",  # Empty excerpt
            start_position=0,
            end_position=0,
            emotional_tones=[EmotionalTone.JOY] * 10,  # Many tones
            intensity_score=1.0,  # Max intensity
            context=""
        )

        service_with_mock_db.save_illustration(
            manuscript_id=manuscript_id,
            chapter_id=chapter_id,
            scene_number=1,
            filename="edge_case.png",
            file_path="/path/edge_case.png",
            prompt="Edge case test",
            image_provider=ImageProvider.DALLE,
            emotional_moment=emotional_moment
        )

        call_args = service_with_mock_db.db.add.call_args[0]
        illustration = call_args[0]

        # Should handle many duplicate tones
        assert "joy" in illustration.emotional_tones
        assert illustration.intensity_score == 1.0
        assert illustration.text_excerpt == ""


class TestIllustrationServiceIntegration:
    """Integration-style tests for IllustrationService."""

    @pytest.fixture
    def service_with_mock_db(self):
        """Create service instance with mock database."""
        return IllustrationService(db=MagicMock())

    def test_full_workflow_single_illustration(self, service_with_mock_db):
        """Test complete workflow for saving and retrieving a single illustration."""
        manuscript_id = str(uuid4())
        chapter_id = str(uuid4())

        emotional_moment = EmotionalMoment(
            text_excerpt="A moment of triumph and joy in the story",
            start_position=0,
            end_position=42,
            emotional_tones=[EmotionalTone.JOY, EmotionalTone.EXCITEMENT],
            intensity_score=0.9,
            context="A moment of triumph and joy in the story"
        )

        # Save illustration
        saved_illustration = service_with_mock_db.save_illustration(
            manuscript_id=manuscript_id,
            chapter_id=chapter_id,
            scene_number=1,
            filename="triumph_scene.jpg",
            file_path="/illustrations/triumph_scene.jpg",
            prompt="A triumphant moment in watercolor style",
            image_provider=ImageProvider.IMAGEN4,
            emotional_moment=emotional_moment,
            style_config={"style": "watercolor"},
            title="The Moment of Triumph",
            description="Hero's triumphant moment",
            file_size=2048,
            width=1024,
            height=768
        )

        # Verify save operation
        service_with_mock_db.db.add.assert_called_once()
        service_with_mock_db.db.commit.assert_called_once()

        # Mock retrieval
        mock_illustration = MagicMock()
        service_with_mock_db.db.query.return_value.filter.return_value.first.return_value = mock_illustration

        test_id = str(uuid4())
        retrieved = service_with_mock_db.get_illustration_by_id(test_id)
        assert retrieved == mock_illustration

    def test_multiple_illustrations_workflow(self, service_with_mock_db):
        """Test workflow with multiple illustrations for the same chapter."""
        manuscript_id = str(uuid4())
        chapter_id = str(uuid4())

        # Create multiple emotional moments
        moments = [
            EmotionalMoment(
                text_excerpt=f"Emotional moment {i}",
                start_position=i * 20,
                end_position=(i * 20) + 18,
                emotional_tones=[EmotionalTone.JOY],
                intensity_score=0.5 + i * 0.1,
                context=f"Emotional moment {i} context"
            )
            for i in range(3)
        ]

        # Save multiple illustrations
        for i, moment in enumerate(moments, 1):
            service_with_mock_db.save_illustration(
                manuscript_id=manuscript_id,
                chapter_id=chapter_id,
                scene_number=i,
                filename=f"scene_{i}.png",
                file_path=f"/illustrations/scene_{i}.png",
                prompt=f"Generate scene {i}",
                image_provider=ImageProvider.DALLE,
                emotional_moment=moment
            )

        # Verify all were saved
        assert service_with_mock_db.db.add.call_count == 3
        assert service_with_mock_db.db.commit.call_count == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])