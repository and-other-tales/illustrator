"""Unit tests for web models."""

from datetime import datetime
from typing import List

import pytest
from pydantic import ValidationError

from illustrator.models import (
    Chapter, ManuscriptMetadata, SavedManuscript,
    IllustrationPrompt, ImageProvider, EmotionalTone
)
from illustrator.web.models.web_models import (
    ManuscriptCreateRequest,
    ManuscriptResponse,
    ChapterCreateRequest,
    ChapterResponse,
    ChapterHeaderResponse,
    ChapterHeaderOptionResponse,
    ImageResponse,
    GalleryResponse,
    SuccessResponse,
    ErrorResponse
)


class TestWebModels:
    """Test web-specific model validation and serialization."""

    def test_manuscript_create_request_valid(self):
        """Test valid manuscript creation request."""
        request = ManuscriptCreateRequest(
            title="Test Novel",
            author="Test Author",
            genre="Fantasy"
        )

        assert request.title == "Test Novel"
        assert request.author == "Test Author"
        assert request.genre == "Fantasy"

    def test_manuscript_create_request_minimal(self):
        """Test manuscript creation request with minimal data."""
        request = ManuscriptCreateRequest(title="Test Novel")

        assert request.title == "Test Novel"
        assert request.author is None
        assert request.genre is None

    def test_manuscript_create_request_validation_errors(self):
        """Test manuscript creation request validation failures."""
        # Empty title should fail
        with pytest.raises(ValidationError):
            ManuscriptCreateRequest(title="")

        # Too long title should fail
        with pytest.raises(ValidationError):
            ManuscriptCreateRequest(title="x" * 201)

        # Too long author should fail
        with pytest.raises(ValidationError):
            ManuscriptCreateRequest(title="Valid Title", author="x" * 101)

        # Too long genre should fail
        with pytest.raises(ValidationError):
            ManuscriptCreateRequest(title="Valid Title", genre="x" * 51)

    def test_manuscript_response_model(self):
        """Test manuscript response model."""
        metadata = ManuscriptMetadata(
            title="Test Novel",
            author="Test Author",
            genre="Fantasy",
            total_chapters=2,
            created_at=datetime.now().isoformat()
        )

        chapters = [
            Chapter(title="Ch1", content="Content1", number=1, word_count=1),
            Chapter(title="Ch2", content="Content2", number=2, word_count=1)
        ]

        response = ManuscriptResponse(
            id="test-id",
            metadata=metadata,
            chapters=chapters,
            total_images=5,
            processing_status="completed",
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-02T00:00:00"
        )

        assert response.id == "test-id"
        assert response.metadata.title == "Test Novel"
        assert len(response.chapters) == 2
        assert response.total_images == 5
        assert response.processing_status == "completed"

    def test_chapter_create_request_valid(self):
        """Test valid chapter creation request."""
        request = ChapterCreateRequest(
            title="Chapter 1",
            content="This is the content of the first chapter.",
            manuscript_id="test-manuscript-id"
        )

        assert request.title == "Chapter 1"
        assert request.manuscript_id == "test-manuscript-id"
        assert len(request.content) > 10

    def test_chapter_create_request_validation_errors(self):
        """Test chapter creation request validation failures."""
        # Empty title should fail
        with pytest.raises(ValidationError):
            ChapterCreateRequest(
                title="",
                content="Valid content",
                manuscript_id="test-id"
            )

        # Too short content should fail
        with pytest.raises(ValidationError):
            ChapterCreateRequest(
                title="Valid Title",
                content="Short",
                manuscript_id="test-id"
            )

        # Too long title should fail
        with pytest.raises(ValidationError):
            ChapterCreateRequest(
                title="x" * 201,
                content="Valid content here",
                manuscript_id="test-id"
            )

    def test_chapter_response_model(self):
        """Test chapter response model."""
        chapter = Chapter(
            title="Test Chapter",
            content="This is test content",
            number=1,
            word_count=4
        )

        response = ChapterResponse(
            id="chapter-id",
            chapter=chapter,
            analysis=None,
            images_generated=3,
            processing_status="draft"
        )

        assert response.id == "chapter-id"
        assert response.chapter.title == "Test Chapter"
        assert response.images_generated == 3
        assert response.processing_status == "draft"

    def test_chapter_header_option_response(self):
        """Test chapter header option response model."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Fantasy chapter header",
            style_modifiers=["watercolor", "mystical"],
            negative_prompt="low quality",
            technical_params={"aspect_ratio": "16:9"}
        )

        option = ChapterHeaderOptionResponse(
            option_number=1,
            title="Mystical Header",
            description="A mystical representation of the chapter",
            visual_focus="magical elements",
            artistic_style="watercolor painting",
            composition_notes="Horizontal layout",
            prompt=prompt
        )

        assert option.option_number == 1
        assert option.title == "Mystical Header"
        assert option.artistic_style == "watercolor painting"
        assert option.prompt.provider == ImageProvider.DALLE

    def test_chapter_header_response(self):
        """Test chapter header response model."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Test prompt",
            style_modifiers=["test"],
        )

        option1 = ChapterHeaderOptionResponse(
            option_number=1,
            title="Option 1",
            description="First option",
            visual_focus="focus1",
            artistic_style="style1",
            composition_notes="notes1",
            prompt=prompt
        )

        option2 = ChapterHeaderOptionResponse(
            option_number=2,
            title="Option 2",
            description="Second option",
            visual_focus="focus2",
            artistic_style="style2",
            composition_notes="notes2",
            prompt=prompt
        )

        response = ChapterHeaderResponse(
            chapter_id="test-chapter-id",
            chapter_title="Test Chapter",
            header_options=[option1, option2]
        )

        assert response.chapter_id == "test-chapter-id"
        assert response.chapter_title == "Test Chapter"
        assert len(response.header_options) == 2
        assert response.header_options[0].option_number == 1
        assert response.header_options[1].option_number == 2

    def test_image_response_model(self):
        """Test image response model."""
        response = ImageResponse(
            id="image-id",
            chapter_number=1,
            scene_number=1,
            image_path="/generated/test_image.png",
            thumbnail_path="/generated/thumbs/test_image.png",
            prompt="A beautiful landscape",
            emotional_moment="peaceful meadow",
            quality_scores={"aesthetic": 0.8, "technical": 0.9},
            metadata={"width": 1024, "height": 768},
            generated_at="2024-01-01T00:00:00"
        )

        assert response.id == "image-id"
        assert response.chapter_number == 1
        assert response.quality_scores["aesthetic"] == 0.8
        assert response.metadata["width"] == 1024

    def test_gallery_response_model(self):
        """Test gallery response model."""
        image = ImageResponse(
            id="img1",
            chapter_number=1,
            scene_number=1,
            image_path="/test.png",
            thumbnail_path="/thumb.png",
            prompt="test prompt",
            emotional_moment="test moment",
            metadata={},
            generated_at="2024-01-01T00:00:00"
        )

        response = GalleryResponse(
            manuscript_id="manuscript-id",
            manuscript_title="Test Novel",
            total_images=1,
            images_by_chapter={"ch1": [image]}
        )

        assert response.manuscript_id == "manuscript-id"
        assert response.manuscript_title == "Test Novel"
        assert response.total_images == 1
        assert len(response.images_by_chapter["ch1"]) == 1

    def test_success_response_model(self):
        """Test success response model."""
        response = SuccessResponse(
            message="Operation completed successfully",
            data={"key": "value"}
        )

        assert response.message == "Operation completed successfully"
        assert response.data["key"] == "value"

    def test_error_response_model(self):
        """Test error response model."""
        response = ErrorResponse(
            error="Validation error",
            detail="Field 'title' is required",
            code="VALIDATION_ERROR"
        )

        assert response.error == "Validation error"
        assert response.detail == "Field 'title' is required"


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_manuscript_response_serialization(self):
        """Test manuscript response can be serialized to JSON."""
        metadata = ManuscriptMetadata(
            title="Test",
            total_chapters=1,
            created_at="2024-01-01T00:00:00"
        )

        chapters = [
            Chapter(title="Ch1", content="Content", number=1, word_count=1)
        ]

        response = ManuscriptResponse(
            id="test-id",
            metadata=metadata,
            chapters=chapters,
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )

        # Should serialize without errors
        serialized = response.model_dump()
        assert serialized["id"] == "test-id"
        assert serialized["metadata"]["title"] == "Test"
        assert len(serialized["chapters"]) == 1

    def test_chapter_header_response_serialization(self):
        """Test chapter header response serialization."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Test prompt",
            style_modifiers=["test"]
        )

        option = ChapterHeaderOptionResponse(
            option_number=1,
            title="Test Option",
            description="Test description",
            visual_focus="test focus",
            artistic_style="test style",
            composition_notes="test notes",
            prompt=prompt
        )

        response = ChapterHeaderResponse(
            chapter_id="test-id",
            chapter_title="Test Chapter",
            header_options=[option]
        )

        serialized = response.model_dump()
        assert serialized["chapter_id"] == "test-id"
        assert len(serialized["header_options"]) == 1
        assert serialized["header_options"][0]["prompt"]["provider"] == "dalle"


class TestModelFieldValidation:
    """Test specific field validation rules."""

    def test_image_provider_validation(self):
        """Test image provider enum validation."""
        # Valid providers
        for provider in [ImageProvider.DALLE, ImageProvider.IMAGEN4, ImageProvider.FLUX, ImageProvider.SEEDREAM]:
            prompt = IllustrationPrompt(
                provider=provider,
                prompt="test",
                style_modifiers=["test"]
            )
            assert prompt.provider == provider

    def test_emotional_tone_validation(self):
        """Test emotional tone enum validation in extended scenarios."""
        from illustrator.models import EmotionalMoment

        # Test with valid emotional tones
        valid_tones = [EmotionalTone.JOY, EmotionalTone.SADNESS, EmotionalTone.MYSTERY]

        moment = EmotionalMoment(
            text_excerpt="Test excerpt",
            start_position=0,
            end_position=12,
            emotional_tones=valid_tones,
            intensity_score=0.8,
            context="Test context"
        )

        assert len(moment.emotional_tones) == 3
        assert EmotionalTone.JOY in moment.emotional_tones

    def test_word_count_consistency(self):
        """Test that word counts are handled consistently."""
        content = "This is a test chapter with exactly ten words here."
        actual_count = len(content.split())

        chapter = Chapter(
            title="Test Chapter",
            content=content,
            number=1,
            word_count=actual_count
        )

        assert chapter.word_count == 10
        assert chapter.word_count == len(chapter.content.split())

    def test_technical_params_flexibility(self):
        """Test that technical parameters can handle various data types."""
        params = {
            "aspect_ratio": "16:9",
            "quality": "high",
            "seed": 12345,
            "temperature": 0.7,
            "steps": 50,
            "cfg_scale": 7.5,
            "custom_flag": True
        }

        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="test",
            style_modifiers=["test"],
            technical_params=params
        )

        assert prompt.technical_params["seed"] == 12345
        assert prompt.technical_params["temperature"] == 0.7
        assert prompt.technical_params["custom_flag"] is True
