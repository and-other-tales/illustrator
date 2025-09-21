"""Unit tests for data models."""

import pytest
from pydantic import ValidationError

from illustrator.models import (
    Chapter,
    ChapterAnalysis,
    EmotionalMoment,
    EmotionalTone,
    IllustrationPrompt,
    ImageProvider,
    ManuscriptMetadata,
)


class TestChapter:
    """Test Chapter model."""

    def test_chapter_creation_valid(self):
        """Test creating a valid chapter."""
        chapter = Chapter(
            title="Test Chapter",
            content="This is a test chapter content.",
            number=1,
            word_count=6
        )
        assert chapter.title == "Test Chapter"
        assert chapter.content == "This is a test chapter content."
        assert chapter.number == 1
        assert chapter.word_count == 6

    def test_chapter_validation_errors(self):
        """Test chapter validation errors."""
        with pytest.raises(ValidationError):
            Chapter(
                # Missing required fields should fail
                title="Test"
                # Missing content, number, word_count
            )


class TestEmotionalMoment:
    """Test EmotionalMoment model."""

    def test_emotional_moment_creation(self):
        """Test creating a valid emotional moment."""
        moment = EmotionalMoment(
            text_excerpt="A dramatic moment in the story.",
            start_position=100,
            end_position=135,
            emotional_tones=[EmotionalTone.TENSION, EmotionalTone.FEAR],
            intensity_score=0.8,
            context="Character facing danger"
        )
        assert moment.text_excerpt == "A dramatic moment in the story."
        assert moment.intensity_score == 0.8
        assert EmotionalTone.TENSION in moment.emotional_tones
        assert EmotionalTone.FEAR in moment.emotional_tones

    def test_intensity_score_validation(self):
        """Test intensity score bounds."""
        # Valid intensity scores
        moment = EmotionalMoment(
            text_excerpt="Test",
            start_position=0,
            end_position=4,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.5,
            context="Test context"
        )
        assert moment.intensity_score == 0.5

        # Should work with edge values
        moment_min = EmotionalMoment(
            text_excerpt="Test",
            start_position=0,
            end_position=4,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.0,
            context="Test context"
        )
        assert moment_min.intensity_score == 0.0

        moment_max = EmotionalMoment(
            text_excerpt="Test",
            start_position=0,
            end_position=4,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=1.0,
            context="Test context"
        )
        assert moment_max.intensity_score == 1.0


class TestIllustrationPrompt:
    """Test IllustrationPrompt model."""

    def test_illustration_prompt_creation(self):
        """Test creating an illustration prompt."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="A beautiful digital painting of a fantasy landscape",
            style_modifiers=["digital art", "fantasy", "detailed"],
            negative_prompt=None,
            technical_params={"size": "1024x1024", "quality": "hd"}
        )

        assert prompt.provider == ImageProvider.DALLE
        assert "fantasy landscape" in prompt.prompt
        assert "digital art" in prompt.style_modifiers
        assert prompt.technical_params["size"] == "1024x1024"

    def test_all_providers_supported(self):
        """Test that all image providers are supported."""
        providers = [ImageProvider.DALLE, ImageProvider.IMAGEN4, ImageProvider.FLUX]

        for provider in providers:
            prompt = IllustrationPrompt(
                provider=provider,
                prompt="Test prompt",
                style_modifiers=["test"],
                technical_params={}
            )
            assert prompt.provider == provider


class TestChapterAnalysis:
    """Test ChapterAnalysis model."""

    def test_complete_chapter_analysis(self):
        """Test creating a complete chapter analysis."""
        chapter = Chapter(
            title="Test Chapter",
            content="A story with emotional moments.",
            number=1,
            word_count=6
        )

        emotional_moment = EmotionalMoment(
            text_excerpt="emotional moments",
            start_position=14,
            end_position=30,
            emotional_tones=[EmotionalTone.ANTICIPATION],
            intensity_score=0.7,
            context="Building tension"
        )

        illustration_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Dramatic scene with tension",
            style_modifiers=["dramatic", "tense"],
            technical_params={"size": "1024x1024"}
        )

        analysis = ChapterAnalysis(
            chapter=chapter,
            emotional_moments=[emotional_moment],
            dominant_themes=["tension", "anticipation"],
            setting_description="A mysterious setting",
            character_emotions={"protagonist": [EmotionalTone.ANTICIPATION]},
            illustration_prompts=[illustration_prompt]
        )

        assert analysis.chapter.title == "Test Chapter"
        assert len(analysis.emotional_moments) == 1
        assert len(analysis.illustration_prompts) == 1
        assert "tension" in analysis.dominant_themes


class TestManuscriptMetadata:
    """Test ManuscriptMetadata model."""

    def test_manuscript_metadata_creation(self):
        """Test creating manuscript metadata."""
        metadata = ManuscriptMetadata(
            title="The Test Novel",
            author="Test Author",
            genre="Fantasy",
            total_chapters=10,
            created_at="2024-01-01T12:00:00"
        )

        assert metadata.title == "The Test Novel"
        assert metadata.author == "Test Author"
        assert metadata.genre == "Fantasy"
        assert metadata.total_chapters == 10

    def test_optional_fields(self):
        """Test optional fields in metadata."""
        metadata = ManuscriptMetadata(
            title="Test Novel",
            total_chapters=5,
            created_at="2024-01-01T12:00:00"
        )

        assert metadata.title == "Test Novel"
        assert metadata.author is None
        assert metadata.genre is None