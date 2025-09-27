"""Comprehensive unit tests for models module."""

import pytest
from datetime import datetime
from typing import List, Dict, Any

from illustrator.models import (
    ImageProvider,
    EmotionalTone,
    Chapter,
    EmotionalMoment,
    IllustrationPrompt,
    StyleConfig,
    ChapterAnalysis,
    ManuscriptMetadata,
    SavedManuscript
)


class TestImageProvider:
    """Test ImageProvider enum."""

    def test_image_provider_values(self):
        """Test that all expected providers exist."""
        assert ImageProvider.DALLE == "dalle"
        assert ImageProvider.IMAGEN4 == "imagen4"
        assert ImageProvider.FLUX == "flux"
        assert ImageProvider.SEEDREAM == "seedream"

    def test_image_provider_string_conversion(self):
        """Test string conversion of providers."""
        assert ImageProvider.DALLE.value == "dalle"
        assert ImageProvider.IMAGEN4.value == "imagen4"
        assert ImageProvider.FLUX.value == "flux"
        assert ImageProvider.SEEDREAM.value == "seedream"


class TestEmotionalTone:
    """Test EmotionalTone enum."""

    def test_basic_emotions(self):
        """Test basic emotional tones."""
        assert EmotionalTone.JOY == "joy"
        assert EmotionalTone.SADNESS == "sadness"
        assert EmotionalTone.ANGER == "anger"
        assert EmotionalTone.FEAR == "fear"
        assert EmotionalTone.SURPRISE == "surprise"
        assert EmotionalTone.DISGUST == "disgust"

    def test_extended_emotions(self):
        """Test extended emotional tones."""
        assert EmotionalTone.ANTICIPATION == "anticipation"
        assert EmotionalTone.TRUST == "trust"
        assert EmotionalTone.MELANCHOLY == "melancholy"
        assert EmotionalTone.EXCITEMENT == "excitement"
        assert EmotionalTone.TENSION == "tension"
        assert EmotionalTone.MYSTERY == "mystery"
        assert EmotionalTone.ROMANCE == "romance"
        assert EmotionalTone.ADVENTURE == "adventure"
        assert EmotionalTone.SUSPENSE == "suspense"
        assert EmotionalTone.COURAGE == "courage"
        assert EmotionalTone.NEUTRAL == "neutral"


class TestChapter:
    """Test Chapter model."""

    def test_chapter_creation(self):
        """Test basic chapter creation."""
        chapter = Chapter(
            title="Test Chapter",
            content="This is test content for the chapter.",
            number=1,
            word_count=8
        )

        assert chapter.title == "Test Chapter"
        assert chapter.content == "This is test content for the chapter."
        assert chapter.number == 1
        assert chapter.word_count == 8

    def test_chapter_validation(self):
        """Test chapter validation."""
        # Test with minimum required fields
        chapter = Chapter(
            title="",
            content="",
            number=0,
            word_count=0
        )

        assert chapter.title == ""
        assert chapter.content == ""
        assert chapter.number == 0
        assert chapter.word_count == 0

    def test_chapter_with_long_content(self):
        """Test chapter with longer content."""
        content = " ".join(["word"] * 100)
        chapter = Chapter(
            title="Long Chapter",
            content=content,
            number=2,
            word_count=100
        )

        assert len(chapter.content.split()) == 100
        assert chapter.word_count == 100


class TestEmotionalMoment:
    """Test EmotionalMoment model."""

    def test_emotional_moment_creation(self):
        """Test emotional moment creation."""
        moment = EmotionalMoment(
            text_excerpt="She felt a surge of hope.",
            start_position=100,
            end_position=128,
            emotional_tones=[EmotionalTone.JOY, EmotionalTone.ANTICIPATION],
            intensity_score=0.8,
            context="Character discovers good news."
        )

        assert moment.text_excerpt == "She felt a surge of hope."
        assert moment.start_position == 100
        assert moment.end_position == 128
        assert len(moment.emotional_tones) == 2
        assert EmotionalTone.JOY in moment.emotional_tones
        assert EmotionalTone.ANTICIPATION in moment.emotional_tones
        assert moment.intensity_score == 0.8
        assert moment.context == "Character discovers good news."

    def test_emotional_moment_multiple_tones(self):
        """Test emotional moment with multiple emotional tones."""
        moment = EmotionalMoment(
            text_excerpt="Bittersweet memories flooded back.",
            start_position=0,
            end_position=31,
            emotional_tones=[EmotionalTone.SADNESS, EmotionalTone.JOY, EmotionalTone.MELANCHOLY],
            intensity_score=0.9,
            context="Character remembers past."
        )

        assert len(moment.emotional_tones) == 3
        assert moment.intensity_score == 0.9

    def test_emotional_moment_boundaries(self):
        """Test emotional moment with boundary values."""
        # Test minimum intensity
        moment_min = EmotionalMoment(
            text_excerpt="Calm.",
            start_position=0,
            end_position=5,
            emotional_tones=[EmotionalTone.PEACE],
            intensity_score=0.0,
            context="Peaceful moment."
        )
        assert moment_min.intensity_score == 0.0

        # Test maximum intensity
        moment_max = EmotionalMoment(
            text_excerpt="Explosive rage!",
            start_position=0,
            end_position=15,
            emotional_tones=[EmotionalTone.ANGER],
            intensity_score=1.0,
            context="Peak anger."
        )
        assert moment_max.intensity_score == 1.0


class TestIllustrationPrompt:
    """Test IllustrationPrompt model."""

    def test_illustration_prompt_creation(self):
        """Test basic illustration prompt creation."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="A beautiful sunset over mountains",
            style_modifiers=["digital art", "vibrant colors"],
            negative_prompt="blurry, low quality",
            technical_params={"quality": "high", "size": "1024x1024"}
        )

        assert prompt.provider == ImageProvider.DALLE
        assert prompt.prompt == "A beautiful sunset over mountains"
        assert len(prompt.style_modifiers) == 2
        assert "digital art" in prompt.style_modifiers
        assert "vibrant colors" in prompt.style_modifiers
        assert prompt.negative_prompt == "blurry, low quality"
        assert prompt.technical_params["quality"] == "high"
        assert prompt.technical_params["size"] == "1024x1024"

    def test_illustration_prompt_no_negative(self):
        """Test illustration prompt without negative prompt."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.FLUX,
            prompt="Character portrait",
            style_modifiers=["sketch"],
            technical_params={}
        )

        assert prompt.provider == ImageProvider.FLUX
        assert prompt.negative_prompt is None
        assert len(prompt.technical_params) == 0

    def test_illustration_prompt_empty_modifiers(self):
        """Test illustration prompt with empty style modifiers."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.IMAGEN4,
            prompt="Simple scene",
            style_modifiers=[],
        )

        assert len(prompt.style_modifiers) == 0
        assert prompt.negative_prompt is None
        assert len(prompt.technical_params) == 0


class TestStyleConfig:
    """Test StyleConfig model."""

    def test_style_config_creation(self):
        """Test style configuration creation."""
        config = StyleConfig(
            image_provider=ImageProvider.DALLE,
            art_style="oil painting",
            color_palette="warm earth tones",
            artistic_influences="Van Gogh, Monet",
            style_config_path="/path/to/config.json"
        )

        assert config.image_provider == ImageProvider.DALLE
        assert config.art_style == "oil painting"
        assert config.color_palette == "warm earth tones"
        assert config.artistic_influences == "Van Gogh, Monet"
        assert config.style_config_path == "/path/to/config.json"

    def test_style_config_defaults(self):
        """Test style configuration with defaults."""
        config = StyleConfig(
            image_provider=ImageProvider.FLUX
        )

        assert config.image_provider == ImageProvider.FLUX
        assert config.art_style == "digital painting"
        assert config.color_palette is None
        assert config.artistic_influences is None
        assert config.style_config_path is None


class TestChapterAnalysis:
    """Test ChapterAnalysis model."""

    def test_chapter_analysis_creation(self):
        """Test chapter analysis creation."""
        chapter = Chapter(
            title="Test Chapter",
            content="Test content",
            number=1,
            word_count=2
        )

        moment = EmotionalMoment(
            text_excerpt="Test moment",
            start_position=0,
            end_position=12,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.5,
            context="Test context"
        )

        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Test prompt",
            style_modifiers=["test"]
        )

        analysis = ChapterAnalysis(
            chapter=chapter,
            emotional_moments=[moment],
            dominant_themes=["friendship", "adventure"],
            setting_description="Fantasy forest",
            character_emotions={"Alice": [EmotionalTone.JOY, EmotionalTone.EXCITEMENT]},
            illustration_prompts=[prompt]
        )

        assert analysis.chapter.title == "Test Chapter"
        assert len(analysis.emotional_moments) == 1
        assert len(analysis.dominant_themes) == 2
        assert "friendship" in analysis.dominant_themes
        assert analysis.setting_description == "Fantasy forest"
        assert "Alice" in analysis.character_emotions
        assert len(analysis.character_emotions["Alice"]) == 2
        assert len(analysis.illustration_prompts) == 1


class TestManuscriptMetadata:
    """Test ManuscriptMetadata model."""

    def test_manuscript_metadata_creation(self):
        """Test manuscript metadata creation."""
        metadata = ManuscriptMetadata(
            title="The Great Adventure",
            author="Jane Doe",
            genre="Fantasy",
            total_chapters=12,
            created_at=datetime.now().isoformat()
        )

        assert metadata.title == "The Great Adventure"
        assert metadata.author == "Jane Doe"
        assert metadata.genre == "Fantasy"
        assert metadata.total_chapters == 12
        assert metadata.created_at is not None

    def test_manuscript_metadata_optional_fields(self):
        """Test manuscript metadata with optional fields."""
        metadata = ManuscriptMetadata(
            title="Simple Story",
            total_chapters=1,
            created_at=datetime.now().isoformat()
        )

        assert metadata.title == "Simple Story"
        assert metadata.author is None
        assert metadata.genre is None
        assert metadata.total_chapters == 1


class TestSavedManuscript:
    """Test SavedManuscript model."""

    def test_saved_manuscript_creation(self):
        """Test saved manuscript creation."""
        metadata = ManuscriptMetadata(
            title="Saved Story",
            total_chapters=1,
            created_at=datetime.now().isoformat()
        )

        chapter = Chapter(
            title="Chapter 1",
            content="Once upon a time...",
            number=1,
            word_count=4
        )

        saved = SavedManuscript(
            metadata=metadata,
            chapters=[chapter],
            saved_at=datetime.now().isoformat(),
            file_path="/path/to/manuscript.json"
        )

        assert saved.metadata.title == "Saved Story"
        assert len(saved.chapters) == 1
        assert saved.chapters[0].title == "Chapter 1"
        assert saved.file_path == "/path/to/manuscript.json"
        assert saved.saved_at is not None


class TestModelIntegration:
    """Test integration between models."""

    def test_complete_workflow_models(self):
        """Test a complete workflow using all models."""
        # Create manuscript metadata
        metadata = ManuscriptMetadata(
            title="Integration Test",
            author="Test Author",
            genre="Test Genre",
            total_chapters=1,
            created_at=datetime.now().isoformat()
        )

        # Create chapter
        chapter = Chapter(
            title="Test Chapter",
            content="This is a test chapter with emotional content. The hero felt joy.",
            number=1,
            word_count=12
        )

        # Create emotional moment
        moment = EmotionalMoment(
            text_excerpt="The hero felt joy.",
            start_position=50,
            end_position=69,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.7,
            context="Hero experiences victory"
        )

        # Create style config
        style = StyleConfig(
            image_provider=ImageProvider.DALLE,
            art_style="fantasy art"
        )

        # Create illustration prompt
        prompt = IllustrationPrompt(
            provider=style.image_provider,
            prompt="Fantasy hero celebrating victory",
            style_modifiers=["fantasy art", "heroic"],
            technical_params={"quality": "high"}
        )

        # Create chapter analysis
        analysis = ChapterAnalysis(
            chapter=chapter,
            emotional_moments=[moment],
            dominant_themes=["heroism", "victory"],
            setting_description="Fantasy realm",
            character_emotions={"Hero": [EmotionalTone.JOY]},
            illustration_prompts=[prompt]
        )

        # Create saved manuscript
        saved = SavedManuscript(
            metadata=metadata,
            chapters=[chapter],
            saved_at=datetime.now().isoformat(),
            file_path="/test/path.json"
        )

        # Verify integration
        assert saved.metadata.title == metadata.title
        assert saved.chapters[0].title == chapter.title
        assert analysis.chapter.number == chapter.number
        assert analysis.emotional_moments[0].intensity_score == moment.intensity_score
        assert analysis.illustration_prompts[0].provider == prompt.provider
        assert len(analysis.dominant_themes) == 2
