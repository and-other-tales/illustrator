"""Data models for manuscript analysis and illustration generation."""

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field


class ImageProvider(str, Enum):
    """Supported image generation providers."""
    DALLE = "dalle"
    IMAGEN4 = "imagen4"
    FLUX = "flux"


class EmotionalTone(str, Enum):
    """Emotional tones detected in text."""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    ANTICIPATION = "anticipation"
    TRUST = "trust"
    MELANCHOLY = "melancholy"
    EXCITEMENT = "excitement"
    TENSION = "tension"
    PEACE = "peace"
    MYSTERY = "mystery"
    ROMANCE = "romance"
    ADVENTURE = "adventure"
    # Additional emotional tones used in tests and other files
    SUSPENSE = "suspense"
    COURAGE = "courage"
    TRIUMPH = "triumph"
    NEUTRAL = "neutral"


class Chapter(BaseModel):
    """Represents a manuscript chapter."""
    title: str = Field(description="Chapter title")
    content: str = Field(description="Full chapter text")
    number: int = Field(description="Chapter number")
    word_count: int = Field(description="Number of words in chapter")


class EmotionalMoment(BaseModel):
    """Represents a highly emotive moment in the text."""
    text_excerpt: str = Field(description="The specific text passage")
    start_position: int = Field(description="Character position where excerpt starts")
    end_position: int = Field(description="Character position where excerpt ends")
    emotional_tones: List[EmotionalTone] = Field(description="Detected emotional tones")
    intensity_score: float = Field(description="Emotional intensity from 0.0 to 1.0")
    context: str = Field(description="Surrounding context for the moment")


class IllustrationPrompt(BaseModel):
    """Generated prompt for image generation."""
    provider: ImageProvider = Field(description="Target image generation provider")
    prompt: str = Field(description="Main generation prompt")
    style_modifiers: List[str] = Field(description="Style and artistic modifiers")
    negative_prompt: str | None = Field(default=None, description="Negative prompt for providers that support it")
    technical_params: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific parameters")


class StyleConfig(BaseModel):
    """Configuration for illustration style and provider."""
    image_provider: ImageProvider = Field(description="Image generation provider")
    art_style: str = Field(default="digital painting", description="Base artistic style")
    color_palette: str | None = Field(default=None, description="Color palette preferences")
    artistic_influences: str | None = Field(default=None, description="Artistic influences or inspirations")
    style_config_path: str | None = Field(default=None, description="Path to custom style config file")


class ChapterAnalysis(BaseModel):
    """Complete analysis of a chapter."""
    chapter: Chapter = Field(description="Original chapter data")
    emotional_moments: List[EmotionalMoment] = Field(description="Top emotional moments")
    dominant_themes: List[str] = Field(description="Key themes in the chapter")
    setting_description: str = Field(description="Physical/atmospheric setting")
    character_emotions: Dict[str, List[EmotionalTone]] = Field(description="Emotions by character")
    illustration_prompts: List[IllustrationPrompt] = Field(description="Generated illustration prompts")


class ManuscriptMetadata(BaseModel):
    """Metadata about the manuscript being processed."""
    title: str = Field(description="Manuscript title")
    author: str | None = Field(default=None, description="Author name")
    genre: str | None = Field(default=None, description="Literary genre")
    total_chapters: int = Field(description="Total number of chapters")
    created_at: str = Field(description="When processing started")


class SavedManuscript(BaseModel):
    """Represents a saved manuscript draft."""
    metadata: ManuscriptMetadata = Field(description="Manuscript metadata")
    chapters: List[Chapter] = Field(description="List of chapters")
    saved_at: str = Field(description="When the manuscript was saved")
    file_path: str = Field(description="Path to the saved file")