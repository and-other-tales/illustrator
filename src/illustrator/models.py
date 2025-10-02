"""Data models for manuscript analysis and illustration generation."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator


class ImageProvider(str, Enum):
    """Supported image generation providers."""
    DALLE = "dalle"
    IMAGEN4 = "imagen4"
    FLUX = "flux"
    FLUX_DEV_VERTEX = "flux_dev_vertex"
    SEEDREAM = "seedream"
    HUGGINGFACE = "huggingface"


class LLMProvider(str, Enum):
    """Supported language model providers for analysis."""

    ANTHROPIC = "anthropic"
    ANTHROPIC_VERTEX = "anthropic_vertex"
    HUGGINGFACE = "huggingface"


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
    CURIOSITY = "curiosity"
    ROMANCE = "romance"
    ADVENTURE = "adventure"
    # Additional emotional tones used in tests and other files
    SUSPENSE = "suspense"
    COURAGE = "courage"
    TRIUMPH = "triumph"
    NEUTRAL = "neutral"
    SERENITY = "serenity"


class Chapter(BaseModel):
    """Represents a manuscript chapter."""
    id: str = Field(description="Chapter id")
    title: str = Field(description="Chapter title")
    summary: str = Field(description="Chapter summary")
    content: str = Field(default="", description="Full chapter text")
    number: int = Field(default=0, description="Chapter number")
    word_count: int = Field(default=0, description="Number of words in chapter")
    emotional_moments: list = Field(default_factory=list, description="Optional emotional moments in chapter")
    
    @model_validator(mode='before')
    @classmethod
    def ensure_required_fields(cls, data: Dict) -> Dict:
        """Ensure required fields are present."""
        if isinstance(data, dict):
            # Handle missing or None ID
            if 'id' not in data or data.get('id') is None:
                # Generate a deterministic ID based on title and number
                title = data.get('title', '')
                number = data.get('number', 0)
                data['id'] = f"ch-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{title}-{number}')}"
            
            # Handle missing or None summary
            if 'summary' not in data or data.get('summary') is None:
                data['summary'] = f"Summary for {data.get('title', 'Untitled Chapter')}"
                
        return data


class EmotionalMoment(BaseModel):
    """Represents a highly emotive moment in the text."""
    text_excerpt: str = Field(default="", description="The specific text passage")
    start_position: int = Field(default=0, description="Character position where excerpt starts")
    end_position: int = Field(default=0, description="Character position where excerpt ends")
    emotional_tones: List[EmotionalTone] = Field(default_factory=list, description="Detected emotional tones")
    intensity_score: float = Field(default=0.0, description="Emotional intensity from 0.0 to 1.0")
    context: str = Field(default="", description="Surrounding context for the moment")


class IllustrationPrompt(BaseModel):
    """Generated prompt for image generation."""
    provider: ImageProvider = Field(description="Target image generation provider")
    prompt: str = Field(description="Main generation prompt")
    style_modifiers: List[str] = Field(description="Style and artistic modifiers")
    # Accept either a string or a list of strings and coerce to a single string for downstream providers
    negative_prompt: str | List[str] | None = Field(default=None, description="Negative prompt for providers that support it")
    technical_params: Dict[str, Any] = Field(default_factory=dict, description="Provider-specific parameters")

    @model_validator(mode='after')
    def _coerce_negative_prompt(self):
        np = self.negative_prompt
        if np is None or (isinstance(np, str) and np.strip() == ""):
            self.negative_prompt = None
        elif isinstance(np, list):
            joined = ", ".join(str(x) for x in np if str(x).strip())
            self.negative_prompt = joined if joined else None
        else:
            self.negative_prompt = str(np)
        return self


class StyleConfig(BaseModel):
    """Configuration for illustration style and provider."""
    style_name: str | None = Field(default=None, description="Style configuration name")
    image_provider: ImageProvider = Field(default=ImageProvider.DALLE, description="Image generation provider")
    art_style: str = Field(default="digital painting", description="Base artistic style")
    color_palette: str | None = Field(default=None, description="Color palette preferences")
    composition: str | None = Field(default=None, description="Composition style")
    lighting: str | None = Field(default=None, description="Lighting style")
    mood: str | None = Field(default=None, description="Mood or atmosphere")
    detail_level: str | None = Field(default=None, description="Level of detail")
    technical_settings: Dict[str, Any] = Field(default_factory=dict, description="Technical configuration settings")
    artistic_influences: str | None = Field(default=None, description="Artistic influences or inspirations")
    style_config_path: str | None = Field(default=None, description="Path to custom style config file")
    huggingface_model_id: str | None = Field(default=None, description="HuggingFace text-to-image model identifier")
    huggingface_provider: str | None = Field(default=None, description="Optional HuggingFace provider override (e.g. fal-ai, replicate)")


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
    description: str | None = Field(default=None, description="Manuscript description")
    genre: str | None = Field(default=None, description="Literary genre")
    total_chapters: int = Field(default=1, description="Total number of chapters")
    created_at: str = Field(default_factory=lambda: "2024-01-01T00:00:00Z", description="When processing started")
    language: str | None = Field(default=None, description="Manuscript language")
    target_audience: str | None = Field(default=None, description="Target audience")
    estimated_length: int | None = Field(default=None, description="Estimated word count")


# Additional enums for backward compatibility
class OutputFormat(str, Enum):
    """Output format options."""
    PNG = "png"
    JPG = "jpg" 
    JPEG = "jpeg"
    WEBP = "webp"


class SceneDetectionResult(BaseModel):
    """Results from scene detection analysis."""
    chapter_id: str | None = Field(default=None, description="Chapter identifier")
    scenes: List[dict] = Field(default_factory=list, description="Detected scenes")
    total_scenes: int = Field(default=0, description="Total number of scenes")
    confidence: float = Field(default=0.8, description="Detection confidence score")


class CharacterDetectionResult(BaseModel):
    """Results from character detection analysis."""
    chapter_id: str | None = Field(default=None, description="Chapter identifier")
    characters: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Detected characters")
    total_characters: int = Field(default=0, description="Total number of characters")
    confidence: float = Field(default=0.8, description="Detection confidence score")


class SavedManuscript(BaseModel):
    """Represents a saved manuscript draft."""
    metadata: ManuscriptMetadata = Field(description="Manuscript metadata")
    chapters: List[Chapter] = Field(description="List of chapters")
    saved_at: str = Field(description="When the manuscript was saved")
    file_path: str = Field(description="Path to the saved file")

    @property
    def title(self) -> str:
        """Convenience access to the manuscript title."""
        return self.metadata.title
