"""Runtime context for the manuscript illustration workflow."""

from pydantic import BaseModel, Field

from illustrator.models import ImageProvider


class ManuscriptContext(BaseModel):
    """Runtime configuration and context for manuscript processing."""

    # User identification
    user_id: str = Field(description="Unique identifier for the user")

    # LLM Configuration
    model: str = Field(default="anthropic/claude-3-5-sonnet-20241022", description="Primary analysis model")

    # System prompts
    analysis_prompt: str = Field(
        default="""You are an expert literary analyst and creative director specializing in identifying the most emotionally resonant moments in fiction. Your task is to:

1. Analyze the provided chapter text for emotional peaks and valleys
2. Identify 3-5 moments with the highest emotional intensity
3. Consider narrative tension, character development, sensory details, and thematic resonance
4. For each moment, determine the dominant emotional tones and their intensity
5. Generate optimal illustration prompts that capture the essence of these moments

Current time: {time}
User preferences: {user_preferences}
Chapter context: {chapter_context}""",
        description="System prompt for chapter analysis"
    )

    illustration_prompt: str = Field(
        default="""You are a master prompt engineer for AI image generation. Your expertise spans DALL-E, Imagen4, and Flux models.

For the given emotional moment and provider, create an optimal generation prompt that:
1. Captures the emotional essence and atmosphere
2. Includes vivid visual details from the text
3. Applies appropriate artistic style modifiers for the provider
4. Considers technical limitations and strengths of each model
5. Balances creative interpretation with textual fidelity

Provider: {provider}
Style preferences: {style_preferences}
Scene context: {scene_context}""",
        description="System prompt for illustration generation"
    )

    # Processing preferences
    image_provider: ImageProvider = Field(default=ImageProvider.DALLE, description="Default image generation provider")
    max_emotional_moments: int = Field(default=10, description="Maximum emotional moments to extract per chapter")
    min_intensity_threshold: float = Field(default=0.6, description="Minimum emotional intensity threshold")

    # Style preferences
    default_art_style: str = Field(default="digital painting", description="Default artistic style")
    color_palette: str | None = Field(default=None, description="Preferred color palette")
    artistic_influences: str | None = Field(default=None, description="Artistic influences or references")

    # API Configuration
    openai_api_key: str | None = Field(default=None, description="OpenAI API key for DALL-E")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key for Claude")
    google_credentials: str | None = Field(default=None, description="Google Cloud credentials for Imagen4")
    google_project_id: str | None = Field(default=None, description="Google Cloud project ID for Imagen4")
    huggingface_api_key: str | None = Field(default=None, description="HuggingFace API key for Flux")

    # Advanced settings
    enable_content_filtering: bool = Field(default=True, description="Enable content filtering for generated images")
    save_intermediate_results: bool = Field(default=True, description="Save analysis results during processing")
    batch_processing: bool = Field(default=False, description="Enable batch processing mode")
    # Analysis mode and concurrency
    analysis_mode: str = Field(default="scene", description="basic | scene | parallel")
    prompt_concurrency: int = Field(default=2, description="Max concurrent prompt generations")
    image_concurrency: int = Field(default=2, description="Max concurrent image generations")

    model_config = {"extra": "allow"}


# Create an alias for backwards compatibility and convenience
IllustratorContext = ManuscriptContext


def get_default_context() -> IllustratorContext:
    """Get default context with environment variables."""
    import os

    return IllustratorContext(
        user_id="default_user",
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        anthropic_api_key=os.getenv('ANTHROPIC_API_KEY'),
        google_credentials=os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        google_project_id=os.getenv('GOOGLE_PROJECT_ID'),
        huggingface_api_key=os.getenv('HUGGINGFACE_API_KEY'),
    )
