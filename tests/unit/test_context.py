"""Comprehensive unit tests for the context module."""

import os
import pytest
from unittest.mock import patch

from illustrator.context import ManuscriptContext, IllustratorContext, get_default_context
from illustrator.models import ImageProvider


class TestManuscriptContext:
    """Test cases for the ManuscriptContext class."""

    def test_context_initialization_defaults(self):
        """Test context initialization with default values."""
        context = ManuscriptContext(user_id="test_user")

        assert context.user_id == "test_user"
        assert context.model == "anthropic/claude-3-5-sonnet-20241022"
        assert context.image_provider == ImageProvider.DALLE
        assert context.max_emotional_moments == 10
        assert context.min_intensity_threshold == 0.6
        assert context.default_art_style == "digital painting"
        assert context.color_palette is None
        assert context.artistic_influences is None
        assert context.enable_content_filtering is True
        assert context.save_intermediate_results is True
        assert context.batch_processing is False
        assert context.analysis_mode == "scene"
        assert context.prompt_concurrency == 2
        assert context.image_concurrency == 2

    def test_context_initialization_custom_values(self):
        """Test context initialization with custom values."""
        context = ManuscriptContext(
            user_id="custom_user",
            model="custom_model",
            image_provider=ImageProvider.IMAGEN4,
            max_emotional_moments=15,
            min_intensity_threshold=0.7,
            default_art_style="watercolor",
            color_palette="warm tones",
            artistic_influences="Van Gogh",
            openai_api_key="openai_key",
            anthropic_api_key="anthropic_key",
            google_credentials="google_creds",
            google_project_id="google_project",
            huggingface_api_key="hf_key",
            enable_content_filtering=False,
            save_intermediate_results=False,
            batch_processing=True,
            analysis_mode="parallel",
            prompt_concurrency=4,
            image_concurrency=3
        )

        assert context.user_id == "custom_user"
        assert context.model == "custom_model"
        assert context.image_provider == ImageProvider.IMAGEN4
        assert context.max_emotional_moments == 15
        assert context.min_intensity_threshold == 0.7
        assert context.default_art_style == "watercolor"
        assert context.color_palette == "warm tones"
        assert context.artistic_influences == "Van Gogh"
        assert context.openai_api_key == "openai_key"
        assert context.anthropic_api_key == "anthropic_key"
        assert context.google_credentials == "google_creds"
        assert context.google_project_id == "google_project"
        assert context.huggingface_api_key == "hf_key"
        assert context.enable_content_filtering is False
        assert context.save_intermediate_results is False
        assert context.batch_processing is True
        assert context.analysis_mode == "parallel"
        assert context.prompt_concurrency == 4
        assert context.image_concurrency == 3

    def test_context_system_prompts(self):
        """Test context system prompts contain expected content."""
        context = ManuscriptContext(user_id="test_user")

        # Analysis prompt should contain key sections
        assert "literary analyst" in context.analysis_prompt.lower()
        assert "emotional peaks" in context.analysis_prompt.lower()
        assert "narrative tension" in context.analysis_prompt.lower()

        # Illustration prompt should contain key sections
        assert "prompt engineer" in context.illustration_prompt.lower()
        assert "emotional essence" in context.illustration_prompt.lower()
        assert "dall-e" in context.illustration_prompt.lower()
        assert "imagen4" in context.illustration_prompt.lower()
        assert "flux" in context.illustration_prompt.lower()

    def test_context_prompt_formatting(self):
        """Test context prompt formatting with placeholders."""
        context = ManuscriptContext(user_id="test_user")

        # Test analysis prompt formatting
        formatted_analysis = context.analysis_prompt.format(
            time="2023-01-01T00:00:00",
            user_preferences="test preferences",
            chapter_context="Chapter 1: Test"
        )

        assert "2023-01-01T00:00:00" in formatted_analysis
        assert "test preferences" in formatted_analysis
        assert "Chapter 1: Test" in formatted_analysis

        # Test illustration prompt formatting
        formatted_illustration = context.illustration_prompt.format(
            provider="DALL-E",
            style_preferences="digital art",
            scene_context="test scene"
        )

        assert "DALL-E" in formatted_illustration
        assert "digital art" in formatted_illustration
        assert "test scene" in formatted_illustration

    def test_context_api_key_handling(self):
        """Test context API key handling."""
        context = ManuscriptContext(
            user_id="test_user",
            openai_api_key="openai_test",
            anthropic_api_key="anthropic_test",
            google_credentials="/path/to/creds.json",
            google_project_id="test_project",
            huggingface_api_key="hf_test"
        )

        assert context.openai_api_key == "openai_test"
        assert context.anthropic_api_key == "anthropic_test"
        assert context.google_credentials == "/path/to/creds.json"
        assert context.google_project_id == "test_project"
        assert context.huggingface_api_key == "hf_test"

    def test_context_api_key_none_values(self):
        """Test context with None API key values."""
        context = ManuscriptContext(
            user_id="test_user",
            openai_api_key=None,
            anthropic_api_key=None,
            google_credentials=None,
            google_project_id=None,
            huggingface_api_key=None
        )

        assert context.openai_api_key is None
        assert context.anthropic_api_key is None
        assert context.google_credentials is None
        assert context.google_project_id is None
        assert context.huggingface_api_key is None

    def test_context_image_provider_enum(self):
        """Test context with different image provider values."""
        # Test DALL-E
        context_dalle = ManuscriptContext(
            user_id="test_user",
            image_provider=ImageProvider.DALLE
        )
        assert context_dalle.image_provider == ImageProvider.DALLE

        # Test Imagen4
        context_imagen = ManuscriptContext(
            user_id="test_user",
            image_provider=ImageProvider.IMAGEN4
        )
        assert context_imagen.image_provider == ImageProvider.IMAGEN4

        # Test Flux
        context_flux = ManuscriptContext(
            user_id="test_user",
            image_provider=ImageProvider.FLUX
        )
        assert context_flux.image_provider == ImageProvider.FLUX

    def test_context_analysis_modes(self):
        """Test context with different analysis modes."""
        modes = ["basic", "scene", "parallel"]

        for mode in modes:
            context = ManuscriptContext(
                user_id="test_user",
                analysis_mode=mode
            )
            assert context.analysis_mode == mode

    def test_context_concurrency_settings(self):
        """Test context concurrency settings."""
        context = ManuscriptContext(
            user_id="test_user",
            prompt_concurrency=8,
            image_concurrency=4
        )

        assert context.prompt_concurrency == 8
        assert context.image_concurrency == 4

    def test_context_boolean_flags(self):
        """Test context boolean flag settings."""
        context = ManuscriptContext(
            user_id="test_user",
            enable_content_filtering=False,
            save_intermediate_results=False,
            batch_processing=True
        )

        assert context.enable_content_filtering is False
        assert context.save_intermediate_results is False
        assert context.batch_processing is True

    def test_context_threshold_settings(self):
        """Test context threshold and limit settings."""
        context = ManuscriptContext(
            user_id="test_user",
            max_emotional_moments=20,
            min_intensity_threshold=0.8
        )

        assert context.max_emotional_moments == 20
        assert context.min_intensity_threshold == 0.8

    def test_context_extra_fields_allowed(self):
        """Test that extra fields are allowed in context."""
        # This should not raise an error due to model_config = {"extra": "allow"}
        context = ManuscriptContext(
            user_id="test_user",
            custom_field="custom_value"
        )

        assert context.user_id == "test_user"
        # Custom field should be accessible if the model allows extra fields
        if hasattr(context, 'custom_field'):
            assert context.custom_field == "custom_value"


class TestIllustratorContextAlias:
    """Test the IllustratorContext alias."""

    def test_illustrator_context_alias(self):
        """Test that IllustratorContext is an alias for ManuscriptContext."""
        assert IllustratorContext is ManuscriptContext

    def test_illustrator_context_usage(self):
        """Test using IllustratorContext alias."""
        context = IllustratorContext(user_id="test_user")
        assert isinstance(context, ManuscriptContext)
        assert context.user_id == "test_user"


class TestGetDefaultContext:
    """Test the get_default_context function."""

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'test_openai_key',
        'ANTHROPIC_API_KEY': 'test_anthropic_key',
        'GOOGLE_APPLICATION_CREDENTIALS': 'test_google_creds',
        'GOOGLE_PROJECT_ID': 'test_google_project',
        'HUGGINGFACE_API_KEY': 'test_hf_key'
    })
    def test_get_default_context_with_env_vars(self):
        """Test get_default_context with environment variables set."""
        context = get_default_context()

        assert context.user_id == "default_user"
        assert context.openai_api_key == "test_openai_key"
        assert context.anthropic_api_key == "test_anthropic_key"
        assert context.google_credentials == "test_google_creds"
        assert context.google_project_id == "test_google_project"
        assert context.huggingface_api_key == "test_hf_key"

    @patch.dict(os.environ, {}, clear=True)
    @patch('dotenv.load_dotenv')
    @patch('dotenv.find_dotenv')
    def test_get_default_context_no_env_vars(self, mock_find_dotenv, mock_load_dotenv):
        """Test get_default_context without environment variables."""
        mock_find_dotenv.return_value = ""
        mock_load_dotenv.return_value = False

        context = get_default_context()

        assert context.user_id == "default_user"
        assert context.openai_api_key is None
        assert context.anthropic_api_key is None
        assert context.google_credentials is None
        assert context.google_project_id is None
        assert context.huggingface_api_key is None

    @patch.dict(os.environ, {
        'OPENAI_API_KEY': 'partial_openai',
        'GOOGLE_PROJECT_ID': 'partial_google'
    }, clear=True)
    @patch('dotenv.load_dotenv')
    @patch('dotenv.find_dotenv')
    def test_get_default_context_partial_env_vars(self, mock_find_dotenv, mock_load_dotenv):
        """Test get_default_context with partial environment variables."""
        mock_find_dotenv.return_value = ""
        mock_load_dotenv.return_value = False

        context = get_default_context()

        assert context.user_id == "default_user"
        assert context.openai_api_key == "partial_openai"
        assert context.anthropic_api_key is None
        assert context.google_credentials is None
        assert context.google_project_id == "partial_google"
        assert context.huggingface_api_key is None

    def test_get_default_context_returns_illustrator_context(self):
        """Test that get_default_context returns IllustratorContext type."""
        context = get_default_context()
        assert isinstance(context, IllustratorContext)
        assert isinstance(context, ManuscriptContext)


class TestContextEdgeCases:
    """Test edge cases and error conditions for context."""

    def test_context_empty_strings(self):
        """Test context with empty string values."""
        context = ManuscriptContext(
            user_id="test_user",
            default_art_style="",
            color_palette="",
            artistic_influences=""
        )

        assert context.default_art_style == ""
        assert context.color_palette == ""
        assert context.artistic_influences == ""

    def test_context_zero_values(self):
        """Test context with zero values."""
        context = ManuscriptContext(
            user_id="test_user",
            max_emotional_moments=0,
            min_intensity_threshold=0.0,
            prompt_concurrency=0,
            image_concurrency=0
        )

        assert context.max_emotional_moments == 0
        assert context.min_intensity_threshold == 0.0
        assert context.prompt_concurrency == 0
        assert context.image_concurrency == 0

    def test_context_extreme_values(self):
        """Test context with extreme values."""
        context = ManuscriptContext(
            user_id="test_user",
            max_emotional_moments=1000,
            min_intensity_threshold=1.0,
            prompt_concurrency=100,
            image_concurrency=50
        )

        assert context.max_emotional_moments == 1000
        assert context.min_intensity_threshold == 1.0
        assert context.prompt_concurrency == 100
        assert context.image_concurrency == 50


class TestContextSerialization:
    """Test context serialization and deserialization."""

    def test_context_dict_serialization(self):
        """Test context dictionary serialization."""
        context = ManuscriptContext(
            user_id="test_user",
            model="test_model",
            image_provider=ImageProvider.FLUX,
            max_emotional_moments=15
        )

        context_dict = context.model_dump()

        assert context_dict["user_id"] == "test_user"
        assert context_dict["model"] == "test_model"
        assert context_dict["image_provider"] == "flux"
        assert context_dict["max_emotional_moments"] == 15

    def test_context_from_dict(self):
        """Test context creation from dictionary."""
        context_data = {
            "user_id": "test_user",
            "model": "test_model",
            "image_provider": "imagen4",
            "max_emotional_moments": 20,
            "default_art_style": "oil painting"
        }

        context = ManuscriptContext(**context_data)

        assert context.user_id == "test_user"
        assert context.model == "test_model"
        assert context.image_provider == ImageProvider.IMAGEN4
        assert context.max_emotional_moments == 20
        assert context.default_art_style == "oil painting"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])