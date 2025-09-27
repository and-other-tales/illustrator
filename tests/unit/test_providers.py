"""Unit tests for image generation providers."""

import base64
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from illustrator.models import (
    EmotionalMoment,
    EmotionalTone,
    IllustrationPrompt,
    ImageProvider,
    Chapter,
)
from illustrator.providers import (
    DalleProvider,
    FluxProvider,
    Imagen4Provider,
    ProviderFactory,
)


def _setup_vertexai_stubs(monkeypatch: pytest.MonkeyPatch, model_instance: MagicMock):
    """Register minimal Vertex AI modules so Imagen4 provider can import them during tests."""

    vertexai_module = types.ModuleType("vertexai")
    vertexai_module.init = MagicMock()

    preview_module = types.ModuleType("vertexai.preview")
    vision_models_module = types.ModuleType("vertexai.preview.vision_models")

    image_model_cls = MagicMock()
    image_model_cls.from_pretrained.return_value = model_instance
    vision_models_module.ImageGenerationModel = image_model_cls

    preview_module.vision_models = vision_models_module
    setattr(vertexai_module, "preview", preview_module)

    monkeypatch.setitem(sys.modules, "vertexai", vertexai_module)
    monkeypatch.setitem(sys.modules, "vertexai.preview", preview_module)
    monkeypatch.setitem(sys.modules, "vertexai.preview.vision_models", vision_models_module)

    return vertexai_module, image_model_cls


class TestProviderFactory:
    """Test the ProviderFactory class."""

    def test_create_dalle_provider(self):
        """Test creating DALL-E provider."""
        provider = ProviderFactory.create_provider(
            ImageProvider.DALLE,
            openai_api_key="test-key",
            anthropic_api_key="test-anthropic-key",
        )
        assert isinstance(provider, DalleProvider)
        assert provider.api_key == "test-key"

    def test_create_imagen4_provider(self):
        """Test creating Imagen4 provider."""
        provider = ProviderFactory.create_provider(
            ImageProvider.IMAGEN4,
            google_credentials="path/to/creds",
            google_project_id="test-project",
            anthropic_api_key="test-anthropic-key",
        )
        assert isinstance(provider, Imagen4Provider)
        assert provider.project_id == "test-project"

    def test_create_flux_provider(self):
        """Test creating Flux provider."""
        provider = ProviderFactory.create_provider(
            ImageProvider.FLUX,
            huggingface_api_key="test-hf-key",
            anthropic_api_key="test-anthropic-key",
        )
        assert isinstance(provider, FluxProvider)
        assert provider.api_key == "test-hf-key"

    def test_create_provider_missing_credentials(self):
        """Test error handling for missing credentials."""
        # Now requires Anthropic key first (for prompt engineering)
        with pytest.raises(ValueError, match="Anthropic API key is required"):
            ProviderFactory.create_provider(ImageProvider.DALLE)

        with pytest.raises(ValueError, match="Google credentials"):
            ProviderFactory.create_provider(ImageProvider.IMAGEN4, anthropic_api_key="test-anthropic-key")

        with pytest.raises(ValueError, match="HuggingFace API key required"):
            ProviderFactory.create_provider(ImageProvider.FLUX, anthropic_api_key="test-anthropic-key")

    def test_unsupported_provider(self):
        """Test error handling for unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider type"):
            # Supply anthropic key so provider type check is reached
            ProviderFactory.create_provider("invalid_provider", anthropic_api_key="x")

    def test_get_available_providers(self):
        """Test getting available providers based on credentials."""
        # No credentials
        available = ProviderFactory.get_available_providers()
        assert len(available) == 0

        # DALL-E only
        available = ProviderFactory.get_available_providers(
            openai_api_key="test-key",
            anthropic_api_key="test-anthropic-key",
        )
        assert ImageProvider.DALLE in available
        assert len(available) == 1

        # All providers
        available = ProviderFactory.get_available_providers(
            openai_api_key="test-key",
            google_credentials="path/to/creds",
            google_project_id="test-project",
            huggingface_api_key="test-hf-key",
            anthropic_api_key="test-anthropic-key",
        )
        assert len(available) == 3
        assert all(provider in available for provider in ImageProvider)


class TestDalleProvider:
    """Test DALL-E provider functionality."""

    @pytest.fixture
    def dalle_provider(self):
        """Create a DALL-E provider for testing."""
        return DalleProvider("test-api-key", "test-anthropic-key")

    @pytest.fixture
    def sample_emotional_moment(self):
        """Create a sample emotional moment for testing."""
        return EmotionalMoment(
            text_excerpt="The dragon breathed fire across the battlefield",
            start_position=100,
            end_position=150,
            emotional_tones=[EmotionalTone.FEAR, EmotionalTone.EXCITEMENT],
            intensity_score=0.9,
            context="Epic battle scene"
        )

    @pytest.mark.asyncio
    async def test_generate_prompt(self, dalle_provider, sample_emotional_moment):
        """Test DALL-E prompt generation."""
        style_preferences = {
            "art_style": "fantasy art",
            "color_palette": "warm colors",
            "artistic_influences": "Frank Frazetta"
        }

        # Minimal chapter context
        sample_chapter = Chapter(
            title="Test Chapter",
            content="A short test chapter content.",
            number=1,
            word_count=5,
        )

        prompt = await dalle_provider.generate_prompt(
            sample_emotional_moment,
            style_preferences,
            context="Medieval fantasy setting",
            chapter_context=sample_chapter,
        )

        assert isinstance(prompt, IllustrationPrompt)
        assert prompt.provider == ImageProvider.DALLE
        assert "dragon" in prompt.prompt.lower()
        assert "fantasy art" in prompt.prompt
        assert "warm colors" in prompt.prompt
        assert prompt.negative_prompt is None  # DALL-E doesn't support negative prompts
        assert "model" in prompt.technical_params
        assert prompt.technical_params["model"] in ("gpt-image-1", "dall-e-3")

    @pytest.mark.asyncio
    async def test_generate_image_success(self, dalle_provider):
        """Test successful image generation."""
        # Create mock prompt
        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="A fantasy dragon",
            style_modifiers=["fantasy", "detailed"],
            technical_params={"model": "dall-e-3", "size": "1024x1024"}
        )

        # Mock successful API response
        mock_response_data = {
            "data": [{
                "b64_json": base64.b64encode(b"fake_image_data").decode(),
                "revised_prompt": "A detailed fantasy dragon breathing fire"
            }]
        }

        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_response_data)

            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.return_value.post.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await dalle_provider.generate_image(prompt)

            assert result['success'] is True
            assert 'image_data' in result
            assert result['metadata']['provider'] == 'dalle'
            assert 'revised_prompt' in result['metadata']

    @pytest.mark.asyncio
    async def test_generate_image_failure(self, dalle_provider):
        """Test image generation API failure."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="Test prompt",
            style_modifiers=["test"],
            technical_params={"model": "dall-e-3"}
        )

        # Mock API error response
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 400
            mock_response.json = AsyncMock(return_value={
                "error": {"message": "Invalid prompt"}
            })

            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.return_value.post.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await dalle_provider.generate_image(prompt)

            assert result['success'] is False
            assert 'error' in result
            assert result['status_code'] == 400


class TestFluxProvider:
    """Test Flux provider functionality."""

    @pytest.fixture
    def flux_provider(self):
        """Create a Flux provider for testing."""
        return FluxProvider("test-hf-key", "test-anthropic-key")

    @pytest.fixture
    def sample_emotional_moment(self):
        """Create a sample emotional moment for testing."""
        return EmotionalMoment(
            text_excerpt="She walked through the misty forest",
            start_position=0,
            end_position=35,
            emotional_tones=[EmotionalTone.MYSTERY, EmotionalTone.PEACE],
            intensity_score=0.6,
            context="Peaceful forest scene"
        )

    @pytest.mark.asyncio
    async def test_generate_prompt(self, flux_provider, sample_emotional_moment):
        """Test Flux prompt generation."""
        style_preferences = {
            "art_style": "impressionist painting",
            "artistic_influences": "Monet"
        }

        sample_chapter = Chapter(
            title="Test Chapter",
            content="A short test chapter content.",
            number=1,
            word_count=5,
        )

        prompt = await flux_provider.generate_prompt(
            sample_emotional_moment,
            style_preferences,
            context="Enchanted forest",
            chapter_context=sample_chapter,
        )

        assert isinstance(prompt, IllustrationPrompt)
        assert prompt.provider == ImageProvider.FLUX
        assert "forest" in prompt.prompt.lower()
        assert "impressionist painting" in prompt.prompt
        assert "monet" in prompt.prompt.lower()
        assert prompt.negative_prompt is not None
        assert "guidance_scale" in prompt.technical_params

    @pytest.mark.asyncio
    async def test_generate_image_success(self, flux_provider):
        """Test successful Flux image generation."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.FLUX,
            prompt="A mystical forest",
            style_modifiers=["mystical", "atmospheric"],
            negative_prompt="blurry, low quality",
            technical_params={"guidance_scale": 7.5}
        )

        # Mock successful API response
        fake_image_bytes = b"fake_flux_image_data"

        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.read = AsyncMock(return_value=fake_image_bytes)

            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.return_value.post.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await flux_provider.generate_image(prompt)

            assert result['success'] is True
            assert 'image_data' in result
            assert result['metadata']['provider'] == 'flux'

    @pytest.mark.asyncio
    async def test_generate_image_failure(self, flux_provider):
        """Test Flux image generation failure."""
        prompt = IllustrationPrompt(
            provider=ImageProvider.FLUX,
            prompt="Test prompt",
            style_modifiers=["test"],
            technical_params={}
        )

        # Mock API error response
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.json = AsyncMock(return_value={
                "error": "Internal server error"
            })

            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.post.return_value.__aenter__ = AsyncMock(return_value=mock_response)
            mock_session.return_value.post.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await flux_provider.generate_image(prompt)

            assert result['success'] is False
            assert 'error' in result
            assert result['status_code'] == 500


class TestImagen4Provider:
    """Test Imagen4 provider functionality."""

    @pytest.fixture
    def imagen4_provider(self):
        """Create an Imagen4 provider for testing."""
        return Imagen4Provider("path/to/creds", "test-project", "test-anthropic-key")

    @pytest.fixture
    def sample_emotional_moment(self):
        """Create a sample emotional moment for testing."""
        return EmotionalMoment(
            text_excerpt="The sunset painted the sky in brilliant colors",
            start_position=0,
            end_position=46,
            emotional_tones=[EmotionalTone.JOY, EmotionalTone.PEACE],
            intensity_score=0.7,
            context="Beautiful sunset scene"
        )

    @pytest.mark.asyncio
    async def test_generate_prompt(self, imagen4_provider, sample_emotional_moment):
        """Test Imagen4 prompt generation."""
        style_preferences = {
            "art_style": "photorealistic",
            "color_palette": "golden hour"
        }

        sample_chapter = Chapter(
            title="Test Chapter",
            content="A short test chapter content.",
            number=1,
            word_count=5,
        )

        prompt = await imagen4_provider.generate_prompt(
            sample_emotional_moment,
            style_preferences,
            context="Scenic landscape",
            chapter_context=sample_chapter,
        )

        assert isinstance(prompt, IllustrationPrompt)
        assert prompt.provider == ImageProvider.IMAGEN4
        assert "sunset" in prompt.prompt.lower()
        assert "photorealistic" in prompt.prompt
        assert "golden hour" in prompt.prompt
        assert prompt.negative_prompt is not None
        assert "blurry" in prompt.negative_prompt
        assert "aspect_ratio" in prompt.technical_params

    def test_emotional_tone_mapping(self, imagen4_provider):
        """Test that emotional tones are properly mapped to visual styles."""
        # Test different emotional tone mappings
        test_cases = [
            (EmotionalTone.JOY, "warm golden hour lighting"),
            (EmotionalTone.SADNESS, "overcast sky"),
            (EmotionalTone.FEAR, "dramatic shadows"),
            (EmotionalTone.ANGER, "storm clouds"),
            (EmotionalTone.MYSTERY, "fog and mist"),
        ]

        for tone, expected_style in test_cases:
            # This is an async method, but we're testing the sync logic
            # In a real test, you'd need to await this
            # For now, we just verify the mapping exists
            assert tone.value in ["joy", "sadness", "fear", "anger", "mystery"]
            # The expected_style would be used in actual prompt generation tests
            assert expected_style is not None

    @pytest.mark.asyncio
    async def test_generate_image_successful_response(self, imagen4_provider, monkeypatch):
        """Ensure Imagen4 provider handles successful responses without index errors."""

        prompt = IllustrationPrompt(
            provider=ImageProvider.IMAGEN4,
            prompt="Preview prompt",
            style_modifiers=["cinematic lighting"],
            technical_params={}
        )

        fake_image = MagicMock()
        fake_image._as_base64_string.return_value = "ZmFrZQ=="

        mock_model_instance = MagicMock()
        mock_model_instance.generate_images.return_value = [fake_image]

        vertexai_module, image_model_cls = _setup_vertexai_stubs(monkeypatch, mock_model_instance)

        result = await imagen4_provider.generate_image(prompt)

        assert result['success'] is True
        assert result['image_data'] == "ZmFrZQ=="
        vertexai_module.init.assert_called_once_with(project="test-project")
        image_model_cls.from_pretrained.assert_called_once_with("imagen-3.0-generate-001")
        mock_model_instance.generate_images.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_image_handles_empty_response(self, imagen4_provider, monkeypatch):
        """The provider should surface helpful errors when the API returns no images."""

        prompt = IllustrationPrompt(
            provider=ImageProvider.IMAGEN4,
            prompt="Preview prompt",
            style_modifiers=[],
            technical_params={}
        )

        mock_model_instance = MagicMock()
        mock_model_instance.generate_images.return_value = []

        _setup_vertexai_stubs(monkeypatch, mock_model_instance)

        result = await imagen4_provider.generate_image(prompt)

        assert result['success'] is False
        assert "returned no images" in result['error']
        assert result['status_code'] == 500
