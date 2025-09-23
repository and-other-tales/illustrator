"""Image generation providers for DALL-E, Imagen4, and Flux."""

import base64
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import aiohttp
from langchain.chat_models import init_chat_model

from illustrator.models import (
    EmotionalMoment,
    IllustrationPrompt,
    ImageProvider,
    Chapter,
)
from illustrator.prompt_engineering import PromptEngineer
from illustrator.error_handling import resilient_async, ErrorRecoveryHandler, safe_execute


def _format_style_modifiers(style_modifiers: List[Any]) -> str:
    """Helper function to safely format style modifiers, handling tuples."""
    formatted = []
    for m in style_modifiers:
        if isinstance(m, tuple):
            # For tuples, join the elements with spaces
            formatted.append(" ".join(str(elem) for elem in m))
        else:
            formatted.append(str(m))
    return ", ".join(formatted)


class ImageGenerationProvider(ABC):
    """Abstract base class for image generation providers."""

    def __init__(self, anthropic_api_key: str):
        """Initialize provider with mandatory LLM for advanced prompt engineering."""
        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required for prompt engineering")

        self.error_handler = ErrorRecoveryHandler(max_attempts=3)

        try:
            llm = init_chat_model(
                model="claude-3-5-sonnet-20241022",
                model_provider="anthropic",
                api_key=anthropic_api_key,
            )
            self.prompt_engineer = PromptEngineer(llm)
        except Exception as e:
            raise ValueError(f"Failed to initialize PromptEngineer: {str(e)}")

    async def generate_prompt(
        self,
        emotional_moment: EmotionalMoment,
        style_preferences: Dict[str, Any],
        context: str = "",
        chapter_context: Chapter = None,
        previous_scenes: List[Dict] = None
    ) -> IllustrationPrompt:
        """Generate an optimized prompt for this provider using advanced prompt engineering."""
        if not chapter_context:
            raise ValueError("Chapter context is required for prompt generation")

        return await self.prompt_engineer.engineer_prompt(
            emotional_moment,
            self.get_provider_type(),
            style_preferences,
            chapter_context,
            previous_scenes or []
        )


    @abstractmethod
    def get_provider_type(self) -> ImageProvider:
        """Return the provider type."""
        pass

    @abstractmethod
    async def generate_image(
        self,
        prompt: IllustrationPrompt,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate an image using this provider."""
        pass


class DalleProvider(ImageGenerationProvider):
    """OpenAI DALL-E image generation provider."""

    def __init__(self, api_key: str, anthropic_api_key: str):
        """Initialize DALL-E provider."""
        super().__init__(anthropic_api_key)
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/images/generations"

    def get_provider_type(self) -> ImageProvider:
        """Return DALL-E provider type."""
        return ImageProvider.DALLE


    @resilient_async(
        max_attempts=3,
        fallback_functions={
            'simplified_processing': lambda: {"success": False, "error": "Used fallback"}
        }
    )
    async def generate_image(
        self,
        prompt: IllustrationPrompt,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using DALL-E API with resilient error handling."""

        async def _generate_dalle_image():
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "prompt": prompt.prompt,
                **prompt.technical_params,
                "n": 1,
                "response_format": "b64_json"
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        image_data = data['data'][0]['b64_json']

                        return {
                            'success': True,
                            'image_data': image_data,
                            'format': 'base64',
                            'metadata': {
                                'provider': 'dalle',
                                'model': prompt.technical_params.get('model', 'dall-e-3'),
                                'prompt': prompt.prompt,
                                'revised_prompt': data['data'][0].get('revised_prompt', prompt.prompt)
                            }
                        }
                    elif response.status == 429:
                        # Rate limit - let error handler deal with it
                        error_data = await response.json()
                        raise Exception(f"Rate limit exceeded: {error_data.get('error', {}).get('message', 'Unknown error')}")
                    elif response.status in [401, 403]:
                        # Authentication error
                        error_data = await response.json()
                        raise Exception(f"Authentication error: {error_data.get('error', {}).get('message', 'Invalid API key')}")
                    else:
                        error_data = await response.json()
                        error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                        raise Exception(f"DALL-E API error (status {response.status}): {error_msg}")

        return await _generate_dalle_image()


class Imagen4Provider(ImageGenerationProvider):
    """Google Cloud Vertex AI Imagen provider (uses Imagen 3 family).

    Note: Despite the class name for backwards compatibility, the underlying
    Vertex AI model currently used is from the Imagen 3 line.
    """

    def __init__(self, credentials_path: str, project_id: str, anthropic_api_key: str):
        """Initialize Imagen4 provider."""
        super().__init__(anthropic_api_key)
        self.credentials_path = credentials_path
        self.project_id = project_id
        # Note: In production, you'd use the Google Cloud AI Platform client library

    def get_provider_type(self) -> ImageProvider:
        """Return Imagen4 provider type."""
        return ImageProvider.IMAGEN4


    from illustrator.error_handling import resilient_async

    @resilient_async(
        max_attempts=3,
        fallback_functions={
            'simplified_processing': lambda: {"success": False, "error": "Used fallback"}
        }
    )
    async def generate_image(
        self,
        prompt: IllustrationPrompt,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using Imagen4 API."""
        try:
            import vertexai
            from vertexai.preview.vision_models import ImageGenerationModel

            # Initialize Vertex AI
            vertexai.init(project=self.project_id)

            # Get the model (Imagen 3 family)
            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

            # Generate image
            images = model.generate_images(
                prompt=prompt.prompt,
                negative_prompt=prompt.negative_prompt,
                number_of_images=1,
                aspect_ratio=prompt.technical_params.get("aspect_ratio", "1:1"),
                safety_filter_level=prompt.technical_params.get("safety_filter_level", "block_most"),
                seed=prompt.technical_params.get("seed")
            )

            # Convert to base64 using the proper API method
            image_b64 = images[0]._as_base64_string()

            return {
                'success': True,
                'image_data': image_b64,
                'format': 'base64',
                'metadata': {
                    'provider': 'imagen4',
                    'prompt': prompt.prompt,
                    'negative_prompt': prompt.negative_prompt,
                }
            }

        except Exception as e:
            return {
                'success': False,
                'error': f"Imagen4 generation failed: {str(e)}",
                'status_code': 500
            }


class FluxProvider(ImageGenerationProvider):
    """HuggingFace Flux 1.1 Pro image generation provider."""

    def __init__(self, api_key: str, anthropic_api_key: str):
        """Initialize Flux provider."""
        super().__init__(anthropic_api_key)
        self.api_key = api_key
        self.base_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-pro"

    def get_provider_type(self) -> ImageProvider:
        """Return Flux provider type."""
        return ImageProvider.FLUX


    from illustrator.error_handling import resilient_async

    @resilient_async(
        max_attempts=3,
        fallback_functions={
            'simplified_processing': lambda: {"success": False, "error": "Used fallback"}
        }
    )
    async def generate_image(
        self,
        prompt: IllustrationPrompt,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using Flux via HuggingFace API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": prompt.prompt,
            "negative_prompt": prompt.negative_prompt,
            "parameters": prompt.technical_params
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    image_bytes = await response.read()
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

                    return {
                        'success': True,
                        'image_data': image_b64,
                        'format': 'base64',
                        'metadata': {
                            'provider': 'flux',
                            'model': 'FLUX.1-pro',
                            'prompt': prompt.prompt,
                            'negative_prompt': prompt.negative_prompt,
                            'parameters': prompt.technical_params,
                        }
                    }
                else:
                    try:
                        error_data = await response.json()
                        error_message = error_data.get('error', 'Unknown error')
                    except Exception:
                        error_message = f"HTTP {response.status}: {await response.text()}"

                    return {
                        'success': False,
                        'error': error_message,
                        'status_code': response.status
                    }


class ProviderFactory:
    """Factory for creating image generation providers."""

    @staticmethod
    def create_provider(
        provider_type: ImageProvider,
        **credentials
    ) -> ImageGenerationProvider:
        """Create a provider instance based on type."""
        anthropic_key = credentials.get('anthropic_api_key')
        if not anthropic_key:
            raise ValueError("Anthropic API key is required for all providers (needed for prompt engineering)")

        if provider_type == ImageProvider.DALLE:
            api_key = credentials.get('openai_api_key')
            if not api_key:
                raise ValueError("OpenAI API key required for DALL-E provider")
            return DalleProvider(api_key, anthropic_key)

        elif provider_type == ImageProvider.IMAGEN4:
            credentials_path = credentials.get('google_credentials')
            project_id = credentials.get('google_project_id')
            if not credentials_path or not project_id:
                raise ValueError("Google credentials and project ID required for Imagen4")
            return Imagen4Provider(credentials_path, project_id, anthropic_key)

        elif provider_type == ImageProvider.FLUX:
            api_key = credentials.get('huggingface_api_key')
            if not api_key:
                raise ValueError("HuggingFace API key required for Flux provider")
            return FluxProvider(api_key, anthropic_key)

        else:
            raise ValueError(f"Unsupported provider type: {provider_type}")

    @staticmethod
    def get_available_providers(**credentials) -> List[ImageProvider]:
        """Get list of available providers based on provided credentials."""
        if not credentials.get('anthropic_api_key'):
            return []  # No providers available without Anthropic key for prompt engineering

        available = []

        if credentials.get('openai_api_key'):
            available.append(ImageProvider.DALLE)

        if credentials.get('google_credentials') and credentials.get('google_project_id'):
            available.append(ImageProvider.IMAGEN4)

        if credentials.get('huggingface_api_key'):
            available.append(ImageProvider.FLUX)

        return available


def get_image_provider(provider_type: str | ImageProvider, **credentials) -> ImageGenerationProvider:
    """Get an image provider instance with credentials from context or environment."""
    from illustrator.context import get_default_context

    # Convert string to ImageProvider enum if needed
    if isinstance(provider_type, str):
        provider_type = ImageProvider(provider_type)

    # Get credentials from context or fallback to provided ones
    context = get_default_context()

    # Prepare credentials dict, filtering out None values
    creds = {
        key: value for key, value in {
            'openai_api_key': context.openai_api_key,
            'huggingface_api_key': context.huggingface_api_key,
            'google_credentials': context.google_credentials,
            'google_project_id': getattr(context, 'google_project_id', None),
            'anthropic_api_key': context.anthropic_api_key,
            **credentials
        }.items() if value is not None
    }

    return ProviderFactory.create_provider(provider_type, **creds)
