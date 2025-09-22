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


class ImageGenerationProvider(ABC):
    """Abstract base class for image generation providers."""

    def __init__(self, anthropic_api_key: str = None):
        """Initialize provider with optional LLM for advanced prompt engineering."""
        self.prompt_engineer = None
        if anthropic_api_key:
            try:
                llm = init_chat_model(
                    model="anthropic/claude-3-5-sonnet-20241022",
                    api_key=anthropic_api_key
                )
                self.prompt_engineer = PromptEngineer(llm)
            except Exception:
                pass  # Fall back to legacy prompt generation

    async def generate_prompt(
        self,
        emotional_moment: EmotionalMoment,
        style_preferences: Dict[str, Any],
        context: str = "",
        chapter_context: Chapter = None,
        previous_scenes: List[Dict] = None
    ) -> IllustrationPrompt:
        """Generate an optimized prompt for this provider."""
        if self.prompt_engineer and chapter_context:
            # Use advanced prompt engineering
            return await self.prompt_engineer.engineer_prompt(
                emotional_moment,
                self.get_provider_type(),
                style_preferences,
                chapter_context,
                previous_scenes or []
            )
        else:
            # Fall back to legacy prompt generation
            return await self._legacy_generate_prompt(
                emotional_moment,
                style_preferences,
                context
            )

    @abstractmethod
    async def _legacy_generate_prompt(
        self,
        emotional_moment: EmotionalMoment,
        style_preferences: Dict[str, Any],
        context: str = "",
    ) -> IllustrationPrompt:
        """Legacy prompt generation method."""
        pass

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

    def __init__(self, api_key: str, anthropic_api_key: str = None):
        """Initialize DALL-E provider."""
        super().__init__(anthropic_api_key)
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/images/generations"

    def get_provider_type(self) -> ImageProvider:
        """Return DALL-E provider type."""
        return ImageProvider.DALLE

    async def _legacy_generate_prompt(
        self,
        emotional_moment: EmotionalMoment,
        style_preferences: Dict[str, Any],
        context: str = "",
    ) -> IllustrationPrompt:
        """Generate DALL-E optimized prompt."""
        # DALL-E specific style modifiers
        style_modifiers = []

        # Check if we have a full style configuration
        if 'style_config' in style_preferences and style_preferences['style_config']:
            config = style_preferences['style_config']

            # Use base prompt modifiers from config
            base_modifiers = config.get('base_prompt_modifiers', [])
            style_modifiers.extend(base_modifiers)

            # Get art style from config or preferences
            art_style = config.get('style_name', style_preferences.get('art_style', 'digital painting'))
            if art_style not in style_modifiers:
                style_modifiers.append(art_style)

        else:
            # Standard style handling
            # Base style
            art_style = style_preferences.get('art_style', 'digital painting')
            style_modifiers.append(art_style)

            # Quality and detail modifiers for DALL-E
            style_modifiers.extend([
                "highly detailed",
                "professional illustration",
                "dramatic lighting",
            ])

        # Color palette (only if not using style config)
        if not ('style_config' in style_preferences and style_preferences['style_config']):
            color_palette = style_preferences.get('color_palette')
            if color_palette:
                style_modifiers.append(f"{color_palette} color palette")

        # Emotional tone modifiers
        tone_modifiers = {
            'joy': 'bright and uplifting',
            'sadness': 'melancholic and somber',
            'fear': 'dark and foreboding',
            'anger': 'intense and fiery',
            'tension': 'suspenseful and dramatic',
            'mystery': 'mysterious and atmospheric',
            'romance': 'romantic and dreamy',
            'adventure': 'dynamic and epic',
        }

        for tone in emotional_moment.emotional_tones:
            if tone.value in tone_modifiers:
                style_modifiers.append(tone_modifiers[tone.value])

        # Build main prompt
        scene_description = emotional_moment.text_excerpt
        prompt_parts = [
            scene_description,
            f"Context: {emotional_moment.context}",
        ]

        if context:
            prompt_parts.append(f"Story context: {context}")

        main_prompt = ". ".join(prompt_parts)

        # Technical parameters for DALL-E
        if 'style_config' in style_preferences and style_preferences['style_config']:
            config = style_preferences['style_config']
            technical_params = config.get('technical_params', {
                "model": "dall-e-3",
                "size": "1024x1024",
                "quality": "hd",
                "style": "natural"
            })
        else:
            technical_params = {
                "model": "dall-e-3",
                "size": "1024x1024",
                "quality": "hd",
                "style": "vivid" if any(tone.value in ['joy', 'excitement', 'adventure'] for tone in emotional_moment.emotional_tones) else "natural"
            }

        # Handle negative prompt from config
        negative_prompt = None
        if 'style_config' in style_preferences and style_preferences['style_config']:
            config = style_preferences['style_config']
            if config.get('negative_prompt'):
                negative_prompt = ', '.join(config['negative_prompt'])

        return IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt=f"{main_prompt}. {', '.join(str(m) for m in style_modifiers)}",
            style_modifiers=style_modifiers,
            negative_prompt=negative_prompt,  # Note: DALL-E doesn't use negative prompts, but stored for compatibility
            technical_params=technical_params,
        )

    async def generate_image(
        self,
        prompt: IllustrationPrompt,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using DALL-E API."""
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
                else:
                    error_data = await response.json()
                    return {
                        'success': False,
                        'error': error_data.get('error', {}).get('message', 'Unknown error'),
                        'status_code': response.status
                    }


class Imagen4Provider(ImageGenerationProvider):
    """Google Cloud Imagen4 image generation provider."""

    def __init__(self, credentials_path: str, project_id: str, anthropic_api_key: str = None):
        """Initialize Imagen4 provider."""
        super().__init__(anthropic_api_key)
        self.credentials_path = credentials_path
        self.project_id = project_id
        # Note: In production, you'd use the Google Cloud AI Platform client library

    def get_provider_type(self) -> ImageProvider:
        """Return Imagen4 provider type."""
        return ImageProvider.IMAGEN4

    async def _legacy_generate_prompt(
        self,
        emotional_moment: EmotionalMoment,
        style_preferences: Dict[str, Any],
        context: str = "",
    ) -> IllustrationPrompt:
        """Generate Imagen4 optimized prompt."""
        style_modifiers = []

        # Imagen4 excels at photorealistic and artistic styles
        art_style = style_preferences.get('art_style', 'cinematic digital art')
        style_modifiers.append(art_style)

        # Imagen4 specific quality modifiers
        style_modifiers.extend([
            "masterpiece quality",
            "ultra high resolution",
            "cinematic composition",
            "professional photography lighting",
        ])

        # Emotional atmosphere
        atmosphere_modifiers = {
            'joy': 'warm golden hour lighting, celebratory atmosphere',
            'sadness': 'overcast sky, muted colors, melancholic mood',
            'fear': 'dramatic shadows, ominous atmosphere, chiaroscuro lighting',
            'anger': 'storm clouds, intense red lighting, turbulent atmosphere',
            'tension': 'dramatic side lighting, film noir atmosphere',
            'mystery': 'fog and mist, ethereal lighting, mysterious ambiance',
            'romance': 'soft romantic lighting, dreamy atmosphere',
            'adventure': 'epic landscape, dynamic composition, heroic lighting',
        }

        for tone in emotional_moment.emotional_tones:
            if tone.value in atmosphere_modifiers:
                style_modifiers.append(atmosphere_modifiers[tone.value])

        # Build detailed scene description
        scene_description = emotional_moment.text_excerpt
        prompt_parts = [
            scene_description,
            f"Emotional context: {emotional_moment.context}",
        ]

        if context:
            prompt_parts.append(f"Story setting: {context}")

        main_prompt = ". ".join(prompt_parts)

        # Negative prompt for Imagen4 (helps avoid unwanted elements)
        negative_elements = [
            "blurry", "low quality", "distorted", "amateur",
            "text", "watermark", "signature", "logo",
        ]

        negative_prompt = ", ".join(negative_elements)

        # Technical parameters
        technical_params = {
            "aspect_ratio": "1:1",
            "safety_filter_level": "block_most",
            "seed": None,  # Random seed for variation
        }

        return IllustrationPrompt(
            provider=ImageProvider.IMAGEN4,
            prompt=f"{main_prompt}. {', '.join(str(m) for m in style_modifiers)}",
            style_modifiers=style_modifiers,
            negative_prompt=negative_prompt,
            technical_params=technical_params,
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

            # Get the model
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

    def __init__(self, api_key: str, anthropic_api_key: str = None):
        """Initialize Flux provider."""
        super().__init__(anthropic_api_key)
        self.api_key = api_key
        self.base_url = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-pro"

    def get_provider_type(self) -> ImageProvider:
        """Return Flux provider type."""
        return ImageProvider.FLUX

    async def _legacy_generate_prompt(
        self,
        emotional_moment: EmotionalMoment,
        style_preferences: Dict[str, Any],
        context: str = "",
    ) -> IllustrationPrompt:
        """Generate Flux optimized prompt."""
        style_modifiers = []

        # Flux excels at artistic and stylized imagery
        art_style = style_preferences.get('art_style', 'detailed digital illustration')
        style_modifiers.append(art_style)

        # Flux-specific style modifiers
        style_modifiers.extend([
            "highly detailed",
            "intricate artwork",
            "trending on artstation",
            "concept art",
        ])

        # Style influence from preferences
        artistic_influences = style_preferences.get('artistic_influences')
        if artistic_influences:
            style_modifiers.append(f"in the style of {artistic_influences}")

        # Emotional style mapping for Flux
        flux_styles = {
            'joy': 'vibrant colors, dynamic energy, uplifting composition',
            'sadness': 'muted palette, soft edges, contemplative mood',
            'fear': 'dark shadows, sharp contrasts, unsettling composition',
            'anger': 'bold reds, aggressive brushstrokes, intense energy',
            'tension': 'diagonal lines, asymmetric composition, dramatic angles',
            'mystery': 'atmospheric depth, subtle details, enigmatic elements',
            'romance': 'soft pastels, flowing lines, intimate composition',
            'adventure': 'epic scale, dynamic movement, heroic proportions',
        }

        for tone in emotional_moment.emotional_tones:
            if tone.value in flux_styles:
                style_modifiers.append(flux_styles[tone.value])

        # Build comprehensive prompt
        scene_description = emotional_moment.text_excerpt
        prompt_parts = [
            scene_description,
            emotional_moment.context,
        ]

        if context:
            prompt_parts.append(context)

        main_prompt = ". ".join(prompt_parts)

        # Negative prompt for Flux
        negative_elements = [
            "low quality", "blurred", "pixelated", "distorted",
            "amateur", "simple", "basic", "ugly",
            "text", "watermark", "signature",
        ]

        negative_prompt = ", ".join(negative_elements)

        # Technical parameters for Flux
        technical_params = {
            "guidance_scale": 7.5,
            "num_inference_steps": 28,
            "width": 1024,
            "height": 1024,
        }

        return IllustrationPrompt(
            provider=ImageProvider.FLUX,
            prompt=f"{main_prompt}. {', '.join(str(m) for m in style_modifiers)}",
            style_modifiers=style_modifiers,
            negative_prompt=negative_prompt,
            technical_params=technical_params,
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