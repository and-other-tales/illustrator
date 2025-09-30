"""Image generation providers for DALL-E, Imagen4, Flux, and Replicate hosts."""

import asyncio
import base64
import io
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Sequence

import aiohttp
from langchain.chat_models import init_chat_model
from langchain_huggingface.llms.huggingface_endpoint import HuggingFaceEndpoint
from huggingface_hub import InferenceClient

from google.auth.credentials import Credentials
from google.oauth2 import service_account

from illustrator.models import (
    Chapter,
    EmotionalMoment,
    IllustrationPrompt,
    ImageProvider,
    LLMProvider,
)
from illustrator.prompt_engineering import PromptEngineer
from illustrator.error_handling import resilient_async, ErrorRecoveryHandler
from illustrator.llm_factory import HuggingFaceConfig, create_chat_model


logger = logging.getLogger(__name__)


DEFAULT_FLUX_ENDPOINT_URL = "https://qj029p0ofvfmjxus.us-east-1.aws.endpoints.huggingface.cloud"


ALLOWED_FLUX_PARAMETERS: set[str] = {
    "guidance_scale",
    "num_inference_steps",
    "width",
    "height",
    "seed",
    "num_images_per_prompt",
    "scheduler",
    "strength",
    "prompt_strength",
    "max_sequence_length",
    "clip_skip",
    "tiling",
    "high_noise_frac",
    "lora_scale",
    "use_refiner",
    "refiner_steps",
    "refiner_guidance_scale",
    "controlnet_conditioning_scale",
    "controlnet_guidance_start",
    "controlnet_guidance_end",
}

MAX_FLUX_PROMPT_TOKENS = 60


REPLICATE_FLUX_PARAMETERS: set[str] = ALLOWED_FLUX_PARAMETERS | {
    "aspect_ratio",
    "output_format",
}

HUGGINGFACE_TTI_PARAMETERS: set[str] = {
    "height",
    "width",
    "num_inference_steps",
    "guidance_scale",
    "scheduler",
    "seed",
    "model_id",
    "provider",
    "extra_body",
}


def _normalise_flux_endpoint(endpoint_url: str | None) -> str:
    """Return a HuggingFace Flux endpoint URL without trailing slashes."""

    if not endpoint_url:
        return DEFAULT_FLUX_ENDPOINT_URL

    cleaned = endpoint_url.strip()
    if not cleaned:
        return DEFAULT_FLUX_ENDPOINT_URL

    if "?" in cleaned:
        base_part, query = cleaned.split("?", 1)
        trimmed = base_part.rstrip("/")
        return f"{trimmed or base_part}?{query}" if trimmed or query else DEFAULT_FLUX_ENDPOINT_URL

    return cleaned.rstrip("/") or DEFAULT_FLUX_ENDPOINT_URL


def _async_generation_failure(provider: str):
    """Return an async fallback function that reports a failed generation."""

    async def _fallback() -> Dict[str, Any]:
        return {
            'success': False,
            'error': f"{provider} generation fallback was triggered",
            'status_code': 503,
        }

    return _fallback


def _format_style_modifiers(style_modifiers: List[Any] | None) -> str:
    """Helper function to safely format style modifiers, handling tuples."""
    if not style_modifiers:
        return ""

    formatted = []
    try:
        for m in style_modifiers:
            if isinstance(m, tuple):
                # For tuples, join the elements with spaces
                formatted.append(" ".join(str(elem) for elem in m))
            else:
                formatted.append(str(m))
    except TypeError:
        logger.warning("Received non-iterable style_modifiers: %r", style_modifiers)
        return ""

    return ", ".join(formatted)


def _credential_error(provider: ImageProvider | str, message: str) -> ValueError:
    """Log and return a ValueError for missing/invalid credentials."""
    provider_name = provider.value if isinstance(provider, ImageProvider) else str(provider)
    logger.error("Cannot initialize %s provider: %s", provider_name, message)
    return ValueError(message)


class ImageGenerationProvider(ABC):
    """Abstract base class for image generation providers."""

    def __init__(
        self,
        *,
        prompt_engineer: PromptEngineer | None = None,
        llm_provider: LLMProvider | str | None = None,
        llm_model: str | None = None,
        anthropic_api_key: str | None = None,
        huggingface_api_key: str | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
        gcp_project_id: str | None = None,
    ) -> None:
        """Initialize provider, preparing prompt engineering resources as needed."""

        self.error_handler = ErrorRecoveryHandler(max_attempts=3)

        resolved_provider: LLMProvider | None
        if llm_provider is None:
            resolved_provider = (
                LLMProvider.ANTHROPIC if anthropic_api_key else LLMProvider.HUGGINGFACE
            )
        elif isinstance(llm_provider, LLMProvider):
            resolved_provider = llm_provider
        else:
            resolved_provider = LLMProvider(llm_provider)

        if prompt_engineer is not None:
            self.prompt_engineer = prompt_engineer
            self.llm_provider = resolved_provider
            return

        if llm_model is None:
            raise ValueError("llm_model must be provided when prompt_engineer is not supplied")

        try:
            llm = create_chat_model(
                provider=resolved_provider,
                model=llm_model,
                anthropic_api_key=anthropic_api_key,
                huggingface_api_key=huggingface_api_key,
                gcp_project_id=gcp_project_id,
                huggingface_config=huggingface_config,
            )
        except Exception as exc:
            raise ValueError(f"Failed to initialize prompt engineering LLM: {exc}") from exc

        self.prompt_engineer = PromptEngineer(llm)
        self.llm_provider = resolved_provider

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

    def __init__(
        self,
        api_key: str,
        *,
        prompt_engineer: PromptEngineer | None = None,
        llm_provider: LLMProvider | str | None = None,
        llm_model: str | None = None,
        anthropic_api_key: str | None = None,
        huggingface_api_key: str | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
        gcp_project_id: str | None = None,
    ) -> None:
        """Initialize DALL-E provider."""
        super().__init__(
            prompt_engineer=prompt_engineer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
            huggingface_config=huggingface_config,
            gcp_project_id=gcp_project_id,
        )
        self.api_key = api_key
        self.base_url = "https://api.openai.com/v1/images/generations"

    def get_provider_type(self) -> ImageProvider:
        """Return DALL-E provider type."""
        return ImageProvider.DALLE


    @resilient_async(
        max_attempts=3,
        fallback_functions={
            'simplified_processing': _async_generation_failure('dalle')
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
                        return {
                            'success': False,
                            'error': error_data.get('error', {}).get('message', 'Rate limit exceeded'),
                            'status_code': 429,
                        }
                    elif response.status in [401, 403]:
                        # Authentication error
                        error_data = await response.json()
                        return {
                            'success': False,
                            'error': error_data.get('error', {}).get('message', 'Authentication error'),
                            'status_code': response.status,
                        }
                    else:
                        error_data = await response.json()
                        error_msg = error_data.get('error', {}).get('message', 'Unknown error')
                        return {
                            'success': False,
                            'error': error_msg,
                            'status_code': response.status,
                        }

        return await _generate_dalle_image()


class Imagen4Provider(ImageGenerationProvider):
    """Google Cloud Vertex AI Imagen provider (uses Imagen 3 family).

    Note: Despite the class name for backwards compatibility, the underlying
    Vertex AI model currently used is from the Imagen 3 line.
    """

    def __init__(
        self,
        credentials_path: str,
        project_id: str,
        anthropic_api_key: str | None = None,
        *,
        prompt_engineer: PromptEngineer | None = None,
        llm_provider: LLMProvider | str | None = None,
        llm_model: str | None = None,
        huggingface_api_key: str | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
        gcp_project_id: str | None = None,
    ) -> None:
        """Initialize Imagen4 provider."""
        super().__init__(
            prompt_engineer=prompt_engineer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
            huggingface_config=huggingface_config,
            gcp_project_id=gcp_project_id,
        )
        self.credentials_path = credentials_path
        self.project_id = project_id
        self._cached_credentials: Credentials | None = None
        # Note: In production, you'd use the Google Cloud AI Platform client library

    def get_provider_type(self) -> ImageProvider:
        """Return Imagen4 provider type."""
        return ImageProvider.IMAGEN4


    from illustrator.error_handling import resilient_async

    @resilient_async(
        max_attempts=3,
        fallback_functions={
            'simplified_processing': _async_generation_failure('imagen4')
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
            credentials = self._get_google_credentials()
            vertexai.init(project=self.project_id, credentials=credentials)

            # Get the model (Imagen 3 family)
            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

            # Generate image
            images_response = model.generate_images(
                prompt=prompt.prompt,
                negative_prompt=prompt.negative_prompt,
                number_of_images=1,
                aspect_ratio=prompt.technical_params.get("aspect_ratio", "1:1"),
                safety_filter_level=prompt.technical_params.get("safety_filter_level", "block_most"),
                seed=prompt.technical_params.get("seed")
            )

            # Normalise different response shapes returned by Vertex AI preview SDK
            if hasattr(images_response, "images"):
                image_candidates = images_response.images
            else:
                image_candidates = images_response

            if image_candidates is None:
                normalized_images: List[Any] = []
            elif isinstance(image_candidates, list):
                normalized_images = image_candidates
            elif isinstance(image_candidates, tuple):
                normalized_images = list(image_candidates)
            else:
                try:
                    normalized_images = list(image_candidates)
                except TypeError as exc:  # pragma: no cover - defensive branch
                    raise ValueError("Imagen4 API returned an unexpected response format") from exc

            if not normalized_images:
                raise ValueError("Imagen4 API returned no images")

            first_image = normalized_images[0]

            # Convert to base64 using the available helpers/fields on the image object
            image_b64: str | None = None
            if hasattr(first_image, "_as_base64_string"):
                image_b64 = first_image._as_base64_string()
            elif hasattr(first_image, "base64_data"):
                image_b64 = first_image.base64_data
            elif hasattr(first_image, "as_base64"):
                image_b64 = first_image.as_base64()
            else:
                raw_data = getattr(first_image, "image_bytes", None)
                if raw_data is None:
                    raw_data = getattr(first_image, "bytes", None)
                if raw_data is None:
                    raw_data = getattr(first_image, "data", None)

                if raw_data is None:
                    raise ValueError("Imagen4 API response missing image data")

                if isinstance(raw_data, str):
                    image_b64 = raw_data
                else:
                    image_b64 = base64.b64encode(raw_data).decode("utf-8")

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

    def _get_google_credentials(self) -> Credentials:
        """Load Google Cloud service account credentials from the provided path or JSON."""
        if self._cached_credentials is not None:
            return self._cached_credentials

        path_or_json = self.credentials_path.strip()

        # Expand environment variables and tilde for filesystem paths
        expanded_path = os.path.expanduser(os.path.expandvars(path_or_json))
        if os.path.exists(expanded_path):
            credentials = service_account.Credentials.from_service_account_file(
                expanded_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self._cached_credentials = credentials
            return credentials

        # Fall back to interpreting the string as raw JSON credentials
        try:
            service_account_info = json.loads(path_or_json)
        except json.JSONDecodeError as exc:
            raise FileNotFoundError(
                "Google credentials file not found and value is not valid JSON. "
                "Set GOOGLE_APPLICATION_CREDENTIALS to the path or JSON content of a service account key."
            ) from exc

        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        self._cached_credentials = credentials
        return credentials


class FluxProvider(ImageGenerationProvider):
    """HuggingFace Flux 1.1 Pro image generation provider."""

    def __init__(
        self,
        api_key: str,
        *,
        prompt_engineer: PromptEngineer | None = None,
        llm_provider: LLMProvider | str | None = None,
        llm_model: str | None = None,
        anthropic_api_key: str | None = None,
        huggingface_api_key: str | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
        gcp_project_id: str | None = None,
        flux_endpoint_url: str | None = None,
    ) -> None:
        """Initialize Flux provider."""
        super().__init__(
            prompt_engineer=prompt_engineer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
            huggingface_config=huggingface_config,
            gcp_project_id=gcp_project_id,
        )
        self.api_key = api_key
        if flux_endpoint_url:
            self.base_url = _normalise_flux_endpoint(flux_endpoint_url)
        elif huggingface_config and huggingface_config.endpoint_url:
            self.base_url = _normalise_flux_endpoint(huggingface_config.endpoint_url)
        else:
            self.base_url = DEFAULT_FLUX_ENDPOINT_URL

        self._request_timeout = (
            huggingface_config.timeout
            if huggingface_config and huggingface_config.timeout is not None
            else None
        )

    @staticmethod
    def _sanitize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Remove parameters unsupported by the Flux inference endpoint."""
        if not parameters:
            return {}

        sanitized: Dict[str, Any] = {}
        dropped: List[str] = []

        for key, value in parameters.items():
            if value is None:
                continue

            if key in ALLOWED_FLUX_PARAMETERS:
                sanitized[key] = value
            else:
                dropped.append(key)

        if dropped:
            logger.debug(
                "Dropped unsupported Flux parameters: %s", ", ".join(sorted(set(dropped)))
            )

        return sanitized

    @staticmethod
    def _truncate_text(text: str | None) -> tuple[str | None, bool]:
        """Restrict text to Flux token limits (approximate by whitespace tokens)."""

        if not text:
            return text, False

        tokens = re.findall(r"\S+", text)
        if len(tokens) <= MAX_FLUX_PROMPT_TOKENS:
            return text, False

        truncated_text = " ".join(tokens[:MAX_FLUX_PROMPT_TOKENS])
        return truncated_text, True

    def get_provider_type(self) -> ImageProvider:
        """Return Flux provider type."""
        return ImageProvider.FLUX


    from illustrator.error_handling import resilient_async

    @resilient_async(
        max_attempts=3,
        fallback_functions={
            'simplified_processing': _async_generation_failure('flux')
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
            "Content-Type": "application/json",
            "Accept": "image/png"
        }

        sanitized_params = self._sanitize_parameters(prompt.technical_params)

        truncated_prompt, prompt_truncated = self._truncate_text(prompt.prompt)
        truncated_negative, negative_truncated = self._truncate_text(prompt.negative_prompt)

        payload = {
            "inputs": truncated_prompt,
            "negative_prompt": truncated_negative,
            "parameters": sanitized_params
        }

        timeout = None
        if self._request_timeout is not None:
            timeout = aiohttp.ClientTimeout(total=self._request_timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.base_url,
                headers=headers,
                json=payload
            ) as response:
                if response.status == 200:
                    content_type = response.headers.get("Content-Type", "").lower()

                    if "application/json" in content_type:
                        json_payload = await response.json()
                        image_field = (
                            json_payload.get("generated_image_base64")
                            or json_payload.get("image_base64")
                            or json_payload.get("image")
                        )
                        if not image_field:
                            raise ValueError("Flux endpoint returned JSON without image data")
                        image_b64 = image_field
                    else:
                        image_bytes = await response.read()
                        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

                    return {
                        'success': True,
                        'image_data': image_b64,
                        'format': 'base64',
                        'metadata': {
                            'provider': 'flux',
                            'model': prompt.technical_params.get('model', 'FLUX.1-pro'),
                            'endpoint_url': self.base_url,
                            'prompt': truncated_prompt,
                            'prompt_truncated': prompt_truncated,
                            'negative_prompt': truncated_negative,
                            'negative_prompt_truncated': negative_truncated,
                            'parameters': sanitized_params,
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


class FluxDevVertexProvider(ImageGenerationProvider):
    """Google Vertex AI Flux Dev model provider."""

    def __init__(
        self,
        gcp_project_id: str,
        *,
        prompt_engineer: PromptEngineer | None = None,
        llm_provider: LLMProvider | str | None = None,
        llm_model: str | None = None,
        anthropic_api_key: str | None = None,
        huggingface_api_key: str | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
        flux_dev_vertex_endpoint_url: str | None = None,
        gcp_credentials: str | None = None,
    ) -> None:
        """Initialize Flux Dev Vertex provider."""
        super().__init__(
            prompt_engineer=prompt_engineer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
            huggingface_config=huggingface_config,
            gcp_project_id=gcp_project_id,
        )
        self.gcp_project_id = gcp_project_id
        self.endpoint_url = flux_dev_vertex_endpoint_url
        self.gcp_credentials = gcp_credentials
        
        # Set up authentication
        if gcp_credentials:
            try:
                import json
                import os
                
                # Check if gcp_credentials is a file path or JSON string
                if os.path.isfile(gcp_credentials):
                    # It's a file path, read the JSON from file
                    with open(gcp_credentials, 'r') as f:
                        credentials_info = json.load(f)
                else:
                    # It's a JSON string, parse it directly
                    credentials_info = json.loads(gcp_credentials)
                
                # Create service account credentials with proper scopes for Vertex AI
                self.credentials = service_account.Credentials.from_service_account_info(
                    credentials_info,
                    scopes=['https://www.googleapis.com/auth/cloud-platform']
                )
                logger.info("Successfully loaded GCP service account credentials")
            except (json.JSONDecodeError, KeyError, FileNotFoundError, OSError) as e:
                logger.warning(f"Invalid GCP credentials format: {e}")
                logger.info("Falling back to default Google Cloud credentials")
                self.credentials = None
        else:
            # Use default credentials
            logger.info("No GCP credentials provided, using default Google Cloud credentials")
            self.credentials = None

    def get_provider_type(self) -> ImageProvider:
        """Return Flux Dev Vertex provider type."""
        return ImageProvider.FLUX_DEV_VERTEX

    @resilient_async(
        max_attempts=3,
        fallback_functions={
            'simplified_processing': _async_generation_failure('flux_dev_vertex')
        }
    )
    async def generate_image(
        self,
        prompt: IllustrationPrompt,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate image using Flux Dev via Google Vertex AI."""
        if not self.endpoint_url:
            return {
                'success': False,
                'error': 'No Vertex AI endpoint URL configured for Flux Dev'
            }
        
        # Handle different endpoint URL formats
        endpoint_url = self.endpoint_url
        
        # For dedicated endpoints (*.prediction.vertexai.goog), use the correct format
        if endpoint_url.endswith('.prediction.vertexai.goog'):
            # Extract components from dedicated endpoint URL
            # Format: https://ENDPOINT_ID.LOCATION_ID-PROJECT_NUMBER.prediction.vertexai.goog
            if endpoint_url.startswith('http'):
                base_url = endpoint_url.replace('https://', '').replace('http://', '')
            else:
                base_url = endpoint_url
                
            # Parse: ENDPOINT_ID.LOCATION_ID-PROJECT_NUMBER.prediction.vertexai.goog
            parts = base_url.split('.prediction.vertexai.goog')[0].split('.')
            if len(parts) >= 2:
                endpoint_id = parts[0]
                location_project = parts[1]
                
                # Split LOCATION_ID-PROJECT_NUMBER
                if '-' in location_project:
                    location_parts = location_project.split('-')
                    location = '-'.join(location_parts[:-1])  # Everything except last part
                    project_number = location_parts[-1]      # Last part
                    
                    # Construct proper dedicated endpoint URL according to Google Cloud docs
                    endpoint_url = f"https://{endpoint_id}.{location}-{project_number}.prediction.vertexai.goog/v1/projects/{project_number}/locations/{location}/endpoints/{endpoint_id}:predict"
                else:
                    return {
                        'success': False,
                        'error': f'Invalid dedicated endpoint URL format: {self.endpoint_url}. Cannot parse location and project number.'
                    }
            else:
                return {
                    'success': False,
                    'error': f'Invalid dedicated endpoint URL format: {self.endpoint_url}. Cannot parse endpoint ID.'
                }
        elif not endpoint_url.startswith(('https://', 'http://')):
            return {
                'success': False,
                'error': f'Invalid endpoint URL format: {endpoint_url}. Must be a full URL or dedicated endpoint hostname'
            }
        
        logger.info(f"Using Flux Dev Vertex endpoint: {endpoint_url}")

        try:
            from google.auth import default
            from google.auth.transport.requests import Request
            import aiohttp

            # Get credentials with proper scopes for Vertex AI
            if self.credentials:
                credentials = self.credentials
            else:
                # Request credentials with Vertex AI scopes
                credentials, _ = default(scopes=[
                    'https://www.googleapis.com/auth/cloud-platform'
                ])

            # Refresh credentials if needed (using sync request for credential refresh)
            if not credentials.valid:
                sync_request = Request()
                credentials.refresh(sync_request)
            
            headers = {
                "Authorization": f"Bearer {credentials.token}",
                "Content-Type": "application/json",
            }

            # Prepare payload for Vertex AI Flux Dev model
            # Based on error message, this is using HuggingFace FluxPipeline directly
            # So we need to match the FluxPipeline.__call__() signature
            
            # For HuggingFace FluxPipeline, the main parameter is the prompt
            instances = [prompt.prompt]
            
            # Prepare parameters that FluxPipeline accepts
            parameters = {}
            
            # FluxPipeline common parameters (based on diffusers FluxPipeline)
            if prompt.technical_params:
                # Map parameters that FluxPipeline actually accepts
                if 'width' in prompt.technical_params:
                    parameters['width'] = prompt.technical_params['width']
                if 'height' in prompt.technical_params:
                    parameters['height'] = prompt.technical_params['height']
                if 'num_inference_steps' in prompt.technical_params:
                    parameters['num_inference_steps'] = prompt.technical_params['num_inference_steps']
                if 'guidance_scale' in prompt.technical_params:
                    parameters['guidance_scale'] = prompt.technical_params['guidance_scale']
                
                # Additional FluxPipeline parameters
                flux_params = ['generator', 'latents', 'output_type', 'return_dict', 'joint_attention_kwargs', 'max_sequence_length']
                for param in flux_params:
                    if param in prompt.technical_params:
                        parameters[param] = prompt.technical_params[param]
            
            # Set reasonable defaults for FluxPipeline
            if 'width' not in parameters:
                parameters['width'] = 512
            if 'height' not in parameters:
                parameters['height'] = 512
            if 'num_inference_steps' not in parameters:
                parameters['num_inference_steps'] = 8
            if 'guidance_scale' not in parameters:
                parameters['guidance_scale'] = 3.5
            
            # Add negative prompt if provided
            if prompt.negative_prompt:
                parameters['negative_prompt'] = prompt.negative_prompt
            
            payload = {
                "instances": instances,
                "parameters": parameters
            }
            
            # Debug logging to see exactly what we're sending
            logger.info(f"Flux Dev Vertex payload: {payload}")
            logger.debug(f"Payload instances: {instances}")
            logger.debug(f"Payload parameters: {parameters}")

            # Try multiple endpoint paths for dedicated endpoints
            endpoint_paths_to_try = [endpoint_url]
            
            # For dedicated endpoints, the primary format should be the one we constructed
            # Add a few alternative formats in case the construction is slightly off
            if '.prediction.vertexai.goog' in endpoint_url and '/v1/projects/' in endpoint_url:
                # Extract base URL and components for alternatives
                base_url = endpoint_url.split('/v1/projects/')[0]
                
                # Extract endpoint ID from the original URL
                original_base = self.endpoint_url.replace('https://', '').replace('http://', '')
                endpoint_id = original_base.split('.')[0]
                
                alternative_paths = [
                    f"{base_url}/predict",  # Simple predict path
                    f"{base_url}/v1/predict",  # v1 predict path
                ]
                
                # Add alternatives that aren't already in the list
                for path in alternative_paths:
                    if path not in endpoint_paths_to_try:
                        endpoint_paths_to_try.append(path)

            async with aiohttp.ClientSession() as session:
                last_error = None
                
                for attempt_url in endpoint_paths_to_try:
                    logger.info(f"Trying Vertex AI endpoint: {attempt_url}")
                    logger.debug(f"Request payload: {payload}")
                    
                    async with session.post(
                        attempt_url,
                        headers=headers,
                        json=payload
                    ) as response:
                        response_text = await response.text()
                        logger.debug(f"Response status: {response.status}")
                        logger.debug(f"Response text: {response_text}")
                        
                        if response.status == 200:
                            try:
                                result = await response.json()
                                logger.debug(f"Response JSON: {result}")
                                
                                # According to HuggingFace documentation, response format is:
                                # output.predictions[0] contains the base64 encoded image directly
                                if 'predictions' in result and result['predictions']:
                                    # The prediction should be a base64 encoded image string
                                    image_data = result['predictions'][0]
                                    logger.info(f"Successfully generated image using endpoint: {attempt_url}")
                                    return {
                                        'success': True,
                                        'image_data': image_data,
                                        'format': 'base64',
                                        'provider': 'flux_dev_vertex'
                                    }
                                
                                last_error = f'No image data in Vertex AI response. Response: {result}'
                            except Exception as json_error:
                                last_error = f'Failed to parse JSON response: {json_error}. Response: {response_text}'
                        else:
                            # Store error for this attempt
                            last_error = f"Vertex AI error {response.status}: {response_text}"
                            logger.warning(f"Endpoint {attempt_url} failed with status {response.status}")
                            logger.warning(f"Response: {response_text}")
                            
                            # For 400 errors, provide more specific guidance about FluxPipeline
                            if response.status == 400:
                                if "unexpected keyword argument" in response_text:
                                    last_error = f"FluxPipeline parameter error (400): {response_text}. This suggests incompatible parameters are being passed to the HuggingFace FluxPipeline."
                                else:
                                    last_error = f"Bad request (400): {response_text}. Please check the request format and parameters."
                                # Don't try other endpoints for 400 errors - it's a payload issue
                                break
                            elif response.status == 404:
                                if "UNIMPLEMENTED" in response_text:
                                    last_error = f"Endpoint not found (404 UNIMPLEMENTED): {attempt_url}. This suggests the Vertex AI endpoint may not be deployed, active, or the URL format is incorrect. Please verify the endpoint is running in Google Cloud Console."
                                else:
                                    last_error = f"Endpoint not found (404): {attempt_url}. Please verify the endpoint URL and deployment status."
                                continue
                            else:
                                # For non-400/404 errors, don't try other endpoints
                                break
                
                # If we get here, all endpoints failed
                return {
                    'success': False,
                    'error': f"All endpoint attempts failed. Last error: {last_error}",
                    'attempted_urls': endpoint_paths_to_try
                }

        except Exception as e:
            logger.error(f"Flux Dev Vertex generation error: {e}")
            return {
                'success': False,
                'error': f"Generation failed: {str(e)}"
            }


class HuggingFaceImageProvider(ImageGenerationProvider):
    """Text-to-image generation using HuggingFace Inference Endpoints."""

    def __init__(
        self,
        api_token: str,
        *,
        model_id: str,
        endpoint_url: str | None = None,
        provider_override: str | None = None,
        prompt_engineer: PromptEngineer | None = None,
        llm_provider: LLMProvider | str | None = None,
        llm_model: str | None = None,
        anthropic_api_key: str | None = None,
        huggingface_api_key: str | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
        gcp_project_id: str | None = None,
    ) -> None:
        if not api_token:
            raise ValueError("HuggingFace API token is required for the HuggingFace image provider")
        if not model_id:
            raise ValueError("A HuggingFace text-to-image model identifier (e.g. 'username/model') is required")

        super().__init__(
            prompt_engineer=prompt_engineer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
            huggingface_config=huggingface_config,
            gcp_project_id=gcp_project_id,
        )

        self._api_token = api_token
        self._default_model_id = model_id
        self._endpoint_url = endpoint_url
        self._provider = provider_override
        timeout = None
        if huggingface_config and huggingface_config.timeout is not None:
            try:
                timeout = int(huggingface_config.timeout)
            except (TypeError, ValueError):
                timeout = None

        endpoint_kwargs: Dict[str, Any] = {
            'model': model_id,
            'huggingfacehub_api_token': api_token,
            'task': 'text-to-image',
        }
        if endpoint_url:
            endpoint_kwargs['endpoint_url'] = endpoint_url
        if provider_override:
            endpoint_kwargs['provider'] = provider_override
        if timeout is not None:
            endpoint_kwargs['timeout'] = timeout

        self._endpoint = HuggingFaceEndpoint(**endpoint_kwargs)
        self._client: InferenceClient = self._endpoint.client
        self._timeout = timeout

    def get_provider_type(self) -> ImageProvider:
        return ImageProvider.HUGGINGFACE

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        if not parameters:
            return {}, {}

        sanitized: Dict[str, Any] = {}
        extra_body: Dict[str, Any] = {}

        for raw_key, value in parameters.items():
            if raw_key not in HUGGINGFACE_TTI_PARAMETERS or value in (None, ''):
                continue

            key = raw_key
            if key == 'extra_body':
                if isinstance(value, Mapping):
                    extra_body.update(value)
                continue

            if key in {'height', 'width', 'num_inference_steps', 'seed'}:
                try:
                    sanitized[key] = int(value)
                except (TypeError, ValueError):
                    logger.debug("Ignoring HuggingFace %s parameter with non-integer value %r", key, value)
            elif key == 'guidance_scale':
                try:
                    sanitized[key] = float(value)
                except (TypeError, ValueError):
                    logger.debug("Ignoring HuggingFace guidance_scale parameter with non-float value %r", value)
            elif key in {'model_id', 'provider', 'scheduler'}:
                sanitized[key] = str(value)
            else:
                sanitized[key] = value

        return sanitized, extra_body

    def _build_client(
        self,
        model_id: str,
        provider_override: str | None,
    ) -> InferenceClient:
        if not provider_override or provider_override == self._provider:
            return self._client

        return InferenceClient(
            model=model_id or self._default_model_id,
            token=self._api_token,
            provider=provider_override,
            timeout=self._timeout,
        )

    @resilient_async(
        max_attempts=3,
        fallback_functions={
            'generation': _async_generation_failure('huggingface'),
        },
    )
    async def generate_image(
        self,
        prompt: IllustrationPrompt,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        sanitized_params, extra_body = self._sanitize_parameters(prompt.technical_params)

        model_id = sanitized_params.pop('model_id', None) or self._default_model_id
        provider_override = sanitized_params.pop('provider', None) or self._provider

        if not model_id:
            return {
                'success': False,
                'error': 'No HuggingFace model id supplied for text-to-image generation',
                'status_code': 400,
            }

        client = self._build_client(model_id, provider_override)

        call_kwargs = sanitized_params.copy()
        if extra_body:
            call_kwargs['extra_body'] = extra_body

        try:
            image = await asyncio.to_thread(
                client.text_to_image,
                prompt=prompt.prompt,
                negative_prompt=prompt.negative_prompt,
                model=model_id,
                **call_kwargs,
            )
        except Exception as exc:
            logger.error("HuggingFace image generation failed", exc_info=exc)
            
            # Check if this is a paused endpoint error
            error_message = str(exc)
            if "paused" in error_message.lower() or "503" in error_message:
                logger.warning("HuggingFace endpoint appears to be paused, image generation failed")
                return {
                    'success': False,
                    'error': f"HuggingFace endpoint is paused or unavailable: {exc}",
                    'status_code': 503,
                    'suggestion': 'The HuggingFace endpoint needs to be restarted. Please check your HuggingFace Inference Endpoint dashboard.'
                }
            
            return {
                'success': False,
                'error': f"HuggingFace image generation failed: {exc}",
                'status_code': 502,
            }

        buffer = io.BytesIO()
        try:
            image.save(buffer, format='PNG')
        except Exception as exc:
            logger.error("Failed to serialise HuggingFace image output", exc_info=exc)
            return {
                'success': False,
                'error': f"Could not encode HuggingFace image output: {exc}",
                'status_code': 502,
            }

        image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        metadata = {
            'provider': ImageProvider.HUGGINGFACE.value,
            'model_id': model_id,
            'endpoint_url': self._endpoint_url,
            'provider_override': provider_override,
            'prompt': prompt.prompt,
            'negative_prompt': prompt.negative_prompt,
            'parameters': sanitized_params,
        }
        if extra_body:
            metadata['extra_body'] = extra_body

        return {
            'success': True,
            'image_data': image_b64,
            'format': 'base64',
            'metadata': metadata,
        }

class ReplicateImageProvider(ImageGenerationProvider):
    """Base provider for models hosted on Replicate."""

    def __init__(
        self,
        api_token: str,
        model_identifier: str,
        *,
        prompt_engineer: PromptEngineer | None = None,
        llm_provider: LLMProvider | str | None = None,
        llm_model: str | None = None,
        anthropic_api_key: str | None = None,
        huggingface_api_key: str | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
        gcp_project_id: str | None = None,
        model_version: str | None = None,
    ) -> None:
        """Initialise the Replicate provider with authentication and prompt tools."""

        super().__init__(
            prompt_engineer=prompt_engineer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
            huggingface_config=huggingface_config,
            gcp_project_id=gcp_project_id,
        )

        try:
            from replicate import Client
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ValueError(
                "The 'replicate' package is required to use Replicate-hosted image providers."
            ) from exc

        self._replicate_client = Client(api_token=api_token)
        self._replicate_model = model_identifier
        self._replicate_model_version = model_version

    def _resolve_model_reference(self) -> str:
        if self._replicate_model_version:
            return f"{self._replicate_model}:{self._replicate_model_version}"
        return self._replicate_model

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Filter technical parameters to those accepted by Replicate models."""
        if not parameters:
            return {}

        sanitized: Dict[str, Any] = {}
        seed = parameters.get('seed')
        if seed is not None:
            sanitized['seed'] = seed

        return sanitized

    def _prepare_input(
        self,
        prompt: IllustrationPrompt,
        **kwargs
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Prepare the payload sent to Replicate along with metadata hints."""
        payload: Dict[str, Any] = {
            "prompt": prompt.prompt,
        }

        if prompt.negative_prompt:
            payload["negative_prompt"] = prompt.negative_prompt

        payload.update(self._sanitize_parameters(prompt.technical_params))

        overrides = kwargs.get("replicate_input_overrides")
        if isinstance(overrides, dict):
            payload.update({key: value for key, value in overrides.items() if value is not None})

        return payload, {}

    def _extract_image_urls(self, output: Any) -> List[str]:
        """Normalise Replicate outputs into a list of downloadable URLs."""
        urls: List[str] = []

        def _append(candidate: str | None) -> None:
            if isinstance(candidate, str) and candidate:
                if candidate not in urls:
                    urls.append(candidate)

        if output is None:
            return urls

        potential_url = getattr(output, "url", None)
        _append(potential_url)

        urls_mapping = getattr(output, "urls", None)
        if isinstance(urls_mapping, Mapping):
            for key in ("get", "content", "self", "download", "url"):
                _append(urls_mapping.get(key))

        if isinstance(output, str):
            if output.startswith(("http://", "https://", "data:")):
                _append(output)
            return urls

        if hasattr(output, "model_dump") and callable(output.model_dump):
            try:
                dumped = output.model_dump()
            except (TypeError, ValueError):
                dumped = None
            if isinstance(dumped, Mapping):
                for value in dumped.values():
                    for candidate in self._extract_image_urls(value):
                        _append(candidate)
                return urls

        if hasattr(output, "dict") and callable(output.dict):
            try:
                dumped = output.dict()
            except (TypeError, ValueError):
                dumped = None
            if isinstance(dumped, Mapping):
                for value in dumped.values():
                    for candidate in self._extract_image_urls(value):
                        _append(candidate)
                return urls

        if isinstance(output, Mapping):
            for value in output.values():
                for candidate in self._extract_image_urls(value):
                    _append(candidate)
            return urls

        if isinstance(output, Sequence) and not isinstance(output, (str, bytes, bytearray)):
            for item in output:
                for candidate in self._extract_image_urls(item):
                    _append(candidate)
            return urls

        if hasattr(output, "__iter__") and not isinstance(output, (str, bytes, bytearray)):
            try:
                sequence = list(output)
            except TypeError:
                sequence = [output]

            for item in sequence:
                for candidate in self._extract_image_urls(item):
                    _append(candidate)
            return urls

        return urls

    async def _download_as_base64(self, url: str) -> str:
        """Download an image URL and return its base64-encoded content."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise ValueError(f"Failed to download Replicate image: HTTP {response.status}")
                data = await response.read()

        return base64.b64encode(data).decode("utf-8")

    async def generate_image(
        self,
        prompt: IllustrationPrompt,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate an image by invoking a Replicate-hosted model."""

        payload, metadata_overrides = self._prepare_input(prompt, **kwargs)
        model_reference = self._resolve_model_reference()

        async def _run_model() -> Any:
            loop = asyncio.get_running_loop()

            def _call_model() -> Any:
                return self._replicate_client.run(model_reference, input=payload)

            return await loop.run_in_executor(None, _call_model)

        try:
            output = await _run_model()
        except Exception as exc:  # pragma: no cover - depends on external service
            logger.error("Replicate generation failed", exc_info=exc)
            return {
                'success': False,
                'error': f"Replicate generation failed: {exc}",
                'status_code': 502,
            }

        image_urls = self._extract_image_urls(output)
        if not image_urls:
            return {
                'success': False,
                'error': 'Replicate returned no image URLs',
                'status_code': 502,
            }

        image_url = image_urls[0]

        try:
            image_b64 = await self._download_as_base64(image_url)
        except Exception as exc:  # pragma: no cover - network dependent
            logger.error("Failed to download Replicate image", exc_info=exc)
            return {
                'success': False,
                'error': f"Could not download image from Replicate: {exc}",
                'status_code': 502,
            }

        metadata = {
            'provider': self.get_provider_type().value,
            'replicate_model': model_reference,
            'prompt': payload.get('prompt', prompt.prompt),
            'image_url': image_url,
        }
        if prompt.style_modifiers:
            metadata['style_modifiers'] = list(prompt.style_modifiers)
        if prompt.technical_params:
            metadata['technical_params'] = dict(prompt.technical_params)
        if prompt.negative_prompt:
            metadata['negative_prompt'] = prompt.negative_prompt
        metadata.update(metadata_overrides)

        return {
            'success': True,
            'image_data': image_b64,
            'format': 'base64',
            'metadata': metadata,
        }


class ReplicateFluxProvider(ReplicateImageProvider):
    """Flux text-to-image generation via Replicate."""

    def __init__(
        self,
        api_token: str,
        *,
        prompt_engineer: PromptEngineer | None = None,
        llm_provider: LLMProvider | str | None = None,
        llm_model: str | None = None,
        anthropic_api_key: str | None = None,
        huggingface_api_key: str | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
        gcp_project_id: str | None = None,
    ) -> None:
        super().__init__(
            api_token,
            "black-forest-labs/flux-1.1-pro",
            prompt_engineer=prompt_engineer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
            huggingface_config=huggingface_config,
            gcp_project_id=gcp_project_id,
        )

    def get_provider_type(self) -> ImageProvider:
        return ImageProvider.FLUX

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        base_params = super()._sanitize_parameters(parameters)
        if not parameters:
            return base_params

        for key in REPLICATE_FLUX_PARAMETERS:
            value = parameters.get(key)
            if value is not None:
                base_params[key] = value

        return base_params

    def _prepare_input(
        self,
        prompt: IllustrationPrompt,
        **kwargs
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        truncated_prompt, prompt_truncated = FluxProvider._truncate_text(prompt.prompt)
        truncated_negative, negative_truncated = FluxProvider._truncate_text(prompt.negative_prompt)

        payload = {
            "prompt": truncated_prompt,
        }

        if truncated_negative:
            payload["negative_prompt"] = truncated_negative
        elif prompt.negative_prompt:
            payload["negative_prompt"] = prompt.negative_prompt

        payload.update(self._sanitize_parameters(prompt.technical_params))

        overrides = kwargs.get("replicate_input_overrides")
        if isinstance(overrides, dict):
            payload.update({key: value for key, value in overrides.items() if value is not None})

        metadata: Dict[str, Any] = {}
        if prompt_truncated:
            metadata["prompt_truncated"] = True
        if negative_truncated:
            metadata["negative_prompt_truncated"] = True

        return payload, metadata


class ReplicateImagenProvider(ReplicateImageProvider):
    """Google Imagen 4 served via Replicate."""

    _ALLOWED_PARAMETERS: set[str] = {"aspect_ratio", "safety_filter_level", "seed"}

    def __init__(
        self,
        api_token: str,
        *,
        prompt_engineer: PromptEngineer | None = None,
        llm_provider: LLMProvider | str | None = None,
        llm_model: str | None = None,
        anthropic_api_key: str | None = None,
        huggingface_api_key: str | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
    ) -> None:
        super().__init__(
            api_token,
            "google/imagen-4",
            prompt_engineer=prompt_engineer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
            huggingface_config=huggingface_config,
        )

    def get_provider_type(self) -> ImageProvider:
        return ImageProvider.IMAGEN4

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        base_params = super()._sanitize_parameters(parameters)
        if not parameters:
            return base_params

        for key in self._ALLOWED_PARAMETERS:
            value = parameters.get(key)
            if value is not None:
                base_params[key] = value

        return base_params


class SeedreamProvider(ReplicateImageProvider):
    """ByteDance Seedream 4 provider via Replicate."""

    _ALLOWED_PARAMETERS: set[str] = {"width", "height", "cfg_scale", "steps", "seed"}

    def __init__(
        self,
        api_token: str,
        *,
        prompt_engineer: PromptEngineer | None = None,
        llm_provider: LLMProvider | str | None = None,
        llm_model: str | None = None,
        anthropic_api_key: str | None = None,
        huggingface_api_key: str | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
    ) -> None:
        super().__init__(
            api_token,
            "bytedance/seedream-4",
            prompt_engineer=prompt_engineer,
            llm_provider=llm_provider,
            llm_model=llm_model,
            anthropic_api_key=anthropic_api_key,
            huggingface_api_key=huggingface_api_key,
            huggingface_config=huggingface_config,
        )

    def get_provider_type(self) -> ImageProvider:
        return ImageProvider.SEEDREAM

    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        base_params = super()._sanitize_parameters(parameters)
        if not parameters:
            return base_params

        for key in self._ALLOWED_PARAMETERS:
            value = parameters.get(key)
            if value is not None:
                base_params[key] = value

        return base_params


class ProviderFactory:
    """Factory for creating image generation providers."""

    @staticmethod
    def create_provider(
        provider_type: ImageProvider,
        **credentials
    ) -> ImageGenerationProvider:
        """Create a provider instance based on type."""
        anthropic_key = credentials.get('anthropic_api_key')
        llm_provider_value = credentials.get('llm_provider')
        llm_provider: LLMProvider
        if llm_provider_value is None:
            llm_provider = LLMProvider.ANTHROPIC if anthropic_key else LLMProvider.HUGGINGFACE
        elif isinstance(llm_provider_value, LLMProvider):
            llm_provider = llm_provider_value
        else:
            llm_provider = LLMProvider(llm_provider_value)

        if llm_provider == LLMProvider.ANTHROPIC and not anthropic_key:
            raise _credential_error(
                provider_type,
                "Anthropic API key is required when using the Anthropic provider for prompt engineering",
            )
        if llm_provider == LLMProvider.HUGGINGFACE and not credentials.get('huggingface_api_key'):
            raise _credential_error(
                provider_type,
                "HuggingFace API key is required when using the HuggingFace provider for prompt engineering",
            )

        replicate_token = credentials.get('replicate_api_token')

        llm_model = (
            credentials.get('model')
            or credentials.get('llm_model')
            or (
                "claude-3-5-sonnet-20241022"
                if llm_provider == LLMProvider.ANTHROPIC
                else "gpt-oss-120b"
            )
        )

        huggingface_config = HuggingFaceConfig(
            endpoint_url=credentials.get('huggingface_endpoint_url'),
            max_new_tokens=credentials.get('huggingface_max_new_tokens', 512),
            temperature=credentials.get('huggingface_temperature', 0.7),
            timeout=credentials.get('huggingface_timeout'),
            model_kwargs=credentials.get('huggingface_model_kwargs'),
        )

        common_llm_kwargs = {
            'llm_provider': llm_provider,
            'llm_model': llm_model,
            'anthropic_api_key': anthropic_key,
            'huggingface_api_key': credentials.get('huggingface_api_key'),
            'gcp_project_id': credentials.get('gcp_project_id'),
            'huggingface_config': huggingface_config,
        }

        if provider_type == ImageProvider.DALLE:
            api_key = credentials.get('openai_api_key')
            if not api_key:
                raise _credential_error(provider_type, "OpenAI API key required for DALL-E provider")
            return DalleProvider(api_key, **common_llm_kwargs)

        elif provider_type == ImageProvider.IMAGEN4:
            if replicate_token:
                try:
                    return ReplicateImagenProvider(
                        replicate_token,
                        **common_llm_kwargs,
                    )
                except ValueError as exc:
                    fallback_available = (
                        credentials.get('google_credentials')
                        and credentials.get('google_project_id')
                    )
                    if fallback_available and "replicate" in str(exc).lower():
                        logger.warning(
                            "Replicate Imagen provider unavailable (%s); falling back to Google Vertex AI",
                            exc,
                        )
                    else:
                        raise

            credentials_path = credentials.get('google_credentials')
            project_id = credentials.get('google_project_id')
            if not credentials_path or not project_id:
                raise _credential_error(
                    provider_type,
                    "Google credentials and project ID required for Imagen4",
                )
            return Imagen4Provider(
                credentials_path,
                project_id,
                **common_llm_kwargs,
            )

        elif provider_type == ImageProvider.FLUX:
            if replicate_token:
                try:
                    return ReplicateFluxProvider(
                        replicate_token,
                        **common_llm_kwargs,
                    )
                except ValueError as exc:
                    fallback_available = credentials.get('huggingface_api_key')
                    if fallback_available and "replicate" in str(exc).lower():
                        logger.warning(
                            "Replicate Flux provider unavailable (%s); falling back to HuggingFace endpoint",
                            exc,
                        )
                    else:
                        raise

            api_key = credentials.get('huggingface_api_key')
            if not api_key:
                raise _credential_error(provider_type, "HuggingFace API key required for Flux provider")
            return FluxProvider(
                api_key,
                flux_endpoint_url=credentials.get('huggingface_flux_endpoint_url'),
                **common_llm_kwargs,
            )

        elif provider_type == ImageProvider.FLUX_DEV_VERTEX:
            project_id = credentials.get('google_project_id') or credentials.get('gcp_project_id')
            if not project_id:
                raise _credential_error(
                    provider_type,
                    "Google Cloud project ID required for Flux Dev Vertex provider"
                )
            return FluxDevVertexProvider(
                gcp_project_id=project_id,
                flux_dev_vertex_endpoint_url=credentials.get('flux_dev_vertex_endpoint_url'),
                gcp_credentials=credentials.get('google_credentials'),
                **common_llm_kwargs,
            )

        elif provider_type == ImageProvider.HUGGINGFACE:
            api_key = credentials.get('huggingface_api_key')
            if not api_key:
                raise _credential_error(
                    provider_type,
                    "HuggingFace API key required for the HuggingFace image provider",
                )

            model_id = credentials.get('huggingface_image_model')
            if not model_id:
                raise _credential_error(
                    provider_type,
                    "HuggingFace image provider requires a model identifier (e.g. 'stabilityai/stable-diffusion-2-1')",
                )

            return HuggingFaceImageProvider(
                api_key,
                model_id=model_id,
                endpoint_url=credentials.get('huggingface_image_endpoint_url'),
                provider_override=credentials.get('huggingface_image_provider'),
                **common_llm_kwargs,
            )

        elif provider_type == ImageProvider.SEEDREAM:
            if not replicate_token:
                raise _credential_error(provider_type, "Replicate API token required for Seedream provider")

            return SeedreamProvider(
                replicate_token,
                **common_llm_kwargs,
            )

        else:
            raise _credential_error(provider_type, f"Unsupported provider type: {provider_type}")

    @staticmethod
    def get_available_providers(**credentials) -> List[ImageProvider]:
        """Get list of available providers based on provided credentials."""
        available = []

        anthropic_key = credentials.get('anthropic_api_key')
        llm_provider_value = credentials.get('llm_provider')
        if llm_provider_value is None and not anthropic_key:
            # Local pipeline can support prompt engineering without Anthropic key
            llm_provider = LLMProvider.HUGGINGFACE
        elif isinstance(llm_provider_value, LLMProvider):
            llm_provider = llm_provider_value
        elif llm_provider_value is not None:
            llm_provider = LLMProvider(llm_provider_value)
        else:
            llm_provider = LLMProvider.ANTHROPIC

        if llm_provider == LLMProvider.ANTHROPIC and not anthropic_key:
            return []

        if credentials.get('openai_api_key'):
            available.append(ImageProvider.DALLE)

        if credentials.get('google_credentials') and credentials.get('google_project_id'):
            available.append(ImageProvider.IMAGEN4)

        if credentials.get('huggingface_api_key'):
            available.append(ImageProvider.FLUX)
            if credentials.get('huggingface_image_model'):
                available.append(ImageProvider.HUGGINGFACE)

        replicate_token = credentials.get('replicate_api_token')
        if replicate_token:
            if ImageProvider.IMAGEN4 not in available:
                available.append(ImageProvider.IMAGEN4)
            if ImageProvider.FLUX not in available:
                available.append(ImageProvider.FLUX)
            available.append(ImageProvider.SEEDREAM)

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
            'gcp_project_id': getattr(context, 'gcp_project_id', None),
            'anthropic_api_key': context.anthropic_api_key,
            'llm_provider': getattr(context, 'llm_provider', None),
            'model': getattr(context, 'model', None),
            'huggingface_task': getattr(context, 'huggingface_task', None),
            'huggingface_device': getattr(context, 'huggingface_device', None),
            'huggingface_max_new_tokens': getattr(context, 'huggingface_max_new_tokens', None),
            'huggingface_temperature': getattr(context, 'huggingface_temperature', None),
            'huggingface_model_kwargs': getattr(context, 'huggingface_model_kwargs', None),
            'huggingface_flux_endpoint_url': getattr(context, 'huggingface_flux_endpoint_url', None),
            'flux_dev_vertex_endpoint_url': getattr(context, 'flux_dev_vertex_endpoint_url', None),
            'huggingface_image_model': getattr(context, 'huggingface_image_model', None),
            'huggingface_image_endpoint_url': getattr(context, 'huggingface_image_endpoint_url', None),
            'huggingface_image_provider': getattr(context, 'huggingface_image_provider', None),
            'replicate_api_token': getattr(context, 'replicate_api_token', None),
            **credentials
        }.items() if value is not None
    }

    return ProviderFactory.create_provider(provider_type, **creds)
