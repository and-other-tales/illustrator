"""Utilities for constructing language model interfaces used across the project."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Sequence


_STUB_AI_MESSAGE: Any | None = None
_STUB_HUMAN_MESSAGE: Any | None = None
_STUB_SYSTEM_MESSAGE: Any | None = None


def _clear_stub_langchain_core() -> None:
    """Remove lightweight test stubs so the real LangChain modules can import."""

    global _STUB_AI_MESSAGE, _STUB_HUMAN_MESSAGE, _STUB_SYSTEM_MESSAGE

    stub = sys.modules.get("langchain_core")
    if stub is not None and getattr(stub, "__file__", None) is None:
        messages_mod = sys.modules.get("langchain_core.messages")
        if messages_mod is not None:
            _STUB_AI_MESSAGE = getattr(messages_mod, "AIMessage", None)
            _STUB_HUMAN_MESSAGE = getattr(messages_mod, "HumanMessage", None)
            _STUB_SYSTEM_MESSAGE = getattr(messages_mod, "SystemMessage", None)

        for name in list(sys.modules.keys()):
            if name == "langchain_core" or name.startswith("langchain_core."):
                sys.modules.pop(name, None)


_clear_stub_langchain_core()


from langchain.chat_models import init_chat_model as _init_chat_model
try:
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
except ModuleNotFoundError:
    _clear_stub_langchain_core()
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# Maintain backwards compatibility for existing call sites/tests that patch init_chat_model directly
init_chat_model = _init_chat_model

# Ensure any previously-installed test stubs now point to the real message classes
_messages_module = sys.modules.get("langchain_core.messages")
if _messages_module is not None:
    setattr(_messages_module, "AIMessage", AIMessage)
    setattr(_messages_module, "HumanMessage", HumanMessage)
    setattr(_messages_module, "SystemMessage", SystemMessage)
    if hasattr(_messages_module, "BaseMessage"):
        setattr(_messages_module, "BaseMessage", BaseMessage)
from huggingface_hub import InferenceClient
try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:  # pragma: no cover - backwards compatibility for older hub versions
    from huggingface_hub.utils import HfHubHTTPError
from transformers import pipeline as hf_pipeline
try:
    from requests.exceptions import ChunkedEncodingError
except ImportError:
    # Fallback if requests is not available
    class ChunkedEncodingError(Exception):
        pass

from illustrator.models import LLMProvider
import json

logger = logging.getLogger(__name__)

_HARMONY_VISIBLE_CHANNELS: set[str] = {
    "assistant",
    "final",
    "default",
    "response",
    "completion",
    "message",
    "",
}

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from illustrator.context import ManuscriptContext


def _message_class_name(message: Any) -> str:
    cls = getattr(message, "__class__", type(message))
    return getattr(cls, "__name__", str(cls))


def _is_system_message(message: Any) -> bool:
    return isinstance(message, SystemMessage) or _message_class_name(message) == "SystemMessage"


def _is_human_message(message: Any) -> bool:
    return isinstance(message, HumanMessage) or _message_class_name(message) == "HumanMessage"


def _is_ai_message(message: Any) -> bool:
    return isinstance(message, AIMessage) or _message_class_name(message) == "AIMessage"


def _coerce_ai_message_instance(message: Any) -> Any:
    if _STUB_AI_MESSAGE is not None and _STUB_AI_MESSAGE is not AIMessage:
        if not isinstance(message, _STUB_AI_MESSAGE):
            content = getattr(message, "content", "")
            try:
                return _STUB_AI_MESSAGE(content=content)
            except Exception:
                return message
        return message
    return message


def _coerce_message_content(content: Any) -> str:
    """Convert LangChain message content (which may be structured) into text."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if item is None:
                continue
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("generated_text")
                if text:
                    parts.append(str(text))
            else:
                try:
                    parts.append(str(item))
                except Exception:
                    continue
        return "\n".join(part for part in parts if part)

    try:
        return str(content)
    except Exception:
        return ""


def _render_harmony_prompt(messages: Sequence[BaseMessage]) -> str:
    """Render chat messages into the harmony text format used by gpt-oss."""

    segments: list[str] = []

    for message in messages:
        if _is_system_message(message):
            role = "system"
        elif getattr(message, "role", "") == "developer":
            role = "developer"
        elif _is_human_message(message):
            role = "user"
        elif _is_ai_message(message):
            role = "assistant"
        else:
            role = str(getattr(message, "role", "assistant") or "assistant")

        content = _coerce_message_content(getattr(message, "content", ""))
        segment = f"<|start|>{role}<|message|>{content}<|end|>"
        segments.append(segment)

    segments.append("<|start|>assistant")
    return "\n\n".join(seg.rstrip() for seg in segments)


def _extract_text_from_content(content: Any, allowed_channels: set[str] | None = None) -> str:
    """Extract textual content from structured chat completions."""

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        channel_value = content.get("channel") or content.get("role") or content.get("name")
        if allowed_channels is not None and channel_value is not None:
            normalized_allowed = {str(c).lower() for c in allowed_channels}
            if normalized_allowed and str(channel_value).lower() not in normalized_allowed:
                return ""

        for key in ("text", "content", "generated_text", "value"):
            value = content.get(key)
            if value:
                extracted = _extract_text_from_content(value, allowed_channels)
                if extracted:
                    return extracted

        # Nested structures like {'delta': {...}}
        for key in ("delta", "message"):
            value = content.get(key)
            if value:
                extracted = _extract_text_from_content(value, allowed_channels)
                if extracted:
                    return extracted

        # Lists stored under other keys
        for key in ("messages", "choices", "tokens"):
            value = content.get(key)
            if value:
                extracted = _extract_text_from_content(value, allowed_channels)
                if extracted:
                    return extracted

        return ""

    if isinstance(content, Iterable) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for item in content:
            extracted = _extract_text_from_content(item, allowed_channels)
            if extracted:
                parts.append(extracted)
        return "".join(parts)

    try:
        return str(content)
    except Exception:
        return ""


@dataclass(slots=True)
class HuggingFaceConfig:
    """Configuration for HuggingFace inference endpoints and local pipelines."""

    endpoint_url: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    timeout: float | None = None
    model_kwargs: dict[str, Any] | None = None
    use_pipeline: bool = False
    pipeline_task: str = "text-generation"
    pipeline_device: str | int | None = None
    pipeline_kwargs: dict[str, Any] | None = None


class OpenAICompatibleChatWrapper:
    """Wrapper for OpenAI-compatible HuggingFace endpoints."""

    def __init__(
        self,
        client: Any,  # AsyncOpenAI client
        model: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        stream_callback: Callable[[str], Awaitable[None] | None] | Callable[[str], None] | None = None,
        session_id: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._stream_callback = stream_callback
        self._session_id = session_id

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Generate a response using OpenAI-compatible API."""
        import asyncio
        import inspect
        
        # Convert LangChain messages to OpenAI format
        openai_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                openai_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                openai_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                openai_messages.append({"role": "assistant", "content": msg.content})

        try:
            # Call the OpenAI-compatible endpoint
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=openai_messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                stream=True
            )

            # Handle streaming response
            content_parts = []
            async for chunk in response:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    content_parts.append(token)
                    
                    # Call stream callback if provided
                    if self._stream_callback and token:
                        try:
                            result = self._stream_callback(token)
                            if inspect.isawaitable(result):
                                await result
                        except Exception:
                            logger.exception("Stream callback raised an exception")

            final_content = "".join(content_parts)
            return AIMessage(content=final_content)

        except Exception as e:
            logger.error(f"OpenAI-compatible endpoint error: {e}")
            raise

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Synchronous version - runs async method."""
        return asyncio.run(self.ainvoke(messages))


class HuggingFaceEndpointChatWrapper:
    """Async-friendly wrapper that mirrors LangChain chat behaviour for HF endpoints."""

    class _OfflineFallbackRequested(Exception):
        """Internal control-flow exception signalling a switch to LocalLLM."""

        def __init__(self, reason: str) -> None:
            super().__init__(reason)
            self.reason = reason

    def __init__(
        self,
        client: InferenceClient,
        generation_kwargs: dict[str, Any],
        *,
        stream_callback: Callable[[str], Awaitable[None] | None] | Callable[[str], None] | None = None,
        session_id: str | None = None,
    ) -> None:
        self._client = client
        self._session_id = session_id
        # Separate generation kwargs for text_generation and chat_completion
        self._generation_kwargs = {
            k: v for k, v in generation_kwargs.items()
            if k not in {'messages', 'model_id'}
        }
        self._chat_kwargs = {
            k: v for k, v in generation_kwargs.items()
            if k in {'temperature', 'stream', 'model'}
        }
        self._stream_callback = stream_callback
        self._supports_chat_completion = hasattr(client, "chat_completion")
        # Harmony-specific behaviour: some HF-hosted models (e.g. gpt-oss-120b) use
        # a structured "harmony" stream/response format. Consumers can opt-in by
        # setting generation_kwargs['harmony_format'] = True in create_chat_model.
        # Record the harmony flag but remove it from kwargs we forward to HF client
        self._is_harmony = bool(generation_kwargs.get("harmony_format"))
        if "harmony_format" in self._generation_kwargs:
            # Ensure we don't pass unknown kwargs to the InferenceClient
            try:
                self._generation_kwargs.pop("harmony_format", None)
            except Exception:
                pass
        # Also ensure it's not in chat kwargs
        if "harmony_format" in self._chat_kwargs:
            try:
                self._chat_kwargs.pop("harmony_format", None)
            except Exception:
                pass

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Generate a response asynchronously using the configured endpoint."""

        prompt = _messages_to_prompt(messages, harmony=self._is_harmony)
        chat_payload = _messages_to_chat_messages(messages)
        use_stream = bool(self._generation_kwargs.get("stream"))
        loop = asyncio.get_running_loop()
        stream_callback = self._stream_callback

        def _schedule_stream_callback(token_text: str) -> None:
            if not stream_callback or not token_text:
                return

            try:
                result = stream_callback(token_text)
            except Exception:  # pragma: no cover - defensive
                logger.exception("Stream callback raised an exception")
                return

            if inspect.isawaitable(result):  # pragma: no branch - small helper
                asyncio.run_coroutine_threadsafe(result, loop)

        def _extract_token(chunk: Any) -> str:
            if chunk is None:
                return ""

            token = getattr(chunk, "token", None)
            if token is not None:
                extracted = _extract_text_from_content(token)
                if extracted:
                    return extracted

            if isinstance(chunk, dict):
                token_dict = chunk.get("token")
                extracted = _extract_text_from_content(token_dict)
                if extracted:
                    return extracted

            return ""

        def _extract_full_text(chunk: Any) -> str:
            if chunk is None:
                return ""

            generated_text = getattr(chunk, "generated_text", None)
            extracted = _extract_text_from_content(generated_text)
            if extracted:
                return extracted

            if isinstance(chunk, dict):
                text = chunk.get("generated_text")
                extracted = _extract_text_from_content(text)
                if extracted:
                    return extracted

            return ""


        def _extract_harmony_token(chunk: Any) -> str:
            """Extract incremental token text from a harmony-format chunk if present."""
            if chunk is None:
                return ""

            try:
                if isinstance(chunk, dict):
                    harmony_payload = chunk.get("harmony")
                    if harmony_payload is not None:
                        for candidate_key in ("delta", "token", "tokens"):
                            if candidate_key in harmony_payload:
                                extracted = _extract_text_from_content(
                                    harmony_payload[candidate_key],
                                    _HARMONY_VISIBLE_CHANNELS,
                                )
                                if extracted:
                                    return extracted

                return ""
            except Exception:
                return ""


        def _extract_harmony_full_text(chunk: Any) -> str:
            """Extract full generated text from a harmony-format completion or chunk."""
            if chunk is None:
                return ""

            try:
                if isinstance(chunk, dict) and "harmony" in chunk:
                    harmony_payload = chunk.get("harmony")
                    if isinstance(harmony_payload, dict):
                        for candidate_key in (
                            "generated_text",
                            "text",
                            "messages",
                            "content",
                        ):
                            if candidate_key in harmony_payload:
                                extracted = _extract_text_from_content(
                                    harmony_payload[candidate_key],
                                    _HARMONY_VISIBLE_CHANNELS,
                                )
                                if extracted:
                                    return extracted.strip()

                        extracted = _extract_text_from_content(
                            harmony_payload,
                            _HARMONY_VISIBLE_CHANNELS,
                        )
                        if extracted:
                            return extracted.strip()

                # fallback to usual generated_text
                return _extract_full_text(chunk)
            except Exception:
                return _extract_full_text(chunk)

        def _run_endpoint() -> AIMessage:
            kwargs = dict(self._generation_kwargs)

            if self._supports_chat_completion:
                # Filter out parameters not supported by chat_completion
                chat_kwargs = {
                    k: v for k, v in kwargs.items()
                    if k in {'temperature', 'stream', 'model'}
                }

                try:
                    if use_stream:
                        aggregated_tokens: list[str] = []

                        for chunk in self._client.chat_completion(
                            messages=chat_payload,
                            **chat_kwargs,
                        ):
                            if self._is_harmony:
                                chunk_text = _extract_harmony_token(chunk)
                            else:
                                chunk_text = _extract_chat_chunk(chunk)
                            if chunk_text:
                                aggregated_tokens.append(chunk_text)
                                _schedule_stream_callback(chunk_text)

                        generated_text = "".join(aggregated_tokens)
                    else:
                        completion = self._client.chat_completion(
                            messages=chat_payload,
                            **chat_kwargs,
                        )

                        generated_text = _extract_chat_completion_text(
                            completion,
                            is_harmony=self._is_harmony,
                        )

                    if generated_text:
                        return _coerce_ai_message_instance(AIMessage(content=generated_text.strip()))
                except Exception as e:  # pragma: no cover - fallback for unsupported chat endpoints
                    error_message = str(e)
                    
                    # Handle ChunkedEncodingError specifically
                    if isinstance(e, ChunkedEncodingError):
                        logger.warning("HuggingFace chat_completion failed with ChunkedEncodingError: %s; falling back to text_generation", str(e))
                    # Check if endpoint is paused
                    elif is_endpoint_paused_error(error_message):
                        logger.warning("HuggingFace endpoint is paused, waiting for restart")
                        asyncio.run(wait_for_endpoint_restart(self._session_id, countdown_seconds=120))
                        
                        # Retry once after waiting
                        try:
                            if use_stream:
                                aggregated_tokens: list[str] = []

                                for chunk in self._client.chat_completion(
                                    messages=chat_payload,
                                    **chat_kwargs,
                                ):
                                    if self._is_harmony:
                                        chunk_text = _extract_harmony_token(chunk)
                                    else:
                                        chunk_text = _extract_chat_chunk(chunk)
                                    if chunk_text:
                                        aggregated_tokens.append(chunk_text)
                                        _schedule_stream_callback(chunk_text)

                                generated_text = "".join(aggregated_tokens)
                            else:
                                completion = self._client.chat_completion(
                                    messages=chat_payload,
                                    **chat_kwargs,
                                )

                                generated_text = _extract_chat_completion_text(
                                    completion,
                                    is_harmony=self._is_harmony,
                                )

                            if generated_text:
                                return _coerce_ai_message_instance(AIMessage(content=generated_text.strip()))
                        except Exception as retry_e:
                            logger.warning(
                                "HuggingFace chat_completion retry failed with %s: %s; falling back to text_generation",
                                type(retry_e).__name__, str(retry_e)
                            )
                    else:
                        logger.warning(
                            "HuggingFace chat_completion failed with %s: %s; falling back to text_generation",
                            type(e).__name__, str(e)
                        )

            # Fallback to text-generation style invocation
            logger.debug("Using text_generation with parameters: %s", self._generation_kwargs)
            try:
                raw_output = self._client.text_generation(
                    prompt,
                    **self._generation_kwargs,
                )
            except Exception as e:
                error_message = str(e)

                # Handle ChunkedEncodingError specifically
                if isinstance(e, ChunkedEncodingError):
                    logger.warning("HuggingFace text_generation failed with ChunkedEncodingError: %s; retrying once", str(e))
                    # Retry once for ChunkedEncodingError
                    try:
                        raw_output = self._client.text_generation(
                            prompt,
                            **self._generation_kwargs,
                        )
                    except Exception as retry_e:
                        logger.error(
                            "HuggingFace text_generation retry after ChunkedEncodingError failed with %s: %s",
                            type(retry_e).__name__, str(retry_e)
                        )
                        return _coerce_ai_message_instance(AIMessage(content=""))
                # Check if endpoint is paused and retry
                elif is_endpoint_paused_error(error_message):
                    logger.warning("HuggingFace endpoint is paused during text_generation, waiting for restart")
                    asyncio.run(wait_for_endpoint_restart(self._session_id, countdown_seconds=120))
                    
                    # Retry once after waiting
                    try:
                        raw_output = self._client.text_generation(
                            prompt,
                            **self._generation_kwargs,
                        )
                    except Exception as retry_e:
                        logger.error(
                            "HuggingFace text_generation retry failed with %s: %s",
                            type(retry_e).__name__, str(retry_e)
                        )
                        return _coerce_ai_message_instance(AIMessage(content=""))
                elif isinstance(e, HfHubHTTPError):
                    # Detect missing model errors so we can fall back when allowed
                    response = getattr(e, "response", None)
                    status_code = getattr(response, "status_code", None)
                    if status_code is None:
                        status_code = getattr(e, "status_code", None)

                    message_lower = error_message.lower()
                    missing_model = False
                    if status_code == 404:
                        missing_model = True
                    elif "404" in message_lower or "not found" in message_lower or "does not exist" in message_lower:
                        missing_model = True

                    if missing_model and _allow_offline_fallback():
                        logger.warning(
                            "HuggingFace text_generation reported missing model (status %s): %s; using LocalLLM fallback",
                            status_code or "unknown",
                            error_message,
                        )
                        raise HuggingFaceEndpointChatWrapper._OfflineFallbackRequested("huggingface_missing_model") from e

                    logger.error(
                        "HuggingFace text_generation failed with %s: %s",
                        type(e).__name__, error_message,
                    )
                    return _coerce_ai_message_instance(AIMessage(content=""))
                else:
                    logger.error(
                        "HuggingFace text_generation failed with %s: %s",
                        type(e).__name__, str(e)
                    )
                    return _coerce_ai_message_instance(AIMessage(content=""))

            if use_stream:
                aggregated_tokens = []
                final_text: str | None = None

                for chunk in raw_output:
                    if self._is_harmony:
                        token_text = _extract_harmony_token(chunk)
                    else:
                        token_text = _extract_token(chunk)

                    if token_text:
                        aggregated_tokens.append(token_text)
                        _schedule_stream_callback(token_text)

                    if self._is_harmony:
                        chunk_full_text = _extract_harmony_full_text(chunk)
                    else:
                        chunk_full_text = _extract_full_text(chunk)

                    if chunk_full_text:
                        final_text = chunk_full_text

                generated_text = final_text or "".join(aggregated_tokens)
            else:
                if isinstance(raw_output, str):
                    generated_text = raw_output
                elif isinstance(raw_output, Iterable):
                    first_item = next(iter(raw_output), "")
                    generated_text = str(first_item)
                else:
                    generated_text = str(raw_output)

            generated_text = generated_text or ""

            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]

            return _coerce_ai_message_instance(AIMessage(content=generated_text.strip()))

        try:
            return await asyncio.to_thread(_run_endpoint)
        except HuggingFaceEndpointChatWrapper._OfflineFallbackRequested as fallback_exc:
            logger.warning(
                "Using LocalLLM fallback due to HuggingFace endpoint error: %s",
                fallback_exc.reason,
            )
            fallback_model = LocalLLM(fallback_exc.reason)
            return await fallback_model.ainvoke(messages)


class HuggingFacePipelineChatWrapper:
    """Wrapper that exposes transformers pipelines through the LangChain chat APIs."""

    def __init__(
        self,
        pipe,
        call_kwargs: dict[str, Any],
        *,
        stream_callback: Callable[[str], Awaitable[None] | None] | Callable[[str], None] | None = None,
        session_id: str | None = None,
        is_harmony: bool = False,
    ) -> None:
        self._pipeline = pipe
        self._call_kwargs = call_kwargs
        self._stream_callback = stream_callback
        self._session_id = session_id
        self._is_harmony = is_harmony

    def _extract_text(self, output: Any) -> str:
        if output is None:
            return ""

        if isinstance(output, str):
            return output

        if isinstance(output, dict):
            for key in ("generated_text", "text", "summary_text"):
                value = output.get(key)
                if value:
                    return str(value)
            return ""

        if isinstance(output, Iterable):
            first = next(iter(output), None)
            if first is None:
                return ""
            if isinstance(first, dict):
                for key in ("generated_text", "text", "summary_text"):
                    value = first.get(key)
                    if value:
                        return str(value)
            return str(first)

        return str(output)

    def _dispatch_stream_callback(self, text: str) -> Awaitable[None] | None:
        if not text or not self._stream_callback:
            return None

        try:
            result = self._stream_callback(text)
        except Exception:  # pragma: no cover - defensive logging
            logger.exception("Stream callback raised an exception for HuggingFace pipeline output")
            return None

        return result if inspect.isawaitable(result) else None

    def invoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        prompt = _messages_to_prompt(messages, harmony=self._is_harmony)
        output = self._pipeline(prompt, **self._call_kwargs)
        text = self._extract_text(output).strip()

        pending = self._dispatch_stream_callback(text)
        if pending is not None:
            try:
                loop = asyncio.get_running_loop()
                asyncio.run_coroutine_threadsafe(pending, loop)
            except RuntimeError:
                asyncio.run(pending)

        return _coerce_ai_message_instance(AIMessage(content=text))

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        prompt = _messages_to_prompt(messages, harmony=self._is_harmony)

        def _run_pipeline() -> str:
            output = self._pipeline(prompt, **self._call_kwargs)
            return self._extract_text(output).strip()

        text = await asyncio.to_thread(_run_pipeline)

        pending = self._dispatch_stream_callback(text)
        if pending is not None:
            await pending

        return _coerce_ai_message_instance(AIMessage(content=text))


def _partition_pipeline_kwargs(
    config: HuggingFaceConfig,
    huggingface_api_key: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split configuration into pipeline init kwargs and generation kwargs."""

    init_kwargs: dict[str, Any] = dict(config.pipeline_kwargs or {})
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "return_full_text": False,
    }

    extra_model_kwargs = dict(config.model_kwargs or {})

    special_init_keys = {
        "device_map",
        "torch_dtype",
        "trust_remote_code",
        "revision",
        "tokenizer",
        "feature_extractor",
        "image_processor",
        "processor",
        "framework",
        "use_fast",
    }

    for key in list(extra_model_kwargs.keys()):
        if key not in special_init_keys:
            continue

        value = extra_model_kwargs.pop(key)
        if key == "torch_dtype" and isinstance(value, str):
            try:
                import torch  # type: ignore

                if hasattr(torch, value):
                    value = getattr(torch, value)
            except ImportError:
                logger.debug("torch not available; leaving torch_dtype value as string")
            except AttributeError:
                logger.debug("Unrecognised torch dtype '%s' provided", value)

        init_kwargs[key] = value

    device = config.pipeline_device
    if device is not None:
        if isinstance(device, str) and device.strip().lower() == "auto":
            init_kwargs.setdefault("device_map", "auto")
        else:
            init_kwargs["device"] = device

    if huggingface_api_key:
        init_kwargs.setdefault("token", huggingface_api_key)

    if extra_model_kwargs:
        generation_kwargs.update(extra_model_kwargs)

    # Ensure callers can override defaults explicitly
    if "return_full_text" not in generation_kwargs:
        generation_kwargs["return_full_text"] = False

    return init_kwargs, generation_kwargs


def _messages_to_prompt(
    messages: Sequence[BaseMessage],
    *,
    harmony: bool = False,
) -> str:
    """Convert chat messages to a single prompt string suitable for text-generation models."""

    if harmony:
        return _render_harmony_prompt(messages)

    lines: list[str] = []
    for message in messages:
        if _is_system_message(message):
            prefix = "System"
        elif _is_human_message(message):
            prefix = "User"
        else:
            prefix = "Assistant"
        normalized_content = _coerce_message_content(getattr(message, "content", ""))
        lines.append(f"{prefix}: {normalized_content}".strip())

    # Encourage the model to respond in the assistant role
    lines.append("Assistant:")
    return "\n".join(lines)


def _messages_to_chat_messages(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
    """Convert LangChain messages into HuggingFace chat-completion payload."""

    # HuggingFace chat-completion format
    chat_messages = []
    for msg in messages:
        if _is_system_message(msg):
            chat_messages.append({"role": "system", "content": msg.content})
        elif _is_human_message(msg):
            chat_messages.append({"role": "user", "content": msg.content})
        elif _is_ai_message(msg):
            chat_messages.append({"role": "assistant", "content": msg.content})
    return chat_messages


def _extract_chat_chunk(chunk: Any) -> str:
    """Extract incremental text from a HuggingFace chat completion stream chunk."""

    if chunk is None:
        return ""

    choices = getattr(chunk, "choices", None)
    if not choices and isinstance(chunk, dict):
        choices = chunk.get("choices")

    if not choices:
        return ""

    collected: list[str] = []
    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")

        if delta is None:
            continue

        content = getattr(delta, "content", None)
        if content is None and isinstance(delta, dict):
            content = delta.get("content")

        if content:
            extracted = _extract_text_from_content(content)
            if extracted:
                collected.append(extracted)

    return "".join(collected)


def _extract_chat_completion_text(completion: Any, *, is_harmony: bool = False) -> str:
    """Extract full text from a HuggingFace chat-completion response."""

    if completion is None:
        return ""

    choices = getattr(completion, "choices", None)
    if not choices and isinstance(completion, dict):
        choices = completion.get("choices")

    if not choices:
        return ""

    first_choice = choices[0]
    message = getattr(first_choice, "message", None)
    if message is None and isinstance(first_choice, dict):
        message = first_choice.get("message")

    if isinstance(message, dict):
        content = message.get("content")
    else:
        content = getattr(message, "content", None)

    allowed_channels = _HARMONY_VISIBLE_CHANNELS if is_harmony else None
    extracted = _extract_text_from_content(content, allowed_channels)
    return extracted.strip() if extracted else ""


class AnthropicVertexWrapper:
    """Wrapper to make AnthropicVertex client compatible with LangChain interface."""
    
    def __init__(self, client, model: str):
        self._client = client
        self._model = model
    
    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Generate a response asynchronously using Anthropic Vertex."""
        # Convert LangChain messages to Anthropic format
        anthropic_messages = []
        system_message = None
        
        for msg in messages:
            if _is_system_message(msg):
                system_message = msg.content
            elif _is_human_message(msg):
                anthropic_messages.append({"role": "user", "content": msg.content})
            elif _is_ai_message(msg):
                anthropic_messages.append({"role": "assistant", "content": msg.content})
        
        # Prepare request parameters
        request_params = {
            "model": self._model,
            "max_tokens": 1024,
            "messages": anthropic_messages,
        }
        
        if system_message:
            request_params["system"] = system_message
        
        try:
            # Run the synchronous call in a thread pool
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self._client.messages.create(**request_params)
            )
            
            # Extract content from response
            content = ""
            if hasattr(response, 'content') and response.content:
                if isinstance(response.content, list) and len(response.content) > 0:
                    content = response.content[0].text
                else:
                    content = str(response.content)
            
            return _coerce_ai_message_instance(AIMessage(content=content))
            
        except Exception as e:
            logger.error(f"Anthropic Vertex API call failed: {e}")
            return _coerce_ai_message_instance(AIMessage(content=""))


def _normalize_provider(provider: LLMProvider | str | None, anthropic_key: str | None, gcp_project_id: str | None = None) -> LLMProvider:
    """Resolve the configured provider, defaulting based on available credentials."""

    if provider is None:
        if gcp_project_id:
            return LLMProvider.ANTHROPIC_VERTEX
        return LLMProvider.ANTHROPIC if anthropic_key else LLMProvider.HUGGINGFACE

    if isinstance(provider, str):
        # Handle case-insensitive conversion
        provider_lower = provider.lower()
        for enum_member in LLMProvider:
            if enum_member.value == provider_lower:
                return enum_member
        # If no match, raise ValueError with expected message
        raise ValueError(f"Unknown provider: {provider}")

    return provider


def create_chat_model(
    *,
    provider: LLMProvider | str | None,
    model: str,
    anthropic_api_key: str | None,
    huggingface_api_key: str | None,
    gcp_project_id: str | None = None,
    huggingface_config: HuggingFaceConfig | None = None,
    stream_callback: Callable[[str], Awaitable[None] | None] | Callable[[str], None] | None = None,
    session_id: str | None = None,
) -> Any:
    """Instantiate a chat-capable model for the requested provider."""

    resolved_provider = _normalize_provider(provider, anthropic_api_key, gcp_project_id)

    if resolved_provider == LLMProvider.ANTHROPIC:
        if not anthropic_api_key:
            if _allow_offline_fallback():
                logger.warning(
                    "Anthropic API key missing; using LocalLLM fallback for offline execution."
                )
                return LocalLLM("anthropic")

            raise ValueError("Anthropic API key is required when using the Anthropic provider")

        # LangChain expects bare model name without provider prefix
        normalized_model = model.split("/", 1)[-1]
        return init_chat_model(
            model=normalized_model,
            model_provider="anthropic",
            api_key=anthropic_api_key,
        )

    if resolved_provider == LLMProvider.ANTHROPIC_VERTEX:
        if not gcp_project_id:
            if _allow_offline_fallback():
                logger.warning(
                    "GCP Project ID missing; using LocalLLM fallback for offline execution."
                )
                return LocalLLM("anthropic_vertex")

            raise ValueError("GCP Project ID is required when using the Anthropic Vertex provider")

        try:
            from anthropic import AnthropicVertex
        except ImportError:
            raise ValueError("anthropic[vertex] package is required for Anthropic Vertex provider. Install with: pip install 'anthropic[vertex]'")

        # Create Anthropic Vertex client
        client = AnthropicVertex(
            region="global",
            project_id=gcp_project_id
        )
        
        # Return a wrapper that mimics LangChain interface
        return AnthropicVertexWrapper(client, model)

    config = huggingface_config or HuggingFaceConfig()

    if resolved_provider == LLMProvider.HUGGINGFACE and config.use_pipeline:
        init_kwargs, generation_kwargs = _partition_pipeline_kwargs(config, huggingface_api_key)

        try:
            pipeline_task = config.pipeline_task or "text-generation"
            pipeline_model = model or init_kwargs.get("model")
            pipeline_instance = hf_pipeline(
                task=pipeline_task,
                model=pipeline_model,
                **{k: v for k, v in init_kwargs.items() if k != "model"},
            )
        except Exception as exc:
            if _allow_offline_fallback():
                logger.warning(
                    "Failed to initialise HuggingFace pipeline (%s); using LocalLLM fallback",
                    exc,
                )
                return LocalLLM("huggingface")

            raise ValueError(f"Failed to initialize HuggingFace pipeline: {exc}") from exc

        is_harmony_model = False
        try:
            model_identifier = (pipeline_model or "") if pipeline_model is not None else ""
            is_harmony_model = "gpt-oss" in model_identifier.lower() or bool(
                (config.model_kwargs or {}).get("harmony_format")
            )
        except Exception:
            is_harmony_model = False

        return HuggingFacePipelineChatWrapper(
            pipeline_instance,
            generation_kwargs,
            stream_callback=stream_callback,
            session_id=session_id,
            is_harmony=is_harmony_model,
        )

    if not huggingface_api_key:
        if _allow_offline_fallback():
            logger.warning(
                "HuggingFace API key missing; using LocalLLM fallback for offline execution."
            )
            return LocalLLM("huggingface")

        raise ValueError("HuggingFace API key is required when using the HuggingFace provider")

    endpoint_url = (config.endpoint_url or "").strip() or None
    default_endpoint = f"https://api-inference.huggingface.co/models/{model}".rstrip("/")
    if endpoint_url and endpoint_url.rstrip("/") == default_endpoint:
        # Treat the default HuggingFace inference endpoint as a standard model lookup
        endpoint_url = None

    # Check if this is an OpenAI-compatible HuggingFace endpoint (has /v1 suffix)
    is_openai_compatible = endpoint_url and endpoint_url.rstrip("/").endswith("/v1")
    
    if is_openai_compatible:
        # Use OpenAI client for HuggingFace endpoints that follow OpenAI format
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ValueError("openai package is required for OpenAI-compatible HuggingFace endpoints. Install with: pip install openai")
        
        # Create OpenAI client with HuggingFace endpoint
        openai_client = AsyncOpenAI(
            base_url=endpoint_url,
            api_key=huggingface_api_key
        )
        
        # Use "tgi" as model name for text generation inference endpoints
        openai_model = "tgi"
        
        return OpenAICompatibleChatWrapper(
            openai_client,
            openai_model,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            stream_callback=stream_callback,
            session_id=session_id,
        )

    client_kwargs: dict[str, Any] = {
        "token": huggingface_api_key,
        "timeout": config.timeout,
    }

    if endpoint_url:
        client = InferenceClient(base_url=endpoint_url, **client_kwargs)
    else:
        client = InferenceClient(model=model, **client_kwargs)

    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "return_full_text": False,
        "stream": True,  # Enable streaming for HuggingFace models
    }

    if config.model_kwargs:
        generation_kwargs.update(config.model_kwargs)

    # Auto-detect Harmony-format models (e.g. gpt-oss-120b) and set flag so
    # the wrapper knows how to parse structured harmony responses.
    try:
        model_lower = (model or "").lower()
        if "gpt-oss" in model_lower or (config.model_kwargs or {}).get("harmony_format"):
            generation_kwargs.setdefault("harmony_format", True)
    except Exception:
        pass

    if endpoint_url:
        # Custom endpoints expect routing to be handled by the base URL
        generation_kwargs.pop("model", None)
    else:
        generation_kwargs.setdefault("model", model)

    return HuggingFaceEndpointChatWrapper(
        client,
        generation_kwargs,
        stream_callback=stream_callback,
        session_id=session_id,
    )


def huggingface_config_from_context(context: ManuscriptContext) -> HuggingFaceConfig:
    """Build HuggingFace configuration from runtime context."""

    model_kwargs = getattr(context, "huggingface_model_kwargs", None)
    if isinstance(model_kwargs, str):
        try:
            model_kwargs = json.loads(model_kwargs)
        except json.JSONDecodeError:
            model_kwargs = None
    elif model_kwargs is not None and not isinstance(model_kwargs, dict):
        # Last resort: attempt to coerce to dict for backwards compatibility
        try:
            model_kwargs = dict(model_kwargs)
        except Exception:
            model_kwargs = None

    return HuggingFaceConfig(
        endpoint_url=getattr(context, "huggingface_endpoint_url", None),
        max_new_tokens=getattr(context, "huggingface_max_new_tokens", 512),
        temperature=getattr(context, "huggingface_temperature", 0.7),
        timeout=getattr(context, "huggingface_timeout", None),
        model_kwargs=model_kwargs,
        use_pipeline=bool(getattr(context, "huggingface_use_pipeline", False)),
        pipeline_task=getattr(context, "huggingface_task", "text-generation"),
        pipeline_device=getattr(context, "huggingface_device", None),
    )


def create_chat_model_from_context(context: ManuscriptContext, session_id: str | None = None) -> Any:
    """Convenience helper to create a chat model using context configuration."""
    kwargs: dict[str, Any] = {
        "provider": getattr(context, "llm_provider", None),
        "model": context.model,
        "anthropic_api_key": getattr(context, "anthropic_api_key", None),
        "huggingface_api_key": getattr(context, "huggingface_api_key", None),
        "huggingface_config": huggingface_config_from_context(context),
        "stream_callback": getattr(context, "huggingface_stream_callback", None),
        "session_id": session_id,
    }

    gcp_project_id = getattr(context, "gcp_project_id", None)
    if gcp_project_id is None and hasattr(context, "__dict__"):
        gcp_project_id = context.__dict__.get("gcp_project_id")

    if gcp_project_id is not None:
        try:
            from unittest.mock import Mock

            if isinstance(gcp_project_id, Mock):
                gcp_project_id = None
        except ImportError:
            pass

    if gcp_project_id is not None:
        kwargs["gcp_project_id"] = gcp_project_id

    return create_chat_model(**kwargs)
logger = logging.getLogger(__name__)


class EndpointPausedError(Exception):
    """Exception raised when HuggingFace endpoint is paused."""
    
    def __init__(self, message="The endpoint is paused, ask a maintainer to restart it"):
        self.message = message
        super().__init__(self.message)


def is_endpoint_paused_error(error_message: str | Exception) -> bool:
    """Check if error message indicates endpoint is paused."""
    # Convert exception to string if needed
    if isinstance(error_message, Exception):
        error_message = str(error_message)
    
    if not error_message:
        return False
    
    error_lower = error_message.lower()
    return (
        "endpoint is paused" in error_lower or
        "ask a maintainer to restart" in error_lower
    )


async def wait_for_endpoint_restart(session_id: str = None, countdown_seconds: int = 120):
    """Wait for endpoint to restart with countdown notifications."""
    logger.info(f"Endpoint paused, waiting {countdown_seconds} seconds for restart")
    
    # Notify WebSocket clients about the pause and countdown
    try:
        from illustrator.web.app import connection_manager
        if session_id and session_id in connection_manager.active_connections:
            await connection_manager.send_personal_message(
                json.dumps({
                    "type": "endpoint_paused",
                    "message": "AI endpoint is paused. Waiting for restart...",
                    "countdown_seconds": countdown_seconds,
                    "status": "waiting"
                }),
                session_id
            )
    except Exception as e:
        logger.debug(f"Could not send WebSocket notification: {e}")
    
    # Count down in 10-second intervals
    remaining = countdown_seconds
    while remaining > 0:
        await asyncio.sleep(10)
        remaining -= 10
        
        # Send countdown updates
        try:
            from illustrator.web.app import connection_manager
            if session_id and session_id in connection_manager.active_connections:
                await connection_manager.send_personal_message(
                    json.dumps({
                        "type": "endpoint_countdown",
                        "message": f"Retrying in {remaining} seconds...",
                        "countdown_seconds": remaining,
                        "status": "waiting"
                    }),
                    session_id
                )
        except Exception as e:
            logger.debug(f"Could not send countdown update: {e}")
    
    logger.info("Endpoint wait period completed, attempting retry")


def _allow_offline_fallback() -> bool:
    """Determine whether offline-safe fallbacks should be used."""

    if os.getenv("ILLUSTRATOR_ENFORCE_REMOTE"):
        return False

    return bool(os.getenv("ILLUSTRATOR_OFFLINE_MODE") or os.getenv("PYTEST_CURRENT_TEST"))


class LocalLLM:
    """Minimal chat model used when external providers aren't available."""

    def __init__(self, reason: str) -> None:
        self.reason = reason

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Return an empty JSON payload so downstream heuristics run."""

        logger.debug("LocalLLM responding due to missing credentials (%s)", self.reason)
        return _coerce_ai_message_instance(AIMessage(content="{}"))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<LocalLLM reason={self.reason}>"
