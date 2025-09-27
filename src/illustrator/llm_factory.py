"""Utilities for constructing language model interfaces used across the project."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
import inspect
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Iterable, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from huggingface_hub import InferenceClient

from illustrator.models import LLMProvider
import json

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from illustrator.context import ManuscriptContext


@dataclass(slots=True)
class HuggingFaceConfig:
    """Configuration for HuggingFace Inference Endpoints."""

    endpoint_url: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    timeout: float | None = None
    model_kwargs: dict[str, Any] | None = None


class HuggingFaceEndpointChatWrapper:
    """Async-friendly wrapper that mirrors LangChain chat behaviour for HF endpoints."""

    def __init__(
        self,
        client: InferenceClient,
        generation_kwargs: dict[str, Any],
        *,
        stream_callback: Callable[[str], Awaitable[None] | None] | Callable[[str], None] | None = None,
    ) -> None:
        self._client = client
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
        self._is_harmony = bool(generation_kwargs.get("harmony_format"))

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Generate a response asynchronously using the configured endpoint."""

        prompt = _messages_to_prompt(messages)
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
                text = getattr(token, "text", None)
                if text:
                    return str(text)

            if isinstance(chunk, dict):
                token_dict = chunk.get("token")
                if isinstance(token_dict, dict):
                    text = token_dict.get("text")
                    if text:
                        return str(text)

            return ""

        def _extract_full_text(chunk: Any) -> str:
            if chunk is None:
                return ""

            generated_text = getattr(chunk, "generated_text", None)
            if generated_text:
                return str(generated_text)

            if isinstance(chunk, dict):
                text = chunk.get("generated_text")
                if text:
                    return str(text)

            return ""


        def _extract_harmony_token(chunk: Any) -> str:
            """Extract incremental token text from a harmony-format chunk if present."""
            if chunk is None:
                return ""

            # Harmony streaming chunks may be dicts with a 'harmony' key containing
            # a nested delta/token structure. Be defensive about shapes.
            try:
                if isinstance(chunk, dict) and "harmony" in chunk:
                    h = chunk.get("harmony")
                    if isinstance(h, dict):
                        # common shapes: {'harmony': {'delta': {'content': '...'}}}
                        delta = h.get("delta") or h.get("token") or {}
                        if isinstance(delta, dict):
                            text = delta.get("content") or delta.get("text")
                            if text:
                                return str(text)
                        # other shape: {'harmony': {'tokens': [{'text': '...'}]}}
                        tokens = h.get("tokens")
                        if isinstance(tokens, list) and tokens:
                            first = tokens[0]
                            if isinstance(first, dict):
                                return str(first.get("text", ""))
                    if isinstance(h, str):
                        return h
            except Exception:
                return ""

            return ""


        def _extract_harmony_full_text(chunk: Any) -> str:
            """Extract full generated text from a harmony-format completion or chunk."""
            if chunk is None:
                return ""

            try:
                if isinstance(chunk, dict) and "harmony" in chunk:
                    h = chunk.get("harmony")
                    if isinstance(h, dict):
                        # try several common keys
                        text = h.get("generated_text") or h.get("text")
                        if text:
                            return str(text)
                        # nested message list
                        messages = h.get("messages") or h.get("content")
                        if isinstance(messages, list):
                            parts = []
                            for m in messages:
                                if isinstance(m, dict):
                                    parts.append(str(m.get("content", "")))
                                else:
                                    parts.append(str(m))
                            return "".join(parts).strip()
                        if isinstance(h, str):
                            return h
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

                        generated_text = _extract_chat_completion_text(completion)

                    if generated_text:
                        return AIMessage(content=generated_text.strip())
                except Exception as e:  # pragma: no cover - fallback for unsupported chat endpoints
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
                logger.error(
                    "HuggingFace text_generation failed with %s: %s",
                    type(e).__name__, str(e)
                )
                return AIMessage(content="")

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

            return AIMessage(content=generated_text.strip())

        return await asyncio.to_thread(_run_endpoint)


def _messages_to_prompt(messages: Sequence[BaseMessage]) -> str:
    """Convert chat messages to a single prompt string suitable for text-generation models."""

    lines: list[str] = []
    for message in messages:
        if isinstance(message, SystemMessage):
            prefix = "System"
        elif isinstance(message, HumanMessage):
            prefix = "User"
        else:
            prefix = "Assistant"
        lines.append(f"{prefix}: {message.content}".strip())

    # Encourage the model to respond in the assistant role
    lines.append("Assistant:")
    return "\n".join(lines)


def _messages_to_chat_messages(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
    """Convert LangChain messages into HuggingFace chat-completion payload."""

    # HuggingFace chat-completion format
    chat_messages = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            chat_messages.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            chat_messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
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
            collected.append(str(content))

    return "".join(collected)


def _extract_chat_completion_text(completion: Any) -> str:
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

    return str(content or "")


def _normalize_provider(provider: LLMProvider | str | None, anthropic_key: str | None) -> LLMProvider:
    """Resolve the configured provider, defaulting based on available credentials."""

    if provider is None:
        return LLMProvider.ANTHROPIC if anthropic_key else LLMProvider.HUGGINGFACE

    if isinstance(provider, str):
        return LLMProvider(provider)

    return provider


def create_chat_model(
    *,
    provider: LLMProvider | str | None,
    model: str,
    anthropic_api_key: str | None,
    huggingface_api_key: str | None,
    huggingface_config: HuggingFaceConfig | None = None,
    stream_callback: Callable[[str], Awaitable[None] | None] | Callable[[str], None] | None = None,
) -> Any:
    """Instantiate a chat-capable model for the requested provider."""

    resolved_provider = _normalize_provider(provider, anthropic_api_key)

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

    if not huggingface_api_key:
        if _allow_offline_fallback():
            logger.warning(
                "HuggingFace API key missing; using LocalLLM fallback for offline execution."
            )
            return LocalLLM("huggingface")

        raise ValueError("HuggingFace API key is required when using the HuggingFace provider")

    config = huggingface_config or HuggingFaceConfig()

    endpoint_url = (config.endpoint_url or "").strip() or None
    default_endpoint = f"https://api-inference.huggingface.co/models/{model}".rstrip("/")
    if endpoint_url and endpoint_url.rstrip("/") == default_endpoint:
        # Treat the default HuggingFace inference endpoint as a standard model lookup
        endpoint_url = None

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
    )


def huggingface_config_from_context(context: ManuscriptContext) -> HuggingFaceConfig:
    """Build HuggingFace configuration from runtime context."""

    return HuggingFaceConfig(
        endpoint_url=getattr(context, "huggingface_endpoint_url", None),
        max_new_tokens=getattr(context, "huggingface_max_new_tokens", 512),
        temperature=getattr(context, "huggingface_temperature", 0.7),
        timeout=getattr(context, "huggingface_timeout", None),
        model_kwargs=getattr(context, "huggingface_model_kwargs", None),
    )


def create_chat_model_from_context(context: ManuscriptContext) -> Any:
    """Convenience helper to create a chat model using context configuration."""

    return create_chat_model(
        provider=getattr(context, "llm_provider", None),
        model=context.model,
        anthropic_api_key=getattr(context, "anthropic_api_key", None),
        huggingface_api_key=getattr(context, "huggingface_api_key", None),
        huggingface_config=huggingface_config_from_context(context),
        stream_callback=getattr(context, "huggingface_stream_callback", None),
    )
logger = logging.getLogger(__name__)


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
        return AIMessage(content="{}")

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<LocalLLM reason={self.reason}>"
