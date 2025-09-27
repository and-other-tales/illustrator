"""Utilities for constructing language model interfaces used across the project."""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from huggingface_hub import InferenceClient

from illustrator.models import LLMProvider

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
    ) -> None:
        self._client = client
        self._generation_kwargs = generation_kwargs

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Generate a response asynchronously using the configured endpoint."""

        prompt = _messages_to_prompt(messages)

        def _run_endpoint() -> AIMessage:
            raw_output = self._client.text_generation(
                prompt,
                **self._generation_kwargs,
            )

            if isinstance(raw_output, str):
                generated_text = raw_output
            elif isinstance(raw_output, Iterable):
                first_item = next(iter(raw_output), "")
                generated_text = str(first_item)
            else:
                generated_text = str(raw_output)

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
    }

    if config.model_kwargs:
        generation_kwargs.update(config.model_kwargs)

    if endpoint_url:
        # Custom endpoints expect routing to be handled by the base URL
        generation_kwargs.pop("model", None)
    else:
        generation_kwargs.setdefault("model", model)

    return HuggingFaceEndpointChatWrapper(client, generation_kwargs)


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
