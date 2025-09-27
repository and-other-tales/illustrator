"""Utilities for constructing language model interfaces used across the project."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

try:
    from langchain_huggingface import HuggingFacePipeline
except Exception as exc:  # pragma: no cover - defensive path when optional dependency missing
    raise RuntimeError(
        "langchain-huggingface is required for local NLP processing. Install with 'pip install langchain-huggingface'."
    ) from exc

from transformers import pipeline as hf_pipeline

from illustrator.models import LLMProvider

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from illustrator.context import ManuscriptContext


@dataclass(slots=True)
class HuggingFaceConfig:
    """Configuration for constructing HuggingFace pipelines."""

    task: str = "text-generation"
    device: str | int | None = None
    max_new_tokens: int = 512
    temperature: float = 0.7
    model_kwargs: dict[str, Any] | None = None


class HuggingFaceChatWrapper:
    """Async-friendly wrapper that emulates LangChain chat model behaviour for pipelines."""

    def __init__(self, pipeline: HuggingFacePipeline, generation_kwargs: dict[str, Any]):
        self._pipeline = pipeline
        self._generation_kwargs = generation_kwargs

    async def ainvoke(self, messages: Sequence[BaseMessage]) -> AIMessage:
        """Generate a response asynchronously using the configured pipeline."""

        prompt = _messages_to_prompt(messages)

        def _run_pipeline() -> AIMessage:
            raw_output = self._pipeline.invoke(prompt, **self._generation_kwargs)

            # HuggingFacePipeline returns either a string or list/dict depending on pipeline configuration
            if isinstance(raw_output, str):
                generated_text = raw_output
            elif isinstance(raw_output, Iterable):
                first_item = next(iter(raw_output), "")
                if isinstance(first_item, dict):
                    generated_text = (
                        first_item.get("generated_text")
                        or first_item.get("summary_text")
                        or ""
                    )
                else:
                    generated_text = str(first_item)
            else:
                generated_text = str(raw_output)

            # Some pipelines include the original prompt; strip it if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):]

            return AIMessage(content=generated_text.strip())

        return await asyncio.to_thread(_run_pipeline)


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
    huggingface_config: HuggingFaceConfig | None = None,
) -> Any:
    """Instantiate a chat-capable model for the requested provider."""

    resolved_provider = _normalize_provider(provider, anthropic_api_key)

    if resolved_provider == LLMProvider.ANTHROPIC:
        if not anthropic_api_key:
            raise ValueError("Anthropic API key is required when using the Anthropic provider")

        # LangChain expects bare model name without provider prefix
        normalized_model = model.split("/", 1)[-1]
        return init_chat_model(
            model=normalized_model,
            model_provider="anthropic",
            api_key=anthropic_api_key,
        )

    # Local HuggingFace pipeline configuration
    config = huggingface_config or HuggingFaceConfig()

    pipeline_kwargs: dict[str, Any] = {}
    if config.device == "auto":
        pipeline_kwargs["device_map"] = "auto"
    elif config.device not in (None, "cpu"):
        pipeline_kwargs["device"] = config.device
    elif config.device == "cpu":
        pipeline_kwargs["device"] = -1

    generator = hf_pipeline(
        task=config.task,
        model=model,
        **pipeline_kwargs,
    )

    if config.device == "cpu" or config.device is None:
        try:
            generator.model.to("cpu")
        except Exception:
            pass  # Model already on CPU or move unsupported

    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "do_sample": config.temperature > 0,
        "return_full_text": False,
    }
    if config.model_kwargs:
        generation_kwargs.update(config.model_kwargs)

    pipeline_wrapper = HuggingFacePipeline(pipeline=generator)
    return HuggingFaceChatWrapper(pipeline_wrapper, generation_kwargs)


def huggingface_config_from_context(context: ManuscriptContext) -> HuggingFaceConfig:
    """Build HuggingFace configuration from runtime context."""

    return HuggingFaceConfig(
        task=getattr(context, "huggingface_task", "text-generation"),
        device=getattr(context, "huggingface_device", None),
        max_new_tokens=getattr(context, "huggingface_max_new_tokens", 512),
        temperature=getattr(context, "huggingface_temperature", 0.7),
        model_kwargs=getattr(context, "huggingface_model_kwargs", None),
    )


def create_chat_model_from_context(context: ManuscriptContext) -> Any:
    """Convenience helper to create a chat model using context configuration."""

    return create_chat_model(
        provider=getattr(context, "llm_provider", None),
        model=context.model,
        anthropic_api_key=getattr(context, "anthropic_api_key", None),
        huggingface_config=huggingface_config_from_context(context),
    )
*** End of File
