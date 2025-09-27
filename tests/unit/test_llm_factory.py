"""Tests for llm_factory offline fallbacks."""

import pytest

from illustrator.llm_factory import LocalLLM, create_chat_model
from illustrator.models import LLMProvider


def test_create_chat_model_offline_fallback(monkeypatch: pytest.MonkeyPatch):
    """Missing credentials should return a local stub when offline mode is enabled."""
    monkeypatch.setenv("ILLUSTRATOR_OFFLINE_MODE", "1")
    monkeypatch.delenv("ILLUSTRATOR_ENFORCE_REMOTE", raising=False)

    model = create_chat_model(
        provider=LLMProvider.HUGGINGFACE,
        model="stub-model",
        anthropic_api_key=None,
        huggingface_api_key=None,
    )

    assert isinstance(model, LocalLLM)


def test_create_chat_model_enforce_remote(monkeypatch: pytest.MonkeyPatch):
    """When remote execution is enforced, missing credentials raise errors."""
    monkeypatch.setenv("ILLUSTRATOR_ENFORCE_REMOTE", "1")
    monkeypatch.delenv("ILLUSTRATOR_OFFLINE_MODE", raising=False)

    with pytest.raises(ValueError, match="Anthropic API key is required"):
        create_chat_model(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet",
            anthropic_api_key=None,
            huggingface_api_key=None,
        )
