"""Unit tests for llm_factory module including endpoint pause handling."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Import the functions we need to test
from illustrator.llm_factory import (
    EndpointPausedError,
    is_endpoint_paused_error,
    wait_for_endpoint_restart,
    HuggingFaceEndpointChatWrapper,
    HuggingFaceConfig,
    create_chat_model,
    create_chat_model_from_context,
    huggingface_config_from_context,
    _normalize_provider,
    _messages_to_prompt,
    _messages_to_chat_messages,
    _allow_offline_fallback,
)
from illustrator.models import LLMProvider


class TestEndpointPausedError:
    """Test EndpointPausedError exception class."""
    
    def test_init_with_default_message(self):
        """Test EndpointPausedError with default message."""
        error = EndpointPausedError()
        assert error.message == "The endpoint is paused, ask a maintainer to restart it"
        assert str(error) == "The endpoint is paused, ask a maintainer to restart it"
    
    def test_init_with_custom_message(self):
        """Test EndpointPausedError with custom message."""
        custom_msg = "Custom endpoint pause message"
        error = EndpointPausedError(custom_msg)
        assert error.message == custom_msg
        assert str(error) == custom_msg


class TestIsEndpointPausedError:
    """Test is_endpoint_paused_error function."""
    
    def test_detects_pause_from_string(self):
        """Test detection of paused endpoint from string."""
        pause_messages = [
            "Bad Request: The endpoint is paused, ask a maintainer to restart it",
            "Error: endpoint is paused",
            "The endpoint is PAUSED and needs restart",
            "Please ask a maintainer to restart the service",
        ]
        
        for msg in pause_messages:
            assert is_endpoint_paused_error(msg) is True
    
    def test_detects_pause_from_exception(self):
        """Test detection of paused endpoint from Exception object."""
        pause_exception = Exception("Bad Request: The endpoint is paused, ask a maintainer to restart it")
        assert is_endpoint_paused_error(pause_exception) is True
        
        normal_exception = Exception("Some other error occurred")
        assert is_endpoint_paused_error(normal_exception) is False
    
    def test_ignores_normal_errors(self):
        """Test that normal errors are not detected as pause errors."""
        normal_messages = [
            "Connection timeout",
            "Invalid API key", 
            "Model not found",
            "Rate limit exceeded",
            "",
            None
        ]
        
        for msg in normal_messages:
            if msg is not None:
                assert is_endpoint_paused_error(msg) is False
    
    def test_handles_empty_input(self):
        """Test handling of empty/None input."""
        assert is_endpoint_paused_error("") is False
        assert is_endpoint_paused_error(None) is False
    
    def test_case_insensitive(self):
        """Test case insensitive detection."""
        assert is_endpoint_paused_error("ENDPOINT IS PAUSED") is True
        assert is_endpoint_paused_error("endpoint is paused") is True
        assert is_endpoint_paused_error("Endpoint Is Paused") is True


class TestWaitForEndpointRestart:
    """Test wait_for_endpoint_restart function."""
    
    @pytest.mark.asyncio
    async def test_wait_without_websocket(self):
        """Test wait function without WebSocket notification."""
        # Mock asyncio.sleep to avoid actual waiting
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            await wait_for_endpoint_restart(session_id=None, countdown_seconds=20)
            
            # Should sleep twice: once for 10 seconds, once for remaining 10
            assert mock_sleep.call_count == 2
            mock_sleep.assert_any_call(10)
    
    @pytest.mark.asyncio
    async def test_wait_with_websocket_notifications(self):
        """Test wait function with WebSocket notifications."""
        mock_connection_manager = Mock()
        mock_connection_manager.active_connections = {"test_session": Mock()}
        mock_connection_manager.send_personal_message = AsyncMock()
        
        with patch('asyncio.sleep', new_callable=AsyncMock), \
             patch('illustrator.web.app.connection_manager', mock_connection_manager):
            
            await wait_for_endpoint_restart(session_id="test_session", countdown_seconds=20)
            
            # Should send initial pause notification and countdown updates
            calls = mock_connection_manager.send_personal_message.call_args_list
            assert len(calls) >= 2
            
            # Check initial pause notification
            initial_call = calls[0]
            message_data = json.loads(initial_call[0][0])
            assert message_data["type"] == "endpoint_paused"
            assert message_data["countdown_seconds"] == 20
    
    @pytest.mark.asyncio
    async def test_wait_with_invalid_session(self):
        """Test wait function with invalid session ID."""
        mock_connection_manager = Mock()
        mock_connection_manager.active_connections = {}
        mock_connection_manager.send_personal_message = AsyncMock()
        
        with patch('asyncio.sleep', new_callable=AsyncMock), \
             patch('illustrator.web.app.connection_manager', mock_connection_manager):
            
            await wait_for_endpoint_restart(session_id="invalid_session", countdown_seconds=10)
            
            # Should not try to send messages
            assert mock_connection_manager.send_personal_message.call_count == 0


class TestHuggingFaceConfig:
    """Test HuggingFaceConfig dataclass."""
    
    def test_default_values(self):
        """Test HuggingFaceConfig default values."""
        config = HuggingFaceConfig()
        assert config.endpoint_url is None
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.timeout is None
        assert config.model_kwargs is None
    
    def test_custom_values(self):
        """Test HuggingFaceConfig with custom values."""
        config = HuggingFaceConfig(
            endpoint_url="https://api.example.com",
            max_new_tokens=1024,
            temperature=0.8,
            timeout=30.0,
            model_kwargs={"do_sample": True}
        )
        assert config.endpoint_url == "https://api.example.com"
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.8
        assert config.timeout == 30.0
        assert config.model_kwargs == {"do_sample": True}


class TestHuggingFaceEndpointChatWrapper:
    """Test HuggingFaceEndpointChatWrapper class."""
    
    def test_init(self):
        """Test wrapper initialization."""
        mock_client = Mock()
        generation_kwargs = {"temperature": 0.7, "max_new_tokens": 512}
        
        wrapper = HuggingFaceEndpointChatWrapper(
            client=mock_client,
            generation_kwargs=generation_kwargs,
            session_id="test_session"
        )
        
        assert wrapper._client is mock_client
        assert wrapper._session_id == "test_session"
        assert wrapper._generation_kwargs == generation_kwargs
    
    @pytest.mark.asyncio
    async def test_ainvoke_successful_response(self):
        """Test successful ainvoke call."""
        mock_client = Mock()
        mock_client.chat_completion.return_value = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]}
        ]
        
        wrapper = HuggingFaceEndpointChatWrapper(
            client=mock_client,
            generation_kwargs={"temperature": 0.7}
        )
        
        messages = [HumanMessage(content="Hi")]
        
        # Mock client methods with proper chat completion structure
        mock_response = {
            "choices": [{"message": {"content": "Hello world"}}]
        }
        mock_client.chat_completion.return_value = mock_response
            
        result = await wrapper.ainvoke(messages)
        
        assert isinstance(result, AIMessage)
        assert result.content == "Hello world"
    
    @pytest.mark.asyncio 
    async def test_ainvoke_endpoint_pause_handling(self):
        """Test ainvoke handling of endpoint pause errors."""
        mock_client = Mock()
        
        # First call raises pause error, second call succeeds
        pause_error = Exception("Bad Request: The endpoint is paused, ask a maintainer to restart it")
        mock_client.chat_completion.side_effect = [
            pause_error,
            [{"choices": [{"delta": {"content": "Success after restart"}}]}]
        ]
        
        wrapper = HuggingFaceEndpointChatWrapper(
            client=mock_client,
            generation_kwargs={"temperature": 0.7},
            session_id="test_session"
        )
        
        messages = [HumanMessage(content="Test")]
        
        with patch('illustrator.llm_factory.wait_for_endpoint_restart', new_callable=AsyncMock) as mock_wait:
            # Mock client method to raise pause error first, then succeed
            success_response = {
                "choices": [{"message": {"content": "Success after restart"}}]
            }
            mock_client.chat_completion.side_effect = [
                pause_error,
                success_response
            ]
            
            result = await wrapper.ainvoke(messages)
            
            # Should have called wait_for_endpoint_restart
            mock_wait.assert_called_once_with("test_session", countdown_seconds=120)
            
            # Should return success result after retry
            assert isinstance(result, AIMessage)
            assert result.content == "Success after restart"


class TestMessageConversion:
    """Test message conversion utility functions."""
    
    def test_messages_to_prompt(self):
        """Test converting messages to prompt string."""
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi there!"),
            HumanMessage(content="How are you?")
        ]
        
        prompt = _messages_to_prompt(messages)
        
        assert "System: You are a helpful assistant." in prompt
        assert "User: Hello!" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "Human: How are you?" in prompt
    
    def test_messages_to_chat_messages(self):
        """Test converting messages to chat format."""
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Hello!"),
            AIMessage(content="Hi!")
        ]
        
        chat_messages = _messages_to_chat_messages(messages)
        
        expected = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi!"}
        ]
        
        assert chat_messages == expected


class TestProviderNormalization:
    """Test provider normalization function."""
    
    def test_normalize_provider_with_string(self):
        """Test normalizing string provider."""
        assert _normalize_provider("anthropic", "key") == LLMProvider.ANTHROPIC
        assert _normalize_provider("huggingface", None) == LLMProvider.HUGGINGFACE
        assert _normalize_provider("ANTHROPIC", "key") == LLMProvider.ANTHROPIC
    
    def test_normalize_provider_with_enum(self):
        """Test normalizing LLMProvider enum."""
        assert _normalize_provider(LLMProvider.ANTHROPIC, "key") == LLMProvider.ANTHROPIC
        assert _normalize_provider(LLMProvider.HUGGINGFACE, None) == LLMProvider.HUGGINGFACE
    
    def test_normalize_provider_fallback_logic(self):
        """Test provider fallback logic based on API key."""
        # Should fall back to ANTHROPIC if key is provided
        assert _normalize_provider(None, "anthropic_key") == LLMProvider.ANTHROPIC
        
        # Should fall back to HUGGINGFACE if no anthropic key
        assert _normalize_provider(None, None) == LLMProvider.HUGGINGFACE
    
    def test_normalize_provider_invalid_string(self):
        """Test handling of invalid provider string."""
        with pytest.raises(ValueError, match="Unknown provider"):
            _normalize_provider("invalid_provider", None)


class TestCreateChatModel:
    """Test create_chat_model function."""
    
    @patch('illustrator.llm_factory.init_chat_model')
    def test_create_anthropic_model(self, mock_init_chat_model):
        """Test creating Anthropic chat model."""
        mock_model = Mock()
        mock_init_chat_model.return_value = mock_model
        
        result = create_chat_model(
            provider=LLMProvider.ANTHROPIC,
            model="claude-3-sonnet-20240229",
            anthropic_api_key="test_key",
            huggingface_api_key=None
        )
        
        assert result is mock_model
        mock_init_chat_model.assert_called_once_with(
            "anthropic/claude-3-sonnet-20240229",
            api_key="test_key"
        )
    
    def test_create_huggingface_model(self):
        """Test creating HuggingFace chat model."""
        with patch('illustrator.llm_factory.InferenceClient') as mock_client_class, \
             patch('illustrator.llm_factory.HuggingFaceEndpointChatWrapper') as mock_wrapper_class:
            
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_wrapper = Mock()
            mock_wrapper_class.return_value = mock_wrapper
            
            config = HuggingFaceConfig(
                endpoint_url="https://api-inference.huggingface.co/models/test",
                max_new_tokens=512,
                temperature=0.7
            )
            
            result = create_chat_model(
                provider=LLMProvider.HUGGINGFACE,
                model="test-model",
                anthropic_api_key=None,
                huggingface_api_key="hf_key",
                huggingface_config=config,
                session_id="test_session"
            )
            
            assert result is mock_wrapper
            mock_client_class.assert_called_once()
            mock_wrapper_class.assert_called_once()


class TestContextHelpers:
    """Test context helper functions."""
    
    def test_huggingface_config_from_context(self):
        """Test creating HuggingFace config from context."""
        # Mock context object
        mock_context = Mock()
        mock_context.huggingface_endpoint_url = "https://api.example.com"
        mock_context.huggingface_max_new_tokens = 1024
        mock_context.huggingface_temperature = 0.8
        mock_context.huggingface_timeout = 30.0
        mock_context.huggingface_model_kwargs = {"do_sample": True}
        
        config = huggingface_config_from_context(mock_context)
        
        assert config.endpoint_url == "https://api.example.com"
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.8
        assert config.timeout == 30.0
        assert config.model_kwargs == {"do_sample": True}
    
    def test_create_chat_model_from_context(self):
        """Test creating chat model from context."""
        mock_context = Mock()
        mock_context.llm_provider = LLMProvider.ANTHROPIC
        mock_context.model = "claude-3-sonnet"
        mock_context.anthropic_api_key = "test_key"
        mock_context.huggingface_api_key = None
        mock_context.huggingface_stream_callback = None
        
        with patch('illustrator.llm_factory.create_chat_model') as mock_create, \
             patch('illustrator.llm_factory.huggingface_config_from_context') as mock_config:
            
            mock_model = Mock()
            mock_create.return_value = mock_model
            mock_hf_config = Mock()
            mock_config.return_value = mock_hf_config
            
            result = create_chat_model_from_context(mock_context, session_id="test_session")
            
            assert result is mock_model
            mock_create.assert_called_once_with(
                provider=LLMProvider.ANTHROPIC,
                model="claude-3-sonnet",
                anthropic_api_key="test_key",
                huggingface_api_key=None,
                huggingface_config=mock_hf_config,
                stream_callback=None,
                session_id="test_session"
            )


class TestOfflineFallback:
    """Test offline fallback functionality."""
    
    @patch.dict('os.environ', {'ILLUSTRATOR_ALLOW_OFFLINE': 'true'})
    def test_allow_offline_fallback_enabled(self):
        """Test offline fallback when enabled."""
        assert _allow_offline_fallback() is True
    
    @patch.dict('os.environ', {'ILLUSTRATOR_ALLOW_OFFLINE': 'false'})
    def test_allow_offline_fallback_disabled(self):
        """Test offline fallback when disabled."""
        assert _allow_offline_fallback() is False
    
    @patch.dict('os.environ', {}, clear=True)
    def test_allow_offline_fallback_default(self):
        """Test offline fallback default value."""
        assert _allow_offline_fallback() is False