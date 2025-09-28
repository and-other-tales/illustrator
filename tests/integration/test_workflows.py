"""Integration tests for key workflows including endpoint pause handling."""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import WebSocket

from illustrator.web.app import app, connection_manager
from illustrator.llm_factory import (
    HuggingFaceEndpointChatWrapper,
    wait_for_endpoint_restart,
    is_endpoint_paused_error
)
from illustrator.models import LLMProvider
from illustrator.context import ManuscriptContext


class TestEndpointPauseIntegration:
    """Integration tests for endpoint pause handling workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @pytest.mark.asyncio
    async def test_complete_endpoint_pause_workflow(self):
        """Test complete endpoint pause and recovery workflow."""
        session_id = "test_session_123"
        
        # Mock HuggingFace client that will simulate endpoint pause
        mock_client = Mock()
        
        # First call raises pause error, second call succeeds
        pause_error = Exception("Bad Request: The endpoint is paused, ask a maintainer to restart it")
        success_response = [
            {"choices": [{"delta": {"content": "Success"}}]}
        ]
        
        mock_client.chat_completion.side_effect = [pause_error, success_response]
        
        # Create wrapper
        wrapper = HuggingFaceEndpointChatWrapper(
            client=mock_client,
            generation_kwargs={"temperature": 0.7},
            session_id=session_id
        )
        
        # Mock WebSocket connection manager
        mock_connection_manager = Mock()
        mock_connection_manager.active_connections = {session_id: Mock()}
        mock_connection_manager.send_personal_message = AsyncMock()
        
        # Mock the wait function to avoid actual delay
        with patch('illustrator.llm_factory.wait_for_endpoint_restart') as mock_wait, \
             patch('illustrator.web.app.connection_manager', mock_connection_manager), \
             patch.object(wrapper, '_run_endpoint') as mock_run:
            
            # Configure mock_run to first raise pause error, then return success
            from langchain_core.messages import AIMessage, HumanMessage
            mock_run.side_effect = [
                pause_error,
                AIMessage(content="Success after restart")
            ]
            
            # Execute the workflow
            messages = [HumanMessage(content="Test message")]
            result = await wrapper.ainvoke(messages)
            
            # Verify pause handling was triggered
            mock_wait.assert_called_once_with(session_id, countdown_seconds=120)
            
            # Verify success after restart
            assert isinstance(result, AIMessage)
            assert result.content == "Success after restart"
    
    @pytest.mark.asyncio
    async def test_websocket_notifications_during_pause(self):
        """Test WebSocket notifications are sent during endpoint pause."""
        session_id = "test_session_456" 
        
        # Mock connection manager with WebSocket
        mock_websocket = AsyncMock()
        mock_connection_manager = Mock()
        mock_connection_manager.active_connections = {session_id: mock_websocket}
        mock_connection_manager.send_personal_message = AsyncMock()
        
        # Test the wait_for_endpoint_restart function directly
        with patch('illustrator.web.app.connection_manager', mock_connection_manager), \
             patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            
            # Call with short countdown for testing
            await wait_for_endpoint_restart(session_id, countdown_seconds=20)
            
            # Verify WebSocket notifications were sent
            calls = mock_connection_manager.send_personal_message.call_args_list
            assert len(calls) >= 2  # Initial pause + at least one countdown
            
            # Check initial pause notification
            initial_call = calls[0]
            initial_message = json.loads(initial_call[0][0])
            assert initial_message["type"] == "endpoint_paused"
            assert initial_message["countdown_seconds"] == 20
            
            # Check countdown notification
            if len(calls) > 1:
                countdown_call = calls[1] 
                countdown_message = json.loads(countdown_call[0][0])
                assert countdown_message["type"] == "endpoint_countdown"
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_pause_handling(self):
        """Test handling multiple concurrent endpoint pause scenarios."""
        session_ids = ["session_1", "session_2", "session_3"]
        
        # Mock connection manager with multiple sessions
        mock_connection_manager = Mock()
        mock_connection_manager.active_connections = {
            sid: AsyncMock() for sid in session_ids
        }
        mock_connection_manager.send_personal_message = AsyncMock()
        
        with patch('illustrator.web.app.connection_manager', mock_connection_manager), \
             patch('asyncio.sleep', new_callable=AsyncMock):
            
            # Start multiple pause handlers concurrently
            tasks = [
                wait_for_endpoint_restart(sid, countdown_seconds=10)
                for sid in session_ids
            ]
            
            await asyncio.gather(*tasks)
            
            # Each session should have received notifications
            calls_per_session = {}
            for call in mock_connection_manager.send_personal_message.call_args_list:
                session_id = call[0][1]  # Second argument is session_id
                calls_per_session.setdefault(session_id, 0)
                calls_per_session[session_id] += 1
            
            # Each session should have received at least one notification
            for sid in session_ids:
                assert calls_per_session.get(sid, 0) >= 1


class TestWebSocketProcessingIntegration:
    """Integration tests for WebSocket processing workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
    
    @pytest.mark.asyncio
    async def test_processing_session_lifecycle(self):
        """Test complete processing session lifecycle via WebSocket."""
        session_id = "integration_test_session"
        manuscript_id = "test_manuscript_123"
        
        # Mock connection manager for testing
        mock_websocket = AsyncMock()
        
        # Test connection establishment
        await connection_manager.connect(mock_websocket, session_id)
        assert session_id in connection_manager.active_connections
        
        # Simulate processing workflow messages
        messages = [
            json.dumps({"type": "progress", "progress": 10, "message": "Starting analysis"}),
            json.dumps({"type": "log", "level": "info", "message": "Analyzing chapter 1"}),
            json.dumps({"type": "progress", "progress": 50, "message": "Generating images"}),
            json.dumps({"type": "image", "url": "/images/test.png", "prompt": "A scene"}),
            json.dumps({"type": "progress", "progress": 100, "message": "Complete"})
        ]
        
        # Send all messages
        for message in messages:
            await connection_manager.send_personal_message(message, session_id)
        
        # Verify all messages were sent
        assert mock_websocket.send_text.call_count == len(messages)
        
        # Test disconnection
        connection_manager.disconnect(session_id)
        assert session_id not in connection_manager.active_connections
    
    @pytest.mark.asyncio 
    async def test_session_pause_resume_integration(self):
        """Test session pause and resume integration."""
        session_id = "pause_test_session"
        
        # Setup session with mock WebSocket
        mock_websocket = AsyncMock()
        await connection_manager.connect(mock_websocket, session_id)
        
        # Create session data
        from illustrator.web.models.web_models import ProcessingSessionData, ProcessingStatus
        session_data = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="processing", message="Processing chapter 1")
        )
        connection_manager.sessions[session_id] = session_data
        
        # Test pause request
        session_data.pause_requested = True
        session_data.status.status = "paused"
        
        pause_message = json.dumps({
            "type": "log",
            "level": "warning", 
            "message": "Processing paused at user request"
        })
        
        await connection_manager.send_personal_message(pause_message, session_id)
        
        # Verify pause state
        assert session_data.pause_requested is True
        assert session_data.status.status == "paused"
        
        # Test resume
        session_data.pause_requested = False
        session_data.status.status = "processing"
        
        resume_message = json.dumps({
            "type": "log",
            "level": "success",
            "message": "Processing resumed"
        })
        
        await connection_manager.send_personal_message(resume_message, session_id)
        
        # Verify resume state
        assert session_data.pause_requested is False
        assert session_data.status.status == "processing"
        
        # Cleanup
        connection_manager.cleanup_session(session_id)
        assert session_id not in connection_manager.sessions


class TestLLMFactoryIntegration:
    """Integration tests for LLM factory with different providers."""
    
    @pytest.mark.asyncio
    async def test_create_chat_model_integration(self):
        """Test creating chat models with different configurations."""
        from illustrator.llm_factory import create_chat_model
        
        # Test with mock context
        mock_context = Mock()
        mock_context.llm_provider = LLMProvider.HUGGINGFACE
        mock_context.model = "microsoft/DialoGPT-medium"
        mock_context.anthropic_api_key = None
        mock_context.huggingface_api_key = "test_key"
        mock_context.huggingface_endpoint_url = "https://api-inference.huggingface.co/models/test"
        mock_context.huggingface_max_new_tokens = 512
        mock_context.huggingface_temperature = 0.7
        mock_context.huggingface_timeout = None
        mock_context.huggingface_model_kwargs = None
        mock_context.huggingface_stream_callback = None
        
        # Mock HuggingFace components
        with patch('illustrator.llm_factory.InferenceClient') as mock_client_class, \
             patch('illustrator.llm_factory.HuggingFaceEndpointChatWrapper') as mock_wrapper_class:
            
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            mock_wrapper = Mock()
            mock_wrapper_class.return_value = mock_wrapper
            
            from illustrator.llm_factory import HuggingFaceConfig
            config = HuggingFaceConfig(
                endpoint_url=mock_context.huggingface_endpoint_url,
                max_new_tokens=mock_context.huggingface_max_new_tokens,
                temperature=mock_context.huggingface_temperature
            )
            
            result = create_chat_model(
                provider=LLMProvider.HUGGINGFACE,
                model=mock_context.model,
                anthropic_api_key=mock_context.anthropic_api_key,
                huggingface_api_key=mock_context.huggingface_api_key,
                huggingface_config=config,
                session_id="test_session"
            )
            
            # Verify HuggingFace model was created
            assert result is mock_wrapper
            mock_client_class.assert_called_once()
            mock_wrapper_class.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_chat_model_from_context_integration(self):
        """Test creating chat model from context with session ID."""
        from illustrator.llm_factory import create_chat_model_from_context
        
        # Create mock context
        mock_context = Mock()
        mock_context.llm_provider = LLMProvider.HUGGINGFACE
        mock_context.model = "test-model"
        mock_context.anthropic_api_key = None
        mock_context.huggingface_api_key = "hf_test_key"
        mock_context.huggingface_stream_callback = None
        
        with patch('illustrator.llm_factory.create_chat_model') as mock_create, \
             patch('illustrator.llm_factory.huggingface_config_from_context') as mock_config:
            
            mock_model = Mock()
            mock_create.return_value = mock_model
            mock_hf_config = Mock() 
            mock_config.return_value = mock_hf_config
            
            result = create_chat_model_from_context(mock_context, session_id="integration_test")
            
            # Verify session_id was passed through
            mock_create.assert_called_once_with(
                provider=LLMProvider.HUGGINGFACE,
                model="test-model",
                anthropic_api_key=None,
                huggingface_api_key="hf_test_key",
                huggingface_config=mock_hf_config,
                stream_callback=None,
                session_id="integration_test"
            )
            
            assert result is mock_model


class TestCharacterTrackingIntegration:
    """Integration tests for character tracking workflows."""
    
    @pytest.mark.asyncio
    async def test_character_tracking_workflow(self):
        """Test complete character tracking workflow across multiple chapters."""
        from illustrator.character_tracking import CharacterTracker
        from illustrator.models import Chapter
        
        # Mock LLM for character analysis
        mock_llm = AsyncMock()
        mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content='{"appearance": "tall with dark hair", "personality": ["brave", "kind"]}'
        ))
        
        tracker = CharacterTracker(llm=mock_llm)
        
        # Create test chapters
        chapter1 = Chapter(
            number=1,
            title="The Beginning",
            content="John walked into the room with confidence. He was a tall man with dark hair."
        )
        
        chapter2 = Chapter(
            number=2,
            title="The Journey",
            content="John showed his brave nature as he faced the challenge."
        )
        
        # Mock the character analysis method
        with patch.object(tracker, '_analyze_character_in_depth', return_value={
            "appearance": "tall with dark hair",
            "personality": ["brave", "confident"],
            "relationships": []
        }) as mock_analyze:
            
            # Create character profile from first chapter
            profile = await tracker._create_character_profile("John", chapter1)
            tracker.characters["John"] = profile
            
            # Verify character was analyzed
            mock_analyze.assert_called_once()
            assert profile.name == "John"
            assert len(profile.appearances) == 1
        
        # Update character from second chapter
        character_data = {
            "appearance": "tall with dark hair, confident posture",
            "personality": ["brave", "confident", "determined"]
        }
        
        tracker._update_character_profile_sync(
            "John",
            character_data,
            "chapter_2",
            "Facing the challenge"
        )
        
        # Verify character was updated across chapters
        updated_profile = tracker.characters["John"]
        assert len(updated_profile.appearances) == 2
        assert updated_profile.appearances[0].chapter_id == "chapter_1"
        assert updated_profile.appearances[1].chapter_id == "chapter_2"
        assert "determined" in updated_profile.personality_traits


class TestErrorHandlingIntegration:
    """Integration tests for error handling across modules."""
    
    @pytest.mark.asyncio
    async def test_endpoint_error_detection_integration(self):
        """Test endpoint error detection across different error formats."""
        # Test various error message formats
        error_cases = [
            Exception("Bad Request: The endpoint is paused, ask a maintainer to restart it"),
            Exception("HTTP 400: endpoint is paused"),
            Exception("Service Unavailable: Ask a maintainer to restart the endpoint"),
            "The endpoint is currently paused",
            "ENDPOINT IS PAUSED - contact maintainer",
        ]
        
        for error in error_cases:
            assert is_endpoint_paused_error(error) is True
        
        # Test non-pause errors
        non_pause_errors = [
            Exception("Connection timeout"),
            Exception("Invalid API key"),
            Exception("Rate limit exceeded"),
            "Model not found",
            "Authentication failed"
        ]
        
        for error in non_pause_errors:
            assert is_endpoint_paused_error(error) is False
    
    @pytest.mark.asyncio
    async def test_websocket_error_recovery_integration(self):
        """Test WebSocket error recovery in processing workflow."""
        session_id = "error_test_session"
        
        # Mock WebSocket that will fail on send
        mock_websocket = AsyncMock()
        mock_websocket.send_text.side_effect = Exception("Connection lost")
        
        await connection_manager.connect(mock_websocket, session_id)
        
        # Create session
        from illustrator.web.models.web_models import ProcessingSessionData, ProcessingStatus
        session_data = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="processing", message="Processing")
        )
        connection_manager.sessions[session_id] = session_data
        
        # Try to send message - should handle error gracefully
        test_message = json.dumps({"type": "log", "message": "Test message"})
        
        # Should not raise exception
        await connection_manager.send_personal_message(test_message, session_id)
        
        # Session should still exist for potential reconnection
        assert session_id in connection_manager.sessions
        
        # Log should still be recorded despite WebSocket failure
        assert len(session_data.logs) == 1
        assert session_data.logs[0].message == test_message
        
        # Cleanup
        connection_manager.cleanup_session(session_id)


class TestFullWorkflowIntegration:
    """Integration tests for complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    async def test_complete_processing_workflow_with_pause(self):
        """Test complete processing workflow including pause/resume and endpoint handling."""
        session_id = "full_workflow_test"
        
        # Setup WebSocket connection
        mock_websocket = AsyncMock()
        await connection_manager.connect(mock_websocket, session_id)
        
        # Create processing session
        from illustrator.web.models.web_models import ProcessingSessionData, ProcessingStatus
        session_data = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="manuscript_123",
            status=ProcessingStatus(status="started", message="Starting processing")
        )
        connection_manager.sessions[session_id] = session_data
        
        # Simulate processing steps
        steps = [
            ("progress", 10, "Initializing"),
            ("log", "info", "Loading manuscript"),
            ("progress", 25, "Analyzing chapters"),
            ("endpoint_paused", 120, "Endpoint paused"),
            ("endpoint_countdown", 110, "Retrying in 110 seconds"),
            ("log", "info", "Endpoint restored"),
            ("progress", 75, "Generating images"),
            ("image", "/images/scene1.png", "A beautiful scene"),
            ("progress", 100, "Processing complete")
        ]
        
        # Send each step
        for step in steps:
            if step[0] == "progress":
                message = json.dumps({
                    "type": "progress",
                    "progress": step[1],
                    "message": step[2]
                })
                session_data.status.progress = step[1]
                session_data.status.message = step[2]
                
            elif step[0] == "log":
                message = json.dumps({
                    "type": "log",
                    "level": step[1],
                    "message": step[2]
                })
                
            elif step[0] == "endpoint_paused":
                message = json.dumps({
                    "type": "endpoint_paused",
                    "countdown_seconds": step[1],
                    "message": "Endpoint paused, waiting for restart"
                })
                
            elif step[0] == "endpoint_countdown":
                message = json.dumps({
                    "type": "endpoint_countdown",
                    "countdown_seconds": step[1],
                    "message": step[2]
                })
                
            elif step[0] == "image":
                message = json.dumps({
                    "type": "image",
                    "url": step[1],
                    "prompt": step[2]
                })
                connection_manager.add_image_entry(session_id, step[1], step[2])
            
            await connection_manager.send_personal_message(message, session_id)
        
        # Verify workflow completion
        assert mock_websocket.send_text.call_count == len(steps)
        assert session_data.status.progress == 100
        assert len(session_data.logs) == len(steps)
        assert len(session_data.images) == 1
        
        # Cleanup
        connection_manager.cleanup_session(session_id)
        assert session_id not in connection_manager.sessions