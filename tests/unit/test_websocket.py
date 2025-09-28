"""Unit tests for web app WebSocket functionality and session management."""

import json
import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
from fastapi import WebSocket
from fastapi.testclient import TestClient

from illustrator.web.models.web_models import (
    ConnectionManager,
    ProcessingStatus,
    ProcessingSessionData,
    ProcessingLogEntry,
    ProcessingImageEntry
)


class TestConnectionManager:
    """Test ConnectionManager class for WebSocket handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.connection_manager = ConnectionManager()
    
    @pytest.mark.asyncio
    async def test_connect_websocket(self):
        """Test WebSocket connection."""
        mock_websocket = AsyncMock(spec=WebSocket)
        session_id = "test_session_123"
        
        await self.connection_manager.connect(mock_websocket, session_id)
        
        mock_websocket.accept.assert_called_once()
        assert session_id in self.connection_manager.active_connections
        assert self.connection_manager.active_connections[session_id] is mock_websocket
    
    def test_disconnect_websocket(self):
        """Test WebSocket disconnection."""
        mock_websocket = Mock()
        session_id = "test_session_123"
        
        # Add connection first
        self.connection_manager.active_connections[session_id] = mock_websocket
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="connected", message="Connected")
        )
        
        # Disconnect
        self.connection_manager.disconnect(session_id)
        
        # WebSocket connection should be removed, but session data preserved
        assert session_id not in self.connection_manager.active_connections
        assert session_id in self.connection_manager.sessions
    
    def test_cleanup_session(self):
        """Test complete session cleanup."""
        mock_websocket = Mock()
        session_id = "test_session_123"
        
        # Add connection and session
        self.connection_manager.active_connections[session_id] = mock_websocket
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="connected", message="Connected")
        )
        
        # Cleanup
        self.connection_manager.cleanup_session(session_id)
        
        # Both connection and session should be removed
        assert session_id not in self.connection_manager.active_connections
        assert session_id not in self.connection_manager.sessions
    
    def test_add_log_entry(self):
        """Test adding log entries to session."""
        session_id = "test_session_123"
        
        # Create session
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="connected", message="Connected")
        )
        
        # Add log entries
        self.connection_manager.add_log_entry(session_id, "info", "Processing started")
        self.connection_manager.add_log_entry(session_id, "error", "An error occurred")
        
        session = self.connection_manager.sessions[session_id]
        assert len(session.logs) == 2
        assert session.logs[0].level == "info"
        assert session.logs[0].message == "Processing started"
        assert session.logs[1].level == "error"
        assert session.logs[1].message == "An error occurred"
    
    def test_add_image_entry(self):
        """Test adding image entries to session."""
        session_id = "test_session_123"
        
        # Create session
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="connected", message="Connected")
        )
        
        # Add image entries
        self.connection_manager.add_image_entry(
            session_id, 
            "/images/test1.png", 
            "A beautiful landscape",
            chapter_number=1,
            scene_number=1
        )
        
        session = self.connection_manager.sessions[session_id]
        assert len(session.images) == 1
        assert session.images[0].url == "/images/test1.png"
        assert session.images[0].prompt == "A beautiful landscape"
        assert session.images[0].chapter_number == 1
    
    @pytest.mark.asyncio
    async def test_send_personal_message(self):
        """Test sending personal messages via WebSocket."""
        mock_websocket = AsyncMock(spec=WebSocket)
        session_id = "test_session_123"
        
        # Setup connection and session
        self.connection_manager.active_connections[session_id] = mock_websocket
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="connected", message="Connected")
        )
        
        message = "Test message"
        await self.connection_manager.send_personal_message(message, session_id)
        
        mock_websocket.send_text.assert_called_once_with(message)
        
        # Should also add log entry
        session = self.connection_manager.sessions[session_id]
        assert len(session.logs) == 1
        assert session.logs[0].message == message
    
    @pytest.mark.asyncio
    async def test_send_message_to_disconnected_session(self):
        """Test sending message when WebSocket is disconnected."""
        session_id = "test_session_123"
        
        # Create session but no active connection
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="disconnected", message="Disconnected")
        )
        
        message = "Test message"
        await self.connection_manager.send_personal_message(message, session_id)
        
        # Should still add log entry even without active connection
        session = self.connection_manager.sessions[session_id]
        assert len(session.logs) == 1
        assert session.logs[0].message == message
    
    @pytest.mark.asyncio
    async def test_send_message_websocket_error(self):
        """Test handling WebSocket send errors."""
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.send_text.side_effect = Exception("Connection closed")
        session_id = "test_session_123"
        
        # Setup connection
        self.connection_manager.active_connections[session_id] = mock_websocket
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="connected", message="Connected")
        )
        
        message = "Test message"
        
        # Should handle the exception gracefully
        await self.connection_manager.send_personal_message(message, session_id)
        
        # Should still add log entry despite WebSocket error
        session = self.connection_manager.sessions[session_id]
        assert len(session.logs) == 1


class TestProcessingStatus:
    """Test ProcessingStatus data model."""
    
    def test_processing_status_creation(self):
        """Test creating ProcessingStatus object."""
        status = ProcessingStatus(
            status="processing",
            message="Analyzing chapter 1",
            progress=25
        )
        
        assert status.status == "processing"
        assert status.message == "Analyzing chapter 1"
        assert status.progress == 25
    
    def test_processing_status_defaults(self):
        """Test ProcessingStatus default values."""
        status = ProcessingStatus(status="started")
        
        assert status.status == "started"
        assert status.message == ""
        assert status.progress == 0


class TestProcessingSessionData:
    """Test ProcessingSessionData model."""
    
    def test_session_data_creation(self):
        """Test creating ProcessingSessionData."""
        status = ProcessingStatus(status="started", message="Session started")
        
        session = ProcessingSessionData(
            session_id="session_123",
            manuscript_id="ms_456",
            status=status,
            start_time="2024-01-01T10:00:00"
        )
        
        assert session.session_id == "session_123"
        assert session.manuscript_id == "ms_456"
        assert session.status.status == "started"
        assert session.start_time == "2024-01-01T10:00:00"
        assert isinstance(session.logs, list)
        assert isinstance(session.images, list)
        assert isinstance(session.step_status, dict)
        assert session.pause_requested is False
    
    def test_session_data_with_websocket(self):
        """Test session data with WebSocket."""
        mock_websocket = Mock(spec=WebSocket)
        status = ProcessingStatus(status="connected")
        
        session = ProcessingSessionData(
            session_id="session_123",
            manuscript_id="ms_456",
            websocket=mock_websocket,
            status=status
        )
        
        assert session.websocket is mock_websocket


class TestProcessingLogEntry:
    """Test ProcessingLogEntry model."""
    
    def test_log_entry_creation(self):
        """Test creating log entry."""
        entry = ProcessingLogEntry(
            timestamp="2024-01-01T10:00:00",
            level="info",
            message="Processing started"
        )
        
        assert entry.timestamp == "2024-01-01T10:00:00"
        assert entry.level == "info"
        assert entry.message == "Processing started"


class TestProcessingImageEntry:
    """Test ProcessingImageEntry model."""
    
    def test_image_entry_creation(self):
        """Test creating image entry."""
        entry = ProcessingImageEntry(
            timestamp="2024-01-01T10:00:00",
            url="/images/test.png",
            prompt="A beautiful landscape",
            chapter_number=1,
            scene_number=2
        )
        
        assert entry.timestamp == "2024-01-01T10:00:00"
        assert entry.url == "/images/test.png"
        assert entry.prompt == "A beautiful landscape"
        assert entry.chapter_number == 1
        assert entry.scene_number == 2
    
    def test_image_entry_optional_fields(self):
        """Test image entry with optional fields."""
        entry = ProcessingImageEntry(
            timestamp="2024-01-01T10:00:00",
            url="/images/test.png",
            prompt="A scene"
        )
        
        assert entry.chapter_number is None
        assert entry.scene_number is None


class TestWebSocketEndpointPauseIntegration:
    """Test integration between WebSocket and endpoint pause functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.connection_manager = ConnectionManager()
    
    @pytest.mark.asyncio
    async def test_endpoint_pause_notification(self):
        """Test endpoint pause notification via WebSocket."""
        mock_websocket = AsyncMock(spec=WebSocket)
        session_id = "test_session_123"
        
        # Setup connection
        await self.connection_manager.connect(mock_websocket, session_id)
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="processing", message="Processing")
        )
        
        # Send endpoint pause notification
        pause_message = json.dumps({
            "type": "endpoint_paused",
            "message": "AI endpoint is paused. Waiting for restart...",
            "countdown_seconds": 120,
            "status": "waiting"
        })
        
        await self.connection_manager.send_personal_message(pause_message, session_id)
        
        mock_websocket.send_text.assert_called_once_with(pause_message)
        
        # Check log was added
        session = self.connection_manager.sessions[session_id]
        assert len(session.logs) == 1
        assert pause_message in session.logs[0].message
    
    @pytest.mark.asyncio
    async def test_endpoint_countdown_updates(self):
        """Test endpoint countdown updates via WebSocket."""
        mock_websocket = AsyncMock(spec=WebSocket)
        session_id = "test_session_123"
        
        # Setup connection
        await self.connection_manager.connect(mock_websocket, session_id)
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="waiting", message="Waiting for endpoint")
        )
        
        # Send countdown updates
        for remaining in [110, 100, 90]:
            countdown_message = json.dumps({
                "type": "endpoint_countdown", 
                "message": f"Retrying in {remaining} seconds...",
                "countdown_seconds": remaining,
                "status": "waiting"
            })
            
            await self.connection_manager.send_personal_message(countdown_message, session_id)
        
        # Should have sent 3 countdown messages
        assert mock_websocket.send_text.call_count == 3
        
        # Check all logs were added
        session = self.connection_manager.sessions[session_id]
        assert len(session.logs) == 3
    
    @pytest.mark.asyncio
    async def test_pause_resume_workflow(self):
        """Test complete pause/resume workflow via WebSocket."""
        mock_websocket = AsyncMock(spec=WebSocket)
        session_id = "test_session_123"
        
        # Setup session
        await self.connection_manager.connect(mock_websocket, session_id)
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="processing", message="Processing chapter 1")
        )
        
        # 1. User requests pause
        session = self.connection_manager.sessions[session_id]
        session.pause_requested = True
        session.status.status = "paused"
        session.status.message = "Processing paused at user request"
        
        pause_notification = json.dumps({
            "type": "log",
            "level": "warning",
            "message": "Processing paused at user request"
        })
        
        await self.connection_manager.send_personal_message(pause_notification, session_id)
        
        # 2. Resume processing
        session.pause_requested = False
        session.status.status = "processing"
        session.status.message = "Processing resumed"
        
        resume_notification = json.dumps({
            "type": "log",
            "level": "success", 
            "message": "Processing resumed"
        })
        
        await self.connection_manager.send_personal_message(resume_notification, session_id)
        
        # Verify workflow
        assert mock_websocket.send_text.call_count == 2
        assert len(session.logs) == 2
        assert session.logs[0].message == pause_notification
        assert session.logs[1].message == resume_notification
        assert session.pause_requested is False
        assert session.status.status == "processing"


class TestWebSocketErrorHandling:
    """Test WebSocket error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.connection_manager = ConnectionManager()
    
    @pytest.mark.asyncio
    async def test_send_to_nonexistent_session(self):
        """Test sending message to non-existent session."""
        # Should handle gracefully without raising exception
        await self.connection_manager.send_personal_message("test", "nonexistent_session")
        
        # No sessions should be created
        assert len(self.connection_manager.sessions) == 0
        assert len(self.connection_manager.active_connections) == 0
    
    @pytest.mark.asyncio
    async def test_websocket_connection_lost_during_send(self):
        """Test handling connection loss during message send."""
        mock_websocket = AsyncMock(spec=WebSocket)
        # Simulate connection loss
        mock_websocket.send_text.side_effect = Exception("Connection lost")
        
        session_id = "test_session_123"
        
        # Setup connection
        await self.connection_manager.connect(mock_websocket, session_id)
        self.connection_manager.sessions[session_id] = ProcessingSessionData(
            session_id=session_id,
            manuscript_id="ms_123",
            status=ProcessingStatus(status="connected", message="Connected")
        )
        
        # Try to send message - should not raise exception
        await self.connection_manager.send_personal_message("test message", session_id)
        
        # Session should still exist (for potential reconnection)
        assert session_id in self.connection_manager.sessions
        
        # Log should still be added despite send failure
        session = self.connection_manager.sessions[session_id]
        assert len(session.logs) == 1
    
    def test_multiple_disconnect_calls(self):
        """Test multiple disconnect calls don't cause errors."""
        session_id = "test_session_123"
        mock_websocket = Mock()
        
        # Add connection
        self.connection_manager.active_connections[session_id] = mock_websocket
        
        # Multiple disconnects should be safe
        self.connection_manager.disconnect(session_id)
        self.connection_manager.disconnect(session_id)
        self.connection_manager.disconnect(session_id)
        
        # Should not raise errors
        assert session_id not in self.connection_manager.active_connections