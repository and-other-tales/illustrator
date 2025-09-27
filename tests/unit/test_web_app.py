"""Tests for the enhanced web application with WebSocket processing."""

import pytest
import json
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from illustrator.models import (
    Chapter,
    EmotionalMoment,
    EmotionalTone,
    ImageProvider,
    IllustrationPrompt,
    ManuscriptMetadata,
    SavedManuscript
)
from src.illustrator.web.app import (
    app,
    WebSocketComprehensiveSceneAnalyzer,
    WebSocketIllustrationGenerator,
    run_processing_workflow
)


class TestWebSocketComprehensiveSceneAnalyzer:
    """Test WebSocket-enhanced scene analyzer."""

    @pytest.fixture
    def connection_manager(self):
        """Mock connection manager."""
        manager = Mock()
        manager.send_personal_message = AsyncMock()
        return manager

    @pytest.fixture
    def analyzer(self, connection_manager):
        """Create analyzer with mocked dependencies."""
        with patch('src.illustrator.web.app.ComprehensiveSceneAnalyzer') as mock_analyzer_class:
            mock_analyzer = Mock()
            mock_analyzer_class.return_value = mock_analyzer

            # Mock analyzer methods
            mock_analyzer._create_detailed_segments.return_value = [
                "When he looked again, the shadow behaved normally",
                "though the air where the man had stood felt oddly charged"
            ]
            mock_analyzer._score_emotional_intensity = AsyncMock(return_value=0.7)
            mock_analyzer._score_visual_potential = AsyncMock(return_value=0.8)
            mock_analyzer._score_narrative_significance = AsyncMock(return_value=0.6)
            mock_analyzer._score_dialogue_richness = AsyncMock(return_value=0.4)

            # Create test emotional moment
            test_moment = EmotionalMoment(
                text_excerpt="When he looked again, the shadow behaved normally",
                start_position=100,
                end_position=150,
                emotional_tones=[EmotionalTone.MYSTERY],
                intensity_score=0.75,
                context="Chapter 1 context"
            )
            mock_analyzer._create_detailed_moment = AsyncMock(return_value=test_moment)
            mock_analyzer._select_diverse_moments = AsyncMock(return_value=[test_moment])

            return WebSocketComprehensiveSceneAnalyzer(connection_manager, "test-session")

    @pytest.fixture
    def sample_chapter(self):
        """Create a sample chapter for testing."""
        return Chapter(
            title="Test Chapter",
            content="When he looked again, the shadow behaved normally, though the air where the man had stood felt oddly charged.",
            number=1,
            word_count=20
        )

    @pytest.mark.asyncio
    async def test_analyze_chapter_comprehensive(self, analyzer, sample_chapter):
        """Test comprehensive chapter analysis with WebSocket updates."""
        moments = await analyzer.analyze_chapter_comprehensive(sample_chapter)

        # Verify we got results
        assert len(moments) == 1
        assert moments[0].emotional_tones == [EmotionalTone.MYSTERY]

        # Verify WebSocket messages were sent
        assert analyzer.connection_manager.send_personal_message.call_count >= 3

        # Check that progress messages were sent
        calls = analyzer.connection_manager.send_personal_message.call_args_list
        messages = [json.loads(call[0][0]) for call in calls]

        # Should have progress messages
        progress_messages = [msg for msg in messages if msg.get('type') == 'log']
        assert len(progress_messages) > 0


class TestWebSocketIllustrationGenerator:
    """Test WebSocket-enhanced illustration generator."""

    @pytest.fixture
    def connection_manager(self):
        """Mock connection manager."""
        manager = Mock()
        manager.send_personal_message = AsyncMock()
        return manager

    @pytest.fixture
    def generator(self, connection_manager):
        """Create generator with mocked dependencies."""
        with patch('src.illustrator.web.app.IllustrationGenerator') as mock_gen_class, \
             patch('src.illustrator.web.app.init_chat_model') as mock_llm, \
             patch('src.illustrator.web.app.PromptEngineer') as mock_engineer_class:

            mock_gen = Mock()
            mock_gen_class.return_value = mock_gen
            mock_gen.output_dir = "/test/output"
            mock_gen.provider = ImageProvider.DALLE

            mock_engineer = Mock()
            mock_engineer_class.return_value = mock_engineer

            # Mock the advanced prompt generation
            test_prompt = IllustrationPrompt(
                provider=ImageProvider.DALLE,
                prompt="A digital painting of an elderly gentleman with mysterious shadows",
                style_modifiers=["mysterious", "atmospheric"],
                negative_prompt="blurry, low quality",
                technical_params={}
            )
            mock_engineer.engineer_prompt = AsyncMock(return_value=test_prompt)

            return WebSocketIllustrationGenerator(
                connection_manager,
                "test-session",
                ImageProvider.DALLE,
                "/test/output"
            )

    @pytest.fixture
    def sample_emotional_moment(self):
        """Create a sample emotional moment."""
        return EmotionalMoment(
            text_excerpt="When he looked again, the shadow behaved normally",
            start_position=100,
            end_position=150,
            emotional_tones=[EmotionalTone.MYSTERY],
            intensity_score=0.75,
            context="Chapter 1 context"
        )

    @pytest.fixture
    def sample_chapter(self):
        """Create a sample chapter."""
        return Chapter(
            title="Test Chapter",
            content="Sample content",
            number=1,
            word_count=20
        )

    @pytest.mark.asyncio
    async def test_create_advanced_prompt(self, generator, sample_emotional_moment, sample_chapter):
        """Test advanced AI-powered prompt creation."""
        style_config = {
            "art_style": "digital painting",
            "color_palette": "warm tones"
        }

        prompt = await generator.create_advanced_prompt(
            sample_emotional_moment,
            ImageProvider.DALLE,
            style_config,
            sample_chapter
        )

        # Verify we got a prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Verify WebSocket messages were sent
        assert generator.connection_manager.send_personal_message.called

        # Verify the AI analysis message was sent
        calls = generator.connection_manager.send_personal_message.call_args_list
        messages = [json.loads(call[0][0]) for call in calls if call[0]]

        analysis_messages = [msg for msg in messages if "AI Scene Analysis" in msg.get('message', '')]
        assert len(analysis_messages) > 0

    @pytest.mark.asyncio
    async def test_create_advanced_prompt_fallback(self, generator, sample_emotional_moment, sample_chapter):
        """Test fallback when AI prompt generation fails."""
        # Make the prompt engineer fail
        generator.prompt_engineer.engineer_prompt.side_effect = Exception("AI error")

        style_config = {
            "art_style": "digital painting",
            "color_palette": "warm tones"
        }

        prompt = await generator.create_advanced_prompt(
            sample_emotional_moment,
            ImageProvider.DALLE,
            style_config,
            sample_chapter
        )

        # Should still get a fallback prompt
        assert isinstance(prompt, str)
        assert len(prompt) > 0

        # Should have warning message
        calls = generator.connection_manager.send_personal_message.call_args_list
        messages = [json.loads(call[0][0]) for call in calls if call[0]]

        warning_messages = [msg for msg in messages if msg.get('level') == 'warning']
        assert len(warning_messages) > 0


class TestWebAppAPI:
    """Test the FastAPI web application."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "manuscript-illustrator"

    def test_process_endpoint_validation(self, client):
        """Test processing endpoint input validation."""
        # Missing required fields
        response = client.post("/api/process", json={})
        assert response.status_code == 422  # Validation error

        # Invalid style config
        response = client.post("/api/process", json={
            "manuscript_id": "test-123",
            "style_config": "invalid",  # Should be dict
            "max_emotional_moments": 5
        })
        assert response.status_code == 422

    @patch('src.illustrator.web.app.get_saved_manuscripts')
    def test_process_endpoint_success(self, mock_get_manuscripts, client):
        """Test successful processing request."""
        # Mock manuscript data
        test_manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test Manuscript",
                author="Test Author",
                genre="Fiction",
                total_chapters=1,
                created_at="2023-01-01T00:00:00"
            ),
            chapters=[
                Chapter(
                    title="Chapter 1",
                    content="Test content",
                    number=1,
                    word_count=2
                )
            ],
            saved_at="2023-01-01T00:00:00",
            file_path="/test/path"
        )
        mock_get_manuscripts.return_value = [test_manuscript]

        # Use the correct manuscript ID that would be generated
        import uuid
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "/test/path"))

        response = client.post("/api/process", json={
            "manuscript_id": manuscript_id,
            "style_config": {
                "image_provider": "dalle",
                "art_style": "digital painting"
            },
            "max_emotional_moments": 1
        })

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "session_id" in data

    def test_style_config_page(self, client):
        """Test style configuration page."""
        response = client.get("/manuscript/test-id/style")
        assert response.status_code == 200
        assert 'style-action-btn' in response.text

    def test_processing_page(self, client):
        """Test processing page."""
        response = client.get("/manuscript/test-id/process")
        assert response.status_code == 200

    @patch('src.illustrator.web.app.subprocess.check_output')
    def test_dashboard_includes_about_modal(self, mock_check_output, client):
        """Ensure the dashboard page renders the About modal."""
        mock_check_output.return_value = b"1234567890\n"

        response = client.get("/")
        assert response.status_code == 200

        html = response.text
        assert 'id="aboutModal"' in html
        assert 'Version 0.1.1234567890' in html
        assert 'Copyright © 2025 PI & Other Tales, Inc. All Rights Reserved.' in html
        assert 'text-uppercase" id="aboutModalLabel"' in html


class TestProcessingWorkflow:
    """Test the complete processing workflow."""

    @pytest.mark.asyncio
    @patch('src.illustrator.web.app.get_saved_manuscripts')
    @patch('src.illustrator.web.app.ComprehensiveSceneAnalyzer')
    @patch('src.illustrator.web.app.IllustrationGenerator')
    async def test_run_processing_workflow(self, mock_gen_class, mock_analyzer_class, mock_get_manuscripts):
        """Test the complete processing workflow."""
        # Setup mocks
        connection_manager = Mock()
        connection_manager.send_personal_message = AsyncMock()

        # Mock manuscript
        test_manuscript = SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test Manuscript",
                author="Test Author",
                genre="Fiction",
                total_chapters=1,
                created_at="2023-01-01T00:00:00"
            ),
            chapters=[
                Chapter(
                    title="Chapter 1",
                    content="Test content with emotional moments",
                    number=1,
                    word_count=5
                )
            ],
            saved_at="2023-01-01T00:00:00",
            file_path="/test/path"
        )

        import uuid
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "/test/path"))
        mock_get_manuscripts.return_value = [test_manuscript]

        # Mock analyzer
        test_moment = EmotionalMoment(
            text_excerpt="emotional moment",
            start_position=0,
            end_position=10,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.8,
            context="test context"
        )

        mock_analyzer = Mock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_chapter_comprehensive = AsyncMock(return_value=[test_moment])

        # Mock generator
        mock_gen = Mock()
        mock_gen_class.return_value = mock_gen
        mock_gen.generate_images = AsyncMock(return_value=[{
            "success": True,
            "file_path": "/test/image.png",
            "prompt": "test prompt"
        }])

        # Run workflow
        with patch('src.illustrator.web.app.connection_manager', connection_manager):
            await run_processing_workflow(
                session_id="test-session",
                manuscript_id=manuscript_id,
                style_config={"art_style": "digital painting", "image_provider": "dalle"},
                max_emotional_moments=1
            )

        # Verify workflow steps were executed
        mock_analyzer.analyze_chapter_comprehensive.assert_called_once()
        mock_gen.generate_images.assert_called_once()

        # Verify WebSocket messages were sent
        assert connection_manager.send_personal_message.call_count > 5


if __name__ == "__main__":
    pytest.main([__file__])
