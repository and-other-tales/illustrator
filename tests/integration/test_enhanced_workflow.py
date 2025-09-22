"""Integration tests for the enhanced workflow with AI prompt engineering."""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from illustrator.models import (
    Chapter,
    EmotionalMoment,
    EmotionalTone,
    ImageProvider,
    IllustrationPrompt,
    ManuscriptMetadata,
    SavedManuscript
)
from src.illustrator.web.app import run_processing_workflow


class TestEnhancedWorkflowIntegration:
    """Test complete enhanced workflow integration."""

    @pytest.fixture
    def sample_manuscript(self):
        """Create a sample manuscript for testing."""
        return SavedManuscript(
            metadata=ManuscriptMetadata(
                title="Test Novel",
                author="Test Author",
                genre="Fantasy",
                total_chapters=1,
                created_at="2023-01-01T00:00:00"
            ),
            chapters=[
                Chapter(
                    title="The Mystery Begins",
                    content="""
                    When he looked again, the shadow behaved normally, though the air where the man had stood
                    felt oddly charged. The melody the man had been humming lingered in the air, hovering at
                    the threshold of hearing, occasionally shifting when he wasn't paying direct attention.

                    Lukas blinked hard and shook his head. The morning light caught the Victorian ironwork
                    of his home, transforming the ornate details into shadow-stories across the aged stone steps.
                    As he watched, the shadows cast by the ironwork appeared to shift independently of the
                    light source, creating an unsettling feeling that the house was observing him.

                    A harried mother wrestled a pram past, its occupant conducting a thorough investigation
                    of every crack in the pavement while a toddler danced an impatient circuit around them,
                    singing an improvised song about dinosaurs. The woman's phone was wedged between her
                    shoulder and ear as she hurried on with her morning routine.
                    """,
                    number=1,
                    word_count=150
                )
            ],
            saved_at="2023-01-01T00:00:00",
            file_path="/test/manuscript.txt"
        )

    @pytest.fixture
    def mock_connection_manager(self):
        """Mock WebSocket connection manager."""
        manager = Mock()
        manager.send_personal_message = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_complete_enhanced_workflow(self, sample_manuscript, mock_connection_manager):
        """Test the complete enhanced workflow from start to finish."""
        import uuid

        # Setup manuscript
        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "/test/manuscript.txt"))

        # Mock the AI analysis components with realistic responses
        # Need to patch the import that happens inside run_processing_workflow
        with patch('illustrator.web.routes.manuscripts.get_saved_manuscripts') as mock_get_manuscripts, \
             patch('src.illustrator.web.app.WebSocketComprehensiveSceneAnalyzer') as MockAnalyzer, \
             patch('src.illustrator.web.app.WebSocketIllustrationGenerator') as MockGenerator:

            mock_get_manuscripts.return_value = [sample_manuscript]

            # Setup analyzer mock
            mock_analyzer_instance = Mock()
            MockAnalyzer.return_value = mock_analyzer_instance

            # Create realistic emotional moments
            test_moments = [
                EmotionalMoment(
                    text_excerpt="When he looked again, the shadow behaved normally, though the air where the man had stood felt oddly charged",
                    start_position=0,
                    end_position=100,
                    emotional_tones=[EmotionalTone.MYSTERY, EmotionalTone.TENSION],
                    intensity_score=0.85,
                    context="supernatural morning scene"
                ),
                EmotionalMoment(
                    text_excerpt="The shadows cast by the ironwork appeared to shift independently of the light source",
                    start_position=200,
                    end_position=290,
                    emotional_tones=[EmotionalTone.FEAR, EmotionalTone.MYSTERY],
                    intensity_score=0.78,
                    context="unsettling house observation"
                ),
                EmotionalMoment(
                    text_excerpt="A harried mother wrestled a pram past, its occupant conducting a thorough investigation",
                    start_position=400,
                    end_position=490,
                    emotional_tones=[EmotionalTone.JOY, EmotionalTone.PEACE],
                    intensity_score=0.65,
                    context="normal morning life contrast"
                )
            ]

            mock_analyzer_instance.analyze_chapter_comprehensive = AsyncMock(return_value=test_moments)

            # Setup generator mock
            mock_generator_instance = Mock()
            MockGenerator.return_value = mock_generator_instance

            # Mock successful image generation
            mock_generator_instance.generate_images = AsyncMock(return_value=[
                {
                    "success": True,
                    "file_path": "/test/output/chapter_01_scene_01.png",
                    "prompt": "A mysterious digital painting of supernatural shadows and Victorian architecture",
                    "chapter_number": 1,
                    "scene_number": 1,
                    "provider": "dalle"
                },
                {
                    "success": True,
                    "file_path": "/test/output/chapter_01_scene_02.png",
                    "prompt": "Dark atmospheric illustration of shifting shadows on ornate ironwork",
                    "chapter_number": 1,
                    "scene_number": 2,
                    "provider": "dalle"
                },
                {
                    "success": True,
                    "file_path": "/test/output/chapter_01_scene_03.png",
                    "prompt": "Warm morning scene of mother with pram and child on urban street",
                    "chapter_number": 1,
                    "scene_number": 3,
                    "provider": "dalle"
                }
            ])

            # Mock the connection manager in the workflow
            with patch('src.illustrator.web.app.connection_manager', mock_connection_manager):

                # Run the complete workflow
                await run_processing_workflow(
                    session_id="test-session-123",
                    manuscript_id=manuscript_id,
                    style_config={
                        "image_provider": "dalle",
                        "art_style": "digital painting",
                        "color_palette": "atmospheric tones",
                        "artistic_influences": "Gothic Victorian"
                    },
                    max_emotional_moments=3
                )

                # Verify the workflow executed correctly
                mock_analyzer_instance.analyze_chapter_comprehensive.assert_called_once()
                mock_generator_instance.generate_images.assert_called_once()

                # Verify WebSocket messages were sent
                assert mock_connection_manager.send_personal_message.call_count >= 10

                # Check for key workflow messages
                calls = mock_connection_manager.send_personal_message.call_args_list
                messages = []
                for call in calls:
                    try:
                        import json
                        msg = json.loads(call[0][0])
                        messages.append(msg)
                    except:
                        pass

                # Verify we have progress messages
                progress_messages = [msg for msg in messages if msg.get('type') == 'progress']
                assert len(progress_messages) > 0

                # Verify we have log messages about AI analysis
                log_messages = [msg for msg in messages if msg.get('type') == 'log']
                ai_messages = [msg for msg in log_messages if 'AI' in msg.get('message', '')]
                assert len(ai_messages) > 0

                # Verify we have completion messages
                completion_messages = [msg for msg in messages if 'completed' in msg.get('message', '')]
                assert len(completion_messages) > 0

    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, sample_manuscript, mock_connection_manager):
        """Test workflow handles errors gracefully."""
        import uuid

        manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "/test/manuscript.txt"))

        # Mock analyzer to fail
        with patch('illustrator.web.routes.manuscripts.get_saved_manuscripts') as mock_get_manuscripts, \
             patch('src.illustrator.web.app.WebSocketComprehensiveSceneAnalyzer') as MockAnalyzer:
            mock_get_manuscripts.return_value = [sample_manuscript]
            mock_analyzer_instance = Mock()
            MockAnalyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_chapter_comprehensive = AsyncMock(
                side_effect=Exception("Analysis failed")
            )

            # Mock the connection manager
            with patch('src.illustrator.web.app.connection_manager', mock_connection_manager):

                # Run workflow - should handle error gracefully
                try:
                    await run_processing_workflow(
                        session_id="test-session-error",
                        manuscript_id=manuscript_id,
                        style_config={
                            "image_provider": "dalle",
                            "art_style": "digital painting"
                        },
                        max_emotional_moments=3
                    )
                except Exception:
                    pass  # Expected to handle errors internally

                # Verify error messages were sent
                calls = mock_connection_manager.send_personal_message.call_args_list
                messages = []
                for call in calls:
                    try:
                        import json
                        msg = json.loads(call[0][0])
                        messages.append(msg)
                    except:
                        pass

                # Should have error messages
                error_messages = [msg for msg in messages if msg.get('level') == 'error']
                assert len(error_messages) > 0

    @pytest.mark.asyncio
    async def test_ai_prompt_engineering_integration(self, mock_connection_manager):
        """Test that AI prompt engineering integrates properly."""
        from src.illustrator.web.app import WebSocketIllustrationGenerator
        from illustrator.models import ImageProvider

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test the WebSocket generator with AI prompt engineering
            generator = WebSocketIllustrationGenerator(
                connection_manager=mock_connection_manager,
                session_id="test-session",
                provider=ImageProvider.DALLE,
                output_dir=Path(temp_dir)
            )

            # Mock the AI prompt engineering
            with patch.object(generator.prompt_engineer, 'engineer_prompt') as mock_engineer:
                mock_prompt = IllustrationPrompt(
                    provider=ImageProvider.DALLE,
                    prompt="A masterfully crafted digital painting of an elderly gentleman with silver hair standing on a Victorian cobblestone street, his shadow rippling unnaturally against the ground like disturbed water, atmospheric morning lighting with supernatural tension, medium shot composition focusing on the mysterious shadow behavior, cinematic mood with otherworldly undertones, professional illustration quality",
                    style_modifiers=["atmospheric", "cinematic", "supernatural"],
                    negative_prompt="blurry, low quality, distorted",
                    technical_params={"quality": "high", "style": "photorealistic"}
                )
                mock_engineer.return_value = mock_prompt

                sample_moment = EmotionalMoment(
                    text_excerpt="When he looked again, the shadow behaved normally",
                    start_position=0,
                    end_position=50,
                    emotional_tones=[EmotionalTone.MYSTERY],
                    intensity_score=0.8,
                    context="supernatural scene"
                )

                sample_chapter = Chapter(
                    title="Test Chapter",
                    content="test",
                    number=1,
                    word_count=1
                )

                # Test AI prompt generation
                result_prompt = await generator.create_advanced_prompt(
                    emotional_moment=sample_moment,
                    provider=ImageProvider.DALLE,
                    style_config={"art_style": "digital painting"},
                    chapter=sample_chapter
                )

                # Verify the result
                assert isinstance(result_prompt, str)
                assert len(result_prompt) > 100  # Should be a detailed AI-generated prompt
                assert "elderly gentleman" in result_prompt or "Victorian" in result_prompt

                # Verify WebSocket messages were sent for AI analysis
                assert mock_connection_manager.send_personal_message.call_count > 0

                # Check for AI analysis messages
                calls = mock_connection_manager.send_personal_message.call_args_list
                messages = []
                for call in calls:
                    try:
                        import json
                        msg = json.loads(call[0][0])
                        messages.append(msg)
                    except:
                        pass

                ai_analysis_messages = [
                    msg for msg in messages
                    if msg.get('type') == 'log' and 'AI Scene Analysis' in msg.get('message', '')
                ]
                assert len(ai_analysis_messages) > 0

    def test_enhanced_workflow_components_exist(self):
        """Test that all enhanced workflow components are properly imported and available."""
        # Test imports work
        from src.illustrator.web.app import (
            WebSocketComprehensiveSceneAnalyzer,
            WebSocketIllustrationGenerator,
            run_processing_workflow
        )
        from illustrator.prompt_engineering import PromptEngineer, SceneAnalyzer
        from illustrator.quality_feedback import QualityAnalyzer, FeedbackSystem

        # Test classes can be instantiated (with mocks)
        mock_llm = Mock()
        mock_connection = Mock()

        # These should not raise exceptions
        analyzer = WebSocketComprehensiveSceneAnalyzer(mock_connection, "test")
        assert analyzer is not None

        generator = WebSocketIllustrationGenerator(
            mock_connection, "test", ImageProvider.DALLE, Path("/tmp")
        )
        assert generator is not None

        prompt_engineer = PromptEngineer(mock_llm)
        assert prompt_engineer is not None

        scene_analyzer = SceneAnalyzer(mock_llm)
        assert scene_analyzer is not None


if __name__ == "__main__":
    pytest.main([__file__])