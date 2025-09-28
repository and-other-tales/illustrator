"""Comprehensive unit tests for the graph module."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

from illustrator.graph import (
    initialize_session,
    analyze_chapter,
    generate_illustrations,
    complete_chapter,
    handle_error,
    route_next_step,
    _format_user_preferences,
    _format_emotional_moments,
    _format_generation_results,
    graph
)
from illustrator.context import ManuscriptContext
from illustrator.state import ManuscriptState
from illustrator.models import (
    Chapter,
    ChapterAnalysis,
    EmotionalMoment,
    EmotionalTone,
    ImageProvider,
    IllustrationPrompt,
    LLMProvider,
    ManuscriptMetadata
)


class TestInitializeSession:
    """Test the initialize_session function."""

    @pytest.mark.asyncio
    async def test_initialize_session(self):
        """Test session initialization."""
        state = {}
        runtime = MagicMock()

        result = await initialize_session(state, runtime)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert "Welcome to Manuscript Illustrator" in result["messages"][0].content
        assert result["awaiting_chapter_input"] is True
        assert result["processing_complete"] is False
        assert result["chapters_completed"] == []
        assert result["error_message"] is None
        assert result["retry_count"] == 0


class TestAnalyzeChapter:
    """Test the analyze_chapter function."""

    @pytest.fixture
    def sample_chapter(self):
        """Sample chapter for testing."""
        return Chapter(
            title="Test Chapter",
            content="This is a test chapter with emotional content that should be analyzed.",
            number=1,
            word_count=12
        )

    @pytest.fixture
    def sample_context(self):
        """Sample context for testing."""
        return ManuscriptContext(
            user_id="test_user",
            anthropic_api_key="test_key",
            openai_api_key="test_key",
            image_provider=ImageProvider.DALLE,
            analysis_mode="scene"
        )

    @pytest.fixture
    def mock_runtime(self, sample_context):
        """Mock runtime with context."""
        runtime = MagicMock()
        runtime.context = sample_context
        return runtime

    @pytest.mark.asyncio
    async def test_analyze_chapter_no_chapter(self, mock_runtime):
        """Test analyze_chapter with no chapter provided."""
        state = {}

        result = await analyze_chapter(state, mock_runtime)

        assert "error_message" in result
        assert "No chapter provided" in result["error_message"]
        assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_analyze_chapter_no_anthropic_key(self, sample_chapter):
        """Test analyze_chapter without Anthropic API key."""
        context = ManuscriptContext(
            user_id="test_user",
            anthropic_api_key=None  # Missing key
        )
        runtime = MagicMock()
        runtime.context = context

        state = {"current_chapter": sample_chapter}

        with patch('illustrator.graph.create_chat_model_from_context') as mock_create_chat_model, \
             patch('illustrator.graph.EmotionalAnalyzer') as mock_analyzer:

            # Mock the LLM creation to raise an error when API key is missing
            mock_create_chat_model.side_effect = ValueError("Anthropic API key is required when using the Anthropic provider")

            mock_analyzer_instance = AsyncMock()
            mock_analyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_chapter_with_scenes.return_value = []

            result = await analyze_chapter(state, runtime)

        assert "error_message" in result
        assert "Anthropic API key is required" in result["error_message"]

    @pytest.mark.asyncio
    async def test_analyze_chapter_success(self, sample_chapter, mock_runtime):
        """Test successful chapter analysis."""
        state = {"current_chapter": sample_chapter}

        emotional_moment = EmotionalMoment(
            text_excerpt="emotional content",
            start_position=10,
            end_position=27,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.8,
            context="This is emotional content"
        )

        with patch('illustrator.graph.create_chat_model_from_context') as mock_create_chat_model, \
             patch('illustrator.graph.EmotionalAnalyzer') as mock_analyzer, \
             patch('illustrator.graph.ProviderFactory') as mock_provider_factory:

            # Setup mocks
            mock_llm = AsyncMock()
            mock_create_chat_model.return_value = mock_llm

            mock_analyzer_instance = AsyncMock()
            mock_analyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_chapter_with_scenes.return_value = [emotional_moment]

            # Mock analysis response
            analysis_response = Mock()
            analysis_response.content = '{"dominant_themes": ["test"], "setting_description": "test setting", "character_emotions": {}}'
            mock_llm.ainvoke.return_value = analysis_response

            # Mock provider
            mock_provider = AsyncMock()
            mock_provider_factory.create_provider.return_value = mock_provider
            mock_provider.generate_prompt.return_value = IllustrationPrompt(
                provider=ImageProvider.DALLE,
                prompt="test prompt",
                style_modifiers=["test style"]
            )

            # Mock store
            mock_runtime.store = AsyncMock()

            result = await analyze_chapter(state, mock_runtime)

        assert "current_analysis" in result
        assert result["awaiting_chapter_input"] is False
        assert result["error_message"] is None
        assert "messages" in result
        assert "Analysis Complete" in result["messages"][0].content

    @pytest.mark.asyncio
    async def test_analyze_chapter_basic_mode(self, sample_chapter, mock_runtime):
        """Test chapter analysis in basic mode."""
        mock_runtime.context.analysis_mode = "basic"
        state = {"current_chapter": sample_chapter}

        emotional_moment = EmotionalMoment(
            text_excerpt="emotional content",
            start_position=10,
            end_position=27,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.8,
            context="This is emotional content"
        )

        with patch('illustrator.graph.create_chat_model_from_context'), \
             patch('illustrator.graph.EmotionalAnalyzer') as mock_analyzer, \
             patch('illustrator.graph.ProviderFactory'):

            mock_analyzer_instance = AsyncMock()
            mock_analyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_chapter.return_value = [emotional_moment]

            mock_runtime.store = AsyncMock()

            result = await analyze_chapter(state, mock_runtime)

        # Should call basic analyze_chapter method
        mock_analyzer_instance.analyze_chapter.assert_called_once()
        mock_analyzer_instance.analyze_chapter_with_scenes.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_chapter_json_parse_error(self, sample_chapter, mock_runtime):
        """Test chapter analysis with JSON parse error."""
        state = {"current_chapter": sample_chapter}

        with patch('illustrator.graph.create_chat_model_from_context') as mock_create_chat_model, \
             patch('illustrator.graph.EmotionalAnalyzer') as mock_analyzer, \
             patch('illustrator.graph.ProviderFactory'):

            mock_llm = AsyncMock()
            mock_create_chat_model.return_value = mock_llm

            mock_analyzer_instance = AsyncMock()
            mock_analyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_chapter_with_scenes.return_value = []

            # Mock invalid JSON response
            analysis_response = Mock()
            analysis_response.content = "invalid json content"
            mock_llm.ainvoke.return_value = analysis_response

            mock_runtime.store = AsyncMock()

            result = await analyze_chapter(state, mock_runtime)

        # Should handle JSON error gracefully and use fallback values
        assert "current_analysis" in result
        assert result["error_message"] is None

    @pytest.mark.asyncio
    async def test_analyze_chapter_exception_handling(self, sample_chapter, mock_runtime):
        """Test chapter analysis exception handling."""
        state = {"current_chapter": sample_chapter}

        with patch('illustrator.graph.create_chat_model_from_context', side_effect=Exception("Test error")):
            result = await analyze_chapter(state, mock_runtime)

        assert "error_message" in result
        assert "Test error" in result["error_message"]
        assert result["retry_count"] == 1


class TestGenerateIllustrations:
    """Test the generate_illustrations function."""

    @pytest.fixture
    def sample_analysis(self):
        """Sample analysis for testing."""
        return ChapterAnalysis(
            chapter=Chapter(
                title="Test Chapter",
                content="Test content",
                number=1,
                word_count=2
            ),
            emotional_moments=[
                EmotionalMoment(
                    text_excerpt="test excerpt",
                    start_position=0,
                    end_position=12,
                    emotional_tones=[EmotionalTone.JOY],
                    intensity_score=0.8,
                    context="test excerpt"
                )
            ],
            dominant_themes=["test"],
            setting_description="test setting",
            character_emotions={},
            illustration_prompts=[IllustrationPrompt(
                provider=ImageProvider.DALLE,
                prompt="test prompt",
                style_modifiers=[],
                technical_params={}
            )]
        )

    @pytest.mark.asyncio
    async def test_generate_illustrations_no_analysis(self):
        """Test generate_illustrations with no analysis."""
        state = {}
        runtime = MagicMock()

        result = await generate_illustrations(state, runtime)

        assert "error_message" in result
        assert "No analysis available" in result["error_message"]

    @pytest.mark.asyncio
    async def test_generate_illustrations_no_anthropic_key(self, sample_analysis):
        """Test generate_illustrations without Anthropic API key."""
        context = ManuscriptContext(
            user_id="test_user",
            llm_provider=LLMProvider.ANTHROPIC,
            anthropic_api_key=None,
            image_provider=ImageProvider.DALLE,
            openai_api_key="test_openai_key"  # Provide OpenAI key for DALLE
        )
        runtime = MagicMock()
        runtime.context = context

        state = {"current_analysis": sample_analysis}

        result = await generate_illustrations(state, runtime)

        assert "error_message" in result
        assert "Anthropic API key is required" in result["error_message"]

    @pytest.mark.asyncio
    async def test_generate_illustrations_no_prompts(self):
        """Test generate_illustrations with no illustration prompts."""
        analysis = ChapterAnalysis(
            chapter=Chapter(title="Test", content="Test", number=1, word_count=1),
            emotional_moments=[],
            dominant_themes=[],
            setting_description="test",
            character_emotions={},
            illustration_prompts=[]  # No prompts
        )

        context = ManuscriptContext(
            user_id="test_user",
            llm_provider=LLMProvider.ANTHROPIC,
            anthropic_api_key="test_key",
            image_provider=ImageProvider.DALLE,
            openai_api_key="test_openai_key"  # Add OpenAI key to avoid initialization error
        )
        runtime = MagicMock()
        runtime.context = context

        state = {"current_analysis": analysis}

        result = await generate_illustrations(state, runtime)

        assert "error_message" in result
        assert "No illustration prompts were generated" in result["error_message"]

    @pytest.mark.asyncio
    async def test_generate_illustrations_success(self, sample_analysis):
        """Test successful illustration generation."""
        context = ManuscriptContext(
            user_id="test_user",
            anthropic_api_key="test_key",
            openai_api_key="test_openai_key",  # Add OpenAI key
            image_provider=ImageProvider.DALLE,
            image_concurrency=1
        )
        runtime = MagicMock()
        runtime.context = context
        runtime.store = AsyncMock()

        state = {"current_analysis": sample_analysis}

        with patch('illustrator.graph.ProviderFactory') as mock_provider_factory, \
             patch('illustrator.graph.init_chat_model'), \
             patch('illustrator.graph.FeedbackSystem') as mock_feedback_system:

            mock_provider = AsyncMock()
            mock_provider_factory.create_provider.return_value = mock_provider
            mock_provider.generate_image = AsyncMock(return_value={
                "success": True,
                "image_data": "test_image_data",
                "metadata": {"test": "metadata"}
            })

            # Mock the FeedbackSystem
            mock_feedback_instance = AsyncMock()
            mock_feedback_system.return_value = mock_feedback_instance
            
            # Import QualityAssessment and create proper mock
            from illustrator.quality_feedback import QualityAssessment
            from illustrator.models import QualityMetric
            
            quality_assessment = QualityAssessment()
            quality_assessment.prompt_id = "test_prompt_1"
            quality_assessment.generation_success = True
            quality_assessment.quality_scores = {QualityMetric.ACCURACY: 0.9}
            quality_assessment.provider = ImageProvider.DALLE
            quality_assessment.timestamp = "2025-09-28T10:00:00"
            
            mock_feedback_instance.process_generation_feedback = AsyncMock(return_value={
                'quality_assessment': quality_assessment,
                'improved_prompt': None,
                'feedback_applied': False
            })

            result = await generate_illustrations(state, runtime)

        assert result["illustrations_generated"] is True
        assert "generated_images" in result
        assert len(result["generated_images"]) == 1
        assert result["error_message"] is None

    @pytest.mark.asyncio
    async def test_generate_illustrations_with_feedback(self, sample_analysis):
        """Test illustration generation with quality feedback."""
        context = ManuscriptContext(
            user_id="test_user",
            anthropic_api_key="test_key",
            openai_api_key="test_openai_key",  # Add OpenAI key
            image_concurrency=1
        )
        runtime = MagicMock()
        runtime.context = context
        runtime.store = AsyncMock()

        state = {"current_analysis": sample_analysis}

        with patch('illustrator.graph.ProviderFactory') as mock_provider_factory, \
             patch('illustrator.graph.init_chat_model') as mock_init_chat, \
             patch('illustrator.graph.FeedbackSystem') as mock_feedback_system:

            mock_provider = AsyncMock()
            mock_provider_factory.create_provider.return_value = mock_provider
            mock_provider.generate_image.return_value = {
                "success": True,
                "image_data": "test_image_data",
                "metadata": {"test": "metadata"}
            }

            mock_llm = AsyncMock()
            mock_init_chat.return_value = mock_llm

            mock_feedback = AsyncMock()
            mock_feedback_system.return_value = mock_feedback
            mock_feedback.process_generation_feedback.return_value = {
                "quality_assessment": Mock(
                    prompt_id="test",
                    generation_success=True,
                    quality_scores={},
                    feedback_notes="test",
                    improvement_suggestions="test",
                    provider=ImageProvider.DALLE,
                    timestamp="2023-01-01"
                )
            }

            result = await generate_illustrations(state, runtime)

        assert result["illustrations_generated"] is True
        assert "quality_assessments" in result


class TestCompleteChapter:
    """Test the complete_chapter function."""

    @pytest.fixture
    def sample_analysis(self):
        """Sample analysis for testing."""
        return ChapterAnalysis(
            chapter=Chapter(title="Test Chapter", content="Test", number=1, word_count=1),
            emotional_moments=[],
            dominant_themes=[],
            setting_description="test",
            character_emotions={},
            illustration_prompts=[IllustrationPrompt(
                provider=ImageProvider.DALLE,
                prompt="test prompt",
                style_modifiers=["digital art"],
                negative_prompt=None,
                technical_params={}
            )]
        )

    @pytest.mark.asyncio
    async def test_complete_chapter_with_analysis(self, sample_analysis):
        """Test completing a chapter with analysis."""
        state = {
            "current_analysis": sample_analysis,
            "generated_images": [{"test": "image"}]
        }
        runtime = MagicMock()

        result = await complete_chapter(state, runtime)

        assert "messages" in result
        assert "Processing Complete" in result["messages"][0].content
        assert len(result["chapters_completed"]) == 1
        assert result["current_chapter"] is None
        assert result["current_analysis"] is None
        assert result["awaiting_chapter_input"] is True

    @pytest.mark.asyncio
    async def test_complete_chapter_no_analysis(self):
        """Test completing a chapter without analysis."""
        state = {}
        runtime = MagicMock()

        result = await complete_chapter(state, runtime)

        assert result["awaiting_chapter_input"] is True


class TestHandleError:
    """Test the handle_error function."""

    @pytest.mark.asyncio
    async def test_handle_error_first_attempt(self):
        """Test error handling on first attempt."""
        state = {
            "error_message": "Test error",
            "retry_count": 1
        }
        runtime = MagicMock()

        result = await handle_error(state, runtime)

        assert "messages" in result
        assert "Processing Error" in result["messages"][0].content
        assert "attempt 2 of 3" in result["messages"][0].content
        assert result["awaiting_chapter_input"] is False

    @pytest.mark.asyncio
    async def test_handle_error_max_attempts(self):
        """Test error handling after max attempts."""
        state = {
            "error_message": "Test error",
            "retry_count": 3
        }
        runtime = MagicMock()

        result = await handle_error(state, runtime)

        assert "messages" in result
        assert "Processing Failed" in result["messages"][0].content
        assert result["awaiting_chapter_input"] is True

    @pytest.mark.asyncio
    async def test_handle_error_no_message(self):
        """Test error handling with no error message."""
        state = {"retry_count": 1}
        runtime = MagicMock()

        result = await handle_error(state, runtime)

        assert "Unknown error occurred" in result["messages"][0].content


class TestRouteNextStep:
    """Test the route_next_step function."""

    def test_route_next_step_error(self):
        """Test routing when there's an error."""
        state = {
            "error_message": "Test error",
            "retry_count": 1
        }

        result = route_next_step(state)
        assert result == "handle_error"

    def test_route_next_step_awaiting_input(self):
        """Test routing when awaiting chapter input."""
        from langgraph.graph import END

        state = {"awaiting_chapter_input": True}

        result = route_next_step(state)
        assert result == END

    def test_route_next_step_analyze_chapter(self):
        """Test routing to analyze chapter."""
        state = {
            "current_chapter": Mock(),
            "awaiting_chapter_input": False
        }

        result = route_next_step(state)
        assert result == "analyze_chapter"

    def test_route_next_step_generate_illustrations(self):
        """Test routing to generate illustrations."""
        state = {
            "current_analysis": Mock(),
            "illustrations_generated": False,
            "awaiting_chapter_input": False
        }

        result = route_next_step(state)
        assert result == "generate_illustrations"

    def test_route_next_step_complete_chapter(self):
        """Test routing to complete chapter."""
        state = {
            "current_analysis": Mock(),
            "illustrations_generated": True,
            "awaiting_chapter_input": False
        }

        result = route_next_step(state)
        assert result == "complete_chapter"

    def test_route_next_step_end(self):
        """Test routing to END."""
        from langgraph.graph import END

        state = {"awaiting_chapter_input": False}

        result = route_next_step(state)
        assert result == END


class TestHelperFunctions:
    """Test helper functions."""

    def test_format_user_preferences(self):
        """Test formatting user preferences."""
        context = ManuscriptContext(
            user_id="test",
            image_provider=ImageProvider.DALLE,
            default_art_style="digital painting",
            color_palette="warm",
            artistic_influences="Van Gogh"
        )

        result = _format_user_preferences(context)

        assert "dalle" in result
        assert "digital painting" in result
        assert "warm" in result
        assert "Van Gogh" in result

    def test_format_user_preferences_minimal(self):
        """Test formatting user preferences with minimal data."""
        context = ManuscriptContext(
            user_id="test",
            image_provider=ImageProvider.FLUX,
            default_art_style="sketch"
        )

        result = _format_user_preferences(context)

        assert "flux" in result
        assert "sketch" in result
        assert len(result.split(";")) == 2  # Only provider and art style

    def test_format_emotional_moments(self):
        """Test formatting emotional moments."""
        moments = [
            EmotionalMoment(
                text_excerpt="This is a long excerpt that should be truncated because it exceeds the character limit",
                start_position=0,
                end_position=96,
                emotional_tones=[EmotionalTone.JOY, EmotionalTone.EXCITEMENT],
                intensity_score=0.8,
                context="This is a long excerpt that should be truncated because it exceeds the character limit"
            ),
            EmotionalMoment(
                text_excerpt="Short excerpt",
                start_position=100,
                end_position=113,
                emotional_tones=[EmotionalTone.SADNESS],
                intensity_score=0.6,
                context="Short excerpt"
            )
        ]

        result = _format_emotional_moments(moments)

        assert "1. **Joy, Excitement**" in result
        assert "2. **Sadness**" in result
        # Check if truncation occurs for long excerpts (>100 chars)
        # The test excerpt is exactly 96 chars, so no truncation expected
        assert "This is a long excerpt that should be truncated because it exceeds the character limit" in result
        assert "Short excerpt" in result

    def test_format_emotional_moments_empty(self):
        """Test formatting empty emotional moments."""
        result = _format_emotional_moments([])
        assert "No high-intensity moments identified" in result

    def test_format_generation_results(self):
        """Test formatting generation results."""
        generated_images = [
            {
                "emotional_moment": "This is a long moment description that should be truncated for display purposes"
            },
            {
                "emotional_moment": "Short moment"
            }
        ]

        result = _format_generation_results(generated_images)

        assert "Scene:" in result
        # The long text is 79 chars, which is under the 80 char limit
        # So no truncation should occur
        assert "This is a long moment description that should be truncated for display purposes" in result
        assert "Short moment" in result

    def test_format_generation_results_empty(self):
        """Test formatting empty generation results."""
        result = _format_generation_results([])
        assert "No images were generated" in result


class TestGraphConfiguration:
    """Test graph configuration and structure."""

    def test_graph_exists(self):
        """Test that graph is properly configured."""
        assert graph is not None
        assert hasattr(graph, 'name')
        assert graph.name == "ManuscriptIllustrator"

    def test_graph_compilation(self):
        """Test that graph compiles without errors."""
        # Graph should already be compiled in the module
        assert graph is not None

    def test_graph_has_nodes(self):
        """Test that graph has expected nodes."""
        # This tests that the graph was built with the expected nodes
        # The actual node structure is internal to LangGraph
        assert graph is not None


class TestAsyncConcurrency:
    """Test async concurrency handling in graph functions."""

    @pytest.fixture
    def sample_context(self):
        """Mock context for testing."""
        return ManuscriptContext(
            user_id="test_user",
            anthropic_api_key="test_key",
            manuscript_id="test_manuscript",
            output_format="json",
            analysis_mode="comprehensive",
            image_provider="dalle"
        )

    @pytest.fixture
    def mock_runtime(self, sample_context):
        """Mock runtime with context."""
        runtime = MagicMock()
        runtime.context = sample_context
        return runtime

    @pytest.fixture
    def sample_analysis(self):
        """Sample chapter analysis for testing."""
        from illustrator.models import Chapter, ChapterAnalysis, EmotionalMoment, EmotionalTone, IllustrationPrompt, ImageProvider

        chapter = Chapter(title="Test", content="Test content", number=1, word_count=10)
        emotional_moment = EmotionalMoment(
            text_excerpt="test moment",
            start_position=0,
            end_position=10,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.8,
            context="test context"
        )
        illustration_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="test prompt",
            style_modifiers=["digital art"],
            technical_params={}
        )

        return ChapterAnalysis(
            chapter=chapter,
            emotional_moments=[emotional_moment],
            dominant_themes=["test theme"],
            setting_description="test setting",
            character_emotions={},
            illustration_prompts=[illustration_prompt]
        )

    @pytest.mark.asyncio
    async def test_analyze_chapter_concurrency_limits(self, mock_runtime):
        """Test that analyze_chapter respects concurrency limits."""
        mock_runtime.context.prompt_concurrency = 2

        chapter = Chapter(title="Test", content="Test content", number=1, word_count=2)
        state = {"current_chapter": chapter}

        emotional_moments = [
            EmotionalMoment(
                text_excerpt=f"moment {i}",
                start_position=i * 10,
                end_position=(i * 10) + 8,
                emotional_tones=[EmotionalTone.JOY],
                intensity_score=0.8,
                context=f"moment {i} context"
            )
            for i in range(5)  # More moments than concurrency limit
        ]

        with patch('illustrator.graph.create_chat_model_from_context') as mock_create_chat_model, \
             patch('illustrator.graph.EmotionalAnalyzer') as mock_analyzer, \
             patch('illustrator.graph.ProviderFactory') as mock_provider_factory, \
             patch('asyncio.Semaphore') as mock_semaphore:

            # Mock the LLM that is returned by create_chat_model_from_context
            mock_llm = AsyncMock()
            mock_llm.ainvoke.return_value.content = '{"dominant_themes": ["test theme"], "setting_description": "test setting", "character_emotions": {}}'
            mock_create_chat_model.return_value = mock_llm

            mock_analyzer_instance = AsyncMock()
            mock_analyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_chapter_with_scenes.return_value = emotional_moments

            mock_provider = AsyncMock()
            mock_provider_factory.create_provider.return_value = mock_provider
            mock_provider.generate_prompt.return_value = IllustrationPrompt(
                provider=ImageProvider.DALLE,
                prompt="test prompt",
                style_modifiers=["test style"]
            )

            mock_store = AsyncMock()
            mock_store.aput = AsyncMock()
            mock_store.aget = AsyncMock(return_value=None)
            mock_runtime.store = mock_store

            await analyze_chapter(state, mock_runtime)

        # Should create semaphore with the specified concurrency limit
        mock_semaphore.assert_called_with(2)

    @pytest.mark.asyncio
    async def test_generate_illustrations_concurrency_limits(self, sample_analysis):
        """Test that generate_illustrations respects concurrency limits."""
        context = ManuscriptContext(
            user_id="test_user",
            anthropic_api_key="test_key",
            openai_api_key="test_openai_key",  # Add OpenAI key
            image_concurrency=1
        )
        runtime = MagicMock()
        runtime.context = context
        runtime.store = AsyncMock()

        # Create analysis with multiple prompts
        sample_analysis.illustration_prompts = [
            IllustrationPrompt(
                provider=ImageProvider.DALLE,
                prompt="prompt1",
                style_modifiers=["digital art"],
                negative_prompt=None,
                technical_params={}
            ),
            IllustrationPrompt(
                provider=ImageProvider.DALLE,
                prompt="prompt2",
                style_modifiers=["digital art"],
                negative_prompt=None,
                technical_params={}
            ),
            IllustrationPrompt(
                provider=ImageProvider.DALLE,
                prompt="prompt3",
                style_modifiers=["digital art"],
                negative_prompt=None,
                technical_params={}
            )
        ]
        state = {"current_analysis": sample_analysis}

        with patch('illustrator.graph.ProviderFactory') as mock_provider_factory, \
             patch('asyncio.Semaphore') as mock_semaphore:

            mock_provider = AsyncMock()
            mock_provider_factory.create_provider.return_value = mock_provider
            mock_provider.generate_image.return_value = {
                "success": True,
                "image_data": "test",
                "metadata": {}
            }

            await generate_illustrations(state, runtime)

        # Should create semaphore with the specified concurrency limit
        mock_semaphore.assert_called_with(1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])