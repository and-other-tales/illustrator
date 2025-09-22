"""Integration tests for the complete workflow."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from illustrator.context import ManuscriptContext
from illustrator.graph import (
    analyze_chapter,
    complete_chapter,
    initialize_session,
    route_next_step,
)
from illustrator.models import Chapter, ImageProvider, ManuscriptMetadata
from illustrator.state import ManuscriptState


class TestWorkflowIntegration:
    """Test the complete manuscript analysis workflow."""

    @pytest.fixture
    def sample_chapter(self):
        """Create a sample chapter for testing."""
        return Chapter(
            title="The Dark Forest",
            content="""
            Sarah stepped into the ancient forest, her heart pounding with fear and anticipation.
            The towering trees seemed to whisper secrets in the wind, their shadows dancing
            menacingly across her path. She felt a surge of determination mixed with terror
            as she pushed deeper into the unknown darkness.

            Suddenly, a branch snapped behind her. She spun around, her breath catching in her
            throat. Nothing but silence greeted her, yet she could feel eyes watching from
            the shadows. The forest seemed alive with malevolent intent.
            """,
            number=1,
            word_count=89
        )

    @pytest.fixture
    def manuscript_context(self):
        """Create a manuscript context for testing."""
        return ManuscriptContext(
            user_id="test-user-123",
            model="anthropic/claude-3-5-sonnet-20241022",
            image_provider=ImageProvider.DALLE,
            max_emotional_moments=3,
            min_intensity_threshold=0.6,
            default_art_style="digital fantasy art",
            openai_api_key="test-openai-key",
            anthropic_api_key="test-anthropic-key"
        )

    @pytest.fixture
    def initial_state(self):
        """Create initial state for testing."""
        return ManuscriptState(
            messages=[],
            manuscript_metadata=None,
            current_chapter=None,
            chapters_completed=[],
            awaiting_chapter_input=False,
            processing_complete=False,
            image_provider=ImageProvider.DALLE,
            style_preferences={},
            analysis_depth="detailed",
            current_analysis=None,
            error_message=None,
            retry_count=0
        )

    @pytest.fixture
    def mock_runtime(self, manuscript_context):
        """Create a mock runtime for testing."""
        runtime = Mock()
        runtime.context = manuscript_context
        runtime.store = AsyncMock()
        return runtime

    @pytest.mark.asyncio
    async def test_initialize_session(self, initial_state, mock_runtime):
        """Test session initialization."""
        result = await initialize_session(initial_state, mock_runtime)

        assert 'messages' in result
        assert len(result['messages']) == 1
        assert result['awaiting_chapter_input'] is True
        assert result['processing_complete'] is False
        assert 'Welcome to Manuscript Illustrator' in result['messages'][0].content

    @pytest.mark.asyncio
    async def test_analyze_chapter_success(self, initial_state, mock_runtime, sample_chapter):
        """Test successful chapter analysis."""
        # Setup state with current chapter
        state = initial_state.copy()
        state['current_chapter'] = sample_chapter

        # Mock LLM responses
        with patch('illustrator.graph.init_chat_model') as mock_init_chat:
            mock_llm = AsyncMock()
            mock_init_chat.return_value = mock_llm

            # Mock intensity scoring response
            intensity_response = Mock()
            intensity_response.content = "0.8"

            # Mock detailed analysis response
            analysis_response = Mock()
            analysis_response.content = json.dumps({
                "dominant_themes": ["fear", "mystery", "journey"],
                "setting_description": "A dark, ancient forest with mysterious atmosphere",
                "character_emotions": {"Sarah": ["fear", "anticipation", "tension"]},
                "visual_highlights": ["Forest shadows", "Character's fearful expression"]
            })

            # Mock detailed emotional moment analysis (for EmotionalAnalyzer)
            moment_analysis_response = Mock()
            moment_analysis_response.content = json.dumps({
                "emotional_tones": ["fear", "tension"],
                "context": "Character entering dangerous territory"
            })

            # Set up mock to return these responses (analyzer makes multiple calls)
            mock_llm.ainvoke.side_effect = [
                intensity_response,        # For LLM intensity scoring
                moment_analysis_response,  # For detailed segment analysis
                analysis_response,         # For main chapter analysis
            ]

            # Mock provider factory
            with patch('illustrator.graph.ProviderFactory') as mock_factory:
                mock_provider = AsyncMock()

                # Create a real IllustrationPrompt instead of a Mock
                from illustrator.models import IllustrationPrompt
                mock_prompt = IllustrationPrompt(
                    provider=ImageProvider.DALLE,
                    prompt="Dark forest scene with mysterious atmosphere",
                    style_modifiers=["dark", "atmospheric"],
                    technical_params={"model": "dall-e-3"}
                )
                mock_provider.generate_prompt.return_value = mock_prompt
                mock_factory.create_provider.return_value = mock_provider

                result = await analyze_chapter(state, mock_runtime)

                assert 'messages' in result
                assert result['awaiting_chapter_input'] is False
                assert result['error_message'] is None
                assert result['current_analysis'] is not None

                # Check that store was called to save analysis
                mock_runtime.store.aput.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_chapter_no_chapter(self, initial_state, mock_runtime):
        """Test chapter analysis with no current chapter."""
        result = await analyze_chapter(initial_state, mock_runtime)

        assert result['error_message'] == "No chapter provided for analysis"
        assert result['retry_count'] == 1

    @pytest.mark.asyncio
    async def test_analyze_chapter_llm_failure(self, initial_state, mock_runtime, sample_chapter):
        """Test chapter analysis with LLM failure."""
        state = initial_state.copy()
        state['current_chapter'] = sample_chapter

        with patch('illustrator.graph.init_chat_model') as mock_init_chat:
            mock_llm = AsyncMock()
            mock_llm.ainvoke.side_effect = Exception("LLM API error")
            mock_init_chat.return_value = mock_llm

            result = await analyze_chapter(state, mock_runtime)

            assert 'error_message' in result
            assert "Analysis failed" in result['error_message']
            assert result['retry_count'] == 1

    @pytest.mark.asyncio
    async def test_complete_chapter(self, initial_state, mock_runtime, sample_chapter):
        """Test chapter completion."""
        # Create a mock analysis
        mock_analysis = Mock()
        mock_analysis.chapter = sample_chapter
        mock_analysis.chapter.number = 1
        mock_analysis.illustration_prompts = [Mock(), Mock()]

        state = initial_state.copy()
        state['current_analysis'] = mock_analysis
        state['chapters_completed'] = []

        result = await complete_chapter(state, mock_runtime)

        assert 'messages' in result
        assert result['awaiting_chapter_input'] is True
        assert len(result['chapters_completed']) == 1
        assert result['current_chapter'] is None
        assert result['current_analysis'] is None
        assert "Processing Complete" in result['messages'][0].content

    def test_route_next_step_awaiting_input(self):
        """Test routing when awaiting chapter input."""
        state = {'awaiting_chapter_input': True}
        result = route_next_step(state)
        assert result == "__end__"

    def test_route_next_step_error(self):
        """Test routing with error condition."""
        state = {
            'error_message': 'Test error',
            'retry_count': 1,
            'awaiting_chapter_input': False
        }
        result = route_next_step(state)
        assert result == "handle_error"

    def test_route_next_step_analyze_chapter(self):
        """Test routing to analyze chapter."""
        state = {
            'current_chapter': Mock(),
            'current_analysis': None,
            'awaiting_chapter_input': False,
            'error_message': None
        }
        result = route_next_step(state)
        assert result == "analyze_chapter"

    def test_route_next_step_complete_chapter(self):
        """Test routing to complete chapter."""
        # Test routing to generate_illustrations when analysis exists but no illustrations yet
        state = {
            'current_analysis': Mock(),
            'awaiting_chapter_input': False,
            'error_message': None
        }
        result = route_next_step(state)
        assert result == "generate_illustrations"

        # Test routing to complete_chapter when analysis and illustrations both exist
        state_with_illustrations = {
            'current_analysis': Mock(),
            'awaiting_chapter_input': False,
            'error_message': None,
            'illustrations_generated': True
        }
        result = route_next_step(state_with_illustrations)
        assert result == "complete_chapter"


class TestEndToEndWorkflow:
    """Test end-to-end workflow scenarios."""

    @pytest.fixture
    def complete_manuscript_data(self):
        """Create complete manuscript data for testing."""
        return {
            'metadata': ManuscriptMetadata(
                title="Test Novel",
                author="Test Author",
                genre="Fantasy",
                total_chapters=2,
                created_at=datetime.now().isoformat()
            ),
            'chapters': [
                Chapter(
                    title="Chapter 1: The Beginning",
                    content="The hero's journey began with a single step into the unknown.",
                    number=1,
                    word_count=11
                ),
                Chapter(
                    title="Chapter 2: The Challenge",
                    content="Facing the dragon, she felt a mixture of terror and excitement.",
                    number=2,
                    word_count=11
                )
            ]
        }

    @pytest.mark.asyncio
    async def test_multi_chapter_workflow(self, complete_manuscript_data):
        """Test processing multiple chapters in sequence."""
        metadata = complete_manuscript_data['metadata']
        chapters = complete_manuscript_data['chapters']

        # This would be a more complex test that processes multiple chapters
        # through the complete workflow. For now, we test the structure.

        assert len(chapters) == 2
        assert metadata.total_chapters == 2

        for i, chapter in enumerate(chapters, 1):
            assert chapter.number == i
            assert chapter.word_count > 0
            assert len(chapter.title) > 0
            assert len(chapter.content) > 0

    def test_state_transitions(self):
        """Test that state transitions follow expected patterns."""
        # Initial state
        initial_state = {
            'awaiting_chapter_input': True,
            'processing_complete': False,
            'current_chapter': None,
            'current_analysis': None,
            'chapters_completed': []
        }

        # After adding chapter
        with_chapter = initial_state.copy()
        with_chapter.update({
            'awaiting_chapter_input': False,
            'current_chapter': Mock()
        })

        # After analysis
        with_analysis = with_chapter.copy()
        with_analysis.update({
            'current_analysis': Mock()
        })

        # After completion
        completed = with_analysis.copy()
        completed.update({
            'awaiting_chapter_input': True,
            'current_chapter': None,
            'current_analysis': None,
            'chapters_completed': [Mock()]
        })

        # Test routing for each state
        assert route_next_step(initial_state) == "__end__"
        assert route_next_step(with_chapter) == "analyze_chapter"
        assert route_next_step(with_analysis) == "generate_illustrations"

        # Test with illustrations generated
        with_illustrations = with_analysis.copy()
        with_illustrations['illustrations_generated'] = True
        assert route_next_step(with_illustrations) == "complete_chapter"