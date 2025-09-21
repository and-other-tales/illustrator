"""Unit tests for emotional analysis functionality."""

import json
from unittest.mock import AsyncMock, Mock

import pytest

from illustrator.analysis import EmotionalAnalyzer, TextSegment
from illustrator.models import Chapter, EmotionalMoment, EmotionalTone


class TestTextSegment:
    """Test TextSegment dataclass."""

    def test_text_segment_creation(self):
        """Test creating a text segment."""
        segment = TextSegment(
            text="This is a test segment.",
            start_pos=0,
            end_pos=24,
            context_before="",
            context_after="More text follows."
        )

        assert segment.text == "This is a test segment."
        assert segment.start_pos == 0
        assert segment.end_pos == 24
        assert segment.context_after == "More text follows."


class TestEmotionalAnalyzer:
    """Test EmotionalAnalyzer class."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM for testing."""
        llm = AsyncMock()
        return llm

    @pytest.fixture
    def analyzer(self, mock_llm):
        """Create an analyzer instance for testing."""
        return EmotionalAnalyzer(mock_llm)

    @pytest.fixture
    def sample_chapter(self):
        """Create a sample chapter for testing."""
        return Chapter(
            title="Test Chapter",
            content="She felt a surge of overwhelming joy as the door opened. The darkness beyond was terrifying, filled with unknown dangers. Her heart pounded with anticipation and fear, a mixture of emotions that made her tremble.",
            number=1,
            word_count=35
        )

    def test_segment_text_basic(self, analyzer):
        """Test basic text segmentation."""
        text = "This is a test. " * 100  # Create text longer than segment size
        segments = analyzer._segment_text(text, segment_size=50, overlap=10)

        assert len(segments) > 1
        assert all(isinstance(seg, TextSegment) for seg in segments)
        assert segments[0].start_pos == 0

    def test_segment_text_short_text(self, analyzer):
        """Test segmenting short text."""
        text = "Short text here."
        segments = analyzer._segment_text(text, segment_size=100, overlap=10)

        assert len(segments) == 1
        assert segments[0].text == text

    def test_calculate_pattern_score(self, analyzer):
        """Test pattern-based scoring."""
        # Text with strong emotional patterns
        emotional_text = "She was absolutely terrified and trembling with fear as the shadow lurked menacingly in the darkness."
        score = analyzer._calculate_pattern_score(emotional_text)
        assert score > 0

        # Neutral text
        neutral_text = "The table was brown and had four legs."
        neutral_score = analyzer._calculate_pattern_score(neutral_text)
        assert neutral_score < score

    def test_identify_primary_emotion(self, analyzer):
        """Test primary emotion identification."""
        fear_text = "She was terrified and trembling with fear as shadows lurked."
        emotion = analyzer._identify_primary_emotion(fear_text)
        assert emotion == EmotionalTone.FEAR

        joy_text = "She laughed with delight and smiled brightly with joy."
        emotion = analyzer._identify_primary_emotion(joy_text)
        assert emotion == EmotionalTone.JOY

    def test_extract_peak_excerpt(self, analyzer):
        """Test extracting peak emotional excerpt."""
        text = "The weather was nice. She was absolutely terrified and trembling! The book was on the table."
        excerpt = analyzer._extract_peak_excerpt(text, max_length=100)

        assert "terrified" in excerpt
        assert len(excerpt) <= 100

    @pytest.mark.asyncio
    async def test_llm_intensity_score_success(self, analyzer, mock_llm):
        """Test LLM-based intensity scoring with successful response."""
        # Mock successful LLM response
        mock_response = Mock()
        mock_response.content = "0.75"
        mock_llm.ainvoke.return_value = mock_response

        segment = TextSegment(
            text="She was terrified",
            start_pos=0,
            end_pos=17,
            context_before="",
            context_after=""
        )

        score = await analyzer._llm_intensity_score(segment)
        assert score == 0.75
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_llm_intensity_score_fallback(self, analyzer, mock_llm):
        """Test LLM scoring fallback to pattern matching."""
        # Mock LLM failure
        mock_llm.ainvoke.side_effect = Exception("LLM error")

        segment = TextSegment(
            text="She was terrified and trembling",
            start_pos=0,
            end_pos=30,
            context_before="",
            context_after=""
        )

        score = await analyzer._llm_intensity_score(segment)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_segment_detailed_success(self, analyzer, mock_llm):
        """Test detailed segment analysis with successful LLM response."""
        # Mock successful LLM response with JSON
        mock_response = Mock()
        mock_response.content = json.dumps({
            "emotional_tones": ["fear", "tension"],
            "context": "Character facing imminent danger"
        })
        mock_llm.ainvoke.return_value = mock_response

        segment = TextSegment(
            text="She was terrified",
            start_pos=0,
            end_pos=17,
            context_before="Earlier context",
            context_after="Later context"
        )

        moment = await analyzer._analyze_segment_detailed(segment, 0.8, "Full chapter text")

        assert isinstance(moment, EmotionalMoment)
        assert moment.intensity_score == 0.8
        assert EmotionalTone.FEAR in moment.emotional_tones
        assert EmotionalTone.TENSION in moment.emotional_tones
        assert "danger" in moment.context.lower()

    @pytest.mark.asyncio
    async def test_analyze_segment_detailed_fallback(self, analyzer, mock_llm):
        """Test detailed segment analysis with LLM failure fallback."""
        # Mock LLM failure
        mock_llm.ainvoke.side_effect = Exception("LLM error")

        segment = TextSegment(
            text="She laughed with joy",
            start_pos=0,
            end_pos=19,
            context_before="",
            context_after=""
        )

        moment = await analyzer._analyze_segment_detailed(segment, 0.6, "Full chapter")

        assert isinstance(moment, EmotionalMoment)
        assert moment.intensity_score == 0.6
        assert len(moment.emotional_tones) > 0
        assert "resonant" in moment.context.lower()

    @pytest.mark.asyncio
    async def test_analyze_chapter_integration(self, analyzer, mock_llm, sample_chapter):
        """Test complete chapter analysis integration."""
        # Mock LLM responses for intensity scoring and detailed analysis
        intensity_response = Mock()
        intensity_response.content = "0.8"

        analysis_response = Mock()
        analysis_response.content = json.dumps({
            "emotional_tones": ["joy", "fear", "anticipation"],
            "context": "Character experiencing mixed emotions"
        })

        mock_llm.ainvoke.side_effect = [intensity_response, analysis_response]

        moments = await analyzer.analyze_chapter(sample_chapter, max_moments=2, min_intensity=0.5)

        assert isinstance(moments, list)
        assert len(moments) <= 2

        if moments:  # If we found emotional moments
            for moment in moments:
                assert isinstance(moment, EmotionalMoment)
                assert moment.intensity_score >= 0.5
                assert len(moment.emotional_tones) > 0

    def test_emotion_patterns_coverage(self, analyzer):
        """Test that emotion patterns cover all expected emotional tones."""
        expected_emotions = [
            EmotionalTone.JOY, EmotionalTone.SADNESS, EmotionalTone.FEAR,
            EmotionalTone.ANGER, EmotionalTone.TENSION, EmotionalTone.MYSTERY
        ]

        for emotion in expected_emotions:
            assert emotion in analyzer.EMOTION_PATTERNS
            assert len(analyzer.EMOTION_PATTERNS[emotion]) > 0

    def test_intensity_modifiers_coverage(self, analyzer):
        """Test intensity modifier patterns."""
        assert 'high' in analyzer.INTENSITY_MODIFIERS
        assert 'medium' in analyzer.INTENSITY_MODIFIERS
        assert 'low' in analyzer.INTENSITY_MODIFIERS

        # Test that each category has modifiers
        for category in analyzer.INTENSITY_MODIFIERS.values():
            assert len(category) > 0
            assert all(isinstance(modifier, str) for modifier in category)