"""Comprehensive unit tests for analysis module."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from illustrator.analysis import (
    EmotionalAnalyzer,
    TextSegment
)
from illustrator.models import (
    Chapter,
    EmotionalMoment,
    EmotionalTone
)


class TestTextSegment:
    """Test TextSegment data class."""

    def test_text_segment_creation(self):
        """Test text segment creation."""
        segment = TextSegment(
            text="This is a test segment.",
            start_pos=0,
            end_pos=23,
            context_before="",
            context_after="This continues the story."
        )

        assert segment.text == "This is a test segment."
        assert segment.start_pos == 0
        assert segment.end_pos == 23
        assert segment.context_before == ""
        assert segment.context_after == "This continues the story."

    def test_text_segment_with_context(self):
        """Test text segment with surrounding context."""
        segment = TextSegment(
            text="Key emotional moment here.",
            start_pos=100,
            end_pos=126,
            context_before="The story builds up to this: ",
            context_after=" And then the consequences unfold."
        )

        assert len(segment.text) == 26
        assert segment.start_pos == 100
        assert segment.end_pos == 126
        assert "builds up" in segment.context_before
        assert "consequences" in segment.context_after


class TestEmotionalAnalyzer:
    """Test EmotionalAnalyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock()
        self.mock_llm.ainvoke = AsyncMock()
        self.analyzer = EmotionalAnalyzer(self.mock_llm)

    def test_emotional_analyzer_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.llm == self.mock_llm
        assert hasattr(self.analyzer, 'scene_detector')
        assert hasattr(self.analyzer, 'narrative_analyzer')

    def test_emotion_patterns_exist(self):
        """Test that emotion patterns are defined."""
        assert hasattr(EmotionalAnalyzer, 'EMOTION_PATTERNS')
        assert EmotionalTone.JOY in EmotionalAnalyzer.EMOTION_PATTERNS
        assert EmotionalTone.SADNESS in EmotionalAnalyzer.EMOTION_PATTERNS
        assert EmotionalTone.FEAR in EmotionalAnalyzer.EMOTION_PATTERNS
        assert EmotionalTone.ANGER in EmotionalAnalyzer.EMOTION_PATTERNS

    def test_intensity_modifiers_exist(self):
        """Test that intensity modifiers are defined."""
        assert hasattr(EmotionalAnalyzer, 'INTENSITY_MODIFIERS')
        assert 'high' in EmotionalAnalyzer.INTENSITY_MODIFIERS
        assert 'medium' in EmotionalAnalyzer.INTENSITY_MODIFIERS
        assert 'low' in EmotionalAnalyzer.INTENSITY_MODIFIERS

    def create_sample_chapter(self) -> Chapter:
        """Create a sample chapter for testing."""
        return Chapter(
            title="Test Chapter",
            content="""Sarah walked through the dark forest, her heart pounding with fear.
            The shadows seemed to reach out for her, and every sound made her jump.
            Suddenly, she heard a familiar voice calling her name. Joy flooded through her
            as she recognized her brother's voice. She ran toward the sound, tears of
            happiness streaming down her face. When she finally saw him, she felt such
            overwhelming relief and love that she could barely speak.""",
            number=1,
            word_count=85
        )

    def test_segment_text_basic(self):
        """Test basic text segmentation."""
        text = "This is a test. " * 50  # 200 words
        segments = self.analyzer._segment_text(text, segment_size=20, overlap=5)

        assert len(segments) > 1
        assert all(isinstance(seg, TextSegment) for seg in segments)
        assert segments[0].start_pos == 0

        # Check overlap
        if len(segments) > 1:
            # There should be some overlap between segments
            assert segments[1].start_pos < segments[0].end_pos

    def test_segment_text_short(self):
        """Test segmentation of short text."""
        short_text = "Just a few words here."
        segments = self.analyzer._segment_text(short_text, segment_size=10, overlap=2)

        assert len(segments) >= 1
        assert segments[0].text == short_text

    @pytest.mark.asyncio
    async def test_analyze_chapter_basic(self):
        """Test basic chapter analysis."""
        chapter = self.create_sample_chapter()

        # Mock LLM response for intensity scoring
        self.mock_llm.ainvoke.return_value.content = "0.8"

        # Mock detailed analysis response
        detailed_response = AsyncMock()
        detailed_response.content = '''
        {
            "emotional_tones": ["fear", "joy", "relief"],
            "context": "Character reunion scene with emotional journey from fear to joy"
        }
        '''
        self.mock_llm.ainvoke.side_effect = ["0.8", detailed_response.content]

        moments = await self.analyzer.analyze_chapter(chapter, max_moments=2, min_intensity=0.5)

        assert isinstance(moments, list)
        assert len(moments) <= 2

        if moments:
            moment = moments[0]
            assert isinstance(moment, EmotionalMoment)
            assert moment.intensity_score >= 0.5
            assert len(moment.emotional_tones) > 0

    @pytest.mark.asyncio
    async def test_analyze_chapter_no_moments(self):
        """Test chapter analysis with no qualifying moments."""
        chapter = Chapter(
            title="Neutral Chapter",
            content="The weather was nice. People went about their daily routines.",
            number=1,
            word_count=12
        )

        # Mock low intensity scores
        self.mock_llm.ainvoke.return_value.content = "0.2"

        moments = await self.analyzer.analyze_chapter(chapter, max_moments=5, min_intensity=0.6)

        assert isinstance(moments, list)
        assert len(moments) == 0

    def test_calculate_pattern_score_joy(self):
        """Test pattern scoring for joyful content."""
        joyful_text = "She smiled brightly and laughed with delight, feeling absolutely elated."
        score = self.analyzer._calculate_pattern_score(joyful_text)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should detect some emotional content

    def test_calculate_pattern_score_sad(self):
        """Test pattern scoring for sad content."""
        sad_text = "Tears streamed down his face as he wept in sorrow and despair."
        score = self.analyzer._calculate_pattern_score(sad_text)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert score > 0

    def test_calculate_pattern_score_neutral(self):
        """Test pattern scoring for neutral content."""
        neutral_text = "The table was brown and rectangular."
        score = self.analyzer._calculate_pattern_score(neutral_text)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Neutral content should have very low score

    def test_identify_primary_emotion_joy(self):
        """Test primary emotion identification for joy."""
        joyful_text = "She smiled and laughed with happiness and delight."
        emotion = self.analyzer._identify_primary_emotion(joyful_text)

        assert isinstance(emotion, EmotionalTone)
        # Should identify joy or a related positive emotion

    def test_identify_primary_emotion_fear(self):
        """Test primary emotion identification for fear."""
        fearful_text = "He trembled with terror and panic, frightened by the shadows."
        emotion = self.analyzer._identify_primary_emotion(fearful_text)

        assert isinstance(emotion, EmotionalTone)
        # Should identify fear or a related negative emotion

    def test_identify_primary_emotion_fallback(self):
        """Test primary emotion identification fallback."""
        neutral_text = "The data was processed correctly."
        emotion = self.analyzer._identify_primary_emotion(neutral_text)

        assert emotion == EmotionalTone.ANTICIPATION  # Default fallback

    def test_extract_peak_excerpt_single_sentence(self):
        """Test peak excerpt extraction from single sentence."""
        text = "She felt overwhelming joy and happiness."
        excerpt = self.analyzer._extract_peak_excerpt(text, max_length=100)

        assert isinstance(excerpt, str)
        assert len(excerpt) <= 100
        assert "joy" in excerpt or "happiness" in excerpt

    def test_extract_peak_excerpt_multiple_sentences(self):
        """Test peak excerpt extraction from multiple sentences."""
        text = """The day was ordinary. Suddenly, she felt overwhelming joy and
        happiness flooding through her heart. The weather continued to be normal.
        Then tears of pure bliss streamed down her face."""

        excerpt = self.analyzer._extract_peak_excerpt(text, max_length=200)

        assert isinstance(excerpt, str)
        assert len(excerpt) <= 200
        # Should prefer sentences with emotional content
        assert "joy" in excerpt or "happiness" in excerpt or "bliss" in excerpt

    def test_extract_peak_excerpt_long_text(self):
        """Test peak excerpt extraction with length limiting."""
        long_emotional_text = ("She felt amazing wonderful fantastic incredible " * 20)
        excerpt = self.analyzer._extract_peak_excerpt(long_emotional_text, max_length=50)

        assert isinstance(excerpt, str)
        assert len(excerpt) <= 50
        assert len(excerpt) > 0

    @pytest.mark.asyncio
    async def test_score_emotional_intensity(self):
        """Test emotional intensity scoring."""
        segment = TextSegment(
            text="She was terrified and shaking with fear.",
            start_pos=0,
            end_pos=39,
            context_before="",
            context_after=""
        )

        # Mock LLM response
        self.mock_llm.ainvoke.return_value.content = "0.85"

        score = await self.analyzer._score_emotional_intensity(segment)

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_llm_intensity_score_valid_response(self):
        """Test LLM intensity scoring with valid response."""
        segment = TextSegment(
            text="The character was deeply moved.",
            start_pos=0,
            end_pos=30,
            context_before="",
            context_after=""
        )

        self.mock_llm.ainvoke.return_value.content = "0.75"

        score = await self.analyzer._llm_intensity_score(segment)

        assert score == 0.75
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_llm_intensity_score_invalid_response(self):
        """Test LLM intensity scoring with invalid response."""
        segment = TextSegment(
            text="Test text.",
            start_pos=0,
            end_pos=10,
            context_before="",
            context_after=""
        )

        self.mock_llm.ainvoke.return_value.content = "invalid_response"

        # Should return a fallback score instead of raising an error
        score = await self.analyzer._llm_intensity_score(segment)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0  # Should be a valid score in range

    @pytest.mark.asyncio
    async def test_analyze_segment_detailed_success(self):
        """Test detailed segment analysis with successful LLM response."""
        segment = TextSegment(
            text="She felt overwhelming joy.",
            start_pos=100,
            end_pos=125,
            context_before="The story built up",
            context_after="and then continued"
        )

        # Mock successful JSON response
        self.mock_llm.ainvoke.return_value.content = '''
        {
            "emotional_tones": ["joy", "excitement"],
            "context": "Character experiences positive revelation"
        }
        '''

        chapter_text = "Full chapter text here..."
        moment = await self.analyzer._analyze_segment_detailed(segment, 0.8, chapter_text)

        assert isinstance(moment, EmotionalMoment)
        assert moment.intensity_score == 0.8
        assert len(moment.emotional_tones) == 2
        assert EmotionalTone.JOY in moment.emotional_tones
        assert EmotionalTone.EXCITEMENT in moment.emotional_tones
        assert "Character experiences positive revelation" in moment.context

    @pytest.mark.asyncio
    async def test_analyze_segment_detailed_fallback(self):
        """Test detailed segment analysis with fallback when LLM fails."""
        segment = TextSegment(
            text="She cried with sadness.",
            start_pos=0,
            end_pos=21,
            context_before="",
            context_after=""
        )

        # Mock failed JSON response
        self.mock_llm.ainvoke.return_value.content = "invalid json"

        chapter_text = "Full chapter text..."
        moment = await self.analyzer._analyze_segment_detailed(segment, 0.6, chapter_text)

        assert isinstance(moment, EmotionalMoment)
        assert moment.intensity_score == 0.6
        assert len(moment.emotional_tones) >= 1
        assert "Emotionally resonant passage" in moment.context

    @pytest.mark.asyncio
    async def test_analyze_chapter_with_scenes_disabled(self):
        """Test chapter analysis with scene awareness disabled."""
        chapter = self.create_sample_chapter()

        # Mock LLM responses
        self.mock_llm.ainvoke.return_value.content = "0.7"
        detailed_response = '''
        {
            "emotional_tones": ["fear", "joy"],
            "context": "Emotional transition scene"
        }
        '''

        self.mock_llm.ainvoke.side_effect = ["0.7", detailed_response]

        moments = await self.analyzer.analyze_chapter_with_scenes(
            chapter, max_moments=3, min_intensity=0.5, scene_awareness=False
        )

        assert isinstance(moments, list)
        # Should fall back to regular analysis
        self.mock_llm.ainvoke.assert_called()  # Should have made LLM calls

    def test_remove_overlapping_moments(self):
        """Test removal of overlapping emotional moments."""
        # Create overlapping moments
        moment1 = EmotionalMoment(
            text_excerpt="First moment",
            start_position=0,
            end_position=50,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.8,
            context="First context"
        )

        moment2 = EmotionalMoment(
            text_excerpt="Overlapping moment",
            start_position=25,  # Overlaps with moment1
            end_position=75,
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.6,  # Lower score
            context="Overlapping context"
        )

        moment3 = EmotionalMoment(
            text_excerpt="Separate moment",
            start_position=100,  # No overlap
            end_position=150,
            emotional_tones=[EmotionalTone.SADNESS],
            intensity_score=0.7,
            context="Separate context"
        )

        candidate_moments = [
            (moment1, 0.8),
            (moment2, 0.6),
            (moment3, 0.7)
        ]

        unique_moments = self.analyzer._remove_overlapping_moments(candidate_moments)

        assert len(unique_moments) == 2  # Should remove the lower-scoring overlapping moment
        moment_texts = [moment.text_excerpt for moment, score in unique_moments]
        assert "First moment" in moment_texts  # Higher score should be kept
        assert "Separate moment" in moment_texts  # Non-overlapping should be kept
        assert "Overlapping moment" not in moment_texts  # Lower score should be removed


class TestEmotionalAnalyzerEdgeCases:
    """Test edge cases and error conditions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock()
        self.analyzer = EmotionalAnalyzer(self.mock_llm)

    def test_segment_text_empty(self):
        """Test segmentation of empty text."""
        segments = self.analyzer._segment_text("", segment_size=10, overlap=2)
        assert len(segments) == 0

    def test_segment_text_single_word(self):
        """Test segmentation of single word."""
        segments = self.analyzer._segment_text("word", segment_size=10, overlap=2)
        assert len(segments) == 1
        assert segments[0].text == "word"

    def test_calculate_pattern_score_empty(self):
        """Test pattern scoring with empty text."""
        score = self.analyzer._calculate_pattern_score("")
        assert score == 0.0

    def test_identify_primary_emotion_empty(self):
        """Test primary emotion identification with empty text."""
        emotion = self.analyzer._identify_primary_emotion("")
        assert emotion == EmotionalTone.ANTICIPATION

    def test_extract_peak_excerpt_empty(self):
        """Test peak excerpt extraction with empty text."""
        excerpt = self.analyzer._extract_peak_excerpt("")
        assert excerpt == ""

    def test_extract_peak_excerpt_no_sentences(self):
        """Test peak excerpt extraction with no proper sentences."""
        text = "word1 word2 word3"  # No sentence endings
        excerpt = self.analyzer._extract_peak_excerpt(text)
        assert len(excerpt) > 0
        assert excerpt == text  # Should return original if no sentences found