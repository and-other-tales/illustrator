"""Tests for the quality feedback system."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from typing import Dict, Any

from illustrator.models import (
    EmotionalMoment,
    EmotionalTone,
    IllustrationPrompt,
    ImageProvider
)
from illustrator.quality_feedback import (
    QualityMetric,
    QualityAssessment,
    PromptPerformance
)


class TestQualityAssessment:
    """Test QualityAssessment data structure."""

    def test_quality_assessment_creation(self):
        """Test creating a quality assessment."""
        assessment = QualityAssessment(
            overall_score=85,
            accuracy_score=80,
            style_consistency=90,
            emotional_alignment=85,
            technical_quality=80,
            prompt_effectiveness=88,
            areas_for_improvement=["lighting could be more dramatic"],
            strengths=["excellent character portrayal"],
            recommendations=["adjust lighting modifiers"]
        )

        assert assessment.overall_score == 85
        assert assessment.accuracy_score == 80
        assert "excellent character portrayal" in assessment.strengths
        assert len(assessment.recommendations) == 1

    def test_quality_assessment_validation(self):
        """Test quality assessment score validation."""
        # Valid scores
        assessment = QualityAssessment(
            overall_score=75,
            accuracy_score=70,
            style_consistency=80,
            emotional_alignment=75,
            technical_quality=70,
            prompt_effectiveness=75,
            areas_for_improvement=[],
            strengths=[],
            recommendations=[]
        )
        assert assessment.overall_score == 75

    def test_needs_improvement(self):
        """Test needs improvement detection."""
        # High quality assessment
        good_assessment = QualityAssessment(
            overall_score=90,
            accuracy_score=85,
            style_consistency=95,
            emotional_alignment=90,
            technical_quality=88,
            prompt_effectiveness=92,
            areas_for_improvement=[],
            strengths=[],
            recommendations=[]
        )
        assert not good_assessment.needs_improvement()

        # Low quality assessment
        poor_assessment = QualityAssessment(
            overall_score=60,
            accuracy_score=55,
            style_consistency=65,
            emotional_alignment=58,
            technical_quality=62,
            prompt_effectiveness=60,
            areas_for_improvement=["multiple issues"],
            strengths=[],
            recommendations=[]
        )
        assert poor_assessment.needs_improvement()


class TestQualityAnalyzer:
    """Test QualityAnalyzer functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.mock_llm = AsyncMock()
        self.analyzer = QualityAnalyzer(self.mock_llm)

    def test_analyze_prompt_structure(self):
        """Test prompt structure analysis."""
        # Well-structured prompt
        good_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="A detailed fantasy castle on a hilltop, dramatic lighting, storm clouds, digital art, high quality",
            style_modifiers=["fantasy", "detailed", "dramatic"],
            negative_prompt=["low quality", "blurry"],
            technical_params={"quality": "hd", "size": "1024x1024"}
        )

        score = self.analyzer.analyze_prompt_structure(good_prompt)
        assert 70 <= score <= 100  # Should be reasonably high

        # Poor prompt
        poor_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="castle",
            style_modifiers=[],
            negative_prompt=[],
            technical_params={}
        )

        score = self.analyzer.analyze_prompt_structure(poor_prompt)
        assert score < 70  # Should be low

    def test_check_emotional_alignment(self):
        """Test emotional alignment checking."""
        emotional_moment = EmotionalMoment(
            text_excerpt="Terror gripped her heart as the monster approached",
            context="scary encounter",
            emotional_tones=[EmotionalTone.FEAR, EmotionalTone.SUSPENSE],
            intensity_score=0.9,
            start_position=0,
            end_position=50
        )

        # Well-aligned prompt
        aligned_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="terrifying monster approaching, dramatic lighting, fear, suspense",
            style_modifiers=["dark", "scary", "dramatic"],
            negative_prompt=["cheerful", "bright"],
            technical_params={}
        )

        score = self.analyzer.check_emotional_alignment(aligned_prompt, emotional_moment)
        assert score > 70  # Should be well-aligned

        # Misaligned prompt
        misaligned_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="beautiful sunny meadow with flowers, bright and cheerful",
            style_modifiers=["bright", "cheerful", "colorful"],
            negative_prompt=["dark"],
            technical_params={}
        )

        score = self.analyzer.check_emotional_alignment(misaligned_prompt, emotional_moment)
        assert score < 50  # Should be poorly aligned

    def test_evaluate_style_consistency(self):
        """Test style consistency evaluation."""
        # Consistent style prompt
        consistent_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="fantasy castle, medieval architecture, digital painting style",
            style_modifiers=["fantasy", "medieval", "digital painting"],
            negative_prompt=["modern", "sci-fi"],
            technical_params={"style": "artistic"}
        )

        score = self.analyzer.evaluate_style_consistency(consistent_prompt)
        assert score > 70

        # Inconsistent style prompt
        inconsistent_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="medieval castle with laser beams and spaceships",
            style_modifiers=["medieval", "sci-fi", "modern"],
            negative_prompt=["fantasy"],
            technical_params={}
        )

        score = self.analyzer.evaluate_style_consistency(inconsistent_prompt)
        assert score < 70

    @pytest.mark.asyncio
    async def test_assess_generation_quality(self):
        """Test complete quality assessment."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '''
        {
            "accuracy_assessment": "Good representation of the described scene",
            "accuracy_score": 85,
            "style_feedback": "Style is consistent and well-executed",
            "style_score": 88,
            "emotional_feedback": "Captures the intended emotional tone well",
            "emotional_score": 82,
            "technical_feedback": "High technical quality with good composition",
            "technical_score": 90,
            "areas_for_improvement": ["Could enhance dramatic lighting"],
            "strengths": ["Excellent character detail", "Strong composition"],
            "overall_score": 86
        }
        '''
        self.mock_llm.ainvoke.return_value = mock_response

        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="fantasy warrior in battle",
            style_modifiers=["fantasy", "dramatic"],
            negative_prompt=["low quality"],
            technical_params={"quality": "hd"}
        )

        emotional_moment = EmotionalMoment(
            text_excerpt="The warrior charged into battle",
            context="epic battle scene",
            emotional_tones=[EmotionalTone.COURAGE, EmotionalTone.EXCITEMENT],
            intensity_score=0.8,
            start_position=0,
            end_position=30
        )

        generation_result = {
            "image_url": "test_url",
            "metadata": {"size": "1024x1024"}
        }

        assessment = await self.analyzer.assess_generation_quality(
            prompt, generation_result, emotional_moment
        )

        assert isinstance(assessment, QualityAssessment)
        assert assessment.overall_score == 86
        assert "Excellent character detail" in assessment.strengths

    @pytest.mark.asyncio
    async def test_assess_generation_quality_fallback(self):
        """Test quality assessment fallback on LLM error."""
        # Mock LLM failure
        self.mock_llm.ainvoke.side_effect = Exception("LLM error")

        prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="test prompt",
            style_modifiers=["test"],
            negative_prompt=[],
            technical_params={}
        )

        emotional_moment = EmotionalMoment(
            text_excerpt="test excerpt",
            context="test",
            emotional_tones=[EmotionalTone.NEUTRAL],
            intensity_score=0.5,
            start_position=0,
            end_position=10
        )

        generation_result = {"image_url": "test_url"}

        assessment = await self.analyzer.assess_generation_quality(
            prompt, generation_result, emotional_moment
        )

        # Should return fallback assessment
        assert isinstance(assessment, QualityAssessment)
        assert 60 <= assessment.overall_score <= 75  # Fallback range


class TestPromptIterator:
    """Test PromptIterator functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.mock_llm = AsyncMock()
        self.iterator = PromptIterator(self.mock_llm)

    def test_identify_iteration_reasons(self):
        """Test iteration reason identification."""
        # Low quality assessment
        poor_assessment = QualityAssessment(
            overall_score=55,
            accuracy_score=50,
            style_consistency=60,
            emotional_alignment=55,
            technical_quality=50,
            prompt_effectiveness=60,
            areas_for_improvement=["poor lighting", "weak composition"],
            strengths=[],
            recommendations=["improve lighting"]
        )

        reasons = self.iterator.identify_iteration_reasons(poor_assessment)

        assert IterationReason.LOW_OVERALL_QUALITY in reasons
        assert IterationReason.POOR_ACCURACY in reasons
        assert IterationReason.WEAK_TECHNICAL_QUALITY in reasons

    @pytest.mark.asyncio
    async def test_improve_prompt(self):
        """Test prompt improvement."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '''
        {
            "improved_prompt": "A majestic fantasy castle on a high mountain peak, dramatic storm clouds, lightning illuminating the towers, epic fantasy art style, high detail, cinematic composition",
            "style_modifiers": ["fantasy", "dramatic", "epic", "cinematic", "high detail"],
            "negative_prompt": ["low quality", "blurry", "amateur", "flat lighting"],
            "technical_params": {"quality": "hd", "size": "1024x1024", "style": "cinematic"},
            "improvements_made": ["enhanced dramatic elements", "improved composition guidance", "added cinematic style"],
            "reasoning": "Added more dramatic and cinematic elements to better capture the epic nature of the scene"
        }
        '''
        self.mock_llm.ainvoke.return_value = mock_response

        original_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="castle on mountain",
            style_modifiers=["fantasy"],
            negative_prompt=["low quality"],
            technical_params={}
        )

        assessment = QualityAssessment(
            overall_score=60,
            accuracy_score=55,
            style_consistency=65,
            emotional_alignment=60,
            technical_quality=55,
            prompt_effectiveness=60,
            areas_for_improvement=["needs more drama"],
            strengths=[],
            recommendations=["add dramatic elements"]
        )

        iteration_reasons = [IterationReason.INSUFFICIENT_DETAIL]

        improved_prompt = await self.iterator.improve_prompt(
            original_prompt, assessment, iteration_reasons
        )

        assert isinstance(improved_prompt, IllustrationPrompt)
        assert len(improved_prompt.prompt) > len(original_prompt.prompt)
        assert "cinematic" in improved_prompt.style_modifiers


class TestFeedbackSystem:
    """Test FeedbackSystem integration."""

    def setup_method(self):
        """Setup for each test."""
        self.mock_llm = AsyncMock()
        self.feedback_system = FeedbackSystem(self.mock_llm)

    @pytest.mark.asyncio
    async def test_process_feedback_cycle(self):
        """Test complete feedback cycle."""
        # Mock analyzer and iterator
        self.feedback_system.analyzer = Mock()
        self.feedback_system.iterator = AsyncMock()

        # Mock quality assessment
        mock_assessment = QualityAssessment(
            overall_score=65,
            accuracy_score=60,
            style_consistency=70,
            emotional_alignment=65,
            technical_quality=60,
            prompt_effectiveness=65,
            areas_for_improvement=["needs improvement"],
            strengths=["good basic structure"],
            recommendations=["enhance details"]
        )
        self.feedback_system.analyzer.assess_generation_quality = AsyncMock(return_value=mock_assessment)

        # Mock improved prompt
        mock_improved_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="improved prompt with more details",
            style_modifiers=["improved", "detailed"],
            negative_prompt=["low quality"],
            technical_params={"quality": "hd"}
        )
        self.feedback_system.iterator.improve_prompt = AsyncMock(return_value=mock_improved_prompt)

        original_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="basic prompt",
            style_modifiers=["basic"],
            negative_prompt=[],
            technical_params={}
        )

        emotional_moment = EmotionalMoment(
            text_excerpt="test scene",
            context="test context",
            emotional_tones=[EmotionalTone.NEUTRAL],
            intensity_score=0.5,
            start_position=0,
            end_position=10
        )

        generation_result = {"image_url": "test_url"}

        iteration = await self.feedback_system.process_feedback_cycle(
            original_prompt,
            generation_result,
            emotional_moment
        )

        assert isinstance(iteration, PromptIteration)
        assert iteration.iteration_number == 1
        assert iteration.improved_prompt == mock_improved_prompt
        assert iteration.quality_assessment == mock_assessment

    @pytest.mark.asyncio
    async def test_iterative_improvement(self):
        """Test iterative improvement process."""
        # Mock components
        self.feedback_system.analyzer = Mock()
        self.feedback_system.iterator = AsyncMock()

        # First iteration - needs improvement
        first_assessment = QualityAssessment(
            overall_score=60,
            accuracy_score=55,
            style_consistency=65,
            emotional_alignment=60,
            technical_quality=55,
            prompt_effectiveness=60,
            areas_for_improvement=["needs work"],
            strengths=[],
            recommendations=["improve quality"]
        )

        # Second iteration - good enough
        second_assessment = QualityAssessment(
            overall_score=85,
            accuracy_score=82,
            style_consistency=88,
            emotional_alignment=85,
            technical_quality=83,
            prompt_effectiveness=87,
            areas_for_improvement=[],
            strengths=["excellent improvement"],
            recommendations=[]
        )

        self.feedback_system.analyzer.assess_generation_quality = AsyncMock(
            side_effect=[first_assessment, second_assessment]
        )

        # Mock improved prompts
        first_improved = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="first improvement",
            style_modifiers=["improved"],
            negative_prompt=["low quality"],
            technical_params={}
        )

        second_improved = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="second improvement, much better",
            style_modifiers=["excellent", "detailed"],
            negative_prompt=["low quality", "amateur"],
            technical_params={"quality": "hd"}
        )

        self.feedback_system.iterator.improve_prompt = AsyncMock(
            side_effect=[first_improved, second_improved]
        )

        original_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="original prompt",
            style_modifiers=["basic"],
            negative_prompt=[],
            technical_params={}
        )

        emotional_moment = EmotionalMoment(
            text_excerpt="test scene",
            context="test",
            emotional_tones=[EmotionalTone.NEUTRAL],
            intensity_score=0.5,
            start_position=0,
            end_position=10
        )

        generation_result = {"image_url": "test_url"}

        report = await self.feedback_system.iterative_improvement(
            original_prompt,
            lambda p: generation_result,  # Mock generation function
            emotional_moment,
            max_iterations=3,
            target_quality=80
        )

        assert isinstance(report, QualityReport)
        assert len(report.iterations) == 2  # Should improve until target reached
        assert report.final_quality_score >= 80

    def test_quality_threshold_checking(self):
        """Test quality threshold checking."""
        # Test different threshold levels
        assert QualityThreshold.meets_threshold(85, QualityThreshold.EXCELLENT)
        assert QualityThreshold.meets_threshold(75, QualityThreshold.GOOD)
        assert QualityThreshold.meets_threshold(65, QualityThreshold.ACCEPTABLE)
        assert not QualityThreshold.meets_threshold(55, QualityThreshold.ACCEPTABLE)

        # Test threshold values
        assert QualityThreshold.EXCELLENT.value == 85
        assert QualityThreshold.GOOD.value == 75
        assert QualityThreshold.ACCEPTABLE.value == 65


class TestPromptIteration:
    """Test PromptIteration data structure."""

    def test_prompt_iteration_creation(self):
        """Test creating a prompt iteration."""
        original_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="original",
            style_modifiers=["basic"],
            negative_prompt=[],
            technical_params={}
        )

        improved_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="improved version",
            style_modifiers=["enhanced", "detailed"],
            negative_prompt=["low quality"],
            technical_params={"quality": "hd"}
        )

        assessment = QualityAssessment(
            overall_score=75,
            accuracy_score=70,
            style_consistency=80,
            emotional_alignment=75,
            technical_quality=70,
            prompt_effectiveness=75,
            areas_for_improvement=[],
            strengths=["good improvement"],
            recommendations=[]
        )

        iteration = PromptIteration(
            iteration_number=1,
            original_prompt=original_prompt,
            improved_prompt=improved_prompt,
            quality_assessment=assessment,
            iteration_reasons=[IterationReason.INSUFFICIENT_DETAIL],
            improvements_made=["added more detail"],
            timestamp=datetime.now()
        )

        assert iteration.iteration_number == 1
        assert iteration.improved_prompt == improved_prompt
        assert iteration.quality_assessment == assessment
        assert len(iteration.improvements_made) == 1


class TestQualityReport:
    """Test QualityReport data structure."""

    def test_quality_report_creation(self):
        """Test creating a quality report."""
        iterations = [
            PromptIteration(
                iteration_number=1,
                original_prompt=Mock(),
                improved_prompt=Mock(),
                quality_assessment=Mock(),
                iteration_reasons=[IterationReason.LOW_OVERALL_QUALITY],
                improvements_made=["first improvement"],
                timestamp=datetime.now()
            )
        ]

        report = QualityReport(
            original_prompt=Mock(),
            final_prompt=Mock(),
            original_quality_score=60,
            final_quality_score=85,
            iterations=iterations,
            total_iterations=1,
            improvement_achieved=25,
            target_quality_reached=True,
            processing_time_seconds=5.2,
            timestamp=datetime.now()
        )

        assert report.total_iterations == 1
        assert report.improvement_achieved == 25
        assert report.target_quality_reached is True
        assert len(report.iterations) == 1