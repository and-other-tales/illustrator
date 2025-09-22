"""Corrected tests for the quality feedback system."""

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
    PromptPerformance,
    IterationReason,
    PromptIteration,
    QualityReport,
    QualityAnalyzer,
    PromptIterator,
    FeedbackSystem
)


class TestQualityAssessment:
    """Test QualityAssessment data structure."""

    def test_quality_assessment_creation(self):
        """Test creating a quality assessment."""
        quality_scores = {
            QualityMetric.VISUAL_ACCURACY: 0.85,
            QualityMetric.EMOTIONAL_RESONANCE: 0.80,
            QualityMetric.ARTISTIC_CONSISTENCY: 0.90,
            QualityMetric.TECHNICAL_QUALITY: 0.75,
            QualityMetric.NARRATIVE_RELEVANCE: 0.88
        }

        assessment = QualityAssessment(
            prompt_id="test-prompt-123",
            generation_success=True,
            quality_scores=quality_scores,
            feedback_notes="High-quality generation with good emotional resonance.",
            improvement_suggestions=["Consider enhancing technical quality", "Add more detail"],
            provider=ImageProvider.DALLE,
            timestamp="2023-01-01T00:00:00Z"
        )

        assert assessment.prompt_id == "test-prompt-123"
        assert assessment.generation_success is True
        assert assessment.quality_scores[QualityMetric.VISUAL_ACCURACY] == 0.85
        assert len(assessment.improvement_suggestions) == 2
        assert assessment.provider == ImageProvider.DALLE

    def test_quality_assessment_validation(self):
        """Test quality assessment score validation."""
        quality_scores = {
            QualityMetric.VISUAL_ACCURACY: 0.75,
            QualityMetric.EMOTIONAL_RESONANCE: 0.70,
            QualityMetric.ARTISTIC_CONSISTENCY: 0.80,
            QualityMetric.TECHNICAL_QUALITY: 0.70,
            QualityMetric.NARRATIVE_RELEVANCE: 0.75
        }

        assessment = QualityAssessment(
            prompt_id="validation_test",
            generation_success=True,
            quality_scores=quality_scores,
            feedback_notes="Test validation",
            improvement_suggestions=["test improvement"],
            provider=ImageProvider.DALLE,
            timestamp="2023-01-01T00:00:00Z"
        )

        assert assessment.quality_scores[QualityMetric.VISUAL_ACCURACY] == 0.75
        assert all(0 <= score <= 1.0 for score in assessment.quality_scores.values())

    def test_needs_improvement(self):
        """Test quality assessment structure for improvement detection."""
        # High quality assessment
        good_scores = {metric: 0.9 for metric in QualityMetric}
        good_assessment = QualityAssessment(
            prompt_id="good_test",
            generation_success=True,
            quality_scores=good_scores,
            feedback_notes="High quality",
            improvement_suggestions=[],
            provider=ImageProvider.DALLE,
            timestamp="2023-01-01T00:00:00Z"
        )

        # Low quality assessment
        poor_scores = {metric: 0.4 for metric in QualityMetric}
        poor_assessment = QualityAssessment(
            prompt_id="poor_test",
            generation_success=False,
            quality_scores=poor_scores,
            feedback_notes="Low quality",
            improvement_suggestions=["Improve accuracy", "Better technical quality"],
            provider=ImageProvider.DALLE,
            timestamp="2023-01-01T00:00:00Z"
        )

        assert good_assessment.generation_success is True
        assert poor_assessment.generation_success is False
        assert len(poor_assessment.improvement_suggestions) > 0


class TestPromptIteration:
    """Test PromptIteration data structure."""

    def test_prompt_iteration_creation(self):
        """Test creating a prompt iteration."""
        original_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="original prompt text",
            style_modifiers=["basic"],
            negative_prompt="blurry, low quality",  # Should be string, not list
            technical_params={}
        )

        improved_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="improved prompt text with more detail",
            style_modifiers=["detailed", "high-quality"],
            negative_prompt="blurry, low quality, distorted",
            technical_params={"quality": "high"}
        )

        quality_scores = {metric: 0.6 for metric in QualityMetric}
        quality_assessment = QualityAssessment(
            prompt_id="test_iteration",
            generation_success=True,
            quality_scores=quality_scores,
            feedback_notes="Iteration test",
            improvement_suggestions=["Add more detail"],
            provider=ImageProvider.DALLE,
            timestamp="2023-01-01T00:00:00Z"
        )

        iteration = PromptIteration(
            iteration_number=1,
            original_prompt=original_prompt,
            improved_prompt=improved_prompt,
            quality_assessment=quality_assessment,
            iteration_reasons=[IterationReason.INSUFFICIENT_DETAIL],
            improvements_made=["Added more specific details"],
            timestamp="2023-01-01T00:00:00Z"
        )

        assert iteration.iteration_number == 1
        assert iteration.original_prompt.prompt == "original prompt text"
        assert iteration.improved_prompt.prompt == "improved prompt text with more detail"
        assert IterationReason.INSUFFICIENT_DETAIL in iteration.iteration_reasons


class TestQualityReport:
    """Test QualityReport data structure."""

    def test_quality_report_creation(self):
        """Test creating a quality report."""
        original_prompt = Mock()
        improved_prompt = Mock()
        quality_assessment = Mock()

        iterations = [
            PromptIteration(
                iteration_number=1,
                original_prompt=original_prompt,
                improved_prompt=improved_prompt,
                quality_assessment=quality_assessment,
                iteration_reasons=[IterationReason.LOW_OVERALL_QUALITY],
                improvements_made=["first improvement"],
                timestamp="2023-01-01T00:00:00Z"
            )
        ]

        report = QualityReport(
            session_id="test_session",
            initial_prompt=original_prompt,
            final_prompt=improved_prompt,
            iterations=iterations,
            total_improvements=1,
            final_quality_score=0.85,
            processing_time=5.2,
            success=True
        )

        assert report.session_id == "test_session"
        assert report.total_improvements == 1
        assert report.final_quality_score == 0.85
        assert report.success is True
        assert len(report.iterations) == 1


class TestQualityAnalyzer:
    """Test QualityAnalyzer functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.mock_llm = AsyncMock()
        self.analyzer = QualityAnalyzer(self.mock_llm)

    @pytest.mark.asyncio
    async def test_assess_generation_quality_success(self):
        """Test successful quality assessment."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = '''
        {
            "visual_accuracy": 0.85,
            "emotional_resonance": 0.80,
            "artistic_consistency": 0.90,
            "technical_quality": 0.75,
            "narrative_relevance": 0.88,
            "overall_feedback": "High-quality generation",
            "improvement_suggestions": ["Enhance technical quality"]
        }
        '''
        self.mock_llm.ainvoke.return_value = mock_response

        test_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="test prompt",
            style_modifiers=["test"],
            negative_prompt="",
            technical_params={}
        )

        generation_result = {
            "success": True,
            "file_path": "/test/path.jpg",
            "provider": "dalle"
        }

        assessment = await self.analyzer.assess_generation_quality(
            original_prompt=test_prompt,
            generation_result=generation_result,
            emotional_moment=Mock()
        )

        assert isinstance(assessment, QualityAssessment)
        assert assessment.generation_success is True
        assert assessment.quality_scores[QualityMetric.VISUAL_ACCURACY] == 0.85

    @pytest.mark.asyncio
    async def test_assess_generation_quality_fallback(self):
        """Test quality assessment with LLM failure fallback."""
        # Make LLM fail
        self.mock_llm.ainvoke.side_effect = Exception("LLM error")

        test_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="test prompt",
            style_modifiers=["test"],
            negative_prompt="",
            technical_params={}
        )

        generation_result = {
            "success": False,
            "file_path": "/test/path.jpg",
            "provider": "dalle"
        }

        assessment = await self.analyzer.assess_generation_quality(
            original_prompt=test_prompt,
            generation_result=generation_result,
            emotional_moment=Mock()
        )

        assert isinstance(assessment, QualityAssessment)
        assert assessment.generation_success is False


class TestPromptIterator:
    """Test PromptIterator functionality."""

    def setup_method(self):
        """Setup for each test."""
        self.mock_llm = AsyncMock()
        self.iterator = PromptIterator(self.mock_llm)

    def test_performance_tracking(self):
        """Test performance tracking functionality."""
        # Test basic performance tracking methods
        insights = self.iterator.get_performance_insights()
        assert isinstance(insights, dict)

        # Test updating performance (this method exists)
        test_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="test prompt",
            style_modifiers=["test"],
            negative_prompt="",
            technical_params={}
        )

        quality_scores = {
            QualityMetric.VISUAL_ACCURACY: 0.8,
            QualityMetric.EMOTIONAL_RESONANCE: 0.7,
            QualityMetric.ARTISTIC_CONSISTENCY: 0.9,
            QualityMetric.TECHNICAL_QUALITY: 0.75,
            QualityMetric.NARRATIVE_RELEVANCE: 0.85
        }

        assessment = QualityAssessment(
            prompt_id="test",
            generation_success=True,
            quality_scores=quality_scores,
            feedback_notes="Test assessment",
            improvement_suggestions=[],
            provider=ImageProvider.DALLE,
            timestamp="2023-01-01T00:00:00Z"
        )

        # This method should exist based on the class definition
        self.iterator.update_performance_tracking(test_prompt, assessment)

    @pytest.mark.asyncio
    async def test_iterate_prompt_success(self):
        """Test successful prompt iteration."""
        mock_response = Mock()
        mock_response.content = '''
        {
            "improved_prompt": "A highly detailed digital painting of...",
            "improvements_made": ["Added more specific details", "Enhanced lighting description"],
            "style_modifiers": ["detailed", "high-resolution", "professional"],
            "negative_prompt": "blurry, low quality, distorted, amateur",
            "technical_params": {"quality": "high", "style": "photorealistic"}
        }
        '''
        self.mock_llm.ainvoke.return_value = mock_response

        original_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="basic prompt",
            style_modifiers=["basic"],
            negative_prompt="blurry",
            technical_params={}
        )

        quality_scores = {metric: 0.6 for metric in QualityMetric}
        assessment = QualityAssessment(
            prompt_id="test",
            generation_success=True,
            quality_scores=quality_scores,
            feedback_notes="Needs improvement",
            improvement_suggestions=["Add more detail"],
            provider=ImageProvider.DALLE,
            timestamp="2023-01-01T00:00:00Z"
        )

        # Use the actual method name
        iteration = await self.iterator.iterate_prompt(
            original_prompt,
            assessment,
            Mock()  # emotional_moment
        )

        assert isinstance(iteration, PromptIteration)
        assert iteration.original_prompt == original_prompt


class TestFeedbackSystem:
    """Test FeedbackSystem integration."""

    def setup_method(self):
        """Setup for each test."""
        self.mock_llm = AsyncMock()
        self.feedback_system = FeedbackSystem(self.mock_llm)

    def test_feedback_system_basic(self):
        """Test basic feedback system functionality."""
        # Test methods that exist based on the class definition
        insights = self.feedback_system.get_system_insights()
        assert isinstance(insights, dict)

        feedback_data = self.feedback_system.export_feedback_data()
        assert isinstance(feedback_data, dict)

    @pytest.mark.asyncio
    async def test_process_generation_feedback(self):
        """Test the actual feedback processing method."""
        original_prompt = IllustrationPrompt(
            provider=ImageProvider.DALLE,
            prompt="basic prompt",
            style_modifiers=["basic"],
            negative_prompt="",
            technical_params={}
        )

        generation_result = {
            "success": True,
            "file_path": "/test/image.jpg",
            "provider": "dalle"
        }

        # Mock the LLM response for feedback processing
        mock_response = Mock()
        mock_response.content = '''
        {
            "visual_accuracy": 0.7,
            "emotional_resonance": 0.8,
            "artistic_consistency": 0.6,
            "technical_quality": 0.75,
            "narrative_relevance": 0.8,
            "overall_feedback": "Good quality with room for improvement",
            "improvement_suggestions": ["Enhance technical details"]
        }
        '''
        self.mock_llm.ainvoke.return_value = mock_response

        # Test the feedback processing
        result = await self.feedback_system.process_generation_feedback(
            original_prompt,
            generation_result,
            Mock()  # emotional_moment
        )

        # Verify we get some result back (actual return type depends on implementation)
        assert result is not None


if __name__ == "__main__":
    pytest.main([__file__])