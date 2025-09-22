"""Simple tests for the quality feedback system."""

import pytest
from datetime import datetime

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


class TestQualityMetric:
    """Test QualityMetric enum."""

    def test_quality_metric_values(self):
        """Test quality metric enum values."""
        assert QualityMetric.VISUAL_ACCURACY == "visual_accuracy"
        assert QualityMetric.EMOTIONAL_RESONANCE == "emotional_resonance"
        assert QualityMetric.ARTISTIC_CONSISTENCY == "artistic_consistency"
        assert QualityMetric.TECHNICAL_QUALITY == "technical_quality"
        assert QualityMetric.NARRATIVE_RELEVANCE == "narrative_relevance"


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
            feedback_notes="Excellent character portrayal with good emotional depth",
            improvement_suggestions=["Enhance background details", "Adjust lighting"],
            provider=ImageProvider.DALLE,
            timestamp=datetime.now().isoformat()
        )

        assert assessment.prompt_id == "test-prompt-123"
        assert assessment.generation_success is True
        assert assessment.quality_scores[QualityMetric.VISUAL_ACCURACY] == 0.85
        assert len(assessment.improvement_suggestions) == 2
        assert assessment.provider == ImageProvider.DALLE

    def test_quality_assessment_failure_case(self):
        """Test quality assessment for failed generation."""
        assessment = QualityAssessment(
            prompt_id="failed-prompt-456",
            generation_success=False,
            quality_scores={},
            feedback_notes="Generation failed due to content policy violation",
            improvement_suggestions=["Revise prompt to avoid policy issues"],
            provider=ImageProvider.DALLE,
            timestamp=datetime.now().isoformat()
        )

        assert assessment.generation_success is False
        assert len(assessment.quality_scores) == 0
        assert "policy violation" in assessment.feedback_notes


class TestPromptPerformance:
    """Test PromptPerformance data structure."""

    def test_prompt_performance_creation(self):
        """Test creating a prompt performance record."""
        avg_scores = {
            QualityMetric.VISUAL_ACCURACY: 0.82,
            QualityMetric.EMOTIONAL_RESONANCE: 0.79,
            QualityMetric.ARTISTIC_CONSISTENCY: 0.85,
            QualityMetric.TECHNICAL_QUALITY: 0.77,
            QualityMetric.NARRATIVE_RELEVANCE: 0.84
        }

        performance = PromptPerformance(
            prompt_template="A {subject} in {setting}, {artistic_style}",
            provider=ImageProvider.DALLE,
            success_rate=0.89,
            avg_quality_scores=avg_scores,
            usage_count=42,
            last_updated=datetime.now().isoformat()
        )

        assert performance.prompt_template == "A {subject} in {setting}, {artistic_style}"
        assert performance.provider == ImageProvider.DALLE
        assert performance.success_rate == 0.89
        assert performance.usage_count == 42
        assert performance.avg_quality_scores[QualityMetric.VISUAL_ACCURACY] == 0.82

    def test_prompt_performance_low_success_rate(self):
        """Test prompt performance with low success rate."""
        performance = PromptPerformance(
            prompt_template="Bad prompt template",
            provider=ImageProvider.FLUX,
            success_rate=0.25,
            avg_quality_scores={
                QualityMetric.VISUAL_ACCURACY: 0.40,
                QualityMetric.TECHNICAL_QUALITY: 0.35
            },
            usage_count=8,
            last_updated=datetime.now().isoformat()
        )

        # Should indicate poor performance
        assert performance.success_rate < 0.5
        assert all(score < 0.6 for score in performance.avg_quality_scores.values())


class TestQualityMetrics:
    """Test quality metrics calculation."""

    def test_average_quality_score(self):
        """Test calculating average quality score."""
        scores = {
            QualityMetric.VISUAL_ACCURACY: 0.85,
            QualityMetric.EMOTIONAL_RESONANCE: 0.80,
            QualityMetric.ARTISTIC_CONSISTENCY: 0.90,
            QualityMetric.TECHNICAL_QUALITY: 0.75,
            QualityMetric.NARRATIVE_RELEVANCE: 0.70
        }

        # Calculate average
        average = sum(scores.values()) / len(scores)
        assert 0.78 <= average <= 0.82  # Should be around 0.8

    def test_quality_threshold_evaluation(self):
        """Test evaluating quality against thresholds."""
        scores = {
            QualityMetric.VISUAL_ACCURACY: 0.92,
            QualityMetric.EMOTIONAL_RESONANCE: 0.88,
            QualityMetric.ARTISTIC_CONSISTENCY: 0.95,
            QualityMetric.TECHNICAL_QUALITY: 0.85,
            QualityMetric.NARRATIVE_RELEVANCE: 0.90
        }

        # Check if all scores meet high threshold
        HIGH_THRESHOLD = 0.80
        meets_threshold = all(score >= HIGH_THRESHOLD for score in scores.values())
        assert meets_threshold is True

        # Check against very high threshold
        VERY_HIGH_THRESHOLD = 0.95
        meets_very_high = all(score >= VERY_HIGH_THRESHOLD for score in scores.values())
        assert meets_very_high is False  # Not all scores are >= 0.95