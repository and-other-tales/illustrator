"""Dataclasses and enums for quality feedback and prompt iteration."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum
from illustrator.models import IllustrationPrompt, ImageProvider

class QualityMetric(str, Enum):
    VISUAL_ACCURACY = "visual_accuracy"
    EMOTIONAL_RESONANCE = "emotional_resonance"
    ARTISTIC_CONSISTENCY = "artistic_consistency"
    TECHNICAL_QUALITY = "technical_quality"
    NARRATIVE_RELEVANCE = "narrative_relevance"

class IterationReason(str, Enum):
    LOW_OVERALL_QUALITY = "low_overall_quality"
    POOR_ACCURACY = "poor_accuracy"
    WEAK_TECHNICAL_QUALITY = "weak_technical_quality"
    INSUFFICIENT_DETAIL = "insufficient_detail"
    WEAK_EMOTIONAL_RESONANCE = "weak_emotional_resonance"
    INCONSISTENT_STYLE = "inconsistent_style"

@dataclass
class QualityAssessment:
    prompt_id: Optional[str] = None
    generation_success: Optional[bool] = None
    quality_scores: Optional[Dict[QualityMetric, float]] = None
    feedback_notes: str = ""
    improvement_suggestions: Optional[List[str]] = None
    provider: Optional[ImageProvider] = None
    timestamp: Optional[str] = None
    overall_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    style_consistency: Optional[float] = None
    emotional_alignment: Optional[float] = None
    technical_quality: Optional[float] = None
    prompt_effectiveness: Optional[float] = None
    areas_for_improvement: Optional[List[str]] = None
    strengths: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    def needs_improvement(self) -> bool:
        if self.overall_score is not None:
            try:
                return float(self.overall_score) < 75.0
            except Exception:
                pass
        if self.quality_scores:
            avg = sum(self.quality_scores.values()) / len(self.quality_scores)
            return avg < 0.75
        return True

@dataclass
class PromptPerformance:
    prompt_template: str
    provider: ImageProvider
    success_rate: float
    avg_quality_scores: Dict[QualityMetric, float]
    usage_count: int
    last_updated: str

@dataclass
class PromptIteration:
    iteration_number: int
    original_prompt: IllustrationPrompt
    improved_prompt: IllustrationPrompt
    quality_assessment: QualityAssessment
    iteration_reasons: List[IterationReason]
    improvements_made: List[str]
    timestamp: str

@dataclass
class QualityReport:
    session_id: str | None = None
    initial_prompt: Optional[IllustrationPrompt] = None
    original_prompt: Optional[IllustrationPrompt] = None
    original_quality_score: Optional[float] = None
    summary: Optional[str] = None
    final_prompt: Optional[IllustrationPrompt] = None
    iterations: Optional[List[PromptIteration]] = None
    total_iterations: int = 0
    total_improvements: int = 0
    improvement_achieved: Optional[float] = None
    final_quality_score: float = 0.0
    target_quality_reached: bool = False
    processing_time_seconds: float = 0.0
    timestamp: Optional[str] = None
    processing_time: float = 0.0
    success: bool = False
