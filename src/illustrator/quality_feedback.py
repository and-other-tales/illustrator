"""Quality feedback and prompt iteration system for continuous improvement."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from illustrator.utils import parse_llm_json

from illustrator.models import (
    EmotionalMoment,
    IllustrationPrompt,
    ImageProvider,
)

logger = logging.getLogger(__name__)


class QualityMetric(str, Enum):
    """Quality assessment metrics for generated images."""
    VISUAL_ACCURACY = "visual_accuracy"
    EMOTIONAL_RESONANCE = "emotional_resonance"
    ARTISTIC_CONSISTENCY = "artistic_consistency"
    TECHNICAL_QUALITY = "technical_quality"
    NARRATIVE_RELEVANCE = "narrative_relevance"


class IterationReason(str, Enum):
    """Reasons for prompt iteration."""
    LOW_OVERALL_QUALITY = "low_overall_quality"
    POOR_ACCURACY = "poor_accuracy"
    WEAK_TECHNICAL_QUALITY = "weak_technical_quality"
    INSUFFICIENT_DETAIL = "insufficient_detail"
    WEAK_EMOTIONAL_RESONANCE = "weak_emotional_resonance"
    INCONSISTENT_STYLE = "inconsistent_style"


@dataclass
class QualityAssessment:
    """Quality assessment for a generated image.

    This class is intentionally flexible to accept both the modern shape
    (quality_scores dict) and legacy flat numeric fields used by older tests
    and code paths.
    """
    # Modern fields
    prompt_id: Optional[str] = None
    generation_success: Optional[bool] = None
    quality_scores: Optional[Dict[QualityMetric, float]] = None  # 0.0 to 1.0
    feedback_notes: str = ""
    improvement_suggestions: Optional[List[str]] = None
    provider: Optional[ImageProvider] = None
    timestamp: Optional[str] = None

    # Legacy / convenience numeric fields (0-100 scale in some tests)
    overall_score: Optional[float] = None
    accuracy_score: Optional[float] = None
    style_consistency: Optional[float] = None
    emotional_alignment: Optional[float] = None
    technical_quality: Optional[float] = None
    prompt_effectiveness: Optional[float] = None

    # legacy lists
    areas_for_improvement: Optional[List[str]] = None
    strengths: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None

    def needs_improvement(self) -> bool:
        """Return True if assessment indicates the prompt needs improvement.

        Priority: use overall_score if present (0-100), otherwise use
        quality_scores average (0.0-1.0) with threshold ~0.75.
        """
        if self.overall_score is not None:
            try:
                return float(self.overall_score) < 75.0
            except Exception:
                pass

        if self.quality_scores:
            avg = sum(self.quality_scores.values()) / len(self.quality_scores)
            return avg < 0.75

        # Fall back to conservative default
        return True


@dataclass
class PromptPerformance:
    """Track prompt performance over time."""
    prompt_template: str
    provider: ImageProvider
    success_rate: float
    avg_quality_scores: Dict[QualityMetric, float]
    usage_count: int
    last_updated: str


@dataclass
class PromptIteration:
    """Represents a single prompt improvement iteration."""
    iteration_number: int
    original_prompt: IllustrationPrompt
    improved_prompt: IllustrationPrompt
    quality_assessment: QualityAssessment
    iteration_reasons: List[IterationReason]
    improvements_made: List[str]
    timestamp: str


@dataclass
class QualityReport:
    """Comprehensive quality report for a prompt improvement session."""
    # Make session_id and initial_prompt optional for backward compatibility with tests
    session_id: str | None = None
    initial_prompt: Optional[IllustrationPrompt] = None
    # legacy/test-expected fields
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
    # modern fields
    processing_time: float = 0.0
    success: bool = False


class QualityAnalyzer:
    """Analyzes generated image quality and provides feedback."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def assess_generation_quality(
        self,
        original_prompt: IllustrationPrompt,
        generation_result: Dict[str, Any],
        emotional_moment: EmotionalMoment
    ) -> QualityAssessment:
        """Assess the quality of a generated image based on multiple criteria."""

        # Some callers/tests omit an explicit 'success' flag. Treat missing 'success'
        # as a successful generation attempt to allow LLM-based analysis.
        if 'success' in generation_result and not generation_result.get('success'):
            return QualityAssessment(
                prompt_id=f"{original_prompt.provider.value}_{hash(original_prompt.prompt)}",
                generation_success=False,
                quality_scores={metric: 0.0 for metric in QualityMetric},
                feedback_notes=f"Generation failed: {generation_result.get('error', 'Unknown error')}",
                improvement_suggestions=[
                    "Check API credentials and connection",
                    "Simplify prompt if too complex",
                    "Verify provider-specific parameter formats"
                ],
                provider=original_prompt.provider,
                timestamp=generation_result.get('timestamp', '')
            )

        # If we have successful generation but no image analysis capability,
        # provide basic assessment based on prompt structure
        # Perform detailed analysis (may call LLM)
        quality_scores, analysis_data = await self._analyze_prompt_quality(
            original_prompt,
            emotional_moment,
            generation_result
        )

        feedback_notes = await self._generate_feedback_notes(
            original_prompt,
            quality_scores,
            generation_result
        )

        improvement_suggestions = await self._generate_improvement_suggestions(
            original_prompt,
            quality_scores,
            emotional_moment
        )

        # Extract legacy/auxiliary fields from analysis if present
        overall_score = None
        strengths = None
        areas_for_improvement = None
        recommendations = None

        if isinstance(analysis_data, dict):
            # LLM may return various keys; support both snake_case and space/upper keys
            overall_score = analysis_data.get('overall_score') or analysis_data.get('overall') or analysis_data.get('overallScore')
            strengths = analysis_data.get('strengths') or analysis_data.get('strengths_list')
            areas_for_improvement = analysis_data.get('areas_for_improvement') or analysis_data.get('areas')
            recommendations = analysis_data.get('recommendations') or analysis_data.get('improvement_suggestions')

        # If LLM didn't provide an overall_score, derive from avg of quality_scores
        if overall_score is None and quality_scores:
            avg = sum(quality_scores.values()) / len(quality_scores)
            overall_score = int(avg * 100)

        return QualityAssessment(
            prompt_id=f"{original_prompt.provider.value}_{hash(original_prompt.prompt)}",
            generation_success=True,
            quality_scores=quality_scores,
            feedback_notes=feedback_notes,
            improvement_suggestions=improvement_suggestions,
            provider=original_prompt.provider,
            timestamp=generation_result.get('metadata', {}).get('timestamp', ''),
            overall_score=overall_score,
            strengths=strengths,
            areas_for_improvement=areas_for_improvement,
            recommendations=recommendations
        )

    async def _analyze_prompt_quality(
        self,
        prompt: IllustrationPrompt,
        emotional_moment: EmotionalMoment,
        generation_result: Dict[str, Any]
    ) -> Dict[QualityMetric, float]:
        """Analyze prompt quality using structural and content analysis."""

        system_prompt = """You are an expert in AI image generation and prompt engineering. Analyze the quality of the provided prompt based on several criteria. Rate each metric from 0.0 to 1.0:

1. VISUAL_ACCURACY: How well the prompt captures specific visual elements from the source text
2. EMOTIONAL_RESONANCE: How effectively the prompt conveys the intended emotional tone
3. ARTISTIC_CONSISTENCY: How well the prompt maintains consistent artistic style and approach
4. TECHNICAL_QUALITY: How well-structured and technically sound the prompt is
5. NARRATIVE_RELEVANCE: How relevant the prompt is to the story context

Consider:
- Clarity and specificity of visual descriptions
- Appropriate use of artistic and technical terms
- Balance between detail and creative freedom
- Provider-specific optimization

Return JSON with scores for each metric."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
Prompt to analyze: {prompt.prompt}

Provider: {prompt.provider.value}
Style modifiers: {', '.join(str(m) if not isinstance(m, tuple) else ' '.join(str(elem) for elem in m) for m in (prompt.style_modifiers or []))}
Technical params: {json.dumps(prompt.technical_params, indent=2)}

Original scene text: {emotional_moment.text_excerpt}
Emotional context: {emotional_moment.context}
Emotional tones: {[tone.value for tone in emotional_moment.emotional_tones]}
Intensity score: {emotional_moment.intensity_score}

Generation metadata: {json.dumps(generation_result.get('metadata', {}), indent=2)}

Analyze the prompt quality and return scores for each metric.""")
            ]

            response = await self.llm.ainvoke(messages)
            analysis_data = parse_llm_json(response.content)

            # Convert to proper enum keys and ensure valid scores
            quality_scores = {}
            for metric in QualityMetric:
                # Support uppercase keys, enum.value keys, or metric names
                score = None
                for key in (metric.value, metric.value.upper(), metric.name.lower(), metric.name):
                    if isinstance(analysis_data, dict) and key in analysis_data:
                        score = analysis_data.get(key)
                        break
                if score is None:
                    score = 0.5
                try:
                    quality_scores[metric] = max(0.0, min(1.0, float(score)))
                except Exception:
                    quality_scores[metric] = 0.5

            return quality_scores, analysis_data

        except Exception as e:
            logger.warning(f"Failed to analyze prompt quality: {e}")
            # Return baseline scores
            return ({metric: 0.6 for metric in QualityMetric}, {})

    # Lightweight helper methods expected by unit tests
    def analyze_prompt_structure(self, prompt: IllustrationPrompt) -> int:
        """Simple heuristic score (0-100) for prompt structure quality."""
        score = 50
        if prompt.prompt and len(prompt.prompt) > 50:
            score += 20
        if prompt.style_modifiers:
            score += min(20, 5 * len(prompt.style_modifiers))
        if prompt.technical_params:
            score += 10
        return max(0, min(100, score))

    def check_emotional_alignment(self, prompt: IllustrationPrompt, moment: EmotionalMoment) -> int:
        """Heuristic emotional alignment based on presence of emotional tone keywords."""
        score = 50
        prompt_text = (prompt.prompt or "").lower()
        matches = 0
        for tone in moment.emotional_tones:
            if tone.value.lower() in prompt_text:
                matches += 1
        if matches:
            score += int(25 * (matches / len(moment.emotional_tones)))
            # stronger intensity slightly boosts the score
            score += int(10 * moment.intensity_score)
        else:
            # No matching emotional indicators -> penalize based on intensity
            score = int(35 + 5 * (1.0 - moment.intensity_score))
        return max(0, min(100, score))

    def evaluate_style_consistency(self, prompt: IllustrationPrompt) -> int:
        """Heuristic for style consistency: penalize conflicting modifiers."""
        modifiers = [m.lower() for m in (prompt.style_modifiers or [])]
        score = 70
        # crude conflict detection: presence of both 'sci-fi' and 'medieval' etc.
        conflicts = 0
        conflict_pairs = [('sci-fi', 'medieval'), ('modern', 'fantasy'), ('cartoon', 'realistic')]
        for a, b in conflict_pairs:
            if a in modifiers and b in modifiers:
                conflicts += 1
        score -= conflicts * 20
        if modifiers and not conflicts:
            score += 10
        return max(0, min(100, score))

    async def _generate_feedback_notes(
        self,
        prompt: IllustrationPrompt,
        quality_scores: Dict[QualityMetric, float],
        generation_result: Dict[str, Any]
    ) -> str:
        """Generate detailed feedback notes based on quality assessment."""

        system_prompt = """Generate concise feedback notes about the prompt and generation quality. Focus on:
1. What worked well in the prompt
2. Areas for improvement
3. Provider-specific considerations
4. Technical or artistic issues

Keep feedback constructive and actionable. Return a brief paragraph of 2-3 sentences."""

        try:
            # Identify strengths and weaknesses
            strengths = [metric.value for metric, score in quality_scores.items() if score >= 0.7]
            weaknesses = [metric.value for metric, score in quality_scores.items() if score < 0.5]

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
Prompt: {prompt.prompt[:200]}...
Provider: {prompt.provider.value}

Quality scores:
{json.dumps({k.value: v for k, v in quality_scores.items()}, indent=2)}

Strengths: {', '.join(strengths) if strengths else 'None identified'}
Weaknesses: {', '.join(weaknesses) if weaknesses else 'None identified'}

Generation success: {generation_result.get('success', False)}
""")
            ]

            response = await self.llm.ainvoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.warning(f"Failed to generate feedback notes: {e}")
            avg_score = sum(quality_scores.values()) / len(quality_scores)
            if avg_score >= 0.7:
                return "Prompt performed well overall with good technical structure and visual clarity."
            elif avg_score >= 0.5:
                return "Prompt showed moderate effectiveness but could benefit from enhanced visual detail and emotional specificity."
            else:
                return "Prompt needs significant improvement in visual accuracy and emotional resonance."

    async def _generate_improvement_suggestions(
        self,
        prompt: IllustrationPrompt,
        quality_scores: Dict[QualityMetric, float],
        emotional_moment: EmotionalMoment
    ) -> List[str]:
        """Generate specific improvement suggestions for the prompt."""

        suggestions = []

        # Add suggestions based on quality scores
        if quality_scores[QualityMetric.VISUAL_ACCURACY] < 0.6:
            suggestions.append("Add more specific visual details from the source text")
            suggestions.append("Include clearer descriptions of character poses and expressions")

        if quality_scores[QualityMetric.EMOTIONAL_RESONANCE] < 0.6:
            suggestions.append("Strengthen emotional language and atmospheric descriptions")
            suggestions.append("Better align visual elements with the emotional tone")

        if quality_scores[QualityMetric.ARTISTIC_CONSISTENCY] < 0.6:
            suggestions.append("Ensure consistent artistic style throughout the prompt")
            suggestions.append("Review style modifiers for coherence")

        if quality_scores[QualityMetric.TECHNICAL_QUALITY] < 0.6:
            suggestions.append("Optimize technical parameters for the target provider")
            suggestions.append("Improve prompt structure and clarity")

        if quality_scores[QualityMetric.NARRATIVE_RELEVANCE] < 0.6:
            suggestions.append("Strengthen connection to story context and narrative")
            suggestions.append("Focus on story-relevant visual elements")

        # Provider-specific suggestions
        if prompt.provider == ImageProvider.DALLE:
            if len(prompt.prompt) > 300:
                suggestions.append("Simplify prompt for DALL-E's preference for concise descriptions")

        elif prompt.provider == ImageProvider.IMAGEN4:
            if "cinematic" not in prompt.prompt.lower():
                suggestions.append("Add cinematic language for better Imagen4 results")

        elif prompt.provider in (ImageProvider.FLUX, ImageProvider.SEEDREAM, ImageProvider.HUGGINGFACE):
            if not any(word in prompt.prompt.lower() for word in ["artistic", "detailed", "style"]):
                suggestions.append("Enhance artistic style descriptions for Flux optimization")

        # Ensure we have at least some suggestions
        if not suggestions:
            suggestions.append("Consider adding more descriptive detail")
            suggestions.append("Review emotional alignment with source text")

        return suggestions[:5]  # Limit to top 5 suggestions


class PromptIterator:
    """Handles prompt iteration and improvement based on quality feedback."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.quality_analyzer = QualityAnalyzer(llm)

        # Performance tracking
        self.prompt_performance: Dict[str, PromptPerformance] = {}
        self.quality_history: List[QualityAssessment] = []

    async def iterate_prompt(
        self,
        original_prompt: IllustrationPrompt,
        quality_assessment: QualityAssessment,
        emotional_moment: EmotionalMoment,
        max_iterations: int = 3
    ) -> Optional[IllustrationPrompt]:
        """Iterate and improve a prompt based on quality feedback."""

        # Check if iteration is needed
        avg_score = sum(quality_assessment.quality_scores.values()) / len(quality_assessment.quality_scores)
        if avg_score >= 0.8:
            # Prompt is already high quality
            return None

        # Generate improved prompt
        improved_prompt = await self._generate_improved_prompt(
            original_prompt,
            quality_assessment,
            emotional_moment
        )

        return improved_prompt

    # Methods expected by tests
    def identify_iteration_reasons(self, assessment: QualityAssessment) -> List[IterationReason]:
        reasons: List[IterationReason] = []
        if assessment.overall_score is not None and assessment.overall_score < 70:
            reasons.append(IterationReason.LOW_OVERALL_QUALITY)
        if assessment.accuracy_score is not None and assessment.accuracy_score < 65:
            reasons.append(IterationReason.POOR_ACCURACY)
        if assessment.technical_quality is not None and assessment.technical_quality < 60:
            reasons.append(IterationReason.WEAK_TECHNICAL_QUALITY)
        if assessment.emotional_alignment is not None and assessment.emotional_alignment < 60:
            reasons.append(IterationReason.WEAK_EMOTIONAL_RESONANCE)
        if assessment.recommendations:
            reasons.append(IterationReason.INSUFFICIENT_DETAIL)
        return reasons

    async def improve_prompt(self, original_prompt: IllustrationPrompt, assessment: QualityAssessment, iteration_reasons: List[IterationReason]) -> IllustrationPrompt:
        # Reuse existing generator logic, but provide compatibility wrapper
        improved = await self._generate_improved_prompt(original_prompt, assessment, EmotionalMoment(text_excerpt="", start_position=0, end_position=0, emotional_tones=[], intensity_score=0.0, context=""))
        return improved

    async def _generate_improved_prompt(
        self,
        original_prompt: IllustrationPrompt,
        quality_assessment: QualityAssessment,
        emotional_moment: EmotionalMoment
    ) -> IllustrationPrompt:
        """Generate an improved version of the prompt based on feedback."""

        system_prompt = f"""You are an expert prompt engineer for {original_prompt.provider.value} image generation.

Improve the provided prompt based on the quality assessment and suggestions. Focus on:
1. Addressing specific weaknesses identified in the quality scores
2. Implementing improvement suggestions
3. Maintaining the core visual and emotional intent
4. Optimizing for the target provider ({original_prompt.provider.value})

Quality issues to address:
{json.dumps({k.value: v for k, v in quality_assessment.quality_scores.items()}, indent=2)}

Improvement suggestions:
{chr(10).join(f"- {suggestion}" for suggestion in quality_assessment.improvement_suggestions)}

Return an improved prompt that maintains the original intent while addressing the identified issues."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""
Original prompt: {original_prompt.prompt}
Style modifiers: {', '.join(str(m) if not isinstance(m, tuple) else ' '.join(str(elem) for elem in m) for m in (original_prompt.style_modifiers or []))}
Technical params: {json.dumps(original_prompt.technical_params)}

Source scene: {emotional_moment.text_excerpt}
Context: {emotional_moment.context}
Emotional tones: {[tone.value for tone in emotional_moment.emotional_tones]}

Generate an improved version of this prompt.""")
            ]

            response = await self.llm.ainvoke(messages)
            improved_prompt_text = response.content.strip()

            # Create improved prompt object
            improved_prompt = IllustrationPrompt(
                provider=original_prompt.provider,
                prompt=improved_prompt_text,
                style_modifiers=(original_prompt.style_modifiers or []).copy(),
                negative_prompt=original_prompt.negative_prompt,
                technical_params=(original_prompt.technical_params or {}).copy()
            )

            return improved_prompt

        except Exception as e:
            logger.error(f"Failed to generate improved prompt: {e}")
            # Return original prompt if improvement fails
            return original_prompt

    def update_performance_tracking(
        self,
        prompt: IllustrationPrompt,
        quality_assessment: QualityAssessment
    ):
        """Update performance tracking for prompts."""

        prompt_key = f"{prompt.provider.value}_{hash(prompt.prompt[:100])}"

        if prompt_key not in self.prompt_performance:
            self.prompt_performance[prompt_key] = PromptPerformance(
                prompt_template=prompt.prompt[:100] + "..." if len(prompt.prompt) > 100 else prompt.prompt,
                provider=prompt.provider,
                success_rate=1.0 if quality_assessment.generation_success else 0.0,
                avg_quality_scores=quality_assessment.quality_scores.copy(),
                usage_count=1,
                last_updated=quality_assessment.timestamp
            )
        else:
            # Update existing performance data
            performance = self.prompt_performance[prompt_key]

            # Update success rate (exponential moving average)
            current_success = 1.0 if quality_assessment.generation_success else 0.0
            performance.success_rate = 0.8 * performance.success_rate + 0.2 * current_success

            # Update quality scores (exponential moving average)
            for metric, score in quality_assessment.quality_scores.items():
                if metric in performance.avg_quality_scores:
                    performance.avg_quality_scores[metric] = (
                        0.8 * performance.avg_quality_scores[metric] + 0.2 * score
                    )
                else:
                    performance.avg_quality_scores[metric] = score

            performance.usage_count += 1
            performance.last_updated = quality_assessment.timestamp

        # Store quality assessment in history
        self.quality_history.append(quality_assessment)

        # Keep only recent history (last 100 assessments)
        if len(self.quality_history) > 100:
            self.quality_history = self.quality_history[-100:]

    def get_performance_insights(self) -> Dict[str, Any]:
        """Get insights about prompt performance across providers."""

        insights = {
            "total_assessments": len(self.quality_history),
            "provider_performance": {},
            "quality_trends": {},
            "top_performing_patterns": []
        }

        # Calculate provider-specific performance
        for provider in ImageProvider:
            provider_assessments = [
                qa for qa in self.quality_history
                if qa.provider == provider
            ]

            if provider_assessments:
                avg_scores = {}
                for metric in QualityMetric:
                    scores = [qa.quality_scores.get(metric, 0.0) for qa in provider_assessments]
                    avg_scores[metric.value] = sum(scores) / len(scores) if scores else 0.0

                insights["provider_performance"][provider.value] = {
                    "total_generations": len(provider_assessments),
                    "success_rate": sum(1 for qa in provider_assessments if qa.generation_success) / len(provider_assessments),
                    "average_quality_scores": avg_scores
                }

        # Quality trends over time
        if len(self.quality_history) >= 10:
            recent_assessments = self.quality_history[-10:]
            older_assessments = self.quality_history[-20:-10] if len(self.quality_history) >= 20 else []

            for metric in QualityMetric:
                recent_avg = sum(qa.quality_scores.get(metric, 0.0) for qa in recent_assessments) / len(recent_assessments)

                if older_assessments:
                    older_avg = sum(qa.quality_scores.get(metric, 0.0) for qa in older_assessments) / len(older_assessments)
                    trend = recent_avg - older_avg
                    insights["quality_trends"][metric.value] = {
                        "recent_average": recent_avg,
                        "trend": "improving" if trend > 0.05 else "declining" if trend < -0.05 else "stable",
                        "change": trend
                    }

        # Top performing prompt patterns
        sorted_performance = sorted(
            self.prompt_performance.values(),
            key=lambda p: sum(p.avg_quality_scores.values()) / len(p.avg_quality_scores) if p.avg_quality_scores else 0,
            reverse=True
        )

        insights["top_performing_patterns"] = [
            {
                "template": perf.prompt_template,
                "provider": perf.provider.value,
                "success_rate": perf.success_rate,
                "avg_quality": sum(perf.avg_quality_scores.values()) / len(perf.avg_quality_scores) if perf.avg_quality_scores else 0,
                "usage_count": perf.usage_count
            }
            for perf in sorted_performance[:5]
        ]

        return insights


class FeedbackSystem:
    """Main system for quality feedback and continuous improvement."""

    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.quality_analyzer = QualityAnalyzer(llm)
        self.analyzer = self.quality_analyzer  # legacy alias used in tests
        self.prompt_iterator = PromptIterator(llm)
        self.iterator = self.prompt_iterator  # legacy alias used in tests

    async def process_generation_feedback(
        self,
        prompt: IllustrationPrompt,
        generation_result: Dict[str, Any],
        emotional_moment: EmotionalMoment,
        enable_iteration: bool = True
    ) -> Dict[str, Any]:
        """Process complete feedback cycle for a generation."""

        # Assess quality
        quality_assessment = await self.quality_analyzer.assess_generation_quality(
            prompt,
            generation_result,
            emotional_moment
        )

        # Update performance tracking
        self.prompt_iterator.update_performance_tracking(prompt, quality_assessment)

        result = {
            "quality_assessment": quality_assessment,
            "improved_prompt": None,
            "feedback_applied": False
        }

        # Optionally generate improved prompt
        if enable_iteration:
            improved_prompt = await self.prompt_iterator.iterate_prompt(
                prompt,
                quality_assessment,
                emotional_moment
            )

            if improved_prompt:
                result["improved_prompt"] = improved_prompt
                result["feedback_applied"] = True

        return result

    # Test-facing APIs expected by unit tests
    async def process_feedback_cycle(self, original_prompt: IllustrationPrompt, generation_result: Dict[str, Any], emotional_moment: EmotionalMoment) -> PromptIteration:
        assessment = await self.analyzer.assess_generation_quality(original_prompt, generation_result, emotional_moment)
        # Use the iterator.improve_prompt API which tests mock
        improved = await self.iterator.improve_prompt(original_prompt, assessment, self.iterator.identify_iteration_reasons(assessment))
        iteration = PromptIteration(
            iteration_number=1,
            original_prompt=original_prompt,
            improved_prompt=improved or original_prompt,
            quality_assessment=assessment,
            iteration_reasons=self.iterator.identify_iteration_reasons(assessment),
            improvements_made=[s for s in (assessment.improvement_suggestions or [])],
            timestamp=generation_result.get('metadata', {}).get('timestamp', '')
        )
        return iteration

    async def iterative_improvement(self, original_prompt: IllustrationPrompt, generate_fn, emotional_moment: EmotionalMoment, max_iterations: int = 3, target_quality: int = 80) -> QualityReport:
        iterations = []
        current_prompt = original_prompt
        final_quality = 0
        for i in range(max_iterations):
            generation_result = generate_fn(current_prompt)
            assessment = await self.analyzer.assess_generation_quality(current_prompt, generation_result, emotional_moment)
            final_quality = assessment.overall_score or final_quality
            improved = await self.iterator.iterate_prompt(current_prompt, assessment, emotional_moment)
            iterations.append(PromptIteration(
                iteration_number=i+1,
                original_prompt=current_prompt,
                improved_prompt=improved or current_prompt,
                quality_assessment=assessment,
                iteration_reasons=self.iterator.identify_iteration_reasons(assessment),
                improvements_made=assessment.improvement_suggestions or [],
                timestamp=generation_result.get('metadata', {}).get('timestamp', '')
            ))
            if final_quality >= target_quality:
                break
            current_prompt = improved or current_prompt

        report = QualityReport(
            session_id="session-1",
            initial_prompt=original_prompt,
            original_prompt=original_prompt,
            summary="Iterative improvement run",
            final_prompt=current_prompt,
            iterations=iterations,
            total_improvements=len(iterations),
            final_quality_score=final_quality,
            processing_time=0.0,
            success=final_quality >= target_quality
        )
        return report


class QualityThreshold(int, Enum):
    EXCELLENT = 85
    GOOD = 75
    ACCEPTABLE = 65

    @classmethod
    def meets_threshold(cls, score: int, threshold: 'QualityThreshold') -> bool:
        return score >= int(threshold.value)
