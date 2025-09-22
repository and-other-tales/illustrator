"""Quality feedback and prompt iteration system for continuous improvement."""

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

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
    """Quality assessment for a generated image."""
    prompt_id: str
    generation_success: bool
    quality_scores: Dict[QualityMetric, float]  # 0.0 to 1.0
    feedback_notes: str
    improvement_suggestions: List[str]
    provider: ImageProvider
    timestamp: str


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
    session_id: str
    initial_prompt: IllustrationPrompt
    final_prompt: IllustrationPrompt
    iterations: List[PromptIteration]
    total_improvements: int
    final_quality_score: float
    processing_time: float
    success: bool


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

        if not generation_result.get('success', False):
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
        quality_scores = await self._analyze_prompt_quality(
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

        return QualityAssessment(
            prompt_id=f"{original_prompt.provider.value}_{hash(original_prompt.prompt)}",
            generation_success=True,
            quality_scores=quality_scores,
            feedback_notes=feedback_notes,
            improvement_suggestions=improvement_suggestions,
            provider=original_prompt.provider,
            timestamp=generation_result.get('metadata', {}).get('timestamp', '')
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
Style modifiers: {', '.join(str(m) for m in prompt.style_modifiers)}
Technical params: {json.dumps(prompt.technical_params, indent=2)}

Original scene text: {emotional_moment.text_excerpt}
Emotional context: {emotional_moment.context}
Emotional tones: {[tone.value for tone in emotional_moment.emotional_tones]}
Intensity score: {emotional_moment.intensity_score}

Generation metadata: {json.dumps(generation_result.get('metadata', {}), indent=2)}

Analyze the prompt quality and return scores for each metric.""")
            ]

            response = await self.llm.ainvoke(messages)
            analysis_data = json.loads(response.content.strip())

            # Convert to proper enum keys and ensure valid scores
            quality_scores = {}
            for metric in QualityMetric:
                score_key = metric.value.upper()
                score = analysis_data.get(score_key, analysis_data.get(metric.value, 0.5))
                quality_scores[metric] = max(0.0, min(1.0, float(score)))

            return quality_scores

        except Exception as e:
            logger.warning(f"Failed to analyze prompt quality: {e}")
            # Return baseline scores
            return {metric: 0.6 for metric in QualityMetric}

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

        elif prompt.provider == ImageProvider.FLUX:
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
Style modifiers: {', '.join(str(m) for m in original_prompt.style_modifiers)}
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
                style_modifiers=original_prompt.style_modifiers.copy(),
                negative_prompt=original_prompt.negative_prompt,
                technical_params=original_prompt.technical_params.copy()
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
        self.prompt_iterator = PromptIterator(llm)

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
        if enable_iteration and not quality_assessment.generation_success:
            improved_prompt = await self.prompt_iterator.iterate_prompt(
                prompt,
                quality_assessment,
                emotional_moment
            )

            if improved_prompt:
                result["improved_prompt"] = improved_prompt
                result["feedback_applied"] = True

        return result

    def get_system_insights(self) -> Dict[str, Any]:
        """Get comprehensive system performance insights."""
        return self.prompt_iterator.get_performance_insights()

    def export_feedback_data(self) -> Dict[str, Any]:
        """Export feedback data for analysis or backup."""
        return {
            "quality_history": [
                {
                    "prompt_id": qa.prompt_id,
                    "generation_success": qa.generation_success,
                    "quality_scores": {k.value: v for k, v in qa.quality_scores.items()},
                    "feedback_notes": qa.feedback_notes,
                    "improvement_suggestions": qa.improvement_suggestions,
                    "provider": qa.provider.value,
                    "timestamp": qa.timestamp
                }
                for qa in self.prompt_iterator.quality_history
            ],
            "prompt_performance": {
                k: {
                    "prompt_template": v.prompt_template,
                    "provider": v.provider.value,
                    "success_rate": v.success_rate,
                    "avg_quality_scores": {mk.value: mv for mk, mv in v.avg_quality_scores.items()},
                    "usage_count": v.usage_count,
                    "last_updated": v.last_updated
                }
                for k, v in self.prompt_iterator.prompt_performance.items()
            }
        }