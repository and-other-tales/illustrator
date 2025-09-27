"""Enhanced emotional analysis and NLP processing for manuscript text with scene-aware analysis."""

import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from illustrator.utils import parse_llm_json

logger = logging.getLogger(__name__)

from illustrator.models import (
    Chapter,
    EmotionalMoment,
    EmotionalTone,
)
from illustrator.scene_detection import LiterarySceneDetector, Scene
from illustrator.narrative_analysis import NarrativeAnalyzer, NarrativeStructure


@dataclass
class TextSegment:
    """Represents a segment of text for analysis."""
    text: str
    start_pos: int
    end_pos: int
    context_before: str
    context_after: str


class EmotionalAnalysisResult(list):
    """List-like emotional analysis result with dictionary-style access."""

    def __init__(self, moments: List[EmotionalMoment], metadata: Dict[str, Any] | None = None):
        super().__init__(moments)
        self.metadata = metadata or {}

    def __contains__(self, item):
        if item in {"emotional_moments", "metadata"}:
            return True
        return super().__contains__(item)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "emotional_moments":
                return list(self)
            if item == "metadata":
                return self.metadata
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key: str, default: Any = None):
        if key == "emotional_moments":
            return list(self)
        if key == "metadata":
            return self.metadata
        return default

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary form."""
        return {
            "emotional_moments": list(self),
            "metadata": self.metadata,
        }


class EmotionalAnalyzer:
    """Analyzes text for emotional content and resonance."""

    # Emotional intensity keywords and patterns
    EMOTION_PATTERNS = {
        EmotionalTone.JOY: [
            r'\b(laugh|smiled?|grin|delight|elat|euphori|bliss|jubil|ecsta)',
            r'\b(bright|radiant|gleaming|sparkling|warm)',
            r'\b(celebrat|triumph|victor|success|accomplish)',
        ],
        EmotionalTone.SADNESS: [
            r'\b(cry|cried|weep|wept|sob|tear|mourn|griev)',
            r'\b(melanchol|depress|despair|sorrow|misery)',
            r'\b(dark|gloomy|bleak|empty|hollow|aching)',
        ],
        EmotionalTone.FEAR: [
            r'\b(terrif|frighten|scare|dread|panic|anxious)',
            r'\b(tremble|shake|shiver|quiver|freeze)',
            r'\b(shadow|lurk|menac|threaten|ominous)',
        ],
        EmotionalTone.ANGER: [
            r'\b(rage|fury|wrath|livid|infuriat|enrag)',
            r'\b(clench|grit|snarl|growl|hiss)',
            r'\b(burn|boil|seethe|storm|thunder)',
        ],
        EmotionalTone.TENSION: [
            r'\b(tense|strain|stress|pressure|tight)',
            r'\b(edge|brink|breaking|snap|crack)',
            r'\b(silence|pause|held.*breath|heart.*pound)',
        ],
        EmotionalTone.MYSTERY: [
            r'\b(mysterious|enigma|puzzle|riddle|secret)',
            r'\b(whisper|murmur|hint|clue|shadow)',
            r'\b(hidden|concealed|veiled|obscure)',
        ],
    }

    # Intensity modifiers
    INTENSITY_MODIFIERS = {
        'high': ['absolutely', 'completely', 'utterly', 'violently', 'intensely', 'overwhelmingly'],
        'medium': ['quite', 'rather', 'fairly', 'somewhat', 'moderately'],
        'low': ['slightly', 'barely', 'faintly', 'mildly', 'gently']
    }

    def __init__(self, llm: BaseChatModel):
        """Initialize the emotional analyzer."""
        self.llm = llm
        self.scene_detector = LiterarySceneDetector(llm)
        self.narrative_analyzer = NarrativeAnalyzer(llm)

    async def analyze_chapter(
        self,
        chapter: Chapter | str,
        max_moments: int = 5,
        min_intensity: float = 0.6,
    ) -> EmotionalAnalysisResult:
        """Analyze a chapter to extract emotional moments."""
        if isinstance(chapter, str):
            chapter = Chapter(
                title="Chapter",
                content=chapter,
                number=1,
                word_count=len(chapter.split()),
            )

        # First pass: segment the text into potential emotional moments
        segments = self._segment_text(chapter.content)

        # Second pass: score segments for emotional intensity
        scored_segments = []
        for segment in segments:
            score = await self._score_emotional_intensity(segment)
            if score >= min_intensity:
                scored_segments.append((segment, score))

        # Sort by intensity and take top moments
        scored_segments.sort(key=lambda x: x[1], reverse=True)
        top_segments = scored_segments[:max_moments]

        # Third pass: detailed analysis of top moments
        emotional_moments = []
        for segment, intensity in top_segments:
            moment = await self._analyze_segment_detailed(segment, intensity, chapter.content)
            emotional_moments.append(moment)

        metadata = {
            "segments_evaluated": len(segments),
            "moments_found": len(emotional_moments),
        }

        return EmotionalAnalysisResult(emotional_moments, metadata)

    async def analyze_chapter_with_scenes(
        self,
        chapter: Chapter,
        max_moments: int = 10,
        min_intensity: float = 0.6,
        scene_awareness: bool = True
    ) -> EmotionalAnalysisResult:
        """Enhanced chapter analysis using scene boundary detection for better emotional moment extraction."""

        if not scene_awareness:
            return await self.analyze_chapter(chapter, max_moments, min_intensity)

        # First, detect scenes in the chapter
        scenes = await self.scene_detector.extract_scenes(chapter.content)

        all_moments = []

        # Analyze each scene individually
        for scene in scenes:
            # Create a scene-specific chapter object
            scene_chapter = Chapter(
                title=f"{chapter.title} - Scene",
                content=scene.text,
                number=chapter.number,
                word_count=len(scene.text.split())
            )

            # Analyze scene for emotional moments with lower threshold
            scene_moments = await self.analyze_chapter(scene_chapter, max_moments=3, min_intensity=0.4)

            # Adjust positions to match original chapter
            for moment in scene_moments:
                moment.start_position += scene.start_position
                moment.end_position += scene.start_position

                # Enhance context with scene metadata
                enhanced_context = f"{moment.context}. Scene context: {scene.scene_type} scene"
                if scene.primary_characters:
                    enhanced_context += f" featuring {', '.join(scene.primary_characters[:2])}"
                if scene.setting_indicators:
                    enhanced_context += f" in {', '.join(scene.setting_indicators[:2])}"

                moment.context = enhanced_context

                # Boost intensity for high-potential scenes
                if scene.visual_potential > 0.7:
                    moment.intensity_score = min(1.0, moment.intensity_score * 1.2)

                # Add narrative significance to the context
                moment.narrative_significance = getattr(moment, 'narrative_significance', 0.0)
                moment.narrative_significance = max(moment.narrative_significance, scene.narrative_importance)

            all_moments.extend(scene_moments)

        # Combine and diversify moments across scenes
        if not all_moments:
            # No fallback - require proper scene analysis
            raise ValueError("Advanced scene analysis failed to identify any emotional moments in chapter")

        # Sort by combined score (intensity + narrative significance)
        scored_moments = []
        for moment in all_moments:
            narrative_score = getattr(moment, 'narrative_significance', 0.5)
            combined_score = (moment.intensity_score * 0.7) + (narrative_score * 0.3)
            scored_moments.append((moment, combined_score))

        scored_moments.sort(key=lambda x: x[1], reverse=True)

        # Select diverse moments ensuring scene variety
        selected_moments = self._select_diverse_scene_moments(scored_moments, scenes, max_moments)

        metadata = {
            "scenes_analyzed": len(scenes),
            "moments_found": len(selected_moments),
        }

        return EmotionalAnalysisResult(selected_moments, metadata)

    def _select_diverse_scene_moments(
        self,
        scored_moments: List[Tuple[EmotionalMoment, float]],
        scenes: List[Scene],
        max_moments: int
    ) -> List[EmotionalMoment]:
        """Select diverse moments ensuring representation across different scenes and emotions."""

        selected = []
        scene_coverage = {}
        emotion_coverage = {}

        # Group moments by scene
        scene_moments = {}
        for moment, score in scored_moments:
            for i, scene in enumerate(scenes):
                if scene.start_position <= moment.start_position < scene.end_position:
                    if i not in scene_moments:
                        scene_moments[i] = []
                    scene_moments[i].append((moment, score))
                    break

        # First pass: select best moment from each scene (up to max_moments)
        scene_indices = list(scene_moments.keys())
        scene_indices.sort(key=lambda i: max(score for _, score in scene_moments[i]), reverse=True)

        for scene_idx in scene_indices[:max_moments]:
            if len(selected) >= max_moments:
                break

            best_moment, _ = max(scene_moments[scene_idx], key=lambda x: x[1])
            selected.append(best_moment)
            scene_coverage[scene_idx] = True

            # Track emotion coverage
            for emotion in best_moment.emotional_tones:
                emotion_coverage[emotion] = emotion_coverage.get(emotion, 0) + 1

        # Second pass: fill remaining slots with emotional diversity priority
        remaining_moments = []
        for moment, score in scored_moments:
            if moment not in selected:
                remaining_moments.append((moment, score))

        for moment, score in remaining_moments:
            if len(selected) >= max_moments:
                break

            # Prioritize moments with underrepresented emotions
            emotion_diversity_bonus = 0.0
            for emotion in moment.emotional_tones:
                current_count = emotion_coverage.get(emotion, 0)
                if current_count == 0:
                    emotion_diversity_bonus += 0.3
                elif current_count == 1:
                    emotion_diversity_bonus += 0.1

            adjusted_score = score + emotion_diversity_bonus

            # Add if it significantly improves diversity or quality
            if emotion_diversity_bonus > 0.2 or adjusted_score > 0.7:
                selected.append(moment)
                for emotion in moment.emotional_tones:
                    emotion_coverage[emotion] = emotion_coverage.get(emotion, 0) + 1

        # Sort final selection by intensity for consistency
        selected.sort(key=lambda m: m.intensity_score, reverse=True)

        return selected[:max_moments]

    async def analyze_chapter_with_narrative_structure(
        self,
        chapter: Chapter,
        max_moments: int = 10,
        min_intensity: float = 0.5,
        full_manuscript_context: str = None
    ) -> Tuple[List[EmotionalMoment], NarrativeStructure]:
        """
        Enhanced chapter analysis that incorporates narrative structure recognition
        for superior emotional moment identification and illustration opportunities.
        """

        # First, detect scenes in the chapter
        scenes = await self.scene_detector.extract_scenes(chapter.content)

        # Perform comprehensive narrative structure analysis
        narrative_structure = await self.narrative_analyzer.analyze_narrative_structure(
            chapter,
            scenes,
            full_manuscript_context
        )

        # Use narrative structure to enhance emotional moment selection
        enhanced_moments = await self._select_narrative_enhanced_moments(
            chapter,
            scenes,
            narrative_structure,
            max_moments,
            min_intensity
        )

        return enhanced_moments, narrative_structure

    async def _select_narrative_enhanced_moments(
        self,
        chapter: Chapter,
        scenes: List[Scene],
        narrative_structure: NarrativeStructure,
        max_moments: int,
        min_intensity: float
    ) -> List[EmotionalMoment]:
        """Select emotional moments enhanced by narrative structure analysis."""

        candidate_moments = []

        # 1. Extract moments from high-priority illustration opportunities
        for opportunity in narrative_structure.illustration_opportunities[:max_moments * 2]:
            if opportunity.get('priority', 0) > 0.6:
                try:
                    # Find the text around this opportunity
                    position = opportunity.get('position', 0)
                    start_pos = max(0, position - 150)
                    end_pos = min(len(chapter.content), position + 150)

                    excerpt = chapter.content[start_pos:end_pos].strip()

                    if len(excerpt) < 50:  # Skip very short excerpts
                        continue

                    # Determine emotional tones
                    emotional_tones = []
                    if 'emotional_focus' in opportunity:
                        for tone_str in opportunity['emotional_focus']:
                            try:
                                emotional_tones.append(EmotionalTone(tone_str))
                            except ValueError:
                                continue

                    # If no specific emotions, analyze the excerpt
                    if not emotional_tones:
                        emotional_tones = await self._analyze_emotional_tones(excerpt)

                    # Create enhanced emotional moment
                    moment = EmotionalMoment(
                        text_excerpt=excerpt,
                        start_position=start_pos,
                        end_position=end_pos,
                        emotional_tones=emotional_tones,
                        intensity_score=min(1.0, opportunity.get('priority', 0.7)),
                        context=self._build_narrative_context(opportunity, narrative_structure),
                        narrative_significance=opportunity.get('priority', 0.7)
                    )

                    candidate_moments.append((moment, opportunity.get('priority', 0.7)))

                except Exception as e:
                    continue

        # 2. Extract moments from narrative arc peaks
        for arc in narrative_structure.narrative_arcs:
            if arc.intensity > 0.6 and arc.illustration_potential > 0.5:
                try:
                    # Find text for this arc
                    start_pos = max(0, arc.start_position)
                    end_pos = min(len(chapter.content), arc.end_position)

                    if end_pos - start_pos < 50:
                        continue

                    excerpt = chapter.content[start_pos:end_pos].strip()

                    # Create moment from arc
                    moment = EmotionalMoment(
                        text_excerpt=excerpt[:300] if len(excerpt) > 300 else excerpt,
                        start_position=start_pos,
                        end_position=min(start_pos + 300, end_pos),
                        emotional_tones=arc.emotional_trajectory or [EmotionalTone.ANTICIPATION],
                        intensity_score=arc.intensity,
                        context=f"Narrative {arc.element.value}: {'; '.join(arc.key_events[:2])}",
                        narrative_significance=arc.significance_score
                    )

                    candidate_moments.append((moment, arc.intensity * arc.significance_score))

                except Exception as e:
                    continue

        # 3. Supplement with scene-aware analysis if needed
        if len(candidate_moments) < max_moments:
            scene_moments = await self.analyze_chapter_with_scenes(
                chapter,
                max_moments=max_moments - len(candidate_moments),
                min_intensity=min_intensity * 0.8,  # Lower threshold for supplemental moments
                scene_awareness=True
            )

            for scene_moment in scene_moments:
                candidate_moments.append((scene_moment, scene_moment.intensity_score))

        # Remove duplicates and overlapping moments
        unique_moments = self._remove_overlapping_moments(candidate_moments)

        # Sort by combined score and select final moments
        scored_moments = []
        for moment, initial_score in unique_moments:
            narrative_score = getattr(moment, 'narrative_significance', 0.5)
            combined_score = (moment.intensity_score * 0.6) + (narrative_score * 0.4)
            scored_moments.append((moment, combined_score))

        scored_moments.sort(key=lambda x: x[1], reverse=True)

        return [moment for moment, score in scored_moments[:max_moments]]

    def _build_narrative_context(
        self,
        opportunity: Dict[str, any],
        narrative_structure: NarrativeStructure
    ) -> str:
        """Build rich context description using narrative analysis."""

        context_parts = []

        if opportunity.get('type'):
            context_parts.append(f"Type: {opportunity['type']}")

        if opportunity.get('description'):
            context_parts.append(opportunity['description'])

        # Add genre context
        if narrative_structure.genre_indicators:
            genres = [g.value for g in narrative_structure.genre_indicators[:2]]
            context_parts.append(f"Genre elements: {', '.join(genres)}")

        # Add structural context
        if narrative_structure.overall_structure:
            context_parts.append(f"Structure: {narrative_structure.overall_structure}")

        # Add thematic context
        if narrative_structure.thematic_elements:
            themes = [t.theme for t in narrative_structure.thematic_elements[:2]]
            context_parts.append(f"Themes: {', '.join(themes)}")

        return "; ".join(context_parts)

    async def _analyze_emotional_tones(self, text: str) -> List[EmotionalTone]:
        """Quick emotional tone analysis for a text excerpt."""

        # Use pattern matching for speed
        detected_emotions = []

        for emotion, patterns in self.EMOTION_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text.lower()):
                    detected_emotions.append(emotion)
                    break

        return detected_emotions if detected_emotions else [EmotionalTone.ANTICIPATION]

    def _remove_overlapping_moments(
        self,
        candidate_moments: List[Tuple[EmotionalMoment, float]]
    ) -> List[Tuple[EmotionalMoment, float]]:
        """Remove overlapping emotional moments, keeping the highest-scored ones."""

        # Sort by score
        sorted_moments = sorted(candidate_moments, key=lambda x: x[1], reverse=True)

        unique_moments = []
        used_positions = set()

        for moment, score in sorted_moments:
            # Check for overlap with already selected moments
            overlaps = False
            for used_start, used_end in used_positions:
                # Check if this moment overlaps significantly (>50%) with a used position
                overlap_start = max(moment.start_position, used_start)
                overlap_end = min(moment.end_position, used_end)

                if overlap_end > overlap_start:
                    overlap_size = overlap_end - overlap_start
                    moment_size = moment.end_position - moment.start_position
                    overlap_ratio = overlap_size / max(1, moment_size)

                    if overlap_ratio >= 0.5:
                        overlaps = True
                        break

            if not overlaps:
                unique_moments.append((moment, score))
                used_positions.add((moment.start_position, moment.end_position))

        return unique_moments

    def _segment_text(self, text: str, segment_size: int = 500, overlap: int = 100) -> List[TextSegment]:
        """Segment text into overlapping chunks for analysis."""
        segments = []
        words = text.split()

        for i in range(0, len(words), segment_size - overlap):
            start_idx = i
            end_idx = min(i + segment_size, len(words))

            segment_text = ' '.join(words[start_idx:end_idx])

            # Calculate character positions
            start_pos = len(' '.join(words[:start_idx]))
            end_pos = start_pos + len(segment_text)

            # Get context
            context_before = ' '.join(words[max(0, start_idx-50):start_idx])
            context_after = ' '.join(words[end_idx:min(len(words), end_idx+50)])

            segments.append(TextSegment(
                text=segment_text,
                start_pos=start_pos,
                end_pos=end_pos,
                context_before=context_before,
                context_after=context_after
            ))

        return segments

    async def _score_emotional_intensity(self, segment: TextSegment) -> float:
        """Score a text segment for emotional intensity using pattern matching and LLM."""
        # Pattern-based scoring
        pattern_score = self._calculate_pattern_score(segment.text)

        # LLM-based scoring for nuance
        llm_score = await self._llm_intensity_score(segment)

        # Combine scores (weighted average)
        final_score = (pattern_score * 0.3) + (llm_score * 0.7)

        return min(1.0, final_score)


    async def _llm_intensity_score(self, segment: TextSegment) -> float:
        """Use LLM to score emotional intensity of a text segment."""
        system_prompt = """You are a literary emotional analysis expert. Rate the emotional intensity of a text passage.

RESPONSE FORMAT: A single decimal number from 0.0 to 1.0.

SCALE:
0.0 = No emotional content, purely descriptive or neutral
0.3 = Mild emotional undertones
0.5 = Moderate emotional content 
0.7 = Strong emotional resonance
1.0 = Peak emotional intensity, highly dramatic

ANALYSIS FACTORS:
- Emotional vocabulary and sensory imagery
- Narrative tension and conflict 
- Character emotional states and reactions
- Environmental and atmospheric details
- Pacing and emotional build-up

CRITICAL: Your response must ONLY be a decimal between 0.0 and 1.0 with no other text or explanation."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Text to analyze:\n\n{segment.text}")
        ]

        def _extract_text(raw_response: Any) -> str:
            if raw_response is None:
                return ""

            if isinstance(raw_response, str):
                return raw_response.strip()

            content = getattr(raw_response, "content", None)
            if content is None:
                return str(raw_response).strip()

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, dict):
                text_value = content.get("text")
                if isinstance(text_value, str):
                    return text_value.strip()
                blocks = content.get("content")
                if isinstance(blocks, (list, tuple)):
                    collected: list[str] = []
                    for block in blocks:
                        if isinstance(block, str):
                            collected.append(block)
                        elif isinstance(block, dict):
                            block_text = block.get("text")
                            if isinstance(block_text, str):
                                collected.append(block_text)
                    if collected:
                        return "\n".join(collected).strip()

            if isinstance(content, (list, tuple)):
                collected = []
                for item in content:
                    if isinstance(item, str):
                        collected.append(item)
                    elif hasattr(item, "text") and isinstance(getattr(item, "text"), str):
                        collected.append(getattr(item, "text"))
                    elif isinstance(item, dict):
                        block_text = item.get("text")
                        if isinstance(block_text, str):
                            collected.append(block_text)
                if collected:
                    return "\n".join(collected).strip()

            return str(content).strip()

        # First try the LLM call itself; on transport/LLM failures, fall back
        try:
            # Set a reasonable timeout
            response = await self.llm.ainvoke(messages, timeout=10.0)
            
            # Extract and validate response
            score_text_raw = _extract_text(response)
            score_text = str(score_text_raw or "").strip()
            
            # Early success path for clean numeric responses
            if score_text and re.match(r'^0?\.\d+$|^1(\.0+)?$', score_text):
                score = float(score_text)
                if 0.0 <= score <= 1.0:
                    return score
                    
        except asyncio.TimeoutError:
            logger.warning("LLM intensity scoring timed out after 10s")
            # Fall through to pattern scoring
        except Exception as e:
            logger.error(f"LLM intensity scoring failed: {str(e)}")
            # Fall through to pattern scoring
            
        score_text = str(score_text_raw or "").strip()

        match_source = score_text

        if not score_text:
            logger.warning("LLM intensity scoring returned empty output; applying fallback parsing.")
        else:
            # Guard against stray wrapping quotes or formatting artefacts like "0.82" or '0.7'
            sanitized_score = score_text.strip("'\"")
            if not sanitized_score:
                logger.warning(
                    "LLM intensity scoring returned only formatting characters; applying fallback parsing."
                )
            else:
                try:
                    score = float(sanitized_score)
                    return max(0.0, min(1.0, score))
                except ValueError as e:
                    logger.warning(
                        "LLM intensity scoring returned non-numeric output: %s. Applying fallback parsing.",
                        e,
                    )
            match_source = sanitized_score

        # Try multiple numeric patterns
        patterns = [
            r'(0?\.\d+|1(?:\.0+)?)',  # Standard decimal
            r'(\d*\.?\d+e[-+]?\d+)',   # Scientific notation
            r'(\d+/\d+)',              # Fractions
            r'(\d+)'                   # Whole numbers
        ]
        
        for pattern in patterns:
            match = re.search(pattern, match_source)
            if match:
                try:
                    raw_value = match.group(1)
                    if '/' in raw_value:  # Handle fractions
                        num, denom = map(float, raw_value.split('/'))
                        score = num / denom
                    else:
                        score = float(raw_value)
                    return max(0.0, min(1.0, score))
                except (ValueError, ZeroDivisionError):
                    continue

        logger.error("LLM intensity scoring failed; using heuristic pattern score as fallback.")
        return self._calculate_pattern_score(segment.text)

    async def _analyze_segment_detailed(
        self,
        segment: TextSegment,
        intensity: float,
        full_chapter_text: str,
    ) -> EmotionalMoment:
        """Perform detailed analysis of an emotional moment."""
        system_prompt = """You are a literary analyst specializing in emotional resonance. Analyze the provided text segment and identify:

1. The dominant emotional tones (up to 3 from the list: joy, sadness, anger, fear, surprise, disgust, anticipation, trust, melancholy, excitement, tension, peace, mystery, romance, adventure)
2. A brief description of the emotional context and what makes this moment resonant

Respond in JSON format:
{
    "emotional_tones": ["emotion1", "emotion2", "emotion3"],
    "context": "Brief description of the emotional context and significance"
}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Text segment to analyze:\n\n{segment.text}\n\nSurrounding context:\n{segment.context_before}\n[SEGMENT]\n{segment.context_after}")
            ]

            response = await self.llm.ainvoke(messages)
            # Parse JSON response robustly
            analysis = parse_llm_json(response.content)

            raw_tones = analysis.get('emotional_tones') or []
            if not isinstance(raw_tones, (list, tuple)):
                raw_tones = [raw_tones]

            emotional_tones: list[EmotionalTone] = []
            for tone in raw_tones:
                if not tone:
                    continue
                try:
                    emotional_tones.append(EmotionalTone(tone))
                except ValueError:
                    logger.debug("Ignoring unrecognised emotional tone from LLM output: %r", tone)

            if not emotional_tones:
                emotional_tones = [self._identify_primary_emotion(segment.text)]

            context = analysis.get('context') or "Emotionally significant moment"
            if not isinstance(context, str):
                context = str(context)

        except Exception:
            # Fallback analysis
            emotional_tones = [self._identify_primary_emotion(segment.text)]
            context = "Emotionally resonant passage with strong sensory and emotional content"

        # Find the most impactful excerpt within the segment
        excerpt = self._extract_peak_excerpt(segment.text)

        return EmotionalMoment(
            text_excerpt=excerpt,
            start_position=segment.start_pos,
            end_position=segment.end_pos,
            emotional_tones=emotional_tones,
            intensity_score=intensity,
            context=context,
        )

    def _identify_primary_emotion(self, text: str) -> EmotionalTone:
        """Identify the primary emotion in text using pattern matching."""
        text_lower = text.lower()
        emotion_scores = {}

        for emotion, patterns in self.EMOTION_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                score += matches
            emotion_scores[emotion] = score

        if emotion_scores:
            top_emotion, top_score = max(emotion_scores.items(), key=lambda x: x[1])
            if top_score > 0:
                return top_emotion

        return EmotionalTone.ANTICIPATION

    def _extract_peak_excerpt(self, text: str, max_length: int = 200) -> str:
        """Extract the most emotionally dense excerpt from a text segment."""
        # If there are no sentence-ending punctuations, return the original (trimmed)
        if not re.search(r'[.!?]', text):
            return text[:max_length]

        sentences = re.split(r'[.!?]+', text)

        if not sentences:
            return text[:max_length]

        # Score each sentence
        scored_sentences = []
        for sentence in sentences:
            if len(sentence.strip()) > 10:  # Skip very short sentences
                score = self._calculate_pattern_score(sentence)
                scored_sentences.append((sentence.strip(), score))

        if not scored_sentences:
            return text[:max_length]

        # Sort by score and build excerpt
        scored_sentences.sort(key=lambda x: x[1], reverse=True)

        # Take top sentences up to max length
        excerpt_parts = []
        current_length = 0

        for sentence, score in scored_sentences:
            if current_length + len(sentence) <= max_length:
                excerpt_parts.append(sentence)
                current_length += len(sentence) + 1  # +1 for space
            else:
                break

        if excerpt_parts:
            return '. '.join(excerpt_parts) + '.'
        else:
            return scored_sentences[0][0][:max_length]

    def _calculate_pattern_score(self, text: str) -> float:
        """Calculate emotional pattern score using regex patterns."""
        text_lower = text.lower()
        total_score = 0.0

        # Score based on emotion patterns
        for emotion, patterns in self.EMOTION_PATTERNS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower))
                total_score += matches * 0.1

        # Apply intensity modifiers
        for intensity_level, modifiers in self.INTENSITY_MODIFIERS.items():
            multiplier = {'high': 1.5, 'medium': 1.0, 'low': 0.5}.get(intensity_level, 1.0)
            for modifier in modifiers:
                if modifier in text_lower:
                    total_score *= multiplier
                    break

        return min(1.0, total_score)
