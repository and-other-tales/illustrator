"""Emotional analysis and NLP processing for manuscript text."""

import re
from dataclasses import dataclass
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from illustrator.models import (
    Chapter,
    EmotionalMoment,
    EmotionalTone,
)


@dataclass
class TextSegment:
    """Represents a segment of text for analysis."""
    text: str
    start_pos: int
    end_pos: int
    context_before: str
    context_after: str


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

    async def analyze_chapter(
        self,
        chapter: Chapter,
        max_moments: int = 5,
        min_intensity: float = 0.6,
    ) -> List[EmotionalMoment]:
        """Analyze a chapter to extract emotional moments."""
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

        return emotional_moments

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

    def _calculate_pattern_score(self, text: str) -> float:
        """Calculate emotional intensity based on pattern matching."""
        text_lower = text.lower()
        total_score = 0.0
        matches = 0

        for emotion, patterns in self.EMOTION_PATTERNS.items():
            for pattern in patterns:
                pattern_matches = len(re.findall(pattern, text_lower))
                if pattern_matches > 0:
                    matches += pattern_matches
                    total_score += pattern_matches * 0.2

        # Apply intensity modifiers
        for intensity, modifiers in self.INTENSITY_MODIFIERS.items():
            for modifier in modifiers:
                if modifier in text_lower:
                    multiplier = {'high': 1.5, 'medium': 1.2, 'low': 0.8}[intensity]
                    total_score *= multiplier
                    break

        # Normalize by text length
        words_count = len(text.split())
        normalized_score = total_score / max(1, words_count / 100)

        return min(1.0, normalized_score)

    async def _llm_intensity_score(self, segment: TextSegment) -> float:
        """Use LLM to score emotional intensity of a text segment."""
        system_prompt = """You are an expert in literary emotional analysis. Rate the emotional intensity of the following text passage on a scale from 0.0 to 1.0, where:

0.0 = No emotional content, purely descriptive or neutral
0.3 = Mild emotional undertones
0.5 = Moderate emotional content
0.7 = Strong emotional resonance
1.0 = Peak emotional intensity, highly dramatic

Consider factors like:
- Emotional vocabulary and imagery
- Narrative tension and conflict
- Character emotional states
- Sensory details that evoke emotion
- Pacing and rhythm that builds emotion

Return ONLY a decimal number between 0.0 and 1.0."""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Text to analyze:\n\n{segment.text}")
            ]

            response = await self.llm.ainvoke(messages)

            # Extract numerical score
            score_text = response.content.strip()
            score = float(score_text)

            return max(0.0, min(1.0, score))

        except (ValueError, AttributeError, Exception):
            # Fallback to pattern-based scoring
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

            # Parse JSON response
            import json
            analysis = json.loads(response.content.strip())

            emotional_tones = [EmotionalTone(tone) for tone in analysis.get('emotional_tones', [])]
            context = analysis.get('context', 'Emotionally significant moment')

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
            return max(emotion_scores.items(), key=lambda x: x[1])[0]

        return EmotionalTone.ANTICIPATION

    def _extract_peak_excerpt(self, text: str, max_length: int = 200) -> str:
        """Extract the most emotionally dense excerpt from a text segment."""
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