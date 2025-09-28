"""Advanced scene boundary detection for literary manuscripts."""

import re
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from illustrator.utils import parse_llm_json


class SceneTransitionType(str, Enum):
    """Types of scene transitions."""
    TIME_JUMP = "time_jump"
    LOCATION_CHANGE = "location_change"
    CHARACTER_CHANGE = "character_change"
    PERSPECTIVE_SHIFT = "perspective_shift"
    DIALOGUE_BREAK = "dialogue_break"
    NARRATIVE_BREAK = "narrative_break"
    CHAPTER_BREAK = "chapter_break"


@dataclass
class SceneBoundary:
    """Represents a detected scene boundary."""
    position: int  # Character position in text
    transition_type: SceneTransitionType
    confidence: float  # 0.0 to 1.0
    context_before: str
    context_after: str
    detected_markers: List[str]
    narrative_significance: float  # How significant this boundary is


@dataclass
class Scene:
    """Represents a literary scene with boundaries and metadata."""
    start_position: int
    end_position: int
    text: str
    scene_type: str  # action, dialogue, exposition, reflection, etc.
    primary_characters: List[str]
    setting_indicators: List[str]
    emotional_intensity: float
    visual_potential: float
    narrative_importance: float
    time_indicators: List[str]
    location_indicators: List[str]


class LiterarySceneDetector:
    """Advanced scene boundary detection using linguistic and narrative analysis."""

    # Time transition markers
    TIME_MARKERS = [
        # Explicit time
        r'\b(later|meanwhile|earlier|afterwards?|before|after|then|next|soon|immediately|suddenly|moments?\s+later)\b',
        r'\b(minutes?|hours?|days?|weeks?|months?|years?)\s+(later|earlier|passed?|ago)\b',
        r'\b(the\s+next|following|previous)\s+(day|morning|afternoon|evening|night|week|month|year)\b',

        # Time of day transitions
        r'\b(dawn|morning|noon|afternoon|evening|night|midnight|sunrise|sunset)\b',
        r'\b(at\s+)?(dawn|daybreak|first\s+light|crack\s+of\s+dawn)\b',

        # Temporal conjunctions
        r'\b(when|while|as|since|until|before|after)\b.*\b(finished|ended|began|started|arrived|left)\b',

        # Narrative time shifts
        r'\b(flashback|flash\s+forward|years?\s+(earlier|later)|back\s+then|in\s+the\s+past)\b'
    ]

    # Location change markers
    LOCATION_MARKERS = [
        # Movement verbs
        r'\b(went|walked|drove|flew|traveled|moved|entered|left|arrived|departed)\s+(to|from|into|out\s+of)\b',
        r'\b(upstairs|downstairs|outside|inside|indoors|outdoors)\b',

        # Location prepositions
        r'\b(at|in|on|near|by|beside|behind|in\s+front\s+of)\s+the\s+\w+',
        r'\b(across|through|over|under|around)\s+the\s+\w+',

        # Specific locations
        r'\b(home|house|office|school|hospital|restaurant|cafe|park|street|road|building)\b',
        r'\b(kitchen|bedroom|living\s+room|bathroom|garage|basement|attic)\b',

        # Geographic markers
        r'\b(city|town|village|country|state|nation|continent)\b',
        r'\b(north|south|east|west|downtown|uptown|suburb)\b'
    ]

    # Character appearance/departure markers
    CHARACTER_MARKERS = [
        r'\b(entered|arrived|appeared|came\s+in|walked\s+in|stepped\s+into)\b',
        r'\b(left|departed|exited|went\s+away|walked\s+away|disappeared)\b',
        r'\b(joined|met|encountered|ran\s+into|bumped\s+into)\b',
        r'\b(said\s+goodbye|farewell|see\s+you\s+later|until\s+next\s+time)\b'
    ]

    # Dialogue transition markers
    DIALOGUE_MARKERS = [
        r'^\s*"[^"]*"',  # Dialogue start
        r'"[^"]*"\s*$',  # Dialogue end
        r'\b(said|asked|replied|answered|whispered|shouted|exclaimed|muttered)\b',
        r'\b(conversation|discussion|talk|chat|argument|debate)\b'
    ]

    # Scene break markers
    SCENE_BREAK_MARKERS = [
        r'\n\s*\*\s*\*\s*\*\s*\n',  # Asterisk breaks
        r'\n\s*-{3,}\s*\n',  # Dash breaks
        r'\n\s*_{3,}\s*\n',  # Underscore breaks
        r'\n\s*#{3,}\s*\n',  # Hash breaks
        r'\n\s{3,}\n',  # Large whitespace
    ]

    def __init__(self, llm: BaseChatModel):
        """Initialize the scene detector."""
        self.llm = llm

    async def detect_scene_boundaries(self, text: str, confidence_threshold: float = 0.6) -> List[SceneBoundary]:
        """Detect all scene boundaries in the text."""
        boundaries = []

        # 1. Pattern-based detection
        pattern_boundaries = self._detect_pattern_boundaries(text)
        boundaries.extend(pattern_boundaries)

        # 2. Structural break detection
        structural_boundaries = self._detect_structural_breaks(text)
        boundaries.extend(structural_boundaries)

        # 3. LLM-based semantic boundary detection
        semantic_boundaries = await self._detect_semantic_boundaries(text)
        boundaries.extend(semantic_boundaries)

        # 4. Consolidate and rank boundaries
        consolidated = self._consolidate_boundaries(boundaries, text)

        # 5. Filter by confidence threshold
        filtered = [b for b in consolidated if b.confidence >= confidence_threshold]

        # 6. Sort by position
        filtered.sort(key=lambda b: b.position)

        return filtered

    def _detect_pattern_boundaries(self, text: str) -> List[SceneBoundary]:
        """Detect boundaries using linguistic patterns."""
        boundaries = []

        # Time-based boundaries
        for pattern in self.TIME_MARKERS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.append(SceneBoundary(
                    position=match.start(),
                    transition_type=SceneTransitionType.TIME_JUMP,
                    confidence=0.7,
                    context_before=self._get_context(text, match.start(), before=True),
                    context_after=self._get_context(text, match.start(), before=False),
                    detected_markers=[match.group()],
                    narrative_significance=0.6
                ))

        # Location-based boundaries
        for pattern in self.LOCATION_MARKERS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.append(SceneBoundary(
                    position=match.start(),
                    transition_type=SceneTransitionType.LOCATION_CHANGE,
                    confidence=0.6,
                    context_before=self._get_context(text, match.start(), before=True),
                    context_after=self._get_context(text, match.start(), before=False),
                    detected_markers=[match.group()],
                    narrative_significance=0.7
                ))

        # Character-based boundaries
        for pattern in self.CHARACTER_MARKERS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                boundaries.append(SceneBoundary(
                    position=match.start(),
                    transition_type=SceneTransitionType.CHARACTER_CHANGE,
                    confidence=0.5,
                    context_before=self._get_context(text, match.start(), before=True),
                    context_after=self._get_context(text, match.start(), before=False),
                    detected_markers=[match.group()],
                    narrative_significance=0.5
                ))

        return boundaries

    def _detect_structural_breaks(self, text: str) -> List[SceneBoundary]:
        """Detect structural scene breaks (asterisks, dashes, etc.)."""
        boundaries = []

        for pattern in self.SCENE_BREAK_MARKERS:
            for match in re.finditer(pattern, text):
                boundaries.append(SceneBoundary(
                    position=match.start(),
                    transition_type=SceneTransitionType.NARRATIVE_BREAK,
                    confidence=0.9,
                    context_before=self._get_context(text, match.start(), before=True),
                    context_after=self._get_context(text, match.start(), before=False),
                    detected_markers=[match.group().strip()],
                    narrative_significance=0.8
                ))

        return boundaries

    async def _detect_semantic_boundaries(self, text: str) -> List[SceneBoundary]:
        """Use LLM to detect semantic scene boundaries."""
        # Split text into paragraphs for analysis
        paragraphs = re.split(r'\n\s*\n', text)
        boundaries = []

        # Analyze paragraph transitions
        for i in range(len(paragraphs) - 1):
            current_para = paragraphs[i].strip()
            next_para = paragraphs[i + 1].strip()

            if len(current_para) < 50 or len(next_para) < 50:
                continue  # Skip very short paragraphs

            # Get position in original text
            position = text.find(next_para)

            # Ask LLM to analyze the transition
            transition_analysis = await self._analyze_paragraph_transition(
                current_para, next_para
            )

            if transition_analysis.get('is_scene_boundary', False):
                boundaries.append(SceneBoundary(
                    position=position,
                    transition_type=SceneTransitionType(
                        transition_analysis.get('transition_type', 'narrative_break')
                    ),
                    confidence=float(transition_analysis.get('confidence', 0.5)),
                    context_before=current_para[-100:],
                    context_after=next_para[:100],
                    detected_markers=transition_analysis.get('detected_markers', []),
                    narrative_significance=float(transition_analysis.get('significance', 0.5))
                ))

        return boundaries

    async def _analyze_paragraph_transition(self, para1: str, para2: str) -> Dict:
        """Analyze transition between two paragraphs."""
        system_prompt = """You are an expert in literary analysis specializing in scene structure. Analyze the transition between two consecutive paragraphs and determine if there's a scene boundary.

A scene boundary occurs when there's a significant shift in:
- Time (hours, days, or longer gaps)
- Location (different rooms, buildings, cities)
- Characters (new characters introduced, others leave)
- Perspective (POV changes)
- Narrative focus (topic/action changes significantly)

Return JSON with:
{
    "is_scene_boundary": boolean,
    "transition_type": "time_jump|location_change|character_change|perspective_shift|narrative_break",
    "confidence": float (0.0-1.0),
    "detected_markers": ["list of specific phrases that indicate transition"],
    "significance": float (0.0-1.0, how important this boundary is narratively)
}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Paragraph 1: {para1}

Paragraph 2: {para2}

Analyze if there's a scene boundary between these paragraphs.""")
            ]

            response = await self.llm.ainvoke(messages)
            return parse_llm_json(response.content)

        except Exception:
            # Fallback analysis
            return {
                "is_scene_boundary": False,
                "transition_type": "narrative_break",
                "confidence": 0.3,
                "detected_markers": [],
                "significance": 0.3
            }

    def _consolidate_boundaries(self, boundaries: List[SceneBoundary], text: str) -> List[SceneBoundary]:
        """Consolidate nearby boundaries and improve confidence scores."""
        if not boundaries:
            return []

        # Sort boundaries by position
        boundaries.sort(key=lambda b: b.position)

        consolidated = []
        merge_distance = 100  # Characters within 100 chars are considered nearby

        i = 0
        while i < len(boundaries):
            current = boundaries[i]

            # Find all boundaries within merge distance
            nearby = [current]
            j = i + 1
            while j < len(boundaries) and boundaries[j].position - current.position <= merge_distance:
                nearby.append(boundaries[j])
                j += 1

            if len(nearby) == 1:
                consolidated.append(current)
            else:
                # Merge nearby boundaries
                merged = self._merge_boundaries(nearby)
                consolidated.append(merged)

            i = j if len(nearby) > 1 else i + 1

        return consolidated

    def _merge_boundaries(self, boundaries: List[SceneBoundary]) -> SceneBoundary:
        """Merge multiple nearby boundaries into one."""
        # Use the boundary with highest confidence as base
        best = max(boundaries, key=lambda b: b.confidence)

        # Combine detected markers
        all_markers = []
        for boundary in boundaries:
            all_markers.extend(boundary.detected_markers)

        # Average confidence weighted by narrative significance
        weighted_confidence = sum(b.confidence * b.narrative_significance for b in boundaries)
        total_weight = sum(b.narrative_significance for b in boundaries)
        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else best.confidence

        # Use highest narrative significance
        max_significance = max(b.narrative_significance for b in boundaries)

        return SceneBoundary(
            position=best.position,
            transition_type=best.transition_type,
            confidence=min(1.0, avg_confidence * 1.2),  # Boost confidence for merged boundaries
            context_before=best.context_before,
            context_after=best.context_after,
            detected_markers=all_markers,
            narrative_significance=max_significance
        )

    def _get_context(self, text: str, position: int, before: bool = True, context_size: int = 100) -> str:
        """Get context around a position in text."""
        if before:
            start = max(0, position - context_size)
            return text[start:position].strip()
        else:
            end = min(len(text), position + context_size)
            return text[position:end].strip()

    async def extract_scenes(self, text: str, min_scene_length: int = 200) -> List[Scene]:
        """Extract complete scenes from text using detected boundaries."""
        boundaries = await self.detect_scene_boundaries(text)

        # Add implicit boundaries at start and end
        if not boundaries or boundaries[0].position > 0:
            boundaries.insert(0, SceneBoundary(
                position=0,
                transition_type=SceneTransitionType.NARRATIVE_BREAK,
                confidence=1.0,
                context_before="",
                context_after=text[:100],
                detected_markers=["text_start"],
                narrative_significance=0.5
            ))

        if not boundaries or boundaries[-1].position < len(text) - 100:
            boundaries.append(SceneBoundary(
                position=len(text),
                transition_type=SceneTransitionType.NARRATIVE_BREAK,
                confidence=1.0,
                context_before=text[-100:],
                context_after="",
                detected_markers=["text_end"],
                narrative_significance=0.5
            ))

        scenes = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i].position
            end = boundaries[i + 1].position
            scene_text = text[start:end].strip()

            # For very short texts, always create at least one scene
            # For normal texts, apply min_scene_length filter
            if len(scene_text) >= min_scene_length or (len(text) < 100 and len(scene_text) > 0):
                scene = await self._analyze_scene(scene_text, start, end)
                scenes.append(scene)

        # Ensure we always return at least one scene for any non-empty content
        if not scenes and text.strip():
            scene = await self._analyze_scene(text.strip(), 0, len(text))
            scenes.append(scene)

        return scenes

    async def _analyze_scene(self, scene_text: str, start_pos: int, end_pos: int) -> Scene:
        """Analyze a scene to extract its properties."""

        # Extract characters (basic name detection)
        characters = self._extract_character_names(scene_text)

        # Extract setting indicators
        setting_indicators = self._extract_setting_indicators(scene_text)

        # Extract time indicators
        time_indicators = self._extract_time_indicators(scene_text)

        # Extract location indicators
        location_indicators = self._extract_location_indicators(scene_text)

        # Use LLM for scene type and quality analysis
        scene_analysis = await self._get_scene_analysis(scene_text)

        return Scene(
            start_position=start_pos,
            end_position=end_pos,
            text=scene_text,
            scene_type=scene_analysis.get('scene_type', 'narrative'),
            primary_characters=characters[:5],  # Top 5 characters
            setting_indicators=setting_indicators,
            emotional_intensity=scene_analysis.get('emotional_intensity', 0.5),
            visual_potential=scene_analysis.get('visual_potential', 0.5),
            narrative_importance=scene_analysis.get('narrative_importance', 0.5),
            time_indicators=time_indicators,
            location_indicators=location_indicators
        )

    def _extract_character_names(self, text: str) -> List[str]:
        """Extract character names from text."""
        # Simple proper noun extraction
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)

        # Count frequency
        name_counts = {}
        for name in proper_nouns:
            if len(name) > 2 and name not in ['The', 'And', 'But', 'For', 'Not', 'Yet', 'So']:
                name_counts[name] = name_counts.get(name, 0) + 1

        # Return most frequent names
        sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)
        return [name for name, count in sorted_names if count >= 2]

    def _extract_setting_indicators(self, text: str) -> List[str]:
        """Extract setting/location indicators."""
        indicators = []

        for pattern in self.LOCATION_MARKERS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicators.extend(matches)

        return list(set(indicators))

    def _extract_time_indicators(self, text: str) -> List[str]:
        """Extract time indicators."""
        indicators = []

        for pattern in self.TIME_MARKERS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            indicators.extend(matches)

        return list(set(indicators))

    def _extract_location_indicators(self, text: str) -> List[str]:
        """Extract location indicators."""
        return self._extract_setting_indicators(text)  # Same as setting for now

    async def _get_scene_analysis(self, scene_text: str) -> Dict:
        """Get LLM analysis of scene properties."""
        system_prompt = """Analyze this scene and provide scores for different qualities.

Scene types: action, dialogue, exposition, reflection, description, conflict, resolution, transition

Return JSON:
{
    "scene_type": "primary scene type",
    "emotional_intensity": float (0.0-1.0),
    "visual_potential": float (0.0-1.0, how suitable for illustration),
    "narrative_importance": float (0.0-1.0, how important to story)
}"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Scene to analyze:\n\n{scene_text[:1000]}...")
            ]

            response = await self.llm.ainvoke(messages)
            return parse_llm_json(response.content)

        except Exception:
            return {
                "scene_type": "narrative",
                "emotional_intensity": 0.5,
                "visual_potential": 0.5,
                "narrative_importance": 0.5
            }
