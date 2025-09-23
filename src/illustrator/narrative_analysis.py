"""Advanced narrative structure recognition and literary analysis for manuscript illustration."""

import re
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from illustrator.models import Chapter, EmotionalMoment, EmotionalTone
from illustrator.scene_detection import Scene

logger = logging.getLogger(__name__)


class NarrativeElement(str, Enum):
    """Core narrative structure elements."""
    EXPOSITION = "exposition"
    INCITING_INCIDENT = "inciting_incident"
    RISING_ACTION = "rising_action"
    CLIMAX = "climax"
    FALLING_ACTION = "falling_action"
    RESOLUTION = "resolution"
    DENOUEMENT = "denouement"

    # Sub-elements
    CONFLICT_INTRODUCTION = "conflict_introduction"
    CHARACTER_DEVELOPMENT = "character_development"
    PLOT_TWIST = "plot_twist"
    REVELATION = "revelation"
    CONFRONTATION = "confrontation"
    RECONCILIATION = "reconciliation"


class LiteraryDevice(str, Enum):
    """Literary devices and techniques."""
    FORESHADOWING = "foreshadowing"
    FLASHBACK = "flashback"
    SYMBOLISM = "symbolism"
    METAPHOR = "metaphor"
    IRONY = "irony"
    PARALLEL = "parallel"
    CONTRAST = "contrast"
    MOTIF = "motif"
    ALLEGORY = "allegory"
    IMAGERY = "imagery"
    DIALOGUE = "dialogue"
    INTERNAL_MONOLOGUE = "internal_monologue"
    DESCRIPTION = "description"


class Genre(str, Enum):
    """Literary genres with specific narrative patterns."""
    FANTASY = "fantasy"
    SCIENCE_FICTION = "science_fiction"
    MYSTERY = "mystery"
    THRILLER = "thriller"
    ROMANCE = "romance"
    HORROR = "horror"
    LITERARY_FICTION = "literary_fiction"
    HISTORICAL_FICTION = "historical_fiction"
    YOUNG_ADULT = "young_adult"
    ADVENTURE = "adventure"
    DRAMA = "drama"
    COMEDY = "comedy"


@dataclass
class NarrativeArc:
    """Represents a narrative arc with structure and pacing."""
    element: NarrativeElement
    start_position: int
    end_position: int
    intensity: float  # 0.0 to 1.0
    emotional_trajectory: List[EmotionalTone]
    key_events: List[str]
    character_involvement: List[str]
    literary_devices: List[LiteraryDevice]
    significance_score: float  # How important this arc is to the overall story
    illustration_potential: float  # How suitable for visual representation


@dataclass
class CharacterArc:
    """Character development arc within the narrative."""
    character_name: str
    arc_type: str  # growth, fall, static, transformation, etc.
    starting_state: Dict[str, str]  # personality, goals, conflicts
    ending_state: Dict[str, str]
    key_moments: List[int]  # positions in text
    emotional_journey: List[Tuple[int, EmotionalTone]]  # position, emotion
    relationships_evolution: Dict[str, str]  # character -> relationship change
    significance: float


@dataclass
class ThematicElement:
    """Thematic analysis of the narrative."""
    theme: str
    evidence_positions: List[int]
    supporting_quotes: List[str]
    character_connections: List[str]
    symbolic_elements: List[str]
    development_arc: str  # how the theme develops through the story
    visual_manifestations: List[str]  # how theme could be shown visually


@dataclass
class NarrativeStructure:
    """Complete narrative structure analysis."""
    overall_structure: str  # three-act, hero's journey, etc.
    narrative_arcs: List[NarrativeArc]
    character_arcs: List[CharacterArc]
    thematic_elements: List[ThematicElement]
    pacing_analysis: Dict[str, float]
    genre_indicators: List[Genre]
    literary_style: Dict[str, str]
    illustration_opportunities: List[Dict[str, any]]


class NarrativeAnalyzer:
    """Advanced narrative structure and literary analysis system."""

    # Narrative structure patterns
    STRUCTURE_PATTERNS = {
        NarrativeElement.EXPOSITION: [
            r'\b(once upon a time|in the beginning|long ago|it was|the story begins)',
            r'\b(introduction|background|setting|establish)',
            r'\b(first|initially|at first|originally)'
        ],
        NarrativeElement.INCITING_INCIDENT: [
            r'\b(suddenly|unexpectedly|then|but then|however)',
            r'\b(changed|disrupted|interrupted|challenged)',
            r'\b(incident|event|moment|discovery|arrival)'
        ],
        NarrativeElement.RISING_ACTION: [
            r'\b(tension|conflict|struggle|challenge|obstacle)',
            r'\b(escalated|intensified|worsened|complicated)',
            r'\b(meanwhile|during|while|as)'
        ],
        NarrativeElement.CLIMAX: [
            r'\b(climax|peak|height|moment of truth|breaking point)',
            r'\b(finally|at last|ultimately|decisive)',
            r'\b(confrontation|showdown|face off|decisive battle)'
        ],
        NarrativeElement.FALLING_ACTION: [
            r'\b(after|following|in the aftermath|consequently)',
            r'\b(resolution|settling|calming|aftermath)',
            r'\b(began to|started to|slowly)'
        ],
        NarrativeElement.RESOLUTION: [
            r'\b(resolved|concluded|ended|finished|settled)',
            r'\b(finally|ultimately|in the end|at last)',
            r'\b(peace|harmony|understanding|acceptance)'
        ]
    }

    # Literary device patterns
    DEVICE_PATTERNS = {
        LiteraryDevice.FORESHADOWING: [
            r'\b(hinted|suggested|implied|foretold)',
            r'\b(ominous|foreboding|prophetic|warning)',
            r'\b(little did.*know|if only.*knew|would later)'
        ],
        LiteraryDevice.SYMBOLISM: [
            r'\b(symbol|symbolic|represent|embody)',
            r'\b(significance|meaning|deeper|metaphorical)',
            r'\b(stood for|represented|embodied)'
        ],
        LiteraryDevice.IRONY: [
            r'\b(ironic|ironically|paradox|contrary)',
            r'\b(unexpected|opposite|reverse|twist)',
            r'\b(bitter irony|cruel twist|fateful)'
        ]
    }

    # Genre indicators
    GENRE_PATTERNS = {
        Genre.FANTASY: [
            r'\b(magic|magical|wizard|spell|enchant|dragon|quest|realm)',
            r'\b(sword|knight|castle|prophecy|ancient|mystical)'
        ],
        Genre.MYSTERY: [
            r'\b(mystery|clue|investigate|detective|suspect|murder)',
            r'\b(evidence|alibi|motive|crime|solve|reveal)'
        ],
        Genre.ROMANCE: [
            r'\b(love|romance|heart|passion|kiss|embrace|beloved)',
            r'\b(romantic|tender|intimate|affection|attraction)'
        ],
        Genre.HORROR: [
            r'\b(horror|terror|fear|frightening|scary|nightmare)',
            r'\b(monster|ghost|haunted|evil|darkness|scream)'
        ]
    }

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def analyze_narrative_structure(
        self,
        chapter: Chapter,
        scenes: List[Scene] = None,
        full_manuscript_context: str = None
    ) -> NarrativeStructure:
        """Perform comprehensive narrative structure analysis."""

        # Basic structural analysis
        narrative_arcs = await self._identify_narrative_arcs(chapter.content, scenes)

        # Character arc analysis
        character_arcs = await self._analyze_character_arcs(chapter.content, narrative_arcs)

        # Thematic analysis
        thematic_elements = await self._analyze_themes(chapter.content)

        # Genre identification
        genre_indicators = self._identify_genre(chapter.content)

        # Pacing analysis
        pacing_analysis = await self._analyze_pacing(chapter.content, narrative_arcs)

        # Literary style analysis
        literary_style = await self._analyze_literary_style(chapter.content)

        # Identify illustration opportunities
        illustration_opportunities = await self._identify_illustration_opportunities(
            narrative_arcs, character_arcs, thematic_elements, scenes
        )

        # Determine overall structure
        overall_structure = self._determine_overall_structure(narrative_arcs)

        return NarrativeStructure(
            overall_structure=overall_structure,
            narrative_arcs=narrative_arcs,
            character_arcs=character_arcs,
            thematic_elements=thematic_elements,
            pacing_analysis=pacing_analysis,
            genre_indicators=genre_indicators,
            literary_style=literary_style,
            illustration_opportunities=illustration_opportunities
        )

    async def _identify_narrative_arcs(
        self,
        text: str,
        scenes: List[Scene] = None
    ) -> List[NarrativeArc]:
        """Identify narrative arcs using pattern matching and LLM analysis."""

        arcs = []

        # Pattern-based identification
        pattern_arcs = self._find_pattern_based_arcs(text)

        # LLM-based analysis for more sophisticated recognition
        llm_arcs = await self._analyze_narrative_structure_llm(text, scenes)

        # Combine and refine
        combined_arcs = self._combine_and_refine_arcs(pattern_arcs, llm_arcs, text)

        return combined_arcs

    def _find_pattern_based_arcs(self, text: str) -> List[NarrativeArc]:
        """Find narrative arcs using pattern matching."""
        arcs = []

        for element, patterns in self.STRUCTURE_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    start_pos = max(0, match.start() - 100)
                    end_pos = min(len(text), match.end() + 200)

                    arc = NarrativeArc(
                        element=element,
                        start_position=start_pos,
                        end_position=end_pos,
                        intensity=0.5,  # Default, will be refined
                        emotional_trajectory=[],
                        key_events=[match.group()],
                        character_involvement=[],
                        literary_devices=[],
                        significance_score=0.5,
                        illustration_potential=0.6
                    )

                    arcs.append(arc)

        return arcs

    async def _analyze_narrative_structure_llm(
        self,
        text: str,
        scenes: List[Scene] = None
    ) -> List[Dict]:
        """Use LLM to identify narrative structure elements."""

        system_prompt = """You are an expert in narrative analysis and story structure. Analyze the provided text and identify narrative structure elements.

        Identify:
        1. Narrative elements (exposition, inciting incident, rising action, climax, falling action, resolution)
        2. Character development moments
        3. Plot progression and pacing
        4. Literary devices used
        5. Emotional intensity at different points

        Return JSON array:
        [
            {
                "element": "exposition|inciting_incident|rising_action|climax|falling_action|resolution",
                "start_position": approximate_character_position,
                "end_position": approximate_character_position,
                "intensity": float (0.0-1.0),
                "description": "brief description of this narrative element",
                "key_events": ["list", "of", "key", "events"],
                "emotional_tones": ["emotion1", "emotion2"],
                "literary_devices": ["device1", "device2"],
                "significance": float (0.0-1.0),
                "illustration_potential": float (0.0-1.0)
            }
        ]"""

        try:
            scene_context = ""
            if scenes:
                scene_context = f"\\nScene information: {len(scenes)} scenes detected with types: {[s.scene_type for s in scenes[:3]]}"

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Text to analyze:{scene_context}\\n\\n{text[:2000]}...")
            ]

            response = await self.llm.ainvoke(messages)
            return json.loads(response.content.strip())

        except Exception as e:
            logger.warning(f"LLM narrative analysis failed: {e}")
            return []

    def _combine_and_refine_arcs(
        self,
        pattern_arcs: List[NarrativeArc],
        llm_arcs: List[Dict],
        text: str
    ) -> List[NarrativeArc]:
        """Combine pattern-based and LLM-based arc identification."""

        refined_arcs = []

        # Convert LLM arcs to NarrativeArc objects
        for arc_data in llm_arcs:
            try:
                emotional_tones = []
                for tone_str in arc_data.get('emotional_tones', []):
                    try:
                        emotional_tones.append(EmotionalTone(tone_str))
                    except ValueError:
                        continue

                literary_devices = []
                for device_str in arc_data.get('literary_devices', []):
                    try:
                        literary_devices.append(LiteraryDevice(device_str))
                    except ValueError:
                        continue

                arc = NarrativeArc(
                    element=NarrativeElement(arc_data.get('element', 'exposition')),
                    start_position=int(arc_data.get('start_position', 0)),
                    end_position=int(arc_data.get('end_position', 100)),
                    intensity=float(arc_data.get('intensity', 0.5)),
                    emotional_trajectory=emotional_tones,
                    key_events=arc_data.get('key_events', []),
                    character_involvement=arc_data.get('characters', []),
                    literary_devices=literary_devices,
                    significance_score=float(arc_data.get('significance', 0.5)),
                    illustration_potential=float(arc_data.get('illustration_potential', 0.5))
                )

                refined_arcs.append(arc)

            except Exception as e:
                logger.warning(f"Error processing LLM arc data: {e}")
                continue

        # Add high-confidence pattern arcs
        for pattern_arc in pattern_arcs:
            # Check if this arc overlaps significantly with LLM arcs
            overlaps = any(
                abs(pattern_arc.start_position - llm_arc.start_position) < 200
                for llm_arc in refined_arcs
            )

            if not overlaps:
                refined_arcs.append(pattern_arc)

        # Sort by position and remove duplicates
        refined_arcs.sort(key=lambda a: a.start_position)

        return refined_arcs

    async def _analyze_character_arcs(
        self,
        text: str,
        narrative_arcs: List[NarrativeArc]
    ) -> List[CharacterArc]:
        """Analyze character development arcs."""

        system_prompt = """Analyze the character development and arcs in the provided text.

        For each significant character, identify:
        1. Character name and role
        2. Type of character arc (growth, fall, transformation, static, etc.)
        3. Starting emotional/psychological state
        4. Key character development moments
        5. Ending state or trajectory
        6. Relationships with other characters and how they evolve

        Return JSON array of character arcs:
        [
            {
                "character_name": "character name",
                "arc_type": "growth|fall|transformation|static|redemption",
                "starting_state": {
                    "personality": "description",
                    "goals": "description",
                    "conflicts": "description"
                },
                "ending_state": {
                    "personality": "description",
                    "goals": "description",
                    "conflicts": "description"
                },
                "key_moments": [position_estimates],
                "emotional_journey": ["emotion_at_key_moments"],
                "relationships": {"character_name": "relationship_evolution"},
                "significance": float (0.0-1.0)
            }
        ]"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Text to analyze:\n\n{text[:1500]}")
            ]

            response = await self.llm.ainvoke(messages)
            character_data = json.loads(response.content.strip())

            character_arcs = []
            for data in character_data:
                try:
                    # Convert emotional journey
                    emotional_journey = []
                    for emotion in data.get('emotional_journey', []):
                        try:
                            emotional_journey.append((0, EmotionalTone(emotion)))
                        except ValueError:
                            continue

                    arc = CharacterArc(
                        character_name=data.get('character_name', 'Unknown'),
                        arc_type=data.get('arc_type', 'static'),
                        starting_state=data.get('starting_state', {}),
                        ending_state=data.get('ending_state', {}),
                        key_moments=data.get('key_moments', []),
                        emotional_journey=emotional_journey,
                        relationships_evolution=data.get('relationships', {}),
                        significance=float(data.get('significance', 0.5))
                    )

                    character_arcs.append(arc)

                except Exception as e:
                    logger.warning(f"Error processing character arc: {e}")
                    continue

            return character_arcs

        except Exception as e:
            logger.warning(f"Character arc analysis failed: {e}")
            return []

    async def _analyze_themes(self, text: str) -> List[ThematicElement]:
        """Analyze thematic elements in the text."""

        system_prompt = """Identify and analyze the major themes in the provided text.

        For each theme:
        1. Identify the central theme
        2. Find evidence and supporting examples
        3. Note how characters relate to this theme
        4. Identify symbolic elements
        5. Analyze how the theme develops
        6. Consider visual representations

        Return JSON array:
        [
            {
                "theme": "theme name",
                "evidence_quotes": ["quote1", "quote2"],
                "character_connections": ["character1", "character2"],
                "symbolic_elements": ["symbol1", "symbol2"],
                "development_arc": "description of how theme develops",
                "visual_manifestations": ["visual_element1", "visual_element2"]
            }
        ]"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Text to analyze:\n\n{text[:1500]}")
            ]

            response = await self.llm.ainvoke(messages)
            theme_data = json.loads(response.content.strip())

            thematic_elements = []
            for data in theme_data:
                element = ThematicElement(
                    theme=data.get('theme', 'Unknown'),
                    evidence_positions=[],  # Would need more sophisticated position finding
                    supporting_quotes=data.get('evidence_quotes', []),
                    character_connections=data.get('character_connections', []),
                    symbolic_elements=data.get('symbolic_elements', []),
                    development_arc=data.get('development_arc', ''),
                    visual_manifestations=data.get('visual_manifestations', [])
                )
                thematic_elements.append(element)

            return thematic_elements

        except Exception as e:
            logger.warning(f"Thematic analysis failed: {e}")
            return []

    def _identify_genre(self, text: str) -> List[Genre]:
        """Identify genre indicators in the text."""
        genre_scores = {}

        for genre, patterns in self.GENRE_PATTERNS.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, text, re.IGNORECASE))
                score += matches

            if score > 0:
                genre_scores[genre] = score

        # Return genres sorted by score
        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        return [genre for genre, score in sorted_genres if score >= 2]

    async def _analyze_pacing(
        self,
        text: str,
        narrative_arcs: List[NarrativeArc]
    ) -> Dict[str, float]:
        """Analyze narrative pacing."""

        # Simple pacing analysis based on sentence length, paragraph structure, etc.
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)

        paragraphs = text.split('\n\n')
        avg_paragraph_length = sum(len(p.split()) for p in paragraphs) / len(paragraphs)

        # Calculate pacing intensity from narrative arcs
        total_intensity = sum(arc.intensity for arc in narrative_arcs)
        avg_intensity = total_intensity / max(1, len(narrative_arcs))

        return {
            'average_sentence_length': avg_sentence_length,
            'average_paragraph_length': avg_paragraph_length,
            'narrative_intensity': avg_intensity,
            'pacing_score': min(1.0, (10.0 - avg_sentence_length / 10.0) + avg_intensity)
        }

    async def _analyze_literary_style(self, text: str) -> Dict[str, str]:
        """Analyze literary style and voice."""

        system_prompt = """Analyze the literary style of this text passage.

        Consider:
        1. Narrative voice (first person, third person, omniscient, etc.)
        2. Writing style (descriptive, minimalist, lyrical, etc.)
        3. Tone (formal, informal, poetic, conversational, etc.)
        4. Literary devices used
        5. Sentence structure patterns

        Return JSON:
        {
            "narrative_voice": "description",
            "writing_style": "description",
            "tone": "description",
            "dominant_devices": ["device1", "device2"],
            "style_category": "literary|commercial|genre|experimental"
        }"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Text to analyze:\n\n{text[:1000]}")
            ]

            response = await self.llm.ainvoke(messages)
            return json.loads(response.content.strip())

        except Exception as e:
            logger.warning(f"Literary style analysis failed: {e}")
            return {
                'narrative_voice': 'unknown',
                'writing_style': 'unknown',
                'tone': 'unknown',
                'dominant_devices': [],
                'style_category': 'unknown'
            }

    async def _identify_illustration_opportunities(
        self,
        narrative_arcs: List[NarrativeArc],
        character_arcs: List[CharacterArc],
        thematic_elements: List[ThematicElement],
        scenes: List[Scene] = None
    ) -> List[Dict[str, any]]:
        """Identify opportunities for compelling illustrations based on narrative analysis."""

        opportunities = []

        # High-intensity narrative moments
        for arc in narrative_arcs:
            if arc.intensity > 0.7 and arc.illustration_potential > 0.6:
                opportunities.append({
                    'type': 'narrative_peak',
                    'position': arc.start_position,
                    'description': f"{arc.element.value} with high intensity",
                    'visual_elements': arc.key_events,
                    'emotional_focus': [tone.value for tone in arc.emotional_trajectory],
                    'priority': arc.significance_score * arc.illustration_potential,
                    'composition_suggestion': self._suggest_composition_for_arc(arc)
                })

        # Character development climax moments
        for char_arc in character_arcs:
            if char_arc.significance > 0.6:
                for moment_pos in char_arc.key_moments:
                    opportunities.append({
                        'type': 'character_development',
                        'position': moment_pos,
                        'description': f"Key development moment for {char_arc.character_name}",
                        'character_focus': char_arc.character_name,
                        'arc_type': char_arc.arc_type,
                        'priority': char_arc.significance,
                        'composition_suggestion': 'character_focused'
                    })

        # Thematic visual manifestations
        for theme in thematic_elements:
            for visual_element in theme.visual_manifestations:
                opportunities.append({
                    'type': 'thematic_illustration',
                    'description': f"Visual representation of theme: {theme.theme}",
                    'theme': theme.theme,
                    'visual_element': visual_element,
                    'symbolic_elements': theme.symbolic_elements,
                    'priority': 0.7,
                    'composition_suggestion': 'symbolic_focused'
                })

        # Scene-based opportunities
        if scenes:
            for scene in scenes:
                if scene.visual_potential > 0.7:
                    opportunities.append({
                        'type': 'high_visual_scene',
                        'position': scene.start_position,
                        'description': f"High visual potential {scene.scene_type} scene",
                        'scene_type': scene.scene_type,
                        'characters': scene.primary_characters,
                        'setting': scene.setting_indicators,
                        'priority': scene.visual_potential * scene.narrative_importance,
                        'composition_suggestion': self._suggest_composition_for_scene(scene)
                    })

        # Sort by priority
        opportunities.sort(key=lambda x: x['priority'], reverse=True)

        return opportunities[:15]  # Top 15 opportunities

    def _suggest_composition_for_arc(self, arc: NarrativeArc) -> str:
        """Suggest visual composition based on narrative arc type."""
        composition_map = {
            NarrativeElement.EXPOSITION: 'wide_establishing_shot',
            NarrativeElement.INCITING_INCIDENT: 'dynamic_medium_shot',
            NarrativeElement.RISING_ACTION: 'tension_building_composition',
            NarrativeElement.CLIMAX: 'dramatic_close_up',
            NarrativeElement.FALLING_ACTION: 'aftermath_wide_shot',
            NarrativeElement.RESOLUTION: 'peaceful_resolution_shot'
        }

        return composition_map.get(arc.element, 'balanced_composition')

    def _suggest_composition_for_scene(self, scene: Scene) -> str:
        """Suggest visual composition based on scene type."""
        composition_map = {
            'action': 'dynamic_action_shot',
            'dialogue': 'character_interaction_shot',
            'exposition': 'establishing_wide_shot',
            'reflection': 'introspective_close_up',
            'description': 'atmospheric_composition',
            'conflict': 'tension_medium_shot'
        }

        return composition_map.get(scene.scene_type, 'balanced_composition')

    def _determine_overall_structure(self, narrative_arcs: List[NarrativeArc]) -> str:
        """Determine the overall narrative structure."""

        arc_types = [arc.element for arc in narrative_arcs]

        # Check for classic three-act structure
        has_setup = any(elem in [NarrativeElement.EXPOSITION, NarrativeElement.INCITING_INCIDENT] for elem in arc_types)
        has_confrontation = any(elem in [NarrativeElement.RISING_ACTION, NarrativeElement.CLIMAX] for elem in arc_types)
        has_resolution = any(elem in [NarrativeElement.FALLING_ACTION, NarrativeElement.RESOLUTION] for elem in arc_types)

        if has_setup and has_confrontation and has_resolution:
            return "three_act_structure"
        elif len(narrative_arcs) >= 5:
            return "five_act_structure"
        elif any(elem == NarrativeElement.CLIMAX for elem in arc_types):
            return "dramatic_arc"
        else:
            return "episodic_structure"