"""Advanced narrative structure recognition and literary analysis for manuscript illustration."""

import re
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Set, Any
from enum import Enum

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from illustrator.utils import parse_llm_json

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

    THEME_KEYWORDS = {
        "family": ["family", "sister", "brother", "mother", "father"],
        "courage": ["courage", "brave", "bold", "fearless"],
        "mystery": ["mystery", "secret", "hidden", "unknown"],
        "loss": ["loss", "grief", "mourning", "missing"],
        "hope": ["hope", "optimism", "faith", "promise"],
    }

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    async def analyze_complete_narrative(self, chapters: List[Chapter]) -> Dict[str, Any]:
        """Provide a manuscript-level narrative analysis across all chapters."""

        if not chapters:
            return {
                "chapters_analyzed": 0,
                "narrative_structure": {
                    "overall_structure": "unknown",
                    "narrative_arcs": [],
                    "pacing_analysis": {},
                },
                "character_arcs": [],
                "thematic_elements": [],
                "genre_classification": {
                    "primary_genre": "unknown",
                    "secondary_genres": [],
                    "detected_indicators": [],
                    "confidence": 0.0,
                },
                "analysis_notes": {
                    "total_word_count": 0,
                    "status": "no_content",
                },
            }

        full_text = "\n\n".join(chapter.content or "" for chapter in chapters)

        try:
            narrative_arcs = self._find_pattern_based_arcs(full_text)
        except Exception as exc:
            logger.warning("Pattern-based arc extraction failed: %s", exc)
            narrative_arcs = []

        try:
            pacing_analysis = await self._analyze_pacing(full_text, narrative_arcs)
        except Exception as exc:
            logger.warning("Pacing analysis failed: %s", exc)
            pacing_analysis = {
                "average_sentence_length": 0.0,
                "average_paragraph_length": 0.0,
                "narrative_intensity": 0.0,
                "pacing_score": 0.0,
            }

        try:
            character_arcs = self._build_character_arcs_from_text(full_text)
        except Exception as exc:
            logger.warning("Character arc heuristics failed: %s", exc)
            character_arcs = []

        try:
            thematic_elements = self._derive_thematic_elements(full_text, character_arcs)
        except Exception as exc:
            logger.warning("Thematic heuristic analysis failed: %s", exc)
            thematic_elements = []

        genre_indicators = self._identify_genre(full_text)
        primary_genre = genre_indicators[0].value if genre_indicators else "unknown"
        secondary_genres = [genre.value for genre in genre_indicators[1:]]

        overall_structure = self._determine_overall_structure(narrative_arcs)

        total_word_count = sum(
            chapter.word_count or len((chapter.content or "").split())
            for chapter in chapters
        )

        return {
            "chapters_analyzed": len(chapters),
            "narrative_structure": {
                "overall_structure": overall_structure,
                "narrative_arcs": narrative_arcs,
                "pacing_analysis": pacing_analysis,
            },
            "character_arcs": character_arcs,
            "thematic_elements": thematic_elements,
            "genre_classification": {
                "primary_genre": primary_genre,
                "secondary_genres": secondary_genres,
                "detected_indicators": [genre.value for genre in genre_indicators],
                "confidence": 0.6 if genre_indicators else 0.0,
            },
            "analysis_notes": {
                "total_word_count": total_word_count,
                "status": "completed",
            },
        }

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

    def _extract_character_names(self, text: str) -> List[str]:
        """Heuristically extract capitalized character names from the text."""

        candidates = re.findall(r"\b[A-Z][a-zA-Z]+\b", text)
        ignore = {
            "Chapter", "The", "And", "But", "When", "Then", "Once", "After",
            "Before", "While", "During", "Into", "From", "Their", "They",
            "She", "He", "Her", "His", "Its", "For", "With", "At",
        }

        seen = set()
        names = []
        for candidate in candidates:
            if candidate in ignore or len(candidate) < 3:
                continue
            if candidate not in seen:
                seen.add(candidate)
                names.append(candidate)

        return names[:10]

    def _build_character_arcs_from_text(self, text: str) -> List[CharacterArc]:
        """Build lightweight character arcs using heuristic analysis."""

        arcs: List[CharacterArc] = []
        for name in self._extract_character_names(text):
            matches = [match.start() for match in re.finditer(rf"\b{name}\b", text)]
            if not matches:
                continue

            significance = min(1.0, 0.3 + 0.1 * len(matches))
            emotional_journey = [(matches[0], EmotionalTone.NEUTRAL)]

            arc = CharacterArc(
                character_name=name,
                arc_type="growth" if len(matches) > 1 else "static",
                starting_state={"summary": "Introduced in narrative."},
                ending_state={"summary": "Continues to influence the story."},
                key_moments=matches[:5],
                emotional_journey=emotional_journey,
                relationships_evolution={},
                significance=significance,
            )
            arcs.append(arc)

        return arcs

    def _derive_thematic_elements(
        self,
        text: str,
        character_arcs: List[CharacterArc],
    ) -> List[ThematicElement]:
        """Infer thematic elements by scanning for known keyword clusters."""

        elements: List[ThematicElement] = []
        lower_text = text.lower()

        for theme, keywords in self.THEME_KEYWORDS.items():
            evidence_positions: List[int] = []
            supporting_quotes: List[str] = []

            for keyword in keywords:
                for match in re.finditer(rf"\b{re.escape(keyword)}\b", lower_text):
                    start = match.start()
                    evidence_positions.append(start)

                    excerpt_start = max(0, start - 60)
                    excerpt_end = min(len(text), start + 120)
                    snippet = text[excerpt_start:excerpt_end].strip()
                    if snippet:
                        supporting_quotes.append(snippet)

            if not evidence_positions:
                continue

            character_connections = [
                arc.character_name
                for arc in character_arcs
                if arc.character_name.lower() in lower_text
            ]

            element = ThematicElement(
                theme=theme,
                evidence_positions=evidence_positions[:5],
                supporting_quotes=supporting_quotes[:3],
                character_connections=list(dict.fromkeys(character_connections))[:5],
                symbolic_elements=[theme],
                development_arc=f"The theme of {theme} recurs {len(evidence_positions)} times across the manuscript.",
                visual_manifestations=[f"Depict {theme} through key emotional moments."],
            )
            elements.append(element)

        return elements

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
            return parse_llm_json(response.content)

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
            theme_data = parse_llm_json(response.content)

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
            return parse_llm_json(response.content)

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


class GenreClassifier:
    """Classifier for automatically identifying literary genres from text."""

    def __init__(self, llm: BaseChatModel):
        """Initialize the genre classifier.

        Args:
            llm: Language model for classification tasks
        """
        self.llm = llm

        # Genre-specific keywords
        self._genre_keywords = {
            Genre.FANTASY: ["wizard", "magic", "dragon", "spell", "enchant", "mystical", "realm", "quest"],
            Genre.MYSTERY: ["detective", "crime", "clues", "investigation", "murder", "suspect", "evidence"],
            Genre.ROMANCE: ["love", "hearts", "kiss", "romantic", "passion", "romance", "relationship"],
            Genre.SCIENCE_FICTION: ["space", "technology", "future", "alien", "robot", "scientific", "galaxy"],
            Genre.HORROR: ["terror", "fear", "ghost", "monster", "scary", "nightmare", "darkness"],
            Genre.THRILLER: ["suspense", "danger", "chase", "tension", "threat", "escape", "pursuit"],
            Genre.ADVENTURE: ["journey", "exploration", "discovery", "adventure", "expedition", "treasure"],
            Genre.DRAMA: ["emotion", "conflict", "relationship", "struggle", "character", "development"],
            Genre.HISTORICAL_FICTION: ["period", "historical", "era", "ancient", "century", "past", "tradition"],
            Genre.LITERARY_FICTION: ["literary", "modern", "contemporary", "character", "style", "prose"],
            Genre.YOUNG_ADULT: ["youth", "teen", "coming", "age", "school", "adolescent", "growing"],
            Genre.COMEDY: ["funny", "humor", "laugh", "comedy", "amusing", "wit", "satire"]
        }

    def _extract_genre_keywords(self, text: str) -> List[str]:
        """Extract genre-relevant keywords from text.

        Args:
            text: The text to analyze

        Returns:
            List of relevant keywords found in text
        """
        text_lower = text.lower()
        found_keywords = []

        for genre, keywords in self._genre_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)

        return found_keywords

    def _calculate_genre_scores(self, keywords: List[str]) -> Dict[Genre, float]:
        """Calculate genre probability scores based on keywords.

        Args:
            keywords: List of keywords found in text

        Returns:
            Dictionary mapping genres to their probability scores
        """
        scores = {genre: 0.0 for genre in Genre}

        for keyword in keywords:
            for genre, genre_keywords in self._genre_keywords.items():
                if keyword in genre_keywords:
                    scores[genre] += 1.0

        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {genre: score / total_score for genre, score in scores.items()}

        return scores

    async def _classify_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM to classify genre with reasoning.

        Args:
            text: Text to classify

        Returns:
            Classification result with confidence and reasoning
        """
        try:
            messages = [
                SystemMessage(content="""You are a literary genre classification expert.
                Analyze the given text and determine its primary and secondary genres.

                Available genres: Fantasy, Mystery, Romance, Science Fiction, Horror,
                Thriller, Adventure, Drama, Historical, Contemporary

                Respond in this exact format:
                Primary Genre: [genre]
                Secondary Genres: [genre1, genre2, ...]
                Confidence: [0.0-1.0]
                Reasoning: [brief explanation]"""),
                HumanMessage(content=f"Classify this text: {text}")
            ]

            response = await self.llm.ainvoke(messages)

            # Parse the response
            lines = response.content.strip().split('\n')
            result = {
                'primary_genre': None,
                'secondary_genres': [],
                'confidence': 0.5,
                'reasoning': ''
            }

            for line in lines:
                if line.startswith('Primary Genre:'):
                    result['primary_genre'] = line.split(':', 1)[1].strip()
                elif line.startswith('Secondary Genres:'):
                    secondary = line.split(':', 1)[1].strip()
                    result['secondary_genres'] = [g.strip() for g in secondary.split(',') if g.strip()]
                elif line.startswith('Confidence:'):
                    try:
                        result['confidence'] = float(line.split(':', 1)[1].strip())
                    except ValueError:
                        result['confidence'] = 0.5
                elif line.startswith('Reasoning:'):
                    result['reasoning'] = line.split(':', 1)[1].strip()

            return result

        except Exception as e:
            logger.error(f"Error in LLM genre classification: {e}")
            return {
                'primary_genre': 'Unknown',
                'secondary_genres': [],
                'confidence': 0.0,
                'reasoning': f'Classification failed: {str(e)}'
            }

    async def classify_genre(self, text: str) -> Dict[str, Any]:
        """Classify the genre of the given text.

        Args:
            text: Text to classify

        Returns:
            Classification result with genre, confidence, and reasoning
        """
        # Extract keywords and calculate initial scores
        keywords = self._extract_genre_keywords(text)
        keyword_scores = self._calculate_genre_scores(keywords)

        # Get LLM classification
        llm_result = await self._classify_with_llm(text)

        # Combine results
        return {
            'primary_genre': llm_result['primary_genre'],
            'secondary_genres': llm_result['secondary_genres'],
            'confidence': llm_result['confidence'],
            'reasoning': llm_result['reasoning'],
            'keyword_scores': keyword_scores,
            'keywords_found': keywords
        }


class ThematicAnalyzer:
    """Analyzer for identifying themes and symbolic elements in text."""

    def __init__(self, llm: BaseChatModel):
        """Initialize the thematic analyzer.

        Args:
            llm: Language model for analysis tasks
        """
        self.llm = llm

        # Common thematic keywords
        self._thematic_keywords = {
            'love': ['love', 'affection', 'romance', 'devotion', 'passion'],
            'death': ['death', 'mortality', 'dying', 'grave', 'funeral'],
            'power': ['power', 'control', 'authority', 'dominance', 'influence'],
            'freedom': ['freedom', 'liberty', 'independence', 'escape', 'release'],
            'identity': ['identity', 'self', 'who', 'being', 'existence'],
            'justice': ['justice', 'fairness', 'right', 'wrong', 'moral'],
            'redemption': ['redemption', 'forgiveness', 'salvation', 'atonement'],
            'sacrifice': ['sacrifice', 'giving', 'offering', 'loss', 'surrender'],
            'betrayal': ['betrayal', 'deception', 'lies', 'trust', 'broken'],
            'hope': ['hope', 'optimism', 'future', 'possibility', 'faith']
        }

    def _extract_thematic_keywords(self, text: str) -> List[str]:
        """Extract theme-related keywords from text.

        Args:
            text: Text to analyze

        Returns:
            List of thematic keywords found
        """
        text_lower = text.lower()
        found_keywords = []

        for theme, keywords in self._thematic_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)

        return found_keywords

    def _identify_common_themes(self, keywords: List[str]) -> List[str]:
        """Identify common themes based on keywords.

        Args:
            keywords: List of thematic keywords

        Returns:
            List of identified themes
        """
        theme_scores = {}

        for keyword in keywords:
            for theme, theme_keywords in self._thematic_keywords.items():
                if keyword in theme_keywords:
                    theme_scores[theme] = theme_scores.get(theme, 0) + 1

        # Return themes with at least 2 keyword matches
        return [theme for theme, score in theme_scores.items() if score >= 2]

    async def _analyze_themes_with_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM to analyze themes in depth.

        Args:
            text: Text to analyze

        Returns:
            Thematic analysis result
        """
        try:
            messages = [
                SystemMessage(content="""You are a literary thematic analysis expert.
                Analyze the given text and identify the major themes, symbols, and motifs.

                Respond in this format:
                Major Themes: [theme1, theme2, ...]
                Symbols: [symbol1, symbol2, ...]
                Motifs: [motif1, motif2, ...]
                Analysis: [detailed thematic analysis]"""),
                HumanMessage(content=f"Analyze themes in this text: {text}")
            ]

            response = await self.llm.ainvoke(messages)

            # Parse response
            lines = response.content.strip().split('\n')
            result = {
                'major_themes': [],
                'symbols': [],
                'motifs': [],
                'analysis': ''
            }

            for line in lines:
                if line.startswith('Major Themes:'):
                    themes = line.split(':', 1)[1].strip()
                    result['major_themes'] = [t.strip() for t in themes.split(',') if t.strip()]
                elif line.startswith('Symbols:'):
                    symbols = line.split(':', 1)[1].strip()
                    result['symbols'] = [s.strip() for s in symbols.split(',') if s.strip()]
                elif line.startswith('Motifs:'):
                    motifs = line.split(':', 1)[1].strip()
                    result['motifs'] = [m.strip() for m in motifs.split(',') if m.strip()]
                elif line.startswith('Analysis:'):
                    result['analysis'] = line.split(':', 1)[1].strip()

            return result

        except Exception as e:
            logger.error(f"Error in LLM thematic analysis: {e}")
            return {
                'major_themes': [],
                'symbols': [],
                'motifs': [],
                'analysis': f'Analysis failed: {str(e)}'
            }

    async def analyze_themes(self, text: str) -> Dict[str, Any]:
        """Analyze themes in the given text.

        Args:
            text: Text to analyze

        Returns:
            Comprehensive thematic analysis
        """
        # Extract keywords and identify common themes
        keywords = self._extract_thematic_keywords(text)
        common_themes = self._identify_common_themes(keywords)

        # Get detailed LLM analysis
        llm_analysis = await self._analyze_themes_with_llm(text)

        return {
            'keywords_found': keywords,
            'common_themes': common_themes,
            'major_themes': llm_analysis['major_themes'],
            'symbols': llm_analysis['symbols'],
            'motifs': llm_analysis['motifs'],
            'detailed_analysis': llm_analysis['analysis']
        }


class CharacterArcAnalyzer:
    """Analyzer for identifying character development patterns and arcs."""

    def __init__(self, llm: BaseChatModel):
        """Initialize the character arc analyzer.

        Args:
            llm: Language model for analysis tasks
        """
        self.llm = llm

        # Character arc types
        self._arc_types = {
            'hero_journey': ['call', 'adventure', 'mentor', 'trials', 'transformation', 'return'],
            'tragic_fall': ['hubris', 'downfall', 'consequence', 'destruction', 'tragic'],
            'redemption': ['mistake', 'guilt', 'journey', 'redemption', 'forgiveness'],
            'coming_of_age': ['youth', 'innocence', 'growth', 'maturity', 'responsibility'],
            'villain_arc': ['corruption', 'evil', 'power', 'darkness', 'antagonist']
        }

    def _identify_character_mentions(self, text: str) -> List[str]:
        """Identify potential character mentions in text.

        Args:
            text: Text to analyze

        Returns:
            List of potential character names/pronouns
        """
        import re

        # Simple pattern to find capitalized words (potential names)
        names = re.findall(r'\b[A-Z][a-z]+\b', text)

        # Add common pronouns
        pronouns = re.findall(r'\b(?:he|she|him|her|his|hers|they|them|their)\b', text.lower())

        return list(set(names + pronouns))

    def _classify_arc_type(self, text: str) -> str:
        """Classify the type of character arc based on text content.

        Args:
            text: Text to analyze

        Returns:
            Most likely arc type
        """
        text_lower = text.lower()
        arc_scores = {}

        for arc_type, keywords in self._arc_types.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                arc_scores[arc_type] = score

        if arc_scores:
            return max(arc_scores.items(), key=lambda x: x[1])[0]
        return 'undefined'

    async def _analyze_character_development(self, text: str, character: str) -> Dict[str, Any]:
        """Analyze character development using LLM.

        Args:
            text: Text containing character
            character: Character to analyze

        Returns:
            Character development analysis
        """
        try:
            messages = [
                SystemMessage(content=f"""You are a character development analyst.
                Analyze the character development for '{character}' in the given text.

                Respond in this format:
                Character: {character}
                Arc Type: [hero_journey/tragic_fall/redemption/coming_of_age/villain_arc/other]
                Development: [description of character growth/change]
                Key Moments: [moment1, moment2, ...]
                Traits: [trait1, trait2, ...]"""),
                HumanMessage(content=f"Analyze character development in: {text}")
            ]

            response = await self.llm.ainvoke(messages)

            # Parse response
            lines = response.content.strip().split('\n')
            result = {
                'character': character,
                'arc_type': 'undefined',
                'development': '',
                'key_moments': [],
                'traits': []
            }

            for line in lines:
                if line.startswith('Arc Type:'):
                    result['arc_type'] = line.split(':', 1)[1].strip()
                elif line.startswith('Development:'):
                    result['development'] = line.split(':', 1)[1].strip()
                elif line.startswith('Key Moments:'):
                    moments = line.split(':', 1)[1].strip()
                    result['key_moments'] = [m.strip() for m in moments.split(',') if m.strip()]
                elif line.startswith('Traits:'):
                    traits = line.split(':', 1)[1].strip()
                    result['traits'] = [t.strip() for t in traits.split(',') if t.strip()]

            return result

        except Exception as e:
            logger.error(f"Error in character development analysis: {e}")
            return {
                'character': character,
                'arc_type': 'undefined',
                'development': f'Analysis failed: {str(e)}',
                'key_moments': [],
                'traits': []
            }

    async def analyze_character_arcs(self, text: str) -> Dict[str, Any]:
        """Analyze character arcs in the given text.

        Args:
            text: Text to analyze

        Returns:
            Character arc analysis results
        """
        # Identify potential characters
        characters = self._identify_character_mentions(text)

        # Classify overall arc type
        arc_type = self._classify_arc_type(text)

        # Analyze development for main characters (limit to first 3)
        character_analyses = []
        for character in characters[:3]:
            analysis = await self._analyze_character_development(text, character)
            character_analyses.append(analysis)

        return {
            'overall_arc_type': arc_type,
            'characters_identified': characters,
            'character_analyses': character_analyses
        }
