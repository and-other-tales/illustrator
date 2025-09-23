"""Comprehensive unit tests for narrative analysis functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from illustrator.narrative_analysis import (
    NarrativeAnalyzer,
    NarrativeStructure,
    Genre,
    NarrativeElement,
    ThematicElement,
    CharacterArc,
    NarrativeArc,
    LiteraryDevice
)
from illustrator.models import Chapter, EmotionalMoment, EmotionalTone


class TestNarrativeStructure:
    """Test narrative structure data class."""

    def test_narrative_structure_creation(self):
        """Test creating narrative structure."""
        structure = NarrativeStructure(
            structure_type=StructureType.THREE_ACT,
            act_boundaries=[0, 1000, 2000, 3000],
            climax_position=2500,
            key_plot_points=[500, 1500, 2500],
            pacing_analysis="Well-paced with clear act divisions"
        )

        assert structure.structure_type == StructureType.THREE_ACT
        assert len(structure.act_boundaries) == 4
        assert structure.climax_position == 2500
        assert len(structure.key_plot_points) == 3

    def test_narrative_structure_defaults(self):
        """Test narrative structure with default values."""
        structure = NarrativeStructure(
            structure_type=StructureType.HERO_JOURNEY
        )

        assert structure.act_boundaries == []
        assert structure.climax_position == 0
        assert structure.key_plot_points == []


class TestNarrativeElement:
    """Test narrative element data class."""

    def test_narrative_element_creation(self):
        """Test creating narrative element."""
        element = NarrativeElement(
            element_type="inciting_incident",
            position=500,
            description="Hero receives call to adventure",
            importance=0.9,
            emotional_impact=0.8
        )

        assert element.element_type == "inciting_incident"
        assert element.position == 500
        assert element.description == "Hero receives call to adventure"
        assert element.importance == 0.9
        assert element.emotional_impact == 0.8


class TestThematicElement:
    """Test thematic element data class."""

    def test_thematic_element_creation(self):
        """Test creating thematic element."""
        element = ThematicElement(
            theme="redemption",
            evidence=["character seeks forgiveness", "makes amends for past"],
            strength=0.8,
            chapter_references=["ch-1", "ch-5", "ch-10"]
        )

        assert element.theme == "redemption"
        assert len(element.evidence) == 2
        assert element.strength == 0.8
        assert len(element.chapter_references) == 3


class TestCharacterArc:
    """Test character arc data class."""

    def test_character_arc_creation(self):
        """Test creating character arc."""
        arc = CharacterArc(
            character_name="John",
            arc_type=CharacterArcType.POSITIVE_CHANGE,
            starting_state="naive and inexperienced",
            ending_state="wise and confident",
            key_moments=[1000, 2000, 3000],
            transformation_triggers=["mentor's death", "major defeat", "final confrontation"]
        )

        assert arc.character_name == "John"
        assert arc.arc_type == CharacterArcType.POSITIVE_CHANGE
        assert arc.starting_state == "naive and inexperienced"
        assert len(arc.key_moments) == 3
        assert len(arc.transformation_triggers) == 3


class TestGenreClassifier:
    """Test genre classification functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock()
        self.classifier = GenreClassifier(self.mock_llm)

    def test_initialization(self):
        """Test classifier initialization."""
        assert self.classifier.llm == self.mock_llm

    def test_extract_genre_keywords(self):
        """Test genre keyword extraction."""
        fantasy_text = "The wizard cast a spell while the dragon soared overhead. Magic filled the air."
        mystery_text = "The detective examined the crime scene carefully, searching for clues."
        romance_text = "Their hearts beat as one as they shared their first kiss under the starlight."

        fantasy_keywords = self.classifier._extract_genre_keywords(fantasy_text)
        mystery_keywords = self.classifier._extract_genre_keywords(mystery_text)
        romance_keywords = self.classifier._extract_genre_keywords(romance_text)

        assert any(keyword in ["wizard", "spell", "dragon", "magic"] for keyword in fantasy_keywords)
        assert any(keyword in ["detective", "crime", "clues"] for keyword in mystery_keywords)
        assert any(keyword in ["hearts", "kiss", "starlight"] for keyword in romance_keywords)

    def test_calculate_genre_scores(self):
        """Test genre score calculation."""
        text = "The detective investigated the mysterious murder in the old mansion."
        keywords = ["detective", "mysterious", "murder", "mansion"]

        scores = self.classifier._calculate_genre_scores(keywords)

        assert isinstance(scores, dict)
        assert Genre.MYSTERY in scores
        assert scores[Genre.MYSTERY] > 0

    @pytest.mark.asyncio
    async def test_classify_with_llm_success(self):
        """Test successful LLM-based genre classification."""
        self.mock_llm.ainvoke.return_value.content = """
        Primary Genre: Mystery/Thriller
        Secondary Genres: Drama, Suspense
        Confidence: 0.85
        Reasoning: Contains detective work, crime investigation, and suspenseful elements.
        """

        text = "Sample mystery text..."
        result = await self.classifier._classify_with_llm(text)

        assert "primary_genre" in result
        assert "confidence" in result
        self.mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_with_llm_failure(self):
        """Test LLM failure handling in genre classification."""
        self.mock_llm.ainvoke.side_effect = Exception("API Error")

        text = "Sample text..."
        result = await self.classifier._classify_with_llm(text)

        assert result == {}

    @pytest.mark.asyncio
    async def test_classify_genre(self):
        """Test complete genre classification."""
        self.mock_llm.ainvoke.return_value.content = """
        Primary Genre: Fantasy
        Secondary Genres: Adventure, Coming-of-Age
        Confidence: 0.9
        """

        text = "The young wizard embarked on a magical quest to save the kingdom."
        classification = await self.classifier.classify_genre(text)

        assert "primary_genre" in classification
        assert "secondary_genres" in classification
        assert "confidence_score" in classification
        assert "keyword_analysis" in classification


class TestThematicAnalyzer:
    """Test thematic analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock()
        self.analyzer = ThematicAnalyzer(self.mock_llm)

    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.llm == self.mock_llm

    def test_extract_thematic_keywords(self):
        """Test thematic keyword extraction."""
        text = """
        The story explores themes of redemption and forgiveness.
        Love conquers all obstacles in the end.
        The protagonist struggles with questions of identity and belonging.
        """

        keywords = self.analyzer._extract_thematic_keywords(text)

        expected_keywords = ["redemption", "forgiveness", "love", "identity", "belonging"]
        assert any(keyword in keywords for keyword in expected_keywords)

    def test_identify_common_themes(self):
        """Test common theme identification."""
        keywords = ["love", "sacrifice", "redemption", "forgiveness", "identity", "belonging"]
        themes = self.analyzer._identify_common_themes(keywords)

        assert len(themes) > 0
        assert any(theme in ["love", "redemption", "identity"] for theme in themes)

    @pytest.mark.asyncio
    async def test_analyze_themes_with_llm_success(self):
        """Test successful LLM-based thematic analysis."""
        self.mock_llm.ainvoke.return_value.content = """
        Major Themes:
        1. Redemption - Character seeks to make amends for past mistakes
        2. Love and Sacrifice - Characters make sacrifices for those they love
        3. Identity - Protagonist struggles with sense of self

        Theme Strength: Redemption (0.9), Love (0.8), Identity (0.7)
        """

        text = "Sample text with thematic content..."
        result = await self.analyzer._analyze_themes_with_llm(text)

        assert isinstance(result, list)
        self.mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_themes_with_llm_failure(self):
        """Test LLM failure handling in thematic analysis."""
        self.mock_llm.ainvoke.side_effect = Exception("API Error")

        text = "Sample text..."
        result = await self.analyzer._analyze_themes_with_llm(text)

        assert result == []

    @pytest.mark.asyncio
    async def test_analyze_themes(self):
        """Test complete thematic analysis."""
        self.mock_llm.ainvoke.return_value.content = """
        Major Themes:
        1. Coming of Age - Protagonist grows from child to adult
        2. Good vs Evil - Clear battle between light and darkness
        3. Friendship - Bonds between characters drive the plot
        """

        text = "A young hero and their friends battle against dark forces."
        themes = await self.analyzer.analyze_themes(text)

        assert len(themes) > 0
        assert all(isinstance(theme, ThematicElement) for theme in themes)
        assert all(theme.strength > 0 for theme in themes)


class TestCharacterArcAnalyzer:
    """Test character arc analysis functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock()
        self.analyzer = CharacterArcAnalyzer(self.mock_llm)

    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.llm == self.mock_llm

    def test_identify_character_mentions(self):
        """Test character mention identification."""
        text = """
        John walked into the room where Mary was waiting.
        Sarah called out to them from the kitchen.
        Dr. Peterson arrived shortly after.
        """

        mentions = self.analyzer._identify_character_mentions(text)

        expected_characters = {"John", "Mary", "Sarah", "Dr. Peterson"}
        found_characters = set(mentions.keys())
        assert expected_characters.issubset(found_characters)

    def test_classify_arc_type(self):
        """Test arc type classification."""
        positive_arc = "character grows from weakness to strength"
        negative_arc = "character falls from grace into corruption"
        flat_arc = "character remains steadfast in their beliefs"

        assert self.analyzer._classify_arc_type(positive_arc) == CharacterArcType.POSITIVE_CHANGE
        assert self.analyzer._classify_arc_type(negative_arc) == CharacterArcType.NEGATIVE_CHANGE
        assert self.analyzer._classify_arc_type(flat_arc) == CharacterArcType.FLAT

    @pytest.mark.asyncio
    async def test_analyze_character_development_success(self):
        """Test successful character development analysis."""
        self.mock_llm.ainvoke.return_value.content = """
        Character: John
        Arc Type: Positive Change
        Starting State: Naive and fearful
        Ending State: Brave and wise
        Key Transformation Points:
        - Chapter 3: Faces first real challenge
        - Chapter 7: Overcomes major fear
        - Chapter 12: Makes ultimate sacrifice
        """

        text = "Sample text with character development..."
        character_mentions = {"John": [100, 500, 1000]}

        arcs = await self.analyzer._analyze_character_development(text, character_mentions)

        assert len(arcs) > 0
        self.mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_character_development_failure(self):
        """Test character development analysis with LLM failure."""
        self.mock_llm.ainvoke.side_effect = Exception("API Error")

        text = "Sample text..."
        character_mentions = {"John": [100, 500]}

        arcs = await self.analyzer._analyze_character_development(text, character_mentions)

        assert arcs == []

    @pytest.mark.asyncio
    async def test_analyze_character_arcs(self):
        """Test complete character arc analysis."""
        self.mock_llm.ainvoke.return_value.content = """
        Character Analysis:
        John - Positive Change Arc: Grows from coward to hero
        Mary - Flat Arc: Remains loyal mentor throughout
        Villain - Negative Change Arc: Becomes increasingly corrupt
        """

        text = "A story of heroes and villains with character development."
        arcs = await self.analyzer.analyze_character_arcs(text)

        assert len(arcs) > 0
        assert all(isinstance(arc, CharacterArc) for arc in arcs)


class TestNarrativeAnalyzer:
    """Test the main narrative analyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock()
        self.analyzer = NarrativeAnalyzer(self.mock_llm)

    def test_initialization(self):
        """Test analyzer initialization."""
        assert self.analyzer.llm == self.mock_llm
        assert isinstance(self.analyzer.genre_classifier, GenreClassifier)
        assert isinstance(self.analyzer.thematic_analyzer, ThematicAnalyzer)
        assert isinstance(self.analyzer.character_arc_analyzer, CharacterArcAnalyzer)

    def test_detect_narrative_structure_patterns(self):
        """Test narrative structure pattern detection."""
        text = """
        Chapter 1: The Call to Adventure
        Our hero lives in the ordinary world until receiving a call to adventure.

        Chapter 5: Crossing the Threshold
        The hero leaves the ordinary world and enters the special world.

        Chapter 10: The Ordeal
        The hero faces their greatest fear and most difficult challenge.

        Chapter 15: The Return
        The hero returns transformed to the ordinary world.
        """

        structure = self.analyzer._detect_narrative_structure_patterns(text)

        assert structure.structure_type == StructureType.HERO_JOURNEY
        assert len(structure.key_plot_points) > 0

    def test_identify_pacing_elements(self):
        """Test pacing element identification."""
        text = """
        The action sequence exploded across three intense chapters.
        Then came a quiet moment of reflection and character development.
        Building tension through subtle clues and mounting danger.
        The climactic battle raged for pages, decisive and final.
        """

        pacing_analysis = self.analyzer._identify_pacing_elements(text)

        assert isinstance(pacing_analysis, str)
        assert len(pacing_analysis) > 0

    @pytest.mark.asyncio
    async def test_analyze_narrative_structure_success(self):
        """Test successful narrative structure analysis."""
        self.mock_llm.ainvoke.return_value.content = """
        Narrative Structure: Three-Act Structure
        Act 1: Setup and inciting incident (0-25%)
        Act 2: Development and complications (25-75%)
        Act 3: Climax and resolution (75-100%)

        Key Plot Points:
        - Inciting Incident: 15%
        - Plot Point 1: 25%
        - Midpoint: 50%
        - Plot Point 2: 75%
        - Climax: 85%
        """

        text = "Sample narrative text..."
        structure = await self.analyzer._analyze_narrative_structure(text)

        assert isinstance(structure, NarrativeStructure)
        self.mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_narrative_structure_failure(self):
        """Test narrative structure analysis with LLM failure."""
        self.mock_llm.ainvoke.side_effect = Exception("API Error")

        text = "Sample text..."
        structure = await self.analyzer._analyze_narrative_structure(text)

        # Should return default structure
        assert isinstance(structure, NarrativeStructure)
        assert structure.structure_type == StructureType.LINEAR

    @pytest.mark.asyncio
    async def test_identify_illustration_opportunities(self):
        """Test illustration opportunity identification."""
        chapters = [
            Chapter(id="ch-1", title="Chapter 1", content="Action scene with battle", emotional_moments=[]),
            Chapter(id="ch-2", title="Chapter 2", content="Quiet character moment", emotional_moments=[]),
            Chapter(id="ch-3", title="Chapter 3", content="Dramatic revelation scene", emotional_moments=[])
        ]

        narrative_structure = NarrativeStructure(
            structure_type=StructureType.THREE_ACT,
            climax_position=2500,
            key_plot_points=[500, 1500, 2500]
        )

        opportunities = self.analyzer._identify_illustration_opportunities(chapters, narrative_structure)

        assert len(opportunities) > 0
        assert all("chapter_id" in opp for opp in opportunities)
        assert all("reason" in opp for opp in opportunities)
        assert all("priority" in opp for opp in opportunities)

    @pytest.mark.asyncio
    async def test_analyze_complete_narrative(self):
        """Test complete narrative analysis."""
        chapters = [
            Chapter(id="ch-1", title="Beginning", content="Hero's ordinary world", emotional_moments=[]),
            Chapter(id="ch-2", title="Adventure", content="Call to adventure", emotional_moments=[]),
            Chapter(id="ch-3", title="Return", content="Hero returns transformed", emotional_moments=[])
        ]

        # Mock all LLM calls
        self.mock_llm.ainvoke.return_value.content = """
        Analysis complete: Fantasy adventure with hero's journey structure.
        Themes: Coming of age, good vs evil, friendship.
        Character arcs: Positive change for protagonist.
        """

        analysis = await self.analyzer.analyze_complete_narrative(chapters)

        assert "genre_classification" in analysis
        assert "narrative_structure" in analysis
        assert "thematic_elements" in analysis
        assert "character_arcs" in analysis
        assert "illustration_opportunities" in analysis

    @pytest.mark.asyncio
    async def test_get_narrative_summary(self):
        """Test narrative summary generation."""
        # Set up some analysis results
        self.analyzer.last_analysis = {
            "genre_classification": {"primary_genre": "Fantasy", "confidence_score": 0.9},
            "narrative_structure": NarrativeStructure(structure_type=StructureType.HERO_JOURNEY),
            "thematic_elements": [ThematicElement(theme="heroism", evidence=[], strength=0.8)],
            "character_arcs": [CharacterArc(character_name="Hero", arc_type=CharacterArcType.POSITIVE_CHANGE)]
        }

        summary = self.analyzer.get_narrative_summary()

        assert "primary_genre" in summary
        assert "structure_type" in summary
        assert "major_themes" in summary
        assert "character_count" in summary

    def test_get_analysis_confidence(self):
        """Test analysis confidence calculation."""
        # Mock some analysis results
        self.analyzer.last_analysis = {
            "genre_classification": {"confidence_score": 0.9},
            "narrative_structure": NarrativeStructure(structure_type=StructureType.THREE_ACT),
            "thematic_elements": [
                ThematicElement(theme="love", evidence=[], strength=0.8),
                ThematicElement(theme="sacrifice", evidence=[], strength=0.7)
            ]
        }

        confidence = self.analyzer.get_analysis_confidence()

        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.7  # Should be reasonably confident