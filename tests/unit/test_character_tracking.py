"""Comprehensive unit tests for character tracking functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from illustrator.character_tracking import (
    CharacterTracker,
    PhysicalDescription,
    CharacterAppearance,
    CharacterRelationship,
    CharacterRelationshipType,
    CharacterProfile
)
from illustrator.models import Chapter, EmotionalMoment, EmotionalTone


class TestPhysicalDescription:
    """Test physical description data class."""

    def test_physical_description_creation(self):
        """Test creating a physical description."""
        desc = PhysicalDescription(
            height="tall",
            build="lean",
            hair_color="brown",
            eye_color="blue",
            distinguishing_features="scar on left cheek"
        )

        assert desc.height == "tall"
        assert desc.build == "lean"
        assert desc.hair_color == "brown"
        assert desc.eye_color == "blue"
        assert desc.distinguishing_features == "scar on left cheek"

    def test_physical_description_defaults(self):
        """Test physical description with default values."""
        desc = PhysicalDescription()

        assert desc.height is None
        assert desc.build is None
        assert desc.hair_color is None
        assert desc.eye_color is None


class TestCharacterAppearance:
    """Test character appearance tracking."""

    def test_character_appearance_creation(self):
        """Test creating character appearance record."""
        physical_desc = PhysicalDescription(hair_color="blonde", eye_color="green")

        appearance = CharacterAppearance(
            chapter_id="ch-1",
            scene_context="morning garden scene",
            physical_description=physical_desc,
            emotional_state="peaceful",
            actions=["walking", "smiling"],
            dialogue_tone="gentle"
        )

        assert appearance.chapter_id == "ch-1"
        assert appearance.scene_context == "morning garden scene"
        assert appearance.physical_description.hair_color == "blonde"
        assert appearance.emotional_state == "peaceful"
        assert "walking" in appearance.actions

    def test_character_appearance_defaults(self):
        """Test character appearance with default values."""
        appearance = CharacterAppearance(
            chapter_id="ch-1",
            scene_context="test scene"
        )

        assert appearance.actions == []
        assert appearance.mentioned_traits == []


class TestCharacterRelationship:
    """Test character relationship tracking."""

    def test_character_relationship_creation(self):
        """Test creating character relationship."""
        relationship = CharacterRelationship(
            other_character="Jane",
            relationship_type=CharacterRelationshipType.FRIENDSHIP,
            description="childhood friends",
            emotional_dynamic="supportive",
            first_mentioned_chapter="ch-1"
        )

        assert relationship.other_character == "Jane"
        assert relationship.relationship_type == CharacterRelationshipType.FRIENDSHIP
        assert relationship.description == "childhood friends"
        assert relationship.emotional_dynamic == "supportive"

    def test_relationship_type_enum(self):
        """Test relationship type enumeration."""
        assert CharacterRelationshipType.FAMILY == "family"
        assert CharacterRelationshipType.ROMANTIC == "romantic"
        assert CharacterRelationshipType.ANTAGONISTIC == "antagonistic"


class TestCharacterProfile:
    """Test complete character profile."""

    def test_character_profile_creation(self):
        """Test creating a character profile."""
        physical_desc = PhysicalDescription(height="medium", hair_color="red")

        profile = CharacterProfile(
            name="Sarah",
            primary_role="protagonist",
            physical_description=physical_desc,
            personality_traits=["determined", "kind"],
            background="grew up in small town"
        )

        assert profile.name == "Sarah"
        assert profile.primary_role == "protagonist"
        assert "determined" in profile.personality_traits
        assert profile.appearances == []
        assert profile.relationships == []

    def test_add_appearance(self):
        """Test adding appearance to profile."""
        profile = CharacterProfile(name="John")
        physical_desc = PhysicalDescription(eye_color="brown")

        appearance = CharacterAppearance(
            chapter_id="ch-1",
            scene_context="office meeting",
            physical_description=physical_desc
        )

        profile.appearances.append(appearance)

        assert len(profile.appearances) == 1
        assert profile.appearances[0].chapter_id == "ch-1"


class TestCharacterTracker:
    """Test the main character tracking class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock()
        self.tracker = CharacterTracker(self.mock_llm)

    def test_initialization(self):
        """Test tracker initialization."""
        assert self.tracker.llm == self.mock_llm
        assert self.tracker.characters == {}
        assert self.tracker.chapter_character_cache == {}

    def test_extract_character_names(self):
        """Test character name extraction from text."""
        text = """
        John walked into the room where Mary was waiting.
        "Hello, Sarah," said Dr. Peterson.
        The cat, Whiskers, meowed loudly.
        """

        names = self.tracker._extract_character_names(text)

        expected_names = {"John", "Mary", "Sarah", "Dr. Peterson"}
        assert expected_names.issubset(set(names))

    def test_extract_character_names_with_titles(self):
        """Test character extraction including titles."""
        text = """
        Professor Smith met with Captain Jones.
        Mrs. Williams spoke to Mr. Anderson about the situation.
        """

        names = self.tracker._extract_character_names(text)

        assert "Professor Smith" in names
        assert "Captain Jones" in names
        assert "Mrs. Williams" in names
        assert "Mr. Anderson" in names
        
    def test_extract_character_names_patterns(self):
        """Test character name extraction using regex patterns."""
        text = """
        Elizabeth Bennett was walking with Mr. Darcy through the garden. 
        Colonel Brandon and Marianne Dashwood were discussing poetry.
        The Duke of Wellington arrived at the party fashionably late.
        """
        
        names = self.tracker._extract_character_names_patterns(text)
        
        assert "Elizabeth" in names
        assert "Bennett" in names
        assert "Darcy" in names
        assert "Brandon" in names
        assert "Marianne" in names
        assert "Dashwood" in names
        assert "Wellington" in names
        assert "Elizabeth Bennett" in names
        assert "Marianne Dashwood" in names
        
        # Common words that start with capitals should be excluded
        assert "The" not in names
        assert "Duke" not in names  # "Duke" alone might be filtered as a title
        
    @pytest.mark.asyncio
    async def test_analyze_character_mentions_success(self):
        """Test successful character mention analysis."""
        self.mock_llm.ainvoke.return_value.content = """
        Character Analysis:
        - John: protagonist, tall, determined, appears angry in this scene
        - Mary: supporting character, kind, blonde hair, seems worried
        - Dr. Peterson: authority figure, elderly, wise
        """

        text = "Sample chapter text with characters..."
        character_names = ["John", "Mary", "Dr. Peterson"]

        analysis = await self.tracker._analyze_character_mentions(text, character_names)

        assert isinstance(analysis, dict)
        self.mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_character_mentions_failure(self):
        """Test character mention analysis with LLM failure."""
        self.mock_llm.ainvoke.side_effect = Exception("API Error")

        text = "Sample text..."
        character_names = ["John", "Mary"]

        analysis = await self.tracker._analyze_character_mentions(text, character_names)

        assert analysis == {}

    def test_parse_character_analysis(self):
        """Test parsing of character analysis response."""
        response = """
        Character Analysis:
        - John: protagonist, tall, brown hair, determined personality, appears confident
        - Mary: supporting character, blonde hair, blue eyes, kind nature, seems worried
        """

        analysis = self.tracker._parse_character_analysis(response)

        assert "John" in analysis
        assert "Mary" in analysis
        assert "tall" in analysis["John"]["physical_traits"]
        assert "blonde hair" in analysis["Mary"]["physical_traits"]

    def test_update_character_profile_new_character(self):
        """Test updating profile for new character."""
        character_data = {
            "role": "protagonist",
            "physical_traits": ["tall", "brown hair"],
            "personality_traits": ["determined", "brave"],
            "emotional_state": "confident"
        }

        self.tracker._update_character_profile(
            "John", character_data, "ch-1", "morning scene"
        )

        assert "John" in self.tracker.characters
        profile = self.tracker.characters["John"]
        assert profile.name == "John"
        assert profile.primary_role == "protagonist"
        assert "tall" in profile.physical_description.__dict__.values()

    def test_update_character_profile_existing_character(self):
        """Test updating existing character profile."""
        # Create initial profile
        profile = CharacterProfile(
            name="John",
            primary_role="protagonist",
            physical_description=PhysicalDescription(hair_color="brown")
        )
        self.tracker.characters["John"] = profile

        # Update with new information
        character_data = {
            "physical_traits": ["tall", "muscular"],
            "emotional_state": "determined"
        }

        self.tracker._update_character_profile(
            "John", character_data, "ch-2", "action scene"
        )

        updated_profile = self.tracker.characters["John"]
        assert len(updated_profile.appearances) == 1
        assert updated_profile.appearances[0].chapter_id == "ch-2"

    def test_merge_physical_descriptions(self):
        """Test merging physical descriptions."""
        existing = PhysicalDescription(hair_color="brown", height="tall")
        new_traits = ["muscular build", "blue eyes", "brown hair"]

        merged = self.tracker._merge_physical_descriptions(existing, new_traits)

        assert merged.hair_color == "brown"  # Keep existing
        assert merged.height == "tall"  # Keep existing
        assert "blue" in str(merged.eye_color)  # Add new

    def test_extract_relationships(self):
        """Test relationship extraction from text."""
        text = """
        John spoke to his sister Mary about their father.
        His best friend Tom was also there with his wife Sarah.
        The enemy soldier Captain Smith watched from afar.
        """

        relationships = self.tracker._extract_relationships(text, ["John", "Mary", "Tom", "Sarah", "Captain Smith"])

        assert len(relationships) > 0
        # Should find family relationship between John and Mary
        john_relationships = [r for r in relationships if r.other_character == "Mary"]
        assert len(john_relationships) > 0
        assert any(r.relationship_type == CharacterRelationshipType.FAMILY for r in john_relationships)

    def test_check_consistency_violations(self):
        """Test consistency violation detection."""
        # Create character with existing traits
        profile = CharacterProfile(
            name="John",
            physical_description=PhysicalDescription(hair_color="brown", eye_color="blue")
        )

        # Add appearance with consistent traits
        consistent_appearance = CharacterAppearance(
            chapter_id="ch-1",
            scene_context="scene1",
            physical_description=PhysicalDescription(hair_color="brown")
        )
        profile.appearances.append(consistent_appearance)

        # Add appearance with inconsistent traits
        inconsistent_appearance = CharacterAppearance(
            chapter_id="ch-2",
            scene_context="scene2",
            physical_description=PhysicalDescription(hair_color="blonde")
        )
        profile.appearances.append(inconsistent_appearance)

        violations = self.tracker._check_consistency_violations(profile)

        assert len(violations) > 0
        assert any("hair_color" in str(v) for v in violations)

    @pytest.mark.asyncio
    async def test_track_characters_in_chapter(self):
        """Test tracking characters in a chapter."""
        chapter = Chapter(
            id="ch-1",
            title="Test Chapter",
            content="John walked into the room where Mary was waiting. She smiled at him warmly.",
            emotional_moments=[]
        )

        self.mock_llm.ainvoke.return_value.content = """
        Character Analysis:
        - John: protagonist, tall, confident
        - Mary: supporting character, kind, welcoming
        """

        await self.tracker.track_characters_in_chapter(chapter)

        assert "John" in self.tracker.characters
        assert "Mary" in self.tracker.characters
        assert "ch-1" in self.tracker.chapter_character_cache

    def test_get_character_profile(self):
        """Test getting character profile."""
        profile = CharacterProfile(name="John", primary_role="protagonist")
        self.tracker.characters["John"] = profile

        retrieved = self.tracker.get_character_profile("John")
        assert retrieved == profile

        # Test non-existent character
        assert self.tracker.get_character_profile("Unknown") is None

    def test_get_characters_in_chapter(self):
        """Test getting characters in specific chapter."""
        self.tracker.chapter_character_cache["ch-1"] = ["John", "Mary"]
        self.tracker.chapter_character_cache["ch-2"] = ["John", "Tom"]

        ch1_characters = self.tracker.get_characters_in_chapter("ch-1")
        assert set(ch1_characters) == {"John", "Mary"}

        ch2_characters = self.tracker.get_characters_in_chapter("ch-2")
        assert set(ch2_characters) == {"John", "Tom"}

    def test_get_character_relationships(self):
        """Test getting character relationships."""
        profile = CharacterProfile(name="John")
        relationship = CharacterRelationship(
            other_character="Mary",
            relationship_type=CharacterRelationshipType.FRIENDSHIP,
            description="best friends"
        )
        profile.relationships.append(relationship)
        self.tracker.characters["John"] = profile

        relationships = self.tracker.get_character_relationships("John")
        assert len(relationships) == 1
        assert relationships[0].other_character == "Mary"

    def test_optimize_for_illustration(self):
        """Test optimization for illustration generation."""
        profile = CharacterProfile(
            name="John",
            physical_description=PhysicalDescription(
                hair_color="brown",
                eye_color="blue",
                height="tall"
            ),
            personality_traits=["determined", "kind"]
        )

        # Add appearance
        appearance = CharacterAppearance(
            chapter_id="ch-1",
            scene_context="action scene",
            emotional_state="determined"
        )
        profile.appearances.append(appearance)

        self.tracker.characters["John"] = profile

        optimized = self.tracker.optimize_for_illustration("John", "ch-1")

        assert "physical_description" in optimized
        assert "emotional_context" in optimized
        assert "personality_hints" in optimized

    def test_get_consistency_report(self):
        """Test getting consistency report."""
        # Add some characters with potential issues
        profile = CharacterProfile(
            name="John",
            physical_description=PhysicalDescription(hair_color="brown")
        )

        # Add inconsistent appearance
        inconsistent_appearance = CharacterAppearance(
            chapter_id="ch-1",
            scene_context="scene1",
            physical_description=PhysicalDescription(hair_color="blonde")
        )
        profile.appearances.append(inconsistent_appearance)

        self.tracker.characters["John"] = profile

        report = self.tracker.get_consistency_report()

        assert "total_characters" in report
        assert "characters_with_issues" in report
        assert "consistency_violations" in report
        assert report["total_characters"] == 1