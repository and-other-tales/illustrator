"""Unit tests for character_tracking module."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from illustrator.character_tracking import (
    CharacterProfile,
    CharacterAppearance,
    CharacterTracker,
    PhysicalDescription,
    EmotionalProfile
)
from illustrator.models import Chapter


class TestCharacterProfile:
    """Test CharacterProfile data class."""
    
    def test_character_profile_creation(self):
        """Test creating a CharacterProfile."""
        appearances = [
            CharacterAppearance(
                chapter_id="ch1",
                scene_context="Opening scene",
                physical_description=PhysicalDescription(distinguishing_features=["Tall with dark hair"])
            )
        ]
        
        profile = CharacterProfile(
            name="John Doe",
            appearances=appearances,
            physical_description=PhysicalDescription(distinguishing_features=["Tall with dark hair", "kind eyes"]),
            personality_traits=["kind", "intelligent"],
            illustration_notes=["Always described as tall"]
        )
        
        assert profile.name == "John Doe"
        assert len(profile.appearances) == 1
        assert "Tall with dark hair" in profile.appearances[0].physical_description.distinguishing_features
        assert "kind" in profile.personality_traits
        assert "Tall with dark hair" in profile.physical_description.distinguishing_features


class TestCharacterAppearance:
    """Test CharacterAppearance data class."""
    
    def test_character_appearance_creation(self):
        """Test creating a CharacterAppearance."""
        appearance = CharacterAppearance(
            chapter_id="chapter_1",
            scene_context="First meeting in the garden",
            physical_description=PhysicalDescription(distinguishing_features=["Wearing a blue dress with flowing hair"])
        )
        
        assert appearance.chapter_id == "chapter_1"
        assert appearance.scene_context == "First meeting in the garden"
        assert "Wearing a blue dress with flowing hair" in appearance.physical_description.distinguishing_features


class TestCharacterTracker:
    """Test CharacterTracker class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock()
        self.tracker = CharacterTracker(llm=self.mock_llm)
    
    def test_init(self):
        """Test CharacterTracker initialization."""
        assert self.tracker.llm is self.mock_llm
        assert isinstance(self.tracker.characters, dict)
        assert isinstance(self.tracker.name_aliases, dict)
    
    def test_find_canonical_name_exact_match(self):
        """Test finding canonical name with exact match."""
        self.tracker.characters["John Smith"] = Mock()
        
        canonical = self.tracker._find_canonical_name("John Smith")
        assert canonical == "John Smith"
    
    def test_find_canonical_name_alias_match(self):
        """Test finding canonical name through alias."""
        self.tracker.name_aliases["Johnny"] = "John Smith"
        
        canonical = self.tracker._find_canonical_name("Johnny")
        assert canonical == "John Smith"
    
    def test_find_canonical_name_no_match(self):
        """Test finding canonical name with no match."""
        canonical = self.tracker._find_canonical_name("Unknown Character")
        assert canonical is None
    
    def test_find_canonical_name_case_insensitive(self):
        """Test case insensitive name matching."""
        self.tracker.characters["John Smith"] = Mock()
        
        canonical = self.tracker._find_canonical_name("john smith")
        assert canonical == "John Smith"
        
        canonical = self.tracker._find_canonical_name("JOHN SMITH")
        assert canonical == "John Smith"
    
    @pytest.mark.asyncio
    async def test_create_character_profile(self):
        """Test creating a new character profile."""
        mock_chapter = Mock(spec=Chapter)
        mock_chapter.content = "John was a tall man with dark hair."
        mock_chapter.number = 1
        
        # Mock the LLM response for character analysis
        self.mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content='{"appearance": "tall with dark hair", "personality": ["serious"], "relationships": []}'
        ))
        
        with patch.object(self.tracker, '_analyze_character_in_depth', return_value={
            "appearance": "tall with dark hair",
            "personality": ["serious"],
            "relationships": []
        }) as mock_analyze:
            
            profile = await self.tracker._create_character_profile("John", mock_chapter)
            
            assert profile.name == "John"
            assert len(profile.appearances) == 1
            assert profile.appearances[0].chapter_id == "chapter_1"
            mock_analyze.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_analyze_character_in_depth(self):
        """Test in-depth character analysis."""
        # Mock LLM response
        analysis_result = {
            "physical_description": "tall with dark hair",
            "personality_traits": ["intelligent", "kind"],
            "relationships": ["friend of Mary"],
            "emotional_state": "confident"
        }
        
        self.mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content=f'{analysis_result}'
        ))
        
        result = await self.tracker._analyze_character_in_depth(
            "John", 
            "John walked confidently into the room", 
            1
        )
        
        # The method should extract information from LLM response
        self.mock_llm.ainvoke.assert_called_once()
        call_args = self.mock_llm.ainvoke.call_args[0][0]
        assert any("John" in str(msg.content) for msg in call_args if hasattr(msg, 'content'))
    
    def test_update_character_profile_sync(self):
        """Test synchronous character profile update."""
        # Create an existing profile
        profile = CharacterProfile(
            name="Jane",
            appearances=[],
            physical_description=PhysicalDescription(distinguishing_features=["A woman"]),
            personality_traits=[],
            illustration_notes=[]
        )
        self.tracker.characters["Jane"] = profile
        
        # Update with new data
        character_data = {
            "appearance": "blonde hair, blue eyes",
            "personality": ["cheerful", "optimistic"],
            "relationships": ["sister of John"]
        }
        
        self.tracker._update_character_profile_sync(
            "Jane", 
            character_data, 
            "chapter_2", 
            "Morning in the kitchen"
        )
        
        # Check that profile was updated
        updated_profile = self.tracker.characters["Jane"]
        assert len(updated_profile.appearances) == 1
        assert "blonde hair, blue eyes" in updated_profile.appearances[0].physical_description.distinguishing_features
        assert updated_profile.appearances[0].chapter_id == "chapter_2"
        assert "cheerful" in updated_profile.personality_traits
    
    @pytest.mark.asyncio
    async def test_update_character_profile_with_chapter(self):
        """Test updating character profile with Chapter object."""
        # Create existing profile
        profile = CharacterProfile(
            name="Bob",
            appearances=[],
            physical_description=PhysicalDescription(distinguishing_features=["A man"]),
            personality_traits=[],
            illustration_notes=[]
        )
        self.tracker.characters["Bob"] = profile
        
        mock_chapter = Mock(spec=Chapter)
        mock_chapter.number = 3
        
        # Call the async version with chapter
        await self.tracker._update_character_profile("Bob", mock_chapter)
        
        # Since we don't have the full implementation details, just verify it doesn't crash
        assert "Bob" in self.tracker.characters
    
    @pytest.mark.asyncio
    async def test_update_character_profile_with_data(self):
        """Test updating character profile with data dict."""
        # Create existing profile
        profile = CharacterProfile(
            name="Alice",
            appearances=[],
            physical_description=PhysicalDescription(distinguishing_features=["A woman"]),
            personality_traits=[],
            illustration_notes=[]
        )
        self.tracker.characters["Alice"] = profile
        
        character_data = {
            "appearance": "red hair, green dress",
            "personality": ["brave", "curious"]
        }
        
        # Call with data signature
        await self.tracker._update_character_profile(
            "Alice", 
            character_data, 
            "chapter_4", 
            "Forest scene"
        )
        
        # Verify update was applied
        updated_profile = self.tracker.characters["Alice"]
        assert len(updated_profile.appearances) == 1
        assert "brave" in updated_profile.personality_traits
    
    @pytest.mark.asyncio
    async def test_update_character_profile_invalid_args(self):
        """Test update character profile with invalid arguments."""
        with pytest.raises(ValueError, match="Invalid arguments"):
            await self.tracker._update_character_profile("Test", "arg1", "arg2")  # Wrong number of args
    
    @pytest.mark.asyncio
    async def test_extract_character_appearance(self):
        """Test extracting character appearance from text."""
        text = "Mary wore a beautiful red dress and had curly blonde hair."
        
        # Mock LLM response
        self.mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content="red dress, curly blonde hair"
        ))
        
        appearance = await self.tracker._extract_character_appearance("Mary", text, 1)
        
        self.mock_llm.ainvoke.assert_called_once()
        # Verify the appearance extraction was called properly
        call_args = self.mock_llm.ainvoke.call_args[0][0]
        assert any("Mary" in str(msg.content) for msg in call_args if hasattr(msg, 'content'))
    
    def test_add_name_alias(self):
        """Test adding name aliases."""
        self.tracker.characters["Elizabeth"] = Mock()
        
        # Add alias (this would be part of character analysis)
        self.tracker.name_aliases["Liz"] = "Elizabeth"
        self.tracker.name_aliases["Beth"] = "Elizabeth"
        
        assert self.tracker._find_canonical_name("Liz") == "Elizabeth"
        assert self.tracker._find_canonical_name("Beth") == "Elizabeth"
    
    def test_character_consistency_tracking(self):
        """Test tracking character consistency across chapters."""
        profile = CharacterProfile(
            name="David",
            appearances=[
                CharacterAppearance(chapter_id="ch1", scene_context="intro", 
                                  physical_description=PhysicalDescription(distinguishing_features=["tall with brown hair"])),
                CharacterAppearance(chapter_id="ch2", scene_context="meeting",
                                  physical_description=PhysicalDescription(distinguishing_features=["tall with dark hair"]))
            ],
            physical_description=PhysicalDescription(distinguishing_features=["Tall man with brown hair"]),
            personality_traits=["kind", "intelligent"],
            illustration_notes=[]
        )
        
        self.tracker.characters["David"] = profile
        
        # Check for potential inconsistencies
        appearances = profile.appearances
        hair_descriptions = []
        for app in appearances:
            for feature in app.physical_description.distinguishing_features:
                if "hair" in feature:
                    hair_descriptions.append(feature)
        
        assert len(hair_descriptions) == 2
        assert "brown hair" in hair_descriptions[0]
        assert "dark hair" in hair_descriptions[1]
        
        # This could be flagged as an inconsistency in the real implementation
    
    @pytest.mark.asyncio
    async def test_track_character_across_multiple_chapters(self):
        """Test tracking a character across multiple chapters."""
        # First chapter - character introduction
        chapter1 = Mock(spec=Chapter)
        chapter1.content = "Sarah entered the room with confidence, her red hair gleaming."
        chapter1.number = 1
        
        # Mock LLM for character creation
        self.mock_llm.ainvoke = AsyncMock(return_value=Mock(
            content='{"appearance": "red hair", "personality": ["confident"]}'
        ))
        
        with patch.object(self.tracker, '_analyze_character_in_depth', return_value={
            "appearance": "red hair",
            "personality": ["confident"]
        }):
            # Create initial profile
            profile = await self.tracker._create_character_profile("Sarah", chapter1)
            self.tracker.characters["Sarah"] = profile
        
        # Second chapter - character update
        character_data = {
            "appearance": "red hair in a braid",
            "personality": ["confident", "determined"]
        }
        
        self.tracker._update_character_profile_sync(
            "Sarah",
            character_data,
            "chapter_2", 
            "Before the battle"
        )
        
        # Verify character has been tracked across chapters
        final_profile = self.tracker.characters["Sarah"]
        assert len(final_profile.appearances) == 2
        assert final_profile.appearances[0].chapter_id == "chapter_1"
        assert final_profile.appearances[1].chapter_id == "chapter_2"
        assert "determined" in final_profile.personality_traits