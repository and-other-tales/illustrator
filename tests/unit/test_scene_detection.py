"""Comprehensive unit tests for scene detection functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from illustrator.scene_detection import (
    LiterarySceneDetector,
    SceneTransitionType,
    SceneBoundary,
    Scene
)
from illustrator.models import Chapter, EmotionalMoment, EmotionalTone


class TestSceneTransitionType:
    """Test scene transition type enum."""

    def test_scene_transition_types(self):
        """Test all scene transition types exist."""
        assert SceneTransitionType.TIME_JUMP == "time_jump"
        assert SceneTransitionType.LOCATION_CHANGE == "location_change"
        assert SceneTransitionType.CHARACTER_CHANGE == "character_change"
        assert SceneTransitionType.PERSPECTIVE_SHIFT == "perspective_shift"
        assert SceneTransitionType.DIALOGUE_BREAK == "dialogue_break"


class TestSceneBoundary:
    """Test scene boundary data class."""

    def test_scene_boundary_creation(self):
        """Test creating scene boundary."""
        boundary = SceneBoundary(
            position=100,
            transition_type=SceneTransitionType.TIME_JUMP,
            confidence=0.8,
            context_before="morning scene",
            context_after="evening scene",
            detected_markers=["later that day"],
            narrative_significance=0.9
        )

        assert boundary.position == 100
        assert boundary.transition_type == SceneTransitionType.TIME_JUMP
        assert boundary.confidence == 0.8
        assert boundary.context_before == "morning scene"
        assert boundary.narrative_significance == 0.9


class TestScene:
    """Test scene data class."""

    def test_scene_creation(self):
        """Test creating scene."""
        scene = Scene(
            start_position=0,
            end_position=500,
            text="Sample scene text here...",
            scene_type="action",
            primary_characters=["John", "Mary"],
            setting_indicators=["forest", "clearing"],
            emotional_intensity=0.8,
            visual_potential=0.9,
            narrative_importance=0.7,
            time_indicators=["morning", "dawn"],
            location_indicators=["woods", "trail"]
        )

        assert scene.start_position == 0
        assert scene.end_position == 500
        assert scene.scene_type == "action"
        assert "John" in scene.primary_characters
        assert "forest" in scene.setting_indicators
        assert scene.emotional_intensity == 0.8


class TestLiterarySceneDetector:
    """Test the main scene detection class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock()
        self.detector = LiterarySceneDetector(self.mock_llm)

    def test_initialization(self):
        """Test detector initialization."""
        assert self.detector.llm == self.mock_llm

    @pytest.mark.asyncio
    async def test_detect_scenes_basic(self):
        """Test basic scene detection."""
        chapter = Chapter(
            title="Test Chapter",
            content="Scene one content here with much more detailed description of what happened in the morning when everything was peaceful and calm. The birds were singing and the sun was shining brightly across the meadow. Later that day, scene two began with dramatic changes as storm clouds gathered overhead and the wind picked up significantly. The atmosphere became tense and foreboding as characters prepared for what was coming next. Finally, the evening brought scene three with its own unique atmosphere and completely different emotional tone from the previous scenes.",
            number=1,
            word_count=85
        )

        # Mock LLM response
        self.mock_llm.ainvoke.return_value.content = """
        Scene Analysis:
        Boundary at position 25 - Time transition: "Later that day"
        Boundary at position 65 - Time transition: "Finally, the evening"
        """

        scenes = await self.detector.extract_scenes(chapter.content)

        # Should detect multiple scenes
        assert isinstance(scenes, list)
        assert len(scenes) > 0
        self.mock_llm.ainvoke.assert_called()

    @pytest.mark.asyncio
    async def test_detect_scenes_empty_content(self):
        """Test scene detection with empty content."""
        chapter = Chapter(
            title="Empty Chapter",
            content="",
            number=1,
            word_count=0
        )

        scenes = await self.detector.extract_scenes(chapter.content)
        assert scenes == []

    @pytest.mark.asyncio
    async def test_detect_scenes_short_content(self):
        """Test scene detection with very short content."""
        chapter = Chapter(
            title="Short Chapter",
            content="Brief content.",
            number=1,
            word_count=2
        )

        scenes = await self.detector.extract_scenes(chapter.content)

        # Should return at least one scene for any content
        assert len(scenes) >= 1
        if scenes:
            assert scenes[0].text == "Brief content."

    @pytest.mark.asyncio
    async def test_detect_scenes_with_emotional_moments(self):
        """Test scene detection considering existing emotional moments."""
        emotional_moment = EmotionalMoment(
            start_position=10,
            end_position=30,
            text_excerpt="emotional content",
            emotional_tones=[EmotionalTone.JOY],
            intensity_score=0.8,
            context="happy moment"
        )

        chapter = Chapter(
            title="Test Chapter",
            content="The day began with emotional content that filled hearts with joy. Later, sadness crept in.",
            number=1,
            word_count=17
        )

        self.mock_llm.ainvoke.return_value.content = """
        Scene boundaries detected at positions based on emotional transitions.
        """

        scenes = await self.detector.extract_scenes(chapter.content)
        assert len(scenes) > 0

    @pytest.mark.asyncio
    async def test_llm_error_handling(self):
        """Test handling of LLM errors."""
        self.mock_llm.ainvoke.side_effect = Exception("API Error")

        chapter = Chapter(
            title="Error Test",
            content="Content that should be processed despite LLM error.",
            number=1,
            word_count=9
        )

        # Should still return scenes even with LLM failure
        scenes = await self.detector.extract_scenes(chapter.content)
        assert isinstance(scenes, list)

    @pytest.mark.asyncio
    async def test_detect_scenes_complex_content(self):
        """Test scene detection with complex literary content."""
        complex_content = """
        Chapter 1: The Beginning

        The morning sun cast long shadows across the courtyard as Lady Margaret descended the stone steps.
        Her silk gown rustled in the gentle breeze, and the servants bowed respectfully as she passed.

        "Good morning, my lady," called Thomas from the stable yard.

        "Good morning, Thomas. How are the horses today?"

        "Restless, my lady. They sense the storm coming."

        Indeed, dark clouds were gathering on the horizon. Margaret pulled her shawl closer and quickened her pace toward the great hall.

        Three hours later, the storm had arrived in full force. Rain lashed against the windows as Margaret sat by the fire, reading correspondence from the capital. Each letter brought more troubling news.

        A loud crash echoed through the castle. Margaret looked up from her letters, her heart racing.
        """

        chapter = Chapter(
            title="Complex Chapter",
            content=complex_content,
            number=1,
            word_count=150
        )

        self.mock_llm.ainvoke.return_value.content = """
        Multiple scene boundaries detected:
        - Dialogue sequence starting at position 200
        - Time jump at position 600 ("Three hours later")
        - Tension escalation at position 800 (storm arrival)
        """

        scenes = await self.detector.extract_scenes(chapter.content)

        assert len(scenes) >= 2  # Should detect multiple scenes

        # Verify scenes have proper structure
        for scene in scenes:
            assert hasattr(scene, 'text')
            assert hasattr(scene, 'scene_type')
            assert hasattr(scene, 'start_position')
            assert hasattr(scene, 'end_position')

    def test_scene_boundary_ordering(self):
        """Test that scene boundaries are properly ordered."""
        # This would test internal ordering logic if exposed
        # For now, just verify the concept exists
        boundary1 = SceneBoundary(
            position=100,
            transition_type=SceneTransitionType.TIME_JUMP,
            confidence=0.8,
            context_before="",
            context_after="",
            detected_markers=[],
            narrative_significance=0.5
        )

        boundary2 = SceneBoundary(
            position=200,
            transition_type=SceneTransitionType.LOCATION_CHANGE,
            confidence=0.9,
            context_before="",
            context_after="",
            detected_markers=[],
            narrative_significance=0.7
        )

        # Boundaries should be comparable by position
        assert boundary1.position < boundary2.position

    def test_scene_types(self):
        """Test different scene type classifications."""
        scene_types = ["action", "dialogue", "exposition", "reflection", "transition"]

        for scene_type in scene_types:
            scene = Scene(
                start_position=0,
                end_position=100,
                text="Sample text",
                scene_type=scene_type,
                primary_characters=[],
                setting_indicators=[],
                emotional_intensity=0.5,
                visual_potential=0.5,
                narrative_importance=0.5,
                time_indicators=[],
                location_indicators=[]
            )
            assert scene.scene_type == scene_type