"""Comprehensive unit tests for visual composition functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import List, Dict, Any

from illustrator.visual_composition import (
    AdvancedVisualComposer,
    CompositionRule,
    LightingSetup,
    ColorHarmony,
    ShotType,
    CameraAngle,
    VisualFocus,
    AdvancedComposition,
    CompositionElement,
    VisualLayer
)
from illustrator.models import Chapter, EmotionalMoment, EmotionalTone


class TestCompositionRule:
    """Test composition rule data class."""

    def test_composition_rule_creation(self):
        """Test creating composition rule."""
        rule = CompositionRule(
            rule_type="rule_of_thirds",
            description="Place key elements along third lines",
            application_weight=0.8,
            emotional_context=["dramatic", "balanced"]
        )

        assert rule.rule_type == "rule_of_thirds"
        assert rule.description == "Place key elements along third lines"
        assert rule.application_weight == 0.8
        assert "dramatic" in rule.emotional_context

    def test_composition_rule_defaults(self):
        """Test composition rule with default values."""
        rule = CompositionRule(rule_type="golden_ratio")

        assert rule.rule_type == "golden_ratio"
        assert rule.emotional_context == []
        assert rule.application_weight == 1.0


class TestLightingSetup:
    """Test lighting setup data class."""

    def test_lighting_setup_creation(self):
        """Test creating lighting setup."""
        setup = LightingSetup(
            lighting_type=LightingType.DRAMATIC,
            key_light_direction="top-left",
            fill_light_intensity=0.3,
            mood_description="mysterious and foreboding",
            color_temperature=3200
        )

        assert setup.lighting_type == LightingType.DRAMATIC
        assert setup.key_light_direction == "top-left"
        assert setup.fill_light_intensity == 0.3
        assert setup.mood_description == "mysterious and foreboding"
        assert setup.color_temperature == 3200

    def test_lighting_setup_defaults(self):
        """Test lighting setup with default values."""
        setup = LightingSetup(lighting_type=LightingType.NATURAL)

        assert setup.lighting_type == LightingType.NATURAL
        assert setup.fill_light_intensity == 0.5
        assert setup.color_temperature == 5600


class TestColorHarmony:
    """Test color harmony data class."""

    def test_color_harmony_creation(self):
        """Test creating color harmony."""
        harmony = ColorHarmony(
            scheme=ColorScheme.COMPLEMENTARY,
            primary_colors=["#FF5733", "#33FF57"],
            accent_colors=["#3357FF"],
            mood_association="energetic and vibrant",
            emotional_impact=0.9
        )

        assert harmony.scheme == ColorScheme.COMPLEMENTARY
        assert len(harmony.primary_colors) == 2
        assert len(harmony.accent_colors) == 1
        assert harmony.emotional_impact == 0.9

    def test_color_harmony_defaults(self):
        """Test color harmony with default values."""
        harmony = ColorHarmony(scheme=ColorScheme.MONOCHROMATIC)

        assert harmony.scheme == ColorScheme.MONOCHROMATIC
        assert harmony.primary_colors == []
        assert harmony.accent_colors == []
        assert harmony.emotional_impact == 0.5


class TestVisualElement:
    """Test visual element data class."""

    def test_visual_element_creation(self):
        """Test creating visual element."""
        element = VisualElement(
            element_type="character",
            position=(0.3, 0.6),
            size_ratio=0.4,
            importance=0.9,
            description="protagonist in dramatic pose"
        )

        assert element.element_type == "character"
        assert element.position == (0.3, 0.6)
        assert element.size_ratio == 0.4
        assert element.importance == 0.9
        assert element.description == "protagonist in dramatic pose"


class TestCompositionGuide:
    """Test composition guide data class."""

    def test_composition_guide_creation(self):
        """Test creating composition guide."""
        guide = CompositionGuide(
            shot_type=ShotType.MEDIUM_SHOT,
            camera_angle=CameraAngle.EYE_LEVEL,
            focal_point=(0.33, 0.33),
            composition_notes="Use rule of thirds for character placement",
            depth_layers=["foreground: character", "background: landscape"]
        )

        assert guide.shot_type == ShotType.MEDIUM_SHOT
        assert guide.camera_angle == CameraAngle.EYE_LEVEL
        assert guide.focal_point == (0.33, 0.33)
        assert len(guide.depth_layers) == 2

    def test_composition_guide_defaults(self):
        """Test composition guide with default values."""
        guide = CompositionGuide()

        assert guide.shot_type == ShotType.MEDIUM_SHOT
        assert guide.camera_angle == CameraAngle.EYE_LEVEL
        assert guide.depth_layers == []


class TestCompositionAnalysis:
    """Test composition analysis data class."""

    def test_composition_analysis_creation(self):
        """Test creating composition analysis."""
        analysis = CompositionAnalysis(
            visual_elements=[
                VisualElement("character", (0.3, 0.6), 0.4, 0.9),
                VisualElement("background", (0.0, 0.0), 1.0, 0.3)
            ],
            composition_guide=CompositionGuide(),
            lighting_setup=LightingSetup(LightingType.DRAMATIC),
            color_harmony=ColorHarmony(ColorScheme.COMPLEMENTARY),
            overall_mood="dramatic confrontation",
            composition_strength=0.85
        )

        assert len(analysis.visual_elements) == 2
        assert isinstance(analysis.composition_guide, CompositionGuide)
        assert isinstance(analysis.lighting_setup, LightingSetup)
        assert analysis.composition_strength == 0.85


class TestAdvancedVisualComposer:
    """Test the main visual composer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_llm = AsyncMock()
        self.composer = AdvancedVisualComposer(self.mock_llm)

    def test_initialization(self):
        """Test composer initialization."""
        assert self.composer.llm == self.mock_llm
        assert len(self.composer.composition_rules) > 0
        assert len(self.composer.lighting_presets) > 0
        assert len(self.composer.color_harmonies) > 0

    def test_select_composition_rules(self):
        """Test composition rule selection."""
        emotional_tone = EmotionalTone.TENSION
        context = {"scene_type": "conflict", "character_count": 2}

        rules = self.composer._select_composition_rules(emotional_tone, context)

        assert len(rules) > 0
        assert all(isinstance(rule, CompositionRule) for rule in rules)

    def test_select_composition_rules_peaceful(self):
        """Test composition rule selection for peaceful scenes."""
        emotional_tone = EmotionalTone.SERENITY
        context = {"scene_type": "contemplative", "character_count": 1}

        rules = self.composer._select_composition_rules(emotional_tone, context)

        # Should include rules appropriate for calm scenes
        rule_types = [rule.rule_type for rule in rules]
        assert any("symmetry" in rule_type or "balance" in rule_type for rule_type in rule_types)

    def test_determine_shot_type(self):
        """Test shot type determination."""
        # Test intimate scene
        intimate_context = {"scene_type": "dialogue", "emotional_intensity": 0.8, "character_count": 2}
        shot_type = self.composer._determine_shot_type(intimate_context)
        assert shot_type in [ShotType.CLOSE_UP, ShotType.MEDIUM_SHOT]

        # Test action scene
        action_context = {"scene_type": "action", "emotional_intensity": 0.9, "character_count": 5}
        shot_type = self.composer._determine_shot_type(action_context)
        assert shot_type in [ShotType.WIDE_SHOT, ShotType.MEDIUM_SHOT]

    def test_determine_camera_angle(self):
        """Test camera angle determination."""
        # Test power dynamic
        power_context = {"emotional_tone": EmotionalTone.TRIUMPH, "character_dynamics": "dominant"}
        angle = self.composer._determine_camera_angle(power_context)
        assert angle in [CameraAngle.LOW_ANGLE, CameraAngle.EYE_LEVEL]

        # Test vulnerable scene
        vulnerable_context = {"emotional_tone": EmotionalTone.FEAR, "character_dynamics": "vulnerable"}
        angle = self.composer._determine_camera_angle(vulnerable_context)
        assert angle in [CameraAngle.HIGH_ANGLE, CameraAngle.EYE_LEVEL]

    def test_select_lighting_setup(self):
        """Test lighting setup selection."""
        # Test dramatic scene
        dramatic_context = {"emotional_tone": EmotionalTone.TENSION, "time_of_day": "night"}
        lighting = self.composer._select_lighting_setup(dramatic_context)
        assert lighting.lighting_type in [LightingType.DRAMATIC, LightingType.LOW_KEY]

        # Test peaceful scene
        peaceful_context = {"emotional_tone": EmotionalTone.SERENITY, "time_of_day": "morning"}
        lighting = self.composer._select_lighting_setup(peaceful_context)
        assert lighting.lighting_type in [LightingType.NATURAL, LightingType.SOFT]

    def test_select_color_harmony(self):
        """Test color harmony selection."""
        # Test warm emotional scene
        warm_context = {"emotional_tone": EmotionalTone.JOY, "mood": "uplifting"}
        harmony = self.composer._select_color_harmony(warm_context)
        assert harmony.scheme in [ColorScheme.WARM, ColorScheme.ANALOGOUS]

        # Test cool dramatic scene
        cool_context = {"emotional_tone": EmotionalTone.SADNESS, "mood": "melancholy"}
        harmony = self.composer._select_color_harmony(cool_context)
        assert harmony.scheme in [ColorScheme.COOL, ColorScheme.MONOCHROMATIC]

    def test_calculate_focal_point_rule_of_thirds(self):
        """Test focal point calculation using rule of thirds."""
        visual_elements = [
            VisualElement("character", (0.5, 0.5), 0.3, 1.0, "main character"),
            VisualElement("object", (0.7, 0.3), 0.1, 0.5, "important object")
        ]

        focal_point = self.composer._calculate_focal_point(visual_elements, ["rule_of_thirds"])

        # Should be near rule of thirds intersection
        assert 0.25 <= focal_point[0] <= 0.75
        assert 0.25 <= focal_point[1] <= 0.75

    def test_calculate_focal_point_golden_ratio(self):
        """Test focal point calculation using golden ratio."""
        visual_elements = [
            VisualElement("character", (0.5, 0.5), 0.4, 1.0, "protagonist")
        ]

        focal_point = self.composer._calculate_focal_point(visual_elements, ["golden_ratio"])

        # Should be positioned according to golden ratio
        assert isinstance(focal_point, tuple)
        assert len(focal_point) == 2

    def test_extract_visual_elements(self):
        """Test visual element extraction from scene text."""
        scene_text = """
        John stood in the center of the room, his sword gleaming in the candlelight.
        Mary watched from the doorway, her face filled with concern.
        Ancient books lined the walls, and a mysterious artifact glowed on the table.
        """

        elements = self.composer._extract_visual_elements(scene_text)

        assert len(elements) > 0
        # Should identify characters and objects
        element_types = [elem.element_type for elem in elements]
        assert "character" in element_types

    def test_analyze_scene_context(self):
        """Test scene context analysis."""
        scene_text = "The battle raged fiercely as warriors clashed under the stormy sky."
        emotional_tone = EmotionalTone.TENSION

        context = self.composer._analyze_scene_context(scene_text, emotional_tone)

        assert "scene_type" in context
        assert "emotional_intensity" in context
        assert "character_count" in context
        assert context["scene_type"] in ["action", "conflict", "battle"]

    def test_generate_depth_layers(self):
        """Test depth layer generation."""
        visual_elements = [
            VisualElement("character", (0.5, 0.6), 0.3, 1.0, "hero in center"),
            VisualElement("background", (0.0, 0.0), 1.0, 0.2, "mountain landscape"),
            VisualElement("object", (0.7, 0.4), 0.1, 0.6, "glowing sword")
        ]

        layers = self.composer._generate_depth_layers(visual_elements)

        assert len(layers) > 0
        assert any("foreground" in layer.lower() for layer in layers)
        assert any("background" in layer.lower() for layer in layers)

    @pytest.mark.asyncio
    async def test_enhance_with_llm_success(self):
        """Test LLM enhancement of composition."""
        self.mock_llm.ainvoke.return_value.content = """
        Enhanced Composition Analysis:
        - Focal point should emphasize character's emotional state
        - Lighting creates dramatic contrast highlighting internal conflict
        - Color palette reinforces the melancholy mood
        - Camera angle subtly conveys vulnerability

        Specific Recommendations:
        - Position character slightly off-center using rule of thirds
        - Use rim lighting to separate character from background
        - Employ cool blue tones with warm accent on character's face
        """

        analysis = CompositionAnalysis(
            visual_elements=[VisualElement("character", (0.5, 0.5), 0.3, 1.0)],
            composition_guide=CompositionGuide(),
            lighting_setup=LightingSetup(LightingType.DRAMATIC),
            color_harmony=ColorHarmony(ColorScheme.COMPLEMENTARY),
            overall_mood="contemplative"
        )

        enhanced = await self.composer._enhance_with_llm(analysis, "Sample scene text")

        assert "llm_enhancement" in enhanced
        assert "specific_recommendations" in enhanced
        self.mock_llm.ainvoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_enhance_with_llm_failure(self):
        """Test LLM enhancement failure handling."""
        self.mock_llm.ainvoke.side_effect = Exception("API Error")

        analysis = CompositionAnalysis(
            visual_elements=[],
            composition_guide=CompositionGuide(),
            lighting_setup=LightingSetup(LightingType.NATURAL),
            color_harmony=ColorHarmony(ColorScheme.NATURAL)
        )

        enhanced = await self.composer._enhance_with_llm(analysis, "Sample scene")

        assert "llm_enhancement" in enhanced
        assert enhanced["llm_enhancement"] == "LLM enhancement unavailable"

    @pytest.mark.asyncio
    async def test_create_advanced_composition(self):
        """Test complete advanced composition creation."""
        scene_text = "Sarah stood at the cliff's edge, watching the sunset paint the sky in brilliant colors."
        emotional_tone = EmotionalTone.SERENITY
        context = {"time_of_day": "sunset", "location": "cliff"}

        self.mock_llm.ainvoke.return_value.content = """
        Composition Enhancement:
        This serene sunset scene should emphasize tranquility and natural beauty.
        The character's contemplative pose against the vast landscape creates
        a perfect moment for visual storytelling.
        """

        composition = await self.composer.create_advanced_composition(
            scene_text, emotional_tone, context
        )

        assert "composition_analysis" in composition
        assert "visual_prompt_enhancement" in composition
        assert "technical_specifications" in composition

        analysis = composition["composition_analysis"]
        assert isinstance(analysis, CompositionAnalysis)
        assert len(analysis.visual_elements) > 0
        assert isinstance(analysis.lighting_setup, LightingSetup)
        assert isinstance(analysis.color_harmony, ColorHarmony)

    @pytest.mark.asyncio
    async def test_optimize_for_emotional_impact(self):
        """Test optimization for emotional impact."""
        base_composition = CompositionAnalysis(
            visual_elements=[VisualElement("character", (0.5, 0.5), 0.3, 1.0)],
            composition_guide=CompositionGuide(),
            lighting_setup=LightingSetup(LightingType.NATURAL),
            color_harmony=ColorHarmony(ColorScheme.NATURAL),
            overall_mood="neutral"
        )

        # Optimize for tension
        optimized = self.composer._optimize_for_emotional_impact(
            base_composition, EmotionalTone.TENSION
        )

        assert optimized.lighting_setup.lighting_type in [LightingType.DRAMATIC, LightingType.LOW_KEY]
        assert optimized.composition_guide.camera_angle != CameraAngle.EYE_LEVEL
        assert optimized.composition_strength > base_composition.composition_strength

    def test_create_technical_specifications(self):
        """Test technical specification generation."""
        composition = CompositionAnalysis(
            visual_elements=[VisualElement("character", (0.33, 0.66), 0.4, 1.0)],
            composition_guide=CompositionGuide(
                shot_type=ShotType.CLOSE_UP,
                camera_angle=CameraAngle.LOW_ANGLE,
                focal_point=(0.33, 0.66)
            ),
            lighting_setup=LightingSetup(
                lighting_type=LightingType.DRAMATIC,
                key_light_direction="top-right",
                color_temperature=3200
            ),
            color_harmony=ColorHarmony(
                scheme=ColorScheme.COMPLEMENTARY,
                primary_colors=["#FF5733", "#33FF57"]
            )
        )

        specs = self.composer._create_technical_specifications(composition)

        assert "shot_composition" in specs
        assert "lighting_details" in specs
        assert "color_specifications" in specs
        assert "camera_settings" in specs

        assert specs["shot_composition"]["type"] == "close-up"
        assert specs["lighting_details"]["primary_light_direction"] == "top-right"
        assert len(specs["color_specifications"]["primary_palette"]) == 2

    def test_get_composition_strength_score(self):
        """Test composition strength scoring."""
        strong_composition = CompositionAnalysis(
            visual_elements=[
                VisualElement("character", (0.33, 0.66), 0.4, 1.0),  # Well positioned
                VisualElement("background", (0.0, 0.0), 1.0, 0.3)
            ],
            composition_guide=CompositionGuide(focal_point=(0.33, 0.66)),
            lighting_setup=LightingSetup(LightingType.DRAMATIC),
            color_harmony=ColorHarmony(ColorScheme.COMPLEMENTARY),
            composition_strength=0.9
        )

        score = self.composer._get_composition_strength_score(strong_composition)

        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should be relatively high

    def test_generate_prompt_enhancements(self):
        """Test visual prompt enhancement generation."""
        composition = CompositionAnalysis(
            visual_elements=[VisualElement("character", (0.33, 0.66), 0.4, 1.0)],
            composition_guide=CompositionGuide(shot_type=ShotType.CLOSE_UP),
            lighting_setup=LightingSetup(LightingType.DRAMATIC),
            color_harmony=ColorHarmony(scheme=ColorScheme.WARM),
            overall_mood="heroic determination"
        )

        enhancements = self.composer._generate_prompt_enhancements(composition)

        assert "composition_terms" in enhancements
        assert "lighting_terms" in enhancements
        assert "color_terms" in enhancements
        assert "mood_descriptors" in enhancements

        assert len(enhancements["composition_terms"]) > 0
        assert "dramatic lighting" in " ".join(enhancements["lighting_terms"])
        assert any("warm" in term for term in enhancements["color_terms"])