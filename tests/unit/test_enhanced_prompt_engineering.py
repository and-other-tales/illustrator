"""Tests for the enhanced prompt engineering system."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from illustrator.models import (
    Chapter,
    EmotionalMoment,
    EmotionalTone,
    ImageProvider,
    IllustrationPrompt
)
from illustrator.prompt_engineering import (
    PromptEngineer,
    SceneAnalyzer,
    VisualElement,
    SceneComposition,
    CompositionType,
    LightingMood,
    StyleTranslator
)


class TestEnhancedPromptEngineering:
    """Test enhanced prompt engineering with AI analysis."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM."""
        llm = AsyncMock()
        return llm

    @pytest.fixture
    def sample_emotional_moment(self):
        """Create sample emotional moment."""
        return EmotionalMoment(
            text_excerpt="When he looked again, the shadow behaved normally, though the air where the man had stood felt oddly charged. The melody the man had been humming lingered in the air.",
            start_position=100,
            end_position=200,
            emotional_tones=[EmotionalTone.MYSTERY, EmotionalTone.TENSION],
            intensity_score=0.85,
            context="Victorian London street scene with supernatural elements"
        )

    @pytest.fixture
    def sample_chapter(self):
        """Create sample chapter."""
        return Chapter(
            title="A City Wakes",
            content="Full chapter content here...",
            number=1,
            word_count=2000
        )

    @pytest.fixture
    def scene_analyzer(self, mock_llm):
        """Create scene analyzer."""
        return SceneAnalyzer(mock_llm)

    @pytest.mark.asyncio
    async def test_scene_analyzer_visual_element_extraction(self, scene_analyzer, sample_emotional_moment, sample_chapter):
        """Test visual element extraction from text."""
        # Mock LLM response for visual elements
        mock_response = Mock()
        mock_response.content = '''[
            {
                "element_type": "character",
                "description": "elderly gentleman with silver hair",
                "importance": 0.9,
                "attributes": {
                    "specific_details": "humming a melody, mysterious presence",
                    "emotional_significance": "central figure creating supernatural atmosphere"
                }
            },
            {
                "element_type": "environment",
                "description": "Victorian street with cobblestones",
                "importance": 0.7,
                "attributes": {
                    "specific_details": "morning light, urban setting",
                    "emotional_significance": "contrasts with supernatural elements"
                }
            },
            {
                "element_type": "atmosphere",
                "description": "oddly charged air with lingering melody",
                "importance": 0.8,
                "attributes": {
                    "specific_details": "supernatural tension, musical element",
                    "emotional_significance": "creates mystery and unease"
                }
            }
        ]'''
        scene_analyzer.llm.ainvoke.return_value = mock_response

        visual_elements = await scene_analyzer._extract_visual_elements(
            sample_emotional_moment.text_excerpt,
            sample_emotional_moment.context,
            sample_chapter
        )

        # Verify visual elements were extracted
        assert len(visual_elements) == 3

        # Check character element
        character_element = next(e for e in visual_elements if e.element_type == "character")
        assert "elderly gentleman" in character_element.description
        assert character_element.importance == 0.9

        # Check environment element
        env_element = next(e for e in visual_elements if e.element_type == "environment")
        assert "Victorian street" in env_element.description

        # Check atmosphere element
        atmo_element = next(e for e in visual_elements if e.element_type == "atmosphere")
        assert "charged air" in atmo_element.description

    @pytest.mark.asyncio
    async def test_scene_composition_analysis(self, scene_analyzer, sample_emotional_moment):
        """Test scene composition analysis."""
        # Create test visual elements
        visual_elements = [
            VisualElement(
                element_type="character",
                description="elderly gentleman",
                importance=0.9,
                attributes={"pose": "standing", "expression": "mysterious"}
            ),
            VisualElement(
                element_type="environment",
                description="Victorian street",
                importance=0.7,
                attributes={"lighting": "morning", "atmosphere": "urban"}
            )
        ]

        # Mock LLM response for composition
        mock_response = Mock()
        mock_response.content = '''{
            "composition_type": "medium_shot",
            "focal_point": "elderly gentleman and his shadow",
            "background_elements": ["Victorian buildings", "cobblestone street"],
            "foreground_elements": ["gentleman figure", "rippling shadow"],
            "lighting_mood": "mysterious",
            "atmosphere": "supernatural tension",
            "color_palette_suggestion": "muted tones with golden accents",
            "emotional_weight": 0.85
        }'''
        scene_analyzer.llm.ainvoke.return_value = mock_response

        composition = await scene_analyzer._analyze_composition(
            sample_emotional_moment,
            visual_elements,
            {}
        )

        # Verify composition analysis
        assert composition.composition_type == CompositionType.MEDIUM_SHOT
        assert composition.focal_point == "elderly gentleman and his shadow"
        assert composition.lighting_mood == LightingMood.MYSTERIOUS
        assert composition.emotional_weight == 0.85

    @pytest.mark.asyncio
    async def test_prompt_engineer_integration(self, mock_llm):
        """Test complete prompt engineering workflow."""
        engineer = PromptEngineer(mock_llm)

        # Mock the scene analyzer
        with patch.object(engineer.scene_analyzer, 'analyze_scene') as mock_analyze:
            # Setup mock scene analysis results
            visual_elements = [
                VisualElement(
                    element_type="character",
                    description="elderly gentleman with silver hair",
                    importance=0.9,
                    attributes={"expression": "mysterious", "pose": "standing"}
                )
            ]

            composition = SceneComposition(
                composition_type=CompositionType.MEDIUM_SHOT,
                focal_point="gentleman and shadow",
                background_elements=["Victorian street"],
                foreground_elements=["gentleman"],
                lighting_mood=LightingMood.MYSTERIOUS,
                atmosphere="supernatural",
                color_palette_suggestion="muted with golden accents",
                emotional_weight=0.85
            )

            mock_analyze.return_value = (composition, visual_elements)

            # Mock the comprehensive prompt building
            with patch.object(engineer, '_build_comprehensive_prompt') as mock_build:
                mock_build.return_value = "A mysterious digital painting of an elderly gentleman with silver hair standing on a Victorian cobblestone street, his shadow rippling unnaturally on the ground, atmospheric lighting with supernatural tension, medium shot composition, muted color palette with golden accents"

                sample_moment = EmotionalMoment(
                    text_excerpt="When he looked again, the shadow behaved normally",
                    start_position=0,
                    end_position=50,
                    emotional_tones=[EmotionalTone.MYSTERY],
                    intensity_score=0.8,
                    context="supernatural scene"
                )

                sample_chapter = Chapter(
                    title="Test Chapter",
                    content="test",
                    number=1,
                    word_count=1
                )

                # Test prompt engineering
                result = await engineer.engineer_prompt(
                    emotional_moment=sample_moment,
                    provider=ImageProvider.DALLE,
                    style_preferences={"art_style": "digital painting"},
                    chapter_context=sample_chapter
                )

                # Verify result
                assert isinstance(result, IllustrationPrompt)
                assert result.provider == ImageProvider.DALLE
                assert len(result.prompt) > 100  # Should be a detailed prompt
                assert "elderly gentleman" in result.prompt
                assert "Victorian" in result.prompt

    def test_style_translator_provider_optimization(self):
        """Test style translation for different providers."""
        translator = StyleTranslator()

        base_style = {
            "art_style": "digital painting",
            "color_palette": "warm tones",
            "artistic_influences": "Van Gogh"
        }

        composition = SceneComposition(
            composition_type=CompositionType.DRAMATIC,
            focal_point="character",
            background_elements=[],
            foreground_elements=[],
            lighting_mood=LightingMood.DRAMATIC,
            atmosphere="intense",
            color_palette_suggestion="bold colors",
            emotional_weight=0.9
        )

        # Test DALL-E optimization
        dalle_style = translator.translate_style_config(base_style, ImageProvider.DALLE, composition)
        assert "digital painting" in dalle_style["style_modifiers"]
        assert dalle_style["technical_params"]["size"] == "1024x1024"

        # Test Imagen4 optimization
        imagen_style = translator.translate_style_config(base_style, ImageProvider.IMAGEN4, composition)
        assert isinstance(imagen_style["style_modifiers"], list)
        assert len(imagen_style["negative_prompt"]) > 0

    def test_rich_style_configuration_applies_provider_directives(self):
        translator = StyleTranslator()

        rich_style = translator.rich_style_configs.get("advanced_eh_shepard")
        assert rich_style is not None

        composition = SceneComposition(
            composition_type=CompositionType.MEDIUM_SHOT,
            focal_point="primary subjects",
            background_elements=["storybook forest"],
            foreground_elements=["joyful characters"],
            lighting_mood=LightingMood.NATURAL,
            atmosphere="whimsical and inviting",
            color_palette_suggestion="soft graphite tones",
            emotional_weight=0.55,
            emotional_tones=[EmotionalTone.JOY]
        )

        translation = translator.translate_style_config(
            rich_style,
            ImageProvider.IMAGEN4,
            composition
        )

        # Base style modifiers should include the core Shepard instructions
        combined_modifiers = " ".join(translation["style_modifiers"]).lower()
        assert "pencil sketch" in combined_modifiers
        assert "shepard" in combined_modifiers

        # Provider-specific technical adjustments should merge into technical params
        assert translation["technical_params"]["guidance_scale"] == 12
        assert translation["technical_params"]["aspect_ratio"] == "1:1"

        # Emotional adaptations should surface atmosphere guidance
        assert any("pencil" in note or "composition" in note for note in translation["atmosphere_guidance"])

        # Provider optimizations should remain accessible for prompt enhancement
        assert translation["provider_optimizations"]["style_emphasis"].startswith("E.H. Shepard")

    @pytest.mark.asyncio
    async def test_context_tracking_characters(self, mock_llm):
        """Test character consistency tracking across scenes."""
        engineer = PromptEngineer(mock_llm)

        # Mock LLM response for character extraction
        mock_response = Mock()
        mock_response.content = '''{
            "characters": [
                {
                    "name": "Lukas",
                    "description": "young man with marketing background, sitting on steps",
                    "action": "watching morning commuters"
                },
                {
                    "name": "elderly gentleman",
                    "description": "silver-haired man with mysterious aura",
                    "action": "walking slowly, humming"
                }
            ]
        }'''
        mock_llm.ainvoke.return_value = mock_response

        sample_moment = EmotionalMoment(
            text_excerpt="Lukas watched the elderly gentleman walk past, humming softly",
            start_position=0,
            end_position=60,
            emotional_tones=[EmotionalTone.PEACE],
            intensity_score=0.6,
            context="morning scene"
        )

        sample_chapter = Chapter(
            title="Test",
            content="test",
            number=1,
            word_count=1
        )

        await engineer._update_context_tracking(sample_moment, sample_chapter, [])

        # Verify character profiles were updated (names may be lowercase)
        assert "lukas" in engineer.character_profiles or "Lukas" in engineer.character_profiles
        assert "elderly gentleman" in engineer.character_profiles

        lukas_profile = engineer.character_profiles.get("lukas") or engineer.character_profiles.get("Lukas")
        assert "marketing background" in lukas_profile.physical_description
        assert "watching morning commuters" in lukas_profile.current_action

    @pytest.mark.asyncio
    async def test_prompt_building_comprehensive(self, mock_llm):
        """Test comprehensive prompt building with all components."""
        engineer = PromptEngineer(mock_llm)

        sample_moment = EmotionalMoment(
            text_excerpt="The shadow rippled like water",
            start_position=0,
            end_position=25,
            emotional_tones=[EmotionalTone.MYSTERY],
            intensity_score=0.9,
            context="supernatural moment"
        )

        visual_elements = [
            VisualElement(
                element_type="atmosphere",
                description="rippling shadow effect",
                importance=0.95,
                attributes={"visual_effect": "water-like movement", "supernatural": "true"}
            )
        ]

        composition = SceneComposition(
            composition_type=CompositionType.CLOSE_UP,
            focal_point="rippling shadow",
            background_elements=["cobblestone ground"],
            foreground_elements=["shadow"],
            lighting_mood=LightingMood.MYSTERIOUS,
            atmosphere="supernatural",
            color_palette_suggestion="dark tones with silver highlights",
            emotional_weight=0.9
        )

        style_translation = {
            "style_modifiers": ["photorealistic", "cinematic"],
            "technical_params": {"quality": "high"},
            "negative_prompt": ["blurry", "low quality"]
        }

        prompt = await engineer._build_comprehensive_prompt(
            sample_moment,
            composition,
            visual_elements,
            style_translation,
            ImageProvider.DALLE
        )

        # Verify comprehensive prompt components
        assert len(prompt) > 50  # Should be detailed
        assert "rippling shadow" in prompt.lower()
        assert "mysterious" in prompt.lower() or "supernatural" in prompt.lower()
        assert "close" in prompt.lower() or "close-up" in prompt.lower()


if __name__ == "__main__":
    pytest.main([__file__])
