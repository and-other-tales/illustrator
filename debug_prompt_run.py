import asyncio
from unittest.mock import AsyncMock
from illustrator.prompt_engineering import PromptEngineer, VisualElement, SceneComposition, CompositionType, LightingMood
from illustrator.models import EmotionalMoment, EmotionalTone, ImageProvider, Chapter

async def run():
    llm = AsyncMock()
    engineer = PromptEngineer(llm)

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
        sample_moment, composition, visual_elements, style_translation, ImageProvider.DALLE
    )

    print('GENERATED PROMPT:\n')
    print(repr(prompt))
    print('\nLOWER:\n')
    print(prompt.lower())

asyncio.run(run())
