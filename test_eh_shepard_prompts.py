#!/usr/bin/env python3
"""
Test script to verify E.H. Shepard style prompt engineering works correctly.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.append("src")

from dotenv import load_dotenv
load_dotenv()

from illustrator.prompt_engineering import PromptEngineer
from illustrator.models import (
    EmotionalMoment, EmotionalTone, ImageProvider, Chapter
)
from langchain.chat_models import init_chat_model

async def test_eh_shepard_prompts():
    """Test the E.H. Shepard prompt engineering."""

    print("🎨 Testing E.H. Shepard Style Prompt Engineering")
    print("=" * 50)

    # Initialize the LLM and prompt engineer
    llm = init_chat_model(model="claude-sonnet-4-20250514", model_provider="anthropic")
    engineer = PromptEngineer(llm)

    # Create test emotional moment
    test_moment = EmotionalMoment(
        text_excerpt="Lukas stood at the doorway, feeling the weight of the old Victorian house watching him with ancient curiosity.",
        start_position=0,
        end_position=112,
        context="First day in a mysterious new house with supernatural elements",
        emotional_tones=[EmotionalTone.MYSTERY, EmotionalTone.ANTICIPATION],
        intensity_score=0.8
    )

    # Create test chapter
    test_chapter = Chapter(
        title="A City Wakes",
        content="Sample chapter content about a mysterious Victorian house...",
        number=1,
        word_count=1000
    )

    # Style preferences with E.H. Shepard configuration
    style_preferences = {
        'art_style': 'pencil sketch',
        'style_name': 'E.H. Shepard',
        'base_prompt_modifiers': [
            'hand-drawn pencil sketch',
            'in the style of E.H. Shepard and Tove Jansson',
            'classic children\'s book illustration',
            'delicate line work',
            'cross-hatching',
            'whimsical characters',
            'gentle shading',
            'expressive facial features',
            'soft pencil textures'
        ],
        'color_palette': 'monochrome pencil with subtle shading',
        'artistic_influences': 'E.H. Shepard, Tove Jansson, classic book illustration'
    }

    print(f"📝 Test Scene: {test_moment.text_excerpt}")
    print(f"🎭 Emotional Tones: {[tone.value for tone in test_moment.emotional_tones]}")
    print(f"⚡ Intensity: {test_moment.intensity_score}")
    print()

    # Test with different providers
    providers = [
        (ImageProvider.DALLE, "DALL-E 3"),
        (ImageProvider.IMAGEN4, "Imagen 4"),
        (ImageProvider.FLUX, "Flux")
    ]

    for provider, name in providers:
        print(f"🤖 Testing with {name}")
        print("-" * 30)

        try:
            # Generate the prompt
            illustration_prompt = await engineer.engineer_prompt(
                emotional_moment=test_moment,
                provider=provider,
                style_preferences=style_preferences,
                chapter_context=test_chapter
            )

            print(f"📋 Generated Prompt:")
            print(f"   {illustration_prompt.prompt}")
            print()
            print(f"🎨 Style Modifiers:")
            for modifier in illustration_prompt.style_modifiers:
                print(f"   • {modifier}")
            print()
            print(f"❌ Negative Prompt: {illustration_prompt.negative_prompt}")
            print()
            print(f"⚙️  Technical Parameters:")
            for key, value in illustration_prompt.technical_params.items():
                print(f"   • {key}: {value}")
            print()

        except Exception as e:
            print(f"❌ Error with {name}: {e}")

        print("=" * 50)

    print("✅ Testing complete!")

if __name__ == "__main__":
    asyncio.run(test_eh_shepard_prompts())