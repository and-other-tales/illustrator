#!/usr/bin/env python3
"""Test script to demonstrate the enhanced prompt engineering capabilities."""

import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
import os

from src.illustrator.models import (
    EmotionalMoment, EmotionalTone, ImageProvider, Chapter
)
from src.illustrator.prompt_engineering import PromptEngineer
from langchain_openai import ChatOpenAI

async def test_enhanced_prompt_generation():
    """Test the enhanced prompt generation with sample scenes."""

    # Load environment
    load_dotenv()

    # Initialize LLM and PromptEngineer
    llm = ChatOpenAI(
        model="gpt-4",
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0.7
    )

    prompt_engineer = PromptEngineer(llm)

    # Test scenes based on your examples
    test_scenes = [
        {
            "title": "Coffee Shop Fear Scene",
            "text": "The barista's smile froze as she looked at the customer, her eyes widening with unmistakable fear. She had just whispered something to him, and now she stood behind the counter, trying to maintain professional composure while clearly terrified.",
            "context": "A tense moment in a cozy coffee shop where a barista has just recognized something disturbing about a customer.",
            "emotional_tones": [EmotionalTone.FEAR, EmotionalTone.TENSION],
            "intensity": 0.8
        },
        {
            "title": "Curious Observer Scene",
            "text": "Lukas watched from the doorway, his head tilted slightly as he observed the strange interaction unfolding before him. Something wasn't quite right about what he was seeing.",
            "context": "A character discovering something mysterious or unsettling.",
            "emotional_tones": [EmotionalTone.MYSTERY, EmotionalTone.CURIOSITY],
            "intensity": 0.6
        }
    ]

    # Chapter context
    test_chapter = Chapter(
        number=1,
        title="The Unexpected Encounter",
        content="A chapter exploring the tension between ordinary moments and hidden fears..."
    )

    # E.H. Shepard style configuration
    eh_shepard_style = {
        "style_name": "Advanced E.H. Shepard Style",
        "art_style": "pencil sketch",
        "base_prompt_modifiers": [
            "hand-drawn pencil sketch",
            "in the style of E.H. Shepard",
            "classic children's book illustration",
            "delicate line work with cross-hatching",
            "expressive character faces",
            "gentle pencil shading"
        ],
        "negative_prompt": [
            "photorealistic", "digital painting", "color", "modern art style"
        ],
        "technical_params": {
            "style": "artistic",
            "quality": "high"
        }
    }

    print("🎨 Testing Enhanced Prompt Generation")
    print("=" * 50)

    for i, scene_data in enumerate(test_scenes, 1):
        print(f"\n[Test Scene {i}] {scene_data['title']}")
        print(f"Original text: {scene_data['text'][:80]}...")

        # Create EmotionalMoment
        emotional_moment = EmotionalMoment(
            text_excerpt=scene_data['text'],
            context=scene_data['context'],
            emotional_tones=scene_data['emotional_tones'],
            intensity_score=scene_data['intensity'],
            narrative_significance="High emotional and visual impact scene"
        )

        try:
            # Generate enhanced prompt
            illustration_prompt = await prompt_engineer.engineer_prompt(
                emotional_moment=emotional_moment,
                provider=ImageProvider.DALLE,
                style_preferences=eh_shepard_style,
                chapter_context=test_chapter
            )

            print(f"\n✨ ENHANCED PROMPT:")
            print("-" * 40)
            print(f"{illustration_prompt.prompt}")
            print("-" * 40)

            print(f"\n🎭 Style Modifiers:")
            for modifier in illustration_prompt.style_modifiers:
                print(f"  • {modifier}")

            if illustration_prompt.negative_prompt:
                print(f"\n🚫 Negative Prompt: {illustration_prompt.negative_prompt}")

            print(f"\n⚙️ Technical Parameters:")
            for key, value in illustration_prompt.technical_params.items():
                print(f"  • {key}: {value}")

        except Exception as e:
            print(f"❌ Error generating prompt: {e}")

    print(f"\n🎨 Enhanced Prompt Generation Test Complete!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_prompt_generation())