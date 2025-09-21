#!/usr/bin/env python3
"""Generate pencil sketch illustrations for classic novels using DALL-E."""

import asyncio
import base64
import os
from pathlib import Path
from typing import Dict, Any

from dotenv import load_dotenv

from src.illustrator.models import EmotionalMoment, EmotionalTone, ImageProvider
from src.illustrator.providers import ProviderFactory

# Classic novel scenes for illustration
CLASSIC_SCENES = [
    {
        "title": "Alice_meets_White_Rabbit",
        "novel": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "prompt": "Alice sitting by riverbank watching a White Rabbit with pink eyes and waistcoat run past, hand-drawn pencil sketch in the style of E.H. Shepard and Tove Jansson, classic children's book illustration, delicate line work, cross-hatching, whimsical characters, traditional illustration technique"
    },
    {
        "title": "Mole_and_Rat_by_river",
        "novel": "The Wind in the Willows",
        "author": "Kenneth Grahame",
        "prompt": "Mole and Water Rat sitting on grassy riverbank watching boats, hand-drawn pencil sketch in the style of E.H. Shepard and Tove Jansson, classic children's book illustration, delicate line work, cross-hatching, whimsical animal characters, peaceful river scene"
    },
    {
        "title": "Mary_enters_secret_garden",
        "novel": "The Secret Garden",
        "author": "Frances Hodgson Burnett",
        "prompt": "Young girl Mary pushing open wooden door into overgrown secret garden with climbing roses, hand-drawn pencil sketch in the style of E.H. Shepard and Tove Jansson, classic children's book illustration, delicate line work, cross-hatching, mysterious garden atmosphere"
    },
    {
        "title": "Long_John_Silver_arrival",
        "novel": "Treasure Island",
        "author": "Robert Louis Stevenson",
        "prompt": "Tall weathered pirate with wooden leg and sea chest arriving at inn, hand-drawn pencil sketch in the style of E.H. Shepard and Tove Jansson, classic adventure book illustration, delicate line work, cross-hatching, dramatic character study"
    },
    {
        "title": "Wolf_family_in_cave",
        "novel": "The Jungle Book",
        "author": "Rudyard Kipling",
        "prompt": "Wolf family with cubs in moonlit cave in Indian jungle, hand-drawn pencil sketch in the style of E.H. Shepard and Tove Jansson, classic children's book illustration, delicate line work, cross-hatching, atmospheric jungle scene"
    },
    {
        "title": "Pooh_and_Piglet_walking",
        "novel": "Winnie-the-Pooh",
        "author": "A.A. Milne",
        "prompt": "Pooh Bear and Piglet walking through Hundred Acre Wood, hand-drawn pencil sketch in the style of E.H. Shepard, classic children's book illustration, delicate line work, cross-hatching, whimsical forest scene, authentic Shepard style"
    },
    {
        "title": "Moomintroll_in_valley",
        "novel": "Finn Family Moomintroll",
        "author": "Tove Jansson",
        "prompt": "Moomintroll character in peaceful valley landscape, hand-drawn pencil sketch in the authentic style of Tove Jansson, classic Nordic children's book illustration, delicate line work, cross-hatching, whimsical creatures, gentle atmosphere"
    },
    {
        "title": "Ratty_in_boat",
        "novel": "The Wind in the Willows",
        "author": "Kenneth Grahame",
        "prompt": "Water Rat in small rowboat on peaceful river with reeds, hand-drawn pencil sketch in the style of E.H. Shepard, classic children's book illustration, delicate line work, cross-hatching, serene water scene"
    },
    {
        "title": "Peter_Rabbit_in_garden",
        "novel": "The Tale of Peter Rabbit",
        "author": "Beatrix Potter",
        "prompt": "Small rabbit in blue jacket sneaking through vegetable garden, hand-drawn pencil sketch in the style of E.H. Shepard and Tove Jansson, classic children's book illustration, delicate line work, cross-hatching, mischievous character"
    },
    {
        "title": "Christopher_Robin_and_toys",
        "novel": "Winnie-the-Pooh",
        "author": "A.A. Milne",
        "prompt": "Young boy Christopher Robin with Pooh, Piglet, and Eeyore in nursery, hand-drawn pencil sketch in the authentic style of E.H. Shepard, classic children's book illustration, delicate line work, cross-hatching, intimate domestic scene"
    }
]

async def generate_pencil_illustrations():
    """Generate pencil sketch illustrations using DALL-E."""

    # Load environment variables
    load_dotenv()

    # Ensure output directory exists
    output_dir = Path("pencil_sketches")
    output_dir.mkdir(exist_ok=True)

    # Initialize DALL-E provider
    provider = ProviderFactory.create_provider(
        ImageProvider.DALLE,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    print(f"🎨 Generating {len(CLASSIC_SCENES)} pencil sketch illustrations...")
    print(f"📁 Output directory: {output_dir.absolute()}")

    successful_generations = 0

    for i, scene in enumerate(CLASSIC_SCENES, 1):
        print(f"\n[{i}/{len(CLASSIC_SCENES)}] 🖼️ Generating: {scene['title']}")
        print(f"   📖 From: {scene['novel']} by {scene['author']}")

        try:
            # Create a simple illustration prompt object
            from src.illustrator.models import IllustrationPrompt

            illustration_prompt = IllustrationPrompt(
                provider=ImageProvider.DALLE,
                prompt=scene['prompt'],
                style_modifiers=["pencil sketch", "E.H. Shepard style", "Tove Jansson style", "hand-drawn"],
                negative_prompt=None,
                technical_params={
                    "model": "dall-e-3",
                    "size": "1024x1024",
                    "quality": "hd",
                    "style": "natural"
                }
            )

            # Generate the image
            result = await provider.generate_image(illustration_prompt)

            if result.get('success'):
                # Save the image
                image_data = result['image_data']
                output_path = output_dir / f"{scene['title']}.png"

                # Decode and save base64 image
                with open(output_path, 'wb') as f:
                    f.write(base64.b64decode(image_data))

                print(f"   ✅ Saved: {output_path}")
                successful_generations += 1

                # Add metadata file
                metadata_path = output_dir / f"{scene['title']}_info.txt"
                with open(metadata_path, 'w') as f:
                    f.write(f"Title: {scene['title']}\n")
                    f.write(f"Novel: {scene['novel']}\n")
                    f.write(f"Author: {scene['author']}\n")
                    f.write(f"Prompt: {scene['prompt']}\n")
                    f.write(f"Generated: {result.get('metadata', {}).get('created', 'Unknown')}\n")

            else:
                print(f"   ❌ Failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print(f"\n🎨 Generation complete!")
    print(f"✅ Successfully generated: {successful_generations}/{len(CLASSIC_SCENES)} illustrations")
    print(f"📁 All images saved in: {output_dir.absolute()}")

if __name__ == "__main__":
    asyncio.run(generate_pencil_illustrations())