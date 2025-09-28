#!/usr/bin/env python3
"""Generate pencil sketch illustrations for classic novels."""

import asyncio
import os
from pathlib import Path

from src.illustrator.context import ManuscriptContext
from src.illustrator.models import Chapter, EmotionalMoment, EmotionalTone, ImageProvider
from src.illustrator.graph import graph

# Classic novel excerpts for illustration
CLASSIC_EXCERPTS = [
    {
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "text": """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it. 'And what is the use of a book,' thought Alice, 'without pictures or conversations?' So she was considering in her own mind, as well as she could, for the hot day made her feel very sleepy and stupid, whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her."""
    },
    {
        "title": "The Wind in the Willows",
        "author": "Kenneth Grahame",
        "text": """The Mole and the Water Rat sat on the grassy bank by the river, watching the lazy water-voles playing in the reeds, and the swallows dipping and skimming over the surface of the water. 'Believe me, my young friend, there is nothing—absolutely nothing—half so much worth doing as simply messing about in boats,' said the Water Rat dreamily. 'Simply messing about in boats—or with boats... In or out of 'em, it doesn't matter.'"""
    },
    {
        "title": "The Secret Garden",
        "author": "Frances Hodgson Burnett",
        "text": """Mary pushed open the heavy wooden door and stepped into the secret garden. It was the sweetest, most mysterious-looking place anyone could imagine. The high walls were covered with the leafless stems of climbing roses, which were so thick that they made a little room, with the sky for a roof. The ground was covered with grass of a wintry brown, and out of it grew clumps of bushes which were surely rose-bushes if they were alive."""
    },
    {
        "title": "Treasure Island",
        "author": "Robert Louis Stevenson",
        "text": """I remember him as if it were yesterday, as he came plodding to the inn door, his sea-chest following behind him in a handbarrow; a tall, strong, heavy, nut-brown man; his tarry pigtail falling over the shoulder of his soiled blue coat; his hands ragged and scarred, with black, broken nails; and the sabre cut across one cheek, a dirty, livid white."""
    },
    {
        "title": "The Jungle Book",
        "author": "Rudyard Kipling",
        "text": """It was seven o'clock of a very warm evening in the Seeonee hills when Father Wolf woke up from his day's rest, scratched himself, yawned, and spread out his paws one after the other to get rid of the sleepy feeling in their tips. Mother Wolf lay with her big gray nose dropped across her four tumbling, squeaking cubs, and the moon shone into the mouth of the cave where they all lived."""
    }
]

async def generate_classic_illustrations():
    """Generate pencil sketch illustrations for classic novel excerpts."""

    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()

    # Create context with pencil sketch style
    context = ManuscriptContext(
        user_id="classic_novels_test",
        image_provider=ImageProvider.DALLE,
        default_art_style="hand-drawn pencil sketch in the style of E.H. Shepard and Tove Jansson, classic children's book illustration, delicate line work, cross-hatching, whimsical characters",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    # Use the illustration workflow graph
    app = graph

    # Process each classic novel excerpt
    for i, excerpt in enumerate(CLASSIC_EXCERPTS, 1):
        print(f"\n📚 Processing: {excerpt['title']} by {excerpt['author']}")

        # Create chapter from excerpt
        chapter = Chapter(
            number=i,
            title=f"{excerpt['title']} - Excerpt",
            content=excerpt["text"],
            word_count=len(excerpt["text"].split())
        )

        # Create initial state
        initial_state = {
            "context": context,
            "chapter": chapter,
            "emotional_moments": [],
            "illustrations": [],
            "status": "analyzing"
        }

        try:
            # Run the workflow
            final_state = await app.ainvoke(initial_state)

            print(f"✅ Generated {len(final_state['illustrations'])} illustrations")

            # Save illustrations if any were generated
            if final_state['illustrations']:
                output_dir = Path(f"classic_illustrations/{excerpt['title'].replace(' ', '_').lower()}")
                output_dir.mkdir(parents=True, exist_ok=True)

                for j, illustration in enumerate(final_state['illustrations']):
                    if 'image_data' in illustration:
                        output_path = output_dir / f"illustration_{j+1}.png"
                        # Save base64 image data
                        import base64
                        with open(output_path, 'wb') as f:
                            f.write(base64.b64decode(illustration['image_data']))
                        print(f"💾 Saved: {output_path}")

        except Exception as e:
            print(f"❌ Error processing {excerpt['title']}: {e}")

if __name__ == "__main__":
    asyncio.run(generate_classic_illustrations())