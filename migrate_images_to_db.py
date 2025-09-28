#!/usr/bin/env python3
"""
Migration script to import existing images into the PostgreSQL database.

This script will:
1. Create database tables if they don't exist
2. Scan existing image files in illustrator_output/generated_images/
3. Create illustration records in the database
4. Handle existing manuscript and chapter data
"""

import os
import sys
import uuid
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from illustrator.db_config import create_tables, get_db
from illustrator.db_models import Manuscript, Chapter, Illustration
from illustrator.services.illustration_service import IllustrationService
from illustrator.models import ImageProvider
from illustrator.web.routes.manuscripts import get_saved_manuscripts


def get_or_create_manuscript(db, manuscript_data) -> Manuscript:
    """Get or create a manuscript record in the database."""
    # Generate the same UUID as used in the web routes
    manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript_data.file_path))

    # Check if manuscript already exists
    existing = db.query(Manuscript).filter(Manuscript.id == manuscript_id).first()
    if existing:
        return existing

    # Create new manuscript
    manuscript = Manuscript(
        id=manuscript_id,
        title=manuscript_data.metadata.title,
        author=manuscript_data.metadata.author,
        genre=manuscript_data.metadata.genre,
        total_chapters=len(manuscript_data.chapters),
        created_at=datetime.fromisoformat(manuscript_data.metadata.created_at.replace('Z', '+00:00')),
        updated_at=datetime.fromisoformat(manuscript_data.saved_at.replace('Z', '+00:00'))
    )

    db.add(manuscript)
    db.commit()
    db.refresh(manuscript)

    return manuscript


def get_or_create_chapter(db, chapter_data, manuscript_id: str) -> Chapter:
    """Get or create a chapter record in the database."""
    # Generate a consistent UUID for the chapter
    chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chapter_data.number}_{chapter_data.title}"))

    # Check if chapter already exists
    existing = db.query(Chapter).filter(Chapter.id == chapter_id).first()
    if existing:
        return existing

    # Create new chapter
    chapter = Chapter(
        id=chapter_id,
        manuscript_id=manuscript_id,
        title=chapter_data.title,
        content=chapter_data.content,
        number=chapter_data.number,
        word_count=chapter_data.word_count
    )

    db.add(chapter)
    db.commit()
    db.refresh(chapter)

    return chapter


def parse_filename(filename: str) -> Dict[str, Optional[int]]:
    """Parse chapter and scene numbers from filename."""
    chapter_num = None
    scene_num = None

    if "chapter_" in filename and "scene_" in filename:
        try:
            parts = filename.split("_")
            chapter_idx = next(i for i, part in enumerate(parts) if part == "chapter")
            scene_idx = next(i for i, part in enumerate(parts) if part == "scene")

            if chapter_idx + 1 < len(parts) and scene_idx + 1 < len(parts):
                chapter_num = int(parts[chapter_idx + 1])
                scene_num = int(parts[scene_idx + 1].split(".")[0])  # Remove extension
        except (ValueError, IndexError, StopIteration):
            pass

    return {"chapter_num": chapter_num, "scene_num": scene_num}


def migrate_images():
    """Main migration function."""
    print("🗄️  Starting image migration to PostgreSQL database...")

    # Create database tables
    print("📋 Creating database tables...")
    create_tables()

    # Get database session
    db = get_db()

    try:
        # Load manuscript data from file system
        print("📚 Loading manuscript data...")
        manuscripts = get_saved_manuscripts()

        if not manuscripts:
            print("❌ No manuscripts found in the file system")
            return

        print(f"📖 Found {len(manuscripts)} manuscripts")

        # Create manuscript and chapter records
        manuscript_map = {}  # manuscript_id -> Manuscript object
        chapter_map = {}     # (manuscript_id, chapter_num) -> Chapter object

        for manuscript_data in manuscripts:
            print(f"📝 Processing manuscript: {manuscript_data.metadata.title}")

            # Create or get manuscript
            manuscript = get_or_create_manuscript(db, manuscript_data)
            manuscript_map[str(manuscript.id)] = manuscript

            # Create or get chapters
            for chapter_data in manuscript_data.chapters:
                chapter = get_or_create_chapter(db, chapter_data, str(manuscript.id))
                chapter_map[(str(manuscript.id), chapter_data.number)] = chapter

        # Scan for image files
        generated_images_dir = Path("illustrator_output") / "generated_images"

        if not generated_images_dir.exists():
            print(f"❌ Generated images directory not found: {generated_images_dir}")
            return

        print(f"🖼️  Scanning for images in: {generated_images_dir}")

        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        image_files = [
            f for f in generated_images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print("❌ No image files found")
            return

        print(f"🎨 Found {len(image_files)} image files")

        # Initialize illustration service
        illustration_service = IllustrationService(db)

        imported_count = 0
        skipped_count = 0

        for image_file in image_files:
            filename = image_file.name
            print(f"🔍 Processing image: {filename}")

            # Parse filename to extract chapter and scene info
            parsed = parse_filename(filename)
            chapter_num = parsed["chapter_num"]
            scene_num = parsed["scene_num"]

            if chapter_num is None or scene_num is None:
                print(f"⚠️  Could not parse chapter/scene from filename: {filename}")
                skipped_count += 1
                continue

            # Find matching manuscript and chapter
            manuscript = None
            chapter = None

            # For now, assume all images belong to the first manuscript
            # (This could be enhanced to match based on filename patterns)
            if manuscripts:
                first_manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscripts[0].file_path))
                manuscript = manuscript_map.get(first_manuscript_id)

                if manuscript:
                    chapter = chapter_map.get((str(manuscript.id), chapter_num))

            if not manuscript or not chapter:
                print(f"⚠️  Could not find matching manuscript/chapter for {filename} (chapter {chapter_num})")
                skipped_count += 1
                continue

            # Check if illustration already exists
            existing = db.query(Illustration).filter(
                Illustration.chapter_id == chapter.id,
                Illustration.scene_number == scene_num
            ).first()

            if existing:
                print(f"⏭️  Illustration already exists: {filename}")
                skipped_count += 1
                continue

            # Get file information
            file_size = image_file.stat().st_size
            created_at = datetime.fromtimestamp(image_file.stat().st_mtime)

            # Create illustration record
            try:
                illustration = illustration_service.save_illustration(
                    manuscript_id=str(manuscript.id),
                    chapter_id=str(chapter.id),
                    scene_number=scene_num,
                    filename=filename,
                    file_path=str(image_file.absolute()),
                    prompt=f"Generated illustration for Chapter {chapter_num}, Scene {scene_num}",
                    image_provider=ImageProvider.IMAGEN4,  # Default assumption
                    title=f"Chapter {chapter_num} - Scene {scene_num}",
                    description=f"Generated illustration for Chapter {chapter_num}, Scene {scene_num}",
                    file_size=file_size
                )

                # Update timestamps to match file timestamps
                illustration.created_at = created_at
                illustration.updated_at = created_at
                db.commit()

                print(f"✅ Imported: {filename}")
                imported_count += 1

            except Exception as e:
                print(f"❌ Error importing {filename}: {e}")
                skipped_count += 1
                continue

        print(f"\n🎉 Migration complete!")
        print(f"   ✅ Successfully imported: {imported_count} images")
        print(f"   ⏭️  Skipped: {skipped_count} images")
        print(f"   📊 Total processed: {len(image_files)} images")

    except Exception as e:
        print(f"❌ Migration failed: {e}")
        raise

    finally:
        db.close()


if __name__ == "__main__":
    migrate_images()