#!/usr/bin/env python3
"""Migration script to import existing images into the MongoDB database."""

from __future__ import annotations

import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from illustrator.db_config import create_tables, get_mongo_database
from illustrator.db_models import CHAPTERS_COLLECTION, MANUSCRIPTS_COLLECTION
from illustrator.models import ImageProvider
from illustrator.services.illustration_service import IllustrationService
from illustrator.web.routes.manuscripts import get_saved_manuscripts


def ensure_manuscript(manuscripts_col, manuscript_data) -> Dict[str, Any]:
    """Ensure a manuscript document exists in MongoDB and return its metadata."""

    manuscript_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, manuscript_data.file_path))
    created_at = datetime.fromisoformat(manuscript_data.metadata.created_at.replace("Z", "+00:00"))
    updated_at = datetime.fromisoformat(manuscript_data.saved_at.replace("Z", "+00:00"))

    payload = {
        "title": manuscript_data.metadata.title,
        "author": manuscript_data.metadata.author,
        "genre": manuscript_data.metadata.genre,
        "total_chapters": len(manuscript_data.chapters),
        "updated_at": updated_at,
    }

    manuscripts_col.update_one(
        {"_id": manuscript_id},
        {
            "$set": payload,
            "$setOnInsert": {"created_at": created_at},
        },
        upsert=True,
    )

    payload.update({"id": manuscript_id, "created_at": created_at})
    return payload


def ensure_chapter(chapters_col, chapter_data, manuscript_id: str) -> Dict[str, Any]:
    """Ensure a chapter document exists in MongoDB and return its metadata."""

    chapter_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{manuscript_id}_{chapter_data.number}_{chapter_data.title}"))
    now = datetime.utcnow()
    payload = {
        "manuscript_id": manuscript_id,
        "title": chapter_data.title,
        "content": chapter_data.content,
        "number": chapter_data.number,
        "word_count": chapter_data.word_count,
        "updated_at": now,
    }

    chapters_col.update_one(
        {"_id": chapter_id},
        {
            "$set": payload,
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

    payload.update({"id": chapter_id})
    return payload


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
                scene_num = int(parts[scene_idx + 1].split(".")[0])
        except (ValueError, IndexError, StopIteration):
            pass

    return {"chapter_num": chapter_num, "scene_num": scene_num}


def migrate_images() -> None:
    """Populate MongoDB with illustration metadata for existing images."""

    print("🗄️  Starting image migration to MongoDB database...")
    print("📋 Ensuring database indexes are present...")
    create_tables()

    db = get_mongo_database()
    manuscripts_col = db[MANUSCRIPTS_COLLECTION]
    chapters_col = db[CHAPTERS_COLLECTION]

    try:
        print("📚 Loading manuscript data...")
        manuscripts = get_saved_manuscripts()
        if not manuscripts:
            print("❌ No manuscripts found in the file system")
            return

        print(f"📖 Found {len(manuscripts)} manuscripts")

        manuscript_map: Dict[str, Dict[str, Any]] = {}
        chapter_map: Dict[tuple[str, int], Dict[str, Any]] = {}

        for manuscript_data in manuscripts:
            print(f"📝 Processing manuscript: {manuscript_data.metadata.title}")
            manuscript_doc = ensure_manuscript(manuscripts_col, manuscript_data)
            manuscript_map[manuscript_doc["id"]] = manuscript_doc

            for chapter_data in manuscript_data.chapters:
                chapter_doc = ensure_chapter(chapters_col, chapter_data, manuscript_doc["id"])
                chapter_map[(manuscript_doc["id"], chapter_data.number)] = chapter_doc

        generated_images_dir = Path("illustrator_output") / "generated_images"
        if not generated_images_dir.exists():
            print(f"❌ Generated images directory not found: {generated_images_dir}")
            return

        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff"}
        image_files = [
            f for f in generated_images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print("❌ No image files found")
            return

        print(f"🎨 Found {len(image_files)} image files")

        illustration_service = IllustrationService(db)
        imported_count = 0
        skipped_count = 0

        default_manuscript_id = next(iter(manuscript_map), None)

        for image_file in image_files:
            filename = image_file.name
            print(f"🔍 Processing image: {filename}")

            parsed = parse_filename(filename)
            chapter_num = parsed["chapter_num"]
            scene_num = parsed["scene_num"]

            if chapter_num is None or scene_num is None:
                print(f"⚠️  Could not parse chapter/scene from filename: {filename}")
                skipped_count += 1
                continue

            manuscript_doc = manuscript_map.get(default_manuscript_id) if default_manuscript_id else None
            chapter_doc = None
            if manuscript_doc:
                chapter_doc = chapter_map.get((manuscript_doc["id"], chapter_num))

            if not manuscript_doc or not chapter_doc:
                print(
                    f"⚠️  Could not find matching manuscript/chapter for {filename} (chapter {chapter_num})"
                )
                skipped_count += 1
                continue

            existing = illustration_service.illustrations.find_one(
                {
                    "chapter_id": chapter_doc["id"],
                    "scene_number": scene_num,
                }
            )
            if existing:
                print(f"⏭️  Illustration already exists: {filename}")
                skipped_count += 1
                continue

            file_size = image_file.stat().st_size if image_file.exists() else None

            try:
                illustration = illustration_service.save_illustration(
                    manuscript_id=manuscript_doc["id"],
                    chapter_id=chapter_doc["id"],
                    scene_number=scene_num,
                    filename=filename,
                    file_path=str(image_file.resolve()),
                    prompt=f"Imported image for Chapter {chapter_num}, Scene {scene_num}",
                    image_provider=ImageProvider.DALLE,
                    emotional_moment=None,
                    style_config=None,
                    title=f"Chapter {chapter_num} Scene {scene_num}",
                    description=f"Imported illustration for Chapter {chapter_num}, Scene {scene_num}",
                    file_size=file_size,
                    width=None,
                    height=None,
                    chapter_number=chapter_num,
                )
                print(f"✅ Imported illustration {illustration['id']}: {illustration['filename']}")
                imported_count += 1
            except Exception as exc:  # noqa: BLE001
                print(f"❌ Error importing {filename}: {exc}")
                skipped_count += 1

        print("\n📊 Migration Summary")
        print("=" * 25)
        print(f"✅ Imported illustrations: {imported_count}")
        print(f"⏭️  Skipped files: {skipped_count}")

    except Exception as exc:  # noqa: BLE001
        print(f"❌ Migration failed: {exc}")
        raise


if __name__ == "__main__":
    migrate_images()
