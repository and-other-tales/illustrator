"""Service layer for managing illustrations stored in MongoDB."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pymongo import ASCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from ..db_config import get_mongo_database
from ..db_models import CHAPTERS_COLLECTION, ILLUSTRATIONS_COLLECTION
from ..models import EmotionalMoment, ImageProvider

IllustrationRecord = Dict[str, Any]


class IllustrationService:
    """Service for managing illustration metadata stored in MongoDB."""

    def __init__(self, db: Optional[Database] = None):
        """Initialise the service with an optional Mongo database instance."""

        self.db: Database = db or get_mongo_database()
        self.illustrations: Collection = self.db[ILLUSTRATIONS_COLLECTION]
        self.chapters: Collection = self.db[CHAPTERS_COLLECTION]

    @staticmethod
    def _to_record(document: Dict[str, Any]) -> IllustrationRecord:
        """Convert a Mongo document into an application-friendly record."""

        record = dict(document)
        record["id"] = str(record.pop("_id"))

        # Normalise optional JSON fields to native structures
        if isinstance(record.get("style_config"), str):
            try:
                record["style_config"] = json.loads(record["style_config"])
            except json.JSONDecodeError:
                record["style_config"] = {}
        record.setdefault("style_config", {})

        emotional = record.get("emotional_tones")
        if isinstance(emotional, str):
            record["emotional_tones"] = [tone for tone in emotional.split(",") if tone]
        else:
            record["emotional_tones"] = emotional or []

        return record

    def save_illustration(
        self,
        manuscript_id: str,
        chapter_id: str,
        scene_number: int,
        filename: str,
        file_path: str,
        prompt: str,
        image_provider: ImageProvider,
        emotional_moment: Optional[EmotionalMoment] = None,
        style_config: Optional[Dict[str, Any]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
        file_size: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        chapter_number: Optional[int] = None,
    ) -> IllustrationRecord:
        """Persist an illustration document and return the stored record."""

        illustration_id = str(uuid4())
        web_url = f"/generated/{filename}"

        emotional_tones: List[str] | None = None
        intensity_score: float | None = None
        text_excerpt: str | None = None

        if emotional_moment:
            emotional_tones = [tone.value for tone in emotional_moment.emotional_tones]
            intensity_score = emotional_moment.intensity_score
            text_excerpt = emotional_moment.text_excerpt

            if not title:
                title = f"Chapter Scene {scene_number}"
            if not description:
                first_tone = emotional_moment.emotional_tones[0].value if emotional_moment.emotional_tones else "neutral"
                excerpt = (emotional_moment.text_excerpt or "").strip()
                snippet = f"{excerpt[:100]}..." if len(excerpt) > 100 else excerpt
                description = f"Illustration depicting: {snippet} (Tone: {first_tone})"

        document = {
            "_id": illustration_id,
            "manuscript_id": manuscript_id,
            "chapter_id": chapter_id,
            "chapter_number": chapter_number,
            "filename": filename,
            "file_path": file_path,
            "web_url": web_url,
            "scene_number": scene_number,
            "title": title or f"Scene {scene_number}",
            "description": description,
            "prompt": prompt,
            "negative_prompt": emotional_moment.negative_prompt if hasattr(emotional_moment, "negative_prompt") else None,
            "style_config": style_config or {},
            "image_provider": image_provider.value,
            "emotional_tones": emotional_tones or [],
            "intensity_score": intensity_score,
            "text_excerpt": text_excerpt,
            "file_size": file_size,
            "width": width,
            "height": height,
            "generation_status": "completed",
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        self.illustrations.insert_one(document)
        stored = self._to_record(document)

        # Enrich with chapter metadata when available
        chapter_doc = self.chapters.find_one({"_id": chapter_id})
        if chapter_doc:
            stored["chapter_title"] = chapter_doc.get("title")
            stored.setdefault("chapter_number", chapter_doc.get("number"))

        return stored

    def get_illustrations_by_manuscript(self, manuscript_id: str) -> List[IllustrationRecord]:
        """Return all illustrations for a manuscript, ordered by chapter then scene."""

        docs = list(
            self.illustrations.find({"manuscript_id": manuscript_id}).sort([
                ("chapter_number", ASCENDING),
                ("scene_number", ASCENDING),
                ("created_at", ASCENDING),
            ])
        )
        records = [self._to_record(doc) for doc in docs]

        if not records:
            return records

        # Attach chapter metadata for convenience
        chapter_ids = {record["chapter_id"] for record in records if record.get("chapter_id")}
        if chapter_ids:
            chapter_docs = self.chapters.find({"_id": {"$in": list(chapter_ids)}})
            chapter_map = {doc["_id"]: doc for doc in chapter_docs}
            for record in records:
                chapter_doc = chapter_map.get(record.get("chapter_id"))
                if chapter_doc:
                    record.setdefault("chapter_number", chapter_doc.get("number"))
                    record.setdefault("chapter_title", chapter_doc.get("title"))

        return records

    def get_illustrations_by_chapter(self, chapter_id: str) -> List[IllustrationRecord]:
        """Return all illustrations for a specific chapter."""

        docs = list(
            self.illustrations.find({"chapter_id": chapter_id}).sort([
                ("scene_number", ASCENDING),
                ("created_at", ASCENDING),
            ])
        )
        records = [self._to_record(doc) for doc in docs]

        chapter_doc = self.chapters.find_one({"_id": chapter_id})
        if chapter_doc:
            for record in records:
                record.setdefault("chapter_number", chapter_doc.get("number"))
                record.setdefault("chapter_title", chapter_doc.get("title"))
        return records

    def get_illustration_by_id(self, illustration_id: str) -> Optional[IllustrationRecord]:
        """Fetch a single illustration by its identifier."""

        doc = self.illustrations.find_one({"_id": illustration_id})
        if not doc:
            return None
        record = self._to_record(doc)
        chapter_doc = self.chapters.find_one({"_id": record.get("chapter_id")})
        if chapter_doc:
            record.setdefault("chapter_number", chapter_doc.get("number"))
            record.setdefault("chapter_title", chapter_doc.get("title"))
        return record

    def delete_illustration(self, illustration_id: str) -> bool:
        """Delete a single illustration record and remove its associated file."""

        record = self.get_illustration_by_id(illustration_id)
        if not record:
            return False

        if record.get("file_path"):
            try:
                Path(record["file_path"]).unlink(missing_ok=True)
            except TypeError:
                file_path = Path(record["file_path"])
                if file_path.exists():
                    file_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass

        self.illustrations.delete_one({"_id": illustration_id})
        return True

    def delete_illustrations_for_manuscript(self, manuscript_id: str) -> int:
        """Remove all illustrations for a manuscript and delete their files."""

        records = self.get_illustrations_by_manuscript(manuscript_id)
        for record in records:
            if record.get("file_path"):
                try:
                    Path(record["file_path"]).unlink(missing_ok=True)
                except TypeError:
                    file_path = Path(record["file_path"])
                    if file_path.exists():
                        file_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

        if not records:
            return 0

        ids = [record["id"] for record in records]
        result = self.illustrations.delete_many({"_id": {"$in": ids}})
        return int(result.deleted_count)

    def close(self) -> None:
        """MongoDB connections are pooled globally; nothing to close."""

        pass
