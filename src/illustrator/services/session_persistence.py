"""
Session persistence service for handling MongoDB-backed session storage.
Provides complete session state management, checkpointing, and recovery capabilities.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pymongo import DESCENDING
from pymongo.collection import Collection
from pymongo.database import Database

from ..db_config import get_db
from ..db_models import (
    PROCESSING_CHECKPOINTS_COLLECTION,
    PROCESSING_LOGS_COLLECTION,
    PROCESSING_SESSIONS_COLLECTION,
    SESSION_IMAGES_COLLECTION,
)


@dataclass
class SessionState:
    """Complete session state for persistence and recovery."""

    session_id: str
    manuscript_id: str
    external_session_id: Optional[str]
    status: str
    progress_percent: int
    current_chapter: Optional[int]
    total_chapters: int
    current_task: Optional[str]
    style_config: Dict[str, Any]
    max_emotional_moments: int
    last_completed_step: Optional[str]
    last_completed_chapter: int
    processed_chapters: List[int]
    current_prompts: List[str]
    generated_images: List[Dict[str, Any]]
    emotional_moments: List[Dict[str, Any]]
    total_images_generated: int
    error_message: Optional[str]
    started_at: str
    paused_at: Optional[str]
    resumed_at: Optional[str]
    last_heartbeat: str


class SessionPersistenceService:
    """Comprehensive session persistence and recovery service."""

    def __init__(self, db: Optional[Database] = None, data_dir: str = "illustrator_output/sessions"):
        self.db: Database = db or get_db()

        self.sessions: Collection = self.db[PROCESSING_SESSIONS_COLLECTION]
        self.checkpoints: Collection = self.db[PROCESSING_CHECKPOINTS_COLLECTION]
        self.logs: Collection = self.db[PROCESSING_LOGS_COLLECTION]
        self.session_images: Collection = self.db[SESSION_IMAGES_COLLECTION]

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sessions_dir = self.data_dir / "active_sessions"
        self.checkpoints_dir = self.data_dir / "checkpoints"
        self.recovery_dir = self.data_dir / "recovery"

        for directory in [self.sessions_dir, self.checkpoints_dir, self.recovery_dir]:
            directory.mkdir(exist_ok=True)

        self.checkpoint_sequence: Dict[str, int] = {}

    @staticmethod
    def _session_record(document: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(document)
        record["id"] = str(record.pop("_id"))
        return record

    @staticmethod
    def _checkpoint_record(document: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(document)
        record["id"] = str(record.pop("_id"))
        return record

    def create_session(
        self,
        manuscript_id: str,
        external_session_id: Optional[str] = None,
        style_config: Optional[Dict[str, Any]] = None,
        max_emotional_moments: int = 10,
        total_chapters: int = 0,
    ) -> Dict[str, Any]:
        """Create a new processing session in MongoDB and file storage."""

        session_id = str(uuid.uuid4())
        now = datetime.utcnow()
        session_doc = {
            "_id": session_id,
            "manuscript_id": manuscript_id,
            "external_session_id": external_session_id,
            "session_type": "illustration_generation",
            "status": "pending",
            "progress_percent": 0,
            "current_task": None,
            "current_chapter": None,
            "total_chapters": total_chapters,
            "style_config": style_config or {},
            "max_emotional_moments": max_emotional_moments,
            "last_completed_step": None,
            "last_completed_chapter": 0,
            "processed_chapters": [],
            "current_prompts": [],
            "total_images_generated": 0,
            "total_chapters_processed": 0,
            "error_message": None,
            "last_checkpoint_id": None,
            "can_resume": True,
            "resume_data": None,
            "started_at": now,
            "completed_at": None,
            "paused_at": None,
            "resumed_at": None,
            "last_heartbeat": now,
            "created_at": now,
            "updated_at": now,
        }

        self.sessions.insert_one(session_doc)
        record = self._session_record(session_doc)

        session_state = SessionState(
            session_id=session_id,
            manuscript_id=manuscript_id,
            external_session_id=external_session_id,
            status="pending",
            progress_percent=0,
            current_chapter=None,
            total_chapters=total_chapters,
            current_task=None,
            style_config=style_config or {},
            max_emotional_moments=max_emotional_moments,
            last_completed_step=None,
            last_completed_chapter=0,
            processed_chapters=[],
            current_prompts=[],
            generated_images=[],
            emotional_moments=[],
            total_images_generated=0,
            error_message=None,
            started_at=now.isoformat(),
            paused_at=None,
            resumed_at=None,
            last_heartbeat=now.isoformat(),
        )

        self._save_session_file(session_state)
        return record

    def update_session_status(
        self,
        session_id: str,
        status: str,
        progress_percent: Optional[int] = None,
        current_task: Optional[str] = None,
        current_chapter: Optional[int] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update session status in MongoDB and file storage."""

        now = datetime.utcnow()
        update_fields: Dict[str, Any] = {
            "status": status,
            "updated_at": now,
            "last_heartbeat": now,
        }
        if progress_percent is not None:
            update_fields["progress_percent"] = progress_percent
        if current_task is not None:
            update_fields["current_task"] = current_task
        if current_chapter is not None:
            update_fields["current_chapter"] = current_chapter
        if error_message is not None:
            update_fields["error_message"] = error_message

        if status == "paused":
            update_fields["paused_at"] = now
        elif status == "running":
            update_fields.setdefault("resumed_at", now)
        elif status == "completed":
            update_fields["completed_at"] = now

        self.sessions.update_one({"_id": session_id}, {"$set": update_fields})

        session_state = self._load_session_file(session_id)
        if session_state:
            session_state.status = status
            if progress_percent is not None:
                session_state.progress_percent = progress_percent
            if current_task is not None:
                session_state.current_task = current_task
            if current_chapter is not None:
                session_state.current_chapter = current_chapter
            if error_message is not None:
                session_state.error_message = error_message

            session_state.last_heartbeat = now.isoformat()

            if status == "paused":
                session_state.paused_at = now.isoformat()
            elif status == "running" and session_state.paused_at:
                session_state.resumed_at = now.isoformat()

            self._save_session_file(session_state)

    def create_checkpoint(
        self,
        session_id: str,
        checkpoint_type: str,
        chapter_number: int,
        step_name: str,
        sequence_number: int,
        checkpoint_data: Dict[str, Any],
        images_generated_count: int = 0,
        total_images_at_checkpoint: int = 0,
        progress_percent: int = 0,
        processing_state: Optional[Dict[str, Any]] = None,
        generated_prompts: Optional[List[str]] = None,
        emotional_moments_data: Optional[List[Dict[str, Any]]] = None,
        is_resumable: bool = True,
        next_action: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a processing checkpoint in MongoDB and on disk."""

        checkpoint_id = str(uuid.uuid4())
        now = datetime.utcnow()
        checkpoint_doc = {
            "_id": checkpoint_id,
            "session_id": session_id,
            "checkpoint_type": checkpoint_type,
            "chapter_number": chapter_number,
            "step_name": step_name,
            "sequence_number": sequence_number,
            "checkpoint_data": checkpoint_data,
            "images_generated_count": images_generated_count,
            "total_images_at_checkpoint": total_images_at_checkpoint,
            "progress_percent": progress_percent,
            "processing_state": processing_state or {},
            "generated_prompts": generated_prompts or [],
            "emotional_moments_data": emotional_moments_data or [],
            "is_resumable": is_resumable,
            "next_action": next_action,
            "created_at": now,
            "completed_at": None,
        }

        self.checkpoints.insert_one(checkpoint_doc)
        self.sessions.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "last_checkpoint_id": checkpoint_id,
                    "last_completed_step": step_name,
                    "last_completed_chapter": chapter_number,
                    "total_images_generated": total_images_at_checkpoint,
                    "updated_at": now,
                }
            },
        )

        record = self._checkpoint_record(checkpoint_doc)

        checkpoint_file = self.checkpoints_dir / f"{session_id}_{sequence_number:04d}.json"
        checkpoint_payload = {
            "checkpoint_id": checkpoint_id,
            "session_id": session_id,
            "checkpoint_type": checkpoint_type,
            "chapter_number": chapter_number,
            "step_name": step_name,
            "sequence_number": sequence_number,
            "created_at": now.isoformat(),
            "checkpoint_data": checkpoint_data,
            "processing_state": processing_state or {},
            "generated_prompts": generated_prompts or [],
            "emotional_moments_data": emotional_moments_data or [],
            "next_action": next_action,
            "images_generated_count": images_generated_count,
            "total_images_at_checkpoint": total_images_at_checkpoint,
            "progress_percent": progress_percent,
        }

        with open(checkpoint_file, "w", encoding="utf-8") as handle:
            json.dump(checkpoint_payload, handle, indent=2)

        return record

    def get_resumable_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions that can be resumed."""

        docs = self.sessions.find(
            {
                "can_resume": True,
                "status": {"$in": ["paused", "failed"]},
            }
        ).sort("updated_at", DESCENDING)
        return [self._session_record(doc) for doc in docs]

    def get_session_for_resume(
        self, session_id: str
    ) -> Optional[Tuple[Dict[str, Any], SessionState, Optional[Dict[str, Any]]]]:
        """Get complete session information for resuming."""

        session_doc = self.sessions.find_one({"_id": session_id})
        if not session_doc or not session_doc.get("can_resume", False):
            return None

        session_state = self._load_session_file(session_id)
        if not session_state:
            return None

        checkpoint_doc = self.checkpoints.find({"session_id": session_id}).sort(
            "sequence_number", DESCENDING
        ).limit(1)
        latest_checkpoint = None
        for doc in checkpoint_doc:
            latest_checkpoint = self._checkpoint_record(doc)
            break

        return self._session_record(session_doc), session_state, latest_checkpoint

    def add_session_image(
        self,
        session_id: str,
        illustration_id: str,
        checkpoint_id: Optional[str] = None,
        generation_order: int = 0,
        chapter_order: int = 0,
        scene_order: int = 0,
        web_url: Optional[str] = None,
        prompt_used: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add an image to a session for tracking and restoration."""

        now = datetime.utcnow()
        record_id = str(uuid.uuid4())
        image_doc = {
            "_id": record_id,
            "session_id": session_id,
            "illustration_id": illustration_id,
            "checkpoint_id": checkpoint_id,
            "generation_order": generation_order,
            "chapter_order": chapter_order,
            "scene_order": scene_order,
            "web_url": web_url,
            "prompt_used": prompt_used,
            "generation_status": "completed",
            "completed_at": now,
            "created_at": now,
        }

        self.session_images.insert_one(image_doc)

        session_state = self._load_session_file(session_id)
        if session_state:
            image_info = {
                "illustration_id": illustration_id,
                "web_url": web_url,
                "prompt_used": prompt_used,
                "generation_order": generation_order,
                "chapter_order": chapter_order,
                "scene_order": scene_order,
                "created_at": now.isoformat(),
            }
            session_state.generated_images.append(image_info)
            session_state.total_images_generated = len(session_state.generated_images)
            self._save_session_file(session_state)

        return image_doc

    def log_session_event(
        self,
        session_id: str,
        level: str,
        message: str,
        chapter_number: Optional[int] = None,
        step_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Log an event for a session."""

        now = datetime.utcnow()
        log_id = str(uuid.uuid4())
        log_doc = {
            "_id": log_id,
            "session_id": session_id,
            "level": level,
            "message": message,
            "chapter_number": chapter_number,
            "step_name": step_name,
            "timestamp": now,
        }
        self.logs.insert_one(log_doc)
        return log_doc

    def get_session_by_external_id(self, external_session_id: str) -> Optional[Dict[str, Any]]:
        """Get a session by its external session ID."""

        doc = self.sessions.find_one({"external_session_id": external_session_id})
        return self._session_record(doc) if doc else None

    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """Clean up old completed sessions and their files."""

        cutoff = datetime.utcnow() - timedelta(days=days_old)
        old_sessions = list(
            self.sessions.find(
                {
                    "status": {"$in": ["completed", "failed", "cancelled"]},
                    "updated_at": {"$lt": cutoff},
                }
            )
        )

        cleaned = 0
        for session_doc in old_sessions:
            session_id = str(session_doc["_id"])

            self.session_images.delete_many({"session_id": session_id})
            self.checkpoints.delete_many({"session_id": session_id})
            self.logs.delete_many({"session_id": session_id})

            session_file = self.sessions_dir / f"{session_id}.json"
            if session_file.exists():
                session_file.unlink()

            for checkpoint_file in self.checkpoints_dir.glob(f"{session_id}_*.json"):
                checkpoint_file.unlink()

            self.sessions.delete_one({"_id": session_id})
            cleaned += 1

        return cleaned

    def _save_session_file(self, session_state: SessionState) -> None:
        session_file = self.sessions_dir / f"{session_state.session_id}.json"
        session_dict = {
            "session_id": session_state.session_id,
            "manuscript_id": session_state.manuscript_id,
            "external_session_id": session_state.external_session_id,
            "status": session_state.status,
            "progress_percent": session_state.progress_percent,
            "current_chapter": session_state.current_chapter,
            "total_chapters": session_state.total_chapters,
            "current_task": session_state.current_task,
            "style_config": session_state.style_config,
            "max_emotional_moments": session_state.max_emotional_moments,
            "last_completed_step": session_state.last_completed_step,
            "last_completed_chapter": session_state.last_completed_chapter,
            "processed_chapters": session_state.processed_chapters,
            "current_prompts": session_state.current_prompts,
            "generated_images": session_state.generated_images,
            "emotional_moments": session_state.emotional_moments,
            "total_images_generated": session_state.total_images_generated,
            "error_message": session_state.error_message,
            "started_at": session_state.started_at,
            "paused_at": session_state.paused_at,
            "resumed_at": session_state.resumed_at,
            "last_heartbeat": session_state.last_heartbeat,
        }

        with open(session_file, "w", encoding="utf-8") as handle:
            json.dump(session_dict, handle, indent=2)

    def _load_session_file(self, session_id: str) -> Optional[SessionState]:
        session_file = self.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return None

        try:
            with open(session_file, "r", encoding="utf-8") as handle:
                session_dict = json.load(handle)

            return SessionState(
                session_id=session_dict["session_id"],
                manuscript_id=session_dict["manuscript_id"],
                external_session_id=session_dict.get("external_session_id"),
                status=session_dict["status"],
                progress_percent=session_dict["progress_percent"],
                current_chapter=session_dict.get("current_chapter"),
                total_chapters=session_dict["total_chapters"],
                current_task=session_dict.get("current_task"),
                style_config=session_dict["style_config"],
                max_emotional_moments=session_dict["max_emotional_moments"],
                last_completed_step=session_dict.get("last_completed_step"),
                last_completed_chapter=session_dict["last_completed_chapter"],
                processed_chapters=session_dict["processed_chapters"],
                current_prompts=session_dict["current_prompts"],
                generated_images=session_dict["generated_images"],
                emotional_moments=session_dict["emotional_moments"],
                total_images_generated=session_dict["total_images_generated"],
                error_message=session_dict.get("error_message"),
                started_at=session_dict["started_at"],
                paused_at=session_dict.get("paused_at"),
                resumed_at=session_dict.get("resumed_at"),
                last_heartbeat=session_dict["last_heartbeat"],
            )
        except (json.JSONDecodeError, KeyError):
            return None

    def close(self) -> None:
        """MongoDB connections are pooled globally; nothing to close."""

        pass
