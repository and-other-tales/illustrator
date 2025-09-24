"""
Session persistence service for handling database and file-based session storage.
Provides complete session state management, checkpointing, and recovery capabilities.
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from sqlalchemy import desc, and_, or_
from sqlalchemy.orm import Session

from ..db_config import get_db
from ..db_models import (
    ProcessingSession,
    ProcessingCheckpoint,
    ProcessingLog,
    SessionImage,
    Manuscript,
    Chapter,
    Illustration
)
from ..models import EmotionalMoment, ImageProvider


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

    # Configuration
    style_config: Dict[str, Any]
    max_emotional_moments: int

    # Processing state
    last_completed_step: Optional[str]
    last_completed_chapter: int
    processed_chapters: List[int]
    current_prompts: List[str]
    generated_images: List[Dict[str, Any]]
    emotional_moments: List[Dict[str, Any]]

    # Results
    total_images_generated: int
    error_message: Optional[str]

    # Timestamps
    started_at: str
    paused_at: Optional[str]
    resumed_at: Optional[str]
    last_heartbeat: str


class SessionPersistenceService:
    """Comprehensive session persistence and recovery service."""

    def __init__(self, db: Optional[Session] = None, data_dir: str = "illustrator_output/sessions"):
        """Initialize the service.

        Args:
            db: Database session (optional, will create if not provided)
            data_dir: Directory for file-based session storage
        """
        self.db = db or get_db()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.sessions_dir = self.data_dir / "active_sessions"
        self.checkpoints_dir = self.data_dir / "checkpoints"
        self.recovery_dir = self.data_dir / "recovery"

        for directory in [self.sessions_dir, self.checkpoints_dir, self.recovery_dir]:
            directory.mkdir(exist_ok=True)

    def create_session(self,
                      manuscript_id: str,
                      external_session_id: Optional[str] = None,
                      style_config: Optional[Dict[str, Any]] = None,
                      max_emotional_moments: int = 10,
                      total_chapters: int = 0) -> ProcessingSession:
        """Create a new processing session in database and file storage.

        Args:
            manuscript_id: UUID of the manuscript
            external_session_id: External session ID for web sessions
            style_config: Style configuration for processing
            max_emotional_moments: Maximum emotional moments per chapter
            total_chapters: Total number of chapters to process

        Returns:
            Created ProcessingSession object
        """
        # Create database session
        db_session = ProcessingSession(
            manuscript_id=uuid.UUID(manuscript_id),
            external_session_id=external_session_id,
            status="pending",
            progress_percent=0,
            total_chapters=total_chapters,
            style_config=json.dumps(style_config or {}),
            max_emotional_moments=max_emotional_moments,
            can_resume=True
        )

        self.db.add(db_session)
        self.db.commit()
        self.db.refresh(db_session)

        # Create file-based session state
        session_state = SessionState(
            session_id=str(db_session.id),
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
            started_at=datetime.utcnow().isoformat(),
            paused_at=None,
            resumed_at=None,
            last_heartbeat=datetime.utcnow().isoformat()
        )

        self._save_session_file(session_state)

        return db_session

    def update_session_status(self,
                            session_id: str,
                            status: str,
                            progress_percent: Optional[int] = None,
                            current_task: Optional[str] = None,
                            current_chapter: Optional[int] = None,
                            error_message: Optional[str] = None) -> None:
        """Update session status in both database and file storage.

        Args:
            session_id: Session ID (UUID string)
            status: New status
            progress_percent: Progress percentage (0-100)
            current_task: Current task description
            current_chapter: Currently processing chapter number
            error_message: Error message if status is 'failed'
        """
        # Update database
        db_session = self.db.query(ProcessingSession).filter(
            ProcessingSession.id == uuid.UUID(session_id)
        ).first()

        if db_session:
            db_session.status = status
            if progress_percent is not None:
                db_session.progress_percent = progress_percent
            if current_task is not None:
                db_session.current_task = current_task
            if current_chapter is not None:
                db_session.current_chapter = current_chapter
            if error_message is not None:
                db_session.error_message = error_message

            db_session.last_heartbeat = datetime.utcnow()

            if status == "paused":
                db_session.paused_at = datetime.utcnow()
            elif status == "running" and db_session.paused_at:
                db_session.resumed_at = datetime.utcnow()
            elif status == "completed":
                db_session.completed_at = datetime.utcnow()

            self.db.commit()

        # Update file state
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

            session_state.last_heartbeat = datetime.utcnow().isoformat()

            if status == "paused":
                session_state.paused_at = datetime.utcnow().isoformat()
            elif status == "running" and session_state.paused_at:
                session_state.resumed_at = datetime.utcnow().isoformat()

            self._save_session_file(session_state)

    def create_checkpoint(self,
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
                         next_action: Optional[str] = None) -> ProcessingCheckpoint:
        """Create a processing checkpoint.

        Args:
            session_id: Session ID (UUID string)
            checkpoint_type: Type of checkpoint (chapter_start, chapter_analyzed, etc.)
            chapter_number: Chapter number being processed
            step_name: Name of the processing step
            sequence_number: Sequence number of this checkpoint
            checkpoint_data: Checkpoint data
            images_generated_count: Number of images generated at this checkpoint
            total_images_at_checkpoint: Total images generated up to this point
            progress_percent: Progress percentage at checkpoint
            processing_state: Detailed processing state
            generated_prompts: List of prompts generated
            emotional_moments_data: Emotional moments data
            is_resumable: Whether this checkpoint allows resuming
            next_action: Next action to take when resuming

        Returns:
            Created ProcessingCheckpoint object
        """
        checkpoint = ProcessingCheckpoint(
            session_id=uuid.UUID(session_id),
            checkpoint_type=checkpoint_type,
            chapter_number=chapter_number,
            step_name=step_name,
            sequence_number=sequence_number,
            checkpoint_data=json.dumps(checkpoint_data),
            images_generated_count=images_generated_count,
            total_images_at_checkpoint=total_images_at_checkpoint,
            progress_percent=progress_percent,
            processing_state=json.dumps(processing_state or {}),
            generated_prompts=json.dumps(generated_prompts or []),
            emotional_moments_data=json.dumps(emotional_moments_data or []),
            next_action=next_action,
            is_resumable=is_resumable
        )

        self.db.add(checkpoint)
        self.db.commit()
        self.db.refresh(checkpoint)

        # Update session's last checkpoint
        db_session = self.db.query(ProcessingSession).filter(
            ProcessingSession.id == uuid.UUID(session_id)
        ).first()

        if db_session:
            db_session.last_checkpoint_id = checkpoint.id
            db_session.last_completed_step = step_name
            db_session.last_completed_chapter = chapter_number
            db_session.total_images_generated = total_images_at_checkpoint
            self.db.commit()

        # Save checkpoint file
        checkpoint_file = self.checkpoints_dir / f"{session_id}_{sequence_number:04d}.json"
        checkpoint_data_with_meta = {
            "checkpoint_id": str(checkpoint.id),
            "session_id": session_id,
            "checkpoint_type": checkpoint_type,
            "chapter_number": chapter_number,
            "step_name": step_name,
            "sequence_number": sequence_number,
            "created_at": checkpoint.created_at.isoformat() if checkpoint.created_at else None,
            "checkpoint_data": checkpoint_data,
            "processing_state": processing_state or {},
            "generated_prompts": generated_prompts or [],
            "emotional_moments_data": emotional_moments_data or [],
            "next_action": next_action,
            "images_generated_count": images_generated_count,
            "total_images_at_checkpoint": total_images_at_checkpoint,
            "progress_percent": progress_percent
        }

        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data_with_meta, f, indent=2)

        return checkpoint

    def get_resumable_sessions(self) -> List[ProcessingSession]:
        """Get all sessions that can be resumed.

        Returns:
            List of resumable ProcessingSession objects
        """
        return (
            self.db.query(ProcessingSession)
            .filter(and_(
                ProcessingSession.can_resume == True,
                or_(
                    ProcessingSession.status == "paused",
                    ProcessingSession.status == "failed"
                )
            ))
            .order_by(desc(ProcessingSession.updated_at))
            .all()
        )

    def get_session_for_resume(self, session_id: str) -> Optional[Tuple[ProcessingSession, SessionState, Optional[ProcessingCheckpoint]]]:
        """Get complete session information for resuming.

        Args:
            session_id: Session ID (UUID string)

        Returns:
            Tuple of (ProcessingSession, SessionState, latest_checkpoint) or None
        """
        # Get database session
        db_session = self.db.query(ProcessingSession).filter(
            ProcessingSession.id == uuid.UUID(session_id)
        ).first()

        if not db_session or not db_session.can_resume:
            return None

        # Get file session state
        session_state = self._load_session_file(session_id)
        if not session_state:
            return None

        # Get latest checkpoint
        latest_checkpoint = (
            self.db.query(ProcessingCheckpoint)
            .filter(ProcessingCheckpoint.session_id == uuid.UUID(session_id))
            .order_by(desc(ProcessingCheckpoint.sequence_number))
            .first()
        )

        return (db_session, session_state, latest_checkpoint)

    def add_session_image(self,
                         session_id: str,
                         illustration_id: str,
                         checkpoint_id: Optional[str] = None,
                         generation_order: int = 0,
                         chapter_order: int = 0,
                         scene_order: int = 0,
                         web_url: Optional[str] = None,
                         prompt_used: Optional[str] = None) -> SessionImage:
        """Add an image to a session for tracking and restoration.

        Args:
            session_id: Session ID (UUID string)
            illustration_id: Illustration ID (UUID string)
            checkpoint_id: Checkpoint ID (UUID string, optional)
            generation_order: Overall generation order
            chapter_order: Order within chapter
            scene_order: Order within scene
            web_url: Web URL for display
            prompt_used: Prompt used for generation

        Returns:
            Created SessionImage object
        """
        session_image = SessionImage(
            session_id=uuid.UUID(session_id),
            illustration_id=uuid.UUID(illustration_id),
            checkpoint_id=uuid.UUID(checkpoint_id) if checkpoint_id else None,
            generation_order=generation_order,
            chapter_order=chapter_order,
            scene_order=scene_order,
            web_url=web_url,
            prompt_used=prompt_used,
            generation_status="completed",
            completed_at=datetime.utcnow()
        )

        self.db.add(session_image)
        self.db.commit()
        self.db.refresh(session_image)

        # Update session state file with image info
        session_state = self._load_session_file(session_id)
        if session_state:
            image_info = {
                "illustration_id": illustration_id,
                "web_url": web_url,
                "prompt_used": prompt_used,
                "generation_order": generation_order,
                "chapter_order": chapter_order,
                "scene_order": scene_order,
                "created_at": datetime.utcnow().isoformat()
            }
            session_state.generated_images.append(image_info)
            session_state.total_images_generated = len(session_state.generated_images)
            self._save_session_file(session_state)

        return session_image

    def log_session_event(self,
                         session_id: str,
                         level: str,
                         message: str,
                         chapter_number: Optional[int] = None,
                         step_name: Optional[str] = None) -> ProcessingLog:
        """Log an event for a session.

        Args:
            session_id: Session ID (UUID string)
            level: Log level (info, warning, error, success)
            message: Log message
            chapter_number: Chapter number (optional)
            step_name: Step name (optional)

        Returns:
            Created ProcessingLog object
        """
        log_entry = ProcessingLog(
            session_id=uuid.UUID(session_id),
            level=level,
            message=message,
            chapter_number=chapter_number,
            step_name=step_name
        )

        self.db.add(log_entry)
        self.db.commit()
        self.db.refresh(log_entry)

        return log_entry

    def get_session_by_external_id(self, external_session_id: str) -> Optional[ProcessingSession]:
        """Get a session by its external session ID.

        Args:
            external_session_id: External session ID

        Returns:
            ProcessingSession object or None
        """
        return (
            self.db.query(ProcessingSession)
            .filter(ProcessingSession.external_session_id == external_session_id)
            .first()
        )

    def cleanup_old_sessions(self, days_old: int = 7) -> int:
        """Clean up old completed sessions and their files.

        Args:
            days_old: Delete sessions older than this many days

        Returns:
            Number of sessions cleaned up
        """
        from datetime import timedelta

        cutoff_date = datetime.utcnow() - timedelta(days=days_old)

        # Get old sessions
        old_sessions = (
            self.db.query(ProcessingSession)
            .filter(and_(
                ProcessingSession.status.in_(["completed", "failed", "cancelled"]),
                ProcessingSession.updated_at < cutoff_date
            ))
            .all()
        )

        cleaned_count = 0
        for session in old_sessions:
            # Delete related records
            self.db.query(SessionImage).filter(
                SessionImage.session_id == session.id
            ).delete()

            self.db.query(ProcessingCheckpoint).filter(
                ProcessingCheckpoint.session_id == session.id
            ).delete()

            self.db.query(ProcessingLog).filter(
                ProcessingLog.session_id == session.id
            ).delete()

            # Delete session files
            session_file = self.sessions_dir / f"{session.id}.json"
            if session_file.exists():
                session_file.unlink()

            # Delete checkpoint files
            for checkpoint_file in self.checkpoints_dir.glob(f"{session.id}_*.json"):
                checkpoint_file.unlink()

            # Delete session record
            self.db.delete(session)
            cleaned_count += 1

        self.db.commit()
        return cleaned_count

    def _save_session_file(self, session_state: SessionState) -> None:
        """Save session state to file.

        Args:
            session_state: Session state to save
        """
        session_file = self.sessions_dir / f"{session_state.session_id}.json"

        # Convert dataclass to dict
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
            "last_heartbeat": session_state.last_heartbeat
        }

        with open(session_file, 'w') as f:
            json.dump(session_dict, f, indent=2)

    def _load_session_file(self, session_id: str) -> Optional[SessionState]:
        """Load session state from file.

        Args:
            session_id: Session ID (UUID string)

        Returns:
            SessionState object or None
        """
        session_file = self.sessions_dir / f"{session_id}.json"

        if not session_file.exists():
            return None

        try:
            with open(session_file, 'r') as f:
                session_dict = json.load(f)

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
                last_heartbeat=session_dict["last_heartbeat"]
            )
        except (json.JSONDecodeError, KeyError) as e:
            # Log error but don't crash
            print(f"Error loading session file {session_file}: {e}")
            return None

    def close(self):
        """Close the database session."""
        if self.db:
            self.db.close()
