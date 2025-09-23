"""SQLAlchemy database models for manuscript illustrations."""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    Float,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class Manuscript(Base):
    """Manuscript table for storing manuscript metadata."""

    __tablename__ = "manuscripts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    author = Column(String(255))
    genre = Column(String(100))
    total_chapters = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Relationships
    chapters = relationship("Chapter", back_populates="manuscript")
    illustrations = relationship("Illustration", back_populates="manuscript")


class Chapter(Base):
    """Chapter table for storing chapter content."""

    __tablename__ = "chapters"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    manuscript_id = Column(UUID(as_uuid=True), ForeignKey("manuscripts.id"), nullable=False)
    title = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    number = Column(Integer, nullable=False)
    word_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Relationships
    manuscript = relationship("Manuscript", back_populates="chapters")
    illustrations = relationship("Illustration", back_populates="chapter")

    # Unique constraint to ensure no duplicate chapter numbers per manuscript
    __table_args__ = (UniqueConstraint('manuscript_id', 'number', name='unique_chapter_per_manuscript'),)


class Illustration(Base):
    """Illustration table for storing generated image metadata."""

    __tablename__ = "illustrations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    manuscript_id = Column(UUID(as_uuid=True), ForeignKey("manuscripts.id"), nullable=False)
    chapter_id = Column(UUID(as_uuid=True), ForeignKey("chapters.id"), nullable=False)

    # Image metadata
    filename = Column(String(255), nullable=False)
    file_path = Column(String(1000), nullable=False)
    web_url = Column(String(1000), nullable=False)  # URL accessible via web (e.g., /generated/filename)

    # Scene information
    scene_number = Column(Integer, nullable=False)
    title = Column(String(500))
    description = Column(Text)

    # Generation metadata
    prompt = Column(Text, nullable=False)
    negative_prompt = Column(Text)
    style_config = Column(Text)  # JSON string of style configuration
    image_provider = Column(String(50), nullable=False)  # dalle, imagen4, flux, etc.

    # Emotional context
    emotional_tones = Column(String(500))  # Comma-separated list of emotional tones
    intensity_score = Column(Float)
    text_excerpt = Column(Text)  # The text excerpt this illustration represents

    # File information
    file_size = Column(Integer)  # File size in bytes
    width = Column(Integer)
    height = Column(Integer)

    # Status and timestamps
    generation_status = Column(String(50), default="completed")  # pending, generating, completed, failed
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Relationships
    manuscript = relationship("Manuscript", back_populates="illustrations")
    chapter = relationship("Chapter", back_populates="illustrations")

    # Unique constraint to ensure no duplicate scene numbers per chapter
    __table_args__ = (UniqueConstraint('chapter_id', 'scene_number', name='unique_scene_per_chapter'),)


class ProcessingSession(Base):
    """Processing session table for tracking illustration generation sessions."""

    __tablename__ = "processing_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    manuscript_id = Column(UUID(as_uuid=True), ForeignKey("manuscripts.id"), nullable=False)
    external_session_id = Column(String(255))  # External session ID for web sessions

    # Session metadata
    session_type = Column(String(50), default="illustration_generation")  # illustration_generation, analysis, etc.
    status = Column(String(50), default="pending")  # pending, running, completed, failed, cancelled, paused, resuming
    progress_percent = Column(Integer, default=0)
    current_task = Column(String(500))
    current_chapter = Column(Integer)
    total_chapters = Column(Integer, default=0)

    # Configuration
    style_config = Column(Text)  # JSON string of style configuration used
    max_emotional_moments = Column(Integer, default=10)

    # Resume state
    last_completed_step = Column(String(50))  # analysis, generation, etc.
    last_completed_chapter = Column(Integer, default=0)
    last_checkpoint_id = Column(UUID(as_uuid=True), ForeignKey("processing_checkpoints.id"))
    resume_data = Column(Text)  # JSON string with comprehensive resume state
    can_resume = Column(Boolean, default=True)

    # Results
    total_images_generated = Column(Integer, default=0)
    total_chapters_processed = Column(Integer, default=0)
    error_message = Column(Text)

    # Timestamps
    started_at = Column(DateTime(timezone=True), default=func.now())
    completed_at = Column(DateTime(timezone=True))
    paused_at = Column(DateTime(timezone=True))
    resumed_at = Column(DateTime(timezone=True))
    last_heartbeat = Column(DateTime(timezone=True), default=func.now())
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Relationships
    manuscript = relationship("Manuscript")
    last_checkpoint = relationship("ProcessingCheckpoint", foreign_keys=[last_checkpoint_id])
    checkpoints = relationship("ProcessingCheckpoint", back_populates="session", order_by="ProcessingCheckpoint.created_at", foreign_keys="ProcessingCheckpoint.session_id")
    logs = relationship("ProcessingLog", back_populates="session", order_by="ProcessingLog.timestamp")
    session_images = relationship("SessionImage", back_populates="session")


class ProcessingCheckpoint(Base):
    """Processing checkpoints for resume capability."""

    __tablename__ = "processing_checkpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("processing_sessions.id"), nullable=False)

    # Checkpoint metadata
    checkpoint_type = Column(String(50), nullable=False)  # chapter_start, chapter_analyzed, chapter_generated, session_complete
    chapter_number = Column(Integer, nullable=False)
    step_name = Column(String(100), nullable=False)
    sequence_number = Column(Integer, nullable=False)  # Order of checkpoints

    # Checkpoint data
    checkpoint_data = Column(Text)  # JSON string with comprehensive checkpoint state
    images_generated_count = Column(Integer, default=0)
    total_images_at_checkpoint = Column(Integer, default=0)
    progress_percent = Column(Integer, default=0)

    # Processing state
    processing_state = Column(Text)  # JSON with detailed processing state
    generated_prompts = Column(Text)  # JSON array of prompts generated so far
    emotional_moments_data = Column(Text)  # JSON of analyzed emotional moments

    # Resume metadata
    is_resumable = Column(Boolean, default=True)
    next_action = Column(String(100))  # What to do next when resuming

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    session = relationship("ProcessingSession", back_populates="checkpoints")


class ProcessingLog(Base):
    """Processing logs for debugging and state tracking."""

    __tablename__ = "processing_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("processing_sessions.id"), nullable=False)

    # Log metadata
    level = Column(String(20), nullable=False)  # info, warning, error, success
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime(timezone=True), default=func.now())

    # Optional context
    chapter_number = Column(Integer)
    step_name = Column(String(100))

    # Relationships
    session = relationship("ProcessingSession", back_populates="logs")


class SessionImage(Base):
    """Track images generated during a session for restore capability."""

    __tablename__ = "session_images"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("processing_sessions.id"), nullable=False)
    illustration_id = Column(UUID(as_uuid=True), ForeignKey("illustrations.id"), nullable=False)
    checkpoint_id = Column(UUID(as_uuid=True), ForeignKey("processing_checkpoints.id"))  # Which checkpoint this image belongs to

    # Generation order for UI restoration
    generation_order = Column(Integer, nullable=False)
    chapter_order = Column(Integer, nullable=False)  # Order within chapter
    scene_order = Column(Integer, nullable=False)    # Order within scene

    # Metadata for restoration
    web_url = Column(String(1000))  # Web URL for immediate display
    prompt_used = Column(Text)      # Prompt used for generation
    generation_status = Column(String(50), default="completed")  # pending, generating, completed, failed

    # Timestamps
    created_at = Column(DateTime(timezone=True), default=func.now())
    completed_at = Column(DateTime(timezone=True))

    # Relationships
    session = relationship("ProcessingSession", back_populates="session_images")
    illustration = relationship("Illustration")
    checkpoint = relationship("ProcessingCheckpoint")