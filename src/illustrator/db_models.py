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

    # Session metadata
    session_type = Column(String(50), default="illustration_generation")  # illustration_generation, analysis, etc.
    status = Column(String(50), default="pending")  # pending, running, completed, failed, cancelled
    progress_percent = Column(Integer, default=0)
    current_task = Column(String(500))

    # Configuration
    style_config = Column(Text)  # JSON string of style configuration used
    max_emotional_moments = Column(Integer, default=10)

    # Results
    total_images_generated = Column(Integer, default=0)
    total_chapters_processed = Column(Integer, default=0)
    error_message = Column(Text)

    # Timestamps
    started_at = Column(DateTime(timezone=True), default=func.now())
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), default=func.now())
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

    # Relationships
    manuscript = relationship("Manuscript")