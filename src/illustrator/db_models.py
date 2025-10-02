"""Document collection names and helpers for MongoDB persistence."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

MANUSCRIPTS_COLLECTION = "manuscripts"
CHAPTERS_COLLECTION = "chapters"
ILLUSTRATIONS_COLLECTION = "illustrations"
PROCESSING_SESSIONS_COLLECTION = "processing_sessions"
PROCESSING_CHECKPOINTS_COLLECTION = "processing_checkpoints"
PROCESSING_LOGS_COLLECTION = "processing_logs"
SESSION_IMAGES_COLLECTION = "session_images"


def now_utc() -> datetime:
    """Return a UTC timestamp helper."""

    return datetime.utcnow()


def normalise_id(document: Dict[str, Any]) -> str:
    """Return the string identifier for a Mongo document."""

    return str(document.get("_id") or document.get("id"))
