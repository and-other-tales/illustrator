"""MongoDB configuration and connection management for Illustrator."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from pymongo import ASCENDING, MongoClient
from pymongo.errors import ServerSelectionTimeoutError
from pymongo.collection import Collection
from pymongo.database import Database

DEFAULT_MONGO_URL = "mongodb://localhost:27017"
DEFAULT_DB_NAME = "illustrator"

_env_mongo_uri = os.getenv("MONGODB_URI")
MONGO_URL = _env_mongo_uri or os.getenv("MONGO_URL", DEFAULT_MONGO_URL)
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", DEFAULT_DB_NAME)

_client: MongoClient | None = None

USE_MOCK = os.getenv("MONGO_USE_MOCK", "false").lower() in {"1", "true", "yes"}


def _build_mock_client() -> MongoClient:
    """Return a mongomock client when running without a real Mongo server."""

    try:
        import mongomock
    except ImportError as exc:  # noqa: F401
        raise RuntimeError(
            "mongomock is required for in-memory MongoDB emulation"
        ) from exc

    return mongomock.MongoClient()


def _initialise_client() -> MongoClient:
    """Create (or reuse) the shared Mongo client."""

    global _client
    if _client is not None:
        return _client

    if USE_MOCK:
        _client = _build_mock_client()
        return _client

    client = MongoClient(
        MONGO_URL,
        appname="illustrator",
        serverSelectionTimeoutMS=3000,
    )

    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        if os.getenv("PYTEST_CURRENT_TEST"):
            client.close()
            _client = _build_mock_client()
            return _client
        raise

    _client = client
    return _client


def get_client() -> MongoClient:
    """Return the shared Mongo client."""

    return _initialise_client()


def get_database() -> Database:
    """Return the configured Mongo database."""

    return get_client()[MONGO_DB_NAME]


def get_collection(name: str) -> Collection:
    """Return a specific collection from the configured database."""

    return get_database()[name]


@contextmanager
def get_db_session() -> Generator[Database, None, None]:
    """Yield the Mongo database to mirror the old SQLAlchemy session helper."""

    db = get_database()
    try:
        yield db
    finally:
        # Mongo connections are pooled; no teardown is required.
        pass


def get_db() -> Database:
    """Backward-compatible helper returning the Mongo database instance."""

    return get_database()


def _ensure_indexes(db: Database) -> None:
    """Create the indexes required for Illustrator collections."""

    manuscripts = db["manuscripts"]

    chapters = db["chapters"]
    chapters.create_index([("manuscript_id", ASCENDING), ("number", ASCENDING)], unique=True)

    illustrations = db["illustrations"]
    illustrations.create_index([("manuscript_id", ASCENDING), ("chapter_id", ASCENDING), ("scene_number", ASCENDING)], unique=True)
    illustrations.create_index("chapter_id")

    sessions = db["processing_sessions"]
    sessions.create_index("external_session_id", unique=True, sparse=True)
    sessions.create_index("updated_at")

    checkpoints = db["processing_checkpoints"]
    checkpoints.create_index([("session_id", ASCENDING), ("sequence_number", ASCENDING)], unique=True)

    session_images = db["session_images"]
    session_images.create_index([("session_id", ASCENDING), ("generation_order", ASCENDING)])

    logs = db["processing_logs"]
    logs.create_index("session_id")
    logs.create_index("timestamp")


def create_tables() -> None:
    """Maintain compatibility with the former SQL setup by creating indexes."""

    _ensure_indexes(get_database())


def close_client() -> None:
    """Close the shared Mongo client (primarily for tests)."""

    global _client
    if _client is not None:
        _client.close()
        _client = None
