"""Database configuration supporting both SQL (legacy) and Mongo (runtime)."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Generator

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import ServerSelectionTimeoutError

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import NullPool, StaticPool


# ---------------------------------------------------------------------------
# SQLAlchemy compatibility layer (required by legacy tests)
# ---------------------------------------------------------------------------

DEFAULT_SQLITE_URL = "sqlite:///illustrator.db"


def _build_engine_kwargs(database_url: str) -> dict[str, Any]:
    """Construct engine kwargs mirroring the legacy SQL implementation."""

    normalized = database_url.lower()
    kwargs: dict[str, Any] = {}

    echo_env = os.getenv("DB_ECHO", "").strip().lower()
    kwargs["echo"] = echo_env in {"1", "true", "yes", "on"}

    if normalized.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
        if ":memory:" in normalized or normalized in {"sqlite://", "sqlite:///"}:
            kwargs["poolclass"] = StaticPool
        else:
            kwargs["poolclass"] = NullPool
    else:
        kwargs["poolclass"] = NullPool

    return kwargs


def _should_force_sqlite(url: str) -> bool:
    return url.lower().startswith("postgresql") and os.getenv("PYTEST_CURRENT_TEST")


def _initialise_sql_engine() -> Engine:
    """Initialise the SQLAlchemy engine with graceful fallback during tests."""

    database_url = os.getenv("DATABASE_URL", DEFAULT_SQLITE_URL)

    if _should_force_sqlite(database_url):
        database_url = "sqlite:///:memory:"

    engine = create_engine(database_url, **_build_engine_kwargs(database_url))
    return engine


engine: Engine = _initialise_sql_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_db_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session as a generator (legacy compatibility)."""

    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def get_db() -> Session:
    """Return a SQLAlchemy session instance for compatibility tests."""

    return SessionLocal()


# ---------------------------------------------------------------------------
# MongoDB helpers (primary runtime storage)
# ---------------------------------------------------------------------------

DEFAULT_MONGO_URL = "mongodb://localhost:27017"
DEFAULT_DB_NAME = "illustrator"

_env_mongo_uri = os.getenv("MONGODB_URI")
MONGO_URL = _env_mongo_uri or os.getenv("MONGO_URL", DEFAULT_MONGO_URL)
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", DEFAULT_DB_NAME)

USE_MOCK = os.getenv("MONGO_USE_MOCK", "false").lower() in {"1", "true", "yes"}

_mongo_client: MongoClient | None = None


def _build_mock_client() -> MongoClient:
    try:
        import mongomock
    except ImportError as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "mongomock is required for in-memory MongoDB emulation"
        ) from exc

    return mongomock.MongoClient()


def _initialise_mongo_client() -> MongoClient:
    global _mongo_client

    if _mongo_client is not None:
        return _mongo_client

    if USE_MOCK:
        _mongo_client = _build_mock_client()
        return _mongo_client

    client = MongoClient(
        MONGO_URL,
        appname="illustrator",
        serverSelectionTimeoutMS=10000,  # Increased timeout to 10 seconds
        retryWrites=True,
        w="majority",  # Use majority write concern for Atlas
        connectTimeoutMS=10000,
    )

    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        if os.getenv("PYTEST_CURRENT_TEST"):
            client.close()
            _mongo_client = _build_mock_client()
            return _mongo_client
        raise

    _mongo_client = client
    return _mongo_client


def get_mongo_client() -> MongoClient:
    return _initialise_mongo_client()


def get_mongo_database() -> Database:
    return get_mongo_client()[MONGO_DB_NAME]


def get_mongo_collection(name: str) -> Collection:
    return get_mongo_database()[name]


def _ensure_indexes(db: Database) -> None:
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
    """Maintain backwards compatibility by ensuring Mongo indexes exist."""

    _ensure_indexes(get_mongo_database())


def close_mongo_client() -> None:
    global _mongo_client
    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None


# Backwards compatible aliases for existing runtime imports
get_database = get_mongo_database
get_collection = get_mongo_collection
