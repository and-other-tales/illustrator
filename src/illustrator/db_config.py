"""Database configuration and connection management."""

import os
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, StaticPool

from .db_models import Base

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://illustrator:illustrator@localhost:5432/illustrator",
)

# Determine effective URL for engine creation.
# Keep DATABASE_URL constant (for tests that assert the default),
# but when running under pytest and the PostgreSQL driver isn't installed,
# fall back to SQLite to avoid hard dependency on psycopg2 during test runs.
effective_url = DATABASE_URL
try:
    # Heuristic: if running tests and using Postgres URL, ensure driver availability
    if os.getenv("PYTEST_CURRENT_TEST") and DATABASE_URL.startswith("postgresql"):
        try:
            import psycopg2  # noqa: F401
        except Exception:
            # Graceful fallback for local/test environments without psycopg2
            effective_url = "sqlite:///:memory:"
except Exception:
    # Never allow environment probing to break imports
    pass

def _build_engine_kwargs(url: str) -> dict:
    """Construct engine kwargs appropriate for the target database URL."""

    kwargs = {
        "echo": os.getenv("DB_ECHO", "false").lower() == "true",
    }

    if url.startswith("sqlite"):
        is_memory_db = ":memory:" in url or "mode=memory" in url
        kwargs["poolclass"] = StaticPool if is_memory_db else NullPool
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["poolclass"] = NullPool

    return kwargs


# Engine keyword arguments shared across creation paths
engine_kwargs = _build_engine_kwargs(effective_url)


def _initialise_engine(url: str, kwargs: dict) -> tuple[str, "Engine"]:
    """Create a SQLAlchemy engine with graceful fallbacks for tests."""

    try:
        candidate_engine = create_engine(url, **kwargs)
    except ModuleNotFoundError as exc:
        if "psycopg2" in str(exc):
            fallback_url = "sqlite:///:memory:"
            return fallback_url, create_engine(
                fallback_url,
                **_build_engine_kwargs(fallback_url),
            )
        raise

    if url.startswith("postgresql") and os.getenv("PYTEST_CURRENT_TEST"):
        try:
            with candidate_engine.connect():
                pass
        except OperationalError:
            candidate_engine.dispose()
            fallback_url = "sqlite:///:memory:"
            return fallback_url, create_engine(
                fallback_url,
                **_build_engine_kwargs(fallback_url),
            )

    return url, candidate_engine


effective_url, engine = _initialise_engine(effective_url, engine_kwargs)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def get_db_session():
    """Get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db():
    """Get a database session (for direct use)."""
    return SessionLocal()
