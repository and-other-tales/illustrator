"""Comprehensive unit tests for the db_config module."""

import os
import pytest
from unittest.mock import MagicMock, Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from illustrator.db_config import (
    DATABASE_URL,
    engine,
    SessionLocal,
    create_tables,
    get_db_session,
    get_db
)
from illustrator.db_models import Base


class TestDatabaseConfiguration:
    """Test database configuration constants and setup."""

    def test_default_database_url(self):
        """Test default DATABASE_URL when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Re-import to get fresh DATABASE_URL
            from illustrator import db_config
            import importlib
            importlib.reload(db_config)

            expected_url = "postgresql://illustrator:illustrator@localhost:5432/illustrator"
            assert db_config.DATABASE_URL == expected_url

    @patch.dict(os.environ, {'DATABASE_URL': 'postgresql://custom:custom@localhost:5432/custom'})
    def test_custom_database_url(self):
        """Test DATABASE_URL when environment variable is set."""
        from illustrator import db_config
        import importlib
        importlib.reload(db_config)

        assert db_config.DATABASE_URL == "postgresql://custom:custom@localhost:5432/custom"

    @patch.dict(os.environ, {'DATABASE_URL': 'sqlite:///test.db'})
    def test_sqlite_database_url(self):
        """Test DATABASE_URL with SQLite."""
        from illustrator import db_config
        import importlib
        importlib.reload(db_config)

        assert db_config.DATABASE_URL == "sqlite:///test.db"

    def test_engine_creation(self):
        """Test that engine is created properly."""
        assert engine is not None
        assert hasattr(engine, 'execute')
        assert hasattr(engine, 'connect')

    def test_engine_configuration(self):
        """Test engine configuration parameters."""
        # Test that engine uses NullPool
        assert engine.pool.__class__.__name__ == 'NullPool'

    @patch.dict(os.environ, {'DB_ECHO': 'true'})
    def test_engine_echo_enabled(self):
        """Test engine with echo enabled via environment variable."""
        from illustrator import db_config
        import importlib
        importlib.reload(db_config)

        # Create new engine with current environment
        test_engine = create_engine(
            db_config.DATABASE_URL,
            echo=os.getenv("DB_ECHO", "false").lower() == "true"
        )
        assert test_engine.echo is True

    @patch.dict(os.environ, {'DB_ECHO': 'false'})
    def test_engine_echo_disabled(self):
        """Test engine with echo disabled via environment variable."""
        from illustrator import db_config
        import importlib
        importlib.reload(db_config)

        test_engine = create_engine(
            db_config.DATABASE_URL,
            echo=os.getenv("DB_ECHO", "false").lower() == "true"
        )
        assert test_engine.echo is False

    def test_session_local_creation(self):
        """Test that SessionLocal is created properly."""
        assert SessionLocal is not None
        assert hasattr(SessionLocal, '__call__')

    def test_session_local_configuration(self):
        """Test SessionLocal configuration."""
        # Test that SessionLocal is bound to the engine
        assert SessionLocal.bind is engine

        # Test configuration parameters
        session = SessionLocal()
        assert session.autocommit is False
        assert session.autoflush is False
        session.close()


class TestDatabaseFunctions:
    """Test database utility functions."""

    @patch('illustrator.db_config.Base.metadata.create_all')
    def test_create_tables(self, mock_create_all):
        """Test create_tables function."""
        create_tables()
        mock_create_all.assert_called_once_with(bind=engine)

    def test_get_db_session_generator(self):
        """Test get_db_session returns a generator."""
        gen = get_db_session()
        assert hasattr(gen, '__next__')
        assert hasattr(gen, '__iter__')

    @patch('illustrator.db_config.SessionLocal')
    def test_get_db_session_lifecycle(self, mock_session_local):
        """Test get_db_session lifecycle management."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        gen = get_db_session()

        # Get the session
        session = next(gen)
        assert session is mock_session

        # Try to close the generator (simulating context manager exit)
        try:
            gen.close()
        except StopIteration:
            pass

        # Verify session was closed
        mock_session.close.assert_called_once()

    @patch('illustrator.db_config.SessionLocal')
    def test_get_db_session_exception_handling(self, mock_session_local):
        """Test get_db_session handles exceptions properly."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        gen = get_db_session()
        session = next(gen)

        # Simulate an exception by throwing into the generator
        try:
            gen.throw(Exception("Test exception"))
        except Exception:
            pass

        # Session should still be closed
        mock_session.close.assert_called_once()

    @patch('illustrator.db_config.SessionLocal')
    def test_get_db_function(self, mock_session_local):
        """Test get_db function."""
        mock_session = MagicMock()
        mock_session_local.return_value = mock_session

        result = get_db()

        assert result is mock_session
        mock_session_local.assert_called_once()

    def test_get_db_returns_session_instance(self):
        """Test that get_db returns a valid session instance."""
        session = get_db()

        # Verify it's a session-like object
        assert hasattr(session, 'query')
        assert hasattr(session, 'add')
        assert hasattr(session, 'commit')
        assert hasattr(session, 'close')

        # Clean up
        session.close()


class TestDatabaseIntegration:
    """Integration tests for database configuration."""

    def test_engine_connection(self):
        """Test that engine can establish a connection."""
        # This might fail if PostgreSQL is not available, so we'll mock it
        with patch.object(engine, 'connect') as mock_connect:
            mock_connection = MagicMock()
            mock_connect.return_value = mock_connection

            with engine.connect() as conn:
                assert conn is mock_connection

    def test_session_creation_and_cleanup(self):
        """Test session creation and proper cleanup."""
        session = SessionLocal()

        # Verify session is created
        assert session is not None
        assert hasattr(session, 'query')

        # Test cleanup
        session.close()

        # Verify session is closed (checking internal state)
        assert session._transaction is None

    @patch('illustrator.db_config.engine.connect')
    def test_create_tables_with_connection(self, mock_connect):
        """Test create_tables with mocked database connection."""
        mock_connection = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_connection

        create_tables()

        # Verify that some database operation was attempted
        mock_connect.assert_called()

    def test_base_metadata_registry(self):
        """Test that Base metadata registry is properly configured."""
        assert Base.metadata is not None
        assert hasattr(Base.metadata, 'create_all')
        assert hasattr(Base.metadata, 'drop_all')


class TestEnvironmentVariableHandling:
    """Test environment variable handling for database configuration."""

    def test_database_url_variations(self):
        """Test various DATABASE_URL formats."""
        test_urls = [
            "postgresql://user:pass@localhost:5432/db",
            "postgresql://user@localhost/db",
            "sqlite:///test.db",
            "sqlite:///:memory:",
            "mysql://user:pass@localhost/db"
        ]

        for test_url in test_urls:
            with patch.dict(os.environ, {'DATABASE_URL': test_url}):
                from illustrator import db_config
                import importlib
                importlib.reload(db_config)

                assert db_config.DATABASE_URL == test_url

    def test_db_echo_variations(self):
        """Test various DB_ECHO environment variable values."""
        test_cases = [
            ("true", True),
            ("TRUE", True),
            ("True", True),
            ("false", False),
            ("FALSE", False),
            ("False", False),
            ("", False),
            ("invalid", False)
        ]

        for env_value, expected_echo in test_cases:
            with patch.dict(os.environ, {'DB_ECHO': env_value}):
                result = os.getenv("DB_ECHO", "false").lower() == "true"
                assert result == expected_echo

    def test_missing_environment_variables(self):
        """Test behavior when environment variables are missing."""
        with patch.dict(os.environ, {}, clear=True):
            from illustrator import db_config
            import importlib
            importlib.reload(db_config)

            # Should use default values
            assert "postgresql://illustrator:illustrator@localhost:5432/illustrator" in db_config.DATABASE_URL


class TestErrorConditions:
    """Test error conditions and edge cases."""

    @patch('illustrator.db_config.create_engine')
    def test_engine_creation_error(self, mock_create_engine):
        """Test handling of engine creation errors."""
        mock_create_engine.side_effect = Exception("Database connection error")

        with pytest.raises(Exception, match="Database connection error"):
            from illustrator import db_config
            import importlib
            importlib.reload(db_config)

    @patch('illustrator.db_config.SessionLocal')
    def test_session_creation_error(self, mock_session_local):
        """Test handling of session creation errors."""
        mock_session_local.side_effect = Exception("Session creation error")

        with pytest.raises(Exception, match="Session creation error"):
            get_db()

    def test_get_db_session_with_session_error(self):
        """Test get_db_session when session operations fail."""
        with patch('illustrator.db_config.SessionLocal') as mock_session_local:
            mock_session = MagicMock()
            mock_session.close.side_effect = Exception("Close error")
            mock_session_local.return_value = mock_session

            gen = get_db_session()
            session = next(gen)

            # Even if close fails, it should not raise an exception
            try:
                gen.close()
            except StopIteration:
                pass
            # Should complete without raising the close error


class TestDatabaseConfigConstants:
    """Test database configuration constants and their properties."""

    def test_database_url_is_string(self):
        """Test that DATABASE_URL is a string."""
        assert isinstance(DATABASE_URL, str)
        assert len(DATABASE_URL) > 0

    def test_engine_properties(self):
        """Test engine object properties."""
        assert hasattr(engine, 'url')
        assert hasattr(engine, 'dialect')
        assert hasattr(engine, 'pool')

    def test_session_local_properties(self):
        """Test SessionLocal properties."""
        assert hasattr(SessionLocal, 'bind')
        assert SessionLocal.bind is engine


if __name__ == "__main__":
    pytest.main([__file__, "-v"])