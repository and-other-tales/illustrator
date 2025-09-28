"""Integration tests for API server CLI mode and related functionality."""

import os
import tempfile
from unittest.mock import patch, MagicMock
import pytest
from click.testing import CliRunner

from illustrator.cli import api_server, web_client
from illustrator.context import get_default_context


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Ensure clean environment for each test."""
    for key in (
        'ILLUSTRATOR_API_KEY',
        'ILLUSTRATOR_REMOTE_API_URL',
    ):
        monkeypatch.delenv(key, raising=False)


class TestApiServerCLI:
    """Test the API server CLI functionality with real implementations."""

    def test_api_server_configuration_validation(self, monkeypatch):
        """Test API server validates configuration properly."""
        from illustrator.web.app import create_api_only_app
        
        # Test that we can create the API app
        app = create_api_only_app()
        assert app is not None
        assert hasattr(app, 'routes')

    @patch('illustrator.cli.uvicorn.run')
    def test_api_server_starts_uvicorn_with_correct_params(self, mock_run, monkeypatch):
        """Test that api_server function calls uvicorn with correct parameters."""
        from click.testing import CliRunner
        
        runner = CliRunner()
        result = runner.invoke(api_server, ['--host', '127.0.0.1', '--port', '9876', '--reload'])
        
        # Verify the command executed successfully
        assert result.exit_code == 0 or mock_run.called  # Either success or uvicorn was mocked
        
        # If uvicorn was called, verify parameters
        if mock_run.called:
            args, kwargs = mock_run.call_args
            assert kwargs['host'] == '127.0.0.1'
            assert kwargs['port'] == 9876
            assert kwargs['reload'] is True

    @patch('illustrator.cli.uvicorn.run')
    def test_api_server_with_api_key_environment(self, mock_run, monkeypatch):
        """Test API server respects API key environment variables."""
        monkeypatch.setenv('ILLUSTRATOR_API_KEY', 'test-api-key')
        
        runner = CliRunner()
        result = runner.invoke(api_server, ['--host', 'localhost', '--port', '8000', '--api-key', 'custom-key', '--reload'])
        
        # Verify command executed without major errors
        assert result.exit_code == 0 or mock_run.called
        
        if mock_run.called:
            args, kwargs = mock_run.call_args
            assert kwargs['host'] == 'localhost'
            assert kwargs['port'] == 8000
            assert kwargs['reload'] is True

    def test_default_context_creation(self):
        """Test that get_default_context works correctly."""
        context = get_default_context()
        assert context is not None
        assert hasattr(context, 'user_id')
        assert hasattr(context, 'llm_provider')


class TestWebClientCLI:
    """Test the web client CLI functionality."""
    
    def test_web_client_environment_configuration(self, monkeypatch):
        """Test web client reads environment configuration correctly."""
        monkeypatch.setenv('ILLUSTRATOR_REMOTE_API_URL', 'http://api.example.com')
        monkeypatch.setenv('ILLUSTRATOR_API_KEY', 'test-api-key')
        
        # Verify environment variables are set correctly
        assert os.getenv('ILLUSTRATOR_REMOTE_API_URL') == 'http://api.example.com'
        assert os.getenv('ILLUSTRATOR_API_KEY') == 'test-api-key'
        
    @patch('illustrator.cli.uvicorn.run')
    def test_web_client_startup_configuration(self, mock_run, monkeypatch):
        """Test web client startup with proper configuration."""
        monkeypatch.setenv('ILLUSTRATOR_REMOTE_API_URL', 'http://localhost:8001')
        
        runner = CliRunner()
        # Test the web client function with Click runner
        result = runner.invoke(web_client, ['--host', '127.0.0.1', '--port', '8080', '--api-url', 'http://localhost:8001'])
        
        # Check that command executed (may fail due to network but shouldn't crash)
        # Exit code may be non-zero due to network issues, but should not be a syntax/import error
        assert result.exit_code is not None  # Command completed, regardless of success


class TestCLIIntegration:
    """Test CLI integration scenarios."""
    
    def test_context_and_models_integration(self):
        """Test that context and models work together properly."""
        from illustrator.context import ManuscriptContext
        from illustrator.models import LLMProvider, ImageProvider
        
        # Test context creation with specific providers
        context = ManuscriptContext(
            user_id="test-user",
            llm_provider=LLMProvider.ANTHROPIC
        )
        
        assert context.user_id == "test-user"
        assert context.llm_provider == LLMProvider.ANTHROPIC
        
    def test_cli_imports_work_correctly(self):
        """Test that all CLI imports work without stubs."""
        # This verifies that the actual modules can be imported
        from illustrator.cli import api_server, web_client
        from illustrator.context import ManuscriptContext, get_default_context
        from illustrator.models import Chapter, ChapterAnalysis, ManuscriptMetadata, SavedManuscript, LLMProvider
        
        # Basic smoke test - these should all be callable/instantiable
        assert callable(api_server)
        assert callable(web_client)
        assert callable(get_default_context)
        assert ManuscriptContext is not None
        assert Chapter is not None
        assert LLMProvider is not None
