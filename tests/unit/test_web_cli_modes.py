"""Unit tests for CLI commands introducing API-only and Web Client modes.

These tests avoid importing heavy web dependencies by patching the factories
and the uvicorn runner inside the CLI command implementations.
"""

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in (
        'ILLUSTRATOR_API_KEY',
        'ILLUSTRATOR_REMOTE_API_URL',
    ):
        monkeypatch.delenv(key, raising=False)


def _install_cli_import_stubs():
    """Install lightweight stubs to satisfy illustrator.cli imports."""
    # Stub illustrator.context
    ctx = types.ModuleType('illustrator.context')
    class ManuscriptContext:  # minimal placeholder
        pass
    ctx.ManuscriptContext = ManuscriptContext
    sys.modules['illustrator.context'] = ctx

    # Stub illustrator.models
    models = types.ModuleType('illustrator.models')
    class Chapter: ...
    class ChapterAnalysis: ...
    class ManuscriptMetadata: ...
    class SavedManuscript: ...
    models.Chapter = Chapter
    models.ChapterAnalysis = ChapterAnalysis
    models.ManuscriptMetadata = ManuscriptMetadata
    models.SavedManuscript = SavedManuscript
    sys.modules['illustrator.models'] = models


def test_cli_api_server_starts_uvicorn(monkeypatch):
    # Patch the app factory imported in the function
    fake_app = object()

    def fake_create_api_only_app():
        return fake_app

    # Ensure uvicorn.run is called with the app and params
    _install_cli_import_stubs()
    # Provide a fake illustrator.web.app module so the in-function import uses it
    web_app_mod = types.ModuleType('illustrator.web.app')
    web_app_mod.create_api_only_app = fake_create_api_only_app
    sys.modules['illustrator.web.app'] = web_app_mod

    import importlib
    cli_mod = importlib.import_module('illustrator.cli')
    with patch.object(cli_mod.uvicorn, 'run') as mock_run:
        api_server = cli_mod.api_server

        api_server(host='127.0.0.1', port=9876, api_key=None, reload=False)

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        # First positional argument is the app instance
        assert args[0] is fake_app
        assert kwargs['host'] == '127.0.0.1'
        assert kwargs['port'] == 9876
        assert kwargs['reload'] is False


def test_cli_api_server_sets_api_key(monkeypatch):
    fake_app = object()

    def fake_create_api_only_app():
        return fake_app

    _install_cli_import_stubs()
    web_app_mod = types.ModuleType('illustrator.web.app')
    web_app_mod.create_api_only_app = fake_create_api_only_app
    sys.modules['illustrator.web.app'] = web_app_mod

    import importlib
    cli_mod = importlib.import_module('illustrator.cli')
    with patch.object(cli_mod.uvicorn, 'run') as mock_run:
        api_server = cli_mod.api_server

        api_server(host='0.0.0.0', port=8001, api_key='secret123', reload=True)

        # Environment variable should be set for downstream app
        assert os.environ.get('ILLUSTRATOR_API_KEY') == 'secret123'
        mock_run.assert_called_once()


def test_cli_web_client_starts_with_remote_settings(monkeypatch):
    fake_app = object()

    def fake_create_web_client_app():
        return fake_app

    class DummyResponse:
        status_code = 200

    _install_cli_import_stubs()
    web_app_mod = types.ModuleType('illustrator.web.app')
    web_app_mod.create_web_client_app = fake_create_web_client_app
    sys.modules['illustrator.web.app'] = web_app_mod

    import importlib
    cli_mod = importlib.import_module('illustrator.cli')
    with patch.object(cli_mod.uvicorn, 'run') as mock_run, \
         patch.object(cli_mod, 'requests') as mock_requests:
        mock_requests.get.return_value = DummyResponse()
        web_client = cli_mod.web_client

        web_client(
            server_url='http://remote:8000',
            api_key='secret123',
            host='127.0.0.1',
            port=3001,
            open_browser=False,
        )

        # Environment variables are exported for the web client factory
        assert os.environ.get('ILLUSTRATOR_REMOTE_API_URL') == 'http://remote:8000'
        assert os.environ.get('ILLUSTRATOR_API_KEY') == 'secret123'

        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert args[0] is fake_app
        assert kwargs['host'] == '127.0.0.1'
        assert kwargs['port'] == 3001
