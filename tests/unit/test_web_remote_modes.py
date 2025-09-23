"""Unit tests for API-only and Web Client app factories and WebSocket proxying."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Ensure clean env for each test
    for key in (
        'ILLUSTRATOR_API_KEY',
        'ILLUSTRATOR_REMOTE_API_URL',
    ):
        monkeypatch.delenv(key, raising=False)


def test_create_api_only_app_routes_and_health(monkeypatch):
    from illustrator.web.app import create_api_only_app

    app = create_api_only_app()
    client = TestClient(app)

    # Health reports mode api-only
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert data['service'] == 'manuscript-illustrator-api'
    assert data['mode'] == 'api-only'

    # API routes are present (basic smoke)
    routes = [r.path for r in app.routes]
    assert any(p.startswith('/api/') for p in routes)
    # No HTML root route in API-only app
    assert '/' not in routes


def test_api_only_app_authentication_required(monkeypatch):
    from illustrator.web.app import create_api_only_app
    from fastapi.testclient import TestClient

    # Enable API key protection
    monkeypatch.setenv('ILLUSTRATOR_API_KEY', 'secret123')
    app = create_api_only_app()
    client = TestClient(app)

    # Health endpoint is public
    assert client.get('/health').status_code == 200

    # API endpoints without key should be unauthorized
    r = client.get('/api/manuscripts')
    assert r.status_code == 401

    # With correct key should pass auth check (endpoint may still 200/empty depending on state)
    r2 = client.get('/api/manuscripts', headers={'X-API-Key': 'secret123'})
    assert r2.status_code in (200, 204, 404)


def test_create_web_client_app_routes_and_health(monkeypatch):
    # Configure remote URL and API key
    monkeypatch.setenv('ILLUSTRATOR_REMOTE_API_URL', 'http://example.com')
    monkeypatch.setenv('ILLUSTRATOR_API_KEY', 'secret123')

    from illustrator.web.app import create_web_client_app
    app = create_web_client_app()
    client = TestClient(app)

    # HTML routes exist
    assert client.get('/').status_code == 200
    assert client.get('/manuscript/new').status_code == 200

    # Health endpoint returns web-client mode and echoes remote API URL
    resp = client.get('/health')
    assert resp.status_code == 200
    data = resp.json()
    assert data['service'] == 'manuscript-illustrator-web-client'
    assert data['mode'] == 'web-client'
    assert data['remote_api_url'] == 'http://example.com'


@pytest.mark.asyncio
async def test_web_client_websocket_proxy_bridges_messages(monkeypatch):
    """The web client WS endpoint should connect to remote and forward messages both ways."""
    monkeypatch.setenv('ILLUSTRATOR_REMOTE_API_URL', 'http://remote-server:8000')

    from illustrator.web.app import create_web_client_app

    # Build a fake websockets connection context manager
    class FakeRemoteWS:
        def __init__(self):
            self.sent = []

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def send(self, message: str):
            self.sent.append(message)

        def __aiter__(self):
            async def gen():
                # Simulate a single message from remote then end
                yield 'remote:hello'
            return gen()

    fake_remote = FakeRemoteWS()

    async def fake_connect(url: str):  # returns async context manager
        return fake_remote

    with patch('src.illustrator.web.app.websockets.connect', new=fake_connect):
        app = create_web_client_app()
        client = TestClient(app)

        with client.websocket_connect('/ws/processing/test-session') as ws:
            # Send a message to be forwarded to remote
            ws.send_text('client:ping')
            # Receive the forwarded message from remote
            received = ws.receive_text()
            assert received == 'remote:hello'

    # Ensure message from client was sent to remote
    assert 'client:ping' in fake_remote.sent

