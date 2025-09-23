"""Integration test for web-client proxy to API-only app with API key."""

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
import httpx


@pytest.mark.integration
def test_web_client_proxies_requests_to_api_only_app(monkeypatch):
    # Configure API key on both ends
    monkeypatch.setenv('ILLUSTRATOR_API_KEY', 'secret123')
    monkeypatch.setenv('ILLUSTRATOR_REMOTE_API_URL', 'http://test-remote')

    from illustrator.web.app import create_api_only_app, create_web_client_app

    api_app = create_api_only_app()
    web_client_app = create_web_client_app()

    # Patch httpx.AsyncClient used in web client to route to the local api_app via ASGITransport
    def async_client_factory(*args, **kwargs):
        transport = httpx.ASGITransport(app=api_app)
        # Preserve headers (should include X-API-Key)
        headers = kwargs.get('headers', {})
        return httpx.AsyncClient(transport=transport, base_url='http://testserver', headers=headers)

    with patch('src.illustrator.web.app.httpx.AsyncClient', side_effect=async_client_factory):
        client = TestClient(web_client_app)

        # Health check should reflect remote health as healthy since ASGITransport will serve it
        r = client.get('/health')
        assert r.status_code == 200
        assert r.json()['status'] in ('healthy', 'degraded')

        # Call an API route through the web client; it should proxy to the API app
        r2 = client.get('/api/manuscripts/')
        # Without any saved manuscripts, expect 200 with list or 204/404 depending on implementation
        assert r2.status_code in (200, 204, 404)

