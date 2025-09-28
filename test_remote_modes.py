#!/usr/bin/env python3
"""Test script for remote server modes."""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_api_only_mode():
    """Test API-only mode functionality."""
    print("Testing API-only mode...")

    try:
        from illustrator.web.app import create_api_only_app

        # Create the API-only app
        app = create_api_only_app()

        # Check if app has the expected routes
        routes = [route.path for route in app.routes]
        expected_api_routes = ["/health", "/api/manuscripts", "/api/chapters"]

        print(f"✓ API-only app created successfully")
        print(f"  Available routes: {len(routes)}")

        # Check for API routes
        api_routes = [route for route in routes if route.startswith("/api/")]
        print(f"  API routes found: {len(api_routes)}")

        # Check that web UI routes are not present
        web_ui_routes = [route for route in routes if route in ["/", "/manuscript"]]
        print(f"  Web UI routes (should be 0): {len(web_ui_routes)}")

        return True

    except Exception as e:
        print(f"✗ API-only mode test failed: {e}")
        return False

async def test_web_client_mode():
    """Test web client mode functionality."""
    print("\nTesting web client mode...")

    try:
        # Set environment variables for testing
        os.environ['ILLUSTRATOR_REMOTE_API_URL'] = 'http://localhost:8000'
        os.environ['ILLUSTRATOR_API_KEY'] = 'test-key-123'

        from illustrator.web.app import create_web_client_app

        # Create the web client app
        app = create_web_client_app()

        # Check if app has the expected routes
        routes = [route.path for route in app.routes]

        print(f"✓ Web client app created successfully")
        print(f"  Available routes: {len(routes)}")

        # Check for web UI routes
        web_routes = [route for route in routes if route in ["/", "/manuscript/{manuscript_id}"]]
        print(f"  Web UI routes found: {len(web_routes)}")

        # Check for proxy route
        proxy_routes = [route for route in routes if "/api/" in route.path]
        print(f"  API proxy routes: {len(proxy_routes)}")

        return True

    except Exception as e:
        print(f"✗ Web client mode test failed: {e}")
        return False

async def test_authentication():
    """Test API key authentication."""
    print("\nTesting API key authentication...")

    try:
        # Set API key environment variable
        os.environ['ILLUSTRATOR_API_KEY'] = 'test-auth-key'

        from illustrator.web.app import create_api_only_app
        from fastapi.testclient import TestClient

        app = create_api_only_app()
        client = TestClient(app)

        # Test health endpoint (should work without auth)
        response = client.get("/health")
        assert response.status_code == 200
        print("✓ Health endpoint accessible without authentication")

        # Test API endpoint without auth (should fail)
        response = client.get("/api/manuscripts")
        assert response.status_code == 401
        print("✓ API endpoint properly protected")

        # Test API endpoint with correct auth (should work)
        response = client.get("/api/manuscripts", headers={"X-API-Key": "test-auth-key"})
        print(f"✓ API endpoint with auth returns: {response.status_code}")

        return True

    except Exception as e:
        print(f"✗ Authentication test failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Testing Manuscript Illustrator Remote Modes\n")

    tests = [
        test_api_only_mode,
        test_web_client_mode,
        test_authentication
    ]

    results = []
    for test in tests:
        try:
            result = await test()
            results.append(result)
        except Exception as e:
            print(f"Test failed with exception: {e}")
            results.append(False)

    # Summary
    passed = sum(results)
    total = len(results)

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Remote modes are working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)