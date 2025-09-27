"""Test the application startup."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import the main app module to see if it works
    from src.illustrator.web.app import app
    print("✅ Application imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")