"""Test script to verify we can import illustrator.web."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # First import the base module
    import src.illustrator.web
    print("✅ Successfully imported illustrator.web")
    
    # Now try importing specific submodules
    from src.illustrator.web import routes
    print("✅ Successfully imported illustrator.web.routes")
    
    from src.illustrator.web import models
    print("✅ Successfully imported illustrator.web.models")
    
    from src.illustrator.web.routes import manuscripts, chapters
    print("✅ Successfully imported route modules")
    
except ImportError as e:
    print(f"❌ Import error: {e}")