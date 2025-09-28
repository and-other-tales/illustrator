"""Test script to verify we can import parse_llm_json."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.illustrator.utils import parse_llm_json
    print("✅ Successfully imported parse_llm_json from illustrator.utils")
except ImportError as e:
    print(f"❌ Import error: {e}")