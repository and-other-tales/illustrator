"""Test script to validate manuscript loading."""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.illustrator.models import SavedManuscript
    from src.illustrator.utils.validation_helpers import ensure_chapter_required_fields
    
    # Create a test manuscript with null IDs
    test_data = {
        "metadata": {
            "title": "Test Novel",
            "author": "Test Author",
            "total_chapters": 2,
            "created_at": "2025-09-28T12:00:00"
        },
        "chapters": [
            {
                "title": "Chapter 1",
                "content": "This is chapter 1",
                "number": 1,
                "id": None,
                "summary": None,
                "word_count": 100
            },
            {
                "title": "Chapter 2",
                "content": "This is chapter 2",
                "number": 2,
                "id": None,
                "summary": None,
                "word_count": 120
            }
        ],
        "saved_at": "2025-09-28T12:00:00",
        "file_path": "test_path.json"
    }
    
    # Apply our fix
    fixed_data = ensure_chapter_required_fields(test_data)
    
    # Try to create a SavedManuscript instance
    manuscript = SavedManuscript(**fixed_data)
    
    print("✅ Successfully created manuscript with fixed data")
    print(f"Chapter 1 ID: {manuscript.chapters[0].id}")
    print(f"Chapter 1 Summary: {manuscript.chapters[0].summary}")
    print(f"Chapter 2 ID: {manuscript.chapters[1].id}")
    print(f"Chapter 2 Summary: {manuscript.chapters[1].summary}")
    
except Exception as e:
    print(f"❌ Error: {e}")