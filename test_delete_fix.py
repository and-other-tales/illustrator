#!/usr/bin/env python3
"""Test script to verify the manuscript deletion functionality works correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from illustrator.web.routes.manuscripts import get_saved_manuscripts, invalidate_manuscripts_cache
import json
from pathlib import Path
from datetime import datetime
from illustrator.models import ManuscriptMetadata, SavedManuscript, Chapter

def test_cache_invalidation():
    """Test that cache invalidation works correctly."""
    print("Testing cache invalidation...")

    # Get initial count
    initial_manuscripts = get_saved_manuscripts()
    initial_count = len(initial_manuscripts)
    print(f"Initial manuscript count: {initial_count}")

    # Create a test manuscript file
    test_metadata = ManuscriptMetadata(
        title="Test Delete Manuscript",
        author="Test Author",
        genre="Test",
        total_chapters=1,
        created_at=datetime.now().isoformat()
    )

    test_chapter = Chapter(
        number=1,
        title="Test Chapter",
        content="This is test content.",
        word_count=4
    )

    test_manuscript = SavedManuscript(
        metadata=test_metadata,
        chapters=[test_chapter],
        saved_at=datetime.now().isoformat(),
        file_path=""
    )

    # Save test manuscript
    saved_manuscripts_dir = Path("saved_manuscripts")
    saved_manuscripts_dir.mkdir(exist_ok=True)
    test_file = saved_manuscripts_dir / f"Test_Delete_Manuscript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    test_manuscript.file_path = str(test_file)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_manuscript.model_dump(), f, indent=2, ensure_ascii=False)

    print(f"Created test manuscript: {test_file}")

    # Get count with cache
    manuscripts_with_new = get_saved_manuscripts()
    count_with_new = len(manuscripts_with_new)
    print(f"Count after adding manuscript (should be {initial_count + 1}): {count_with_new}")

    # Delete the test file
    test_file.unlink()
    print(f"Deleted test manuscript file: {test_file}")

    # Without cache invalidation, this would still show the old count
    manuscripts_without_invalidation = get_saved_manuscripts()
    count_without_invalidation = len(manuscripts_without_invalidation)
    print(f"Count without invalidation (might still show old count due to cache): {count_without_invalidation}")

    # Now invalidate cache
    invalidate_manuscripts_cache()
    print("Cache invalidated")

    # Get count after invalidation
    manuscripts_after_invalidation = get_saved_manuscripts()
    count_after_invalidation = len(manuscripts_after_invalidation)
    print(f"Count after cache invalidation (should be {initial_count}): {count_after_invalidation}")

    # Verify the fix works
    if count_after_invalidation == initial_count:
        print("✅ Cache invalidation test PASSED")
        return True
    else:
        print("❌ Cache invalidation test FAILED")
        return False

def test_manuscript_list():
    """Test that we can list manuscripts without errors."""
    print("\nTesting manuscript listing...")
    try:
        manuscripts = get_saved_manuscripts()
        print(f"Successfully loaded {len(manuscripts)} manuscripts")

        for i, manuscript in enumerate(manuscripts[:3]):  # Show first 3
            print(f"  {i+1}. {manuscript.metadata.title} by {manuscript.metadata.author}")

        print("✅ Manuscript listing test PASSED")
        return True
    except Exception as e:
        print(f"❌ Manuscript listing test FAILED: {e}")
        return False

if __name__ == "__main__":
    print("Testing manuscript delete fix...\n")

    test1_passed = test_manuscript_list()
    test2_passed = test_cache_invalidation()

    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED! The delete fix should work correctly.")
        sys.exit(0)
    else:
        print("\n💥 Some tests FAILED. Please check the implementation.")
        sys.exit(1)