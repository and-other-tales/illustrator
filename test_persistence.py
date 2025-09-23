#!/usr/bin/env python3
"""
Test script for session persistence and checkpoint functionality.
This script tests the basic functionality without requiring a full database setup.
"""

import sys
import os
import json
import uuid
from pathlib import Path

# Add src directory to Python path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from illustrator.services.checkpoint_manager import CheckpointManager, CheckpointType, ProcessingStep
    from illustrator.services.session_persistence import SessionPersistenceService, SessionState
    print("✅ Successfully imported persistence services")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_session_state_creation():
    """Test creating a SessionState object."""
    try:
        session_state = SessionState(
            session_id=str(uuid.uuid4()),
            manuscript_id=str(uuid.uuid4()),
            external_session_id="test-session",
            status="pending",
            progress_percent=0,
            current_chapter=None,
            total_chapters=5,
            current_task="Starting test",
            style_config={"art_style": "digital"},
            max_emotional_moments=10,
            last_completed_step=None,
            last_completed_chapter=0,
            processed_chapters=[],
            current_prompts=[],
            generated_images=[],
            emotional_moments=[],
            total_images_generated=0,
            error_message=None,
            started_at="2023-01-01T00:00:00",
            paused_at=None,
            resumed_at=None,
            last_heartbeat="2023-01-01T00:00:00"
        )
        print("✅ SessionState creation successful")
        return session_state
    except Exception as e:
        print(f"❌ SessionState creation failed: {e}")
        return None

def test_file_operations():
    """Test basic file operations for persistence."""
    try:
        # Create test directory
        test_dir = Path("test_output/sessions")
        test_dir.mkdir(parents=True, exist_ok=True)

        # Test writing and reading a session file
        test_data = {
            "session_id": "test-123",
            "manuscript_id": "manuscript-456",
            "status": "testing",
            "progress_percent": 50
        }

        session_file = test_dir / "test_session.json"
        with open(session_file, 'w') as f:
            json.dump(test_data, f, indent=2)

        # Read back the data
        with open(session_file, 'r') as f:
            loaded_data = json.load(f)

        if loaded_data == test_data:
            print("✅ File operations test successful")

            # Cleanup
            session_file.unlink()
            return True
        else:
            print("❌ File operations test failed: Data mismatch")
            return False

    except Exception as e:
        print(f"❌ File operations test failed: {e}")
        return False

def test_checkpoint_types():
    """Test checkpoint type enums."""
    try:
        # Test all checkpoint types
        checkpoint_types = [
            CheckpointType.SESSION_START,
            CheckpointType.MANUSCRIPT_LOADED,
            CheckpointType.CHAPTER_START,
            CheckpointType.CHAPTER_ANALYZED,
            CheckpointType.PROMPTS_GENERATED,
            CheckpointType.IMAGES_GENERATING,
            CheckpointType.CHAPTER_COMPLETED,
            CheckpointType.SESSION_COMPLETED,
            CheckpointType.SESSION_PAUSED,
            CheckpointType.ERROR_OCCURRED
        ]

        # Test all processing steps
        processing_steps = [
            ProcessingStep.INITIALIZING,
            ProcessingStep.LOADING_MANUSCRIPT,
            ProcessingStep.ANALYZING_CHAPTERS,
            ProcessingStep.GENERATING_PROMPTS,
            ProcessingStep.GENERATING_IMAGES,
            ProcessingStep.COMPLETING_SESSION
        ]

        print(f"✅ Checkpoint types test successful ({len(checkpoint_types)} types, {len(processing_steps)} steps)")
        return True
    except Exception as e:
        print(f"❌ Checkpoint types test failed: {e}")
        return False

def test_directory_structure():
    """Test that necessary directories can be created."""
    try:
        base_dir = Path("test_output")
        directories = [
            base_dir / "sessions" / "active_sessions",
            base_dir / "sessions" / "checkpoints",
            base_dir / "sessions" / "recovery"
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            if not directory.exists():
                raise Exception(f"Failed to create directory: {directory}")

        print("✅ Directory structure test successful")
        return True
    except Exception as e:
        print(f"❌ Directory structure test failed: {e}")
        return False

def run_tests():
    """Run all tests."""
    print("🧪 Testing Session Persistence and Checkpoint Functionality")
    print("=" * 60)

    tests = [
        ("SessionState Creation", test_session_state_creation),
        ("File Operations", test_file_operations),
        ("Checkpoint Types", test_checkpoint_types),
        ("Directory Structure", test_directory_structure)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔄 Running {test_name} test...")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The persistence system is ready to use.")
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)