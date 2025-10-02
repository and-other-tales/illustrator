#!/usr/bin/env python3
"""
Dynamic version generation utility for the datasets repository.
Generates version numbers in format: MAJOR.MINOR.PATCH.BUILD
where BUILD is based on the unix timestamp of the latest git commit.
"""

import subprocess
import time
from pathlib import Path
from typing import Tuple, Optional

# Base semantic version (update manually for major/minor/patch releases)
MAJOR_VERSION = 1
MINOR_VERSION = 0
PATCH_VERSION = 0

def get_git_commit_timestamp() -> Optional[int]:
    """Get the unix timestamp of the latest git commit."""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%ct'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=5
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except (subprocess.SubprocessError, ValueError, FileNotFoundError):
        pass
    return None

def get_git_commit_hash() -> Optional[str]:
    """Get the short hash of the latest git commit."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return None

def get_build_number() -> int:
    """
    Generate build number from git commit timestamp.
    Falls back to current timestamp if git is unavailable.
    """
    timestamp = get_git_commit_timestamp()
    if timestamp is None:
        timestamp = int(time.time())
    return timestamp

def get_version_tuple() -> Tuple[int, int, int, int]:
    """Get version as tuple (major, minor, patch, build)."""
    build = get_build_number()
    return (MAJOR_VERSION, MINOR_VERSION, PATCH_VERSION, build)

def get_version_string() -> str:
    """Get version as string in format: MAJOR.MINOR.PATCH.BUILD"""
    major, minor, patch, build = get_version_tuple()
    return f"{major}.{minor}.{patch}.{build}"

def get_short_version() -> str:
    """Get short version without build number: MAJOR.MINOR.PATCH"""
    return f"{MAJOR_VERSION}.{MINOR_VERSION}.{PATCH_VERSION}"

def get_version_info() -> dict:
    """Get comprehensive version information."""
    major, minor, patch, build = get_version_tuple()
    commit_hash = get_git_commit_hash()
    
    return {
        'version': f"{major}.{minor}.{patch}.{build}",
        'short_version': f"{major}.{minor}.{patch}",
        'major': major,
        'minor': minor,
        'patch': patch,
        'build': build,
        'commit_hash': commit_hash,
        'timestamp': get_git_commit_timestamp() or int(time.time())
    }

def print_version_info():
    """Print formatted version information."""
    info = get_version_info()
    print(f"Version: {info['version']}")
    print(f"Short Version: {info['short_version']}")
    if info['commit_hash']:
        print(f"Git Commit: {info['commit_hash']}")
    print(f"Build Timestamp: {info['timestamp']}")

if __name__ == "__main__":
    print_version_info()