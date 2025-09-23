"""Test configuration for path setup.

Ensures the `src` directory is on sys.path so the `illustrator` package
can be imported without installing the project in editable mode.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

