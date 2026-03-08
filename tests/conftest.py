"""
Pytest configuration for basketball app tests.

Ensures project root is on sys.path so the "api" package is findable.
Backend tests require numpy, lightgbm, etc.; each test module uses
pytest.importorskip("numpy") so missing deps yield a clear skip message.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
