"""Put the memory-research project root on sys.path so ``import eval`` resolves
when pytest is invoked from anywhere."""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Register the gating markers used by the DB/embedder-gated consolidation
    sweep test (mirrors tests/conftest.py) so ``-m db``/``-m embedder`` select
    cleanly and no unknown-marker warning fires."""
    config.addinivalue_line(
        "markers", "embedder: needs the real in-process ONNX MiniLM-384 embedder (no DB)"
    )
    config.addinivalue_line(
        "markers", "db: needs a reachable Postgres + pgvector (MEMORY_DATABASE_URL)"
    )
