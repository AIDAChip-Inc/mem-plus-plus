"""Migration integrity.

``test_single_alembic_head`` is always-runnable — it reads the versions directory
only (no DB), so a clean clone can prove there is exactly one head before any
Postgres exists. The upgrade/downgrade round-trip against a live server is
DB-gated (``test_e2e_pgvector.py`` covers real-schema behavior).
"""
from __future__ import annotations

from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory

_ROOT = Path(__file__).resolve().parents[1]


def test_single_alembic_head():
    """Exactly one migration head — the repo-wide migration-heads gate, provable
    offline (no DSN, no connection)."""
    cfg = Config(str(_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ROOT / "alembic"))
    script = ScriptDirectory.from_config(cfg)
    heads = script.get_heads()
    assert len(heads) == 1, f"expected exactly one migration head, got {heads}"


def test_initial_migration_is_base():
    """The single revision is a base (no down_revision) — a clean linear history."""
    cfg = Config(str(_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(_ROOT / "alembic"))
    script = ScriptDirectory.from_config(cfg)
    bases = script.get_bases()
    assert list(bases) == ["0001"]
