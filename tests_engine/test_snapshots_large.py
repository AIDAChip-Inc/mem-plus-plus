"""Opt-in actual >256 MiB round trip; uses about 1 GiB of temporary memory."""
import json
import os
import uuid

import pytest
from sqlalchemy import func

from memory.models import AgentMemory
from memory.snapshots import SnapshotSession
from test_snapshots import objects as objects


@pytest.mark.skipif(os.environ.get("MEMORY_TEST_LARGE_SNAPSHOT") != "1", reason="Set MEMORY_TEST_LARGE_SNAPSHOT=1 for >256 MiB round trip")
def test_corpus_larger_than_previous_cap(objects):
    with SnapshotSession(objects, writer=True) as writer:
        writer.execute(AgentMemory.__table__.insert().values(
            user_id=uuid.UUID(int=1), agent_type="assistant", content=func.zeroblob(257 * 1024**2)))
        writer.commit()
    pointer = json.loads(objects.get("latest.json").data)
    manifest = json.loads(objects.get(f"manifests/{pointer['sha256']}.json").data)
    assert manifest["size"] > 256 * 1024**2
    with SnapshotSession(objects) as reader:
        assert reader.query(AgentMemory).count() == 1
        assert reader.query(func.length(AgentMemory.content)).scalar() == 257 * 1024**2
