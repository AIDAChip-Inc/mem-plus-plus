"""Pinned in-process Mem++ gateway; no public HTTP endpoint or paid ingestion."""
from __future__ import annotations
import importlib.metadata
import json
from pathlib import Path
import subprocess
import uuid

ENGINE_COMMIT = 'd952129834f1749c412a04528a4f9138d94a7d74'
PROFILE = {'embedding': 'onnx384', 'weights': [1, 1, 2], 'candidates': 200, 'lexical_or': True}


class MemppStore:
    def __init__(self, world):
        self.world = world
        self._service = None

    def _ready(self):
        if self._service is not None:
            return
        dist = importlib.metadata.distribution('memory-research')
        direct = json.loads(dist.read_text('direct_url.json') or '{}')
        commit = direct.get('vcs_info', {}).get('commit_id')
        if not commit and direct.get('dir_info', {}).get('editable'):
            from urllib.parse import urlparse, unquote
            root = Path(unquote(urlparse(direct['url']).path))
            commit = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '--', 'memory', 'pyproject.toml', 'uv.lock'], text=True).strip()
            if dirty:
                commit += '.dirty'
        if commit != ENGINE_COMMIT:
            raise RuntimeError('Install the documented pinned Mem++ engine before live use')
        from memory import config
        from memory.db import get_engine, get_session
        from memory.aml_api import AMLService
        from sqlalchemy import text
        if get_engine().dialect.name != 'postgresql':
            raise RuntimeError('Use migrated PostgreSQL/pgvector')
        actual = (config.MEMORY_RRF_W_FUZZY, config.MEMORY_RRF_W_TAG, config.MEMORY_RRF_W_VECTOR,
                  config.CANDIDATE_LIMIT, config.MEMORY_LEXICAL_OR)
        if actual != (1., 1., 2., 200, True):
            raise RuntimeError('MemoryBench profile requires RRF1/1/2, candidates200, lexicalORon')
        if config.MEMORY_EMBEDDING_BACKEND != 'onnx' or not config.MEMORY_EMBEDDINGS_ENABLED:
            raise RuntimeError('Use enabled local ONNX embeddings')
        with get_session() as db:
            dimension = db.execute(text("SELECT format_type(atttypid,atttypmod) FROM pg_attribute WHERE attrelid='agent_memory'::regclass AND attname='embedding'")).scalar_one()
            if dimension != 'vector(384)':
                raise RuntimeError('Use vector384 schema')
        # Internal reuse of the reviewed ingestion/recall transaction code. This
        # object is called directly, never bound to a network listener.
        self._service = AMLService(get_session, 'memorybench-in-process-token-unused-' + uuid.uuid4().hex)

    def record_id(self, source_id):
        self._ready()
        from memory.aml_api import _identity
        _, pid = _identity(self.world)
        return str(uuid.uuid5(uuid.uuid5(pid, source_id), '0'))

    def put(self, source_id, content):
        self._ready()
        from memory.aml_api import AddRequest, Message
        self._service.add(AddRequest(request_id=source_id, user_id=self.world,
                                    session_id='memorybench', messages=[Message(role='user', content=content)]))

    def search(self, query, k):
        self._ready()
        from memory.aml_api import SearchRequest
        return self._service.search(SearchRequest(query=query, user_id=self.world, top_k=k))['data']

    def verify(self, documents, *, allow_subset=False):
        """Fail closed on absent/wrong DB, extra rows, missing vectors or drift."""
        self._ready()
        from memory.aml_api import _identity, _AGENT
        from memory.models import AgentMemory
        uid, pid = _identity(self.world)
        with self._service.sessions() as db:
            rows = db.query(AgentMemory).filter_by(user_id=uid, project_id=pid, agent_type=_AGENT).all()
            expected = {self.record_id(k): 'user: ' + v for k, v in documents.items()}
            if (len(rows) > len(expected) if allow_subset else len(rows) != len(expected)) or any(str(row.id) not in expected or row.content != expected[str(row.id)] or row.embedding is None for row in rows):
                raise RuntimeError('Database does not match the saved MemoryBench snapshot')
