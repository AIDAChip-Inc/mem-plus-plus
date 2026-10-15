"""Native MemoryBench agent, copied into src/agent by install_native.py."""
from __future__ import annotations
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
from typing import Literal
import uuid

from pydantic import BaseModel, ConfigDict, Field
from src.agent.base_agent import BaseAgent
from src.llms import LlmFactory
from .mempp_store import MemppStore, ENGINE_COMMIT, PROFILE

_NS = uuid.uuid5(uuid.NAMESPACE_DNS, 'memorybench.mem-plus-plus.aidachip.com')


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def world_id(run_id, cache_path):
    return str(uuid.uuid5(_NS, canonical([str(uuid.UUID(run_id)), str(Path(cache_path).resolve())])))


class MemppAgentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    llm_provider: Literal['openai', 'vllm', 'anthropic'] = 'anthropic'
    llm_config: dict = Field(default_factory=lambda: {'model': 'claude-sonnet-4-6', 'temperature': 1, 'top_p': None, 'max_tokens': 4096})
    memory_cache_dir: str
    run_id: str = Field(default_factory=lambda: os.environ.get('MEMPP_MEMORYBENCH_RUN_ID', ''))
    retrieve_k: int = Field(default=50, ge=1, le=100, strict=True)


class MemppAgent(BaseAgent):
    def __init__(self, config, *, store_factory=MemppStore, llm=None):
        self.config = config
        self.run_id = str(uuid.UUID(config.run_id))
        self.cache_path = Path(config.memory_cache_dir).resolve()
        self.world = world_id(self.run_id, self.cache_path)
        self.store = store_factory(self.world)
        self.llm = llm if llm is not None else LlmFactory.create(config.llm_provider, copy.deepcopy(config.llm_config))
        self._docs = {}
        self._loaded = False
        self._lock = threading.RLock()
        self._memory_trace_local = threading.local()

    @property
    def manifest(self):
        return self.cache_path / 'mempp-cache.json'

    def _ensure_loaded(self):
        if not self._loaded:
            if self.manifest.exists():
                self.load_memories()
            else:
                self.store.verify({})
                self._loaded = True

    def add_memory(self, content, doc_id):
        if not isinstance(content, str) or not content.strip():
            raise ValueError('Memory content must be nonblank source text')
        if not isinstance(doc_id, str) or not doc_id or len(doc_id) > 2048:
            raise ValueError('Memory source ID must contain 1..2048 characters')
        with self._lock:
            self._ensure_loaded()
            if doc_id in self._docs and self._docs[doc_id] != content:
                raise ValueError('Source identity reused for changed content')
            self.store.put(doc_id, content)
            self._docs[doc_id] = content

    def add_conversation_to_memory(self, messages, conversation_idx=0):
        # Content digest distinguishes new on-policy observations without
        # overwriting prior history; exact retry remains idempotent.
        for index, msg in enumerate(messages):
            if not isinstance(msg.get('role'), str) or not isinstance(msg.get('content'), str):
                raise ValueError('Expected role/content string messages')
            content = f"Speaker {msg['role']} says: {msg['content']}"
            identity = canonical([conversation_idx, index, msg['role'], msg['content']])
            self.add_memory(content, 'dialog:' + hashlib.sha256(identity.encode()).hexdigest())

    def retrieve_memory(self, query, k=50):
        if type(k) is not int or not 1 <= k <= 100:
            raise ValueError('retrieve_k must be an integer in1..100')
        with self._lock:
            self._ensure_loaded()
            by_id = {self.store.record_id(key): (key, value) for key, value in self._docs.items()}
        hits = self.store.search(query, k)
        trace, context = [], []
        for hit in hits:
            source_id, content = by_id[hit['id']]
            if hit['content'] != 'user: ' + content:
                raise RuntimeError('Recall content differs from recorded source')
            trace.append({'id': hit['id'], 'source_id': source_id, 'content': content})
            context.append(f'[{source_id}]\n{content}')
        self.set_last_memory_trace(trace)
        return context

    def generate_response(self, messages, lang='en', retrieve_k=None):
        if lang not in ('en', 'zh') or not messages or messages[-1].get('role') != 'user':
            raise ValueError('Expected en/zh and a final user message')
        question = messages[-1]['content']
        context = '\n'.join(self.retrieve_memory(question, self.config.retrieve_k if retrieve_k is None else retrieve_k))
        if lang == 'en':
            prompt = f"Context:\n{context}\n\nUser: \n{question}\n\nBased on the context provided, respond naturally and appropriately to the user's input above."
        else:
            prompt = f'相关知识：\n{context}\n\n用户输入：\n{question}\n\n请根据提供的相关知识准确、自然地回答用户的输入。'
        # Native reading.py retries the same history on failure, then uses its
        # augmented first turn for follow-ups. Commit only a successful prompt.
        request = copy.deepcopy(messages)
        request[-1]['content'] = prompt
        response = self.llm.generate_response(messages=request)
        messages[-1]['content'] = prompt
        return response

    def export_memory_records(self):
        with self._lock:
            self._ensure_loaded()
            return [{'id': self.store.record_id(key), 'source_id': key, 'content': value}
                    for key, value in sorted(self._docs.items())]

    def memory_records_storage_format(self):
        return 'both'

    def save_memories(self):
        with self._lock:
            self._ensure_loaded()
            self.store.verify(self._docs)
            payload = {'version': 1, 'run_id': self.run_id, 'origin_cache': str(self.cache_path),
                       'world': self.world, 'engine_commit': ENGINE_COMMIT, 'profile': PROFILE,
                       'documents': self._docs}
            envelope = {'payload': payload, 'sha256': hashlib.sha256(canonical(payload).encode()).hexdigest()}
            self.cache_path.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile('w', dir=self.cache_path, delete=False, encoding='utf-8') as file:
                temp = Path(file.name)
                try:
                    file.write(canonical(envelope)); file.flush(); os.fsync(file.fileno()); file.close()
                    os.replace(temp, self.manifest)
                finally:
                    temp.unlink(missing_ok=True)

    def load_memories(self):
        with self._lock:
            envelope = json.loads(self.manifest.read_text(encoding='utf-8'))
            payload = envelope['payload']
            if hashlib.sha256(canonical(payload).encode()).hexdigest() != envelope['sha256']:
                raise ValueError('Memory snapshot checksum mismatch')
            if (payload['version'] != 1 or payload['run_id'] != self.run_id or
                    payload['engine_commit'] != ENGINE_COMMIT or payload['profile'] != PROFILE or
                    world_id(self.run_id, payload['origin_cache']) != payload['world']):
                raise ValueError('Snapshot run/profile/world mismatch')
            documents = payload['documents']
            if not isinstance(documents, dict) or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() or not 1 <= len(k) <= 2048 for k,v in documents.items()):
                raise ValueError('Invalid source snapshot')
            if self.world != payload['world']:
                # Native runner copied training cache to a corpus-specific path.
                # Fork immutable training snapshot; do not reuse its mutable DB.
                self.store.verify(documents, allow_subset=True)
                for key, content in documents.items():
                    self.store.put(key, content)
            self.store.verify(documents)
            self._docs = dict(documents)
            self._loaded = True
            if self.world != payload['world']:
                self.save_memories()
