"""Authenticated MemoryArena initialize/add/wrap_user_prompt evaluation facade.

Requires MEMORY_ARENA_RUN_ID (a fresh UUID for each independent evaluation run).
Use MemoryArena's requests.Session hook or add the token header to its client.
No agent answering, web search, scoring, or paid inference is performed here.
"""
from __future__ import annotations

import hashlib
from html import escape
import json
import os
import threading
from typing import Literal
import uuid

from pydantic import Field, ValidationError

from memory.aml_api import AMLService, APIError, AddRequest, SearchRequest, _Payload, main as serve


class InitializeRequest(_Payload):
    user_id: str = Field(min_length=1, max_length=1024)
    memory_system_name: Literal["mempp"]


class ChunkRequest(InitializeRequest):
    chunk: str = Field(min_length=1)


class PromptRequest(InitializeRequest):
    question: str = Field(min_length=1)


class MemoryArenaService(AMLService):
    paths = {"/memory/initialize", "/memory/add", "/memory/wrap_user_prompt"}

    def __init__(self, sessions, token: str, *, run_id: str, top_k: int = 10):
        super().__init__(sessions, token)
        self.run_id = str(uuid.UUID(run_id))
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 100:
            raise ValueError("top_k must be an integer from 1 to 100")
        self.top_k = top_k
        self._initialized: set[str] = set()
        self._lock = threading.Lock()

    def dispatch(self, path: str, raw: bytes) -> dict:
        model = {"/memory/initialize": InitializeRequest,
                 "/memory/add": ChunkRequest,
                 "/memory/wrap_user_prompt": PromptRequest}[path]
        try:
            request = model.model_validate_json(raw)
        except ValidationError:
            raise APIError(422, "invalid request fields") from None
        if not request.user_id.strip():
            raise APIError(422, "user_id must not be blank")
        if path == "/memory/initialize":
            # Reinitialization by another session must not erase shared history.
            with self._lock:
                self._initialized.add(request.user_id)
            return {"status": "ok", "user_id": request.user_id, "memory_system_name": "mempp"}
        with self._lock:
            initialized = request.user_id in self._initialized
        if not initialized:
            raise APIError(404, "user not initialized")
        # JSON encoding prevents delimiter ambiguity, including adversarial IDs.
        scope = json.dumps(["memoryarena", self.run_id, request.user_id], ensure_ascii=True)
        world = "memoryarena:" + hashlib.sha256(scope.encode("utf-8")).hexdigest()
        if path == "/memory/add":
            # The upstream API has no request ID. Exact duplicate chunks within a
            # world are intentionally deduplicated, including after a lost reply.
            receipt = hashlib.sha256(request.chunk.encode("utf-8")).hexdigest()
            payload = AddRequest(request_id=receipt, user_id=world, session_id="memoryarena",
                                 messages=[{"role": "user", "content": request.chunk}])
            # Upstream MemoryClient does not retry. Retry only the engine's
            # atomic canonical-tag race; permanent conflicts remain errors.
            for attempt in range(3):
                try:
                    self.add(payload)
                    break
                except APIError as exc:
                    if (exc.status != 409 or exc.message != "concurrent ingestion conflict; retry request"
                            or attempt == 2):
                        raise
            return {"status": "ok", "user_id": request.user_id, "response": {"success": True}}
        rows = self.search(SearchRequest(user_id=world, query=request.question,
                                         top_k=self.top_k))["data"]
        lines = ["<memory_context>"]
        if rows:
            for row in rows:
                lines.append(f'<memory id="{row["id"]}">{escape(row["content"])}</memory>')
        else:
            lines.append("None")
        lines.extend(["</memory_context>", f"User: {request.question}"])
        return {"status": "ok", "user_id": request.user_id, "prompt": "\n".join(lines)}


def main(argv=None):
    run_id = os.environ.get("MEMORY_ARENA_RUN_ID", "")
    if not run_id:
        raise RuntimeError("MEMORY_ARENA_RUN_ID is required; use a fresh UUID per independent run")
    top_k = int(os.environ.get("MEMORY_ARENA_TOP_K", "10"))
    def factory(sessions, token):
        return MemoryArenaService(sessions, token, run_id=run_id, top_k=top_k)
    serve(argv, service_factory=factory, default_port=8000, description=__doc__)


if __name__ == "__main__":
    main()
