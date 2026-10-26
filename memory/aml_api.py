"""Textual AML Add/Search adapter over Mem++ (no answering or extraction).

Run ``python -m memory.aml_api`` with a storage configuration and a dedicated
MEMORY_API_TOKEN. Storage is migrated PostgreSQL or opt-in object-store snapshots;
both service profiles use local ONNX embeddings.
Serve behind a TLS reverse proxy for remote evaluation; the default is loopback.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from http import HTTPStatus
from socketserver import ThreadingMixIn
from typing import Literal
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from memory import config
from memory.db import get_engine, get_session
from memory.embeddings import get_embedding_service
from memory.extraction import reconcile_tags
from memory.models import AgentMemory, MemoryAPIRequest
from memory.object_store import PublicationConflict
from memory.store import PostgresMemoryStore, make_scope_key

_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "aml.mem-plus-plus.aidachip.com")
_AGENT = "aml-textual"
_MAX_BODY = 4 * 1024 * 1024
logger = logging.getLogger(__name__)


class _Payload(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class Message(_Payload):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)
    timestamp: int | None = None  # AML Unix milliseconds, not seconds


class AddRequest(_Payload):
    request_id: str = Field(min_length=1, max_length=2048)
    user_id: str = Field(min_length=1, max_length=2048)
    session_id: str = Field(min_length=1, max_length=2048)
    messages: list[Message] = Field(min_length=1)


class SearchRequest(_Payload):
    query: str = Field(min_length=1)
    user_id: str = Field(min_length=1, max_length=2048)
    top_k: int = Field(ge=1, le=100)
    options: list[str] | None = None


class APIError(Exception):
    def __init__(self, status: int, message: str):
        self.status, self.message = status, message


def _identity(user_id: str):
    """One platform user_id is one independent organizational memory world.

    Sessions and contributors within that world share evidence. No other
    platform user_id, nor the demo project, can enter this scope.
    """
    key = json.dumps([user_id], ensure_ascii=True)
    return uuid.uuid5(_NS, "user:" + key), uuid.uuid5(_NS, "project:" + key)


class AMLService:
    paths = {"/add", "/search"}

    def __init__(self, sessions, token: str):
        if len(token) < 32 or any(not (33 <= ord(c) <= 126) for c in token):
            raise ValueError("MEMORY_API_TOKEN must be at least 32 printable ASCII characters without whitespace")
        self.sessions = sessions
        self.authorization = ("Bearer " + token).encode("utf-8")

    def add(self, request: AddRequest) -> dict:
        # Validate all timestamps before starting a transaction or doing inference.
        try:
            dates = [datetime.fromtimestamp(m.timestamp / 1000, UTC)
                     if m.timestamp is not None else None for m in request.messages]
        except (ValueError, OverflowError, OSError):
            raise APIError(422, "timestamp outside supported datetime range") from None
        if any(not m.content.strip() for m in request.messages):
            raise APIError(422, "message content must not be blank")
        uid, pid = _identity(request.user_id)
        rid = uuid.uuid5(pid, request.request_id)
        digest = hashlib.sha256(request.model_dump_json().encode("utf-8")).hexdigest()
        sid = uuid.uuid5(pid, request.session_id)

        def check_existing(db):
            previous = db.get(MemoryAPIRequest, rid)
            if previous is None:
                return False
            if previous.payload_sha256 != digest:
                raise APIError(409, "request_id already used with a different payload")
            return True

        try:
            with self.sessions() as db:
                if db.info.get("snapshot_reader"):
                    raise APIError(403, "read-only replica; send ingestion to the writer")
                if not check_existing(db):
                    # The PK serializes concurrent retries. Ledger and records are
                    # committed together, so a successful retry cannot be partial.
                    db.add(MemoryAPIRequest(id=rid, payload_sha256=digest))
                    db.flush()
                    store = PostgresMemoryStore(db)
                    for i, (message, occurred_at) in enumerate(zip(request.messages, dates)):
                        content = f"{message.role}: {message.content}"
                        mid = uuid.uuid5(rid, str(i))
                        row = AgentMemory(
                            id=mid, user_id=uid, project_id=pid, customer_id=_NS,
                            agent_type=_AGENT, session_id=sid, source_message_id=mid,
                            content=content, context_summary=content, occurred_at=occurred_at,
                        )
                        if occurred_at is not None:
                            row.valid_from = occurred_at
                        store._populate_search_fields(row, tsv_source=content, embed_source=content)
                        if row.embedding is None:
                            raise APIError(503, "embedding unavailable; ingestion rolled back")
                        row.tags = reconcile_tags(db, make_scope_key(uid, _AGENT, pid),
                                                  [f"by:{message.role}"])
                        db.add(row)
                    db.commit()
        except IntegrityError:
            # Another process may have committed this exact request while this
            # transaction waited on the PK. Other conflicts are retryable 409s.
            with self.sessions() as db:
                if not check_existing(db):
                    raise APIError(409, "concurrent ingestion conflict; retry request") from None
        return {"success": True, "request_id": request.request_id,
                "user_id": request.user_id, "session_id": request.session_id}

    def search(self, request: SearchRequest) -> dict:
        if not request.query.strip():
            raise APIError(422, "query must not be blank")
        uid, pid = _identity(request.user_id)
        with self.sessions() as db:
            rows = PostgresMemoryStore(db).recall(
                user_id=uid, project_id=pid, agent_type=_AGENT, query=request.query,
                k=request.top_k, update_hits=False,
            )
            evidence = []
            for row in rows:
                # Use event time when supplied; retain the original source text.
                when = row.occurred_at or row.created_at
                if when.tzinfo is None:
                    when = when.replace(tzinfo=UTC)
                when = when.astimezone(UTC)
                content = (f"[{when.isoformat()}] {row.content}"
                           if row.occurred_at is not None else row.content)
                evidence.append({"id": str(row.id), "content": content,
                                 "created_at": when.isoformat()})
        return {"data": evidence}

    def dispatch(self, path: str, raw: bytes) -> dict:
        model = AddRequest if path == "/add" else SearchRequest
        try:
            request = model.model_validate_json(raw)
        except ValidationError:
            raise APIError(422, "invalid request fields") from None
        return self.add(request) if path == "/add" else self.search(request)

    def __call__(self, environ, start_response):
        try:
            path, method = environ.get("PATH_INFO", ""), environ.get("REQUEST_METHOD", "")
            if path == "/health" and method == "GET":
                with self.sessions() as db:
                    db.execute(text("SELECT 1"))
                payload = {"status": "ok"}
            else:
                supplied = environ.get("HTTP_AUTHORIZATION", "").encode("utf-8")
                if not hmac.compare_digest(supplied, self.authorization):
                    raise APIError(401, "unauthorized")
                if path not in self.paths:
                    raise APIError(404, "not found")
                if method != "POST":
                    raise APIError(405, "POST required")
                if environ.get("CONTENT_TYPE", "").split(";")[0].strip().lower() != "application/json":
                    raise APIError(415, "application/json required")
                try:
                    length = int(environ.get("CONTENT_LENGTH", ""))
                except ValueError:
                    raise APIError(411, "Content-Length required") from None
                if length < 1 or length > _MAX_BODY:
                    raise APIError(413, "request body outside supported size")
                raw = environ["wsgi.input"].read(length)
                if len(raw) != length:
                    raise APIError(400, "incomplete request body")
                payload = self.dispatch(path, raw)
            status = 200
        except APIError as exc:
            status, payload = exc.status, {"detail": exc.message}
        except PublicationConflict:
            status, payload = 409, {"detail": "snapshot publication conflict; retry the same request"}
        except Exception as exc:
            # Do not log SQL parameters, request content or authorization headers.
            logger.error("AML request failed (%s)", type(exc).__name__)
            status, payload = 503, {"detail": "service unavailable"}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        start_response(f"{status} {HTTPStatus(status).phrase}",
                       [("Content-Type", "application/json; charset=utf-8"),
                        ("Content-Length", str(len(body))), ("Cache-Control", "no-store")])
        return [body]


class _Server(ThreadingMixIn, WSGIServer):
    daemon_threads = True


class _Handler(WSGIRequestHandler):
    def setup(self):
        self.request.settimeout(30)  # bound header/body reads, not backend processing
        super().setup()

    def log_message(self, format, *args):
        pass  # evaluation payloads/URLs are not access-logged


def main(argv=None, *, service_factory=AMLService, default_port=8787, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=default_port)
    args = parser.parse_args(argv)
    service = service_factory(get_session, os.environ.get("MEMORY_API_TOKEN", ""))
    snapshots = os.environ.get("MEMORY_STORAGE_BACKEND", "postgres") == "object_store"
    if not snapshots and get_engine().dialect.name != "postgresql":
        raise RuntimeError("AML service requires migrated PostgreSQL + pgvector")
    if config.MEMORY_EMBEDDING_BACKEND != "onnx" or not config.MEMORY_EMBEDDINGS_ENABLED:
        raise RuntimeError("This no-paid-inference service profile requires enabled ONNX embeddings")
    if not get_embedding_service("memory").warm_load():
        raise RuntimeError("Local embedding model unavailable")
    with get_session() as db:
        if snapshots:
            if not db.info["snapshot_reader"] and db.etag is None:
                db.commit()  # publish an empty corpus before accepting readers
        else:
            dimension = db.execute(text("SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                                        "WHERE attrelid='agent_memory'::regclass AND attname='embedding'"))
            if dimension.scalar_one() != "vector(384)":
                raise RuntimeError("AML service requires the 384-dimensional embedding schema")
        db.query(MemoryAPIRequest.id).limit(1).all()  # verify migration before serving
    with make_server(args.host, args.port, service, server_class=_Server,
                     handler_class=_Handler) as server:
        print(f"{type(service).__name__} listening on {args.host}:{args.port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
