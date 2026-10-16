"""Bench'd evidence-only adapter for the authenticated Mem++ Add/Search API."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import ipaddress
import json
import os
from urllib.parse import urlsplit
import uuid

import requests
from benchd_harness.adapters.base import BaseAdapter


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def endpoint(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("Use a root HTTP(S) origin without credentials, query or fragment")
    if parsed.scheme == "http":
        try:
            local = ipaddress.ip_address(parsed.hostname).is_loopback
        except ValueError:
            local = parsed.hostname == "localhost"
        if not local:
            raise ValueError("Remote Mem++ endpoints require HTTPS")
    # Validate the port before sending an authenticated request.
    parsed.port
    return value.rstrip("/")


def message(turn):
    if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant", "system"}:
        raise ValueError("Expected a user, assistant or system turn")
    content = turn.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Turn content must be nonempty text")
    metadata = turn.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("Turn metadata must be an object")
    # AML has no system-role enum. Preserve the original role explicitly in
    # source text; user here is the transport role, not an attribution claim.
    role = turn["role"]
    if role == "system":
        content = "Original role: system\n" + content
        role = "user"
    if metadata:
        content += "\nOriginal metadata: " + canonical(metadata)
    result = {"role": role, "content": content}
    timestamp = turn.get("timestamp")
    if timestamp is not None:
        if not isinstance(timestamp, str):
            raise ValueError("Timestamp must be timezone-aware ISO 8601 text")
        when = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if when.tzinfo is None or when.utcoffset() is None:
            raise ValueError("Timestamp must include a timezone")
        delta = when.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        result["timestamp"] = delta // timedelta(milliseconds=1)
    return result


class MemppAdapter(BaseAdapter):
    """One fresh memory world per reset; shared evidence within each example."""
    name = "mempp"
    version = "0.1.0"

    def __init__(self):
        self._session = None
        self._world = None
        self.last_evidence = []

    def setup(self):
        if self._session is not None:
            raise RuntimeError("Adapter already initialized")
        self._url = endpoint(os.environ.get("MEMPP_BENCHD_URL", "http://127.0.0.1:8787"))
        token = os.environ.get("MEMORY_API_TOKEN", "")
        if len(token) < 32 or any(not 33 <= ord(c) <= 126 for c in token):
            raise ValueError("Configure a 32+ printable-ASCII-character MEMORY_API_TOKEN without whitespace")
        self._k = int(os.environ.get("MEMPP_BENCHD_TOP_K", "50"))
        if not 1 <= self._k <= 100:
            raise ValueError("MEMPP_BENCHD_TOP_K must be between 1 and 100")
        self._session = requests.Session()
        self._session.trust_env = False
        self._session.headers["Authorization"] = "Bearer " + token
        try:
            self.reset()
            health = self._request("GET", "/health")
            if health != {"status": "ok"}:
                raise RuntimeError("Unexpected Mem++ health response")
            if self.recall("benchd_canary_xyz789"):
                raise RuntimeError("Fresh Mem++ world returned canary evidence")
        except Exception:
            self.teardown()
            raise

    def _ready(self):
        if self._session is None or self._world is None:
            raise RuntimeError("Call setup() before using Mem++")

    def _request(self, method, path, payload=None):
        self._ready()
        body = canonical(payload).encode("utf-8") if payload is not None else None
        response = self._session.request(method, self._url + path, data=body,
                                         headers={"Content-Type": "application/json"},
                                         timeout=(10, 1800), allow_redirects=False)
        if 300 <= response.status_code < 400:
            raise RuntimeError("Mem++ redirects are not followed with credentials")
        response.raise_for_status()
        return response.json()

    def reset(self):
        if self._session is None:
            raise RuntimeError("Call setup() before reset()")
        # The pinned harness resets for each independent BenchmarkItem. This
        # does not delete any organization/project or change Mem++ sharing.
        self._world = "benchd:" + str(uuid.uuid4())
        self.last_evidence = []

    def ingest(self, turns):
        self._ready()
        if not isinstance(turns, list):
            raise ValueError("Expected a list of turns")
        if not turns:
            return
        messages = [message(turn) for turn in turns]
        # An exact batch retry (including after a lost response) has the same
        # request ID. Repeated identical turns inside this batch retain indices.
        batch = canonical(messages)
        payload = {"request_id": "benchd:" + hashlib.sha256(batch.encode()).hexdigest(),
                   "user_id": self._world, "session_id": "benchd", "messages": messages}
        if len(canonical(payload).encode()) > 4 * 1024 * 1024:
            raise ValueError("Turn batch exceeds the Mem++ API 4 MiB limit")
        receipt = self._request("POST", "/add", payload)
        expected = {"success": True, **{k: payload[k] for k in ["request_id", "user_id", "session_id"]}}
        if receipt != expected:
            raise RuntimeError("Mem++ ingestion receipt does not match the request")

    def recall(self, query):
        self._ready()
        self.last_evidence = []
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be nonempty text")
        result = self._request("POST", "/search", {"user_id": self._world, "query": query, "top_k": self._k})
        rows = result.get("data") if isinstance(result, dict) else None
        if not isinstance(rows, list) or len(rows) > self._k:
            raise RuntimeError("Unexpected Mem++ search response")
        seen = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not isinstance(row.get("content"), str):
                raise RuntimeError("Invalid Mem++ evidence row")
            uuid.UUID(row["id"])
            if row["id"] in seen:
                raise RuntimeError("Duplicate Mem++ evidence identity")
            seen.add(row["id"])
        self.last_evidence = rows
        return "\n\n".join(f'[{row["id"]}]\n{row["content"]}' for row in rows)

    def teardown(self):
        if self._session is not None:
            self._session.close()
        self._session = None
        self._world = None
        self.last_evidence = []
