"""Adapter contract tests use an in-memory fake transport, no models."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest
import requests

from integrations.benchd import mempp_adapter as adapter


TOKEN = "local-test-token-not-a-secret-0123456789"


class Session:
    def __init__(self):
        self.headers = {}
        self.calls = []
        self.worlds = {}
        self.closed = False
        self.failure = None

    def request(self, method, url, **kw):
        self.calls.append((method, url, copy.deepcopy(kw)))
        response = requests.Response()
        response.status_code = self.failure or 200
        data = json.loads(kw["data"]) if kw["data"] else None
        if method == "GET":
            result = {"status": "ok"}
        elif url.endswith("/add"):
            bucket = self.worlds.setdefault(data["user_id"], {})
            bucket[data["request_id"]] = data["messages"]
            result = {k: data[k] for k in ["request_id", "user_id", "session_id"]}
            result["success"] = True
        else:
            result = {"data": [{"id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f'{rid}:{i}')),
                                "content": msg["content"]}
                               for rid, messages in self.worlds.get(data["user_id"], {}).items()
                               for i, msg in enumerate(messages)][:data["top_k"]]}
        response._content = json.dumps(result).encode()
        return response

    def close(self):
        self.closed = True


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setenv("MEMORY_API_TOKEN", TOKEN)
    monkeypatch.delenv("MEMPP_BENCHD_URL", raising=False)
    monkeypatch.delenv("MEMPP_BENCHD_TOP_K", raising=False)
    session = Session()
    monkeypatch.setattr(adapter.requests, "Session", lambda: session)
    a = adapter.MemppAdapter()
    a.setup()
    yield a, session
    a.teardown()


def test_reset_sharing_retries_and_evidence(live):
    a, session = live
    turns = [{"role": "user", "content": "Atlas launch approved"},
             {"role": "assistant", "content": "Finance confirmed"}]
    first = a._world
    a.ingest(turns)
    initial = a.recall("Atlas")
    a.ingest(turns)
    assert a.recall("Atlas") == initial
    assert len(a.last_evidence) == 2
    assert "Atlas launch approved" in initial and "Finance confirmed" in initial
    a.ingest([{"role": "user", "content": "Legal also approved"}])
    assert "Legal also approved" in a.recall("Atlas")
    assert len(a.last_evidence) == 3
    a.reset()
    assert a._world != first and a.recall("Atlas") == ""
    assert first in session.worlds  # No broad deletion of organization data.
    assert all(call[2]["allow_redirects"] is False for call in session.calls)
    assert session.headers["Authorization"] == "Bearer " + TOKEN
    assert session.trust_env is False


def test_original_role_metadata_and_utc_time(live):
    a, session = live
    a.ingest([{"role": "system", "content": "保留原文", "timestamp": "2024-01-01T01:00:00+01:00",
               "metadata": {"author": "Team B", "tags": ["Atlas"]}}])
    msg = json.loads(session.calls[-1][2]["data"])["messages"][0]
    assert msg["role"] == "user"
    assert "Original role: system\n保留原文" in msg["content"]
    assert json.loads(msg["content"].split("Original metadata: ")[1]) == {"author": "Team B", "tags": ["Atlas"]}
    assert msg["timestamp"] == 1704067200000
    assert "保留原文".encode() in session.calls[-1][2]["data"]


@pytest.mark.parametrize("timestamp", ["2024-01-01T00:00:00", "nonsense", 1704067200, True])
def test_invalid_times_fail_before_ingestion(live, timestamp):
    a, session = live
    before = len(session.calls)
    with pytest.raises(ValueError):
        a.ingest([{"role": "user", "content": "first"},
                  {"role": "user", "content": "invalid", "timestamp": timestamp}])
    assert len(session.calls) == before


def test_duplicate_turns_in_one_batch_are_preserved(live):
    a, _ = live
    a.ingest([{"role": "user", "content": "Same"}] * 2)
    a.recall("Same")
    assert len(a.last_evidence) == 2
    assert len({r["id"] for r in a.last_evidence}) == 2


@pytest.mark.parametrize("status", [302, 401, 429, 503])
def test_transport_errors_propagate(live, status):
    a, session = live
    session.failure = status
    with pytest.raises((requests.HTTPError, RuntimeError)):
        a.recall("Atlas")


@pytest.mark.parametrize("url", ["http://example.com", "https://user:secret@example.com", "https://example.com/?token=x",
                                 "https://example.com/#x", "https://example.com/path", "ftp://localhost", "http://localhost:bad"])
def test_unsafe_endpoint(url):
    with pytest.raises(ValueError):
        adapter.endpoint(url)


@pytest.mark.parametrize("url", ["http://localhost:8787/", "http://127.0.0.1:8787", "http://[::1]:8787", "https://example.com"])
def test_safe_endpoint(url):
    assert adapter.endpoint(url) == url.rstrip("/")


def test_no_silent_truncation(live):
    a, session = live
    before = len(session.calls)
    with pytest.raises(ValueError, match="4 MiB"):
        a.ingest([{"role": "user", "content": "字" * 1_500_000}])
    assert len(session.calls) == before


def test_teardown_requires_new_setup(live):
    a, session = live
    a.teardown()
    assert session.closed
    with pytest.raises(RuntimeError):
        a.recall("Atlas")


def test_wrapper_registers_without_upstream_edit(monkeypatch):
    from integrations.benchd import run_harness
    registry = {"llm-baseline": object(), "other": object()}
    seen = []
    monkeypatch.setattr(run_harness, "_BUILTIN_ADAPTERS", registry)
    monkeypatch.setattr(run_harness.sys, "argv", ["run_harness", "--help"])
    monkeypatch.setattr(run_harness, "cli", lambda: seen.append(registry["mempp"]))
    run_harness.main()
    assert seen == [adapter.MemppAdapter]
    assert registry == {"mempp": adapter.MemppAdapter}
    with pytest.raises(RuntimeError, match="replace"):
        run_harness.main()


@pytest.mark.parametrize("flag", ["--judge", "--judge=true"])
def test_wrapper_blocks_unguarded_paid_judging(monkeypatch, flag):
    from integrations.benchd import run_harness
    registry = {}
    monkeypatch.setattr(run_harness, "_BUILTIN_ADAPTERS", registry)
    monkeypatch.setattr(run_harness.sys, "argv", ["run_harness", "run", flag])
    with pytest.raises(RuntimeError, match="shared-budget"):
        run_harness.main()
    assert registry == {}


@pytest.mark.parametrize("command", ["submit", "auto-run"])
def test_wrapper_excludes_submission_and_batch_routing(monkeypatch, command):
    from integrations.benchd import run_harness
    monkeypatch.setattr(run_harness.sys, "argv", ["run_harness", command])
    with pytest.raises(RuntimeError, match="local driver"):
        run_harness.main()


@pytest.mark.parametrize("args", [["adapter", "validate", "llm-baseline"],
                                  ["run", "--adapter=llm-baseline", "-b", "smoke-memory-v0", "--no-judge"],
                                  ["run", "-allm-baseline", "-b", "smoke-memory-v0", "--no-judge"]])
def test_actual_cli_cannot_route_to_model_backed_builtin(args):
    # No caller credentials are inherited even if a future routing regression
    # occurs. Installed pinned harness and this checkout are the only imports.
    result = subprocess.run([sys.executable, "-m", "integrations.benchd.run_harness", *args],
                            cwd=Path(__file__).resolve().parents[2],
                            env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 1
    assert "Adapter 'llm-baseline' requires the openai package" in result.stdout + result.stderr
