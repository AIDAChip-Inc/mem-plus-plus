"""Configurable pre/post-hook LLM model — resolution + threading.

Proves the model knob (config.resolve_model) maps friendly names to the
authoritative production ids, defaults to Haiku, and is actually threaded from
``store_facts`` / the extractors down to ``llm_call`` — with ZERO behavior change
when a model is not passed. LLM-free: ``llm_call`` is monkeypatched to capture the
model kwarg, so no SDK / credential / network is needed.
"""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from memory import config, extraction, recall
from memory.models import Base


# ── resolve_model (pure) ─────────────────────────────────────────────────────────
def test_resolve_model_maps_names_to_production_ids():
    assert config.resolve_model("haiku") == config.EXTRACTION_MODEL  # zero-diff default
    assert config.resolve_model("sonnet") == "claude-sonnet-5"
    assert config.resolve_model("opus") == "claude-opus-4-8"
    assert config.resolve_model("OPUS") == "claude-opus-4-8"  # case-insensitive


def test_resolve_model_default_is_haiku():
    assert config.resolve_model(None) == config.EXTRACTION_MODEL
    assert config.resolve_model("") == config.EXTRACTION_MODEL


def test_resolve_model_passes_through_unknown_id_idempotently():
    # A full id (or unknown name) is returned unchanged -> safe to resolve twice.
    assert config.resolve_model("claude-sonnet-5") == "claude-sonnet-5"
    assert config.resolve_model(config.resolve_model("opus")) == "claude-opus-4-8"


# ── threading: extractor -> llm_call ──────────────────────────────────────────────
def test_extraction_threads_chosen_model_to_llm_call(monkeypatch):
    captured: dict = {}

    def fake_llm_call(prompt, *, system_prompt="", model=None, max_tokens=256):
        captured["model"] = model
        return '{"summary": "a durable fact", "tags": ["t"]}'

    monkeypatch.setattr(extraction, "llm_call", fake_llm_call)
    extraction.extract_summary_and_tags("some content", gate=False, model="opus")
    assert captured["model"] == "claude-opus-4-8"  # name resolved to the id


def test_extraction_defaults_to_haiku_when_model_unset(monkeypatch):
    captured: dict = {}

    def fake_llm_call(prompt, *, system_prompt="", model=None, max_tokens=256):
        captured["model"] = model
        return '{"summary": "x", "tags": []}'

    monkeypatch.setattr(extraction, "llm_call", fake_llm_call)
    extraction.extract_atomic_facts("some content")  # no model -> default
    assert captured["model"] == config.EXTRACTION_MODEL


# ── threading: store_facts -> extractor -> llm_call (SQLite-degraded) ──────────────
@pytest.fixture()
def sqlite_env(monkeypatch):
    monkeypatch.setenv("MEMORY_USER", "awsi-model")
    engine = create_engine("sqlite://", future=True)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=True, future=True)
    monkeypatch.setattr("memory.db._engine", engine, raising=False)
    monkeypatch.setattr("memory.db._SessionLocal", Session, raising=False)
    yield
    engine.dispose()


def test_store_facts_threads_model_end_to_end(sqlite_env, monkeypatch):
    captured: dict = {}

    def fake_llm_call(prompt, *, system_prompt="", model=None, max_tokens=256):
        captured["model"] = model
        return '{"worth_storing": true, "summary": "The tapeout is set for Q3", "tags": ["tapeout"]}'

    monkeypatch.setattr(extraction, "llm_call", fake_llm_call)
    res = recall.store_facts("awsi", "User: the tapeout is set for Q3.", model="sonnet")
    assert captured["model"] == "claude-sonnet-5"  # chat's choice threaded to the post-hook
    assert res["written"] == 1
