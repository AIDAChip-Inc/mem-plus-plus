"""Embedder-gated: the REAL in-process ONNX MiniLM-384 model fires.

This is the half of the "ONNX + pgvector" path that Awsi could not verify — but
it needs NO container: the embedder is in-process (onnxruntime + numpy). These
tests load the real fp32 all-MiniLM-L6-v2 export and prove the vectors are (a)
the right shape/norm, (b) deterministic, and (c) SEMANTICALLY meaningful — a
related pair scores far above an unrelated pair, which a hash-based stand-in
(like the unit suite's fake) could never do. SKIPS cleanly if the model cannot
load (see the ``real_embedder`` fixture).
"""
from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.embedder

from memory import config


def _cos(a, b) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb)  # both L2-normalized → dot == cosine


def test_embedding_shape_and_l2_norm(real_embedder):
    vec = real_embedder.embed("Adopt pgvector for semantic recall over embeddings")
    assert vec is not None
    assert len(vec) == config.EMBEDDING_DIM == 384
    assert abs(np.linalg.norm(np.asarray(vec)) - 1.0) < 1e-4  # unit vector


def test_embedding_is_deterministic(real_embedder):
    text = "The loop filter reduces PLL phase noise"
    assert real_embedder.embed(text) == real_embedder.embed(text)


def test_blank_text_returns_none(real_embedder):
    assert real_embedder.embed("   ") is None


def test_semantic_similarity_separates_related_from_unrelated(real_embedder):
    """The decisive proof it is a real semantic model, not a hash: paraphrase
    similarity ≫ unrelated similarity, with a clear margin."""
    anchor = real_embedder.embed("Adopt a vector database for semantic similarity search")
    related = real_embedder.embed("We should use pgvector to find nearest-neighbor embeddings")
    unrelated = real_embedder.embed("The quarterly sales budget meeting is on Tuesday")

    sim_related = _cos(anchor, related)
    sim_unrelated = _cos(anchor, unrelated)
    assert sim_related > 0.3, f"paraphrase similarity too low ({sim_related:.3f})"
    assert sim_related > sim_unrelated + 0.2, (
        f"no semantic separation: related={sim_related:.3f} unrelated={sim_unrelated:.3f}"
    )


def test_query_intent_path_produces_usable_vector(real_embedder):
    """The query-intent branch (used by the recall vector leg) yields a vector
    that ranks the semantically-matching document above a distractor."""
    q = real_embedder.embed("which database for vector similarity?", intent="query")
    match = real_embedder.embed("pgvector stores embeddings for similarity search")
    distractor = real_embedder.embed("the office coffee machine is broken again")
    assert _cos(q, match) > _cos(q, distractor) + 0.2
