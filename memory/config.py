"""Static engine configuration — plain constants, NOT Flagsmith.

The production store/extraction paths gate behavior on Flagsmith flags and read
tunables from a pydantic ``Settings`` object. In this research replica those are
replaced by the constants below (user decision: Flagsmith stubbed to constants
here). **Only the flag SOURCE changes** (Flagsmith registry -> config constant) —
the store/extraction logic stays byte-faithful.

The flag defaults MATCH PRODUCTION (verified against
``aidachip-mvp/src/backend/app/flags/registry.py`` lines 231-239): every flag is
env-overridable so a study can flip it (e.g. turn atomic-fact granularity on)
without editing code — set the same-named env var to 1/true/on or 0/false/off.

The ranking constants are PRESERVED EXACTLY from production for faithfulness:
top-k=24, RRF K=60, weights fuzzy/tag/vector = 1.0/1.0/4.0, recency half-life
7.0d, recent-reserve 3, candidate cap 50, priority MATCH->HITS->RECENCY, salience
band 1000.0, MiniLM 384-d.
"""
import os

_TRUTHY = {"1", "true", "yes", "on"}


def _flag(name: str, default: bool) -> bool:
    """Production default, overridable by the same-named env var."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUTHY


# ── Feature flags — PRODUCTION DEFAULTS (registry.py DEFAULTS), env-overridable ─
# Atomic-fact fan-out ships OFF in production: store_facts(mode="auto") is the
# gated SINGLE-summary path, not N-fact fan-out. Flip MEMORY_ATOMIC_FACTS=1 to
# study atomic granularity.
MEMORY_ATOMIC_FACTS = _flag("MEMORY_ATOMIC_FACTS", False)
MEMORY_EMBEDDINGS_ENABLED = _flag("MEMORY_EMBEDDINGS_ENABLED", True)  # vector leg + write embedding
MEMORY_PROJECT_ENABLED = _flag("MEMORY_PROJECT_ENABLED", True)        # two-section (own + team) recall
MEMORY_EVENT_RESERVE = _flag("MEMORY_EVENT_RESERVE", True)            # event-fresh recent-reserve (v1occ)
MEMORY_TWO_SECTION = _flag("MEMORY_TWO_SECTION", True)                # extraction emits the agent-stated section
# Production default True. The supersession WRITE path (consolidation) is out of
# this replica's scope (recall+write core only) so no ported code reads it; kept
# to record the production default for a study that ports consolidation later.
MEMORY_SUPERSESSION = _flag("MEMORY_SUPERSESSION", True)

# ── Recall ranking constants (memory_design.md §5) — PRESERVE EXACTLY ──────────
MEMORY_RECALL_K = 24               # top-k memories returned by recall_facts
MEMORY_RRF_K = 60                  # Reciprocal Rank Fusion decay constant
# RRF weights for the lexical/fuzzy and entity-tag legs. Env-overridable on the
# same pattern as MEMORY_RRF_W_VECTOR below, and for the same reason: the leg
# ABLATION (set one weight to 0.0) has to be expressible without editing code.
# _rrf_fuse drops a 0.0-weight list outright (store.py:91-93), so 0.0 is a true
# no-op on the leg rather than a very small contribution. Defaults UNCHANGED at
# 1.0 -- every published number reproduces with the vars unset.
MEMORY_RRF_W_FUZZY = float(os.environ.get("MEMORY_RRF_W_FUZZY", "1.0"))  # lexical/fuzzy leg
MEMORY_RRF_W_TAG = float(os.environ.get("MEMORY_RRF_W_TAG", "1.0"))      # entity-tag leg
# RRF weight: vector-ANN leg. Was 4.0, which made the fusion DEGENERATE: with
# rrf_k=60 and a 50-row candidate cap, a document ranked #1 by BOTH non-vector
# legs scored 2/60 = 0.0333, below the WORST member of the vector top-50 at
# 4/(60+49) = 0.0367. The lexical and tag legs therefore could not promote any
# document -- the returned set was the vector top-50, merely reordered. The 4.0
# "eval-tuned knee" was tuned while the lexical leg was also broken (see the
# websearch_to_tsquery note in store.py), so it baked the bug in. At 2.0 with
# the larger pool below, best fuzzy+tag (0.0333) now clears the worst vector row
# (2/259 = 0.0077). Measured A/B on LoCoMo (n=180, k=50): recall@50 +0.039,
# MRR +0.086 (+26.4%), nDCG +0.068, every category up.
# CORPUS-DEPENDENT. The 2.0/200 pair above was tuned on LoCoMo (124-char turns).
# OrgMemBench (2,022-char artefacts) was measured under the ORIGINAL 4.0/50 and
# does not reproduce at 2.0/200 -- base falls 40.49 -> 27.66, C1 31.6 -> 5.3.
# Both settings are correct for their own corpus, so neither is hard-coded now:
# the LoCoMo-tuned values remain the DEFAULT (every published LoCoMo number
# reproduces untouched) and the ORG arms set MEMORY_RRF_W_VECTOR=4 /
# MEMORY_CANDIDATE_LIMIT=50 / MEMORY_LEXICAL_OR=0 explicitly.
MEMORY_RRF_W_VECTOR = float(os.environ.get("MEMORY_RRF_W_VECTOR", "2.0"))
RECENCY_HALF_LIFE_DAYS = 7.0       # recency_decay halves every 7 days
MEMORY_RECALL_RECENT_RESERVE = 3   # k-slots reserved for the freshest facts
# Per-method candidate pool cap. Was 50, i.e. == k, so the fused pool could
# never be larger than the vector leg's own top-k. Raising it alone buys nothing
# (+0.0009 measured) because the weights above forbade the extra candidates from
# cracking the top-k; the two changes only pay off together.
CANDIDATE_LIMIT = int(os.environ.get("MEMORY_CANDIDATE_LIMIT", "200"))
SALIENCE_MATCH_BAND = 1000.0       # matched-row salience band (dominates hits+recency)
# Lexical leg: OR the query terms (LoCoMo default, fixes a 70% zero-hit rate) or
# fall back to conjunctive websearch_to_tsquery, which is what the ORG matrix was
# measured under.
MEMORY_LEXICAL_OR = _flag("MEMORY_LEXICAL_OR", True)

# ── Project Memory (two-section split) ─────────────────────────────────────────
MEMORY_PROJECT_RECALL_K = 8            # team-section cap out of the k budget
MEMORY_PROJECT_RECALL_STRICT_SPLIT = False  # flex split (own backfills)

# ── Embedding (MiniLM-384, ONNX-pinned) ────────────────────────────────────────
# Backend is switchable so mem++ can be measured on the same embedding stack as
# the systems it is compared against. Defaults are the published configuration
# (ONNX MiniLM-384); MEMORY_EMBEDDING_BACKEND=openai selects
# text-embedding-3-small at 1536-d, which needs its own database because the
# pgvector column is fixed-width (models.py Vector(384), alembic 0001).
MEMORY_EMBEDDING_BACKEND = os.environ.get("MEMORY_EMBEDDING_BACKEND", "onnx").strip().lower()
if MEMORY_EMBEDDING_BACKEND == "openai":
    EMBEDDING_MODEL = os.environ.get("MEMORY_EMBEDDING_MODEL", "text-embedding-3-small")
    EMBEDDING_DIM = int(os.environ.get("MEMORY_EMBEDDING_DIM", "1536"))
else:
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384

# ── Extraction / pre-post-hook LLM (pluggable via env API key; degrades if absent) ─
# Default Haiku, pinned to the DATED id so engine extraction and the eval judge
# share ONE reproducible pin (a bare alias could drift). The name->id map lets the
# chat pick a model for BOTH the reply (llm_call) and the post-hook store
# (store_facts -> extraction).
EXTRACTION_MODEL = "claude-haiku-4-5-20251001"

# Selectable models — ids are the AUTHORITATIVE production ones (app/models.yaml:
# haiku/sonnet/opus). "haiku" == the current EXTRACTION_MODEL default (zero-diff).
MEMORY_LLM_MODELS = {
    "haiku": EXTRACTION_MODEL,
    "sonnet": "claude-sonnet-5",
    "opus": "claude-opus-4-8",
}
# Single source of truth for the friendly name->id map; eval/ imports this alias
# so the judge/answer-model selector and the engine share the same pins.
MODEL_ALIASES = MEMORY_LLM_MODELS


def resolve_model(name_or_id: str | None) -> str:
    """Resolve a friendly name (haiku/sonnet/opus) OR a full model id to an id.

    ``None`` -> the Haiku default (``EXTRACTION_MODEL``); a known short name -> its
    id; anything else is passed through UNCHANGED (already a full id). Idempotent,
    so it is safe to call more than once along the LLM path.
    """
    if not name_or_id:
        return EXTRACTION_MODEL
    return MEMORY_LLM_MODELS.get(name_or_id.strip().lower(), name_or_id)
