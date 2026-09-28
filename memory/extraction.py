"""Memory extraction (write path) — ported from production ``extraction.py``.

One Haiku call per turn produces (context_summary, candidate tags, occurred_at);
candidates reconcile against the per-scope canonical tag registry. ``occurred_at``
is the EVENT time. All LLM access goes through ``memory.llm.llm_call`` (the
pluggable shim), which returns ``None`` when no LLM is available — so every
extractor degrades to "nothing" rather than crashing.

Trimmed vs production: the ``extract_turn_facts``/``extract_and_store`` POST-hook
plumbing and the ``isolate_subprocess`` SDK option are dropped (this replica's
write path calls the extractors directly — see ``recall.store_facts``). The
prompt templates, occurred_at contract, JSON parsing, and anti-confabulation
grounding are preserved verbatim.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime, time

from sqlalchemy.orm import Session

from memory import config
from memory.llm import llm_call
from memory.models import MemoryTag

logger = logging.getLogger(__name__)

MAX_EXCERPT_CHARS = 4000
MAX_TOKENS = 300
MAX_ATOMIC_TOKENS = 600
_FALLBACK_SUMMARY_CHARS = 240
_MAX_TAGS = 8
_MAX_ATOMIC_FACTS = 8

_SYSTEM_PROMPT = (
    "You extract structured memory metadata for AIDAChip, an EDA-first AI agent "
    "platform. Content may concern chip design or any other domain (business, "
    "operations, planning, personal) — extract faithfully in the content's own "
    "domain. Respond with exactly the JSON value the instructions request "
    "(object or array) — no prose, no markdown fences."
)

_PROMPT_TEMPLATE = """Summarize this content and extract entity tags. The content is \
one agent conversation turn OR a document (meeting notes, an email thread, a chat \
log, a wiki page, call notes).

Content (first {max_chars} chars):
---
{excerpt}
---

Return ONLY a JSON object with these fields:
- "summary": the durable facts, decisions, or results in this content — at most 2 \
sentences for a short conversation turn, up to 4 sentences for a long document \
covering multiple decisions
- "tags": array of 1-6 short entity tags naming the concrete entities involved, \
matching the content's domain (e.g. ["pll", "phase-noise", "loop-filter"] for \
circuit design; ["vendor-contract", "q3-budget", "sales-portal"] for business or \
operations). Lowercase, hyphenated.
__OCCURRED_AT_CLAUSE__

Return ONLY the JSON object, no markdown fences."""

_OCCURRED_AT_CLAUSE = (
    '- "occurred_at": the date the described event happened, as ISO "YYYY-MM-DD". '
    'Fill it from an explicit date, a dated context line like "[7 May 2023] ...", '
    'or a clear relative reference ("yesterday", "10 days ago", "last Tuesday") '
    "resolved against the reference date given above — resolve relative references "
    "ONLY when that reference date is present; without it, or if the timing is "
    'vague ("recently", "a while back") or unstated, null. Never use your own '
    "idea of today's date, and never invent a date."
)
_PROMPT_TEMPLATE = _PROMPT_TEMPLATE.replace("__OCCURRED_AT_CLAUSE__", _OCCURRED_AT_CLAUSE, 1)

_GATED_PROMPT_TEMPLATE = """Decide whether this agent conversation turn contains a \
DURABLE fact worth remembering, then summarize it.

Turn content (first {max_chars} chars):
---
{excerpt}
---

Set "worth_storing" to false (and leave summary "") when the turn has NO durable, \
reusable fact — e.g. the agent refusing or saying it doesn't know, the agent talking \
about its own memory/capabilities, the user merely asking a question or floating an \
UNCONFIRMED guess, greetings, acknowledgements, or chit-chat. Set it true only when the \
turn states a stable fact the user would want recalled later: a goal, constraint, \
decision, identifier, preference, outcome, or an explicit open question the user is \
tracking.

CRITICAL — record only facts GROUNDED IN THE USER'S OWN WORDS. The user's messages are \
marked "User:"; the assistant's are marked "Agent:". Capture what the user stated or \
explicitly confirmed. Do NOT record specifics the ASSISTANT introduced that the user did \
not say — invented file names, document references, numbers, dates, or names the user \
never mentioned. If the assistant elaborated, restated, or speculated beyond the user's \
words, exclude that elaboration; if the turn's only "fact" is the assistant's own \
unconfirmed elaboration, set worth_storing to false. When in doubt, prefer the user's \
literal statement over the assistant's paraphrase.

Return ONLY a JSON object with these fields:
- "worth_storing": true or false per the rule above
- "summary": if worth_storing, at most 2 sentences capturing the durable fact(s); else ""
- "tags": if worth_storing, 1-6 short lowercase hyphenated entity tags; else []
__OCCURRED_AT_CLAUSE__

Return ONLY the JSON object, no markdown fences."""
_GATED_PROMPT_TEMPLATE = _GATED_PROMPT_TEMPLATE.replace(
    "__OCCURRED_AT_CLAUSE__", _OCCURRED_AT_CLAUSE, 1
)

_MAX_AGENT_SUMMARY_CHARS = 300

# Two-section cell (A2 shape): the SAME gated call gains ONE output field.
_GATED_TWO_SECTION_TEMPLATE = _GATED_PROMPT_TEMPLATE.replace(
    '- "occurred_at":',
    '- "agent_summary": if worth_storing and the assistant\'s reply (marked "Agent:") '
    "said, did, or confirmed something DIRECTLY relevant to the stored fact, one tight "
    "sentence (max 200 characters) summarizing what the AGENT stated — kept strictly "
    'OUT of "summary", which stays grounded in the user\'s own words; else null\n'
    '- "occurred_at":',
    1,
)
if "agent_summary" not in _GATED_TWO_SECTION_TEMPLATE:
    raise RuntimeError("_GATED_TWO_SECTION_TEMPLATE: agent_summary anchor splice failed")

_ATOMIC_PROMPT_TEMPLATE = """Extract the DISTINCT durable facts stated in this \
agent conversation turn — the separate, standalone, reusable pieces of \
information it contains.

Turn content (first {max_chars} chars):
---
{excerpt}
---

Return a JSON LIST (array) of fact objects, ONE object per GENUINELY DISTINCT \
durable fact (a goal, constraint, decision, identifier, preference, outcome, or \
explicit open question). Most turns yield ONE or TWO facts.

PREFER FEWER, COMPLETE FACTS OVER MANY FRAGMENTS. Combine closely-related \
details into a SINGLE self-contained fact — do NOT shatter one idea into \
separate fragments. For example, "User adopted a dog named Biscuit on Tuesday" \
is ONE fact, not three (adopted a dog / named Biscuit / on Tuesday). Emit a \
SEPARATE fact only when the turn states a genuinely UNRELATED piece of \
information. If the turn contains no durable fact, return an empty list [].

CRITICAL — every fact MUST be GROUNDED IN THE USER'S OWN WORDS. The user's \
messages are marked "User:"; the assistant's are marked "Agent:". Record only \
what the user stated or explicitly confirmed. Do NOT record specifics the \
ASSISTANT introduced that the user did not say — invented file names, document \
references, numbers, dates, or names the user never mentioned. If the assistant \
elaborated, restated, or speculated beyond the user's words, exclude that \
elaboration. When in doubt, prefer the user's literal statement over the \
assistant's paraphrase.

Each object has these fields:
- "summary": ONE complete fact, one or two sentences, self-contained (fold in \
the closely-related detail rather than emitting a second fragment for it)
- "tags": 1-6 short lowercase hyphenated entity tags for that fact
__OCCURRED_AT_CLAUSE__

Return ONLY the JSON array, no prose, no markdown fences."""
_ATOMIC_PROMPT_TEMPLATE = _ATOMIC_PROMPT_TEMPLATE.replace(
    "__OCCURRED_AT_CLAUSE__",
    _OCCURRED_AT_CLAUSE.replace("the described event", "THAT fact's event", 1),
    1,
)
for _t in (_PROMPT_TEMPLATE, _GATED_PROMPT_TEMPLATE, _ATOMIC_PROMPT_TEMPLATE):
    if "__OCCURRED_AT_CLAUSE__" in _t or '- "occurred_at":' not in _t:
        raise RuntimeError("occurred_at clause substitution failed")


def _fallback(content: str) -> tuple[str, list[str], None]:
    """Failure result (ungated path): truncated content, no tags, no date."""
    return content[:_FALLBACK_SUMMARY_CHARS].strip(), [], None


_SKIP: tuple[str, list[str], None] = ("", [], None)
_SKIP_TWO_SECTION: tuple[str, list[str], None, None] = (*_SKIP, None)


def _parse_iso_date(value: object) -> datetime | None:
    """Strict ISO date/datetime from model output; missing/garbage -> None."""
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        return datetime.combine(date.fromisoformat(raw), time.min, tzinfo=UTC)
    except ValueError:
        pass
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _parse_extraction_response(
    text: str,
) -> tuple[str, list[str], datetime | None, bool, str | None] | None:
    """Parse the Haiku JSON object response. Returns None if unusable."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("memory extraction: failed to parse JSON response")
        return None
    if not isinstance(data, dict):
        return None

    worth = bool(data.get("worth_storing", True))
    summary = data.get("summary")
    if not isinstance(summary, str):
        summary = ""
    summary = summary.strip()
    if worth and not summary:
        return None
    raw_tags = data.get("tags", [])
    tags: list[str] = []
    if isinstance(raw_tags, list):
        tags = [t.strip() for t in raw_tags if isinstance(t, str) and t.strip()][:_MAX_TAGS]
    agent_summary = data.get("agent_summary")
    if not isinstance(agent_summary, str) or not agent_summary.strip():
        agent_summary = None
    else:
        agent_summary = agent_summary.strip()[:_MAX_AGENT_SUMMARY_CHARS]
    return summary, tags, _parse_iso_date(data.get("occurred_at")), worth, agent_summary


def _anchor_line(reference_date: datetime | None) -> str:
    """Date-anchor prologue enabling relative-reference resolution."""
    if reference_date is None:
        return (
            "No reference date is provided for this content. Do NOT resolve "
            'relative time references ("yesterday", "10 days ago", "last '
            'Tuesday") into dates — fill occurred_at only from an explicit '
            "date or a dated context line in the text; otherwise null.\n\n"
        )
    return (
        f"Reference date: this content is from {reference_date:%Y-%m-%d} "
        f"({reference_date:%A}). Resolve relative time references against it.\n\n"
    )


_EXPLICIT_DATE_HINT = re.compile(
    r"\b(19|20)\d{2}\b"
    r"|\b(january|february|march|april|may|june|july|august|september|october"
    r"|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\b",
    re.IGNORECASE,
)


def _guard_unanchored_date(
    occurred_at: datetime | None, content: str, reference_date: datetime | None
) -> datetime | None:
    """Null a model date when there is NO anchor and no explicit date hint."""
    if occurred_at is None or reference_date is not None:
        return occurred_at
    return occurred_at if _EXPLICIT_DATE_HINT.search(content[:MAX_EXCERPT_CHARS]) else None


def _call_extraction(
    template: str, content: str, model: str | None, reference_date: datetime | None = None
) -> tuple[str, list[str], datetime | None, bool, str | None] | None:
    """One LLM call with ``template`` -> parsed 5-tuple, or None on failure."""
    resolved_model = config.resolve_model(model)
    prompt = _anchor_line(reference_date) + template.format(
        max_chars=MAX_EXCERPT_CHARS, excerpt=content[:MAX_EXCERPT_CHARS]
    )
    try:
        raw = llm_call(
            prompt, system_prompt=_SYSTEM_PROMPT, max_tokens=MAX_TOKENS, model=resolved_model
        )
    except Exception as exc:
        logger.warning("memory extraction llm_call raised: %s", exc)
        raw = None
    if raw is None:
        return None
    parsed = _parse_extraction_response(raw)
    if parsed is None:
        return None
    summary, tags, occurred_at, worth, agent_summary = parsed
    occurred_at = _guard_unanchored_date(occurred_at, content, reference_date)
    return summary, tags, occurred_at, worth, agent_summary


class MemoryExtractionUnavailable(RuntimeError):
    """Extraction FAILED (LLM unreachable/unusable) — distinct from "no facts".

    Raised only when a caller opts in via ``raise_on_unavailable=True``. The
    default contracts (``[]`` / ``_SKIP``) are unchanged.
    """


def extract_summary_and_tags(
    content: str,
    *,
    model: str | None = None,
    gate: bool = False,
    raise_on_unavailable: bool = False,
    reference_date: datetime | None = None,
) -> tuple[str, list[str], datetime | None]:
    """One LLM call -> (context_summary, candidate tags, occurred_at)."""
    template = _GATED_PROMPT_TEMPLATE if gate else _PROMPT_TEMPLATE
    parsed = _call_extraction(template, content, model, reference_date=reference_date)
    if parsed is None:
        if gate and raise_on_unavailable:
            raise MemoryExtractionUnavailable("gated extraction failed (LLM unreachable/unusable)")
        return _SKIP if gate else _fallback(content)
    summary, tags, occurred_at, worth, _ = parsed
    if gate and (not worth or not summary):
        return _SKIP
    return summary, tags, occurred_at


def extract_two_section(
    content: str,
    *,
    model: str | None = None,
    raise_on_unavailable: bool = False,
    reference_date: datetime | None = None,
) -> tuple[str, list[str], datetime | None, str | None]:
    """Gated extraction + the agent-stated section (two-section cell, A2 shape)."""
    parsed = _call_extraction(
        _GATED_TWO_SECTION_TEMPLATE, content, model, reference_date=reference_date
    )
    if parsed is None:
        if raise_on_unavailable:
            raise MemoryExtractionUnavailable(
                "two-section extraction failed (LLM unreachable/unusable)"
            )
        return _SKIP_TWO_SECTION
    summary, tags, occurred_at, worth, agent_summary = parsed
    if not worth or not summary:
        return _SKIP_TWO_SECTION
    return summary, tags, occurred_at, agent_summary


def _loads_lazy(cleaned: str) -> object | None:
    """Best-effort JSON parse for the atomic-fact response (#26 robustness)."""
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    candidates = []
    arr_open, arr_close = cleaned.find("["), cleaned.rfind("]")
    if 0 <= arr_open < arr_close:
        candidates.append((arr_open, cleaned[arr_open : arr_close + 1]))
    obj_open, obj_close = cleaned.find("{"), cleaned.rfind("}")
    if 0 <= obj_open < obj_close:
        candidates.append((obj_open, cleaned[obj_open : obj_close + 1]))
    for _open, snippet in sorted(candidates):
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return None


def _parse_atomic_response(text: str) -> list[tuple[str, list[str], datetime | None]] | None:
    """Parse the Haiku JSON-LIST response into atomic-fact tuples (#25, #26)."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    data = _loads_lazy(cleaned)
    if data is None:
        logger.warning("memory atomic extraction: failed to parse JSON list")
        return None
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    facts: list[tuple[str, list[str], datetime | None]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        summary = item.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            continue
        raw_tags = item.get("tags", [])
        tags: list[str] = []
        if isinstance(raw_tags, list):
            tags = [t.strip() for t in raw_tags if isinstance(t, str) and t.strip()][:_MAX_TAGS]
        facts.append((summary.strip(), tags, _parse_iso_date(item.get("occurred_at"))))
        if len(facts) >= _MAX_ATOMIC_FACTS:
            break
    return facts


def extract_atomic_facts(
    content: str,
    *,
    model: str | None = None,
    raise_on_unavailable: bool = False,
    reference_date: datetime | None = None,
) -> list[tuple[str, list[str], datetime | None]]:
    """One LLM call -> a LIST of atomic facts (#25), mem0-style granularity."""
    resolved_model = config.resolve_model(model)
    prompt = _anchor_line(reference_date) + _ATOMIC_PROMPT_TEMPLATE.format(
        max_chars=MAX_EXCERPT_CHARS, excerpt=content[:MAX_EXCERPT_CHARS]
    )
    try:
        raw = llm_call(
            prompt, system_prompt=_SYSTEM_PROMPT, max_tokens=MAX_ATOMIC_TOKENS, model=resolved_model
        )
    except Exception as exc:
        logger.warning("memory atomic extraction llm_call raised: %s", exc)
        if raise_on_unavailable:
            raise MemoryExtractionUnavailable(f"extraction LLM raised: {exc}") from exc
        return []
    if raw is None:
        if raise_on_unavailable:
            raise MemoryExtractionUnavailable("extraction LLM unreachable (llm_call returned None)")
        return []
    parsed = _parse_atomic_response(raw)
    if parsed is None:
        if raise_on_unavailable:
            raise MemoryExtractionUnavailable("extraction response unparseable (not a JSON list)")
        return []
    null_unanchored = reference_date is None and not _EXPLICIT_DATE_HINT.search(
        content[:MAX_EXCERPT_CHARS]
    )
    return [(s, t, None if null_unanchored else o) for s, t, o in parsed]


def normalize_label(raw: str) -> str:
    """Normalize a candidate tag: lowercase, strip, collapse [ _-]+ to '-'."""
    label = raw.strip().lower()
    label = re.sub(r"[ _-]+", "-", label)
    return label.strip("-")


def reconcile_tags(db: Session, scope_key: str, candidates: list[str]) -> list[MemoryTag]:
    """Reconcile candidates against the per-scope canonical tag registry.

    REUSE the existing canonical tag on match, INSERT a new one on miss.
    Duplicates and empty candidates are dropped.
    """
    tags: list[MemoryTag] = []
    seen: set[str] = set()
    for candidate in candidates or []:
        if not isinstance(candidate, str):
            continue
        label = normalize_label(candidate)
        if not label or label in seen:
            continue
        seen.add(label)
        tag = (
            db.query(MemoryTag)
            .filter(MemoryTag.scope_key == scope_key, MemoryTag.label == label)
            .first()
        )
        if tag is None:
            tag = MemoryTag(scope_key=scope_key, label=label)
            db.add(tag)
            db.flush()
        tags.append(tag)
    return tags
