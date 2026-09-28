"""Answer-generation LLM callable + the frozen answer prompt.

The answer model DEFAULTS to the same dated id as the judge (HAND's production
eval baseline) so answer and judge sit on one comparable protocol — but it is
now user-SELECTABLE (haiku / sonnet / opus, or a raw model id). The default is
the pin, so ``None`` reproduces the frozen baseline exactly. Only the MODEL
swaps; the answer PROMPT stays pinned + versioned (``ANSWER_PROMPT_VERSION``) —
changing the prompt text starts a new baseline, changing the model does not.

``make_answer_fn`` lazily imports ``anthropic`` so the harness runs with a stub
answerer and no API key; token counts flow into the systems metrics.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import NamedTuple

ANSWER_MODEL = "claude-haiku-4-5-20251001"
ANSWER_PROMPT_VERSION = "memory-research-answer-v2"
ANSWER_MAX_TOKENS = 512

# Friendly-name -> model-id map, SHARED by the answer + judge sides so a run can
# select haiku / sonnet / opus without knowing dated ids. Reproducibility is
# preserved because "haiku" resolves to the same pin used as the default.
#
# Integration note: prefer the single shared map Awsi is adding to
# ``memory.config`` this wave; this local fallback carries the SAME friendly
# names so Nizam can reconcile to one source without a rename. Ids verified
# against production ``app/config.py`` / the claude-api model table (2026-07-23):
# haiku=claude-haiku-4-5-20251001 (the pin), sonnet=claude-sonnet-5,
# opus=claude-opus-4-8.
try:  # pragma: no cover - depends on whether Awsi's map has landed in this branch
    from memory.config import MODEL_ALIASES  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001 - any import failure means "not in this branch yet"
    MODEL_ALIASES: dict[str, str] = {
        "haiku": ANSWER_MODEL,  # the pinned reproducibility default (dated)
        "sonnet": "claude-sonnet-5",
        "opus": "claude-opus-4-8",
    }


def resolve_model(name_or_id: str | None, default: str) -> str:
    """Resolve a friendly model NAME (``haiku`` / ``sonnet`` / ``opus``,
    case-insensitive) to its model id.

      * ``None`` -> ``default`` (the pinned reproducibility default) — so a caller
        that passes nothing reproduces the frozen baseline byte-for-byte.
      * a known friendly name -> its mapped id.
      * anything else -> returned UNCHANGED (treated as a raw model id).
    """
    if name_or_id is None:
        return default
    return MODEL_ALIASES.get(name_or_id.strip().lower(), name_or_id)

# v2. v1 ended with an absolute refusal instruction ("If the excerpts do not
# contain the information needed to answer, reply exactly: I don't know"), which
# is wrong for LoCoMo's open_domain category: those questions are inferential by
# construction ("Would Caroline likely have Dr. Seuss books on her bookshelf?" ->
# "Yes, since she collects classic children's books") and the answer is never
# stated in the transcript. The instruction suppressed that whole category for
# EVERY system evaluated through this harness -- running mem0 through it dropped
# its open_domain from 72.93 (paper) to 33.33, while the same run BEAT the paper
# on the other three categories. It also produced 103 refusals with the gold card
# already in the prompt. v2 permits grounded inference while still requiring
# abstention when the excerpts give no basis at all, which the 446 adversarial
# items depend on.
ANSWER_PROMPT_TEMPLATE = (
    "You are answering a question about past conversations using the memory "
    "excerpts below.\n\n"
    "Memory excerpts:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer concisely (a short phrase or sentence), grounded in the excerpts "
    "above. You may reason from what the excerpts say to reach a well-supported "
    "conclusion, including a likely answer to a question the excerpts do not "
    "state outright. If the excerpts give no basis for any answer, reply "
    "exactly: I don't know."
)

# v3-short: the Mem0/LoCoMo answer-LENGTH protocol. Mem0's evaluation prompt
# (their deleted evals/, recovered from git history) instructs "The answer should
# be less than 5-6 words". F1 and BLEU-1 are answer-length metrics -- our v2
# answers average ~255 characters and score F1 ~17 against their ~39 for the
# same content -- so Table 1 is only comparable under the same constraint.
# Opt-in via MEMORY_ANSWER_STYLE=short so every stored v2 result reproduces.

# Mem0's evaluation answer prompt, VERBATIM. Source: mem0 repo, evaluation/prompts.py
# at commit 9315e303^ (the file was deleted on 2026-06-14; recovered read-only from
# git history). This is ANSWER_PROMPT_ZEP -- the single "Memories:" block variant
# Mem0 used for systems whose memory is not split per speaker (ours is not);
# its instructions 1-8 and the step-by-step approach are identical to the
# two-speaker ANSWER_PROMPT. Only the Jinja {{ }} placeholders were changed to
# Python {context}/{question}. Their runner sends it as the SYSTEM message with
# temperature 0.0 (evaluation/src/memzero/search.py:113-115), model = $MODEL
# (gpt-4o-mini in the paper), top_k default 30 (run_experiments.py:29).
MEM0_ANSWER_PROMPT = """
    You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

    # CONTEXT:
    You have access to memories from a conversation. These memories contain
    timestamped information that may be relevant to answering the question.

    # INSTRUCTIONS:
    1. Carefully analyze all provided memories
    2. Pay special attention to the timestamps to determine the answer
    3. If the question asks about a specific event or fact, look for direct evidence in the memories
    4. If the memories contain contradictory information, prioritize the most recent memory
    5. If there is a question about time references (like "last year", "two months ago", etc.),
       calculate the actual date based on the memory timestamp. For example, if a memory from
       4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
    6. Always convert relative time references to specific dates, months, or years. For example,
       convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory
       timestamp. Ignore the reference while answering the question.
    7. Focus only on the content of the memories. Do not confuse character
       names mentioned in memories with the actual users who created those memories.
    8. The answer should be less than 5-6 words.

    # APPROACH (Think step by step):
    1. First, examine all memories that contain information related to the question
    2. Examine the timestamps and content of these memories carefully
    3. Look for explicit mentions of dates, times, locations, or events that answer the question
    4. If the answer requires calculation (e.g., converting relative time references), show your work
    5. Formulate a precise, concise answer based solely on the evidence in the memories
    6. Double-check that your answer directly addresses the question asked
    7. Ensure your final answer is specific and avoids vague time references

    Memories:

    {context}

    Question: {question}
    Answer:
    """

# Where the prompt goes. Mem0 sends the whole rendered prompt as the SYSTEM
# message; our v2/v3 prompts are user turns. Set per style below.
ANSWER_AS_SYSTEM = False
ANSWER_STYLE = os.environ.get("MEMORY_ANSWER_STYLE", "").strip().lower()
if ANSWER_STYLE == "short":
    ANSWER_PROMPT_VERSION = "memory-research-answer-v3-short"
    ANSWER_PROMPT_TEMPLATE = ANSWER_PROMPT_TEMPLATE + (
        "\n\nThe answer should be less than 5-6 words.")
elif ANSWER_STYLE == "mem0":
    ANSWER_PROMPT_VERSION = "mem0-eval-answer-prompt-verbatim(zep-variant)"
    ANSWER_PROMPT_TEMPLATE = MEM0_ANSWER_PROMPT
    ANSWER_AS_SYSTEM = True
elif ANSWER_STYLE not in ("", "v2", "default"):
    raise ValueError(f"MEMORY_ANSWER_STYLE={ANSWER_STYLE!r}: expected mem0|short|v2")


class LLMReply(NamedTuple):
    """Uniform reply shape for injected answer callables (real or stubbed)."""

    text: str
    input_tokens: int = 0
    output_tokens: int = 0


def build_answer_prompt(*, context: str, question: str) -> str:
    return ANSWER_PROMPT_TEMPLATE.format(context=context, question=question)


def make_answer_fn(
    model: str | None = None, max_tokens: int = ANSWER_MAX_TOKENS
) -> Callable[[str], LLMReply]:
    """Return ``prompt -> LLMReply`` backed by the Anthropic API (lazy import).

    ``model`` accepts a friendly name (haiku / sonnet / opus) or a raw id;
    ``None`` uses the pinned default ``ANSWER_MODEL``. The prompt is fixed
    (``ANSWER_PROMPT_VERSION``) regardless of which model is chosen.
    """
    # Cross-provider answering, mirroring make_judge_fn: an "openai:" or "gpt"
    # prefix selects the OpenAI chat API at temperature 0 with a fixed seed.
    # Needed to reproduce Mem0's protocol (gpt-4o-mini answerer). With the
    # mem0 prompt style the prompt is the SYSTEM message, as in their runner.
    _m = (model or "").strip().lower()
    if _m.startswith("openai:") or _m.startswith("gpt"):
        from openai import OpenAI  # noqa: PLC0415 - deliberately lazy

        resolved_oai = model.split(":", 1)[1] if ":" in model else model
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        seed = int(os.environ.get("MEMORY_ANSWER_SEED", "20260904"))
        role = "system" if ANSWER_AS_SYSTEM else "user"

        def answer_fn_openai(prompt: str) -> LLMReply:
            resp = oai.chat.completions.create(
                model=resolved_oai, temperature=0, seed=seed,
                max_completion_tokens=max_tokens,
                messages=[{"role": role, "content": prompt}])
            u = resp.usage
            return LLMReply(text=resp.choices[0].message.content or "",
                            input_tokens=u.prompt_tokens, output_tokens=u.completion_tokens)

        return answer_fn_openai

    import anthropic  # noqa: PLC0415 — deliberately lazy

    resolved = resolve_model(model, ANSWER_MODEL)
    client = anthropic.Anthropic()

    def answer_fn(prompt: str) -> LLMReply:
        _kw = ({"system": prompt, "messages": [{"role": "user", "content": "Answer:"}]}
               if ANSWER_AS_SYSTEM else
               {"messages": [{"role": "user", "content": prompt}]})
        resp = client.messages.create(
            model=resolved,
            max_tokens=max_tokens,
            # Was unset, i.e. the Anthropic default of 1.0, while the mem0 arm
            # (mem0_arm.py:92) pinned temperature=0 -- so every mem++ number was
            # single-seed sampling noise against a deterministic competitor, and
            # make_tables.py published "temperature 0" over both rows regardless.
            temperature=0,
            **_kw,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        return LLMReply(text=text, input_tokens=resp.usage.input_tokens,
                        output_tokens=resp.usage.output_tokens)

    return answer_fn
