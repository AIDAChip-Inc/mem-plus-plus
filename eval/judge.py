"""LLM-judge answer metric — selectable model + published, versioned prompt.

LLM-judge variance is the #1 reproducibility risk, so the protocol is frozen
BY DEFAULT while the model stays user-selectable:

  * JUDGE_MODEL is the pinned DEFAULT (a dated model id — a score is only
    comparable under the identical judge). We reuse HAND's production eval judge
    (``claude-haiku-4-5-20251001``) so default numbers sit on the same baseline.
    ``make_judge_fn`` now accepts a friendly name (haiku / sonnet / opus) or a
    raw id to swap the judge model; ``None`` keeps the pin. The chosen model is
    RECORDED in every results-JSON config block (see ``harness.run_benchmark``).
  * The judge PROMPT is published byte-exact in ``judge_prompt.txt`` and
    exposed here as JUDGE_PROMPT_TEMPLATE. Only the MODEL is user-selected — the
    prompt text is fixed; changing it requires bumping JUDGE_PROMPT_VERSION.

Prompt building + verdict parsing + accuracy are pure and unit tested. The
actual model call lives in ``make_judge_fn`` (lazy anthropic import) so the
harness can be exercised with a stub judge and no API key.
"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

from .llm import resolve_model  # shared friendly-name -> model-id map

# Pinned protocol identifiers — recorded in every results JSON's config block.
JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_PROMPT_VERSION = "memory-research-judge-v1"

JUDGE_PROMPT_TEMPLATE = (Path(__file__).parent / "judge_prompt.txt").read_text()

# Adversarial / abstention questions are unanswerable from the conversation —
# the correct behavior is to abstain, so the judge must reward abstention, not
# the trap answer.
ABSTENTION_NOTE = (
    "\nNote: this question is NOT answerable from the conversation. The generated "
    "answer is CORRECT only if it abstains (says it does not know, or that the "
    "information is not available). Any confidently asserted answer is INCORRECT.\n"
)


def build_judge_prompt(
    question: str, gold_answer: str, generated_answer: str, *, abstention: bool = False
) -> str:
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
        abstention_note=ABSTENTION_NOTE if abstention else "",
    )


def parse_verdict(text: str) -> bool:
    """Parse the judge reply into pass/fail.

    INCORRECT is checked first because 'CORRECT' is a substring of it.
    Unparseable replies count as failures (conservative).
    """
    upper = (text or "").upper()
    if re.search(r"\bINCORRECT\b", upper):
        return False
    if re.search(r"\bCORRECT\b", upper):
        return True
    return False


def judge_one(
    judge_fn: Callable[[str], str],
    *,
    question: str,
    gold_answer: str,
    generated_answer: str,
    abstention: bool = False,
) -> bool:
    """Build the frozen prompt, call the injected judge, parse the verdict."""
    prompt = build_judge_prompt(question, gold_answer, generated_answer, abstention=abstention)
    return parse_verdict(judge_fn(prompt))


def judge_accuracy(verdicts: Sequence[bool]) -> float:
    """Fraction of CORRECT verdicts; 0.0 over an empty set."""
    return (sum(1 for v in verdicts if v) / len(verdicts)) if verdicts else 0.0


def make_judge_fn(model: str | None = None) -> Callable[[str], str]:
    """Return a callable ``prompt -> reply text`` backed by the Anthropic API.

    ``model`` accepts a friendly name (haiku / sonnet / opus) or a raw id;
    ``None`` uses the pinned default ``JUDGE_MODEL``. Lazy import so the
    harness/tests run without ``anthropic`` installed or an API key present.

    No ``thinking`` config is sent: a one-word CORRECT/INCORRECT verdict needs
    no reasoning, and on the 4.6+ tiers (sonnet / opus) omitting ``thinking``
    leaves it off — so the judge stays a cheap, deterministic classifier
    whichever model is selected.
    """
    # Cross-provider judging. The answerer is Claude for EVERY system compared
    # here, so a Claude judge means Claude grading Claude -- a conflict a
    # reviewer will raise. An OpenAI judge removes it, and because the answerer
    # is identical across systems the judge never sees which memory system
    # produced the context, so it cannot favour either. Selected by an "openai:"
    # or "gpt" prefix on `model`; everything else keeps the Anthropic path and
    # the published default is unchanged.
    _m = (model or "").strip().lower()
    if _m.startswith("openai:") or _m.startswith("gpt"):
        import os

        from openai import OpenAI  # noqa: PLC0415 — deliberately lazy

        resolved_oai = model.split(":", 1)[1] if ":" in model else model
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        def judge_fn_openai(prompt: str) -> str:
            resp = oai.chat.completions.create(
                model=resolved_oai,
                max_completion_tokens=20,
                # Was UNSET, i.e. the OpenAI default of 1.0 -- so this judge was
                # SAMPLING while the Anthropic branch below pinned 0. Measured
                # consequence on two arms whose retrieval was 99.7% identical:
                # the judge still flipped 3.7-5.4% of verdicts, giving a 95% CI
                # of +/-1.0 point on any pairwise LoCoMo delta and +/-8.4 on
                # OrgMemBench at n=73 -- wider than every graph, hop-depth and
                # consolidation effect the matrices were built to measure.
                temperature=0,
                # Determinism also needs a fixed seed: temperature=0 alone is
                # not a guarantee on the OpenAI API (batching and MoE routing
                # still admit variation). seed + temperature together give the
                # closest thing to a reproducible verdict the API offers.
                seed=int(os.environ.get("MEMORY_JUDGE_SEED", "20260904")),
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        return judge_fn_openai

    import anthropic  # noqa: PLC0415 — deliberately lazy

    resolved = resolve_model(model, JUDGE_MODEL)
    client = anthropic.Anthropic()

    def judge_fn(prompt: str) -> str:
        resp = client.messages.create(
            model=resolved,
            # Was 8. A verdict with any preamble truncates, parse_verdict then
            # fails closed to INCORRECT -- an asymmetry that only ever cost the
            # arm being judged here, since the mem0 arm judges at 20.
            max_tokens=20,
            # Was unset (Anthropic default 1.0): the JUDGE itself was sampling.
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    return judge_fn
