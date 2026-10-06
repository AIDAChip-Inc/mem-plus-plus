"""LLM-judge answer metric — selectable model + selectable, versioned prompt.

LLM-judge variance is the #1 reproducibility risk, so the protocol is frozen
BY DEFAULT while the model stays user-selectable:

  * JUDGE_MODEL is the pinned DEFAULT (a dated model id — a score is only
    comparable under the identical judge). We reuse HAND's production eval judge
    (``claude-haiku-4-5-20251001``) so default numbers sit on the same baseline.
    ``make_judge_fn`` now accepts a friendly name (haiku / sonnet / opus) or a
    raw id to swap the judge model; ``None`` keeps the pin. The chosen model is
    RECORDED in every results-JSON config block (see ``harness.run_benchmark``).
  * The judge PROMPT is one of three published, versioned prompts
    (``JUDGE_PROMPTS``), selected per run with ``--judge-prompt``:

      - ``repo``: the in-house prompt, byte-exact in ``judge_prompt.txt``
        (``JUDGE_PROMPT_VERSION``). Stricter than the two below.
      - ``mem0``: Mem0's LoCoMo LLM-judge prompt (``ACCURACY_PROMPT``), copied
        verbatim. Used by the paper for LoCoMo and by every copied LoCoMo
        baseline. JSON reply ``{"label": "CORRECT" | "WRONG"}``.
      - ``longmemeval``: the official LongMemEval per-question-type answer-check
        prompts (``get_anscheck_prompt``), copied verbatim, including the
        abstention prompt. Used by the paper for LongMemEval_S (following Zep).
        Reply ``yes`` / ``no``.

    The default per benchmark matches the paper (``default_judge_prompt``):
    locomo -> mem0, longmemeval -> longmemeval, everything else -> repo. The
    prompt name and its version are recorded in every results-JSON config.

Prompt building + verdict parsing + accuracy are pure and unit tested. The
actual model call lives in ``make_judge_fn`` (lazy SDK import) so the harness
can be exercised with a stub judge and no API key.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from pathlib import Path

from .llm import resolve_model  # shared friendly-name -> model-id map

# Pinned protocol identifiers — recorded in every results JSON's config block.
JUDGE_MODEL = "claude-haiku-4-5-20251001"
JUDGE_PROMPT_VERSION = "memory-research-judge-v1"

JUDGE_PROMPT_TEMPLATE = (Path(__file__).parent / "judge_prompt.txt").read_text()

# ── Upstream judge prompts (copied verbatim; do not edit) ─────────────────────
#
# Mem0 LoCoMo judge. Source: mem0ai/mem0, evaluation/metrics/llm_judge.py,
# ACCURACY_PROMPT, at commit aae5989e78a6188b3b047c104d960c9ad0927e75 (the last
# commit touching that file before mem0 moved its evaluation code out of the
# repo).
# https://github.com/mem0ai/mem0/blob/aae5989e78a6188b3b047c104d960c9ad0927e75/evaluation/metrics/llm_judge.py
# Copyright (c) Mem0 authors. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Reproduced unmodified; the
# surrounding code in this file is not part of the upstream work.
# Trailing spaces and the curly quotes are part of the upstream text; a test
# pins its sha256.
MEM0_ACCURACY_PROMPT = (
    "\n"
    "Your task is to label an answer to a question as ’CORRECT’ or ’WRONG’. You will be given the following data:\n"
    "    (1) a question (posed by one user to another user), \n"
    "    (2) a ’gold’ (ground truth) answer, \n"
    "    (3) a generated answer\n"
    "which you will score as CORRECT/WRONG.\n"
    "\n"
    "The point of the question is to ask about something one user should know about the other user based on their prior conversations.\n"
    "The gold answer will usually be a concise and short answer that includes the referenced topic, for example:\n"
    "Question: Do you remember what I got the last time I went to Hawaii?\n"
    "Gold answer: A shell necklace\n"
    "The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT. \n"
    "\n"
    'For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it\'s the same date.\n'
    "\n"
    "Now it's time for the real question:\n"
    "Question: {question}\n"
    "Gold answer: {gold_answer}\n"
    "Generated answer: {generated_answer}\n"
    "\n"
    "First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG. \n"
    "Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.\n"
    "\n"
    'Just return the label CORRECT or WRONG in a json format with the key as "label".\n'
)

# LongMemEval answer-check prompts. Source: xiaowu0162/LongMemEval,
# src/evaluation/evaluate_qa.py, get_anscheck_prompt(), at commit
# d6dc8b50a2d9ac0c99485ea28fa5755c62414c34.
# https://github.com/xiaowu0162/LongMemEval/blob/d6dc8b50a2d9ac0c99485ea28fa5755c62414c34/src/evaluation/evaluate_qa.py
# Copyright (c) 2024 Di Wu. Licensed under the MIT License. Reproduced
# unmodified; a test pins each template's sha256.
_LME_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. "
    "Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the answer, answer no. "
    "\n"
    "\n"
    "Question: {}\n"
    "\n"
    "Correct Answer: {}\n"
    "\n"
    "Model Response: {}\n"
    "\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. "
    "Otherwise, answer no. "
    "If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. "
    "If the response only contains a subset of the information required by the answer, answer no. "
    "In addition, do not penalize off-by-one errors for the number of days. "
    "If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. "
    "\n"
    "\n"
    "Question: {}\n"
    "\n"
    "Correct Answer: {}\n"
    "\n"
    "Model Response: {}\n"
    "\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_KNOWLEDGE_UPDATE = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. "
    "Otherwise, answer no. "
    "If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n"
    "\n"
    "Question: {}\n"
    "\n"
    "Correct Answer: {}\n"
    "\n"
    "Model Response: {}\n"
    "\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, and a response from a model. "
    "Please answer yes if the response satisfies the desired response. "
    "Otherwise, answer no. "
    "The model does not need to reflect all the points in the rubric. "
    "The response is correct as long as it recalls and utilizes the user's personal information correctly.\n"
    "\n"
    "Question: {}\n"
    "\n"
    "Rubric: {}\n"
    "\n"
    "Model Response: {}\n"
    "\n"
    "Is the model response correct? Answer yes or no only."
)
_LME_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response from a model. "
    "Please answer yes if the model correctly identifies the question as unanswerable. "
    "The model could say that the information is incomplete, or some other information is given but the asked information is not.\n"
    "\n"
    "Question: {}\n"
    "\n"
    "Explanation: {}\n"
    "\n"
    "Model Response: {}\n"
    "\n"
    "Does the model correctly identify the question as unanswerable? Answer yes or no only."
)

# Prompt name -> recorded version string.
JUDGE_PROMPT_VERSIONS = {
    "repo": JUDGE_PROMPT_VERSION,
    "mem0": "mem0-llm-judge@aae5989",
    "longmemeval": "longmemeval-anscheck@d6dc8b5",
}
JUDGE_PROMPTS = tuple(JUDGE_PROMPT_VERSIONS)

# The paper's judging setup: LoCoMo with Mem0's judge, LongMemEval_S with the
# official per-type prompts. Everything else keeps the in-house prompt.
_DEFAULT_JUDGE_PROMPT_BY_BENCHMARK = {"locomo": "mem0", "longmemeval": "longmemeval"}

# Max reply tokens per prompt. mem0 asks for a one-sentence explanation plus a
# JSON label; LongMemEval's own script uses max_tokens=10 for a yes/no.
_JUDGE_MAX_TOKENS = {"repo": 20, "mem0": 256, "longmemeval": 10}

_LME_TASK_TEMPLATES = {
    "single-session-user": _LME_DEFAULT,
    "single-session-assistant": _LME_DEFAULT,
    "multi-session": _LME_DEFAULT,
    "temporal-reasoning": _LME_TEMPORAL,
    "knowledge-update": _LME_KNOWLEDGE_UPDATE,
    "single-session-preference": _LME_PREFERENCE,
}


def default_judge_prompt(benchmark: str) -> str:
    """The judge prompt the paper used for ``benchmark`` (``repo`` if none)."""
    return _DEFAULT_JUDGE_PROMPT_BY_BENCHMARK.get(benchmark, "repo")


def _check_prompt_name(judge_prompt: str) -> None:
    if judge_prompt not in JUDGE_PROMPT_VERSIONS:
        raise ValueError(f"unknown judge prompt {judge_prompt!r}; choose from {JUDGE_PROMPTS}")


def build_longmemeval_prompt(
    task: str, question: str, answer: str, response: str, *, abstention: bool = False
) -> str:
    """Upstream ``get_anscheck_prompt``. ``task`` is the LongMemEval question
    type; the adapter's underscore form (``temporal_reasoning``) is accepted."""
    if abstention:
        return _LME_ABSTENTION.format(question, answer, response)
    template = _LME_TASK_TEMPLATES.get(task.replace("_", "-"))
    if template is None:
        raise ValueError(
            f"no LongMemEval judge prompt for question type {task!r}; the "
            "'longmemeval' judge prompt only applies to LongMemEval questions"
        )
    return template.format(question, answer, response)

# Adversarial / abstention questions are unanswerable from the conversation —
# the correct behavior is to abstain, so the judge must reward abstention, not
# the trap answer.
ABSTENTION_NOTE = (
    "\nNote: this question is NOT answerable from the conversation. The generated "
    "answer is CORRECT only if it abstains (says it does not know, or that the "
    "information is not available). Any confidently asserted answer is INCORRECT.\n"
)


def build_judge_prompt(
    question: str,
    gold_answer: str,
    generated_answer: str,
    *,
    abstention: bool = False,
    judge_prompt: str = "repo",
    category: str | None = None,
) -> str:
    """Render the selected judge prompt. ``category`` is the LongMemEval
    question type and is required for ``judge_prompt='longmemeval'``. The mem0
    prompt has no abstention variant, so ``abstention`` is ignored for it."""
    _check_prompt_name(judge_prompt)
    if judge_prompt == "mem0":
        return MEM0_ACCURACY_PROMPT.format(
            question=question, gold_answer=gold_answer, generated_answer=generated_answer
        )
    if judge_prompt == "longmemeval":
        return build_longmemeval_prompt(
            category or "", question, gold_answer, generated_answer, abstention=abstention
        )
    return JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
        abstention_note=ABSTENTION_NOTE if abstention else "",
    )


_JSON_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_LABEL_FIELD = re.compile(r'"label"\s*:\s*"([A-Za-z]+)"')


def _parse_mem0_label(text: str) -> bool:
    """Mem0 reads ``json.loads(extract_json(reply))["label"] == "CORRECT"``.
    Same rule here; a reply that is not valid JSON falls back to the first
    ``"label": "..."`` field, and anything else fails closed."""
    raw = _JSON_FENCE.sub("", (text or "").strip())
    try:
        label = json.loads(raw).get("label")
    except (ValueError, AttributeError):
        m = _LABEL_FIELD.search(text or "")
        label = m.group(1) if m else None
    return label == "CORRECT"


def parse_verdict(text: str, judge_prompt: str = "repo") -> bool:
    """Parse the judge reply into pass/fail for the selected prompt.

    repo: INCORRECT is checked first because 'CORRECT' is a substring of it.
    mem0: the JSON ``label`` must equal ``CORRECT``.
    longmemeval: upstream rule, ``'yes' in reply.lower()``.
    Unparseable replies count as failures (conservative).
    """
    _check_prompt_name(judge_prompt)
    if judge_prompt == "mem0":
        return _parse_mem0_label(text)
    if judge_prompt == "longmemeval":
        return "yes" in (text or "").lower()
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
    judge_prompt: str = "repo",
    category: str | None = None,
) -> bool:
    """Build the selected prompt, call the injected judge, parse the verdict."""
    prompt = build_judge_prompt(
        question, gold_answer, generated_answer,
        abstention=abstention, judge_prompt=judge_prompt, category=category,
    )
    return parse_verdict(judge_fn(prompt), judge_prompt)


def judge_accuracy(verdicts: Sequence[bool]) -> float:
    """Fraction of CORRECT verdicts; 0.0 over an empty set."""
    return (sum(1 for v in verdicts if v) / len(verdicts)) if verdicts else 0.0


def make_judge_fn(model: str | None = None, judge_prompt: str = "repo") -> Callable[[str], str]:
    """Return a callable ``prompt -> reply text`` backed by the Anthropic API.

    ``judge_prompt`` sets the call parameters the prompt needs (reply length;
    JSON mode for ``mem0`` on OpenAI, as Mem0's own judge does). Temperature is
    0 for every prompt.

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
    _check_prompt_name(judge_prompt)
    max_tokens = _JUDGE_MAX_TOKENS[judge_prompt]
    _m = (model or "").strip().lower()
    if _m.startswith("openai:") or _m.startswith("gpt"):
        import os

        from openai import OpenAI  # noqa: PLC0415 — deliberately lazy

        resolved_oai = model.split(":", 1)[1] if ":" in model else model
        oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

        # Mem0's judge requests a JSON object reply; mirror it.
        extra = {"response_format": {"type": "json_object"}} if judge_prompt == "mem0" else {}

        def judge_fn_openai(prompt: str) -> str:
            resp = oai.chat.completions.create(
                model=resolved_oai,
                max_completion_tokens=max_tokens,
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
                **extra,
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
            # (20 for the repo prompt; the mem0/longmemeval prompts set their own.)
            max_tokens=max_tokens,
            # Was unset (Anthropic default 1.0): the JUDGE itself was sampling.
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")

    return judge_fn
