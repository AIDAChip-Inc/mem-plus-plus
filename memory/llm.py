"""Pluggable one-shot LLM caller for extraction — degrades cleanly if absent.

Production routes extraction through the Claude Agent SDK (``app.services.llm``).
This replica uses the official Anthropic Python SDK directly, resolving
credentials from the environment (``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN``,
or an ``ant auth login`` profile). When the SDK is not installed OR no credential
is available OR the call fails, ``llm_call`` returns ``None`` — never raises — so
the write path degrades to "store nothing" rather than crashing.

Consumers: ``store_facts`` (extraction) and the consolidation write path
(``memory.consolidation``), which uses the ``llm_complete`` / ``structured_call``
shims below and the ``llm_available`` probe to choose its faithful LLM path vs a
degraded deterministic-union fallback. ``store_facts_verbatim`` never touches this
module.
"""
from __future__ import annotations

import logging
import os

from memory import config

logger = logging.getLogger(__name__)

# Credentials the Anthropic SDK reads from the environment. The probe only checks
# these two — a profile-only (``ant auth login``) session is NOT detected and
# conservatively reads as "unavailable" (⇒ consolidation takes its safe
# deterministic-union path), which never corrupts data, only forgoes LLM synthesis.
_CRED_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def llm_call(
    prompt: str,
    *,
    system_prompt: str = "",
    model: str | None = None,
    max_tokens: int = 256,
) -> str | None:
    """One-shot text completion, or ``None`` when unavailable. Never raises.

    ``model`` accepts a friendly name (haiku/sonnet/opus) OR a full model id;
    ``None`` falls back to the Haiku default (``config.resolve_model``), so the
    behavior is unchanged when a caller does not pass one.
    """
    resolved_model = config.resolve_model(model)
    try:
        import anthropic
    except ImportError:
        logger.warning("anthropic SDK not installed — extraction unavailable")
        return None
    try:
        client = anthropic.Anthropic()  # resolves key/token/profile from env
        kwargs: dict = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        message = client.messages.create(**kwargs)
        text = "".join(b.text for b in message.content if b.type == "text").strip()
        return text or None
    except Exception as exc:  # missing credential, network, API error — all soft-fail
        logger.warning("extraction llm_call failed: %s", exc)
        return None


def llm_available() -> bool:
    """True iff the Anthropic SDK is importable AND a credential env var is set.

    Cheap, deterministic environment-level probe (no network call). Consolidation
    uses it to pick the faithful LLM path (Haiku synthesis + supersession
    classification) vs the degraded deterministic-union fallback — so an offline
    run (the unit suite, a no-key self-verify) is reproducible and never blocks on
    an LLM. See ``_CRED_ENV_VARS`` for the profile-only caveat.
    """
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return any(os.environ.get(v) for v in _CRED_ENV_VARS)


def llm_complete(
    prompt: str, *, model: str | None = None, max_tokens: int = 256
) -> tuple[str, dict]:
    """One-shot completion -> ``(text, usage)`` (mirrors production ``llm_complete``).

    Reuses ``llm_call``; returns ``("", {})`` on any failure (never raises), so the
    consolidation caller treats an empty result as a merge failure exactly as
    production does. ``usage`` is a stub here (no token accounting in the replica).
    """
    text = llm_call(prompt, model=model, max_tokens=max_tokens)
    return (text or "", {})


def structured_call(
    prompt: str,
    *,
    tool_name: str,
    tool_description: str,
    input_schema: dict,
    system_prompt: str = "",
    model: str | None = None,
    max_tokens: int = 256,
) -> dict | None:
    """Forced-tool structured call -> schema-shaped dict, or ``None`` (never raises).

    Faithful to production ``app.services.llm.structured_call`` (team standard:
    never regex-on-prose): a single forced tool whose ``input_schema`` the model
    must fill. Returns the tool-use ``input`` dict; ``None`` on missing SDK, missing
    credential, truncation (``stop_reason == 'max_tokens'``), or any error — a
    missing structured result is "no data", never a fabricated value.
    """
    resolved_model = config.resolve_model(model)
    try:
        import anthropic  # noqa: F401
    except ImportError:
        logger.warning("anthropic SDK not installed — structured_call unavailable")
        return None
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(
            model=resolved_model,
            max_tokens=max_tokens,
            system=system_prompt or "You are a helpful assistant.",
            tools=[{
                "name": tool_name,
                "description": tool_description,
                "input_schema": input_schema,
            }],
            tool_choice={"type": "tool", "name": tool_name},
            messages=[{"role": "user", "content": prompt}],
        )
        if getattr(response, "stop_reason", None) == "max_tokens":
            logger.warning(
                "structured_call: max_tokens=%d truncated the forced tool_use for %r",
                max_tokens, tool_name,
            )
            return None
        for block in response.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        return None
    except Exception as exc:
        logger.warning("structured_call failed: %s", exc)
        return None
