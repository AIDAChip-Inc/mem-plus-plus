"""Local chat demo for the memory-research engine.

A Gradio chat app that replicates the PRODUCTION per-turn team-memory hooks
(``.claude/hooks/memory_hook.py``) against this replica's in-process API:

  PRE-hook  (production ``UserPromptSubmit`` -> recall):  on each user message,
            ``recall_facts(persona, msg, k=24)`` -> a context block injected into
            the LLM prompt (mirrors the production ``additionalContext`` build).
  POST-hook (production ``Stop`` -> write):  after the turn, ``store_facts(persona,
            turn_text, mode='auto')`` extracts + stores the turn.

``chat.engine`` is the headless, dependency-light hook replication (unit-tested
against a stub client); ``chat.app`` is the Gradio UI + debug panel over it.
"""
