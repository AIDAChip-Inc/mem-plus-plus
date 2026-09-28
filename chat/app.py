"""Gradio chat demo over the memory-research engine.

Left: a chat pane. Right: the recall/write DEBUG panel, so a student can SEE the
production per-turn hooks fire — the memory recalled and injected on the PRE-hook,
and what the POST-hook wrote back. Wiring lives in ``chat.engine`` (``run_turn``);
this file is presentation + the Gradio glue only.

Run:  uv run python -m chat.app         (or the one-command ``run_demo.sh``)
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import gradio as gr

from chat.engine import DEFAULT_PERSONA, RECALL_K, EngineBackend, TurnResult, run_turn
from chat.eval_tab import EVAL_CSS, build_eval_tab
from chat.memory_tab import MEM_CSS, build_memory_tab
from chat.system_tab import SYS_CSS, build_system_tab
from memory.config import MEMORY_LLM_MODELS

_ROOT = Path(__file__).resolve().parent.parent  # memory-research/
_ENV_PATH = _ROOT / ".env"
_BACKEND = EngineBackend()

# Selectable reply/extraction models (friendly names -> ids in memory.config).
# "haiku" is the pinned default (zero-diff vs the prior hardcoded behavior).
DEFAULT_MODEL = "haiku"


def _load_dotenv() -> None:
    """Populate os.environ from memory-research/.env (stdlib only), without
    overriding anything already set. Lets ``python -m chat.app`` work standalone,
    not only under run_demo.sh. Never commits or echoes a secret."""
    env = _ENV_PATH
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, val = s.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def persist_api_key(value: str, *, env_path: Path | None = None) -> bool:
    """Write ``ANTHROPIC_API_KEY=<value>`` into ``.env``, creating or updating
    that single line and preserving every other line. Returns True when a key was
    saved, False for an empty value. The key is NEVER logged or echoed; the file
    is written 0600 (owner-only) since it holds a secret, and ``.env`` is
    gitignored so it is never committed. ``env_path`` resolves at call time so the
    live ``_ENV_PATH`` (patchable in tests) is honoured."""
    env_path = env_path or _ENV_PATH
    value = (value or "").strip()
    if not value:
        return False
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    new_line = f"ANTHROPIC_API_KEY={value}"
    replaced = False
    for i, line in enumerate(lines):
        if line.strip().split("=", 1)[0].strip() == "ANTHROPIC_API_KEY":
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        env_path.chmod(0o600)  # secret file — restrict to owner
    except OSError:
        pass  # best-effort on platforms without POSIX perms
    return True


# ── debug-panel rendering (the "show the recall" surface) ──────────────────────

def _badge(text: str, kind: str) -> str:
    return f'<span class="mr-badge mr-badge-{kind}">{html.escape(text)}</span>'


def _facts_table(result: TurnResult) -> str:
    """The recalled-facts table — a full superset of production's recall block.

    Surfaces, per row: scope (own/team), salience (the ranking score), the
    match-vs-recency REASON the row was kept, hit_count, and the TWO distinct
    timestamps — ``event`` (occurred_at, when the fact happened) vs ``written``
    (created_at, the write/recency time the recency-decay leg keys off)."""
    if not result.recalled:
        return '<p class="mr-empty">No memories recalled for this turn.</p>'
    head = (
        "<table class='mr-table'><thead><tr>"
        "<th>#</th><th>scope</th>"
        "<th title='Ranking score: matched-band + relevance + hits + recency-decay'>salience</th>"
        "<th title='Why the row was kept: matched the query, or held by the recency reserve'>reason</th>"
        "<th title='Times this memory has been recalled'>hits</th>"
        "<th title='EVENT time — when the fact happened (occurred_at)'>event</th>"
        "<th title='WRITE / RECENCY time — when the row was stored (created_at)'>written</th>"
        "<th>summary</th>"
        "</tr></thead><tbody>"
    )
    body = []
    for f in result.recalled:
        scope = _badge(f.scope, "team" if f.scope == "team" else "own") if f.scope != "-" else "-"
        salience = "-" if f.score is None else f"{f.score:.1f}"
        if f.matched is None:
            reason = "-"
        else:
            tag = "match" if f.matched else "recency"
            reason = _badge(tag, tag)
        event = html.escape(f.occurred_at or "-")
        written = html.escape(f.created_at or "-")
        summary = html.escape(f.summary)
        if f.by:
            summary += f' <span class="mr-by">by {html.escape(f.by)}</span>'
        body.append(
            f"<tr><td>{f.rank}</td><td>{scope}</td>"
            f"<td class='mr-mono'>{salience}</td><td>{reason}</td>"
            f"<td class='mr-mono'>{f.hit_count}</td>"
            f"<td class='mr-mono'>{event}</td><td class='mr-mono'>{written}</td>"
            f"<td>{summary}</td></tr>"
        )
    return head + "".join(body) + "</tbody></table>"


def _write_line(result: TurnResult) -> str:
    s = result.stored or {}
    written = s.get("written", 0)
    mode = s.get("mode", "?")
    if s.get("degraded"):
        reason = html.escape(str(s.get("reason") or "extraction unavailable"))
        return f'{_badge("degraded", "recency")} mode <code>{html.escape(str(mode))}</code> — {reason}'
    noun = "fact" if written == 1 else "facts"
    return f'{_badge(f"{written} {noun}", "match" if written else "own")} written · mode <code>{html.escape(str(mode))}</code>'


def render_debug(result: TurnResult | None, error: str | None = None) -> str:
    if error:
        return (
            '<div class="mr-panel"><div class="mr-sec-title">Memory unavailable</div>'
            f'<p class="mr-err">{html.escape(error)}</p>'
            '<p class="mr-hint">Start the local Postgres and migrate: '
            '<code>docker compose up -d</code> then <code>uv run alembic upgrade head</code> '
            '(or just run <code>./run_demo.sh</code>).</p></div>'
        )
    if result is None:
        return (
            '<div class="mr-panel"><div class="mr-sec-title">Recall debug</div>'
            '<p class="mr-hint">Send a message to see the memory recalled and injected '
            'on the PRE-hook, then what the POST-hook writes back.</p></div>'
        )
    llm = _badge("LLM reply", "match") if result.llm_used else _badge("recall-only (no key)", "recency")
    legend = ('<p class="mr-hint"><b>event</b> = when the fact happened '
              "(occurred_at) · <b>written</b> = when it was stored / recency time "
              "(created_at) · <b>salience</b> = ranking score · <b>reason</b> = "
              "matched the query vs. held by the recency reserve.</p>")
    return f"""<div class="mr-panel">
  <div class="mr-sec-title">PRE-hook · recalled &amp; injected {llm}</div>
  {_facts_table(result)}
  {legend if result.recalled else ""}
  <div class="mr-sec-title">Injected context block</div>
  <pre class="mr-block">{html.escape(result.context_block)}</pre>
  <div class="mr-sec-title">POST-hook · write</div>
  <p class="mr-write">{_write_line(result)}</p>
</div>"""


# ── event handlers ─────────────────────────────────────────────────────────────

def on_submit(user_msg: str, history: list[dict], persona: str, k: float,
              model: str, api_key: str):
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return history, "", render_debug(None)
    if (api_key or "").strip():
        os.environ["ANTHROPIC_API_KEY"] = api_key.strip()
    history = list(history or [])
    try:
        # ``model`` (haiku/sonnet/opus) drives BOTH the reply and the POST-hook store.
        result = run_turn(_BACKEND, persona or DEFAULT_PERSONA, user_msg,
                          k=int(k), model=model or None)
    except Exception as exc:  # DB down / not migrated / engine error — surface, don't crash
        history += [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": "Memory backend unavailable — see the debug panel."},
        ]
        return history, "", render_debug(None, error=f"{type(exc).__name__}: {exc}")
    history += [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": result.reply},
    ]
    return history, "", render_debug(result)


def on_save_key(api_key: str) -> str:
    """Persist the entered key to .env (and use it now). Confirmation only — the
    key material is never echoed back."""
    key = (api_key or "").strip()
    if not key:
        return "Enter a key first, then press Save."
    os.environ["ANTHROPIC_API_KEY"] = key
    persist_api_key(key)
    return f"✓ Saved to `{_ENV_PATH.name}` (owner-only, gitignored) — future launches auto-load it."


def _status_line() -> str:
    db = "set" if os.environ.get("MEMORY_DATABASE_URL") else "MISSING"
    key = "set" if (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")) else "unset (recall-only)"
    return f"DB: {db} · LLM key: {key}"


_CSS = """
:root { --mr-fg:#1c1e26; --mr-muted:#6b7280; --mr-line:#e5e7eb; --mr-panel:#f7f8fa;
        --mr-accent:#3b6ea5; --mr-own:#e8eef6; --mr-team:#f0e9f7; --mr-match:#e3f2e6;
        --mr-recency:#fbeee0; --mr-code:#eef0f3; }
/* Follow GRADIO's active theme (it toggles `.dark`), NOT the OS preference —
   OS-vs-Gradio mismatch was painting light text on Gradio's light background. */
.dark { --mr-fg:#e5e7eb; --mr-muted:#9aa3b2; --mr-line:#2b2f3a; --mr-panel:#171a21;
        --mr-accent:#8ab4e8; --mr-own:#20303f; --mr-team:#2c2440; --mr-match:#1f3326;
        --mr-recency:#3a2c1c; --mr-code:#1e222b; }
.mr-panel { border:1px solid var(--mr-line); border-radius:12px; padding:16px 18px;
            background:var(--mr-panel); color:var(--mr-fg); font-size:14px; }
.mr-sec-title { font-weight:600; letter-spacing:.01em; margin:14px 0 8px; color:var(--mr-fg);
                border-top:1px solid var(--mr-line); padding-top:12px; }
.mr-sec-title:first-child { border-top:0; padding-top:0; margin-top:0; }
.mr-table { width:100%; border-collapse:collapse; font-size:13px; }
.mr-table th { text-align:left; color:var(--mr-muted); font-weight:500; padding:4px 8px;
               border-bottom:1px solid var(--mr-line); }
.mr-table td { padding:6px 8px; border-bottom:1px solid var(--mr-line); vertical-align:top; }
.mr-mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; white-space:nowrap; }
.mr-block { background:var(--mr-code); border-radius:8px; padding:10px 12px; font-size:12.5px;
            white-space:pre-wrap; overflow-x:auto; color:var(--mr-fg); }
.mr-badge { display:inline-block; padding:1px 8px; border-radius:999px; font-size:11px;
            font-weight:600; }
.mr-badge-own { background:var(--mr-own); color:var(--mr-accent); }
.mr-badge-team { background:var(--mr-team); color:#7c5cbf; }
.mr-badge-match { background:var(--mr-match); color:#2e7d43; }
.mr-badge-recency { background:var(--mr-recency); color:#b5701a; }
.mr-by { color:var(--mr-muted); font-size:11px; }
.mr-empty, .mr-hint, .mr-err { color:var(--mr-muted); font-size:13px; }
.mr-err { color:#c0392b; } .mr-write code, .mr-hint code { background:var(--mr-code);
          padding:1px 5px; border-radius:4px; font-size:12px; }
.mr-subtitle { color:var(--mr-muted); font-size:13.5px; margin-top:-4px; }
"""


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="memory-research · chat demo", css=_CSS + EVAL_CSS + MEM_CSS + SYS_CSS,
                   theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
        gr.Markdown("## AIDAChip team-memory — live chat demo")
        gr.Markdown(
            "Each turn replays the production per-turn hooks against the local replica: "
            "**PRE-hook** recalls the persona's memory (`recall_facts`, k=24) and injects "
            "it into the prompt; the reply is generated with that context; the **POST-hook** "
            "extracts and stores the finished turn (`store_facts`, mode=auto).",
            elem_classes=["mr-subtitle"],
        )

        with gr.Accordion("Settings & API key", open=False):
            with gr.Row():
                persona = gr.Textbox(value=DEFAULT_PERSONA, label="Persona (agent_type)",
                                     scale=2, info="Recall + write scope for this chat.")
                k = gr.Slider(1, 50, value=RECALL_K, step=1, label="Recall top-k", scale=2)
                model = gr.Dropdown(
                    list(MEMORY_LLM_MODELS), value=DEFAULT_MODEL, label="LLM model", scale=2,
                    info="Drives BOTH the reply and the post-hook memory extraction.")
            api_key = gr.Textbox(
                label="ANTHROPIC_API_KEY", type="password", placeholder="sk-ant-…",
                info="Used this session immediately. Press Save to also write it to .env "
                     "(gitignored) so the next launch auto-loads it. Blank = recall-only view.",
            )
            with gr.Row():
                save_key = gr.Button("Save key for future launches", size="sm", scale=1)
                key_saved = gr.Markdown(value="", elem_classes=["mr-subtitle"])
            status = gr.Markdown(value=_status_line(), elem_classes=["mr-subtitle"])

        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(type="messages", height=460, label="Chat",
                                             avatar_images=None, show_copy_button=True)
                        with gr.Row():
                            msg = gr.Textbox(
                                placeholder="Tell the assistant something, then ask about it…",
                                show_label=False, scale=8, autofocus=True)
                            send = gr.Button("Send", variant="primary", scale=1)
                        clear = gr.Button("Clear chat", size="sm")
                    with gr.Column(scale=2):
                        debug = gr.HTML(value=render_debug(None))
            with gr.Tab("Evaluation"):
                build_eval_tab()
            with gr.Tab("Memory"):
                build_memory_tab()
            with gr.Tab("System"):
                build_system_tab()

        inputs = [msg, chatbot, persona, k, model, api_key]
        outputs = [chatbot, msg, debug]
        send.click(on_submit, inputs, outputs).then(_status_line, None, status)
        msg.submit(on_submit, inputs, outputs).then(_status_line, None, status)
        clear.click(lambda: ([], render_debug(None)), None, [chatbot, debug])
        save_key.click(on_save_key, api_key, key_saved).then(_status_line, None, status)
    return demo


def main() -> None:
    _load_dotenv()
    name = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    print(f"\n  memory-research chat demo → http://{name}:{port}\n")
    build_demo().launch(server_name=name, server_port=port, show_api=False)


if __name__ == "__main__":
    main()
