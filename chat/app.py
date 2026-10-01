"""Gradio chat demo over the memory-research engine.

Left: a chat pane. Right: the recall/write DEBUG panel, so a student can SEE the
production per-turn hooks fire — the memory recalled and injected on the PRE-hook,
and what the POST-hook wrote back. Wiring lives in ``chat.engine`` (``run_turn``);
this file is presentation + the Gradio glue only.

Run:  uv run python -m chat.app         (or the one-command ``run_demo.sh``)
"""
from __future__ import annotations

import html
import ipaddress
import os
import re
import sys
import tempfile
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


_ENV_KEY_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Conservative token whitelist: no whitespace, control chars, quotes, `$`, backticks,
# `;`, `#`, `=` or other shell/dotenv metacharacters. No provider prefix required so
# dummy/test keys keep working.
_API_KEY_RE = re.compile(r"[A-Za-z0-9._-]{1,512}")
_API_KEY_VAR = "ANTHROPIC_API_KEY"


def validate_api_key(value: str) -> str:
    """Return ``value`` unchanged if it is a safe key token, else raise ValueError.
    Must run before ANY file write or ``os.environ`` assignment. The message never
    contains the value."""
    if not isinstance(value, str) or not _API_KEY_RE.fullmatch(value):
        raise ValueError(
            "Invalid API key: only letters, digits, '.', '_' and '-' are allowed "
            "(max 512 chars; no spaces, quotes, newlines or shell characters)."
        )
    return value


def parse_dotenv(text: str) -> dict[str, str]:
    """Non-executing ``.env`` parser (no shell, no eval, no expansion).

    Supports ``KEY=value``, ``export KEY=value``, matched ``"..."`` / ``'...'``
    values (taken literally, no escapes or ``$`` expansion), blank lines, ``#``
    comments, and trailing `` # comment`` on unquoted values. Invalid lines are
    skipped. Later duplicates win. Keep in sync with ``load_env_file`` in run_demo.sh."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export ") or line.startswith("export\t"):
            line = line[6:].lstrip()
        key, sep, val = line.partition("=")
        key = key.strip()
        if not sep or not _ENV_KEY_RE.fullmatch(key):
            continue
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        else:
            val = re.split(r"\s#", val, maxsplit=1)[0].rstrip()
        out[key] = val
    return out


def _load_dotenv() -> None:
    """Populate os.environ from memory-research/.env (stdlib only), without
    overriding anything already set. Lets ``python -m chat.app`` work standalone,
    not only under run_demo.sh. Never commits or echoes a secret."""
    env = _ENV_PATH
    if not env.exists():
        return
    for key, val in parse_dotenv(env.read_text(encoding="utf-8")).items():
        if key not in os.environ:
            os.environ[key] = val


def persist_api_key(value: str, *, env_path: Path | None = None) -> bool:
    """Write ``ANTHROPIC_API_KEY=<value>`` into ``.env``, creating or updating
    that single line and preserving every other line. Returns True when a key was
    saved, False for an empty value. Raises ValueError (before touching the file)
    if the key fails ``validate_api_key``. The key is NEVER logged or echoed. The
    file is replaced atomically (temp file in the same dir + ``os.replace``) with
    0600 perms, so a secret is never briefly world-readable; OS/permission errors
    propagate. ``.env`` is gitignored. ``env_path`` resolves at call time so the
    live ``_ENV_PATH`` (patchable in tests) is honoured."""
    env_path = env_path or _ENV_PATH
    if not (value or "").strip():
        return False
    value = validate_api_key(value)
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    new_line = f"{_API_KEY_VAR}={value}"
    out: list[str] = []
    replaced = False
    for line in lines:
        head = line.strip().removeprefix("export ").split("=", 1)[0].strip()
        if head == _API_KEY_VAR:
            if not replaced:  # keep the first slot, drop later duplicates
                out.append(new_line)
                replaced = True
            continue
        out.append(line)
    if not replaced:
        out.append(new_line)
    fd, tmp = tempfile.mkstemp(dir=env_path.parent, prefix=f".{env_path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            os.fchmod(fh.fileno(), 0o600)
            fh.write("\n".join(out) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, env_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
    return True


def require_loopback(host: str) -> str:
    """Return ``host`` if it is a loopback address/name, else raise ValueError.
    This demo has no authentication, so it must only listen locally."""
    h = (host or "").strip().strip("[]")
    if h.lower() == "localhost":
        return host
    try:
        if ipaddress.ip_address(h).is_loopback:
            return host
    except ValueError:
        pass
    raise ValueError(
        f"Refusing to bind Gradio to non-loopback host {host!r}: this local-only demo has "
        "no authentication. Use 127.0.0.1, ::1 or localhost (see SECURITY.md)."
    )


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
        try:
            validate_api_key(api_key)
        except ValueError as exc:  # reject before any env assignment; keep the typed message
            return history, user_msg, render_debug(None, error=str(exc))
        os.environ[_API_KEY_VAR] = api_key
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
    key = api_key or ""
    if not key.strip():
        return "Enter a key first, then press Save."
    try:
        validate_api_key(key)
    except ValueError as exc:  # never assign env or write the file for an invalid key
        return f"✗ {exc}"
    try:
        persist_api_key(key)
    except OSError as exc:
        return f"✗ Could not save to `{_ENV_PATH.name}` ({type(exc).__name__}); key not saved."
    os.environ[_API_KEY_VAR] = key
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
    try:
        require_loopback(name)
    except ValueError as exc:
        sys.exit(f"error: {exc}")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    print(f"\n  memory-research chat demo → http://{name}:{port}\n")
    # share=False explicit: GRADIO_SHARE=1 would otherwise open a public tunnel.
    build_demo().launch(server_name=name, server_port=port, show_api=False, share=False)


if __name__ == "__main__":
    main()
