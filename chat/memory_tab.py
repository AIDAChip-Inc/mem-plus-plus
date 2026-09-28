"""Memory tab — browse the ``agent_memory`` DB directly.

Presentation + Gradio glue over the SHIPPED read-only query API; it reimplements
no query logic. The flow reuses, verbatim:

  * ``memory.browse.browse_memories`` — the strictly read-only, filterable SELECT
    over ``agent_memory`` (never bumps ``hit_count``, never writes). Lexical or
    pgvector-semantic text, plus structured filters + pagination.

Two axes are ORTHOGONAL (Awsi's design note, mirrored from ``browse.py``): ``persona``
is the ROLE axis (``agent_type`` — which persona authored the memory) and ``scope`` is
the USER axis (``own``/``team``/``all``). They are independent filters here — NOT the
coupled two-section split recall uses — so any persona can be viewed in any scope.

The core (``run_browse``) is dependency-injectable — a headless test drives the
filter → ``browse_memories`` kwargs mapping with a stub (or the real SQLite-degraded
path) needing no Postgres and no API key.
"""
from __future__ import annotations

import html
from collections.abc import Callable

import gradio as gr

from memory.browse import browse_memories
from memory.consolidation import consolidate
from memory.mutate import merge_memories, reset_db, supersede_memory

SCOPE_CHOICES = ["all", "own", "team"]
ORDER_CHOICES = ["recency", "hits", "occurred"]
DEFAULT_LIMIT = 50

# Memory-tab-only classes, composed into the app's shared <style> (app.py) so the
# whole demo reads as one system. Reuses the .mr-* vars (light + dark) verbatim.
MEM_CSS = """
.mr-tablewrap { overflow-x:auto; border:1px solid var(--mr-line); border-radius:12px; }
.mr-tablewrap .mr-table { margin:0; }
.mr-tablewrap .mr-table th { position:sticky; top:0; background:var(--mr-panel);
                             padding:8px 10px; white-space:nowrap; z-index:1; }
.mr-tablewrap .mr-table td { padding:8px 10px; }
.mr-tablewrap .mr-table tr:hover td { background:var(--mr-code); }
.mr-summary { min-width:280px; max-width:520px; }
.mr-chips { display:flex; flex-wrap:wrap; gap:4px; }
.mr-chip { display:inline-block; padding:1px 7px; border-radius:999px; font-size:11px;
           background:var(--mr-code); color:var(--mr-muted); white-space:nowrap; }
.mr-count { color:var(--mr-muted); font-size:12.5px; margin:2px 0 8px; }
/* Full-cell table — the complete agent_memory row, grouped + tooltip'd, horizontally
   scrollable (leans on .mr-tablewrap). Group row scrolls; the field-name row stays
   pinned (overrides the shared sticky-th rule for the group row only). */
.mr-fullcell { font-size:12.5px; }
.mr-fullcell th, .mr-fullcell td { white-space:nowrap; }
.mr-tablewrap .mr-fullcell .mr-grouprow th { position:static; top:auto;
     text-transform:uppercase; font-size:10px; letter-spacing:.05em; font-weight:600;
     color:var(--mr-accent); background:var(--mr-panel); padding:6px 10px 3px;
     border-bottom:2px solid var(--mr-line); }
.mr-fullcell .mr-colrow th { font-size:11px; }
.mr-group-start { border-left:2px solid var(--mr-line); }
.mr-celltext { display:inline-block; max-width:320px; overflow:hidden;
               text-overflow:ellipsis; white-space:nowrap; vertical-align:bottom; }
.mr-id { cursor:help; }
.mr-bool-t { color:#2e7d43; font-weight:600; }
.mr-bool-f { color:var(--mr-muted); }
/* Action bar — makes the WRITE actions below the read-only table unmissable, and
   styles the confirm/error notice panel (_action_panel kind=warn/err). Reuses the
   shared .mr-* vars + the .mr-chip pill so it reads as one system in light + dark. */
.mr-actionbar { border:1px solid var(--mr-line); border-left:4px solid var(--mr-accent);
                border-radius:12px; background:var(--mr-panel); color:var(--mr-fg);
                padding:12px 16px; margin:10px 0 4px; }
.mr-actionbar .mr-actions-title { font-weight:700; font-size:15px; margin:0 0 8px;
                letter-spacing:.01em; }
.mr-actionlegend { display:flex; flex-wrap:wrap; align-items:center; gap:4px 8px;
                margin:0; font-size:12.5px; color:var(--mr-muted); }
.mr-actionlegend .mr-chip { background:var(--mr-code); color:var(--mr-accent); font-weight:600; }
.mr-actionbar .mr-hint { margin:8px 0 0; }
.mr-warn { border:1px solid var(--mr-line); border-left:4px solid #b5701a;
           border-radius:12px; background:var(--mr-recency); color:var(--mr-fg);
           padding:14px 16px; font-size:14px; }
/* Quick (one-click default) vs explicit (configurable) paths for supersede/merge.
   A labeled rule separates the two so the default and the form never read as one
   control. Reuses the shared .mr-* vars (light + dark). */
.mr-op-head { font-weight:600; color:var(--mr-fg); font-size:13px; margin:2px 0 0; }
.mr-quick-hint { color:var(--mr-muted); font-size:12px; margin:2px 0 4px; }
.mr-op-or { text-align:center; color:var(--mr-muted); font-size:10.5px; font-weight:600;
            text-transform:uppercase; letter-spacing:.09em; margin:8px 0 2px;
            border-top:1px solid var(--mr-line); padding-top:8px; }
"""


def _clean(text: str | None) -> str | None:
    """Trim to a value or None (blank field = filter off)."""
    text = (text or "").strip()
    return text or None


def _split_tags(raw: str | None) -> list[str] | None:
    """Comma/whitespace-separated tag box → list (ANY-match), or None when empty."""
    tags = [t.strip() for t in (raw or "").replace(",", " ").split()]
    return tags or None


def run_browse(
    *,
    text: str | None = None,
    semantic: bool = False,
    persona: str | None = None,
    scope: str = "all",
    tags: str | None = None,
    occurred_after: str | None = None,
    occurred_before: str | None = None,
    min_hit_count: float | int | None = 0,
    order: str = "recency",
    limit: float | int | None = DEFAULT_LIMIT,
    offset: float | int | None = 0,
    browse_fn: Callable[..., list[dict]] = browse_memories,
) -> list[dict]:
    """Map the UI controls onto ``browse_memories`` kwargs and return the rows.

    Parses the free-text controls (tags box → list, numbers → clamped ints, blank →
    filter off) and delegates the query. ``browse_fn`` is injectable so a headless
    test can stub the query and assert the wiring, or point at the real read-only
    ``browse_memories`` on the SQLite-degraded path.
    """
    return browse_fn(
        text=_clean(text),
        semantic=bool(semantic),
        persona=_clean(persona),
        scope=scope or "all",
        tags=_split_tags(tags),
        occurred_after=_clean(occurred_after),
        occurred_before=_clean(occurred_before),
        min_hit_count=max(int(min_hit_count or 0), 0),
        order=order or "recency",
        limit=max(int(limit or DEFAULT_LIMIT), 1),
        offset=max(int(offset or 0), 0),
    )


# ── rendering (reuses the chat/eval .mr-* design system) ────────────────────────

def _chips(items: list[str]) -> str:
    if not items:
        return '<span class="mr-chip">—</span>'
    return ('<div class="mr-chips">'
            + "".join(f'<span class="mr-chip">{html.escape(t)}</span>' for t in items)
            + "</div>")


def _ts(value: str | None) -> str:
    """ISO timestamp -> ``YYYY-MM-DD HH:MM`` (monospace-friendly), full ISO in a
    ``title`` so no precision is lost. ``None`` -> ``-``. Escaped for HTML."""
    if not value:
        return '<span class="mr-mono">-</span>'
    disp = html.escape(value[:16].replace("T", " "))
    return f'<span class="mr-mono" title="{html.escape(value)}">{disp}</span>'


def _idcell(value: str | None) -> str:
    """GUID -> monospace 8-char prefix + ellipsis, full value in a ``title``. Long
    ids are what make a wide row unreadable, so we truncate the display only."""
    if not value:
        return '<span class="mr-mono">-</span>'
    v = str(value)
    disp = html.escape(v[:8] + ("…" if len(v) > 8 else ""))
    return f'<span class="mr-mono mr-id" title="{html.escape(v)}">{disp}</span>'


def _text(value: str | None) -> str:
    """Free text -> ellipsis-clamped inline cell, full value in a ``title``."""
    v = (value or "").strip()
    if not v:
        return "—"
    return f'<span class="mr-celltext" title="{html.escape(v)}">{html.escape(v)}</span>'


def _mono(value) -> str:
    return f"<span class='mr-mono'>{html.escape(str(value)) if value not in (None, '') else '-'}</span>"


def _bool(value, *, tip: str = "") -> str:
    t = html.escape(tip)
    if value is True:
        return f'<span class="mr-bool-t" title="{t}">✔ true</span>'
    if value is False:
        return f'<span class="mr-bool-f" title="{t}">✘ false</span>'
    return '<span class="mr-bool-f">-</span>'


def _num(value) -> str:
    if value is None:
        return "<span class='mr-mono'>-</span>"
    v = f"{value:.1f}" if isinstance(value, float) else str(value)
    return f"<span class='mr-mono'>{html.escape(v)}</span>"


def _scope_badge(scope: str) -> str:
    kind = "team" if scope == "team" else "own"
    return f'<span class="mr-badge mr-badge-{kind}">{html.escape(scope)}</span>'


# The COMPLETE agent_memory cell, laid out as tooltip'd columns grouped by concern
# so a ~30-field row stays scannable. Each column: (key, renderer, header-tooltip).
# ``kind`` renderers keep ids/times monospace + truncated (full value in a title),
# long text ellipsis-clamped, booleans as ✔/✘. The order mirrors browse._full_cell.
_COLUMN_GROUPS: list[tuple[str, list[tuple[str, str, str]]]] = [
    ("Identity & content", [
        ("id", "id", "Primary key (uuid) — truncated; hover for the full value"),
        ("content", "text", "Raw stored text of the memory"),
        ("context_summary", "text", "Recall/embedding summary — first_nonblank(summary, content) is embedded"),
        ("agent_summary", "text", "Team-facing summary — EXCLUDED from content_tsv + the embedding"),
        ("tags", "chips", "Semantic entity tags (authorship by:<slug> split into the 'by' column)"),
        ("by", "mono", "Authorship — the by:<slug> contributors"),
    ]),
    ("Scope coordinates", [
        ("user_id", "id", "Human identity (uuid5 off the replica namespace)"),
        ("agent_type", "mono", "Persona — the isolation dimension for recall/write"),
        ("project_id", "id", "Fixed project uuid5 (one value for the whole replica)"),
        ("customer_id", "id", "Fixed customer uuid5 — its presence is what makes the TEAM pass run"),
        ("session_id", "id", "Session — OMITTED from the recall scope, so recall is cross-session"),
        ("scope", "scope", "own = this human's row · team = a teammate's row (USER axis)"),
    ]),
    ("Classification", [
        ("memory_tier", "mono", "Memory tier classification"),
        ("discipline", "mono", "Engineering discipline"),
        ("authority", "mono", "Authority / trust level"),
    ]),
    ("Temporal (bi-temporal)", [
        ("occurred_at", "ts", "EVENT time — when the fact actually happened"),
        ("created_at", "ts", "WRITE / RECENCY time — when the row was stored (recency-decay keys off THIS, not occurred_at)"),
        ("last_used_at", "ts", "Recency of recall — when a query last returned this row"),
        ("valid_from", "ts", "Bi-temporal start — occurred_at else created_at"),
        ("valid_to", "ts", "Bi-temporal end — set when superseded; blank = still current"),
    ]),
    ("Lifecycle", [
        ("is_active", "bool", "Recallable? Automatic conflict-supersession keeps TRUE; manual retire/merge sets FALSE"),
        ("superseded_by_id", "id", "The newest-wins winner row that superseded this one"),
        ("parent_id", "id", "Merge parent — the root this row was archived into (parent_id forest)"),
    ]),
    ("Ranking", [
        ("hit_count", "num", "Times a relevance-matched recall returned this row"),
        ("last_hit_by", "mono", "Who last recalled this on a matched hit"),
        ("salience", "num", "Ranking score = 1000·matched + 100·rrf + 10·log1p(hits) + recency-decay"),
    ]),
    ("Provenance & search", [
        ("source_message_id", "id", "Origin message — blank = synthetic / merged root"),
        ("has_embedding", "bool", "MiniLM-384 vector present (raw 384 floats are never dumped)"),
        ("embedding_dim", "num", "Embedding dimensionality (384)"),
        ("has_tsv", "bool", "content_tsv full-text vector present"),
    ]),
]


def _render_cell(r: dict, key: str, kind: str, tip: str) -> str:
    v = r.get(key)
    if kind == "id":
        return _idcell(v)
    if kind == "text":
        return _text(v)
    if kind == "chips":
        return _chips(r.get("tags") or [])
    if kind == "scope":
        return _scope_badge(v or "-")
    if kind == "ts":
        return _ts(v)
    if kind == "bool":
        return _bool(v, tip=tip)
    if kind == "num":
        return _num(v)
    return _mono(v)


def render_memory_table(rows: list[dict], *, offset: int = 0, semantic: bool = False) -> str:
    """Render browse rows as the COMPLETE agent_memory cell — every field
    ``browse_memories`` returns — in one horizontally-scrollable table.

    ~30 columns are grouped by concern (identity/content · scope · classification ·
    temporal · lifecycle · ranking · provenance+search) under a two-tier header: an
    uppercase GROUP row plus a pinned field-name row (each field-name cell carries a
    ``title`` explaining the column — event vs written vs valid_from/to vs last_used,
    the is_active recallability contrast, the salience formula, …). ids/times are
    monospace + truncated with the full value in a ``title``, long text is
    ellipsis-clamped, booleans render ✔/✘. Everything is escaped; read-only (row
    selection + the write actions live in the action menu below). The wide table
    scrolls inside ``.mr-tablewrap`` (overflow-x) so the page body never does."""
    order_note = " · ranked by semantic similarity" if semantic else ""
    if not rows:
        return ('<div class="mr-panel"><div class="mr-sec-title">Memory browser</div>'
                '<p class="mr-empty">No memories match these filters. Widen the scope, '
                "clear the text box, or check the DB is migrated.</p></div>")
    count = (f'<div class="mr-count">Showing {len(rows)} row(s) from offset '
             f"{offset}{order_note} · full agent_memory cell (scroll →).</div>")

    group_ths, col_ths = [], []
    for gname, cols in _COLUMN_GROUPS:
        group_ths.append(f'<th class="mr-group-start" colspan="{len(cols)}">{html.escape(gname)}</th>')
        for i, (key, _kind, tip) in enumerate(cols):
            cls = ' class="mr-group-start"' if i == 0 else ""
            col_ths.append(f'<th{cls} title="{html.escape(tip)}">{html.escape(key)}</th>')
    head = (
        '<div class="mr-tablewrap"><table class="mr-table mr-fullcell"><thead>'
        f'<tr class="mr-grouprow">{"".join(group_ths)}</tr>'
        f'<tr class="mr-colrow">{"".join(col_ths)}</tr>'
        "</thead><tbody>"
    )
    body = []
    for r in rows:
        tds = []
        for _gname, cols in _COLUMN_GROUPS:
            for i, (key, kind, tip) in enumerate(cols):
                cls = ' class="mr-group-start"' if i == 0 else ""
                tds.append(f"<td{cls}>{_render_cell(r, key, kind, tip)}</td>")
        body.append(f"<tr>{''.join(tds)}</tr>")
    table = head + "".join(body) + "</tbody></table></div>"
    return f'<div class="mr-panel"><div class="mr-sec-title">Memory browser</div>{count}{table}</div>'


# ── action menu (supersede / merge / reset) — WRITE ops via memory.mutate ───────

def _action_panel(title: str, body_html: str, *, kind: str = "ok") -> str:
    """Result/notice panel. ``kind`` ``warn`` styles a validation notice; ``err`` an
    exception; ``ok`` a success — all reuse the shared .mr-* system."""
    box = "mr-warn" if kind in ("warn", "err") else "mr-panel"
    return (f'<div class="{box}"><div class="mr-sec-title">{html.escape(title)}</div>'
            f"{body_html}</div>")


def _distinct(ids: list[str] | None) -> list[str]:
    """Order-preserving distinct, dropping blanks (a double-select is not two rows)."""
    seen: set = set()
    out: list[str] = []
    for i in ids or []:
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _guard(exc: Exception) -> tuple[bool, str]:
    """A mutate op raised — surface it, never crash the app."""
    body = (f'<p class="mr-err">{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</p>'
            '<p class="mr-hint">The row(s) may be out of scope or the DB may be down. '
            "Re-Browse to refresh, then retry.</p>")
    return False, _action_panel("Action failed", body, kind="err")


def do_supersede(
    selected_ids: list[str] | None, reason: str | None, confirm: bool,
    *, supersede_fn: Callable[..., dict] = supersede_memory,
) -> tuple[bool, str]:
    """Retire each selected memory from recall via ``memory.mutate.supersede_memory``.

    Requires ≥1 selection and an explicit ``confirm`` (destructive — recall stops
    returning the row). Returns ``(ok, result_html)``; ``ok=False`` (no write) for a
    missing selection / unconfirmed action / raised error. ``supersede_fn`` is
    injectable so a headless test asserts the exact call without a DB.
    """
    ids = _distinct(selected_ids)
    if not ids:
        return False, _action_panel(
            "Nothing selected",
            '<p class="mr-hint">Pick one or more rows in <b>Selected rows</b> above, then '
            "Supersede.</p>", kind="warn")
    if not confirm:
        return False, _action_panel(
            "Confirm required",
            f'<p class="mr-hint">Tick <b>Confirm supersede</b> — this retires the '
            f"selected {len(ids)} memory row(s) so recall no longer returns them "
            "(a soft, auditable retire — the row is not deleted).</p>", kind="warn")
    try:
        results = [supersede_fn(mid, reason=(reason or None)) for mid in ids]
    except Exception as exc:  # out-of-scope / DB down — surface, don't crash
        return _guard(exc)
    rows = "".join(
        f"<tr><td class='mr-mono'>{html.escape(str(r.get('id')))}</td>"
        f"<td class='mr-mono'>{html.escape(str(r.get('valid_to') or '-'))}</td></tr>"
        for r in results
    )
    body = (f'<p class="mr-count">Retired {len(results)} row(s) from recall'
            f"{' · reason logged' if reason else ''}.</p>"
            "<table class='mr-table'><thead><tr><th>superseded id</th><th>valid_to</th>"
            f"</tr></thead><tbody>{rows}</tbody></table>")
    return True, _action_panel("Superseded", body)


def do_merge(
    selected_ids: list[str] | None, summary: str | None, confirm: bool,
    *, merge_fn: Callable[..., dict] = merge_memories,
) -> tuple[bool, str]:
    """Consolidate ≥2 selected memories into one new parent via
    ``memory.mutate.merge_memories`` (deterministic union summary when ``summary`` is
    blank). Requires ≥2 selections + ``confirm``. Returns ``(ok, result_html)`` and
    surfaces the new parent id. ``merge_fn`` is injectable for a headless test.
    """
    ids = _distinct(selected_ids)
    if len(ids) < 2:
        return False, _action_panel(
            "Select at least two",
            '<p class="mr-hint">Merge consolidates <b>two or more</b> rows into one new '
            "parent. Select more rows, then Merge.</p>", kind="warn")
    if not confirm:
        return False, _action_panel(
            "Confirm required",
            f'<p class="mr-hint">Tick <b>Confirm merge</b> — this consolidates the '
            f"selected {len(ids)} rows into one new parent and archives the children "
            "(auditable; children stay in the table, superseded).</p>", kind="warn")
    try:
        res = merge_fn(ids, summary=(summary.strip() or None) if summary else None)
    except Exception as exc:  # mixed-scope / out-of-scope / DB down
        return _guard(exc)
    body = (f'<p class="mr-count">Merged {res.get("merged_count")} row(s) into a new '
            "parent.</p><table class='mr-table'><tbody>"
            f"<tr><td>new parent id</td><td class='mr-mono'>{html.escape(str(res.get('parent_id')))}</td></tr>"
            f"<tr><td>merged children</td><td class='mr-mono'>{res.get('merged_count')}</td></tr>"
            "</tbody></table>")
    return True, _action_panel("Merged", body)


def do_reset(
    confirm: bool, *, reset_fn: Callable[[], dict] = reset_db,
) -> tuple[bool, str]:
    """Empty the memory tables via ``memory.mutate.reset_db`` — GLOBAL and
    destructive. Requires the strong ``confirm``. Returns ``(ok, result_html)`` and
    surfaces ``{deleted_memories, deleted_tags}``. ``reset_fn`` is injectable.
    """
    if not confirm:
        return False, _action_panel(
            "Confirm required",
            '<p class="mr-hint">Tick <b>Yes — permanently delete ALL stored '
            "memory</b>. This empties every memory row in the DB. It cannot be "
            "undone.</p>", kind="warn")
    try:
        res = reset_fn()
    except Exception as exc:  # DB down
        return _guard(exc)
    body = ("<table class='mr-table'><tbody>"
            f"<tr><td>deleted memories</td><td class='mr-mono'>{res.get('deleted_memories')}</td></tr>"
            f"<tr><td>deleted tags</td><td class='mr-mono'>{res.get('deleted_tags')}</td></tr>"
            "</tbody></table>")
    return True, _action_panel("Database reset", body)


def do_consolidate(
    persona: str | None, threshold: float | int | None, confirm: bool,
    *, consolidate_fn: Callable[..., dict] = consolidate,
) -> tuple[bool, str]:
    """Manually consolidate one persona's memory pool via
    ``memory.consolidation.consolidate`` — the on-demand synthesize-merge of
    near-duplicates (τ = cosine near-dup threshold, default 0.85). Per-persona (NOT
    the selected rows). Requires a ``persona`` + explicit ``confirm``. Returns
    ``(ok, result_html)`` surfacing ``{run_id, pools_merged, memories_before,
    memories_after, superseded}``. ``consolidate_fn`` is injectable for a headless
    test (mirrors the other ``do_*`` ops).
    """
    name = _clean(persona)
    if not name:
        return False, _action_panel(
            "Persona required",
            '<p class="mr-hint">Enter the <b>persona</b> whose pool to consolidate '
            "(e.g. <code>researcher</code>) — consolidation runs per-persona, not on the "
            "selected rows.</p>", kind="warn")
    if not confirm:
        return False, _action_panel(
            "Confirm required",
            f'<p class="mr-hint">Tick <b>Confirm consolidate</b> — this synthesize-merges '
            f"near-duplicate memories in <b>{html.escape(name)}</b>'s pool (τ = cosine "
            "near-dup threshold). Merged children are archived (auditable, not "
            "deleted).</p>", kind="warn")
    tau = float(threshold) if threshold is not None else 0.85
    try:
        res = consolidate_fn(name, threshold=tau)
    except Exception as exc:  # DB down / no embeddings on the degraded path
        return _guard(exc)
    body = (f'<p class="mr-count">Consolidated <b>{html.escape(name)}</b>\'s pool at '
            f"τ={tau:.2f} (synthesize-merge of near-duplicates).</p>"
            "<table class='mr-table'><tbody>"
            f"<tr><td>run id</td><td class='mr-mono'>{html.escape(str(res.get('run_id')))}</td></tr>"
            f"<tr><td>pools merged</td><td class='mr-mono'>{res.get('pools_merged')}</td></tr>"
            f"<tr><td>memories before</td><td class='mr-mono'>{res.get('memories_before')}</td></tr>"
            f"<tr><td>memories after</td><td class='mr-mono'>{res.get('memories_after')}</td></tr>"
            f"<tr><td>superseded</td><td class='mr-mono'>{res.get('superseded')}</td></tr>"
            "</tbody></table>")
    return True, _action_panel("Consolidated", body)


# ── Gradio tab ───────────────────────────────────────────────────────────────

def _placeholder() -> str:
    return ('<div class="mr-panel"><div class="mr-sec-title">Memory browser</div>'
            '<p class="mr-hint">Set filters and press <b>Browse</b> to page through the '
            "<code>agent_memory</code> table. This view is strictly read-only — browsing "
            "never bumps <code>hit_count</code> or perturbs recall stats.</p></div>")


def _do_browse(text, semantic, persona, scope, tags, occurred_after, occurred_before,
               min_hit_count, order, limit, offset) -> tuple[list[dict] | None, str | None]:
    """Run the filtered browse; return ``(rows, None)`` or ``(None, error_html)``."""
    try:
        rows = run_browse(
            text=text, semantic=semantic, persona=persona, scope=scope, tags=tags,
            occurred_after=occurred_after, occurred_before=occurred_before,
            min_hit_count=min_hit_count, order=order, limit=limit, offset=offset,
        )
    except ValueError as exc:  # bad scope/order — shouldn't happen via the UI, but guard
        return None, ('<div class="mr-panel"><div class="mr-sec-title">Invalid filter</div>'
                      f'<p class="mr-err">{html.escape(str(exc))}</p></div>')
    except Exception as exc:  # DB down / not migrated — surface, do not crash the app
        return None, ('<div class="mr-panel"><div class="mr-sec-title">Memory unavailable</div>'
                      f'<p class="mr-err">{html.escape(type(exc).__name__)}: {html.escape(str(exc))}</p>'
                      '<p class="mr-hint">Start the local Postgres and migrate: '
                      "<code>docker compose up -d</code> then <code>uv run alembic upgrade head</code> "
                      "(or just run <code>./run_demo.sh</code>).</p></div>")
    return rows, None


def _persona_choices(*, browse_fn: Callable[..., list[dict]] = browse_memories) -> list[str]:
    """Distinct ``agent_type`` values PRESENT in the DB — the personas offered in the
    Consolidate dropdown. Derived from one read-only ``browse_memories`` (scope=all,
    generous limit) so it needs no dedicated engine primitive; returns ``[]`` when the
    DB is unavailable (the dropdown stays typeable via ``allow_custom_value``). Any
    persona past the limit tail can still be typed. ``browse_fn`` is injectable for a
    headless test."""
    try:
        rows = browse_fn(scope="all", limit=2000)
    except Exception:  # DB down / not migrated — degrade to a typeable empty list
        return []
    return sorted({p for r in rows if (p := r.get("agent_type"))})


def _default_persona(choices: list[str]) -> str:
    """Consolidate-dropdown default: the chat tab's ``researcher`` persona when present,
    else the first available persona, else ``researcher`` (custom value)."""
    return "researcher" if "researcher" in choices else (choices[0] if choices else "researcher")


def _choices(rows: list[dict]) -> list[tuple[str, str]]:
    """Browse rows -> (label, memory_id) options for the selection dropdown. The
    label is a short, scannable summary + persona + id-prefix; the value is the id
    the mutate ops take."""
    out: list[tuple[str, str]] = []
    for r in rows:
        rid = r.get("id")
        if not rid:
            continue
        summ = (r.get("summary") or "").strip() or "—"
        label = (f"{summ[:56]}{'…' if len(summ) > 56 else ''} · "
                 f"{r.get('agent_type') or '?'} · {rid[:8]}")
        out.append((label, rid))
    return out


def on_browse(text, semantic, persona, scope, tags, occurred_after, occurred_before,
              min_hit_count, order, limit, offset):
    """Run a browse and refresh the table, the selection dropdown, and the rows
    state in one shot. Returns ``(table_html, dropdown_update, rows)``."""
    rows, err = _do_browse(text, semantic, persona, scope, tags, occurred_after,
                           occurred_before, min_hit_count, order, limit, offset)
    if err is not None:
        return err, gr.update(choices=[], value=[]), []
    table = render_memory_table(rows, offset=max(int(offset or 0), 0), semantic=bool(semantic))
    return table, gr.update(choices=_choices(rows), value=[]), rows


# Filter-input order shared by the browse handler and the post-action refresh.
def _refresh(filters: tuple):
    """Re-run the current filter set after a write op — refresh table + choices."""
    rows, err = _do_browse(*filters)
    if err is not None:
        return err, gr.update(choices=[], value=[]), []
    offset = filters[-1]
    table = render_memory_table(rows, offset=max(int(offset or 0), 0), semantic=bool(filters[1]))
    return table, gr.update(choices=_choices(rows), value=[]), rows


def _after_action(ok: bool, msg: str, filters: tuple):
    """Shared action epilogue -> ``(action_out, table, selection, rows_state,
    con_persona)``. On a failed/unconfirmed action nothing is refreshed (the other
    four keep their current values); on success the table + row choices + rows_state
    are re-browsed and the Consolidate persona dropdown is repopulated from the
    personas now present in the DB (an action can add/retire a persona's rows)."""
    if not ok:
        return msg, gr.update(), gr.update(), gr.update(), gr.update()
    table, choices_upd, rows = _refresh(filters)
    return msg, table, choices_upd, rows, gr.update(choices=_persona_choices())


def on_supersede(selected, reason, confirm, *filters):
    return _after_action(*do_supersede(selected, reason, confirm), filters)


def on_supersede_default(selected, *filters):
    """One-click Supersede: retire the selected row(s) with default args
    (``reason=""``, ``confirm=True``) — the button press IS the intent. Soft +
    auditable (rows retired, not deleted). Reuses ``do_supersede`` + ``_after_action``
    unchanged."""
    return _after_action(*do_supersede(selected, "", True), filters)


def on_merge(selected, summary, confirm, *filters):
    return _after_action(*do_merge(selected, summary, confirm), filters)


def on_merge_default(selected, *filters):
    """One-click Merge: fold the ≥2 selected rows into one parent with default args
    (``summary=None`` -> deterministic union, ``confirm=True``). Reuses ``do_merge`` +
    ``_after_action`` unchanged."""
    return _after_action(*do_merge(selected, None, True), filters)


def on_reset(confirm, *filters):
    return _after_action(*do_reset(confirm), filters)


def on_consolidate(persona, threshold, confirm, *filters):
    return _after_action(*do_consolidate(persona, threshold, confirm), filters)


def build_memory_tab() -> None:
    """Build the Memory tab contents (call inside a gr.Tab / gr.Blocks)."""
    gr.Markdown("### Memory — browse the team-memory DB")
    gr.Markdown(
        "The **browser** below is a strictly **read-only** window on the `agent_memory` "
        "table (via `memory.browse.browse_memories`) — it never perturbs recall stats. "
        "**Persona** (which persona authored the memory) and **scope** (own/team/all) "
        "are *independent* filters, unlike recall's coupled two-section split. Each row "
        "shows the **complete cell** — every `agent_memory` field, grouped and "
        "tooltip'd (hover any column header), including the distinct **occurred_at** "
        "(event) vs **created_at** (written/recency) vs **last_used_at** timestamps, the "
        "bi-temporal `valid_from`/`valid_to`, the `is_active` recallability flag, and the "
        "`salience` ranking score. Scroll the table sideways to see them all. The WRITE "
        "**actions** (🛠 consolidate · supersede · merge · reset) live in the highlighted "
        "**Memory actions** panel below the table.",
        elem_classes=["mr-subtitle"],
    )
    with gr.Row():
        text = gr.Textbox(label="Search text", scale=5,
                          placeholder="Filter by content — blank = no text filter…")
        semantic = gr.Checkbox(value=False, label="Semantic (pgvector)",
                               info="Rank by embedding similarity (PG only; else lexical).")
    with gr.Row():
        persona = gr.Textbox(label="Persona (agent_type)", scale=2,
                             placeholder="e.g. awsi — blank = all personas")
        scope = gr.Radio(SCOPE_CHOICES, value="all", label="Scope (own/team/all)", scale=2)
        order = gr.Radio(ORDER_CHOICES, value="recency", label="Order", scale=2)
    with gr.Row():
        tags = gr.Textbox(label="Tags (any-match)", scale=3,
                          placeholder="comma/space separated, e.g. pgvector, pll")
        occurred_after = gr.Textbox(label="Occurred after", scale=2, placeholder="YYYY-MM-DD")
        occurred_before = gr.Textbox(label="Occurred before", scale=2, placeholder="YYYY-MM-DD")
    with gr.Row():
        min_hit_count = gr.Number(value=0, precision=0, label="Min hit count", scale=2)
        limit = gr.Number(value=DEFAULT_LIMIT, precision=0, label="Limit", scale=2)
        offset = gr.Number(value=0, precision=0, label="Offset", scale=2)
        browse_btn = gr.Button("Browse", variant="primary", scale=2)
    out = gr.HTML(value=_placeholder())

    # ── Memory actions — highlighted so the WRITE ops are impossible to miss ─────
    # A single accent-bordered legend banner names all four ops + what each targets
    # (selected row(s) vs per-persona vs global), then the controls group by target.
    rows_state = gr.State([])
    gr.HTML(
        '<div class="mr-actionbar">'
        '<div class="mr-actions-title">🛠 Memory actions</div>'
        '<div class="mr-actionlegend">'
        '<span class="mr-chip">Consolidate</span> synthesize-merge near-duplicates in a '
        "persona's pool <em>(per-persona)</em>"
        '<span class="mr-chip">Supersede</span> retire selected row(s) from recall '
        "<em>(selected)</em>"
        '<span class="mr-chip">Merge</span> fold 2+ selected rows into one parent '
        "<em>(selected)</em>"
        '<span class="mr-chip">Reset</span> wipe every memory row <em>(global)</em>'
        "</div>"
        '<p class="mr-hint">Supersede &amp; Merge act on the row(s) '
        "you pick below — each offers a <b>one-click default</b> (soft, auditable "
        "retire) <em>or</em> an explicit configurable form; Consolidate runs "
        "per-persona and Reset is global, both confirm-gated. The table refreshes "
        "after each.</p>"
        "</div>")

    # Consolidate — per-persona, so it stands apart from the row-selection ops below.
    gr.Markdown(
        "**Consolidate** is the manual, on-demand synthesize-merge of a persona's "
        "near-duplicate memories (τ = cosine near-dup threshold) via "
        "`memory.consolidation.consolidate`.",
        elem_classes=["mr-subtitle"],
    )
    _personas = _persona_choices()
    with gr.Row():
        con_persona = gr.Dropdown(
            choices=_personas, value=_default_persona(_personas), allow_custom_value=True,
            label="Consolidate persona", scale=3,
            info="Personas present in the DB (refreshes after each action) — type a "
                 "new one if it isn't listed yet.")
        con_threshold = gr.Number(value=0.85, label="Threshold τ (cosine near-dup)", scale=2)
        con_confirm = gr.Checkbox(value=False, label="Confirm consolidate", scale=2)
        con_btn = gr.Button("Consolidate", variant="primary", scale=2)

    # Supersede / Merge — operate on the selected browsed row(s).
    selection = gr.Dropdown(
        choices=[], multiselect=True, label="Selected rows",
        info="Populated from the last Browse. Pick 1+ to supersede, 2+ to merge.")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("**Supersede** — retire selected row(s) from recall.",
                        elem_classes=["mr-op-head"])
            sup_default_btn = gr.Button("⚡ Supersede selected (defaults)", variant="primary")
            gr.Markdown("One click · no reason · retires the selected row(s) now "
                        "(soft, auditable — not deleted).", elem_classes=["mr-quick-hint"])
            gr.Markdown("or configure", elem_classes=["mr-op-or"])
            sup_reason = gr.Textbox(label="Supersede reason (optional, logged not stored)",
                                    placeholder="e.g. duplicate of a newer fact")
            sup_confirm = gr.Checkbox(value=False, label="Confirm supersede (retire from recall)")
            sup_btn = gr.Button("Supersede selected", variant="stop")
        with gr.Column(scale=1):
            gr.Markdown("**Merge** — fold 2+ selected rows into one parent.",
                        elem_classes=["mr-op-head"])
            merge_default_btn = gr.Button("⚡ Merge selected (defaults)", variant="primary")
            gr.Markdown("One click · deterministic union summary · needs ≥2 selected "
                        "(children archived, auditable).", elem_classes=["mr-quick-hint"])
            gr.Markdown("or configure", elem_classes=["mr-op-or"])
            merge_summary = gr.Textbox(label="Merge summary (optional — blank = deterministic union)",
                                       placeholder="parent summary, or leave blank")
            merge_confirm = gr.Checkbox(value=False, label="Confirm merge (≥2 rows into one parent)")
            merge_btn = gr.Button("Merge selected", variant="stop")
    with gr.Row():
        reset_confirm = gr.Checkbox(value=False, label="Yes — permanently delete ALL stored memory")
        reset_btn = gr.Button("Reset DB", variant="stop")
    action_out = gr.HTML(value="")

    filters = [text, semantic, persona, scope, tags, occurred_after, occurred_before,
               min_hit_count, order, limit, offset]
    browse_out = [out, selection, rows_state]
    browse_btn.click(on_browse, filters, browse_out)
    # Enter in the search box also runs the current filter set.
    text.submit(on_browse, filters, browse_out)

    # con_persona is BOTH a control read by Consolidate and an output refreshed after
    # every action (its choices track the personas present in the DB).
    action_out_targets = [action_out, out, selection, rows_state, con_persona]
    con_btn.click(on_consolidate, [con_persona, con_threshold, con_confirm, *filters], action_out_targets)
    sup_btn.click(on_supersede, [selection, sup_reason, sup_confirm, *filters], action_out_targets)
    sup_default_btn.click(on_supersede_default, [selection, *filters], action_out_targets)
    merge_btn.click(on_merge, [selection, merge_summary, merge_confirm, *filters], action_out_targets)
    merge_default_btn.click(on_merge_default, [selection, *filters], action_out_targets)
    reset_btn.click(on_reset, [reset_confirm, *filters], action_out_targets)
