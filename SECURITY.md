# Security and deployment notes

Mem++ is a **local research/demo tool**. It has optional single-owner UI authentication, no tenant
isolation, and no hardening for shared or internet-facing use. Nothing here is a
production security assurance.

## Intended deployment

- Run on a single-user machine you control.
- Chat UI (Gradio) listens on loopback only. `chat/app.py` and `run_demo.sh` refuse
  any `GRADIO_SERVER_NAME` that is not `127.0.0.1`, `localhost`, or `::1`.
  `share=False` is set explicitly, so `GRADIO_SHARE` cannot open a public tunnel.
- `run_demo.sh` default DB is a project-local Postgres reachable only over a unix
  socket (no TCP).
- `docker-compose.yml` publishes Postgres as `127.0.0.1:5433`. It uses a fixed demo
  credential (`memory`/`memory`); do not expose it or reuse that password.
- Do not put the UI behind a reverse proxy or port-forward to make it reachable by
  others. The optional UI login alone does not make remote deployment supported.

## Optional UI authentication

- Set `MEMORY_UI_AUTH_ENABLED=1` with `MEMORY_UI_USERNAME` and
  `MEMORY_UI_PASSWORD` to enable Gradio session authentication for the demo.
- Missing or blank credentials and invalid enable flags fail startup. Secrets
  are never included in configuration errors. Use a unique, long password.
- All authenticated access uses the same process and memory scope; this is not
  tenant isolation or role-based authorization. Restart to change credentials.
- Loopback-only binding and disabled public sharing still apply.

## Secrets and `.env`

- `.env` is gitignored. Never commit it or paste keys into issues/logs.
- Keys saved from the UI must match `[A-Za-z0-9._-]{1,512}`. Anything else (spaces,
  quotes, newlines, `$`, `;`, backticks, ...) is rejected before any file write or
  environment assignment.
- The UI writes `.env` atomically (temp file + `os.replace`) with mode `0600`.
  Permission/IO errors are reported, not ignored.
- `.env` is parsed, never executed: `run_demo.sh` and `chat/app.py` accept
  `KEY=value`, `export KEY=value`, and matched `"..."`/`'...'` values taken
  literally (no `$`/backtick expansion, no escapes, single-line only). Invalid
  lines are skipped.
- `run_demo.sh` assigns `.env` values over already-exported variables (as before);
  `chat/app.py` alone does not override existing environment. The socket
  `MEMORY_DATABASE_URL` computed by `run_demo.sh` always wins.
- The session key is held in process environment; anyone with access to your
  account or the process can read it.

## Known limits

- Recalled/stored memory text and LLM output are rendered in the UI; HTML is
  escaped in the debug panel, but this has not had a full review.
- Stored memories are sent to the configured LLM provider when a key is set.
- Dependencies are not audited or pinned beyond `uv.lock`.
