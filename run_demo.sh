#!/usr/bin/env bash
# One-command launcher for the memory-research chat demo.
#   ./run_demo.sh         start the project-local DB + migrate + launch the chat app
#   ./run_demo.sh stop    stop the project-local DB cluster
#   ./run_demo.sh reset   WIPE the DB (delete ./.pgdata + ./.pgsock) after a confirm
#
# DEFAULT PATH: a Docker-FREE, project-local Postgres cluster living inside this
# folder (./.pgdata) and reachable ONLY over a unix socket in ./.pgsock — so the
# whole folder is self-contained and can be handed to a student to "just run".
# It never binds a TCP port, so it cannot clash with any Postgres already running
# on :5432. docker-compose.yml remains as a documented alternative for Docker users.
#
# PERSISTENCE: the cluster is created ONCE (initdb only when ./.pgdata is absent);
# every later run just starts the existing cluster, so all stored memory PERSISTS
# across runs. `./run_demo.sh reset` is the ONLY thing that wipes it.
#
# Idempotent — safe to re-run; each step is a no-op when already satisfied.
set -euo pipefail

cd "$(dirname "$0")"          # memory-research/
ROOT="$(pwd -P)"             # absolute path to this folder
PGDATA="$ROOT/.pgdata"
PGSOCK="$ROOT/.pgsock"
PGLOG="$PGDATA/server.log"
PGUSER="$(id -un)"           # cluster superuser = current OS user (trust auth)
DBNAME="memory_research"

# Resolve a pgvector-capable Postgres bin dir. The Homebrew `postgresql@17`
# formula is what ships pgvector on macOS (the default `postgres` on PATH may be
# an older major without it), so prefer it; override with MEMORY_PG_BIN.
#   1) $MEMORY_PG_BIN   2) Homebrew postgresql@17   3) whatever is on PATH
resolve_pgbin() {
  if [ -n "${MEMORY_PG_BIN:-}" ]; then printf '%s' "$MEMORY_PG_BIN"; return; fi
  if command -v brew >/dev/null 2>&1; then
    local p; p="$(brew --prefix postgresql@17 2>/dev/null)/bin"
    if [ -x "$p/initdb" ]; then printf '%s' "$p"; return; fi
  fi
  local i; i="$(command -v initdb 2>/dev/null || true)"
  [ -n "$i" ] && dirname "$i"
}
PGBIN="$(resolve_pgbin)"

# ── stop subcommand ────────────────────────────────────────────────────────────
if [ "${1:-}" = "stop" ]; then
  if [ -n "$PGBIN" ] && [ -d "$PGDATA" ] && "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    "$PGBIN/pg_ctl" -D "$PGDATA" stop -m fast
    echo "✓ stopped project-local Postgres."
  else
    echo "→ project-local Postgres is not running (nothing to stop)."
  fi
  exit 0
fi

# ── reset / wipe subcommand ──────────────────────────────────────────────────────
# Stop the cluster (if running) then DELETE ./.pgdata + ./.pgsock so the NEXT run
# re-inits a fresh, empty DB. Guarded by an interactive confirmation so it cannot
# nuke stored memory by accident. `--force` skips the prompt (required when stdin
# is not a TTY, e.g. CI/scripts) — the ONLY way to wipe non-interactively.
if [ "${1:-}" = "reset" ] || [ "${1:-}" = "wipe" ]; then
  force=0; [ "${2:-}" = "--force" ] && force=1
  if [ "$force" != "1" ]; then
    if [ ! -t 0 ]; then
      echo "✗ refusing to wipe: stdin is not a TTY. Re-run with '--force' to confirm:" >&2
      echo "    ./run_demo.sh reset --force" >&2
      exit 1
    fi
    echo "⚠️  This deletes ALL stored memory in ./.pgdata (the project-local DB)."
    printf 'Type YES to confirm: '
    read -r reply
    if [ "$reply" != "YES" ]; then echo "→ aborted (nothing deleted)."; exit 1; fi
  fi
  if [ -n "$PGBIN" ] && [ -d "$PGDATA" ] && "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
    "$PGBIN/pg_ctl" -D "$PGDATA" stop -m fast >/dev/null
    echo "✓ stopped project-local Postgres."
  fi
  rm -rf "$PGDATA" "$PGSOCK"
  echo "✓ wiped ./.pgdata + ./.pgsock — next './run_demo.sh' re-inits a fresh DB."
  exit 0
fi

PORT="${GRADIO_SERVER_PORT:-7860}"
HOSTNAME_="${GRADIO_SERVER_NAME:-127.0.0.1}"

# 1. Config — seed .env from the template on first run, then load it. NOTE: the
#    project-local socket DSN is computed and exported below (step 5), overriding
#    any MEMORY_DATABASE_URL in .env, so the local cluster is authoritative.
if [ ! -f .env ]; then
  cp .env.example .env
  echo "→ created .env from .env.example"
fi
set -a; . ./.env; set +a

# 2. Prerequisites.
if [ -z "$PGBIN" ] || [ ! -x "$PGBIN/initdb" ]; then
  echo "✗ Postgres binaries not found. Install Postgres 17 + pgvector, e.g.:" >&2
  echo "    brew install postgresql@17 pgvector        # macOS (no Docker needed)" >&2
  echo "  or set MEMORY_PG_BIN to a pgvector-capable bin dir. See README 'Local database'." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "✗ uv not found. Install it: https://docs.astral.sh/uv/  and retry." >&2
  exit 1
fi

# 3. Socket-path length guard. A unix socket path must fit the OS sockaddr_un
#    limit (~104 on macOS); Postgres appends "/.s.PGSQL.<port>". Fail early with a
#    clear message rather than an opaque bind error deep in startup.
SOCKFILE="$PGSOCK/.s.PGSQL.5432"
if [ "${#SOCKFILE}" -ge 100 ]; then
  echo "✗ socket path too long (${#SOCKFILE} ≥ 100 chars): $SOCKFILE" >&2
  echo "  Move this folder to a shorter path (e.g. ~/memory-research) and re-run." >&2
  exit 1
fi

# 4. Project-local, Docker-free Postgres cluster. Idempotent.
mkdir -p "$PGSOCK"
if [ ! -d "$PGDATA" ]; then
  echo "→ initializing project-local Postgres cluster in ./.pgdata (trust auth, user $PGUSER)…"
  "$PGBIN/initdb" -D "$PGDATA" -U "$PGUSER" -A trust >/dev/null
fi
if "$PGBIN/pg_ctl" -D "$PGDATA" status >/dev/null 2>&1; then
  echo "✓ project-local Postgres already running."
else
  echo "→ starting project-local Postgres (socket-only, no TCP)…"
  "$PGBIN/pg_ctl" -D "$PGDATA" -l "$PGLOG" \
    -o "-c listen_addresses='' -k $PGSOCK" -w start
fi

echo "→ waiting for Postgres to be ready…"
for _ in $(seq 1 40); do
  if "$PGBIN/pg_isready" -h "$PGSOCK" -U "$PGUSER" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 1
done
if [ "${ready:-0}" != "1" ]; then
  echo "✗ Postgres did not become ready. Check the log: $PGLOG" >&2
  exit 1
fi
echo "✓ Postgres ready."

# 5. Create the DB (idempotent) and export the project-local socket DSN. The
#    socket dir is an absolute path that varies per machine, so we compute the
#    DSN here rather than hard-coding it in .env (see .env.example).
if ! "$PGBIN/psql" -h "$PGSOCK" -U "$PGUSER" -d postgres -tAc \
    "SELECT 1 FROM pg_database WHERE datname='$DBNAME'" | grep -q 1; then
  "$PGBIN/createdb" -h "$PGSOCK" -U "$PGUSER" "$DBNAME"
  echo "→ created database $DBNAME"
fi
export MEMORY_DATABASE_URL="postgresql+psycopg2://$PGUSER@/$DBNAME?host=$PGSOCK"

# 6. Dependencies (engine + chat UI + extraction LLM). Idempotent.
echo "→ syncing dependencies (uv sync --extra chat --extra llm)…"
uv sync --extra chat --extra llm

# 7. Schema — create the pgvector extension + memory tables. Idempotent.
echo "→ applying migrations (alembic upgrade head)…"
uv run alembic upgrade head

# 8. Launch.
echo ""
echo "✓ ready → http://${HOSTNAME_}:${PORT}"
echo "  (Ctrl-C to stop the app; './run_demo.sh stop' to stop Postgres.)"
echo ""
GRADIO_SERVER_NAME="$HOSTNAME_" GRADIO_SERVER_PORT="$PORT" uv run python -m chat.app
