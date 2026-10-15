"""Run from the repository root with PYTHONPATH=. and MEMORY_DATABASE_URL set.

Requires a migrated PostgreSQL/pgvector DB and local ONNX model. The script runs
LoCoMo recall-only at k=50, reusing existing ingest if present, and asserts every
returned ORM row belongs to the requested project and customer. It does not call
an answer model or judge. Output is written under gitignored runs/.
"""
import json
from pathlib import Path
from threading import Lock
from memory.store import PostgresMemoryStore
from eval.run_locomo import main

original = PostgresMemoryStore.recall_project_sections
lock = Lock()
audit = {"queries_checked": 0, "returned_records_checked": 0,
         "foreign_project_records": 0, "projects": set()}

def checked(self, **kwargs):
    result = original(self, **kwargs)
    # API shape: (own ORM rows, own hit-writeback plan, team ORM rows, team plan).
    # Plans contain only (UUID, salience, matched) tuples, not additional ORM rows.
    own, _, team, _ = result
    rows = own + team
    for row in rows:
        assert row.project_id == kwargs["project_id"], "foreign project record"
        assert row.customer_id == kwargs["customer_id"], "foreign customer record"
    with lock:
        audit["queries_checked"] += 1
        audit["returned_records_checked"] += len(rows)
        audit["projects"].add(str(kwargs["project_id"]))
    return result

PostgresMemoryStore.recall_project_sections = checked
status = main(["--recall-only", "--no-count-tokens", "--k", "50", "--workers", "2",
               "--reuse-ingest", "--out", "runs/locomo_scope_onnx_audited_recall_k50.json"])
assert status == 0
assert audit["queries_checked"] == 1540
assert len(audit["projects"]) == 10
audit["projects"] = sorted(audit["projects"])
Path("runs/locomo_scope_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps(audit))
