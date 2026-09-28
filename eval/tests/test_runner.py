"""CLI-runner tests: stub-backed end-to-end run, JSON output, error paths."""
from __future__ import annotations

import json

from eval import run_locomo, run_longbench

LOCOMO_SAMPLE = [{
    "conversation": {
        "session_1_date_time": "1:56 pm on 7 May, 2023",
        "session_1": [
            {"speaker": "Alice", "dia_id": "D1:1", "text": "my favorite color is blue"},
            {"speaker": "Bob", "dia_id": "D1:2", "text": "the weather is sunny"},
        ],
    },
    "qa": [
        {"question": "what is Alice's favorite color?", "answer": "blue",
         "category": 4, "evidence": ["D1:1"]},
    ],
}]


def test_run_locomo_stub_writes_json(tmp_path, capsys):
    data = tmp_path / "locomo10.json"
    data.write_text(json.dumps(LOCOMO_SAMPLE))
    out = tmp_path / "res.json"
    rc = run_locomo.main(["--stub", "--data", str(data), "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["config"]["backend"] == "stub"
    assert payload["config"]["n_questions"] == 1
    assert payload["summary"]["recall_at_k"] == 1.0  # stub ranks the blue turn top
    printed = capsys.readouterr().out
    assert "DRY-RUN" in printed and "locomo" in printed


def test_run_recall_only_stub(tmp_path):
    data = tmp_path / "locomo10.json"
    data.write_text(json.dumps(LOCOMO_SAMPLE))
    out = tmp_path / "res.json"
    rc = run_locomo.main(["--stub", "--recall-only", "--data", str(data), "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["config"]["recall_only"] is True
    assert payload["summary"]["token_f1"] is None  # answer skipped


def test_run_missing_data_returns_2(capsys):
    rc = run_locomo.main(["--data", "/nonexistent/locomo10.json"])
    assert rc == 2
    assert "fetch it" in capsys.readouterr().out


def test_run_longbench_stub_directory(tmp_path):
    (tmp_path / "qasper.jsonl").write_text(json.dumps(
        {"input": "what color?", "context": "the color is blue", "answers": ["blue"]}) + "\n")
    out = tmp_path / "lb.json"
    rc = run_longbench.main(["--stub", "--data", str(tmp_path), "--out", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["config"]["benchmark"] == "longbench"
    assert payload["summary"]["token_f1"] is not None
