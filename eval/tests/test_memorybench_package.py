"""Structural tests use synthetic responses, never benchmark predictions."""
import io
import json
import zipfile

import pytest
from integrations.memorybench import package_submission as package


def manifest():
    return {"schema_version": 1, "repository": package.REPO, "revision": "a" * 40,
            "datasets": {"Locomo-0": {"test_indices": [129, 189]}}}


def responses():
    return [{"test_idx": 189, "dataset": "Locomo-0", "response": "", "gold": "DO NOT UPLOAD"},
            {"test_idx": 129, "dataset": "Locomo-0", "response": "Synthetic: مرحبا"}]


def test_roundtrip_and_determinism():
    data = package.make_package(manifest(), responses())
    assert data == package.make_package(manifest(), list(reversed(responses())))
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        assert archive.namelist() == ["dataset/Locomo-0/predict.json"]
        rows = json.loads(archive.read(archive.namelist()[0]))
    assert [r["test_idx"] for r in rows] == [129, 189]
    assert rows[0]["response"] == "Synthetic: مرحبا"
    assert rows[1]["response"] == ""
    assert all(set(r) == {"test_idx", "dataset", "response"} for r in rows)
    assert b"DO NOT UPLOAD" not in data


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate", "boolean", "float", "unknown", "unsafe", "nonstring"])
def test_bad_predictions(change):
    rows = responses()
    if change == "missing": rows.pop()
    elif change == "extra": rows.append({"test_idx": 0, "dataset": "Locomo-0", "response": "synthetic"})
    elif change == "duplicate": rows.append(dict(rows[0]))
    elif change == "boolean": rows[0]["test_idx"] = True
    elif change == "float": rows[0]["test_idx"] = 189.0
    elif change == "unknown": rows[0]["dataset"] = "Unknown"
    elif change == "unsafe": rows[0]["dataset"] = "../Locomo-0"
    elif change == "nonstring": rows[0]["response"] = None
    with pytest.raises(ValueError): package.make_package(manifest(), rows)


def test_subset_datasets_allowed():
    spec = manifest()
    spec["datasets"]["Locomo-1"] = {"test_indices": [7]}
    package.make_package(spec, responses())
    rows = responses() + [{"dataset": "Locomo-1", "test_idx": 7, "response": "synthetic"}]
    with zipfile.ZipFile(io.BytesIO(package.make_package(spec, rows))) as archive:
        assert len(archive.namelist()) == 2


@pytest.mark.parametrize("value", ['[{"response":NaN}]', '{"x":1,"x":2}'])
def test_strict_json(tmp_path, value):
    path = tmp_path / "input.json"
    path.write_text(value)
    with pytest.raises(ValueError): package.read_json(path)


def test_no_overwrite_and_no_temporary_leaks(tmp_path):
    path = tmp_path / "result.zip"
    package.save_new(path, b"original")
    with pytest.raises(FileExistsError): package.save_new(path, b"replacement")
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_size_limit(monkeypatch):
    monkeypatch.setattr(package, "MAX_BYTES", 1)
    with pytest.raises(ValueError, match="500 MB"):
        package.make_package(manifest(), responses())


@pytest.mark.parametrize("ids", [[True], [1, 1], [], [-1], [1.0]])
def test_invalid_manifest_indices(ids):
    spec = manifest()
    spec["datasets"]["Locomo-0"]["test_indices"] = ids
    with pytest.raises(ValueError): package.make_package(spec, responses())
