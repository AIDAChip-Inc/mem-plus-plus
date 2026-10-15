"""Validate MemoryBench answers and create a deterministic website-upload ZIP.

No model execution, upload, credentials or score calculation occurs here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

MAX_BYTES = 500_000_000  # Conservative decimal interpretation of the website's 500 MB.
REPO = "THUIR/MemoryBench"


def dataset_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
        raise ValueError("Unsafe or unsupported dataset name")
    return value


def indices(values):
    if not isinstance(values, list) or not values:
        raise ValueError("Expected a nonempty list of test indices")
    if any(type(v) is not int or v < 0 for v in values):
        raise ValueError("test_idx must be a nonnegative integer, not a boolean")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate test indices")
    return sorted(values)


def reject_constant(value):
    raise ValueError(f"Nonstandard JSON constant: {value}")


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=reject_constant,
                      object_pairs_hook=reject_duplicate_keys)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def save_new(path, data):
    """Publish only a complete file; never overwrite an existing destination."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def fetch_manifest(names, revision):
    """Read all official test Arrow shards at an explicit immutable HF commit."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full 40-character Hugging Face dataset commit SHA")
    if len(set(names)) != len(names) or not names:
        raise ValueError("Specify unique dataset names")
    # Optional dependencies are needed only to fetch the index manifest.
    from huggingface_hub import hf_hub_download
    import pyarrow.ipc as ipc

    def download(filename):
        return Path(hf_hub_download(REPO, filename, repo_type="dataset",
                                   revision=revision, token=False))

    datasets = {}
    for name in names:
        dataset_name(name)
        prefix = f"dataset/{name}/test/"
        state_path = download(prefix + "state.json")
        state = read_json(state_path)
        shards, found = [], []
        for entry in state["_data_files"]:
            filename = entry["filename"]
            if not re.fullmatch(r"[A-Za-z0-9_-]+\.arrow", filename):
                raise ValueError("Unsupported upstream Arrow filename")
            path = download(prefix + filename)
            with path.open("rb") as stream:
                table = ipc.open_stream(stream).read_all()
            found.extend(table.column("test_idx").to_pylist())
            shards.append({"path": prefix + filename,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        datasets[name] = {"test_indices": indices(found), "shards": shards,
                          "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest()}
    return {"schema_version": 1, "repository": REPO, "revision": revision,
            "datasets": datasets}


def make_package(manifest, predictions):
    """Require exact coverage for each included dataset; omit all extra row fields."""
    if (manifest.get("schema_version") != 1 or manifest.get("repository") != REPO
            or not re.fullmatch(r"[0-9a-f]{40}", manifest.get("revision", ""))):
        raise ValueError("Expected a pinned official MemoryBench index manifest")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Expected a nonempty array of actual model responses")
    grouped = {}
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError("Each prediction must be an object")
        name = dataset_name(row.get("dataset"))
        if name not in manifest["datasets"]:
            raise ValueError(f"Dataset absent from manifest: {name}")
        idx = row.get("test_idx")
        if type(idx) is not int or idx < 0 or not isinstance(row.get("response"), str):
            raise ValueError("Each row needs integer test_idx and string response")
        rows = grouped.setdefault(name, {})
        if idx in rows:
            raise ValueError(f"Duplicate prediction: {name}/{idx}")
        rows[idx] = {"test_idx": idx, "response": row["response"], "dataset": name}
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, rows in sorted(grouped.items()):
            expected = set(indices(manifest["datasets"][name]["test_indices"]))
            actual = set(rows)
            if expected != actual:
                raise ValueError(f"{name}: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
            info = zipfile.ZipInfo(f"dataset/{name}/predict.json", (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, encoded([rows[idx] for idx in sorted(rows)]))
    payload = output.getvalue()
    if len(payload) > MAX_BYTES:
        raise ValueError("ZIP exceeds the conservative 500 MB upload limit")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch-manifest")
    fetch.add_argument("--dataset", action="append", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--output", type=Path, required=True)
    pack = commands.add_parser("package")
    pack.add_argument("--manifest", type=Path, required=True)
    pack.add_argument("--predictions", type=Path, required=True)
    pack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "fetch-manifest":
            payload = encoded(fetch_manifest(args.dataset, args.revision))
        else:
            payload = make_package(read_json(args.manifest), read_json(args.predictions))
        save_new(args.output, payload)
    except (ValueError, KeyError, TypeError, FileExistsError) as exc:
        parser.exit(2, f"Validation failed: {exc}\n")
    print(f"Saved {args.output}; sha256={hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
