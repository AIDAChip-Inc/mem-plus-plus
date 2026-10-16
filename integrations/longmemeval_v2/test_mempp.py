"""Use actual pinned native Memory/trajectory contracts; gateway is synthetic."""
import copy
import json
from pathlib import Path
import shutil
import uuid
import base64
import hashlib
from importlib.metadata import PackagePath
from types import SimpleNamespace

from PIL import Image
import pytest

from memory_modules.memory import build_memory, load_memory
from integrations.longmemeval_v2 import mempp


class Gateway:
    data = {}
    uncertain_once = False

    def __init__(self, world):
        self.world = world
        self.rows = self.data.setdefault(world, {})

    def identities(self, source, count):
        return [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{self.world}:{source}:{i}")) for i in range(count)]

    def put(self, source, texts):
        ids = self.identities(source, len(texts))
        for key, text in zip(ids, texts):
            if key in self.rows and self.rows[key] != text:
                raise RuntimeError("Changed retry")
        self.rows.update(zip(ids, texts))
        if self.uncertain_once:
            self.uncertain_once = False
            raise OSError("Synthetic lost acknowledgement after atomic commit")

    def search(self, query, k):
        return [{"id": key, "content": "user: " + text} for key, text in sorted(self.rows.items())][:k]

    def verify(self, documents):
        if self.rows != {key: value["text"] for key, value in documents.items()}:
            raise RuntimeError("Database differs from the saved evidence snapshot")


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    Gateway.data = {}
    monkeypatch.setattr(mempp, "_Gateway", Gateway)
    root = tmp_path.resolve()
    Image.new("RGB", (8, 8), color="blue").save(root / "frame.png")
    trajectory = {"id": "synthetic-trajectory", "goal": "Find Atlas settings", "outcome": "success",
                  "start_url": "https://example.invalid/settings", "states": [
                      {"step": 3, "url": "https://example.invalid/settings", "action": "click Atlas",
                       "thought": "Read the settings page", "accessibility_tree": "保留原文 Atlas settings",
                       "screenshot": "frame.png"}]}
    config = {"memory_type": "mempp", "memory_params": {"trajectories_root_dir": str(root), "workspace_dir": str(root / "cache"), "top_k": 10}}
    return root, trajectory, config


def test_native_factory_sharing_and_independent_worlds(fixture):
    root, trajectory, config = fixture
    a = build_memory(config)
    b = build_memory(config)
    a.insert(trajectory)
    a.insert(trajectory)
    other = copy.deepcopy(trajectory); other["id"] = "second"
    a.insert(other)
    context = a.query("Atlas")
    assert len(context) == 4 and context[0]["type"] == "text" and context[1]["type"] == "image"
    state = json.loads(context[0]["value"].split("user: ", 1)[1])
    assert state["state"]["text"] == "保留原文 Atlas settings" and state["state"]["step"] == 3
    assert len(a._docs) == 2 and b.query("Atlas") == []
    a.set_query_context(query_invocation_id="opaque-only")
    assert a.query("Atlas") == context  # No question metadata/gold is consulted.
    assert Path(context[1]["value"]).read_bytes() == (root / "frame.png").read_bytes()
    assert not Path(context[1]["value"]).is_symlink()


def test_move_snapshot_delete_original_assets_and_load(fixture):
    root, trajectory, config = fixture
    a = build_memory(config); a.insert(trajectory)
    ids = set(a._docs)
    out = root / "snapshot"; a.save_memory(out)
    with pytest.raises(FileExistsError): a.save_memory(out)
    with pytest.raises(RuntimeError, match="immutable"): a.insert(trajectory)
    moved = root / "moved"; shutil.move(out, moved)
    shutil.rmtree(root / "cache"); (root / "frame.png").unlink()
    b = load_memory(moved, requested_config=config)
    assert b.world == a.world and set(b._docs) == ids
    assert Path(b.query("Atlas")[1]["value"]).is_file()
    assert Path(b.query("Atlas")[1]["value"]).parent == moved / "assets"
    with pytest.raises(RuntimeError, match="immutable"): b.insert(trajectory)


@pytest.mark.parametrize("change", ["text", "image"])
def test_same_id_changed_payload_rejected(fixture, change):
    root, trajectory, config = fixture
    a = build_memory(config); a.insert(trajectory)
    if change == "text": trajectory["states"][0]["accessibility_tree"] = "changed"
    else: Image.new("RGB", (8, 8), color="red").save(root / "frame.png")
    with pytest.raises(RuntimeError, match="changed"):
        a.insert(trajectory)
    assert len(a._docs) == 1


def test_uncertain_ack_retry_is_idempotent(fixture):
    _, trajectory, config = fixture
    a = build_memory(config); a._gateway.uncertain_once = True
    with pytest.raises(OSError): a.insert(trajectory)
    assert not a._docs
    a.insert(trajectory)
    assert len(a._docs) == len(a._gateway.rows) == 1


@pytest.mark.parametrize("change", ["missing_image", "image_bytes", "image_symlink", "db_missing", "db_foreign", "config"])
def test_load_fails_on_evidence_or_config_drift(fixture, change):
    root, trajectory, config = fixture
    a = build_memory(config); a.insert(trajectory)
    out = root / "snapshot"; a.save_memory(out)
    image = next((out / "assets").iterdir())
    if change == "missing_image": image.unlink()
    elif change == "image_bytes": image.write_bytes(b"changed")
    elif change == "image_symlink": image.unlink(); image.symlink_to(root / "frame.png")
    elif change == "db_missing": a._gateway.rows.clear()
    elif change == "db_foreign": a._gateway.rows[str(uuid.uuid4())] = "foreign"
    elif change == "config": config["memory_params"]["top_k"] = 1
    with pytest.raises(RuntimeError): load_memory(out, requested_config=config)


def test_query_detects_screenshot_change_and_foreign_record(fixture):
    _, trajectory, config = fixture
    a = build_memory(config); a.insert(trajectory)
    a._gateway.rows[str(uuid.uuid4())] = "foreign"
    with pytest.raises(RuntimeError, match="outside"): a.query("Atlas")
    a._gateway.rows = {k: v["text"] for k, v in a._docs.items()}
    next(a._assets.iterdir()).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="screenshot"): a.query("Atlas")


def test_native_legacy_content_shape(fixture):
    _, _, config = fixture
    a = build_memory(config)
    a.insert({"id": "legacy", "metadata": {"original_goal": "Legacy goal"}, "outcome": "success",
              "content": [{"step": 4, "url": "https://example.invalid/legacy", "action": None,
                           "thoughts": None, "observation": {"text": "Legacy state", "screenshot": "frame.png"}}]})
    context = a.query("Legacy")
    assert "Legacy goal" in context[0]["value"] and "Legacy state" in context[0]["value"]


@pytest.mark.parametrize("params", [{"api_key": "not-allowed"}, {"top_k": True}, {"top_k": 101}])
def test_config_rejects_secret_fields_and_invalid_k(fixture, params):
    _, _, config = fixture
    config["memory_params"].update(params)
    with pytest.raises(ValueError): build_memory(config)


def test_build_driver_rejects_model_execution(monkeypatch):
    from integrations.longmemeval_v2 import build_only
    monkeypatch.setattr(build_only.sys, "argv", ["build_only", "--save-memory"])
    with pytest.raises(RuntimeError, match="skip-evaluation"):
        build_only.main()


def test_imported_engine_sources_checked_against_wheel_record(tmp_path):
    file = tmp_path / "__init__.py"
    file.write_bytes(b"SYNTHETIC original source")
    record = PackagePath("memory/__init__.py")
    record.hash = SimpleNamespace(mode="sha256", value=base64.urlsafe_b64encode(hashlib.sha256(file.read_bytes()).digest()).decode().rstrip("="))
    distribution = SimpleNamespace(files=[record])
    mempp.verify_engine_sources(distribution, tmp_path)
    file.write_bytes(b"SYNTHETIC changed source")
    with pytest.raises(RuntimeError, match="RECORD"):
        mempp.verify_engine_sources(distribution, tmp_path)
    file.write_bytes(b"SYNTHETIC original source")
    (tmp_path / "extra.py").write_text("SYNTHETIC extra")
    with pytest.raises(RuntimeError, match="file set"):
        mempp.verify_engine_sources(distribution, tmp_path)
