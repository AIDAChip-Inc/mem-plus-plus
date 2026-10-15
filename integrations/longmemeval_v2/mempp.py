"""Text-guided Mem++ trajectory retrieval with paired, checksummed screenshots."""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from PIL import Image

from memory_modules.memory import Memory, register_memory
from memory_modules.trajectory_store import prepare_trajectory_insert

ENGINE_COMMIT = "d952129834f1749c412a04528a4f9138d94a7d74"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_engine_sources(distribution, actual_root):
    expected = {str(f)[len("memory/"):]: f for f in distribution.files or []
                if str(f).startswith("memory/") and str(f).endswith(".py")}
    actual_files = {str(f.relative_to(actual_root)) for f in actual_root.rglob("*.py")}
    if not expected or set(expected) != actual_files:
        raise RuntimeError("Imported Mem++ source file set differs from the pinned wheel")
    for name, record in expected.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256((actual_root / name).read_bytes()).digest()).decode().rstrip("=")
        if record.hash is None or record.hash.mode != "sha256" or record.hash.value != encoded:
            raise RuntimeError("Imported Mem++ source differs from its installed wheel RECORD hash")


class _Gateway:
    """Use the pinned engine directly; no network listener or generative model."""
    def __init__(self, world):
        self.world = world
        self.service = None

    def ready(self):
        if self.service is not None:
            return
        if os.environ.get("PYTHON_DOTENV_DISABLED") != "1":
            raise RuntimeError("Set PYTHON_DOTENV_DISABLED=1 and configure the dedicated database explicitly")
        distribution = importlib.metadata.distribution("memory-research")
        origin = json.loads(distribution.read_text("direct_url.json") or "{}")
        if origin.get("vcs_info", {}).get("commit_id") != ENGINE_COMMIT:
            raise RuntimeError("Install the documented pinned Mem++ engine")
        import memory
        actual_root = Path(memory.__file__).resolve().parent
        verify_engine_sources(distribution, actual_root)
        from memory import config
        from memory.db import get_engine, get_session
        from memory.embeddings import get_embedding_service
        from memory.aml_api import AMLService
        from sqlalchemy import text
        if get_engine().dialect.name != "postgresql":
            raise RuntimeError("Use migrated UTF-8 PostgreSQL/pgvector")
        if (config.MEMORY_EMBEDDING_BACKEND != "onnx" or not config.MEMORY_EMBEDDINGS_ENABLED
                or config.EMBEDDING_MODEL != "all-MiniLM-L6-v2" or config.EMBEDDING_DIM != 384):
            raise RuntimeError("Use enabled local ONNX MiniLM-L6-v2 384 embeddings")
        actual = (config.MEMORY_RRF_W_FUZZY, config.MEMORY_RRF_W_TAG, config.MEMORY_RRF_W_VECTOR,
                  config.CANDIDATE_LIMIT, config.MEMORY_LEXICAL_OR)
        if actual != (1., 1., 2., 200, True):
            raise RuntimeError("Use RRF1/1/2, candidates200 and lexicalOR")
        with get_session() as db:
            if db.execute(text("SHOW server_encoding")).scalar_one() != "UTF8":
                raise RuntimeError("Use a UTF-8 database")
            dimension = db.execute(text("SELECT format_type(atttypid,atttypmod) FROM pg_attribute WHERE attrelid='agent_memory'::regclass AND attname='embedding'")).scalar_one()
            if dimension != "vector(384)":
                raise RuntimeError("Use vector384 schema")
        if not get_embedding_service("memory").warm_load():
            raise RuntimeError("Local embedding model unavailable")
        self.service = AMLService(get_session, "unused-in-process-token-" + uuid.uuid4().hex)

    def identities(self, source, count):
        self.ready()
        from memory.aml_api import _identity
        _, project = _identity(self.world)
        request = uuid.uuid5(project, source)
        return [str(uuid.uuid5(request, str(i))) for i in range(count)]

    def put(self, source, texts):
        self.ready()
        from memory.aml_api import AddRequest, Message
        self.service.add(AddRequest(request_id=source, user_id=self.world,
                                    session_id="longmemeval-v2",
                                    messages=[Message(role="user", content=t) for t in texts]))

    def search(self, query, k):
        self.ready()
        from memory.aml_api import SearchRequest
        return self.service.search(SearchRequest(user_id=self.world, query=query, top_k=k))["data"]

    def verify(self, documents):
        self.ready()
        from memory.aml_api import _identity, _AGENT
        from memory.models import AgentMemory
        uid, pid = _identity(self.world)
        with self.service.sessions() as db:
            rows = db.query(AgentMemory).filter_by(user_id=uid, project_id=pid, agent_type=_AGENT).all()
            if len(rows) != len(documents) or any(
                str(r.id) not in documents or r.content != "user: " + documents[str(r.id)]["text"]
                or r.embedding is None or len(r.embedding) != 384 for r in rows
            ):
                raise RuntimeError("Database differs from the saved evidence snapshot")


@register_memory
class MemppMemory(Memory):
    memory_type = "mempp"

    def __init__(self, memory_params):
        super().__init__(memory_params)
        if set(memory_params) - {"top_k", "trajectories_root_dir", "workspace_dir"}:
            raise ValueError("Unknown memory parameter; keep credentials in the environment")
        for key in ["trajectories_root_dir", "workspace_dir"]:
            if not isinstance(memory_params.get(key), str) or not memory_params[key].strip():
                raise ValueError(f"Configure {key}")
        self.k = memory_params.get("top_k", 10)
        if type(self.k) is not int or not 1 <= self.k <= 100:
            raise ValueError("top_k must be an integer in 1..100")
        self.world = "longmemeval-v2:" + str(uuid.uuid4())
        self._gateway = _Gateway(self.world)
        self._assets = Path(memory_params["workspace_dir"]).resolve() / self.world.split(":")[1] / "assets"
        self._docs = {}
        self._trajectories = {}
        self._sealed = False

    def insert(self, trajectory):
        if self._sealed:
            raise RuntimeError("Saved/loaded evidence snapshots are immutable; build a new experiment")
        prepared = prepare_trajectory_insert(trajectory, trajectories_root_dir=Path(self.memory_params["trajectories_root_dir"]).resolve())
        image_hashes = [digest(p) for p in prepared.screenshot_sources]
        fingerprint = hashlib.sha256(canonical([prepared.fingerprint, image_hashes]).encode()).hexdigest()
        old = self._trajectories.get(prepared.trajectory_id)
        if old is not None:
            if old != fingerprint:
                raise RuntimeError("Trajectory ID reused with changed text or screenshot bytes")
            return
        texts, images = [], []
        data = prepared.simplified
        for index, state in enumerate(data["states"]):
            source = prepared.screenshot_sources[index]
            with Image.open(source) as image:
                extension = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif"}.get(image.format)
                image.verify()
            if extension is None:
                raise ValueError("Unsupported screenshot image format")
            name = image_hashes[index] + extension
            # Explicit copies, never native relative symlinks to source data.
            self._assets.mkdir(parents=True, exist_ok=True)
            target = self._assets / name
            if not target.exists():
                shutil.copyfile(source, target)
            if digest(target) != image_hashes[index]:
                raise RuntimeError("Screenshot changed during materialization")
            images.append(name)
            texts.append(canonical({"trajectory_id": data["id"], "goal": data["goal"],
                                    "outcome": data["outcome"], "start_url": data["start_url"],
                                    "state": {k: v for k, v in state.items() if k != "screenshot"}}))
        source_id = "trajectory:" + hashlib.sha256(prepared.trajectory_id.encode()).hexdigest()
        ids = self._gateway.identities(source_id, len(texts))
        self._gateway.put(source_id, texts)
        self._docs.update({key: {"text": text, "image": image, "sha256": sha}
                           for key, text, image, sha in zip(ids, texts, images, image_hashes)})
        self._trajectories[prepared.trajectory_id] = fingerprint

    def query(self, query, query_image=None):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be nonempty text")
        if query_image is not None and not Path(query_image).is_file():
            raise ValueError("Missing query image")
        # Text-guided retrieval only: paired screenshots are returned to the
        # native reader, but neither source nor query images are image-embedded.
        rows = self._gateway.search(query, self.k)
        result, seen = [], set()
        for row in rows:
            key = row["id"]
            doc = self._docs.get(key)
            if key in seen or doc is None or row["content"] != "user: " + doc["text"]:
                raise RuntimeError("Retrieved evidence is outside the indexed snapshot")
            seen.add(key)
            image = self._checked_image(doc)
            result.extend([{"type": "text", "value": f"[{key}]\n{row['content']}"},
                           {"type": "image", "value": str(image)}])
        return result

    def _checked_image(self, doc):
        name = doc["image"]
        if (Path(name).suffix not in {".png", ".jpg", ".webp", ".gif"}
                or name != doc["sha256"] + Path(name).suffix
                or len(doc["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in doc["sha256"])):
            raise RuntimeError("Unsafe screenshot reference")
        path = self._assets / name
        if path.is_symlink() or not path.is_file() or digest(path) != doc["sha256"]:
            raise RuntimeError("Missing or changed evidence screenshot")
        return path

    def save_memory(self, output_dir):
        output = Path(output_dir)
        if output.exists():
            raise FileExistsError("Refusing to overwrite a saved memory")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".mempp-save-", dir=output.parent) as temporary:
            stage = Path(temporary)
            (stage / "memory_config.json").write_text(canonical(self.memory_config) + "\n", encoding="utf-8")
            self._save_backend(stage)
            os.rename(stage, output)
        self._sealed = True

    def _save_backend(self, output_dir):
        self._gateway.verify(self._docs)
        assets = output_dir / "assets"
        assets.mkdir()
        for doc in self._docs.values():
            shutil.copyfile(self._checked_image(doc), assets / doc["image"])
        snapshot = {"schema": 1, "world": self.world, "engine_commit": ENGINE_COMMIT,
                    "memory_config": self.memory_config, "documents": self._docs, "trajectories": self._trajectories}
        (output_dir / "mempp_snapshot.json").write_text(canonical(snapshot) + "\n", encoding="utf-8")

    def _load_backend(self, input_dir):
        snapshot = json.loads((input_dir / "mempp_snapshot.json").read_text(encoding="utf-8"))
        if (snapshot["schema"] != 1 or snapshot["engine_commit"] != ENGINE_COMMIT
                or snapshot["memory_config"] != self.memory_config
                or not snapshot["world"].startswith("longmemeval-v2:")):
            raise RuntimeError("Incompatible evidence snapshot")
        uuid.UUID(snapshot["world"].split(":", 1)[1])
        self.world = snapshot["world"]
        self._assets = input_dir.resolve() / "assets"
        if self._assets.is_symlink():
            raise RuntimeError("Snapshot assets must be copied, not symlinked")
        self._docs = snapshot["documents"]
        self._trajectories = snapshot["trajectories"]
        for doc in self._docs.values():
            self._checked_image(doc)
        self._gateway = _Gateway(self.world)
        self._gateway.verify(self._docs)
        self._sealed = True
