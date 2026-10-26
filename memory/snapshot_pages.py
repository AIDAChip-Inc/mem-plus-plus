"""Content-addressed snapshot chunks and verified, disposable immutable caches."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from memory.object_store import PublicationConflict, object_limit

CHUNK_SIZE = 64 * 1024
_cache: OrderedDict[tuple[str, str], bytes] = OrderedDict()
_cache_bytes = 0
_lock = threading.Lock()


def reset_cache():
    """Discard the process cache; durable objects remain authoritative."""
    global _cache_bytes
    with _lock:
        _cache.clear()
        _cache_bytes = 0


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _disk_path(store, key):
    directory = os.environ.get("MEMORY_OBJECT_STORE_CACHE_DIR")
    if not directory:
        return None
    namespace = digest(store.cache_key.encode())
    return Path(directory) / namespace / digest(key.encode())


def remember(store, key, data):
    """Only immutable objects belong here. Never cache latest.json."""
    global _cache_bytes
    limit = int(os.environ.get("MEMORY_OBJECT_STORE_CACHE_BYTES", str(64 * 1024**2)))
    if limit < 0:
        raise ValueError("MEMORY_OBJECT_STORE_CACHE_BYTES must be nonnegative")
    with _lock:
        cache_key = (store.cache_key, key)
        previous = _cache.pop(cache_key, None)
        if previous is not None:
            _cache_bytes -= len(previous)
        if len(data) <= limit:
            _cache[cache_key] = data
            _cache_bytes += len(data)
        while _cache and _cache_bytes > limit:
            _cache_bytes -= len(_cache.popitem(last=False)[1])
    path = _disk_path(store, key)
    if path is not None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            fd, name = tempfile.mkstemp(dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                os.replace(name, path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        except OSError:
            # Cache durability/availability is never a prerequisite for commit.
            pass


def immutable(store, key, expected, *, max_bytes=None):
    limit = object_limit(max_bytes)
    with _lock:
        cache_key = (store.cache_key, key)
        data = _cache.get(cache_key)
        if data is not None:
            _cache.move_to_end(cache_key)
    if data is not None and len(data) <= limit and digest(data) == expected:
        return data
    path = _disk_path(store, key)
    if path is not None:
        try:
            # Caches are disposable and may be corrupt. Check size before any
            # allocation, then bound the read even if another process replaces it.
            if path.stat().st_size > limit:
                raise OSError("Oversized disposable cache entry")
            with path.open("rb") as stream:
                data = stream.read(limit + 1)
            if len(data) <= limit and digest(data) == expected:
                remember(store, key, data)
                return data
        except OSError:
            pass
    obj = store.get(key, max_bytes=limit)
    if obj is None or digest(obj.data) != expected:
        raise ValueError("Published snapshot is missing or has an invalid checksum")
    remember(store, key, obj.data)
    return obj.data


def put_immutable(store, key, data):
    try:
        store.put(key, data)
    except PublicationConflict:
        obj = store.get(key)
        if obj is None or obj.data != data:
            raise ValueError("Immutable object key contains different data")
    remember(store, key, data)


def size_check(size):
    limit = int(os.environ.get("MEMORY_OBJECT_STORE_MAX_BYTES", "0"))
    if limit < 0:
        raise ValueError("MEMORY_OBJECT_STORE_MAX_BYTES must be nonnegative")
    if limit and size > limit:
        raise ValueError("Corpus exceeds MEMORY_OBJECT_STORE_MAX_BYTES")


def load(store, profile):
    pointer = store.get("latest.json")
    if pointer is None:
        return None, None, set()
    manifest = json.loads(pointer.data)
    sha = manifest.get("sha256", "")
    if (manifest.get("format") not in {1, 2} or manifest.get("embedding") != profile
            or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha)):
        raise ValueError("Unsupported snapshot format or embedding profile")
    if manifest["format"] == 1:
        # Read compatibility with the first PR revision; next commit upgrades it.
        data = immutable(store, f"snapshots/{sha}.sqlite", sha)
        size_check(len(data))
        return data, pointer.etag, set()
    index = json.loads(immutable(store, f"manifests/{sha}.json", sha))
    chunks, size = index.get("chunks"), index.get("size")
    if (index.get("format") != 2 or index.get("embedding") != profile
            or index.get("chunk_size") != CHUNK_SIZE or type(size) is not int or size <= 0
            or not isinstance(chunks, list) or len(chunks) != (size + CHUNK_SIZE - 1) // CHUNK_SIZE
            or any(not isinstance(h, str) or not re.fullmatch(r"[0-9a-f]{64}", h) for h in chunks)):
        raise ValueError("Invalid chunk manifest")
    size_check(size)
    sizes = {}
    for index, sha in enumerate(chunks):
        length = min(CHUNK_SIZE, size - index * CHUNK_SIZE)
        if sha in sizes and sizes[sha] != length:
            raise ValueError("Invalid chunk lengths in manifest")
        sizes[sha] = length
    unique = list(sizes)
    def fetch(sha):
        chunk = immutable(store, f"chunks/{sha}", sha, max_bytes=sizes[sha])
        if len(chunk) != sizes[sha]:
            raise ValueError("Invalid chunk length")
        return chunk
    with ThreadPoolExecutor(max_workers=min(16, len(unique))) as pool:
        loaded = dict(zip(unique, pool.map(fetch, unique)))
    data = b"".join(loaded[h] for h in chunks)
    if len(data) != size:
        raise ValueError("Invalid snapshot size")
    return data, pointer.etag, set(chunks)


def publish(store, data, profile, known, expected):
    size_check(len(data))
    hashes = []
    for offset in range(0, len(data), CHUNK_SIZE):
        chunk = data[offset:offset + CHUNK_SIZE]
        sha = digest(chunk)
        if sha not in known:
            put_immutable(store, f"chunks/{sha}", chunk)
            known.add(sha)
        hashes.append(sha)
    index = encoded({"format": 2, "embedding": profile, "size": len(data),
                     "chunk_size": CHUNK_SIZE, "chunks": hashes})
    sha = digest(index)
    put_immutable(store, f"manifests/{sha}.json", index)
    pointer = encoded({"format": 2, "embedding": profile, "sha256": sha})
    return store.put("latest.json", pointer, expected=expected)
