"""Small conditional-write object interface used by the snapshot backend."""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse


class PublicationConflict(RuntimeError):
    """Another writer published first. Close the session and retry from latest."""


def object_limit(max_bytes: int | None = None) -> int:
    limit = int(os.environ.get("MEMORY_OBJECT_STORE_MAX_OBJECT_BYTES", str(1024**3)))
    if limit <= 0 or (max_bytes is not None and max_bytes <= 0):
        raise ValueError("Object byte limits must be positive")
    return min(limit, max_bytes) if max_bytes is not None else limit


@dataclass(frozen=True)
class Object:
    data: bytes
    etag: str


class ObjectStore(Protocol):
    """Keys are relative; expected=None means create only, never unconditional PUT.

    Implementations must provide atomic conditional writes and read-after-write
    consistency. Missing keys return None; auth/network failures must propagate.
    """

    cache_key: str

    def get(self, key: str, *, max_bytes: int | None = None) -> Object | None: ...

    def put(self, key: str, data: bytes, *, expected: str | None = None) -> str: ...


class FileObjectStore:
    """POSIX test/deployment adapter. Requires flock, atomic rename and fsync.

    Do not point this at an S3/FUSE mount without these filesystem guarantees;
    use the S3 adapter for object stores instead.
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.cache_key = self.root.as_uri()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Object key must stay within the configured directory")
        return path

    def get(self, key: str, *, max_bytes: int | None = None) -> Object | None:
        limit = object_limit(max_bytes)
        try:
            path = self._path(key)
            if path.stat().st_size > limit:
                raise ValueError("Object exceeds its byte limit")
            with path.open("rb") as stream:
                data = stream.read(limit + 1)
            if len(data) > limit:
                raise ValueError("Object exceeds its byte limit")
        except FileNotFoundError:
            return None
        return Object(data, hashlib.sha256(data).hexdigest())

    def put(self, key: str, data: bytes, *, expected: str | None = None) -> str:
        import fcntl

        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # A stable lock inode coordinates separate processes, including initial
        # creation. Never lock the pointer itself: atomic rename changes its inode.
        with (self.root / ".publication.lock").open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            current = self.get(key)
            if (current.etag if current else None) != expected:
                raise PublicationConflict("Object changed; reload latest and retry")
            fd, name = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(name, path)
                directory = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        return hashlib.sha256(data).hexdigest()


class S3ObjectStore:
    """S3 or compatible endpoint with GetObject and conditional PutObject support."""

    def __init__(self, bucket: str, prefix: str = "", *, client=None):
        if client is None:
            import boto3

            client = boto3.client("s3", endpoint_url=os.environ.get("MEMORY_S3_ENDPOINT_URL"))
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.cache_key = f"s3:{self.client.meta.endpoint_url}:{bucket}/{self.prefix}"

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def get(self, key: str, *, max_bytes: int | None = None) -> Object | None:
        from botocore.exceptions import ClientError

        try:
            response = self.client.get_object(Bucket=self.bucket, Key=self._key(key))
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "NoSuchKey":
                return None
            raise
        body = response["Body"]
        try:
            # Bound allocations even for a corrupt/untrusted remote object.
            limit = object_limit(max_bytes)
            if response["ContentLength"] > limit:
                raise ValueError("Object exceeds MEMORY_OBJECT_STORE_MAX_OBJECT_BYTES")
            data = body.read(limit + 1)
            if len(data) > limit:
                raise ValueError("Object exceeds MEMORY_OBJECT_STORE_MAX_OBJECT_BYTES")
            return Object(data, response["ETag"])
        finally:
            body.close()

    def put(self, key: str, data: bytes, *, expected: str | None = None) -> str:
        from botocore.exceptions import ClientError

        condition = {"IfNoneMatch": "*"} if expected is None else {"IfMatch": expected}
        try:
            result = self.client.put_object(
                Bucket=self.bucket, Key=self._key(key), Body=data, **condition,
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"PreconditionFailed", "ConditionalRequestConflict"}:
                raise PublicationConflict("Object changed; reload latest and retry") from exc
            raise
        return result["ETag"]


def open_object_store(uri: str) -> ObjectStore:
    location = urlparse(uri)
    if location.query or location.fragment:
        raise ValueError("Object-store URI cannot include a query or fragment")
    if location.scheme == "s3" and location.netloc:
        return S3ObjectStore(location.netloc, location.path.lstrip("/"))
    if location.scheme == "file" and not location.netloc and location.path.startswith("/"):
        return FileObjectStore(Path(unquote(location.path)))
    raise ValueError("MEMORY_OBJECT_STORE_URI must be s3://bucket/prefix or file:///absolute/path")
