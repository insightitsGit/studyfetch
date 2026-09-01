from __future__ import annotations

import hashlib
from pathlib import Path

from app.config import settings


class BlobStore:
    """Local filesystem by default; Azure Blob when a connection string is set."""

    def __init__(self) -> None:
        self._azure = None
        self._container = settings.azure_blob_container
        if settings.azure_storage_connection_string:
            from azure.storage.blob import BlobServiceClient

            client = BlobServiceClient.from_connection_string(
                settings.azure_storage_connection_string
            )
            self._azure = client.get_container_client(self._container)
            try:
                self._azure.create_container()
            except Exception:
                pass
        settings.blob_dir.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        if self._azure is not None:
            self._azure.upload_blob(key, data, overwrite=True, content_type=content_type)
            return f"az://{self._container}/{key}"
        dest = settings.blob_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return f"file://{dest.as_posix()}"

    def get(self, uri: str) -> bytes:
        if uri.startswith("az://"):
            if self._azure is None:
                raise RuntimeError("Azure URI stored but no connection string configured")
            key = uri.split("/", 3)[-1]
            return self._azure.download_blob(key).readall()
        if uri.startswith("file://"):
            return Path(uri.removeprefix("file://")).read_bytes()
        path = Path(uri)
        if path.exists():
            return path.read_bytes()
        raise FileNotFoundError(uri)

    def delete_prefix(self, prefix: str) -> None:
        if self._azure is not None:
            for blob in self._azure.list_blobs(name_starts_with=prefix):
                self._azure.delete_blob(blob.name)
            return
        root = settings.blob_dir / prefix
        if root.exists():
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def local_path(self, uri: str) -> Path | None:
        if uri.startswith("file://"):
            return Path(uri.removeprefix("file://"))
        return None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


blob_store = BlobStore()
