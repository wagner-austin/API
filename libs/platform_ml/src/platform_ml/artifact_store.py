from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Final

from platform_core import DataBankClient
from platform_core.data_bank_protocol import FileUploadResponse

from . import _test_hooks
from .tarball import TarballError, extract_tarball


class ArtifactStoreError(Exception):
    """Base error for artifact store operations."""


class ArtifactStore:
    """ML artifact storage via data-bank-api.

    All artifacts are stored remotely. No local-only mode.

    Args:
        client: A DataBankClient instance. For testing, create with a fake
            HTTP transport: `DataBankClient(..., client=httpx.Client(transport=fake))`
    """

    def __init__(self, client: DataBankClient) -> None:
        self._client: Final[DataBankClient] = client

    def upload_artifact(
        self,
        dir_path: Path,
        *,
        artifact_name: str,
        request_id: str,
    ) -> FileUploadResponse:
        """Upload a directory as a tarball artifact.

        Raises:
            ArtifactStoreError: On validation or upload failure
        """
        if not dir_path.exists() or not dir_path.is_dir():
            raise ArtifactStoreError("artifact directory does not exist")
        name = artifact_name.strip()
        if name == "":
            raise ArtifactStoreError("artifact_name must be non-empty")
        # Stage tarball in a temporary directory for upload
        with tempfile.TemporaryDirectory(prefix="artifact-") as tmp:
            tmp_dir = Path(tmp)
            tar_path = tmp_dir / f"{name}.tar.gz"
            try:
                _test_hooks.create_tarball(dir_path, tar_path, root_name=name)
            except TarballError as exc:
                raise ArtifactStoreError(f"tarball creation failed: {exc}") from exc
            # Compute sha256 for observability (server returns its own)
            sha = _sha256_file(tar_path)
            # Upload; content-type is a best-effort hint, server decides
            with tar_path.open("rb") as f:
                resp = self._client.upload(
                    file_id=f"{name}.tar.gz",
                    stream=f,
                    content_type="application/gzip",
                    request_id=request_id,
                )
            # Basic cross-check: ensure server size matches our file size
            size = tar_path.stat().st_size
            if int(resp["size"]) != int(size):
                raise ArtifactStoreError("uploaded size mismatch")
            # No attempt to reconcile sha256s; trust server
            _ = sha
            return resp

    def download_artifact(
        self,
        file_id: str,
        *,
        dest_dir: Path,
        request_id: str,
        expected_root: str,
    ) -> Path:
        """Download and extract a tarball artifact.

        Raises:
            ArtifactStoreError: On download, extraction, or validation failure
        """
        fid = file_id.strip()
        if fid == "":
            raise ArtifactStoreError("file_id must be non-empty")
        dest_dir.mkdir(parents=True, exist_ok=True)
        # Use a deterministic temporary file name under dest_dir
        tmp_path = dest_dir / f".{os.path.basename(fid)}.part"
        # Check file size before attempting download
        head_info = self._client.head(fid, request_id=request_id)
        if head_info["size"] <= 0:
            raise ArtifactStoreError("downloaded file has invalid size")
        self._client.download_to_path(fid, tmp_path, request_id=request_id, verify_etag=True)
        # Extract ensuring the archive contains the expected root
        try:
            out_root = extract_tarball(tmp_path, dest_dir, expected_root=expected_root)
        except TarballError as exc:
            raise ArtifactStoreError(f"tarball extraction failed: {exc}") from exc
        # Leave tmp tarball for caller to manage if needed
        return out_root


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["ArtifactStore", "ArtifactStoreError"]
