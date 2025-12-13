"""Tests for artifact_store module using fake HTTP transport.

These tests exercise real ArtifactStore and DataBankClient code,
only faking the HTTP responses at the transport layer.
"""

from __future__ import annotations

import hashlib
import tarfile
from pathlib import Path

import httpx
from platform_core import DataBankClient

from platform_ml import _test_hooks
from platform_ml._test_hooks import _CreateTarballProtocol
from platform_ml.artifact_store import ArtifactStore, ArtifactStoreError
from platform_ml.tarball import TarballError

from .http_fakes import FakeHttpTransport


def _compute_sha256(data: bytes) -> str:
    """Compute SHA256 hash of bytes."""
    return hashlib.sha256(data).hexdigest()


def _make_client(transport: FakeHttpTransport) -> DataBankClient:
    """Create a DataBankClient with fake HTTP transport."""
    http_client = httpx.Client(transport=transport)
    return DataBankClient("http://data-bank", "test-key", client=http_client)


class EchoSizeTransport(FakeHttpTransport):
    """Transport that extracts actual file size from multipart upload for tests."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Handle upload by extracting actual file size from multipart body."""
        if request.method == "POST" and "/files" in str(request.url):
            # Read streaming request body
            request.read()
            body = request.content
            # Find tarball content between boundaries (look for gzip magic bytes)
            # Gzip files start with \x1f\x8b
            gzip_start = body.find(b"\x1f\x8b")
            if gzip_start >= 0:
                # Find the end boundary (starts with \r\n--)
                end_boundary = body.find(b"\r\n--", gzip_start)
                if end_boundary >= 0:
                    file_size = end_boundary - gzip_start
                else:
                    file_size = len(body) - gzip_start
            else:
                file_size = 0

            response_json: dict[str, str | int | None] = {
                "file_id": "my-model.tar.gz",
                "size": file_size,
                "sha256": "abc123",
                "content_type": "application/gzip",
                "created_at": None,
            }
            return httpx.Response(
                status_code=200,
                json=response_json,
                request=request,
            )
        return super().handle_request(request)


def test_artifact_store_upload_success(tmp_path: Path) -> None:
    """Test successful upload using real code with fake HTTP."""
    # Prepare artifact directory
    src_dir = tmp_path / "model_dir"
    (src_dir / "sub").mkdir(parents=True)
    (src_dir / "sub" / "a.txt").write_text("hello", encoding="utf-8")
    (src_dir / "weights.bin").write_bytes(b"\x00\x01\x02\x03")

    # Create transport that echoes content-length as response size
    transport = EchoSizeTransport()

    # Create store with DataBankClient using fake transport
    http_client = httpx.Client(transport=transport)
    client = DataBankClient("http://data-bank", "test-key", client=http_client)
    store = ArtifactStore(client)

    # This tests REAL code: ArtifactStore -> DataBankClient -> (fake HTTP)
    resp = store.upload_artifact(src_dir, artifact_name="my-model", request_id="req-1")

    assert resp["file_id"] == "my-model.tar.gz"
    assert resp["size"] > 0


def test_artifact_store_upload_empty_name_fails(tmp_path: Path) -> None:
    """Test that empty artifact name raises error - no HTTP needed."""
    src_dir = tmp_path / "model"
    src_dir.mkdir()
    (src_dir / "data.bin").write_bytes(b"\x00")

    transport = FakeHttpTransport()
    client = _make_client(transport)
    store = ArtifactStore(client)

    import pytest

    with pytest.raises(ArtifactStoreError, match="artifact_name must be non-empty"):
        store.upload_artifact(src_dir, artifact_name="  ", request_id="req-1")


def test_artifact_store_upload_missing_dir_fails(tmp_path: Path) -> None:
    """Test that missing directory raises error - no HTTP needed."""
    transport = FakeHttpTransport()
    client = _make_client(transport)
    store = ArtifactStore(client)

    import pytest

    with pytest.raises(ArtifactStoreError, match="artifact directory does not exist"):
        store.upload_artifact(tmp_path / "missing", artifact_name="model", request_id="req-1")


def test_artifact_store_upload_size_mismatch(tmp_path: Path) -> None:
    """Test that size mismatch from server raises error."""
    src_dir = tmp_path / "model"
    src_dir.mkdir()
    (src_dir / "data.bin").write_bytes(b"\x00" * 100)

    transport = FakeHttpTransport()
    # Return wrong size to trigger mismatch
    transport.add_response(
        "POST",
        "/files",
        200,
        json_body={
            "file_id": "model.tar.gz",
            "size": 999999,  # Wrong size
            "sha256": "abc",
            "content_type": "application/gzip",
            "created_at": None,
        },
    )

    client = _make_client(transport)
    store = ArtifactStore(client)

    import pytest

    with pytest.raises(ArtifactStoreError, match="uploaded size mismatch"):
        store.upload_artifact(src_dir, artifact_name="model", request_id="req-1")


def test_artifact_store_download_success(tmp_path: Path) -> None:
    """Test successful download using real code with fake HTTP."""
    # Create a real tarball to serve
    src_dir = tmp_path / "original"
    src_dir.mkdir()
    (src_dir / "data.txt").write_text("test content", encoding="utf-8")

    tarball_path = tmp_path / "test.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tf:
        tf.add(src_dir, arcname="my-artifact")

    tarball_bytes = tarball_path.read_bytes()
    tarball_sha = _compute_sha256(tarball_bytes)

    transport = FakeHttpTransport()

    # HEAD request for file metadata
    transport.add_response(
        "HEAD",
        "/files/my-artifact.tar.gz",
        200,
        headers={
            "content-length": str(len(tarball_bytes)),
            "etag": tarball_sha,
            "content-type": "application/gzip",
        },
    )

    # GET request for file download
    transport.add_response(
        "GET",
        "/files/my-artifact.tar.gz",
        200,
        body=tarball_bytes,
        headers={
            "content-length": str(len(tarball_bytes)),
            "etag": tarball_sha,
            "content-type": "application/gzip",
        },
    )

    client = _make_client(transport)
    store = ArtifactStore(client)

    dest_dir = tmp_path / "downloaded"
    out_root = store.download_artifact(
        "my-artifact.tar.gz",
        dest_dir=dest_dir,
        request_id="req-1",
        expected_root="my-artifact",
    )

    assert out_root.exists()
    assert (out_root / "data.txt").read_text(encoding="utf-8") == "test content"


def test_artifact_store_download_empty_file_id_fails(tmp_path: Path) -> None:
    """Test that empty file_id raises error - no HTTP needed."""
    transport = FakeHttpTransport()
    client = _make_client(transport)
    store = ArtifactStore(client)

    import pytest

    with pytest.raises(ArtifactStoreError, match="file_id must be non-empty"):
        store.download_artifact(
            "  ",
            dest_dir=tmp_path / "dl",
            request_id="req-1",
            expected_root="model",
        )


def test_artifact_store_download_zero_size_fails(tmp_path: Path) -> None:
    """Test that zero size from server raises error before download."""
    transport = FakeHttpTransport()

    # HEAD returns zero size - we should catch this before attempting download
    transport.add_response(
        "HEAD",
        "/files/model.tar.gz",
        200,
        headers={
            "content-length": "0",  # Zero size
            "etag": "abc123",
            "content-type": "application/gzip",
        },
    )

    client = _make_client(transport)
    store = ArtifactStore(client)

    import pytest

    with pytest.raises(ArtifactStoreError, match="downloaded file has invalid size"):
        store.download_artifact(
            "model.tar.gz",
            dest_dir=tmp_path / "dl",
            request_id="req-1",
            expected_root="model",
        )


def test_artifact_store_download_extraction_fails(tmp_path: Path) -> None:
    """Test that tarball extraction failure raises error."""
    # Create tarball with wrong root name
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "x.txt").write_text("x", encoding="utf-8")

    tarball_path = tmp_path / "test.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tf:
        tf.add(src_dir, arcname="wrong-root")  # Wrong root name

    tarball_bytes = tarball_path.read_bytes()
    tarball_sha = _compute_sha256(tarball_bytes)

    transport = FakeHttpTransport()

    transport.add_response(
        "HEAD",
        "/files/model.tar.gz",
        200,
        headers={
            "content-length": str(len(tarball_bytes)),
            "etag": tarball_sha,
            "content-type": "application/gzip",
        },
    )

    transport.add_response(
        "GET",
        "/files/model.tar.gz",
        200,
        body=tarball_bytes,
        headers={
            "content-length": str(len(tarball_bytes)),
            "etag": tarball_sha,
            "content-type": "application/gzip",
        },
    )

    client = _make_client(transport)
    store = ArtifactStore(client)

    import pytest

    with pytest.raises(ArtifactStoreError, match="tarball extraction failed"):
        store.download_artifact(
            "model.tar.gz",
            dest_dir=tmp_path / "dl",
            request_id="req-1",
            expected_root="expected-root",  # Doesn't match "wrong-root"
        )


def test_artifact_store_upload_tarball_creation_fails(tmp_path: Path) -> None:
    """Test that tarball creation failure raises ArtifactStoreError."""
    src_dir = tmp_path / "model"
    src_dir.mkdir()
    (src_dir / "data.bin").write_bytes(b"\x00")

    # Inject a fake create_tarball that raises TarballError
    class FailingTarballCreator:
        """Fake tarball creator that always fails."""

        def __call__(self, src_dir: Path, dest_file: Path, *, root_name: str) -> Path:
            raise TarballError("simulated failure")

    failing_creator: _CreateTarballProtocol = FailingTarballCreator()
    _test_hooks.create_tarball = failing_creator

    transport = FakeHttpTransport()
    client = _make_client(transport)
    store = ArtifactStore(client)

    import pytest

    with pytest.raises(ArtifactStoreError, match="tarball creation failed"):
        store.upload_artifact(src_dir, artifact_name="model", request_id="req-1")
