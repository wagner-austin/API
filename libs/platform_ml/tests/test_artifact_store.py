from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Protocol

import pytest
from platform_core.data_bank_client import HeadInfo
from platform_core.data_bank_protocol import FileUploadResponse

from platform_ml.artifact_store import ArtifactStore, ArtifactStoreError


class _ClientProto(Protocol):
    def upload(
        self,
        file_id: str,
        stream: io.BufferedReader,
        *,
        content_type: str = "application/octet-stream",
        request_id: str | None = None,
    ) -> FileUploadResponse: ...

    def download_to_path(
        self,
        file_id: str,
        dest: Path,
        *,
        resume: bool = True,
        request_id: str | None = None,
        verify_etag: bool = True,
        chunk_size: int = 1024 * 1024,
    ) -> HeadInfo: ...


class _FakeClient:
    def __init__(self, backing_dir: Path) -> None:
        self._dir = backing_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def upload(
        self,
        file_id: str,
        stream: io.BufferedReader,
        *,
        content_type: str = "application/octet-stream",
        request_id: str | None = None,
    ) -> FileUploadResponse:
        fid = file_id
        dest = self._dir / fid
        data = stream.read()
        dest.write_bytes(data)
        import hashlib

        sha = hashlib.sha256(data).hexdigest()
        return {
            "file_id": fid,
            "size": len(data),
            "sha256": sha,
            "content_type": content_type,
            "created_at": None,
        }

    def download_to_path(
        self,
        file_id: str,
        dest: Path,
        *,
        resume: bool = True,
        request_id: str | None = None,
        verify_etag: bool = True,
        chunk_size: int = 1024 * 1024,
    ) -> HeadInfo:
        src = self._dir / file_id
        data = src.read_bytes()
        dest.write_bytes(data)
        import hashlib

        sha = hashlib.sha256(data).hexdigest()
        return {"size": len(data), "etag": sha, "content_type": "application/gzip"}


def test_artifact_store_upload_and_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Prepare an artifact directory
    src_dir = tmp_path / "model_dir"
    (src_dir / "sub").mkdir(parents=True)
    (src_dir / "sub" / "a.txt").write_text("hi", encoding="utf-8")
    (src_dir / "b.bin").write_bytes(b"\x00\x02")

    # Patch DataBankClient used inside ArtifactStore to our fake client
    import platform_ml.artifact_store as mod

    fake = _FakeClient(tmp_path / "remote")

    def _mk_client(base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> _ClientProto:
        return fake

    monkeypatch.setattr(mod, "DataBankClient", _mk_client, raising=True)

    store = ArtifactStore(base_url="http://x", api_key="k")
    # Invalid artifact name (empty)
    with pytest.raises(ArtifactStoreError):
        store.upload_artifact(src_dir, artifact_name=" \t ", request_id="r")
    resp = store.upload_artifact(src_dir, artifact_name="art-x", request_id="r1")
    assert resp["file_id"].endswith(".tar.gz")
    assert resp["size"] > 0
    # The remote should contain a tarball we can read
    remote_file = (tmp_path / "remote") / resp["file_id"]
    with tarfile.open(remote_file.as_posix(), mode="r:gz") as tf:
        names = tf.getnames()
        assert any(n.endswith("a.txt") for n in names)

    # Download and extract
    out_root = store.download_artifact(
        resp["file_id"], dest_dir=tmp_path / "dl", request_id="r2", expected_root="art-x"
    )
    assert (out_root / "sub" / "a.txt").read_text(encoding="utf-8") == "hi"

    # Invalid invocation
    with pytest.raises(ArtifactStoreError):
        store.upload_artifact(tmp_path / "missing", artifact_name="x", request_id="r")
    with pytest.raises(ArtifactStoreError):
        store.download_artifact("", dest_dir=tmp_path, request_id="r", expected_root="x")

    # Force head size invalid by monkeypatching
    def _download_to_path_zero(
        file_id: str,
        dest: Path,
        *,
        resume: bool = True,
        request_id: str | None = None,
        verify_etag: bool = True,
        chunk_size: int = 1024 * 1024,
    ) -> HeadInfo:
        remote_file = (tmp_path / "remote") / file_id
        dest.write_bytes(remote_file.read_bytes())
        return {"size": 0, "etag": "x", "content_type": "application/gzip"}

    monkeypatch.setattr(fake, "download_to_path", _download_to_path_zero, raising=True)
    with pytest.raises(ArtifactStoreError):
        _ = store.download_artifact(
            resp["file_id"], dest_dir=tmp_path / "dl2", request_id="r3", expected_root="art-x"
        )


# Coverage for line 74: uploaded size mismatch
def test_artifact_store_upload_size_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "model"
    src_dir.mkdir()
    (src_dir / "data.bin").write_bytes(b"\x00" * 100)

    import platform_ml.artifact_store as mod

    class _SizeMismatchClient:
        def upload(
            self,
            file_id: str,
            stream: io.BufferedReader,
            *,
            content_type: str = "application/octet-stream",
            request_id: str | None = None,
        ) -> FileUploadResponse:
            data = stream.read()
            # Return wrong size to trigger mismatch
            return {
                "file_id": file_id,
                "size": len(data) + 999,  # deliberately wrong
                "sha256": "fake",
                "content_type": content_type,
                "created_at": None,
            }

    def _mk_bad_client(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> _SizeMismatchClient:
        return _SizeMismatchClient()

    monkeypatch.setattr(mod, "DataBankClient", _mk_bad_client, raising=True)
    store = ArtifactStore(base_url="http://x", api_key="k")
    with pytest.raises(ArtifactStoreError, match="uploaded size mismatch"):
        store.upload_artifact(src_dir, artifact_name="art", request_id="r")


# Coverage for lines 59-60: TarballError during upload (create_tarball fails)
def test_artifact_store_upload_tarball_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "model"
    src_dir.mkdir()
    (src_dir / "data.bin").write_bytes(b"\x00" * 10)

    import platform_ml.artifact_store as mod
    from platform_ml.tarball import TarballError

    def _create_tarball_fail(src_dir: Path, dest_file: Path, *, root_name: str) -> Path:
        raise TarballError("simulated tarball creation failure")

    # Patch DataBankClient (unused but needed for init)
    fake = _FakeClient(tmp_path / "remote")

    def _mk_client(base_url: str, api_key: str, *, timeout_seconds: float = 600.0) -> _ClientProto:
        return fake

    monkeypatch.setattr(mod, "DataBankClient", _mk_client, raising=True)
    monkeypatch.setattr(mod, "create_tarball", _create_tarball_fail, raising=True)

    store = ArtifactStore(base_url="http://x", api_key="k")
    with pytest.raises(ArtifactStoreError, match="tarball creation failed"):
        store.upload_artifact(src_dir, artifact_name="art", request_id="r")


# Coverage for lines 102-103: TarballError during download (extract_tarball fails)
def test_artifact_store_download_tarball_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import platform_ml.artifact_store as mod
    from platform_ml.tarball import TarballError

    # Create a fake remote file
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir()
    fake_tarball = remote_dir / "model.tar.gz"
    fake_tarball.write_bytes(b"not-a-real-tarball")

    class _DownloadClient:
        def download_to_path(
            self,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            dest.write_bytes(fake_tarball.read_bytes())
            return {"size": 100, "etag": "x", "content_type": "application/gzip"}

    def _extract_tarball_fail(src_file: Path, dest_dir: Path, *, expected_root: str) -> Path:
        raise TarballError("simulated extraction failure")

    def _mk_client(
        base_url: str, api_key: str, *, timeout_seconds: float = 600.0
    ) -> _DownloadClient:
        return _DownloadClient()

    monkeypatch.setattr(mod, "DataBankClient", _mk_client, raising=True)
    monkeypatch.setattr(mod, "extract_tarball", _extract_tarball_fail, raising=True)

    store = ArtifactStore(base_url="http://x", api_key="k")
    with pytest.raises(ArtifactStoreError, match="tarball extraction failed"):
        store.download_artifact(
            "model.tar.gz",
            dest_dir=tmp_path / "dl",
            request_id="r",
            expected_root="model",
        )
