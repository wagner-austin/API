from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.data_bank_client import (
    AuthorizationError,
    HeadInfo,
    NotFoundError,
)

from model_trainer.core.services.data.corpus_fetcher import CorpusFetcher


def test_fetcher_head_404_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = CorpusFetcher("http://db", "k", tmp_path)

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C:
        def __init__(
            self: _C, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            raise NotFoundError("not found")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C)
    with pytest.raises(NotFoundError):
        _ = f.fetch("deadbeef")


def test_fetcher_get_401_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = CorpusFetcher("http://db", "k", tmp_path)

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C2:
        def __init__(
            self: _C2, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C2, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return HeadInfo(size=10, etag="abcd", content_type="text/plain")

        def download_to_path(
            self: _C2,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            raise AuthorizationError("unauthorized")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C2)
    with pytest.raises(AuthorizationError):
        _ = f.fetch("deadbeef")


def test_fetcher_size_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = CorpusFetcher("http://db", "k", tmp_path)
    payload = b"abcde"

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C3:
        def __init__(
            self: _C3, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C3, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return HeadInfo(size=10, etag="abcd", content_type="text/plain")

        def download_to_path(
            self: _C3,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            with dest.open("wb") as out:
                out.write(payload)
            return HeadInfo(size=10, etag="abcd", content_type="text/plain")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C3)
    with pytest.raises(RuntimeError):
        _ = f.fetch("deadbeef")


def test_fetcher_resume_size_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = CorpusFetcher("http://db", "k", tmp_path)
    payload = b"abc"

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C4:
        def __init__(
            self: _C4, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C4, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            # Expect total 6 bytes, but we'll only provide 3 including resume
            return HeadInfo(size=6, etag="abcd", content_type="text/plain")

        def download_to_path(
            self: _C4,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            # Simulate Range resume but return empty to force mismatch
            with dest.open("ab" if dest.exists() else "wb"):
                pass
            return HeadInfo(size=6, etag="abcd", content_type="text/plain")

    # Seed partial temp file to trigger Range branch
    fid = "resume"
    cache_path = tmp_path / f"{fid}.txt"
    tmp = cache_path.with_suffix(".tmp")
    tmp.write_bytes(payload)

    monkeypatch.setattr(cf_mod, "DataBankClient", _C4)
    with pytest.raises(RuntimeError):
        _ = f.fetch(fid)


def test_fetcher_head_without_content_length_allows_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    f = CorpusFetcher("http://db", "k", tmp_path)

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C5:
        def __init__(
            self: _C5, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C5, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            # No Content-Length equivalent: return zero size
            return HeadInfo(size=0, etag="", content_type="application/octet-stream")

        def download_to_path(
            self: _C5,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            with dest.open("wb") as out:
                out.write(b"")
            return HeadInfo(size=0, etag="", content_type="application/octet-stream")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C5)
    out = f.fetch("zero")
    assert out.exists() and out.stat().st_size == 0
