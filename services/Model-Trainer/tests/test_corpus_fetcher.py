from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.data_bank_client import (
    HeadInfo,
)

from model_trainer.core.services.data.corpus_fetcher import CorpusFetcher


@pytest.mark.parametrize("resume", [False, True])
def test_fetcher_download_and_cache_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resume: bool
) -> None:
    payload = b"hello world" * 100
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)

    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C:
        def __init__(
            self: _C, base_url: str, api_key: str, *, timeout_seconds: float = 60.0
        ) -> None:
            pass

        def head(self: _C, file_id: str, *, request_id: str | None = None) -> HeadInfo:
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

        def download_to_path(
            self: _C,
            file_id: str,
            dest: Path,
            *,
            resume: bool = True,
            request_id: str | None = None,
            verify_etag: bool = True,
            chunk_size: int = 1024 * 1024,
        ) -> HeadInfo:
            start = dest.stat().st_size if dest.exists() else 0
            if start > 0:
                with dest.open("ab") as f_bin:
                    f_bin.write(payload[start:])
            else:
                with dest.open("wb") as f_bin:
                    f_bin.write(payload)
            return HeadInfo(size=len(payload), etag="abcd", content_type="text/plain")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C)

    fid = "deadbeef"
    cache_path = cache / f"{fid}.txt"
    if resume:
        tmp = cache_path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(payload[:50])

    out = f.fetch(fid)
    assert out == cache_path
    assert out.read_bytes() == payload

    out2 = f.fetch(fid)
    assert out2 == out


def test_fetcher_uses_cache_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    f = CorpusFetcher("http://test", "k", cache)
    fid = "cafebabe"
    cache_path = cache / f"{fid}.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("cached", encoding="utf-8")

    # If DataBankClient is created, fail the test
    import model_trainer.core.services.data.corpus_fetcher as cf_mod

    class _C2:
        def __init__(self: _C2) -> None:
            raise AssertionError("DataBankClient should not be constructed on cache hit")

    monkeypatch.setattr(cf_mod, "DataBankClient", _C2)

    out = f.fetch(fid)
    assert out == cache_path
    assert out.read_text(encoding="utf-8") == "cached"
