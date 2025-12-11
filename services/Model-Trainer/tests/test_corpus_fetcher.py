from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import httpx
import pytest

from model_trainer.core import _test_hooks
from model_trainer.core.services.data.corpus_fetcher import CorpusFetcher


class _MemStore:
    """In-memory file store for testing."""

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}

    def put(self, fid: str, data: bytes) -> None:
        self._files[fid] = data

    def get(self, fid: str) -> bytes:
        return self._files[fid]

    def exists(self, fid: str) -> bool:
        return fid in self._files


class _MockServer:
    """Mock HTTP server for DataBankClient testing."""

    def __init__(self, store: _MemStore, expect_key: str) -> None:
        self._store = store
        self._key = expect_key

    def handle(self, request: httpx.Request) -> httpx.Response:
        hdrs: dict[str, str] = {k.lower(): v for k, v in request.headers.items()}
        if hdrs.get("x-api-key") != self._key:
            return httpx.Response(401, text="unauthorized")

        path = request.url.path
        if request.method == "HEAD" and path.startswith("/files/"):
            fid = path.split("/")[-1]
            if not self._store.exists(fid):
                return httpx.Response(404, text="not found")
            data = self._store.get(fid)
            etag = sha256(data).hexdigest()
            return httpx.Response(
                200,
                headers={
                    "Content-Length": str(len(data)),
                    "ETag": etag,
                    "Content-Type": "text/plain",
                },
            )
        if request.method == "GET" and path.startswith("/files/"):
            fid = path.split("/")[-1]
            if not self._store.exists(fid):
                return httpx.Response(404, text="not found")
            data = self._store.get(fid)
            rng_header = hdrs.get("range")
            if rng_header and rng_header.startswith("bytes="):
                start = int(rng_header.split("=")[1].split("-")[0])
                part = data[start:]
                return httpx.Response(
                    206,
                    content=part,
                    headers={
                        "Content-Length": str(len(part)),
                        "Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}",
                        "ETag": sha256(data).hexdigest(),
                    },
                )
            return httpx.Response(
                200,
                content=data,
                headers={
                    "Content-Length": str(len(data)),
                    "ETag": sha256(data).hexdigest(),
                },
            )
        return httpx.Response(500, text="unhandled")


def _make_mock_client(store: _MemStore, api_key: str = "k") -> httpx.Client:
    """Create httpx.Client with mock transport."""
    server = _MockServer(store, api_key)
    return httpx.Client(transport=httpx.MockTransport(server.handle))


@pytest.mark.parametrize("resume", [False, True])
def test_fetcher_download_and_cache_resume(tmp_path: Path, resume: bool) -> None:
    payload = b"hello world" * 100
    cache = tmp_path / "cache"
    store = _MemStore()
    fid = "deadbeef"
    store.put(fid, payload)

    # Set up httpx client factory with mock transport
    _test_hooks.httpx_client_factory = lambda *, timeout_seconds=30.0: _make_mock_client(store, "k")

    f = CorpusFetcher("http://test", "k", cache)
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


def test_fetcher_uses_cache_without_network(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    fid = "cafebabe"
    cache_path = cache / f"{fid}.txt"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("cached", encoding="utf-8")

    # If httpx client is created, fail the test (cache should bypass network)
    def _fail_factory(*, timeout_seconds: float = 30.0) -> httpx.Client:
        raise AssertionError("httpx.Client should not be created on cache hit")

    _test_hooks.httpx_client_factory = _fail_factory

    f = CorpusFetcher("http://test", "k", cache)
    out = f.fetch(fid)
    assert out == cache_path
    assert out.read_text(encoding="utf-8") == "cached"
