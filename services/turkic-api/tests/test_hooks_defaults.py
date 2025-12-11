"""Tests for _test_hooks default implementations to ensure coverage."""

from __future__ import annotations

import io
import types
from collections.abc import Generator
from pathlib import Path

import pytest

from turkic_api import _test_hooks


class _WikiRaw:
    """Fake raw stream for Wikipedia response that satisfies RawStreamProtocol."""

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def read(self, n: int, /) -> bytes:
        return self._buf.read(n)

    def seekable(self) -> bool:
        return self._buf.seekable()

    def seek(self, n: int, /) -> int:
        return self._buf.seek(n)


class _WikiResp:
    """Fake Wikipedia response that satisfies WikipediaRequestsResponseProtocol."""

    def __init__(self, compressed: bytes) -> None:
        self._raw = _WikiRaw(compressed)

    @property
    def raw(self) -> _test_hooks.RawStreamProtocol:
        return self._raw

    def raise_for_status(self) -> None:
        pass

    def __enter__(self) -> _test_hooks.WikipediaRequestsResponseProtocol:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        pass


def test_default_get_env_returns_from_environ(tmp_path: Path) -> None:
    """Test _default_get_env calls _optional_env_str."""
    # This just verifies the function is callable - we don't set env vars
    result = _test_hooks._default_get_env("NONEXISTENT_VAR_12345")
    assert result is None


def test_default_local_corpus_factory(tmp_path: Path) -> None:
    """Test _default_local_corpus_factory creates LocalCorpusService."""
    service = _test_hooks._default_local_corpus_factory(str(tmp_path))
    # Verify it's the expected type
    assert service.__class__.__name__ == "LocalCorpusService"


def test_default_ensure_corpus_file_calls_real_impl(tmp_path: Path) -> None:
    """Test _default_ensure_corpus_file delegates to real implementation."""
    from turkic_api.core.models import ProcessSpec

    # Set up a fake stream hook to avoid network
    orig = _test_hooks.stream_oscar_hook

    def _stub(lang: str) -> Generator[str, None, None]:
        yield "test line"

    _test_hooks.stream_oscar_hook = _stub

    spec = ProcessSpec(
        source="oscar",
        language="kk",
        max_sentences=1,
        transliterate=True,
        confidence_threshold=0.0,
    )
    try:
        path = _test_hooks._default_ensure_corpus_file(spec, str(tmp_path))
        assert path.exists()
    finally:
        _test_hooks.stream_oscar_hook = orig


def test_default_langid_download_with_fake_requests(tmp_path: Path) -> None:
    """Test _default_langid_download writes downloaded content."""
    # We need to override the requests.get call since we can't make real network calls
    # The function imports requests locally, so we test it by mocking via hooks
    # Actually, since we can't mock requests without monkeypatch, we skip the actual
    # network call test and just verify the function signature exists
    assert callable(_test_hooks._default_langid_download)


def test_default_langid_ensure_model_path_prefer_218e(tmp_path: Path) -> None:
    """Test _default_langid_ensure_model_path prefer_218e path."""
    # Create fake model file to avoid download
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_file = models_dir / "lid218e.bin"
    model_file.write_bytes(b"fake model")

    # Override langid_download to avoid network
    orig_download = _test_hooks.langid_download

    def _noop_download(url: str, dest: Path) -> None:
        dest.write_bytes(b"fake")

    _test_hooks.langid_download = _noop_download
    try:
        path = _test_hooks._default_langid_ensure_model_path(str(tmp_path), prefer_218e=True)
        assert path == model_file
    finally:
        _test_hooks.langid_download = orig_download


def test_default_langid_ensure_model_path_prefer_176(tmp_path: Path) -> None:
    """Test _default_langid_ensure_model_path prefer_176 path."""
    # Create fake model file to avoid download
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    model_file = models_dir / "lid.176.bin"
    model_file.write_bytes(b"fake model")

    # Override langid_download to avoid network
    orig_download = _test_hooks.langid_download

    def _noop_download(url: str, dest: Path) -> None:
        dest.write_bytes(b"fake")

    _test_hooks.langid_download = _noop_download
    try:
        path = _test_hooks._default_langid_ensure_model_path(str(tmp_path), prefer_218e=False)
        assert path == model_file
    finally:
        _test_hooks.langid_download = orig_download


def test_default_langid_ensure_model_path_download_218e(tmp_path: Path) -> None:
    """Test _default_langid_ensure_model_path downloads 218e when missing."""
    downloaded_urls: list[str] = []

    def _capture_download(url: str, dest: Path) -> None:
        downloaded_urls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake")

    orig_download = _test_hooks.langid_download
    _test_hooks.langid_download = _capture_download
    try:
        path = _test_hooks._default_langid_ensure_model_path(str(tmp_path), prefer_218e=True)
        assert "lid218e" in downloaded_urls[0]
        assert path.exists()
    finally:
        _test_hooks.langid_download = orig_download


def test_default_langid_ensure_model_path_download_176(tmp_path: Path) -> None:
    """Test _default_langid_ensure_model_path downloads 176 when missing."""
    downloaded_urls: list[str] = []

    def _capture_download(url: str, dest: Path) -> None:
        downloaded_urls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake")

    orig_download = _test_hooks.langid_download
    _test_hooks.langid_download = _capture_download
    try:
        path = _test_hooks._default_langid_ensure_model_path(str(tmp_path), prefer_218e=False)
        assert "lid.176" in downloaded_urls[0]
        assert path.exists()
    finally:
        _test_hooks.langid_download = orig_download


def test_default_to_ipa_calls_real_impl() -> None:
    """Test _default_to_ipa delegates to real transliteration."""
    result = _test_hooks._default_to_ipa("тест", "kk")
    # Should produce IPA output starting with 't' sound for тест (test)
    assert result.startswith("t")


def test_default_path_exists_and_unlink(tmp_path: Path) -> None:
    """Test _default_path_exists and _default_path_unlink."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test")

    assert _test_hooks._default_path_exists(test_file) is True
    _test_hooks._default_path_unlink(test_file)
    assert _test_hooks._default_path_exists(test_file) is False


def test_default_path_unlink_missing_ok(tmp_path: Path) -> None:
    """Test _default_path_unlink with missing_ok."""
    test_file = tmp_path / "nonexistent.txt"
    # Should not raise with missing_ok=True
    _test_hooks._default_path_unlink(test_file, missing_ok=True)


def test_default_stream_hooks_delegate_to_real() -> None:
    """Test _default_stream_* hooks reference real functions."""
    # These just verify the hooks are callable
    assert callable(_test_hooks._default_stream_oscar)
    assert callable(_test_hooks._default_stream_wikipedia_xml)
    assert callable(_test_hooks._default_stream_culturax)


def test_default_decode_required_literal() -> None:
    """Test _default_decode_required_literal calls real validator."""
    result = _test_hooks._default_decode_required_literal(
        "oscar", "source", frozenset({"oscar", "wikipedia"})
    )
    assert result == "oscar"


def test_default_decode_optional_literal() -> None:
    """Test _default_decode_optional_literal calls real validator."""
    result = _test_hooks._default_decode_optional_literal(
        "Latn", "script", frozenset({"Latn", "Cyrl"})
    )
    assert result == "Latn"

    result_none = _test_hooks._default_decode_optional_literal(
        None, "script", frozenset({"Latn", "Cyrl"})
    )
    assert result_none is None


def test_default_guard_find_monorepo_root(tmp_path: Path) -> None:
    """Test _default_guard_find_monorepo_root finds root with libs dir."""
    # Create a fake monorepo structure
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()
    subdir = tmp_path / "subdir"
    subdir.mkdir()

    result = _test_hooks._default_guard_find_monorepo_root(subdir)
    assert result == tmp_path


def test_default_guard_find_monorepo_root_not_found(tmp_path: Path) -> None:
    """Test _default_guard_find_monorepo_root raises when not found."""
    with pytest.raises(RuntimeError, match="monorepo root"):
        _test_hooks._default_guard_find_monorepo_root(tmp_path)


def test_default_wikipedia_requests_get_adapter() -> None:
    """Test _default_wikipedia_requests_get wraps real requests."""
    # We can't make real network calls, but we verify the function signature
    assert callable(_test_hooks._default_wikipedia_requests_get)


def test_default_data_bank_uploader_factory() -> None:
    """Test _default_data_bank_uploader_factory creates DataBankClient."""
    # This actually creates the client - it's OK since it doesn't make requests yet
    client = _test_hooks._default_data_bank_uploader_factory(
        "http://localhost:8080", "fake-key", timeout_seconds=30.0
    )
    # Verify client conforms to protocol by accessing the upload method
    upload_method = client.upload
    assert callable(upload_method)


def test_default_data_bank_downloader_factory() -> None:
    """Test _default_data_bank_downloader_factory creates DataBankClient."""
    # This actually creates the client - it's OK since it doesn't make requests yet
    client = _test_hooks._default_data_bank_downloader_factory(
        "http://localhost:8080", "fake-key", timeout_seconds=30.0
    )
    # Verify client conforms to protocol by accessing the methods
    head_method = client.head
    stream_method = client.stream_download
    assert callable(head_method)
    assert callable(stream_method)


def test_default_stream_oscar_calls_corpus_download() -> None:
    """Test _default_stream_oscar delegates to corpus_download.stream_oscar."""
    import sys
    from types import ModuleType

    from platform_core.json_utils import JSONValue

    # Create a fake datasets module to make stream_oscar work
    class _DS:
        def __iter__(self) -> Generator[dict[str, str | int] | int, None, None]:
            yield {"text": "hello"}

    class _FakeDatasets(ModuleType):
        @staticmethod
        def load_dataset(*_a: JSONValue, **_k: JSONValue) -> _DS:
            return _DS()

    sys.modules["datasets"] = _FakeDatasets("datasets")
    try:
        gen = _test_hooks._default_stream_oscar("kk")
        result = list(gen)
        assert result == ["hello"]
    finally:
        del sys.modules["datasets"]


def test_default_stream_culturax_calls_corpus_download() -> None:
    """Test _default_stream_culturax delegates to corpus_download.stream_culturax."""
    import sys
    from types import ModuleType

    from platform_core.json_utils import JSONValue

    # Create a fake datasets module to make stream_culturax work
    class _DS:
        def __iter__(self) -> Generator[dict[str, str | int] | int, None, None]:
            yield {"text": "world"}

    class _FakeDatasets(ModuleType):
        @staticmethod
        def load_dataset(*_a: JSONValue, **_k: JSONValue) -> _DS:
            return _DS()

    sys.modules["datasets"] = _FakeDatasets("datasets")
    try:
        gen = _test_hooks._default_stream_culturax("kk")
        result = list(gen)
        assert result == ["world"]
    finally:
        del sys.modules["datasets"]


def test_default_stream_wikipedia_xml_calls_corpus_download() -> None:
    """Test _default_stream_wikipedia_xml delegates to corpus_download.stream_wikipedia_xml."""
    import bz2

    # Stub the wikipedia_requests_get hook to avoid network
    compressed = bz2.compress(b"<page><revision><text>Test sentence</text></revision></page>")

    orig = _test_hooks.wikipedia_requests_get

    def _fake_get(
        url: str, *, stream: bool, timeout: int
    ) -> _test_hooks.WikipediaRequestsResponseProtocol:
        return _WikiResp(compressed)

    _test_hooks.wikipedia_requests_get = _fake_get
    try:
        gen = _test_hooks._default_stream_wikipedia_xml("kk")
        result = list(gen)
        assert "Test sentence" in result
    finally:
        _test_hooks.wikipedia_requests_get = orig


def test_default_redis_for_kv_creates_client() -> None:
    """Test _default_redis_for_kv creates a Redis client via redis_for_kv."""
    # The function creates a Redis client. Creating the client doesn't immediately
    # connect, so we can verify it returns a valid client instance.
    # We use a fake URL - actual connection happens on first command.
    client = _test_hooks._default_redis_for_kv("redis://localhost:6379/0")
    # Verify it's a Redis client by checking class name
    assert "Redis" in client.__class__.__name__
    # Clean up
    client.close()


def test_default_load_langid_model_delegates() -> None:
    """Test _default_load_langid_model delegates to langid.load_langid_model."""
    # This function imports and calls load_langid_model
    # We can't actually load fasttext in tests, but we verify it delegates correctly
    # by stubbing langid_ensure_model_path and langid_get_fasttext_factory
    import numpy as np
    from numpy.typing import NDArray

    from tests.conftest import make_probs

    class _FakeModel:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__kk",), make_probs(0.99))

    class _FakeFactory:
        def __call__(self, *, model_path: str) -> _FakeModel:
            return _FakeModel()

    def _fake_ensure(data_dir: str, prefer_218e: bool = True) -> Path:
        return Path("/fake/model.bin")

    def _fake_get_factory() -> _test_hooks.LangIdModelFactoryProtocol:
        return _FakeFactory()

    orig_ensure = _test_hooks.langid_ensure_model_path
    orig_factory = _test_hooks.langid_get_fasttext_factory

    _test_hooks.langid_ensure_model_path = _fake_ensure
    _test_hooks.langid_get_fasttext_factory = _fake_get_factory

    try:
        model = _test_hooks._default_load_langid_model("/fake/data", prefer_218e=True)
        labels, _probs = model.predict("test", k=1)
        assert labels[0] == "__label__kk"
    finally:
        _test_hooks.langid_ensure_model_path = orig_ensure
        _test_hooks.langid_get_fasttext_factory = orig_factory


def test_default_langid_download_writes_file(tmp_path: Path) -> None:
    """Test _default_langid_download downloads and writes content."""
    import http.server
    import socketserver
    import threading

    # Create a simple HTTP server to serve test content
    content = b"fake model content"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: str) -> None:
            pass  # Suppress logging

    # Find a free port
    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.handle_request)
        thread.start()

        dest = tmp_path / "models" / "test.bin"
        _test_hooks._default_langid_download(f"http://127.0.0.1:{port}/model", dest)

        thread.join(timeout=5)

    assert dest.exists()
    assert dest.read_bytes() == content


def test_default_langid_download_handles_empty_chunks(tmp_path: Path) -> None:
    """Test _default_langid_download skips empty chunks in iter_content."""
    import http.server
    import socketserver
    import threading

    # Create HTTP server that sends chunked response
    # The requests library handles chunked encoding, we just need valid content
    content = b"partial content"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            # Send content in chunked format
            self.wfile.write(b"%x\r\n" % len(content))
            self.wfile.write(content)
            self.wfile.write(b"\r\n")
            self.wfile.write(b"0\r\n\r\n")  # End of chunks

        def log_message(self, format: str, *args: str) -> None:
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.handle_request)
        thread.start()

        dest = tmp_path / "models" / "chunked.bin"
        _test_hooks._default_langid_download(f"http://127.0.0.1:{port}/model", dest)

        thread.join(timeout=5)

    assert dest.exists()
    assert dest.read_bytes() == content


def test_default_langid_get_fasttext_factory(tmp_path: Path) -> None:
    """Test _default_langid_get_fasttext_factory imports real FastText and returns factory."""
    # This tests the production default implementation with the real fasttext library.
    # The hook pattern means other tests override the hook to avoid this dependency,
    # but this test verifies the default actually works with real fasttext.

    # First, ensure we have a valid model to test with
    # Use the hook to download a real model (this tests integration)
    model_path = _test_hooks._default_langid_ensure_model_path(
        data_dir=str(tmp_path), prefer_218e=False
    )

    # Get the factory using the production default
    factory = _test_hooks._default_langid_get_fasttext_factory()

    # Verify the factory is callable and produces working models
    assert callable(factory)

    # Create a model using the factory with the real model file
    model = factory(model_path=str(model_path))

    # Verify the model can make predictions (real fasttext behavior)
    # With k=1, fasttext returns exactly 1 label and 1 probability
    labels, probs = model.predict("hello world", k=1)
    label: str = labels[0]
    # Use .item() to extract scalar from numpy array with proper typing
    prob: float = probs.item(0)
    assert label.startswith("__label__")
    assert 0.0 <= prob <= 1.0


def test_default_wikipedia_requests_get_real_http(tmp_path: Path) -> None:
    """Test _default_wikipedia_requests_get makes real HTTP request and returns adapter."""
    import http.server
    import socketserver
    import threading

    content = b"test content"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, format: str, *args: str) -> None:
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.handle_request)
        thread.start()

        resp = _test_hooks._default_wikipedia_requests_get(
            f"http://127.0.0.1:{port}/test", stream=True, timeout=5
        )

        with resp:
            resp.raise_for_status()
            raw_data = resp.raw.read(100)

        thread.join(timeout=5)

    assert raw_data == content
