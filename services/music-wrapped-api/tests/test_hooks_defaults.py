"""Tests for _test_hooks default implementations.

These tests exercise the production default implementations to ensure they work
correctly and to achieve full code coverage.
"""

from __future__ import annotations

import types
import urllib.request
from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue
from platform_workers.rq_harness import _RedisBytesClient

from music_wrapped_api import _test_hooks


def test_default_get_env() -> None:
    """Test _default_get_env reads from environment via hook."""
    # The function delegates to platform_core's _optional_env_str which uses
    # the config hook. We verify that the delegation works by setting a
    # specific value in the hook and checking the result.
    from platform_core.config import _test_hooks as config_test_hooks
    from platform_core.testing import FakeEnv

    # Create a FakeEnv with known values
    env = FakeEnv({"TEST_VAR": "test_value"})
    original = config_test_hooks.get_env
    config_test_hooks.get_env = env

    # Test retrieval of set value
    result = _test_hooks._default_get_env("TEST_VAR")
    assert result == "test_value"

    # Test retrieval of missing value returns None
    missing = _test_hooks._default_get_env("MISSING_VAR")
    assert missing is None

    # Restore original hook
    config_test_hooks.get_env = original


def test_default_require_env_missing() -> None:
    """Test _default_require_env raises for missing env var."""
    with pytest.raises(RuntimeError):
        _test_hooks._default_require_env("DEFINITELY_MISSING_VAR_12345")


def test_default_rand_state() -> None:
    """Test _default_rand_state generates random state."""
    state1 = _test_hooks._default_rand_state()
    state2 = _test_hooks._default_rand_state()
    # States should be different (random)
    assert state1 != state2
    # token_urlsafe(16) produces 22 chars base64
    assert state1.isascii()


def test_default_make_request() -> None:
    """Test _default_make_request creates a valid Request."""
    req = _test_hooks._default_make_request("https://example.com", b"data")
    # Verify it's a urllib Request by calling add_header (should not raise)
    req.add_header("X-Test", "value")


def test_default_urlopen_post_type_error() -> None:
    """Test _default_urlopen_post raises TypeError for non-Request."""

    class _FakeRequest:
        def add_header(self, name: str, value: str) -> None:
            pass

    with pytest.raises(TypeError, match=r"req must be a urllib\.request\.Request"):
        _test_hooks._default_urlopen_post(_FakeRequest(), 5.0)


def test_default_guard_find_monorepo_root() -> None:
    """Test _default_guard_find_monorepo_root finds monorepo root."""
    # Start from this file's directory, which is inside the monorepo
    start = Path(__file__).resolve().parent
    root = _test_hooks._default_guard_find_monorepo_root(start)
    # Verify it found a directory with 'libs'
    assert (root / "libs").is_dir()


def test_default_guard_find_monorepo_root_failure() -> None:
    """Test _default_guard_find_monorepo_root raises for invalid path."""
    # Use root directory which doesn't have 'libs'
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with pytest.raises(RuntimeError, match="monorepo root"):
            _test_hooks._default_guard_find_monorepo_root(tmp_path)


def test_default_guard_load_orchestrator() -> None:
    """Test _default_guard_load_orchestrator loads the orchestrator."""
    start = Path(__file__).resolve().parent
    root = _test_hooks._default_guard_find_monorepo_root(start)
    run_for_project = _test_hooks._default_guard_load_orchestrator(root)
    # Verify it returns a callable
    assert callable(run_for_project)


class _FakeHttpResponse:
    """Fake HTTP response for testing default implementations."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _FakeHttpResponse:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        pass


def test_default_lfm_get_session_json() -> None:
    """Test _default_lfm_get_session_json builds correct request."""
    requests_made: list[str] = []
    responses: list[_FakeHttpResponse] = [
        _FakeHttpResponse(b'{"session": {"key": "sk", "name": "user"}}'),
    ]

    def _fake_urlopen(url: str, timeout: float) -> _FakeHttpResponse:
        requests_made.append(url)
        return responses.pop(0)

    # Temporarily override urlopen_get
    original = _test_hooks.urlopen_get
    _test_hooks.urlopen_get = _fake_urlopen

    result = _test_hooks._default_lfm_get_session_json("key1", "secret1", "tok1")

    _test_hooks.urlopen_get = original

    assert result == {"session": {"key": "sk", "name": "user"}}
    assert len(requests_made) == 1
    url = requests_made[0]
    assert "api_key=key1" in url
    assert "method=auth.getSession" in url


def test_default_lfm_get_session_json_invalid() -> None:
    """Test _default_lfm_get_session_json raises for non-dict response."""
    from platform_core.errors import AppError

    responses: list[_FakeHttpResponse] = [
        _FakeHttpResponse(b"[]"),  # Not a dict
    ]

    def _fake_urlopen(url: str, timeout: float) -> _FakeHttpResponse:
        return responses.pop(0)

    original = _test_hooks.urlopen_get
    _test_hooks.urlopen_get = _fake_urlopen

    with pytest.raises(AppError):
        _test_hooks._default_lfm_get_session_json("key", "secret", "token")

    _test_hooks.urlopen_get = original


def test_default_spotify_exchange_code() -> None:
    """Test _default_spotify_exchange_code builds correct request."""
    requests_made: list[urllib.request.Request] = []
    responses: list[_FakeHttpResponse] = [
        _FakeHttpResponse(b'{"access_token": "tok", "refresh_token": "ref"}'),
    ]

    def _fake_make_request(url: str, data: bytes) -> urllib.request.Request:
        return urllib.request.Request(url, data)

    def _fake_urlopen_post(
        req: _test_hooks.HttpRequestProtocol, timeout: float
    ) -> _FakeHttpResponse:
        if isinstance(req, urllib.request.Request):
            requests_made.append(req)
        return responses.pop(0)

    original_make = _test_hooks.make_request
    original_post = _test_hooks.urlopen_post
    _test_hooks.make_request = _fake_make_request
    _test_hooks.urlopen_post = _fake_urlopen_post

    result = _test_hooks._default_spotify_exchange_code(
        "code1", "http://redirect", client_id="cid", client_secret="csec"
    )

    _test_hooks.make_request = original_make
    _test_hooks.urlopen_post = original_post

    assert result == {"access_token": "tok", "refresh_token": "ref"}
    assert len(requests_made) == 1
    assert requests_made[0].full_url == "https://accounts.spotify.com/api/token"


def test_default_spotify_exchange_code_invalid() -> None:
    """Test _default_spotify_exchange_code raises for non-dict response."""
    from platform_core.errors import AppError

    responses: list[_FakeHttpResponse] = [
        _FakeHttpResponse(b'"string_response"'),  # Not a dict
    ]

    def _fake_make_request(url: str, data: bytes) -> urllib.request.Request:
        return urllib.request.Request(url, data)

    def _fake_urlopen_post(
        req: _test_hooks.HttpRequestProtocol, timeout: float
    ) -> _FakeHttpResponse:
        return responses.pop(0)

    original_make = _test_hooks.make_request
    original_post = _test_hooks.urlopen_post
    _test_hooks.make_request = _fake_make_request
    _test_hooks.urlopen_post = _fake_urlopen_post

    with pytest.raises(AppError):
        _test_hooks._default_spotify_exchange_code(
            "code1", "http://redirect", client_id="cid", client_secret="csec"
        )

    _test_hooks.make_request = original_make
    _test_hooks.urlopen_post = original_post


def test_default_redis_for_kv() -> None:
    """Test _default_redis_for_kv creates Redis client."""
    # Creating a Redis client doesn't connect immediately
    # It connects on first command, so we can verify the client is created
    client = _test_hooks._default_redis_for_kv("redis://localhost:6379/0")
    # Verify it has Redis-like interface
    assert "Redis" in client.__class__.__name__
    client.close()


def test_default_rq_conn() -> None:
    """Test _default_rq_conn creates RQ Redis connection."""
    # Creating connection doesn't connect immediately
    conn = _test_hooks._default_rq_conn("redis://localhost:6379/0")
    # Verify it's a Redis bytes client
    assert "Redis" in conn.__class__.__name__
    conn.close()


def test_default_rq_queue() -> None:
    """Test _default_rq_queue creates RQ queue."""
    # Create a connection (won't connect immediately)
    conn = _test_hooks._default_rq_conn("redis://localhost:6379/0")
    queue = _test_hooks._default_rq_queue("test_queue", connection=conn)
    # Verify it has queue-like interface (wrapped as _RQQueueAdapter)
    assert "Queue" in queue.__class__.__name__
    conn.close()


def test_rq_queue_factory_hook_delegates() -> None:
    """Test _rq_queue_factory_hook delegates to _default_rq_queue."""
    from platform_workers.rq_harness import RQClientQueue
    from platform_workers.testing import FakeQueue, FakeRedisBytesClient

    calls: list[tuple[str, _RedisBytesClient]] = []

    def _capture(name: str, *, connection: _RedisBytesClient) -> RQClientQueue:
        calls.append((name, connection))
        return FakeQueue(job_id="captured")

    # Replace the default temporarily
    import music_wrapped_api._test_hooks as hooks_mod

    original = hooks_mod._default_rq_queue
    hooks_mod._default_rq_queue = _capture

    # Call the wrapper hook
    conn = FakeRedisBytesClient()
    _ = hooks_mod._rq_queue_factory_hook("test_queue", conn)

    # Restore
    hooks_mod._default_rq_queue = original

    assert calls == [("test_queue", conn)]


def test_get_job_hook_delegates() -> None:
    """Test _get_job_hook delegates to _default_get_job."""
    from platform_workers.testing import FakeRedisBytesClient

    calls: list[tuple[str, _RedisBytesClient]] = []

    class _FakeJob:
        def get_status(self) -> str:
            return "test"

        @property
        def is_finished(self) -> bool:
            return True

        @property
        def meta(self) -> dict[str, JSONValue]:
            return {}

        @property
        def result(self) -> str | None:
            return None

    def _capture(job_id: str, *, connection: _RedisBytesClient) -> _test_hooks.RQJobProtocol:
        calls.append((job_id, connection))
        return _FakeJob()

    import music_wrapped_api._test_hooks as hooks_mod

    original = hooks_mod._default_get_job
    hooks_mod._default_get_job = _capture

    conn = FakeRedisBytesClient()
    _ = hooks_mod._get_job_hook("job-123", conn)

    hooks_mod._default_get_job = original

    assert calls == [("job-123", conn)]


def test_spotify_exchange_code_hook_delegates() -> None:
    """Test _spotify_exchange_code_hook delegates to _default_spotify_exchange_code."""

    calls: list[tuple[str, str, str, str]] = []

    def _capture(
        code: str, redirect_uri: str, *, client_id: str, client_secret: str
    ) -> dict[str, JSONValue]:
        calls.append((code, redirect_uri, client_id, client_secret))
        return {"captured": True}

    import music_wrapped_api._test_hooks as hooks_mod

    original = hooks_mod._default_spotify_exchange_code
    hooks_mod._default_spotify_exchange_code = _capture

    result = hooks_mod._spotify_exchange_code_hook("code1", "http://r", "cid", "csec")

    hooks_mod._default_spotify_exchange_code = original

    assert calls == [("code1", "http://r", "cid", "csec")]
    assert result == {"captured": True}


def test_default_get_job_calls_rq_fetch() -> None:
    """Test _default_get_job calls RQ Job.fetch with correct args."""
    import sys

    # Save original module if present
    orig_rq_job = sys.modules.get("rq.job")
    orig_rq = sys.modules.get("rq")

    fetch_calls: list[tuple[str, _RedisBytesClient]] = []

    class _FakeJob:
        def get_status(self) -> str:
            return "queued"

        @property
        def is_finished(self) -> bool:
            return False

        @property
        def meta(self) -> dict[str, JSONValue]:
            return {}

        @property
        def result(self) -> str | None:
            return None

        @classmethod
        def fetch(cls, job_id: str, *, connection: _RedisBytesClient) -> _FakeJob:
            fetch_calls.append((job_id, connection))
            return cls()

    # Create fake rq.job module as a class with Job attribute
    class _FakeRqJobModule(types.ModuleType):
        Job = _FakeJob

    class _FakeRqModule(types.ModuleType):
        def __init__(self, name: str, job_mod: types.ModuleType) -> None:
            super().__init__(name)
            self.job = job_mod

    fake_rq_job: types.ModuleType = _FakeRqJobModule("rq.job")
    fake_rq: types.ModuleType = _FakeRqModule("rq", fake_rq_job)

    sys.modules["rq"] = fake_rq
    sys.modules["rq.job"] = fake_rq_job

    from platform_workers.testing import FakeRedisBytesClient

    conn = FakeRedisBytesClient()
    job = _test_hooks._default_get_job("test-job-id", connection=conn)

    # Restore original modules
    if orig_rq is not None:
        sys.modules["rq"] = orig_rq
    else:
        sys.modules.pop("rq", None)
    if orig_rq_job is not None:
        sys.modules["rq.job"] = orig_rq_job
    else:
        sys.modules.pop("rq.job", None)

    assert len(fetch_calls) == 1
    assert fetch_calls[0][0] == "test-job-id"
    assert fetch_calls[0][1] is conn
    assert job.get_status() == "queued"


def test_default_build_renderer_creates_renderer() -> None:
    """Test _default_build_renderer creates a working renderer."""
    renderer = _test_hooks._default_build_renderer()
    # Actually call render_wrapped to verify it works
    from platform_music import WrappedResult

    result: WrappedResult = {
        "service": "lastfm",
        "year": 2024,
        "generated_at": "2024-12-10T00:00:00Z",
        "total_scrobbles": 1000,
        "top_artists": [],
        "top_songs": [],
        "top_by_month": [],
    }
    png_bytes = renderer.render_wrapped(result)
    # Verify it returns PNG bytes (starts with PNG signature)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_default_urlopen_get() -> None:
    """Test _default_urlopen_get with local http server."""
    import http.server
    import socketserver
    import threading

    content = b'{"test": "data"}'

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(200)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, fmt: str, *args: str) -> None:
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.handle_request)
        thread.start()

        resp = _test_hooks._default_urlopen_get(f"http://127.0.0.1:{port}/test", 5.0)
        with resp:
            data = resp.read()

        thread.join(timeout=5)

    assert data == content


def test_default_urlopen_post() -> None:
    """Test _default_urlopen_post with local http server."""
    import http.server
    import socketserver
    import threading

    received_data: list[bytes] = []
    response_content = b'{"response": "ok"}'

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", 0))
            received_data.append(self.rfile.read(length))
            self.send_response(200)
            self.send_header("Content-Length", str(len(response_content)))
            self.end_headers()
            self.wfile.write(response_content)

        def log_message(self, fmt: str, *args: str) -> None:
            pass

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.handle_request)
        thread.start()

        req = urllib.request.Request(f"http://127.0.0.1:{port}/test", data=b"test_body")
        resp = _test_hooks._default_urlopen_post(req, 5.0)
        with resp:
            data = resp.read()

        thread.join(timeout=5)

    assert data == response_content
    assert received_data == [b"test_body"]
