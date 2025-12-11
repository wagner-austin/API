"""Test hooks for music-wrapped-api - allows injecting test dependencies.

This module provides hooks for dependency injection in tests. Production code
sets hooks to real implementations at startup; tests set them to fakes.

Hooks are module-level callables that production code calls directly. Tests
assign fake implementations before running the code under test.

Usage in production code:
    from music_wrapped_api import _test_hooks
    client = _test_hooks.redis_factory(url)

Usage in tests:
    from music_wrapped_api import _test_hooks
    from platform_workers.testing import FakeRedis
    _test_hooks.redis_factory = lambda url: FakeRedis()
"""

from __future__ import annotations

import hashlib
import types
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from platform_core.config import _optional_env_str, _require_env_str
from platform_core.json_utils import JSONValue
from platform_music import WrappedResult
from platform_workers.redis import RedisStrProto, redis_for_kv
from platform_workers.rq_harness import (
    RQClientQueue,
    WorkerConfig,
    _RedisBytesClient,
    redis_raw_for_rq,
    rq_queue,
)

# =============================================================================
# Protocols for hookable dependencies
# =============================================================================


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


class RQJobProtocol(Protocol):
    """Protocol for RQ Job instance."""

    def get_status(self) -> str:
        """Get the job status."""
        ...

    @property
    def is_finished(self) -> bool:
        """Check if job is finished."""
        ...

    @property
    def meta(self) -> dict[str, JSONValue]:
        """Get job metadata."""
        ...

    @property
    def result(self) -> str | None:
        """Get job result."""
        ...


class RendererProtocol(Protocol):
    """Protocol for wrapped result renderer."""

    def render_wrapped(self, result: WrappedResult) -> bytes:
        """Render wrapped result to PNG bytes."""
        ...


class RendererFactoryProtocol(Protocol):
    """Protocol for renderer factory."""

    def __call__(self) -> RendererProtocol:
        """Create a renderer instance."""
        ...


class HttpResponseProtocol(Protocol):
    """Protocol for HTTP response context manager."""

    def read(self) -> bytes:
        """Read response body."""
        ...

    def __enter__(self) -> HttpResponseProtocol:
        """Enter context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None:
        """Exit context manager."""
        ...


class UrlOpenProtocol(Protocol):
    """Protocol for urllib.request.urlopen-like functions (GET requests)."""

    def __call__(self, url: str, timeout: float) -> HttpResponseProtocol:
        """Open a URL and return a response."""
        ...


class HttpRequestProtocol(Protocol):
    """Protocol for HTTP request object."""

    def add_header(self, name: str, value: str) -> None:
        """Add a header to the request."""
        ...


class UrlOpenPostProtocol(Protocol):
    """Protocol for urllib.request.urlopen for POST requests."""

    def __call__(self, req: HttpRequestProtocol, timeout: float) -> HttpResponseProtocol:
        """Open a request and return a response."""
        ...


class GuardRunForProjectProtocol(Protocol):
    """Protocol for run_for_project function from monorepo_guards."""

    def __call__(self, *, monorepo_root: Path, project_root: Path) -> int:
        """Run guards for a project."""
        ...


class GuardFindMonorepoRootProtocol(Protocol):
    """Protocol for _find_monorepo_root function."""

    def __call__(self, start: Path) -> Path:
        """Find the monorepo root from a starting path."""
        ...


class GuardLoadOrchestratorProtocol(Protocol):
    """Protocol for _load_orchestrator function."""

    def __call__(self, monorepo_root: Path) -> GuardRunForProjectProtocol:
        """Load the orchestrator module and return run_for_project."""
        ...


# =============================================================================
# Default implementations
# =============================================================================


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads from os.environ via platform_core."""
    return _optional_env_str(key)


def _default_require_env(key: str) -> str:
    """Production implementation - reads required env var via platform_core."""
    return _require_env_str(key)


def _default_redis_for_kv(url: str) -> RedisStrProto:
    """Production implementation - creates real Redis client for KV."""
    return redis_for_kv(url)


def _default_rq_conn(url: str) -> _RedisBytesClient:
    """Production implementation - creates RQ Redis connection."""
    return redis_raw_for_rq(url)


def _default_rq_queue(name: str, *, connection: _RedisBytesClient) -> RQClientQueue:
    """Production implementation - creates RQ queue."""
    return rq_queue(name, connection=connection)


class _RQJobFetchProtocol(Protocol):
    """Protocol for RQ Job.fetch class method."""

    def fetch(self, job_id: str, *, connection: _RedisBytesClient) -> RQJobProtocol: ...


class _RQJobModuleProtocol(Protocol):
    """Protocol for rq.job module."""

    Job: _RQJobFetchProtocol


def _default_get_job(job_id: str, *, connection: _RedisBytesClient) -> RQJobProtocol:
    """Production implementation - fetches RQ job."""
    rq_job_mod: _RQJobModuleProtocol = __import__("rq.job", fromlist=["Job"])
    job: RQJobProtocol = rq_job_mod.Job.fetch(job_id, connection=connection)
    return job


def _default_build_renderer() -> RendererProtocol:
    """Production implementation - creates image renderer."""
    img_mod = __import__("platform_music.image_gen.renderer", fromlist=["build_renderer"])
    factory: RendererFactoryProtocol = img_mod.build_renderer
    return factory()


def _default_urlopen_get(url: str, timeout: float) -> HttpResponseProtocol:
    """Production implementation - performs HTTP GET via urllib."""
    resp: HttpResponseProtocol = urllib.request.urlopen(url, timeout=timeout)
    return resp


def _default_urlopen_post(req: HttpRequestProtocol, timeout: float) -> HttpResponseProtocol:
    """Production implementation - performs HTTP POST via urllib.

    Note: req parameter is typed as HttpRequestProtocol for hook compatibility,
    but at runtime production code passes urllib.request.Request which urlopen accepts.
    """
    # At runtime, req is always a urllib.request.Request instance.
    # We verify this and use the verified instance directly.
    if not isinstance(req, urllib.request.Request):
        raise TypeError("req must be a urllib.request.Request instance")
    resp: HttpResponseProtocol = urllib.request.urlopen(req, timeout=timeout)
    return resp


def _default_make_request(url: str, data: bytes) -> HttpRequestProtocol:
    """Production implementation - creates urllib Request."""
    req: HttpRequestProtocol = urllib.request.Request(url, data)
    return req


def _default_lfm_get_session_json(
    api_key: str, api_secret: str, token: str
) -> dict[str, JSONValue]:
    """Production implementation - calls Last.fm auth.getSession API."""
    raw = f"api_key{api_key}methodauth.getSessiontoken{token}{api_secret}"
    sig = hashlib.md5(raw.encode("utf-8")).hexdigest()
    params = {
        "method": "auth.getSession",
        "api_key": api_key,
        "api_sig": sig,
        "token": token,
        "format": "json",
    }
    url = "https://ws.audioscrobbler.com/2.0/?" + urllib.parse.urlencode(params)
    with urlopen_get(url, 10.0) as resp:
        data = resp.read().decode("utf-8")
    from platform_core.json_utils import load_json_str

    doc = load_json_str(data)
    if not isinstance(doc, dict):
        from platform_core.errors import AppError, ErrorCode

        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="invalid lastfm json",
            http_status=502,
        )
    return doc


def _default_spotify_exchange_code(
    code: str, redirect_uri: str, *, client_id: str, client_secret: str
) -> dict[str, JSONValue]:
    """Production implementation - exchanges Spotify OAuth code for tokens."""
    import base64 as _b64

    token_url = "https://accounts.spotify.com/api/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    encoded = urllib.parse.urlencode(data).encode("utf-8")

    req = make_request(token_url, encoded)
    auth = (client_id + ":" + client_secret).encode("utf-8")
    hdr = "Basic " + _b64.b64encode(auth).decode("utf-8")
    req.add_header("Authorization", hdr)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")

    with urlopen_post(req, 10.0) as resp:
        payload = resp.read().decode("utf-8")

    from platform_core.json_utils import load_json_str

    doc = load_json_str(payload)
    if not isinstance(doc, dict):
        from platform_core.errors import AppError, ErrorCode

        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="invalid spotify json",
            http_status=502,
        )
    return doc


def _default_rand_state() -> str:
    """Production implementation - generates random state for OAuth."""
    import secrets as _secrets

    return _secrets.token_urlsafe(16)


def _default_guard_find_monorepo_root(start: Path) -> Path:
    """Production implementation - finds monorepo root by climbing directories."""
    current = start
    while True:
        if (current / "libs").is_dir():
            return current
        if current.parent == current:
            raise RuntimeError("monorepo root with 'libs' directory not found")
        current = current.parent


def _default_guard_load_orchestrator(monorepo_root: Path) -> GuardRunForProjectProtocol:
    """Production implementation - loads the orchestrator module."""
    import sys

    libs_path = monorepo_root / "libs"
    guards_src = libs_path / "monorepo_guards" / "src"
    sys.path.insert(0, str(guards_src))
    sys.path.insert(0, str(libs_path))
    mod = __import__("monorepo_guards.orchestrator", fromlist=["run_for_project"])
    run_for_project: GuardRunForProjectProtocol = mod.run_for_project
    return run_for_project


# =============================================================================
# Module-level hooks
# =============================================================================

# Hook for worker runner (used by worker_entry.py)
# Tests set this BEFORE running worker_entry as __main__.
test_runner: WorkerRunnerProtocol | None = None

# Hook for environment variable access. Tests can override to provide fake values.
get_env: Callable[[str], str | None] = _default_get_env

# Hook for required environment variable access.
require_env: Callable[[str], str] = _default_require_env

# Hook for Redis KV client factory. Tests can override with FakeRedis.
redis_factory: Callable[[str], RedisStrProto] = _default_redis_for_kv

# Hook for RQ Redis connection. Tests can override with FakeRedisBytesClient.
rq_conn: Callable[[str], _RedisBytesClient] = _default_rq_conn


# Hook for RQ queue factory. Tests can override with FakeQueue.
def _rq_queue_factory_hook(name: str, connection: _RedisBytesClient) -> RQClientQueue:
    """Default RQ queue factory hook."""
    return _default_rq_queue(name, connection=connection)


rq_queue_factory: Callable[[str, _RedisBytesClient], RQClientQueue] = _rq_queue_factory_hook


# Hook for RQ job fetcher. Tests can override with fake jobs.
def _get_job_hook(job_id: str, connection: _RedisBytesClient) -> RQJobProtocol:
    """Default RQ job fetcher hook."""
    return _default_get_job(job_id, connection=connection)


get_job: Callable[[str, _RedisBytesClient], RQJobProtocol] = _get_job_hook

# Hook for wrapped image renderer factory.
build_renderer: Callable[[], RendererProtocol] = _default_build_renderer

# Hook for HTTP GET requests. Tests can override to avoid network.
urlopen_get: UrlOpenProtocol = _default_urlopen_get

# Hook for HTTP POST requests. Tests can override to avoid network.
urlopen_post: UrlOpenPostProtocol = _default_urlopen_post

# Hook for creating HTTP requests. Tests can override.
make_request: Callable[[str, bytes], HttpRequestProtocol] = _default_make_request

# Hook for Last.fm session JSON getter. Tests can override to avoid network.
lfm_get_session_json: Callable[[str, str, str], dict[str, JSONValue]] = (
    _default_lfm_get_session_json
)


# Hook for Spotify code exchange. Tests can override to avoid network.
def _spotify_exchange_code_hook(
    code: str, redirect_uri: str, client_id: str, client_secret: str
) -> dict[str, JSONValue]:
    """Default Spotify code exchange hook."""
    return _default_spotify_exchange_code(
        code, redirect_uri, client_id=client_id, client_secret=client_secret
    )


spotify_exchange_code: Callable[[str, str, str, str], dict[str, JSONValue]] = (
    _spotify_exchange_code_hook
)

# Hook for random state generation. Tests can override for determinism.
rand_state: Callable[[], str] = _default_rand_state

# Hook for guard find_monorepo_root. Tests can override to return fake paths.
guard_find_monorepo_root: GuardFindMonorepoRootProtocol = _default_guard_find_monorepo_root

# Hook for guard load_orchestrator. Tests can override to return fake orchestrators.
guard_load_orchestrator: GuardLoadOrchestratorProtocol = _default_guard_load_orchestrator


__all__ = [
    # Protocols
    "GuardFindMonorepoRootProtocol",
    "GuardLoadOrchestratorProtocol",
    "GuardRunForProjectProtocol",
    "HttpRequestProtocol",
    "HttpResponseProtocol",
    "RQJobProtocol",
    "RendererFactoryProtocol",
    "RendererProtocol",
    "UrlOpenPostProtocol",
    "UrlOpenProtocol",
    "WorkerRunnerProtocol",
    # Default implementations
    "_default_build_renderer",
    "_default_get_env",
    "_default_get_job",
    "_default_guard_find_monorepo_root",
    "_default_guard_load_orchestrator",
    "_default_lfm_get_session_json",
    "_default_make_request",
    "_default_rand_state",
    "_default_redis_for_kv",
    "_default_require_env",
    "_default_rq_conn",
    "_default_rq_queue",
    "_default_spotify_exchange_code",
    "_default_urlopen_get",
    "_default_urlopen_post",
    # Module-level hooks
    "build_renderer",
    "get_env",
    "get_job",
    "guard_find_monorepo_root",
    "guard_load_orchestrator",
    "lfm_get_session_json",
    "make_request",
    "rand_state",
    "redis_factory",
    "require_env",
    "rq_conn",
    "rq_queue_factory",
    "spotify_exchange_code",
    "test_runner",
    "urlopen_get",
    "urlopen_post",
]
