"""Testing utilities for services using platform_core."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NoReturn, Protocol

from platform_core.comparability import RunFingerprint
from platform_core.config import _test_hooks
from platform_core.determinism_record import (
    UNPINNED_STACK,
    DeterminismRecord,
    determinism_record,
)
from platform_core.environment_record import HostRecord, PackageVersion
from platform_core.http_client import HttpxAsyncClient, HttpxClient, HttpxResponse, Timeout
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

# =============================================================================
# Environment Fakes
# =============================================================================


class FakeEnv:
    """Fake environment for testing config loaders."""

    def __init__(self, env_vars: dict[str, str] | None = None) -> None:
        self._env: dict[str, str] = env_vars or {}

    def get(self, key: str) -> str | None:
        return self._env.get(key)

    def set(self, key: str, value: str) -> None:
        self._env[key] = value

    def delete(self, key: str) -> None:
        self._env.pop(key, None)

    def clear(self) -> None:
        self._env.clear()

    def __call__(self, key: str) -> str | None:
        return self.get(key)


def make_fake_env(env_vars: dict[str, str] | None = None) -> FakeEnv:
    """Create a FakeEnv and install it as the config hook."""
    env = FakeEnv(env_vars)
    _test_hooks.get_env = env
    return env


# =============================================================================
# HTTP Client Fakes
# =============================================================================


class FakeTimeout:
    """Protocol-compliant fake for httpx.Timeout."""

    __slots__ = ("_timeout",)

    def __init__(self, timeout: float) -> None:
        self._timeout = float(timeout)

    def __repr__(self) -> str:
        return f"Timeout({self._timeout})"


class FakeHttpxResponse:
    """Protocol-compliant fake for httpx.Response.

    Satisfies HttpxResponse Protocol from platform_core.http_client.
    Supports initialization from JSON body, raw bytes, or text.
    """

    __slots__ = ("_json", "content", "headers", "status_code", "text")

    def __init__(
        self,
        status: int,
        json_body: JSONValue | None = None,
        *,
        content: bytes | None = None,
        text: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.status_code: int = int(status)
        self._json: JSONValue | None = json_body
        self.headers: Mapping[str, str] = dict(headers) if headers else {}

        # Determine content and text from inputs
        if content is not None:
            self.content: bytes | bytearray = content
            self.text: str = text if text is not None else content.decode("utf-8", errors="replace")
        elif json_body is not None:
            self.text = dump_json_str(json_body)
            self.content = self.text.encode("utf-8")
        elif text is not None:
            self.text = text
            self.content = text.encode("utf-8")
        else:
            self.text = ""
            self.content = b""

    def json(self) -> JSONValue:
        if self._json is not None:
            return self._json
        return load_json_str(self.text)

    def raise_for_status(self) -> None:
        """Raise for non-2xx status codes.

        Raises:
            RuntimeError: If status code is 4xx or 5xx.
        """
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHttpxAsyncClient:
    """Protocol-compliant fake for httpx.AsyncClient.

    Satisfies HttpxAsyncClient Protocol from platform_core.http_client.
    Supports configurable responses and exception raising for testing.
    """

    __slots__ = (
        "_call_count",
        "_exception_count",
        "_exception_to_raise",
        "_response",
        "seen_headers",
        "seen_urls",
    )

    def __init__(
        self,
        response: HttpxResponse | None = None,
        *,
        exception_to_raise: Exception | None = None,
        exception_count: int = 0,
    ) -> None:
        self._response: HttpxResponse | None = response
        self._exception_to_raise: Exception | None = exception_to_raise
        self._exception_count: int = exception_count
        self._call_count: int = 0
        self.seen_headers: dict[str, str] = {}
        self.seen_urls: list[str] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    async def aclose(self) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        self._call_count += 1
        self.seen_urls.append(url)
        self.seen_headers.update(headers)
        _ = (json, files)  # Unused but part of protocol

        # Raise exception if configured (for first N calls)
        should_raise = self._exception_to_raise is not None and (
            self._exception_count == 0 or self._call_count <= self._exception_count
        )
        if should_raise and self._exception_to_raise is not None:
            raise self._exception_to_raise

        if self._response is None:
            raise RuntimeError("No response configured for FakeHttpxAsyncClient")
        return self._response

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
    ) -> HttpxResponse:
        return await self.post(url, headers=headers, json=None, files=None)


class FakeHttpxAsyncClientRaises:
    """Fake async client that always raises on post/get."""

    __slots__ = ("_exception",)

    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    async def aclose(self) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> NoReturn:
        _ = (url, headers, json, files)
        raise self._exception

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
    ) -> NoReturn:
        raise self._exception


class FakeHttpxClient:
    """Protocol-compliant fake for httpx.Client (sync).

    Satisfies HttpxClient Protocol from platform_core.http_client.
    """

    __slots__ = ("_response", "seen_headers", "seen_urls")

    def __init__(self, response: HttpxResponse | None = None) -> None:
        self._response: HttpxResponse | None = response
        self.seen_headers: dict[str, str] = {}
        self.seen_urls: list[str] = []

    def close(self) -> None:
        return None

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, str | int] | None = None,
    ) -> HttpxResponse:
        self.seen_urls.append(url)
        if headers is not None:
            self.seen_headers.update(headers)
        _ = params

        if self._response is None:
            raise RuntimeError("No response configured for FakeHttpxClient")
        return self._response

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        self.seen_urls.append(url)
        self.seen_headers.update(headers)
        _ = (json, files)

        if self._response is None:
            raise RuntimeError("No response configured for FakeHttpxClient")
        return self._response


class FakeHttpxClientRaises:
    """Fake sync client that always raises on post."""

    __slots__ = ("_exception",)

    def __init__(self, exception: Exception) -> None:
        self._exception = exception

    def close(self) -> None:
        return None

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> NoReturn:
        _ = (url, headers, json, files)
        raise self._exception


# =============================================================================
# Factory Protocols and Helpers
# =============================================================================


class TimeoutCtor(Protocol):
    """Protocol for Timeout constructor."""

    def __call__(self, timeout: float) -> Timeout: ...


class AsyncClientCtor(Protocol):
    """Protocol for AsyncClient constructor."""

    def __call__(self, *, timeout: Timeout) -> HttpxAsyncClient: ...


class ClientCtor(Protocol):
    """Protocol for Client constructor."""

    def __call__(self, *, timeout: Timeout) -> HttpxClient: ...


def make_timeout_ctor() -> TimeoutCtor:
    """Create a typed Timeout constructor returning FakeTimeout."""

    def ctor(timeout: float) -> Timeout:
        return FakeTimeout(timeout)

    return ctor


def make_async_client_ctor(response: HttpxResponse) -> AsyncClientCtor:
    """Create a typed AsyncClient constructor returning FakeHttpxAsyncClient."""

    def ctor(*, timeout: Timeout) -> HttpxAsyncClient:
        _ = timeout
        return FakeHttpxAsyncClient(response)

    return ctor


def make_client_ctor(response: HttpxResponse) -> ClientCtor:
    """Create a typed Client constructor returning FakeHttpxClient."""

    def ctor(*, timeout: Timeout) -> HttpxClient:
        _ = timeout
        return FakeHttpxClient(response)

    return ctor


class FakeHttpxModule:
    """Fake httpx module for test hook injection.

    Provides Timeout/AsyncClient/Client constructors that return fakes.
    """

    def __init__(
        self,
        response: HttpxResponse,
        *,
        async_client: bool = False,
    ) -> None:
        object.__setattr__(self, "Timeout", make_timeout_ctor())
        if async_client:
            object.__setattr__(self, "AsyncClient", make_async_client_ctor(response))
        else:
            object.__setattr__(self, "Client", make_client_ctor(response))


class FakeHttpxModuleSyncOnly:
    """Fake httpx module with only sync Client (for sync-only code)."""

    def __init__(self, response: HttpxResponse) -> None:
        object.__setattr__(self, "Timeout", make_timeout_ctor())
        object.__setattr__(self, "Client", make_client_ctor(response))


# =============================================================================
# Run-fingerprint Fakes
# =============================================================================

#: The machine a fingerprint built by :func:`sample_run_fingerprint` reports.
#:
#: Stated once here rather than in each consumer's test module. Every repo
#: that builds a :class:`~platform_core.comparability.RunFingerprint` needs
#: one, and thirty hand-written copies is how the LAST axis addition became a
#: thirty-site sweep across three repositories.
SAMPLE_HOST: HostRecord = HostRecord(
    platform="Linux-5.14.0-x86_64-with-glibc2.34",
    machine="x86_64",
    logical_cores=8,
)

#: The library versions a fingerprint built by :func:`sample_run_fingerprint`
#: reports.
SAMPLE_PACKAGES: tuple[PackageVersion, ...] = (PackageVersion(name="numpy", version="2.3.5"),)


def sample_run_fingerprint(
    *,
    image_digest: str = "sha256:sample",
    gpu_model: str = "A100",
    driver_version: str = "580.82.07",
    determinism: DeterminismRecord | None = None,
    host: HostRecord | None = None,
    packages: tuple[PackageVersion, ...] | None = None,
) -> RunFingerprint:
    """Build a fingerprint for a test that is not about the fingerprint.

    THE POINT IS THE DEFAULTS, and specifically that a test which does not
    care about an axis does not mention it. A fingerprint has six axes and
    most tests are about one of them; when every test hand-built the whole
    record, adding an axis meant editing every test that had no opinion about
    it. A test that IS about an axis passes it and the default never applies.

    Args:
        image_digest: The image axis.
        gpu_model: The card axis.
        driver_version: The driver axis.
        determinism: The determinism axis, defaulting to a torch run with
            nothing pinned.
        host: The machine axis, defaulting to :data:`SAMPLE_HOST`.
        packages: The library axis, defaulting to :data:`SAMPLE_PACKAGES`.

    Returns:
        The fingerprint.
    """
    return RunFingerprint(
        image_digest=image_digest,
        gpu_model=gpu_model,
        driver_version=driver_version,
        determinism=(
            determinism if determinism is not None else determinism_record(UNPINNED_STACK, {})
        ),
        host=host if host is not None else SAMPLE_HOST,
        packages=packages if packages is not None else SAMPLE_PACKAGES,
    )


class FakeHostProbe:
    """A machine stated rather than owned, for fingerprint tests.

    Every consumer of :class:`~platform_core.comparability.RunFingerprint`
    needs a host to build one, and a test that read the real machine would
    assert different values on every developer's box.
    """

    def __init__(self, *, platform: str, machine: str, logical_cores: int) -> None:
        """Store the machine this probe reports.

        Args:
            platform: The operating system, release and build to report.
            machine: The instruction-set architecture to report.
            logical_cores: The logical processor count to report.
        """
        self._platform = platform
        self._machine = machine
        self._logical_cores = logical_cores

    def platform(self) -> str:
        """Return the stated platform string.

        Returns:
            The platform this probe was built with.
        """
        return self._platform

    def machine(self) -> str:
        """Return the stated architecture.

        Returns:
            The architecture this probe was built with.
        """
        return self._machine

    def logical_cores(self) -> int:
        """Return the stated logical processor count.

        Returns:
            The count this probe was built with.
        """
        return self._logical_cores


class FakeVersionReader:
    """Installed versions stated rather than resolved, for fingerprint tests.

    Raises on an unnamed distribution rather than inventing a version: a test
    that asked for a library it did not state has a mistake in it, and a
    default would hide the mistake behind a plausible-looking fingerprint.
    """

    def __init__(self, versions: Mapping[str, str]) -> None:
        """Store the versions this reader reports.

        Args:
            versions: Version by distribution name.
        """
        self._versions = dict(versions)

    def __call__(self, distribution: str) -> str:
        """Return the stated version of a distribution.

        Args:
            distribution: The distribution name.

        Returns:
            Its stated version.

        Raises:
            KeyError: When the distribution was not stated.
        """
        return self._versions[distribution]


__all__ = [
    "SAMPLE_HOST",
    "SAMPLE_PACKAGES",
    "AsyncClientCtor",
    "ClientCtor",
    "FakeEnv",
    "FakeHostProbe",
    "FakeHttpxAsyncClient",
    "FakeHttpxAsyncClientRaises",
    "FakeHttpxClient",
    "FakeHttpxClientRaises",
    "FakeHttpxModule",
    "FakeHttpxModuleSyncOnly",
    "FakeHttpxResponse",
    "FakeTimeout",
    "FakeVersionReader",
    "TimeoutCtor",
    "make_async_client_ctor",
    "make_client_ctor",
    "make_fake_env",
    "make_timeout_ctor",
    "sample_run_fingerprint",
]
