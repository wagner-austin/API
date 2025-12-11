"""Test hooks for turkic-api - allows injecting test dependencies.

This module provides hooks for dependency injection in tests. Production code
sets hooks to real implementations at startup; tests set them to fakes.

Hooks are module-level callables that production code calls directly. Tests
assign fake implementations before running the code under test.

Usage in production code:
    from turkic_api import _test_hooks
    client = _test_hooks.redis_factory(url)

Usage in tests:
    from turkic_api import _test_hooks
    from platform_workers.testing import FakeRedis
    _test_hooks.redis_factory = lambda url: FakeRedis()
"""

from __future__ import annotations

import types
from collections.abc import Callable, Generator
from pathlib import Path
from typing import BinaryIO, Protocol

import httpx
import numpy as np
import requests
from numpy.typing import NDArray
from platform_core.config import _optional_env_str
from platform_core.data_bank_client import DataBankClient, HeadInfo
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.json_utils import JSONValue
from platform_workers.redis import RedisStrProto, redis_for_kv
from platform_workers.rq_harness import WorkerConfig

from turkic_api.core.models import ProcessSpec


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


# =============================================================================
# Protocols for hookable dependencies
# =============================================================================


class LangIdModelProtocol(Protocol):
    """Protocol for language identification model with predict method."""

    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        """Predict language labels and probabilities for the given text."""
        ...


class LocalCorpusServiceProtocol(Protocol):
    """Protocol for LocalCorpusService - allows injecting fakes for testing."""

    def stream(self, spec: ProcessSpec) -> Generator[str, None, None]:
        """Stream corpus lines for the given spec."""
        ...


class DataBankUploaderProtocol(Protocol):
    """Protocol for DataBankClient upload method - allows injecting fakes for testing."""

    def upload(
        self,
        file_id: str,
        stream: BinaryIO,
        *,
        content_type: str,
        request_id: str | None,
    ) -> FileUploadResponse:
        """Upload a file to the data bank."""
        ...


class StreamDownloadResponseProtocol(Protocol):
    """Protocol for stream download response."""

    @property
    def status_code(self) -> int:
        """HTTP status code of the response."""
        ...

    def iter_bytes(self) -> Generator[bytes, None, None]:
        """Iterate over response bytes."""
        ...

    def close(self) -> None:
        """Close the response."""
        ...


class DataBankDownloaderProtocol(Protocol):
    """Protocol for DataBankClient download methods - allows injecting fakes for testing."""

    def head(self, file_id: str, *, request_id: str | None = None) -> HeadInfo:
        """Get file metadata."""
        ...

    def stream_download(
        self,
        file_id: str,
        *,
        request_id: str | None = None,
        chunk_size: int = 8192,
    ) -> httpx.Response:
        """Stream download a file."""
        ...


class EnsureCorpusProtocol(Protocol):
    """Protocol for ensure_corpus_file function."""

    def __call__(
        self,
        spec: ProcessSpec,
        data_dir: str,
        script: str | None = None,
        *,
        langid_model: LangIdModelProtocol | None = None,
    ) -> Path:
        """Ensure corpus file exists, downloading if necessary."""
        ...


class ToIpaProtocol(Protocol):
    """Protocol for to_ipa transliteration function."""

    def __call__(self, text: str, language: str) -> str:
        """Transliterate text to IPA."""
        ...


# =============================================================================
# Factory Protocols
# =============================================================================


class LocalCorpusServiceFactoryProtocol(Protocol):
    """Protocol for LocalCorpusService class factory."""

    def __call__(self, data_dir: str) -> LocalCorpusServiceProtocol:
        """Create a LocalCorpusService for the given data directory."""
        ...


class DataBankUploaderFactoryProtocol(Protocol):
    """Protocol for DataBankClient factory - allows injecting fakes for testing."""

    def __call__(
        self, api_url: str, api_key: str, *, timeout_seconds: float
    ) -> DataBankUploaderProtocol:
        """Create a DataBankClient for the given URL and API key."""
        ...


class DataBankDownloaderFactoryProtocol(Protocol):
    """Protocol for DataBankClient factory for downloading - allows injecting fakes."""

    def __call__(
        self, api_url: str, api_key: str, *, timeout_seconds: float
    ) -> DataBankDownloaderProtocol:
        """Create a DataBankClient for downloading files."""
        ...


class LangIdModelLoaderProtocol(Protocol):
    """Protocol for load_langid_model function."""

    def __call__(self, data_dir: str, prefer_218e: bool = True) -> LangIdModelProtocol:
        """Load a language identification model from the data directory."""
        ...


# =============================================================================
# Default implementations
# =============================================================================


def _default_get_env(key: str) -> str | None:
    """Production implementation - reads from os.environ."""
    return _optional_env_str(key)


def _default_redis_for_kv(url: str) -> RedisStrProto:
    """Production implementation - creates real Redis client."""
    return redis_for_kv(url)


def _default_local_corpus_factory(data_dir: str) -> LocalCorpusServiceProtocol:
    """Production implementation - creates real LocalCorpusService."""
    # Import lazily to avoid circular import
    from turkic_api.core.corpus import LocalCorpusService

    return LocalCorpusService(data_dir)


def _default_data_bank_uploader_factory(
    api_url: str, api_key: str, *, timeout_seconds: float
) -> DataBankUploaderProtocol:
    """Production implementation - creates real DataBankClient."""
    return DataBankClient(api_url, api_key, timeout_seconds=timeout_seconds)


def _default_data_bank_downloader_factory(
    api_url: str, api_key: str, *, timeout_seconds: float
) -> DataBankDownloaderProtocol:
    """Production implementation - creates real DataBankClient for downloading."""
    return DataBankClient(api_url, api_key, timeout_seconds=timeout_seconds)


def _default_ensure_corpus_file(
    spec: ProcessSpec,
    data_dir: str,
    script: str | None = None,
    *,
    langid_model: LangIdModelProtocol | None = None,
) -> Path:
    """Production implementation - calls real ensure_corpus_file."""
    from turkic_api.core.corpus_download import ensure_corpus_file as _ensure

    return _ensure(spec, data_dir, script, langid_model=langid_model)


def _default_load_langid_model(data_dir: str, prefer_218e: bool = True) -> LangIdModelProtocol:
    """Production implementation - loads real langid model."""
    from turkic_api.core.langid import load_langid_model as _load

    return _load(data_dir, prefer_218e=prefer_218e)


# =============================================================================
# Langid module hooks
# =============================================================================


class LangIdDownloadProtocol(Protocol):
    """Protocol for langid download function."""

    def __call__(self, url: str, dest: Path) -> None:
        """Download a file from url to dest."""
        ...


class LangIdEnsureModelPathProtocol(Protocol):
    """Protocol for langid ensure_model_path function."""

    def __call__(self, data_dir: str, prefer_218e: bool = True) -> Path:
        """Ensure model file exists, downloading if necessary."""
        ...


class LangIdModelFactoryProtocol(Protocol):
    """Protocol for FastText model factory function."""

    def __call__(self, *, model_path: str) -> LangIdModelProtocol:
        """Create a LangIdModel from the given path."""
        ...


class LangIdFastTextFactoryProtocol(Protocol):
    """Protocol for langid _get_fasttext_model_factory function."""

    def __call__(self) -> LangIdModelFactoryProtocol:
        """Get the FastText model factory."""
        ...


def _default_langid_download(url: str, dest: Path) -> None:
    """Production implementation - downloads file via requests."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with dest.open("wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


def _default_langid_ensure_model_path(data_dir: str, prefer_218e: bool = True) -> Path:
    """Production implementation - ensures model path exists."""
    from turkic_api.core.langid import (
        _MODEL_DIRNAME,
        _URL_176,
        _URL_218E,
    )

    base = Path(data_dir) / _MODEL_DIRNAME
    path_218e = base / "lid218e.bin"
    path_176 = base / "lid.176.bin"
    if prefer_218e:
        if not path_218e.exists():
            langid_download(url=_URL_218E, dest=path_218e)
        return path_218e
    if not path_176.exists():
        langid_download(url=_URL_176, dest=path_176)
    return path_176


def _default_langid_get_fasttext_factory() -> LangIdModelFactoryProtocol:
    """Production implementation - gets FastText model factory."""
    ft_module = __import__("fasttext.FastText", fromlist=["_FastText"])
    factory: LangIdModelFactoryProtocol = ft_module._FastText
    return factory


# =============================================================================
# Corpus download hooks (for requests.get in wikipedia streaming)
# =============================================================================


class RawStreamProtocol(Protocol):
    """Protocol for raw stream used in Wikipedia streaming.

    Must satisfy _compression._Reader which bz2.BZ2File requires:
    - read(n: int) -> bytes
    - seekable() -> bool
    - seek(n: int) -> int
    """

    def read(self, n: int, /) -> bytes:
        """Read up to n bytes."""
        ...

    def seekable(self) -> bool:
        """Return whether the stream supports seeking."""
        ...

    def seek(self, n: int, /) -> int:
        """Seek to position n."""
        ...


class WikipediaRequestsResponseProtocol(Protocol):
    """Protocol for requests response used in Wikipedia streaming."""

    @property
    def raw(self) -> RawStreamProtocol:
        """Raw response body."""
        ...

    def raise_for_status(self) -> None:
        """Raise exception for non-2xx status."""
        ...

    def __enter__(self) -> WikipediaRequestsResponseProtocol:
        """Context manager entry."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        """Context manager exit."""
        ...


class WikipediaRequestsGetProtocol(Protocol):
    """Protocol for requests.get function used in Wikipedia streaming."""

    def __call__(
        self, url: str, *, stream: bool, timeout: int
    ) -> WikipediaRequestsResponseProtocol:
        """Make a GET request."""
        ...


def _default_wikipedia_requests_get(
    url: str, *, stream: bool, timeout: int
) -> WikipediaRequestsResponseProtocol:
    """Production implementation - uses real requests.get.

    Note: requests.Response structurally implements WikipediaRequestsResponseProtocol
    even though mypy can't prove it due to __enter__ returning Self. At runtime
    this works correctly because we only access .raw and .raise_for_status().
    """
    # The response implements the protocol structurally at runtime.
    # We use a factory function to satisfy the hook type while letting
    # the real requests.Response be used.
    resp = requests.get(url, stream=stream, timeout=timeout)

    class _Adapter:
        """Adapter to satisfy WikipediaRequestsResponseProtocol."""

        @property
        def raw(self) -> RawStreamProtocol:
            """Return the raw response stream."""
            return resp.raw

        def raise_for_status(self) -> None:
            resp.raise_for_status()

        def __enter__(self) -> WikipediaRequestsResponseProtocol:
            resp.__enter__()
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: types.TracebackType | None,
        ) -> None:
            resp.__exit__(exc_type, exc_val, exc_tb)

    return _Adapter()


def _default_to_ipa(text: str, language: str) -> str:
    """Production implementation - calls real to_ipa."""
    from turkic_api.core.translit import to_ipa as _to_ipa

    return _to_ipa(text, language)


# =============================================================================
# Models hooks (for testing defensive error paths in validators)
# =============================================================================


class DecodeRequiredLiteralProtocol(Protocol):
    """Protocol for _decode_required_literal function."""

    def __call__(
        self,
        val: JSONValue,
        field: str,
        allowed: frozenset[str],
    ) -> str:
        """Decode and validate a required literal value."""
        ...


class DecodeOptionalLiteralProtocol(Protocol):
    """Protocol for _decode_optional_literal function."""

    def __call__(
        self,
        val: JSONValue,
        field: str,
        allowed: frozenset[str],
    ) -> str | None:
        """Decode and validate an optional literal value."""
        ...


def _default_decode_required_literal(
    val: JSONValue,
    field: str,
    allowed: frozenset[str],
) -> str:
    """Production implementation - uses real validator."""
    from turkic_api.api.validators import _decode_required_literal as _decode

    return _decode(val, field, allowed)


def _default_decode_optional_literal(
    val: JSONValue,
    field: str,
    allowed: frozenset[str],
) -> str | None:
    """Production implementation - uses real validator."""
    from turkic_api.api.validators import _decode_optional_literal as _decode

    return _decode(val, field, allowed)


def _default_path_exists(path: Path) -> bool:
    """Production implementation - checks if path exists on filesystem."""
    return path.exists()


def _default_path_unlink(path: Path, *, missing_ok: bool = False) -> None:
    """Production implementation - unlinks a file from the filesystem."""
    path.unlink(missing_ok=missing_ok)


def _default_stream_oscar(lang: str) -> Generator[str, None, None]:
    """Production implementation - streams from OSCAR dataset."""
    from turkic_api.core.corpus_download import stream_oscar as _stream

    yield from _stream(lang)


def _default_stream_wikipedia_xml(lang: str) -> Generator[str, None, None]:
    """Production implementation - streams from Wikipedia XML dump."""
    from turkic_api.core.corpus_download import stream_wikipedia_xml as _stream

    yield from _stream(lang)


def _default_stream_culturax(lang: str) -> Generator[str, None, None]:
    """Production implementation - streams from CulturaX dataset."""
    from turkic_api.core.corpus_download import stream_culturax as _stream

    yield from _stream(lang)


# =============================================================================
# Module-level hooks
# =============================================================================

# Hook for worker runner (used by worker_entry.py)
# Tests set this BEFORE running worker_entry as __main__.
test_runner: WorkerRunnerProtocol | None = None

# Hook for environment variable access. Tests can override to provide fake values.
get_env: Callable[[str], str | None] = _default_get_env

# Hook for Redis client factory. Tests can override with FakeRedis.
redis_factory: Callable[[str], RedisStrProto] = _default_redis_for_kv

# Hook for LocalCorpusService factory. Tests can inject fake services.
local_corpus_service_factory: LocalCorpusServiceFactoryProtocol = _default_local_corpus_factory

# Hook for DataBankClient factory (for uploads). Tests can inject fake clients.
data_bank_client_factory: DataBankUploaderFactoryProtocol = _default_data_bank_uploader_factory

# Hook for DataBankClient factory (for downloads). Tests can inject fake clients.
data_bank_downloader_factory: DataBankDownloaderFactoryProtocol = (
    _default_data_bank_downloader_factory
)

# Hook for ensure_corpus_file. Tests can override to no-op.
ensure_corpus_file: EnsureCorpusProtocol = _default_ensure_corpus_file

# Hook for load_langid_model. Tests can inject fake models.
load_langid_model: LangIdModelLoaderProtocol = _default_load_langid_model

# Hook for to_ipa transliteration. Tests can inject identity function.
to_ipa: ToIpaProtocol = _default_to_ipa

# Hook for path existence checking. Tests can override to fake filesystem state.
path_exists: Callable[[Path], bool] = _default_path_exists


def _path_unlink_with_missing_ok(path: Path) -> None:
    """Default path_unlink hook - unlinks with missing_ok=True."""
    _default_path_unlink(path, missing_ok=True)


# Hook for path unlink. Tests can override to simulate filesystem errors.
path_unlink: Callable[[Path], None] = _path_unlink_with_missing_ok

# Hook for corpus streaming functions. Tests can inject fake data sources.
stream_oscar_hook: Callable[[str], Generator[str, None, None]] = _default_stream_oscar
stream_wikipedia_xml_hook: Callable[[str], Generator[str, None, None]] = (
    _default_stream_wikipedia_xml
)
stream_culturax_hook: Callable[[str], Generator[str, None, None]] = _default_stream_culturax

# Hook for requests.get used in Wikipedia streaming. Tests can override to avoid network.
wikipedia_requests_get: WikipediaRequestsGetProtocol = _default_wikipedia_requests_get

# Hook for langid download. Tests can override to avoid network.
langid_download: LangIdDownloadProtocol = _default_langid_download

# Hook for langid ensure_model_path. Tests can override to return fake paths.
langid_ensure_model_path: LangIdEnsureModelPathProtocol = _default_langid_ensure_model_path

# Hook for langid get_fasttext_factory. Tests can override to return fake factories.
langid_get_fasttext_factory: LangIdFastTextFactoryProtocol = _default_langid_get_fasttext_factory

# Hook for models literal decoder functions. Tests can override for defensive error path testing.
decode_required_literal: DecodeRequiredLiteralProtocol = _default_decode_required_literal
decode_optional_literal: DecodeOptionalLiteralProtocol = _default_decode_optional_literal

# Hook for source map. Tests can override to test defensive error paths.
source_map: dict[str, str] = {
    "oscar": "oscar",
    "wikipedia": "wikipedia",
    "culturax": "culturax",
}

# Hook for language map. Tests can override to test defensive error paths.
language_map: dict[str, str] = {
    "kk": "kk",
    "ky": "ky",
    "uz": "uz",
    "tr": "tr",
    "ug": "ug",
    "fi": "fi",
    "az": "az",
    "en": "en",
}


# =============================================================================
# Guard script hooks
# =============================================================================


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


# Hook for guard find_monorepo_root. Tests can override to return fake paths.
guard_find_monorepo_root: GuardFindMonorepoRootProtocol = _default_guard_find_monorepo_root

# Hook for guard load_orchestrator. Tests can override to return fake orchestrators.
guard_load_orchestrator: GuardLoadOrchestratorProtocol = _default_guard_load_orchestrator


__all__ = [
    # Protocols
    "DataBankDownloaderFactoryProtocol",
    "DataBankDownloaderProtocol",
    "DataBankUploaderFactoryProtocol",
    "DataBankUploaderProtocol",
    "DecodeOptionalLiteralProtocol",
    "DecodeRequiredLiteralProtocol",
    "EnsureCorpusProtocol",
    "GuardFindMonorepoRootProtocol",
    "GuardLoadOrchestratorProtocol",
    "GuardRunForProjectProtocol",
    "LangIdDownloadProtocol",
    "LangIdEnsureModelPathProtocol",
    "LangIdFastTextFactoryProtocol",
    "LangIdModelFactoryProtocol",
    "LangIdModelLoaderProtocol",
    "LangIdModelProtocol",
    "LocalCorpusServiceFactoryProtocol",
    "LocalCorpusServiceProtocol",
    "StreamDownloadResponseProtocol",
    "ToIpaProtocol",
    "WikipediaRequestsGetProtocol",
    "WikipediaRequestsResponseProtocol",
    "WorkerRunnerProtocol",
    # Default implementations
    "_default_data_bank_downloader_factory",
    "_default_data_bank_uploader_factory",
    "_default_decode_optional_literal",
    "_default_decode_required_literal",
    "_default_ensure_corpus_file",
    "_default_get_env",
    "_default_guard_find_monorepo_root",
    "_default_guard_load_orchestrator",
    "_default_langid_download",
    "_default_langid_ensure_model_path",
    "_default_langid_get_fasttext_factory",
    "_default_load_langid_model",
    "_default_local_corpus_factory",
    "_default_path_exists",
    "_default_path_unlink",
    "_default_redis_for_kv",
    "_default_to_ipa",
    "_default_wikipedia_requests_get",
    # Module-level hooks
    "data_bank_client_factory",
    "data_bank_downloader_factory",
    "decode_optional_literal",
    "decode_required_literal",
    "ensure_corpus_file",
    "get_env",
    "guard_find_monorepo_root",
    "guard_load_orchestrator",
    "langid_download",
    "langid_ensure_model_path",
    "langid_get_fasttext_factory",
    "language_map",
    "load_langid_model",
    "local_corpus_service_factory",
    "path_exists",
    "path_unlink",
    "redis_factory",
    "source_map",
    "stream_culturax_hook",
    "stream_oscar_hook",
    "stream_wikipedia_xml_hook",
    "test_runner",
    "to_ipa",
    "wikipedia_requests_get",
]
