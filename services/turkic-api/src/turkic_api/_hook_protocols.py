"""The contracts every injectable dependency satisfies.

A hook is only as strict as the shape it promises, so each one is declared as a
Protocol whose signature matches the real implementation exactly — same
parameter names, same keyword-only markers, same return type. Production wires
the real thing in and a test wires a fake in, and neither can drift from the
other without the type checker saying so.

Nothing here executes. The production implementations are in
:mod:`turkic_api._hook_defaults` and the hooks themselves in
:mod:`turkic_api._test_hooks`.
"""

from __future__ import annotations

import types
from collections.abc import Callable, Generator
from pathlib import Path
from typing import BinaryIO, Protocol

import httpx
import numpy as np
from numpy.typing import NDArray
from platform_core.data_bank_client import HeadInfo
from platform_core.data_bank_protocol import FileUploadResponse
from platform_core.json_utils import JSONValue
from platform_workers.rq_harness import WorkerConfig

from turkic_api.core.models import ProcessSpec


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


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

    def __call__(self, text: str, lang: str) -> str:
        """Transliterate text to IPA.

        Args:
            text: Source text to transliterate.
            lang: ISO 639 language code controlling the transliteration table.

        Returns:
            IPA representation of ``text`` for the given language.
        """
        ...


class BuildLangScriptFilterProtocol(Protocol):
    """Protocol for build_lang_script_filter predicate factory."""

    def __call__(
        self,
        *,
        target_lang: str,
        script: str | None,
        threshold: float,
        model: LangIdModelProtocol,
    ) -> Callable[[str], bool]:
        """Build a predicate keeping lines that match language and script.

        Args:
            target_lang: ISO 639 language code to keep.
            script: Optional ISO 15924 script code that lines must also match.
            threshold: Minimum FastText probability for the language label.
            model: Language identification model.

        Returns:
            Callable taking a line and returning True if it passes the filter.
        """
        ...


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


__all__ = [
    "BuildLangScriptFilterProtocol",
    "DataBankDownloaderFactoryProtocol",
    "DataBankDownloaderProtocol",
    "DataBankUploaderFactoryProtocol",
    "DataBankUploaderProtocol",
    "DecodeOptionalLiteralProtocol",
    "DecodeRequiredLiteralProtocol",
    "EnsureCorpusProtocol",
    "LangIdDownloadProtocol",
    "LangIdEnsureModelPathProtocol",
    "LangIdFastTextFactoryProtocol",
    "LangIdModelFactoryProtocol",
    "LangIdModelLoaderProtocol",
    "LangIdModelProtocol",
    "LocalCorpusServiceFactoryProtocol",
    "LocalCorpusServiceProtocol",
    "RawStreamProtocol",
    "StreamDownloadResponseProtocol",
    "ToIpaProtocol",
    "WikipediaRequestsGetProtocol",
    "WikipediaRequestsResponseProtocol",
    "WorkerRunnerProtocol",
]
