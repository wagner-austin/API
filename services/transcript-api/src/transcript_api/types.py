from __future__ import annotations

from collections.abc import Mapping
from types import TracebackType
from typing import Protocol, runtime_checkable

from platform_core.json_utils import JSONValue
from platform_core.logging import get_logger
from platform_workers.rq_harness import RQJobLike, RQRetryLike
from typing_extensions import TypedDict

# Re-export JSONValue for local convenience
JsonValue = JSONValue

# Public JSON type for API boundaries - non-recursive, one-level deep
JsonDict = dict[str, str | int | float | bool | None | list[str | int | float | bool | None]]


class _EnqCallable(Protocol):
    def __call__(
        self,
        *args: JsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike: ...


class LoggerProtocol(Protocol):
    """Protocol for a minimal structured logger interface."""

    def debug(
        self,
        msg: str,
        *args: JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, JsonValue] | None = None,
    ) -> None: ...

    def info(
        self,
        msg: str,
        *args: JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, JsonValue] | None = None,
    ) -> None: ...

    def warning(
        self,
        msg: str,
        *args: JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, JsonValue] | None = None,
    ) -> None: ...

    def error(
        self,
        msg: str,
        *args: JsonValue,
        exc_info: bool
        | BaseException
        | tuple[type[BaseException], BaseException, TracebackType | None]
        | tuple[None, None, None]
        | None = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        extra: Mapping[str, JsonValue] | None = None,
    ) -> None: ...


class QueueProtocol(Protocol):
    """Minimal interface for a background job queue."""

    def enqueue(
        self,
        func: str | _EnqCallable,
        *args: JsonValue,
        job_timeout: int | None = None,
        result_ttl: int | None = None,
        failure_ttl: int | None = None,
        retry: RQRetryLike | None = None,
        description: str | None = None,
    ) -> RQJobLike: ...


# Default language preference list for transcript fetching
DEFAULT_TRANSCRIPT_LANGS = ["en", "en-US", "en-GB"]


class TranscriptSegment(TypedDict):
    text: str
    start: float
    duration: float


class AudioChunk(TypedDict):
    """Represents a physical audio file chunk and its time window in the source."""

    path: str
    start_seconds: float
    duration_seconds: float
    size_bytes: int


# Alias for readability in signatures
TranscriptSegmentList = list[TranscriptSegment]


class TranscriptOptions(TypedDict):
    preferred_langs: list[str]


class TranscriptResult(TypedDict):
    url: str
    video_id: str
    text: str


class RawTranscriptItem(TypedDict):
    text: str
    start: float
    duration: float


# OpenAI verbose response (canonical typed shape)
class VerboseSegmentTD(TypedDict):
    text: str
    start: float
    end: float


class VerboseResponseTD(TypedDict):
    text: str
    segments: list[VerboseSegmentTD]


# yt-dlp info structures (only the fields we consume)
class RequestedDownloadTD(TypedDict, total=False):
    filepath: str


class FormatTD(TypedDict, total=False):
    vcodec: str
    acodec: str
    abr: float
    filesize: int
    filesize_approx: int


class SubtitleTrackTD(TypedDict, total=False):
    """Single subtitle track metadata from yt-dlp."""

    ext: str
    url: str
    name: str


class YtInfoTD(TypedDict, total=False):
    id: str
    duration: float
    formats: list[FormatTD]
    requested_downloads: list[RequestedDownloadTD]
    subtitles: dict[str, list[SubtitleTrackTD]]
    automatic_captions: dict[str, list[SubtitleTrackTD]]


class SubtitleInfoTD(TypedDict, total=False):
    """Subtitle metadata returned by yt-dlp info extraction."""

    subtitles: dict[str, list[SubtitleTrackTD]]
    automatic_captions: dict[str, list[SubtitleTrackTD]]


class SubtitleResultTD(TypedDict):
    """Result of subtitle download: path to file and detected language."""

    path: str
    lang: str
    is_auto: bool


class CaptionsPayload(TypedDict):
    """Payload for captions endpoint."""

    url: str
    preferred_langs: list[str] | None


class STTPayload(TypedDict):
    """Payload for speech-to-text endpoint."""

    url: str


class TranscriptOut(TypedDict):
    """Output format for transcript endpoints."""

    url: str
    video_id: str
    text: str


@runtime_checkable
class SupportsToDictRecursive(Protocol):
    def to_dict_recursive(
        self,
    ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]: ...


@runtime_checkable
class SupportsModelDump(Protocol):
    def model_dump(
        self,
    ) -> dict[str, str | int | float | bool | None | list[dict[str, str | int | float]]]: ...


@runtime_checkable
class SupportsEstimate(Protocol):
    def estimate(self, url: str) -> tuple[int, float]: ...


@runtime_checkable
class _BinaryFileProto(Protocol):
    def read(self, size: int = -1) -> bytes: ...
    def close(self) -> None: ...


@runtime_checkable
class _TranscriptionsProto(Protocol):
    def create(
        self,
        *,
        model: str,
        file: _BinaryFileProto,
        response_format: str,
        timeout: float | None,
    ) -> SupportsToDictRecursive | SupportsModelDump: ...


@runtime_checkable
class _AudioProto(Protocol):
    @property
    def transcriptions(self) -> _TranscriptionsProto: ...


@runtime_checkable
class OpenAIClientProto(Protocol):
    @property
    def audio(self) -> _AudioProto: ...


@runtime_checkable
class _TracebackProto(Protocol):
    pass


@runtime_checkable
class YtDlpProto(Protocol):
    """Protocol for yt-dlp YoutubeDL instance.

    Note: extract_info returns raw dict (JsonValue) since that's what yt-dlp returns.
    The caller is responsible for coercing to YtInfoTD via _coerce_yt_info().
    """

    def __enter__(self) -> YtDlpProto: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: _TracebackProto | None,
    ) -> None: ...

    def extract_info(self, url: str, download: bool) -> dict[str, JsonValue]: ...
    def prepare_filename(self, info: dict[str, JsonValue]) -> str: ...


logger = get_logger(__name__)
