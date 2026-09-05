"""Test hooks for transcript-api - allows injecting test dependencies."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from typing import BinaryIO, Protocol

from platform_core.json_utils import JSONValue
from platform_stt import VerboseResponse
from platform_stt import _test_hooks as stt_hooks
from platform_workers.redis import RedisStrProto, redis_for_kv
from platform_workers.rq_harness import WorkerConfig, run_rq_worker

from .types import (
    AudioChunk,
    OpenAIClientProto,
    SubtitleResultTD,
    TranscriptOptions,
    TranscriptSegment,
    YtDlpProto,
    YtInfoTD,
)


class WorkerRunnerProtocol(Protocol):
    """Protocol for worker runner function."""

    def __call__(self, config: WorkerConfig) -> None:
        """Run the worker with the given config."""
        ...


class YTFetchedProto(Protocol):
    """Protocol for a fetched transcript (youtube_transcript_api >= 1.x).

    The 1.x library returns a FetchedTranscript object of snippet objects;
    ``to_raw_data`` is its own conversion back to the plain dicts the
    adapter validates and coerces to RawTranscriptItem.
    """

    def to_raw_data(self) -> list[dict[str, JSONValue]]: ...


class YTTranscriptResourceProto(Protocol):
    """Protocol for a YouTube transcript resource."""

    def fetch(self) -> YTFetchedProto: ...


class YTListingProto(Protocol):
    """Protocol for YouTube transcript listing.

    The 1.x library's ``find_transcript`` raises ``NoTranscriptFound``
    rather than returning None; the adapter maps that exception, so the
    protocol return stays non-optional.
    """

    def find_transcript(self, languages: list[str]) -> YTTranscriptResourceProto: ...
    def translate(self, language: str) -> YTTranscriptResourceProto: ...


class YTApiProto(Protocol):
    """Protocol for the YouTube Transcript API (youtube_transcript_api >= 1.x).

    The 1.x library is instance-based: ``YouTubeTranscriptApi()`` with
    ``fetch``/``list`` methods, replacing the 0.x static
    ``get_transcript``/``list_transcripts``. ``fetch`` returns an object
    whose ``to_raw_data`` yields the plain dicts the adapter validates and
    coerces to RawTranscriptItem.
    """

    def fetch(self, video_id: str, languages: list[str]) -> YTFetchedProto: ...

    def list(self, video_id: str) -> YTListingProto: ...


class YTExceptionsProto(Protocol):
    """Protocol for YouTube transcript exceptions tuple."""

    NoTranscriptFound: type[Exception]
    TranscriptsDisabled: type[Exception]
    VideoUnavailable: type[Exception]


class AudioChunkerProto(Protocol):
    """Protocol for audio chunker."""

    def chunk_audio(
        self, audio_path: str, total_duration: float, estimated_mb: float
    ) -> list[AudioChunk]: ...


class OpenAIClientFactoryProto(Protocol):
    """Protocol for OpenAI client factory."""

    def __call__(self, *, api_key: str, timeout: float, max_retries: int) -> OpenAIClientProto: ...


class AudioChunkerFactoryProto(Protocol):
    """Protocol for audio chunker factory."""

    def __call__(
        self,
        *,
        target_chunk_mb: float,
        max_chunk_duration_seconds: float,
        silence_threshold_db: float,
        silence_duration_seconds: float,
    ) -> AudioChunkerProto: ...


class STTProviderFactoryProto(Protocol):
    """Protocol for STT provider factory."""

    def __call__(
        self,
        *,
        stt_client: STTClientProto,
        probe_client: ProbeDownloadClientProto,
        max_video_seconds: int,
        max_file_mb: int,
        enable_chunking: bool,
        chunk_threshold_mb: float,
        target_chunk_mb: float,
        max_chunk_duration: float,
        max_concurrent_chunks: int,
        silence_threshold_db: float,
        silence_duration: float,
        stt_rtf: float,
        dl_mib_per_sec: float,
        cookies_text: str | None,
    ) -> STTProviderProto: ...


class STTClientProto(Protocol):
    """Protocol for STT client (e.g., OpenAI Whisper)."""

    def transcribe_verbose(
        self,
        *,
        file: BinaryIO,
        timeout: float | None,
    ) -> VerboseResponse: ...


class ProbeDownloadClientProto(Protocol):
    """Protocol for probe/download client (e.g., yt-dlp)."""

    def probe(self, url: str) -> YtInfoTD: ...

    def download_audio(self, url: str, *, cookies_path: str | None) -> str: ...

    def download_subtitles(
        self,
        url: str,
        *,
        cookies_path: str | None,
        preferred_langs: list[str],
    ) -> SubtitleResultTD | None: ...


class STTProviderProto(Protocol):
    """Protocol for STT transcript provider."""

    def fetch(self, video_id: str, opts: TranscriptOptions) -> list[TranscriptSegment]: ...


class YtDlpFactoryProto(Protocol):
    """Protocol for yt-dlp factory function."""

    def __call__(self, opts: dict[str, JSONValue]) -> YtDlpProto: ...


class SubprocessRunResult(Protocol):
    """Protocol for subprocess.run result."""

    returncode: int
    stdout: bytes | str | None
    stderr: bytes | str | None


class SubprocessRunProtocol(Protocol):
    """Protocol for subprocess.run function."""

    def __call__(
        self,
        args: list[str],
        *,
        capture_output: bool = False,
        check: bool = False,
        timeout: float | None = None,
        text: bool = False,
        input: bytes | str | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> SubprocessRunResult: ...


# =========================================================================
# Default implementations
# =========================================================================


def _default_redis_for_kv(url: str) -> RedisStrProto:
    """Production implementation - creates real Redis client."""
    return redis_for_kv(url)


def _default_os_stat(path: str) -> os.stat_result:
    """Production implementation - calls os.stat."""
    return os.stat(path)


def _default_os_path_getsize(path: str) -> int:
    """Production implementation - calls os.path.getsize."""
    return os.path.getsize(path)


def _default_os_remove(path: str) -> None:
    """Production implementation - calls os.remove."""
    os.remove(path)


def _default_mkdtemp(prefix: str | None = None, dir: str | None = None) -> str:
    """Production implementation - calls tempfile.mkdtemp."""
    return tempfile.mkdtemp(prefix=prefix, dir=dir)


class _SubprocessRunResultImpl:
    """Concrete implementation of SubprocessRunResult from subprocess.run output."""

    __slots__ = ("returncode", "stderr", "stdout")

    def __init__(
        self,
        returncode: int,
        stdout: bytes | str | None,
        stderr: bytes | str | None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _default_openai_client_factory(
    *, api_key: str, timeout: float, max_retries: int
) -> OpenAIClientProto:
    """Production implementation - creates real OpenAI client."""
    mod = __import__("openai")
    client: OpenAIClientProto = mod.OpenAI(
        api_key=api_key, timeout=timeout, max_retries=max_retries
    )
    return client


def _default_yt_api_factory() -> YTApiProto:
    """Production implementation - creates real YouTube API."""
    mod = __import__("youtube_transcript_api")
    api: YTApiProto = mod.YouTubeTranscriptApi()
    return api


def _default_yt_exceptions_factory() -> tuple[type[Exception], type[Exception], type[Exception]]:
    """Production implementation - returns real YouTube exception classes."""
    mod = __import__("youtube_transcript_api")
    # Assign to typed variables to satisfy mypy
    no_transcript_found: type[Exception] = mod.NoTranscriptFound
    transcripts_disabled: type[Exception] = mod.TranscriptsDisabled
    video_unavailable: type[Exception] = mod.VideoUnavailable
    return (no_transcript_found, transcripts_disabled, video_unavailable)


def _default_yt_translate_exceptions_factory() -> tuple[type[Exception], type[Exception]]:
    """Production implementation - returns the real translate exception classes.

    The 1.x library's ``Transcript.translate`` raises these two rather than
    the RuntimeError the 0.x adapter mapped.
    """
    mod = __import__("youtube_transcript_api")
    not_translatable: type[Exception] = mod.NotTranslatable
    translation_unavailable: type[Exception] = mod.TranslationLanguageNotAvailable
    return (not_translatable, translation_unavailable)


def _default_yt_dlp_factory(opts: dict[str, JSONValue]) -> YtDlpProto:
    """Production implementation - creates real yt-dlp client."""
    mod = __import__("yt_dlp")
    client: YtDlpProto = mod.YoutubeDL(opts)
    return client


def _default_audio_chunker_factory(
    *,
    target_chunk_mb: float,
    max_chunk_duration_seconds: float,
    silence_threshold_db: float,
    silence_duration_seconds: float,
) -> AudioChunkerProto:
    """Production implementation - creates real AudioChunker."""
    from platform_stt.chunker import AudioChunker

    chunker: AudioChunkerProto = AudioChunker(
        target_chunk_mb=target_chunk_mb,
        max_chunk_duration_seconds=max_chunk_duration_seconds,
        silence_threshold_db=silence_threshold_db,
        silence_duration_seconds=silence_duration_seconds,
    )
    return chunker


def _default_stt_client_builder(api_key: str) -> STTClientProto:
    """Production implementation - creates real OpenAI STT client."""
    from .adapters.openai_client import OpenAISttClient

    client: STTClientProto = OpenAISttClient(api_key=api_key)
    return client


def _default_probe_client_builder() -> ProbeDownloadClientProto:
    """Production implementation - creates real yt-dlp adapter."""
    from .adapters.yt_dlp_client import YtDlpAdapter

    adapter: ProbeDownloadClientProto = YtDlpAdapter()
    return adapter


def _default_stt_provider_factory(
    *,
    stt_client: STTClientProto,
    probe_client: ProbeDownloadClientProto,
    max_video_seconds: int,
    max_file_mb: int,
    enable_chunking: bool,
    chunk_threshold_mb: float,
    target_chunk_mb: float,
    max_chunk_duration: float,
    max_concurrent_chunks: int,
    silence_threshold_db: float,
    silence_duration: float,
    stt_rtf: float,
    dl_mib_per_sec: float,
    cookies_text: str | None,
) -> STTProviderProto:
    """Production implementation - creates real STTTranscriptProvider."""
    from .stt_provider import STTTranscriptProvider

    provider: STTProviderProto = STTTranscriptProvider(
        stt_client=stt_client,
        probe_client=probe_client,
        max_video_seconds=max_video_seconds,
        max_file_mb=max_file_mb,
        enable_chunking=enable_chunking,
        chunk_threshold_mb=chunk_threshold_mb,
        target_chunk_mb=target_chunk_mb,
        max_chunk_duration=max_chunk_duration,
        max_concurrent_chunks=max_concurrent_chunks,
        silence_threshold_db=silence_threshold_db,
        silence_duration=silence_duration,
        stt_rtf=stt_rtf,
        dl_mib_per_sec=dl_mib_per_sec,
        cookies_text=cookies_text,
    )
    return provider


# =========================================================================
# Module-level hooks
# =========================================================================

# Hook for the worker runner, bound to the real RQ runner. Tests rebind
# it BEFORE running worker_entry as __main__; because this is a separate
# module, the binding persists across runpy.run_module.
worker_runner: WorkerRunnerProtocol = run_rq_worker

# Hook for Redis client factory
redis_factory: Callable[[str], RedisStrProto] = _default_redis_for_kv

# Hook for os.stat
os_stat: Callable[[str], os.stat_result] = _default_os_stat

# Hook for os.path.getsize
os_path_getsize: Callable[[str], int] = _default_os_path_getsize

# Hook for tempfile.mkdtemp
mkdtemp: Callable[[str | None, str | None], str] = _default_mkdtemp

# Hook for os.remove
os_remove: Callable[[str], None] = _default_os_remove

# Hook for subprocess.run - typed loosely to allow various return types
subprocess_run: SubprocessRunProtocol = stt_hooks._default_subprocess_run

# Hook for OpenAI client factory
openai_client_factory: OpenAIClientFactoryProto = _default_openai_client_factory

# Hook for YouTube Transcript API factory
yt_api_factory: Callable[[], YTApiProto] = _default_yt_api_factory

# Hook for YouTube exception classes factory
yt_exceptions_factory: Callable[[], tuple[type[Exception], type[Exception], type[Exception]]] = (
    _default_yt_exceptions_factory
)

# Hook for YouTube translate exception classes factory
yt_translate_exceptions_factory: Callable[[], tuple[type[Exception], type[Exception]]] = (
    _default_yt_translate_exceptions_factory
)

# Hook for yt-dlp factory
yt_dlp_factory: YtDlpFactoryProto = _default_yt_dlp_factory

# Hook for AudioChunker factory
audio_chunker_factory: AudioChunkerFactoryProto = _default_audio_chunker_factory

# Hook for STT client builder
stt_client_builder: Callable[[str], STTClientProto] = _default_stt_client_builder

# Hook for probe client builder
probe_client_builder: Callable[[], ProbeDownloadClientProto] = _default_probe_client_builder

# Hook for STT provider factory
stt_provider_factory: STTProviderFactoryProto = _default_stt_provider_factory


# Hook for ffmpeg availability check
ffmpeg_available: Callable[[], bool] = stt_hooks._default_ffmpeg_available


__all__ = [
    "AudioChunkerFactoryProto",
    "AudioChunkerProto",
    "OpenAIClientFactoryProto",
    "ProbeDownloadClientProto",
    "STTClientProto",
    "STTProviderFactoryProto",
    "STTProviderProto",
    "SubprocessRunProtocol",
    "SubprocessRunResult",
    "WorkerRunnerProtocol",
    "YTApiProto",
    "YTExceptionsProto",
    "YTFetchedProto",
    "YTListingProto",
    "YTTranscriptResourceProto",
    "YtDlpFactoryProto",
    "_default_audio_chunker_factory",
    "_default_mkdtemp",
    "_default_openai_client_factory",
    "_default_os_path_getsize",
    "_default_os_remove",
    "_default_os_stat",
    "_default_probe_client_builder",
    "_default_redis_for_kv",
    "_default_stt_client_builder",
    "_default_stt_provider_factory",
    "_default_yt_api_factory",
    "_default_yt_dlp_factory",
    "_default_yt_exceptions_factory",
    "_default_yt_translate_exceptions_factory",
    "audio_chunker_factory",
    "ffmpeg_available",
    "mkdtemp",
    "openai_client_factory",
    "os_path_getsize",
    "os_remove",
    "os_stat",
    "probe_client_builder",
    "redis_factory",
    "stt_client_builder",
    "stt_provider_factory",
    "subprocess_run",
    "worker_runner",
    "yt_api_factory",
    "yt_dlp_factory",
    "yt_exceptions_factory",
    "yt_translate_exceptions_factory",
]
