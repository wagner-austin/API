"""Internal test hooks for platform_stt - allows injecting test dependencies.

This module provides dependency injection hooks following the pattern:
- Production code sets hooks to real implementations at startup
- Tests set hooks to fakes before running

Usage in production:
    # At startup, hooks are already set to defaults (production implementations)

Usage in tests:
    from platform_stt import _test_hooks
    _test_hooks.subprocess_run = fake_subprocess_run
    # ... run test ...
    # Reset after test if needed
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .types import (
    AudioChunk,
    BinaryFileProtocol,
    RawVerboseDict,
    VerboseResponse,
    WhisperTask,
)

# =============================================================================
# Subprocess Protocol
# =============================================================================


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
    ) -> SubprocessRunResult:
        """Run subprocess with given arguments."""
        ...


# =============================================================================
# OpenAI Client Protocols
# =============================================================================


class TranscriptionsCreateProtocol(Protocol):
    """Protocol for OpenAI transcriptions.create method."""

    def create(
        self,
        *,
        model: str,
        file: BinaryFileProtocol,
        response_format: str,
        language: str | None = None,
        timeout: float | None = None,
    ) -> RawVerboseDict:
        """Create transcription from audio file."""
        ...


class TranslationsCreateProtocol(Protocol):
    """Protocol for OpenAI translations.create method."""

    def create(
        self,
        *,
        model: str,
        file: BinaryFileProtocol,
        response_format: str,
        timeout: float | None = None,
    ) -> RawVerboseDict:
        """Create translation from audio file."""
        ...


class AudioNamespaceProtocol(Protocol):
    """Protocol for OpenAI audio namespace."""

    @property
    def transcriptions(self) -> TranscriptionsCreateProtocol:
        """Get transcriptions interface."""
        ...

    @property
    def translations(self) -> TranslationsCreateProtocol:
        """Get translations interface."""
        ...


class OpenAIClientProtocol(Protocol):
    """Protocol for OpenAI client."""

    @property
    def audio(self) -> AudioNamespaceProtocol:
        """Get audio namespace."""
        ...


class OpenAIClientFactoryProtocol(Protocol):
    """Protocol for OpenAI client factory."""

    def __call__(self, *, api_key: str, timeout: float, max_retries: int) -> OpenAIClientProtocol:
        """Create OpenAI client with given configuration."""
        ...


# =============================================================================
# STT Client Protocol
# =============================================================================


class STTClientProtocol(Protocol):
    """Protocol for STT client (e.g., OpenAI Whisper)."""

    def transcribe(
        self,
        *,
        file: BinaryIO,
        language: str | None = None,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Transcribe audio file to text in source language."""
        ...

    def translate(
        self,
        *,
        file: BinaryIO,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Translate audio file to English text."""
        ...


# =============================================================================
# Audio Chunker Protocol
# =============================================================================


class AudioChunkerProtocol(Protocol):
    """Protocol for audio chunker."""

    def chunk_audio(
        self, audio_path: str, total_duration: float, estimated_mb: float
    ) -> list[AudioChunk]:
        """Split audio file into chunks for processing."""
        ...


class AudioChunkerFactoryProtocol(Protocol):
    """Protocol for audio chunker factory."""

    def __call__(
        self,
        *,
        target_chunk_mb: float,
        max_chunk_duration_seconds: float,
        silence_threshold_db: float,
        silence_duration_seconds: float,
    ) -> AudioChunkerProtocol:
        """Create audio chunker with given configuration."""
        ...


# =============================================================================
# Transcriber Protocol
# =============================================================================


class TranscribeFnProtocol(Protocol):
    """Protocol for transcription function used by parallel transcriber."""

    def __call__(
        self,
        *,
        model: str,
        file: BinaryIO,
        response_format: Literal["verbose_json"],
        language: str | None = None,
        task: WhisperTask = "transcribe",
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Transcribe or translate audio file."""
        ...


# =============================================================================
# Language Detection Protocol
# =============================================================================


class LangIdModelProtocol(Protocol):
    """Protocol for language identification model."""

    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        """Predict language labels and probabilities for the given text."""
        ...


class LangIdModelFactoryProtocol(Protocol):
    """Protocol for language identification model factory."""

    def __call__(self, *, model_path: str) -> LangIdModelProtocol:
        """Load language identification model from path."""
        ...


class LangIdDownloadProtocol(Protocol):
    """Protocol for downloading language identification model."""

    def __call__(self, url: str, dest: Path) -> None:
        """Download model from URL to destination path."""
        ...


# =============================================================================
# File I/O Protocols
# =============================================================================


class WriteTextFileProtocol(Protocol):
    """Protocol for writing text to a file."""

    def __call__(self, path: Path, content: str) -> None:
        """Write text content to file path.

        Args:
            path: Destination file path.
            content: Text content to write.

        Raises:
            OSError: If file cannot be written.
        """
        ...


# =============================================================================
# Default Implementations
# =============================================================================


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


def _run_subprocess_bytes(
    args: list[str],
    capture_output: bool,
    check: bool,
    timeout: float | None,
    input_data: bytes | None,
    cwd: str | None,
    env: dict[str, str] | None,
) -> _SubprocessRunResultImpl:
    """Run subprocess and return bytes output."""
    stdout_pipe = subprocess.PIPE if capture_output else None
    stderr_pipe = subprocess.PIPE if capture_output else None
    stdin_pipe = subprocess.PIPE if input_data is not None else None

    proc: subprocess.Popen[bytes] = subprocess.Popen(
        args,
        stdout=stdout_pipe,
        stderr=stderr_pipe,
        stdin=stdin_pipe,
        cwd=cwd,
        env=env,
    )
    stdout_bytes, stderr_bytes = proc.communicate(input=input_data, timeout=timeout)
    returncode: int = proc.returncode

    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, args, stdout_bytes, stderr_bytes)

    return _SubprocessRunResultImpl(returncode, stdout_bytes, stderr_bytes)


def _run_subprocess_text(
    args: list[str],
    capture_output: bool,
    check: bool,
    timeout: float | None,
    input_data: str | None,
    cwd: str | None,
    env: dict[str, str] | None,
) -> _SubprocessRunResultImpl:
    """Run subprocess and return text output."""
    stdout_pipe = subprocess.PIPE if capture_output else None
    stderr_pipe = subprocess.PIPE if capture_output else None
    stdin_pipe = subprocess.PIPE if input_data is not None else None

    proc: subprocess.Popen[str] = subprocess.Popen(
        args,
        stdout=stdout_pipe,
        stderr=stderr_pipe,
        stdin=stdin_pipe,
        text=True,
        cwd=cwd,
        env=env,
    )
    stdout_str, stderr_str = proc.communicate(input=input_data, timeout=timeout)
    returncode: int = proc.returncode

    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, args, stdout_str, stderr_str)

    return _SubprocessRunResultImpl(returncode, stdout_str, stderr_str)


def _default_subprocess_run(
    args: list[str],
    *,
    capture_output: bool = False,
    check: bool = False,
    timeout: float | None = None,
    text: bool = False,
    input: bytes | str | None = None,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> SubprocessRunResult:
    """Production implementation - uses typed Popen to avoid Any types."""
    if text:
        input_str: str | None = input if isinstance(input, str) else None
        return _run_subprocess_text(args, capture_output, check, timeout, input_str, cwd, env)
    input_bytes: bytes | None = None
    if isinstance(input, str):
        input_bytes = input.encode()
    elif isinstance(input, bytes):
        input_bytes = input
    return _run_subprocess_bytes(args, capture_output, check, timeout, input_bytes, cwd, env)


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


def _default_ffmpeg_available() -> bool:
    """Production implementation - checks if ffmpeg/ffprobe are available."""
    from shutil import which

    ffmpeg = which("ffmpeg")
    ffprobe = which("ffprobe")
    return bool(ffmpeg and ffprobe)


def _default_openai_client_factory(
    *, api_key: str, timeout: float, max_retries: int
) -> OpenAIClientProtocol:
    """Production implementation - creates real OpenAI client."""
    mod = __import__("openai")
    client: OpenAIClientProtocol = mod.OpenAI(
        api_key=api_key, timeout=timeout, max_retries=max_retries
    )
    return client


def _default_audio_chunker_factory(
    *,
    target_chunk_mb: float,
    max_chunk_duration_seconds: float,
    silence_threshold_db: float,
    silence_duration_seconds: float,
) -> AudioChunkerProtocol:
    """Production implementation - creates real AudioChunker."""
    from .chunker import AudioChunker

    chunker: AudioChunkerProtocol = AudioChunker(
        target_chunk_mb=target_chunk_mb,
        max_chunk_duration_seconds=max_chunk_duration_seconds,
        silence_threshold_db=silence_threshold_db,
        silence_duration_seconds=silence_duration_seconds,
    )
    return chunker


def _default_langid_download(url: str, dest: Path) -> None:
    """Production implementation - downloads model file from URL."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, str(dest))


def _default_langid_ensure_model_path(data_dir: str, prefer_218e: bool = True) -> Path:
    """Production implementation - ensures model file exists."""
    from .langid import ensure_model_path as _ensure

    return _ensure(data_dir, prefer_218e=prefer_218e)


def _default_langid_get_fasttext_factory() -> LangIdModelFactoryProtocol:
    """Production implementation - gets FastText model factory."""
    mod = __import__("fasttext.FastText", fromlist=["_FastText"])
    factory: LangIdModelFactoryProtocol = mod._FastText
    return factory


def _default_write_text_file(path: Path, content: str) -> None:
    """Production implementation - writes text to file with UTF-8 encoding.

    Args:
        path: Destination file path.
        content: Text content to write.

    Raises:
        OSError: If file cannot be written.
    """
    path.write_text(content, encoding="utf-8")


# =============================================================================
# Module-level Hooks
# =============================================================================

# Hook for subprocess.run
subprocess_run: SubprocessRunProtocol = _default_subprocess_run

# Hook for os.stat
os_stat: Callable[[str], os.stat_result] = _default_os_stat

# Hook for os.path.getsize
os_path_getsize: Callable[[str], int] = _default_os_path_getsize

# Hook for os.remove
os_remove: Callable[[str], None] = _default_os_remove

# Hook for tempfile.mkdtemp
mkdtemp: Callable[[str | None, str | None], str] = _default_mkdtemp

# Hook for ffmpeg availability check
ffmpeg_available: Callable[[], bool] = _default_ffmpeg_available

# Hook for OpenAI client factory
openai_client_factory: OpenAIClientFactoryProtocol = _default_openai_client_factory

# Hook for AudioChunker factory
audio_chunker_factory: AudioChunkerFactoryProtocol = _default_audio_chunker_factory

# Hook for language ID model download
langid_download: LangIdDownloadProtocol = _default_langid_download

# Hook for language ID model path resolution
langid_ensure_model_path: Callable[[str, bool], Path] = _default_langid_ensure_model_path

# Hook for language ID FastText factory
langid_get_fasttext_factory: Callable[[], LangIdModelFactoryProtocol] = (
    _default_langid_get_fasttext_factory
)

# Hook for writing text files
write_text_file: WriteTextFileProtocol = _default_write_text_file


__all__ = [
    "AudioChunkerFactoryProtocol",
    "AudioChunkerProtocol",
    "LangIdDownloadProtocol",
    "LangIdModelFactoryProtocol",
    "LangIdModelProtocol",
    "OpenAIClientFactoryProtocol",
    "OpenAIClientProtocol",
    "STTClientProtocol",
    "SubprocessRunProtocol",
    "SubprocessRunResult",
    "TranscribeFnProtocol",
    "WriteTextFileProtocol",
    "_default_audio_chunker_factory",
    "_default_ffmpeg_available",
    "_default_langid_download",
    "_default_langid_ensure_model_path",
    "_default_langid_get_fasttext_factory",
    "_default_mkdtemp",
    "_default_openai_client_factory",
    "_default_os_path_getsize",
    "_default_os_remove",
    "_default_os_stat",
    "_default_subprocess_run",
    "_default_write_text_file",
    "audio_chunker_factory",
    "ffmpeg_available",
    "langid_download",
    "langid_ensure_model_path",
    "langid_get_fasttext_factory",
    "mkdtemp",
    "openai_client_factory",
    "os_path_getsize",
    "os_remove",
    "os_stat",
    "subprocess_run",
    "write_text_file",
]
