"""Public test utilities for platform_stt consumers.

Provides fake implementations and test helpers for services using platform_stt.

Usage:
    from platform_stt.testing import (
        FakeSTTClient,
        FakeAudioChunker,
        FakeLangIdModel,
        reset_hooks,
    )

    # Set up fakes for testing
    from platform_stt import _test_hooks
    _test_hooks.openai_client_factory = lambda **kw: FakeOpenAIClient()

    # Reset to production after test
    reset_hooks()
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import BinaryIO

import numpy as np
from numpy.typing import NDArray

from . import _test_hooks
from .types import (
    AudioChunk,
    VerboseResponse,
    VerboseSegment,
    WhisperTask,
)

# =============================================================================
# Fake STT Client
# =============================================================================


class FakeSTTClient:
    """Fake STT client for testing.

    Returns configurable responses without making real API calls.
    """

    __slots__ = ("_language", "_response", "_translate_response", "call_count")

    def __init__(
        self,
        response: VerboseResponse | None = None,
        translate_response: VerboseResponse | None = None,
    ) -> None:
        """Initialize fake client.

        Args:
            response: Response to return from transcribe().
            translate_response: Response to return from translate().
                               Falls back to response if not specified.
        """
        default = VerboseResponse(
            text="Test transcription",
            language="en",
            segments=[VerboseSegment(text="Test", start=0.0, end=1.0)],
        )
        self._response = response or default
        self._translate_response = translate_response or self._response
        self._language: str | None = None
        self.call_count = 0

    def transcribe(
        self,
        *,
        file: BinaryIO,
        language: str | None = None,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Fake transcription."""
        _ = (file, timeout)
        self._language = language
        self.call_count += 1
        return self._response

    def translate(
        self,
        *,
        file: BinaryIO,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Fake translation."""
        _ = (file, timeout)
        self.call_count += 1
        return self._translate_response

    def process(
        self,
        *,
        file: BinaryIO,
        task: WhisperTask = "transcribe",
        language: str | None = None,
        timeout: float | None = None,
    ) -> VerboseResponse:
        """Fake process."""
        if task == "translate":
            return self.translate(file=file, timeout=timeout)
        return self.transcribe(file=file, language=language, timeout=timeout)


# =============================================================================
# Fake Audio Chunker
# =============================================================================


class FakeAudioChunker:
    """Fake audio chunker for testing.

    Returns configurable chunks without running ffmpeg.
    """

    __slots__ = ("_chunks",)

    def __init__(self, chunks: list[AudioChunk] | None = None) -> None:
        """Initialize fake chunker.

        Args:
            chunks: Chunks to return. If None, returns single pass-through chunk.
        """
        self._chunks = chunks

    def chunk_audio(
        self, audio_path: str, total_duration: float, estimated_mb: float
    ) -> list[AudioChunk]:
        """Return configured chunks or single pass-through chunk."""
        if self._chunks is not None:
            return self._chunks
        # Default: return single chunk pointing to source file. A fake is
        # routinely handed a path that was never written, so absence is the
        # expected case and is asked about directly rather than caught.
        size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
        return [
            AudioChunk(
                path=audio_path,
                start_seconds=0.0,
                duration_seconds=total_duration,
                size_bytes=size,
            )
        ]


# =============================================================================
# Fake Language ID Model
# =============================================================================


class FakeLangIdModel:
    """Fake language identification model for testing.

    Returns configurable language detection results.
    """

    __slots__ = ("_confidence", "_label")

    def __init__(self, label: str = "__label__en", confidence: float = 0.99) -> None:
        """Initialize fake model.

        Args:
            label: Label to return from predict().
            confidence: Confidence score to return.
        """
        self._label = label
        self._confidence = confidence

    def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
        """Return configured prediction."""
        del text, k  # unused
        labels: tuple[str, ...] = (self._label,)
        # Use zeros + assignment to avoid np.array's Any return type
        probs: NDArray[np.float64] = np.zeros(1, dtype=np.float64)
        probs[0] = self._confidence
        return labels, probs


# =============================================================================
# Fake Subprocess Result
# =============================================================================


class FakeSubprocessResult:
    """Fake subprocess result for testing ffmpeg operations."""

    __slots__ = ("returncode", "stderr", "stdout")

    def __init__(
        self,
        returncode: int = 0,
        stdout: str | bytes | None = None,
        stderr: str | bytes | None = None,
    ) -> None:
        """Initialize fake result."""
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeSubprocessRun:
    """Fake subprocess.run for testing.

    Records calls and returns configurable results.
    """

    __slots__ = ("_result", "calls")

    def __init__(self, result: FakeSubprocessResult | None = None) -> None:
        """Initialize fake subprocess runner.

        Args:
            result: Result to return from calls.
        """
        self._result = result or FakeSubprocessResult()
        self.calls: list[list[str]] = []

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
    ) -> FakeSubprocessResult:
        """Record call and return configured result."""
        _ = (capture_output, timeout, text, input, cwd, env)
        self.calls.append(args)
        if check and self._result.returncode != 0:
            import subprocess

            raise subprocess.CalledProcessError(
                self._result.returncode, args, self._result.stdout, self._result.stderr
            )
        return self._result


# =============================================================================
# Fake File Writer
# =============================================================================


class FakeWriteTextFile:
    """Fake file writer for testing.

    Records all write operations instead of writing to disk.
    """

    __slots__ = ("writes",)

    def __init__(self) -> None:
        """Initialize fake writer with empty writes list."""
        self.writes: list[tuple[Path, str]] = []

    def __call__(self, path: Path, content: str) -> None:
        """Record write operation.

        Args:
            path: Destination file path.
            content: Content that would be written.
        """
        self.writes.append((path, content))


# =============================================================================
# Hook Management
# =============================================================================


def set_production_hooks() -> None:
    """Set all hooks to production implementations."""
    _test_hooks.subprocess_run = _test_hooks._default_subprocess_run
    _test_hooks.os_stat = _test_hooks._default_os_stat
    _test_hooks.os_path_getsize = _test_hooks._default_os_path_getsize
    _test_hooks.os_remove = _test_hooks._default_os_remove
    _test_hooks.mkdtemp = _test_hooks._default_mkdtemp
    _test_hooks.ffmpeg_available = _test_hooks._default_ffmpeg_available
    _test_hooks.openai_client_factory = _test_hooks._default_openai_client_factory
    _test_hooks.audio_chunker_factory = _test_hooks._default_audio_chunker_factory
    _test_hooks.langid_download = _test_hooks._default_langid_download
    _test_hooks.langid_ensure_model_path = _test_hooks._default_langid_ensure_model_path
    _test_hooks.langid_get_fasttext_factory = _test_hooks._default_langid_get_fasttext_factory
    _test_hooks.write_text_file = _test_hooks._default_write_text_file


def reset_hooks() -> None:
    """Reset all hooks to production implementations."""
    set_production_hooks()


def make_fake_subprocess_run(
    result: FakeSubprocessResult | None = None,
) -> FakeSubprocessRun:
    """Create a fake subprocess runner and install it as the hook.

    Args:
        result: Result to return from subprocess calls.

    Returns:
        The installed FakeSubprocessRun instance.
    """
    fake = FakeSubprocessRun(result)
    _test_hooks.subprocess_run = fake
    return fake


def make_fake_audio_chunker_factory(
    chunks: list[AudioChunk] | None = None,
) -> _test_hooks.AudioChunkerFactoryProtocol:
    """Create a fake audio chunker factory.

    Args:
        chunks: Chunks to return from the chunker.

    Returns:
        Factory function that creates FakeAudioChunker.
    """

    def factory(
        *,
        target_chunk_mb: float,
        max_chunk_duration_seconds: float,
        silence_threshold_db: float,
        silence_duration_seconds: float,
    ) -> _test_hooks.AudioChunkerProtocol:
        del target_chunk_mb, max_chunk_duration_seconds
        del silence_threshold_db, silence_duration_seconds
        return FakeAudioChunker(chunks)

    return factory


def make_fake_langid_model_factory(
    label: str = "__label__en",
    confidence: float = 0.99,
) -> _test_hooks.LangIdModelFactoryProtocol:
    """Create a fake language ID model factory.

    Args:
        label: Label to return from predictions.
        confidence: Confidence score to return.

    Returns:
        Factory function that creates FakeLangIdModel.
    """

    def factory(*, model_path: str) -> _test_hooks.LangIdModelProtocol:
        del model_path  # unused
        return FakeLangIdModel(label=label, confidence=confidence)

    return factory


__all__ = [
    "FakeAudioChunker",
    "FakeLangIdModel",
    "FakeSTTClient",
    "FakeSubprocessResult",
    "FakeSubprocessRun",
    "FakeWriteTextFile",
    "make_fake_audio_chunker_factory",
    "make_fake_langid_model_factory",
    "make_fake_subprocess_run",
    "reset_hooks",
    "set_production_hooks",
]
