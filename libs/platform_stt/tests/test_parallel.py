"""Tests for platform_stt.parallel module."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Literal

import pytest

from platform_stt.parallel import ParallelTranscriber
from platform_stt.types import AudioChunk, VerboseResponse, VerboseSegment, WhisperTask


class FakeTranscribeFn:
    """Fake transcription function implementing TranscribeFnProtocol."""

    __slots__ = ("_call_count", "_calls", "_responses")

    def __init__(self, responses: list[VerboseResponse] | None = None) -> None:
        """Initialize with optional list of responses to return sequentially."""
        default = VerboseResponse(
            text="Test",
            segments=[VerboseSegment(text="Test", start=0.0, end=1.0)],
        )
        self._responses = responses or [default]
        self._call_count = 0
        self._calls: list[dict[str, str | float | None]] = []

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
        """Record call and return next response."""
        self._calls.append(
            {
                "model": model,
                "response_format": response_format,
                "language": language,
                "task": task,
                "timeout": timeout,
            }
        )
        idx = self._call_count % len(self._responses)
        self._call_count += 1
        return self._responses[idx]


class FailingThenSuccessFn:
    """Fake that fails first then succeeds."""

    __slots__ = ("call_count",)

    def __init__(self) -> None:
        self.call_count = 0

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
        """Fail first call, succeed after."""
        self.call_count += 1
        if self.call_count < 2:
            raise OSError("Transient error")
        return VerboseResponse(
            text="Success",
            segments=[VerboseSegment(text="Success", start=0.0, end=1.0)],
        )


class AlwaysFailsFn:
    """Fake that always raises TimeoutError."""

    __slots__ = ()

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
        """Always raise TimeoutError."""
        raise TimeoutError("Always fails")


class TestParallelTranscriber:
    """Tests for ParallelTranscriber class."""

    def test_init_defaults(self) -> None:
        """Initialize with default values."""
        fake_fn = FakeTranscribeFn()
        transcriber = ParallelTranscriber(transcribe=fake_fn)
        assert transcriber._max_concurrent == 3
        assert transcriber._max_retries == 2
        assert transcriber._timeout == 900.0
        assert transcriber._language is None
        assert transcriber._task == "transcribe"

    def test_init_custom_values(self) -> None:
        """Initialize with custom values."""
        fake_fn = FakeTranscribeFn()
        transcriber = ParallelTranscriber(
            transcribe=fake_fn,
            max_concurrent=5,
            max_retries=3,
            timeout_seconds=600.0,
            language="vi",
            task="translate",
        )
        assert transcriber._max_concurrent == 5
        assert transcriber._max_retries == 3
        assert transcriber._timeout == 600.0
        assert transcriber._language == "vi"
        assert transcriber._task == "translate"

    def test_init_clamps_negative_values(self) -> None:
        """Clamp negative concurrent and retry values to minimums."""
        fake_fn = FakeTranscribeFn()
        transcriber = ParallelTranscriber(
            transcribe=fake_fn,
            max_concurrent=-1,
            max_retries=-1,
        )
        assert transcriber._max_concurrent == 1
        assert transcriber._max_retries == 0

    def test_transcribe_chunks_empty_list(self) -> None:
        """Handle empty chunk list."""
        fake_fn = FakeTranscribeFn()
        transcriber = ParallelTranscriber(transcribe=fake_fn)
        result = transcriber.transcribe_chunks([])
        assert result == []

    def test_transcribe_chunks_single_chunk(self, tmp_path: Path) -> None:
        """Transcribe a single chunk."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio data")

        response = VerboseResponse(
            text="Hello world",
            segments=[
                VerboseSegment(text="Hello", start=0.0, end=0.5),
                VerboseSegment(text="world", start=0.5, end=1.0),
            ],
        )
        fake_fn = FakeTranscribeFn([response])
        transcriber = ParallelTranscriber(transcribe=fake_fn)

        chunks = [
            AudioChunk(
                path=str(audio_file),
                start_seconds=0.0,
                duration_seconds=10.0,
                size_bytes=100,
            )
        ]
        result = transcriber.transcribe_chunks(chunks)

        assert len(result) == 1
        assert len(result[0]) == 2
        assert result[0][0]["text"] == "Hello"
        assert result[0][1]["text"] == "world"

    def test_transcribe_chunks_multiple_chunks(self, tmp_path: Path) -> None:
        """Transcribe multiple chunks in parallel."""
        audio_file1 = tmp_path / "chunk1.mp3"
        audio_file1.write_bytes(b"fake audio 1")
        audio_file2 = tmp_path / "chunk2.mp3"
        audio_file2.write_bytes(b"fake audio 2")

        responses = [
            VerboseResponse(
                text="First",
                segments=[VerboseSegment(text="First", start=0.0, end=1.0)],
            ),
            VerboseResponse(
                text="Second",
                segments=[VerboseSegment(text="Second", start=0.0, end=1.0)],
            ),
        ]
        fake_fn = FakeTranscribeFn(responses)
        transcriber = ParallelTranscriber(transcribe=fake_fn)

        chunks = [
            AudioChunk(
                path=str(audio_file1),
                start_seconds=0.0,
                duration_seconds=10.0,
                size_bytes=100,
            ),
            AudioChunk(
                path=str(audio_file2),
                start_seconds=10.0,
                duration_seconds=10.0,
                size_bytes=100,
            ),
        ]
        result = transcriber.transcribe_chunks(chunks)

        assert len(result) == 2
        # Results maintain chunk order
        assert len(fake_fn._calls) == 2

    def test_transcribe_chunks_passes_language(self, tmp_path: Path) -> None:
        """Verify language parameter is passed through."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        fake_fn = FakeTranscribeFn()
        transcriber = ParallelTranscriber(
            transcribe=fake_fn,
            language="vi",
        )

        chunks = [
            AudioChunk(
                path=str(audio_file),
                start_seconds=0.0,
                duration_seconds=10.0,
                size_bytes=100,
            )
        ]
        transcriber.transcribe_chunks(chunks)

        assert fake_fn._calls[0]["language"] == "vi"

    def test_transcribe_chunks_passes_task(self, tmp_path: Path) -> None:
        """Verify task parameter is passed through."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        fake_fn = FakeTranscribeFn()
        transcriber = ParallelTranscriber(
            transcribe=fake_fn,
            task="translate",
        )

        chunks = [
            AudioChunk(
                path=str(audio_file),
                start_seconds=0.0,
                duration_seconds=10.0,
                size_bytes=100,
            )
        ]
        transcriber.transcribe_chunks(chunks)

        assert fake_fn._calls[0]["task"] == "translate"

    def test_transcribe_chunks_retries_on_failure(self, tmp_path: Path) -> None:
        """Verify retry logic on transient errors."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        failing_fn = FailingThenSuccessFn()
        transcriber = ParallelTranscriber(
            transcribe=failing_fn,
            max_retries=2,
        )

        chunks = [
            AudioChunk(
                path=str(audio_file),
                start_seconds=0.0,
                duration_seconds=10.0,
                size_bytes=100,
            )
        ]
        result = transcriber.transcribe_chunks(chunks)

        assert len(result) == 1
        assert result[0][0]["text"] == "Success"
        assert failing_fn.call_count == 2

    def test_transcribe_chunks_raises_after_max_retries(self, tmp_path: Path) -> None:
        """Raise error after exhausting retries."""
        audio_file = tmp_path / "test.mp3"
        audio_file.write_bytes(b"fake audio")

        always_fails_fn = AlwaysFailsFn()
        transcriber = ParallelTranscriber(
            transcribe=always_fails_fn,
            max_retries=1,
        )

        chunks = [
            AudioChunk(
                path=str(audio_file),
                start_seconds=0.0,
                duration_seconds=10.0,
                size_bytes=100,
            )
        ]

        with pytest.raises(TimeoutError, match="Always fails"):
            transcriber.transcribe_chunks(chunks)

    def test_transcribe_chunks_file_not_found(self) -> None:
        """Raise OSError for missing chunk file."""
        fake_fn = FakeTranscribeFn()
        transcriber = ParallelTranscriber(transcribe=fake_fn, max_retries=0)

        chunks = [
            AudioChunk(
                path="/nonexistent/file.mp3",
                start_seconds=0.0,
                duration_seconds=10.0,
                size_bytes=100,
            )
        ]

        with pytest.raises(OSError):
            transcriber.transcribe_chunks(chunks)
