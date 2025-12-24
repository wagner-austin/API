"""Parallel transcription of audio chunks with bounded concurrency.

Provides thread-based parallel processing of audio chunks with retry support.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import BinaryIO, Literal, Protocol

from platform_core.logging import get_logger

from .types import AudioChunk, TranscriptSegment, VerboseResponse, WhisperTask
from .whisper_parse import convert_verbose_to_segments

# Type alias for list of segments
TranscriptSegmentList = list[TranscriptSegment]


class TranscribeFn(Protocol):
    """Protocol for transcription function.

    Callables matching this signature can be used with ParallelTranscriber.
    """

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
        """Transcribe or translate audio file.

        Args:
            model: Model name (e.g., "whisper-1").
            file: Binary file object containing audio data.
            response_format: Response format (must be "verbose_json").
            language: Optional source language hint.
            task: Task type ("transcribe" or "translate").
            timeout: Optional request timeout.

        Returns:
            VerboseResponse with text and segments.
        """
        ...


class ParallelTranscriber:
    """Parallel transcription of audio chunks with bounded concurrency.

    Uses a thread pool to process multiple chunks simultaneously while
    respecting concurrency limits and retry policies.

    Attributes:
        max_concurrent: Maximum number of concurrent transcriptions.
        max_retries: Maximum retry attempts per chunk.
        timeout_seconds: Request timeout for each chunk.
    """

    __slots__ = (
        "_language",
        "_logger",
        "_max_concurrent",
        "_max_retries",
        "_task",
        "_timeout",
        "_transcribe",
    )

    def __init__(
        self,
        *,
        transcribe: TranscribeFn,
        max_concurrent: int = 3,
        max_retries: int = 2,
        timeout_seconds: float = 900.0,
        language: str | None = None,
        task: WhisperTask = "transcribe",
    ) -> None:
        """Initialize parallel transcriber.

        Args:
            transcribe: Function to call for transcription.
            max_concurrent: Maximum concurrent transcriptions (default: 3).
            max_retries: Maximum retries per chunk (default: 2).
            timeout_seconds: Request timeout in seconds (default: 900).
            language: Optional source language hint.
            task: Task type ("transcribe" or "translate").
        """
        self._transcribe = transcribe
        self._max_concurrent = max(1, int(max_concurrent))
        self._max_retries = max(0, int(max_retries))
        self._timeout = float(timeout_seconds)
        self._language = language
        self._task = task
        self._logger = get_logger(__name__)

    def transcribe_chunks(self, chunks: list[AudioChunk]) -> list[TranscriptSegmentList]:
        """Transcribe all chunks with bounded parallelism and retries.

        Uses thread pool to process chunks concurrently. Failed chunks are
        retried up to max_retries times before the error propagates.

        Args:
            chunks: List of AudioChunk descriptors to process.

        Returns:
            List of segment lists, one per chunk, in chunk order.

        Raises:
            OSError: If file operations fail after retries.
            TimeoutError: If transcription times out after retries.
            ValueError: If transcription response is invalid after retries.
        """
        total = len(chunks)

        def work(idx: int, chunk: AudioChunk) -> TranscriptSegmentList:
            attempt = 0
            while True:
                attempt += 1
                try:
                    self._logger.info(
                        "Transcribing chunk %d/%d: path=%s size=%d bytes",
                        idx + 1,
                        total,
                        chunk["path"],
                        chunk["size_bytes"],
                    )
                    with open(chunk["path"], "rb") as f:
                        resp = self._transcribe(
                            model="whisper-1",
                            file=f,
                            response_format="verbose_json",
                            language=self._language,
                            task=self._task,
                            timeout=self._timeout,
                        )
                    segments = convert_verbose_to_segments(resp)
                    self._logger.info(
                        "Chunk %d/%d complete: segments=%d start=%.1fs duration=%.1fs",
                        idx + 1,
                        total,
                        len(segments),
                        chunk["start_seconds"],
                        chunk["duration_seconds"],
                    )
                    return segments
                except (OSError, TimeoutError, ValueError) as e:
                    if attempt <= self._max_retries:
                        self._logger.debug(
                            "Retrying chunk start=%.2fs attempt=%d error=%s",
                            chunk["start_seconds"],
                            attempt,
                            e,
                        )
                        continue
                    raise

        out: list[TranscriptSegmentList] = [[] for _ in chunks]
        with ThreadPoolExecutor(max_workers=self._max_concurrent) as pool:
            futures = {pool.submit(work, i, c): i for i, c in enumerate(chunks)}
            for fut in as_completed(futures):
                idx = futures[fut]
                out[idx] = fut.result()
        return out


__all__ = ["ParallelTranscriber", "TranscribeFn", "TranscriptSegmentList"]
