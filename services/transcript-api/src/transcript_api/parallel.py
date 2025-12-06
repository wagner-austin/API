from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import BinaryIO, Literal, Protocol, runtime_checkable

from platform_core.logging import get_logger

from .types import AudioChunk, TranscriptSegmentList, VerboseResponseTD
from .whisper_parse import convert_verbose_to_segments


@runtime_checkable
class TranscribeFn(Protocol):
    def __call__(
        self,
        *,
        model: str,
        file: BinaryIO,
        response_format: Literal["verbose_json"],
        timeout: float | None = None,
    ) -> VerboseResponseTD: ...


class ParallelTranscriber:
    def __init__(
        self,
        *,
        transcribe: TranscribeFn,
        max_concurrent: int = 3,
        max_retries: int = 2,
        timeout_seconds: float = 900.0,
    ) -> None:
        self._transcribe = transcribe
        self._max_concurrent = max(1, int(max_concurrent))
        self._max_retries = max(0, int(max_retries))
        self._timeout = float(timeout_seconds)
        self._logger = get_logger(__name__)

    def transcribe_chunks(self, chunks: list[AudioChunk]) -> list[TranscriptSegmentList]:
        """Transcribe all chunks with bounded parallelism and retries (threads)."""
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
