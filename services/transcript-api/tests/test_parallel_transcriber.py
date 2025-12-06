from __future__ import annotations

import logging
import os
import tempfile
from typing import BinaryIO, Literal

import pytest

from transcript_api.parallel import ParallelTranscriber
from transcript_api.types import AudioChunk, VerboseResponseTD


def _make_chunk(contents: bytes, start: float, dur: float) -> AudioChunk:
    fd, path = tempfile.mkstemp(prefix="chunk_", suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(contents)
    except OSError as e:
        logging.getLogger(__name__).exception("failed to write temp chunk: %s", e)
        os.close(fd)
        raise
    return AudioChunk(
        path=path,
        start_seconds=start,
        duration_seconds=dur,
        size_bytes=len(contents),
    )


def _cleanup_chunks(chunks: list[AudioChunk]) -> None:
    for c in chunks:
        try:
            os.remove(c["path"])
        except OSError:
            logging.getLogger(__name__).exception("cleanup failed for %s", c["path"])
            raise


def test_parallel_transcriber_returns_segments_per_chunk() -> None:
    # Fake transcribe that returns one segment per input file
    def fake_transcribe(
        *,
        model: str,
        file: BinaryIO,
        response_format: Literal["verbose_json"],
        timeout: float | None = None,
    ) -> VerboseResponseTD:
        data = file.read()
        return {"text": "", "segments": [{"text": f"{len(data)} bytes", "start": 0.0, "end": 1.0}]}

    pt = ParallelTranscriber(transcribe=fake_transcribe, max_concurrent=2, max_retries=0)
    chunks = [
        _make_chunk(b"aaa", 0.0, 1.0),
        _make_chunk(b"bbbb", 1.0, 1.0),
        _make_chunk(b"cc", 2.0, 1.0),
    ]
    try:
        out = pt.transcribe_chunks(chunks)
        assert len(out) == len(chunks)
        assert [len(s) for s in out] == [1, 1, 1]
        assert out[0][0]["text"].endswith("3 bytes")
    finally:
        _cleanup_chunks(chunks)


def test_parallel_transcriber_retries_then_succeeds() -> None:
    attempts: list[int] = []

    def flaky_transcribe(
        *,
        model: str,
        file: BinaryIO,
        response_format: Literal["verbose_json"],
        timeout: float | None = None,
    ) -> VerboseResponseTD:
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("transient")
        data = file.read()
        return {"text": "", "segments": [{"text": f"{len(data)} bytes", "start": 0.0, "end": 1.0}]}

    pt = ParallelTranscriber(
        transcribe=flaky_transcribe,
        max_concurrent=1,
        max_retries=2,
        timeout_seconds=5.0,
    )
    chunks = [_make_chunk(b"abc", 0.0, 1.0)]

    try:
        out = pt.transcribe_chunks(chunks)
        assert len(out) == 1
        assert len(out[0]) == 1
        assert len(attempts) == 2
    finally:
        _cleanup_chunks(chunks)


def test_parallel_transcriber_raises_after_retry_exhausted() -> None:
    calls: list[int] = []

    def always_fail(
        *,
        model: str,
        file: BinaryIO,
        response_format: Literal["verbose_json"],
        timeout: float | None = None,
    ) -> VerboseResponseTD:
        calls.append(1)
        raise OSError("persistent failure")

    pt = ParallelTranscriber(
        transcribe=always_fail,
        max_concurrent=1,
        max_retries=1,
        timeout_seconds=5.0,
    )
    chunks = [_make_chunk(b"x", 0.0, 1.0)]

    try:
        with pytest.raises(OSError):
            pt.transcribe_chunks(chunks)
        assert len(calls) == 2
    finally:
        _cleanup_chunks(chunks)
