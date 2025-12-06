from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol

from transcript_api.merger import TranscriptMerger
from transcript_api.types import AudioChunk, TranscriptSegment


class _CapLogLike(Protocol):
    records: Sequence[logging.LogRecord]

    def set_level(self, level: int | str) -> None: ...


def _make_chunk(path: str, start: float, duration: float) -> AudioChunk:
    return AudioChunk(
        path=path,
        start_seconds=start,
        duration_seconds=duration,
        size_bytes=0,
    )


def _make_segment(text: str, start: float, duration: float) -> TranscriptSegment:
    return TranscriptSegment(text=text, start=start, duration=duration)


def test_merger_skips_empty_chunks_and_sorts(caplog: _CapLogLike) -> None:
    caplog.set_level("DEBUG")
    chunk_a = _make_chunk("a.m4a", start=0.0, duration=5.0)
    chunk_b = _make_chunk("b.m4a", start=10.0, duration=5.0)

    seg_b1 = _make_segment("b1", start=0.0, duration=1.0)
    seg_b2 = _make_segment("b2", start=3.0, duration=1.0)

    merger = TranscriptMerger()
    merged = merger.merge(
        [
            (chunk_a, []),
            (chunk_b, [seg_b2, seg_b1]),
        ]
    )

    assert [s["text"] for s in merged] == ["b1", "b2"]
    assert [round(s["start"], 1) for s in merged] == [10.0, 13.0]

    messages = [record.getMessage() for record in caplog.records]
    assert any("has no segments" in msg for msg in messages)
    assert any("Merge complete" in msg for msg in messages)


def test_adjust_timestamps_clamps_negative_start() -> None:
    chunk = _make_chunk("c.m4a", start=-5.0, duration=5.0)
    seg = _make_segment("neg", start=-3.0, duration=1.0)

    merger = TranscriptMerger()
    adjusted = merger._adjust_timestamps([seg], offset_seconds=chunk["start_seconds"])

    assert len(adjusted) == 1
    assert adjusted[0]["start"] == 0.0
    assert adjusted[0]["text"] == "neg"
