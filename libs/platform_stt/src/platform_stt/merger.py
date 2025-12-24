"""Merge transcription segments from multiple audio chunks.

Handles timestamp adjustment and ordering when combining chunk results.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.logging import get_logger

from .types import AudioChunk, TranscriptSegment


class TranscriptMerger:
    """Merge segments from multiple audio chunks into a single transcript.

    Adjusts timestamps based on each chunk's offset in the original audio
    and produces a properly ordered segment list.
    """

    __slots__ = ("_logger",)

    def __init__(self) -> None:
        """Initialize transcript merger."""
        self._logger = get_logger(__name__)

    def merge(
        self,
        chunk_results: list[tuple[AudioChunk, list[TranscriptSegment]]],
    ) -> list[TranscriptSegment]:
        """Merge segments from all chunks into a single ordered transcript.

        Adjusts start timestamps by each chunk's start offset, concatenates
        all segments, then sorts by start time.

        Args:
            chunk_results: List of (chunk, segments) tuples.

        Returns:
            Merged and sorted list of TranscriptSegment.
        """
        adjusted: list[TranscriptSegment] = []
        for idx, (chunk, segs) in enumerate(chunk_results):
            if not segs:
                self._logger.warning(
                    "Chunk %d has no segments (start=%.1fs)", idx, chunk["start_seconds"]
                )
                continue
            self._logger.debug(
                "Merging chunk %d: %d segments (start=%.1fs, duration=%.1fs)",
                idx,
                len(segs),
                chunk["start_seconds"],
                chunk["duration_seconds"],
            )
            adjusted.extend(self._adjust_timestamps(segs, chunk["start_seconds"]))
        adjusted.sort(key=lambda s: s["start"])
        self._logger.info(
            "Merge complete: %d total segments from %d chunks", len(adjusted), len(chunk_results)
        )
        return adjusted

    def _adjust_timestamps(
        self, segments: Sequence[TranscriptSegment], offset_seconds: float
    ) -> list[TranscriptSegment]:
        """Adjust segment timestamps by adding offset.

        Args:
            segments: List of segments to adjust.
            offset_seconds: Time offset to add to each segment's start.

        Returns:
            New list of segments with adjusted timestamps.
        """
        out: list[TranscriptSegment] = []
        for seg in segments:
            out.append(
                TranscriptSegment(
                    text=seg["text"],
                    start=max(0.0, seg["start"] + offset_seconds),
                    duration=seg["duration"],
                )
            )
        return out


def merge_segment_text(segments: list[TranscriptSegment]) -> str:
    """Concatenate text from all segments into a single string.

    Args:
        segments: List of transcript segments.

    Returns:
        Combined text with segments separated by spaces.
    """
    parts: list[str] = []
    for seg in segments:
        text = seg["text"].strip()
        if text:
            parts.append(text)
    return " ".join(parts)


__all__ = ["TranscriptMerger", "merge_segment_text"]
