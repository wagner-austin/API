"""Judge a live-view MJPEG stream from its frames and their arrival times.

WHY THIS EXISTS. "The video looks laggy" was diagnosed four times in one
session by argument, and three of those answers were wrong: that the
game paints three times a second (measured on a tank standing alone,
the one case where it is true), that a page-side JPEG encoder was
starving the render loop (it costs the page nothing), and that the tick
loop blocks frame delivery (it waits inside Playwright, which pumps
events, so it does not). Each was plausible and each cost a rebuild
that would have fixed nothing.

The numbers that settled it are the ones below, and none of them is
hard to compute -- what was missing was a place to put them. Rate alone
cannot tell a slow stream from a stuttering one, and average rate hides
the distinction entirely: a stream delivering a smooth 3/s and a stream
delivering 30/s for a tenth of a second then nothing for a second both
report "3 frames per second".

WHAT THE FIELDS MEAN, and what a bad value points at:

``duplicate_share``
    Frames byte-identical to the one before. Anything above zero is
    bandwidth spent re-sending a picture the viewer already has. It was
    71 per cent before the caster learned to compare frames.

``at_sampling_floor``
    Gaps that landed within a small tolerance of the caster's own
    interval. This is the one that is easy to misread as content: when
    it is large, the stream is reporting the SAMPLING RATE rather than
    the paint rate, and the fix is to sample faster. At 12 Hz this was
    71 of 148 gaps, all piled at 83 ms.

``burst_share``
    Gaps under :data:`BURST_GAP_SECONDS`, i.e. real motion captured at
    a watchable rate. High burst share with a low overall rate means
    the source is genuinely idle between events and no transport change
    will help.

``stalls``
    Gaps over :data:`STALL_GAP_SECONDS`. Their COUNT and total duration
    are what a viewer experiences as lag.
"""

from __future__ import annotations

import hashlib
from itertools import pairwise

from typing_extensions import TypedDict

BURST_GAP_SECONDS = 0.1
"""Gap at or under which consecutive frames read as continuous motion.

Ten frames a second is the rate at which a viewer stops seeing steps
and starts seeing movement. Used to separate "captured an animation"
from "sent two pictures a second apart".
"""

STALL_GAP_SECONDS = 1.0
"""Gap at or over which a viewer perceives the picture as frozen."""

SAMPLING_FLOOR_TOLERANCE = 0.25
"""Fractional window around the sampling interval counted as "at the floor".

A ``setInterval`` does not fire on an exact period, so a gap produced by
the sampler rather than by the content lands near the interval rather
than on it. Twenty-five per cent is wide enough to catch the pile and
narrow enough not to swallow genuinely different gaps: at 83 ms it spans
62 to 104 ms, which excludes both the sub-50 ms burst bucket and
anything past 125 ms.
"""


class StreamReportDict(TypedDict):
    """Everything measurable about one sampled stream.

    Attributes:
        seconds: Length of the observation window.
        frames: Frames received.
        frames_per_second: Frames received per second of the window.
        distinct: Frames whose bytes differ from the frame before.
        distinct_per_second: Distinct frames per second.
        duplicate_share: Fraction of frames identical to the previous
            one, in [0, 1]. Zero once the sender suppresses repeats.
        burst_gaps: Gaps at or under :data:`BURST_GAP_SECONDS`.
        burst_share: Those as a fraction of all gaps, in [0, 1].
        at_sampling_floor: Gaps within
            :data:`SAMPLING_FLOOR_TOLERANCE` of the declared sampling
            interval. Zero when no interval was declared.
        stalls: Gaps at or over :data:`STALL_GAP_SECONDS`.
        stalled_seconds: Their total duration.
        median_gap: Median inter-frame gap, or 0.0 with under two
            frames.
        min_gap: Shortest gap, or 0.0 with under two frames.
        max_gap: Longest gap, or 0.0 with under two frames.
        bytes_per_second: Wire bytes per second of the window.
    """

    seconds: float
    frames: int
    frames_per_second: float
    distinct: int
    distinct_per_second: float
    duplicate_share: float
    burst_gaps: int
    burst_share: float
    at_sampling_floor: int
    stalls: int
    stalled_seconds: float
    median_gap: float
    min_gap: float
    max_gap: float
    bytes_per_second: float


def _median(values: list[float]) -> float:
    """Return the median of a non-empty sorted-able list.

    Args:
        values: Samples; must not be empty.

    Returns:
        The middle value for an odd count, the mean of the two middle
        values for an even one.
    """
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def summarize_stream(
    frames: list[bytes],
    arrivals: list[float],
    seconds: float,
    sampling_interval: float,
) -> StreamReportDict:
    """Reduce a sampled stream to the numbers that distinguish its faults.

    Args:
        frames: Frame payloads in arrival order.
        arrivals: Arrival time of each frame, in seconds from the start
            of the window. Must be the same length as ``frames``.
        seconds: Length of the observation window. Must be positive.
        sampling_interval: The sender's declared inter-frame interval in
            seconds, used for :attr:`StreamReportDict.at_sampling_floor`.
            Pass ``0.0`` when it is unknown, which reports zero rather
            than guessing.

    Returns:
        The report.

    Raises:
        ValueError: If the two lists differ in length, or the window is
            not positive. Both are caller bugs that would otherwise
            produce a plausible-looking report from misaligned data.
    """
    if len(frames) != len(arrivals):
        raise ValueError(f"{len(frames)} frames against {len(arrivals)} arrival times")
    if seconds <= 0:
        raise ValueError(f"observation window must be positive, got {seconds}")

    digests = [hashlib.sha256(frame).digest() for frame in frames]
    distinct = sum(1 for i, d in enumerate(digests) if i == 0 or d != digests[i - 1])
    duplicates = len(frames) - distinct

    gaps = [b - a for a, b in pairwise(arrivals)]
    burst = sum(1 for g in gaps if g <= BURST_GAP_SECONDS)
    stall_gaps = [g for g in gaps if g >= STALL_GAP_SECONDS]
    floor = 0
    if sampling_interval > 0:
        low = sampling_interval * (1.0 - SAMPLING_FLOOR_TOLERANCE)
        high = sampling_interval * (1.0 + SAMPLING_FLOOR_TOLERANCE)
        floor = sum(1 for g in gaps if low <= g <= high)

    return StreamReportDict(
        seconds=seconds,
        frames=len(frames),
        frames_per_second=len(frames) / seconds,
        distinct=distinct,
        distinct_per_second=distinct / seconds,
        duplicate_share=duplicates / len(frames) if frames else 0.0,
        burst_gaps=burst,
        burst_share=burst / len(gaps) if gaps else 0.0,
        at_sampling_floor=floor,
        stalls=len(stall_gaps),
        stalled_seconds=sum(stall_gaps),
        median_gap=_median(gaps) if gaps else 0.0,
        min_gap=min(gaps) if gaps else 0.0,
        max_gap=max(gaps) if gaps else 0.0,
        bytes_per_second=sum(len(f) for f in frames) / seconds,
    )


def render_report(report: StreamReportDict) -> str:
    """Render a report as the operator-facing block.

    The verdict lines are the point of the tool. Each names the ONE
    reading that a number supports, so the report cannot be used to
    argue for a fix it does not evidence.

    Args:
        report: The measured report.

    Returns:
        A multi-line block, no trailing newline.
    """
    lines = [
        f"window            {report['seconds']:.1f} s",
        f"frames            {report['frames']} = {report['frames_per_second']:.2f}/s",
        f"distinct          {report['distinct']} = {report['distinct_per_second']:.2f}/s",
        f"duplicate share   {report['duplicate_share'] * 100:.0f}%",
        f"bandwidth         {report['bytes_per_second'] / 1024:.0f} KB/s",
        f"gaps  median      {report['median_gap'] * 1000:.0f} ms"
        f"   min {report['min_gap'] * 1000:.0f}"
        f"   max {report['max_gap'] * 1000:.0f}",
        f"bursts (<={BURST_GAP_SECONDS * 1000:.0f} ms)  "
        f"{report['burst_gaps']} = {report['burst_share'] * 100:.0f}% of gaps",
        f"at sampling floor {report['at_sampling_floor']}",
        f"stalls (>={STALL_GAP_SECONDS:.0f} s)    "
        f"{report['stalls']}, {report['stalled_seconds']:.1f} s total",
        "",
    ]
    lines.extend(verdicts(report))
    return "\n".join(lines)


def verdicts(report: StreamReportDict) -> list[str]:
    """Name what each out-of-range number points at, and nothing more.

    Args:
        report: The measured report.

    Returns:
        Zero or more lines. An empty list means nothing in the report
        supports a complaint, which is itself an answer.
    """
    found: list[str] = []
    if report["duplicate_share"] > 0.05:
        found.append(
            f"SENDER REPEATS ITSELF: {report['duplicate_share'] * 100:.0f}% of frames are "
            "byte-identical to the one before. That is bandwidth, not motion."
        )
    if report["at_sampling_floor"] > report["frames"] * 0.25:
        found.append(
            f"UNDERSAMPLED: {report['at_sampling_floor']} gaps sit at the sampler's own "
            "interval, so this measures the CAPTURE rate, not the source. Sample faster "
            "before concluding anything about the source."
        )
    if report["stalls"] > 0 and report["burst_share"] > 0.5:
        found.append(
            f"BURSTY SOURCE: {report['burst_share'] * 100:.0f}% of gaps are motion but "
            f"{report['stalls']} stalls cover {report['stalled_seconds']:.1f} s. The "
            "source is idle between events; a faster transport cannot fill that."
        )
    if report["stalls"] > 0 and report["burst_share"] <= 0.5:
        found.append(
            f"SLOW THROUGHOUT: {report['stalls']} stalls and only "
            f"{report['burst_share'] * 100:.0f}% of gaps are motion. Suspect the "
            "delivery path before the source."
        )
    return found


__all__ = [
    "BURST_GAP_SECONDS",
    "SAMPLING_FLOOR_TOLERANCE",
    "STALL_GAP_SECONDS",
    "StreamReportDict",
    "render_report",
    "summarize_stream",
    "verdicts",
]
