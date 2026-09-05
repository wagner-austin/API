"""Measure a live HLS stream from its playlist and say what the numbers support.

The successor to the MJPEG ``stream_quality`` module, kept for the same
reason it existed: "the video looks laggy" must be answered by
measurement, not argument. For HLS the observable is the PLAYLIST — a
healthy live stream advances ``#EXT-X-MEDIA-SEQUENCE`` by one every
segment length, so sampling the playlist tells you directly whether
the encoder is producing on cadence, independent of any player.

What the numbers can and cannot support, stated up front:

* Sequence advancing on cadence proves the ENCODER is healthy. It says
  nothing about a viewer's network.
* A stall here (no advance for well past the target duration) is
  upstream of every viewer — the encoder or its display died — which
  is exactly the fault this probe exists to separate from "my phone's
  connection hiccuped".
* An unreachable playlist is its own row, not a stall: 503 means the
  encoder is warming, 404 means nothing is streaming at all.
"""

from __future__ import annotations

from typing_extensions import TypedDict


class ParsedPlaylistDict(TypedDict):
    """The fields of one media playlist this probe reasons about.

    Attributes:
        media_sequence: ``#EXT-X-MEDIA-SEQUENCE`` value — the index of
            the first segment in the window, which increments once per
            rotated segment and is therefore the live edge's odometer.
        target_duration: ``#EXT-X-TARGETDURATION`` in seconds.
        segment_durations: Each ``#EXTINF`` duration in the window,
            in order.
    """

    media_sequence: int
    target_duration: float
    segment_durations: list[float]


class PlaylistSampleDict(TypedDict):
    """One observation of the playlist URL.

    Attributes:
        at_ms: Sample wall-clock time.
        status: HTTP status of the fetch.
        playlist: The parsed playlist for a 200, else ``None``.
    """

    at_ms: int
    status: int
    playlist: ParsedPlaylistDict | None


class HlsReportDict(TypedDict):
    """What a sampling window supports saying about the stream.

    Attributes:
        samples: Playlist fetches attempted.
        ok_samples: Fetches that returned a parsable playlist.
        warming_samples: 503 answers (encoder not producing yet).
        missing_samples: 404 answers (no stream being served).
        advances: Media-sequence increments observed across the window.
        advance_gaps_ms: Milliseconds between consecutive advances,
            in observation order.
        max_gap_ms: The longest such gap (0 with fewer than two
            advances).
        stalls: Gaps longer than twice the target duration — the
            encoder missing its own cadence, not a viewer problem.
        target_duration: The playlist's declared target duration in
            seconds (0.0 when never observed).
    """

    samples: int
    ok_samples: int
    warming_samples: int
    missing_samples: int
    advances: int
    advance_gaps_ms: list[int]
    max_gap_ms: int
    stalls: int
    target_duration: float


def parse_playlist(text: str) -> ParsedPlaylistDict:
    """Parse the fields this probe needs out of an m3u8 media playlist.

    Args:
        text: The playlist body.

    Returns:
        The parsed fields. A playlist with no ``EXT-X-MEDIA-SEQUENCE``
        tag has sequence 0 — that is the tag's own defined default,
        not a guess.

    Raises:
        ValueError: ``text`` is not an m3u8 playlist at all, or a tag
            this probe reads carries a non-numeric value.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines or lines[0] != "#EXTM3U":
        raise ValueError("not an m3u8 playlist: missing #EXTM3U header")
    media_sequence = 0
    target_duration = 0.0
    segment_durations: list[float] = []
    for line in lines[1:]:
        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            media_sequence = int(line.partition(":")[2])
        elif line.startswith("#EXT-X-TARGETDURATION:"):
            target_duration = float(line.partition(":")[2])
        elif line.startswith("#EXTINF:"):
            segment_durations.append(float(line.partition(":")[2].partition(",")[0]))
    return ParsedPlaylistDict(
        media_sequence=media_sequence,
        target_duration=target_duration,
        segment_durations=segment_durations,
    )


def _advance_gaps(observed: list[tuple[int, ParsedPlaylistDict]]) -> list[int]:
    """Fold the observations into gaps between media-sequence advances.

    Args:
        observed: Successful observations in time order.

    Returns:
        Milliseconds between consecutive advances, measured between
        the FIRST sample at which each new sequence value was seen —
        an upper bound on encoder cadence quantised by the sampling
        interval, honest as long as the sampler polls several times
        per segment.
    """
    gaps: list[int] = []
    last_sequence: int | None = None
    last_advance_ms = 0
    for at_ms, playlist in observed:
        sequence = playlist["media_sequence"]
        if last_sequence is None:
            last_sequence = sequence
            last_advance_ms = at_ms
            continue
        if sequence > last_sequence:
            gaps.append(at_ms - last_advance_ms)
            last_advance_ms = at_ms
            last_sequence = sequence
    return gaps


def summarize_samples(samples: list[PlaylistSampleDict]) -> HlsReportDict:
    """Reduce a sampling window to the report's numbers.

    Args:
        samples: Observations in time order.

    Returns:
        The report.
    """
    observed: list[tuple[int, ParsedPlaylistDict]] = []
    warming = 0
    missing = 0
    for sample in samples:
        playlist = sample["playlist"]
        if playlist is not None:
            observed.append((sample["at_ms"], playlist))
        elif sample["status"] == 503:
            warming += 1
        elif sample["status"] == 404:
            missing += 1
    target_duration = observed[-1][1]["target_duration"] if observed else 0.0
    advance_gaps_ms = _advance_gaps(observed)
    stall_floor_ms = target_duration * 2000.0
    stalls = 0
    if target_duration > 0:
        for gap in advance_gaps_ms:
            if gap > stall_floor_ms:
                stalls += 1
    return HlsReportDict(
        samples=len(samples),
        ok_samples=len(observed),
        warming_samples=warming,
        missing_samples=missing,
        advances=len(advance_gaps_ms),
        advance_gaps_ms=advance_gaps_ms,
        max_gap_ms=max(advance_gaps_ms) if advance_gaps_ms else 0,
        stalls=stalls,
        target_duration=target_duration,
    )


def render_report(report: HlsReportDict) -> str:
    """Render the report for a terminal.

    Args:
        report: The summarized window.

    Returns:
        A multi-line report, one finding per line, ending with what
        the numbers do and do not support.
    """
    lines = [
        f"playlist samples: {report['samples']}"
        f" (ok {report['ok_samples']}, warming {report['warming_samples']},"
        f" missing {report['missing_samples']})",
        f"target duration: {report['target_duration']:.1f}s",
        f"segment advances: {report['advances']}",
    ]
    if report["advance_gaps_ms"]:
        gaps = sorted(report["advance_gaps_ms"])
        median = gaps[len(gaps) // 2]
        lines.append(
            f"advance cadence: median {median} ms, max {report['max_gap_ms']} ms"
            f" (healthy is ~{report['target_duration'] * 1000:.0f} ms)"
        )
    lines.append(f"encoder stalls (gap > 2x target): {report['stalls']}")
    if report["ok_samples"] == 0:
        lines.append("verdict: no playlist observed - nothing here says anything about smoothness")
    elif report["stalls"] == 0 and report["advances"] > 0:
        lines.append(
            "verdict: encoder is producing on cadence; any stutter a viewer sees is"
            " downstream of the stream itself"
        )
    else:
        lines.append(
            "verdict: the encoder itself missed its cadence - the fault is upstream of every viewer"
        )
    return "\n".join(lines)


__all__ = [
    "HlsReportDict",
    "ParsedPlaylistDict",
    "PlaylistSampleDict",
    "parse_playlist",
    "render_report",
    "summarize_samples",
]
