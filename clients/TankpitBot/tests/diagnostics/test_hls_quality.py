"""The HLS playlist probe: parsing, reduction, and honest rendering."""

from __future__ import annotations

import pytest

from tankpit_bot.diagnostics.hls_quality import (
    HlsReportDict,
    PlaylistSampleDict,
    parse_playlist,
    render_report,
    summarize_samples,
)


def _playlist_text(sequence: int, target: float = 2.0) -> str:
    """One well-formed live media playlist.

    Args:
        sequence: The ``EXT-X-MEDIA-SEQUENCE`` value.
        target: The target duration.

    Returns:
        The playlist body.
    """
    return (
        "#EXTM3U\n"
        "#EXT-X-VERSION:3\n"
        f"#EXT-X-TARGETDURATION:{target:.0f}\n"
        f"#EXT-X-MEDIA-SEQUENCE:{sequence}\n"
        "#EXTINF:2.000000,\n"
        f"seg{sequence:05d}.ts\n"
        "#EXTINF:2.000000,\n"
        f"seg{sequence + 1:05d}.ts\n"
    )


def _sample(at_ms: int, sequence: int) -> PlaylistSampleDict:
    """One successful observation.

    Args:
        at_ms: Sample time.
        sequence: Media sequence observed.

    Returns:
        The sample.
    """
    return PlaylistSampleDict(
        at_ms=at_ms, status=200, playlist=parse_playlist(_playlist_text(sequence))
    )


def _failed(at_ms: int, status: int) -> PlaylistSampleDict:
    """One unsuccessful observation.

    Args:
        at_ms: Sample time.
        status: The HTTP status received.

    Returns:
        The sample.
    """
    return PlaylistSampleDict(at_ms=at_ms, status=status, playlist=None)


class TestParsePlaylist:
    """Field extraction and refusals."""

    def test_reads_sequence_target_and_durations(self) -> None:
        """Every field this probe reasons about is extracted."""
        parsed = parse_playlist(_playlist_text(41))
        assert parsed["media_sequence"] == 41
        assert parsed["target_duration"] == 2.0
        assert parsed["segment_durations"] == [2.0, 2.0]

    def test_missing_sequence_tag_defaults_to_zero(self) -> None:
        """The tag's own defined default, not a guess."""
        parsed = parse_playlist("#EXTM3U\n#EXT-X-TARGETDURATION:2\n")
        assert parsed["media_sequence"] == 0

    def test_non_playlist_text_is_refused(self) -> None:
        """A 200 that is not m3u8 points at a misrouted URL."""
        with pytest.raises(ValueError, match="missing #EXTM3U"):
            parse_playlist("<html>not a stream</html>")

    def test_empty_text_is_refused(self) -> None:
        """No lines at all is not a playlist either."""
        with pytest.raises(ValueError, match="missing #EXTM3U"):
            parse_playlist("")

    def test_non_numeric_tag_value_propagates(self) -> None:
        """A malformed tag fails loudly instead of being skipped."""
        with pytest.raises(ValueError):
            parse_playlist("#EXTM3U\n#EXT-X-MEDIA-SEQUENCE:soon\n")


class TestSummarizeSamples:
    """The reduction's counts, gaps, and stall arithmetic."""

    def test_a_healthy_window_counts_advances_on_cadence(self) -> None:
        """Sequence stepping every ~2 s reads as zero stalls."""
        samples = [
            _sample(0, 10),
            _sample(1000, 10),
            _sample(2000, 11),
            _sample(3000, 11),
            _sample(4000, 12),
        ]
        report = summarize_samples(samples)
        assert report["samples"] == 5
        assert report["ok_samples"] == 5
        assert report["advances"] == 2
        assert report["advance_gaps_ms"] == [2000, 2000]
        assert report["max_gap_ms"] == 2000
        assert report["stalls"] == 0
        assert report["target_duration"] == 2.0

    def test_a_gap_past_twice_the_target_is_a_stall(self) -> None:
        """The encoder missing its own cadence is counted upstream."""
        samples = [_sample(0, 10), _sample(2000, 11), _sample(9000, 12)]
        report = summarize_samples(samples)
        assert report["advance_gaps_ms"] == [2000, 7000]
        assert report["stalls"] == 1
        assert report["max_gap_ms"] == 7000

    def test_warming_and_missing_answers_are_their_own_rows(self) -> None:
        """503 and 404 are counted, never folded into stalls."""
        samples = [
            _failed(0, 503),
            _failed(500, 503),
            _failed(1000, 404),
            _sample(1500, 1),
        ]
        report = summarize_samples(samples)
        assert report["warming_samples"] == 2
        assert report["missing_samples"] == 1
        assert report["ok_samples"] == 1
        assert report["advances"] == 0
        assert report["max_gap_ms"] == 0

    def test_an_unexpected_status_counts_in_neither_bucket(self) -> None:
        """A 500 is a sample that is neither warming nor missing."""
        report = summarize_samples([_failed(0, 500)])
        assert report["samples"] == 1
        assert report["warming_samples"] == 0
        assert report["missing_samples"] == 0

    def test_an_empty_window_is_all_zeroes(self) -> None:
        """No observations produce no claims."""
        report = summarize_samples([])
        assert report["ok_samples"] == 0
        assert report["target_duration"] == 0.0
        assert report["stalls"] == 0


def _healthy_report(*, stalls: int = 0) -> HlsReportDict:
    """A report from an on-cadence window.

    Args:
        stalls: Stall count to stamp in.

    Returns:
        The report.
    """
    return HlsReportDict(
        samples=10,
        ok_samples=10,
        warming_samples=0,
        missing_samples=0,
        advances=4,
        advance_gaps_ms=[2000, 2100, 1900, 2000],
        max_gap_ms=2100,
        stalls=stalls,
        target_duration=2.0,
    )


class TestRenderReport:
    """Each verdict states only what the numbers support."""

    def test_healthy_window_blames_nothing_upstream(self) -> None:
        """On-cadence advances put any stutter downstream of the stream."""
        text = render_report(_healthy_report())
        assert "median 2000 ms" in text
        assert "downstream of the stream itself" in text

    def test_stalls_blame_the_encoder(self) -> None:
        """A missed cadence is an upstream fault, said plainly."""
        text = render_report(_healthy_report(stalls=2))
        assert "the fault is upstream" in text

    def test_no_observations_claims_nothing(self) -> None:
        """A window that saw no playlist supports no smoothness claim."""
        text = render_report(
            HlsReportDict(
                samples=10,
                ok_samples=0,
                warming_samples=8,
                missing_samples=2,
                advances=0,
                advance_gaps_ms=[],
                max_gap_ms=0,
                stalls=0,
                target_duration=0.0,
            )
        )
        assert "nothing here says anything about smoothness" in text
        assert "median" not in text
