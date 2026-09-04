"""Tests for the stream-quality report and the MJPEG reader.

The report exists to stop a stream being diagnosed by argument, so the
tests are written as the arguments it has to be able to settle: is the
sender repeating itself, are we measuring our own sampling rate, is the
source bursty, or is the delivery path slow. Each case is a stream
shaped to make exactly one of those true.
"""

from __future__ import annotations

import pytest

from tankpit_bot.diagnostics.mjpeg_reader import (
    boundary_from_content_type,
    frames_from_buffer,
)
from tankpit_bot.diagnostics.stream_quality import (
    render_report,
    summarize_stream,
    verdicts,
)

BOUNDARY = b"--tankpitbotframe"


def _jpeg(marker: int) -> bytes:
    """Build a distinguishable pseudo-JPEG.

    Args:
        marker: Byte that makes this frame differ from another.

    Returns:
        Bytes beginning with the JPEG magic.
    """
    return b"\xff\xd8\xff" + bytes([marker]) * 32


def _part(body: bytes) -> bytes:
    """Wrap a body in one multipart part.

    Args:
        body: The frame payload.

    Returns:
        The boundary, headers and body.
    """
    return (
        BOUNDARY
        + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
        + str(len(body)).encode()
        + b"\r\n\r\n"
        + body
        + b"\r\n"
    )


class TestMjpegReader:
    def test_a_part_is_only_complete_once_the_next_boundary_arrives(self) -> None:
        """A trailing part is held back, not yielded truncated.

        This is the property that keeps a half-arrived JPEG out of the
        numbers. Counted as a frame it would drag the byte rate down
        and add a phantom gap.
        """
        stream = _part(_jpeg(1)) + _part(_jpeg(2))
        cut = len(stream) - 10

        frames, rest = frames_from_buffer(stream[:cut], BOUNDARY)

        assert frames == [_jpeg(1)]
        assert rest.startswith(BOUNDARY)

    def test_the_held_remainder_completes_on_the_next_read(self) -> None:
        """Feeding the rest back yields the frame that was pending.

        The held part is only released once a boundary follows it, so
        the second read appends the tail AND the next boundary -- which
        is exactly what arriving bytes do on a live stream.
        """
        stream = _part(_jpeg(1)) + _part(_jpeg(2))
        cut = len(stream) - 15
        first, rest = frames_from_buffer(stream[:cut], BOUNDARY)

        second, _ = frames_from_buffer(rest + stream[cut:] + BOUNDARY, BOUNDARY)

        assert first == [_jpeg(1)]
        assert second == [_jpeg(2)]

    def test_bytes_with_no_boundary_yield_nothing_and_are_kept(self) -> None:
        """A read that lands mid-body is not data loss."""
        frames, rest = frames_from_buffer(b"garbage-with-no-boundary", BOUNDARY)

        assert frames == []
        assert rest == b"garbage-with-no-boundary"

    def test_a_part_whose_body_is_not_a_jpeg_is_skipped(self) -> None:
        """An error page inside the stream is not counted as a frame.

        The relay passes upstream bytes through, so a child that
        answered with text mid-stream would otherwise be measured as
        video.
        """
        stream = _part(b"<html>child is unwell</html>") + _part(_jpeg(4)) + BOUNDARY

        frames, _ = frames_from_buffer(stream, BOUNDARY)

        assert frames == [_jpeg(4)]

    def test_a_part_without_a_header_terminator_is_skipped(self) -> None:
        """Headers that never end carry no body to take."""
        stream = BOUNDARY + b"\r\nContent-Type: image/jpeg" + BOUNDARY + b"x"

        frames, _ = frames_from_buffer(stream, BOUNDARY)

        assert frames == []

    def test_an_empty_boundary_is_refused(self) -> None:
        """Zero-length boundaries would match everywhere, forever."""
        with pytest.raises(ValueError, match="must not be empty"):
            frames_from_buffer(b"anything", b"")

    def test_the_boundary_comes_from_the_sender(self) -> None:
        """The token is read off the header, never assumed."""
        assert (
            boundary_from_content_type("multipart/x-mixed-replace; boundary=childframe42")
            == b"--childframe42"
        )

    def test_a_quoted_boundary_is_unwrapped(self) -> None:
        """Quoted parameter values are legal and must still parse."""
        assert boundary_from_content_type('multipart/x-mixed-replace; boundary="abc"') == b"--abc"

    def test_a_content_type_without_a_boundary_is_refused(self) -> None:
        """Guessing one would silently produce an empty stream."""
        with pytest.raises(ValueError, match="no boundary"):
            boundary_from_content_type("image/jpeg")

    def test_an_empty_boundary_parameter_is_refused(self) -> None:
        """``boundary=`` with nothing after it is malformed, not blank."""
        with pytest.raises(ValueError, match="empty boundary"):
            boundary_from_content_type("multipart/x-mixed-replace; boundary=")


class TestSummarize:
    def test_a_sender_repeating_itself_is_named(self) -> None:
        """Identical consecutive frames are duplicates, not motion.

        The exact shape the caster produced before it learned to
        compare: a steady rate, every frame the same picture.
        """
        frames = [_jpeg(1)] * 10
        arrivals = [i * 0.083 for i in range(10)]

        report = summarize_stream(frames, arrivals, 1.0, 0.0)

        assert report["distinct"] == 1
        assert report["duplicate_share"] == pytest.approx(0.9)
        assert any("SENDER REPEATS ITSELF" in v for v in verdicts(report))

    def test_gaps_piled_on_the_sampling_interval_are_named_as_undersampling(self) -> None:
        """The reading that is easiest to mistake for the source's rate.

        Every gap sits at the declared 83 ms interval, which means the
        stream is reporting how fast it was ASKED to capture. Concluding
        anything about the game from this is the error the field exists
        to prevent.
        """
        frames = [_jpeg(i) for i in range(12)]
        arrivals = [i * 0.083 for i in range(12)]

        report = summarize_stream(frames, arrivals, 1.0, 0.083)

        assert report["at_sampling_floor"] == 11
        assert any("UNDERSAMPLED" in v for v in verdicts(report))

    def test_a_bursty_source_is_distinguished_from_a_slow_one(self) -> None:
        """Motion then silence: the transport cannot fix the silence.

        Ten frames 30 ms apart, then a two-second hole. Average rate
        alone would call this slow; the burst share is what says the
        source went idle.
        """
        frames = [_jpeg(i) for i in range(12)]
        arrivals = [i * 0.03 for i in range(10)] + [2.3, 2.33]

        report = summarize_stream(frames, arrivals, 3.0, 0.0)

        assert report["stalls"] == 1
        assert report["burst_share"] > 0.5
        assert any("BURSTY SOURCE" in v for v in verdicts(report))

    def test_a_uniformly_slow_stream_points_at_the_delivery_path(self) -> None:
        """No bursts and repeated stalls is the opposite diagnosis."""
        frames = [_jpeg(i) for i in range(4)]
        arrivals = [0.0, 1.5, 3.0, 4.5]

        report = summarize_stream(frames, arrivals, 5.0, 0.0)

        assert report["stalls"] == 3
        assert any("SLOW THROUGHOUT" in v for v in verdicts(report))

    def test_a_healthy_stream_supports_no_complaint(self) -> None:
        """An empty verdict list is an answer: nothing here is wrong."""
        frames = [_jpeg(i) for i in range(30)]
        arrivals = [i * 0.033 for i in range(30)]

        assert verdicts(summarize_stream(frames, arrivals, 1.0, 0.0)) == []

    def test_no_declared_interval_reports_zero_rather_than_guessing(self) -> None:
        """Sampling-floor counting needs the sender's rate to mean anything."""
        frames = [_jpeg(i) for i in range(5)]
        arrivals = [i * 0.083 for i in range(5)]

        assert summarize_stream(frames, arrivals, 1.0, 0.0)["at_sampling_floor"] == 0

    def test_a_single_frame_has_no_gaps_and_says_so(self) -> None:
        """One frame is a rate, not a distribution."""
        report = summarize_stream([_jpeg(1)], [0.0], 1.0, 0.0)

        assert (report["median_gap"], report["min_gap"], report["max_gap"]) == (0.0, 0.0, 0.0)
        assert report["burst_share"] == 0.0

    def test_the_median_of_an_even_count_averages_the_middle_pair(self) -> None:
        """Four frames give three gaps; six give five. Both branches run."""
        even = summarize_stream([_jpeg(i) for i in range(5)], [0.0, 0.1, 0.3, 0.6, 1.0], 1.0, 0.0)
        odd = summarize_stream([_jpeg(i) for i in range(4)], [0.0, 0.1, 0.3, 0.6], 1.0, 0.0)

        assert even["median_gap"] == pytest.approx(0.25)
        assert odd["median_gap"] == pytest.approx(0.2)

    def test_misaligned_inputs_are_refused(self) -> None:
        """A report built from mismatched lists would look plausible."""
        with pytest.raises(ValueError, match="against"):
            summarize_stream([_jpeg(1)], [0.0, 1.0], 1.0, 0.0)

    def test_a_non_positive_window_is_refused(self) -> None:
        """Every rate divides by it."""
        with pytest.raises(ValueError, match="must be positive"):
            summarize_stream([_jpeg(1)], [0.0], 0.0, 0.0)

    def test_an_empty_stream_reports_zeroes_rather_than_dividing_by_none(self) -> None:
        """Nothing arriving is a measurement, not a crash."""
        report = summarize_stream([], [], 5.0, 0.083)

        assert report["frames"] == 0
        assert report["duplicate_share"] == 0.0
        assert report["bytes_per_second"] == 0.0

    def test_the_rendered_block_carries_the_numbers_and_the_verdict(self) -> None:
        """The report is read by a human; both halves have to be in it."""
        frames = [_jpeg(1)] * 8
        arrivals = [i * 0.083 for i in range(8)]

        text = render_report(summarize_stream(frames, arrivals, 1.0, 0.083))

        assert "duplicate share   88%" in text
        assert "SENDER REPEATS ITSELF" in text
