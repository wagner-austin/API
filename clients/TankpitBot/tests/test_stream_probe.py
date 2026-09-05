"""The HLS stream probe: argument grammar, collection loop, and report."""

from __future__ import annotations

from collections.abc import Generator

import pytest
from scripts.stream_probe import (
    _USAGE,
    DEFAULT_INTERVAL,
    DEFAULT_SECONDS,
    DEFAULT_URL,
    collect,
    main,
    parse_args,
)

from scripts import _test_hooks as script_hooks
from tankpit_bot import _test_hooks as core_hooks

_PLAYLIST = (
    b"#EXTM3U\n#EXT-X-TARGETDURATION:2\n#EXT-X-MEDIA-SEQUENCE:5\n#EXTINF:2.0,\nseg00005.ts\n"
)


@pytest.fixture(autouse=True)
def _restore_script_hooks() -> Generator[None, None, None]:
    """Put the scripts-package hooks back after every test.

    The central ``_restore_hooks`` fixture covers the process-wide
    hooks these tests also touch (``get_argv``,
    ``get_current_time_ms``); the two scripts-package seams are this
    file's own to guard.

    Yields:
        Nothing — the fixture exists for its restore side.
    """
    original_http_get = script_hooks.http_get
    original_sleep = script_hooks.sleep_seconds
    yield
    script_hooks.http_get = original_http_get
    script_hooks.sleep_seconds = original_sleep


class _FakeResponse:
    """One HTTP answer with the two fields the probe reads."""

    def __init__(self, status_code: int, content: bytes) -> None:
        """Bind status and body.

        Args:
            status_code: HTTP status.
            content: Body bytes.
        """
        self.status_code = status_code
        self.content = content


class _ScriptedHttp:
    """``http_get`` over a scripted answer sequence."""

    def __init__(self, answers: list[_FakeResponse]) -> None:
        """Bind the answers, served in order.

        Args:
            answers: One per expected call.
        """
        self._answers = answers
        self.urls: list[str] = []

    def __call__(self, url: str) -> _FakeResponse:
        """Serve the next answer.

        Args:
            url: Recorded for assertion.

        Returns:
            The next scripted response.
        """
        self.urls.append(url)
        return self._answers[len(self.urls) - 1]


class _MsClock:
    """``get_current_time_ms`` advancing a fixed step per read."""

    def __init__(self, step_ms: int) -> None:
        """Start at zero.

        Args:
            step_ms: Milliseconds each read advances.
        """
        self._now = 0
        self._step = step_ms

    def __call__(self) -> int:
        """Read and advance.

        Returns:
            The pre-advance reading.
        """
        now = self._now
        self._now += self._step
        return now


def test_the_real_sleep_hook_blocks_for_real() -> None:
    """``_real_sleep_seconds`` spends actual wall clock, observed once.

    50 ms rather than 1: this box's clock read identical values across
    a 1 ms sleep under a loaded xdist run (2026-09-05), and a duration
    past the ~15.6 ms Windows scheduler tick makes the strict
    inequality a fact rather than a race.
    """
    from tankpit_bot.stream import _test_hooks as stream_hooks

    before = stream_hooks._real_monotonic_seconds()
    script_hooks._real_sleep_seconds(0.05)
    after = stream_hooks._real_monotonic_seconds()
    assert after > before


class TestParseArgs:
    """The command-line grammar, refusals included."""

    def test_defaults(self) -> None:
        """No arguments measure the demo slot for the default window."""
        assert parse_args([]) == (DEFAULT_URL, DEFAULT_SECONDS, DEFAULT_INTERVAL)

    def test_all_flags(self) -> None:
        """Every flag replaces its default."""
        url, seconds, interval = parse_args(
            ["--url", "http://x/index.m3u8", "--seconds", "12", "--interval", "0.25"]
        )
        assert url == "http://x/index.m3u8"
        assert seconds == 12.0
        assert interval == 0.25

    def test_unknown_flag_is_refused(self) -> None:
        """A typo'd flag is an error, not a silently different run."""
        with pytest.raises(ValueError, match="unknown argument"):
            parse_args(["--fps", "30"])

    def test_flag_without_value_is_refused(self) -> None:
        """A dangling flag is an error."""
        with pytest.raises(ValueError, match="needs a value"):
            parse_args(["--seconds"])

    def test_non_positive_number_is_refused(self) -> None:
        """Zero seconds is not a measurement."""
        with pytest.raises(ValueError, match="must be positive"):
            parse_args(["--seconds", "0"])

    def test_non_numeric_number_propagates(self) -> None:
        """A malformed number fails loudly."""
        with pytest.raises(ValueError):
            parse_args(["--interval", "fast"])


class TestCollect:
    """The sampling loop against scripted answers and clocks."""

    def test_collects_until_the_window_closes(self) -> None:
        """Each poll records status + parsed playlist, then sleeps."""
        http = _ScriptedHttp(
            [
                _FakeResponse(503, b"warming"),
                _FakeResponse(200, _PLAYLIST),
            ]
        )
        script_hooks.http_get = http
        slept: list[float] = []

        def record_sleep(seconds: float) -> None:
            slept.append(seconds)

        script_hooks.sleep_seconds = record_sleep
        # Reads: window-check, at_ms, window-check, at_ms, window-check(closes).
        core_hooks.get_current_time_ms = _MsClock(500)

        samples = collect("http://x/index.m3u8", 2.0, 0.5)

        assert http.urls == ["http://x/index.m3u8", "http://x/index.m3u8"]
        assert [sample["status"] for sample in samples] == [503, 200]
        assert samples[0]["playlist"] is None
        playlist = samples[1]["playlist"]
        if playlist is None:
            raise AssertionError("the 200 sample must carry a parsed playlist")
        assert playlist["media_sequence"] == 5
        assert slept == [0.5, 0.5]

    def test_a_200_that_is_not_a_playlist_propagates(self) -> None:
        """A misrouted URL fails the run rather than producing a report."""
        script_hooks.http_get = _ScriptedHttp([_FakeResponse(200, b"<html></html>")])
        script_hooks.sleep_seconds = _noop_sleep
        core_hooks.get_current_time_ms = _MsClock(100)

        with pytest.raises(ValueError, match="missing #EXTM3U"):
            collect("http://x/wrong", 2.0, 0.5)


def _noop_sleep(seconds: float) -> None:
    """Sleep hook that spends no wall clock.

    Args:
        seconds: Ignored.
    """
    del seconds


class TestMain:
    """Entry point wiring: argv, exit codes, report on stdout."""

    def test_bad_arguments_exit_two(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The usage complaint reaches stderr and the exit code says so."""
        core_hooks.get_argv = lambda: ["tankpit-stream-probe", "--nope"]
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 2
        assert capsys.readouterr().err == f"unknown argument '--nope'\n\n{_USAGE}\n"

    def test_a_measurement_prints_the_report(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A completed window renders the URL and the numbers."""
        core_hooks.get_argv = lambda: [
            "tankpit-stream-probe",
            "--url",
            "http://x/index.m3u8",
            "--seconds",
            "1",
        ]
        script_hooks.http_get = _ScriptedHttp([_FakeResponse(200, _PLAYLIST)])
        script_hooks.sleep_seconds = _noop_sleep
        core_hooks.get_current_time_ms = _MsClock(400)

        main()

        out = capsys.readouterr().out
        assert "http://x/index.m3u8" in out
        assert "playlist samples: 1" in out
        assert "target duration: 2.0s" in out
