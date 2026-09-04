"""Tests for ``tankpit-stream-probe``.

The stream is supplied through the ``open_http_stream`` hook, so the
argument parsing, the incremental read, the clock arithmetic and the
rendering all execute against bytes a test wrote. No socket, no bot.
"""

from __future__ import annotations

import threading
from collections.abc import Generator
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from scripts.stream_probe import DEFAULT_SECONDS, DEFAULT_URL, collect, main, parse_args

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot._test_hooks.http_stream import (
    _HttpClientStream,
    _real_open_http_stream,
    resolve_target,
)

BOUNDARY = b"--tankpitbotframe"
CONTENT_TYPE = "multipart/x-mixed-replace; boundary=tankpitbotframe"


def _part(marker: int) -> bytes:
    """One multipart part carrying a distinguishable pseudo-JPEG.

    Args:
        marker: Byte that makes this frame differ from another.

    Returns:
        The encoded part.
    """
    body = b"\xff\xd8\xff" + bytes([marker]) * 24
    return BOUNDARY + b"\r\nContent-Type: image/jpeg\r\n\r\n" + body + b"\r\n"


class _FakeStream:
    """An :class:`HttpStreamProtocol` over a fixed chunk list."""

    def __init__(
        self,
        chunks: list[bytes],
        content_type: str = CONTENT_TYPE,
        status: int = 200,
    ) -> None:
        """Bind the stream to the bytes it will hand out.

        Args:
            chunks: Returned one per ``read``; exhaustion returns b"".
            content_type: The header the reader takes its boundary from.
            status: The response status the probe checks first.
        """
        self._chunks = list(chunks)
        self._content_type = content_type
        self._status = status
        self.closed = 0

    @property
    def status(self) -> int:
        """The bound status.

        Returns:
            The status code.
        """
        return self._status

    @property
    def content_type(self) -> str:
        """The bound content type.

        Returns:
            The header value.
        """
        return self._content_type

    def read(self, size: int) -> bytes:
        """Return the next chunk.

        Args:
            size: Ignored; the test controls chunking directly.

        Returns:
            The next chunk, or empty when exhausted.
        """
        _ = size
        return self._chunks.pop(0) if self._chunks else b""

    def close(self) -> None:
        """Record one release."""
        self.closed += 1


class _StepClock:
    """A millisecond clock that advances a fixed step per reading."""

    def __init__(self, step_ms: int) -> None:
        """Start at zero.

        Args:
            step_ms: Milliseconds added on every call.
        """
        self.now = 0
        self._step = step_ms

    def __call__(self) -> int:
        """Read and advance.

        Returns:
            The current millisecond reading.
        """
        value = self.now
        self.now += self._step
        return value


@pytest.fixture()
def hooked() -> Generator[list[_FakeStream], None, None]:
    """Swap the stream opener and clock, restoring both afterwards.

    Yields:
        A list the test appends its stream to before running the probe.
    """
    streams: list[_FakeStream] = []
    original_open = core_hooks.open_http_stream
    original_clock = core_hooks.get_current_time_ms
    original_argv = core_hooks.get_argv
    core_hooks.open_http_stream = lambda url: streams[0]
    yield streams
    core_hooks.open_http_stream = original_open
    core_hooks.get_current_time_ms = original_clock
    core_hooks.get_argv = original_argv


class TestParseArgs:
    def test_no_arguments_uses_the_demo_slot(self) -> None:
        """The default target is the thing an operator usually means."""
        assert parse_args([]) == (DEFAULT_URL, DEFAULT_SECONDS, 0.0)

    def test_every_flag_is_read(self) -> None:
        """All three together, since they are parsed in one loop."""
        assert parse_args(["--url", "http://x/v", "--seconds", "5", "--fps", "30"]) == (
            "http://x/v",
            5.0,
            30.0,
        )

    def test_an_unknown_flag_is_refused_with_usage(self) -> None:
        """Silently ignoring it would measure something else."""
        with pytest.raises(ValueError, match="unknown argument"):
            parse_args(["--windows", "3"])

    def test_a_flag_without_its_value_is_refused(self) -> None:
        """A trailing flag would otherwise read past the end."""
        with pytest.raises(ValueError, match="needs a value"):
            parse_args(["--seconds"])

    def test_a_non_positive_number_is_refused(self) -> None:
        """Every rate divides by the window."""
        with pytest.raises(ValueError, match="must be positive"):
            parse_args(["--seconds", "0"])

    def test_a_non_numeric_value_is_refused(self) -> None:
        """``--fps fast`` is a typo, not a request."""
        with pytest.raises(ValueError):
            parse_args(["--fps", "fast"])


class TestCollect:
    def test_frames_are_timestamped_as_they_arrive(self, hooked: list[_FakeStream]) -> None:
        """Arrival order and count survive an arbitrary chunking.

        The parts are split across reads at boundaries that do not
        align with them, which is what a socket does.
        """
        stream = b"".join(_part(i) for i in range(4)) + BOUNDARY
        hooked.append(_FakeStream([stream[:40], stream[40:90], stream[90:]]))
        core_hooks.get_current_time_ms = _StepClock(100)

        frames, arrivals, elapsed = collect("http://x/v", 60.0)

        assert len(frames) == 4
        assert len(arrivals) == 4
        assert elapsed > 0
        assert hooked[0].closed == 1

    def test_the_window_ends_the_read(self, hooked: list[_FakeStream]) -> None:
        """A live stream never ends, so the clock has to stop it."""
        endless = [_part(1)] * 100
        hooked.append(_FakeStream(endless))
        core_hooks.get_current_time_ms = _StepClock(400)

        frames, _, elapsed = collect("http://x/v", 1.0)

        assert elapsed >= 1.0
        assert len(frames) < 100
        assert hooked[0].closed == 1

    def test_an_ended_stream_returns_what_it_had(self, hooked: list[_FakeStream]) -> None:
        """Exhaustion before the window is not an error."""
        hooked.append(_FakeStream([_part(1) + _part(2) + BOUNDARY]))
        core_hooks.get_current_time_ms = _StepClock(10)

        frames, _, _ = collect("http://x/v", 60.0)

        assert len(frames) == 2
        assert hooked[0].closed == 1

    def test_a_refusal_is_named_by_its_status_not_by_the_boundary(self) -> None:
        """A 404 is a valid text/plain response, and must read as a 404.

        Measured against the live endpoint: a demo slot whose bot had
        ended answered 404, and the probe reported "no boundary in
        content type text/plain". That blames the stream for a target
        that was simply not there, which is the same misdirection the
        https bug produced.
        """
        streams: list[_FakeStream] = [_FakeStream([], content_type="text/plain", status=404)]
        original = core_hooks.open_http_stream
        core_hooks.open_http_stream = lambda url: streams[0]
        core_hooks.get_current_time_ms = _StepClock(10)
        try:
            with pytest.raises(ValueError, match="answered 404, not a stream"):
                collect("http://x/v", 5.0)
        finally:
            core_hooks.open_http_stream = original

        assert streams[0].closed == 1

    def test_a_response_without_a_boundary_is_refused_and_still_closed(
        self, hooked: list[_FakeStream]
    ) -> None:
        """The connection is released even when the header is wrong."""
        hooked.append(_FakeStream([_part(1)], content_type="image/jpeg"))
        core_hooks.get_current_time_ms = _StepClock(10)

        with pytest.raises(ValueError, match="no boundary"):
            collect("http://x/v", 5.0)

        assert hooked[0].closed == 1


class TestMain:
    def test_it_prints_the_report(
        self, hooked: list[_FakeStream], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The measured block reaches stdout."""
        hooked.append(_FakeStream([b"".join(_part(i) for i in range(6)) + BOUNDARY]))
        core_hooks.get_current_time_ms = _StepClock(50)
        core_hooks.get_argv = lambda: ["tankpit-stream-probe", "--url", "http://x/v"]

        main()

        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "http://x/v"
        assert lines[2] == "frames            6 = 40.00/s"

    def test_an_empty_stream_says_so_rather_than_dividing_by_nothing(
        self, hooked: list[_FakeStream], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """No frames is a result, and it is reported as one."""
        hooked.append(_FakeStream([]))
        core_hooks.get_current_time_ms = _StepClock(50)
        core_hooks.get_argv = lambda: ["tankpit-stream-probe"]

        main()

        head = capsys.readouterr().out.splitlines()[0]
        assert head.split(" in ")[0] == f"no frames from {DEFAULT_URL}"

    def test_a_bad_command_line_exits_two(
        self, hooked: list[_FakeStream], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Usage errors are distinguishable from a measured result."""
        _ = hooked
        core_hooks.get_argv = lambda: ["tankpit-stream-probe", "--nope", "1"]

        with pytest.raises(SystemExit) as exit_info:
            main()

        assert exit_info.value.code == 2
        assert capsys.readouterr().err.splitlines()[0] == "unknown argument '--nope'"

    def test_a_declared_fps_reaches_the_sampling_floor_count(
        self, hooked: list[_FakeStream], capsys: pytest.CaptureFixture[str]
    ) -> None:
        """--fps is the flag that turns rate into a diagnosis."""
        hooked.append(_FakeStream([_part(i) for i in range(8)] + [BOUNDARY]))
        core_hooks.get_current_time_ms = _StepClock(50)
        core_hooks.get_argv = lambda: ["tankpit-stream-probe", "--fps", "10"]

        main()

        floor_line = capsys.readouterr().out.splitlines()[8]
        assert floor_line == "at sampling floor 7"


class TestRealHttpStream:
    """The production hook, against a real loopback server.

    Everything above substitutes the stream. This exercises the socket
    path -- the request, the typed response, the header read, the
    chunked read and the release -- because a hook nothing ever runs is
    a hook that can be wrong in production and green in CI.
    """

    @staticmethod
    def _serve(body: bytes, content_type: str) -> tuple[int, HTTPServer]:
        """Start a one-response server on an ephemeral loopback port.

        Args:
            body: Bytes to write as the response body.
            content_type: Value for the ``Content-Type`` header.

        Returns:
            The bound port and the server, for the caller to shut down.
        """

        class _Handler(BaseHTTPRequestHandler):
            # Bounded, so a peer that opens a socket and says nothing
            # cannot park this handler forever. The stdlib default is
            # no timeout at all.
            timeout = 5.0

            def do_GET(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: str) -> None:
                """Swallow the access log.

                An earlier note here said silencing this would need an
                ``object``-typed signature the typing guard rejects.
                It does not: the base signature is ``*args: Any``, and
                mypy accepts a narrower override against ``Any``. The
                stderr line it writes comes back out of ``make check``
                wrapped in a PS 5.1 NativeCommandError record, which
                reads as a failure in a passing run.

                Args:
                    format: Printf-style template; ignored.
                    args: Template arguments; ignored.
                """
                _ = (format, args)

        server = HTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return server.server_address[1], server

    def test_it_reads_a_real_response_and_reports_its_content_type(self) -> None:
        """The header and the body both come off the wire."""
        payload = _part(1) + _part(2) + BOUNDARY
        port, server = self._serve(payload, CONTENT_TYPE)
        chunks: list[bytes] = []
        # The concrete class, not the protocol: the context-manager pair
        # is part of the implementation and the protocol deliberately
        # does not require it, so `with` has to name the real type. It
        # is handed an already-open connection because that is now its
        # contract — dialling belongs to the opener, which owns the
        # socket until this object takes it.
        connection = HTTPConnection("127.0.0.1", port, timeout=30.0)
        connection.request("GET", "/video?x=1")
        try:
            with _HttpClientStream(connection, connection.getresponse()) as stream:
                assert stream.status == 200
                assert stream.content_type == CONTENT_TYPE
                while True:
                    chunk = stream.read(16)
                    if not chunk:
                        break
                    chunks.append(chunk)
        finally:
            server.shutdown()
            server.server_close()

        assert b"".join(chunks) == payload

    def test_a_response_without_a_content_type_reads_as_empty(self) -> None:
        """An absent header is the empty string, not a crash.

        The caller then fails on the missing boundary, which names the
        real problem rather than an attribute error.
        """
        port, server = self._serve(b"x", "")
        try:
            stream = _real_open_http_stream(f"http://127.0.0.1:{port}/")
            content_type = stream.content_type
            stream.close()
        finally:
            server.shutdown()
            server.server_close()

        assert content_type == ""

    def test_an_https_url_resolves_to_tls_on_443(self) -> None:
        """The bug that shipped: https fell through to cleartext on 80.

        The probe's whole point is measuring the PUBLIC stream, which is
        https, and the failure was silent in the worst way: the request
        reached an endpoint, came back as a text/plain error page, and
        the probe then blamed the stream for having no multipart
        boundary.
        """
        assert resolve_target("https://tankpit.austinwagner.org/demo/video/demo-1") == (
            True,
            "tankpit.austinwagner.org",
            443,
            "/demo/video/demo-1",
        )

    def test_plain_http_keeps_port_80_and_no_tls(self) -> None:
        """The loopback case, which is the other half of the branch."""
        assert resolve_target("http://127.0.0.1:27300/demo/video/demo-1?a=2") == (
            False,
            "127.0.0.1",
            27300,
            "/demo/video/demo-1?a=2",
        )

    def test_an_empty_path_becomes_root(self) -> None:
        """``http://host`` is a legal URL and must still request something."""
        assert resolve_target("http://example.org")[3] == "/"

    def test_a_non_http_scheme_is_refused(self) -> None:
        """A ws:// or file:// URL is a caller error, named here."""
        with pytest.raises(ValueError, match="scheme must be http or https"):
            _real_open_http_stream("ws://127.0.0.1:1/video")

    def test_a_refused_connection_closes_the_socket_it_opened(self) -> None:
        """The open owns the socket until the stream takes it.

        ``_HttpClientStream`` used to dial inside ``__init__``, so a
        refusal raised with the socket already created and the
        half-built object discarded — nothing closed it but the garbage
        collector, whenever it got round to it, with a
        ``ResourceWarning``. The dial now lives in the opener, which
        closes what it opened when the handover never happens.

        Bound to a real refused port rather than a fabricated failure:
        the branch under test is the one a warming child hits every
        time it is polled before its service binds, which is the
        ordinary case for this probe.
        """
        # A port this test held and then released: known, and provably
        # empty, which a hardcoded number cannot promise on a machine
        # running a fleet and several MCP services.
        port, server = self._serve(b"", "text/plain")
        server.shutdown()
        server.server_close()

        with pytest.raises(OSError):
            _real_open_http_stream(f"http://127.0.0.1:{port}/video")

    def test_a_url_without_a_host_is_refused(self) -> None:
        """A caller bug, named where it happens.

        Left to http.client this surfaces much later as a connection
        error against an empty host, which reads as the server being
        down rather than the URL being wrong.
        """
        with pytest.raises(ValueError, match="missing host"):
            _real_open_http_stream("not-a-url")
