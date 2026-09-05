"""The production hook implementations, exercised for real.

The fakes elsewhere assert what this package ASKS FOR. Nothing there would
notice if the real poster sent no headers, raised on a 401, or buffered its
output forever -- so these run the real implementations against a real
socket, a real directory and real standard output.

THE 401 CASE IS THE REASON THIS FILE EXISTS. ``urllib`` raises on a non-2xx
by default, and a poster that raised would turn the ordinary rotated-key
failure into a traceback with no status in it. The pass-through processor is
what prevents that, and only a real server answering 401 proves it works.

The server writes HTTP by hand on a :class:`socketserver.StreamRequestHandler`
rather than subclassing :class:`http.server.BaseHTTPRequestHandler`. That base
class requires a method named ``do_POST``, which no lowercase-name rule will
accept on a definition, and suppressing a rule to satisfy a base class is not
a trade this repository makes. Writing the status line directly needs no
suppression -- and since these tests are about status codes, an explicit
status line is a gain rather than a cost.
"""

from __future__ import annotations

import pathlib
import socketserver
import threading
from collections.abc import Generator

import pytest
from platform_core.config import config_test_hooks

from board_watch import _test_hooks

#: Body the handler answers with, echoing what it was sent.
_TEMPLATE = '{{"seen":"{body}","key":"{key}"}}'


class _Handler(socketserver.StreamRequestHandler):
    """Answers one HTTP request, echoing the body and the api-key header."""

    def handle(self) -> None:
        """Read one request and write one response.

        The path decides the status: ``/ok`` answers 200 and anything else
        answers 401, so a single server covers both branches of the poster.
        """
        request_line = self.rfile.readline().decode("utf-8").strip()
        parts = request_line.split(" ")
        path = parts[1] if len(parts) > 1 else ""
        length = 0
        api_key = ""
        while True:
            raw = self.rfile.readline().decode("utf-8").strip()
            if raw == "":
                break
            name, _, value = raw.partition(":")
            if name.lower() == "content-length":
                length = int(value.strip())
            if name.lower() == "x-api-key":
                api_key = value.strip()
        body = self.rfile.read(length).decode("utf-8")
        payload = _TEMPLATE.format(body=body, key=api_key).encode("utf-8")
        status = "200 OK" if path == "/ok" else "401 Unauthorized"
        self.wfile.write(f"HTTP/1.1 {status}\r\n".encode())
        self.wfile.write(b"Content-Type: application/json; charset=utf-8\r\n")
        self.wfile.write(f"Content-Length: {len(payload)}\r\n".encode())
        self.wfile.write(b"Connection: close\r\n\r\n")
        self.wfile.write(payload)


@pytest.fixture(name="server_url")
def _server_url() -> Generator[str, None, None]:
    """Run a real HTTP server on a loopback port for the duration of a test.

    Yields:
        The base URL, with no trailing slash.
    """
    server = socketserver.TCPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port: int = server.socket.getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


def test_the_real_poster_sends_headers_and_body(server_url: str) -> None:
    """A poster that dropped either would fail only against the live board."""
    response = _test_hooks.http_post(
        f"{server_url}/ok",
        headers={"x-api-key": "secret-value", "Content-Type": "application/json"},
        body=b"payload-bytes",
        timeout_seconds=5,
    )
    assert response["status"] == 200
    assert "payload-bytes" in response["body"]
    assert "secret-value" in response["body"]
    assert response["content_type"] == "application/json; charset=utf-8"


def test_the_real_poster_returns_a_401_instead_of_raising(server_url: str) -> None:
    """The rotated-key case, which is how this fails in practice.

    Without the pass-through error processor ``urllib`` raises here, and the
    caller could not attach ``HTTP_STATUS`` to a status it never saw.
    """
    response = _test_hooks.http_post(
        f"{server_url}/refused",
        headers={"x-api-key": "stale"},
        body=b"{}",
        timeout_seconds=5,
    )
    assert response["status"] == 401


def test_the_real_environment_reader_normalises_blank_to_unset() -> None:
    """It delegates to the monorepo's one permitted environment reader.

    Rebinding that reader's own hook rather than setting a real variable is
    what keeps this package from growing a second ``os.environ`` access, and
    it exercises the delegation rather than assuming it.
    """
    config_test_hooks.get_env = {"SET": "present", "BLANK": "   "}.get
    assert _test_hooks.env("SET") == "present"
    assert _test_hooks.env("BLANK") is None
    assert _test_hooks.env("ABSENT") is None


def test_the_real_emitter_writes_a_line_and_flushes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Monitor reads this stream, so a buffered line is an event that has not
    happened yet as far as the subscriber is concerned."""
    _test_hooks.emit("one")
    _test_hooks.emit("two")
    assert capsys.readouterr().out == "one\ntwo\n"


def test_the_real_writer_creates_missing_parents(tmp_path: pathlib.Path) -> None:
    """The default state directory does not exist on a fresh machine."""
    target = tmp_path / "a" / "b" / "c.json"
    _test_hooks.write_text(target, "body")
    assert target.read_text(encoding="utf-8") == "body"
    assert _test_hooks.read_text(target) == "body"


def test_the_real_existence_check_distinguishes_a_directory(
    tmp_path: pathlib.Path,
) -> None:
    """A directory at the cursor's path is not a cursor document."""
    directory = tmp_path / "not-a-file"
    directory.mkdir()
    assert _test_hooks.file_exists(directory) is False
    assert _test_hooks.file_exists(tmp_path / "absent.json") is False


__all__ = [
    "test_the_real_emitter_writes_a_line_and_flushes",
    "test_the_real_environment_reader_normalises_blank_to_unset",
    "test_the_real_existence_check_distinguishes_a_directory",
    "test_the_real_poster_returns_a_401_instead_of_raising",
    "test_the_real_poster_sends_headers_and_body",
    "test_the_real_writer_creates_missing_parents",
]
