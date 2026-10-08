"""The production urllib hooks, run for real through their failures.

``test_hooks_defaults.py`` runs both against a local server that answers 200,
and every service test replaces them with fakes, so none shows the real GET
raising on a failing status or the real POST raising when nothing answers.
These call the implementations themselves (``effect-seam-twin``, board task
cc7222ca).
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import threading
import urllib.error
import urllib.request

import pytest

from music_wrapped_api import _test_hooks


class _Unavailable(http.server.BaseHTTPRequestHandler):
    """Answers every GET the way Last.fm does while it is down."""

    def do_GET(self) -> None:
        body = b'{"error": 16, "message": "temporarily unavailable"}'
        self.send_response(503)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: str) -> None:
        """Keep the server quiet under pytest.

        Args:
            fmt: The format string.
            args: Its arguments.
        """


def test_a_get_answered_with_a_failing_status_is_raised() -> None:
    """A 503 from the upstream is an HTTPError, never a body read as data."""
    with socketserver.TCPServer(("127.0.0.1", 0), _Unavailable) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.handle_request)
        thread.start()
        with pytest.raises(urllib.error.HTTPError) as caught:
            _test_hooks._default_urlopen_get(f"http://127.0.0.1:{port}/2.0/", 10.0)
        thread.join(timeout=10.0)
    assert caught.value.code == 503
    caught.value.close()


def test_a_post_to_a_host_that_refuses_is_raised() -> None:
    """The port is one a listener held and released, so nothing is bound to it."""
    released = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    released.bind(("127.0.0.1", 0))
    bound: tuple[str, int] = released.getsockname()
    released.close()
    request = urllib.request.Request(f"http://127.0.0.1:{bound[1]}/2.0/", b"method=auth")
    with pytest.raises(urllib.error.URLError):
        _test_hooks._default_urlopen_post(request, 10.0)
