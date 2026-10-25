"""The exchange's opener speaks http alone and builds no SSL context (MCPs board task a1f9d16d).

``urllib.request.build_opener`` adds an HTTPS handler whose construction loads
the CA bundle on Python 3.12, 40.7 ms of cpu a call on the hub. These cases
run the real opener: against a real loopback server for the scheme it
serves, and against the schemes it must refuse by name instead of carrying.
"""

from __future__ import annotations

import http.server
import threading
import urllib.error
import urllib.request
from collections.abc import Generator
from typing import Protocol

import pytest

from platform_core.mcp_exchange import ExchangeDeadline, exchange_opener


class _Answer(http.server.BaseHTTPRequestHandler):
    """Answers every POST with its body, under a 503 for ``/busy``."""

    def do_POST(self) -> None:
        """Echo the body back."""
        length = int(self.headers["Content-Length"])
        body = self.rfile.read(length)
        self.send_response(503 if self.path == "/busy" else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: str | int) -> None:
        """Keep the test's output to its own lines.

        Args:
            format: The message's format.
            *args: Its arguments.
        """


@pytest.fixture(name="server_url")
def _server_url() -> Generator[str, None, None]:
    """Run a real HTTP server on a loopback port for one case.

    Yields:
        Its base URL, with no trailing slash.
    """
    server = http.server.HTTPServer(("127.0.0.1", 0), _Answer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


class _Opened(Protocol):
    """The members a case reads off an opened URL."""

    status: int

    def read(self) -> bytes:
        """Read the whole body.

        Returns:
            The body bytes.
        """
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


def _post(url: str) -> tuple[int, bytes]:
    deadline = ExchangeDeadline(5)
    request = urllib.request.Request(url=url, data=b"payload", method="POST")
    response: _Opened = exchange_opener(deadline).open(request, timeout=5)
    answer = (response.status, response.read())
    response.close()
    assert deadline.finish() is False
    return answer


def test_an_http_exchange_answers_through_the_deadline_handler(server_url: str) -> None:
    assert _post(f"{server_url}/ok") == (200, b"payload")


def test_a_non_2xx_status_comes_back_instead_of_raising(server_url: str) -> None:
    assert _post(f"{server_url}/busy") == (503, b"payload")


@pytest.mark.parametrize("scheme", ["https", "ftp", "file"])
def test_every_other_scheme_is_refused_by_name(scheme: str) -> None:
    """No handler for it exists, so no SSL context or file read is ever reached."""
    opener = exchange_opener(ExchangeDeadline(5))

    with pytest.raises(urllib.error.URLError) as refused:
        opener.open(f"{scheme}://127.0.0.1/mcp", timeout=5)

    assert str(refused.value.reason) == f"unknown url type: {scheme}"
