"""The production download hooks, run for real through their failures.

``test_hooks_defaults.py`` runs both against a local server that answers 200,
and every corpus and language-id test replaces them with fakes, so none shows
the real download refusing a missing model or the real Wikipedia fetch
raising when nothing answers. These call the implementations themselves
(``effect-seam-twin``, board task cc7222ca).
"""

from __future__ import annotations

import http.server
import socket
import socketserver
import threading
from pathlib import Path

import pytest
import requests

from turkic_api import _test_hooks


class _Missing(http.server.BaseHTTPRequestHandler):
    """Answers every GET the way a moved model URL does."""

    def do_GET(self) -> None:
        body = b"Not Found"
        self.send_response(404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: str) -> None:
        """Keep the server quiet under pytest.

        Args:
            format: The format string.
            args: Its arguments.
        """


def test_a_model_download_answered_404_is_raised_and_writes_nothing(tmp_path: Path) -> None:
    """A moved model URL must not leave a 9-byte "model" for fastText to load."""
    dest = tmp_path / "models" / "lid.218e.bin"
    with socketserver.TCPServer(("127.0.0.1", 0), _Missing) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.handle_request)
        thread.start()
        with pytest.raises(requests.HTTPError) as caught:
            _test_hooks._default_langid_download(f"http://127.0.0.1:{port}/lid.218e.bin", dest)
        thread.join(timeout=10.0)
    assert str(caught.value).startswith("404 Client Error: Not Found for url: ")
    assert not dest.exists()


def test_a_wikipedia_fetch_to_a_host_that_refuses_is_raised() -> None:
    """The port is one a listener held and released, so nothing is bound to it."""
    released = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    released.bind(("127.0.0.1", 0))
    bound: tuple[str, int] = released.getsockname()
    released.close()
    with pytest.raises(requests.ConnectionError):
        _test_hooks._default_wikipedia_requests_get(
            f"http://127.0.0.1:{bound[1]}/kkwiki-latest-pages-articles.xml.bz2",
            stream=True,
            timeout=10,
        )
