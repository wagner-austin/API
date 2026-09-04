"""Correlation headers, and the DELETE that two libraries each carried a copy of.

``http_delete`` is exercised against a real HTTP server on a real socket
rather than a stub. The thing worth proving is that it composes a request
urllib will send and a server will recognise as a DELETE carrying the headers
it was given -- none of which a fake request object can be wrong about,
because a fake would agree with whatever the caller did.
"""

from __future__ import annotations

import threading
import urllib.error
from collections.abc import Callable, Generator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from platform_core.http_utils import add_correlation_header, http_delete

_RECEIVED: list[tuple[str, str, dict[str, str]]] = []
"""(method, path, headers) for each request the test server handled."""

_REFUSE_PATH = "/refuse"
"""A path the server answers 404, so the failure path is a real 404."""


class _Handler(BaseHTTPRequestHandler):
    """Records what arrived and answers."""

    def record_and_answer(self) -> None:
        """Record the request and answer it."""
        _RECEIVED.append((self.command, self.path, dict(self.headers.items())))
        self.send_response(404 if self.path == _REFUSE_PATH else 204)
        self.end_headers()

    def __getattr__(self, name: str) -> Callable[[], None]:
        """Answer the stdlib's lookup for ``do_DELETE``.

        BaseHTTPRequestHandler dispatches by looking up ``do_`` plus the HTTP
        method, so the handler has to answer to a name no naming convention
        here allows. Resolved on lookup rather than defined under that name,
        which satisfies the stdlib's contract without a suppression.

        Args:
            name: The attribute the stdlib is looking for.

        Returns:
            The recording handler, for the one name this serves.

        Raises:
            AttributeError: For every other name, as normal lookup would.
        """
        if name == "do_DELETE":
            return self.record_and_answer
        raise AttributeError(name)

    def log_message(self, format: str, *args: str | int) -> None:
        """Silence the server's stderr logging during tests.

        Args:
            format: The stdlib's printf-style template.
            args: Its substitutions, which this handler passes on to nothing.
        """
        return


@pytest.fixture()
def server_url() -> Generator[str, None, None]:
    """Run a real HTTP server for one test.

    Yields:
        The base URL the server is listening on.
    """
    _RECEIVED.clear()
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class TestAddCorrelationHeader:
    def test_it_sets_the_header_on_a_copy(self) -> None:
        original = {"Accept": "application/json"}

        result = add_correlation_header(original, "abc123")

        assert result == {"Accept": "application/json", "X-Request-ID": "abc123"}
        assert original == {"Accept": "application/json"}

    def test_none_headers_yield_just_the_correlation_header(self) -> None:
        assert add_correlation_header(None, "abc123") == {"X-Request-ID": "abc123"}

    def test_the_header_name_is_configurable(self) -> None:
        result = add_correlation_header({}, "abc123", header_name="X-Trace")

        assert result == {"X-Trace": "abc123"}


class TestHttpDelete:
    """One implementation, proven against a socket."""

    def test_it_sends_a_delete_to_the_url(self, server_url: str) -> None:
        http_delete(f"{server_url}/events/42", {})

        method, path, _ = _RECEIVED[0]
        assert (method, path) == ("DELETE", "/events/42")

    def test_it_sends_the_headers_it_was_given(self, server_url: str) -> None:
        """Authorization is the whole reason a caller passes headers here; a
        delete that arrived unauthenticated would 401 rather than delete."""
        http_delete(f"{server_url}/events/42", {"Authorization": "Bearer token"})

        _, _, headers = _RECEIVED[0]
        assert headers["Authorization"] == "Bearer token"

    def test_a_refusal_propagates(self, server_url: str) -> None:
        """A delete that failed and reported success is the failure this call
        exists to make visible, so the 404 is not swallowed."""
        with pytest.raises(urllib.error.HTTPError):
            http_delete(f"{server_url}{_REFUSE_PATH}", {})

    def test_an_unreachable_host_propagates(self) -> None:
        with pytest.raises(urllib.error.URLError):
            http_delete("http://127.0.0.1:1/gone", {})


__all__ = ["TestAddCorrelationHeader", "TestHttpDelete"]
