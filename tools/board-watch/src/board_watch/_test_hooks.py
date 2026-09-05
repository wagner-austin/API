"""Injection points for everything this package cannot do in a test.

Module-level names, rebound by :mod:`tests.conftest` before each test and
restored after. There is no conditional anywhere in the package asking
whether it is under test: production binds the real implementations at
import and a test binds fakes, and the call site just calls the hook.

Four kinds of seam, one per thing that reaches outside the process: the
network, the environment, the filesystem holding the cursor, and standard
output. Nothing else is hooked, because nothing else leaves.

There is deliberately no clock and no sleep. This command reads once and
exits; the interval belongs to the shell loop that calls it, where it is
visible at the call site. See :mod:`board_watch.cli.watch`.
"""

from __future__ import annotations

import http.client
import pathlib
import sys
import urllib.request
from typing import Final, Protocol, TypedDict

from platform_core.config import _optional_env_str


class HttpResponse(TypedDict):
    """What a POST to the board's HTTP surface answered.

    Attributes:
        status: The HTTP status line's code.
        content_type: The ``Content-Type`` header, lowercased, or the empty
            string when the response carried none.
        body: The decoded response body.
    """

    status: int
    content_type: str
    body: str


class HttpPostProtocol(Protocol):
    """Post a JSON body and read the whole response."""

    def __call__(
        self,
        url: str,
        *,
        headers: dict[str, str],
        body: bytes,
        timeout_seconds: int,
    ) -> HttpResponse:
        """Perform the request.

        Args:
            url: Absolute URL to post to.
            headers: Every request header, already complete.
            body: The encoded request body.
            timeout_seconds: How long to wait for the whole exchange.

        Returns:
            The response.
        """
        ...


class EnvProtocol(Protocol):
    """Read one process environment variable.

    Implementations MUST normalise a variable that is set to whitespace to
    None. An exported-but-blank variable is the unset case as far as every
    caller here is concerned, and a fake that returned ``""`` where the real
    reader returns None would let a blank credential reach the board.
    """

    def __call__(self, name: str) -> str | None:
        """Read it.

        Args:
            name: The variable name.

        Returns:
            Its trimmed value, or None when unset or blank.
        """
        ...


class ReadTextProtocol(Protocol):
    """Read a UTF-8 file that is known to exist."""

    def __call__(self, path: pathlib.Path) -> str:
        """Read it.

        Args:
            path: The file.

        Returns:
            Its whole contents.
        """
        ...


class WriteTextProtocol(Protocol):
    """Write a UTF-8 file, creating its parent directory."""

    def __call__(self, path: pathlib.Path, content: str) -> None:
        """Write it.

        Args:
            path: The file.
            content: What to write.
        """
        ...


class FileExistsProtocol(Protocol):
    """Report whether a path is an existing file."""

    def __call__(self, path: pathlib.Path) -> bool:
        """Check it.

        Args:
            path: The path.

        Returns:
            True when it exists and is a file.
        """
        ...


class EmitProtocol(Protocol):
    """Write one line to the watcher's event stream."""

    def __call__(self, line: str) -> None:
        """Write it.

        Args:
            line: The line, without a trailing newline.
        """
        ...


class _ResponseHeaders(Protocol):
    """The one header lookup this package makes."""

    def get(self, name: str, failobj: str) -> str:
        """Read a header.

        Args:
            name: The header name.
            failobj: What to answer when the header is absent.

        Returns:
            The header value, or ``failobj``.
        """
        ...


class _UrlResponse(Protocol):
    """The four members this package uses from an opened URL.

    A Protocol rather than :class:`http.client.HTTPResponse` because
    ``OpenerDirector.open`` is typed loosely enough that everything read off
    its result would otherwise be ``Any``. Annotating the assignment with
    this narrows it at the boundary, which is where parsing belongs.
    """

    status: int
    headers: _ResponseHeaders

    def read(self) -> bytes:
        """Read the whole body.

        Returns:
            The body bytes.
        """
        ...

    def close(self) -> None:
        """Release the connection."""
        ...


class _PassThroughErrorProcessor(urllib.request.HTTPErrorProcessor):
    """An error processor that hands back non-2xx responses instead of raising.

    ``urllib``'s default turns a 401 into a raised
    :class:`urllib.error.HTTPError`, which would force this module to catch an
    exception in order to read a status code the caller needs. The status is
    not an exceptional condition here -- a rotated API key is the ordinary way
    this fails, and the caller's job is to attach
    ``BoardWatchErrorCode.HTTP_STATUS`` to it.

    Replacing the processor removes the exception at its source rather than
    handling it downstream, so no ``except`` appears anywhere in this package.
    """

    def http_response(
        self, request: urllib.request.Request, response: http.client.HTTPResponse
    ) -> http.client.HTTPResponse:
        """Return the response unchanged, whatever its status.

        Args:
            request: The request that produced it, unused.
            response: The response as received.

        Returns:
            That same response.
        """
        return response

    https_response = http_response


def _default_http_post(
    url: str,
    *,
    headers: dict[str, str],
    body: bytes,
    timeout_seconds: int,
) -> HttpResponse:
    """Post with the standard library.

    ``urllib`` rather than ``httpx`` because this is a single loopback POST
    from a short-lived CLI. ``platform_core.http_client`` exists for services
    that hold a client across many requests and need transports injected;
    borrowing it here would add a dependency to buy nothing.

    Args:
        url: Absolute URL to post to.
        headers: Every request header, already complete.
        body: The encoded request body.
        timeout_seconds: How long to wait for the whole exchange.

    Returns:
        The response, whatever its status.
    """
    request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
    opener = urllib.request.build_opener(_PassThroughErrorProcessor)
    # Annotated at the assignment so the Protocol, not ``urlopen``'s loose
    # return type, is what the rest of this function is checked against.
    response: _UrlResponse = opener.open(request, timeout=timeout_seconds)
    answer = HttpResponse(
        status=response.status,
        content_type=response.headers.get("Content-Type", "").lower(),
        body=response.read().decode("utf-8"),
    )
    response.close()
    return answer


def _default_env(name: str) -> str | None:
    """Read a process environment variable.

    Delegates to ``platform_core.config``, which is the monorepo's single
    permitted reader of the process environment -- the ``env`` guard rule
    names it explicitly rather than exempting it. A second reader here would
    be the fork that rule exists to prevent.

    Args:
        name: The variable name.

    Returns:
        Its trimmed value, or None when unset OR set to whitespace. The
        normalisation is the shared reader's, and it is why callers here
        test only for None.
    """
    return _optional_env_str(name)


def _default_read_text(path: pathlib.Path) -> str:
    """Read a UTF-8 file.

    Args:
        path: The file.

    Returns:
        Its whole contents.
    """
    return path.read_text(encoding="utf-8")


def _default_write_text(path: pathlib.Path, content: str) -> None:
    """Write a UTF-8 file, creating its parent directory.

    Args:
        path: The file.
        content: What to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _default_file_exists(path: pathlib.Path) -> bool:
    """Report whether a path is an existing file.

    Args:
        path: The path.

    Returns:
        True when it exists and is a file.
    """
    return path.is_file()


def _default_emit(line: str) -> None:
    """Write one line to standard output and flush it.

    The flush is required, not tidiness. Monitor reads this process's stdout
    as a stream of events, and a buffered line is an event that has not
    happened yet as far as the subscriber is concerned.

    Args:
        line: The line, without a trailing newline.
    """
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


#: How long a single board call may take before it is abandoned.
DEFAULT_TIMEOUT_SECONDS: Final = 20

http_post: HttpPostProtocol = _default_http_post
env: EnvProtocol = _default_env
read_text: ReadTextProtocol = _default_read_text
write_text: WriteTextProtocol = _default_write_text
file_exists: FileExistsProtocol = _default_file_exists
emit: EmitProtocol = _default_emit
