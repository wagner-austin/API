from __future__ import annotations

from collections.abc import Mapping

_DELETE_TIMEOUT_SECONDS = 30
"""How long to wait on a delete before giving up, in seconds."""


def add_correlation_header(
    headers: Mapping[str, str] | None,
    request_id: str,
    *,
    header_name: str = "X-Request-ID",
) -> dict[str, str]:
    """Return a copy of headers including a correlation header.

    Accepts any mapping for input and returns a new plain dict with the
    correlation header set to the provided request_id.
    """
    base: dict[str, str] = dict(headers or {})
    base[header_name] = request_id
    return base


def http_delete(url: str, headers: dict[str, str]) -> None:
    """Send a DELETE request and discard the response body.

    Written once here because ``platform_calendar`` and ``platform_email``
    each carried a byte-identical copy in their production hooks. Both used
    urllib rather than this workspace's httpx client, which is the reason it
    is worth keeping as its own function rather than folding into
    :mod:`platform_core.http_client`: it is the dependency-free path a
    production hook can take without pulling the async stack into a CLI.

    Args:
        url: The resource to delete.
        headers: Headers to send, typically carrying authorization.

    Raises:
        urllib.error.HTTPError: If the server refuses. Not caught: a delete
            that failed and reported success is the failure this call exists
            to make visible.
        urllib.error.URLError: If the request cannot be made at all.
    """
    import urllib.request
    from http.client import HTTPResponse

    request = urllib.request.Request(url, method="DELETE")
    for key, value in headers.items():
        request.add_header(key, value)
    response = urllib.request.urlopen(request, timeout=_DELETE_TIMEOUT_SECONDS)
    assert isinstance(response, HTTPResponse)
    response.close()


__all__ = ["add_correlation_header", "http_delete"]
