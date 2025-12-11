"""Fake HTTP transport for testing artifact_store.

This module lives in tests/ so it can import httpx directly,
avoiding the httpx-direct-import guard rule in src/ files.
"""

from __future__ import annotations

import httpx
from platform_core.json_utils import JSONValue, dump_json_str


class FakeHttpTransport(httpx.BaseTransport):
    """Fake HTTP transport that returns pre-configured responses.

    This allows testing real DataBankClient and ArtifactStore code
    by simulating HTTP responses at the transport layer.
    """

    def __init__(self) -> None:
        self._responses: list[tuple[str, str, int, bytes, dict[str, str]]] = []
        self._request_log: list[httpx.Request] = []

    def add_response(
        self,
        method: str,
        path_contains: str,
        status_code: int,
        *,
        body: bytes = b"",
        json_body: dict[str, JSONValue] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Add a response to return for matching requests."""
        if json_body is not None:
            body = dump_json_str(json_body).encode()
            headers = headers or {}
            headers.setdefault("content-type", "application/json")
        self._responses.append(
            (
                method.upper(),
                path_contains,
                status_code,
                body,
                headers or {},
            )
        )

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Handle a request by returning a matching pre-configured response."""
        self._request_log.append(request)

        for method, path_contains, status_code, body, headers in self._responses:
            if request.method == method and path_contains in str(request.url):
                return httpx.Response(
                    status_code=status_code,
                    content=body,
                    headers=headers,
                    request=request,
                )

        # No matching response configured
        return httpx.Response(
            status_code=500,
            content=b"No fake response configured for this request",
            request=request,
        )

    @property
    def requests(self) -> list[httpx.Request]:
        """Get log of all requests made."""
        return self._request_log

    def clear(self) -> None:
        """Clear responses and request log."""
        self._responses.clear()
        self._request_log.clear()


__all__ = ["FakeHttpTransport"]
