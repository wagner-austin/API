"""Production implementations for Devpost API access.

This module contains production implementations that use the hooks system
for testability. Tests can override hooks to provide fakes.
"""

from __future__ import annotations

from collections.abc import Callable

from platform_core.http_client import HttpxClient, SyncTransport, build_client
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from platform_devpost.types import (
    DevpostApiProtocol,
    DevpostClientProtocol,
    HackathonListResponse,
    decode_list_response,
)

# -----------------------------------------------------------------------------
# HTTP Client Builder Hook
# -----------------------------------------------------------------------------


_http_client_builder: Callable[[float, SyncTransport | None], HttpxClient] = build_client


def _set_http_client_builder(builder: Callable[[float, SyncTransport | None], HttpxClient]) -> None:
    """Set HTTP client builder for testing.

    Args:
        builder: Function that creates HTTP client.
    """
    global _http_client_builder
    _http_client_builder = builder


def _reset_http_client_builder() -> None:
    """Reset HTTP client builder to production implementation."""
    global _http_client_builder
    _http_client_builder = build_client


# -----------------------------------------------------------------------------
# HTTP-based Devpost API
# -----------------------------------------------------------------------------

DEVPOST_API_URL = "https://devpost.com/api/hackathons"
DEFAULT_TIMEOUT_SECONDS = 30.0


class _HttpDevpostApi:
    """Production HTTP-based Devpost API.

    This class fetches hackathon data from the Devpost API using httpx.
    """

    __slots__ = ("_client",)

    def __init__(self) -> None:
        """Initialize the HTTP client."""
        self._client: HttpxClient = _http_client_builder(DEFAULT_TIMEOUT_SECONDS, None)

    def fetch_hackathons(
        self,
        *,
        page: int = 1,
        search: str | None = None,
    ) -> HackathonListResponse:
        """Fetch hackathons from Devpost API.

        Args:
            page: Page number (1-indexed).
            search: Optional search query.

        Returns:
            HackathonListResponse with hackathons and metadata.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        params: dict[str, str | int] = {"page": page}
        if search is not None:
            params["search"] = search

        response = self._client.get(DEVPOST_API_URL, params=params)
        response.raise_for_status()
        data = narrow_json_to_dict(load_json_str(response.text))

        result: HackathonListResponse = decode_list_response(data)
        return result


def create_devpost_api() -> DevpostApiProtocol:
    """Create production Devpost API client.

    Returns:
        DevpostApiProtocol instance.
    """
    api: DevpostApiProtocol = _HttpDevpostApi()
    return api


def make_devpost_client() -> DevpostClientProtocol:
    """Production factory for DevpostClient.

    Returns:
        DevpostClientProtocol instance.
    """
    from platform_devpost.client import DevpostClient

    client: DevpostClientProtocol = DevpostClient()
    return client
