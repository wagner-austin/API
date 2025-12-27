"""Devpost API client wrapper.

This module provides a high-level client for interacting with the Devpost API.
"""

from __future__ import annotations

from platform_devpost.types import (
    DevpostApiProtocol,
    Hackathon,
    HackathonState,
)


class DevpostClient:
    """Production Devpost API client.

    This client wraps the low-level Devpost API and provides a higher-level
    interface for listing and searching hackathons.

    Attributes:
        _api: The underlying Devpost API instance.
    """

    __slots__ = ("_api",)

    def __init__(self) -> None:
        """Initialize client with API from hooks."""
        from platform_devpost.testing import hooks

        self._api: DevpostApiProtocol = hooks.devpost_api_factory()

    def list_hackathons(
        self,
        *,
        search: str | None = None,
        state: HackathonState | None = None,
    ) -> tuple[Hackathon, ...]:
        """List hackathons with optional filters.

        Args:
            search: Optional search query.
            state: Optional state filter.

        Returns:
            Tuple of matching hackathons.
        """
        response = self._api.fetch_hackathons(search=search)
        result = response.hackathons
        if state is not None:
            result = tuple(h for h in result if h.open_state == state)
        return result

    def get_hackathon(self, hackathon_id: int) -> Hackathon | None:
        """Get a specific hackathon by ID.

        This method fetches all hackathons and searches for the one with
        the matching ID. For efficiency, consider caching results if you
        need to look up multiple hackathons.

        Args:
            hackathon_id: Hackathon identifier.

        Returns:
            Hackathon if found, None otherwise.
        """
        response = self._api.fetch_hackathons()
        for hackathon in response.hackathons:
            if hackathon.id == hackathon_id:
                return hackathon
        return None
