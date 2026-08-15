"""platform_devpost Devpost API and client protocols."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from platform_devpost._types_hackathon import Hackathon
from platform_devpost._types_listing import HackathonListResponse
from platform_devpost._types_validation import HackathonState

# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


class DevpostApiProtocol(Protocol):
    """Protocol for Devpost API client."""

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
        """
        ...


@runtime_checkable
class DevpostClientProtocol(Protocol):
    """Protocol for Devpost client."""

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
        ...

    def get_hackathon(self, hackathon_id: int) -> Hackathon | None:
        """Get a specific hackathon by ID.

        Args:
            hackathon_id: Hackathon identifier.

        Returns:
            Hackathon if found, None otherwise.
        """
        ...
