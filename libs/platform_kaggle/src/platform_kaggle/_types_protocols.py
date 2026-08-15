"""types: KaggleTagProtocol and related definitions."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from platform_kaggle._types_competition import Competition
from platform_kaggle._types_pages import CompetitionPages
from platform_kaggle._types_validation import CompetitionCategory

# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


class KaggleTagProtocol(Protocol):
    """Protocol for Kaggle API tag object (ApiCategory)."""

    ref: str


class KaggleCompetitionProtocol(Protocol):
    """Protocol for Kaggle API competition object (ApiCompetition).

    The real Kaggle API returns ApiCompetition objects with these attributes.
    Types match the real kagglesdk.competitions.types.ApiCompetition.
    """

    @property
    def ref(self) -> str:
        """Competition reference URL (full Kaggle URL)."""
        ...

    @property
    def title(self) -> str:
        """Competition title."""
        ...

    @property
    def category(self) -> str:
        """Competition category."""
        ...

    @property
    def reward(self) -> str:
        """Prize description."""
        ...

    @property
    def deadline(self) -> datetime:
        """Deadline (datetime object from Kaggle API)."""
        ...

    @property
    def team_count(self) -> int:
        """Number of teams."""
        ...

    @property
    def tags(self) -> Sequence[KaggleTagProtocol | None] | None:
        """Competition tags (may contain None items or be None)."""
        ...

    @property
    def description(self) -> str:
        """Short description."""
        ...

    @property
    def url(self) -> str:
        """Full Kaggle URL."""
        ...


class CompetitionsResponseProtocol(Protocol):
    """Protocol for competitions_list response (new Kaggle API format)."""

    @property
    def competitions(self) -> Sequence[KaggleCompetitionProtocol | None] | None:
        """Get sequence of competition objects (may contain None or be None)."""
        ...


class KaggleApiProtocol(Protocol):
    """Protocol for Kaggle API instance."""

    def authenticate(self) -> None:
        """Authenticate with Kaggle API using credentials."""
        ...

    def competitions_list(
        self,
        group: str | None = None,
        category: str | None = None,
        sort_by: str | None = None,
        page: int | None = None,
        search: str | None = None,
        page_size: int | None = None,
        page_token: str | None = None,
    ) -> CompetitionsResponseProtocol | None:
        """List competitions with optional filters.

        Args:
            group: Competition group filter.
            category: Category filter.
            sort_by: Sort order.
            page: Page number.
            search: Search query.
            page_size: Results per page.
            page_token: Pagination token.

        Returns:
            Response wrapper with competitions list, or None.
        """
        ...


class KaggleApiClassProtocol(Protocol):
    """Protocol for KaggleApi class (not instance)."""

    def __call__(self) -> KaggleApiProtocol:
        """Instantiate a new KaggleApi.

        Returns:
            New unauthenticated KaggleApiProtocol instance.
        """
        ...


class KaggleModuleProtocol(Protocol):
    """Protocol for the kaggle API module."""

    KaggleApi: KaggleApiClassProtocol


class KagglePreAuthModuleProtocol(Protocol):
    """Protocol for kaggle module with pre-authenticated global api.

    The kaggle package creates and authenticates a global `api` object
    at import time in its __init__.py. This protocol allows typed access.
    """

    api: KaggleApiProtocol


class KaggleApiFactoryProtocol(Protocol):
    """Protocol for Kaggle API factory."""

    def __call__(self) -> KaggleApiProtocol:
        """Create authenticated Kaggle API instance.

        Returns:
            Authenticated KaggleApiProtocol instance.
        """
        ...


@runtime_checkable
class KaggleClientProtocol(Protocol):
    """Protocol for Kaggle API client."""

    def list_competitions(
        self,
        *,
        search: str | None = None,
        category: CompetitionCategory | None = None,
    ) -> tuple[Competition, ...]:
        """List active competitions with optional filters.

        Args:
            search: Optional search query.
            category: Optional category filter.

        Returns:
            Tuple of matching competitions.
        """
        ...

    def get_competition(self, ref: str) -> Competition | None:
        """Get a specific competition by ref.

        Args:
            ref: Competition reference slug.

        Returns:
            Competition if found, None otherwise.
        """
        ...


@runtime_checkable
class KagglePageFetcherProtocol(Protocol):
    """Protocol for fetching competition pages from Kaggle's internal API."""

    def fetch_pages(self, competition_id: int) -> CompetitionPages:
        """Fetch all pages for a competition.

        Args:
            competition_id: Numeric Kaggle competition ID.

        Returns:
            CompetitionPages containing all page content.

        Raises:
            RuntimeError: If the API request fails.
        """
        ...

    def get_competition_id(self, slug: str) -> int:
        """Get the numeric competition ID from a slug.

        Args:
            slug: Competition slug (e.g., "google-gemma-3n-hackathon").

        Returns:
            Numeric competition ID.

        Raises:
            RuntimeError: If the API request fails.
            JSONTypeError: If response parsing fails.
        """
        ...
