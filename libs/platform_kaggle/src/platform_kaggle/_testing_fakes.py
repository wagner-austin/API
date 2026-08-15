"""testing: FakeApiTag and related definitions."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from platform_kaggle.types import (
    Competition,
    CompetitionCategory,
    CompetitionPages,
    CompetitionsResponseProtocol,
    KaggleCompetitionProtocol,
)

# -----------------------------------------------------------------------------
# Fake Kaggle API Implementation
# -----------------------------------------------------------------------------


class FakeApiTag:
    """Fake Kaggle API tag object matching ApiCategory structure.

    Attributes:
        ref: Tag reference slug.
    """

    __slots__ = ("ref",)

    def __init__(self, ref: str) -> None:
        """Initialize fake tag.

        Args:
            ref: Tag reference slug.
        """
        self.ref = ref


class FakeKaggleCompetition:
    """Fake Kaggle competition object matching ApiCompetition structure.

    Attributes:
        ref: Competition reference URL (full Kaggle URL).
        title: Competition title.
        category: Competition category string.
        reward: Prize description.
        deadline: Deadline as datetime.
        team_count: Number of teams.
        tags: Sequence of FakeApiTag objects (may contain None or be None).
        description: Short description.
        url: Full Kaggle URL.
    """

    __slots__ = (
        "category",
        "deadline",
        "description",
        "ref",
        "reward",
        "tags",
        "team_count",
        "title",
        "url",
    )

    def __init__(
        self,
        *,
        ref: str,
        title: str,
        category: str,
        reward: str,
        deadline: datetime,
        team_count: int,
        tags: Sequence[FakeApiTag | None] | None,
        description: str,
        url: str,
    ) -> None:
        """Initialize fake competition.

        Args:
            ref: Competition reference URL (full Kaggle URL).
            title: Competition title.
            category: Competition category string.
            reward: Prize description.
            deadline: Deadline as datetime.
            team_count: Number of teams.
            tags: Sequence of FakeApiTag objects (may contain None or be None).
            description: Short description.
            url: Full Kaggle URL.
        """
        self.ref = ref
        self.title = title
        self.category = category
        self.reward = reward
        self.deadline = deadline
        self.team_count = team_count
        self.tags = tags
        self.description = description
        self.url = url


class FakeCompetitionsResponse:
    """Fake response wrapper for competitions_list (matches new Kaggle API format)."""

    __slots__ = ("_competitions",)

    def __init__(self, competitions: Sequence[KaggleCompetitionProtocol | None] | None) -> None:
        """Initialize fake response.

        Args:
            competitions: Sequence of competition objects (may contain None or be None).
        """
        self._competitions = competitions

    @property
    def competitions(self) -> Sequence[KaggleCompetitionProtocol | None] | None:
        """Get competitions sequence."""
        return self._competitions


class FakeKaggleApi:
    """Fake Kaggle API for testing.

    Attributes:
        _competitions: Configured competitions to return.
        _list_calls: Record of calls to competitions_list.
        _authenticated: Whether authenticate() was called.
    """

    __slots__ = ("_authenticated", "_competitions", "_list_calls")

    def __init__(
        self,
        competitions: Sequence[KaggleCompetitionProtocol | None] | None = (),
    ) -> None:
        """Initialize fake API.

        Args:
            competitions: Competitions to return from competitions_list.
        """
        if competitions is None:
            self._competitions: list[KaggleCompetitionProtocol | None] = []
        else:
            self._competitions = list(competitions)
        self._list_calls: list[dict[str, str]] = []
        self._authenticated = False

    def authenticate(self) -> None:
        """Mark API as authenticated."""
        self._authenticated = True

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
        """Return configured competitions, optionally filtered.

        Args:
            group: Competition group filter (unused in fake).
            category: Optional category filter.
            sort_by: Sort order (unused in fake).
            page: Page number (unused in fake).
            search: Optional search query.
            page_size: Results per page (unused in fake).
            page_token: Pagination token (unused in fake).

        Returns:
            Response wrapper with competitions list (matches new Kaggle API format).
        """
        self._list_calls.append({"search": search or "", "category": category or ""})
        result: list[KaggleCompetitionProtocol | None] = []
        for c in self._competitions:
            if c is None:
                continue
            include = True
            if search:
                search_lower = search.lower()
                include = search_lower in c.title.lower() or search_lower in c.ref.lower()
            if include and category:
                include = c.category.lower() == category.lower()
            if include:
                result.append(c)

        return FakeCompetitionsResponse(result)


# -----------------------------------------------------------------------------
# Fake Client Implementation
# -----------------------------------------------------------------------------


class FakeKaggleClient:
    """Fake Kaggle client for testing.

    Attributes:
        _competitions: Configured competitions to return.
        _list_calls: Record of calls to list_competitions.
        _get_calls: Record of calls to get_competition.
    """

    __slots__ = ("_competitions", "_get_calls", "_list_calls")

    def __init__(self, competitions: tuple[Competition, ...] = ()) -> None:
        """Initialize fake client.

        Args:
            competitions: Competitions to return from list_competitions.
        """
        self._competitions = competitions
        self._list_calls: list[dict[str, str | CompetitionCategory | None]] = []
        self._get_calls: list[str] = []

    def list_competitions(
        self,
        *,
        search: str | None = None,
        category: CompetitionCategory | None = None,
    ) -> tuple[Competition, ...]:
        """Return configured competitions, optionally filtered.

        Args:
            search: Optional search query (filters by title).
            category: Optional category filter.

        Returns:
            Tuple of matching competitions.
        """
        self._list_calls.append({"search": search, "category": category})
        result = self._competitions
        if search:
            search_lower = search.lower()
            result = tuple(c for c in result if search_lower in c.title.lower())
        if category:
            result = tuple(c for c in result if c.category == category)
        return result

    def get_competition(self, ref: str) -> Competition | None:
        """Get competition by ref.

        Args:
            ref: Competition reference slug.

        Returns:
            Competition if found, None otherwise.
        """
        self._get_calls.append(ref)
        for c in self._competitions:
            if c.ref == ref:
                return c
        return None


# -----------------------------------------------------------------------------
# Fake Page Fetcher Implementation
# -----------------------------------------------------------------------------


class FakeKagglePageFetcher:
    """Fake page fetcher for testing.

    Attributes:
        _pages: Mapping of competition ID to pages.
        _competition_ids: Mapping of slug to competition ID.
        _fetch_calls: Record of calls to fetch_pages.
        _id_calls: Record of calls to get_competition_id.
    """

    __slots__ = ("_competition_ids", "_fetch_calls", "_id_calls", "_pages")

    def __init__(
        self,
        pages: dict[int, CompetitionPages] | None = None,
        competition_ids: dict[str, int] | None = None,
    ) -> None:
        """Initialize fake page fetcher.

        Args:
            pages: Mapping of competition ID to CompetitionPages.
            competition_ids: Mapping of slug to competition ID.
        """
        self._pages: dict[int, CompetitionPages] = pages if pages is not None else {}
        self._competition_ids: dict[str, int] = (
            competition_ids if competition_ids is not None else {}
        )
        self._fetch_calls: list[int] = []
        self._id_calls: list[str] = []

    def fetch_pages(self, competition_id: int) -> CompetitionPages:
        """Fetch pages for a competition.

        Args:
            competition_id: Numeric Kaggle competition ID.

        Returns:
            CompetitionPages for the competition.

        Raises:
            RuntimeError: If competition ID not configured.
        """
        self._fetch_calls.append(competition_id)
        if competition_id not in self._pages:
            raise RuntimeError(f"Competition {competition_id} not found")
        return self._pages[competition_id]

    def get_competition_id(self, slug: str) -> int:
        """Get competition ID from slug.

        Args:
            slug: Competition slug.

        Returns:
            Numeric competition ID.

        Raises:
            RuntimeError: If competition slug not configured.
        """
        self._id_calls.append(slug)
        comp_id = self._competition_ids.get(slug)
        if comp_id is None:
            raise RuntimeError(f"Competition '{slug}' not found")
        return comp_id
