"""Public test utilities for platform_kaggle consumers.

This module provides hooks for dependency injection and fake implementations
for testing. Production code sets hooks at startup; tests set them to fakes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from platform_kaggle.types import (
    CapabilityStrength,
    CodebaseCapability,
    CodebaseProfile,
    Competition,
    CompetitionCategory,
    CompetitionPage,
    CompetitionPages,
    CompetitionsResponseProtocol,
    KaggleApiFactoryProtocol,
    KaggleApiProtocol,
    KaggleClientProtocol,
    KaggleCompetitionProtocol,
    KagglePageFetcherProtocol,
)

# -----------------------------------------------------------------------------
# Hook Types
# -----------------------------------------------------------------------------

KaggleClientHook = Callable[[], KaggleClientProtocol]
PageFetcherHook = Callable[[], KagglePageFetcherProtocol]
ProfileScannerHook = Callable[[Path], CodebaseProfile]


# -----------------------------------------------------------------------------
# Hooks Container
# -----------------------------------------------------------------------------


class HooksContainer:
    """Container for dependency injection hooks.

    Attributes:
        kaggle_api_factory: Factory for Kaggle API (returns pre-authenticated api).
        kaggle_client: Factory for Kaggle client.
        page_fetcher: Factory for page fetcher.
        profile_scanner: Factory for codebase profile scanner.
    """

    kaggle_api_factory: KaggleApiFactoryProtocol
    kaggle_client: KaggleClientHook
    page_fetcher: PageFetcherHook
    profile_scanner: ProfileScannerHook


hooks = HooksContainer()


def _init_hooks() -> None:
    """Initialize hooks with production implementations."""
    from platform_kaggle._production import _get_kaggle_api, make_kaggle_client
    from platform_kaggle.capabilities import scan_codebase
    from platform_kaggle.internal_api import create_page_fetcher

    hooks.kaggle_api_factory = _get_kaggle_api
    hooks.kaggle_client = make_kaggle_client
    hooks.page_fetcher = create_page_fetcher
    hooks.profile_scanner = scan_codebase


def reset_hooks() -> None:
    """Reset hooks to production implementations (for test teardown)."""
    _init_hooks()


# Initialize hooks on module load
_init_hooks()


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


# -----------------------------------------------------------------------------
# Factory Functions for Tests
# -----------------------------------------------------------------------------


def make_fake_competition(
    *,
    ref: str = "test-competition",
    title: str = "Test Competition",
    category: CompetitionCategory = "Playground",
    reward: str = "Knowledge",
    deadline: str = "2025-12-31",
    team_count: int = 100,
    tags: tuple[str, ...] = ("tabular",),
    description: str = "Test description",
) -> Competition:
    """Factory for creating test Competition instances.

    Args:
        ref: Competition reference slug.
        title: Competition title.
        category: Competition category.
        reward: Prize description.
        deadline: Deadline as ISO 8601 date string.
        team_count: Number of teams.
        tags: Tuple of tags.
        description: Short description.

    Returns:
        Competition instance.
    """
    return Competition(
        ref=ref,
        title=title,
        category=category,
        reward=reward,
        deadline=deadline,
        team_count=team_count,
        tags=tags,
        description=description,
        url=f"https://www.kaggle.com/competitions/{ref}",
    )


def make_fake_kaggle_competition(
    *,
    ref: str = "test-competition",
    title: str = "Test Competition",
    category: str = "Playground",
    reward: str = "Knowledge",
    deadline: datetime | None = None,
    team_count: int = 100,
    tags: tuple[str, ...] = ("tabular",),
    description: str = "Test description",
) -> FakeKaggleCompetition:
    """Factory for creating test FakeKaggleCompetition instances.

    Matches Kaggle API 1.8.3 format where ref is a full URL.

    Args:
        ref: Competition reference slug (converted to full URL internally).
        title: Competition title.
        category: Competition category string.
        reward: Prize description.
        deadline: Deadline as datetime (defaults to 2025-12-31).
        team_count: Number of teams.
        tags: Tuple of tag strings (converted to FakeApiTag objects).
        description: Short description.

    Returns:
        FakeKaggleCompetition instance with ref as full URL.
    """
    url = f"https://www.kaggle.com/competitions/{ref}"
    if deadline is None:
        deadline = datetime(2025, 12, 31, 23, 59, 59)
    return FakeKaggleCompetition(
        ref=url,  # Kaggle API 1.8.3 returns full URL in ref field
        title=title,
        category=category,
        reward=reward,
        deadline=deadline,
        team_count=team_count,
        tags=[FakeApiTag(t) for t in tags],
        description=description,
        url=url,
    )


def make_fake_capability(
    *,
    name: str = "test_capability",
    strength: CapabilityStrength = "moderate",
    tags: tuple[str, ...] = ("test",),
    description: str = "Test capability",
) -> CodebaseCapability:
    """Factory for creating test CodebaseCapability instances.

    Args:
        name: Capability identifier.
        strength: Capability strength level.
        tags: Tuple of tags.
        description: Human-readable description.

    Returns:
        CodebaseCapability instance.
    """
    return CodebaseCapability(
        name=name,
        strength=strength,
        tags=tags,
        description=description,
    )


def make_fake_profile(
    *,
    capabilities: tuple[CodebaseCapability, ...] = (),
    ml_backends: tuple[str, ...] = ("xgboost",),
    data_formats: tuple[str, ...] = ("csv",),
    task_types: tuple[str, ...] = ("binary_classification",),
) -> CodebaseProfile:
    """Factory for creating test CodebaseProfile instances.

    Args:
        capabilities: Tuple of capabilities.
        ml_backends: Tuple of ML backend names.
        data_formats: Tuple of data format names.
        task_types: Tuple of task type names.

    Returns:
        CodebaseProfile instance.
    """
    return CodebaseProfile(
        capabilities=capabilities,
        ml_backends=ml_backends,
        data_formats=data_formats,
        task_types=task_types,
    )


def make_fake_competition_page(
    *,
    id: int = 1,
    name: str = "Description",
    content: str = "Test content",
) -> CompetitionPage:
    """Factory for creating test CompetitionPage instances.

    Args:
        id: Page ID.
        name: Page name (e.g., "Description", "Evaluation").
        content: Markdown content.

    Returns:
        CompetitionPage instance.
    """
    return CompetitionPage(
        id=id,
        name=name,
        content=content,
    )


def make_fake_competition_pages(
    *,
    competition_id: int = 12345,
    pages: tuple[CompetitionPage, ...] | None = None,
    description: str = "Test description",
    evaluation: str = "Test evaluation",
    timeline: str = "Test timeline",
    rules: str = "Test rules",
) -> CompetitionPages:
    """Factory for creating test CompetitionPages instances.

    Args:
        competition_id: Numeric competition ID.
        pages: Tuple of pages. If None, creates default pages from content args.
        description: Description page content.
        evaluation: Evaluation page content.
        timeline: Timeline page content.
        rules: Rules page content.

    Returns:
        CompetitionPages instance.
    """
    if pages is None:
        pages = (
            CompetitionPage(id=1, name="Description", content=description),
            CompetitionPage(id=2, name="Evaluation", content=evaluation),
            CompetitionPage(id=3, name="Timeline", content=timeline),
            CompetitionPage(id=4, name="Rules", content=rules),
        )
    return CompetitionPages(
        competition_id=competition_id,
        pages=pages,
        description=description,
        evaluation=evaluation,
        timeline=timeline,
        rules=rules,
    )


__all__ = [
    "FakeApiTag",
    "FakeCompetitionsResponse",
    "FakeKaggleApi",
    "FakeKaggleClient",
    "FakeKaggleCompetition",
    "FakeKagglePageFetcher",
    "HooksContainer",
    "KaggleApiFactoryProtocol",
    "KaggleApiProtocol",
    "KaggleClientHook",
    "PageFetcherHook",
    "ProfileScannerHook",
    "hooks",
    "make_fake_capability",
    "make_fake_competition",
    "make_fake_competition_page",
    "make_fake_competition_pages",
    "make_fake_kaggle_competition",
    "make_fake_profile",
    "reset_hooks",
]
