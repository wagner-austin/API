"""Tests for client: _NoneReturningApi."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import pytest

from platform_kaggle.client import (
    _extract_ref_slug,
    _normalize_category,
    _to_api_category,
)
from platform_kaggle.testing import (
    FakeApiTag,
)
from platform_kaggle.types import (
    CompetitionsResponseProtocol,
    KaggleCompetitionProtocol,
    KaggleTagProtocol,
)


class _NoneReturningApi:
    """Fake API that returns None from competitions_list."""

    def authenticate(self) -> None:
        """No-op authenticate."""

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
        """Return None to simulate API failure."""
        return None


class _NoneCompetitionsResponse:
    """Response with None competitions."""

    @property
    def competitions(self) -> Sequence[KaggleCompetitionProtocol | None] | None:
        """Return None competitions."""
        return None


class _NoneCompetitionsApi:
    """Fake API that returns response with None competitions."""

    def authenticate(self) -> None:
        """No-op authenticate."""

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
        """Return response with None competitions."""
        return _NoneCompetitionsResponse()


class _MixedCompetitionsResponse:
    """Response with None item in competitions list."""

    def __init__(self, comps: Sequence[KaggleCompetitionProtocol | None]) -> None:
        self._comps = comps

    @property
    def competitions(self) -> Sequence[KaggleCompetitionProtocol | None] | None:
        """Return competitions with None item."""
        return self._comps


class _CompetitionWithNoneTags:
    """Competition with None tags."""

    def __init__(self, ref_slug: str) -> None:
        self._ref_slug = ref_slug
        self._deadline = datetime(2025, 12, 31)

    @property
    def ref(self) -> str:
        return f"https://www.kaggle.com/competitions/{self._ref_slug}"

    @property
    def title(self) -> str:
        return "Null Tags Comp"

    @property
    def category(self) -> str:
        return "Playground"

    @property
    def reward(self) -> str:
        return "Knowledge"

    @property
    def deadline(self) -> datetime:
        return self._deadline

    @property
    def team_count(self) -> int:
        return 100

    @property
    def tags(self) -> Sequence[KaggleTagProtocol | None] | None:
        return None

    @property
    def description(self) -> str:
        return "Test"

    @property
    def url(self) -> str:
        return f"https://www.kaggle.com/competitions/{self._ref_slug}"


class _CompetitionWithNoneInTags:
    """Competition with None item in tags list."""

    def __init__(self) -> None:
        self._deadline = datetime(2025, 12, 31)

    @property
    def ref(self) -> str:
        return "https://www.kaggle.com/competitions/mixed-tags-comp"

    @property
    def title(self) -> str:
        return "Mixed Tags Comp"

    @property
    def category(self) -> str:
        return "Playground"

    @property
    def reward(self) -> str:
        return "Knowledge"

    @property
    def deadline(self) -> datetime:
        return self._deadline

    @property
    def team_count(self) -> int:
        return 100

    @property
    def tags(self) -> Sequence[KaggleTagProtocol | None] | None:
        return [FakeApiTag("valid-tag"), None, FakeApiTag("another-tag")]

    @property
    def description(self) -> str:
        return "Test"

    @property
    def url(self) -> str:
        return "https://www.kaggle.com/competitions/mixed-tags-comp"


class TestNormalizeCategory:
    """Tests for _normalize_category function."""

    def test_normalize_featured(self) -> None:
        """Test normalizing Featured category."""
        assert _normalize_category("Featured") == "Featured"

    def test_normalize_research(self) -> None:
        """Test normalizing Research category."""
        assert _normalize_category("Research") == "Research"

    def test_normalize_playground(self) -> None:
        """Test normalizing Playground category."""
        assert _normalize_category("Playground") == "Playground"

    def test_normalize_getting_started(self) -> None:
        """Test normalizing Getting Started category."""
        assert _normalize_category("Getting Started") == "Getting Started"

    def test_normalize_masters(self) -> None:
        """Test normalizing Masters category."""
        assert _normalize_category("Masters") == "Masters"

    def test_normalize_kudos(self) -> None:
        """Test normalizing Kudos category."""
        assert _normalize_category("Kudos") == "Kudos"

    def test_normalize_unknown(self) -> None:
        """Test normalizing unknown category defaults to Playground."""
        assert _normalize_category("Unknown") == "Playground"


class TestToApiCategory:
    """Tests for _to_api_category function."""

    def test_to_api_featured(self) -> None:
        """Test converting Featured to API format."""
        assert _to_api_category("Featured") == "featured"

    def test_to_api_research(self) -> None:
        """Test converting Research to API format."""
        assert _to_api_category("Research") == "research"

    def test_to_api_playground(self) -> None:
        """Test converting Playground to API format."""
        assert _to_api_category("Playground") == "playground"

    def test_to_api_getting_started(self) -> None:
        """Test converting Getting Started to API format."""
        assert _to_api_category("Getting Started") == "gettingStarted"

    def test_to_api_masters(self) -> None:
        """Test converting Masters to API format."""
        assert _to_api_category("Masters") == "masters"

    def test_to_api_kudos(self) -> None:
        """Test converting Kudos to API format."""
        assert _to_api_category("Kudos") == "kudos"


class TestExtractRefSlug:
    """Tests for _extract_ref_slug function."""

    def test_extract_from_https_url(self) -> None:
        """Test extracting slug from HTTPS URL."""
        url = "https://www.kaggle.com/competitions/gemini-3"
        assert _extract_ref_slug(url) == "gemini-3"

    def test_extract_from_http_url(self) -> None:
        """Test extracting slug from HTTP URL."""
        url = "http://www.kaggle.com/competitions/my-competition"
        assert _extract_ref_slug(url) == "my-competition"

    def test_extract_with_trailing_slash(self) -> None:
        """Test extracting slug from URL with trailing slash."""
        url = "https://www.kaggle.com/competitions/test-comp/"
        assert _extract_ref_slug(url) == "test-comp"

    def test_extract_complex_slug(self) -> None:
        """Test extracting complex slug with hyphens and numbers."""
        url = "https://www.kaggle.com/competitions/amex-default-prediction-2024"
        assert _extract_ref_slug(url) == "amex-default-prediction-2024"

    def test_invalid_url_raises_error(self) -> None:
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Kaggle competition URL"):
            _extract_ref_slug("not-a-valid-url")

    def test_url_without_competitions_raises_error(self) -> None:
        """Test that URL without /competitions/ raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Kaggle competition URL"):
            _extract_ref_slug("https://www.kaggle.com/datasets/something")
