"""Tests for platform_kaggle.client module."""

from __future__ import annotations

import pytest

from platform_kaggle.client import (
    KaggleClient,
    _extract_ref_slug,
    _normalize_category,
    _to_api_category,
)
from platform_kaggle.testing import (
    FakeKaggleApi,
    hooks,
    make_fake_kaggle_competition,
)


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


class TestKaggleClient:
    """Tests for KaggleClient class."""

    def test_list_competitions(self) -> None:
        """Test listing competitions."""
        # Setup fake API
        fake_api = FakeKaggleApi(
            competitions=(
                make_fake_kaggle_competition(
                    ref="comp-1",
                    title="Competition 1",
                    category="Playground",
                    tags=("tabular",),
                ),
                make_fake_kaggle_competition(
                    ref="comp-2",
                    title="Competition 2",
                    category="Featured",
                    tags=("nlp",),
                ),
            )
        )
        original = hooks.kaggle_api_factory

        def fake_factory() -> FakeKaggleApi:
            return fake_api

        hooks.kaggle_api_factory = fake_factory

        try:
            client = KaggleClient()
            comps = client.list_competitions()

            assert len(comps) == 2
            assert comps[0].ref == "comp-1"
            assert comps[1].ref == "comp-2"
        finally:
            hooks.kaggle_api_factory = original

    def test_list_competitions_with_search(self) -> None:
        """Test listing competitions with search filter."""
        fake_api = FakeKaggleApi(
            competitions=(
                make_fake_kaggle_competition(ref="tabular-comp", title="Tabular Comp"),
                make_fake_kaggle_competition(ref="nlp-comp", title="NLP Comp"),
            )
        )
        original = hooks.kaggle_api_factory

        def fake_factory() -> FakeKaggleApi:
            return fake_api

        hooks.kaggle_api_factory = fake_factory

        try:
            client = KaggleClient()
            comps = client.list_competitions(search="tabular")

            assert len(comps) == 1
            assert comps[0].ref == "tabular-comp"
        finally:
            hooks.kaggle_api_factory = original

    def test_list_competitions_with_category(self) -> None:
        """Test listing competitions with category filter."""
        fake_api = FakeKaggleApi(
            competitions=(
                make_fake_kaggle_competition(
                    ref="featured-comp",
                    title="Featured Comp",
                    category="Featured",
                ),
                make_fake_kaggle_competition(
                    ref="playground-comp",
                    title="Playground Comp",
                    category="Playground",
                ),
            )
        )
        original = hooks.kaggle_api_factory

        def fake_factory() -> FakeKaggleApi:
            return fake_api

        hooks.kaggle_api_factory = fake_factory

        try:
            client = KaggleClient()
            comps = client.list_competitions(category="Featured")

            assert len(comps) == 1
            assert comps[0].ref == "featured-comp"
        finally:
            hooks.kaggle_api_factory = original

    def test_get_competition_found(self) -> None:
        """Test getting a competition by ref when it exists."""
        fake_api = FakeKaggleApi(
            competitions=(
                make_fake_kaggle_competition(ref="target-comp", title="Target Comp"),
                make_fake_kaggle_competition(ref="other-comp", title="Other Comp"),
            )
        )
        original = hooks.kaggle_api_factory

        def fake_factory() -> FakeKaggleApi:
            return fake_api

        hooks.kaggle_api_factory = fake_factory

        try:
            client = KaggleClient()
            comp = client.get_competition("target-comp")

            # Verify competition was found - raise if None
            if comp is None:
                raise AssertionError("Expected competition to be found")
            assert comp.ref == "target-comp"
            assert comp.title == "Target Comp"
        finally:
            hooks.kaggle_api_factory = original

    def test_get_competition_not_found(self) -> None:
        """Test getting a competition by ref when it doesn't exist."""
        fake_api = FakeKaggleApi(competitions=(make_fake_kaggle_competition(ref="other-comp"),))
        original = hooks.kaggle_api_factory

        def fake_factory() -> FakeKaggleApi:
            return fake_api

        hooks.kaggle_api_factory = fake_factory

        try:
            client = KaggleClient()
            comp = client.get_competition("nonexistent")

            assert comp is None
        finally:
            hooks.kaggle_api_factory = original

    def test_get_competition_found_after_skip(self) -> None:
        """Test getting a competition when it's not the first result.

        This tests the for loop continuation branch in get_competition.
        The search returns multiple results where the target is not first.
        """
        # FakeKaggleApi filters by whether search string is in title OR ref
        # So searching for "comp-target" will match both:
        # - "comp-target-wrong" (contains "comp-target" in ref)
        # - "comp-target" (exact match)
        fake_api = FakeKaggleApi(
            competitions=(
                make_fake_kaggle_competition(ref="comp-target-wrong", title="Wrong One"),
                make_fake_kaggle_competition(ref="comp-target", title="Target One"),
            )
        )
        original = hooks.kaggle_api_factory

        def fake_factory() -> FakeKaggleApi:
            return fake_api

        hooks.kaggle_api_factory = fake_factory

        try:
            client = KaggleClient()
            # Search for "comp-target" returns both competitions
            # Loop will skip "comp-target-wrong" (ref != "comp-target")
            # Then find "comp-target" (ref == "comp-target")
            comp = client.get_competition("comp-target")

            if comp is None:
                raise AssertionError("Expected competition to be found")
            assert comp.ref == "comp-target"
        finally:
            hooks.kaggle_api_factory = original
