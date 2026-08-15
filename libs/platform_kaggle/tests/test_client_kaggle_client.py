"""Tests for client: KaggleClient."""

from __future__ import annotations

from collections.abc import Sequence

from platform_kaggle.client import (
    KaggleClient,
)
from platform_kaggle.testing import (
    FakeKaggleApi,
    hooks,
    make_fake_kaggle_competition,
)
from platform_kaggle.types import (
    CompetitionsResponseProtocol,
    KaggleCompetitionProtocol,
)
from tests.test_client_fakes_and_helpers import (
    _CompetitionWithNoneInTags,
    _CompetitionWithNoneTags,
    _MixedCompetitionsResponse,
    _NoneCompetitionsApi,
    _NoneReturningApi,
)


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

    def test_list_competitions_api_returns_none(self) -> None:
        """Test list_competitions handles None response from API."""
        original = hooks.kaggle_api_factory
        hooks.kaggle_api_factory = lambda: _NoneReturningApi()

        try:
            client = KaggleClient()
            result = client.list_competitions()
            assert result == ()
        finally:
            hooks.kaggle_api_factory = original

    def test_list_competitions_competitions_is_none(self) -> None:
        """Test list_competitions handles None competitions in response."""
        original = hooks.kaggle_api_factory
        hooks.kaggle_api_factory = lambda: _NoneCompetitionsApi()

        try:
            client = KaggleClient()
            result = client.list_competitions()
            assert result == ()
        finally:
            hooks.kaggle_api_factory = original

    def test_list_competitions_skips_none_competition(self) -> None:
        """Test list_competitions skips None items in competitions list."""

        class _MixedCompetitionsApi:
            """Fake API that returns response with None item in list."""

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
                """Return response with None item."""
                valid_comp = make_fake_kaggle_competition(ref="valid-comp")
                return _MixedCompetitionsResponse([None, valid_comp, None])

        original = hooks.kaggle_api_factory
        hooks.kaggle_api_factory = lambda: _MixedCompetitionsApi()

        try:
            client = KaggleClient()
            result = client.list_competitions()
            # Should skip None items and only return valid competition
            assert result[0].ref == "valid-comp"
        finally:
            hooks.kaggle_api_factory = original

    def test_list_competitions_handles_none_tags(self) -> None:
        """Test list_competitions handles None tags on competition."""

        class _NoneTagsResponse:
            """Response with competitions having None tags."""

            @property
            def competitions(self) -> Sequence[KaggleCompetitionProtocol | None] | None:
                return [_CompetitionWithNoneTags("null-tags-comp"), _CompetitionWithNoneInTags()]

        class _NoneTagsApi:
            """Fake API for testing None tags handling."""

            def authenticate(self) -> None:
                pass

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
                return _NoneTagsResponse()

        original = hooks.kaggle_api_factory
        hooks.kaggle_api_factory = lambda: _NoneTagsApi()

        try:
            client = KaggleClient()
            result = client.list_competitions()

            # First competition has None tags -> empty tuple
            assert result[0].tags == ()
            # Second competition has [valid, None, valid] -> (valid, valid)
            assert result[1].tags == ("valid-tag", "another-tag")
        finally:
            hooks.kaggle_api_factory = original

    def test_get_competition_api_returns_none(self) -> None:
        """Test get_competition handles None response from API."""
        original = hooks.kaggle_api_factory
        hooks.kaggle_api_factory = lambda: _NoneReturningApi()

        try:
            client = KaggleClient()
            result = client.get_competition("any-ref")
            assert result is None
        finally:
            hooks.kaggle_api_factory = original

    def test_get_competition_competitions_is_none(self) -> None:
        """Test get_competition handles None competitions in response."""
        original = hooks.kaggle_api_factory
        hooks.kaggle_api_factory = lambda: _NoneCompetitionsApi()

        try:
            client = KaggleClient()
            result = client.get_competition("any-ref")
            assert result is None
        finally:
            hooks.kaggle_api_factory = original

    def test_get_competition_skips_none_in_list(self) -> None:
        """Test get_competition skips None items when searching."""

        class _MixedApi:
            """Fake API with None items in response."""

            def authenticate(self) -> None:
                pass

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
                target = make_fake_kaggle_competition(ref="target-comp")
                return _MixedCompetitionsResponse([None, target, None])

        original = hooks.kaggle_api_factory
        hooks.kaggle_api_factory = lambda: _MixedApi()

        try:
            client = KaggleClient()
            result = client.get_competition("target-comp")
            if result is None:
                raise AssertionError("Expected to find competition")
            assert result.ref == "target-comp"
        finally:
            hooks.kaggle_api_factory = original

    def test_get_competition_handles_none_tags(self) -> None:
        """Test get_competition handles None tags on found competition."""

        class _NoneTagsResponse:
            """Response with competition with None tags."""

            @property
            def competitions(self) -> Sequence[KaggleCompetitionProtocol | None] | None:
                return [_CompetitionWithNoneTags("target")]

        class _NoneTagsApi:
            """Fake API for None tags test."""

            def authenticate(self) -> None:
                pass

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
                return _NoneTagsResponse()

        original = hooks.kaggle_api_factory
        hooks.kaggle_api_factory = lambda: _NoneTagsApi()

        try:
            client = KaggleClient()
            result = client.get_competition("target")
            if result is None:
                raise AssertionError("Expected to find competition")
            assert result.tags == ()
        finally:
            hooks.kaggle_api_factory = original
