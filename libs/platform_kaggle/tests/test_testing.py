"""Tests for platform_kaggle.testing module."""

from __future__ import annotations

from platform_kaggle.testing import (
    FakeApiTag,
    FakeKaggleApi,
    FakeKaggleClient,
    FakeKaggleCompetition,
    KaggleApiProtocol,
    hooks,
    make_fake_capability,
    make_fake_competition,
    make_fake_kaggle_competition,
    make_fake_profile,
)
from platform_kaggle.types import KaggleClientProtocol


class TestFakeKaggleCompetition:
    """Tests for FakeKaggleCompetition class."""

    def test_creation(self) -> None:
        """Test creating a FakeKaggleCompetition."""
        from datetime import datetime

        tag = FakeApiTag("tabular")
        deadline = datetime(2025, 12, 31, 23, 59, 59)
        comp = FakeKaggleCompetition(
            ref="test-comp",
            title="Test Competition",
            category="Playground",
            reward="Knowledge",
            deadline=deadline,
            team_count=100,
            tags=[tag],
            description="Test description",
            url="https://example.com",
        )
        assert comp.ref == "test-comp"
        assert comp.title == "Test Competition"
        assert comp.category == "Playground"
        assert comp.reward == "Knowledge"
        assert comp.deadline == deadline
        assert comp.team_count == 100
        tags = comp.tags
        if tags is None:
            raise AssertionError("tags should not be None")
        assert len(tags) == 1
        first_tag = tags[0]
        if first_tag is None:
            raise AssertionError("first tag should not be None")
        assert first_tag.ref == "tabular"
        assert comp.description == "Test description"
        assert comp.url == "https://example.com"


class TestFakeKaggleApi:
    """Tests for FakeKaggleApi class."""

    def test_creation_empty(self) -> None:
        """Test creating empty FakeKaggleApi."""
        api = FakeKaggleApi()
        assert api._competitions == []
        assert api._list_calls == []
        assert api._authenticated is False

    def test_creation_with_none_competitions(self) -> None:
        """Test creating FakeKaggleApi with competitions=None."""
        api = FakeKaggleApi(competitions=None)
        assert api._competitions == []

    def test_creation_with_competitions(self) -> None:
        """Test creating FakeKaggleApi with competitions."""
        comp = make_fake_kaggle_competition(ref="test")
        api = FakeKaggleApi(competitions=(comp,))
        assert api._competitions[0] is comp

    def test_authenticate(self) -> None:
        """Test authenticate method."""
        api = FakeKaggleApi()
        assert api._authenticated is False
        api.authenticate()
        assert api._authenticated is True

    def test_competitions_list_skips_none_in_list(self) -> None:
        """Test competitions_list skips None items in the competitions list."""
        comp = make_fake_kaggle_competition(ref="valid-comp", title="Valid")
        # Create list with None items - type allows Sequence[...Protocol | None]
        comps_with_nones: list[FakeKaggleCompetition | None] = [None, comp, None]
        api = FakeKaggleApi(competitions=comps_with_nones)

        response = api.competitions_list()

        if response is None:
            raise AssertionError("response should not be None")
        competitions = response.competitions
        if competitions is None:
            raise AssertionError("competitions should not be None")
        # Should only contain the valid competition (None items skipped)
        assert competitions[0] is comp

    def test_competitions_list_all(self) -> None:
        """Test listing all competitions."""
        comp1 = make_fake_kaggle_competition(ref="comp1", title="Comp 1")
        comp2 = make_fake_kaggle_competition(ref="comp2", title="Comp 2")
        api = FakeKaggleApi(competitions=(comp1, comp2))

        response = api.competitions_list()

        if response is None:
            raise AssertionError("response should not be None")
        competitions = response.competitions
        if competitions is None:
            raise AssertionError("competitions should not be None")
        assert len(competitions) == 2
        assert api._list_calls == [{"search": "", "category": ""}]

    def test_competitions_list_with_search(self) -> None:
        """Test listing competitions with search filter."""
        comp1 = make_fake_kaggle_competition(ref="tabular-comp", title="Tabular")
        comp2 = make_fake_kaggle_competition(ref="nlp-comp", title="NLP Comp")
        api = FakeKaggleApi(competitions=(comp1, comp2))

        response = api.competitions_list(search="tabular")

        if response is None:
            raise AssertionError("response should not be None")
        competitions = response.competitions
        if competitions is None:
            raise AssertionError("competitions should not be None")
        assert len(competitions) == 1
        first = competitions[0]
        if first is None:
            raise AssertionError("first competition should not be None")
        # Kaggle API 1.8.3 returns full URL in ref field
        assert first.ref == "https://www.kaggle.com/competitions/tabular-comp"

    def test_competitions_list_search_by_ref(self) -> None:
        """Test listing competitions with search matching ref."""
        comp1 = make_fake_kaggle_competition(ref="my-special-comp", title="Other")
        comp2 = make_fake_kaggle_competition(ref="regular", title="Regular")
        api = FakeKaggleApi(competitions=(comp1, comp2))

        response = api.competitions_list(search="special")

        if response is None:
            raise AssertionError("response should not be None")
        competitions = response.competitions
        if competitions is None:
            raise AssertionError("competitions should not be None")
        assert len(competitions) == 1
        first = competitions[0]
        if first is None:
            raise AssertionError("first competition should not be None")
        # Kaggle API 1.8.3 returns full URL in ref field
        assert first.ref == "https://www.kaggle.com/competitions/my-special-comp"

    def test_competitions_list_with_category(self) -> None:
        """Test listing competitions with category filter."""
        comp1 = make_fake_kaggle_competition(
            ref="featured-comp", title="Featured", category="Featured"
        )
        comp2 = make_fake_kaggle_competition(
            ref="playground-comp", title="Playground", category="Playground"
        )
        api = FakeKaggleApi(competitions=(comp1, comp2))

        response = api.competitions_list(category="featured")

        if response is None:
            raise AssertionError("response should not be None")
        competitions = response.competitions
        if competitions is None:
            raise AssertionError("competitions should not be None")
        assert len(competitions) == 1
        first = competitions[0]
        if first is None:
            raise AssertionError("first competition should not be None")
        # Kaggle API 1.8.3 returns full URL in ref field
        assert first.ref == "https://www.kaggle.com/competitions/featured-comp"


class TestFakeKaggleClient:
    """Tests for FakeKaggleClient class."""

    def test_creation_empty(self) -> None:
        """Test creating empty FakeKaggleClient."""
        client = FakeKaggleClient()
        assert client._competitions == ()
        assert client._list_calls == []
        assert client._get_calls == []

    def test_creation_with_competitions(self) -> None:
        """Test creating FakeKaggleClient with competitions."""
        comp = make_fake_competition(ref="test")
        client = FakeKaggleClient(competitions=(comp,))
        assert len(client._competitions) == 1

    def test_list_competitions_all(self) -> None:
        """Test listing all competitions."""
        comp1 = make_fake_competition(ref="comp1")
        comp2 = make_fake_competition(ref="comp2")
        client = FakeKaggleClient(competitions=(comp1, comp2))

        result = client.list_competitions()

        assert len(result) == 2
        assert len(client._list_calls) == 1

    def test_list_competitions_with_search(self) -> None:
        """Test listing competitions with search filter."""
        comp1 = make_fake_competition(ref="tabular", title="Tabular Comp")
        comp2 = make_fake_competition(ref="nlp", title="NLP Comp")
        client = FakeKaggleClient(competitions=(comp1, comp2))

        result = client.list_competitions(search="tabular")

        assert len(result) == 1
        assert result[0].ref == "tabular"

    def test_list_competitions_with_category(self) -> None:
        """Test listing competitions with category filter."""
        comp1 = make_fake_competition(ref="featured", category="Featured")
        comp2 = make_fake_competition(ref="playground", category="Playground")
        client = FakeKaggleClient(competitions=(comp1, comp2))

        result = client.list_competitions(category="Featured")

        assert len(result) == 1
        assert result[0].ref == "featured"

    def test_get_competition_found(self) -> None:
        """Test getting competition by ref when it exists."""
        comp1 = make_fake_competition(ref="target")
        comp2 = make_fake_competition(ref="other")
        client = FakeKaggleClient(competitions=(comp1, comp2))

        result = client.get_competition("target")

        # Verify competition was found - raise if None
        if result is None:
            raise AssertionError("Expected competition to be found")
        assert result.ref == "target"
        assert client._get_calls == ["target"]

    def test_get_competition_not_found(self) -> None:
        """Test getting competition by ref when it doesn't exist."""
        comp = make_fake_competition(ref="other")
        client = FakeKaggleClient(competitions=(comp,))

        result = client.get_competition("nonexistent")

        assert result is None
        assert client._get_calls == ["nonexistent"]


class TestMakeFakeCompetition:
    """Tests for make_fake_competition function."""

    def test_default_values(self) -> None:
        """Test creating competition with default values."""
        comp = make_fake_competition()
        assert comp.ref == "test-competition"
        assert comp.title == "Test Competition"
        assert comp.category == "Playground"
        assert comp.reward == "Knowledge"
        # Far future on purpose: a default deadline that expires turns every
        # active_only consumer red on a date rather than on a change.
        assert comp.deadline == "2999-12-31"
        assert comp.team_count == 100
        assert comp.tags == ("tabular",)
        assert comp.description == "Test description"
        assert comp.url == "https://www.kaggle.com/competitions/test-competition"

    def test_custom_values(self) -> None:
        """Test creating competition with custom values."""
        comp = make_fake_competition(
            ref="custom-comp",
            title="Custom Title",
            category="Featured",
            reward="$100,000",
            deadline="2025-06-15",
            team_count=5000,
            tags=("nlp", "classification"),
            description="Custom description",
        )
        assert comp.ref == "custom-comp"
        assert comp.title == "Custom Title"
        assert comp.category == "Featured"
        assert comp.reward == "$100,000"
        assert comp.deadline == "2025-06-15"
        assert comp.team_count == 5000
        assert comp.tags == ("nlp", "classification")
        assert comp.description == "Custom description"


class TestMakeFakeKaggleCompetition:
    """Tests for make_fake_kaggle_competition function."""

    def test_default_values(self) -> None:
        """Test creating kaggle competition with default values."""
        comp = make_fake_kaggle_competition()
        # Kaggle API 1.8.3 returns full URL in ref field
        assert comp.ref == "https://www.kaggle.com/competitions/test-competition"
        assert comp.title == "Test Competition"
        assert comp.category == "Playground"
        assert comp.reward == "Knowledge"
        assert comp.team_count == 100
        tags = comp.tags
        if tags is None:
            raise AssertionError("tags should not be None")
        assert len(tags) == 1
        first_tag = tags[0]
        if first_tag is None:
            raise AssertionError("first tag should not be None")
        assert first_tag.ref == "tabular"

    def test_custom_values(self) -> None:
        """Test creating kaggle competition with custom values."""
        comp = make_fake_kaggle_competition(
            ref="custom",
            title="Custom",
            team_count=500,
            tags=("a", "b"),
        )
        # Kaggle API 1.8.3 returns full URL in ref field
        assert comp.ref == "https://www.kaggle.com/competitions/custom"
        assert comp.title == "Custom"
        assert comp.team_count == 500
        tags = comp.tags
        if tags is None:
            raise AssertionError("tags should not be None")
        assert len(tags) == 2
        tag0 = tags[0]
        tag1 = tags[1]
        if tag0 is None:
            raise AssertionError("tag 0 should not be None")
        if tag1 is None:
            raise AssertionError("tag 1 should not be None")
        assert tag0.ref == "a"
        assert tag1.ref == "b"

    def test_custom_deadline(self) -> None:
        """Test creating kaggle competition with custom deadline."""
        from datetime import datetime

        custom_deadline = datetime(2026, 6, 15, 12, 0, 0)
        comp = make_fake_kaggle_competition(deadline=custom_deadline)
        assert comp.deadline == custom_deadline


class TestMakeFakeCapability:
    """Tests for make_fake_capability function."""

    def test_default_values(self) -> None:
        """Test creating capability with default values."""
        cap = make_fake_capability()
        assert cap.name == "test_capability"
        assert cap.strength == "moderate"
        assert cap.tags == ("test",)
        assert cap.description == "Test capability"

    def test_custom_values(self) -> None:
        """Test creating capability with custom values."""
        cap = make_fake_capability(
            name="xgboost_tabular",
            strength="strong",
            tags=("tabular", "classification"),
            description="XGBoost for tabular data",
        )
        assert cap.name == "xgboost_tabular"
        assert cap.strength == "strong"
        assert cap.tags == ("tabular", "classification")
        assert cap.description == "XGBoost for tabular data"


class TestMakeFakeProfile:
    """Tests for make_fake_profile function."""

    def test_default_values(self) -> None:
        """Test creating profile with default values."""
        profile = make_fake_profile()
        assert profile.capabilities == ()
        assert profile.ml_backends == ("xgboost",)
        assert profile.data_formats == ("csv",)
        assert profile.task_types == ("binary_classification",)

    def test_custom_values(self) -> None:
        """Test creating profile with custom values."""
        cap = make_fake_capability()
        profile = make_fake_profile(
            capabilities=(cap,),
            ml_backends=("pytorch", "tensorflow"),
            data_formats=("parquet",),
            task_types=("regression",),
        )
        assert profile.capabilities == (cap,)
        assert profile.ml_backends == ("pytorch", "tensorflow")
        assert profile.data_formats == ("parquet",)
        assert profile.task_types == ("regression",)


class TestHooks:
    """Tests for hooks functionality."""

    def test_hooks_kaggle_api_factory_returns_api(self) -> None:
        """Test hooks kaggle_api_factory returns functional API."""
        # Install a fake API to test the hook mechanism
        fake_api = FakeKaggleApi()
        original = hooks.kaggle_api_factory

        def fake_factory() -> KaggleApiProtocol:
            return fake_api

        hooks.kaggle_api_factory = fake_factory
        try:
            factory = hooks.kaggle_api_factory
            result = factory()
            # Verify the result is our fake API
            result.authenticate()
            assert fake_api._authenticated is True
        finally:
            hooks.kaggle_api_factory = original

    def test_hooks_kaggle_client_returns_client(self) -> None:
        """Test hooks kaggle_client returns functional client."""
        # Install a fake client to test the hook mechanism
        comp = make_fake_competition(ref="test-ref")
        fake_client = FakeKaggleClient(competitions=(comp,))
        original = hooks.kaggle_client

        def fake_factory() -> KaggleClientProtocol:
            return fake_client

        hooks.kaggle_client = fake_factory
        try:
            factory = hooks.kaggle_client
            client = factory()
            # Verify the result is our fake client
            comps = client.list_competitions()
            assert comps[0].ref == "test-ref"
        finally:
            hooks.kaggle_client = original


class TestProductionHooks:
    """Tests for production hook implementations."""

    def test_production_kaggle_client_uses_factory(self) -> None:
        """Test production kaggle_client hook creates client with factory."""
        # Set up a fake API factory that the client will use
        fake_api = FakeKaggleApi(competitions=(make_fake_kaggle_competition(ref="prod-test"),))
        original_factory = hooks.kaggle_api_factory

        def test_factory() -> KaggleApiProtocol:
            return fake_api

        hooks.kaggle_api_factory = test_factory

        try:
            # Import the production client class and create one
            from platform_kaggle.client import KaggleClient

            client = KaggleClient()
            # The client should use our fake factory
            comps = client.list_competitions()
            assert comps[0].ref == "prod-test"
        finally:
            hooks.kaggle_api_factory = original_factory
