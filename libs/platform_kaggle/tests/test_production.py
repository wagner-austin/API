"""Tests for platform_kaggle._production module."""

from __future__ import annotations

from platform_kaggle._production import (
    _get_kaggle_api,
    default_kaggle_api_factory,
    make_kaggle_client,
)
from platform_kaggle.testing import (
    FakeKaggleApi,
    hooks,
    make_fake_kaggle_competition,
)
from platform_kaggle.types import KaggleApiProtocol


class TestGetKaggleApi:
    """Tests for _get_kaggle_api function."""

    def test_get_kaggle_api_returns_real_kaggle_api(self) -> None:
        """Test _get_kaggle_api returns the real kaggle.api singleton.

        Requires KAGGLE_API_TOKEN to be set in environment.
        """
        api: KaggleApiProtocol = _get_kaggle_api()

        # Verify the API returns expected response type by calling it
        response = api.competitions_list(page_size=1)

        # Response should not be None with valid credentials
        if response is None:
            raise AssertionError("Expected response from competitions_list")

        # Verify response has competitions property
        competitions = response.competitions

        # With valid API token, competitions should not be None
        if competitions is None:
            raise AssertionError("Expected competitions to not be None")

        # Kaggle always has active competitions - get the first one
        first_comp = competitions[0]
        if first_comp is None:
            raise AssertionError("Expected first competition to not be None")

        # Verify the competition has expected properties (actual values from Kaggle)
        # ref is a full URL like "https://www.kaggle.com/competitions/..."
        assert "/competitions/" in first_comp.ref
        # title is a non-empty string
        assert first_comp.title != ""
        # category is a string
        assert first_comp.category != ""


class TestMakeKaggleClient:
    """Tests for make_kaggle_client function."""

    def test_make_kaggle_client_returns_client(self) -> None:
        """Test make_kaggle_client creates a working client."""
        fake_api = FakeKaggleApi(competitions=(make_fake_kaggle_competition(ref="prod-test"),))
        original_factory = hooks.kaggle_api_factory

        def test_factory() -> KaggleApiProtocol:
            return fake_api

        hooks.kaggle_api_factory = test_factory

        try:
            client = make_kaggle_client()
            comps = client.list_competitions()
            assert comps[0].ref == "prod-test"
        finally:
            hooks.kaggle_api_factory = original_factory


class TestDefaultKaggleApiFactory:
    """Tests for default_kaggle_api_factory."""

    def test_default_kaggle_api_factory_uses_hook(self) -> None:
        """Test default_kaggle_api_factory returns pre-authenticated api."""
        fake_api = FakeKaggleApi(competitions=(make_fake_kaggle_competition(ref="factory-test"),))
        original_factory = hooks.kaggle_api_factory

        def test_factory() -> KaggleApiProtocol:
            return fake_api

        hooks.kaggle_api_factory = test_factory

        try:
            api = default_kaggle_api_factory()
            response = api.competitions_list()
            if response is None:
                raise AssertionError("response should not be None")
            competitions = response.competitions
            if competitions is None:
                raise AssertionError("competitions should not be None")
            first = competitions[0]
            if first is None:
                raise AssertionError("first competition should not be None")
            assert first.ref == "https://www.kaggle.com/competitions/factory-test"
        finally:
            hooks.kaggle_api_factory = original_factory
