"""Tests for platform_kaggle._production module."""

from __future__ import annotations

from platform_kaggle._production import (
    create_kaggle_api,
    default_kaggle_api_factory,
    make_kaggle_client,
)
from platform_kaggle.testing import (
    FakeKaggleApi,
    FakeKaggleModule,
    hooks,
    make_fake_kaggle_competition,
)
from platform_kaggle.types import KaggleApiProtocol, KaggleModuleProtocol


class TestMakeKaggleClient:
    """Tests for make_kaggle_client function."""

    def test_make_kaggle_client_returns_client(self) -> None:
        """Test make_kaggle_client creates a working client."""
        # Set up fake API factory first
        fake_api = FakeKaggleApi(competitions=(make_fake_kaggle_competition(ref="prod-test"),))
        original_factory = hooks.kaggle_api_factory

        def test_factory() -> KaggleApiProtocol:
            return fake_api

        hooks.kaggle_api_factory = test_factory

        try:
            # Call the production function
            client = make_kaggle_client()

            # Verify it works
            comps = client.list_competitions()
            assert comps[0].ref == "prod-test"
        finally:
            hooks.kaggle_api_factory = original_factory


class TestCreateKaggleApi:
    """Tests for create_kaggle_api and default_kaggle_api_factory."""

    def test_create_kaggle_api_uses_hook(self) -> None:
        """Test create_kaggle_api uses the kaggle_module hook."""
        # Create a fake API that will be returned
        fake_api = FakeKaggleApi(competitions=(make_fake_kaggle_competition(ref="hook-test"),))

        # Create fake module that returns the fake API
        fake_module = FakeKaggleModule(api=fake_api)

        # Set up the hook
        original_module = hooks.kaggle_module

        def test_module_factory() -> KaggleModuleProtocol:
            return fake_module

        hooks.kaggle_module = test_module_factory

        try:
            api = create_kaggle_api()
            # Verify authenticate was called
            assert fake_api._authenticated is True
            # Verify we got the right API - Kaggle API 1.8.3 returns full URL in ref
            response = api.competitions_list()
            assert response.competitions[0].ref == "https://www.kaggle.com/competitions/hook-test"
        finally:
            hooks.kaggle_module = original_module

    def test_default_kaggle_api_factory_uses_hook(self) -> None:
        """Test default_kaggle_api_factory uses the kaggle_module hook."""
        # Create a fake API
        fake_api = FakeKaggleApi(competitions=(make_fake_kaggle_competition(ref="factory-test"),))

        # Create fake module
        fake_module = FakeKaggleModule(api=fake_api)

        # Set up the hook
        original_module = hooks.kaggle_module

        def test_module_factory() -> KaggleModuleProtocol:
            return fake_module

        hooks.kaggle_module = test_module_factory

        try:
            api = default_kaggle_api_factory()
            assert fake_api._authenticated is True
            # Kaggle API 1.8.3 returns full URL in ref
            response = api.competitions_list()
            assert (
                response.competitions[0].ref == "https://www.kaggle.com/competitions/factory-test"
            )
        finally:
            hooks.kaggle_module = original_module
