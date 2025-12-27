"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_codebase import CodebaseProfile, LibInfo, ServiceInfo
from platform_codebase.testing import make_fake_lib_info, make_fake_profile, make_fake_service_info
from platform_devpost import Hackathon
from platform_devpost.testing import FakeDevpostClient, make_fake_hackathon
from platform_kaggle import Competition
from platform_kaggle.testing import (
    FakeKaggleClient,
    FakeKagglePageFetcher,
    make_fake_competition,
    make_fake_competition_pages,
)

from opportunity_radar_api.api.container import ServiceContainer
from opportunity_radar_api.config import OpportunityRadarSettings


def _make_fake_competition() -> Competition:
    """Create a fake Kaggle competition."""
    return make_fake_competition(
        ref="test-comp",
        title="Test Competition",
        category="Playground",
        tags=("tabular", "classification"),
    )


def _make_fake_hackathon() -> Hackathon:
    """Create a fake Devpost hackathon."""
    return make_fake_hackathon(
        id=123,
        title="Test Hackathon",
        open_state="open",
    )


def _make_fake_profile() -> CodebaseProfile:
    """Create a fake codebase profile."""
    return make_fake_profile(
        technologies=("python",),
        frameworks=("fastapi",),
        ml_backends=("xgboost",),
    )


def _make_fake_lib_info() -> LibInfo:
    """Create a fake library info."""
    return make_fake_lib_info(
        name="test-lib",
        dependencies=("fastapi", "httpx"),
    )


def _make_fake_service_info() -> ServiceInfo:
    """Create a fake service info."""
    return make_fake_service_info(
        name="test-service",
        dependencies=("flask",),
    )


def _make_fake_kaggle_client(fake_competition: Competition) -> FakeKaggleClient:
    """Create a fake Kaggle client with test data."""
    return FakeKaggleClient(competitions=(fake_competition,))


def _make_fake_page_fetcher(fake_competition: Competition) -> FakeKagglePageFetcher:
    """Create a fake page fetcher with test data."""
    pages = make_fake_competition_pages(description="Test competition description")
    return FakeKagglePageFetcher(
        competition_ids={fake_competition.ref: 1},
        pages={1: pages},
    )


def _make_fake_devpost_client(fake_hackathon: Hackathon) -> FakeDevpostClient:
    """Create a fake Devpost client with test data."""
    return FakeDevpostClient(hackathons=(fake_hackathon,))


def _make_fake_container(
    fake_kaggle_client: FakeKaggleClient,
    fake_page_fetcher: FakeKagglePageFetcher,
    fake_devpost_client: FakeDevpostClient,
    fake_profile: CodebaseProfile,
    fake_lib_info: LibInfo,
    fake_service_info: ServiceInfo,
    tmp_path: Path,
) -> ServiceContainer:
    """Create a fake service container."""
    return ServiceContainer(
        monorepo_root=tmp_path,
        kaggle_client_factory=lambda: fake_kaggle_client,
        kaggle_page_fetcher_factory=lambda: fake_page_fetcher,
        devpost_client_factory=lambda: fake_devpost_client,
        codebase_profile_factory=lambda root: fake_profile,
        libs_scanner=lambda root: (fake_lib_info,),
        services_scanner=lambda root: (fake_service_info,),
    )


def _make_fake_settings() -> OpportunityRadarSettings:
    """Create fake settings for testing."""
    return OpportunityRadarSettings(
        kaggle_api_token="",
        port=8010,
        log_level="INFO",
        log_format="json",
        github_token=None,
        github_repo=None,
    )


fake_competition = pytest.fixture(_make_fake_competition)
fake_hackathon = pytest.fixture(_make_fake_hackathon)
fake_profile = pytest.fixture(_make_fake_profile)
fake_lib_info = pytest.fixture(_make_fake_lib_info)
fake_service_info = pytest.fixture(_make_fake_service_info)
fake_kaggle_client = pytest.fixture(_make_fake_kaggle_client)
fake_page_fetcher = pytest.fixture(_make_fake_page_fetcher)
fake_devpost_client = pytest.fixture(_make_fake_devpost_client)
fake_container = pytest.fixture(_make_fake_container)
fake_settings = pytest.fixture(_make_fake_settings)
