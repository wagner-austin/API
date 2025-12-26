from __future__ import annotations

import pytest

from github_stats_api.github_client import GitHubLanguageData, GitHubUserData
from github_stats_api.settings import Settings


def _make_fake_settings() -> Settings:
    """Fake settings for testing.

    Returns:
        Settings TypedDict with test values.
    """
    return {
        "github_token": "test-token-12345",
        "cache_ttl_seconds": 60,
        "port": 8000,
    }


def _make_fake_user_data() -> GitHubUserData:
    """Fake GitHub user data for testing.

    Returns:
        GitHubUserData with test values.
    """
    return {
        "login": "testuser",
        "name": "Test User",
        "total_commits": 150,
        "total_prs": 25,
        "total_issues": 10,
        "total_stars": 500,
        "total_contributions": 200,
    }


def _make_fake_language_data() -> list[GitHubLanguageData]:
    """Fake GitHub language data for testing.

    Returns:
        List of GitHubLanguageData with test values.
    """
    return [
        {"name": "Python", "size": 50000, "color": "#3572A5"},
        {"name": "TypeScript", "size": 30000, "color": "#3178c6"},
        {"name": "JavaScript", "size": 15000, "color": "#f1e05a"},
        {"name": "Shell", "size": 5000, "color": "#89e051"},
    ]


fake_settings = pytest.fixture(_make_fake_settings)
fake_user_data = pytest.fixture(_make_fake_user_data)
fake_language_data = pytest.fixture(_make_fake_language_data)
