"""Tests for container module."""

from __future__ import annotations

from pathlib import Path

import pytest

from opportunity_radar_api import _test_hooks
from opportunity_radar_api.api.container import (
    ServiceContainer,
    _find_monorepo_root,
    create_production_container,
)
from opportunity_radar_api.config import OpportunityRadarSettings


def test_container_get_kaggle_client(fake_container: ServiceContainer) -> None:
    """Test that get_kaggle_client returns configured client."""
    client = fake_container.get_kaggle_client()

    # Should be able to list competitions
    competitions = client.list_competitions()
    assert len(competitions) == 1


def test_container_get_devpost_client(fake_container: ServiceContainer) -> None:
    """Test that get_devpost_client returns configured client."""
    client = fake_container.get_devpost_client()

    # Should be able to list hackathons
    hackathons = client.list_hackathons()
    assert len(hackathons) == 1


def test_container_get_codebase_profile(fake_container: ServiceContainer) -> None:
    """Test that get_codebase_profile returns profile."""
    profile = fake_container.get_codebase_profile()

    assert profile.technologies == ("python",)
    assert profile.frameworks == ("fastapi",)


def test_container_scan_libs(fake_container: ServiceContainer) -> None:
    """Test that scan_libs returns lib info."""
    libs = fake_container.scan_libs()

    assert len(libs) == 1
    assert libs[0].name == "test-lib"


def test_container_scan_services(fake_container: ServiceContainer) -> None:
    """Test that scan_services returns service info."""
    services = fake_container.scan_services()

    assert len(services) == 1
    assert services[0].name == "test-service"


def test_real_find_monorepo_root_finds_libs_dir(tmp_path: Path) -> None:
    """Test that _real_find_monorepo_root finds directory with libs."""
    # Create libs directory
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()

    # Create nested path
    nested = tmp_path / "services" / "my-service" / "src"
    nested.mkdir(parents=True)

    # Temporarily patch the file location

    original_file = _test_hooks.__file__

    try:
        # Mock __file__ to be in nested path
        _test_hooks.__file__ = str(nested / "_test_hooks.py")

        # Find should work from nested path
        result = _test_hooks._real_find_monorepo_root()
        assert result == tmp_path
    finally:
        _test_hooks.__file__ = original_file


def test_real_find_monorepo_root_raises_if_not_found(tmp_path: Path) -> None:
    """Test that _real_find_monorepo_root raises if libs not found."""
    # Create a deeply nested path in tmp_path
    # We need to ensure no "libs" directory exists above tmp_path
    # To do this reliably, we check from the actual root level

    original_file = _test_hooks.__file__

    # Create path without libs
    nested = tmp_path / "some" / "path"
    nested.mkdir(parents=True)

    # Create a fake file path at the filesystem root level
    # On Windows, this would be like C:\nonexistent\container.py
    # On Unix, this would be /nonexistent/container.py
    # The loop will hit root and raise RuntimeError
    root = Path(tmp_path.anchor)
    fake_path = root / "nonexistent_test_path_xyz" / "container.py"

    try:
        _test_hooks.__file__ = str(fake_path)

        # Should raise RuntimeError since there's no libs above the root
        with pytest.raises(RuntimeError, match="Could not find monorepo root"):
            _test_hooks._real_find_monorepo_root()
    finally:
        _test_hooks.__file__ = original_file


def test_create_production_container_with_root(
    tmp_path: Path,
    fake_settings: OpportunityRadarSettings,
) -> None:
    """Test creating production container with explicit root."""
    # Create libs directory
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()

    # This will fail at runtime because the real libs don't exist
    # but we can at least verify the container is created
    container = create_production_container(fake_settings, monorepo_root=tmp_path)

    assert container.monorepo_root == tmp_path


def test_create_production_container_auto_detects_root(
    tmp_path: Path,
    fake_settings: OpportunityRadarSettings,
) -> None:
    """Test creating production container with auto-detected root."""
    original_hook = _test_hooks.container_find_monorepo_root

    def fake_find() -> Path:
        return tmp_path

    _test_hooks.container_find_monorepo_root = fake_find

    try:
        container = create_production_container(fake_settings)  # No root provided
        assert container.monorepo_root == tmp_path
    finally:
        _test_hooks.container_find_monorepo_root = original_hook


def test_default_codebase_profile_factory(tmp_path: Path) -> None:
    """Test _default_codebase_profile_factory creates profile."""
    from opportunity_radar_api.api.container import _default_codebase_profile_factory

    # Create minimal structure
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()

    profile = _default_codebase_profile_factory(tmp_path)
    # Profile should be created (empty tuple for capabilities with no libs)
    assert profile.capabilities == ()
    assert profile.technologies == ()


def test_find_monorepo_root_uses_hook() -> None:
    """Test _find_monorepo_root uses hook when set."""
    expected = Path("/fake/monorepo/root")

    def fake_find() -> Path:
        return expected

    original = _test_hooks.container_find_monorepo_root
    _test_hooks.container_find_monorepo_root = fake_find

    try:
        result = _find_monorepo_root()
        assert result == expected
    finally:
        _test_hooks.container_find_monorepo_root = original


def test_find_monorepo_root_hook_is_bound_to_the_real_search() -> None:
    """The hook holds the real search, so the caller needs no fallback branch."""
    assert _test_hooks.container_find_monorepo_root is _test_hooks._real_find_monorepo_root


def test_create_production_container_with_github_scanning() -> None:
    """Test creating production container with GitHub scanning enabled."""
    settings = OpportunityRadarSettings(
        kaggle_api_token="",
        port=8010,
        log_level="INFO",
        log_format="json",
        github_token="ghp_test_token",
        github_repo="wagner-austin/API",
    )

    container = create_production_container(settings)

    # Should use GitHub-based path (use as_posix for cross-platform comparison)
    assert container.monorepo_root.as_posix() == "/github/wagner-austin/API"


def test_create_production_container_github_libs_scanner() -> None:
    """Test that GitHub libs scanner works via container."""
    from platform_codebase import FakeGitHubClient
    from platform_codebase.github_scanner import scan_libs_from_github

    # Create a fake GitHub client
    fake_client = FakeGitHubClient(
        directories={"libs": ["test-lib"]},
        files={
            "libs/test-lib/pyproject.toml": """
[tool.poetry]
name = "test-lib"

[tool.poetry.dependencies]
python = "^3.11"
""",
        },
    )

    # Test the underlying scanner function
    result = scan_libs_from_github(fake_client, "owner", "repo")
    assert len(result) == 1
    assert result[0].name == "test-lib"


def test_create_production_container_github_services_scanner() -> None:
    """Test that GitHub services scanner works via container."""
    from platform_codebase import FakeGitHubClient
    from platform_codebase.github_scanner import scan_services_from_github

    # Create a fake GitHub client
    fake_client = FakeGitHubClient(
        directories={"services": ["test-api"]},
        files={
            "services/test-api/pyproject.toml": """
[tool.poetry]
name = "test-api"

[tool.poetry.dependencies]
python = "^3.11"
""",
        },
        path_patterns={("services/test-api", ".rules"): True},
    )

    # Test the underlying scanner function
    result = scan_services_from_github(fake_client, "owner", "repo")
    assert len(result) == 1
    assert result[0].name == "test-api"
    assert result[0].has_rules_files is True


def test_github_scanning_via_container_scan_libs() -> None:
    """Test that scan_libs works on container with GitHub scanning."""
    from platform_codebase import FakeGitHubClient

    from opportunity_radar_api import _test_hooks

    # Create a fake GitHub client
    fake_client = FakeGitHubClient(
        directories={"libs": ["my-lib"]},
        files={
            "libs/my-lib/pyproject.toml": """
[tool.poetry]
name = "my-lib"

[tool.poetry.dependencies]
python = "^3.11"
""",
        },
    )

    def fake_factory(token: str) -> FakeGitHubClient:
        _ = token  # Unused
        return fake_client

    original = _test_hooks.container_github_client_factory
    _test_hooks.container_github_client_factory = fake_factory

    try:
        settings = OpportunityRadarSettings(
            kaggle_api_token="",
            port=8010,
            log_level="INFO",
            log_format="json",
            github_token="ghp_test",
            github_repo="owner/repo",
        )
        container = create_production_container(settings)

        # Call scan_libs to exercise the closure
        libs = container.scan_libs()
        assert len(libs) == 1
        assert libs[0].name == "my-lib"
    finally:
        _test_hooks.container_github_client_factory = original


def test_github_scanning_via_container_scan_services() -> None:
    """Test that scan_services works on container with GitHub scanning."""
    from platform_codebase import FakeGitHubClient

    from opportunity_radar_api import _test_hooks

    # Create a fake GitHub client
    fake_client = FakeGitHubClient(
        directories={"services": ["my-api"]},
        files={
            "services/my-api/pyproject.toml": """
[tool.poetry]
name = "my-api"

[tool.poetry.dependencies]
python = "^3.11"
""",
        },
        path_patterns={("services/my-api", ".rules"): False},
    )

    def fake_factory(token: str) -> FakeGitHubClient:
        _ = token  # Unused
        return fake_client

    original = _test_hooks.container_github_client_factory
    _test_hooks.container_github_client_factory = fake_factory

    try:
        settings = OpportunityRadarSettings(
            kaggle_api_token="",
            port=8010,
            log_level="INFO",
            log_format="json",
            github_token="ghp_test",
            github_repo="owner/repo",
        )
        container = create_production_container(settings)

        # Call scan_services to exercise the closure
        services = container.scan_services()
        assert len(services) == 1
        assert services[0].name == "my-api"
        assert services[0].has_rules_files is False
    finally:
        _test_hooks.container_github_client_factory = original


def test_github_scanning_via_container_get_profile() -> None:
    """Test that get_codebase_profile works on container with GitHub scanning."""
    from platform_codebase import FakeGitHubClient

    from opportunity_radar_api import _test_hooks

    # Create a fake GitHub client with ML dependencies
    fake_client = FakeGitHubClient(
        directories={"libs": ["ml-lib"], "services": []},
        files={
            "libs/ml-lib/pyproject.toml": """
[tool.poetry]
name = "ml-lib"

[tool.poetry.dependencies]
python = "^3.11"
xgboost = "^2.0.0"
lightgbm = "^4.0.0"
""",
        },
    )

    def fake_factory(token: str) -> FakeGitHubClient:
        _ = token  # Unused
        return fake_client

    original = _test_hooks.container_github_client_factory
    _test_hooks.container_github_client_factory = fake_factory

    try:
        settings = OpportunityRadarSettings(
            kaggle_api_token="",
            port=8010,
            log_level="INFO",
            log_format="json",
            github_token="ghp_test",
            github_repo="owner/repo",
        )
        container = create_production_container(settings)

        # Call get_codebase_profile to exercise github_profile_factory
        profile = container.get_codebase_profile()

        # Verify capabilities were detected from GitHub-scanned data
        assert "xgboost" in profile.ml_backends
        assert "lightgbm" in profile.ml_backends
        cap_names = [c.name for c in profile.capabilities]
        assert "xgboost_tabular" in cap_names
        assert "lightgbm_tabular" in cap_names
    finally:
        _test_hooks.container_github_client_factory = original
