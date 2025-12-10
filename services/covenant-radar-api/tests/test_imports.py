"""Tests for package imports and exports."""

from __future__ import annotations


def test_main_package_imports() -> None:
    """Test main package can be imported."""
    import covenant_radar_api

    assert covenant_radar_api.__all__ == []


def test_core_package_exports() -> None:
    """Test core package exports ServiceContainer, Settings, and settings_from_env."""
    from covenant_radar_api.core import ServiceContainer, Settings, settings_from_env

    assert Settings is not None
    assert callable(settings_from_env)
    assert ServiceContainer is not None


def test_api_package_imports() -> None:
    """Test api package can be imported."""
    import covenant_radar_api.api

    assert covenant_radar_api.api.__all__ == []


def test_api_routes_package_imports() -> None:
    """Test api.routes package can be imported."""
    import covenant_radar_api.api.routes

    assert covenant_radar_api.api.routes.__all__ == []


def test_infra_package_imports() -> None:
    """Test infra package can be imported."""
    import covenant_radar_api.infra

    assert covenant_radar_api.infra.__all__ == []


def test_worker_package_imports() -> None:
    """Test worker package exports main."""
    from covenant_radar_api.worker import main

    assert callable(main)


def test_health_routes_build_router_exported() -> None:
    """Test health routes exports build_router."""
    from covenant_radar_api.api.routes.health import build_router

    assert callable(build_router)


def test_scripts_package_imports() -> None:
    """Test scripts package can be imported."""
    import scripts

    assert scripts.__all__ == []
