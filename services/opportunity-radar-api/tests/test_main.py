"""Tests for main module."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from opportunity_radar_api.api.container import ServiceContainer
from opportunity_radar_api.api.main import create_app


def test_create_app_with_container(fake_container: ServiceContainer) -> None:
    """Test creating app with custom container."""
    app = create_app(container=fake_container)

    assert app.title == "opportunity-radar-api"
    assert app.version == "0.1.0"


def test_create_app_includes_routers(fake_container: ServiceContainer) -> None:
    """Test that app includes all routers."""
    app = create_app(container=fake_container)
    client = TestClient(app)

    # Health route should work
    response = client.get("/healthz")
    assert response.status_code == 200
    data = narrow_json_to_dict(load_json_str(response.text))
    assert data["status"] == "ok"

    # Codebase route should work
    response = client.get("/codebase/profile")
    assert response.status_code == 200

    # Kaggle route should work
    response = client.get("/kaggle/competitions")
    assert response.status_code == 200

    # Devpost route should work
    response = client.get("/devpost/hackathons")
    assert response.status_code == 200


def test_create_app_without_container(tmp_path: Path) -> None:
    """Test creating app without container creates production container."""
    # Create libs directory for monorepo detection
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()

    # Create app with explicit monorepo_root (triggers production container path)
    app = create_app(monorepo_root=tmp_path)

    assert app.title == "opportunity-radar-api"


def test_create_app_with_settings_but_no_container(tmp_path: Path) -> None:
    """Test creating app with settings but no container."""
    from opportunity_radar_api.config import OpportunityRadarSettings

    # Create libs directory for monorepo detection
    libs_dir = tmp_path / "libs"
    libs_dir.mkdir()

    settings = OpportunityRadarSettings(
        kaggle_api_token="",
        port=8010,
        log_level="INFO",
        log_format="json",
        github_token=None,
        github_repo=None,
    )

    # Pass settings directly (skips load_settings)
    app = create_app(settings=settings, monorepo_root=tmp_path)

    assert app.title == "opportunity-radar-api"
