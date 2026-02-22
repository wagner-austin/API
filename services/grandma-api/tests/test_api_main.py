"""Tests for grandma_api.api.main module."""

from __future__ import annotations

from typing import Protocol

from fastapi.testclient import TestClient

from grandma_api.api.main import create_app
from grandma_api.config import GrandmaApiSettings
from grandma_api.core.container import ServiceContainer

from .conftest import make_test_container, set_fake_env


class _AppStateProto(Protocol):
    """Protocol for accessing typed container from app.state."""

    container: ServiceContainer


def _make_test_settings() -> GrandmaApiSettings:
    """Create test settings."""
    return GrandmaApiSettings(
        openai_api_key="sk-test",
        api_token="test-token",
        port=8080,
        log_level="INFO",
        log_format="json",
    )


def test_create_app_with_settings() -> None:
    """Test create_app with explicit settings."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    settings = _make_test_settings()
    container, _, _, _ = make_test_container(settings)
    app = create_app(settings, container=container)

    assert app.title == "Grandma API"
    assert app.version == "0.1.0"


def test_create_app_loads_from_env() -> None:
    """Test create_app loads settings from environment when None."""
    set_fake_env(
        {
            "OPENAI_API_KEY": "sk-from-env",
            "API_TOKEN": "token-from-env",
        }
    )

    app = create_app(None)

    assert app.title == "Grandma API"


def test_create_app_has_healthz_route() -> None:
    """Test create_app includes healthz route by making a request."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    settings = _make_test_settings()
    container, _, _, _ = make_test_container(settings)
    app = create_app(settings, container=container)
    client = TestClient(app)

    # Verify route exists by making a request
    response = client.get("/healthz")
    assert response.status_code == 200


def test_create_app_has_translate_route() -> None:
    """Test create_app includes translate route by making a request."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    settings = _make_test_settings()
    container, _, _, _ = make_test_container(settings)
    app = create_app(settings, container=container)
    client = TestClient(app)

    # Verify route exists - OPTIONS request returns 405 (method not allowed)
    response = client.options("/translate")
    assert response.status_code == 405


def test_create_app_with_text_format() -> None:
    """Test create_app with text log format."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    settings = GrandmaApiSettings(
        openai_api_key="sk-test",
        api_token="test-token",
        port=8080,
        log_level="DEBUG",
        log_format="text",
    )
    container, _, _, _ = make_test_container(settings)
    app = create_app(settings, container=container)

    # Verify app was created with expected metadata
    assert app.title == "Grandma API"
    assert app.description == "Vietnamese to English audio translation API"


def test_create_app_exposes_container() -> None:
    """Test create_app exposes ServiceContainer on app.state."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    settings = _make_test_settings()
    container, _, _, _ = make_test_container(settings)
    app = create_app(settings, container=container)

    state: _AppStateProto = app.state
    assert state.container is container


def test_create_app_creates_container_from_settings() -> None:
    """Test create_app creates container from settings when not provided."""
    set_fake_env({"OPENAI_API_KEY": "sk-test", "API_TOKEN": "token"})

    settings = _make_test_settings()
    app = create_app(settings)

    # Container should be created with matching settings
    state: _AppStateProto = app.state
    assert state.container.settings == settings
