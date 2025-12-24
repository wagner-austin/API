"""Tests for grandma_api.asgi module."""

from __future__ import annotations

from .conftest import set_fake_env


def test_asgi_app_has_expected_title() -> None:
    """Test that asgi.app has expected title."""
    set_fake_env(
        {
            "OPENAI_API_KEY": "sk-test-asgi",
            "API_TOKEN": "test-token-asgi",
        }
    )

    # Import after setting up hooks to ensure settings load correctly
    # Use importlib to avoid module caching issues
    import importlib

    import grandma_api.asgi

    importlib.reload(grandma_api.asgi)

    from grandma_api.asgi import app

    assert app.title == "Grandma API"
    assert app.version == "0.1.0"
