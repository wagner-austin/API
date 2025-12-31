"""Shared fixtures for Google AI integration tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from covenant_radar_api.integrations.google_ai import _test_hooks


def _reset_google_ai_hooks_impl() -> Generator[None, None, None]:
    """Reset Google AI test hooks after each test."""
    orig_factory = _test_hooks.gemini_client_factory

    yield

    _test_hooks.gemini_client_factory = orig_factory


reset_google_ai_hooks = pytest.fixture(autouse=True)(_reset_google_ai_hooks_impl)


__all__ = [
    "reset_google_ai_hooks",
]
