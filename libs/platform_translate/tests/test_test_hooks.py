"""Tests for platform_translate._test_hooks module."""

from __future__ import annotations

import pytest

from platform_translate import _test_hooks
from platform_translate.testing import reset_hooks
from platform_translate.types import TranslatorConfig


class TestDefaultHooks:
    """Tests for default hook values."""

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_backend_factory_default(self) -> None:
        """backend_factory defaults to _default_backend_factory."""
        reset_hooks()
        assert _test_hooks.backend_factory is _test_hooks._default_backend_factory


class TestDefaultBackendFactory:
    """Tests for _default_backend_factory function."""

    def teardown_method(self) -> None:
        """Reset hooks after each test."""
        reset_hooks()

    def test_unsupported_backend_raises(self) -> None:
        """Raises ValueError for unsupported backend."""
        config = TranslatorConfig(
            backend="unsupported",
            api_key="key",
            model="model",
        )
        with pytest.raises(ValueError, match="Unsupported backend"):
            _test_hooks._default_backend_factory(config)

    def test_anthropic_backend_creation(self) -> None:
        """Creates AnthropicBackend for anthropic backend type."""
        config = TranslatorConfig(
            backend="anthropic",
            api_key="test-api-key",
            model="claude-3-haiku-20240307",
        )
        backend = _test_hooks._default_backend_factory(config)
        assert backend.backend_id == "anthropic"

    def test_openai_backend_creation(self) -> None:
        """Creates OpenAIBackend for openai backend type."""
        config = TranslatorConfig(
            backend="openai",
            api_key="test-api-key",
            model="gpt-4o-mini",
        )
        backend = _test_hooks._default_backend_factory(config)
        assert backend.backend_id == "openai"


class TestExports:
    """Tests for module exports."""

    def test_all_protocols_exported(self) -> None:
        """All protocol types are in __all__."""
        assert "BackendFactoryProtocol" in _test_hooks.__all__

    def test_all_defaults_exported(self) -> None:
        """All default implementations are in __all__."""
        assert "_default_backend_factory" in _test_hooks.__all__

    def test_all_hooks_exported(self) -> None:
        """All hook variables are in __all__."""
        assert "backend_factory" in _test_hooks.__all__
