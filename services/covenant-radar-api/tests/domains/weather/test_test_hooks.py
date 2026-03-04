"""Tests for weather domain _test_hooks module."""

from __future__ import annotations

from covenant_radar_api.domains.weather._test_hooks import __all__


class TestWeatherTestHooks:
    """Tests for weather _test_hooks module."""

    def test_no_hooks_exported(self) -> None:
        """No hookable dependencies exist in weather domain."""
        assert __all__ == []
