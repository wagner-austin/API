"""Tests for domains _test_hooks module."""

from __future__ import annotations

from covenant_radar_api.domains._test_hooks import __all__


class TestDomainsTestHooks:
    """Tests for domains base _test_hooks module."""

    def test_no_hooks_exported(self) -> None:
        """No hookable dependencies exist in domains base layer."""
        assert __all__ == []
