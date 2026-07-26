"""Tests for esports domain _test_hooks module."""

from __future__ import annotations

from covenant_radar_api.domains.esports._test_hooks import __all__


class TestEsportsTestHooks:
    """Tests for esports _test_hooks module."""

    def test_no_hooks_exported(self) -> None:
        """No hookable dependencies exist in the esports domain.

        The extractor is pure computation over a single event, so there is
        nothing external to substitute.
        """
        assert __all__ == []
