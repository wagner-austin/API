"""Tests for :mod:`tankpit_bot.physics.combat`.

Reference: ``wiki/pages/shoot-event-format.md`` (global action queue
and post-departure reroute TTL, measured 2026-07-19).
"""

from __future__ import annotations

from tankpit_bot.physics.combat import REROUTE_TTL_MS


class TestRerouteTtl:
    """The homing-reroute TTL estimate and its measured boundary."""

    def test_estimate_is_12_seconds(self) -> None:
        """The working estimate is the midpoint of the measured window."""
        assert REROUTE_TTL_MS == 12_000

    def test_estimate_sits_inside_the_measured_boundary(self) -> None:
        """Hits observed through +11.0 s, first miss at +13.0 s — the
        constant must stay inside [11.0, 13.0] s until a live pursuit
        miss narrows it."""
        assert 11_000 <= REROUTE_TTL_MS <= 13_000
