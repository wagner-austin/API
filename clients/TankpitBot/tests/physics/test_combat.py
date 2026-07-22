"""Tests for :mod:`tankpit_bot.physics.combat`.

Reference: ``wiki/pages/shoot-event-format.md`` (global action queue
and post-departure reroute TTL, measured 2026-07-19).
"""

from __future__ import annotations

from tankpit_bot.physics.combat import REROUTE_TTL_MS


class TestRerouteTtl:
    """The homing-reroute TTL and its corpus-measured boundary."""

    def test_value_is_the_corpus_swept_midpoint(self) -> None:
        """The constant is the midpoint of the swept [12.91, 12.93] s.

        2026-07-22 archive sweep: 704 echo-paired hits dense to
        +12.91 s, zero later, dense misses from +12.93 s.
        """
        assert REROUTE_TTL_MS == 12_920

    def test_value_sits_inside_the_measured_boundary(self) -> None:
        """Latest hit +12.91 s, earliest boundary miss +12.93 s — the
        constant must stay inside until a future sweep narrows it."""
        assert 12_910 <= REROUTE_TTL_MS <= 12_930
