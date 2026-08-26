"""Tests for the decoration award-name table.

The table is the client's ``nb`` array ([[decoration-encoding]]); the
flagship pin is the first live 0x4E ever received (2026-08-26
05:11:16): Arterial's slot=1 level=1 — the BRONZE TANK AWARD for 100
career deactivations.
"""

from __future__ import annotations

from tankpit_bot.protocol.decorations import DECORATION_NAMES, decoration_name


def test_the_first_live_award_resolves_to_the_bronze_tank_award() -> None:
    """Slot 1 level 1 is the 100-deactivation bronze — Arterial's medal."""
    assert decoration_name(1, 1) == "BRONZE TANK AWARD"


def test_every_table_cell_resolves_by_slot_and_level() -> None:
    """All 27 known awards resolve to their table row."""
    for slot, row in enumerate(DECORATION_NAMES):
        for level in (1, 2, 3):
            assert decoration_name(slot, level) == row[level - 1]


def test_corner_cells_of_the_table() -> None:
    """First and last cells match the client's nb order."""
    assert decoration_name(0, 1) == "SINGLE STAR"
    assert decoration_name(8, 3) == "LIGHTBULB 3"


def test_unknown_pairs_resolve_to_none() -> None:
    """Outside the known table the wire numbers stand alone.

    The fields are raw bytes; a future server-side award category
    must render as numbers, never crash the decode.
    """
    assert decoration_name(9, 1) is None
    assert decoration_name(-1, 1) is None
    assert decoration_name(1, 0) is None
    assert decoration_name(1, 4) is None
