"""Tests for the decoration award-name table.

The table is the client's ``nb`` array ([[decoration-encoding]]); the
flagship pin is the first live 0x4E ever received (2026-08-26
05:11:16): Arterial's slot=1 level=1 — the BRONZE TANK AWARD for 100
career deactivations.
"""

from __future__ import annotations

from tankpit_bot.protocol.decorations import (
    DECORATION_NAMES,
    decoration_name,
    decoration_names_from_state,
    unpack_decoration_state,
)


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


def test_arterials_live_state_unpacks_to_the_bronze() -> None:
    """``04000000`` (Arterial, live 2026-08-26) is slot 1 level 1."""
    state = bytes.fromhex("04000000")
    assert unpack_decoration_state(state) == (0, 1, 0, 0, 0, 0, 0, 0, 0)
    assert decoration_names_from_state(state) == ("BRONZE TANK AWARD",)


def test_artaxs_live_state_unpacks_to_three_medals() -> None:
    """``1e000000`` (Artax, live 2026-08-26) carries three awards.

    DOUBLE STAR (reached Colonel), GOLDEN TANK AWARD (500 career
    deactivations), COMBAT HONOR MEDAL (deactivated 20+ times) — the
    account had been carrying all three unread while the bytes were
    mislabeled "cosmetic skin".
    """
    state = bytes.fromhex("1e000000")
    assert unpack_decoration_state(state) == (2, 3, 1, 0, 0, 0, 0, 0, 0)
    assert decoration_names_from_state(state) == (
        "DOUBLE STAR",
        "GOLDEN TANK AWARD",
        "COMBAT HONOR MEDAL",
    )


def test_undecorated_state_unpacks_to_nothing() -> None:
    """All-zero bytes carry no awards."""
    assert unpack_decoration_state(bytes(4)) == (0,) * 9
    assert decoration_names_from_state(bytes(4)) == ()


def test_high_slots_reach_through_the_third_byte() -> None:
    """Slot 8 lives in bits 16-17 — the third wire byte."""
    # slot 8 level 3 -> bits 16-17 = 0b11 -> byte 2 = 0x03
    state = bytes.fromhex("00000300")
    assert unpack_decoration_state(state)[8] == 3
    assert decoration_names_from_state(state) == ("LIGHTBULB 3",)


def test_wrong_width_state_raises() -> None:
    """Anything but 4 bytes is a decode bug and raises."""
    import pytest

    with pytest.raises(ValueError, match="must be 4 bytes, got 3"):
        unpack_decoration_state(bytes(3))


def test_unknown_pairs_resolve_to_none() -> None:
    """Outside the known table the wire numbers stand alone.

    The fields are raw bytes; a future server-side award category
    must render as numbers, never crash the decode.
    """
    assert decoration_name(9, 1) is None
    assert decoration_name(-1, 1) is None
    assert decoration_name(1, 0) is None
    assert decoration_name(1, 4) is None
