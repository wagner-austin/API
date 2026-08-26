"""Decoration award names: the client's ``nb`` table, slot by level.

The 0x4E decoration event carries ``(tank_id, slot, level)`` and the
real client resolves the award name as ``nb[3 * slot + level - 1]``
([[decoration-encoding]], tpclient.js Sf/V.N, traced 2026-06-19).
This module carries that table so the bot can NAME an award at decode
time — the first live 0x4E (2026-08-26 05:11:16, Arterial's BRONZE
TANK AWARD for 100 career deactivations) logged as bare numbers.

Slots 0-8, levels 1-3. The earning criteria (also from the in-client
guide) live on the wiki page; only the names are wire-relevant.
"""

from __future__ import annotations

DECORATION_NAMES: tuple[tuple[str, str, str], ...] = (
    ("SINGLE STAR", "DOUBLE STAR", "TRIPLE STAR"),
    ("BRONZE TANK AWARD", "SILVER TANK AWARD", "GOLDEN TANK AWARD"),
    ("COMBAT HONOR MEDAL", "BATTLE HONOR MEDAL", "HEROIC HONOR MEDAL"),
    ("SHINING SWORD", "BATTERED SWORD", "RUSTY SWORD"),
    ("BRONZE SHIELD", "SILVER SHIELD", "DEFENDER OF THE TRUTH"),
    ("BRONZE CUP", "SILVER CUP", "GOLDEN CUP"),
    ("PURPLE HEART", "PURPLE HEART 2", "PURPLE HEART 3"),
    ("WAR CORRESPONDENT", "WAR CORRESPONDENT 2", "WAR CORRESPONDENT 3"),
    ("LIGHTBULB AWARD", "LIGHTBULB 2", "LIGHTBULB 3"),
)
"""The nine award categories, three tiers each, client order."""


def unpack_decoration_state(state: bytes) -> tuple[int, ...]:
    """Unpack the 4 packed decoration bytes into 9 award levels.

    The client's ``yg`` law ([[decoration-encoding]]): the 4 bytes form
    a little-endian 32-bit integer, 2 bits per slot, slots 0-8. Live
    verification 2026-08-26: Arterial's ``04000000`` unpacks to slot 1
    level 1 (the bronze earned an hour earlier) and Artax's
    ``1e000000`` to slots (2, 3, 1) — DOUBLE STAR, GOLDEN TANK AWARD,
    COMBAT HONOR MEDAL.

    Args:
        state: The 4 decoration bytes from TankInfo / TankStatusFull.

    Returns:
        Nine award levels, slot order, each 0 (none) through 3.

    Raises:
        ValueError: When ``state`` is not exactly 4 bytes — the wire
            layout fixes the width, so anything else is a decode bug.
    """
    if len(state) != 4:
        raise ValueError(f"decoration state must be 4 bytes, got {len(state)}")
    value = state[0] | (state[1] << 8) | (state[2] << 16) | (state[3] << 24)
    return tuple((value >> (2 * slot)) & 3 for slot in range(9))


def decoration_names_from_state(state: bytes) -> tuple[str, ...]:
    """Return the award names a packed decoration state carries.

    Args:
        state: The 4 decoration bytes from TankInfo / TankStatusFull.

    Returns:
        Names of every held award, slot order; empty when undecorated.
    """
    return tuple(
        DECORATION_NAMES[slot][level - 1]
        for slot, level in enumerate(unpack_decoration_state(state))
        if level > 0
    )


def decoration_name(slot: int, level: int) -> str | None:
    """Return the award name for a slot/level pair.

    Args:
        slot: Award category, 0-8 in the known client table.
        level: Award tier, 1-3.

    Returns:
        The client's award name, or ``None`` when the pair is outside
        the known table — the wire fields are raw bytes and a future
        server-side award category must render as its numbers, not
        crash the decode.
    """
    if 0 <= slot < len(DECORATION_NAMES) and 1 <= level <= 3:
        return DECORATION_NAMES[slot][level - 1]
    return None


__all__ = [
    "DECORATION_NAMES",
    "decoration_name",
    "decoration_names_from_state",
    "unpack_decoration_state",
]
