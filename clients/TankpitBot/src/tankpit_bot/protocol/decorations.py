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
]
