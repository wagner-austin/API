"""TypedDicts for the bot service's HTTP request payloads.

What the SPA sends. What a session reports back is the shared
contract in :mod:`tankpit_bot.bus.session_status`.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from tankpit_bot.bus.session_status import WireMode


class ModeCommandDict(TypedDict):
    """Wire payload for ``POST /api/tankbot/mode``.

    Attributes:
        manual_mode: The mode the SPA is asking the bot to hold. Use
            ``"AUTO"`` to restore auto-arbitration; the three
            :data:`AIMode` literals pin the durable HFSM to that mode.
    """

    manual_mode: WireMode


def make_mode_command(manual_mode: WireMode) -> ModeCommandDict:
    """Create a :class:`ModeCommandDict`.

    Args:
        manual_mode: Wire-level mode literal.

    Returns:
        Populated :class:`ModeCommandDict`.
    """
    return ModeCommandDict(manual_mode=manual_mode)


__all__ = [
    "ModeCommandDict",
    "make_mode_command",
]
