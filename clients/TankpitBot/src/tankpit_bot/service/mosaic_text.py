"""The label over each tile of the fleet's four-up picture.

Each tile carries one line over its top margin
(:mod:`tankpit_bot.stream.mosaic`): which bot it is and the room it
plays in, or, for an empty position, that the tile is waiting. What
the bot is doing is not here; it is in the game's own panel in the
picture (board task c896a291).

The encoder reads each label from a file it reloads every frame, so
this module's other half is writing those files whole: through the
atomic replace hook, and only when the words changed, because a file
the encoder is mid-way through reading must never be torn and a
rewrite nobody needed is a write per tile per second for nothing.
"""

from __future__ import annotations

from pathlib import Path

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.types.rooms import DEFAULT_LOBBY_ROOM


def waiting_label(position: int) -> str:
    """Return the label of an empty position.

    Args:
        position: Grid position, 0 to 3.

    Returns:
        The position, counted from one as a viewer counts, and that it
        is waiting.
    """
    return f"Slot {position + 1}  Waiting for a bot"


def bot_label(instance: str, room: str) -> str:
    """Return the label of a live tile.

    Args:
        instance: The bot's instance name.
        room: The room the bot was spawned into; empty means the
            default room, which is named rather than left blank.

    Returns:
        The instance and the room it plays in.
    """
    return f"{instance}  {room or DEFAULT_LOBBY_ROOM}"


class TileTextWriter:
    """Writes label files whole, and only when their words change."""

    def __init__(self) -> None:
        """Start with nothing written."""
        self._written: dict[Path, str] = {}

    def write(self, path: Path, text: str) -> None:
        """Make ``path`` hold ``text``, unless it already does.

        Args:
            path: The label file the encoder draws from.
            text: The words it should draw.
        """
        if self._written.get(path) == text:
            return
        top_hooks.replace_text(path, text)
        self._written[path] = text


__all__ = [
    "TileTextWriter",
    "bot_label",
    "waiting_label",
]
