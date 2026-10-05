"""Reading the tank's fuel total off the bot's ``Fuel: X -> Y`` lines.

The bot logs every fuel change as one WORLD line, ``Fuel: 1100 -> 1055
(-45)``. Two folds of the fleet surface read the total from it: the
operator page's activity tail and the public demo's captions. One
parser here, so the two can never disagree about what the tank holds.
"""

from __future__ import annotations

FUEL_PREFIX = "Fuel: "
"""How a fuel-change message starts."""


def is_fuel_line(message: str) -> bool:
    """Return whether an event message is a fuel-change line.

    Args:
        message: The event's message text.

    Returns:
        True when the message starts with :data:`FUEL_PREFIX`.
    """
    return message.startswith(FUEL_PREFIX)


def fuel_total(message: str) -> int:
    """Parse the total out of a ``Fuel: X -> Y`` line.

    Args:
        message: The event message, known to start with ``Fuel: ``.

    Returns:
        The post-arrow total, or ``-1`` when the line does not end in
        a plain number.
    """
    tail = message[len(FUEL_PREFIX) :].split("->")[-1].strip()
    total = tail.split(" ")[0].split("(")[0].strip()
    return int(total) if total.isdigit() else -1


__all__ = [
    "FUEL_PREFIX",
    "fuel_total",
    "is_fuel_line",
]
