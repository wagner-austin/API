"""The reflex layer's table row, sent over the order wire.

Split from :mod:`rw_bot.wire.command` because a posture is not an order and
never was: every order addresses a unit by its engine identity, while this
addresses a *type*, and the agent stores it in the reflex layer's table
rather than dispatching it ([[community-play-strategies]]). The line format
and the strict parser on the far side are shared with the orders, which is
why the field validators are imported from there -- a value is legal on this
verb exactly when it is legal on any other.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from rw_bot.wire.command import CommandError, require_finite, require_type_name

_NOT_FINITE = "RW-CMD-002"
_BAD_FLOOR = "RW-CMD-005"


class PostureOrder(TypedDict):
    """Set one unit type's standing reflex posture on the agent.

    Not an order to a unit: the reflex layer runs on the game thread between
    samples, and this line is a row of its table -- the type's reach (from
    the catalogue the planner owns, so the agent never guesses a stat) and
    the two reflexes the doctrine plays: kiting the reach band, and fleeing
    below the health floor ([[community-play-strategies]]).

    Attributes:
        kind: Discriminator, always ``"posture"``.
        type: The unit type name the row describes.
        reach: The type's attack range, in world units.
        speed: The type's movement speed. The kite gate: a slower unit
            cannot out-step what outruns it, so the agent kites only with
            a speed advantage over the chaser.
        kite: Whether owned units of the type hold the reach band.
        hp_floor: Percent of health below which owned units flee, zero for
            never.
    """

    kind: Literal["posture"]
    type: str
    reach: float
    speed: float
    kite: bool
    hp_floor: int


def posture_order(
    *, type_name: str, reach: float, speed: float, kite: bool, hp_floor: int
) -> PostureOrder:
    """Build a validated posture row.

    Args:
        type_name: The unit type the row describes.
        reach: The type's attack range.
        speed: The type's movement speed.
        kite: Whether owned units of the type hold the reach band.
        hp_floor: Percent of health below which owned units flee.

    Returns:
        The order.

    Raises:
        CommandError: ``RW-CMD-001`` on an uncarryable type name,
            ``RW-CMD-002`` on a non-finite or negative reach,
            ``RW-CMD-005`` on a floor outside 0-100.
    """
    require_type_name(type_name, "posture")
    require_finite(reach, "reach")
    require_finite(speed, "speed")
    if reach < 0.0 or speed < 0.0:
        raise CommandError(
            _NOT_FINITE,
            f"a posture reach and speed must be non-negative, got {reach!r}/{speed!r}",
        )
    if hp_floor < 0 or hp_floor > 100:
        raise CommandError(
            _BAD_FLOOR,
            f"a posture hp_floor is a percent, 0-100, got {hp_floor}",
        )
    return PostureOrder(
        kind="posture",
        type=type_name,
        reach=reach,
        speed=speed,
        kite=kite,
        hp_floor=hp_floor,
    )


def encode_posture(order: PostureOrder) -> str:
    """Render a posture row as one wire line.

    Args:
        order: The order to encode.

    Returns:
        One JSON object, without a trailing newline.
    """
    return (
        f'{{"kind":"posture","type":"{order["type"]}","reach":{order["reach"]},'
        f'"speed":{order["speed"]},'
        f'"kite":{1 if order["kite"] else 0},"hp_floor":{order["hp_floor"]}}}'
    )


__all__ = ["PostureOrder", "encode_posture", "posture_order"]
