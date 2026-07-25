"""Orders the planner sends to the agent.

The outbound half of the wire contract. Each order is one flat JSON object on
one line, in exactly the shape ``rwbot.agent.CommandRecord`` accepts — the
agent rejects anything else loudly rather than skipping it, so this encoder
exists to make malformed lines unrepresentable rather than merely unlikely.

Two verbs are defined, matching the two the agent can dispatch: move a unit to
a world position, and have a builder place a structure there. Both address a
unit by its engine identity, never by roster position — position renumbers
whenever anything is built or dies.
"""

from __future__ import annotations

from typing import Literal, TypedDict

from rw_bot import RwBotError

_BLANK_TYPE = "RW-CMD-001"
_NOT_FINITE = "RW-CMD-002"


class CommandError(RwBotError):
    """An order could not be encoded because it is not well formed.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending field.
    """


class MoveOrder(TypedDict):
    """Send one unit to a world position.

    Attributes:
        kind: Discriminator, always ``"move"``.
        unit_id: Engine identity of the unit to order.
        x: Destination world x.
        y: Destination world y.
    """

    kind: Literal["move"]
    unit_id: int
    x: float
    y: float


class BuildOrder(TypedDict):
    """Have one builder place a structure at a world position.

    Attributes:
        kind: Discriminator, always ``"build"``.
        unit_id: Engine identity of the builder.
        type_name: Registry name of the structure, e.g. ``"landFactory"``.
        x: Placement world x.
        y: Placement world y.
    """

    kind: Literal["build"]
    unit_id: int
    type_name: str
    x: float
    y: float


def move_order(*, unit_id: int, x: float, y: float) -> MoveOrder:
    """Build a validated move order.

    Args:
        unit_id: Engine identity of the unit to order.
        x: Destination world x.
        y: Destination world y.

    Returns:
        The order.

    Raises:
        CommandError: ``RW-CMD-002`` when a coordinate is not finite.
    """
    _require_finite(x, "x")
    _require_finite(y, "y")
    return MoveOrder(kind="move", unit_id=unit_id, x=x, y=y)


def build_order(*, unit_id: int, type_name: str, x: float, y: float) -> BuildOrder:
    """Build a validated build order.

    Args:
        unit_id: Engine identity of the builder.
        type_name: Registry name of the structure to place.
        x: Placement world x.
        y: Placement world y.

    Returns:
        The order.

    Raises:
        CommandError: ``RW-CMD-001`` when the type name is blank,
            ``RW-CMD-002`` when a coordinate is not finite.
    """
    if type_name.strip() == "":
        raise CommandError(_BLANK_TYPE, "a build order needs a unit-type name")
    _require_finite(x, "x")
    _require_finite(y, "y")
    return BuildOrder(kind="build", unit_id=unit_id, type_name=type_name, x=x, y=y)


def encode_move(order: MoveOrder) -> str:
    """Render a move order as one wire line.

    Args:
        order: The order to encode.

    Returns:
        One JSON object, without a trailing newline.
    """
    return f'{{"kind":"move","unit_id":{order["unit_id"]},"x":{order["x"]!r},"y":{order["y"]!r}}}'


def encode_build(order: BuildOrder) -> str:
    """Render a build order as one wire line.

    The type name is emitted unescaped: registry names are drawn from the
    engine's own ``.ini`` vocabulary and contain no quotes or backslashes. A
    name that did would be rejected by the agent's parser rather than
    misinterpreted, because the parser reads one flat object and fails on
    anything ambiguous.

    Args:
        order: The order to encode.

    Returns:
        One JSON object, without a trailing newline.

    Raises:
        CommandError: ``RW-CMD-001`` when the type name contains a character
            the flat wire format cannot carry.
    """
    type_name = order["type_name"]
    for forbidden in ('"', "\\", "\n", "\r"):
        if forbidden in type_name:
            raise CommandError(
                _BLANK_TYPE,
                f"unit-type name {type_name!r} contains a character the wire format does not carry",
            )
    return (
        f'{{"kind":"build","unit_id":{order["unit_id"]},'
        f'"x":{order["x"]!r},"y":{order["y"]!r},"type":"{type_name}"}}'
    )


def _require_finite(value: float, field: str) -> None:
    """Reject a coordinate JSON cannot carry.

    Args:
        value: The coordinate.
        field: Field name, for the message.

    Raises:
        CommandError: ``RW-CMD-002`` when the value is not finite.
    """
    if value != value or value in (float("inf"), float("-inf")):
        raise CommandError(_NOT_FINITE, f"order field {field!r} must be finite, got {value}")


__all__ = [
    "BuildOrder",
    "CommandError",
    "MoveOrder",
    "build_order",
    "encode_build",
    "encode_move",
    "move_order",
]
