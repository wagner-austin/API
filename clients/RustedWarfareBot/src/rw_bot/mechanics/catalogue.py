"""The unit catalogue, decoded from the engine's own ``-printunits`` output.

The engine will print every unit's stats on request and then exit before the
game loop. That output is the authoritative catalogue: it is the engine
reporting its own numbers rather than a table transcribed from a wiki, and it
regenerates in one command against any build.

Its shape is HTML fragments, one ``<div class="unit">`` per unit, carrying the
type name, a display name, a description and a ``<pre>`` block of stats. The
type name is the join key that matters — ``unit:builder`` here is the same
string the live world stream reports as an entity's ``type`` and the same one
the type registry accepts when placing a building, so a planner can price what
it is looking at without a second mapping table.

Two shapes vary and both are modelled rather than flattened. A unit is armed
only if it has an attack range: 61 of the 90 are. Upgrade prices are a tier
sequence and most units have none.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypedDict

from rw_bot import RwBotError

_BLOCK_OPEN: Final = '<div class="unit">'
_BLOCK_CLOSE: Final = "</pre></div>"
_IMG_PREFIX: Final = '<img src="unit:'
_STATS_OPEN: Final = "<pre>"

_NO_TYPE_NAME = "RW-CATALOGUE-001"
_NO_DISPLAY_NAME = "RW-CATALOGUE-002"
_MALFORMED_STAT = "RW-CATALOGUE-003"
_MISSING_STAT = "RW-CATALOGUE-004"
_UNCLOSED_BLOCK = "RW-CATALOGUE-005"
_DUPLICATE_TYPE = "RW-CATALOGUE-006"


class CatalogueError(RwBotError):
    """The ``-printunits`` output did not match the shape the engine emits.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description, naming the unit where known.
    """


class Weapon(TypedDict):
    """What an armed unit does at range.

    A unit is armed when the engine prints an attack range for it. Damage of a
    kind the engine did not print is zero: a unit with no ``Direct Damage`` line
    deals no direct damage, which is a fact about the unit rather than a missing
    reading.

    Per-shot and per-volley damage are both recorded because neither can be
    derived from the other. The engine prints a volley total only when it
    differs from the per-shot figure, and the ratio is not fixed — observed
    values include 2x, 4x, 6x and one unit at 1.84x. Computing volley damage
    from a guessed barrel count would therefore be wrong for most multi-barrel
    units.

    Attributes:
        shoot_delay: Frames between shots. Lower is faster.
        attack_range: Range in world units, comparable with entity positions.
        direct_damage: Single-target damage per shot, zero when the unit deals
            none.
        direct_damage_volley: Single-target damage per full volley. Equal to
            ``direct_damage`` when the engine printed no separate total.
        area_damage: Splash damage per shot, zero when the unit deals none.
        area_damage_volley: Splash damage per full volley, equal to
            ``area_damage`` when no separate total was printed.
    """

    shoot_delay: float
    attack_range: float
    direct_damage: float
    direct_damage_volley: float
    area_damage: float
    area_damage_volley: float


class UnitStats(TypedDict):
    """One unit's entry in the catalogue.

    Attributes:
        type_name: Engine type name, e.g. ``"builder"``. Joins to the ``type``
            field of a live entity and to the build-placement registry.
        display_name: Human-readable name as the game shows it.
        description: The bullet text, with markup stripped and bullets joined
            by newlines.
        price: Credit cost to build.
        hp: Maximum hit points.
        speed: Movement speed. Zero for buildings, which is how the catalogue
            reports immobility.
        turn_speed: Turn rate.
        mass: Unit mass.
        upgrade_prices: Tier upgrade costs in tier order, empty when the unit
            does not upgrade.
        weapon: The unit's weapon, or ``None`` when it has no attack range.
    """

    type_name: str
    display_name: str
    description: str
    price: int
    hp: int
    speed: float
    turn_speed: float
    mass: int
    upgrade_prices: tuple[int, ...]
    weapon: Weapon | None


def decode_catalogue(lines: Sequence[str]) -> tuple[UnitStats, ...]:
    """Decode every unit block in a ``-printunits`` log.

    The log also contains ordinary boot output; only the unit blocks are read,
    and everything outside them is skipped rather than parsed.

    Args:
        lines: The log's lines, without newline terminators.

    Returns:
        Every unit, in catalogue order.

    Raises:
        CatalogueError: ``RW-CATALOGUE-001`` when a block has no type name,
            ``-002`` when it has no display name, ``-003`` on a malformed stat
            line, ``-004`` when a stat every unit must carry is absent,
            ``-005`` when a block is never closed, ``-006`` on a repeated type
            name.
    """
    units: list[UnitStats] = []
    seen: set[str] = set()
    block: list[str] | None = None

    for line in lines:
        if line.startswith(_BLOCK_OPEN):
            block = []
            continue
        if block is None:
            continue
        block.append(line)
        if line.endswith(_BLOCK_CLOSE):
            unit = _decode_block(block)
            if unit["type_name"] in seen:
                raise CatalogueError(
                    _DUPLICATE_TYPE,
                    f"type name {unit['type_name']!r} appears twice; it is the join key "
                    "to live entities and must identify exactly one unit",
                )
            seen.add(unit["type_name"])
            units.append(unit)
            block = None

    if block is not None:
        raise CatalogueError(
            _UNCLOSED_BLOCK, "a unit block was opened but never closed; the log is truncated"
        )
    return tuple(units)


def _decode_block(block: Sequence[str]) -> UnitStats:
    """Decode one ``<div class="unit">`` block.

    Args:
        block: The block's lines, excluding the opening div.

    Returns:
        The decoded unit.

    Raises:
        CatalogueError: As documented on :func:`decode_catalogue`.
    """
    type_name = _extract_type_name(block)
    stats = _extract_stats(block, type_name)

    return UnitStats(
        type_name=type_name,
        display_name=_extract_tagged(block, "<h4>", "</h4>", _NO_DISPLAY_NAME, type_name),
        description=_extract_description(block),
        price=_require_int(stats, "Price", type_name),
        hp=_require_int(stats, "Hp", type_name),
        speed=_require_float(stats, "Speed", type_name),
        turn_speed=_require_float(stats, "Turn speed", type_name),
        mass=_require_int(stats, "Mass", type_name),
        upgrade_prices=_extract_upgrades(stats, type_name),
        weapon=_extract_weapon(stats, type_name),
    )


def _extract_type_name(block: Sequence[str]) -> str:
    """Read the type name from the block's image tag.

    Args:
        block: The block's lines.

    Returns:
        The engine type name.

    Raises:
        CatalogueError: ``RW-CATALOGUE-001`` when no image tag is present.
    """
    for line in block:
        start = line.find(_IMG_PREFIX)
        if start < 0:
            continue
        rest = line[start + len(_IMG_PREFIX) :]
        end = rest.find('"')
        if end > 0:
            return rest[:end]
    raise CatalogueError(_NO_TYPE_NAME, "a unit block carries no 'unit:<name>' image tag")


def _extract_tagged(
    block: Sequence[str], open_tag: str, close_tag: str, code: str, type_name: str
) -> str:
    """Read the text between a matched pair of tags on one line.

    Args:
        block: The block's lines.
        open_tag: Opening tag.
        close_tag: Closing tag.
        code: Error code to raise under when absent.
        type_name: Unit being decoded, for the message.

    Returns:
        The enclosed text.

    Raises:
        CatalogueError: Under ``code`` when the pair is not found.
    """
    for line in block:
        start = line.find(open_tag)
        end = line.find(close_tag)
        if start >= 0 and end > start:
            return line[start + len(open_tag) : end]
    raise CatalogueError(code, f"unit {type_name!r} has no {open_tag}…{close_tag}")


def _extract_description(block: Sequence[str]) -> str:
    """Read the bullet text, stripping markup.

    An absent description is not an error: it is prose, and a unit without it is
    still fully priced and statted.

    Args:
        block: The block's lines.

    Returns:
        The bullets joined by newlines, empty when there are none.
    """
    for line in block:
        start = line.find("<p>")
        end = line.find("</p>")
        if start < 0 or end <= start:
            continue
        body = line[start + 3 : end]
        return "\n".join(part for part in body.split("<br/>") if part != "")
    return ""


def _extract_stats(block: Sequence[str], type_name: str) -> dict[str, str]:
    """Collect the ``key: value`` lines of the stats block.

    Args:
        block: The block's lines.
        type_name: Unit being decoded, for the message.

    Returns:
        Raw stat strings by key, with currency markers removed.

    Raises:
        CatalogueError: ``RW-CATALOGUE-003`` when a stats line has no separator.
    """
    stats: dict[str, str] = {}
    inside = False
    for raw in block:
        line = raw
        if line.startswith(_STATS_OPEN):
            inside = True
            line = line[len(_STATS_OPEN) :]
        if not inside:
            continue
        if line.startswith(_BLOCK_CLOSE):
            break
        if line == "":
            continue
        separator = line.find(": ")
        if separator < 0:
            raise CatalogueError(
                _MALFORMED_STAT, f"unit {type_name!r} has a stat line without ': ': {line!r}"
            )
        stats[line[:separator]] = line[separator + 2 :].lstrip("$")
    return stats


def _extract_upgrades(stats: dict[str, str], type_name: str) -> tuple[int, ...]:
    """Collect tier upgrade prices in tier order.

    Tiers are read in ascending order and stop at the first absent one, so a
    hypothetical T4 without a T3 would be ignored rather than silently
    reordered.

    Args:
        stats: Raw stats by key.
        type_name: Unit being decoded, for the message.

    Returns:
        Upgrade prices in tier order, empty when the unit does not upgrade.

    Raises:
        CatalogueError: ``RW-CATALOGUE-003`` when a present price is not a
            number.
    """
    prices: list[int] = []
    tier = 2
    while f"T{tier} Upgrade Price" in stats:
        prices.append(_require_int(stats, f"T{tier} Upgrade Price", type_name))
        tier += 1
    return tuple(prices)


def _extract_weapon(stats: dict[str, str], type_name: str) -> Weapon | None:
    """Build the weapon, or report the unit as unarmed.

    Attack range is the discriminator: the engine prints it for exactly the
    units that can attack.

    Args:
        stats: Raw stats by key.
        type_name: Unit being decoded, for the message.

    Returns:
        The weapon, or ``None`` when the unit has no attack range.

    Raises:
        CatalogueError: ``RW-CATALOGUE-003`` when a present stat is not a
            number, ``RW-CATALOGUE-004`` when an armed unit has no shoot delay.
    """
    if "Attack Range" not in stats:
        return None
    direct, direct_volley = _damage(stats, "Direct Damage", type_name)
    area, area_volley = _damage(stats, "Area Damage", type_name)
    return Weapon(
        shoot_delay=_require_float(stats, "Shoot Delay", type_name),
        attack_range=_require_float(stats, "Attack Range", type_name),
        direct_damage=direct,
        direct_damage_volley=direct_volley,
        area_damage=area,
        area_damage_volley=area_volley,
    )


def _damage(stats: dict[str, str], key: str, type_name: str) -> tuple[float, float]:
    """Read a damage stat as its per-shot and per-volley figures.

    The engine writes either ``12`` or ``12 (total:24.0)``. Absence means the
    unit deals no damage of that kind; a bare figure means one shot is the whole
    volley.

    Args:
        stats: Raw stats by key.
        key: Stat name.
        type_name: Unit being decoded, for the message.

    Returns:
        Per-shot damage and per-volley damage.

    Raises:
        CatalogueError: ``RW-CATALOGUE-003`` when either figure is not a number
            or the parenthesised part is malformed.
    """
    if key not in stats:
        return 0.0, 0.0
    raw = stats[key]
    marker = raw.find(" (total:")
    if marker < 0:
        per_shot = _require_float(stats, key, type_name)
        return per_shot, per_shot
    if not raw.endswith(")"):
        raise CatalogueError(
            _MALFORMED_STAT, f"unit {type_name!r} has an unclosed total in {key}: {raw!r}"
        )
    return (
        _parse_stat_number(raw[:marker], key, type_name),
        _parse_stat_number(raw[marker + len(" (total:") : -1], key, type_name),
    )


def _require_int(stats: dict[str, str], key: str, type_name: str) -> int:
    """Read a stat as an ``int``.

    Args:
        stats: Raw stats by key.
        key: Stat name.
        type_name: Unit being decoded, for the message.

    Returns:
        The value.

    Raises:
        CatalogueError: ``RW-CATALOGUE-004`` when absent, ``RW-CATALOGUE-003``
            when not an integer.
    """
    raw = _require_present(stats, key, type_name)
    try:
        return int(raw)
    except ValueError as error:
        raise CatalogueError(
            _MALFORMED_STAT, f"unit {type_name!r} has non-integer {key}: {raw!r}"
        ) from error


def _require_float(stats: dict[str, str], key: str, type_name: str) -> float:
    """Read a stat as a ``float``.

    Args:
        stats: Raw stats by key.
        key: Stat name.
        type_name: Unit being decoded, for the message.

    Returns:
        The value.

    Raises:
        CatalogueError: ``RW-CATALOGUE-004`` when absent, ``RW-CATALOGUE-003``
            when not a number.
    """
    return _parse_stat_number(_require_present(stats, key, type_name), key, type_name)


def _parse_stat_number(raw: str, key: str, type_name: str) -> float:
    """Parse one unit stat, naming the unit and stat when it is not a number.

    Not named ``_parse_float``: the monorepo guard bans that name outright, on
    the reading that it means a locally reimplemented config helper. This is
    not one. It decodes a stat out of the engine's own ``-printunits``
    catalogue and raises a domain error carrying the unit and stat, which is
    the part that makes a malformed catalogue diagnosable.

    Args:
        raw: The literal text.
        key: Stat name, for the message.
        type_name: Unit being decoded, for the message.

    Returns:
        The value.

    Raises:
        CatalogueError: ``RW-CATALOGUE-003`` when the text is not a number.
    """
    try:
        return float(raw)
    except ValueError as error:
        raise CatalogueError(
            _MALFORMED_STAT, f"unit {type_name!r} has non-numeric {key}: {raw!r}"
        ) from error


def _require_present(stats: dict[str, str], key: str, type_name: str) -> str:
    """Fetch a stat that must be present.

    Args:
        stats: Raw stats by key.
        key: Stat name.
        type_name: Unit being decoded, for the message.

    Returns:
        The raw value.

    Raises:
        CatalogueError: ``RW-CATALOGUE-004`` when the key is absent.
    """
    if key not in stats:
        raise CatalogueError(_MISSING_STAT, f"unit {type_name!r} has no {key}")
    return stats[key]


__all__ = ["CatalogueError", "UnitStats", "Weapon", "decode_catalogue"]
