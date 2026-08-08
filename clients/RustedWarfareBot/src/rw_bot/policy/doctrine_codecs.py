"""Reading and writing a doctrine's flat payload form.

Split from :mod:`rw_bot.policy.doctrine` at the size cap: the TypedDict and
its field lore are what a policy reader consults, and the codec walk below
is what a file or a queue consults -- they grow for different reasons. The
decode validates every field through the shared require helpers, so a
doctrine from a file, a queue row or a test literal is one and the same
shape ([[policy-loop]]).
"""

from __future__ import annotations

from collections.abc import Mapping

from rw_bot.policy.doctrine import (
    DERIVE_RESERVE,
    NAVTILT_BEHIND,
    NAVTILT_OFF,
    NO_HEAVIES,
    Doctrine,
    DoctrineError,
)
from rw_bot.validation import (
    require_bool,
    require_int,
    require_non_empty_str,
    require_positive_int,
)

_BLANK_GOAL = "RW-DOCTRINE-006"
_BAD_RESERVE = "RW-DOCTRINE-007"
_BAD_GUARD_CAP = "RW-DOCTRINE-008"
_BAD_RAID_SIZE = "RW-DOCTRINE-009"
_BLANK_HEAVY = "RW-DOCTRINE-010"
_BAD_TECH_CAP = "RW-DOCTRINE-012"
_BAD_LURK_COUNT = "RW-DOCTRINE-013"
_BAD_ALLIN_SAMPLE = "RW-DOCTRINE-014"
_BAD_CREEP_HOLD = "RW-DOCTRINE-015"
_BAD_DECOY_COUNT = "RW-DOCTRINE-016"
_BAD_HP_FLOOR = "RW-DOCTRINE-017"
_BAD_STRIKE_RATIO = "RW-DOCTRINE-018"
_BAD_MEDIC_COUNT = "RW-DOCTRINE-019"
_BAD_BUNKER_COUNT = "RW-DOCTRINE-020"
_BAD_FLAME_COUNT = "RW-DOCTRINE-021"
_BAD_CLOSE_RATIO = "RW-DOCTRINE-022"
_BAD_GUN_COUNT = "RW-DOCTRINE-023"
_BAD_NUKE_COUNT = "RW-DOCTRINE-024"
_BAD_NAVTILT = "RW-DOCTRINE-025"


def _count(
    payload: Mapping[str, str | int | float | bool],
    field: str,
    code: str,
    meaning: str,
) -> int:
    """Read one non-negative count field, or refuse it with its own code.

    Every count shares the shape -- zero is a meaningful off-state, below
    zero is a typo -- and each keeps its own error code so a refusal names
    the field's semantics rather than a generic bound.

    Args:
        payload: Field values by name.
        field: The count's name.
        code: The DoctrineError code a refusal carries.
        meaning: The field's semantics, for the message.

    Returns:
        The validated count.

    Raises:
        DoctrineError: When the value is negative.
    """
    value = require_int(payload, field)
    if value < 0:
        raise DoctrineError(code, f"field {field!r} must be >= 0, {meaning}, got {value}")
    return value


def decode_doctrine(payload: Mapping[str, str | int | float | bool]) -> Doctrine:
    """Decode a flat payload into a :class:`Doctrine`.

    Args:
        payload: Field values by name. ``goals`` is comma-separated text, as a
            job line carries it.

    Returns:
        The validated doctrine.

    Raises:
        DecodeError: When a field is absent, mistyped, blank or non-positive.
        DoctrineError: ``RW-DOCTRINE-006`` when the goals carry a blank entry,
            which is a stray comma rather than a unit.
    """
    goals = tuple(part.strip() for part in require_non_empty_str(payload, "goals").split(","))
    if any(goal == "" for goal in goals):
        raise DoctrineError(
            _BLANK_GOAL,
            f"the goals carry a blank entry: {payload['goals']!r}",
        )
    heavies_raw = require_non_empty_str(payload, "heavies")
    heavies = (
        () if heavies_raw == NO_HEAVIES else tuple(part.strip() for part in heavies_raw.split(","))
    )
    if any(heavy == "" for heavy in heavies):
        raise DoctrineError(
            _BLANK_HEAVY,
            f"the heavies carry a blank entry: {payload['heavies']!r}",
        )
    reserve = require_int(payload, "reserve")
    if reserve < DERIVE_RESERVE:
        raise DoctrineError(
            _BAD_RESERVE,
            f"field 'reserve' must be >= 0, or {DERIVE_RESERVE} to derive it, got {reserve}",
        )
    guard_cap = _count(payload, "guard_cap", _BAD_GUARD_CAP, "with 0 meaning the whole reserve")
    raid = _count(payload, "raid", _BAD_RAID_SIZE, "a party size with 0 meaning no raiding")
    tech = _count(payload, "tech", _BAD_TECH_CAP, "factories to unlock with 0 meaning none")
    lurk = _count(payload, "lurk", _BAD_LURK_COUNT, "lurkers to keep alive with 0 meaning none")
    decoys = _count(payload, "decoys", _BAD_DECOY_COUNT, "scatter scouts with 0 for none")
    allin = _count(payload, "allin", _BAD_ALLIN_SAMPLE, "a release observation with 0 for never")
    strike = _count(payload, "strike", _BAD_STRIKE_RATIO, "a rival army-value drop with 0 for off")
    medics = _count(payload, "medics", _BAD_MEDIC_COUNT, "combat engineers to keep alive, 0 none")
    bunkers = _count(payload, "bunkers", _BAD_BUNKER_COUNT, "mobile turrets to keep alive, 0 none")
    flame = _count(payload, "flame", _BAD_FLAME_COUNT, "flame turrets to hold, 0 none")
    close = _count(payload, "close", _BAD_CLOSE_RATIO, "a dominance multiple with 0 for never")
    guns = _count(payload, "guns", _BAD_GUN_COUNT, "top-tier gun turrets to hold, 0 none")
    nukes = _count(payload, "nukes", _BAD_NUKE_COUNT, "nuke launchers to stand, 0 none")
    hp_floor = require_int(payload, "hp_floor")
    if hp_floor < 0 or hp_floor > 100:
        raise DoctrineError(
            _BAD_HP_FLOOR,
            f"field 'hp_floor' is a percent of health, 0-100 with 0 for never, got {hp_floor}",
        )
    navtilt = require_int(payload, "navtilt")
    if navtilt < NAVTILT_OFF or navtilt > NAVTILT_BEHIND:
        raise DoctrineError(
            _BAD_NAVTILT,
            f"field 'navtilt' is 0 off, 1 always, or 2 only-when-behind, got {navtilt}",
        )
    creep = require_int(payload, "creep")
    if creep < 0 or creep > 100:
        raise DoctrineError(
            _BAD_CREEP_HOLD,
            f"field 'creep' must be 0-100, the percent of the way to hold at, got {creep}",
        )
    return Doctrine(
        name=require_non_empty_str(payload, "name"),
        goals=goals,
        heavies=heavies,
        max_workers=require_positive_int(payload, "max_workers"),
        mass=require_positive_int(payload, "mass"),
        reserve=reserve,
        expand=require_bool(payload, "expand"),
        counter=require_bool(payload, "counter"),
        cover=require_bool(payload, "cover"),
        intercept=require_bool(payload, "intercept"),
        guard_cap=guard_cap,
        aa_cover=require_bool(payload, "aa_cover"),
        forward=require_bool(payload, "forward"),
        scout=require_bool(payload, "scout"),
        raid=raid,
        rush=require_bool(payload, "rush"),
        creep=creep,
        riposte=require_bool(payload, "riposte"),
        navtilt=navtilt,
        tech=tech,
        lurk=lurk,
        allin=allin,
        decoys=decoys,
        kite=require_bool(payload, "kite"),
        income_ladder=require_bool(payload, "income_ladder"),
        hp_floor=hp_floor,
        strike=strike,
        medics=medics,
        bunkers=bunkers,
        flame=flame,
        close=close,
        guns=guns,
        nukes=nukes,
    )


def encode_doctrine(doctrine: Doctrine) -> dict[str, str | int | bool]:
    """Encode a :class:`Doctrine` back to a flat payload.

    Round-trips with :func:`decode_doctrine`.

    Args:
        doctrine: The doctrine to encode.

    Returns:
        Field values by name, as :func:`decode_doctrine` reads them.
    """
    return {
        "name": doctrine["name"],
        "goals": ",".join(doctrine["goals"]),
        "heavies": ",".join(doctrine["heavies"]) if doctrine["heavies"] else NO_HEAVIES,
        "max_workers": doctrine["max_workers"],
        "mass": doctrine["mass"],
        "reserve": doctrine["reserve"],
        "expand": doctrine["expand"],
        "counter": doctrine["counter"],
        "cover": doctrine["cover"],
        "intercept": doctrine["intercept"],
        "guard_cap": doctrine["guard_cap"],
        "aa_cover": doctrine["aa_cover"],
        "forward": doctrine["forward"],
        "scout": doctrine["scout"],
        "raid": doctrine["raid"],
        "rush": doctrine["rush"],
        "creep": doctrine["creep"],
        "riposte": doctrine["riposte"],
        "navtilt": doctrine["navtilt"],
        "tech": doctrine["tech"],
        "lurk": doctrine["lurk"],
        "allin": doctrine["allin"],
        "decoys": doctrine["decoys"],
        "kite": doctrine["kite"],
        "income_ladder": doctrine["income_ladder"],
        "hp_floor": doctrine["hp_floor"],
        "strike": doctrine["strike"],
        "medics": doctrine["medics"],
        "bunkers": doctrine["bunkers"],
        "flame": doctrine["flame"],
        "close": doctrine["close"],
        "guns": doctrine["guns"],
        "nukes": doctrine["nukes"],
    }


__all__ = ["decode_doctrine", "encode_doctrine"]
