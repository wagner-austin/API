"""A gameplay style as data, so trying one is an argument rather than an edit.

Every knob here already existed; what was missing was a single carrier. The
goals, the worker ceiling, the wave mass and the expansion switch were spread
across nine positional CLI slots, and the ninth slot only exists because the
eighth did -- each new question threaded one more position through the entry
point, the Makefile and the sweep harness ([[policy-loop]]).

A doctrine is one file naming all of them. Two arms of an experiment are two
files that differ in one line, which is the same discipline the sweep already
enforces for jobs: the arm that ran last week can be re-run, because the file
that defined it was never edited into the next one.

Every field is required. A doctrine file with a missing field is an error
naming the field, not a default quietly changing what the arm means -- the same
rule the sweep's job lines follow, for the same reason.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final, TypedDict

from rw_bot import RwBotError
from rw_bot.policy.combat import WAVE_SIZES
from rw_bot.policy.workforce import DEFAULT_MAX_WORKERS
from rw_bot.validation import (
    require_bool,
    require_int,
    require_non_empty_str,
    require_positive_int,
)

_FIELD_SHAPE = "RW-DOCTRINE-001"
_UNKNOWN_FIELD = "RW-DOCTRINE-002"
_NOT_A_NUMBER = "RW-DOCTRINE-003"
_NOT_A_FLAG = "RW-DOCTRINE-004"
_REPEATED_FIELD = "RW-DOCTRINE-005"
_BLANK_GOAL = "RW-DOCTRINE-006"
_BAD_RESERVE = "RW-DOCTRINE-007"
_BAD_GUARD_CAP = "RW-DOCTRINE-008"
_BAD_RAID_SIZE = "RW-DOCTRINE-009"
_BLANK_HEAVY = "RW-DOCTRINE-010"
_BAD_TECH_CAP = "RW-DOCTRINE-012"

#: The ``heavies`` value that means "no extra composition entries".
#:
#: A word rather than a blank, because a doctrine line cannot carry an empty
#: value and a missing field is an error by design.
NO_HEAVIES: Final = "none"

#: The ``reserve`` value that means "derive it from the composition".
#:
#: Negative rather than zero, because zero is a real reserve a doctrine may
#: want, and conflating "reserve nothing" with "decide for me" is how an arm
#: stops testing what it claims ([[policy-economy]]).
DERIVE_RESERVE: Final = -1

#: Fields carried as whole numbers in a doctrine file.
_INT_FIELDS: Final = ("max_workers", "mass", "reserve", "guard_cap", "raid", "tech")

#: Fields carried as ``0`` or ``1`` in a doctrine file.
_FLAG_FIELDS: Final = (
    "expand",
    "counter",
    "cover",
    "intercept",
    "aa_cover",
    "forward",
    "scout",
    "rush",
    "creep",
    "riposte",
)

#: Fields carried as text in a doctrine file.
_STR_FIELDS: Final = ("name", "goals", "heavies")

#: Every field a doctrine file must carry, in the order presets write them.
DOCTRINE_FIELDS: Final = (*_STR_FIELDS, *_INT_FIELDS, *_FLAG_FIELDS)


class DoctrineError(RwBotError):
    """A doctrine file could not be read as a gameplay style.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending line.
    """


class Doctrine(TypedDict):
    """One gameplay style, complete.

    Attributes:
        name: What this style is called, for the run log and the result file.
        goals: What to ask the planner for, in order. Repeats are a ratio, not
            a preference stated twice ([[policy-production]]).
        max_workers: The most builders worth holding ([[policy-production]]).
        mass: Units the sustained wave waits for. Values at or below the
            shipped ladder's last fixed rung leave the ladder unchanged, so the
            shipped behaviour is a value rather than a special case
            ([[engine-ai-triggers]]).
        reserve: Credits held back from expansion for the army, or
            :data:`DERIVE_RESERVE` to derive it from the composition. A fixed
            figure is what keeps a composition A/B from silently also being a
            reserve A/B ([[policy-economy]]).
        expand: Whether to play the economy at all. False is the control arm
            of the expansion A/B ([[policy-economy]]).
        counter: Whether production tilts toward what the opponent is seen to
            field, or holds the stated mix regardless
            ([[mechanics-combat-profile]]).
        cover: Whether the economy buys turrets beside bare structures at
            all. On is the behaviour every measurement carries somewhere in
            its lineage -- but for most of that lineage the orders were
            silently refused, so "defence on" historically meant "defence
            attempted". The first batch where turrets actually landed spent
            25-45k a match on them and won 6/24 at a rung the
            attempted-defence bot won 10/12, which is what makes on-vs-off
            a question at last ([[policy-holding-ground]]).
        intercept: Whether the reserve turns on a raider standing inside the
            outpost radius of one of our structures, or keeps gathering
            regardless ([[policy-holding-ground]]).
        aa_cover: Whether an anti-air turret joins the cover once the
            opponent has shown aircraft. Nothing the bot could place before
            this touched an aircraft at all -- the whole army and the ground
            turret declare ``canAttackFlyingUnits: false``
            ([[policy-holding-ground]]).
        guard_cap: The most reserve units an interception commits, or zero for
            all of them -- the behaviour every guard measurement was taken
            under, so the shipped figure is a value rather than a special
            case. The cost case that makes it a question: one match logged
            870 intercepts and never massed an attack
            ([[policy-holding-ground]]).
        forward: Whether the reserve posts at the frontier extractor
            instead of the base. The one invariant six batches have not
            moved is that matches are decided by extractor drops far from
            where the army gathers; this is the corpus's forward-posture
            answer to it ([[policy-holding-ground]],
            [[community-play-strategies]]).
        scout: Whether one scout is kept alive walking the pool circuit, its
            sightings remembered through the fog and fed to the counter tilt
            ([[community-play-strategies]]).
        rush: Whether released waves march at the estimated enemy start
            while nothing is visible to fight. The all-in verb: against an
            income-multiplier opponent whose advantage compounds with time,
            the earliest possible fight is the fairest one, and without this
            the first wave stood at the rally point waiting for an opponent
            who never needed to come ([[policy-holding-ground]]).
        raid: The raid party's size, or zero for no raiding. A size rather
            than a flag, because the size is the open question the v2
            measure left: at the first-wave size the raid is free and wins
            nothing, so whether a heavier party converts is a doctrine arm,
            not a code edit ([[policy-raid]]).
        creep: Whether the economy walks turrets toward the enemy start, one
            covered step at a time. The documented human answer to the
            cheating difficulties: turrets outrange and outlast anything the
            AI fields inside its thousand-tick opening delay, and ground
            taken this way is ground its random-target attack groups cannot
            answer ([[ai-opponent-strategy]], [[community-play-strategies]]).
        riposte: Whether the whole reserve releases the moment an intrusion
            ends -- the human counter-punch: let the attack burn itself on
            the defences, then push into the window before the opponent's
            next group finishes its thousand-tick delay and seventeen-second
            staging ([[ai-opponent-strategy]]).
        tech: How many factories unlock their next tier, or zero for none.
            The land factory's 2,000-credit upgrade flips a flag on the same
            building and opens the heavy roster -- reachable only through
            the ability verb, because it converts into no type
            ([[mechanics-build-actions]]). A count rather than a flag,
            because the unlock is per building and the first one already
            opens production: the flag form bought all four factories'
            unlocks in one probe -- 8,000 credits of saving pauses for a
            roster the first 2,000 had opened ([[policy-budget]]).
        heavies: Composition entries outside the plan, repeats a ratio like
            the goals. The channel the unlocked roster joins the army mix
            through: production orders only what the engine offers, so an
            entry here is inert until its factory's tier opens -- and it
            must NOT be a goal, because the plan derives prerequisites from
            the static build tree, which would insert the experimental
            factory rather than wait for the unlock
            ([[mechanics-build-actions]], [[policy-production]]).
    """

    name: str
    goals: tuple[str, ...]
    heavies: tuple[str, ...]
    max_workers: int
    mass: int
    reserve: int
    expand: bool
    counter: bool
    cover: bool
    intercept: bool
    guard_cap: int
    aa_cover: bool
    forward: bool
    scout: bool
    raid: int
    rush: bool
    creep: bool
    riposte: bool
    tech: int


#: The style everything so far was measured under, exactly.
#:
#: Extractors first because they pay for everything after them; no factory
#: named because the build tree inserts prerequisites; the shipped AI's wave
#: mass; expansion on; the mix held as stated. A doctrine file is only ever
#: compared against this, so it is a constant rather than a file that could
#: drift.
DEFAULT_DOCTRINE: Final[Doctrine] = Doctrine(
    name="default",
    goals=(
        "extractorT1",
        "extractorT1",
        "extractorT1",
        "c_tank",
        "c_tank",
        "c_tank",
        "c_tank",
    ),
    heavies=(),
    max_workers=DEFAULT_MAX_WORKERS,
    mass=WAVE_SIZES[-1],
    reserve=DERIVE_RESERVE,
    expand=True,
    counter=False,
    cover=True,
    intercept=False,
    guard_cap=0,
    aa_cover=False,
    forward=False,
    scout=False,
    raid=0,
    rush=False,
    creep=False,
    riposte=False,
    tech=0,
)


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
    guard_cap = require_int(payload, "guard_cap")
    if guard_cap < 0:
        raise DoctrineError(
            _BAD_GUARD_CAP,
            f"field 'guard_cap' must be >= 0, with 0 meaning the whole reserve, got {guard_cap}",
        )
    raid = require_int(payload, "raid")
    if raid < 0:
        raise DoctrineError(
            _BAD_RAID_SIZE,
            f"field 'raid' must be >= 0, a party size with 0 meaning no raiding, got {raid}",
        )
    tech = require_int(payload, "tech")
    if tech < 0:
        raise DoctrineError(
            _BAD_TECH_CAP,
            f"field 'tech' must be >= 0, factories to unlock with 0 meaning none, got {tech}",
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
        creep=require_bool(payload, "creep"),
        riposte=require_bool(payload, "riposte"),
        tech=tech,
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
        "tech": doctrine["tech"],
    }


def parse_doctrine_lines(lines: Sequence[str]) -> Doctrine:
    """Read a doctrine file: one ``field value`` pair per line.

    Blank lines and ``#`` comments are skipped, so a preset can record why its
    values are what they are beside the values themselves. Fields may appear in
    any order; each exactly once.

    Args:
        lines: The file's lines, without newlines.

    Returns:
        The doctrine it describes.

    Raises:
        DoctrineError: When a line is malformed, names an unknown field,
            repeats one, or carries a value of the wrong shape.
        DecodeError: When a field is absent or out of range.
    """
    payload: dict[str, str | int | float | bool] = {}
    for line in lines:
        bare = line.strip()
        if not bare or bare.startswith("#"):
            continue
        field, _, raw = bare.partition(" ")
        raw = raw.strip()
        if not raw:
            raise DoctrineError(_FIELD_SHAPE, f"a doctrine line is 'field value', got {line!r}")
        if field in payload:
            raise DoctrineError(_REPEATED_FIELD, f"field {field!r} appears twice")
        if field in _STR_FIELDS:
            payload[field] = raw
        elif field in _INT_FIELDS:
            try:
                payload[field] = int(raw)
            except ValueError as error:
                raise DoctrineError(
                    _NOT_A_NUMBER, f"field {field!r} must be a whole number, got {raw!r}"
                ) from error
        elif field in _FLAG_FIELDS:
            if raw not in ("0", "1"):
                raise DoctrineError(_NOT_A_FLAG, f"field {field!r} must be 0 or 1, got {raw!r}")
            payload[field] = raw == "1"
        else:
            raise DoctrineError(
                _UNKNOWN_FIELD,
                f"field {field!r} is not one of {', '.join(DOCTRINE_FIELDS)}",
            )
    return decode_doctrine(payload)


def format_doctrine(doctrine: Doctrine) -> tuple[str, ...]:
    """Render a doctrine as the lines :func:`parse_doctrine_lines` reads.

    What a probe or a test writes when it needs a preset on disk, so the two
    formats cannot drift.

    Args:
        doctrine: The doctrine to render.

    Returns:
        One line per field, in :data:`DOCTRINE_FIELDS` order.
    """
    flat = encode_doctrine(doctrine)
    rendered: list[str] = []
    for field in DOCTRINE_FIELDS:
        value = flat[field]
        if isinstance(value, bool):
            rendered.append(f"{field} {int(value)}")
        else:
            rendered.append(f"{field} {value}")
    return tuple(rendered)


__all__ = [
    "DEFAULT_DOCTRINE",
    "DERIVE_RESERVE",
    "DOCTRINE_FIELDS",
    "NO_HEAVIES",
    "Doctrine",
    "DoctrineError",
    "decode_doctrine",
    "encode_doctrine",
    "format_doctrine",
    "parse_doctrine_lines",
]
