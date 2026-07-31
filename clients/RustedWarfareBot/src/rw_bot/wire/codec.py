"""Folding the agent's NDJSON records back into whole samples, and back again.

The stream is a sequence of records discriminated by ``kind``. A ``frame``
record opens a sample and declares how many ``entity``, ``pool``, ``option`` and
``player`` records follow; those carry the visible roster, the visible resource
pools, what each unit can make, and the scoreboard. Folding them into samples is
this module's whole job. What a sample *contains* is
:mod:`rw_bot.wire.state`, which is vocabulary rather than behaviour and is
imported by nearly every policy module -- the reason the two are apart.

**The declared counts are checked rather than trusted.** A sample that promises
three entities and delivers two is a truncated capture -- the ordinary result of
reading a stream while the agent is still writing it -- and silently yielding
the short sample would let a planner decide on a roster it cannot see all of.

Encoding is here beside decoding deliberately, and the architecture suite
enforces it: a decoder without an encoder is a format that cannot be round
tripped, so a fixture has to be hand written and drifts from what the engine
actually sends.

Nothing here reads a file. Both directions are pure functions of what they are
given, which is what lets the same code serve a live tail and an archived
replay corpus without a branch between them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rw_bot import RwBotError
from rw_bot.validation import (
    require_bool,
    require_finite_float,
    require_int,
    require_non_empty_str,
    require_str,
)
from rw_bot.wire.ndjson import parse_object
from rw_bot.wire.state import (
    CHILD_COUNT_FIELDS,
    KIND_ENTITY,
    KIND_FRAME,
    KIND_OPTION,
    KIND_PLAYER,
    KIND_POOL,
    BuildOption,
    Entity,
    PlayerStat,
    ResourcePool,
    Sample,
)

_UNKNOWN_KIND = "RW-WIRE-001"
_RECORD_BEFORE_FRAME = "RW-WIRE-002"
_COUNT_MISMATCH = "RW-WIRE-003"
_FRAME_MISMATCH = "RW-WIRE-004"
_POOL_COUNT_MISMATCH = "RW-WIRE-005"
_OPTION_COUNT_MISMATCH = "RW-WIRE-006"
_PLAYER_COUNT_MISMATCH = "RW-WIRE-007"


class WireError(RwBotError):
    """The record sequence did not form whole samples.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of the offending record.
    """


def _decode_entity(record: Mapping[str, str | int | float | bool]) -> Entity:
    """Decode one entity record.

    Args:
        record: The record's fields.

    Returns:
        The entity.

    Raises:
        DecodeError: When a field is absent or carries a wrong type.
    """
    return Entity(
        index=require_int(record, "index"),
        unit_id=require_int(record, "id"),
        type_name=require_non_empty_str(record, "type"),
        class_name=require_non_empty_str(record, "class"),
        x=require_finite_float(record, "x"),
        y=require_finite_float(record, "y"),
        team=require_int(record, "team"),
        mine=require_bool(record, "mine"),
        hostile=require_bool(record, "hostile"),
        movement=require_non_empty_str(record, "movement"),
        group=require_int(record, "group"),
        flying=require_bool(record, "flying"),
        submerged=require_bool(record, "submerged"),
        touching_water=require_bool(record, "touching_water"),
        hp=require_finite_float(record, "hp"),
        max_hp=require_finite_float(record, "max_hp"),
        complete=require_bool(record, "complete"),
        queued=require_int(record, "queued"),
    )


def _decode_pool(record: Mapping[str, str | int | float | bool]) -> ResourcePool:
    """Decode one resource-pool record.

    Args:
        record: The record's fields.

    Returns:
        The pool.

    Raises:
        DecodeError: When a field is absent or carries a wrong type.
    """
    return ResourcePool(
        index=require_int(record, "index"),
        tile_x=require_int(record, "tile_x"),
        tile_y=require_int(record, "tile_y"),
        x=require_finite_float(record, "x"),
        y=require_finite_float(record, "y"),
        group_land=require_int(record, "group_land"),
    )


def _decode_option(record: Mapping[str, str | int | float | bool]) -> BuildOption:
    """Decode one build-option record.

    Args:
        record: The record's fields.

    Returns:
        The option.

    Raises:
        DecodeError: When a field is absent or carries a wrong type.
    """
    return BuildOption(
        index=require_int(record, "index"),
        unit_id=require_int(record, "unit_id"),
        # Not required to be non-empty. An action concerning no type at all is
        # published now rather than dropped, because filtering those in the
        # agent is what hid upgrades ([[policy-holding-ground]]).
        produces=require_str(record, "produces"),
        key=require_str(record, "key"),
        placed=require_bool(record, "placed"),
        available=require_bool(record, "available"),
        makes_something=require_bool(record, "makes_something"),
        price=require_int(record, "price"),
    )


def _decode_player(record: Mapping[str, str | int | float | bool]) -> PlayerStat:
    """Decode one player-scoreboard record.

    Args:
        record: The record's fields.

    Returns:
        The scoreboard entry.

    Raises:
        DecodeError: When a field is absent or carries a wrong type.
    """
    return PlayerStat(
        index=require_int(record, "index"),
        team=require_int(record, "team"),
        local=require_bool(record, "local"),
        hostile=require_bool(record, "hostile"),
        defeated=require_bool(record, "defeated"),
        wiped=require_bool(record, "wiped"),
        income=require_int(record, "income"),
        army_value=require_int(record, "army_value"),
        building_value=require_int(record, "building_value"),
    )


class _OpenSample:
    """A sample being assembled, from its frame record to its last child.

    This is a class rather than a dozen locals threaded through helpers, and the
    reason is drift. Every record kind the stream gains adds a declared count, a
    list, a count check and an argument to whatever closes the sample, and when
    that was spelled out positionally the closing helper reached fourteen
    parameters — an arrangement where adding the next kind means editing four
    call sites and where two of the arguments were one transposition away from
    being silently swapped. Here a new kind is a list, a declared count and one
    line in :meth:`close`.

    Attributes:
        frame: The engine frame this sample reports.
        clock_ms: The engine clock at the same moment.
        credits: Credits the local player holds.
        defeated: Whether the local player has been defeated.
        wiped: Whether the local player has been wiped out.
        players_left: How many players are still in the match.
        entities: Entity records seen so far.
        pools: Resource-pool records seen so far.
        options: Build-option records seen so far.
        players: Player scoreboard records seen so far.
    """

    def __init__(self, record: Mapping[str, str | int | float | bool]) -> None:
        """Open a sample from its frame record.

        Args:
            record: The frame record's fields.

        Raises:
            DecodeError: When the frame record is missing a field or carries a
                wrong type.
        """
        self.frame = require_int(record, "frame")
        self.clock_ms = require_int(record, "clock_ms")
        self.credits = require_int(record, "credits")
        self.defeated = require_bool(record, "defeated")
        self.wiped = require_bool(record, "wiped")
        self.players_left = require_int(record, "players_left")
        self._declared_entities = require_int(record, "visible")
        self._declared_pools = require_int(record, "pools")
        self._declared_options = require_int(record, "options")
        self._declared_players = require_int(record, "players")
        self.entities: list[Entity] = []
        self.pools: list[ResourcePool] = []
        self.options: list[BuildOption] = []
        self.players: list[PlayerStat] = []

    def add(self, kind: str, record: Mapping[str, str | int | float | bool]) -> None:
        """Decode one child record into this sample.

        Every kind the stream carries is decoded here rather than in the reading
        loop. That keeps the loop to "open a sample or add to one", and it means
        a new record kind is a single branch in a single method instead of a
        branch in the loop, a list, a declared count and a count check spread
        across four places.

        Args:
            kind: The record's ``kind``.
            record: The record's fields.

        Raises:
            DecodeError: When the record is missing a field or carries a wrong
                type.
            WireError: ``RW-WIRE-004`` when the record's own frame disagrees
                with this sample's, ``RW-WIRE-001`` on a kind the stream does
                not define.
        """
        reported = require_int(record, "frame")
        if reported != self.frame:
            raise WireError(
                _FRAME_MISMATCH,
                f"{kind} reports frame {reported} inside the sample for frame {self.frame}; "
                "the records have been interleaved",
            )
        if kind == KIND_ENTITY:
            self.entities.append(_decode_entity(record))
        elif kind == KIND_POOL:
            self.pools.append(_decode_pool(record))
        elif kind == KIND_OPTION:
            self.options.append(_decode_option(record))
        elif kind == KIND_PLAYER:
            self.players.append(_decode_player(record))
        else:
            raise WireError(_UNKNOWN_KIND, f"unknown record kind {kind!r}")

    def close(self) -> Sample:
        """Finish the sample, checking it against its own declared counts.

        Returns:
            The completed sample.

        Raises:
            WireError: ``RW-WIRE-003`` when the entity counts disagree,
                ``RW-WIRE-005`` when the pool counts do, ``RW-WIRE-006`` when
                the option counts do, ``RW-WIRE-007`` when the player counts do.
        """
        self._require_count(
            len(self.entities), self._declared_entities, "visible entities", _COUNT_MISMATCH
        )
        self._require_count(
            len(self.pools), self._declared_pools, "visible resource pools", _POOL_COUNT_MISMATCH
        )
        self._require_count(
            len(self.options), self._declared_options, "build options", _OPTION_COUNT_MISMATCH
        )
        self._require_count(
            len(self.players), self._declared_players, "players", _PLAYER_COUNT_MISMATCH
        )
        return Sample(
            frame=self.frame,
            clock_ms=self.clock_ms,
            credits=self.credits,
            defeated=self.defeated,
            wiped=self.wiped,
            players_left=self.players_left,
            entities=tuple(self.entities),
            pools=tuple(self.pools),
            options=tuple(self.options),
            players=tuple(self.players),
        )

    def _require_count(self, carried: int, declared: int, what: str, code: str) -> None:
        """Check one carried count against what the frame record promised.

        The declared counts are checked rather than trusted. A sample that
        promises three entities and delivers two is a truncated capture — the
        ordinary result of reading a stream while the agent is still writing it
        — and silently yielding the short sample would let a planner decide on a
        roster it cannot see all of.

        Args:
            carried: How many records of this kind arrived.
            declared: How many the frame record promised.
            what: Human-readable name of the kind, for the message.
            code: Stable identifier for this kind's mismatch.

        Raises:
            WireError: The given ``code`` when the counts disagree.
        """
        if carried != declared:
            raise WireError(
                code,
                f"frame {self.frame} declared {declared} {what} but carried {carried}; "
                "the capture is truncated or interleaved",
            )


def declared_children(record: Mapping[str, str | int | float | bool]) -> int:
    """Return how many child records a frame record says will follow.

    Args:
        record: A frame record's fields.

    Returns:
        The total across every declared kind.

    Raises:
        DecodeError: When a count is absent or not an int.
    """
    return sum(require_int(record, field) for field in CHILD_COUNT_FIELDS)


def decode_samples(lines: Sequence[str]) -> tuple[Sample, ...]:
    """Fold a run of NDJSON lines into whole samples.

    Blank lines are skipped, which is what a file ending in a newline produces.

    Args:
        lines: NDJSON lines, without newline terminators.

    Returns:
        Every complete sample, in stream order.

    Raises:
        NdjsonError: When a line does not parse.
        DecodeError: When a record is missing a field or carries a wrong type.
        WireError: ``RW-WIRE-001`` on an unknown ``kind``, ``RW-WIRE-002`` when
            a child record precedes any frame, ``RW-WIRE-003`` when a sample's
            entity count disagrees with its declared count, ``RW-WIRE-004`` when
            a record's frame disagrees with the sample it falls in,
            ``RW-WIRE-005`` when the pool count disagrees, ``RW-WIRE-006`` when
            the option count does, ``RW-WIRE-007`` when the player count does.
    """
    samples: list[Sample] = []
    open_sample: _OpenSample | None = None

    for line in lines:
        if line.strip() == "":
            continue
        record = parse_object(line)
        kind = require_non_empty_str(record, "kind")

        if kind == KIND_FRAME:
            if open_sample is not None:
                samples.append(open_sample.close())
            open_sample = _OpenSample(record)
            continue

        if open_sample is None:
            raise WireError(
                _RECORD_BEFORE_FRAME,
                f"a {kind} record appeared before any frame record; the stream does "
                "not begin at a sample boundary",
            )
        open_sample.add(kind, record)

    if open_sample is not None:
        samples.append(open_sample.close())
    return tuple(samples)


def encode_sample(sample: Sample) -> tuple[str, ...]:
    """Render a sample back to NDJSON lines.

    Round-trips with :func:`decode_samples`, which is what makes a decoded
    corpus re-emittable as a fixture.

    Args:
        sample: The sample to encode.

    Returns:
        One frame line, then one line per entity, then one line per pool — the
        order the agent writes them in, which is what makes the round trip
        byte-for-byte rather than merely equivalent.
    """
    frame = sample["frame"]
    lines = [
        f'{{"kind":"{KIND_FRAME}","frame":{frame},'
        f'"clock_ms":{sample["clock_ms"]},"visible":{len(sample["entities"])},'
        f'"pools":{len(sample["pools"])},"options":{len(sample["options"])},'
        f'"players":{len(sample["players"])},'
        f'"credits":{sample["credits"]},'
        f'"defeated":{str(sample["defeated"]).lower()},'
        f'"wiped":{str(sample["wiped"]).lower()},'
        f'"players_left":{sample["players_left"]}}}'
    ]
    for entity in sample["entities"]:
        name = _escape(entity["class_name"])
        type_name = _escape(entity["type_name"])
        lines.append(
            f'{{"kind":"{KIND_ENTITY}","frame":{frame},"index":{entity["index"]},'
            f'"id":{entity["unit_id"]},"type":"{type_name}",'
            f'"class":"{name}","x":{entity["x"]!r},"y":{entity["y"]!r},'
            f'"team":{entity["team"]},"mine":{str(entity["mine"]).lower()},'
            f'"hostile":{str(entity["hostile"]).lower()},'
            f'"movement":"{entity["movement"]}","group":{entity["group"]},'
            f'"flying":{str(entity["flying"]).lower()},'
            f'"submerged":{str(entity["submerged"]).lower()},'
            f'"touching_water":{str(entity["touching_water"]).lower()},'
            f'"hp":{entity["hp"]!r},"max_hp":{entity["max_hp"]!r},'
            f'"complete":{str(entity["complete"]).lower()},'
            f'"queued":{entity["queued"]}}}'
        )
    for pool in sample["pools"]:
        lines.append(
            f'{{"kind":"{KIND_POOL}","frame":{frame},"index":{pool["index"]},'
            f'"tile_x":{pool["tile_x"]},"tile_y":{pool["tile_y"]},'
            f'"x":{pool["x"]!r},"y":{pool["y"]!r},'
            f'"group_land":{pool["group_land"]}}}'
        )
    for option in sample["options"]:
        produces = _escape(option["produces"])
        key = _escape(option["key"])
        lines.append(
            f'{{"kind":"{KIND_OPTION}","frame":{frame},"index":{option["index"]},'
            f'"unit_id":{option["unit_id"]},"produces":"{produces}",'
            f'"key":"{key}",'
            f'"placed":{str(option["placed"]).lower()},'
            f'"available":{str(option["available"]).lower()},'
            f'"makes_something":{str(option["makes_something"]).lower()},'
            f'"price":{option["price"]}}}'
        )
    for player in sample["players"]:
        lines.append(
            f'{{"kind":"{KIND_PLAYER}","frame":{frame},"index":{player["index"]},'
            f'"team":{player["team"]},"local":{str(player["local"]).lower()},'
            f'"hostile":{str(player["hostile"]).lower()},'
            f'"defeated":{str(player["defeated"]).lower()},'
            f'"wiped":{str(player["wiped"]).lower()},'
            f'"income":{player["income"]},"army_value":{player["army_value"]},'
            f'"building_value":{player["building_value"]}}}'
        )
    return tuple(lines)


def _escape(text: str) -> str:
    """Escape a string for inclusion in a JSON string literal.

    Engine class names contain nothing that needs escaping, so this never fires
    in practice. It is here because :func:`encode_sample` claims to round-trip,
    and a claim that holds only for the values seen so far is not a round trip.

    Args:
        text: The raw string.

    Returns:
        The escaped string, without surrounding quotes.
    """
    out: list[str] = []
    for char in text:
        if char in ('"', "\\"):
            out.append("\\" + char)
        elif char in ("\n", "\r", "\t"):
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[char])
        elif ord(char) < 0x20:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)
    return "".join(out)


__all__ = [
    "CHILD_COUNT_FIELDS",
    "KIND_ENTITY",
    "KIND_FRAME",
    "KIND_OPTION",
    "KIND_PLAYER",
    "KIND_POOL",
    "BuildOption",
    "Entity",
    "PlayerStat",
    "ResourcePool",
    "Sample",
    "WireError",
    "declared_children",
    "decode_samples",
    "encode_sample",
]
