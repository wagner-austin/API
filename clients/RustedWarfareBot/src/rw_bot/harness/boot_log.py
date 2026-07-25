"""Structured decoding of the engine's boot log.

Every headless run writes an engine log via ``-log``. That log is the project's
primary evidence source: it pins the build, names each subsystem as it is
constructed, prints at least one obfuscated class mapping outright, records
which map loaded, and captures crashes with their stack traces. Reading it by
hand does not scale, and grep answers cannot be asserted on in a test.

Boundary note: :class:`BootLog` and its members are *derived* types, not wire
types. Their sole constructor is :func:`parse_boot_log`, which is itself the
validator — it rejects a log missing its version header or carrying a crash
marker with no stack trace. They therefore carry no ``encode_*``/``decode_*``
pair: their members are already JSON-shaped scalars and tuples that
``json.dumps`` serialises directly, and no untyped payload is ever turned back
into one, so an encoder would be a thin wrapper over identity.
:class:`~rw_bot.harness.launch.LaunchConfig` does cross an untyped boundary and
carries the full encode/decode/``require_*`` chain.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypedDict

from rw_bot import RwBotError

_TIMESTAMP_LEN: Final = 25
_LOADING_PREFIX: Final = "--Now loading:"
_MAPFILE_PREFIX: Final = "Mapfile: "
_CREATED_PREFIX: Final = "Created new "
_CREATED_INFIX: Final = " of:"
_CRASH_MARKER: Final = "uncaughtException start"
_FRAME_PREFIX: Final = "at "
_VERSION_PREFIX: Final = "Game Version: "
_CODE_PREFIX: Final = "Game Code: "
_BUILD_PREFIX: Final = "Build Number: "

_NO_VERSION: Final = "RW-BOOTLOG-001"
_CRASH_NO_TRACE: Final = "RW-BOOTLOG-002"
_BAD_GAME_CODE: Final = "RW-BOOTLOG-003"


class BootLogError(RwBotError):
    """An engine boot log could not be decoded into structured records.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what was missing or malformed.
    """


class EngineVersion(TypedDict):
    """The build identity the engine prints during early boot.

    Attributes:
        version: Marketing version, e.g. ``"1.15"``.
        game_code: Numeric build code, e.g. ``176``.
        build_number: Build label as printed, e.g. ``"#28"``.
    """

    version: str
    game_code: int
    build_number: str


class SubsystemLoad(TypedDict):
    """One ``--Now loading:<name>`` line.

    Attributes:
        line_number: 1-indexed line in the source log.
        name: Subsystem name exactly as printed, e.g. ``"CommandController"``.
    """

    line_number: int
    name: str


class ClassMapping(TypedDict):
    """A subsystem printed together with its obfuscated implementation class.

    These lines are the cheapest deobfuscation source available: the engine
    names the readable subsystem and its single-letter class in one line.

    Attributes:
        line_number: 1-indexed line in the source log.
        subsystem: Readable name, e.g. ``"gameEngine"``.
        java_class: Fully-qualified class, e.g. ``"com.corrodinggames.rts.game.i"``.
    """

    line_number: int
    subsystem: str
    java_class: str


class MapLoad(TypedDict):
    """One ``Mapfile: <path>`` line.

    Attributes:
        line_number: 1-indexed line in the source log.
        map_file: Asset path of the loaded map.
    """

    line_number: int
    map_file: str


class EngineCrash(TypedDict):
    """An uncaught exception together with the frame it was thrown from.

    Attributes:
        line_number: 1-indexed line of the ``uncaughtException start`` marker.
        exception_type: Fully-qualified throwable, e.g.
            ``"java.lang.NullPointerException"``.
        top_frame: Topmost stack frame with its ``at `` prefix removed.
    """

    line_number: int
    exception_type: str
    top_frame: str


class BootLog(TypedDict):
    """Everything structured that a single engine log carries.

    Attributes:
        version: Build identity from the log header.
        subsystems: Every ``--Now loading:`` line, in file order.
        class_mappings: Every readable-name-to-obfuscated-class line.
        maps: Every map load, in file order.
        crashes: Every uncaught exception, in file order.
    """

    version: EngineVersion
    subsystems: tuple[SubsystemLoad, ...]
    class_mappings: tuple[ClassMapping, ...]
    maps: tuple[MapLoad, ...]
    crashes: tuple[EngineCrash, ...]


def strip_timestamp(line: str) -> str:
    """Remove the engine's ``YYYY-MM-DD HH:MM:SS.mmm: `` prefix if present.

    Not every line carries one: raw stack frames and Slick's own INFO lines are
    written straight to the log without the engine's prefix.

    Args:
        line: A single raw log line, without its trailing newline.

    Returns:
        The line with its timestamp prefix removed, or unchanged when the line
        does not begin with one.
    """
    if len(line) < _TIMESTAMP_LEN:
        return line
    if (
        line[4] == "-"
        and line[7] == "-"
        and line[10] == " "
        and line[13] == ":"
        and line[16] == ":"
        and line[19] == "."
        and line[23:25] == ": "
    ):
        return line[_TIMESTAMP_LEN:]
    return line


def _parse_version(bodies: Sequence[str]) -> EngineVersion:
    """Extract the build identity from the log's header lines.

    Args:
        bodies: Every log line with timestamps already stripped.

    Returns:
        The build identity.

    Raises:
        BootLogError: ``RW-BOOTLOG-001`` when any of the three header lines is
            absent. ``RW-BOOTLOG-003`` when the game code is not an integer.
    """
    version = ""
    build_number = ""
    code_text = ""
    for body in bodies:
        if body.startswith(_VERSION_PREFIX):
            version = body[len(_VERSION_PREFIX) :].strip()
        elif body.startswith(_BUILD_PREFIX):
            build_number = body[len(_BUILD_PREFIX) :].strip()
        elif body.startswith(_CODE_PREFIX):
            code_text = body[len(_CODE_PREFIX) :].strip()
    missing = [
        name
        for name, value in (
            ("Game Version", version),
            ("Build Number", build_number),
            ("Game Code", code_text),
        )
        if value == ""
    ]
    if missing:
        raise BootLogError(
            _NO_VERSION,
            f"boot log is missing its version header: {', '.join(missing)} not found",
        )
    if not code_text.isdigit():
        raise BootLogError(
            _BAD_GAME_CODE,
            f"Game Code must be an integer, got {code_text!r}",
        )
    return EngineVersion(version=version, game_code=int(code_text), build_number=build_number)


def _parse_crash(bodies: Sequence[str], marker_index: int) -> EngineCrash:
    """Extract one crash record starting at a marker line.

    The engine prints the marker, a handler line, the throwable, a cause line,
    then the raw trace. The bare line immediately preceding the first ``at ``
    frame is the throwable that was actually thrown.

    Args:
        bodies: Every log line with timestamps already stripped.
        marker_index: Index of the ``uncaughtException start`` line.

    Returns:
        The crash record.

    Raises:
        BootLogError: ``RW-BOOTLOG-002`` when no stack frame follows the marker
            before the next marker or end of file.
    """
    scan = marker_index + 1
    while scan < len(bodies):
        body = bodies[scan]
        if body.startswith(_CRASH_MARKER):
            break
        if body.strip().startswith(_FRAME_PREFIX) and bodies[scan].startswith(("\t", " ")):
            frame = body.strip()[len(_FRAME_PREFIX) :]
            return EngineCrash(
                line_number=marker_index + 1,
                exception_type=bodies[scan - 1].strip(),
                top_frame=frame,
            )
        scan += 1
    raise BootLogError(
        _CRASH_NO_TRACE,
        f"crash marker at line {marker_index + 1} has no stack frame following it",
    )


def parse_boot_log(lines: Sequence[str]) -> BootLog:
    """Decode a full engine log into structured records.

    Args:
        lines: The log's lines, without trailing newlines, in file order.

    Returns:
        Every structured record the log carries.

    Raises:
        BootLogError: ``RW-BOOTLOG-001`` when the version header is incomplete,
            ``RW-BOOTLOG-002`` when a crash marker carries no stack trace,
            ``RW-BOOTLOG-003`` when the game code is not an integer.
    """
    bodies = [strip_timestamp(line) for line in lines]
    subsystems: list[SubsystemLoad] = []
    mappings: list[ClassMapping] = []
    maps: list[MapLoad] = []
    crashes: list[EngineCrash] = []

    for index, body in enumerate(bodies):
        if body.startswith(_LOADING_PREFIX):
            subsystems.append(
                SubsystemLoad(line_number=index + 1, name=body[len(_LOADING_PREFIX) :].strip())
            )
        elif body.startswith(_MAPFILE_PREFIX):
            maps.append(
                MapLoad(line_number=index + 1, map_file=body[len(_MAPFILE_PREFIX) :].strip())
            )
        elif body.startswith(_CREATED_PREFIX) and _CREATED_INFIX in body:
            subject, _, java_class = body[len(_CREATED_PREFIX) :].partition(_CREATED_INFIX)
            mappings.append(
                ClassMapping(
                    line_number=index + 1,
                    subsystem=subject.strip(),
                    java_class=java_class.strip(),
                )
            )
        elif body.startswith(_CRASH_MARKER):
            crashes.append(_parse_crash(bodies, index))

    return BootLog(
        version=_parse_version(bodies),
        subsystems=tuple(subsystems),
        class_mappings=tuple(mappings),
        maps=tuple(maps),
        crashes=tuple(crashes),
    )


def find_subsystem(log: BootLog, name: str) -> SubsystemLoad | None:
    """Return the load record for one subsystem, or ``None`` if it never loaded.

    ``None`` here is an answer, not a soft failure: "did CommandController
    construct during this boot?" is a question whose negative result is
    meaningful and is asserted on directly by harness tests.

    Args:
        log: The parsed log to search.
        name: Exact subsystem name, e.g. ``"CommandController"``.

    Returns:
        The first matching record, or ``None`` when the subsystem never loaded.
    """
    for item in log["subsystems"]:
        if item["name"] == name:
            return item
    return None


__all__ = [
    "BootLog",
    "BootLogError",
    "ClassMapping",
    "EngineCrash",
    "EngineVersion",
    "MapLoad",
    "SubsystemLoad",
    "find_subsystem",
    "parse_boot_log",
    "strip_timestamp",
]
