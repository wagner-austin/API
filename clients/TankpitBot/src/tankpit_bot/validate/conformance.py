"""Capture-replay conformance: does the sim answer each tick as the server did?

Every archived capture is replayed through the sim one server tick at a
time. Before each tick the sim is anchored to the archive's latest
statements (:mod:`tankpit_bot.validate.conformance_mirror`), given the
exact commands the real client sent into that tick, and advanced once;
the self-caused wire it emits is reduced to the response-shape alphabet
(:func:`~tankpit_bot.analysis.response_shapes.shape_token`) and compared
with the same reduction of the batch the real server sent. A tick
matches when the two token sequences are equal.

What this measures is the sim's LAWS, tick by tick, from the real tick's
starting state, which is the property a multiplayer server built on the
sim must have before a real client can be pointed at it. It is not a
byte comparison: positions, ids and clocks legitimately differ between
the sim's world and the recorded one, and the token alphabet is exactly
the part of the wire that does not.

A tick is compared only when the client sent it a command other than a
keepalive. A tick holding a command the sim has no law for (an ``OTHER``
kind) is counted ``unmodelled`` and not compared.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

from tankpit_bot import _test_hooks
from tankpit_bot.analysis.response_shapes import shape_token
from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.resources import field_gif_path
from tankpit_bot.sim.commands import ClientCommandKind
from tankpit_bot.sim.ghost import compile_ghost_spec
from tankpit_bot.sim.run_boot import seed_ghost_world
from tankpit_bot.sim.scenarios import SIM_CLIENT_ID
from tankpit_bot.sim.server import SUPPORTED_KINDS, SimServer
from tankpit_bot.sim.world import make_sim_world
from tankpit_bot.validate.conformance_mirror import ArchiveMirror
from tankpit_bot.validate.conformance_types import (
    ConformanceReportDict,
    DivergenceDict,
    GroupTallyDict,
    ReplaySkipReason,
    SessionResultDict,
    SkippedSessionDict,
)
from tankpit_bot.validate.conformance_wire import ReplayCapture, read_replay

CAPTURE_SUFFIX = ".capture_session.json"
"""What every archived capture file is named."""

LATEST_CAPTURE_NAME = "latest" + CAPTURE_SUFFIX
"""A run directory's copy of its newest capture, which is never a new one."""


class TickDivergence(NamedTuple):
    """One compared tick whose sim wire differed from the archive's.

    Attributes:
        commands: The tick's command group (``+``-joined kinds).
        live: The real server's self-caused tokens.
        sim: The sim's self-caused tokens.
        timestamp_ms: The archived batch's capture time.
    """

    commands: str
    live: tuple[str, ...]
    sim: tuple[str, ...]
    timestamp_ms: int


class CaptureReplayResult(NamedTuple):
    """One capture's replay: its totals, and every tick compared.

    Attributes:
        totals: The per-capture counts.
        groups: Every compared tick's command group, with whether it
            matched, in tick order.
        divergences: Every tick that did not match, in tick order.
    """

    totals: SessionResultDict
    groups: tuple[tuple[str, bool], ...]
    divergences: tuple[TickDivergence, ...]


def _tokens(batch: Sequence[BinaryMessage], self_id: int) -> tuple[str, ...]:
    """Reduce a batch to its self-caused shape tokens.

    Args:
        batch: One tick's messages, in order.
        self_id: The tank whose self-caused tokens they are.

    Returns:
        The tokens, in batch order.
    """
    return tuple(token for token in (shape_token(m, self_id) for m in batch) if token is not None)


def places_self(capture: ReplayCapture) -> bool:
    """Whether the capture introduces its own tank and then places it.

    The ghost compiler needs both (it raises otherwise), and so does the
    replay: without a placed client there is no tick to anchor.

    Args:
        capture: The capture as ticks.

    Returns:
        True when an 0x21 introduction precedes a 0x3D or 0x47 naming it.
    """
    mirror = ArchiveMirror()
    for tick in capture.ticks:
        for message in tick.received:
            mirror.observe(message)
            if (
                message["msg_type"] in (0x3D, 0x47)
                and mirror.self_id is not None
                and message["tank_id"] == mirror.self_id
            ):
                return True
    return False


def replay_capture(
    capture: ReplayCapture,
    capture_text: str,
    terrain: _test_hooks.TerrainMapProtocol,
    *,
    session: str,
    field: str,
) -> CaptureReplayResult:
    """Replay one capture tick by tick and compare every commanded tick.

    Args:
        capture: The capture as ticks (it must place its own tank; see
            :func:`places_self`).
        capture_text: The capture file's text, for the ghost compiler that
            seeds the sim world with the recording's tanks and containers.
        terrain: The field's terrain.
        session: The capture's label in the report.
        field: The field's terrain GIF name (``field01_r.gif``).

    Returns:
        The capture's totals, compared groups and divergences.

    Raises:
        RuntimeError: If the capture never placed its own tank.
        SimError: If the sim refuses a command it claims to support,
            which is a sim defect and is not counted as a mismatch.
    """
    world = make_sim_world(field)
    seed_ghost_world(world, terrain, compile_ghost_spec(capture_text), None)
    server = SimServer(world, terrain)
    server.connect(SIM_CLIENT_ID)
    server.handshake(SIM_CLIENT_ID)
    viewport = server.require_session(SIM_CLIENT_ID).viewport
    client = world["tanks"][SIM_CLIENT_ID]
    mirror = ArchiveMirror()
    totals = SessionResultDict(session=session, field=field, ticks=0, matched=0, unmodelled=0)
    groups: list[tuple[str, bool]] = []
    divergences: list[TickDivergence] = []
    for tick in capture.ticks:
        commanded = [c for c in tick.commands if c["kind"] is not ClientCommandKind.KEEPALIVE]
        self_id = mirror.self_id
        if commanded and self_id is not None:
            group = "+".join(c["kind"].value for c in commanded)
            if any(c["kind"] not in SUPPORTED_KINDS for c in commanded):
                totals["unmodelled"] += 1
            else:
                mirror.anchor(world, client)
                window = mirror.window
                if window is not None:
                    viewport.window = window
                for command in tick.commands:
                    server.queue_command(SIM_CLIENT_ID, command)
                sim = _tokens(server.advance_tick()[SIM_CLIENT_ID], SIM_CLIENT_ID)
                live = _tokens(tick.received, self_id)
                matched = sim == live
                totals["ticks"] += 1
                totals["matched"] += int(matched)
                groups.append((group, matched))
                if not matched:
                    divergences.append(TickDivergence(group, live, sim, tick.timestamp_ms))
        for message in tick.received:
            mirror.observe(message)
    return CaptureReplayResult(totals, tuple(groups), tuple(divergences))


def discover_captures(roots: list[Path]) -> list[Path]:
    """Every archived capture under the given directories, in path order.

    A run directory's ``latest`` copy duplicates a capture already in the
    list, so it is left out.

    Args:
        roots: Directories to search, recursively.

    Returns:
        The capture files.
    """
    found: set[Path] = set()
    for root in roots:
        for path in _test_hooks.glob_paths(root, "**/*" + CAPTURE_SUFFIX):
            if path.name != LATEST_CAPTURE_NAME:
                found.add(path)
    return sorted(found)


class _ShapeKey(NamedTuple):
    """A divergence's identity: the group and the two token sequences."""

    commands: str
    live: tuple[str, ...]
    sim: tuple[str, ...]


def _group_order(item: tuple[str, int]) -> tuple[int, str]:
    """Sort key: most-compared group first, then by name.

    Args:
        item: A (group, tick count) pair.

    Returns:
        The key.
    """
    return (-item[1], item[0])


def _divergence_order(item: tuple[_ShapeKey, int]) -> tuple[int, _ShapeKey]:
    """Sort key: most frequent divergence first, then by its key.

    Args:
        item: A (divergence key, count) pair.

    Returns:
        The key.
    """
    return (-item[1], item[0])


class _Archive:
    """The run's accumulating results, and its terrain cache."""

    def __init__(self) -> None:
        self.sessions: list[SessionResultDict] = []
        self.skipped: list[SkippedSessionDict] = []
        self.groups: Counter[str] = Counter()
        self.group_matches: Counter[str] = Counter()
        self.divergences: Counter[_ShapeKey] = Counter()
        self.examples: dict[_ShapeKey, tuple[str, int]] = {}
        self.terrains: dict[Path, _test_hooks.TerrainMapProtocol] = {}

    def skip(self, path: Path, reason: ReplaySkipReason, detail: str) -> None:
        self.skipped.append(SkippedSessionDict(session=str(path), reason=reason, detail=detail))

    def terrain(self, gif: Path) -> _test_hooks.TerrainMapProtocol:
        known = self.terrains.get(gif)
        if known is None:
            known = _test_hooks.load_terrain_map(gif)
            self.terrains[gif] = known
        return known

    def add(self, result: CaptureReplayResult) -> None:
        self.sessions.append(result.totals)
        for group, matched in result.groups:
            self.groups[group] += 1
            self.group_matches[group] += int(matched)
        for d in result.divergences:
            key = _ShapeKey(d.commands, d.live, d.sim)
            self.divergences[key] += 1
            self.examples.setdefault(key, (result.totals["session"], d.timestamp_ms))

    def report(self) -> ConformanceReportDict:
        groups = [
            GroupTallyDict(commands=g, ticks=n, matched=self.group_matches[g])
            for g, n in sorted(self.groups.items(), key=_group_order)
        ]
        divergences = [
            DivergenceDict(
                commands=key.commands,
                live=list(key.live),
                sim=list(key.sim),
                count=n,
                example_session=self.examples[key][0],
                example_timestamp_ms=self.examples[key][1],
            )
            for key, n in sorted(self.divergences.items(), key=_divergence_order)
        ]
        return ConformanceReportDict(
            sessions=self.sessions, skipped=self.skipped, groups=groups, divergences=divergences
        )


def _replay_path(archive: _Archive, path: Path) -> None:
    """Replay one capture file into the run, or record why it cannot be.

    Args:
        archive: The run (mutated).
        path: The capture file.
    """
    scanned = scan_session(path)
    if scanned["kind"] == "skipped":
        archive.skip(path, ReplaySkipReason(scanned["reason"].value), "the archive scan's skip")
        return
    capture = read_replay(scanned["frames"])
    if capture.field_image is None:
        archive.skip(path, ReplaySkipReason.NO_ROOM, "no confirmed join of a listed room")
        return
    gif = field_gif_path(capture.field_image)
    if gif is None:
        archive.skip(path, ReplaySkipReason.FIELD_MISSING, capture.field_image)
        return
    if not places_self(capture):
        archive.skip(path, ReplaySkipReason.NO_SELF, "no 0x21 followed by its placement")
        return
    archive.add(
        replay_capture(
            capture,
            _test_hooks.read_text(path),
            archive.terrain(gif),
            session=str(path),
            field=gif.name,
        )
    )


def run_conformance(paths: list[Path]) -> ConformanceReportDict:
    """Replay every capture and gather the run's report.

    Args:
        paths: The capture files, in the order to replay them.

    Returns:
        The run.

    Raises:
        DecodeError: If a capture holds a sent command that will not
            decode (:func:`~tankpit_bot.validate.conformance_wire.read_replay`).
    """
    archive = _Archive()
    for path in paths:
        _replay_path(archive, path)
    return archive.report()


__all__ = [
    "CAPTURE_SUFFIX",
    "LATEST_CAPTURE_NAME",
    "CaptureReplayResult",
    "TickDivergence",
    "discover_captures",
    "places_self",
    "replay_capture",
    "run_conformance",
]
