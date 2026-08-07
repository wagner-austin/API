"""Ghost replay: a recorded session's opponents, replayed in the sim.

The archive holds the complete OBSERVABLE behavior of every tank the
recorded client ever saw — identities (0x21 with names), positions
(0x3D absolute, 0x47 start+path), every shot (0x53 with its aim
tile), chats (0x4D, the consent signals), all on absolute timestamps.
This module compiles one capture into a tick-indexed ghost spec and
drives the sim's non-client tanks through it verbatim while the
PRODUCTION bot plays live against them ([[capture-differ]] stage 4):
the Yuppler fight, the arena gank — the actual recorded pressure.

Semantics and honest limits (v1):

* Ghosts follow the RECORDING, not the live bot — the reproduction is
  exact while the live bot tracks the recorded session, and the
  divergence point is itself the measurement (the ``ghost_summary``
  diagnostic reports how long the live run tracked the recording).
* Other tanks' positions are viewport-scoped on the wire, so ghosts
  hold still between recorded sightings and jump on the next one —
  the in-viewport FIGHT is what replays faithfully.
* Damage resolves by SIM law from the replayed shots: a ghost the
  recording saw die may survive (the live bot fought differently),
  and a ghost the live bot kills early simply stops consuming its
  remaining timeline (dead tanks' events are skipped).
* Ghost fuel is unobservable on the wire (0x2E carries fuel for SELF
  only) — ghosts seed at rank capacity.
* Container exposure comes from the recording's own first 0x4C dot
  atlas (plus visible-layer first reads): radar-revealed hidden fuel
  stays hidden, atlas dots without a volume read seed as drained
  dots — the live bot's first map open shows the map the RECORDED
  bot saw (the first cut marked every first-read dotted and the
  replay hopped at a phantom atlas on round 2).
"""

from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.json_utils import (
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_list,
    narrow_json_to_str,
)
from platform_core.logging import get_logger

from tankpit_bot._test_hooks.terrain import TerrainMapProtocol
from tankpit_bot.capture.frames import split_payload_frames
from tankpit_bot.capture.xor import build_session_xor_table, xor_decode_body
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.protocol.commands import TICK_RATE_MS
from tankpit_bot.protocol.framing import FramingError
from tankpit_bot.protocol.types import BinaryMessage
from tankpit_bot.runtime_logging import emit_diagnostic
from tankpit_bot.sim.world import SimContainerDict, SimEquipmentDict
from tankpit_bot.sniffer.decoders import _is_text_route
from tankpit_bot.state.viewport_geometry import viewport_patch_world_coords
from tankpit_bot.wire.helpers import DecodeError

log = get_logger(__name__)

_STEP_DELTAS: dict[str, tuple[int, int]] = {
    "n": (0, -1),
    "s": (0, 1),
    "e": (1, 0),
    "w": (-1, 0),
}
_TICK_MS = TICK_RATE_MS
_TRACK_DRIFT_TILES = 4
"""Chebyshev drift beyond which the live bot no longer 'tracks' the
recorded client — the ghost_summary's divergence threshold."""


class GhostTankDict(TypedDict):
    """One replayable opponent: identity plus its first sighted tile."""

    tank_id: int
    team: int
    rank: int
    name: str
    x: int
    y: int


class GhostEventDict(TypedDict):
    """One tick-indexed recorded action of a ghost.

    ``place`` relocates to (x, y); ``shoot`` fires at the recorded aim
    tile (x, y); ``chat`` sends ``message_id`` (x/y carry the sender's
    recorded tile).
    """

    tick: int
    tank_id: int
    kind: Literal["place", "shoot", "chat"]
    x: int
    y: int
    message_id: int


class GhostContainerDict(TypedDict):
    """A container's first-observed state in the recording."""

    x: int
    y: int
    volume: int
    dotted: bool


class GhostSpecDict(TypedDict):
    """Everything one capture contributes to a replayable world."""

    client_team: int
    client_rank: int
    client_x: int
    client_y: int
    client_fuel: int
    client_counts: list[int]
    ghosts: list[GhostTankDict]
    events: list[GhostEventDict]
    recorded_path: dict[int, tuple[int, int]]
    containers: list[GhostContainerDict]
    equipment: list[tuple[int, int]]
    dot_atlas: list[tuple[int, int]]
    ticks: int
    unplaced_tanks: int


class _Walk:
    """Mutable per-capture accumulation state for the compiler."""

    def __init__(self) -> None:
        self.self_id: int | None = None
        self.self_team = 0
        self.t0: int | None = None
        self.identities: dict[int, tuple[int, str]] = {}
        self.ranks: dict[int, int] = {}
        self.positions: dict[int, list[tuple[int, int, int]]] = {}
        self.shots: list[tuple[int, int, int, int]] = []
        self.chats: list[tuple[int, int, int, int, int]] = []
        self.first_reads: dict[tuple[int, int], tuple[int, bool]] = {}
        self.dot_atlas: set[tuple[int, int]] | None = None
        self.client_fuel: int | None = None
        self.client_counts: list[int] | None = None

    def tick(self, t: int) -> int:
        """Map an absolute timestamp onto the session tick index."""
        if self.t0 is None:
            self.t0 = t
        return max(0, (t - self.t0) // _TICK_MS)


def _note_position(walk: _Walk, tank_id: int, t: int, x: int, y: int) -> None:
    walk.positions.setdefault(tank_id, []).append((walk.tick(t), x, y))


def _note_first_read(walk: _Walk, x: int, y: int, value: int, visible: bool) -> None:
    walk.first_reads.setdefault((x, y), (value, visible))


def _consume_tank_message(walk: _Walk, t: int, decoded: BinaryMessage) -> bool:
    """Fold one tank-flavored message into the compiler state.

    Args:
        walk: Accumulation state (mutated).
        t: The message's absolute timestamp.
        decoded: The decoded message.

    Returns:
        True when the message was consumed here.
    """
    match decoded:
        case {"msg_type": 0x21, "tank_id": int(tank_id), "team": int(team), "name": str(name)}:
            if walk.self_id is None:
                walk.self_id = tank_id
            elif tank_id != walk.self_id:
                walk.identities.setdefault(tank_id, (team, name))
            return True
        case {
            "msg_type": 0x3D,
            "tank_id": int(tank_id),
            "team": int(team),
            "rank": int(rank),
            "x": int(x),
            "y": int(y),
        }:
            walk.ranks.setdefault(tank_id, rank)
            if tank_id == walk.self_id:
                walk.self_team = team
            _note_position(walk, tank_id, t, x, y)
            return True
        case {
            "msg_type": 0x47,
            "tank_id": int(tank_id),
            "rank": int(rank),
            "start_x": int(x),
            "start_y": int(y),
            "path": str(path),
        }:
            walk.ranks.setdefault(tank_id, rank)
            for step in path:
                dx, dy = _STEP_DELTAS[step]
                x, y = x + dx, y + dy
            _note_position(walk, tank_id, t, x, y)
            return True
    return False


def _consume_combat_social(walk: _Walk, t: int, decoded: BinaryMessage) -> bool:
    """Fold one shot or chat into the compiler state.

    Args:
        walk: Accumulation state (mutated).
        t: The message's absolute timestamp.
        decoded: The decoded message.

    Returns:
        True when the message was consumed here.
    """
    match decoded:
        case {
            "msg_type": 0x53,
            "shooter_id": int(shooter_id),
            "aim_x": int(aim_x),
            "aim_y": int(aim_y),
        }:
            if shooter_id != walk.self_id:
                walk.shots.append((walk.tick(t), shooter_id, aim_x, aim_y))
            return True
        case {
            "msg_type": 0x4D,
            "sender_id": int(sender_id),
            "message_type": int(message_id),
            "x": chat_x,
            "y": chat_y,
        }:
            if sender_id != walk.self_id:
                x = chat_x if isinstance(chat_x, int) else 0
                y = chat_y if isinstance(chat_y, int) else 0
                walk.chats.append((walk.tick(t), sender_id, message_id, x, y))
            return True
    return False


def _consume_world_reads(walk: _Walk, decoded: BinaryMessage) -> bool:
    """Fold one container-bearing message into the first-read map.

    Args:
        walk: Accumulation state (mutated).
        decoded: The decoded message.

    Returns:
        True when the message was consumed here.
    """
    if decoded["msg_type"] == 0x4F:
        for container in decoded["containers"]:
            _note_first_read(walk, container["x"], container["y"], container["volume"], False)
        return True
    if decoded["msg_type"] == 0x43:
        for x, y, value in decoded["updates"]:
            _note_first_read(walk, x, y, value, True)
        return True
    if decoded["msg_type"] == 0x5A:
        left, top = decoded["viewport_left"], decoded["viewport_top"]
        for entity in decoded["entities"]:
            x, y = viewport_patch_world_coords(left, top, entity["col"], entity["row"])
            _note_first_read(walk, x, y, entity["cache_value"], True)
        return True
    if decoded["msg_type"] == 0x4C:
        if walk.dot_atlas is None:
            walk.dot_atlas = {(x, y) for x, y in decoded["fuel_dots"]}
        return True
    return False


def _consume(walk: _Walk, t: int, decoded: BinaryMessage) -> None:
    """Fold one decoded received message into the compiler state.

    Args:
        walk: Accumulation state (mutated).
        t: The message's absolute timestamp.
        decoded: The decoded message.
    """
    if _consume_tank_message(walk, t, decoded):
        return
    if _consume_combat_social(walk, t, decoded):
        return
    if _consume_world_reads(walk, decoded):
        return
    if decoded["msg_type"] == 0x44:
        if walk.client_fuel is None:
            walk.client_fuel = decoded["fuel_total"]
    elif decoded["msg_type"] == 0x49 and walk.client_counts is None:
        walk.client_counts = list(decoded["counts"])


def _assemble_ghosts(walk: _Walk) -> tuple[list[GhostTankDict], list[GhostEventDict], int]:
    """Build the ghost roster and its tick-ordered event stream.

    Args:
        walk: The walked capture state.

    Returns:
        The ghosts, their events sorted by (tick, tank), and how many
        identities had no sighting to spawn from.
    """
    ghosts: list[GhostTankDict] = []
    events: list[GhostEventDict] = []
    unplaced = 0
    for tank_id, (team, name) in sorted(walk.identities.items()):
        sightings = walk.positions.get(tank_id)
        if not sightings:
            unplaced += 1
            continue
        first_tick, first_x, first_y = sightings[0]
        ghosts.append(
            GhostTankDict(
                tank_id=tank_id,
                team=team,
                rank=walk.ranks.get(tank_id, 1),
                name=name,
                x=first_x,
                y=first_y,
            )
        )
        last_per_tick: dict[int, tuple[int, int]] = {}
        for tick, x, y in sightings:
            last_per_tick[tick] = (x, y)
        for tick in sorted(last_per_tick):
            if tick == first_tick:
                continue
            x, y = last_per_tick[tick]
            events.append(
                GhostEventDict(tick=tick, tank_id=tank_id, kind="place", x=x, y=y, message_id=0)
            )
    ghost_ids = {ghost["tank_id"] for ghost in ghosts}
    for tick, shooter_id, aim_x, aim_y in walk.shots:
        if shooter_id in ghost_ids:
            events.append(
                GhostEventDict(
                    tick=tick, tank_id=shooter_id, kind="shoot", x=aim_x, y=aim_y, message_id=0
                )
            )
    for tick, sender_id, message_id, x, y in walk.chats:
        if sender_id in ghost_ids:
            events.append(
                GhostEventDict(
                    tick=tick, tank_id=sender_id, kind="chat", x=x, y=y, message_id=message_id
                )
            )
    events.sort(key=lambda event: (event["tick"], event["tank_id"]))
    return ghosts, events, unplaced


def _assemble(walk: _Walk) -> GhostSpecDict:
    """Turn the walked capture state into the replayable spec."""
    if walk.self_id is None or walk.self_id not in walk.positions:
        raise RuntimeError("ghost capture never identified or placed its own tank")
    self_path = walk.positions[walk.self_id]
    recorded_path: dict[int, tuple[int, int]] = {}
    for tick, x, y in self_path:
        recorded_path[tick] = (x, y)
    ghosts, events, unplaced = _assemble_ghosts(walk)
    containers: list[GhostContainerDict] = []
    equipment: list[tuple[int, int]] = []
    dot_atlas = walk.dot_atlas if walk.dot_atlas is not None else set()
    for (x, y), (value, visible) in sorted(walk.first_reads.items()):
        if value == -1:
            equipment.append((x, y))
        elif value > 0:
            containers.append(
                GhostContainerDict(x=x, y=y, volume=value, dotted=(x, y) in dot_atlas or visible)
            )
    # Dots the recording's atlas held but whose volume was never read
    # seed as drained dots — the map shows them, pickups answer code 4,
    # exactly the live experience of exposure memory.
    for x, y in sorted(dot_atlas - set(walk.first_reads)):
        containers.append(GhostContainerDict(x=x, y=y, volume=0, dotted=True))
    last_tick = max([event["tick"] for event in events] + list(recorded_path), default=0)
    first_self = self_path[0]
    return GhostSpecDict(
        client_team=walk.self_team,
        client_rank=walk.ranks.get(walk.self_id, 1),
        client_x=first_self[1],
        client_y=first_self[2],
        client_fuel=walk.client_fuel if walk.client_fuel is not None else 1000,
        client_counts=walk.client_counts if walk.client_counts is not None else [25] * 5,
        ghosts=ghosts,
        events=events,
        recorded_path=recorded_path,
        containers=containers,
        equipment=equipment,
        dot_atlas=sorted(dot_atlas),
        ticks=last_tick + 1,
        unplaced_tanks=unplaced,
    )


def compile_ghost_spec(capture_text: str) -> GhostSpecDict:
    """Compile one capture session into a replayable ghost spec.

    Rebuilds the capture's XOR table — callers that need a different
    session key afterwards (the sim boot) must rebuild their own.

    Args:
        capture_text: The raw ``capture_session.json`` contents.

    Returns:
        The tick-indexed spec.

    Raises:
        RuntimeError: If the capture has no magic/messages or never
            placed its own tank — an unreplayable recording, loudly.
    """
    session = narrow_json_to_dict(load_json_str(capture_text))
    magic = session.get("magic")
    raw_messages = session.get("messages")
    if not isinstance(magic, str) or not magic or raw_messages is None:
        raise RuntimeError("ghost capture is missing its magic or messages")
    xor_table = build_session_xor_table(magic)
    walk = _Walk()
    messages = [narrow_json_to_dict(m) for m in narrow_json_to_list(raw_messages)]
    messages.sort(key=lambda m: narrow_json_to_int(m["timestamp_ms"]))
    for message in messages:
        if message.get("direction") == "sent":
            continue
        t = narrow_json_to_int(message["timestamp_ms"])
        # The ghost compiler reads a recording to replay it; a payload
        # it cannot parse is reported and skipped, not silently
        # truncated ([[session-state-deglobalisation]]).
        try:
            frames = split_payload_frames(narrow_json_to_str(message.get("payload") or ""))
        except FramingError as error:
            log.warning("ghost compile: skipping unparseable payload: %s", error)
            continue
        for body in frames:
            if not body or try_decode_plaintext_ack(body) is not None:
                continue
            if _is_text_route(body[0], body):
                continue
            try:
                decoded = decode_message(body[0], xor_decode_body(body, xor_table, offset=1))
            except DecodeError as error:
                log.debug("ghost compile: undecodable frame skipped (%s)", error)
                continue
            _consume(walk, t, decoded)
    return _assemble(walk)


class GhostTracker:
    """Tracks how long the live bot follows the recorded client."""

    def __init__(self, recorded_path: dict[int, tuple[int, int]]) -> None:
        """Bind the tracker to the recorded per-tick client path."""
        self._path = recorded_path
        self._last_known: tuple[int, int] | None = None
        self.tracked_ticks = 0
        self.compared_ticks = 0
        self.first_divergence_tick = -1
        self.final_drift = 0

    def note_round(self, tick: int, live_x: int, live_y: int) -> None:
        """Compare one round's live position against the recording.

        Args:
            tick: The session tick just played.
            live_x: The live bot's X after the round.
            live_y: The live bot's Y after the round.
        """
        recorded = self._path.get(tick, self._last_known)
        if recorded is None:
            return
        self._last_known = recorded
        drift = max(abs(live_x - recorded[0]), abs(live_y - recorded[1]))
        self.compared_ticks += 1
        self.final_drift = drift
        if drift <= _TRACK_DRIFT_TILES:
            self.tracked_ticks += 1
        elif self.first_divergence_tick < 0:
            self.first_divergence_tick = tick

    def emit_summary(self) -> None:
        """Emit the run's tracking verdict as a diagnostic."""
        emit_diagnostic(
            diagnostic_kind="ghost_summary",
            compared_ticks=self.compared_ticks,
            tracked_ticks=self.tracked_ticks,
            first_divergence_tick=self.first_divergence_tick,
            final_drift=self.final_drift,
        )


def ghost_events_for_tick(spec: GhostSpecDict, tick: int) -> list[GhostEventDict]:
    """The recorded events due at one session tick, in replay order.

    Args:
        spec: The compiled spec.
        tick: The session tick about to be played.

    Returns:
        The tick's events (possibly empty).
    """
    return [event for event in spec["events"] if event["tick"] == tick]


def seed_ghost_world_population(
    world_containers: list[SimContainerDict],
    world_equipment: list[SimEquipmentDict],
    spec: GhostSpecDict,
    terrain: TerrainMapProtocol,
) -> int:
    """Append the capture's first-observed containers to a world.

    Wire-real reads can still land outside the sim's legality (a
    ferry-borne read patching a water tile the validator's rock rule
    would flag) — impassable non-water tiles are skipped with a tally.

    Args:
        world_containers: The world's container list (appended).
        world_equipment: The world's equipment list (appended).
        spec: The compiled spec.
        terrain: The loaded field terrain.

    Returns:
        How many observed containers were skipped as unseedable.
    """
    skipped = 0

    def seedable(x: int, y: int) -> bool:
        return terrain.is_passable(x, y) or terrain.get_terrain(x, y) == terrain.WATER

    for container in spec["containers"]:
        if not seedable(container["x"], container["y"]):
            skipped += 1
            continue
        world_containers.append(
            SimContainerDict(
                x=container["x"],
                y=container["y"],
                volume=container["volume"],
                dotted=container["dotted"],
            )
        )
    for x, y in spec["equipment"]:
        if not seedable(x, y):
            skipped += 1
            continue
        world_equipment.append(SimEquipmentDict(x=x, y=y))
    return skipped


__all__ = [
    "GhostEventDict",
    "GhostSpecDict",
    "GhostTankDict",
    "GhostTracker",
    "compile_ghost_spec",
    "ghost_events_for_tick",
    "seed_ghost_world_population",
]
