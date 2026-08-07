"""Boot a sim session and queue its opponents.

The tick-paced clock, the ghost-world seed, the bootstrap that returns
a bot wired to a live :class:`~tankpit_bot.sim.server.SimServer`, and
the two round-queueing helpers. The session loop that drives them is
:mod:`tankpit_bot.sim.run`.
"""

from __future__ import annotations

import zlib
from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot.bot.base import Bot
from tankpit_bot.browser.room_join import join_room
from tankpit_bot.physics.capacity import fuel_capacity
from tankpit_bot.protocol.naming import is_practice_bot_name
from tankpit_bot.sim.atlas_seed import seed_atlas_population
from tankpit_bot.sim.commands import ClientCommandDict
from tankpit_bot.sim.ghost import (
    GhostSpecDict,
    ghost_events_for_tick,
    seed_ghost_world_population,
)
from tankpit_bot.sim.lobby import SIM_ACCOUNT, SimLobby
from tankpit_bot.sim.opponent import decide_opponent, maybe_revive_opponent
from tankpit_bot.sim.practice_room import PracticeRoomDriver, seed_practice_roster
from tankpit_bot.sim.scenarios import (
    _FERRY_CLIENT_FUEL,
    SIM_CLIENT_ID,
    SIM_MAGIC,
    _require_seeds_passable,
)
from tankpit_bot.sim.server import SimServer
from tankpit_bot.sim.session import SimCDPSession, deliver_batch
from tankpit_bot.sim.spawn import find_open_tile_near
from tankpit_bot.sim.world import SimWorldDict, make_sim_tank
from tankpit_bot.sim.world_seed import (
    MINE_DENSITY,
    seed_ferries,
    seed_field_population,
    seed_minefield,
    seed_practice_client,
    select_practice_layout,
)
from tankpit_bot.sniffer.world_state import reset_world_state

log = get_logger(__name__)


class TickPacedClock:
    """The sim session's decision clock: one real tick per round.

    Wall time is the wrong clock for a sim session — an in-process
    round takes microseconds, so live TTLs (forage coverage 180 s,
    harvest memory 10 min, belief freshness gates) either never age
    or age at the mercy of machine load (the 2026-08-01 flake: the
    same 600-round session exited at round 54 solo and never under
    xdist). Pacing the decision clock at the measured 2 s server tick
    makes sessions deterministic AND live-realistic: a 300-round soak
    now ages exactly like a 10-minute session.
    """

    def __init__(self, start_ms: int) -> None:
        """Anchor the clock at the session's real start time."""
        self._now_ms = start_ms

    def __call__(self) -> int:
        """Return the paced session time."""
        return self._now_ms

    def advance(self, delta_ms: int) -> None:
        """Advance the session by one round's worth of time."""
        self._now_ms += delta_ms


def _seed_ghost_world(
    world: SimWorldDict,
    terrain: _test_hooks.TerrainMapProtocol,
    ghost_spec: GhostSpecDict,
    atlas_path: Path | None,
) -> None:
    """Seed a recorded session's world: client, ghosts, containers.

    Args:
        world: Simulated world (mutated).
        terrain: The loaded field terrain.
        ghost_spec: The compiled recording.
        atlas_path: Optional mined-atlas underlay — fills tiles the
            recording never observed; the capture's own reads stay
            per-tile authoritative.
    """
    client = make_sim_tank(
        SIM_CLIENT_ID,
        ghost_spec["client_team"],
        ghost_spec["client_rank"],
        ghost_spec["client_x"],
        ghost_spec["client_y"],
        ghost_spec["client_fuel"],
    )
    client["counts"] = list(ghost_spec["client_counts"])
    world["tanks"][SIM_CLIENT_ID] = client
    placed = 0
    for ghost in ghost_spec["ghosts"]:
        spawn_x, spawn_y = ghost["x"], ghost["y"]
        if not terrain.is_passable(spawn_x, spawn_y):
            spot = find_open_tile_near(
                world, terrain, spawn_x, spawn_y, world["tick"], min_radius=1, max_radius=4
            )
            if spot is None:
                continue
            spawn_x, spawn_y = spot
        world["tanks"][ghost["tank_id"]] = make_sim_tank(
            ghost["tank_id"],
            ghost["team"],
            ghost["rank"],
            spawn_x,
            spawn_y,
            fuel_capacity(ghost["rank"]),
            name=ghost["name"],
        )
        placed += 1
    if atlas_path is not None:
        atlas_tally = seed_atlas_population(
            world, terrain, atlas_path, frozenset(ghost_spec["dot_atlas"])
        )
        ghost_tiles = {(c["x"], c["y"]) for c in ghost_spec["containers"]}
        ghost_tiles.update(ghost_spec["equipment"])
        world["containers"] = [
            c for c in world["containers"] if (c["x"], c["y"]) not in ghost_tiles
        ]
        world["equipment"] = [e for e in world["equipment"] if (e["x"], e["y"]) not in ghost_tiles]
        log.info("ghost atlas underlay: %s", atlas_tally)
    skipped = seed_ghost_world_population(
        world["containers"], world["equipment"], ghost_spec, terrain
    )
    log.info(
        "ghost world: %d ghosts placed (%d identities unsighted), "
        "%d containers/equipment seeded (%d unseedable), %d ticks of timeline",
        placed,
        ghost_spec["unplaced_tanks"],
        len(world["containers"]) + len(world["equipment"]),
        skipped,
        ghost_spec["ticks"],
    )


def _boot(
    world: SimWorldDict,
    *,
    practice: bool = False,
    stamp: str = "",
    atlas_path: Path | None = None,
    ghost_spec: GhostSpecDict | None = None,
) -> tuple[Bot, SimServer, SimCDPSession, PracticeRoomDriver | None]:
    """Wire a real Bot to the sim over the CDP seam.

    Args:
        world: The world the server will own. In practice mode this
            arrives EMPTY of tanks and containers — the stamp-selected
            real layout seeds the client spawn and the full 36-bot
            roster, and ``seed_field_population`` lays down the static
            container field ([[game-economy]] 2026-07-25: the world
            never spawns at runtime).
        practice: When True, build the practice-room world before the
            handshake so the join roster dump includes the bots, and
            hand the server their ids for the corpse-window
            reactivation hook.
        stamp: The run stamp (the layout selector for practice and
            atlas spawns; unused otherwise).
        atlas_path: When set, the mined longitudinal atlas replaces
            the statistical container field ([[game-economy]]
            2026-08-01): in practice mode the roster still seeds, but
            the ground truth is the REAL room; standalone it is a
            pure-forage world on the real field.
        ghost_spec: When set, seed a recorded session's world: the
            client at its recorded spawn state, every sighted
            opponent as a replayable ghost (at its first PASSABLE
            sighting; ferry riders and other water sightings spawn at
            their first dry tile), and the capture's first-observed
            containers.

    Returns:
        The bot, the server, the seam link, and the practice-room
        driver (None outside practice mode), with the join handshake
        already delivered.

    Raises:
        XorStaticKeyUnavailableError: If the XOR static key cannot be
            read — the sim's binary seam needs the real cipher.
        RuntimeError: If the field terrain GIF is unavailable, or a
            scenario seed sits on impassable ground — a sim run needs
            both right, loudly.
    """
    reset_world_state()
    gif_path = Path(world["field"])
    if not _test_hooks.path_exists(gif_path):
        raise RuntimeError(f"terrain GIF {gif_path} not found — run `make download-fields` first")
    terrain = _test_hooks.load_terrain_map(gif_path)
    driver: PracticeRoomDriver | None = None
    roster_ids: frozenset[int] = frozenset()
    if practice:
        layout = select_practice_layout(stamp)
        log.info(
            "practice layout %s: client spawn %s, %d bots",
            layout["provenance"],
            layout["client_spawn"],
            len(layout["roster"]),
        )
        seed_practice_client(world, terrain, layout, SIM_CLIENT_ID)
        roster_ids = seed_practice_roster(world, terrain, layout["roster"])
        driver = PracticeRoomDriver(roster_ids)
        if atlas_path is not None:
            tally = seed_atlas_population(world, terrain, atlas_path)
            log.info("atlas field %s: %s", atlas_path, tally)
        else:
            seed_field_population(world, terrain, seed=zlib.crc32(stamp.encode("utf-8")))
    elif ghost_spec is not None:
        _seed_ghost_world(world, terrain, ghost_spec, atlas_path)
        # Reactive ghosts (2026-08-03): bot-named ghosts carry the
        # certified roster policy UNDER their recorded timeline — the
        # live bot's shots draw the mined shot-for-shot return fire
        # and team aggro even where the recording has no answer, and
        # a killed bot ghost reactivates by the corpse-window law.
        # Human-named ghosts stay pure recordings.
        roster_ids = frozenset(
            ghost["tank_id"]
            for ghost in ghost_spec["ghosts"]
            if is_practice_bot_name(ghost["name"]) and ghost["tank_id"] in world["tanks"]
        )
        if roster_ids:
            driver = PracticeRoomDriver(roster_ids)
    elif atlas_path is not None:
        layout = select_practice_layout(stamp)
        seed_practice_client(world, terrain, layout, SIM_CLIENT_ID)
        # A pure-forage world has no targets, so a hunt-ready spawn
        # would exit ``no_viable_targets`` on tick 2 (first standalone
        # atlas run did exactly that). Start lean: the session's work
        # IS the real field's larder economics.
        world["tanks"][SIM_CLIENT_ID]["fuel"] = _FERRY_CLIENT_FUEL
        tally = seed_atlas_population(world, terrain, atlas_path)
        log.info("atlas forage world %s: %s", atlas_path, tally)
    # The room's standing minefield, laid LAST so it can share tiles
    # with the containers already seeded — which the game does, and
    # which is exactly where the bot's clearance and landing-
    # displacement machinery earns its keep. Every scenario gets one:
    # the archive's opening viewport patch is 27-49% mined in every
    # session that has one ([[session-state-deglobalisation]]).
    laid = seed_minefield(world, terrain)
    log.info("minefield: %d mines at density %.2f", laid, MINE_DENSITY)
    # Ferries too: the sim floated exactly one, at a hardcoded tile, in
    # one scenario, so every other room had none and the 205 archived
    # 0x4A drift frames had nothing to answer them. A scenario that
    # placed its own keeps them ([[ferry-mechanics]]).
    log.info("ferries: %d afloat", seed_ferries(world, terrain))
    _require_seeds_passable(world, terrain)
    server = SimServer(world, terrain, client_id=SIM_CLIENT_ID, roster_ids=roster_ids)
    bot = Bot("https://sim.tankpit.local/", headless=True)
    # The bot lifts the magic off the page client's AUTH frame live, via
    # a CDP event stream the sim has no counterpart for; the link sends
    # that frame but cannot deliver it back through that path, so the
    # magic is handed over directly and the frame stands for itself on
    # the wire ([[session-state-deglobalisation]]).
    bot._magic = SIM_MAGIC
    bot._on_magic_captured(SIM_MAGIC)
    link = SimCDPSession(server, SIM_MAGIC, SimLobby(SIM_ACCOUNT))
    bot._cdp = link
    # The link is the page too: it satisfies the narrow page protocols
    # the poll-and-read flows take, so the PRODUCTION autoscroll
    # enforcement and account-stats capture run from the tick body
    # instead of being skipped for want of a browser. Pointing the
    # bot's capture list at the link's wire log is what lets the
    # enforcer find its ack — they are the same session's traffic.
    bot._page = link
    bot._messages = link.wire_log
    link.open_lobby()
    # The PRODUCTION lobby flow, against the sim's plaintext channel.
    # This used to be skipped and its one durable effect — a selected
    # room, without which the bot's decision terrain stays None for the
    # whole run and every terrain-gated behaviour silently self-disables
    # (2026-07-31 human-opponent soak) — was hand-installed instead. Now
    # it comes from the room list the way it does live, and 1,571
    # archived lobby frames per session finally have a sim counterpart.
    if not join_room(link, link):
        raise RuntimeError("sim lobby: the production join flow did not reach a room")
    deliver_batch(bot._cdp_message_buffer, server.handshake(), link)
    return bot, server, link, driver


def _queue_ghost_round(server: SimServer, spec: GhostSpecDict, round_index: int) -> frozenset[int]:
    """Feed one round's recorded ghost actions into the server.

    Dead ghosts skip their remaining timeline (the live fight may
    have killed them earlier than the recording did).

    Args:
        server: The live sim server.
        spec: The compiled ghost spec.
        round_index: The session tick about to be played.

    Returns:
        Ids of ghosts that acted from the recording this tick — the
        reactive-policy layer yields to them (recorded authority).
    """
    acted: set[int] = set()
    for event in ghost_events_for_tick(spec, round_index):
        tank = server.world["tanks"].get(event["tank_id"])
        if tank is None or not tank["alive"]:
            continue
        acted.add(event["tank_id"])
        if event["kind"] == "place":
            server.relocate_tank(event["tank_id"], event["x"], event["y"])
        elif event["kind"] == "shoot":
            server.queue_command(
                event["tank_id"],
                ClientCommandDict(
                    kind="shoot",
                    command=115,
                    x=event["x"],
                    y=event["y"],
                    target_id=0,
                    slot=0,
                    message_id=0,
                    direction=0,
                ),
            )
        else:
            server.queue_command(
                event["tank_id"],
                ClientCommandDict(
                    kind="chat",
                    command=109,
                    x=event["x"],
                    y=event["y"],
                    target_id=0,
                    slot=0,
                    message_id=event["message_id"],
                    direction=0,
                ),
            )
    return frozenset(acted)


def _queue_round_opponents(
    server: SimServer,
    driver: PracticeRoomDriver | None,
    opponent: bool,
    ghost_spec: GhostSpecDict | None,
    enemy_id: int,
    round_index: int,
) -> int:
    """Queue this round's non-client actions for the active mode.

    Args:
        server: The live sim server.
        driver: The roster-policy driver — practice mode's seeded
            roster, or ghost mode's reactive bot-named ghosts.
        opponent: Whether the scripted opponent plays.
        ghost_spec: The ghost timeline, when in ghost mode.
        enemy_id: The scripted opponent's current wire id.
        round_index: The session tick about to be played.

    Returns:
        The (possibly revived) scripted opponent id.
    """
    recorded_actors: frozenset[int] = frozenset()
    if ghost_spec is not None:
        recorded_actors = _queue_ghost_round(server, ghost_spec, round_index)
    if driver is not None:
        decisions = driver.decide_all(server.world, server.terrain, hold_ids=recorded_actors)
        for bot_id, command in decisions:
            server.queue_command(bot_id, command)
    elif opponent:
        enemy_id = maybe_revive_opponent(server, enemy_id, SIM_CLIENT_ID)
        opponent_command = decide_opponent(server.world, enemy_id, SIM_CLIENT_ID)
        if opponent_command is not None:
            server.queue_command(enemy_id, opponent_command)
    return enemy_id


__all__ = [
    "TickPacedClock",
    "log",
]
