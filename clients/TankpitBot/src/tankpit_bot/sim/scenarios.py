"""Sim scenario worlds: the default arena, the ferry lake, and the
mode resolver that turns CLI flags into one seeded world.

Split from ``sim/run.py`` 2026-08-01 (the 400-600 line rule): this
module owns WHAT world a session plays on; ``run.py`` owns HOW the
session runs (boot, loop, clock, CLI).
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from tankpit_bot import _test_hooks
from tankpit_bot.sim.atlas_seed import DEFAULT_ATLAS_PATH
from tankpit_bot.sim.ghost import GhostSpecDict, compile_ghost_spec
from tankpit_bot.sim.world import (
    SimContainerDict,
    SimEquipmentDict,
    SimFerryDict,
    SimWorldDict,
    make_sim_tank,
    make_sim_world,
)

SIM_MAGIC = "simmagic5uk3et4epiexu"
"""The sim session's magic, in the shape the real server issues.

Twenty lowercase alphanumerics, like every magic in the archive
(``fjoodiu5uk3et4epiexu``, ``jhw7j98myv3i0g2qx5b9``, …). This was
``"simmagic"`` — eight characters — which the PRODUCTION extractor
refuses outright (``extract_magic_from_auth_payload`` requires ten or
more), so the moment the sim started putting a real AUTH frame on the
wire, the bot could not have read its own cipher out of it. A shape
the real reader rejects is not a simulation of it
([[session-state-deglobalisation]])."""

SIM_CLIENT_ID = 9
SIM_ENEMY_ID = 11
SIM_FIELD = "field01_r.gif"
_SIM_DIR = Path("runs") / "sim"
_DEFAULT_ROUNDS = 150

# The default arena is the most open ground on field01: a fully
# passable 21x21 clearing centered on (216, 108) (verified against
# the real GIF — the naive (100, 100) region is coastal, and six of
# its would-be container tiles sit in water).
_ARENA_X = 216
_ARENA_Y = 108
_DEFAULT_FUEL_CONTAINERS: tuple[tuple[int, int, int], ...] = (
    (219, 108, 300),
    (213, 112, 400),
    (222, 103, 400),
    (210, 105, 400),
    (225, 112, 400),
    (216, 116, 400),
    (208, 111, 300),
    (221, 100, 300),
)
# Enough equipment that the bot can actually rebuild its radar
# buffer: collect hops burn extras on scan-on-landing roughly as fast
# as one container's radar stack refills them, so a sparse seeding
# strands the session below the hunt threshold (first clearing run:
# exit no_productive_collect at round 39 with all three containers
# drained).
_DEFAULT_EQUIPMENT: tuple[tuple[int, int], ...] = (
    (219, 111),
    (214, 106),
    (210, 114),
    (224, 105),
    (212, 100),
    (218, 115),
    (208, 104),
    (223, 113),
)


def make_default_sim_world() -> SimWorldDict:
    """Build the default scenario: a sustainable fight-and-collect map.

    The client spawns rank 1 in the field01 clearing combat-ready
    (the fight runs first; the extras it burns scanning pull the
    equipment-collection paths in afterwards), the armed rank-8
    opponent holds ten tiles east, and enough fuel and equipment is
    on the ground that a long session stays productive. A finite
    world still drains eventually — container respawns are an
    uncertified gap — so the production ``no_productive_collect``
    exit is this scenario's natural old age.

    Returns:
        The seeded world.
    """
    world = make_sim_world(SIM_FIELD)
    world["tanks"][SIM_CLIENT_ID] = make_sim_tank(SIM_CLIENT_ID, 2, 1, _ARENA_X, _ARENA_Y, 1100)
    world["tanks"][SIM_CLIENT_ID]["counts"] = [25, 25, 25, 25, 25]
    # A winnable fight, not an execution. The bot FIGHTS WITH ARMOR
    # OFF by policy (desired_equipment is dual/homing/radar — the
    # scenario-harness baseline pins armor disabled), so its
    # effective HP is its fuel — and an out-of-ammo opponent still
    # lands unlimited 45-fuel singles, so ammo does not cap damage.
    # A 500-fuel opponent (six kill-hits needed) lets the client win
    # the deterministic knife fight while still eating real return
    # fire; heavier seedings killed the bot by round ~20 every run
    # (that ``deactivated`` ending is itself covered by the exit
    # law's tests).
    world["tanks"][SIM_ENEMY_ID] = make_sim_tank(SIM_ENEMY_ID, 1, 8, _ARENA_X + 10, _ARENA_Y, 500)
    world["tanks"][SIM_ENEMY_ID]["counts"] = [0, 4, 0, 2, 3]
    for x, y, volume in _DEFAULT_FUEL_CONTAINERS:
        world["containers"].append(SimContainerDict(x=x, y=y, volume=volume, dotted=True))
    for x, y in _DEFAULT_EQUIPMENT:
        world["equipment"].append(SimEquipmentDict(x=x, y=y))
    return world


# The ferry scenario's lake: the field01 water body at (112,112)
# (4,456 connected water tiles, verified against the real GIF
# 2026-08-01). The client spawns on the west shore; the water-locked
# container floats 6 east (inside the join window, so the landing
# radar believes it); the ferry idles 12 east on its own water — the
# doctrine's canonical shape (user, [[ferry-mechanics]]: "many times
# it will be on its own area in the water. not touching land") —
# OUTSIDE the join window, so only the scope scout's east pan can
# reveal it. A second container floats deeper in for the riding
# pickup law.
_FERRY_SHORE_X = 106
_FERRY_SHORE_Y = 112
# Fuel arithmetic: the land stock (2x150) tops the 400-fuel spawn to
# ~700 at best — always below the hunt-readiness fuel bar — so the
# only route to a full tank runs across the water. Without that
# scarcity the first cut of this scenario ended round 11: land fuel
# alone reached 1062, hunt engaged, and the empty roster exited
# ``no_viable_targets`` before the ferry work ever mattered.
_FERRY_CLIENT_FUEL = 400
_FERRY_TILE: tuple[int, int] = (118, 112)
_FERRY_WATER_FUEL: tuple[tuple[int, int, int], ...] = (
    (112, 112, 700),
    (120, 115, 500),
)
_FERRY_LAND_FUEL: tuple[tuple[int, int, int], ...] = (
    (104, 110, 150),
    (103, 114, 150),
)
_FERRY_LAND_EQUIPMENT: tuple[tuple[int, int], ...] = (
    (105, 115),
    (102, 111),
    (108, 114),
    (104, 116),
)


def make_ferry_sim_world() -> SimWorldDict:
    """Build the ferry scenario: a water-locked larder behind one pan.

    No opponent — the session is pure forage. The intended chain is
    the full F5 doctrine ([[ferry-mechanics]]): the landing radar
    believes the floating container -> the larder declines it
    ``no_landing`` -> the scope scout pans east ([[viewport-shift-
    protocol]]) -> the 0x5A patch carries the ferry as wire terrain 5
    -> the next larder tick hops ``ferry_served`` onto the boarding
    tile -> the held fuel lock rides the water to the pickup. The
    client spawns BELOW capacity so the larder has a deficit to
    serve, with land stock around the shore keeping the session
    productive on either side of the ferry work.

    Returns:
        The seeded world.
    """
    world = make_sim_world(SIM_FIELD)
    client = make_sim_tank(SIM_CLIENT_ID, 2, 1, _FERRY_SHORE_X, _FERRY_SHORE_Y, _FERRY_CLIENT_FUEL)
    client["counts"] = [25, 25, 25, 25, 25]
    world["tanks"][SIM_CLIENT_ID] = client
    world["ferries"].append(SimFerryDict(x=_FERRY_TILE[0], y=_FERRY_TILE[1]))
    for x, y, volume in _FERRY_WATER_FUEL + _FERRY_LAND_FUEL:
        world["containers"].append(SimContainerDict(x=x, y=y, volume=volume, dotted=True))
    for x, y in _FERRY_LAND_EQUIPMENT:
        world["equipment"].append(SimEquipmentDict(x=x, y=y))
    return world


def _require_seeds_passable(world: SimWorldDict, terrain: _test_hooks.TerrainMapProtocol) -> None:
    """Reject a scenario whose seeds sit on illegal ground — loudly.

    The first real-terrain run (2026-07-22) seeded the coastal
    (100, 100) region and drowned six containers: the bot starved
    among dots it could never reach. Nothing on the real map spawns
    on impassable ground UNSERVED: a container floating on WATER is a
    real live population ([[ferry-mechanics]] F5 — 12 of 13 believed
    containers water-locked in run bot-20260728-093011), but only a
    world that also seeds a ferry within boarding range of it offers
    the harvest path the real map does. Ferries themselves must float
    (a ferry on land is a harness typo, not a scenario).

    Args:
        world: The seeded world.
        terrain: The loaded field terrain.

    Raises:
        RuntimeError: Listing every illegal seed.
    """

    offenders: list[str] = []
    for tank_id, tank in sorted(world["tanks"].items()):
        if not terrain.is_passable(tank["x"], tank["y"]):
            offenders.append(f"tank {tank_id} at ({tank['x']},{tank['y']})")
    for container in world["containers"]:
        if not _container_seed_legal(world, terrain, container["x"], container["y"]):
            offenders.append(f"fuel container at ({container['x']},{container['y']})")
    for equipment in world["equipment"]:
        if not _container_seed_legal(world, terrain, equipment["x"], equipment["y"]):
            offenders.append(f"equipment at ({equipment['x']},{equipment['y']})")
    for ferry in world["ferries"]:
        if terrain.get_terrain(ferry["x"], ferry["y"]) != terrain.WATER:
            offenders.append(f"ferry at ({ferry['x']},{ferry['y']})")
    if offenders:
        raise RuntimeError(
            "impassable scenario seeds on " + world["field"] + ": " + "; ".join(offenders)
        )


def _container_seed_legal(
    world: SimWorldDict,
    terrain: _test_hooks.TerrainMapProtocol,
    x: int,
    y: int,
) -> bool:
    """A ground seed or a floating one; only rock/void is a typo.

    The longitudinal atlas ([[game-economy]] 2026-08-01) proved the
    water-locked container population is REAL live state (12 of 13
    believed containers in run bot-20260728-093011 floated), and
    ferries DRIFT — the live map guarantees no boarding tile near a
    floating container at any given instant either. So water seeds
    are legal with or without a ferry; the harvest path is the ferry
    doctrine's business ([[ferry-mechanics]]), not the validator's.

    Args:
        world: The seeded world (unused — kept for signature parity
            with earlier ferry-service semantics).
        terrain: The loaded field terrain.
        x: Seed tile X.
        y: Seed tile Y.

    Returns:
        True when the container/equipment seed is a legal world.
    """
    del world
    if terrain.is_passable(x, y):
        return True
    return terrain.get_terrain(x, y) == terrain.WATER


def _select_scenario_world(
    *,
    practice: bool,
    ferry_mode: bool,
    atlas_mode: bool,
    opponent: bool,
    opponent_name: str,
) -> tuple[SimWorldDict, bool]:
    """Build the requested scenario's world and resolve the opponent flag.

    Args:
        practice: Practice-roster session (the world seeds in
            ``_boot``, so it starts empty here).
        ferry_mode: Ferry forage scenario (never has an opponent).
        atlas_mode: Standalone atlas forage world (client + the mined
            real field, no opponent; seeds in ``_boot``).
        opponent: The caller's opponent request.
        opponent_name: Optional scripted-opponent wire name.

    Returns:
        The world and the effective opponent flag.
    """
    if practice:
        return make_sim_world(SIM_FIELD), opponent
    if ferry_mode:
        return make_ferry_sim_world(), False
    if atlas_mode:
        return make_sim_world(SIM_FIELD), False
    world = make_default_sim_world()
    if opponent_name:
        world["tanks"][SIM_ENEMY_ID]["name"] = opponent_name
    return world, opponent


def _resolve_session_mode(
    *,
    opponent: bool,
    practice: bool,
    ferry: bool,
    atlas: str | None,
    ghost: str | None,
    opponent_name: str,
) -> tuple[SimWorldDict, bool, bool, GhostSpecDict | None, Path | None, bool]:
    """Resolve the requested flags into one scenario's inputs.

    Ghost replay takes precedence over every other flag (a recording
    IS a complete scenario); then practice > ferry > atlas > default.

    Args:
        opponent: The caller's opponent request.
        practice: Practice-roster session flag.
        ferry: Ferry scenario flag.
        atlas: Optional atlas path string.
        ghost: Optional capture path string.
        opponent_name: Optional scripted-opponent wire name.

    Returns:
        ``(world, opponent, practice, ghost_spec, atlas_path,
        ferry_mode)`` ready for ``_boot``.
    """
    if ghost is not None:
        ghost_spec = compile_ghost_spec(_test_hooks.read_text(Path(ghost)))
        # Ghost composes with the atlas: the mined room fills every
        # tile the recording never observed (a replay world seeded
        # only from first-reads starves collect long before the
        # recorded session did — the self-replay validation exited at
        # round 21 of 300), and the capture's own reads stay
        # per-tile authoritative.
        ghost_atlas = Path(atlas) if atlas is not None else None
        return make_sim_world(SIM_FIELD), False, False, ghost_spec, ghost_atlas, False
    ferry_mode = ferry and not practice
    atlas_path = Path(atlas) if atlas is not None and not ferry_mode else None
    world, opponent = _select_scenario_world(
        practice=practice,
        ferry_mode=ferry_mode,
        atlas_mode=atlas_path is not None and not practice,
        opponent=opponent,
        opponent_name=opponent_name,
    )
    return world, opponent, practice, None, atlas_path, ferry_mode


class _CliArgsDict(TypedDict):
    """The sim CLI's parsed flags.

    ``out`` is the archive directory the session's capture and world
    land in; it defaults to the shared ``runs/sim`` archive and is
    pointed elsewhere by ``scripts.build_sim_baseline``, which needs a
    directory holding exactly one generation of the sim.
    """

    rounds: int
    opponent: bool
    practice: bool
    ferry: bool
    atlas: str | None
    ghost: str | None
    stamp: str | None
    opponent_name: str
    out: str


def _apply_valued_flag(parsed: _CliArgsDict, token: str, value: str) -> bool:
    """Apply one flag whose meaning is carried by the NEXT token.

    Args:
        parsed: The bundle being filled (mutated on a match).
        token: The flag token.
        value: The token following it.

    Returns:
        True when ``token`` is a valued flag and both tokens are
        consumed; False when it is not one, leaving it for
        :func:`_apply_bare_flag`.

    Raises:
        ValueError: If ``--rounds`` names a non-integer. A tick count
            the caller mistyped is not something to guess at.
    """
    if token == "--rounds":
        parsed["rounds"] = int(value)
    elif token == "--ghost":
        parsed["ghost"] = value
    elif token == "--stamp":
        parsed["stamp"] = value
    elif token == "--human-opponent":
        parsed["opponent_name"] = value
    elif token == "--out":
        parsed["out"] = value
    else:
        return False
    return True


def _apply_bare_flag(parsed: _CliArgsDict, token: str, rest: list[str]) -> int:
    """Apply one flag that stands on its own.

    ``--from-atlas`` is the reason this takes ``rest`` rather than a
    single value: its path is OPTIONAL, so it reads the next token
    only when that token is not itself a flag.

    Args:
        parsed: The bundle being filled (mutated on a match).
        token: The flag token.
        rest: The tokens after it.

    Returns:
        How many tokens were consumed — 2 for ``--from-atlas PATH``,
        1 for everything else including tokens this does not
        recognise, which are skipped.
    """
    if token == "--no-opponent":
        parsed["opponent"] = False
    elif token in ("--practice", "--ferry"):
        parsed["practice" if token == "--practice" else "ferry"] = True
    elif token == "--from-atlas":
        if rest and not rest[0].startswith("--"):
            parsed["atlas"] = rest[0]
            return 2
        parsed["atlas"] = str(DEFAULT_ATLAS_PATH)
    return 1


def _parse_cli(args: list[str]) -> _CliArgsDict:
    """Parse the manual flag loop into one typed bundle.

    The two flag SHAPES are parsed separately — a flag whose value is
    the next token, and a flag that stands alone — because a single
    chain covering both grew past the branch ceiling the moment a
    seventh flag arrived, and the two shapes have genuinely different
    consumption rules.

    Args:
        args: Raw CLI tokens.

    Returns:
        The parsed flags (unknown tokens are skipped).
    """
    parsed = _CliArgsDict(
        rounds=_DEFAULT_ROUNDS,
        opponent=True,
        practice=False,
        ferry=False,
        atlas=None,
        ghost=None,
        stamp=None,
        opponent_name="",
        out=str(_SIM_DIR),
    )
    index = 0
    while index < len(args):
        token = args[index]
        if index + 1 < len(args) and _apply_valued_flag(parsed, token, args[index + 1]):
            index += 2
            continue
        index += _apply_bare_flag(parsed, token, args[index + 1 :])
    return parsed


__all__ = [
    "SIM_CLIENT_ID",
    "SIM_ENEMY_ID",
    "SIM_FIELD",
    "SIM_MAGIC",
    "_DEFAULT_ROUNDS",
    "_FERRY_CLIENT_FUEL",
    "_SIM_DIR",
    "_parse_cli",
    "_require_seeds_passable",
    "_resolve_session_mode",
    "_select_scenario_world",
    "make_default_sim_world",
    "make_ferry_sim_world",
]
