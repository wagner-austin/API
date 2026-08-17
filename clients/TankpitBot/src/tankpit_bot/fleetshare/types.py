"""Fleet knowledge-sharing records: the per-bot report and its rows.

The shared knowledge layer (fleet ruling 2026-08-14): every bot writes
one ``knowledge.json`` beside its ``hud.json`` each tick — its own
fresh beliefs, offered to same-team siblings — and reads the reports
its teammates wrote. The transport is the run-directory filesystem,
the same channel the fleet page already reads, so a single tank runs
identically with zero siblings and a fleet coordinates with no
manager process required. Codecs live in
:mod:`tankpit_bot.fleetshare.codecs`.
"""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict

FleetRole = Literal[
    "fighter",
    "gatherer",
]
"""What a fleet bot does with its ticks.

``fighter`` runs the full HUNT/COLLECT doctrine. ``gatherer`` never
enters HUNT: it lives in the COLLECT cascade — scan, sweep, hop —
roaming the map and publishing what it finds for the fighters of its
color.
"""


FLEET_ROLES: tuple[FleetRole, ...] = (
    "fighter",
    "gatherer",
)


class FleetEnemySightingDict(TypedDict):
    """One enemy tank a reporting bot has a fresh positional belief for.

    Attributes:
        tank_id: The enemy's tank id.
        name: Display name from the reporter's registry.
        team: The enemy's team (0-3).
        rank: Military rank (0 recruit .. 8 general).
        x: Last known X.
        y: Last known Y.
        damage_state: Fuel-quartile damage tier (0 near death .. 3 full).
        observed_ms: The reporter's ``last_position_update_ms`` for the
            tank — the belief's own age, NOT the report's write time,
            so receivers merge with true freshness.
    """

    tank_id: int
    name: str
    team: int
    rank: int
    x: int
    y: int
    damage_state: int
    observed_ms: int


class FleetContainerSightingDict(TypedDict):
    """One container a reporting bot believes exists.

    Attributes:
        x: Container X.
        y: Container Y.
        is_fuel: True for fuel, False for equipment.
        volume: Fuel volume (0 for equipment).
        observed_ms: The reporter's belief timestamp for the container.
    """

    x: int
    y: int
    is_fuel: bool
    volume: int
    observed_ms: int


class FleetContainerRemovalDict(TypedDict):
    """One container tile the reporting bot disproved or consumed.

    The negative half of container knowledge ([[fleet-coordination]]):
    a pickup that emptied the tile, a code-4 disproof, an unreachable
    verdict, or radar stating empty. Receivers drop any belief in the
    tile OBSERVED BEFORE the removal and inherit the tombstone, so
    one bot's consumption stops the whole fleet from chasing the
    ghost.

    Attributes:
        x: Tile X.
        y: Tile Y.
        removed_ms: When the reporter's removal happened.
    """

    x: int
    y: int
    removed_ms: int


class FleetScannedTileDict(TypedDict):
    """One tile the reporting bot holds live radar coverage for.

    The negative space of the worldview: "I looked here at
    ``observed_ms``" -- what a scan found rides the enemy and
    container rows; this row says the ground itself is known, so a
    sibling's forage and sweep stop paying radars for it
    ([[fleet-coordination]] scanner division of labor).

    Attributes:
        x: Tile X.
        y: Tile Y.
        observed_ms: When the reporter's radar covered the tile.
    """

    x: int
    y: int
    observed_ms: int


class FleetReportDict(TypedDict):
    """One bot's knowledge offer to its same-team siblings.

    Attributes:
        instance: The reporting bot's instance name ("" for the
            sole-bot namespace).
        team: The reporter's team — receivers merge SAME-TEAM reports
            only (knowledge sharing is an alliance, [[fleet-coordination]]).
        tank_id: The reporter's own tank id, so receivers never merge
            a sighting of themselves or the reporter as an "enemy".
        role: The reporter's :data:`FleetRole`.
        x: The reporter's X at write time.
        y: The reporter's Y at write time.
        engaged_target_id: The reporter's held combat lock (-1 for
            none) — the focus-fire signal teammates' acquisition
            prefers.
        written_ms: Wall-clock ms of the write. Receivers drop reports
            older than the freshness TTL, so a dead bot's last file
            ages out instead of steering the living.
        enemies: Fresh enemy sightings.
        containers: Fresh container sightings.
        removed: Container tiles recently disproved or consumed --
            the negative knowledge that stops teammates chasing
            ghosts.
        scanned: Tiles under live radar coverage (within the forage
            coverage TTL) -- the shared scan map.
    """

    instance: str
    team: int
    tank_id: int
    role: FleetRole
    x: int
    y: int
    engaged_target_id: int
    written_ms: int
    enemies: list[FleetEnemySightingDict]
    containers: list[FleetContainerSightingDict]
    removed: list[FleetContainerRemovalDict]
    scanned: list[FleetScannedTileDict]


__all__ = [
    "FLEET_ROLES",
    "FleetContainerRemovalDict",
    "FleetContainerSightingDict",
    "FleetEnemySightingDict",
    "FleetReportDict",
    "FleetRole",
    "FleetScannedTileDict",
]
