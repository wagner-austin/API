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


EngagementDoctrine = Literal[
    "skirmish",
    "swarm",
    "duelist",
    "passive",
]
"""How a bot times its human engagements (operator order 2026-09-01:
"will we have pluggable strategies that we can swap in per bot or per
group of bots at a whim?"). A doctrine is DATA selecting between
existing tested gates — never loaded code.

``skirmish`` is today's behavior: the wartime readiness floor and
focus fire, each bot joining as soon as its own bars clear.
``swarm`` adds the muster: engage a consented human immediately when
a sibling is already fighting it, otherwise keep farming until the
war-ready quorum stands and strike together — the serial trickle's
fix. ``duelist`` engages only when NO sibling holds the human (first
come duels, the rest keep farming). ``passive`` never initiates
against humans; consent-based return fire is untouched.
"""


ENGAGEMENT_DOCTRINES: tuple[EngagementDoctrine, ...] = (
    "skirmish",
    "swarm",
    "duelist",
    "passive",
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


class FleetMineSightingDict(TypedDict):
    """One hostile mine a reporting bot believes is laid.

    The mine-aware layer between the bots (operator order 2026-09-01:
    "have a mine aware layer between the bots"): a sibling's hostile
    mine lands in the receiver's mine registry, so its composed
    decision terrain avoids the walk-over (45 fuel, movement
    arrested) and its teleports expect the displacement — BEFORE the
    receiver ever windows the tile. Only mines hostile to the
    reporting TEAM are published (reports merge same-team only, and
    own-color mines are passable to every sibling by game physics,
    [[mine-mechanics]]). No removal rows: mines never drift, and a
    cleared mine's phantom re-import self-limits to the share
    horizon — the clearer stops believing it, the sighted copy ages
    out of the reporter's publication within
    :data:`~tankpit_bot.fleetshare.report.MINE_SIGHTING_TTL_MS`, and
    contact disproofs (0x45 in view, the exact-landing receipt) prune
    on touch as they always have.

    Attributes:
        x: Mine X.
        y: Mine Y.
        mine_type: Wire mine-type byte from the reporter's registry.
        tank_id: Layer's tank id as the reporter recorded it.
        team: The mine's team (hostile to the reporting team).
        observed_ms: The reporter's belief timestamp for the mine.
    """

    x: int
    y: int
    mine_type: int
    tank_id: int
    team: int
    observed_ms: int


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
        room: The room id the reporter is playing in — receivers
            merge SAME-ROOM reports only (coordinates are per-field;
            a Desert sighting merged into a Practice belief set is
            poison, found 2026-08-26 when the first cross-room recon
            launched).
        tank_id: The reporter's own tank id, so receivers never merge
            a sighting of themselves or the reporter as an "enemy".
        role: The reporter's :data:`FleetRole`.
        x: The reporter's X at write time.
        y: The reporter's Y at write time.
        forage_goal_x: X of the reporter's latched forage-frontier
            goal block center (-1 for none). Siblings skip claimed
            blocks so a fleet divides the map instead of dogpiling
            one stale block (operator observation 2026-08-28: "no
            awareness of who's collecting what").
        forage_goal_y: Y of that goal (-1 for none).
        collect_claim_x: X of the reporter's held collect-plan
            container (-1 for none). Siblings treat claimed
            containers as taken -- the race otherwise resolves only
            when the winner's removal row lands, after the loser has
            already paid the travel.
        collect_claim_y: Y of that claim (-1 for none).
        combat_consent_ids: Tank ids whose combat-consent evidence
            this reporter holds (they chatted to it, or struck it).
            Operator ruling 2026-08-26: "if one has consent the other
            doesn't need it" — a human who engages one tank of our
            color has consented to fighting the COLOR, so the
            evidence rides the report like sightings do. Receivers
            still gate on ``is_human_name`` at the call sites, and
            fire-authorization still requires own-viewport
            confirmation.
        engaged_target_id: The reporter's held combat lock (-1 for
            none) — the focus-fire signal teammates' acquisition
            prefers.
        war_ready: True when the reporter clears the wartime
            readiness floor AND runs a war-joining doctrine
            (skirmish/swarm) — the swarm muster's quorum signal
            (operator order 2026-09-01): a swarm bot strikes first
            only when itself plus its war-ready siblings reach the
            quorum, so the fleet hits together instead of trickling.
        written_ms: Wall-clock ms of the write. Receivers drop reports
            older than the freshness TTL, so a dead bot's last file
            ages out instead of steering the living.
        enemies: Fresh enemy sightings.
        containers: Fresh container sightings.
        removed: Container tiles recently disproved or consumed --
            the negative knowledge that stops teammates chasing
            ghosts.
        mines: Fresh hostile-mine sightings -- the fleet mine map.
        scanned: Tiles under live radar coverage (within the forage
            coverage TTL) -- the shared scan map.
    """

    instance: str
    team: int
    room: str
    tank_id: int
    role: FleetRole
    x: int
    y: int
    engaged_target_id: int
    war_ready: bool
    forage_goal_x: int
    forage_goal_y: int
    collect_claim_x: int
    collect_claim_y: int
    combat_consent_ids: list[int]
    written_ms: int
    enemies: list[FleetEnemySightingDict]
    containers: list[FleetContainerSightingDict]
    removed: list[FleetContainerRemovalDict]
    mines: list[FleetMineSightingDict]
    scanned: list[FleetScannedTileDict]


__all__ = [
    "ENGAGEMENT_DOCTRINES",
    "EngagementDoctrine",
    "FLEET_ROLES",
    "FleetContainerRemovalDict",
    "FleetContainerSightingDict",
    "FleetEnemySightingDict",
    "FleetMineSightingDict",
    "FleetReportDict",
    "FleetRole",
    "FleetScannedTileDict",
]
