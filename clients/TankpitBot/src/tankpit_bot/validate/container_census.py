"""Where containers stand on each field, read from the archive's radar scans.

Every radar scan the client sent reveals a known set of tiles: the sim's
own footprint law (:func:`tankpit_bot.sim.actions.radar_covers`) applied
to the client's window, tile and rank as the archive last stated them
(:class:`~tankpit_bot.validate.conformance_mirror.ArchiveMirror`), with an
extra radar recognised by the 0x49 that precedes its 0x4F in the batch.
Each revealed tile is an observation, and each container the 0x4F lists
in it is a sighting. A listed container outside the computed footprint is
counted as a footprint miss, which is the check that the law is right.

From those observations a census states, per field, which tiles were
seen, which held a container at least once, the density at scan time,
and how the sites split by terrain class; and for each ordered pair of
fields it fits the terrain shape on one and scores it on the other. That
is the held-out test the multiplayer track's Phase 4 asked for (board
task ``b008ab91``): a fitted distribution, which one field cannot
distinguish from an algorithm, scored on a field it was not fitted on.
"""

from __future__ import annotations

import math
from pathlib import Path

from tankpit_bot import _test_hooks
from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol.types import BinaryMessage, RadarScanResultDict
from tankpit_bot.resources import field_gif_path
from tankpit_bot.sim.actions import VIEWPORT_RADIUS, radar_covers
from tankpit_bot.sim.commands import ClientCommandKind
from tankpit_bot.validate.conformance_mirror import EQUIPMENT_VOLUME, ArchiveMirror
from tankpit_bot.validate.conformance_wire import ReplayCapture, ReplayTick, read_replay
from tankpit_bot.validate.container_census_types import (
    CensusReportDict,
    FieldCensusDict,
    TerrainCountsDict,
    TransferScoreDict,
)

MAP_SPAN = 256
"""A field is 256 tiles on a side."""

SITE_BLOCK = 16
"""The block side the dispersion statistic counts sites in: one viewport."""


class CensusError(ValueError):
    """A held-out score the fitted shape cannot assign (``CENSUS_*`` codes)."""


class FieldTally:
    """One field's accumulating observations.

    Attributes:
        captures: Captures that contributed at least one scan.
        scans: Scans counted.
        footprint_misses: Listed containers outside the computed footprint.
        tile_reads: Revealed tiles summed over scans.
        container_reads: Listed containers inside the footprint, summed.
        observed: Every tile some scan revealed.
        equipment: Tiles seen holding equipment.
        fuel: Tiles seen holding fuel.
    """

    def __init__(self) -> None:
        self.captures = 0
        self.scans = 0
        self.footprint_misses = 0
        self.tile_reads = 0
        self.container_reads = 0
        self.observed: set[tuple[int, int]] = set()
        self.equipment: set[tuple[int, int]] = set()
        self.fuel: set[tuple[int, int]] = set()

    def sites(self) -> set[tuple[int, int]]:
        """Every tile seen holding a container at least once."""
        return self.equipment | self.fuel


def _scan_result(tick: ReplayTick) -> tuple[RadarScanResultDict, bool] | None:
    """The tick's radar answer and whether it spent an extra radar.

    The server reports an extra radar's consumption with an inventory
    snapshot ahead of the scan in the same batch.

    Args:
        tick: A tick the client sent only a radar into.

    Returns:
        The 0x4F and the extra-radar flag, or None when the batch holds no
        0x4F (a refused or unanswered scan).
    """
    spent = False
    message: BinaryMessage
    for message in tick.received:
        if message["msg_type"] == 0x49:
            spent = True
        elif message["msg_type"] == 0x4F:
            return message, spent
    return None


def _note_scan(tick: ReplayTick, mirror: ArchiveMirror, tally: FieldTally) -> bool:
    """Record one radar tick's observations, when its footprint is known.

    Args:
        tick: A tick the client sent only a radar into.
        mirror: The archive's statements before this tick.
        tally: The field's observations (mutated).

    Returns:
        True when the scan was counted.
    """
    window, tile, rank = mirror.window, mirror.tile, mirror.rank
    answer = _scan_result(tick)
    if window is None or tile is None or rank is None or answer is None:
        return False
    scan, spent = answer
    left, top = window
    revealed = {
        (x, y)
        for x in range(max(left, 0), min(left + 2 * VIEWPORT_RADIUS, MAP_SPAN))
        for y in range(max(top, 0), min(top + 2 * VIEWPORT_RADIUS, MAP_SPAN))
        if radar_covers(window, tile, rank, spent, (x, y))
    }
    tally.scans += 1
    tally.tile_reads += len(revealed)
    tally.observed |= revealed
    for container in scan["containers"]:
        site = (container["x"], container["y"])
        if site not in revealed:
            tally.footprint_misses += 1
            continue
        tally.container_reads += 1
        if container["volume"] == EQUIPMENT_VOLUME:
            tally.equipment.add(site)
        else:
            tally.fuel.add(site)
    return True


def tally_capture(capture: ReplayCapture, tally: FieldTally) -> None:
    """Add one capture's radar scans to its field's tally.

    Only ticks the client sent a lone radar into are read, so the batch's
    0x4F is unambiguously that scan's answer.

    Args:
        capture: The capture as ticks.
        tally: Its field's observations (mutated).
    """
    mirror = ArchiveMirror()
    counted = False
    for tick in capture.ticks:
        commanded = [
            c["kind"] for c in tick.commands if c["kind"] is not ClientCommandKind.KEEPALIVE
        ]
        if commanded == [ClientCommandKind.RADAR]:
            counted = _note_scan(tick, mirror, tally) or counted
        for message in tick.received:
            mirror.observe(message)
    tally.captures += int(counted)


def _by_terrain(
    tiles: set[tuple[int, int]], terrain: _test_hooks.TerrainMapProtocol
) -> TerrainCountsDict:
    """Split tiles by their static terrain class.

    Args:
        tiles: The tiles.
        terrain: The field.

    Returns:
        The split.
    """
    counts = TerrainCountsDict(ground=0, water=0, rock=0)
    for x, y in tiles:
        kind = terrain.get_terrain(x, y)
        if kind == terrain.WATER:
            counts["water"] += 1
        elif kind == terrain.ROCK:
            counts["rock"] += 1
        else:
            counts["ground"] += 1
    return counts


def block_dispersion(
    observed: set[tuple[int, int]],
    sites: set[tuple[int, int]],
    terrain: _test_hooks.TerrainMapProtocol,
) -> float:
    """Pearson chi-square per degree of freedom of sites over 16x16 blocks.

    Each block's expectation is the field's site rate times its observed
    non-rock tiles. Independent placement reads near one minus the rate;
    clustering at the block scale reads above it.

    Args:
        observed: Every observed tile.
        sites: Every observed site.
        terrain: The field.

    Returns:
        The statistic, or 0.0 when fewer than two blocks hold open tiles.
    """
    open_tiles: dict[tuple[int, int], int] = {}
    for x, y in observed:
        if terrain.get_terrain(x, y) != terrain.ROCK:
            block = (x // SITE_BLOCK, y // SITE_BLOCK)
            open_tiles[block] = open_tiles.get(block, 0) + 1
    if len(open_tiles) < 2 or not sites:
        return 0.0
    counts: dict[tuple[int, int], int] = {}
    for x, y in sites:
        block = (x // SITE_BLOCK, y // SITE_BLOCK)
        counts[block] = counts.get(block, 0) + 1
    rate = len(sites) / sum(open_tiles.values())
    chi = sum(
        (counts.get(block, 0) - rate * n) ** 2 / (rate * n) for block, n in open_tiles.items()
    )
    return chi / (len(open_tiles) - 1)


def census_field(
    field: str, tally: FieldTally, terrain: _test_hooks.TerrainMapProtocol
) -> FieldCensusDict:
    """Reduce one field's tally to its census record.

    Args:
        field: The field image.
        tally: Its observations.
        terrain: The field.

    Returns:
        The record.
    """
    sites = tally.sites()
    return FieldCensusDict(
        field=field,
        captures=tally.captures,
        scans=tally.scans,
        footprint_misses=tally.footprint_misses,
        tile_reads=tally.tile_reads,
        container_reads=tally.container_reads,
        observed=_by_terrain(tally.observed, terrain),
        sites=_by_terrain(sites, terrain),
        equipment_sites=len(tally.equipment),
        fuel_sites=len(tally.fuel),
        block_dispersion=block_dispersion(tally.observed, sites, terrain),
    )


def _bernoulli_loss(hits: int, trials: int, p: float) -> float:
    """Summed log loss of ``hits`` successes in ``trials`` at probability ``p``.

    Raises:
        CensusError: If ``p`` gives probability zero to an outcome that
            happened (``CENSUS_IMPOSSIBLE_SITE``).
    """
    if (p <= 0.0 and hits > 0) or (p >= 1.0 and hits < trials):
        raise CensusError(f"CENSUS_IMPOSSIBLE_SITE: {hits}/{trials} sites at probability {p}")
    loss = 0.0
    if hits:
        loss -= hits * math.log(p)
    if trials - hits:
        loss -= (trials - hits) * math.log(1.0 - p)
    return loss


def transfer_score(train: FieldCensusDict, test: FieldCensusDict) -> TransferScoreDict:
    """Fit the terrain shape on one field and score it on another.

    The shape: rock holds no site, and a water tile holds one at
    ``water_factor`` times a ground tile's rate, both read off the train
    field. Only the level is fitted on the test field, by its observed
    site mass, so the comparison with the test field's own constant rate
    (also one fitted parameter) isolates what the shape carries over.

    Args:
        train: The field the shape is fitted on (it must have observed
            sites on ground).
        test: The held-out field.

    Returns:
        Both losses, per observed test tile.

    Raises:
        CensusError: If the train field saw no ground site, so the shape
            is undefined (``CENSUS_NO_GROUND``), or the shape forbids a
            site the test field holds.
    """
    t_obs, t_sites = train["observed"], train["sites"]
    if t_sites["ground"] == 0:
        raise CensusError(f"CENSUS_NO_GROUND: {train['field']} saw no site on ground")
    ground_rate = t_sites["ground"] / t_obs["ground"]
    water_rate = t_sites["water"] / t_obs["water"] if t_obs["water"] else 0.0
    rock_rate = t_sites["rock"] / t_obs["rock"] if t_obs["rock"] else 0.0
    water_factor = water_rate / ground_rate
    rock_factor = rock_rate / ground_rate
    obs, sites = test["observed"], test["sites"]
    total_obs = obs["ground"] + obs["water"] + obs["rock"]
    total_sites = sites["ground"] + sites["water"] + sites["rock"]
    constant = _bernoulli_loss(total_sites, total_obs, total_sites / total_obs)
    mass = obs["ground"] + water_factor * obs["water"] + rock_factor * obs["rock"]
    level = total_sites / mass
    shape = (
        _bernoulli_loss(sites["ground"], obs["ground"], level)
        + _bernoulli_loss(sites["water"], obs["water"], level * water_factor)
        + _bernoulli_loss(sites["rock"], obs["rock"], level * rock_factor)
    )
    return TransferScoreDict(
        train_field=train["field"],
        test_field=test["field"],
        water_factor=water_factor,
        train_density=train["container_reads"] / train["tile_reads"],
        test_density=test["container_reads"] / test["tile_reads"],
        log_loss_constant=constant / total_obs,
        log_loss_shape=shape / total_obs,
    )


def run_census(paths: list[Path]) -> CensusReportDict:
    """Census every capture's field and score each field against each other.

    A capture with no magic, an unframed payload, no joined field or a
    field this distribution carries no minimap for contributes nothing.

    Args:
        paths: Capture files.

    Returns:
        The census: fields by name, then every ordered pair of fields
        whose train side saw a site on ground.
    """
    tallies: dict[str, tuple[Path, FieldTally]] = {}
    for path in paths:
        scanned = scan_session(path)
        if scanned["kind"] == "skipped":
            continue
        capture = read_replay(scanned["frames"])
        if capture.field_image is None:
            continue
        gif = field_gif_path(capture.field_image)
        if gif is None:
            continue
        _, tally = tallies.setdefault(capture.field_image, (gif, FieldTally()))
        tally_capture(capture, tally)
    fields = [
        census_field(field, tally, _test_hooks.load_terrain_map(gif))
        for field, (gif, tally) in sorted(tallies.items())
        if tally.scans
    ]
    transfers = [
        transfer_score(train, test)
        for train in fields
        for test in fields
        if train is not test and train["sites"]["ground"]
    ]
    return CensusReportDict(fields=fields, transfers=transfers)


__all__ = [
    "MAP_SPAN",
    "SITE_BLOCK",
    "CensusError",
    "FieldTally",
    "block_dispersion",
    "census_field",
    "run_census",
    "tally_capture",
    "transfer_score",
]
