"""The typed shapes of a container census, and their codecs.

A census (:mod:`tankpit_bot.validate.container_census`) reads every radar
scan the archive holds, works out which tiles each scan revealed, and
records which of them held a container. These are its records: one per
field, and one per held-out comparison of the fitted distribution
between two fields. The report is written to disk as JSON, so each
record has an encoder and a validating decoder.
"""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    narrow_json_to_dict,
    require_dict,
    require_float,
    require_int,
    require_list,
    require_str,
)


class TerrainCountsDict(TypedDict):
    """A count split by the tile's static terrain class.

    Attributes:
        ground: Tiles of ground (``.``).
        water: Tiles of water (``W``).
        rock: Tiles of rock (``#``).
    """

    ground: int
    water: int
    rock: int


class FieldCensusDict(TypedDict):
    """What the archive's radar scans saw of one field.

    Attributes:
        field: The field image (``field01.gif``).
        captures: Captures that played the field and scanned it.
        scans: Radar scans whose footprint could be computed.
        footprint_misses: Containers a scan listed outside its computed
            footprint; a check on the footprint law, near zero when it holds.
        tile_reads: Revealed tiles summed over every scan.
        container_reads: Containers listed summed over every scan, so
            ``container_reads / tile_reads`` is the density at scan time.
        observed: Distinct tiles some scan revealed, by terrain class.
        sites: Distinct observed tiles some scan found holding a container
            at least once, by terrain class.
        equipment_sites: Sites seen holding equipment at least once.
        fuel_sites: Sites seen holding fuel (any volume) at least once.
        block_dispersion: Pearson chi-square per degree of freedom of site
            counts over 16x16 blocks against a uniform rate on the block's
            observed non-rock tiles; a binomial field reads near one minus
            the rate, clustering reads above it.
    """

    field: str
    captures: int
    scans: int
    footprint_misses: int
    tile_reads: int
    container_reads: int
    observed: TerrainCountsDict
    sites: TerrainCountsDict
    equipment_sites: int
    fuel_sites: int
    block_dispersion: float


class TransferScoreDict(TypedDict):
    """The terrain-shaped distribution fitted on one field, scored on another.

    Attributes:
        train_field: The field the shape is fitted on.
        test_field: The held-out field it is scored on.
        water_factor: A water tile's site rate over a ground tile's, fitted
            on the train field; rock holds no site on either.
        train_density: Container density at scan time on the train field.
        test_density: The same on the test field.
        log_loss_constant: Mean log loss of a site at every observed test
            tile at the test field's own site rate, the one-parameter
            baseline.
        log_loss_shape: Mean log loss of the train field's terrain shape,
            with only its level fitted on the test field.
    """

    train_field: str
    test_field: str
    water_factor: float
    train_density: float
    test_density: float
    log_loss_constant: float
    log_loss_shape: float


class CensusReportDict(TypedDict):
    """A whole census.

    Attributes:
        fields: One record per field the archive scanned, by field name.
        transfers: One record per ordered pair of fields.
    """

    fields: list[FieldCensusDict]
    transfers: list[TransferScoreDict]


def _encode_counts(counts: TerrainCountsDict) -> JSONObject:
    """Encode a terrain split."""
    return {"ground": counts["ground"], "water": counts["water"], "rock": counts["rock"]}


def _decode_counts(obj: JSONObject, key: str) -> TerrainCountsDict:
    """Decode a terrain split held under ``key``.

    Raises:
        JSONTypeError: If the split is missing or a class is not an int.
    """
    counts = require_dict(obj, key)
    return TerrainCountsDict(
        ground=require_int(counts, "ground"),
        water=require_int(counts, "water"),
        rock=require_int(counts, "rock"),
    )


def encode_census(report: CensusReportDict) -> JSONObject:
    """Encode a census for writing.

    Args:
        report: The census.

    Returns:
        Its JSON object.
    """
    fields: list[JSONValue] = [
        {
            "field": f["field"],
            "captures": f["captures"],
            "scans": f["scans"],
            "footprint_misses": f["footprint_misses"],
            "tile_reads": f["tile_reads"],
            "container_reads": f["container_reads"],
            "observed": _encode_counts(f["observed"]),
            "sites": _encode_counts(f["sites"]),
            "equipment_sites": f["equipment_sites"],
            "fuel_sites": f["fuel_sites"],
            "block_dispersion": f["block_dispersion"],
        }
        for f in report["fields"]
    ]
    transfers: list[JSONValue] = [
        {
            "train_field": t["train_field"],
            "test_field": t["test_field"],
            "water_factor": t["water_factor"],
            "train_density": t["train_density"],
            "test_density": t["test_density"],
            "log_loss_constant": t["log_loss_constant"],
            "log_loss_shape": t["log_loss_shape"],
        }
        for t in report["transfers"]
    ]
    return {"fields": fields, "transfers": transfers}


def decode_census(data: JSONObject) -> CensusReportDict:
    """Decode and validate a written census.

    Args:
        data: The JSON object :func:`encode_census` wrote.

    Returns:
        The census.

    Raises:
        JSONTypeError: If any field is missing or of the wrong type.
    """
    fields: list[FieldCensusDict] = []
    for raw in require_list(data, "fields"):
        obj = narrow_json_to_dict(raw)
        fields.append(
            FieldCensusDict(
                field=require_str(obj, "field"),
                captures=require_int(obj, "captures"),
                scans=require_int(obj, "scans"),
                footprint_misses=require_int(obj, "footprint_misses"),
                tile_reads=require_int(obj, "tile_reads"),
                container_reads=require_int(obj, "container_reads"),
                observed=_decode_counts(obj, "observed"),
                sites=_decode_counts(obj, "sites"),
                equipment_sites=require_int(obj, "equipment_sites"),
                fuel_sites=require_int(obj, "fuel_sites"),
                block_dispersion=require_float(obj, "block_dispersion"),
            )
        )
    transfers: list[TransferScoreDict] = []
    for raw in require_list(data, "transfers"):
        obj = narrow_json_to_dict(raw)
        transfers.append(
            TransferScoreDict(
                train_field=require_str(obj, "train_field"),
                test_field=require_str(obj, "test_field"),
                water_factor=require_float(obj, "water_factor"),
                train_density=require_float(obj, "train_density"),
                test_density=require_float(obj, "test_density"),
                log_loss_constant=require_float(obj, "log_loss_constant"),
                log_loss_shape=require_float(obj, "log_loss_shape"),
            )
        )
    return CensusReportDict(fields=fields, transfers=transfers)


__all__ = [
    "CensusReportDict",
    "FieldCensusDict",
    "TerrainCountsDict",
    "TransferScoreDict",
    "decode_census",
    "encode_census",
]
