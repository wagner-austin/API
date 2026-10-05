"""The container census report survives a write and a read, and bad records are refused."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from tankpit_bot.validate.container_census_types import (
    CensusReportDict,
    FieldCensusDict,
    TerrainCountsDict,
    TransferScoreDict,
    decode_census,
    encode_census,
)


def _report() -> CensusReportDict:
    """A census holding one field and one transfer."""
    return CensusReportDict(
        fields=[
            FieldCensusDict(
                field="field05.gif",
                captures=19,
                scans=2963,
                footprint_misses=60,
                tile_reads=609978,
                container_reads=4880,
                observed=TerrainCountsDict(ground=51628, water=4071, rock=9307),
                sites=TerrainCountsDict(ground=4212, water=266, rock=0),
                equipment_sites=2725,
                fuel_sites=1753,
                block_dispersion=1.474,
            )
        ],
        transfers=[
            TransferScoreDict(
                train_field="field01.gif",
                test_field="field05.gif",
                water_factor=0.794,
                train_density=0.0066,
                test_density=0.008,
                log_loss_constant=0.25075,
                log_loss_shape=0.23958,
            )
        ],
    )


def test_the_census_round_trips_through_json() -> None:
    """Every field written is read back identical."""
    text = dump_json_str(encode_census(_report()))
    assert decode_census(narrow_json_to_dict(load_json_str(text))) == _report()


def test_a_terrain_split_missing_a_class_is_refused() -> None:
    """A field whose site split lacks rock does not decode."""
    data: JSONObject = {
        "fields": [
            {
                "field": "f",
                "captures": 1,
                "scans": 1,
                "footprint_misses": 0,
                "tile_reads": 1,
                "container_reads": 0,
                "observed": {"ground": 1, "water": 0, "rock": 0},
                "sites": {"ground": 0, "water": 0},
                "equipment_sites": 0,
                "fuel_sites": 0,
                "block_dispersion": 0.0,
            }
        ],
        "transfers": [],
    }
    with pytest.raises(JSONTypeError):
        decode_census(data)


def test_a_transfer_with_a_string_loss_is_refused() -> None:
    """Losses are numbers."""
    data: JSONObject = {
        "fields": [],
        "transfers": [
            {
                "train_field": "a",
                "test_field": "b",
                "water_factor": 0.8,
                "train_density": 0.1,
                "test_density": 0.1,
                "log_loss_constant": "low",
                "log_loss_shape": 0.2,
            }
        ],
    }
    with pytest.raises(JSONTypeError):
        decode_census(data)
