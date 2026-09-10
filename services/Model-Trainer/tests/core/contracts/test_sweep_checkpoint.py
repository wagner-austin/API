"""The sweep checkpoint contract, and the resumes it must refuse.

A sweep checkpoint is read once, by a process about to skip hours of work on
its authority, and every mistake it can make is silent: a resume against the
wrong corpus produces a complete table in which every number is plausible and
some cells were measured over text nobody is looking at. So most of what is
below is about what does NOT decode.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError
from platform_core.run_record import Observation

from model_trainer.core.contracts.sweep_checkpoint import (
    SWEEP_CHECKPOINT_SCHEMA_VERSION,
    CellRecord,
    SweepCheckpoint,
    completed_cells,
    decode_sweep_checkpoint,
    encode_cell_record,
    encode_sweep_checkpoint,
    sweep_checkpoint_mismatches,
    with_cell,
)

_MEASUREMENT = "cartridge-solo-seeds"
_DIGEST = "a1b2c3d4e5f6"


def _cell(name: str, *, gain: float = 0.25) -> CellRecord:
    """Build one completed cell.

    Args:
        name: What the unit is.
        gain: The value its observation carries.

    Returns:
        The record.
    """
    return CellRecord(
        cell=name,
        observations=[
            Observation(name=f"{name}_gain", value=gain),
            Observation(name=f"{name}_windows", value=8.0),
        ],
    )


def _checkpoint(*cells: CellRecord) -> SweepCheckpoint:
    """Build a checkpoint holding the given cells.

    Args:
        *cells: Completed cells.

    Returns:
        The checkpoint.
    """
    return SweepCheckpoint(
        schema_version=SWEEP_CHECKPOINT_SCHEMA_VERSION,
        measurement=_MEASUREMENT,
        inputs_digest=_DIGEST,
        cells=list(cells),
    )


class TestTheRoundTrip:
    def test_a_checkpoint_survives_encoding_and_decoding(self) -> None:
        original = _checkpoint(_cell("seed7"), _cell("seed8", gain=0.5))

        assert decode_sweep_checkpoint(encode_sweep_checkpoint(original)) == original

    def test_every_observation_survives_with_its_name_and_value(self) -> None:
        """A cell's observations ARE its contribution. Dropping one would let a
        resumed sweep report a table missing a column nobody notices is gone,
        because the cell it came from is marked done."""
        decoded = decode_sweep_checkpoint(encode_sweep_checkpoint(_checkpoint(_cell("seed7"))))

        observations = decoded["cells"][0]["observations"]
        assert [o["name"] for o in observations] == ["seed7_gain", "seed7_windows"]
        assert [o["value"] for o in observations] == [0.25, 8.0]

    def test_an_empty_checkpoint_is_valid(self) -> None:
        """A sweep evicted before its first cell finished has nothing to skip,
        which is a legitimate state and not a corrupt file."""
        assert decode_sweep_checkpoint(encode_sweep_checkpoint(_checkpoint()))["cells"] == []


class TestTheSchemaVersionIsCheckedInsideDecode:
    def test_a_different_version_is_refused(self) -> None:
        payload = encode_sweep_checkpoint(_checkpoint(_cell("seed7")))
        payload["schema_version"] = SWEEP_CHECKPOINT_SCHEMA_VERSION + 1

        with pytest.raises(JSONTypeError, match="cannot be assumed to mean the same thing"):
            decode_sweep_checkpoint(payload)

    def test_the_refusal_names_re_running_rather_than_repair(self) -> None:
        payload = encode_sweep_checkpoint(_checkpoint())
        payload["schema_version"] = 99

        with pytest.raises(JSONTypeError, match="Re-run rather than resume"):
            decode_sweep_checkpoint(payload)


class TestMalformedCells:
    def test_cells_that_are_not_a_list_are_refused(self) -> None:
        payload = encode_sweep_checkpoint(_checkpoint())
        payload["cells"] = "seed7"

        with pytest.raises(JSONTypeError):
            decode_sweep_checkpoint(payload)

    def test_a_malformed_cell_is_named_by_its_position(self) -> None:
        """A cell is anonymous until it decodes: its own name lives inside the
        object that would not, so the index is the only usable pointer."""
        payload: JSONObject = {
            **encode_sweep_checkpoint(_checkpoint()),
            "cells": [encode_cell_record(_cell("seed7")), "not an object"],
        }

        with pytest.raises(JSONTypeError, match=r"'cells\[1\]' must be an object"):
            decode_sweep_checkpoint(payload)

    def test_an_observation_without_a_name_is_refused(self) -> None:
        """The refusal comes from `decode_observation` in platform_core, which
        this contract reuses rather than re-implements. Asserted here so the
        reuse cannot be quietly replaced by a looser local check."""
        payload: JSONObject = {
            **encode_sweep_checkpoint(_checkpoint()),
            "cells": [{"cell": "seed7", "observations": [{"name": "", "value": 1.0}]}],
        }

        with pytest.raises(JSONTypeError, match="must say what was measured"):
            decode_sweep_checkpoint(payload)


class TestTheFingerprintRefusesADifferentMeasurement:
    def test_the_same_measurement_reports_no_mismatch(self) -> None:
        assert (
            sweep_checkpoint_mismatches(
                _checkpoint(_cell("seed7")),
                measurement=_MEASUREMENT,
                inputs_digest=_DIGEST,
            )
            == []
        )

    def test_a_different_corpus_is_named_with_both_digests(self) -> None:
        mismatches = sweep_checkpoint_mismatches(
            _checkpoint(_cell("seed7")), measurement=_MEASUREMENT, inputs_digest="ffffffff"
        )

        assert mismatches == [f"inputs_digest: checkpoint '{_DIGEST}' != current 'ffffffff'"]

    def test_both_fields_are_checked_at_once(self) -> None:
        """An operator who fixes the corpus and resubmits should not then
        discover the measurement name also disagreed."""
        mismatches = sweep_checkpoint_mismatches(
            _checkpoint(), measurement="something-else", inputs_digest="ffffffff"
        )

        assert [line.split(":")[0] for line in mismatches] == ["measurement", "inputs_digest"]


class TestAccumulating:
    def test_adding_a_cell_leaves_the_original_untouched(self) -> None:
        """The caller holds a checkpoint already written to disk. Mutating it
        in place lets the object and the file diverge between the append and
        the next save, and a failure in that window leaves a checkpoint
        claiming a cell whose file does not record it."""
        before = _checkpoint(_cell("seed7"))

        after = with_cell(before, _cell("seed8"))

        assert completed_cells(before) == frozenset({"seed7"})
        assert completed_cells(after) == frozenset({"seed7", "seed8"})

    def test_the_identity_travels_unchanged(self) -> None:
        after = with_cell(_checkpoint(), _cell("seed7"))

        assert after["measurement"] == _MEASUREMENT
        assert after["inputs_digest"] == _DIGEST
        assert after["schema_version"] == SWEEP_CHECKPOINT_SCHEMA_VERSION

    def test_an_empty_checkpoint_skips_nothing(self) -> None:
        assert completed_cells(_checkpoint()) == frozenset()
