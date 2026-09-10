"""What the solo-seeds sweep does when a run is cut in half.

SPLIT FROM ``test_cartridge_solo_seeds`` by role, when the eviction tests took
that suite over the 600-line ceiling. That one asserts what the measurement
MEASURES -- one gain row per seed, the summary derived from exactly those
rows, the recorded knobs threading through. This one asserts what survives an
eviction, which is a different question about the same code.

EVERY TEST HERE RUNS THE REAL PAYLOAD: real tiny GPT-2, real cartridge
training, real scoring, real files in ``tmp_path``. The checkpoint's whole
value is a claim about what happens to REAL work when a process stops between
cells, and a faked payload would assert that the fake stops where the fake was
told to.

THE HARD PART IS MAKING A SKIP OBSERVABLE. A resume that re-trains a cell and
then discards the result returns exactly what a straight run returns, costs
exactly as much, and exits zero -- equality cannot tell the two apart. So the
first test below plants a value seed 7 could not possibly measure and asserts
it comes back out: the only path by which that number reaches the observations
is the checkpoint being believed and the training being skipped.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.run_record import Observation

from model_trainer.cli import cartridge_solo_seeds as solo
from model_trainer.core.contracts.sweep_checkpoint import (
    SWEEP_CHECKPOINT_SCHEMA_VERSION,
    CellRecord,
    SweepCheckpoint,
)
from model_trainer.core.services.model.cartridge_plans import corpus_digest
from model_trainer.core.services.model.cartridge_sweep_checkpoint import (
    checkpoint_exists,
    save_sweep_checkpoint,
)
from tests._solo_seeds_harness import documents, staged, wired

#: Imported for the autouse fixture's side effect; pytest collects it from
#: this module's namespace, and naming it here is what makes that explicit
#: rather than a star-import accident.
__all__ = ["wired"]

#: What the CLI calls this measurement, derived the same way it derives it.
_MEASUREMENT = "solo-seeds-gpt2"


def _planted(directory: pathlib.Path, *, gain: float, marker: str = "a") -> None:
    """Write a checkpoint claiming seed 7 was already measured.

    Args:
        directory: Where the checkpoint file goes.
        gain: The value to record for seed 7.
        marker: Which corpus the checkpoint claims to be over; the default is
            the one the tests measure, so the resume is accepted.
    """
    save_sweep_checkpoint(
        directory,
        SweepCheckpoint(
            schema_version=SWEEP_CHECKPOINT_SCHEMA_VERSION,
            measurement=_MEASUREMENT,
            inputs_digest=corpus_digest(documents(marker)),
            cells=[
                CellRecord(
                    cell="seed7",
                    observations=[Observation(name="solo-gpt2-seed7_gain", value=gain)],
                )
            ],
        ),
    )


class TestResumingARealSweep:
    def test_a_recorded_seed_is_read_from_the_checkpoint_not_retrained(
        self, tmp_path: pathlib.Path
    ) -> None:
        """THE SKIP, MADE OBSERVABLE ON THE REAL PAYLOAD.

        ``-12345.0`` is not a gain any cartridge produces, so its presence in
        the output is proof the checkpoint was believed rather than
        re-measured. Asserting equality with a straight run instead would
        pass just as happily against a resume that redid all the work.
        """
        alpha = staged(tmp_path, "alpha")
        checkpoints = tmp_path / "ckpt"
        planted = -12345.0
        _planted(checkpoints, gain=planted)

        observations, _digest = solo.measure_solo_seeds(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7, 8),
            device="cpu",
            checkpoints=checkpoints,
        )

        recorded = {o["name"]: o["value"] for o in observations}
        assert recorded["solo-gpt2-seed7_gain"] == planted
        # AND THE REDUCTION USED IT, so a resumed run's mean and spread are
        # over the draws the record names rather than over the cells that
        # happened to run this time.
        assert recorded["solo-gpt2_gain_mean"] == (planted + recorded["solo-gpt2-seed8_gain"]) / 2

    def test_a_resumed_run_reproduces_the_run_it_resumes(self, tmp_path: pathlib.Path) -> None:
        """THE PROPERTY THE WHOLE MECHANISM IS WORTHLESS WITHOUT.

        An eviction stops a process between cells; finishing the work later
        must yield exactly what finishing it in one go would have. Otherwise
        resuming quietly changes the numbers and no record says which kind of
        run produced them.

        Built from a straight run's OWN seed-7 row, so the second run trains
        only seed 8 and every row must still match.
        """
        alpha = staged(tmp_path, "alpha")
        straight, _digest = solo.measure_solo_seeds(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7, 8),
            device="cpu",
            checkpoints=tmp_path / "ckpt-straight",
        )
        by_name = {o["name"]: o["value"] for o in straight}

        resumed_from = tmp_path / "ckpt-resumed"
        _planted(resumed_from, gain=by_name["solo-gpt2-seed7_gain"])

        resumed, _resumed_digest = solo.measure_solo_seeds(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7, 8),
            device="cpu",
            checkpoints=resumed_from,
        )

        assert resumed == straight

    def test_a_completed_run_leaves_no_checkpoint_behind(self, tmp_path: pathlib.Path) -> None:
        """A leftover file is indistinguishable from an interrupted run, so
        the NEXT submission of the same measurement would skip seeds it
        should have retrained and report an earlier execution's numbers as
        its own."""
        alpha = staged(tmp_path, "alpha")
        checkpoints = tmp_path / "ckpt"

        solo.measure_solo_seeds(
            alpha,
            model_id="gpt2",
            load_precision=None,
            seeds=(7, 8),
            device="cpu",
            checkpoints=checkpoints,
        )

        assert checkpoint_exists(checkpoints, _MEASUREMENT) is False

    def test_a_checkpoint_over_another_corpus_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Resuming across it would produce a complete table in which some
        seeds were trained on text nobody is looking at -- every number
        plausible, exit zero, nothing downstream able to see it."""
        alpha = staged(tmp_path, "alpha")
        checkpoints = tmp_path / "ckpt"
        _planted(checkpoints, gain=1.0, marker="z")

        with pytest.raises(AppError) as excinfo:
            solo.measure_solo_seeds(
                alpha,
                model_id="gpt2",
                load_precision=None,
                seeds=(7, 8),
                device="cpu",
                checkpoints=checkpoints,
            )

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN
        assert "measured over other inputs" in excinfo.value.message
