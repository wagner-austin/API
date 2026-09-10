"""Persisting a sweep's completed cells, and skipping the work they cost.

REAL FILES IN ``tmp_path``. The property this module exists for is that a
save is atomic, and that property lives in the operating system rather than
in this code -- a test against a fake would assert that the fake behaves as
the author imagined.

AND THE SKIP IS ASSERTED BY COUNTING WORK, not by reading a return value. A
resume that re-runs the cell and then discards the result costs exactly as
much as no resume and looks identical from outside; the only thing that tells
them apart is whether the expensive callable ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str
from platform_core.run_record import Observation

from model_trainer.core.contracts.sweep_checkpoint import (
    SWEEP_CHECKPOINT_SCHEMA_VERSION,
    CellRecord,
    SweepCheckpoint,
    completed_cells,
    encode_sweep_checkpoint,
)
from model_trainer.core.services.model.cartridge_sweep_checkpoint import (
    bind_cells,
    cell_or_resume,
    checkpoint_exists,
    checkpoint_path,
    checkpointed_cells,
    delete_sweep_checkpoint,
    load_sweep_checkpoint,
    require_distinct_cells,
    resume_or_start,
    save_sweep_checkpoint,
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
    return CellRecord(cell=name, observations=[Observation(name=f"{name}_gain", value=gain)])


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


class TestSavingAndLoading:
    def test_a_saved_checkpoint_loads_back_identically(self, tmp_path: Path) -> None:
        original = _checkpoint(_cell("seed7"), _cell("seed8"))

        save_sweep_checkpoint(tmp_path, original)

        assert load_sweep_checkpoint(tmp_path, _MEASUREMENT) == original

    def test_saving_creates_the_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "runs" / "checkpoints"

        save_sweep_checkpoint(nested, _checkpoint(_cell("seed7")))

        assert checkpoint_exists(nested, _MEASUREMENT)

    def test_no_temporary_file_survives_a_save(self, tmp_path: Path) -> None:
        """A leftover .pending means the rename did not happen, so the
        published file is the OLD one -- which is the failure the atomic
        write exists to prevent."""
        save_sweep_checkpoint(tmp_path, _checkpoint(_cell("seed7")))

        assert list(tmp_path.glob("*.pending")) == []

    def test_the_published_file_is_a_sibling_of_the_temporary(self, tmp_path: Path) -> None:
        """NOT a system temporary directory. ``os.replace`` is atomic only
        within a filesystem, and on the cluster the checkpoint lives on /pub
        scratch while /tmp is routinely another device, where a cross-device
        rename degrades to a copy."""
        published = save_sweep_checkpoint(tmp_path, _checkpoint(_cell("seed7")))

        assert published.parent == tmp_path
        assert published == checkpoint_path(tmp_path, _MEASUREMENT)

    def test_a_second_save_replaces_the_first(self, tmp_path: Path) -> None:
        save_sweep_checkpoint(tmp_path, _checkpoint(_cell("seed7")))

        save_sweep_checkpoint(tmp_path, _checkpoint(_cell("seed7"), _cell("seed8")))

        assert len(load_sweep_checkpoint(tmp_path, _MEASUREMENT)["cells"]) == 2
        assert len(list(tmp_path.glob("sweep-checkpoint-*"))) == 1


class TestRefusals:
    def test_a_file_holding_a_json_array_is_refused(self, tmp_path: Path) -> None:
        checkpoint_path(tmp_path, _MEASUREMENT).write_text("[]", encoding="utf-8")

        with pytest.raises(TypeError, match="does not hold a JSON object"):
            load_sweep_checkpoint(tmp_path, _MEASUREMENT)

    def test_a_checkpoint_from_another_schema_version_is_refused(self, tmp_path: Path) -> None:
        payload = encode_sweep_checkpoint(_checkpoint(_cell("seed7")))
        payload["schema_version"] = SWEEP_CHECKPOINT_SCHEMA_VERSION + 98
        checkpoint_path(tmp_path, _MEASUREMENT).write_text(dump_json_str(payload), encoding="utf-8")

        with pytest.raises(JSONTypeError, match="Re-run rather than resume"):
            load_sweep_checkpoint(tmp_path, _MEASUREMENT)

    def test_loading_an_absent_checkpoint_raises_rather_than_inventing_one(
        self, tmp_path: Path
    ) -> None:
        """No empty-checkpoint fallback: a first run and a lost checkpoint
        would then look identical to a caller that never asked."""
        with pytest.raises(OSError):
            load_sweep_checkpoint(tmp_path, _MEASUREMENT)


class TestExistenceAndDeletion:
    def test_a_directory_with_no_checkpoint_reports_none(self, tmp_path: Path) -> None:
        assert checkpoint_exists(tmp_path, _MEASUREMENT) is False

    def test_a_directory_named_like_the_file_is_not_a_checkpoint(self, tmp_path: Path) -> None:
        checkpoint_path(tmp_path, _MEASUREMENT).mkdir(parents=True)

        assert checkpoint_exists(tmp_path, _MEASUREMENT) is False

    def test_a_completed_sweep_deletes_its_checkpoint(self, tmp_path: Path) -> None:
        save_sweep_checkpoint(tmp_path, _checkpoint(_cell("seed7")))

        delete_sweep_checkpoint(tmp_path, _MEASUREMENT)

        assert checkpoint_exists(tmp_path, _MEASUREMENT) is False

    def test_deleting_an_absent_checkpoint_is_not_an_error(self, tmp_path: Path) -> None:
        delete_sweep_checkpoint(tmp_path, _MEASUREMENT)

        assert checkpoint_exists(tmp_path, _MEASUREMENT) is False

    def test_two_measurements_do_not_share_a_file(self, tmp_path: Path) -> None:
        assert checkpoint_path(tmp_path, "sweep-a") != checkpoint_path(tmp_path, "sweep-b")


class TestResumeOrStart:
    def test_no_file_starts_an_empty_sweep(self, tmp_path: Path) -> None:
        resumed = resume_or_start(tmp_path, measurement=_MEASUREMENT, inputs_digest=_DIGEST)

        assert resumed["cells"] == []
        assert resumed["measurement"] == _MEASUREMENT

    def test_a_matching_checkpoint_is_resumed(self, tmp_path: Path) -> None:
        save_sweep_checkpoint(tmp_path, _checkpoint(_cell("seed7"), _cell("seed8")))

        resumed = resume_or_start(tmp_path, measurement=_MEASUREMENT, inputs_digest=_DIGEST)

        assert completed_cells(resumed) == frozenset({"seed7", "seed8"})

    def test_a_foreign_checkpoint_is_refused_not_discarded(self, tmp_path: Path) -> None:
        """Resuming across a checkpoint from other inputs produces a COMPLETE
        table in which some cells were measured over text nobody is looking
        at. Every number plausible, exit zero, nothing downstream able to see
        it -- which is strictly worse than a crash."""
        save_sweep_checkpoint(tmp_path, _checkpoint(_cell("seed7")))

        with pytest.raises(AppError) as excinfo:
            resume_or_start(tmp_path, measurement=_MEASUREMENT, inputs_digest="ffffffff")

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN
        assert "measured over other inputs" in excinfo.value.message

    def test_the_refusal_names_the_field_the_file_and_the_way_out(self, tmp_path: Path) -> None:
        """Refusing without naming the field sends the operator to delete the
        checkpoint, which spends exactly the hours it existed to save."""
        save_sweep_checkpoint(tmp_path, _checkpoint(_cell("seed7")))

        with pytest.raises(AppError) as excinfo:
            resume_or_start(tmp_path, measurement=_MEASUREMENT, inputs_digest="ffffffff")

        assert f"inputs_digest: checkpoint '{_DIGEST}' != current 'ffffffff'" in (
            excinfo.value.message
        )
        assert str(checkpoint_path(tmp_path, _MEASUREMENT)) in excinfo.value.message
        assert "delete the checkpoint" in excinfo.value.message


class TestCellOrResume:
    """THE SEAM EVERY SWEEP DRIVES, so the skip is asserted by counting work."""

    def test_an_unmeasured_cell_runs_and_is_recorded(self, tmp_path: Path) -> None:
        ran: list[str] = []

        def _measure() -> tuple[Observation, ...]:
            """Measure the cell, recording that it ran.

            Returns:
                One observation.
            """
            ran.append("seed7")
            return (Observation(name="seed7_gain", value=0.5),)

        checkpoint, produced = cell_or_resume(_checkpoint(), tmp_path, "seed7", _measure)

        assert ran == ["seed7"]
        assert produced == (Observation(name="seed7_gain", value=0.5),)
        assert completed_cells(checkpoint) == frozenset({"seed7"})
        assert checkpoint_exists(tmp_path, _MEASUREMENT), "the cell was not saved immediately"

    def test_a_recorded_cell_does_not_run_at_all(self, tmp_path: Path) -> None:
        """THE ASSERTION THE WHOLE MECHANISM IS FOR. A skip that re-runs the
        work and discards it is indistinguishable from no skip except by
        whether the callable executed, so that is what is asserted."""

        def _refuse() -> tuple[Observation, ...]:
            """Fail if the cell is measured again.

            Returns:
                Never; always raises.

            Raises:
                AssertionError: Always.
            """
            raise AssertionError("a checkpointed cell was measured again")

        checkpoint, produced = cell_or_resume(
            _checkpoint(_cell("seed7", gain=0.75)), tmp_path, "seed7", _refuse
        )

        assert produced == (Observation(name="seed7_gain", value=0.75),)
        assert completed_cells(checkpoint) == frozenset({"seed7"})

    def test_the_saved_file_holds_the_cell_before_the_next_one_starts(self, tmp_path: Path) -> None:
        """Saved immediately after measuring, not at the end of the sweep: the
        window between finishing a cell and recording it is exactly the work
        an eviction can still take."""

        def _first() -> tuple[Observation, ...]:
            """Measure the first cell.

            Returns:
                One observation.
            """
            return (Observation(name="a_gain", value=1.0),)

        checkpoint, _produced = cell_or_resume(_checkpoint(), tmp_path, "a", _first)

        assert completed_cells(load_sweep_checkpoint(tmp_path, _MEASUREMENT)) == frozenset({"a"})
        assert completed_cells(checkpoint) == frozenset({"a"})


def _measure_unit(unit: int) -> tuple[Observation, ...]:
    """Measure one integer unit, naming its observation after it.

    The value is derived from the unit, so a cell that measured the WRONG
    unit is visible in the number as well as the name.

    Args:
        unit: What this cell measures.

    Returns:
        One observation.
    """
    return (Observation(name=f"unit{unit}_gain", value=float(unit) / 10.0),)


class TestBindCells:
    """THE CLOSURE BUG, PINNED, because nothing else would catch it."""

    def test_each_cell_measures_its_own_unit(self) -> None:
        """A ``lambda: measure(unit)`` built in a loop captures the VARIABLE,
        so every cell would measure the LAST unit -- producing one number
        under N distinct labels, which is complete, plausible and wrong.
        Asserted by running the cells AFTER the whole list is built, which is
        when the broken version collapses."""
        cells = bind_cells([(f"unit{unit}", unit) for unit in (7, 8, 9)], _measure_unit)

        produced = [measure() for _name, measure in cells]

        assert produced == [
            (Observation(name="unit7_gain", value=0.7),),
            (Observation(name="unit8_gain", value=0.8),),
            (Observation(name="unit9_gain", value=0.9),),
        ]

    def test_the_names_travel_in_the_order_given(self) -> None:
        cells = bind_cells([(f"unit{unit}", unit) for unit in (9, 7, 8)], _measure_unit)

        assert [name for name, _measure in cells] == ["unit9", "unit7", "unit8"]

    def test_nothing_is_measured_until_a_cell_is_called(self) -> None:
        """The whole skip depends on this: a cell already in the checkpoint
        must cost nothing, and it only costs nothing if building the list did
        not already run it."""
        ran: list[int] = []

        def _recording(unit: int) -> tuple[Observation, ...]:
            """Measure a unit, recording that it ran.

            Args:
                unit: What this cell measures.

            Returns:
                One observation.
            """
            ran.append(unit)
            return _measure_unit(unit)

        bind_cells([(f"unit{unit}", unit) for unit in (7, 8)], _recording)

        assert ran == []


def _never_runs() -> tuple[Observation, ...]:
    """Fail if a cell runs when the sweep should have been refused.

    Returns:
        Never; always raises.

    Raises:
        AssertionError: Always.
    """
    raise AssertionError("a cell ran despite the sweep being refused")


class TestRequireDistinctCells:
    def test_distinct_names_are_accepted(self) -> None:
        require_distinct_cells(bind_cells([("a", 1), ("b", 2)], _measure_unit))

    def test_a_repeated_name_is_refused(self) -> None:
        """A resume matches by NAME, so the second cell would inherit the
        first's observations and never run -- reporting one cell's numbers
        under two labels. Invisible on a first run, wrong after an eviction."""
        with pytest.raises(AppError) as excinfo:
            require_distinct_cells([("a", _never_runs), ("a", _never_runs)])

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_DUPLICATE_CELL
        assert "'a'" in excinfo.value.message or "a are used" in excinfo.value.message

    def test_a_name_used_three_times_is_named_once(self) -> None:
        """Otherwise the message repeats the same name for every extra use,
        and an operator reading it counts collisions that do not exist."""
        with pytest.raises(AppError) as excinfo:
            require_distinct_cells([("a", _never_runs), ("a", _never_runs), ("a", _never_runs)])

        assert excinfo.value.message.count("a are used") == 1

    def test_every_duplicate_is_named_not_only_the_first(self) -> None:
        with pytest.raises(AppError) as excinfo:
            require_distinct_cells(
                [("a", _never_runs), ("b", _never_runs), ("a", _never_runs), ("b", _never_runs)]
            )

        assert "a, b" in excinfo.value.message


class TestCheckpointedCells:
    """The driver every sweep calls, so the discipline is tested once."""

    def test_it_runs_every_cell_and_keys_the_results_by_name(self, tmp_path: Path) -> None:
        produced = checkpointed_cells(
            tmp_path,
            measurement=_MEASUREMENT,
            inputs_digest=_DIGEST,
            cells=bind_cells([(f"unit{unit}", unit) for unit in (7, 8)], _measure_unit),
        )

        assert dict(produced) == {
            "unit7": (Observation(name="unit7_gain", value=0.7),),
            "unit8": (Observation(name="unit8_gain", value=0.8),),
        }

    def test_the_keys_keep_the_order_the_cells_were_given(self, tmp_path: Path) -> None:
        """Callers that want their rows flat take ``.values()``, and the
        record's row order is the order the cells were declared in."""
        produced = checkpointed_cells(
            tmp_path,
            measurement=_MEASUREMENT,
            inputs_digest=_DIGEST,
            cells=bind_cells([(f"unit{unit}", unit) for unit in (9, 7, 8)], _measure_unit),
        )

        assert list(produced) == ["unit9", "unit7", "unit8"]

    def test_a_completed_sweep_deletes_its_own_checkpoint(self, tmp_path: Path) -> None:
        """A leftover file is indistinguishable from an interrupted run, so
        the NEXT submission would skip cells it should have re-measured and
        report an earlier execution's numbers as its own."""
        checkpointed_cells(
            tmp_path,
            measurement=_MEASUREMENT,
            inputs_digest=_DIGEST,
            cells=bind_cells([("unit7", 7)], _measure_unit),
        )

        assert checkpoint_exists(tmp_path, _MEASUREMENT) is False

    def test_the_checkpoint_survives_a_cell_that_raises(self, tmp_path: Path) -> None:
        """THE WHOLE POINT, in miniature. An eviction is a process that stops
        mid-cell; what must survive is every cell that finished before it. If
        the delete ran on the way out, or the saves waited until the end,
        this file would be absent and the resubmission would start at zero."""
        ran: list[int] = []

        def _second_fails(unit: int) -> tuple[Observation, ...]:
            """Measure the first unit and die on the second.

            Args:
                unit: What this cell measures.

            Returns:
                The first unit's observation.

            Raises:
                RuntimeError: On any unit after the first.
            """
            ran.append(unit)
            if len(ran) > 1:
                raise RuntimeError("evicted")
            return _measure_unit(unit)

        with pytest.raises(RuntimeError, match="evicted"):
            checkpointed_cells(
                tmp_path,
                measurement=_MEASUREMENT,
                inputs_digest=_DIGEST,
                cells=bind_cells([("unit7", 7), ("unit8", 8)], _second_fails),
            )

        assert completed_cells(load_sweep_checkpoint(tmp_path, _MEASUREMENT)) == frozenset(
            {"unit7"}
        )

    def test_a_resumed_sweep_reruns_only_what_was_not_finished(self, tmp_path: Path) -> None:
        """ASSERTED BY COUNTING WORK. A resume that re-runs a cell and then
        discards the result costs exactly as much as no resume and returns
        the same mapping; only whether the callable ran tells them apart."""
        save_sweep_checkpoint(
            tmp_path,
            SweepCheckpoint(
                schema_version=SWEEP_CHECKPOINT_SCHEMA_VERSION,
                measurement=_MEASUREMENT,
                inputs_digest=_DIGEST,
                cells=[CellRecord(cell="unit7", observations=list(_measure_unit(7)))],
            ),
        )
        ran: list[int] = []

        def _recording(unit: int) -> tuple[Observation, ...]:
            """Measure a unit, recording that it ran.

            Args:
                unit: What this cell measures.

            Returns:
                One observation.
            """
            ran.append(unit)
            return _measure_unit(unit)

        produced = checkpointed_cells(
            tmp_path,
            measurement=_MEASUREMENT,
            inputs_digest=_DIGEST,
            cells=bind_cells([("unit7", 7), ("unit8", 8)], _recording),
        )

        assert ran == [8], "the checkpointed cell was measured again"
        assert dict(produced) == {
            "unit7": (Observation(name="unit7_gain", value=0.7),),
            "unit8": (Observation(name="unit8_gain", value=0.8),),
        }

    def test_colliding_cells_are_refused_before_any_work_runs(self, tmp_path: Path) -> None:
        """Failing in seconds beats reporting one cell's numbers under two
        labels after forty hours, so the check is not deferred to the loop."""
        with pytest.raises(AppError) as excinfo:
            checkpointed_cells(
                tmp_path,
                measurement=_MEASUREMENT,
                inputs_digest=_DIGEST,
                cells=[("a", _never_runs), ("a", _never_runs)],
            )

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_DUPLICATE_CELL
        assert checkpoint_exists(tmp_path, _MEASUREMENT) is False

    def test_a_foreign_checkpoint_is_refused_before_any_work_runs(self, tmp_path: Path) -> None:
        save_sweep_checkpoint(tmp_path, _checkpoint(_cell("unit7")))

        with pytest.raises(AppError) as excinfo:
            checkpointed_cells(
                tmp_path,
                measurement=_MEASUREMENT,
                inputs_digest="ffffffff",
                cells=[("unit7", _never_runs)],
            )

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN
