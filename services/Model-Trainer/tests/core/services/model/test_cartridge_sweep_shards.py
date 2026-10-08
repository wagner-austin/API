"""Seed blocks, shards and merges, against real checkpoint files.

REAL FILES IN ``tmp_path``, for the reason the checkpoint suite gives. The
cells are trivial and COUNTED, because the property a merge exists for is that
it measures nothing -- every cell resumes -- and the only evidence of that is
the work that did not run.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.run_record import Observation

from model_trainer.core.contracts.sweep_checkpoint import (
    SWEEP_CHECKPOINT_SCHEMA_VERSION,
    CellRecord,
    SweepCheckpoint,
)
from model_trainer.core.services.model.cartridge_sweep_checkpoint import (
    bind_cells,
    checkpoint_exists,
    load_sweep_checkpoint,
    save_sweep_checkpoint,
)
from model_trainer.core.services.model.cartridge_sweep_shards import (
    SEED_BLOCK_SIZE,
    SweepMerge,
    SweepShard,
    adopt_shards,
    assembled_cells,
    block_cell,
    gather_blocks,
    measure_shard,
    owned_cells,
    seed_blocks,
    shard_directory,
)

_MEASUREMENT = "trait-tiny-composed"
_DIGEST = "d1g3st"


class _Counter:
    """Measures a unit by naming it, and keeps every unit it measured."""

    def __init__(self) -> None:
        """Start empty."""
        self.measured: list[int] = []

    def measure(self, unit: int) -> tuple[Observation, ...]:
        """Measure one unit.

        Args:
            unit: The unit.

        Returns:
            One row naming it.
        """
        self.measured.append(unit)
        return (Observation(name=f"u_seed{unit}_gain", value=float(unit)),)


def _cells(counter: _Counter, count: int) -> list[tuple[str, Callable[[], Sequence[Observation]]]]:
    """Bind ``count`` units, one per block, to the counter.

    Args:
        counter: Records which units ran.
        count: How many cells.

    Returns:
        The cells, named as block cells of ``"u"``.
    """
    return bind_cells([(block_cell("u", index), index) for index in range(count)], counter.measure)


def _shard(root: Path, index: int, count: int) -> SweepShard:
    """Name one shard.

    Args:
        root: The sweep's shard root.
        index: The shard.
        count: How many shards.

    Returns:
        The shard.
    """
    return SweepShard(root=root, count=count, index=index)


class TestSeedBlocks:
    """The unit every sharded cell is cut to."""

    def test_seeds_are_cut_into_consecutive_blocks(self) -> None:
        """Order kept, every block the minimum replicate count."""
        assert seed_blocks((7, 8, 9, 10, 11, 12)) == ((7, 8, 9), (10, 11, 12))
        assert SEED_BLOCK_SIZE == 3

    @pytest.mark.parametrize("seeds", [(), (7, 8, 9, 10)])
    def test_a_ragged_or_empty_seed_set_is_refused(self, seeds: tuple[int, ...]) -> None:
        """A last block of one seed is a replicate set nothing accepts.

        Args:
            seeds: The seeds to cut.
        """
        with pytest.raises(AppError) as excinfo:
            seed_blocks(seeds)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_SEED_BLOCKS_UNEVEN
        assert f"{len(seeds)} seed(s)" in excinfo.value.message

    def test_gathering_concatenates_every_block_in_order(self) -> None:
        """The rows a rebuild reads, block by block."""
        produced = {
            "n2@b0": (Observation(name="a", value=1.0),),
            "n2@b1": (Observation(name="b", value=2.0),),
            "n4@b0": (Observation(name="c", value=3.0),),
        }
        rows = gather_blocks(produced, "n2", blocks=2)
        assert [row["name"] for row in rows] == ["a", "b"]


class TestOwnership:
    """A partition of the cell list, by position."""

    def test_the_shards_partition_the_cells(self) -> None:
        """Every cell to exactly one shard, every count-th from the index."""
        cells = _cells(_Counter(), 5)
        owned = [
            [name for name, _measure in owned_cells(cells, _shard(Path("r"), index, 2))]
            for index in range(2)
        ]
        assert owned == [["u@b0", "u@b2", "u@b4"], ["u@b1", "u@b3"]]

    def test_each_shard_has_its_own_directory_per_cut(self) -> None:
        """Two cuts of one sweep must never share a directory."""
        root = Path("root")
        assert shard_directory(root, index=0, count=2) != shard_directory(root, index=0, count=3)
        assert shard_directory(root, index=1, count=2) == root / "shard-1-of-2"


class TestAShardAndAMerge:
    """Shards measure their share; a merge measures nothing."""

    def test_a_merge_resumes_every_cell_the_shards_measured(self, tmp_path: Path) -> None:
        """Two shards run the five cells between them; the merge runs none.

        Args:
            tmp_path: The test's temporary directory.
        """
        root = tmp_path / "shards"
        workers = [_Counter(), _Counter()]
        for index, worker in enumerate(workers):
            measure_shard(
                _shard(root, index, 2),
                measurement=_MEASUREMENT,
                inputs_digest=_DIGEST,
                cells=_cells(worker, 5),
            )
        assert [worker.measured for worker in workers] == [[0, 2, 4], [1, 3]]

        merger = _Counter()
        produced = assembled_cells(
            tmp_path / "merge",
            merge=SweepMerge(root=root, count=2),
            measurement=_MEASUREMENT,
            inputs_digest=_DIGEST,
            cells=_cells(merger, 5),
        )
        assert merger.measured == []
        assert list(produced) == [f"u@b{index}" for index in range(5)]
        assert produced["u@b3"][0]["value"] == 3.0
        assert not checkpoint_exists(tmp_path / "merge", _MEASUREMENT)
        assert checkpoint_exists(shard_directory(root, index=0, count=2), _MEASUREMENT)

    def test_a_shard_with_no_cells_still_publishes(self, tmp_path: Path) -> None:
        """So a merge can tell it from a shard that never ran.

        Args:
            tmp_path: The test's temporary directory.
        """
        directory = measure_shard(
            _shard(tmp_path, 2, 3),
            measurement=_MEASUREMENT,
            inputs_digest=_DIGEST,
            cells=_cells(_Counter(), 2),
        )
        assert load_sweep_checkpoint(directory, _MEASUREMENT)["cells"] == []

    def test_a_straight_run_needs_no_shards(self, tmp_path: Path) -> None:
        """No merge is the ordinary checkpointed run.

        Args:
            tmp_path: The test's temporary directory.
        """
        counter = _Counter()
        produced = assembled_cells(
            tmp_path,
            merge=None,
            measurement=_MEASUREMENT,
            inputs_digest=_DIGEST,
            cells=_cells(counter, 2),
        )
        assert counter.measured == [0, 1]
        assert list(produced) == ["u@b0", "u@b1"]


def _publish(directory: Path, *cells: str, digest: str = _DIGEST) -> None:
    """Write a shard checkpoint holding the named cells.

    Args:
        directory: The shard's directory.
        *cells: The cells it holds.
        digest: What it measured over.
    """
    save_sweep_checkpoint(
        directory,
        SweepCheckpoint(
            schema_version=SWEEP_CHECKPOINT_SCHEMA_VERSION,
            measurement=_MEASUREMENT,
            inputs_digest=digest,
            cells=[
                CellRecord(cell=name, observations=[Observation(name=name, value=1.0)])
                for name in cells
            ],
        ),
    )


def _adopt(tmp_path: Path, cells: Sequence[str]) -> AppError[ModelTrainerErrorCode]:
    """Adopt two shards under ``tmp_path`` and return the refusal.

    Args:
        tmp_path: The test's temporary directory; shards live under ``shards``.
        cells: What the sweep needs.

    Returns:
        The error the adoption raised.
    """
    with pytest.raises(AppError) as excinfo:
        adopt_shards(
            tmp_path / "merge",
            root=tmp_path / "shards",
            count=2,
            measurement=_MEASUREMENT,
            inputs_digest=_DIGEST,
            cells=cells,
        )
    refusal: AppError[ModelTrainerErrorCode] = excinfo.value
    return refusal


class TestAMergeRefuses:
    """Rather than assemble from whatever is there."""

    def test_an_absent_shard_and_its_cells(self, tmp_path: Path) -> None:
        """Shard 1 never ran: both it and its cell are named.

        Args:
            tmp_path: The test's temporary directory.
        """
        _publish(shard_directory(tmp_path / "shards", index=0, count=2), "u@b0")
        error = _adopt(tmp_path, ["u@b0", "u@b1"])
        assert error.code is ModelTrainerErrorCode.CARTRIDGE_SHARD_INCOMPLETE
        assert "shard-1-of-2" in error.message
        assert "u@b1" in error.message
        assert not checkpoint_exists(tmp_path / "merge", _MEASUREMENT)

    def test_a_shard_over_other_inputs(self, tmp_path: Path) -> None:
        """Cells measured over another corpus must not be mixed in.

        Args:
            tmp_path: The test's temporary directory.
        """
        root = tmp_path / "shards"
        _publish(shard_directory(root, index=0, count=2), "u@b0")
        _publish(shard_directory(root, index=1, count=2), "u@b1", digest="other")
        error = _adopt(tmp_path, ["u@b0", "u@b1"])
        assert error.code is ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN
        assert "inputs_digest" in error.message

    def test_a_cell_measured_twice(self, tmp_path: Path) -> None:
        """Two shards claiming one cell were cut from different lists.

        Args:
            tmp_path: The test's temporary directory.
        """
        root = tmp_path / "shards"
        _publish(shard_directory(root, index=0, count=2), "u@b0")
        _publish(shard_directory(root, index=1, count=2), "u@b0")
        error = _adopt(tmp_path, ["u@b0"])
        assert error.code is ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_DUPLICATE_CELL

    def test_a_cell_the_sweep_does_not_have(self, tmp_path: Path) -> None:
        """A shard of some other grid.

        Args:
            tmp_path: The test's temporary directory.
        """
        root = tmp_path / "shards"
        _publish(shard_directory(root, index=0, count=2), "u@b0")
        _publish(shard_directory(root, index=1, count=2), "v@b0")
        error = _adopt(tmp_path, ["u@b0"])
        assert error.code is ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN
        assert "v@b0" in error.message
