"""The naive trait grid sharded across jobs, against the same grid run straight.

THE ONE CLAIM THAT MAKES SHARDING SAFE is that a merged record IS the straight
record: the same rows, the same names, the same floats bit for bit. It is
asserted here over two blocks on a real tiny GPT-2, because every way it could
fail -- a block drawing different partners, a rebuild reading a block's own
mean, a merge assembling from fewer seeds -- produces a complete record that
only a comparison with the straight run can catch.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator, Mapping

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import cartridge_trait_sweep as sweep
from model_trainer.core.contracts.trait_plan import TraitPlan
from model_trainer.core.services.model.cartridge_sweep_checkpoint import checkpoint_exists
from model_trainer.core.services.model.cartridge_sweep_shards import (
    SweepMerge,
    SweepShard,
    shard_directory,
)
from tests._trait_sweep_support import TINY_TRAIT_PLAN, install_fakes, restore_fakes

#: Two blocks, so a shard holds half the seeds and a block's partners would
#: land on the other block's seeds if the stride came from the call.
_TWO_BLOCKS: TraitPlan = {**TINY_TRAIT_PLAN, "seeds": (7, 8, 9, 10, 11, 12)}


def _plans() -> Mapping[str, TraitPlan]:
    """Stand in for the production table with the two-block plan.

    Returns:
        One plan.
    """
    return {"two": _TWO_BLOCKS}


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes and the two-block table, and restore them afterwards.

    Yields:
        None, once the fakes are installed.
    """
    install_fakes()
    measurement_hooks.trait_sweep_plans = _plans
    yield None
    restore_fakes()


def _shard_both(root: pathlib.Path, corpus: pathlib.Path) -> None:
    """Run both shards of the two-shard cut.

    Args:
        root: The shard root.
        corpus: The (faked) corpus directory.
    """
    for index in range(2):
        sweep.measure_trait_shard(
            _TWO_BLOCKS,
            plan_name="two",
            corpus=corpus,
            device="cpu",
            shard=SweepShard(root=root, count=2, index=index),
        )


class TestAMergedRecordIsTheStraightRecord:
    """Row for row, bit for bit."""

    def test_two_shards_and_a_merge_reproduce_the_straight_run(
        self, tmp_path: pathlib.Path
    ) -> None:
        """And the record carries every seed of both blocks.

        Args:
            tmp_path: The test's temporary directory.
        """
        straight, _digest = sweep.measure_grid(
            _TWO_BLOCKS,
            plan_name="two",
            corpus=tmp_path,
            device="cpu",
            checkpoints=tmp_path / "straight",
            merge=None,
        )
        root = tmp_path / "shards"
        _shard_both(root, tmp_path)
        merged, _digest = sweep.measure_grid(
            _TWO_BLOCKS,
            plan_name="two",
            corpus=tmp_path,
            device="cpu",
            checkpoints=tmp_path / "merge",
            merge=SweepMerge(root=root, count=2),
        )
        assert merged == straight
        names = {row["name"] for row in merged}
        assert "bullets-n3-composed-expression_seed12_gain" in names
        assert "bullets-solo-expression_seed7_gain" in names
        assert checkpoint_exists(shard_directory(root, index=1, count=2), "trait-two-composed")

    def test_a_merge_with_a_shard_missing_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Assembling from one shard would report means over half the seeds.

        Args:
            tmp_path: The test's temporary directory.
        """
        root = tmp_path / "shards"
        sweep.measure_trait_shard(
            _TWO_BLOCKS,
            plan_name="two",
            corpus=tmp_path,
            device="cpu",
            shard=SweepShard(root=root, count=2, index=0),
        )
        with pytest.raises(AppError) as excinfo:
            sweep.measure_grid(
                _TWO_BLOCKS,
                plan_name="two",
                corpus=tmp_path,
                device="cpu",
                checkpoints=tmp_path / "merge",
                merge=SweepMerge(root=root, count=2),
            )
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_SHARD_INCOMPLETE
        assert "shard-1-of-2" in excinfo.value.message


def _flags(tmp_path: pathlib.Path, *extra: str) -> list[str]:
    """Build one invocation's flags.

    Args:
        tmp_path: The test's temporary directory.
        *extra: Flags appended.

    Returns:
        The argument list.
    """
    return ["--plan", "two", "--corpus", str(tmp_path), "--device", "cpu", *extra]


class TestTheCommandLine:
    """The shard and merge forms through ``main``."""

    def test_a_shard_writes_checkpoints_and_no_record(self, tmp_path: pathlib.Path) -> None:
        """Its checkpoints are its output.

        Args:
            tmp_path: The test's temporary directory.
        """
        root = tmp_path / "shards"
        shard = ("--shards", str(root), "--shard-count", "2", "--shard", "1")
        assert sweep.main(_flags(tmp_path, *shard)) == 0
        directory = shard_directory(root, index=1, count=2)
        assert checkpoint_exists(directory, "trait-two-anchor")
        assert checkpoint_exists(directory, "trait-two-composed")
        assert not list(tmp_path.glob("**/*record*.json"))

    def test_a_shard_given_an_out_path_is_refused(self, tmp_path: pathlib.Path) -> None:
        """A shard has no record to write, and a path for one is a mistake.

        Args:
            tmp_path: The test's temporary directory.
        """
        flags = _flags(
            tmp_path,
            *("--shards", str(tmp_path), "--shard-count", "2", "--shard", "0"),
            *("--out", str(tmp_path / "record.json")),
        )
        with pytest.raises(ValueError, match="drop --out"):
            sweep.main(flags)

    def test_the_merge_writes_the_record(self, tmp_path: pathlib.Path) -> None:
        """After both shards, the merge form writes the plan's record.

        Args:
            tmp_path: The test's temporary directory.
        """
        root = tmp_path / "shards"
        _shard_both(root, tmp_path)
        out = tmp_path / "merged" / "record.json"
        merge = ("--shards", str(root), "--shard-count", "2", "--out", str(out))
        assert sweep.main(_flags(tmp_path, *merge)) == 0
        restored = decode_run_record(load_json_str(out.read_text(encoding="utf-8")))
        assert "-seeds7..12-" in restored["label"]
