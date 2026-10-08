"""The trait-repair grid in shards, on a real tiny GPT-2 over authored corpora.

A merged repair record must be the straight one, adapter and all; the fakes
are ``tests/_trait_repair_support.py``'s. Split from
``test_cartridge_trait_repair_sweep.py`` (MCPs board task 2f90d785): these
tests share no fixture, so under ``--dist loadgroup`` each takes whichever
worker is free rather than queueing behind the plain-base and adapter runs,
and the reproduction test alone was 56.5 s in CI job 113153367247.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator

import pytest

from model_trainer.cli import cartridge_trait_repair_sweep as sweep
from model_trainer.core.services.model.cartridge_sweep_checkpoint import checkpoint_exists
from model_trainer.core.services.model.cartridge_sweep_shards import (
    SweepMerge,
    SweepShard,
    shard_directory,
)
from model_trainer.core.services.model.cartridge_trait_repair_plans import (
    TraitRepairAdapter,
    TraitRepairPlan,
)
from tests._trait_repair_support import (
    CROWD_WINDOWS,
    TINY_CROWD,
    install_repair_fakes,
    repair_argv,
    restore_repair_hooks,
    staged_pool,
)
from tests._trait_sweep_support import TINY_TRAIT_PLAN


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards.

    Yields:
        None, once the fakes are installed.
    """
    install_repair_fakes()
    yield None
    restore_repair_hooks()


class TestShardingTheRepairGrid:
    """A merged repair record is the straight one, adapter and all."""

    def test_shards_and_a_merge_reproduce_the_straight_run(self, tmp_path: pathlib.Path) -> None:
        """Two blocks behind the language-modeling adapter, bit for bit.

        The adapter is retrained in every job, and its epoch rows are part of
        what every shard's cells were measured over -- so the merge is also
        the check that three trainings of it agreed.

        Args:
            tmp_path: The test's temporary directory.
        """
        plan = TraitRepairPlan(
            trait={**TINY_TRAIT_PLAN, "seeds": (7, 8, 9, 10, 11, 12)},
            crowd=TINY_CROWD,
            adapter=TraitRepairAdapter.LANGUAGE_MODELING,
            crowd_windows_per_corpus=CROWD_WINDOWS,
        )
        pool = staged_pool(tmp_path)
        corpus = tmp_path / "traits"
        straight, _digest = sweep.measure_grid(
            plan,
            plan_name="tiny",
            corpus=corpus,
            pool_corpora=pool,
            device="cpu",
            checkpoints=tmp_path / "straight",
            merge=None,
        )
        root = tmp_path / "shards"
        for index in range(2):
            sweep.measure_repair_shard(
                plan,
                plan_name="tiny",
                corpus=corpus,
                pool_corpora=pool,
                device="cpu",
                shard=SweepShard(root=root, count=2, index=index),
            )
        merged, _digest = sweep.measure_grid(
            plan,
            plan_name="tiny",
            corpus=corpus,
            pool_corpora=pool,
            device="cpu",
            checkpoints=tmp_path / "merge",
            merge=SweepMerge(root=root, count=2),
        )
        assert merged == straight
        names = {row["name"] for row in merged}
        assert "bullets-lora-diverse-n3-composed-style_seed12_gain" in names
        assert "bullets-companion-cross-1-expression_seed10_gain" in names

    def test_a_shard_writes_its_checkpoint_and_refuses_an_out_path(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The shard form through ``main``, both ways.

        Args:
            tmp_path: The test's temporary directory.
        """
        root = tmp_path / "shards"
        flags = repair_argv(tmp_path)
        shard = ["--shards", str(root), "--shard-count", "1", "--shard", "0"]
        with pytest.raises(ValueError, match="drop --out"):
            sweep.main([*flags, *shard])
        assert sweep.main([*flags[:-2], *shard]) == 0
        assert checkpoint_exists(
            shard_directory(root, index=0, count=1), "trait-repair-tiny-diverse"
        )
        assert not (tmp_path / "nested" / "record.json").exists()
