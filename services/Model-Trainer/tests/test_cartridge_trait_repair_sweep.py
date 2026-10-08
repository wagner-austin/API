"""The trait-repair entry, on a real tiny GPT-2 over authored corpora.

The arms, the crowding pool, the PEFT adapter and both of its objectives are
REAL; the faked seams are the hub loaders, the trait and corpus readers and
the plan tables, as in every cartridge sweep's suite.

THE PROPERTY THAT MATTERS MOST is asserted against another entry point: the
adapter a trait-repair record is measured on must be the adapter the corpus
arc's own sweep builds from the same pool, or the question "does the recorded
lever transfer" is asked of a different lever. Its epoch rows are compared
bit for bit with the corpus sweep's.
"""

from __future__ import annotations

import pathlib
import runpy
import sys
from collections.abc import Generator, Mapping

import pytest
from platform_core.json_utils import load_json_str
from platform_core.run_record import decode_run_record

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import _trait_hooks as trait_hooks
from model_trainer.cli import cartridge_base_lora_sweep as corpus_sweep
from model_trainer.cli import cartridge_trait_repair_sweep as sweep
from model_trainer.core.services.model.cartridge_pool_plans import (
    BASE_LORA_SWEEP_PLANS,
    CONTENT_LORA_SWEEP_PLANS,
    BaseLoraSweepPlan,
)
from model_trainer.core.services.model.cartridge_sweep_checkpoint import checkpoint_exists
from model_trainer.core.services.model.cartridge_sweep_shards import (
    SweepMerge,
    SweepShard,
    shard_directory,
)
from model_trainer.core.services.model.cartridge_trait_plans import TRAIT_SWEEP_PLANS
from model_trainer.core.services.model.cartridge_trait_repair_plans import (
    TRAIT_REPAIR_SWEEP_EXPERIMENT,
    TRAIT_REPAIR_SWEEP_PLANS,
    TraitRepairAdapter,
    TraitRepairPlan,
    trait_repair_plan_label,
)
from tests._trait_sweep_support import TINY_TRAIT_PLAN, install_fakes, restore_fakes

#: The crowd knobs shrunk to the tiny rung, shaped like the recorded rows.
_TINY_CROWD: BaseLoraSweepPlan = {
    "model_id": "gpt2",
    "window": 8,
    "held_out_stride": 3,
    "compartment_counts": (2, 3),
    "slots": 2,
    "probability": 0.5,
    "max_companions": 2,
    "lora_rank": 2,
    "lora_alpha": 4,
    "lora_epochs": 1,
    "lora_learning_rate": 0.05,
    "max_drawn": 2,
    "pool_members_per_corpus": 1,
    "seeds": (7, 8, 9),
    "epochs": 1,
    "learning_rate": 0.05,
}

#: Every fake corpus yields eight training windows at window 8 and stride 3,
#: so declaring eight makes the pool the one the corpus sweep builds.
_CROWD_WINDOWS = 8


def _plan(adapter: TraitRepairAdapter) -> TraitRepairPlan:
    """Build the tiny plan for one adapter.

    Args:
        adapter: Which base the families are measured on.

    Returns:
        The plan.
    """
    return TraitRepairPlan(
        trait=TINY_TRAIT_PLAN,
        crowd=_TINY_CROWD,
        adapter=adapter,
        crowd_windows_per_corpus=_CROWD_WINDOWS,
    )


def _fake_repair_plans() -> Mapping[str, TraitRepairPlan]:
    """Stand in for the production repair table.

    Returns:
        One plan per adapter.
    """
    return {
        "tiny-diverse": _plan(TraitRepairAdapter.NONE),
        "tiny-base-lora": _plan(TraitRepairAdapter.LANGUAGE_MODELING),
        "tiny-content-lora": _plan(TraitRepairAdapter.CROWD_INVARIANCE),
    }


def _fake_corpus_reader(corpus_dir: pathlib.Path, /) -> tuple[str, ...]:
    """Four documents of 24 characters each, keyed on the directory's name.

    Args:
        corpus_dir: The corpus directory; its first letter marks the text.

    Returns:
        The corpus bodies.
    """
    return tuple(f"{corpus_dir.name[0]}{index}" * 12 for index in range(4))


def _fake_corpus_plans() -> Mapping[str, BaseLoraSweepPlan]:
    """Stand in for the corpus arc's base-LoRA table.

    Returns:
        The tiny crowd row, so the corpus sweep builds the same adapter.
    """
    return {"tiny": _TINY_CROWD}


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards.

    Yields:
        None, once the fakes are installed.
    """
    install_fakes()
    trait_hooks.trait_repair_plans = _fake_repair_plans
    cli_hooks.read_corpus_documents = _fake_corpus_reader
    measurement_hooks.base_lora_sweep_plans = _fake_corpus_plans
    yield None
    restore_fakes()
    trait_hooks.trait_repair_plans = trait_hooks._default_trait_repair_plans
    cli_hooks.read_corpus_documents = cli_hooks._default_read_corpus_documents
    measurement_hooks.base_lora_sweep_plans = measurement_hooks._default_base_lora_sweep_plans


def _pool(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    """Create the two pool corpus directories.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The pool directories, in order.
    """
    created = [tmp_path / "delta", tmp_path / "echo"]
    for path in created:
        path.mkdir(parents=True)
    return created


def _values(tmp_path: pathlib.Path, adapter: TraitRepairAdapter) -> dict[str, float]:
    """Run one tiny plan and map its rows by name.

    Args:
        tmp_path: The test's temporary directory.
        adapter: Which plan to run.

    Returns:
        Every row's value, keyed by name.
    """
    observations, _digest = sweep.measure_grid(
        _plan(adapter),
        plan_name="tiny",
        corpus=tmp_path / "traits",
        pool_corpora=_pool(tmp_path),
        device="cpu",
        checkpoints=tmp_path / "checkpoints",
        merge=None,
    )
    names = [row["name"] for row in observations]
    assert len(names) == len(set(names))
    return {row["name"]: row["value"] for row in observations}


class TestThePlainBase:
    """The diverse family only: the naive grid already is the plain family."""

    def test_the_record_carries_the_cross_arms_and_the_diverse_cells(
        self, tmp_path: pathlib.Path
    ) -> None:
        """Every pool member scored alone; every count of the one family.

        Args:
            tmp_path: The test's temporary directory.
        """
        values = _values(tmp_path, TraitRepairAdapter.NONE)
        assert values["held_out_pairs"] == 6.0
        assert values["max_drawn"] == 2.0
        assert values["bullets-companion-cross-1-expression_spread"] >= 0.0
        assert values["bullets-diverse-n3-cross-1-coherence_spread"] >= 0.0
        assert values["diverse_composed_noise_floor"] >= 0.0
        assert not [name for name in values if "lora-" in name or "epoch" in name]


class TestTheAdaptedBases:
    """Both families, behind the adapter the corpus arc recorded."""

    def test_the_language_modeling_adapter_runs_both_families(self, tmp_path: pathlib.Path) -> None:
        """Plain and diverse cells, and the adapter's convergence rows.

        Args:
            tmp_path: The test's temporary directory.
        """
        values = _values(tmp_path, TraitRepairAdapter.LANGUAGE_MODELING)
        assert values["lora-train-epoch-0_loss"] > 0.0
        assert values["lora-plain_composed_noise_floor"] >= 0.0
        assert values["lora-diverse_composed_noise_floor"] >= 0.0
        assert values["bullets-lora-plain-n2-alone-expression_spread"] >= 0.0

    def test_the_invariance_adapter_records_its_distillation(self, tmp_path: pathlib.Path) -> None:
        """The KL rows are the distillation's own, under the recorded name.

        Args:
            tmp_path: The test's temporary directory.
        """
        values = _values(tmp_path, TraitRepairAdapter.CROWD_INVARIANCE)
        assert values["invariance-train-epoch-0_kl"] > 0.0
        assert not [name for name in values if name.startswith("lora-train-epoch")]

    def test_the_adapter_is_the_one_the_corpus_sweep_builds(self, tmp_path: pathlib.Path) -> None:
        """Same pool, same plan row: bit-identical epoch rows from both entries.

        The corpus sweep's primary corpus yields eight training windows, the
        count the repair plan declares, so both truncate the pool alike.

        Args:
            tmp_path: The test's temporary directory.
        """
        trait_values = _values(tmp_path / "trait", TraitRepairAdapter.LANGUAGE_MODELING)
        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        primary, beta, gamma = (corpus_root / name for name in ("alpha", "beta", "gamma"))
        for path in (primary, beta, gamma):
            path.mkdir()
        corpus_rows, _digest = corpus_sweep.measure_grid(
            _TINY_CROWD,
            plan_name="tiny",
            corpus=primary,
            other_corpora=[beta, gamma],
            pool_corpora=_pool(corpus_root),
            device="cpu",
            checkpoints=corpus_root / "checkpoints",
        )
        corpus_values = {row["name"]: row["value"] for row in corpus_rows}
        assert trait_values["lora-train-epoch-0_loss"] == corpus_values["lora-train-epoch-0_loss"]


class TestTheContaminationWall:
    """The pool must be the plan's size and held out from the traits."""

    def test_a_pool_of_the_wrong_size_is_refused(self, tmp_path: pathlib.Path) -> None:
        """One corpus where the plan adapts against two.

        Args:
            tmp_path: The test's temporary directory.
        """
        with pytest.raises(ValueError, match="pool of 2 corpora; 1 supplied"):
            sweep.measure_grid(
                _plan(TraitRepairAdapter.NONE),
                plan_name="tiny",
                corpus=tmp_path / "traits",
                pool_corpora=_pool(tmp_path)[:1],
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
                merge=None,
            )

    def test_the_trait_corpus_cannot_be_a_pool_corpus(self, tmp_path: pathlib.Path) -> None:
        """A base adapted on the traits would carry the answer in its LoRA.

        Args:
            tmp_path: The test's temporary directory.
        """
        traits = tmp_path / "traits"
        with pytest.raises(ValueError, match="also measured"):
            sweep.measure_grid(
                _plan(TraitRepairAdapter.NONE),
                plan_name="tiny",
                corpus=traits,
                pool_corpora=[traits, tmp_path / "echo"],
                device="cpu",
                checkpoints=tmp_path / "checkpoints",
                merge=None,
            )


class TestThePlanTable:
    """Every row references the recorded rows by identity."""

    def test_every_trait_row_is_the_naive_grids_row(self) -> None:
        """The repairs are measured on exactly the cartridges the naive grid ran."""
        for plan in TRAIT_REPAIR_SWEEP_PLANS.values():
            assert plan["trait"] is TRAIT_SWEEP_PLANS["gpt2-traits"]

    def test_each_adapter_reads_the_corpus_arcs_own_row(self) -> None:
        """Base-LoRA from the base-LoRA table, invariance from the content-LoRA table."""
        assert (
            TRAIT_REPAIR_SWEEP_PLANS["gpt2-traits-base-lora"]["crowd"]
            is BASE_LORA_SWEEP_PLANS["gpt2-base-lora"]
        )
        assert (
            TRAIT_REPAIR_SWEEP_PLANS["gpt2-traits-content-lora"]["crowd"]
            is CONTENT_LORA_SWEEP_PLANS["gpt2-content-lora"]
        )
        assert [plan["adapter"] for plan in TRAIT_REPAIR_SWEEP_PLANS.values()] == list(
            TraitRepairAdapter
        )

    def test_the_label_names_the_adapter_and_the_crowd(self) -> None:
        """Two adapters on one roster must never share a label."""
        label = trait_repair_plan_label(
            "gpt2-traits-content-lora",
            TRAIT_REPAIR_SWEEP_PLANS["gpt2-traits-content-lora"],
            digest="53b2249185311be5",
        )
        assert "-crowd-invariance-w256-s4-cw42-p0.5-K3-R8-a16-le3-llr0.0001-D8-m3" in label
        assert label.startswith("gpt2-traits-content-lora-gpt2-traitsbullets.")


class TestHookDefaults:
    """The production table sits behind the hook the fakes replace."""

    def test_the_repair_plan_hook_serves_the_declared_table(self) -> None:
        """The committed plans, not a copy of them."""
        assert trait_hooks._default_trait_repair_plans() is TRAIT_REPAIR_SWEEP_PLANS


def _argv(tmp_path: pathlib.Path) -> list[str]:
    """Build the flags one run takes.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The argument list, pool directories created.
    """
    pool = ",".join(str(path) for path in _pool(tmp_path))
    return [
        "--plan",
        "tiny-diverse",
        "--corpus",
        str(tmp_path / "traits"),
        "--pool-corpora",
        pool,
        "--device",
        "cpu",
        "--out",
        str(tmp_path / "nested" / "record.json"),
    ]


class TestInvocationForms:
    """The `main` call and `python -m` must both measure and write."""

    def test_main_writes_a_decodable_record(self, tmp_path: pathlib.Path) -> None:
        """The artifact is the deliverable.

        Args:
            tmp_path: The test's temporary directory.
        """
        assert sweep.main(_argv(tmp_path)) == 0
        restored = decode_run_record(
            load_json_str((tmp_path / "nested" / "record.json").read_text(encoding="utf-8"))
        )
        assert restored["experiment"] == TRAIT_REPAIR_SWEEP_EXPERIMENT
        assert restored["label"].startswith("tiny-diverse-gpt2-traitsbullets.")

    def test_running_it_as_a_module_actually_measures(self, tmp_path: pathlib.Path) -> None:
        """Without the ``__main__`` guard this imports, runs nothing and exits 0.

        Args:
            tmp_path: The test's temporary directory.
        """
        module_name = "model_trainer.cli.cartridge_trait_repair_sweep"
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["x", *_argv(tmp_path)]
        try:
            with pytest.raises(SystemExit) as raised:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module
        assert raised.value.code == 0
        assert (tmp_path / "nested" / "record.json").is_file()


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
            crowd=_TINY_CROWD,
            adapter=TraitRepairAdapter.LANGUAGE_MODELING,
            crowd_windows_per_corpus=_CROWD_WINDOWS,
        )
        pool = _pool(tmp_path)
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
        flags = _argv(tmp_path)
        shard = ["--shards", str(root), "--shard-count", "1", "--shard", "0"]
        with pytest.raises(ValueError, match="drop --out"):
            sweep.main([*flags, *shard])
        assert sweep.main([*flags[:-2], *shard]) == 0
        assert checkpoint_exists(
            shard_directory(root, index=0, count=1), "trait-repair-tiny-diverse"
        )
        assert not (tmp_path / "nested" / "record.json").exists()
