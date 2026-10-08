"""The trait-repair entry, on a real tiny GPT-2 over authored corpora.

The arms, the crowding pool, the PEFT adapter and both of its objectives are
REAL; the faked seams are the hub loaders, the trait and corpus readers and
the plan tables, as in every cartridge sweep's suite, and they live in
``tests/_trait_repair_support.py``.

This file holds the plain base walked from the command line, the
contamination wall and the plan table. The adapted bases, with the bit-for-bit
comparison against the corpus arc's own adapter, are
``test_cartridge_trait_repair_adapters.py``, and the shard-and-merge
reproduction is ``test_cartridge_trait_repair_shards.py``: the three were one
file that ran 137 s on one xdist worker (MCPs board task 2f90d785).
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator
from typing import TypedDict

import pytest
from platform_core.json_utils import load_json_str
from platform_core.run_record import RunRecord, decode_run_record

from model_trainer.cli import _trait_hooks as trait_hooks
from model_trainer.cli import cartridge_trait_repair_sweep as sweep
from model_trainer.core.services.model.cartridge_pool_plans import (
    BASE_LORA_SWEEP_PLANS,
    CONTENT_LORA_SWEEP_PLANS,
)
from model_trainer.core.services.model.cartridge_trait_plans import TRAIT_SWEEP_PLANS
from model_trainer.core.services.model.cartridge_trait_repair_plans import (
    TRAIT_REPAIR_SWEEP_EXPERIMENT,
    TRAIT_REPAIR_SWEEP_PLANS,
    TraitRepairAdapter,
    trait_repair_plan_label,
)
from tests._module_run import run_module_as_main
from tests._trait_repair_support import (
    install_repair_fakes,
    repair_argv,
    restore_repair_hooks,
    staged_pool,
    tiny_plan,
    unique_values,
)

#: The module-scoped walk is shared by two tests, so this file's tests run on
#: one xdist worker (tests/test_xdist_grouping.py says why).
pytestmark = pytest.mark.xdist_group("test_cartridge_trait_repair_sweep.py")


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards.

    Yields:
        None, once the fakes are installed.
    """
    install_repair_fakes()
    yield None
    restore_repair_hooks()


class _Walk(TypedDict):
    """One ``python -m`` run of the plain-base plan and the record it wrote."""

    code: int | str | None
    record: RunRecord


@pytest.fixture(name="walk", scope="module")
def _walk(tmp_path_factory: pytest.TempPathFactory) -> _Walk:
    """Run the plain-base plan once through ``python -m``.

    ONE RUN SERVES TestThePlainBase AND TestInvocationForms (MCPs board task
    2f90d785), where three tests each ran this same plan, 7.4 to 7.7 s each
    in CI job 113153367247: ``python -m`` runs the ``__main__`` guard, which
    calls ``entrypoint()``, which calls ``main()`` on the process arguments,
    which records ``measure_grid``'s rows unchanged. Module-scoped, so it
    installs the fakes itself, ahead of the function-scoped ``wired``.

    Args:
        tmp_path_factory: Source of the run's own directory.

    Returns:
        The exit code and the decoded record.
    """
    root = tmp_path_factory.mktemp("walk")
    argv = repair_argv(root)
    install_repair_fakes()
    try:
        code = run_module_as_main("model_trainer.cli.cartridge_trait_repair_sweep", argv)
    finally:
        restore_repair_hooks()
    text = (root / "nested" / "record.json").read_text(encoding="utf-8")
    return {"code": code, "record": decode_run_record(load_json_str(text))}


class TestThePlainBase:
    """The diverse family only: the naive grid already is the plain family."""

    def test_the_record_carries_the_cross_arms_and_the_diverse_cells(self, walk: _Walk) -> None:
        """Every pool member scored alone; every count of the one family."""
        values = unique_values(walk["record"]["observations"])
        assert values["held_out_pairs"] == 6.0
        assert values["max_drawn"] == 2.0
        assert values["bullets-companion-cross-1-expression_spread"] >= 0.0
        assert values["bullets-diverse-n3-cross-1-coherence_spread"] >= 0.0
        assert values["diverse_composed_noise_floor"] >= 0.0
        assert not [name for name in values if "lora-" in name or "epoch" in name]


class TestTheContaminationWall:
    """The pool must be the plan's size and held out from the traits."""

    def test_a_pool_of_the_wrong_size_is_refused(self, tmp_path: pathlib.Path) -> None:
        """One corpus where the plan adapts against two.

        Args:
            tmp_path: The test's temporary directory.
        """
        with pytest.raises(ValueError, match="pool of 2 corpora; 1 supplied"):
            sweep.measure_grid(
                tiny_plan(TraitRepairAdapter.NONE),
                plan_name="tiny",
                corpus=tmp_path / "traits",
                pool_corpora=staged_pool(tmp_path)[:1],
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
                tiny_plan(TraitRepairAdapter.NONE),
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


class TestInvocationForms:
    """``python -m``, and so the ``entrypoint()`` and ``main()`` it calls, write the record."""

    def test_running_it_as_a_module_writes_a_decodable_record(self, walk: _Walk) -> None:
        """Without the ``__main__`` guard this imports, runs nothing and exits 0."""
        assert walk["code"] == 0
        assert walk["record"]["experiment"] == TRAIT_REPAIR_SWEEP_EXPERIMENT
        assert walk["record"]["label"].startswith("tiny-diverse-gpt2-traitsbullets.")
