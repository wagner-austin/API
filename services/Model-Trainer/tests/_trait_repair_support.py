"""The fakes the trait-repair suites share, in one place rather than three.

EXTRACTED WHEN THE SUITE SPLIT (MCPs board task 2f90d785), not copied. The
one file held three costly concerns, the plain base walked from the command
line, the two adapted bases, and the shard-and-merge reproduction, and on
one xdist worker they ran 137 s back to back. Split by concern, each lands
on its own worker; every split file needs the same tiny plans, the same
corpus reader and the same pool, and two copies drifting would let one
suite pass against a world the other no longer describes.

Not a ``conftest.py``: these are imported by name, so a reader of any of the
suites can see where its world comes from.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping, Sequence

from platform_core.run_record import Observation

from model_trainer.cli import _measurement_hooks as measurement_hooks
from model_trainer.cli import _test_hooks as cli_hooks
from model_trainer.cli import _trait_hooks as trait_hooks
from model_trainer.cli import cartridge_trait_repair_sweep as sweep
from model_trainer.core.services.model.cartridge_pool_plans import BaseLoraSweepPlan
from model_trainer.core.services.model.cartridge_trait_repair_plans import (
    TraitRepairAdapter,
    TraitRepairPlan,
)
from tests._trait_sweep_support import TINY_TRAIT_PLAN, install_fakes, restore_fakes

#: The crowd knobs shrunk to the tiny rung, shaped like the recorded rows.
TINY_CROWD: BaseLoraSweepPlan = {
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
CROWD_WINDOWS = 8


def tiny_plan(adapter: TraitRepairAdapter) -> TraitRepairPlan:
    """Build the tiny plan for one adapter.

    Args:
        adapter: Which base the families are measured on.

    Returns:
        The plan.
    """
    return TraitRepairPlan(
        trait=TINY_TRAIT_PLAN,
        crowd=TINY_CROWD,
        adapter=adapter,
        crowd_windows_per_corpus=CROWD_WINDOWS,
    )


def _fake_repair_plans() -> Mapping[str, TraitRepairPlan]:
    """Stand in for the production repair table.

    Returns:
        One plan per adapter.
    """
    return {
        "tiny-diverse": tiny_plan(TraitRepairAdapter.NONE),
        "tiny-base-lora": tiny_plan(TraitRepairAdapter.LANGUAGE_MODELING),
        "tiny-content-lora": tiny_plan(TraitRepairAdapter.CROWD_INVARIANCE),
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
    return {"tiny": TINY_CROWD}


def install_repair_fakes() -> None:
    """Point the trait, repair-plan, corpus and corpus-plan hooks at fakes."""
    install_fakes()
    trait_hooks.trait_repair_plans = _fake_repair_plans
    cli_hooks.read_corpus_documents = _fake_corpus_reader
    measurement_hooks.base_lora_sweep_plans = _fake_corpus_plans


def restore_repair_hooks() -> None:
    """Put every production hook back."""
    restore_fakes()
    trait_hooks.trait_repair_plans = trait_hooks._default_trait_repair_plans
    cli_hooks.read_corpus_documents = cli_hooks._default_read_corpus_documents
    measurement_hooks.base_lora_sweep_plans = measurement_hooks._default_base_lora_sweep_plans


def staged_pool(tmp_path: pathlib.Path) -> list[pathlib.Path]:
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


def unique_values(observations: Sequence[Observation]) -> dict[str, float]:
    """Map rows by name, refusing a name two rows share.

    Args:
        observations: The rows one run produced.

    Returns:
        Every row's value, keyed by name.
    """
    names = [row["name"] for row in observations]
    assert len(names) == len(set(names))
    return {row["name"]: row["value"] for row in observations}


def measured_values(tmp_path: pathlib.Path, adapter: TraitRepairAdapter) -> dict[str, float]:
    """Run one tiny plan and map its rows by name.

    Args:
        tmp_path: The test's temporary directory.
        adapter: Which plan to run.

    Returns:
        Every row's value, keyed by name.
    """
    observations, _digest = sweep.measure_grid(
        tiny_plan(adapter),
        plan_name="tiny",
        corpus=tmp_path / "traits",
        pool_corpora=staged_pool(tmp_path),
        device="cpu",
        checkpoints=tmp_path / "checkpoints",
        merge=None,
    )
    return unique_values(observations)


def repair_argv(tmp_path: pathlib.Path) -> list[str]:
    """Build the flags one plain-base run takes.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The argument list, pool directories created.
    """
    pool = ",".join(str(path) for path in staged_pool(tmp_path))
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
