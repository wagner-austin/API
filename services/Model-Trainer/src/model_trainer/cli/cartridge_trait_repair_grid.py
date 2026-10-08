"""The trait-repair grid's cells, bound to the base each plan measures on.

Split from :mod:`~model_trainer.cli.cartridge_trait_repair_sweep` when seed
blocks and shards took that module past the file ceiling. The concern here is
CONSTRUCTION -- the gates, the pool, the adapter, the families and every cell
bound to its seeds -- and nothing in it measures anything until a cell is
called. The sweep module RUNS a grid: straight, as a shard, or as a merge.

THE ADAPTER IS RETRAINED BY EVERY JOB THAT BUILDS A GRID, and that is
deliberate: it is deterministic and minutes long, and its epoch rows enter
the grid's inputs digest, so a merge REFUSES shards whose adapters came out
differently rather than mixing cells measured on two bases.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable, Sequence

import torch
from platform_core.logging import get_logger
from platform_core.run_record import Observation
from typing_extensions import TypedDict

from model_trainer.cli import _trait_hooks
from model_trainer.cli.cartridge_base_lora_sweep import require_held_out_pool
from model_trainer.cli.cartridge_composition_sweep import staged_partner_trains
from model_trainer.cli.cartridge_crowd_adapters import (
    crowd_invariance_adapter,
    language_modeling_adapter,
)
from model_trainer.cli.cartridge_lora_policy import quantization_for
from model_trainer.cli.cartridge_pool_provider import SeededPoolProvider
from model_trainer.core.contracts.trait_corpus import trait_corpus_digest
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_measurement import ReplicateBuilderProto
from model_trainer.core.services.model.cartridge_plans import digest_parts
from model_trainer.core.services.model.cartridge_sweep_checkpoint import bind_cells
from model_trainer.core.services.model.cartridge_sweep_shards import block_cell, seed_blocks
from model_trainer.core.services.model.cartridge_trait_repair_plans import (
    TraitRepairAdapter,
    TraitRepairPlan,
    trait_repair_plan_label,
)
from model_trainer.core.services.model.cartridge_varied import CompanionPoolProviderProto
from model_trainer.core.services.model.trait_arms import (
    measure_trait_composition,
    trait_arm_observations,
    trait_cell_observations,
)
from model_trainer.core.services.model.trait_families import (
    companioned_trait_build,
    measure_trait_companion_cross,
    plain_trait_build,
)
from model_trainer.core.services.model.trait_roster import (
    SplitTrait,
    gate_primary_trait,
    prepare_traits,
)
from model_trainer.core.types import CacheCapableLMProto

_log = get_logger(__name__)


class RepairGrid(TypedDict):
    """One repair plan's cells, bound to its adapted base, before any runs.

    Attributes:
        digest: The trait corpora's digest.
        inputs_digest: What the grid measures over: the plan's label, the
            pool corpora and the adapter's epoch rows.
        trait: The primary trait's name, which prefixes every arm.
        families: The families this plan runs, in record order.
        pool_size: How many members the companion pool holds.
        head_rows: The gates' rows, ``max_drawn`` and the adapter's rows.
        cells: Every companion-cross block, then every family's blocks.
    """

    digest: str
    inputs_digest: str
    trait: str
    families: tuple[str, ...]
    pool_size: int
    head_rows: tuple[Observation, ...]
    cells: list[tuple[str, Callable[[], Sequence[Observation]]]]


def adapted_base(
    base: CacheCapableLMProto,
    pool_trains: Sequence[Sequence[torch.Tensor]],
    plan: TraitRepairPlan,
    *,
    device: str,
) -> tuple[CacheCapableLMProto, tuple[Observation, ...]]:
    """Return the base the families are measured on, and its adapter's rows.

    Args:
        base: The plain base, loaded and on the device. Consumed when the plan
            adapts it, because PEFT injects into its module tree.
        pool_trains: One training-window sequence per pool corpus.
        plan: The plan, which names the adapter.
        device: Where a teacher base is placed.

    Returns:
        ``(measured_base, epoch_rows)``; the plain base and no rows when the
        plan adapts nothing.

    Raises:
        ValueError: Propagated from the adapter trainers.
        AppError: Propagated from the PEFT and model-loading layers.
    """
    if plan["adapter"] is TraitRepairAdapter.LANGUAGE_MODELING:
        return language_modeling_adapter(base, pool_trains, plan["crowd"])
    if plan["adapter"] is TraitRepairAdapter.CROWD_INVARIANCE:
        return crowd_invariance_adapter(base, pool_trains, plan["crowd"], device=device)
    return base, ()


def repair_families(
    measured: CacheCapableLMProto,
    plan: TraitRepairPlan,
    *,
    prepared: Sequence[SplitTrait],
    pool_for_seed: CompanionPoolProviderProto,
) -> tuple[tuple[str, Callable[[int, tuple[int, ...]], ReplicateBuilderProto]], ...]:
    """Name each family this plan runs and bind its recipe per count and block.

    THE PLAIN BASE RUNS ONLY THE DIVERSE FAMILY, because its plain family is
    the naive trait grid, already recorded by ``cartridge_trait_sweep``.
    Running it again here would be the same cells under a second name. An
    adapted base runs both, as the corpus arc's LoRA sweeps did, under the
    names they used.

    Args:
        measured: The base every cartridge trains in front of.
        plan: The plan being run.
        prepared: The roster's splits, primary first.
        pool_for_seed: The replicate's frozen companion pool provider.

    Returns:
        ``(family_name, builder_for(count, block))`` per family, in record
        order.
    """
    trait = plan["trait"]
    crowd = plan["crowd"]
    primary = prepared[0]

    def _plain(count: int, block: tuple[int, ...]) -> ReplicateBuilderProto:
        """Bind the naive recipe at one compartment count over one block.

        Args:
            count: How many trait cartridges are composed.
            block: The seeds to measure.

        Returns:
            The builder.
        """
        return plain_trait_build(
            measured,
            first_train=primary.train,
            other_trains=[other.train for other in prepared[1:count]],
            num_slots=trait["slots"],
            seeds=block,
            seed_stride=len(trait["seeds"]),
            epochs=trait["epochs"],
            learning_rate=trait["learning_rate"],
        )

    def _diverse(count: int, block: tuple[int, ...]) -> ReplicateBuilderProto:
        """Bind the diverse-companion recipe at one compartment count over one block.

        Args:
            count: How many trait cartridges are composed.
            block: The seeds to measure.

        Returns:
            The builder.
        """
        return companioned_trait_build(
            measured,
            first_train=primary.train,
            other_trains=[other.train for other in prepared[1:count]],
            num_slots=trait["slots"],
            seeds=block,
            seed_stride=len(trait["seeds"]),
            epochs=trait["epochs"],
            learning_rate=trait["learning_rate"],
            pool_for_seed=pool_for_seed,
            companion_probability=crowd["probability"],
        )

    if plan["adapter"] is TraitRepairAdapter.NONE:
        return (("diverse", _diverse),)
    return (("lora-plain", _plain), ("lora-diverse", _diverse))


def repair_grid(
    plan: TraitRepairPlan,
    *,
    plan_name: str,
    corpus: pathlib.Path,
    pool_corpora: Sequence[pathlib.Path],
    device: str,
) -> RepairGrid:
    """Gate the plan, build its base, and bind every cell.

    Args:
        plan: The measurement to run.
        plan_name: Which plan this is.
        corpus: Directory holding one authored JSON file per trait.
        pool_corpora: The recorded pool corpora, one per companion; disjoint
            from the traits.
        device: Device to measure on.

    Returns:
        The grid, with no cell measured yet.

    Raises:
        ValueError: When the pool is the wrong size, repeats, or overlaps the
            traits, and propagated from the adapter trainers.
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` from the corpus layer,
            ``CARTRIDGE_QA_UNDERPOWERED`` from the gates,
            ``CARTRIDGE_SEED_BLOCKS_UNEVEN`` when the seeds do not cut into
            blocks, and from the PEFT layer.
    """
    trait = plan["trait"]
    crowd = plan["crowd"]
    require_held_out_pool(crowd, pool_corpora=pool_corpora, measured=[corpus])

    corpora = _trait_hooks.read_trait_corpora(corpus, trait["traits"])
    digest = trait_corpus_digest(corpora)
    prepared = prepare_traits(corpora, trait, device=device)
    primary = prepared[0]
    gate_rows = gate_primary_trait(trait, primary)
    blocks = seed_blocks(trait["seeds"])

    pool_trains, pool_digests = staged_partner_trains(
        pool_corpora,
        tokenizer=hf_hooks.Hooks.load_hf_tokenizer(trait["model_id"]),
        window=crowd["window"],
        held_out_stride=crowd["held_out_stride"],
        required=plan["crowd_windows_per_corpus"],
        device=device,
    )

    base = require_cache_capable(
        hf_hooks.Hooks.load_hf_model(trait["model_id"], quantization_for(trait["model_id"]))
    )
    base.to(device)
    measured, epoch_rows = adapted_base(base, pool_trains, plan, device=device)

    # Constructed over the WHOLE plan's seeds whichever blocks this job runs:
    # the provider spaces each replicate's member seeds by that count, which
    # is what makes a block's pool the pool a straight run would have drawn.
    provider = SeededPoolProvider(
        measured,
        pool_trains,
        slots=crowd["slots"],
        seeds=trait["seeds"],
        epochs=crowd["epochs"],
        learning_rate=crowd["learning_rate"],
    )
    families = repair_families(measured, plan, prepared=prepared, pool_for_seed=provider.pool)

    def _companion_cross(block: tuple[int, ...]) -> tuple[Observation, ...]:
        """Score every pool member alone on the primary's pairs, over one block.

        Args:
            block: The seeds to measure.

        Returns:
            Each member's readings over the block.
        """
        arms = measure_trait_companion_cross(
            measured,
            pool_for_seed=provider.pool,
            pool_size=len(pool_trains),
            seeds=block,
            held_out=primary.held_out,
            arm=f"{primary.trait}-companion-cross",
        )
        return tuple(row for arm in arms for row in trait_arm_observations(arm))

    def _cell(
        unit: tuple[
            str, Callable[[int, tuple[int, ...]], ReplicateBuilderProto], int, tuple[int, ...]
        ],
    ) -> tuple[Observation, ...]:
        """Measure one family at one count over one block.

        Args:
            unit: ``(family, recipe, count, block)``.

        Returns:
            Every arm's rows over the block.
        """
        family, recipe, count, block = unit
        arm = f"{primary.trait}-{family}-n{count}"
        cell = measure_trait_composition(
            measured,
            build=recipe(count, block),
            partners=count - 1,
            held_out=primary.held_out,
            arm=arm,
        )
        _log.info(
            "%s seeds %s: %+.4f alone -> %+.4f composed",
            arm,
            block,
            cell["alone"]["expression"]["mean"],
            cell["composed"]["expression"]["mean"],
        )
        return trait_cell_observations(arm, cell)

    return RepairGrid(
        digest=digest,
        inputs_digest=digest_parts(
            [
                trait_repair_plan_label(plan_name, plan, digest=digest),
                *pool_digests,
                *(f"{row['name']}={row['value']!r}" for row in epoch_rows),
            ]
        ),
        trait=primary.trait,
        families=tuple(family for family, _recipe in families),
        pool_size=len(pool_trains),
        head_rows=(
            *gate_rows,
            Observation(name="max_drawn", value=float(crowd["max_drawn"])),
            *epoch_rows,
        ),
        cells=[
            *bind_cells(
                [
                    (block_cell("companion-cross", index), block)
                    for index, block in enumerate(blocks)
                ],
                _companion_cross,
            ),
            *bind_cells(
                [
                    (block_cell(f"{family}-n{count}", index), (family, recipe, count, block))
                    for family, recipe in families
                    for count in trait["compartment_counts"]
                    for index, block in enumerate(blocks)
                ],
                _cell,
            ),
        ],
    )


__all__ = ["RepairGrid", "adapted_base", "repair_families", "repair_grid"]
