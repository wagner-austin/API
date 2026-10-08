"""The two base-side adapters, built once for every sweep that measures behind one.

WHY THIS MODULE EXISTS. ``cartridge_base_lora_sweep`` and
``cartridge_content_lora_sweep`` each built their adapter inline -- the same
crowding pool from the same seed formula, the same PEFT wrap, and then one of
two training objectives -- and the trait-repair sweep (board task
``83c25b86``) needs both adapters again, exactly as recorded, because its
question is whether THE RECORDED levers transfer from corpus compartments to
disposition compartments. A third inline copy would be a fork of the thing
being tested; one owner means the adapter a trait record carries is the
adapter a corpus record carries, and the epoch rows each emits can be
compared bit for bit.

THE SEQUENCE IS PART OF THE ADAPTER, and that is why it is one function per
objective rather than three helpers a caller strings together. Loading a
second base for the teacher and wrapping the base in PEFT both initialise
weights from the process-wide generator, and the crowding pool's training
leaves that generator in a state the next step consumes. So the invariance
adapter is pool, then teacher, then wrap, then distil -- the order its
recorded sweep used -- and a caller that reordered them would get a
different adapter with the same name.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from platform_core.logging import get_logger
from platform_core.run_record import Observation

from model_trainer.cli.cartridge_lora_policy import quantization_for, target_modules_for
from model_trainer.core.services.finetuning.strategies import _test_hooks as strategy_hooks
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.finetuning.strategies.cartridge_slots import CartridgeSlots
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_base_lora import (
    freeze_adapted,
    train_composition_lora,
)
from model_trainer.core.services.model.cartridge_content_lora import (
    train_composition_lora_invariant,
)
from model_trainer.core.services.model.cartridge_measurement import train_cartridge
from model_trainer.core.services.model.cartridge_pool_plans import BaseLoraSweepPlan
from model_trainer.core.types import CacheCapableLMProto

_log = get_logger(__name__)

#: Seed for the LoRA's own training draw. Sits in the gap between the
#: measurement offsets (at most 48 under the recorded plans) and the
#: crowding pool's seeds (61 up).
LORA_TRAIN_SEED = 53

#: First seed of the crowding pool; member (corpus j, variant m) trains from
#: ``POOL_SEED_BASE + j * pool_members_per_corpus + m``. Chosen past every
#: seed anything else in this measurement uses.
POOL_SEED_BASE = 61


def train_crowding_pool(
    base: CacheCapableLMProto,
    pool_trains: Sequence[Sequence[torch.Tensor]],
    plan: BaseLoraSweepPlan,
) -> tuple[CartridgeSlots, ...]:
    """Train the frozen cartridges an adapter learns to read behind.

    Args:
        base: The plain base the pool trains in front of.
        pool_trains: One training-window sequence per pool corpus.
        plan: Supplies the slot count, schedule and variants per corpus.

    Returns:
        ``pool_members_per_corpus`` cartridges per corpus, nested (corpus,
        variant) in that order.
    """
    return tuple(
        train_cartridge(
            base,
            pool_train,
            num_slots=plan["slots"],
            seed=POOL_SEED_BASE + position * plan["pool_members_per_corpus"] + member,
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
        )
        for position, pool_train in enumerate(pool_trains)
        for member in range(plan["pool_members_per_corpus"])
    )


def _wrapped(base: CacheCapableLMProto, plan: BaseLoraSweepPlan) -> CacheCapableLMProto:
    """Wrap the base in the plan's LoRA.

    Args:
        base: The base to adapt. PEFT injects into its module tree, so after
            this call the base itself answers as the adapted model.
        plan: Supplies the rank and scaling.

    Returns:
        The adapted model, cache-capable.

    Raises:
        AppError: Propagated from the PEFT layer.
    """
    return require_cache_capable(
        strategy_hooks.Hooks.create_peft_model(
            base,
            r=plan["lora_rank"],
            lora_alpha=plan["lora_alpha"],
            lora_dropout=0.0,
            target_modules=target_modules_for(plan["model_id"]),
            bias="none",
        )
    )


def language_modeling_adapter(
    base: CacheCapableLMProto,
    pool_trains: Sequence[Sequence[torch.Tensor]],
    plan: BaseLoraSweepPlan,
) -> tuple[CacheCapableLMProto, tuple[Observation, ...]]:
    """Adapt the base to do language modeling behind drawn composed cartridges.

    Args:
        base: The plain base. Consumed: it is the module PEFT adapts.
        pool_trains: One training-window sequence per pool corpus; also the
            adapter's training text, concatenated in pool order.
        plan: The recorded plan whose crowding pool and LoRA this builds.

    Returns:
        ``(adapted, rows)``: the frozen adapted base, and one
        ``lora-train-epoch-{i}_loss`` row per epoch so the record shows the
        adaptation converged rather than asserting it did.

    Raises:
        ValueError: Propagated from the crowded-prefix model.
        AppError: Propagated from the PEFT layer.
    """
    crowding_pool = train_crowding_pool(base, pool_trains, plan)
    adapted = _wrapped(base, plan)
    lora_corpus = [window for pool_train in pool_trains for window in pool_train]
    epoch_losses = train_composition_lora(
        adapted,
        crowding_pool,
        lora_corpus,
        max_drawn=plan["max_drawn"],
        seed=LORA_TRAIN_SEED,
        epochs=plan["lora_epochs"],
        learning_rate=plan["lora_learning_rate"],
    )
    freeze_adapted(adapted)
    for position, loss in enumerate(epoch_losses):
        _log.info("lora epoch %d mean loss %.4f", position, loss)
    return adapted, tuple(
        Observation(name=f"lora-train-epoch-{position}_loss", value=loss)
        for position, loss in enumerate(epoch_losses)
    )


def crowd_invariance_adapter(
    base: CacheCapableLMProto,
    pool_trains: Sequence[Sequence[torch.Tensor]],
    plan: BaseLoraSweepPlan,
    *,
    device: str,
) -> tuple[CacheCapableLMProto, tuple[Observation, ...]]:
    """Distil the base to hold its own predictions invariant under a crowd.

    Args:
        base: The plain base. Consumed: it is the module PEFT adapts.
        pool_trains: One training-window sequence per pool corpus. Member
            ``i`` of the crowding pool draws its target windows from the
            corpus it was trained on.
        plan: The recorded plan whose crowding pool and LoRA this builds.
        device: Where the teacher is placed.

    Returns:
        ``(adapted, rows)``: the frozen adapted base, and one
        ``invariance-train-epoch-{i}_kl`` row per epoch.

    Raises:
        ValueError: Propagated from the invariance trainer.
        AppError: Propagated from the PEFT and model-loading layers.
    """
    crowding_pool = train_crowding_pool(base, pool_trains, plan)
    # A SECOND plain instance for the teacher: PEFT injects its adapters
    # into the wrapped module tree, so after adaptation the one loaded base
    # cannot also answer as the un-adapted base. Frozen before first use --
    # the teacher is a fixed reference, and a teacher that could drift under
    # the student's optimizer would make the objective chase itself.
    teacher_base = require_cache_capable(
        hf_hooks.Hooks.load_hf_model(plan["model_id"], quantization_for(plan["model_id"]))
    )
    teacher_base.to(device)
    freeze_adapted(teacher_base)
    adapted = _wrapped(base, plan)
    # member_windows[i] is the corpus pool[i] was trained on: the crowding
    # pool nests (corpus, variant), so members of one corpus share one
    # window list by reference.
    member_windows = [
        pool_trains[position]
        for position in range(len(pool_trains))
        for _member in range(plan["pool_members_per_corpus"])
    ]
    epoch_kls = train_composition_lora_invariant(
        adapted,
        teacher_base,
        crowding_pool,
        member_windows,
        max_drawn=plan["max_drawn"],
        seed=LORA_TRAIN_SEED,
        epochs=plan["lora_epochs"],
        learning_rate=plan["lora_learning_rate"],
    )
    freeze_adapted(adapted)
    for position, loss in enumerate(epoch_kls):
        _log.info("invariance epoch %d mean kl %.6f", position, loss)
    return adapted, tuple(
        Observation(name=f"invariance-train-epoch-{position}_kl", value=loss)
        for position, loss in enumerate(epoch_kls)
    )


__all__ = [
    "LORA_TRAIN_SEED",
    "POOL_SEED_BASE",
    "crowd_invariance_adapter",
    "language_modeling_adapter",
    "train_crowding_pool",
]
