"""Which repair families the trait grid runs, and what each one carries over.

THE QUESTION THESE PLANS ASK (board task ``83c25b86``): crowd-invariance
distillation repaired the CONTENT half of crowded-prefix interference between
corpus compartments; does it -- and do the two levers recorded before it --
transfer when the compartments carry a disposition? That is only asked if the
levers are the RECORDED ones, so a plan here is a reference, not a copy: its
``trait`` field IS a row of the trait table and its ``crowd`` field IS a row
of the corpus arc's LoRA tables, by identity. Nothing about the crowding
pool, the adapter or the companion recipe can drift from the record it
claims to mirror, because there is no second statement of it to drift.

WHY ``crowd_windows_per_corpus`` IS DECLARED AND NOT DERIVED. The recorded
sweeps truncated every pool corpus to the primary corpus's training-window
count -- 42 for the me-wiki corpus at window 256 and stride 4 (digest
``e2f23c635583``, counted with the recorded windowing on 2026-10-08). A trait
grid has no such primary: its compartments are authored pairs, not windows.
Declaring the recorded count is what makes the adapter the recorded adapter,
and the record proves it rather than asserting it -- each adapter's epoch rows
(``lora-train-epoch-{i}_loss``, ``invariance-train-epoch-{i}_kl``) are the
recorded records' rows when the pool is the recorded pool.

THE COUNTS ARE (2, 4), NOT THE RECORDED (4, 8). Eight compartments would need
eight traits, and the admissible roster has six; the four it does not exclude
for positional concentration are exactly what an n4 cell consumes. That is a
deviation forced by the roster, and the record states it in its label.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from typing_extensions import TypedDict

from model_trainer.core.contracts.trait_plan import TraitPlan, trait_plan_label
from model_trainer.core.services.model.cartridge_pool_plans import (
    BASE_LORA_SWEEP_PLANS,
    CONTENT_LORA_SWEEP_PLANS,
    BaseLoraSweepPlan,
)
from model_trainer.core.services.model.cartridge_trait_plans import TRAIT_SWEEP_PLANS


class TraitRepairAdapter(StrEnum):
    """Which base the repair families are measured on.

    Attributes:
        NONE: The plain base. Only the diverse-companion family runs, because
            its plain family IS the naive grid the trait sweep already ran.
        LANGUAGE_MODELING: A LoRA trained to do language modeling behind drawn
            composed cartridges -- the recorded base-LoRA lever.
        CROWD_INVARIANCE: A LoRA distilled to hold the base's own predictions
            invariant under a crowd -- the recorded content lever, and the one
            the task's title asks about.
    """

    NONE = "none"
    LANGUAGE_MODELING = "language-modeling"
    CROWD_INVARIANCE = "crowd-invariance"


class TraitRepairPlan(TypedDict):
    """One repair-family measurement on the trait grid.

    Attributes:
        trait: The trait plan whose roster, split, schedule and seeds every
            trait cartridge uses. A row of the trait table by identity.
        crowd: The recorded corpus-arc plan whose pool knobs, companion
            probability and LoRA this mirrors. A row of a LoRA table by
            identity.
        adapter: Which base the families are measured on.
        crowd_windows_per_corpus: Training windows each pool corpus is
            truncated to -- the recorded primary corpus's count, so the pool
            and adapter are the recorded ones.
    """

    trait: TraitPlan
    crowd: BaseLoraSweepPlan
    adapter: TraitRepairAdapter
    crowd_windows_per_corpus: int


#: Fixed for the reason every experiment name is: a trait-repair record
#: differenced against a trait-composition record would be two questions
#: sharing arm names.
TRAIT_REPAIR_SWEEP_EXPERIMENT = "cartridge-trait-repair"

#: The recorded primary corpus's training-window count; see the module
#: docstring for how it was counted and how the record certifies it.
_RECORDED_CROWD_WINDOWS: Final[int] = 42


def trait_repair_plan_label(name: str, plan: TraitRepairPlan, *, digest: str) -> str:
    """Build the label that identifies one repair plan's numbers on one corpus.

    Args:
        name: The plan's name.
        plan: The plan.
        digest: Digest of the trait corpora.

    Returns:
        The trait plan's label, extended with every crowd field that moves a
        number and the adapter.
    """
    crowd = plan["crowd"]
    return (
        f"{trait_plan_label(name, plan['trait'], digest=digest)}"
        f"-{plan['adapter'].value}"
        f"-w{crowd['window']}-s{crowd['held_out_stride']}-cw{plan['crowd_windows_per_corpus']}"
        f"-p{crowd['probability']}-K{crowd['max_companions']}"
        f"-R{crowd['lora_rank']}-a{crowd['lora_alpha']}-le{crowd['lora_epochs']}"
        f"-llr{crowd['lora_learning_rate']}-D{crowd['max_drawn']}"
        f"-m{crowd['pool_members_per_corpus']}"
    )


TRAIT_REPAIR_SWEEP_PLANS: Final[dict[str, TraitRepairPlan]] = {
    # The recorded diverse recipe on the plain base. Its crowd row is the
    # base-LoRA plan because that plan "matches the diverse grid on every
    # measurement field" (cartridge_pool_plans); only its pool knobs are read.
    "gpt2-traits-diverse": TraitRepairPlan(
        trait=TRAIT_SWEEP_PLANS["gpt2-traits"],
        crowd=BASE_LORA_SWEEP_PLANS["gpt2-base-lora"],
        adapter=TraitRepairAdapter.NONE,
        crowd_windows_per_corpus=_RECORDED_CROWD_WINDOWS,
    ),
    "gpt2-traits-base-lora": TraitRepairPlan(
        trait=TRAIT_SWEEP_PLANS["gpt2-traits"],
        crowd=BASE_LORA_SWEEP_PLANS["gpt2-base-lora"],
        adapter=TraitRepairAdapter.LANGUAGE_MODELING,
        crowd_windows_per_corpus=_RECORDED_CROWD_WINDOWS,
    ),
    "gpt2-traits-content-lora": TraitRepairPlan(
        trait=TRAIT_SWEEP_PLANS["gpt2-traits"],
        crowd=CONTENT_LORA_SWEEP_PLANS["gpt2-content-lora"],
        adapter=TraitRepairAdapter.CROWD_INVARIANCE,
        crowd_windows_per_corpus=_RECORDED_CROWD_WINDOWS,
    ),
}


__all__ = [
    "TRAIT_REPAIR_SWEEP_EXPERIMENT",
    "TRAIT_REPAIR_SWEEP_PLANS",
    "TraitRepairAdapter",
    "TraitRepairPlan",
    "trait_repair_plan_label",
]
