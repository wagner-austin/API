"""What the two base-side LoRA sweeps measure, once instead of twice.

``cartridge_base_lora_sweep`` and ``cartridge_content_lora_sweep`` differ in
exactly one thing: how the adapter is trained. The first trains it to do
language modeling behind drawn composed cartridges; the second trains it to
hold the base's own predictions invariant under them. Everything AFTER that
-- the companion-cross arms, the two cell families at each compartment count,
and the per-family noise floors -- was byte-identical between them, sixty-three
lines each, and this module is those sixty-three lines with one owner.

WHY IT WAS WORTH MOVING RATHER THAN LEAVING. Two copies of a measurement are
two copies of its arm names, and the arm name is what pairs a run against the
recorded ladder. A change to one copy does not fail: it produces a complete
record whose rows no longer line up with the other sweep's, and the two are
compared precisely because they are supposed to differ only in the adapter.
The pool provider they both used moved out too, further, into
:mod:`model_trainer.cli.cartridge_pool_provider` -- the diverse sweep had a
third copy of it, so it belongs to none of the three.

THE CELLS AND THE CHECKPOINT ARE HERE FOR THE SAME REASON. Both sweeps run on
``free-gpu``, where ``PreemptMode=CANCEL`` means an evicted job is killed
outright and Slurm resubmits nothing, and both are plans measured in hours.
Writing the resume discipline once means the two cannot drift into resuming
differently, which is the failure that would be hardest to see: a sweep that
skips the wrong cell still emits a complete table.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import torch
from platform_core.logging import get_logger
from platform_core.run_record import Observation

from model_trainer.cli.cartridge_benchmark import sweep_observations
from model_trainer.cli.cartridge_companion_sweep import (
    cell_observations,
)
from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    gain_observations,
    noise_floor,
    per_seed_observations,
    replicate,
    replicated_from_observations,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.cartridge_measurement import (
    held_out_gain,
    measure_composition_scaling,
)
from model_trainer.core.services.model.cartridge_pool_plans import BaseLoraSweepPlan
from model_trainer.core.services.model.cartridge_sweep_checkpoint import (
    bind_cells,
    checkpointed_cells,
)
from model_trainer.core.services.model.cartridge_varied import (
    CompanionPoolProviderProto,
    measure_varied_companioned_scaling,
)
from model_trainer.core.types import CacheCapableLMProto

_log = get_logger(__name__)


def lora_arm_families(
    adapted: CacheCapableLMProto,
    *,
    plan: BaseLoraSweepPlan,
    pool_for_seed: CompanionPoolProviderProto,
    pool_size: int,
    train: Sequence[torch.Tensor],
    other_trains: Sequence[Sequence[torch.Tensor]],
    held_out: Sequence[torch.Tensor],
    checkpoints: pathlib.Path,
    measurement: str,
    inputs_digest: str,
) -> tuple[Observation, ...]:
    """Run both cell families against an adapted base, resumably.

    RESUMABLE PER ARM FAMILY AND COMPARTMENT COUNT. A ``(family, count)``
    cell is the smallest unit whose rows are complete -- it trains
    ``len(seeds) * count`` cartridges and reduces them into arms -- and on
    the larger plans it is also hours, which is what an eviction takes. The
    companion-cross arms are one cell for all members together, because
    ``pool_for_seed`` trains a seed's whole pool in one call: splitting per
    member would retrain the pool once per member and discard all but one of
    them each time.

    Args:
        adapted: The frozen, adapted base every arm is measured against.
        plan: The plan being run; supplies the seeds, slot count, epochs,
            learning rate, compartment counts and companion probability.
        pool_for_seed: One replicate's frozen companion pool, cached by the
            caller so cells sharing a seed share a pool by identity.
        pool_size: How many members a pool has, which is how many
            companion-cross arms there are.
        train: Training windows for the cartridge whose retention is the
            finding.
        other_trains: One window sequence per composition partner, in the
            order the counts consume them.
        held_out: Items every arm is scored on.
        checkpoints: Directory holding this sweep's checkpoint.
        measurement: Names this sweep, so one sweep's cells can never be
            adopted by another.
        inputs_digest: Digest of everything the cells are measured over.

    Returns:
        The companion-cross arms, both families' cells at every compartment
        count, and each family's noise floor -- in the order the record has
        always carried them.

    Raises:
        AppError: Propagated from the measurement layer, and from the
            checkpoint layer when a checkpoint describes different inputs.
    """

    def _companion_cross() -> tuple[Observation, ...]:
        """Score every measurement-pool member alone on the primary held-out.

        Returns:
            Each member's arm, with its mean, spread and per-seed gains.
        """
        companion_gains: list[list[tuple[int, float]]] = [[] for _ in range(pool_size)]
        for seed in plan["seeds"]:
            for member, slots in enumerate(pool_for_seed(seed)):
                companion_gains[member].append(
                    (seed, held_out_gain(CartridgeModel(base=adapted, slots=slots), held_out))
                )
        scored: list[Observation] = []
        for member, gains in enumerate(companion_gains):
            arm = replicate(f"lora-companion-cross-{member}", gains)
            _log.info("lora-companion-cross-%d: %+.4f on the primary held-out", member, arm["mean"])
            scored.extend(gain_observations(arm))
            scored.extend(per_seed_observations(arm))
        return tuple(scored)

    def _plain(count: int) -> tuple[Observation, ...]:
        """Measure the plain composition arm at one compartment count.

        Args:
            count: How many cartridges are composed.

        Returns:
            Every arm's mean, spread and per-seed gains, the retention ratio
            where it is readable, and the interference verdict.
        """
        plain_name = f"lora-plain-n{count}"
        alone, composed, untrained_composed, cross = measure_composition_scaling(
            adapted,
            first_train=train,
            other_trains=other_trains[: count - 1],
            held_out=held_out,
            arm=plain_name,
            num_slots=plan["slots"],
            seeds=plan["seeds"],
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
        )
        _log.info("%s: %+.4f alone -> %+.4f composed", plain_name, alone["mean"], composed["mean"])
        return cell_observations(plain_name, alone, composed, untrained_composed, cross)

    def _diverse(count: int) -> tuple[Observation, ...]:
        """Measure the diverse-companion arm at one compartment count.

        Args:
            count: How many cartridges are composed.

        Returns:
            The cell's rows, shaped exactly as :func:`_plain`'s.
        """
        diverse_name = f"lora-diverse-n{count}"
        alone, composed, untrained_composed, cross = measure_varied_companioned_scaling(
            adapted,
            first_train=train,
            other_trains=other_trains[: count - 1],
            held_out=held_out,
            arm=diverse_name,
            num_slots=plan["slots"],
            seeds=plan["seeds"],
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
            pool_for_seed=pool_for_seed,
            companion_probability=plan["probability"],
        )
        _log.info(
            "%s: %+.4f alone -> %+.4f composed", diverse_name, alone["mean"], composed["mean"]
        )
        return cell_observations(diverse_name, alone, composed, untrained_composed, cross)

    produced = checkpointed_cells(
        checkpoints,
        measurement=measurement,
        inputs_digest=inputs_digest,
        cells=[
            ("companion-cross", _companion_cross),
            *bind_cells(
                [(f"plain-n{count}", count) for count in plan["compartment_counts"]], _plain
            ),
            *bind_cells(
                [(f"diverse-n{count}", count) for count in plan["compartment_counts"]], _diverse
            ),
        ],
    )

    # THE CELLS ARE WALKED IN THE DECLARED ORDER, not in the order they came
    # back, so a resumed sweep emits the rows a straight run would have, in
    # the same places.
    observations: list[Observation] = list(produced["companion-cross"])
    plain_arms: list[ReplicatedGain] = []
    diverse_arms: list[ReplicatedGain] = []
    for count in plan["compartment_counts"]:
        for family, arms in (("plain", plain_arms), ("diverse", diverse_arms)):
            rows = produced[f"{family}-n{count}"]
            observations.extend(rows)
            # REBUILT FROM THE ROWS RATHER THAN KEPT FROM THE CALL, because a
            # resumed cell never made the call. The rebuild is exact -- every
            # field of an arm is a function of its per-seed gains, and those
            # are in the rows -- so the floors below are computed over the
            # same arms either way.
            arms.append(
                replicated_from_observations(
                    rows, arm=f"lora-{family}-n{count}-composed", seeds=plan["seeds"]
                )
            )

    plain_floor = noise_floor(plain_arms)
    observations.append(Observation(name="lora-plain_composed_noise_floor", value=plain_floor))
    observations.extend(sweep_observations(plain_arms, plain_floor))
    diverse_floor = noise_floor(diverse_arms)
    observations.append(Observation(name="lora-diverse_composed_noise_floor", value=diverse_floor))
    observations.extend(sweep_observations(diverse_arms, diverse_floor))
    return tuple(observations)


__all__ = ["lora_arm_families"]
