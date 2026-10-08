"""Measure what composing DISPOSITIONS costs, against the published baseline.

THE QUESTION (board task ``83c25b86``). Every compartment this arc has ever
composed is a CORPUS: the dependent variable is held-out loss on the
compartment's own text. A persona has no held-out corpus in that sense, and
the published activation-space answer is that composing dispositions is
expensive -- two steering vectors already cost a large fraction of trait
expression, and every common composition scheme degrades as vectors are added.
Nobody has asked what two CARTRIDGES cost on the same traits, and this arc has
a lever the steering literature does not: its compartments are trained.

WHAT THIS RUNS, in order, and the order is the design. The solo cell first,
because the arc stops if it fails -- at the 7B rung the corpus programme's
solo gain nearly vanished and every retention ratio computed on those records
became a division artefact. Then the steering arm at matched counts, tuned by
the published rule, which divides by nothing the cartridge produced. Then,
ONLY if the solo arm cleared its own spread, the composed cells at each
compartment count with their untrained-composed and cross controls. A failed
precondition still writes a record -- the solo and steering rows with
``solo_precondition_cleared`` at 0 -- because that failure is the first result
the task names.

THIS IS THE NAIVE GRID. The repairs -- diverse companions, a base-side LoRA,
crowd-invariance distillation -- are
:mod:`~model_trainer.cli.cartridge_trait_repair_sweep`, which reads the same
plan row so its cells subtract against these.

IN SEED BLOCKS, AND SHARDABLE. The variance pilot puts the smallest effect of
interest more seeds away than one job can measure, so every cell is one block
of seeds (:mod:`~model_trainer.core.services.model.cartridge_sweep_shards`).
A straight run measures every block itself; ``--shards DIR --shard-count N
--shard K`` measures one share and writes no record; ``--shards DIR
--shard-count N`` adopts every share and writes the record a straight run
would have. Every arm in the record is rebuilt over all seeds from its blocks'
per-seed rows, so no block's own mean or spread reaches it.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable, Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import dump_json_str
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
)
from typing_extensions import TypedDict

from model_trainer.cli import _measurement_hooks, _trait_hooks
from model_trainer.cli._sweep_shard_flags import SHARD_FLAGS, read_shard_role
from model_trainer.cli.cartridge_benchmark import sweep_observations
from model_trainer.cli.cartridge_lora_policy import quantization_for
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.replicated_measurement import ReplicatedGain, noise_floor
from model_trainer.core.contracts.trait_corpus import trait_corpus_digest
from model_trainer.core.contracts.trait_plan import (
    TRAIT_SWEEP_EXPERIMENT,
    TraitPlan,
    trait_plan_label,
)
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_plans import require_cartridge_plan
from model_trainer.core.services.model.cartridge_sweep_checkpoint import bind_cells
from model_trainer.core.services.model.cartridge_sweep_shards import (
    SweepMerge,
    SweepShard,
    assembled_cells,
    block_cell,
    gather_blocks,
    measure_shard,
    seed_blocks,
)
from model_trainer.core.services.model.steering_vectors import require_steerable
from model_trainer.core.services.model.trait_arms import (
    measure_trait_composition,
    measure_trait_solo,
    measure_trait_steering,
    solo_precondition_cleared,
    steering_observations,
    trait_arm_observations,
    trait_cell_observations,
)
from model_trainer.core.services.model.trait_families import plain_trait_build
from model_trainer.core.services.model.trait_rebuild import rebuild_trait_arm, rebuild_trait_cell
from model_trainer.core.services.model.trait_roster import gate_primary_trait, prepare_traits

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (PLAN_FLAG, CORPUS_FLAG, DEVICE_FLAG, OUT_FLAG, *SHARD_FLAGS)


class _TraitGrid(TypedDict):
    """One plan's cells, bound to a loaded base, before any of them runs.

    Attributes:
        label: The plan's label, every phase's inputs digest.
        digest: The trait corpora's digest.
        trait: The primary trait's name, which prefixes every arm.
        gate_rows: The power gates' rows.
        anchor: The solo blocks, then every steering configuration.
        composed: Every compartment count's blocks.
    """

    label: str
    digest: str
    trait: str
    gate_rows: tuple[Observation, ...]
    anchor: list[tuple[str, Callable[[], Sequence[Observation]]]]
    composed: list[tuple[str, Callable[[], Sequence[Observation]]]]


def describe_solo_precondition(solo: ReplicatedGain, untrained: ReplicatedGain) -> str:
    """State the solo verdict in the words a reader of the job log needs.

    BOTH ARMS, ALWAYS. A solo gain that merely matches an untrained prefix is
    the other way this fails, and a reader needs both numbers to tell the two
    apart; the verdict itself is decided by
    :func:`~model_trainer.core.services.model.trait_arms.solo_precondition_cleared`
    and lands in the record as ``solo_precondition_cleared``.

    Args:
        solo: The trained solo arm's expression reading.
        untrained: The untrained-prefix control's, under the same seeds.

    Returns:
        One sentence naming the verdict, the gain, its spread and the control.
    """
    verdict = "CLEARED" if solo_precondition_cleared(solo) else "FAILED"
    return (
        f"solo precondition {verdict}: {solo['arm']!r} expressed its trait by "
        f"{solo['mean']:+.4f} across {len(solo['seeds'])} seed(s) against its own per-seed "
        f"spread {solo['spread']:.4f}; the untrained-prefix control sits at "
        f"{untrained['mean']:+.4f}"
    )


def _trait_grid(
    plan: TraitPlan, *, plan_name: str, corpus: pathlib.Path, device: str
) -> _TraitGrid:
    """Read the traits, gate them, load the base, and bind every cell.

    Args:
        plan: The measurement to run.
        plan_name: Which plan this is.
        corpus: Directory holding one authored JSON file per trait.
        device: Device to measure on.

    Returns:
        The grid, with nothing measured yet.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` from the corpus layer,
            ``CARTRIDGE_QA_UNDERPOWERED`` when the pairs or the seeds cannot
            resolve what the plan declares, and ``CARTRIDGE_SEED_BLOCKS_UNEVEN``
            when the seeds do not cut into blocks.
    """
    corpora = _trait_hooks.read_trait_corpora(corpus, plan["traits"])
    digest = trait_corpus_digest(corpora)
    prepared = prepare_traits(corpora, plan, device=device)
    primary = prepared[0]
    gate_rows = gate_primary_trait(plan, primary)
    blocks = seed_blocks(plan["seeds"])

    base = require_cache_capable(
        hf_hooks.Hooks.load_hf_model(plan["model_id"], quantization_for(plan["model_id"]))
    )
    base.to(device)

    def _solo(block: tuple[int, ...]) -> tuple[Observation, ...]:
        """Measure the trait alone over one block, with the untrained prefix beside it.

        Args:
            block: The seeds to measure.

        Returns:
            Both arms' rows over the block.
        """
        trained, untrained = measure_trait_solo(
            base,
            train=primary.train,
            held_out=primary.held_out,
            arm=f"{primary.trait}-solo",
            num_slots=plan["slots"],
            seeds=block,
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
        )
        _log.info(
            "%s seeds %s: %+.4f expression, %+.4f coherence",
            trained["expression"]["arm"],
            block,
            trained["expression"]["mean"],
            trained["coherence"]["mean"],
        )
        return (*trait_arm_observations(trained), *trait_arm_observations(untrained))

    def _composed(unit: tuple[int, tuple[int, ...]]) -> tuple[Observation, ...]:
        """Measure one compartment count over one block.

        Args:
            unit: ``(count, block)``.

        Returns:
            Every arm's rows over the block.
        """
        count, block = unit
        arm = f"{primary.trait}-n{count}"
        cell = measure_trait_composition(
            base,
            build=plain_trait_build(
                base,
                first_train=primary.train,
                other_trains=[other.train for other in prepared[1:count]],
                num_slots=plan["slots"],
                seeds=block,
                seed_stride=len(plan["seeds"]),
                epochs=plan["epochs"],
                learning_rate=plan["learning_rate"],
            ),
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

    def _steering(count: int) -> tuple[Observation, ...]:
        """Measure the published intervention at one trait count.

        Args:
            count: How many trait directions are summed.

        Returns:
            The configuration's rows.
        """
        reading = measure_trait_steering(
            require_steerable(base),
            trait_trains=[other.train for other in prepared[:count]],
            held_out=primary.held_out,
            arm=f"{primary.trait}-steer-n{count}",
            module_name=plan["steering_module"],
            strengths=plan["steering_strengths"],
            coherence_bar=plan["steering_coherence_bar"],
        )
        _log.info(
            "%s at strength %g: %+.4f expression on %d pair(s), p=%.4f",
            reading["arm"],
            reading["tuning"]["strength"],
            reading["expression"]["mean_baseline"] - reading["expression"]["mean_treatment"],
            reading["expression"]["items"],
            reading["expression"]["p_value"],
        )
        return steering_observations(reading)

    # The steering arm runs at one and at every cartridge count, because its
    # solo reading is what a composed steering number is read against, and in
    # the ANCHOR phase, before the verdict: it divides by nothing the solo
    # cartridge produced, so an activation substrate that expresses the trait
    # where the cartridge failed is itself a finding.
    return _TraitGrid(
        label=trait_plan_label(plan_name, plan, digest=digest),
        digest=digest,
        trait=primary.trait,
        gate_rows=gate_rows,
        anchor=[
            *bind_cells(
                [(block_cell("solo", index), block) for index, block in enumerate(blocks)], _solo
            ),
            *bind_cells(
                [(f"steer-n{count}", count) for count in (1, *plan["compartment_counts"])],
                _steering,
            ),
        ],
        composed=bind_cells(
            [
                (block_cell(f"n{count}", index), (count, block))
                for count in plan["compartment_counts"]
                for index, block in enumerate(blocks)
            ],
            _composed,
        ),
    )


def measure_grid(
    plan: TraitPlan,
    *,
    plan_name: str,
    corpus: pathlib.Path,
    device: str,
    checkpoints: pathlib.Path,
    merge: SweepMerge | None,
) -> tuple[tuple[Observation, ...], str]:
    """Run the solo blocks and the steering arms, then the composed cells if earned.

    Args:
        plan: The measurement to run.
        plan_name: Which plan this is, so the checkpoint can tell one plan's
            cells from another's.
        corpus: Directory holding one authored JSON file per trait.
        device: Device to measure on.
        checkpoints: Directory holding this run's checkpoints.
        merge: The shards to assemble from, or None to measure every block.

    Returns:
        ``(observations, digest)``: the gates' rows, the solo precondition's
        verdict, the solo arm and its control over every seed, and one
        steering reading per matched count -- and, only when the verdict
        cleared, every composed cell with its controls and the composed
        family's noise floor and step verdicts.

    Raises:
        AppError: Propagated from :func:`_trait_grid`, from the shards when a
            merge's are incomplete or foreign, and from the checkpoint layer.
    """
    grid = _trait_grid(plan, plan_name=plan_name, corpus=corpus, device=device)
    seeds = plan["seeds"]
    blocks = len(seed_blocks(seeds))
    trait = grid["trait"]
    anchor = assembled_cells(
        checkpoints,
        merge=merge,
        measurement=f"trait-{plan_name}-anchor",
        inputs_digest=grid["label"],
        cells=grid["anchor"],
    )
    solo_rows = gather_blocks(anchor, "solo", blocks=blocks)
    trained = rebuild_trait_arm(solo_rows, name=f"{trait}-solo", seeds=seeds)
    untrained = rebuild_trait_arm(solo_rows, name=f"{trait}-solo-untrained", seeds=seeds)
    cleared = solo_precondition_cleared(trained["expression"])
    _log.info("%s", describe_solo_precondition(trained["expression"], untrained["expression"]))

    observations: list[Observation] = [
        *grid["gate_rows"],
        Observation(name="steering_coherence_bar", value=plan["steering_coherence_bar"]),
        Observation(name="solo_precondition_cleared", value=float(cleared)),
        *trait_arm_observations(trained),
        *trait_arm_observations(untrained),
    ]
    for count in (1, *plan["compartment_counts"]):
        observations.extend(anchor[f"steer-n{count}"])

    # THE ARC STOPS HERE IF THE SOLO ARM FAILED, and the record says so rather
    # than a job log: every composed retention divides by the solo gain.
    if not cleared:
        return tuple(observations), grid["digest"]

    # A SECOND CHECKPOINT, not a continuation of the first: the anchor's file
    # is deleted once its phase completes, so an eviction in this phase
    # re-measures the anchor on resume -- at a cost of minutes, and with no
    # effect on a number, since every block is a function of its seeds.
    produced = assembled_cells(
        checkpoints,
        merge=merge,
        measurement=f"trait-{plan_name}-composed",
        inputs_digest=grid["label"],
        cells=grid["composed"],
    )
    composed_arms: list[ReplicatedGain] = []
    for count in plan["compartment_counts"]:
        arm = f"{trait}-n{count}"
        cell = rebuild_trait_cell(
            gather_blocks(produced, f"n{count}", blocks=blocks),
            arm=arm,
            partners=count - 1,
            seeds=seeds,
        )
        observations.extend(trait_cell_observations(arm, cell))
        composed_arms.append(cell["composed"]["expression"])

    composed_floor = noise_floor(composed_arms)
    observations.append(Observation(name="composed_expression_noise_floor", value=composed_floor))
    observations.extend(sweep_observations(composed_arms, composed_floor))
    return tuple(observations), grid["digest"]


def measure_trait_shard(
    plan: TraitPlan, *, plan_name: str, corpus: pathlib.Path, device: str, shard: SweepShard
) -> None:
    """Measure one shard's share of both phases and leave it for the merge.

    BOTH PHASES, WITHOUT THE VERDICT. A shard holds a few of the solo blocks,
    so it cannot know whether the solo arm cleared; the merge decides, and
    composed blocks it does not need are simply never adopted.

    Args:
        plan: The measurement to run.
        plan_name: Which plan this is.
        corpus: Directory holding one authored JSON file per trait.
        device: Device to measure on.
        shard: Which share to measure.

    Raises:
        AppError: Propagated from :func:`_trait_grid` and the shard layer.
    """
    grid = _trait_grid(plan, plan_name=plan_name, corpus=corpus, device=device)
    for phase in ("anchor", "composed"):
        directory = measure_shard(
            shard,
            measurement=f"trait-{plan_name}-{phase}",
            inputs_digest=grid["label"],
            cells=grid[phase],
        )
        _log.info(
            "trait shard %d of %d, %s phase -> %s", shard["index"], shard["count"], phase, directory
        )


def trait_sweep_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    device: str,
    checkpoints: pathlib.Path,
    merge: SweepMerge | None,
) -> RunRecord:
    """Pin determinism, run or assemble the grid, and record it.

    Args:
        plan_name: Which plan to run.
        corpus: Directory holding one authored JSON file per trait.
        device: Device to measure on.
        checkpoints: Directory holding this run's checkpoints, so an evicted
            run resumes at its last completed cell rather than at zero.
        merge: The shards to assemble from, or None for a straight run.

    Returns:
        The record.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        AppError: Propagated from :func:`measure_grid`.
    """
    plan = require_cartridge_plan(_measurement_hooks.trait_sweep_plans(), plan_name)
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )
    observations, digest = measure_grid(
        plan,
        plan_name=plan_name,
        corpus=corpus,
        device=device,
        checkpoints=checkpoints,
        merge=merge,
    )
    return run_record(
        experiment=TRAIT_SWEEP_EXPERIMENT,
        label=trait_plan_label(plan_name, plan, digest=digest),
        fingerprint=fingerprint,
        observations=observations,
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one plan straight, as a shard, or as a merge.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the record -- or, for a shard, its checkpoints -- is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, a
            required flag is absent, the shard flags are not one of their
            three forms, or a shard is given ``--out``.
        KeyError: If the plan name is unknown.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    role = read_shard_role(parsed)
    plan_name = cli_args.require_flag(parsed, PLAN_FLAG)
    corpus = pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG))
    device = cli_args.require_flag(parsed, DEVICE_FLAG)

    shard = role["shard"]
    if shard is not None:
        if OUT_FLAG in parsed:
            raise ValueError(f"a shard writes checkpoints, not a record; drop {OUT_FLAG}")
        plan = require_cartridge_plan(_measurement_hooks.trait_sweep_plans(), plan_name)
        fingerprint = capture_run_fingerprint(
            device, probe_determinism(device, remove_split_k=False, math_attention=False)
        )
        _log.info("trait shard on %s", describe_run_fingerprint(fingerprint))
        measure_trait_shard(plan, plan_name=plan_name, corpus=corpus, device=device, shard=shard)
        return 0

    # RESOLVED BEFORE THE RUN, because the checkpoint directory derives from
    # where the artifact goes. A DERIVED path rather than a flag of its own: a
    # second flag could disagree with this one, and a resumed sweep would then
    # look for its work in a directory nothing wrote to.
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    record = trait_sweep_run_record(
        plan_name,
        corpus=corpus,
        device=device,
        checkpoints=out.parent / "checkpoints",
        merge=role["merge"],
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "trait sweep %s %s -> %s",
        record["label"],
        describe_run_fingerprint(record["fingerprint"]),
        out,
    )
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="cartridge-trait-sweep",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "describe_solo_precondition",
    "entrypoint",
    "main",
    "measure_grid",
    "measure_trait_shard",
    "trait_sweep_run_record",
]


# Without this, `python -m model_trainer.cli.cartridge_trait_sweep` imports the
# module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
