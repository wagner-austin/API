"""Measure whether the recorded composition repairs transfer to DISPOSITIONS.

THE QUESTION (board task ``83c25b86``, its title). Crowd-invariance
distillation repaired the content half of crowded-prefix interference between
CORPUS compartments -- gpt2 n8 +49.6% and gpt2-medium n8 +38.1% where the
language-modeling objective recorded -79.4%. The naive trait grid
(``cartridge_trait_sweep``) measures what composing trait cartridges costs;
this sweep runs the three levers the corpus arc recorded against that cost,
each as its own plan and its own record:

* the diverse-companion recipe on the plain base;
* the base-side LoRA trained to do language modeling behind a crowd;
* the base-side LoRA distilled to crowd-invariance.

THE LEVERS ARE THE RECORDED ONES, not reimplementations. The pool corpora,
the crowding pool's seed formula, the adapter's objective and its schedule
come from the corpus arc's own plan rows, by identity
(:mod:`~model_trainer.core.services.model.cartridge_trait_repair_plans`), and
the adapter is built by the same function the corpus sweeps call
(:mod:`~model_trainer.cli.cartridge_crowd_adapters`). The record carries the
adapter's epoch rows under the recorded names, so a reader can hold them
against the corpus records' rows and see the adapter is the same adapter.

WHAT DIFFERS FROM THE CORPUS SWEEPS, all forced: the compartments are traits
scored by the contrastive instrument, the counts are (2, 4) because the
admissible roster holds six traits and eight compartments would need eight,
and the plain base has no plain family here because the naive trait grid
already is it.

IN SEED BLOCKS, AND SHARDABLE, exactly as the naive grid is and with the same
three flags. The grid's construction -- gates, pool, adapter, families, cells
-- is :mod:`~model_trainer.cli.cartridge_trait_repair_grid`; this module runs
one straight, as a shard, or as a merge, and reduces it into the record.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

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

from model_trainer.cli import _trait_hooks
from model_trainer.cli._sweep_shard_flags import SHARD_FLAGS, read_shard_role
from model_trainer.cli.cartridge_benchmark import sweep_observations
from model_trainer.cli.cartridge_trait_repair_grid import repair_grid
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.replicated_measurement import noise_floor
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.cartridge_plans import require_cartridge_plan
from model_trainer.core.services.model.cartridge_sweep_shards import (
    SweepMerge,
    SweepShard,
    assembled_cells,
    gather_blocks,
    measure_shard,
    seed_blocks,
)
from model_trainer.core.services.model.cartridge_trait_repair_plans import (
    TRAIT_REPAIR_SWEEP_EXPERIMENT,
    TraitRepairPlan,
    trait_repair_plan_label,
)
from model_trainer.core.services.model.trait_arms import (
    trait_arm_observations,
    trait_cell_observations,
)
from model_trainer.core.services.model.trait_rebuild import rebuild_trait_arm, rebuild_trait_cell

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
POOL_CORPORA_FLAG = "--pool-corpora"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (PLAN_FLAG, CORPUS_FLAG, POOL_CORPORA_FLAG, DEVICE_FLAG, OUT_FLAG, *SHARD_FLAGS)


def measure_grid(
    plan: TraitRepairPlan,
    *,
    plan_name: str,
    corpus: pathlib.Path,
    pool_corpora: Sequence[pathlib.Path],
    device: str,
    checkpoints: pathlib.Path,
    merge: SweepMerge | None,
) -> tuple[tuple[Observation, ...], str]:
    """Build the plan's base, then run or assemble every cell and reduce them.

    Args:
        plan: The measurement to run.
        plan_name: Which plan this is, so the checkpoint can tell one plan's
            cells from another's.
        corpus: Directory holding one authored JSON file per trait.
        pool_corpora: The recorded pool corpora.
        device: Device to measure on.
        checkpoints: Directory holding this run's checkpoint.
        merge: The shards to assemble from, or None to measure every block.

    Returns:
        ``(observations, digest)``: the gates' rows, the adapter's epoch rows,
        one ``companion-cross`` arm per pool member, every family's cell at
        every count with its controls, and each family's noise floor and step
        verdicts -- every arm over the whole plan's seeds.

    Raises:
        ValueError: Propagated from the grid's construction.
        AppError: Propagated from the grid's construction, from the shards
            when a merge's are incomplete or foreign, and from the checkpoint
            layer.
    """
    grid = repair_grid(
        plan, plan_name=plan_name, corpus=corpus, pool_corpora=pool_corpora, device=device
    )
    trait = plan["trait"]
    seeds = trait["seeds"]
    blocks = len(seed_blocks(seeds))
    produced = assembled_cells(
        checkpoints,
        merge=merge,
        measurement=f"trait-repair-{plan_name}",
        inputs_digest=grid["inputs_digest"],
        cells=grid["cells"],
    )

    observations: list[Observation] = list(grid["head_rows"])
    cross_rows = gather_blocks(produced, "companion-cross", blocks=blocks)
    for member in range(grid["pool_size"]):
        observations.extend(
            trait_arm_observations(
                rebuild_trait_arm(
                    cross_rows, name=f"{grid['trait']}-companion-cross-{member}", seeds=seeds
                )
            )
        )
    for family in grid["families"]:
        composed_arms = []
        for count in trait["compartment_counts"]:
            arm = f"{grid['trait']}-{family}-n{count}"
            cell = rebuild_trait_cell(
                gather_blocks(produced, f"{family}-n{count}", blocks=blocks),
                arm=arm,
                partners=count - 1,
                seeds=seeds,
            )
            observations.extend(trait_cell_observations(arm, cell))
            composed_arms.append(cell["composed"]["expression"])
        floor = noise_floor(composed_arms)
        observations.append(Observation(name=f"{family}_composed_noise_floor", value=floor))
        observations.extend(sweep_observations(composed_arms, floor))
    return tuple(observations), grid["digest"]


def measure_repair_shard(
    plan: TraitRepairPlan,
    *,
    plan_name: str,
    corpus: pathlib.Path,
    pool_corpora: Sequence[pathlib.Path],
    device: str,
    shard: SweepShard,
) -> None:
    """Measure one shard's share of the plan and leave it for the merge.

    Args:
        plan: The measurement to run.
        plan_name: Which plan this is.
        corpus: Directory holding one authored JSON file per trait.
        pool_corpora: The recorded pool corpora.
        device: Device to measure on.
        shard: Which share to measure.

    Raises:
        ValueError: Propagated from the grid's construction.
        AppError: Propagated from the grid's construction and the shard layer.
    """
    grid = repair_grid(
        plan, plan_name=plan_name, corpus=corpus, pool_corpora=pool_corpora, device=device
    )
    directory = measure_shard(
        shard,
        measurement=f"trait-repair-{plan_name}",
        inputs_digest=grid["inputs_digest"],
        cells=grid["cells"],
    )
    _log.info("trait repair shard %d of %d -> %s", shard["index"], shard["count"], directory)


def trait_repair_sweep_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    pool_corpora: Sequence[pathlib.Path],
    device: str,
    checkpoints: pathlib.Path,
    merge: SweepMerge | None,
) -> RunRecord:
    """Pin determinism, run or assemble the grid, and record it.

    Args:
        plan_name: Which plan to run.
        corpus: Directory holding one authored JSON file per trait.
        pool_corpora: The recorded pool corpora.
        device: Device to measure on.
        checkpoints: Directory holding this run's checkpoint.
        merge: The shards to assemble from, or None for a straight run.

    Returns:
        The record.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        ValueError: Propagated from :func:`measure_grid`.
        AppError: Propagated from :func:`measure_grid`.
    """
    plan = require_cartridge_plan(_trait_hooks.trait_repair_plans(), plan_name)
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )
    observations, digest = measure_grid(
        plan,
        plan_name=plan_name,
        corpus=corpus,
        pool_corpora=pool_corpora,
        device=device,
        checkpoints=checkpoints,
        merge=merge,
    )
    return run_record(
        experiment=TRAIT_REPAIR_SWEEP_EXPERIMENT,
        label=trait_repair_plan_label(plan_name, plan, digest=digest),
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
        0 once the record -- or, for a shard, its checkpoint -- is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, a
            required flag is absent, the shard flags are not one of their
            three forms, a shard is given ``--out``, or the pool layout is
            refused.
        KeyError: If the plan name is unknown.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    role = read_shard_role(parsed)
    plan_name = cli_args.require_flag(parsed, PLAN_FLAG)
    corpus = pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG))
    device = cli_args.require_flag(parsed, DEVICE_FLAG)
    pool = [
        pathlib.Path(entry)
        for entry in cli_args.require_flag(parsed, POOL_CORPORA_FLAG).split(",")
        if entry
    ]

    shard = role["shard"]
    if shard is not None:
        if OUT_FLAG in parsed:
            raise ValueError(f"a shard writes a checkpoint, not a record; drop {OUT_FLAG}")
        plan = require_cartridge_plan(_trait_hooks.trait_repair_plans(), plan_name)
        fingerprint = capture_run_fingerprint(
            device, probe_determinism(device, remove_split_k=False, math_attention=False)
        )
        _log.info("trait repair shard on %s", describe_run_fingerprint(fingerprint))
        measure_repair_shard(
            plan,
            plan_name=plan_name,
            corpus=corpus,
            pool_corpora=pool,
            device=device,
            shard=shard,
        )
        return 0

    # A DERIVED checkpoint path, for the reason the trait sweep gives: a
    # second flag could disagree with this one.
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    record = trait_repair_sweep_run_record(
        plan_name,
        corpus=corpus,
        pool_corpora=pool,
        device=device,
        checkpoints=out.parent / "checkpoints",
        merge=role["merge"],
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")
    _log.info(
        "trait repair sweep %s %s -> %s",
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
        service_name="cartridge-trait-repair-sweep",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "entrypoint",
    "main",
    "measure_grid",
    "measure_repair_shard",
    "trait_repair_sweep_run_record",
]


# Without this, `python -m model_trainer.cli.cartridge_trait_repair_sweep`
# imports the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
