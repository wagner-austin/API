"""Measure whether a composition-trained base rescues crowded prefixes.

THE ARM THIS RUNS (board task ``6c752568``). The cartridge-side arc ended
with a settled attribution: the residual many-compartment cost is
STRUCTURAL -- the base was never trained to read a crowded prefix -- and it
inverts with depth. This sweep adapts the base itself: a small LoRA on the
attention projections trains to do language modeling behind a DRAWN number
of frozen composed cartridges, then the RECORDED grid's own arms run with
the adapted base underneath.

TWO CELL FAMILIES, TWO QUESTIONS. ``lora-plain-n{count}`` trains ordinary
cartridges against the adapted base and composes them -- does base-side
training alone rescue composition (naive baseline: n4 -45.4%, n8 -7.0%)?
``lora-diverse-n{count}`` trains diverse-companioned cartridges against the
adapted base -- do the two sides compose (diverse baseline: n4 +55.5%, n8
+28.0%)? Every alone arm is measured against the ADAPTED base, so the
solo-cost axis prices the LoRA itself.

THE CONTAMINATION WALL. The LoRA and its crowding pool train ONLY on the
pool corpora, which must be disjoint from the primary, the partners, and
each other -- the same refusals the diverse sweep enforces, plus a
pool-count check against the plan. The adapted base never sees a measured
corpus before measurement.

SEED GEOGRAPHY, so nothing collides: measurement offsets reach
``seed + (COMPANION_SEED_STRIDE + j) * len(seeds)`` (at most 48 under the
recorded plans); the LoRA trains at
:data:`~model_trainer.cli.cartridge_crowd_adapters.LORA_TRAIN_SEED` (53);
crowding pool members start at
:data:`~model_trainer.cli.cartridge_crowd_adapters.POOL_SEED_BASE` (61). The
adapter itself is built by
:func:`~model_trainer.cli.cartridge_crowd_adapters.language_modeling_adapter`,
which the trait-repair sweep drives too.
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

from model_trainer.cli import _measurement_hooks, _test_hooks
from model_trainer.cli.cartridge_composition_sweep import staged_partner_trains
from model_trainer.cli.cartridge_crowd_adapters import language_modeling_adapter
from model_trainer.cli.cartridge_lora_families import lora_arm_families
from model_trainer.cli.cartridge_lora_policy import quantization_for
from model_trainer.cli.cartridge_pool_provider import SeededPoolProvider
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies.cartridge import (
    require_cache_capable,
)
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_plans import (
    corpus_digest,
    digest_parts,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_pool_plans import (
    BASE_LORA_SWEEP_EXPERIMENT,
    BaseLoraSweepPlan,
    base_lora_sweep_label,
)

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
OTHER_CORPORA_FLAG = "--other-corpora"
POOL_CORPORA_FLAG = "--pool-corpora"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (
    PLAN_FLAG,
    CORPUS_FLAG,
    OTHER_CORPORA_FLAG,
    POOL_CORPORA_FLAG,
    DEVICE_FLAG,
    OUT_FLAG,
)


def _require_admissible_corpora(
    plan: BaseLoraSweepPlan,
    *,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    pool_corpora: Sequence[pathlib.Path],
) -> None:
    """Refuse a corpus layout the measurement could not honestly run on.

    Args:
        plan: The measurement being run.
        corpus: The primary corpus.
        other_corpora: Composition partners.
        pool_corpora: The LoRA's own corpora, one per companion.

    Raises:
        ValueError: If too few other corpora are supplied, the pool count
            mismatches the plan, or a pool corpus repeats or overlaps the
            measured corpora.
    """
    largest = max(plan["compartment_counts"])
    if len(other_corpora) < largest - 1:
        raise ValueError(
            f"the plan composes up to {largest} compartments, which needs "
            f"{largest - 1} other corpora; {len(other_corpora)} supplied"
        )
    require_held_out_pool(plan, pool_corpora=pool_corpora, measured=[corpus, *other_corpora])


def require_held_out_pool(
    plan: BaseLoraSweepPlan,
    *,
    pool_corpora: Sequence[pathlib.Path],
    measured: Sequence[pathlib.Path],
) -> None:
    """Refuse a crowd pool that is the wrong size or is not held out.

    THE CONTAMINATION WALL, with one owner. Every sweep that trains a crowding
    pool, an adapter or a companion pool on these corpora -- the corpus grid's
    LoRA sweeps and the trait grid's repair sweep -- must keep them disjoint
    from everything it measures, or the adapted base carries the answer.

    Args:
        plan: Supplies the pool size.
        pool_corpora: The pool's corpora, one per companion.
        measured: Every corpus the sweep scores or composes.

    Raises:
        ValueError: If the pool count mismatches the plan, or a pool corpus
            repeats or is also measured.
    """
    if len(pool_corpora) != plan["max_companions"]:
        raise ValueError(
            f"the plan adapts against a pool of {plan['max_companions']} corpora; "
            f"{len(pool_corpora)} supplied"
        )
    names = [str(entry) for entry in pool_corpora]
    if len(set(names)) != len(names):
        raise ValueError(
            f"the pool corpora repeat ({', '.join(names)}); a repeated corpus "
            f"narrows the crowd the base learns to read"
        )
    measured_names = {str(entry) for entry in measured}
    overlapping = [name for name in names if name in measured_names]
    if overlapping:
        raise ValueError(
            f"the pool corpora {', '.join(overlapping)} are also measured "
            f"corpora; a base adapted on text it is later measured against "
            f"would carry the answer in its LoRA -- supply corpora held out "
            f"from the composition"
        )


def measure_grid(
    plan: BaseLoraSweepPlan,
    *,
    plan_name: str,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    pool_corpora: Sequence[pathlib.Path],
    device: str,
    checkpoints: pathlib.Path,
) -> tuple[tuple[Observation, ...], str]:
    """Adapt the base, then run both cell families against it.

    RESUMABLE PER ARM FAMILY AND COMPARTMENT COUNT. A ``(family, count)``
    cell is the smallest unit whose rows are complete -- it trains
    ``len(seeds) * count`` cartridges and reduces them into arms -- and on
    the largest plans it is also hours, which is what an eviction takes.

    WHAT A RESUME PAYS AGAIN, STATED PLAINLY. The setup above the cells is
    NOT checkpointed: the corpora are re-read, the base is re-loaded, the
    crowding pool is re-trained and the LoRA is re-trained over it. That is
    deliberate. All of it is deterministic -- every training call re-seeds
    from its own seed -- so reproducing it costs time but cannot change a
    number, and serialising an adapter plus a pool of cartridges would be a
    weights checkpoint, which is a different and much larger contract. The
    cells are where the hours are; the setup is the toll a resume pays.

    Args:
        plan: The measurement to run.
        plan_name: Which plan this is, so the checkpoint can tell one plan's
            cells from another's.
        corpus: Directory of markdown documents whose retention is the
            finding.
        other_corpora: Composition partners, in the order the counts consume
            them.
        pool_corpora: The adaptation corpora, one per companion; disjoint
            from everything measured.
        device: Device to measure on.
        checkpoints: Directory holding this sweep's checkpoint.

    Returns:
        ``(observations, digest)``: the LoRA's epoch losses, one
        ``lora-companion-cross-{j}`` arm per measurement-pool member, and
        both cell families' arms with per-family floors.

    Raises:
        ValueError: Propagated from :func:`_require_admissible_corpora`.
        AppError: Propagated from the corpus, PEFT and measurement layers,
            and from the checkpoint layer when a checkpoint on disk
            describes different inputs.
    """
    _require_admissible_corpora(
        plan,
        corpus=corpus,
        other_corpora=other_corpora,
        pool_corpora=pool_corpora,
    )
    largest = max(plan["compartment_counts"])

    documents = _test_hooks.read_corpus_documents(corpus)
    digest = corpus_digest(documents)
    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(plan["model_id"])
    encoded = [tokenizer.encode(document) for document in documents]
    train, held_out = split_by_stride(
        build_windows(encoded, window=plan["window"], device=device),
        held_out_stride=plan["held_out_stride"],
    )
    _log.info(
        "primary corpus %s: %d documents, %d train / %d held-out windows",
        corpus,
        len(documents),
        len(train),
        len(held_out),
    )

    other_trains, other_digests = staged_partner_trains(
        other_corpora[: largest - 1],
        tokenizer=tokenizer,
        window=plan["window"],
        held_out_stride=plan["held_out_stride"],
        required=len(train),
        device=device,
    )
    pool_trains, pool_digests = staged_partner_trains(
        pool_corpora,
        tokenizer=tokenizer,
        window=plan["window"],
        held_out_stride=plan["held_out_stride"],
        required=len(train),
        device=device,
    )

    base = require_cache_capable(
        hf_hooks.Hooks.load_hf_model(plan["model_id"], quantization_for(plan["model_id"]))
    )
    base.to(device)

    adapted, epoch_rows = language_modeling_adapter(base, pool_trains, plan)

    observations: list[Observation] = [
        Observation(name="slots_per_cartridge", value=float(plan["slots"])),
        Observation(name="max_drawn", value=float(plan["max_drawn"])),
        *epoch_rows,
    ]

    provider = SeededPoolProvider(
        adapted,
        pool_trains,
        slots=plan["slots"],
        seeds=plan["seeds"],
        epochs=plan["epochs"],
        learning_rate=plan["learning_rate"],
    )
    observations.extend(
        lora_arm_families(
            adapted,
            plan=plan,
            pool_for_seed=provider.pool,
            pool_size=len(pool_trains),
            train=train,
            other_trains=other_trains,
            held_out=held_out,
            checkpoints=checkpoints,
            measurement=f"base-lora-{plan_name}",
            # THE LABEL IS THE IDENTITY, and using it here is deliberate
            # reuse rather than a second list of knobs to keep in step. It
            # already carries every measurement field and the primary digest,
            # and it is what two runs are paired by -- so if two runs would
            # carry the same label they are the same measurement, and if they
            # would not, one must not resume the other. The partner digests
            # are appended because the label does not reach them.
            inputs_digest=digest_parts(
                [
                    base_lora_sweep_label(plan_name, plan, digest=digest),
                    *other_digests,
                    *pool_digests,
                ]
            ),
        )
    )
    return tuple(observations), digest


def base_lora_sweep_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    pool_corpora: Sequence[pathlib.Path],
    device: str,
    checkpoints: pathlib.Path,
) -> RunRecord:
    """Pin determinism, run the grid, and record it.

    Args:
        plan_name: Which plan to run.
        corpus: Directory of markdown documents whose retention is measured.
        other_corpora: Composition partners.
        pool_corpora: The adaptation corpora.
        device: Device to measure on.
        checkpoints: Directory holding this sweep's checkpoint, so an evicted
            run resumes at its last completed cell rather than at zero.

    Returns:
        The record.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        ValueError: Propagated from :func:`measure_grid`.
        AppError: Propagated from the corpus, PEFT and measurement layers.
    """
    plan = require_cartridge_plan(_measurement_hooks.base_lora_sweep_plans(), plan_name)
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )
    observations, digest = measure_grid(
        plan,
        plan_name=plan_name,
        corpus=corpus,
        other_corpora=other_corpora,
        pool_corpora=pool_corpora,
        device=device,
        checkpoints=checkpoints,
    )
    return run_record(
        experiment=BASE_LORA_SWEEP_EXPERIMENT,
        label=base_lora_sweep_label(plan_name, plan, digest=digest),
        fingerprint=fingerprint,
        observations=observations,
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run one plan and write the record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, a
            required flag is absent, or the corpus layout is refused.
        KeyError: If the plan name is unknown.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    others = [
        pathlib.Path(entry)
        for entry in cli_args.require_flag(parsed, OTHER_CORPORA_FLAG).split(",")
        if entry
    ]
    pool = [
        pathlib.Path(entry)
        for entry in cli_args.require_flag(parsed, POOL_CORPORA_FLAG).split(",")
        if entry
    ]
    # RESOLVED BEFORE THE RUN, because the checkpoint directory derives from
    # where the artifact goes. A DERIVED path rather than a flag of its own:
    # a second flag could disagree with this one, and a resumed sweep would
    # then look for its work in a directory nothing wrote to.
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = base_lora_sweep_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        other_corpora=others,
        pool_corpora=pool,
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
        checkpoints=out.parent / "checkpoints",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "base lora sweep %s %s -> %s",
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
        service_name="cartridge-base-lora-sweep",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "base_lora_sweep_run_record",
    "entrypoint",
    "main",
    "measure_grid",
    "require_held_out_pool",
]


# Without this, `python -m model_trainer.cli.cartridge_base_lora_sweep`
# imports the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
