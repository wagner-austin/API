"""Measure whether a crowd-invariance-trained base closes the content gap.

THE ARM THIS RUNS (board task ``a85fbabe``, baseline record ``372cee59``,
cross-node bit-identical). The base-LoRA arm proved the LM objective repairs
the STRUCTURAL half of crowded-prefix interference at both scales and left
the CONTENT half standing: gpt2-medium's real-content n8 composition sits
1.04 below its own repaired noise control. This sweep trains the SAME LoRA
by crowd-invariance distillation instead -- behind a drawn roster, on the
drawn target's own text, match the plain base's predictions behind the
target alone (:mod:`~model_trainer.core.services.model.cartridge_content_lora`)
-- then the RECORDED grid's own arms run with the adapted base underneath.

THE GRID IS THE BASE-LORA GRID, cell for cell and name for name --
``lora-plain-n{count}`` and ``lora-diverse-n{count}`` with the same arms,
controls, seeds and floors -- so every observation subtracts against the
``372cee59`` record directly and the two records isolate exactly one
difference: what the LoRA was trained to do. Every alone arm is measured
against the ADAPTED base, so the solo-cost axis prices the distillation.

THE CONTAMINATION WALL AND SEED GEOGRAPHY ARE THE BASE-LORA SWEEP'S OWN,
imported rather than restated: the refusals, the provider, the target
modules and every seed constant come from
:mod:`~model_trainer.cli.cartridge_base_lora_sweep`, because a second copy
of a wall is two walls that can drift apart. The teacher runs on a SECOND
plain instance of the base, loaded here and frozen before use -- PEFT
injects its adapters into the wrapped module tree, so the adapted model
cannot also serve as the un-adapted teacher.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import (
    NO_PAYLOAD,
    Observation,
    RunRecord,
    encode_run_record,
    run_record,
)

from model_trainer.cli import _measurement_hooks, _test_hooks
from model_trainer.cli.cartridge_base_lora_sweep import (
    LORA_TRAIN_SEED,
    POOL_SEED_BASE,
    _require_admissible_corpora,
)
from model_trainer.cli.cartridge_composition_sweep import staged_partner_trains
from model_trainer.cli.cartridge_lora_families import lora_arm_families
from model_trainer.cli.cartridge_lora_policy import quantization_for, target_modules_for
from model_trainer.cli.cartridge_pool_provider import SeededPoolProvider
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies import _test_hooks as strategy_hooks
from model_trainer.core.services.finetuning.strategies.cartridge import (
    require_cache_capable,
)
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_base_lora import freeze_adapted
from model_trainer.core.services.model.cartridge_content_lora import (
    train_composition_lora_invariant,
)
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import (
    train_cartridge,
)
from model_trainer.core.services.model.cartridge_plans import (
    corpus_digest,
    digest_parts,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_pool_plans import (
    CONTENT_LORA_SWEEP_EXPERIMENT,
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
    """Distil the base to crowd-invariance, then run both cell families.

    RESUMABLE PER ARM FAMILY AND COMPARTMENT COUNT, through the same
    :func:`~model_trainer.cli.cartridge_lora_families.lora_arm_families` the
    base sweep drives, which is where that grain is argued.

    WHAT A RESUME PAYS AGAIN, STATED PLAINLY. The distillation above the
    cells is NOT checkpointed: the corpora are re-read, both bases are
    re-loaded, the crowding pool is re-trained and the adapter is distilled
    again. All of it is deterministic -- every training call re-seeds from
    its own seed -- so reproducing it costs time but cannot change a number,
    and serialising an adapter plus a pool of cartridges would be a weights
    checkpoint, a different and much larger contract. The cells are where
    the hours are; the distillation is the toll a resume pays.

    Args:
        plan: The measurement to run.
        plan_name: Which plan this is, so the checkpoint can tell one plan's
            cells from another's.
        corpus: Directory of markdown documents whose retention is the
            finding.
        other_corpora: Composition partners, in the order the counts consume
            them.
        pool_corpora: The distillation corpora, one per companion; disjoint
            from everything measured.
        device: Device to measure on.
        checkpoints: Directory holding this sweep's checkpoint.

    Returns:
        ``(observations, digest)``: the distillation's epoch KLs, one
        ``lora-companion-cross-{j}`` arm per measurement-pool member, and
        both cell families' arms with per-family floors.

    Raises:
        ValueError: Propagated from the refusals and the invariance trainer.
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

    crowding_pool = tuple(
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

    adapted = require_cache_capable(
        strategy_hooks.Hooks.create_peft_model(
            base,
            r=plan["lora_rank"],
            lora_alpha=plan["lora_alpha"],
            lora_dropout=0.0,
            target_modules=target_modules_for(plan["model_id"]),
            bias="none",
        )
    )
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

    observations: list[Observation] = [
        Observation(name="slots_per_cartridge", value=float(plan["slots"])),
        Observation(name="max_drawn", value=float(plan["max_drawn"])),
        *[
            Observation(name=f"invariance-train-epoch-{position}_kl", value=loss)
            for position, loss in enumerate(epoch_kls)
        ],
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
            measurement=f"content-lora-{plan_name}",
            # THE LABEL IS THE IDENTITY, for the reason the base sweep states
            # where it does the same thing: it already carries every
            # measurement field and the primary digest, and it is what two
            # runs are paired by. The partner digests are appended because
            # the label does not reach them.
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


def content_lora_sweep_run_record(
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
        pool_corpora: The distillation corpora.
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
    plan = require_cartridge_plan(_measurement_hooks.content_lora_sweep_plans(), plan_name)
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
        experiment=CONTENT_LORA_SWEEP_EXPERIMENT,
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

    record = content_lora_sweep_run_record(
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
        "content lora sweep %s %s -> %s",
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
        level="INFO",
        format_mode="text",
        service_name="cartridge-content-lora-sweep",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "content_lora_sweep_run_record",
    "entrypoint",
    "main",
    "measure_grid",
]


# Without this, `python -m model_trainer.cli.cartridge_content_lora_sweep`
# imports the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
