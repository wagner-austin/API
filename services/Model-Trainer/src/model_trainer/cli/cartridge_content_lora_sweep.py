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

import torch
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
    LORA_TARGET_MODULES,
    LORA_TRAIN_SEED,
    POOL_SEED_BASE,
    _MeasurementPoolProvider,
    _require_admissible_corpora,
)
from model_trainer.cli.cartridge_benchmark import sweep_observations
from model_trainer.cli.cartridge_companion_sweep import (
    cell_observations,
)
from model_trainer.cli.cartridge_composition_sweep import matched_other_train
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    gain_observations,
    noise_floor,
    per_seed_observations,
    replicate,
)
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies import _test_hooks as strategy_hooks
from model_trainer.core.services.finetuning.strategies.cartridge import (
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_base_lora import freeze_adapted
from model_trainer.core.services.model.cartridge_content_lora import (
    train_composition_lora_invariant,
)
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import (
    held_out_gain,
    measure_composition_scaling,
    train_cartridge,
)
from model_trainer.core.services.model.cartridge_plans import (
    corpus_digest,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_pool_plans import (
    CONTENT_LORA_SWEEP_EXPERIMENT,
    BaseLoraSweepPlan,
    base_lora_sweep_label,
)
from model_trainer.core.services.model.cartridge_varied import (
    measure_varied_companioned_scaling,
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
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    pool_corpora: Sequence[pathlib.Path],
    device: str,
) -> tuple[tuple[Observation, ...], str]:
    """Distil the base to crowd-invariance, then run both cell families.

    Args:
        plan: The measurement to run.
        corpus: Directory of markdown documents whose retention is the
            finding.
        other_corpora: Composition partners, in the order the counts consume
            them.
        pool_corpora: The distillation corpora, one per companion; disjoint
            from everything measured.
        device: Device to measure on.

    Returns:
        ``(observations, digest)``: the distillation's epoch KLs, one
        ``lora-companion-cross-{j}`` arm per measurement-pool member, and
        both cell families' arms with per-family floors.

    Raises:
        ValueError: Propagated from the refusals and the invariance trainer.
        AppError: Propagated from the corpus, PEFT and measurement layers.
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

    other_trains: list[list[torch.Tensor]] = []
    for other in other_corpora[: largest - 1]:
        other_documents = _test_hooks.read_corpus_documents(other)
        other_encoded = [tokenizer.encode(document) for document in other_documents]
        other_trains.append(
            matched_other_train(
                str(other),
                build_windows(other_encoded, window=plan["window"], device=device),
                held_out_stride=plan["held_out_stride"],
                required=len(train),
            )
        )
    pool_trains: list[list[torch.Tensor]] = []
    for pool_corpus in pool_corpora:
        pool_documents = _test_hooks.read_corpus_documents(pool_corpus)
        pool_encoded = [tokenizer.encode(document) for document in pool_documents]
        pool_trains.append(
            matched_other_train(
                str(pool_corpus),
                build_windows(pool_encoded, window=plan["window"], device=device),
                held_out_stride=plan["held_out_stride"],
                required=len(train),
            )
        )

    base = require_cache_capable(hf_hooks.Hooks.load_hf_model(plan["model_id"], None))
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
    teacher_base = require_cache_capable(hf_hooks.Hooks.load_hf_model(plan["model_id"], None))
    teacher_base.to(device)
    freeze_adapted(teacher_base)

    adapted = require_cache_capable(
        strategy_hooks.Hooks.create_peft_model(
            base,
            r=plan["lora_rank"],
            lora_alpha=plan["lora_alpha"],
            lora_dropout=0.0,
            target_modules=LORA_TARGET_MODULES,
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

    provider = _MeasurementPoolProvider(adapted, pool_trains, plan)
    companion_gains: list[list[tuple[int, float]]] = [[] for _ in pool_trains]
    for seed in plan["seeds"]:
        for member, slots in enumerate(provider.pool(seed)):
            companion_gains[member].append(
                (seed, held_out_gain(CartridgeModel(base=adapted, slots=slots), held_out))
            )
    for member, gains in enumerate(companion_gains):
        arm = replicate(f"lora-companion-cross-{member}", gains)
        _log.info("lora-companion-cross-%d: %+.4f on the primary held-out", member, arm["mean"])
        observations.extend(gain_observations(arm))
        observations.extend(per_seed_observations(arm))

    plain_arms: list[ReplicatedGain] = []
    diverse_arms: list[ReplicatedGain] = []
    for count in plan["compartment_counts"]:
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
        plain_arms.append(composed)
        observations.extend(
            cell_observations(plain_name, alone, composed, untrained_composed, cross)
        )
        # The seed pairing is the evidence (the noise-floor finding on task
        # 1fc5afed): means and spreads cannot answer a paired question after
        # the fact, so every arm's per-seed gains join the record. Additive
        # rows only; every recorded cell name is untouched.
        for measurement in (alone, composed, untrained_composed, *cross):
            observations.extend(per_seed_observations(measurement))

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
            pool_for_seed=provider.pool,
            companion_probability=plan["probability"],
        )
        _log.info(
            "%s: %+.4f alone -> %+.4f composed", diverse_name, alone["mean"], composed["mean"]
        )
        diverse_arms.append(composed)
        observations.extend(
            cell_observations(diverse_name, alone, composed, untrained_composed, cross)
        )
        # The seed pairing is the evidence (the noise-floor finding on task
        # 1fc5afed): means and spreads cannot answer a paired question after
        # the fact, so every arm's per-seed gains join the record. Additive
        # rows only; every recorded cell name is untouched.
        for measurement in (alone, composed, untrained_composed, *cross):
            observations.extend(per_seed_observations(measurement))

    plain_floor = noise_floor(plain_arms)
    observations.append(Observation(name="lora-plain_composed_noise_floor", value=plain_floor))
    observations.extend(sweep_observations(plain_arms, plain_floor))
    diverse_floor = noise_floor(diverse_arms)
    observations.append(Observation(name="lora-diverse_composed_noise_floor", value=diverse_floor))
    observations.extend(sweep_observations(diverse_arms, diverse_floor))
    return tuple(observations), digest


def content_lora_sweep_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    pool_corpora: Sequence[pathlib.Path],
    device: str,
) -> RunRecord:
    """Pin determinism, run the grid, and record it.

    Args:
        plan_name: Which plan to run.
        corpus: Directory of markdown documents whose retention is measured.
        other_corpora: Composition partners.
        pool_corpora: The distillation corpora.
        device: Device to measure on.

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
        corpus=corpus,
        other_corpora=other_corpora,
        pool_corpora=pool_corpora,
        device=device,
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
    record = content_lora_sweep_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        other_corpora=others,
        pool_corpora=pool,
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
    )

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
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
