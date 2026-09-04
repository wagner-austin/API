"""Measure how cartridge composition scales with the compartment count.

THE QUESTION. The recorded two-cartridge result (``gpt2-wiki``, 2026-09-03)
is that a cartridge retains ~59% of its held-out gain when one unrelated
cartridge is concatenated in front of it. A compartmental serving design
wants to wire SEVERAL wiki compartments into one request, and nothing has
measured three or more. If retention collapses with the count, the practical
limit is two and the serving design must know that before it is built.

TWO POLICIES, BECAUSE THE DESIGN SPACE HAS TWO EDGES. Holding each
compartment's slot count fixed grows the composed prefix with the count, so
it measures interference plus attention over a longer prefix. Holding the
TOTAL budget fixed shrinks each compartment as the count grows, so it
measures interference plus per-compartment capacity loss. A deployment picks
a point between these edges; the sweep measures the edges.

THE FIXED-POLICY ALONE ARMS DOUBLE AS AN INTEGRITY CHECK. Under the fixed
policy the alone arm is the same cartridge -- same corpus, same slot count,
same seeds -- at every compartment count, trained independently each time.
With determinism pinned those means must agree exactly; a record where they
differ is a record whose training was not a function of its seeds, which is
the defect that invalidated this measurement's first ancestor.

CROSS-GAIN ARMS GUARD THE PREMISE. Every retention here is only meaningful
if the other corpora are genuinely unrelated to the primary -- the two-halves
artifact (94% retention, entirely overlap) is the standing warning. Each
other-corpus cartridge is therefore scored alone on the primary held-out
items, and a positive cross gain marks the retention it participated in as
inflated. The check rides in the record, not in a caveat somebody remembers.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

import torch
from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.errors import (
    AppError,
    ModelTrainerErrorCode,
    model_trainer_status_for,
)
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
from model_trainer.cli.cartridge_benchmark import sweep_observations
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    gain_observations,
    noise_floor,
    retention,
)
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import measure_composition_scaling
from model_trainer.core.services.model.cartridge_plans import (
    COMPOSITION_SWEEP_EXPERIMENT,
    CompositionSweepPlan,
    composition_sweep_label,
    corpus_digest,
    require_cartridge_plan,
)

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
OTHER_CORPORA_FLAG = "--other-corpora"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (PLAN_FLAG, CORPUS_FLAG, OTHER_CORPORA_FLAG, DEVICE_FLAG, OUT_FLAG)

#: The two slot policies, in the order they run and are recorded.
POLICIES = ("fixed", "budget")


def policy_slots(plan: CompositionSweepPlan, policy: str, count: int) -> int:
    """Slots per cartridge for one policy at one compartment count.

    Args:
        plan: The plan being run.
        policy: ``"fixed"`` or ``"budget"``.
        count: How many compartments compose.

    Returns:
        Prefix positions for EACH cartridge.

    Raises:
        ValueError: If the policy is unknown, or the budget policy cannot
            divide its total evenly at this count. An uneven division would
            silently hand some compartment more capacity than its siblings
            and report the asymmetry as a property of composing.
    """
    if policy == "fixed":
        return plan["fixed_slots"]
    if policy != "budget":
        raise ValueError(f"unknown slot policy {policy!r}; policies: {', '.join(POLICIES)}")
    budget = plan["total_slot_budget"]
    if budget % count != 0:
        raise ValueError(
            f"a total slot budget of {budget} does not divide evenly across "
            f"{count} compartments; every count in the plan must divide the "
            f"budget, or the compartments are not the same size"
        )
    return budget // count


def matched_other_train(
    name: str,
    windows: Sequence[torch.Tensor],
    *,
    held_out_stride: int,
    required: int,
) -> list[torch.Tensor]:
    """Split one other corpus and match its training set to the primary's.

    Matching is a truncation, exactly as the two-cartridge benchmark does it,
    so every cartridge in a composition sees the same amount of training
    signal. A SHORTER corpus is refused rather than accepted: its cartridge
    would be weaker for want of text, and that weakness would be reported as
    a property of composing.

    Args:
        name: The corpus, for the refusal message.
        windows: Every window the corpus yielded.
        held_out_stride: The plan's stride; the held-out share is discarded.
        required: Training windows the primary corpus has.

    Returns:
        Exactly ``required`` training windows.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` if the corpus yields
            fewer training windows than the primary.
    """
    train, _held = split_by_stride(windows, held_out_stride=held_out_stride)
    if len(train) < required:
        raise AppError(
            ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE,
            (
                f"other corpus {name} yields {len(train)} training window(s) against "
                f"the primary's {required}; its cartridge would be weaker for want "
                f"of text and the gap would be misread as a composition cost -- "
                f"supply more text or a smaller primary corpus"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CARTRIDGE_CORPUS_UNUSABLE),
        )
    return train[:required]


def measure_sweep(
    plan: CompositionSweepPlan,
    *,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    device: str,
) -> tuple[tuple[Observation, ...], str]:
    """Run both policies at every compartment count and name what they produced.

    Args:
        plan: The measurement to run.
        corpus: Directory of markdown documents whose retention is the
            finding.
        other_corpora: Directories of mutually unrelated documents, one per
            additional compartment. The largest count uses the first
            ``count - 1`` of them, so their order is part of the run.
        device: Device to measure on.

    Returns:
        ``(observations, digest)`` -- the named numbers, and the digest of
        the primary corpus.

    Raises:
        ValueError: If too few other corpora are supplied for the plan's
            largest count, or the budget does not divide a count.
        AppError: Propagated from the corpus and measurement layers.
    """
    largest = max(plan["compartment_counts"])
    if len(other_corpora) < largest - 1:
        raise ValueError(
            f"the plan composes up to {largest} compartments, which needs "
            f"{largest - 1} other corpora; {len(other_corpora)} supplied"
        )

    documents = _test_hooks.read_corpus_documents(corpus)
    digest = corpus_digest(documents)
    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(plan["model_id"])
    encoded = [tokenizer.encode(document) for document in documents]
    train, held_out = split_by_stride(
        build_windows(encoded, window=plan["window"], device=device),
        held_out_stride=plan["held_out_stride"],
    )
    _log.info(
        "primary corpus %s: %d documents, %d train / %d held-out windows of %d tokens",
        corpus,
        len(documents),
        len(train),
        len(held_out),
        plan["window"],
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
        _log.info("other corpus %s: %d matched training windows", other, len(train))

    base = require_cache_capable(hf_hooks.Hooks.load_hf_model(plan["model_id"], None))
    base.to(device)

    observations: list[Observation] = []
    for policy in POLICIES:
        composed_arms: list[ReplicatedGain] = []
        for count in plan["compartment_counts"]:
            slots = policy_slots(plan, policy, count)
            arm = f"{policy}-n{count}"
            alone, composed, untrained_composed, cross = measure_composition_scaling(
                base,
                first_train=train,
                other_trains=other_trains[: count - 1],
                held_out=held_out,
                arm=arm,
                num_slots=slots,
                seeds=plan["seeds"],
                epochs=plan["epochs"],
                learning_rate=plan["learning_rate"],
            )
            _log.info(
                "%s (%d slots each): %+.4f alone -> %+.4f composed, %+.4f untrained-composed",
                arm,
                slots,
                alone["mean"],
                composed["mean"],
                untrained_composed["mean"],
            )
            composed_arms.append(composed)
            for measured in [alone, composed, untrained_composed, *cross]:
                observations.extend(gain_observations(measured))
            observations.append(Observation(name=f"{arm}_slots_per_cartridge", value=float(slots)))
            observations.append(
                Observation(name=f"{arm}_retention", value=retention(alone, composed))
            )
            observations.append(
                Observation(
                    name=f"{arm}_untrained_retention",
                    value=retention(alone, untrained_composed),
                )
            )
            # The interference verdict: what the strangers' CONTENT costs
            # beyond their presence. Judged against a floor from this pair
            # alone -- both arms run the same prefix length, so they are one
            # kind, and importing the sweep-wide floor would judge them
            # against noise from differently shaped prefixes.
            observations.extend(
                sweep_observations(
                    [composed, untrained_composed],
                    noise_floor([composed, untrained_composed]),
                )
            )
        # One floor per policy, over its composed arms only: they are the
        # arms the retention trend is read from, and mixing the two policies
        # would judge one edge of the design space against the other's noise.
        floor = noise_floor(composed_arms)
        observations.append(Observation(name=f"{policy}_composed_noise_floor", value=floor))
        observations.extend(sweep_observations(composed_arms, floor))
    return tuple(observations), digest


def composition_sweep_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    device: str,
) -> RunRecord:
    """Pin determinism, run the sweep, and record it.

    Args:
        plan_name: Which plan to run.
        corpus: Directory of markdown documents whose retention is measured.
        other_corpora: Directories of mutually unrelated documents.
        device: Device to measure on.

    Returns:
        The record.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        ValueError: Propagated from :func:`measure_sweep`.
        AppError: Propagated from the corpus and measurement layers.
    """
    plan = require_cartridge_plan(_measurement_hooks.composition_sweep_plans(), plan_name)
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )
    observations, digest = measure_sweep(
        plan, corpus=corpus, other_corpora=other_corpora, device=device
    )
    return run_record(
        experiment=COMPOSITION_SWEEP_EXPERIMENT,
        label=composition_sweep_label(plan_name, plan, digest=digest),
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
            required flag is absent, or too few other corpora are named.
        KeyError: If the plan name is unknown.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    others = [
        pathlib.Path(entry)
        for entry in cli_args.require_flag(parsed, OTHER_CORPORA_FLAG).split(",")
        if entry
    ]
    record = composition_sweep_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        other_corpora=others,
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
    )

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "composition sweep %s %s -> %s",
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
        service_name="cartridge-composition-sweep",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "POLICIES",
    "composition_sweep_run_record",
    "entrypoint",
    "main",
    "matched_other_train",
    "measure_sweep",
    "policy_slots",
]


# Without this, `python -m model_trainer.cli.cartridge_composition_sweep`
# imports the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
