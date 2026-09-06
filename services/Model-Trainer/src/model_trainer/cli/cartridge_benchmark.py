"""Measure a cartridge against a REAL base model, and record what it found.

The run the cartridge work needed and did not have. Its unit tests measure a
two-layer, two-head model with random weights, which is the right thing for a
suite that must finish in seconds -- and three of the conclusions drawn from
that model did not survive a real one. The differences are set out in
:mod:`~model_trainer.core.services.model.cartridge_measurement`; the short
version is that an untrained prefix is harmless on a random model and does
real damage on a trained one, which inverts the control every other arm was
read against.

WHAT IT EMITS. One :class:`RunRecord` carrying, per arm, a mean and a spread.
The spread is not decoration: this measurement's own noise turned out to be
about 0.02, and the first pass at it reported several differences of that size
as findings. Every arm is replicated across the plan's seeds, and
:func:`separates` judges each step of the sweep against the largest spread the
run itself produced rather than against a constant somebody chose.

A PLAN IS A FUNCTION OF ITS SEEDS AND NOTHING ELSE, and it took a wrong
explanation to get there. Two runs of one plan reported the eight-slot arm's
spread as 0.0049 and then 0.0268; the runs happened to differ in whether a
test suite was also using the GPU, and this paragraph said contention was the
cause. It was not. Training was drawing dropout from a process-wide generator
nobody seeded, so ANY two runs differed and the load correlation was a
coincidence with a plausible story attached.
:func:`~model_trainer.core.services.model.cartridge_measurement.train_cartridge`
now seeds per arm, after the geometry probe. Whether contention affects the
numbers is a real question and remains unmeasured -- it is simply not what
those two runs showed.

WHY TWO CORPORA. The composition arm asks what a cartridge retains when
another is concatenated in front of it, and that question has a wrong answer
available: composing two cartridges trained on two halves of ONE corpus
measured 94% retention, because each half already predicted the other. Against
an unrelated corpus the same code reports 59%. So the second corpus is a
required flag -- there is no default that would be honest.
"""

from __future__ import annotations

import itertools
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
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    gain_observations,
    noise_floor,
    per_seed_observations,
    retention,
    separates,
)
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import (
    measure_composition,
    measure_slot_count,
    measure_untrained,
)
from model_trainer.core.services.model.cartridge_plans import (
    CARTRIDGE_EXPERIMENT,
    CartridgePlan,
    corpus_digest,
    plan_label,
    require_cartridge_plan,
)
from model_trainer.core.services.model.control_arms import CONTROLS_FLAG, require_control_arm

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
SECOND_CORPUS_FLAG = "--second-corpus"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (PLAN_FLAG, CORPUS_FLAG, SECOND_CORPUS_FLAG, DEVICE_FLAG, OUT_FLAG, CONTROLS_FLAG)


def sweep_observations(sweep: Sequence[ReplicatedGain], floor: float) -> tuple[Observation, ...]:
    """Name each step of the capacity sweep and whether it cleared the noise.

    The separation verdicts are recorded as numbers, not left to a reader with
    the means in front of them. Whether one slot count beat another is the
    claim the sweep exists to make, and a record that carried only the means
    would leave every future reader to re-derive it against a floor they would
    have to re-derive too.

    Args:
        sweep: The sweep's arms, in increasing slot order.
        floor: The run's noise floor.

    Returns:
        Two observations per adjacent pair: the signed difference, and 1.0 or
        0.0 for whether it separated.
    """
    named: list[Observation] = []
    for smaller, larger in itertools.pairwise(sweep):
        verdict = separates(larger, smaller, floor=floor)
        step = f"{verdict['second']}_to_{verdict['first']}"
        named.append(Observation(name=f"{step}_difference", value=verdict["difference"]))
        named.append(
            Observation(name=f"{step}_separated", value=1.0 if verdict["separated"] else 0.0)
        )
    return tuple(named)


def measure_plan(
    plan: CartridgePlan,
    *,
    corpus: pathlib.Path,
    second_corpus: pathlib.Path,
    device: str,
) -> tuple[tuple[Observation, ...], str]:
    """Run every arm of one plan and name what they produced.

    Args:
        plan: The measurement to run.
        corpus: Directory of markdown documents the cartridge is trained and
            scored on.
        second_corpus: Directory of UNRELATED documents, for the composition
            arm's second cartridge.
        device: Device to measure on.

    Returns:
        ``(observations, digest)`` -- the named numbers, and the digest of the
        primary corpus, which the label is built from.

    Raises:
        AppError: With ``CARTRIDGE_CORPUS_UNUSABLE`` if either corpus yields
            no windows, or ``CARTRIDGE_MEASUREMENT_UNREPLICATED`` if the plan
            names too few seeds.
    """
    documents = _test_hooks.read_corpus_documents(corpus)
    second_documents = _test_hooks.read_corpus_documents(second_corpus)
    digest = corpus_digest(documents)

    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(plan["model_id"])
    encoded = [tokenizer.encode(document) for document in documents]
    second_encoded = [tokenizer.encode(document) for document in second_documents]

    train, held_out = split_by_stride(
        build_windows(encoded, window=plan["window"], device=device),
        held_out_stride=plan["held_out_stride"],
    )
    second_train, _second_held = split_by_stride(
        build_windows(second_encoded, window=plan["window"], device=device),
        held_out_stride=plan["held_out_stride"],
    )
    # Matched in size to the primary corpus, so the composition arm's two
    # cartridges see the same amount of training signal. Without this the
    # weaker cartridge would be the one with less text, and its weakness
    # would be reported as a property of composing.
    second_train = second_train[: len(train)]
    _log.info(
        "corpus %s: %d documents, %d train / %d held-out windows of %d tokens",
        corpus,
        len(documents),
        len(train),
        len(held_out),
        plan["window"],
    )

    base = require_cache_capable(hf_hooks.Hooks.load_hf_model(plan["model_id"], None))
    base.to(device)

    untrained = measure_untrained(
        base, held_out, num_slots=plan["composition_slots"], seeds=plan["seeds"]
    )
    _log.info("untrained prefix: %+.4f (spread %.4f)", untrained["mean"], untrained["spread"])

    sweep: list[ReplicatedGain] = []
    for num_slots in plan["slot_counts"]:
        arm = measure_slot_count(
            base,
            train,
            held_out,
            num_slots=num_slots,
            seeds=plan["seeds"],
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
        )
        sweep.append(arm)
        _log.info("%s: %+.4f (spread %.4f)", arm["arm"], arm["mean"], arm["spread"])

    alone, composed = measure_composition(
        base,
        first_train=train,
        second_train=second_train,
        held_out=held_out,
        arm="composition",
        num_slots=plan["composition_slots"],
        seeds=plan["seeds"],
        epochs=plan["epochs"],
        learning_rate=plan["learning_rate"],
    )
    _log.info("composition: %+.4f alone -> %+.4f composed", alone["mean"], composed["mean"])

    # TWO FLOORS, NOT ONE, AND THIS WAS A DEFECT BEFORE IT WAS A DESIGN. The
    # first record this CLI emitted judged the sweep against the largest
    # spread of ANY arm, and the largest belonged to the composed arm -- which
    # trains two cartridges and runs a prefix twice as long, so it is noisier
    # for reasons that have nothing to do with the sweep. At 0.0671 it buried
    # a sweep step of +0.0584 that two independent runs had found real.
    #
    # A floor estimates how much ONE KIND of arm varies between seeds.
    # Importing another kind's noise does not make a claim more conservative,
    # it makes it wrong in an unpredictable direction.
    sweep_floor = noise_floor(sweep)
    composition_floor = noise_floor([alone, composed])
    _log.info(
        "noise floor %.4f across %d sweep arms, %.4f across the composition pair",
        sweep_floor,
        len(sweep),
        composition_floor,
    )

    observations: list[Observation] = []
    for arm in [untrained, *sweep, alone, composed]:
        observations.extend(gain_observations(arm))
        # THE PER-SEED GAINS, WITHOUT WHICH THIS RECORD CANNOT ANSWER ITS OWN
        # HARDEST QUESTION. Every arm runs under the same seeds, so a seed's
        # gain at 32 slots and at 128 slots are two measurements of one draw
        # -- and the sweep's top steps are decided by those paired
        # differences, not by two means with their pairing discarded. The
        # record carried mean and spread only, so the numbers existed in this
        # process and were dropped on the way out.
        observations.extend(per_seed_observations(arm))
    observations.extend(sweep_observations(sweep, sweep_floor))
    observations.append(Observation(name="sweep_noise_floor", value=sweep_floor))
    observations.append(Observation(name="composition_noise_floor", value=composition_floor))
    observations.append(Observation(name="composition_retention", value=retention(alone, composed)))
    return tuple(observations), digest


def cartridge_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    second_corpus: pathlib.Path,
    device: str,
    remove_split_k: bool,
    math_attention: bool,
) -> RunRecord:
    """Pin determinism, run every arm, and record it.

    WHY THE POSTURE IS AN ARGUMENT RATHER THAN FIXED AT ``none``. It was
    fixed, at ``(False, False)``, and that made this command unable to run
    the one arm it most needed. The first cross-card check -- an RTX 3090 Ti
    against a V100, identical code, identical corpus digest, identical seeds
    -- found every trained arm disagreeing by 1e-3 to 7e-3 while the
    UNTRAINED arm, which does forward passes and no optimisation, agreed to
    9.1e-08. That is float32 last-bit disagreement in the forward kernels,
    integrated through training. The two controls in
    :mod:`~model_trainer.core.services.model.control_arms` are the measured
    fix for exactly those kernels, and this command could not reach them, so
    the obvious next measurement was a code change rather than a run.

    An argument, not a flip of the constant, for the reason
    :func:`~model_trainer.cli.known_answer_probe.probe_determinism` states:
    an instrument that imposes an intervention cannot measure it. The
    ``none`` arm has to stay reachable or the treated numbers have nothing
    to be compared against.

    Args:
        plan_name: Which plan to run.
        corpus: Directory of markdown documents to measure.
        second_corpus: Unrelated documents for the composition arm.
        device: Device to measure on.
        remove_split_k: Whether to take split-K out of cuBLASLt's options.
        math_attention: Whether to restrict attention to the math kernel.
            On a 16 GB card this is the control that can turn a run that
            fits into one that does not: the math path materialises the whole
            ``[batch, heads, seq, seq]`` score matrix, so its cost grows with
            the square of the plan's window.

    Returns:
        The record. Its fingerprint carries whichever controls were applied,
        so two records measured under different arms are distinguishable
        without reading the command line that produced them.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        AppError: Propagated from the arms when a corpus or a seed count
            cannot support the measurement.
    """
    plan = require_cartridge_plan(_measurement_hooks.cartridge_plans(), plan_name)
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device,
        probe_determinism(device, remove_split_k=remove_split_k, math_attention=math_attention),
    )
    observations, digest = measure_plan(
        plan, corpus=corpus, second_corpus=second_corpus, device=device
    )
    return run_record(
        experiment=CARTRIDGE_EXPERIMENT,
        label=plan_label(plan_name, plan, digest=digest),
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
        ValueError: When a flag is unknown, repeated, missing its value, or a
            required flag is absent.
        KeyError: If the plan name is unknown.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    remove_split_k, math_attention = require_control_arm(
        cli_args.require_flag(parsed, CONTROLS_FLAG)
    )

    record = cartridge_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        second_corpus=pathlib.Path(cli_args.require_flag(parsed, SECOND_CORPUS_FLAG)),
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
        remove_split_k=remove_split_k,
        math_attention=math_attention,
    )

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "cartridge measurement %s %s -> %s",
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
        service_name="cartridge-benchmark",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "cartridge_run_record",
    "entrypoint",
    "main",
    "measure_plan",
    "sweep_observations",
]


# Without this, `python -m model_trainer.cli.cartridge_benchmark` imports the
# module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
