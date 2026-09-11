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
became a division artefact, so a composition number measured against a solo
arm indistinguishable from noise is a ratio with no reading. Then the
composed cells at each compartment count with their untrained-composed and
cross controls. Then the steering arm at matched counts, which is what makes
the result a comparison rather than a number.

THE GRID IS THE RECORDED ONE, WITH A STATED DEVIATION. Solo, n2 and n4 with
every control run here; the diverse-companion, base-LoRA and crowd-invariance
families are NOT, and that is deliberate rather than an omission. Those three
are interventions that repair composition, and the question they answer --
does base-side adaptation repair DISPOSITIONAL interference -- is only
askable once there is a naive interference number to repair. The corpus arc
was built in exactly that order, each sweep importing the one before it, and
building the repair before the baseline would be the same mistake the solo
precondition exists to prevent one level up.

RESUMABLE PER CELL. Both rungs run on ``free-gpu``, where ``PreemptMode=CANCEL``
means an evicted job is killed outright and Slurm resubmits nothing. A cell is
the smallest unit whose observations are complete, and the corpus reading,
tokenisation and power gate above the cells are re-derived on a resume: all of
it is deterministic, so reproducing it costs seconds and cannot change a
number.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

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

from model_trainer.cli import _measurement_hooks, _trait_hooks
from model_trainer.cli.cartridge_benchmark import sweep_observations
from model_trainer.cli.cartridge_lora_policy import quantization_for
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    noise_floor,
    replicated_from_observations,
)
from model_trainer.core.contracts.trait_corpus import TraitCorpus, trait_corpus_digest
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
from model_trainer.core.services.model.cartridge_qa_power import require_resolvable_pairs
from model_trainer.core.services.model.cartridge_scoring import TraitPair
from model_trainer.core.services.model.cartridge_sweep_checkpoint import (
    bind_cells,
    checkpointed_cells,
)
from model_trainer.core.services.model.steering_vectors import require_steerable
from model_trainer.core.services.model.trait_arms import (
    TraitArm,
    measure_trait_composition,
    measure_trait_solo,
    measure_trait_steering,
    steering_observations,
    trait_arm_observations,
    trait_cell_observations,
)
from model_trainer.core.services.model.trait_corpus import (
    split_trait_pairs,
    tokenise_trait_pairs,
)

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (PLAN_FLAG, CORPUS_FLAG, DEVICE_FLAG, OUT_FLAG)


class _SplitTrait:
    """One trait's pairs, already tokenised and split.

    Attributes:
        trait: The trait's name, which names every arm it appears in.
        train: Pairs the cartridge trains on and the direction is read from.
        held_out: Pairs every arm is scored on.
    """

    def __init__(self, trait: str, train: list[TraitPair], held_out: list[TraitPair]) -> None:
        """Hold one trait's split.

        Args:
            trait: The trait's name.
            train: Training pairs.
            held_out: Held-out pairs.
        """
        self.trait = trait
        self.train = train
        self.held_out = held_out


def prepare_traits(
    corpora: Sequence[TraitCorpus], plan: TraitPlan, *, device: str
) -> list[_SplitTrait]:
    """Tokenise and split every trait in the roster, in roster order.

    Args:
        corpora: The authored corpora, in roster order.
        plan: The measurement being run.
        device: Torch device string to build the tensors on.

    Returns:
        One split per trait, in roster order.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` propagated from tokenisation
            or the split.
    """
    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(plan["model_id"])
    prepared: list[_SplitTrait] = []
    for corpus in corpora:
        pairs = tokenise_trait_pairs(
            corpus, tokenizer, max_seq_len=plan["max_seq_len"], device=device
        )
        train, held_out = split_trait_pairs(pairs, held_out_stride=plan["held_out_stride"])
        _log.info(
            "trait %s: %d pair(s), %d train / %d held out",
            corpus["trait"],
            len(pairs),
            len(train),
            len(held_out),
        )
        prepared.append(_SplitTrait(corpus["trait"], train, held_out))
    return prepared


def require_solo_precondition(solo: TraitArm, untrained: TraitArm) -> None:
    """Stop the arc when the solo arm did not clear its own noise.

    RAISED BETWEEN CELLS, WHICH IS THE ENTIRE VALUE OF IT. Checking this after
    the composed cells would cost the hours the composed cells take and then
    report that none of them can be read. The corpus programme's 7B rung
    failed at exactly this step, and the composition numbers measured beside
    it were reported for days before anybody noticed that their denominator
    was noise.

    THE BAR IS THE ARM'S OWN SPREAD, and it is deliberately the weakest
    defensible one. A solo gain smaller than the range its own seeds produced
    cannot be told from which cartridge happened to be drawn; clearing it is
    not evidence the effect is large, only that there is an effect to divide
    by. The untrained arm is named in the refusal because a solo gain that
    merely matches an untrained prefix is the other way this fails, and a
    reader needs both numbers to tell them apart.

    Args:
        solo: The trained solo arm.
        untrained: The untrained-prefix control, measured under the same seeds.

    Raises:
        AppError: With ``TRAIT_SOLO_PRECONDITION_FAILED`` when the trained
            arm's mean expression gain does not exceed its own spread.
    """
    expression = solo["expression"]
    if expression["mean"] > expression["spread"]:
        return
    raise AppError(
        ModelTrainerErrorCode.TRAIT_SOLO_PRECONDITION_FAILED,
        (
            f"the solo arm {expression['arm']!r} expressed its trait by "
            f"{expression['mean']:+.4f} across seeds {expression['seeds']} while its own "
            f"per-seed spread is {expression['spread']:.4f}, so the gain cannot be told "
            f"from which cartridge happened to be drawn (the untrained-prefix control "
            f"sits at {untrained['expression']['mean']:+.4f}). Every composed cell below "
            f"this divides by that gain, so their retentions would be division "
            f"artefacts; this is the substrate failing, not composition, and it is a "
            f"result to report rather than a run to repair"
        ),
        model_trainer_status_for(ModelTrainerErrorCode.TRAIT_SOLO_PRECONDITION_FAILED),
    )


def measure_grid(
    plan: TraitPlan,
    *,
    plan_name: str,
    corpus: pathlib.Path,
    device: str,
    checkpoints: pathlib.Path,
) -> tuple[tuple[Observation, ...], str]:
    """Run the solo cell, the composed cells and the steering arms.

    Args:
        plan: The measurement to run.
        plan_name: Which plan this is, so the checkpoint can tell one plan's
            cells from another's.
        corpus: Directory holding one authored JSON file per trait.
        device: Device to measure on.
        checkpoints: Directory holding this sweep's checkpoint.

    Returns:
        ``(observations, digest)``: the realised pair count and resolvable
        floor, the solo cell and its control, every composed cell with its
        controls, the composed family's noise floor and step verdicts, and one
        steering reading per matched count.

    Raises:
        AppError: With ``TRAIT_CORPUS_UNUSABLE`` from the corpus layer,
            ``CARTRIDGE_QA_UNDERPOWERED`` when the realised pair set cannot
            resolve what the plan declares, ``TRAIT_SOLO_PRECONDITION_FAILED``
            when the solo arm does not clear its own noise, and from the
            checkpoint layer when a checkpoint describes different inputs.
    """
    corpora = _trait_hooks.read_trait_corpora(corpus, plan["traits"])
    digest = trait_corpus_digest(corpora)
    label = trait_plan_label(plan_name, plan, digest=digest)
    prepared = prepare_traits(corpora, plan, device=device)
    primary = prepared[0]

    # THE GATE RUNS BEFORE A MODEL LOADS, on the REALISED held-out count. A
    # cap in a plan is an upper bound the corpus is free to fall short of, and
    # the retracted question-set headline came from a plan whose cap said 120
    # over a set of 32.
    floor = require_resolvable_pairs(
        len(primary.held_out),
        alpha=plan["alpha"],
        test=plan["mcnemar_test"],
        smallest_effect_of_interest=plan["smallest_effect_of_interest"],
        subject="pair",
    )
    _log.info(
        "%d held-out pair(s) on %s resolve nothing smaller than %.4f; the plan declares %.4f",
        len(primary.held_out),
        primary.trait,
        floor,
        plan["smallest_effect_of_interest"],
    )

    base = require_cache_capable(
        hf_hooks.Hooks.load_hf_model(plan["model_id"], quantization_for(plan["model_id"]))
    )
    base.to(device)

    def _solo() -> tuple[Observation, ...]:
        """Measure the trait alone, with the untrained prefix beside it.

        Returns:
            Both arms' rows, and the precondition verdict as a number.

        Raises:
            AppError: With ``TRAIT_SOLO_PRECONDITION_FAILED`` when the trained
                arm does not clear its own spread.
        """
        trained, untrained = measure_trait_solo(
            base,
            train=primary.train,
            held_out=primary.held_out,
            arm=f"{primary.trait}-solo",
            num_slots=plan["slots"],
            seeds=plan["seeds"],
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
        )
        _log.info(
            "%s: %+.4f expression, %+.4f coherence, spread %.4f",
            trained["expression"]["arm"],
            trained["expression"]["mean"],
            trained["coherence"]["mean"],
            trained["expression"]["spread"],
        )
        require_solo_precondition(trained, untrained)
        return (
            *trait_arm_observations(trained),
            *trait_arm_observations(untrained),
        )

    def _composed(count: int) -> tuple[Observation, ...]:
        """Measure one compartment count's cell.

        Args:
            count: How many trait cartridges are composed.

        Returns:
            Every arm's rows and the expression retention where readable.
        """
        arm = f"{primary.trait}-n{count}"
        cell = measure_trait_composition(
            base,
            first_train=primary.train,
            other_trains=[other.train for other in prepared[1:count]],
            held_out=primary.held_out,
            arm=arm,
            num_slots=plan["slots"],
            seeds=plan["seeds"],
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
        )
        _log.info(
            "%s: %+.4f alone -> %+.4f composed",
            arm,
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
            strength=plan["steering_strength"],
        )
        _log.info(
            "%s: %+.4f expression on %d pair(s), p=%.4f",
            reading["arm"],
            reading["expression"]["mean_baseline"] - reading["expression"]["mean_treatment"],
            reading["expression"]["items"],
            reading["expression"]["p_value"],
        )
        return steering_observations(reading)

    # THE STEERING ARM RUNS AT EVERY CARTRIDGE COUNT AND AT ONE. Its solo
    # reading is the anchor: without it, a composed steering number has
    # nothing of its own kind to be read against and the comparison would be
    # between one arc's composed arm and another arc's solo arm.
    steering_counts = (1, *plan["compartment_counts"])
    produced = checkpointed_cells(
        checkpoints,
        measurement=f"trait-{plan_name}",
        inputs_digest=label,
        cells=[
            ("solo", _solo),
            *bind_cells([(f"n{count}", count) for count in plan["compartment_counts"]], _composed),
            *bind_cells([(f"steer-n{count}", count) for count in steering_counts], _steering),
        ],
    )

    observations: list[Observation] = [
        Observation(name="held_out_pairs", value=float(len(primary.held_out))),
        Observation(name="training_pairs", value=float(len(primary.train))),
        Observation(name="resolvable_floor", value=floor),
        Observation(name="declared_effect", value=plan["smallest_effect_of_interest"]),
        Observation(name="slots_per_cartridge", value=float(plan["slots"])),
        Observation(name="steering_strength", value=plan["steering_strength"]),
        *produced["solo"],
    ]
    composed_arms: list[ReplicatedGain] = []
    for count in plan["compartment_counts"]:
        rows = produced[f"n{count}"]
        observations.extend(rows)
        # REBUILT FROM THE ROWS RATHER THAN KEPT FROM THE CALL, because a
        # resumed cell never made the call. The rebuild is exact: every field
        # of an arm is a function of its per-seed gains, and those are in the
        # rows.
        composed_arms.append(
            replicated_from_observations(
                rows,
                arm=f"{primary.trait}-n{count}-composed-expression",
                seeds=plan["seeds"],
            )
        )
    for count in steering_counts:
        observations.extend(produced[f"steer-n{count}"])

    composed_floor = noise_floor(composed_arms)
    observations.append(Observation(name="composed_expression_noise_floor", value=composed_floor))
    observations.extend(sweep_observations(composed_arms, composed_floor))
    return tuple(observations), digest


def trait_sweep_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    device: str,
    checkpoints: pathlib.Path,
) -> RunRecord:
    """Pin determinism, run the grid, and record it.

    Args:
        plan_name: Which plan to run.
        corpus: Directory holding one authored JSON file per trait.
        device: Device to measure on.
        checkpoints: Directory holding this sweep's checkpoint, so an evicted
            run resumes at its last completed cell rather than at zero.

    Returns:
        The record.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        AppError: Propagated from the corpus, power, measurement and
            checkpoint layers.
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
    )
    return run_record(
        experiment=TRAIT_SWEEP_EXPERIMENT,
        label=trait_plan_label(plan_name, plan, digest=digest),
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

    # RESOLVED BEFORE THE RUN, because the checkpoint directory derives from
    # where the artifact goes. A DERIVED path rather than a flag of its own: a
    # second flag could disagree with this one, and a resumed sweep would then
    # look for its work in a directory nothing wrote to.
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = trait_sweep_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
        checkpoints=out.parent / "checkpoints",
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
        level="INFO",
        format_mode="text",
        service_name="cartridge-trait-sweep",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "entrypoint",
    "main",
    "measure_grid",
    "prepare_traits",
    "require_solo_precondition",
    "trait_sweep_run_record",
]


# Without this, `python -m model_trainer.cli.cartridge_trait_sweep` imports the
# module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
