"""Measure whether a content-diverse companion pool closes the count-decay.

THE CHAIN THIS EXTENDS. The companion p-sweep (``bc29dc3e``) proved
composition-aware training moves the compartment ceiling; the n8 extension
(``684492dd``) measured content-companionship as the load-bearing kind and
the recipe's decay with count (44.6% at four, 26.5% at eight); the
varied-count sweep (``7815a0fd``) refuted count-exposure as the fix -- its
same-corpus pool taught count-invariance and the decay stayed, naming
CONTENT interference as the cause. This sweep (``d2c03dd4``) runs the named
lever: the pool's K members each train on a DIFFERENT held-out corpus, so
the trainee learns to share attention with different voices, which is what
seven real strangers are.

THE POOL'S FIRST CORPUS SHOULD BE THE RECORDED COMPANION'S. Member zero
trains from the exact seed formula the recorded grids used, so when the
first ``--companion-corpora`` entry is the single-companion grid's corpus,
member zero IS the recorded companion byte for byte and the three records
isolate exactly one difference each.

COMPANION CROSS-GAIN IS NEW, AND IT CLOSES A GAP. Every earlier grid
measured PARTNER relatedness (the cross-gain arm caught a leaked roster)
but ASSUMED its companion clean. Here every pool member is scored alone on
the primary held-out text, per seed, so a companion whose corpus secretly
predicts the primary is convicted by the record instead of trusted.

THE SOLO-COST AXIS IS HALF THE ANSWER, as always.
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
from model_trainer.cli.cartridge_benchmark import sweep_observations
from model_trainer.cli.cartridge_companion_sweep import (
    cell_observations,
)
from model_trainer.cli.cartridge_composition_sweep import staged_partner_trains
from model_trainer.cli.cartridge_pool_provider import SeededPoolProvider
from model_trainer.cli.cartridge_solo_seeds import resolve_precision
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision
from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    gain_observations,
    noise_floor,
    per_seed_observations,
    replicate,
    replicated_from_observations,
)
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies.cartridge import (
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import (
    held_out_gain,
    train_cartridge,
)
from model_trainer.core.services.model.cartridge_plans import (
    corpus_digest,
    digest_parts,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_pool_plans import (
    DIVERSE_COMPANION_SWEEP_EXPERIMENT,
    VariedCompanionSweepPlan,
    varied_companion_sweep_label,
)
from model_trainer.core.services.model.cartridge_sweep_checkpoint import (
    bind_cells,
    checkpointed_cells,
)
from model_trainer.core.services.model.cartridge_varied import (
    measure_varied_companioned_scaling,
)

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
OTHER_CORPORA_FLAG = "--other-corpora"
COMPANION_CORPORA_FLAG = "--companion-corpora"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (
    PLAN_FLAG,
    CORPUS_FLAG,
    OTHER_CORPORA_FLAG,
    COMPANION_CORPORA_FLAG,
    DEVICE_FLAG,
    OUT_FLAG,
)


def _require_admissible_corpora(
    plan: VariedCompanionSweepPlan,
    *,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    companion_corpora: Sequence[pathlib.Path],
) -> None:
    """Refuse a corpus layout the measurement could not honestly run on.

    Split from :func:`measure_grid` when the fourth refusal pushed it over
    the complexity ceiling; the refusals are one decision and read as one.

    Args:
        plan: The measurement being run.
        corpus: The primary corpus.
        other_corpora: Composition partners.
        companion_corpora: One corpus per pool member.

    Raises:
        ValueError: If too few other corpora are supplied, the companion
            count mismatches the plan, or a companion corpus repeats or
            overlaps the measured corpora.
    """
    largest = max(plan["compartment_counts"])
    if len(other_corpora) < largest - 1:
        raise ValueError(
            f"the plan composes up to {largest} compartments, which needs "
            f"{largest - 1} other corpora; {len(other_corpora)} supplied"
        )
    if len(companion_corpora) != plan["max_companions"]:
        raise ValueError(
            f"the plan draws from a pool of {plan['max_companions']} companions, "
            f"one corpus each; {len(companion_corpora)} companion corpora supplied"
        )
    names = [str(entry) for entry in companion_corpora]
    if len(set(names)) != len(names):
        raise ValueError(
            f"the companion corpora repeat ({', '.join(names)}); a repeated corpus "
            f"is the varied sweep's same-content pool wearing a diverse label"
        )
    measured = {str(corpus), *(str(other) for other in other_corpora)}
    overlapping = [name for name in names if name in measured]
    if overlapping:
        raise ValueError(
            f"the companion corpora {', '.join(overlapping)} are also measured "
            f"corpora; a cartridge trained beside its future partner would be "
            f"measured on partner memorisation, not composition robustness -- "
            f"supply corpora held out from the composition"
        )


def measure_grid(
    plan: VariedCompanionSweepPlan,
    *,
    plan_name: str,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    companion_corpora: Sequence[pathlib.Path],
    device: str,
    load_precision: QuantizationConfig | StoredBf16Precision | None,
    precision_token: str,
    checkpoints: pathlib.Path,
) -> tuple[tuple[Observation, ...], str]:
    """Run every count cell, score every pool member, and name it all.

    RESUMABLE PER CELL: the naive baseline, the companion-cross arms, and
    each compartment count. Every one of those is a block of rows that is
    complete on its own and costs ``len(seeds)`` cartridge trainings or more,
    which is what an eviction takes.

    WHAT A RESUME PAYS AGAIN, STATED PLAINLY. The setup above the cells is
    NOT checkpointed: the corpora are re-read and the base is re-loaded. Both
    are deterministic, so reproducing them costs time but cannot change a
    number. Note that the companion POOL is not setup -- it is trained lazily
    inside the cells that ask for it, and cached per seed -- so a resume
    retrains only the pools its remaining cells actually need.

    Args:
        plan: The measurement to run.
        plan_name: Which plan this is, so the checkpoint can tell one plan's
            cells from another's.
        corpus: Directory of markdown documents whose retention is the
            finding.
        other_corpora: Composition partners, one per additional compartment,
            in the order the counts consume them.
        companion_corpora: One corpus per pool member, in pool order. Their
            count must equal the plan's ``max_companions``, and every entry
            must be disjoint from the primary, the partners, and each other.
        device: Device to measure on.
        load_precision: What the loader is handed, resolved from the plan's
            ``precision_selector`` by the caller -- a declared value, so a
            record can always say what precision it measured.
        precision_token: The label segment that separates two precisions'
            records, from the same resolution. Part of what identifies the
            checkpoint, so a run at one precision cannot resume cells
            measured at the other.
        checkpoints: Directory holding this sweep's checkpoint.

    Returns:
        ``(observations, digest)``. Beside the cells' arms, one
        ``companion-cross-{j}`` arm per pool member scores that member
        alone on the primary held-out text -- the companion-leakage
        instrument -- and the ``naive-solo`` arm trains one cartridge per
        seed by the exact solo formula behind the same base load: the
        in-record naive baseline every companioned arm pairs against, and
        the readability gate where a certified solo record exists for the
        plan's base.

    Raises:
        ValueError: Propagated from :func:`_require_admissible_corpora`.
        AppError: Propagated from the corpus and measurement layers, and
            from the checkpoint layer when a checkpoint on disk describes
            different inputs.
    """
    _require_admissible_corpora(
        plan,
        corpus=corpus,
        other_corpora=other_corpora,
        companion_corpora=companion_corpora,
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
    companion_trains, companion_digests = staged_partner_trains(
        companion_corpora,
        tokenizer=tokenizer,
        window=plan["window"],
        held_out_stride=plan["held_out_stride"],
        required=len(train),
        device=device,
    )

    base = require_cache_capable(hf_hooks.Hooks.load_hf_model(plan["model_id"], load_precision))
    base.to(device)
    provider = SeededPoolProvider(
        base,
        companion_trains,
        slots=plan["slots"],
        seeds=plan["seeds"],
        epochs=plan["epochs"],
        learning_rate=plan["learning_rate"],
    )

    observations: list[Observation] = [
        Observation(name="slots_per_cartridge", value=float(plan["slots"])),
        Observation(name="max_companions", value=float(plan["max_companions"])),
    ]

    def _naive_solo() -> tuple[Observation, ...]:
        """Train one plain cartridge per seed and score it alone.

        The in-record naive baseline, and the record's readability gate: one
        cartridge per seed trained by the exact solo formula -- plain seed,
        the plan's own knobs -- behind the same base load. Where a certified
        solo record exists for this base at these knobs, these per-seed gains
        must reproduce it bit-for-bit, and every companioned arm's per-seed
        subtraction pairs against them inside one record.

        Returns:
            The ``naive-solo`` arm's mean, spread and per-seed gains.
        """
        naive_gains: list[tuple[int, float]] = []
        for seed in plan["seeds"]:
            naive_slots = train_cartridge(
                base,
                train,
                num_slots=plan["slots"],
                seed=seed,
                epochs=plan["epochs"],
                learning_rate=plan["learning_rate"],
            )
            naive_gains.append(
                (seed, held_out_gain(CartridgeModel(base=base, slots=naive_slots), held_out))
            )
        naive = replicate("naive-solo", naive_gains)
        _log.info("naive-solo: %+.4f mean over %d seeds", naive["mean"], len(naive_gains))
        return (*gain_observations(naive), *per_seed_observations(naive))

    def _companion_cross() -> tuple[Observation, ...]:
        """Score every pool member alone on the primary held-out text.

        ONE CELL FOR ALL THE MEMBERS, because ``provider.pool(seed)`` trains
        a seed's whole pool in a single call: splitting per member would
        retrain the pool once per member and discard all but one each time.

        Returns:
            Each member's arm, with its mean, spread and per-seed gains.
        """
        companion_gains: list[list[tuple[int, float]]] = [[] for _ in companion_trains]
        for seed in plan["seeds"]:
            for member, slots in enumerate(provider.pool(seed)):
                companion_gains[member].append(
                    (seed, held_out_gain(CartridgeModel(base=base, slots=slots), held_out))
                )
        scored: list[Observation] = []
        for member, gains in enumerate(companion_gains):
            arm = replicate(f"companion-cross-{member}", gains)
            _log.info("companion-cross-%d: %+.4f on the primary held-out", member, arm["mean"])
            scored.extend(gain_observations(arm))
            scored.extend(per_seed_observations(arm))
        return tuple(scored)

    def _count_cell(count: int) -> tuple[Observation, ...]:
        """Measure the diverse-companion arm at one compartment count.

        Args:
            count: How many cartridges are composed.

        Returns:
            Every arm's mean, spread and per-seed gains, the retention ratio
            where it is readable, and the interference verdict.
        """
        alone, composed, untrained_composed, cross = measure_varied_companioned_scaling(
            base,
            first_train=train,
            other_trains=other_trains[: count - 1],
            held_out=held_out,
            arm=_arm_name(count),
            num_slots=plan["slots"],
            seeds=plan["seeds"],
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
            pool_for_seed=provider.pool,
            companion_probability=plan["probability"],
        )
        _log.info(
            "%s: %+.4f alone -> %+.4f composed, %+.4f untrained-composed",
            _arm_name(count),
            alone["mean"],
            composed["mean"],
            untrained_composed["mean"],
        )
        return cell_observations(_arm_name(count), alone, composed, untrained_composed, cross)

    def _arm_name(count: int) -> str:
        """Name the cell at one compartment count.

        ONE SPELLING FOR THE ARM, used by the measurement, the log line and
        the rebuild below. Three copies of an f-string is three chances for
        the rebuild to look for an arm the measurement never wrote.

        Args:
            count: How many cartridges are composed.

        Returns:
            The arm name.
        """
        return f"diverse-K{plan['max_companions']}-p{plan['probability']}-n{count}"

    produced = checkpointed_cells(
        checkpoints,
        measurement=f"diverse-companion-{plan_name}",
        # THE LABEL IS THE IDENTITY: it already carries every measurement
        # field, the precision token and the primary digest, and it is what
        # two runs are paired by. The partner and companion digests are
        # appended because the label does not reach them.
        inputs_digest=digest_parts(
            [
                varied_companion_sweep_label(
                    plan_name, plan, digest=digest, precision_token=precision_token
                ),
                *other_digests,
                *companion_digests,
            ]
        ),
        cells=[
            ("naive-solo", _naive_solo),
            ("companion-cross", _companion_cross),
            *bind_cells(
                [(f"n{count}", count) for count in plan["compartment_counts"]], _count_cell
            ),
        ],
    )

    # WALKED IN THE DECLARED ORDER rather than over what came back, so a
    # resumed sweep emits the rows a straight run would have, in the same
    # places.
    observations.extend(produced["naive-solo"])
    observations.extend(produced["companion-cross"])
    composed_arms: list[ReplicatedGain] = []
    for count in plan["compartment_counts"]:
        rows = produced[f"n{count}"]
        observations.extend(rows)
        # REBUILT FROM THE ROWS RATHER THAN KEPT FROM THE CALL, because a
        # resumed cell never made the call. The rebuild is exact -- every
        # field of an arm is a function of its per-seed gains, and those are
        # in the rows -- so the floor below is computed over the same arms
        # either way.
        composed_arms.append(
            replicated_from_observations(
                rows, arm=f"{_arm_name(count)}-composed", seeds=plan["seeds"]
            )
        )
    floor = noise_floor(composed_arms)
    observations.append(Observation(name="diverse_composed_noise_floor", value=floor))
    observations.extend(sweep_observations(composed_arms, floor))
    return tuple(observations), digest


def diverse_companion_sweep_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    companion_corpora: Sequence[pathlib.Path],
    device: str,
    checkpoints: pathlib.Path,
) -> RunRecord:
    """Pin determinism, run the grid, and record it.

    Args:
        plan_name: Which plan to run.
        corpus: Directory of markdown documents whose retention is measured.
        other_corpora: Composition partners.
        companion_corpora: One held-out corpus per pool member.
        device: Device to measure on.
        checkpoints: Directory holding this sweep's checkpoint, so an evicted
            run resumes at its last completed cell rather than at zero.

    Returns:
        The record.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        ValueError: Propagated from :func:`measure_grid`, or from
            :func:`resolve_precision` for a plan whose selector or
            (model, precision) pair is undeclared.
        AppError: Propagated from the corpus and measurement layers.
    """
    plan = require_cartridge_plan(_measurement_hooks.diverse_companion_sweep_plans(), plan_name)
    load_precision, precision_token = resolve_precision(
        plan["model_id"], plan["precision_selector"]
    )
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )
    observations, digest = measure_grid(
        plan,
        plan_name=plan_name,
        corpus=corpus,
        other_corpora=other_corpora,
        companion_corpora=companion_corpora,
        device=device,
        load_precision=load_precision,
        precision_token=precision_token,
        checkpoints=checkpoints,
    )
    return run_record(
        experiment=DIVERSE_COMPANION_SWEEP_EXPERIMENT,
        label=varied_companion_sweep_label(
            plan_name, plan, digest=digest, precision_token=precision_token
        ),
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
            required flag is absent, too few other corpora are named, the
            companion count mismatches the plan, or a companion corpus is
            repeated or not held out.
        KeyError: If the plan name is unknown.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    others = [
        pathlib.Path(entry)
        for entry in cli_args.require_flag(parsed, OTHER_CORPORA_FLAG).split(",")
        if entry
    ]
    companions = [
        pathlib.Path(entry)
        for entry in cli_args.require_flag(parsed, COMPANION_CORPORA_FLAG).split(",")
        if entry
    ]
    # RESOLVED BEFORE THE RUN, because the checkpoint directory derives from
    # where the artifact goes. A DERIVED path rather than a flag of its own:
    # a second flag could disagree with this one, and a resumed sweep would
    # then look for its work in a directory nothing wrote to.
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))

    record = diverse_companion_sweep_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        other_corpora=others,
        companion_corpora=companions,
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
        checkpoints=out.parent / "checkpoints",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "diverse companion sweep %s %s -> %s",
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
        service_name="cartridge-diverse-companion-sweep",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "diverse_companion_sweep_run_record",
    "entrypoint",
    "main",
    "measure_grid",
]


# Without this, `python -m model_trainer.cli.cartridge_diverse_companion_sweep`
# imports the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
