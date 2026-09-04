"""Measure whether composition-aware training moves the two-compartment ceiling.

THE BASELINE THIS SUBTRACTS AGAINST. The composition-scaling sweep (board
task ``a67d6038``) measured plain-trained cartridges at fixed 64 slots
retaining 62.8% of their held-out gain at two compartments and -45.4% at
four, and its untrained-composed control attributed the small-scale cost to
STRUCTURE: the base was never trained to read a prefix with company in it.
This sweep runs the intervention (board task ``bc29dc3e``): every cartridge
is trained with a frozen companion present at a swept probability, and the
same arms are measured. Probability zero is not a row here -- it IS the
baseline record, and re-running it would register those numbers twice.

TWO COMPANION KINDS, BECAUSE THE INTERVENTION HAS TWO READINGS. A noise
companion teaches tolerance of foreign structure and nothing else; a trained
companion teaches tolerance of foreign CONTENT. The trained companion's
corpus is a flag of its own and MUST be disjoint from the composition
partners: a cartridge trained beside its future partner would be measured on
partner memorisation, not composition robustness, which is the same shape of
artifact as the two-halves 94%.

THE SOLO-COST AXIS IS HALF THE ANSWER. Companioned training could buy
composed retention by spending alone-performance. Every combination's alone
arm is emitted beside its composed arm, and a verdict that reads only the
composed axis is wrong by construction.
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
from model_trainer.cli.cartridge_benchmark import sweep_observations
from model_trainer.cli.cartridge_composition_sweep import matched_other_train
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
from model_trainer.core.services.finetuning.strategies.cartridge import (
    measure_geometry,
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_slots import (
    CartridgeSlots,
    initialise_slots,
)
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_companioned import measure_companioned_scaling
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import train_cartridge
from model_trainer.core.services.model.cartridge_plans import (
    COMPANION_SWEEP_EXPERIMENT,
    CompanionSweepPlan,
    companion_sweep_label,
    corpus_digest,
    require_cartridge_plan,
)
from model_trainer.core.types import CacheCapableLMProto

_log = get_logger(__name__)

PLAN_FLAG = "--plan"
CORPUS_FLAG = "--corpus"
OTHER_CORPORA_FLAG = "--other-corpora"
COMPANION_CORPUS_FLAG = "--companion-corpus"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (
    PLAN_FLAG,
    CORPUS_FLAG,
    OTHER_CORPORA_FLAG,
    COMPANION_CORPUS_FLAG,
    DEVICE_FLAG,
    OUT_FLAG,
)

#: The two companion kinds, in the order they run and are recorded.
COMPANION_KINDS = ("noise", "trained")

#: Seed offset for the companion's own draw or training, chosen past every
#: offset the measurement's other cartridges use: the k-th other corpus draws
#: at ``seed + (k + 1) * len(seeds)``, so with counts up to eight this offset
#: stays clear of them all.
COMPANION_SEED_STRIDE = 11


class _CompanionProviders:
    """Deterministic per-seed companions, one trained cartridge per seed.

    A class rather than closures so the trained companion's cache is a named
    thing a test can inspect: the trained companion is expensive, must be
    identical wherever one seed's replicate uses it, and training it once
    per seed is only correct because :func:`train_cartridge` re-seeds the
    global generator itself -- reuse changes no downstream draw.
    """

    _base: CacheCapableLMProto
    _companion_train: Sequence[torch.Tensor]
    _plan: CompanionSweepPlan
    _trained: dict[int, CartridgeSlots]

    def __init__(
        self,
        base: CacheCapableLMProto,
        companion_train: Sequence[torch.Tensor],
        plan: CompanionSweepPlan,
    ) -> None:
        """Hold what both providers need.

        Args:
            base: The frozen base.
            companion_train: Training windows for the trained companion,
                from the held-out companion corpus.
            plan: The plan being run.
        """
        self._base = base
        self._companion_train = companion_train
        self._plan = plan
        self._trained = {}

    def _companion_seed(self, seed: int) -> int:
        """The seed a replicate's companion draws or trains from.

        Args:
            seed: The replicate's base seed.

        Returns:
            The offset seed.
        """
        return seed + COMPANION_SEED_STRIDE * len(self._plan["seeds"])

    def noise(self, seed: int) -> CartridgeSlots:
        """A fresh untrained companion for one replicate.

        Args:
            seed: The replicate's base seed.

        Returns:
            Newly drawn slots, never trained.
        """
        geometry = measure_geometry(self._base, num_slots=self._plan["slots"])
        return initialise_slots(geometry, seed=self._companion_seed(seed))

    def trained(self, seed: int) -> CartridgeSlots:
        """The plain-trained held-out-corpus companion for one replicate.

        Args:
            seed: The replicate's base seed.

        Returns:
            The trained slots, cached so every combination in the grid that
            shares this seed shares one companion object.
        """
        if seed not in self._trained:
            self._trained[seed] = train_cartridge(
                self._base,
                self._companion_train,
                num_slots=self._plan["slots"],
                seed=self._companion_seed(seed),
                epochs=self._plan["epochs"],
                learning_rate=self._plan["learning_rate"],
            )
        return self._trained[seed]


def cell_observations(
    arm: str,
    alone: ReplicatedGain,
    composed: ReplicatedGain,
    untrained_composed: ReplicatedGain,
    cross: Sequence[ReplicatedGain],
) -> tuple[Observation, ...]:
    """Name one grid cell's numbers for the record.

    Pure assembly, split from the grid walk so the ratio-absence rule is
    testable against constructed arms: forcing a real cartridge to fail its
    own corpus deterministically turned out to be harder than the failure
    itself (the tiny rung learns positively even at learning rate 50).

    THE RATIO EXISTS ONLY WHERE THE ALONE ARM IMPROVED ON THE BASE -- the
    same condition retention() itself refuses on. A heavily companioned
    cartridge can fail to learn its own corpus (measured: noise at p=1.0
    scores -0.68 alone), and that is a RESULT this grid exists to find, not
    an error: the raw arms carry it, and an absent retention reads as
    'alone did not improve on base', checkable from the alone mean's sign
    in the same record.

    Args:
        arm: The cell's name, e.g. ``"noise-p0.5-n4"``.
        alone: The solo-cost arm.
        composed: The full trained composition.
        untrained_composed: The noise-composition control.
        cross: One cross-gain arm per other corpus.

    Returns:
        Gain observations for every arm, the retention ratio where it is
        readable, and the composed-versus-untrained interference verdict.
    """
    named: list[Observation] = []
    for measured in [alone, composed, untrained_composed, *cross]:
        named.extend(gain_observations(measured))
    if alone["mean"] > 0.0:
        named.append(Observation(name=f"{arm}_retention", value=retention(alone, composed)))
    named.extend(
        sweep_observations(
            [composed, untrained_composed],
            noise_floor([composed, untrained_composed]),
        )
    )
    return tuple(named)


def measure_grid(
    plan: CompanionSweepPlan,
    *,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    companion_corpus: pathlib.Path,
    device: str,
) -> tuple[tuple[Observation, ...], str]:
    """Run every (kind, probability, count) cell and name what it produced.

    Args:
        plan: The measurement to run.
        corpus: Directory of markdown documents whose retention is the
            finding.
        other_corpora: Composition partners, one per additional compartment,
            in the order the counts consume them.
        companion_corpus: The trained companion's corpus. Must not appear
            among the partners or as the primary.
        device: Device to measure on.

    Returns:
        ``(observations, digest)`` -- the named numbers, and the digest of
        the primary corpus. A cell's ``_retention`` observation is absent
        when its alone arm did not improve on the base; the raw arm means
        are always present and carry that verdict in their sign.

    Raises:
        ValueError: If too few other corpora are supplied, or the companion
            corpus is also a partner or the primary.
        AppError: Propagated from the corpus and measurement layers.
    """
    largest = max(plan["compartment_counts"])
    if len(other_corpora) < largest - 1:
        raise ValueError(
            f"the plan composes up to {largest} compartments, which needs "
            f"{largest - 1} other corpora; {len(other_corpora)} supplied"
        )
    overlapping = [str(other) for other in other_corpora if str(other) == str(companion_corpus)]
    if str(companion_corpus) == str(corpus) or overlapping:
        raise ValueError(
            f"the companion corpus {companion_corpus} is also a measured corpus; a "
            f"cartridge trained beside its future partner would be measured on "
            f"partner memorisation, not composition robustness -- supply a corpus "
            f"held out from the composition"
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
    companion_documents = _test_hooks.read_corpus_documents(companion_corpus)
    companion_encoded = [tokenizer.encode(document) for document in companion_documents]
    companion_train = matched_other_train(
        str(companion_corpus),
        build_windows(companion_encoded, window=plan["window"], device=device),
        held_out_stride=plan["held_out_stride"],
        required=len(train),
    )

    base = require_cache_capable(hf_hooks.Hooks.load_hf_model(plan["model_id"], None))
    base.to(device)
    providers = _CompanionProviders(base, companion_train, plan)

    observations: list[Observation] = [
        Observation(name="slots_per_cartridge", value=float(plan["slots"]))
    ]
    for kind in COMPANION_KINDS:
        provider = providers.noise if kind == "noise" else providers.trained
        for probability in plan["probabilities"]:
            composed_arms: list[ReplicatedGain] = []
            for count in plan["compartment_counts"]:
                arm = f"{kind}-p{probability}-n{count}"
                alone, composed, untrained_composed, cross = measure_companioned_scaling(
                    base,
                    first_train=train,
                    other_trains=other_trains[: count - 1],
                    held_out=held_out,
                    arm=arm,
                    num_slots=plan["slots"],
                    seeds=plan["seeds"],
                    epochs=plan["epochs"],
                    learning_rate=plan["learning_rate"],
                    companion_for_seed=provider,
                    companion_probability=probability,
                )
                _log.info(
                    "%s: %+.4f alone -> %+.4f composed, %+.4f untrained-composed",
                    arm,
                    alone["mean"],
                    composed["mean"],
                    untrained_composed["mean"],
                )
                composed_arms.append(composed)
                observations.extend(
                    cell_observations(arm, alone, composed, untrained_composed, cross)
                )
            floor = noise_floor(composed_arms)
            observations.append(
                Observation(name=f"{kind}-p{probability}_composed_noise_floor", value=floor)
            )
            observations.extend(sweep_observations(composed_arms, floor))
    return tuple(observations), digest


def companion_sweep_run_record(
    plan_name: str,
    *,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    companion_corpus: pathlib.Path,
    device: str,
) -> RunRecord:
    """Pin determinism, run the grid, and record it.

    Args:
        plan_name: Which plan to run.
        corpus: Directory of markdown documents whose retention is measured.
        other_corpora: Composition partners.
        companion_corpus: The trained companion's held-out corpus.
        device: Device to measure on.

    Returns:
        The record.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        ValueError: Propagated from :func:`measure_grid`.
        AppError: Propagated from the corpus and measurement layers.
    """
    plan = require_cartridge_plan(_measurement_hooks.companion_sweep_plans(), plan_name)
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )
    observations, digest = measure_grid(
        plan,
        corpus=corpus,
        other_corpora=other_corpora,
        companion_corpus=companion_corpus,
        device=device,
    )
    return run_record(
        experiment=COMPANION_SWEEP_EXPERIMENT,
        label=companion_sweep_label(plan_name, plan, digest=digest),
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
            required flag is absent, too few other corpora are named, or the
            companion corpus is not held out.
        KeyError: If the plan name is unknown.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    others = [
        pathlib.Path(entry)
        for entry in cli_args.require_flag(parsed, OTHER_CORPORA_FLAG).split(",")
        if entry
    ]
    record = companion_sweep_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        other_corpora=others,
        companion_corpus=pathlib.Path(cli_args.require_flag(parsed, COMPANION_CORPUS_FLAG)),
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
    )

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "companion sweep %s %s -> %s",
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
        service_name="cartridge-companion-sweep",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "COMPANION_KINDS",
    "COMPANION_SEED_STRIDE",
    "cell_observations",
    "companion_sweep_run_record",
    "entrypoint",
    "main",
    "measure_grid",
]


# Without this, `python -m model_trainer.cli.cartridge_companion_sweep`
# imports the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
