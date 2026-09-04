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
from model_trainer.cli.cartridge_companion_sweep import (
    COMPANION_SEED_STRIDE,
    cell_observations,
)
from model_trainer.cli.cartridge_composition_sweep import matched_other_train
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.contracts.replicated_measurement import (
    ReplicatedGain,
    gain_observations,
    noise_floor,
    replicate,
)
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies.cartridge import (
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.finetuning.strategies.cartridge_slots import CartridgeSlots
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import (
    held_out_gain,
    train_cartridge,
)
from model_trainer.core.services.model.cartridge_plans import (
    DIVERSE_COMPANION_SWEEP_EXPERIMENT,
    VariedCompanionSweepPlan,
    corpus_digest,
    require_cartridge_plan,
    varied_companion_sweep_label,
)
from model_trainer.core.services.model.cartridge_varied import (
    measure_varied_companioned_scaling,
)
from model_trainer.core.types import CacheCapableLMProto

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


class _DiversePoolProvider:
    """Deterministic per-seed pools, one corpus per member.

    A class for the reason the sibling providers are. Member ``j`` of seed
    ``s``'s pool trains ON THE j-TH COMPANION CORPUS from
    ``s + (COMPANION_SEED_STRIDE + j) * len(seeds)`` -- the exact formula
    the varied provider uses, so with the recorded companion's corpus first
    the pools nest the recorded configuration while every later member
    carries a different voice.
    """

    _base: CacheCapableLMProto
    _companion_trains: Sequence[Sequence[torch.Tensor]]
    _plan: VariedCompanionSweepPlan
    _pools: dict[int, tuple[CartridgeSlots, ...]]

    def __init__(
        self,
        base: CacheCapableLMProto,
        companion_trains: Sequence[Sequence[torch.Tensor]],
        plan: VariedCompanionSweepPlan,
    ) -> None:
        """Hold what the pool builds need.

        Args:
            base: The frozen base.
            companion_trains: One training-window sequence per pool member,
                in pool order.
            plan: The plan being run.
        """
        self._base = base
        self._companion_trains = companion_trains
        self._plan = plan
        self._pools = {}

    def pool(self, seed: int) -> tuple[CartridgeSlots, ...]:
        """The frozen pool for one replicate.

        Args:
            seed: The replicate's base seed.

        Returns:
            One plain-trained companion per corpus, cached so every cell
            that shares this seed shares one pool by identity.
        """
        if seed not in self._pools:
            self._pools[seed] = tuple(
                train_cartridge(
                    self._base,
                    companion_train,
                    num_slots=self._plan["slots"],
                    seed=seed + (COMPANION_SEED_STRIDE + member) * len(self._plan["seeds"]),
                    epochs=self._plan["epochs"],
                    learning_rate=self._plan["learning_rate"],
                )
                for member, companion_train in enumerate(self._companion_trains)
            )
        return self._pools[seed]


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
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    companion_corpora: Sequence[pathlib.Path],
    device: str,
) -> tuple[tuple[Observation, ...], str]:
    """Run every count cell, score every pool member, and name it all.

    Args:
        plan: The measurement to run.
        corpus: Directory of markdown documents whose retention is the
            finding.
        other_corpora: Composition partners, one per additional compartment,
            in the order the counts consume them.
        companion_corpora: One corpus per pool member, in pool order. Their
            count must equal the plan's ``max_companions``, and every entry
            must be disjoint from the primary, the partners, and each other.
        device: Device to measure on.

    Returns:
        ``(observations, digest)``. Beside the cells' arms, one
        ``companion-cross-{j}`` arm per pool member scores that member
        alone on the primary held-out text -- the companion-leakage
        instrument.

    Raises:
        ValueError: Propagated from :func:`_require_admissible_corpora`.
        AppError: Propagated from the corpus and measurement layers.
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
    companion_trains: list[list[torch.Tensor]] = []
    for companion_corpus in companion_corpora:
        companion_documents = _test_hooks.read_corpus_documents(companion_corpus)
        companion_encoded = [tokenizer.encode(document) for document in companion_documents]
        companion_trains.append(
            matched_other_train(
                str(companion_corpus),
                build_windows(companion_encoded, window=plan["window"], device=device),
                held_out_stride=plan["held_out_stride"],
                required=len(train),
            )
        )

    base = require_cache_capable(hf_hooks.Hooks.load_hf_model(plan["model_id"], None))
    base.to(device)
    provider = _DiversePoolProvider(base, companion_trains, plan)

    observations: list[Observation] = [
        Observation(name="slots_per_cartridge", value=float(plan["slots"])),
        Observation(name="max_companions", value=float(plan["max_companions"])),
    ]
    companion_gains: list[list[tuple[int, float]]] = [[] for _ in companion_trains]
    for seed in plan["seeds"]:
        for member, slots in enumerate(provider.pool(seed)):
            companion_gains[member].append(
                (seed, held_out_gain(CartridgeModel(base=base, slots=slots), held_out))
            )
    for member, gains in enumerate(companion_gains):
        arm = replicate(f"companion-cross-{member}", gains)
        _log.info("companion-cross-%d: %+.4f on the primary held-out", member, arm["mean"])
        observations.extend(gain_observations(arm))

    composed_arms: list[ReplicatedGain] = []
    for count in plan["compartment_counts"]:
        arm_name = f"diverse-K{plan['max_companions']}-p{plan['probability']}-n{count}"
        alone, composed, untrained_composed, cross = measure_varied_companioned_scaling(
            base,
            first_train=train,
            other_trains=other_trains[: count - 1],
            held_out=held_out,
            arm=arm_name,
            num_slots=plan["slots"],
            seeds=plan["seeds"],
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
            pool_for_seed=provider.pool,
            companion_probability=plan["probability"],
        )
        _log.info(
            "%s: %+.4f alone -> %+.4f composed, %+.4f untrained-composed",
            arm_name,
            alone["mean"],
            composed["mean"],
            untrained_composed["mean"],
        )
        composed_arms.append(composed)
        observations.extend(cell_observations(arm_name, alone, composed, untrained_composed, cross))
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
) -> RunRecord:
    """Pin determinism, run the grid, and record it.

    Args:
        plan_name: Which plan to run.
        corpus: Directory of markdown documents whose retention is measured.
        other_corpora: Composition partners.
        companion_corpora: One held-out corpus per pool member.
        device: Device to measure on.

    Returns:
        The record.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        ValueError: Propagated from :func:`measure_grid`.
        AppError: Propagated from the corpus and measurement layers.
    """
    plan = require_cartridge_plan(_measurement_hooks.diverse_companion_sweep_plans(), plan_name)
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )
    observations, digest = measure_grid(
        plan,
        corpus=corpus,
        other_corpora=other_corpora,
        companion_corpora=companion_corpora,
        device=device,
    )
    return run_record(
        experiment=DIVERSE_COMPANION_SWEEP_EXPERIMENT,
        label=varied_companion_sweep_label(plan_name, plan, digest=digest),
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
    record = diverse_companion_sweep_run_record(
        cli_args.require_flag(parsed, PLAN_FLAG),
        corpus=pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        other_corpora=others,
        companion_corpora=companions,
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
    )

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
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
