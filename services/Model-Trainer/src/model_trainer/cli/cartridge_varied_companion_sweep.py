"""Measure whether varied-count companionship closes the count-decay.

THE BASELINE THIS SUBTRACTS AGAINST. The companion p-sweep (board task
``bc29dc3e``) proved composition-aware training moves the two-compartment
ceiling, and its n8 extension (``684492dd``) measured the recipe's decay
with deployment count: trained-p0.5 retention is 44.6% at four compartments
and 26.5% at eight, where every cartridge trained beside exactly ONE
64-slot companion. This sweep runs the intervention over that decay (board
task ``7815a0fd``): every cartridge trains through a pool of frozen
companions with the PER-STEP COUNT drawn, so gradients see the prefix at
several lengths instead of one.

ONE KIND, ONE PROBABILITY, BECAUSE BOTH KNOBS ARE ALREADY MEASURED.
Content-companionship dominated noise on every axis in the p-sweep and
noise's benefit vanished entirely at n8, so the pool is trained-kind only;
p=0.5 was the best dose twice over. The plan varies exactly the new thing:
the count distribution.

THE POOL HOLDS CONTENT FIXED. Every member is a seed-variant cartridge
plain-trained on the SAME held-out corpus, so the swept axis is count, not
content diversity -- a pool of different corpora would smuggle a second
variable into the cell. The pool's first member trains from the exact seed
the single-companion grid's companion used, so the pools nest the recorded
configuration. The companion corpus MUST be disjoint from the composition
partners, refused exactly as the sibling sweep refuses it.

THE SOLO-COST AXIS IS HALF THE ANSWER, as always: every combination's
alone arm is emitted beside its composed arm.
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
    noise_floor,
)
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies.cartridge import (
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_slots import CartridgeSlots
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_measurement import train_cartridge
from model_trainer.core.services.model.cartridge_plans import (
    VARIED_COMPANION_SWEEP_EXPERIMENT,
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


class _PoolProvider:
    """Deterministic per-seed pools of seed-variant trained companions.

    A class for the reason the sibling sweep's ``_CompanionProviders`` is: a
    trained pool is expensive, must be identical wherever one seed's
    replicate uses it, and the cache is a named thing a test can inspect.
    Member ``j`` of seed ``s``'s pool trains from
    ``s + (COMPANION_SEED_STRIDE + j) * len(seeds)``: at ``j = 0`` that is
    exactly the single-companion grid's companion seed, so the pools nest
    the recorded configuration, and the stride by ``len(seeds)`` keeps every
    member seed distinct across replicates (each is congruent to its
    replicate's seed modulo the seed count).
    """

    _base: CacheCapableLMProto
    _companion_train: Sequence[torch.Tensor]
    _plan: VariedCompanionSweepPlan
    _pools: dict[int, tuple[CartridgeSlots, ...]]

    def __init__(
        self,
        base: CacheCapableLMProto,
        companion_train: Sequence[torch.Tensor],
        plan: VariedCompanionSweepPlan,
    ) -> None:
        """Hold what the pool builds need.

        Args:
            base: The frozen base.
            companion_train: Training windows for every pool member, from
                the held-out companion corpus.
            plan: The plan being run.
        """
        self._base = base
        self._companion_train = companion_train
        self._plan = plan
        self._pools = {}

    def pool(self, seed: int) -> tuple[CartridgeSlots, ...]:
        """The frozen pool for one replicate.

        Args:
            seed: The replicate's base seed.

        Returns:
            ``max_companions`` plain-trained seed-variant companions, cached
            so every cell that shares this seed shares one pool by identity.
        """
        if seed not in self._pools:
            self._pools[seed] = tuple(
                train_cartridge(
                    self._base,
                    self._companion_train,
                    num_slots=self._plan["slots"],
                    seed=seed + (COMPANION_SEED_STRIDE + member) * len(self._plan["seeds"]),
                    epochs=self._plan["epochs"],
                    learning_rate=self._plan["learning_rate"],
                )
                for member in range(self._plan["max_companions"])
            )
        return self._pools[seed]


def measure_grid(
    plan: VariedCompanionSweepPlan,
    *,
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    companion_corpus: pathlib.Path,
    device: str,
) -> tuple[tuple[Observation, ...], str]:
    """Run every count cell and name what it produced.

    Args:
        plan: The measurement to run.
        corpus: Directory of markdown documents whose retention is the
            finding.
        other_corpora: Composition partners, one per additional compartment,
            in the order the counts consume them.
        companion_corpus: The pool's corpus. Must not appear among the
            partners or as the primary.
        device: Device to measure on.

    Returns:
        ``(observations, digest)`` -- the named numbers, and the digest of
        the primary corpus. A cell's ``_retention`` observation is absent
        when its alone arm did not improve on the base, per the sibling
        sweep's ratio rule.

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
    provider = _PoolProvider(base, companion_train, plan)

    observations: list[Observation] = [
        Observation(name="slots_per_cartridge", value=float(plan["slots"])),
        Observation(name="max_companions", value=float(plan["max_companions"])),
    ]
    composed_arms: list[ReplicatedGain] = []
    for count in plan["compartment_counts"]:
        arm = f"varied-K{plan['max_companions']}-p{plan['probability']}-n{count}"
        alone, composed, untrained_composed, cross = measure_varied_companioned_scaling(
            base,
            first_train=train,
            other_trains=other_trains[: count - 1],
            held_out=held_out,
            arm=arm,
            num_slots=plan["slots"],
            seeds=plan["seeds"],
            epochs=plan["epochs"],
            learning_rate=plan["learning_rate"],
            pool_for_seed=provider.pool,
            companion_probability=plan["probability"],
        )
        _log.info(
            "%s: %+.4f alone -> %+.4f composed, %+.4f untrained-composed",
            arm,
            alone["mean"],
            composed["mean"],
            untrained_composed["mean"],
        )
        composed_arms.append(composed)
        observations.extend(cell_observations(arm, alone, composed, untrained_composed, cross))
    floor = noise_floor(composed_arms)
    observations.append(Observation(name="varied_composed_noise_floor", value=floor))
    observations.extend(sweep_observations(composed_arms, floor))
    return tuple(observations), digest


def varied_companion_sweep_run_record(
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
        companion_corpus: The pool's held-out corpus.
        device: Device to measure on.

    Returns:
        The record.

    Raises:
        KeyError: If the plan name is unknown, naming the plans that exist.
        ValueError: Propagated from :func:`measure_grid`.
        AppError: Propagated from the corpus and measurement layers.
    """
    plan = require_cartridge_plan(_measurement_hooks.varied_companion_sweep_plans(), plan_name)
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
        experiment=VARIED_COMPANION_SWEEP_EXPERIMENT,
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
    record = varied_companion_sweep_run_record(
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
        "varied companion sweep %s %s -> %s",
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
        service_name="cartridge-varied-companion-sweep",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "entrypoint",
    "main",
    "measure_grid",
    "varied_companion_sweep_run_record",
]


# Without this, `python -m model_trainer.cli.cartridge_varied_companion_sweep`
# imports the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
