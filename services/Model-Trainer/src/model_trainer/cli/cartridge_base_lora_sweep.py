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
recorded plans); the LoRA trains at :data:`LORA_TRAIN_SEED` (53); crowding
pool members start at :data:`POOL_SEED_BASE` (61).
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
from model_trainer.core.services.finetuning.strategies import _test_hooks as strategy_hooks
from model_trainer.core.services.finetuning.strategies.cartridge import (
    require_cache_capable,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.finetuning.strategies.cartridge_slots import CartridgeSlots
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_base_lora import (
    freeze_adapted,
    train_composition_lora,
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
    BASE_LORA_SWEEP_EXPERIMENT,
    BaseLoraSweepPlan,
    base_lora_sweep_label,
)
from model_trainer.core.services.model.cartridge_varied import (
    measure_varied_companioned_scaling,
)
from model_trainer.core.types import CacheCapableLMProto

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

#: The GPT-2 family fuses query, key and value into one projection; adapting
#: it is adapting attention. A constant rather than a plan field because it
#: is a property of the architecture the plan's ``model_id`` names, not a
#: knob anyone sweeps.
LORA_TARGET_MODULES = ("c_attn",)

#: Seed for the LoRA's own training draw. Sits in the gap between the
#: measurement offsets (at most 48 under the recorded plans) and the
#: crowding pool's seeds (61 up).
LORA_TRAIN_SEED = 53

#: First seed of the crowding pool; member (corpus j, variant m) trains from
#: ``POOL_SEED_BASE + j * pool_members_per_corpus + m``. Chosen past every
#: seed anything else in this measurement uses.
POOL_SEED_BASE = 61


class _MeasurementPoolProvider:
    """Deterministic per-seed companion pools against the ADAPTED base.

    The diverse sweep's provider, rebuilt against the adapted base with its
    knobs passed explicitly: one plain-trained companion per pool corpus,
    member ``j`` from ``seed + (COMPANION_SEED_STRIDE + j) * len(seeds)`` --
    the recorded formula, so the diverse cells here are structured exactly
    as the recorded diverse grid's, differing only in the base underneath.
    """

    _base: CacheCapableLMProto
    _companion_trains: Sequence[Sequence[torch.Tensor]]
    _plan: BaseLoraSweepPlan
    _pools: dict[int, tuple[CartridgeSlots, ...]]

    def __init__(
        self,
        base: CacheCapableLMProto,
        companion_trains: Sequence[Sequence[torch.Tensor]],
        plan: BaseLoraSweepPlan,
    ) -> None:
        """Hold what the pool builds need.

        Args:
            base: The frozen adapted base.
            companion_trains: One training-window sequence per pool corpus.
            plan: The plan being run.
        """
        self._base = base
        self._companion_trains = companion_trains
        self._plan = plan
        self._pools = {}

    def pool(self, seed: int) -> tuple[CartridgeSlots, ...]:
        """The frozen companion pool for one replicate.

        Args:
            seed: The replicate's base seed.

        Returns:
            One plain-trained companion per pool corpus, cached so every
            cell that shares this seed shares one pool by identity.
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
    measured = {str(corpus), *(str(other) for other in other_corpora)}
    overlapping = [name for name in names if name in measured]
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
    corpus: pathlib.Path,
    other_corpora: Sequence[pathlib.Path],
    pool_corpora: Sequence[pathlib.Path],
    device: str,
) -> tuple[tuple[Observation, ...], str]:
    """Adapt the base, then run both cell families against it.

    Args:
        plan: The measurement to run.
        corpus: Directory of markdown documents whose retention is the
            finding.
        other_corpora: Composition partners, in the order the counts consume
            them.
        pool_corpora: The adaptation corpora, one per companion; disjoint
            from everything measured.
        device: Device to measure on.

    Returns:
        ``(observations, digest)``: the LoRA's epoch losses, one
        ``lora-companion-cross-{j}`` arm per measurement-pool member, and
        both cell families' arms with per-family floors.

    Raises:
        ValueError: Propagated from :func:`_require_admissible_corpora`.
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
    lora_corpus = [window for pool_train in pool_trains for window in pool_train]
    epoch_losses = train_composition_lora(
        adapted,
        crowding_pool,
        lora_corpus,
        max_drawn=plan["max_drawn"],
        seed=LORA_TRAIN_SEED,
        epochs=plan["lora_epochs"],
        learning_rate=plan["lora_learning_rate"],
    )
    freeze_adapted(adapted)
    for position, loss in enumerate(epoch_losses):
        _log.info("lora epoch %d mean loss %.4f", position, loss)

    observations: list[Observation] = [
        Observation(name="slots_per_cartridge", value=float(plan["slots"])),
        Observation(name="max_drawn", value=float(plan["max_drawn"])),
        *[
            Observation(name=f"lora-train-epoch-{position}_loss", value=loss)
            for position, loss in enumerate(epoch_losses)
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

    plain_floor = noise_floor(plain_arms)
    observations.append(Observation(name="lora-plain_composed_noise_floor", value=plain_floor))
    observations.extend(sweep_observations(plain_arms, plain_floor))
    diverse_floor = noise_floor(diverse_arms)
    observations.append(Observation(name="lora-diverse_composed_noise_floor", value=diverse_floor))
    observations.extend(sweep_observations(diverse_arms, diverse_floor))
    return tuple(observations), digest


def base_lora_sweep_run_record(
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
        pool_corpora: The adaptation corpora.
        device: Device to measure on.

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
        corpus=corpus,
        other_corpora=other_corpora,
        pool_corpora=pool_corpora,
        device=device,
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
    record = base_lora_sweep_run_record(
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
        level="INFO",
        format_mode="text",
        service_name="cartridge-base-lora-sweep",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "LORA_TARGET_MODULES",
    "LORA_TRAIN_SEED",
    "POOL_SEED_BASE",
    "base_lora_sweep_run_record",
    "entrypoint",
    "main",
    "measure_grid",
]


# Without this, `python -m model_trainer.cli.cartridge_base_lora_sweep`
# imports the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
