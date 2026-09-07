"""Nine-seed solo cartridge gains on one base: the reliability measurement.

WHY THIS EXISTS (board task ``b89cd348``, successor to the headroom arm
``afee6162``). Headroom explains the 7B solo-gain CEILING -- ~0.4 nats
available on Pythia-6.9B where the GPT-2 family had ~0.8 -- but it
cannot explain seed 9 going negative beside seeds 7 and 8 at ~0.4 in
the same cell. Whether that is a real failure rate or a three-seed
accident decides whether any 7B composition rung is worth a card, and
three seeds cannot answer it. Nine can put a denominator under it.

THE FIRST THREE SEEDS ARE THE ORIGINAL DRAW, DELIBERATELY, per the
``gpt2-wiki-9seed`` precedent: the 7/8/9 subset of this run must
reproduce the recorded cells' behaviour under the same knobs; if it
does not, something other than replication changed and no number here
can be read until that is explained.

Every knob is read off the recorded plan row through the same plan hook
the sweeps use -- window, stride, slot count, epochs, learning rate --
so this measurement trains exactly the cartridge every record trained,
just more of them, alone, with nothing else in the frame: no LoRA, no
pool, no composition. The variance in question is cartridge training
itself.
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
from model_trainer.cli.cartridge_headroom import GEOMETRY_PLAN_NAME, base_short
from model_trainer.cli.cartridge_lora_policy import quantization_for
from model_trainer.cli.known_answer_probe import probe_determinism
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
    require_cartridge_plan,
)

_log = get_logger(__name__)

MODEL_FLAG = "--model-id"
CORPUS_FLAG = "--corpus"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (MODEL_FLAG, CORPUS_FLAG, DEVICE_FLAG, OUT_FLAG)

#: Fixed for the reason every experiment name here is: the string is what
#: two records are grouped by, and a drifting one silently unpairs them.
SOLO_SEEDS_EXPERIMENT = "cartridge-solo-seeds"

#: Nine seeds, the first three being the original recorded draw so the
#: 7/8/9 subset reproduces the recorded cells or loudly fails to.
SOLO_SEEDS = (7, 8, 9, 10, 11, 12, 13, 14, 15)


def solo_seeds_label(*, model_id: str, seeds: Sequence[int], digest: str) -> str:
    """Build the label identifying one solo-seeds measurement.

    Args:
        model_id: The base measured.
        seeds: The seeds trained.
        digest: The corpus digest.

    Returns:
        The label.
    """
    return f"cartridge-solo-seeds-{base_short(model_id)}-n{len(seeds)}-{digest}"


def measure_solo_seeds(
    corpus: pathlib.Path,
    *,
    model_id: str,
    seeds: Sequence[int],
    device: str,
) -> tuple[tuple[Observation, ...], str]:
    """Train one solo cartridge per seed and score each alone.

    Args:
        corpus: Directory of markdown documents; its held-out split is
            what every cartridge is scored on.
        model_id: The base, resolvable by the loading policy.
        seeds: Seeds to train, one cartridge each.
        device: Device to measure on.

    Returns:
        ``(observations, digest)``: one gain row per seed, the mean and
        spread over them, and the corpus/window counts behind them.

    Raises:
        ValueError: When no seed is named; a reliability measurement
            with no draws would still carry a label and read as one.
        AppError: Propagated from the corpus, loading and training
            layers.
    """
    if len(seeds) == 0:
        raise ValueError(
            "no seeds named; a reliability measurement is a denominator and zero draws is not one"
        )
    geometry = require_cartridge_plan(
        _measurement_hooks.base_lora_sweep_plans(), GEOMETRY_PLAN_NAME
    )
    documents = _test_hooks.read_corpus_documents(corpus)
    digest = corpus_digest(documents)
    tokenizer = hf_hooks.Hooks.load_hf_tokenizer(model_id)
    encoded = [tokenizer.encode(document) for document in documents]
    train, held_out = split_by_stride(
        build_windows(encoded, window=geometry["window"], device=device),
        held_out_stride=geometry["held_out_stride"],
    )
    base = require_cache_capable(hf_hooks.Hooks.load_hf_model(model_id, quantization_for(model_id)))
    # The windows are on ``device``; the loader answers for where the
    # model materialised. Same boundary the headroom CLI shipped without
    # (job 55812484, 19 seconds), pinned by the same recording fake.
    base.to(device)

    short = base_short(model_id)
    observations: list[Observation] = [
        Observation(
            name=f"solo-{short}_characters",
            value=float(sum(len(document) for document in documents)),
        ),
        Observation(name=f"solo-{short}_train_windows", value=float(len(train))),
        Observation(name=f"solo-{short}_held_out_windows", value=float(len(held_out))),
    ]
    gains: list[float] = []
    for seed in seeds:
        slots = train_cartridge(
            base,
            train,
            num_slots=geometry["slots"],
            seed=seed,
            epochs=geometry["epochs"],
            learning_rate=geometry["learning_rate"],
        )
        gain = held_out_gain(CartridgeModel(base=base, slots=slots), held_out)
        gains.append(gain)
        observations.append(Observation(name=f"solo-{short}-seed{seed}_gain", value=gain))
        _log.info("solo %s seed %d: gain %.4f", short, seed, gain)
    observations.append(Observation(name=f"solo-{short}_gain_mean", value=sum(gains) / len(gains)))
    observations.append(
        Observation(name=f"solo-{short}_gain_spread", value=max(gains) - min(gains))
    )
    return tuple(observations), digest


def solo_seeds_run_record(corpus: pathlib.Path, *, model_id: str, device: str) -> RunRecord:
    """Pin determinism, run the measurement, and record it.

    Args:
        corpus: The corpus directory.
        model_id: The base to measure.
        device: Device to measure on.

    Returns:
        The record.

    Raises:
        ValueError: Propagated from :func:`measure_solo_seeds`.
        AppError: Propagated from the corpus, loading and training
            layers.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )
    observations, digest = measure_solo_seeds(
        corpus, model_id=model_id, seeds=SOLO_SEEDS, device=device
    )
    return run_record(
        experiment=SOLO_SEEDS_EXPERIMENT,
        label=solo_seeds_label(model_id=model_id, seeds=SOLO_SEEDS, digest=digest),
        fingerprint=fingerprint,
        observations=observations,
        payload_digest=NO_PAYLOAD,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the measurement and write the record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults
            to the process arguments.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value,
            or a required flag is absent.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    record = solo_seeds_run_record(
        pathlib.Path(cli_args.require_flag(parsed, CORPUS_FLAG)),
        model_id=cli_args.require_flag(parsed, MODEL_FLAG),
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
    )

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "solo seeds %s %s -> %s",
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
        service_name="cartridge-solo-seeds",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "SOLO_SEEDS",
    "SOLO_SEEDS_EXPERIMENT",
    "entrypoint",
    "main",
    "measure_solo_seeds",
    "solo_seeds_label",
    "solo_seeds_run_record",
]


# Without this, `python -m model_trainer.cli.cartridge_solo_seeds` imports
# the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
