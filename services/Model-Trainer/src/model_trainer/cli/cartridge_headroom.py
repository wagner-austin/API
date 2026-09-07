"""Absolute plain-base held-out loss per (base, corpus): the headroom term.

WHY THIS EXISTS (board task ``afee6162``, successor to the 7B rung
``af35fc20``). Every recorded cartridge gain is ``base_loss -
adapted_loss`` on held-out windows, in the base's own tokenization. The
7B rung found the solo gain nearly vanishes on Pythia-6.9B (~0.07-0.23
nats against ~0.81 on every GPT-2 rung) and filed four candidate
mechanisms unattributed. The cheapest decisive test is this measurement:
the base-loss TERM alone, for every measured base, on the same held-out
split every record scored against. A base already sitting near the
smaller bases' adapted losses never had the recorded ~0.8 nats to give;
a base with nats to spare pushes the attribution to training instead.

UNITS, stated because the comparison crosses tokenizers: each base is
scored in its own tokenization, exactly as every recorded gain was, so
these rows subtract against the records directly. Cross-family readings
(gpt2's BPE against GPT-NeoX's) additionally need the per-corpus
character and token rows this record carries: the same text tokenized
shorter shows a higher per-token loss at equal compression, and a reader
without those rows would attribute the tokenizer to the model.

The window geometry is read off the recorded plan row rather than
declared again, so this measurement cannot drift from what the records
scored: same window, same stride, same split arithmetic.
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
from model_trainer.cli.cartridge_lora_policy import quantization_for
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.finetuning.strategies.cartridge import (
    require_cache_capable,
)
from model_trainer.core.services.model.backends.hf_lm import _test_hooks as hf_hooks
from model_trainer.core.services.model.cartridge_corpus import build_windows, split_by_stride
from model_trainer.core.services.model.cartridge_plans import (
    corpus_digest,
    require_cartridge_plan,
)
from model_trainer.core.services.model.cartridge_scoring import base_loss

_log = get_logger(__name__)

CORPORA_FLAG = "--corpora"
DEVICE_FLAG = "--device"
OUT_FLAG = "--out"

_FLAGS = (CORPORA_FLAG, DEVICE_FLAG, OUT_FLAG)

#: Fixed for the reason every experiment name here is: the string is what
#: two records are grouped by, and a drifting one silently unpairs them.
HEADROOM_EXPERIMENT = "cartridge-headroom"

#: Every base a cartridge record exists for, in ladder order. Explicit
#: like the policy maps this feeds: a base outside the policy is a loud
#: refusal in :func:`quantization_for`, never a silent fp32 load.
MEASURED_BASES = ("gpt2", "gpt2-medium", "gpt2-xl", "EleutherAI/pythia-6.9b")

#: The plan row whose geometry this measurement must match, resolved
#: through the same plan hook every sweep reads, so the two cannot
#: drift: every cartridge record's held-out split used exactly this
#: row's window and stride.
GEOMETRY_PLAN_NAME = "gpt2-base-lora"


def base_short(model_id: str) -> str:
    """Collapse a hub id to its observation-name segment.

    Args:
        model_id: The hub id.

    Returns:
        The id's last path segment: ``EleutherAI/pythia-6.9b`` names its
        rows ``pythia-6.9b`` and the gpt2 family is unchanged.
    """
    return model_id.rsplit("/", 1)[-1]


def headroom_label(
    *,
    bases: Sequence[str],
    corpora: Sequence[pathlib.Path],
    window: int,
    held_out_stride: int,
    digest: str,
) -> str:
    """Build the label identifying one headroom measurement.

    Args:
        bases: The bases measured.
        corpora: The corpora measured.
        window: Tokens per window, as measured.
        held_out_stride: The held-out stride, as measured.
        digest: The primary (first) corpus's digest, per the sweeps'
            convention.

    Returns:
        The label.
    """
    return f"cartridge-headroom-w{window}-s{held_out_stride}-b{len(bases)}-c{len(corpora)}-{digest}"


def measure_headroom(
    corpora: Sequence[pathlib.Path],
    *,
    bases: Sequence[str],
    window: int,
    held_out_stride: int,
    device: str,
) -> tuple[tuple[Observation, ...], str]:
    """Score every corpus's held-out windows on every base, plain.

    Args:
        corpora: Corpus directories, the first being the primary whose
            digest names the record.
        bases: Hub ids to measure, each resolvable by the loading policy.
        window: Tokens per window. Production passes the recorded plan
            row's value; anything else measures a different split than
            the records scored.
        held_out_stride: The held-out stride, same provenance.
        device: Device to measure on.

    Returns:
        ``(observations, digest)``: per corpus, its character and
        document counts (base-independent, for the cross-tokenizer
        reading); per (base, corpus), the mean held-out loss and the
        window and token counts behind it. The digest is the primary
        corpus's.

    Raises:
        ValueError: When no corpus or no base is named; a headroom record
            over nothing would read as a measurement.
        AppError: Propagated from the corpus layer when a corpus cannot
            fill the recorded geometry.
    """
    if len(corpora) == 0:
        raise ValueError(
            "no corpora named; a headroom record scoring nothing would still "
            "carry a label and read as a measurement"
        )
    if len(bases) == 0:
        raise ValueError(
            "no bases named; the measurement exists to compare bases and one "
            "list of corpus rows compares nothing"
        )
    staged = [(path, _test_hooks.read_corpus_documents(path)) for path in corpora]
    digest = corpus_digest(staged[0][1])

    observations: list[Observation] = []
    for path, documents in staged:
        observations.append(
            Observation(
                name=f"headroom-{path.name}_characters",
                value=float(sum(len(document) for document in documents)),
            )
        )
        observations.append(
            Observation(name=f"headroom-{path.name}_documents", value=float(len(documents)))
        )

    for model_id in bases:
        short = base_short(model_id)
        tokenizer = hf_hooks.Hooks.load_hf_tokenizer(model_id)
        base = require_cache_capable(
            hf_hooks.Hooks.load_hf_model(model_id, quantization_for(model_id))
        )
        # The windows are built on ``device`` and the loader hands the
        # model back wherever it materialised it -- measured 2026-09-07,
        # job 55812484: cuda windows against cpu weights fail the first
        # embedding lookup, nineteen seconds into a five-hour allocation.
        base.to(device)
        base.eval()
        for path, documents in staged:
            encoded = [tokenizer.encode(document) for document in documents]
            _train, held_out = split_by_stride(
                build_windows(encoded, window=window, device=device),
                held_out_stride=held_out_stride,
            )
            losses = [base_loss(base, item) for item in held_out]
            mean = sum(losses) / len(losses)
            observations.append(
                Observation(name=f"headroom-{short}-{path.name}_base_loss_mean", value=mean)
            )
            observations.append(
                Observation(
                    name=f"headroom-{short}-{path.name}_held_out_windows",
                    value=float(len(held_out)),
                )
            )
            observations.append(
                Observation(
                    name=f"headroom-{short}-{path.name}_held_out_tokens",
                    value=float(len(held_out) * window),
                )
            )
            _log.info(
                "headroom %s on %s: %.4f over %d held-out windows",
                short,
                path.name,
                mean,
                len(held_out),
            )
    return tuple(observations), digest


def headroom_run_record(corpora: Sequence[pathlib.Path], *, device: str) -> RunRecord:
    """Pin determinism, run the measurement, and record it.

    Args:
        corpora: Corpus directories, primary first.
        device: Device to measure on.

    Returns:
        The record.

    Raises:
        ValueError: Propagated from :func:`measure_headroom`.
        AppError: Propagated from the corpus and loading layers.
    """
    fingerprint: RunFingerprint = capture_run_fingerprint(
        device, probe_determinism(device, remove_split_k=False, math_attention=False)
    )
    geometry = require_cartridge_plan(
        _measurement_hooks.base_lora_sweep_plans(), GEOMETRY_PLAN_NAME
    )
    window = geometry["window"]
    held_out_stride = geometry["held_out_stride"]
    observations, digest = measure_headroom(
        corpora,
        bases=MEASURED_BASES,
        window=window,
        held_out_stride=held_out_stride,
        device=device,
    )
    return run_record(
        experiment=HEADROOM_EXPERIMENT,
        label=headroom_label(
            bases=MEASURED_BASES,
            corpora=corpora,
            window=window,
            held_out_stride=held_out_stride,
            digest=digest,
        ),
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
            a required flag is absent, or no corpus is named.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    corpora = [
        pathlib.Path(entry)
        for entry in cli_args.require_flag(parsed, CORPORA_FLAG).split(",")
        if entry
    ]
    record = headroom_run_record(corpora, device=cli_args.require_flag(parsed, DEVICE_FLAG))

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    _log.info(
        "headroom %s %s -> %s",
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
        service_name="cartridge-headroom",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = [
    "GEOMETRY_PLAN_NAME",
    "HEADROOM_EXPERIMENT",
    "MEASURED_BASES",
    "base_short",
    "entrypoint",
    "headroom_label",
    "headroom_run_record",
    "main",
    "measure_headroom",
]


# Without this, `python -m model_trainer.cli.cartridge_headroom` imports
# the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
