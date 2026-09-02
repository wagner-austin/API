"""Score a cloze item set under the FULLY-owned model, attention included.

The closure instrument for the sm_75 residual. Model-Trainer's
``score_baseline`` scores under arms that own the projections and leave
attention to the vendor; that arm's records broke on the GTX 1630 at 15- and
16-token options. This command scores the same items through the same
scorer, the same determinism pin and the same record schema, with ONE
difference: every attention module is swapped onto
:class:`~ordered_kernels.modules.OrderedSdpaAttention` before the
projections are swapped ordered -- so every matmul in the model and every
reduction in attention runs program-ordered.

The records this writes are a NEW corpus, not a match for the fixed-order
one: owned attention computes different bits than the vendor's math kernel
everywhere, on every card. The claim it exists to test is cross-card:
score-tuples bit-identical between sm_75 and everything else, with the
DECISIONS unchanged against the old corpus -- the arms change bits, never
answers, or something is wrong.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from model_trainer.cli import _test_hooks
from model_trainer.cli.score_baseline import encode_outcomes, outcomes_digest
from model_trainer.core.run_fingerprint import (
    capture_run_fingerprint,
    describe_run_fingerprint,
)
from model_trainer.core.services.model.kernel_arm_modules import require_swappable
from model_trainer.worker.cloze_job import parse_items
from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger, setup_logging
from platform_core.run_record import Observation, RunRecord, encode_run_record, run_record

from ordered_kernels.cli.train_step import require_swapped
from ordered_kernels.modules import use_ordered_attention, use_ordered_kernels

_log = get_logger(__name__)

MODEL_FLAG = "--model"
ITEMS_FLAG = "--items"
DEVICE_FLAG = "--device"
MAX_SEQ_LEN_FLAG = "--max-seq-len"
EXPERIMENT_FLAG = "--experiment"
LABEL_FLAG = "--label"
OUT_FLAG = "--out"
OUTCOMES_FLAG = "--outcomes"

_FLAGS = (
    MODEL_FLAG,
    ITEMS_FLAG,
    DEVICE_FLAG,
    MAX_SEQ_LEN_FLAG,
    EXPERIMENT_FLAG,
    LABEL_FLAG,
    OUT_FLAG,
    OUTCOMES_FLAG,
)


def score_fully_owned(
    *,
    hub_model_id: str,
    items_path: pathlib.Path,
    device: str,
    max_seq_len: int,
    experiment: str,
    label: str,
) -> tuple[RunRecord, str]:
    """Pin determinism, swap EVERYTHING ordered, score, and record it.

    The same shape as ``score_baseline.score_with_outcomes``, arm included
    in the function's identity instead of a flag: this command has exactly
    one posture and a flag whose only value is a constant would be a place
    to hold the record's name wrong.

    Args:
        hub_model_id: HuggingFace model id to score untrained.
        items_path: Newline-delimited JSON cloze items, already staged.
        device: Device to score on.
        max_seq_len: Token budget per item.
        experiment: What this measurement belongs to.
        label: Which measurement within it.

    Returns:
        The run record and the encoded per-item outcomes beside it.

    Raises:
        RuntimeError: When either swap matched nothing -- a record claiming
            full ownership over a model that kept vendor arithmetic
            somewhere is the defect every arm here refuses.
        ValueError: Propagated from the loader or the swaps.
    """
    determinism = _test_hooks.apply_determinism_hook(remove_split_k=True, math_attention=True)
    fingerprint: RunFingerprint = capture_run_fingerprint(device, determinism)

    items = parse_items(items_path.read_text(encoding="utf-8"))
    model = _test_hooks.load_hub_model(hub_model_id)
    # Attention first, then the projections: the attention wrapper holds the
    # original Conv1Ds as submodules, so the projections walk still finds
    # and swaps them inside it. Both counts are enforced non-zero.
    target = require_swappable(model.model)
    attn_swapped = require_swapped(use_ordered_attention(target))
    proj_swapped = require_swapped(use_ordered_kernels(target))
    _log.info(
        "fully-owned arm: %d attention module(s), %d projection(s) swapped",
        attn_swapped,
        proj_swapped,
    )
    result = _test_hooks.score_cloze(
        items=items, model=model, device=device, max_seq_len=max_seq_len
    )

    record = run_record(
        experiment=experiment,
        label=label,
        fingerprint=fingerprint,
        observations=(
            Observation(name="cloze_accuracy", value=result["accuracy"]),
            Observation(name="cloze_chance", value=result["chance"]),
            Observation(name="cloze_correct", value=float(result["correct"])),
            Observation(name="cloze_total", value=float(result["total"])),
        ),
        payload_digest=outcomes_digest(result["outcomes"]),
    )
    return record, encode_outcomes(result["outcomes"])


def main(argv: Sequence[str] | None = None) -> int:
    """Score one item set fully owned and write record plus outcomes.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the record and its outcomes are written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, a
            required flag is absent, or ``--max-seq-len`` is not a positive
            integer -- resolved before anything is scored.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    raw_len = cli_args.require_flag(parsed, MAX_SEQ_LEN_FLAG)
    if not raw_len.isdigit() or int(raw_len) == 0:
        raise ValueError(f"{MAX_SEQ_LEN_FLAG} must be a positive integer, got {raw_len!r}")

    record, outcomes = score_fully_owned(
        hub_model_id=cli_args.require_flag(parsed, MODEL_FLAG),
        items_path=pathlib.Path(cli_args.require_flag(parsed, ITEMS_FLAG)),
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
        max_seq_len=int(raw_len),
        experiment=cli_args.require_flag(parsed, EXPERIMENT_FLAG),
        label=cli_args.require_flag(parsed, LABEL_FLAG),
    )

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    outcomes_path = pathlib.Path(cli_args.require_flag(parsed, OUTCOMES_FLAG))
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path.write_text(outcomes, encoding="utf-8")

    _log.info(
        "fully-owned scored experiment=%s label=%s accuracy=%.6f %s -> %s outcomes %s",
        record["experiment"],
        record["label"],
        record["observations"][0]["value"],
        describe_run_fingerprint(record["fingerprint"]),
        out,
        outcomes_path,
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
        service_name="ordered-score",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["entrypoint", "main", "score_fully_owned"]


if __name__ == "__main__":
    entrypoint()
