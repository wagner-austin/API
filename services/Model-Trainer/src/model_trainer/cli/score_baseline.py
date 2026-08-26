"""Score an untrained model on a cloze item set and emit a run record.

The floor every arm accuracy is read as lift over, produced somewhere a
cluster can reach. ``POST /runs/baselines/cloze`` already does this, and
needs an API, redis and an RQ worker to do it; a Slurm compute node has
none of those. This runs the same scorer in-process.

WHY THE ITEMS ARE A PATH AND NOT A FILE ID. The worker fetches its item set
from the data-bank service by id. On a compute node there is no data-bank, so
the item set is a file that the job staged. That is the only difference in
what is scored -- the parser, the scorer and the determinism pin are the
same code the worker calls.

WHAT IT LEAVES BEHIND is a :class:`~platform_core.run_record.RunRecord`:
the accuracy and its companions as named observations, the configuration that
produced them, and a digest of the per-item outcomes. A later run can then be
compared against this one, or refused, on evidence rather than assumption --
which is the whole point, since the open question is whether a floor measured
on one card holds on another.
"""

from __future__ import annotations

import hashlib
import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.run_record import Observation, RunRecord, encode_run_record, run_record

from model_trainer.cli import _test_hooks
from model_trainer.core.contracts.cloze import ClozeItemOutcome, encode_cloze_item_outcome
from model_trainer.core.run_fingerprint import capture_run_fingerprint, describe_run_fingerprint
from model_trainer.worker.cloze_job import parse_items

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


def outcomes_digest(outcomes: Sequence[ClozeItemOutcome]) -> str:
    """Digest WHICH ITEMS were answered correctly, and nothing else.

    Two runs agreeing on an accuracy can still disagree on which items they
    got right, and the aggregate cannot tell you. This digest is what
    distinguishes them.

    IT DELIBERATELY EXCLUDES ``scores``. They were included until 2026-08-25,
    and the first cross-card comparison showed why that was wrong: gpt2 on
    the same 2,627 items scored 1374 correct on both a 3090 Ti and an A100
    80GB -- accuracy identical to all fifteen digits -- and the digests
    differed. ``scores`` are raw negative log-likelihoods, so they differ in
    their low bits between two cards whatever the answers were. A digest over
    them therefore ALWAYS differs across hardware, which means it cannot
    distinguish "these runs disagreed about an item" from "these runs agreed
    completely and one of them ran on a different card". A check that fires
    on every comparison carries no information.

    The decisions are the measurement; the scores are how it got there. The
    scores are not discarded -- ``main`` writes every outcome, scores
    included, beside the record, which is where an item-by-item diagnosis
    belongs.

    Args:
        outcomes: The per-item outcomes, in scoring order.

    Returns:
        ``sha256:`` followed by the hex digest of the canonical JSON of
        ``[item_id, correct]`` pairs, in scoring order. Order is preserved
        rather than sorted: two runs over one item set score it in the same
        order, and a reordering is itself a difference worth catching.
    """
    canonical = dump_json_str([[o["item_id"], o["correct"]] for o in outcomes])
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def encode_outcomes(outcomes: Sequence[ClozeItemOutcome]) -> str:
    """Render every per-item outcome, scores included, for a later diagnosis.

    ``ClozeItemOutcome`` documents itself as carried "so that two arms scored
    on the same item set can be compared item by item" -- and the scorer
    reduced them to one digest and dropped them, so that comparison was not
    possible for anything it produced. When the A100 floor's digest differed
    from the 3090 Ti's, there was no way to ask which items moved.

    Args:
        outcomes: The per-item outcomes, in scoring order.

    Returns:
        Canonical JSON of the encoded outcomes.
    """
    return dump_json_str([encode_cloze_item_outcome(o) for o in outcomes])


def score_with_outcomes(
    *,
    hub_model_id: str,
    items_path: pathlib.Path,
    device: str,
    max_seq_len: int,
    experiment: str,
    label: str,
) -> tuple[RunRecord, str]:
    """Pin determinism, score the model, and record what it ran on.

    Determinism is pinned FIRST, before the model is loaded, because
    ``CUBLAS_WORKSPACE_CONFIG`` is read when the cuBLAS handle is created and
    loading weights is enough to create it. A pin after that is accepted in
    silence and does nothing.

    Args:
        hub_model_id: HuggingFace model id to score untrained.
        items_path: Newline-delimited JSON cloze items, already staged.
        device: Device to score on.
        max_seq_len: Token budget per item.
        experiment: What this measurement belongs to.
        label: Which measurement within it.

    Returns:
        The run record -- accuracy, correct, total and chance as
        observations, the fingerprint, and a digest of which items were
        answered correctly -- and the encoded outcomes beside it.

        The outcomes are returned rather than reduced away because the digest
        answers "did these two runs decide the same things" and nothing else.
        When the answer is no, the next question is which items moved, and
        only the outcomes can answer that.
    """
    determinism = _test_hooks.apply_determinism_hook()
    fingerprint: RunFingerprint = capture_run_fingerprint(device, determinism)

    items = parse_items(items_path.read_text(encoding="utf-8"))
    model = _test_hooks.load_hub_model(hub_model_id)
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
    """Score one baseline and write its record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the record and its outcomes are written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, a
            required flag is absent, or ``--max-seq-len`` is not a positive
            integer. Nothing is scored on a command line that was not
            understood: a run under a mistyped flag is a different run, and
            it would otherwise write a record claiming to be this one.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    raw_len = cli_args.require_flag(parsed, MAX_SEQ_LEN_FLAG)
    if not raw_len.isdigit() or int(raw_len) == 0:
        raise ValueError(f"{MAX_SEQ_LEN_FLAG} must be a positive integer, got {raw_len!r}")

    record, outcomes = score_with_outcomes(
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

    # Required rather than optional. The per-item outcomes are the only thing
    # that can say WHICH items two runs disagreed about, and a flag nobody
    # remembers to pass is not there on the run that turns out to need it --
    # which is exactly what happened to the first A100 floor.
    outcomes_path = pathlib.Path(cli_args.require_flag(parsed, OUTCOMES_FLAG))
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path.write_text(outcomes, encoding="utf-8")

    _log.info(
        "baseline scored experiment=%s label=%s accuracy=%.6f %s -> %s outcomes %s",
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

    Matches ``modeltrainer-cluster-train``: an hpc3 run document names a
    command, and a console script is what that command can be.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main())


__all__ = [
    "encode_outcomes",
    "entrypoint",
    "main",
    "outcomes_digest",
    "score_with_outcomes",
]
