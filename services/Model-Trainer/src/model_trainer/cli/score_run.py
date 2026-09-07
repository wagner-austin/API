"""Score a TRAINED run's artifact on a cloze item set, on a compute node.

THE GAP THIS CLOSES. A compute node could train an arm and could score an
UNTRAINED baseline (:mod:`~model_trainer.cli.score_baseline`), and could do
nothing with the model it had just produced. Every published arm accuracy
therefore went through the deployed service's ``POST /runs/{id}/cloze`` --
which needs an API, redis and an RQ worker, none of which exist on HPC3. So a
cluster campaign trained on the cluster and then had to come home to be read,
and the twelve arms of the 2026-09-04 ablation redo are the first that could
not: they live in a `/pub` artifact directory and there was no command that
would look at them.

WHY NOT JUST POINT ``score_baseline`` AT THE DIRECTORY, which would run. It
loads through :func:`~...hf_lm.io.load_prepared_hf_lm_from_hub`, whose whole
contract is "nothing applied" -- *a baseline is defined by having nothing
applied to it*, so it reports no strategy and reapplies none.

For a ``full`` finetune that happens to give the right weights, because the
whole model IS the artifact. **For an adapter arm it would silently score the
BASE MODEL and report the number as the arm's.** A LoRA or QLoRA run saves an
adapter beside a base it does not own; loading the directory as a hub model
picks up whatever base is there and never attaches the adapter. The run
completes, the accuracy is plausible, and it measures a model that was never
trained. There would be nothing in the record to notice it by.

So this loads through :func:`~...hf_lm.io.load_prepared_hf_lm_from_handle`,
which reads the artifact's own metadata, rebuilds the base under that run's
own quantization, and asks the finetuning strategy that produced it to
reattach what it trained. Right for ``full`` for the same reason it is right
for an adapter, rather than by coincidence.

WHAT IS DELIBERATELY IDENTICAL to ``score_baseline``: the determinism pin and
its ordering, the kernel arm, the item parser, the scorer, the observation
names, and the outcomes digest. An arm and a floor are subtracted from each
other, so anything that differed between the two paths would land in that
difference.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.comparability import RunFingerprint
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.run_record import Observation, RunRecord, encode_run_record, run_record

from model_trainer.cli import _test_hooks
from model_trainer.cli.score_baseline import encode_outcomes, outcomes_digest
from model_trainer.core.run_fingerprint import capture_run_fingerprint, describe_run_fingerprint
from model_trainer.core.services.model.kernel_arm_modules import apply_kernel_arm_to_model
from model_trainer.worker.cloze_job import parse_items

_log = get_logger(__name__)

ARTIFACT_FLAG = "--artifact-dir"
KERNEL_FLAG = "--kernel"
ITEMS_FLAG = "--items"
DEVICE_FLAG = "--device"
MAX_SEQ_LEN_FLAG = "--max-seq-len"
EXPERIMENT_FLAG = "--experiment"
LABEL_FLAG = "--label"
OUT_FLAG = "--out"
OUTCOMES_FLAG = "--outcomes"

_FLAGS = (
    ARTIFACT_FLAG,
    KERNEL_FLAG,
    ITEMS_FLAG,
    DEVICE_FLAG,
    MAX_SEQ_LEN_FLAG,
    EXPERIMENT_FLAG,
    LABEL_FLAG,
    OUT_FLAG,
    OUTCOMES_FLAG,
)


def score_run_with_outcomes(
    *,
    artifact_dir: pathlib.Path,
    items_path: pathlib.Path,
    device: str,
    max_seq_len: int,
    experiment: str,
    label: str,
    kernel: str,
) -> tuple[RunRecord, str]:
    """Pin determinism, score a trained artifact, and record what ran.

    Determinism is pinned FIRST, before the artifact is read, for the reason
    ``score_baseline`` gives: ``CUBLAS_WORKSPACE_CONFIG`` is read when the
    cuBLAS handle is created and loading weights is enough to create it, so a
    pin after that is accepted in silence and does nothing.

    Args:
        artifact_dir: Directory holding the trained run's saved model and its
            metadata, as ``modeltrainer-cluster-train`` wrote them.
        items_path: Newline-delimited JSON cloze items, already staged.
        device: Device to score on.
        max_seq_len: Token budget per item.
        experiment: What this measurement belongs to.
        label: Which measurement within it.
        kernel: Which arithmetic the model's matmuls use, by
            :data:`~deterministic_gemm.KERNEL_ARMS` name. Applied AFTER the
            weights are loaded and before anything is scored, and required
            with no default, exactly as for a baseline: a record that cannot
            say which arithmetic produced it is a record nobody can read
            against another.

    Returns:
        The run record -- accuracy, correct, total and chance as observations,
        the fingerprint, and a digest of which items were answered correctly
        -- and the encoded outcomes beside it.

    Raises:
        FileNotFoundError: If the artifact directory holds no metadata, which
            means it is not a finished run rather than a run that scored
            badly.
    """
    # Both controls on, matching `score_baseline` exactly. An arm scored under
    # one posture and a floor under another would put the difference between
    # the postures into every lift computed from the pair.
    determinism = _test_hooks.apply_determinism_hook(remove_split_k=True, math_attention=True)
    fingerprint: RunFingerprint = capture_run_fingerprint(device, determinism)

    items = parse_items(items_path.read_text(encoding="utf-8"))
    model = _test_hooks.load_run_model(str(artifact_dir))
    swapped = apply_kernel_arm_to_model(model.model, kernel)
    _log.info("kernel arm %s replaced %d module(s)", kernel, swapped)
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
    """Score one trained run and write its record.

    Args:
        argv: Command-line arguments excluding the program name. Defaults to
            the process arguments.

    Returns:
        0 once the record and its outcomes are written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing its value, a
            required flag is absent, or ``--max-seq-len`` is not a positive
            integer. Nothing is scored on a command line that was not
            understood.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)

    raw_len = cli_args.require_flag(parsed, MAX_SEQ_LEN_FLAG)
    if not raw_len.isdigit() or int(raw_len) == 0:
        raise ValueError(f"{MAX_SEQ_LEN_FLAG} must be a positive integer, got {raw_len!r}")

    record, outcomes = score_run_with_outcomes(
        artifact_dir=pathlib.Path(cli_args.require_flag(parsed, ARTIFACT_FLAG)),
        items_path=pathlib.Path(cli_args.require_flag(parsed, ITEMS_FLAG)),
        device=cli_args.require_flag(parsed, DEVICE_FLAG),
        max_seq_len=int(raw_len),
        experiment=cli_args.require_flag(parsed, EXPERIMENT_FLAG),
        label=cli_args.require_flag(parsed, LABEL_FLAG),
        kernel=cli_args.require_flag(parsed, KERNEL_FLAG),
    )

    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")

    # Required rather than optional, for the reason `score_baseline` states:
    # the per-item outcomes are the only thing that can say WHICH items two
    # runs disagreed about, and a flag nobody remembers to pass is absent on
    # the run that turns out to need it.
    outcomes_path = pathlib.Path(cli_args.require_flag(parsed, OUTCOMES_FLAG))
    outcomes_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes_path.write_text(outcomes, encoding="utf-8")

    _log.info(
        "run scored experiment=%s label=%s accuracy=%.6f %s -> %s outcomes %s",
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
    "entrypoint",
    "main",
    "score_run_with_outcomes",
]


if __name__ == "__main__":
    entrypoint()
