"""Apply a written edit to the stuck policy and count successes before and after.

Board task 7964f9f6, criterion A1, second half. The policy is loaded from the
file the training run wrote, the edit is applied through the shared weight
program applier, and the same seeded episodes are run three times: the
untouched policy, the edited one, and the policy again after the edit is
taken back out. The first is the untouched control; the third is the control
on the harness itself, recorded so a reader can see it reproduce the first.

Usage:
    python -m model_trainer.cli.weight_space_policy_edit --policy DIR/policy.json
        --program EDIT.json --episodes 1024 --eval-seed 7 --out REC.json
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger
from platform_core.run_record import Observation, RunRecord, encode_run_record, run_record

from model_trainer.cli._weight_space_flags import require_int_flag
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.cli.weight_space_policy_train import CONTROL_EXPERIMENT
from model_trainer.core.contracts.weight_program import WeightProgram, decode_weight_program
from model_trainer.core.run_fingerprint import capture_run_fingerprint
from model_trainer.core.services.model.weight_space.control_policy import (
    decode_policy_weights,
    load_policy,
)
from model_trainer.core.services.model.weight_space.mountain_car import (
    SuccessReport,
    evaluate_policy,
    start_positions,
)
from model_trainer.core.services.model.weight_space.program import (
    apply_weight_program,
    remove_weight_program,
)

_log = get_logger(__name__)

POLICY_FLAG = "--policy"
PROGRAM_FLAG = "--program"
EPISODES_FLAG = "--episodes"
EVAL_SEED_FLAG = "--eval-seed"
OUT_FLAG = "--out"

_FLAGS = (POLICY_FLAG, PROGRAM_FLAG, EPISODES_FLAG, EVAL_SEED_FLAG, OUT_FLAG)


def read_program(path: pathlib.Path) -> WeightProgram:
    """Read a weight program off disk.

    Args:
        path: The program's JSON file.

    Returns:
        The validated program.

    Raises:
        JSONTypeError: If the file does not hold a valid program.
    """
    return decode_weight_program(narrow_json_to_dict(load_json_str(path.read_text("utf-8"))))


def _report_observations(prefix: str, report: SuccessReport) -> tuple[Observation, ...]:
    """Name one evaluation's numbers.

    Args:
        prefix: ``base``, ``edited`` or ``restored``.
        report: The evaluation.

    Returns:
        Its successes, success rate and mean steps to the flag.
    """
    return (
        Observation(name=f"{prefix}_successes", value=float(report["successes"])),
        Observation(
            name=f"{prefix}_success_rate",
            value=float(report["successes"]) / float(report["episodes"]),
        ),
        Observation(name=f"{prefix}_mean_steps_to_goal", value=report["mean_steps_to_goal"]),
    )


def edit_and_measure(
    *, policy_path: pathlib.Path, program: WeightProgram, episodes: int, eval_seed: int
) -> RunRecord:
    """Measure the policy, edit it, measure it, restore it, measure it again.

    Args:
        policy_path: The training run's ``policy.json``.
        program: The edit.
        episodes: Evaluation episodes.
        eval_seed: Seed of the evaluation starts.

    Returns:
        The record, its payload digest the edited weights' digest.

    Raises:
        AppError: Propagated from the applier when the program was written for
            other weights or wrote outside what it names, or from the restore
            when the restored weights differ from the untouched ones. The
            restored evaluation is RECORDED rather than compared: once the
            restore has proved every parameter's bytes are back, the same
            starts on the same weights are the same episodes, and the record
            lets a reader see that rather than take it on trust.
    """
    determinism = probe_determinism("cpu", remove_split_k=False, math_attention=False)
    fingerprint = capture_run_fingerprint("cpu", determinism)
    policy = load_policy(
        decode_policy_weights(narrow_json_to_dict(load_json_str(policy_path.read_text("utf-8"))))
    )
    starts = start_positions(episodes, eval_seed)
    base = evaluate_policy(policy.greedy, starts)
    applied = apply_weight_program(policy, program)
    edited = evaluate_policy(policy.greedy, starts)
    remove_weight_program(policy, applied)
    restored = evaluate_policy(policy.greedy, starts)
    effect = applied["effect"]
    return run_record(
        experiment=CONTROL_EXPERIMENT,
        label=f"mountain-car-edit-{program['program_id']}",
        fingerprint=fingerprint,
        observations=(
            Observation(name="episodes", value=float(episodes)),
            *_report_observations("base", base),
            *_report_observations("edited", edited),
            *_report_observations("restored", restored),
            Observation(name="elements_changed", value=float(effect["elements_changed"])),
            Observation(name="elements_total", value=float(effect["elements_total"])),
            Observation(
                name="fraction_changed",
                value=float(effect["elements_changed"]) / float(effect["elements_total"]),
            ),
        ),
        payload_digest=effect["weights_after"],
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Apply one edit and write the before-and-after record.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the record is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing or malformed.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    record = edit_and_measure(
        policy_path=pathlib.Path(cli_args.require_flag(parsed, POLICY_FLAG)),
        program=read_program(pathlib.Path(cli_args.require_flag(parsed, PROGRAM_FLAG))),
        episodes=require_int_flag(parsed, EPISODES_FLAG, 1),
        eval_seed=require_int_flag(parsed, EVAL_SEED_FLAG, 0),
    )
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")
    _log.info("edited %s -> %s", record["label"], out)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main())


__all__ = ["edit_and_measure", "entrypoint", "main", "read_program"]


# Without this, `python -m model_trainer.cli.weight_space_policy_edit` imports
# the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
