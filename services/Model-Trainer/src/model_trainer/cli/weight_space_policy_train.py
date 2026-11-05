"""Train the stuck MountainCar policy and write what a weight reader is given.

Board task 7964f9f6, criterion A1, first half. The post's Experiment 4 gives
its reader "the weights, a description of the scene and one recorded
episode"; this produces exactly those, plus the record of how the policy does
before anyone edits it.

WHAT IT LEAVES IN ``--out-dir``:

* ``policy.json`` -- the weights, exactly, for the edit command to load;
* ``weights.txt`` -- the same weights as the reader sees them;
* ``episode.txt`` -- one greedy episode from the first evaluation start;
* ``record.json`` -- a :class:`~platform_core.run_record.RunRecord` whose
  payload digest is the policy's weights digest, the value an edit written
  from ``weights.txt`` must carry as ``written_against``.

Usage:
    python -m model_trainer.cli.weight_space_policy_train --seed 1 --hidden 32
        --iterations 200 --batch 64 --discount 0.9 --learning-rate 0.01
        --episodes 1024 --eval-seed 7 --out-dir DIR
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_core.run_record import Observation, RunRecord, encode_run_record, run_record

from model_trainer.cli._weight_space_flags import require_float_flag, require_int_flag
from model_trainer.cli.known_answer_probe import probe_determinism
from model_trainer.core.run_fingerprint import capture_run_fingerprint
from model_trainer.core.services.model.editing.verify import parameter_digests, weights_digest
from model_trainer.core.services.model.weight_space.control_policy import (
    encode_policy_weights,
    policy_weights,
    render_weights_text,
)
from model_trainer.core.services.model.weight_space.mountain_car import (
    evaluate_policy,
    record_episode,
    render_trace,
    start_positions,
)
from model_trainer.core.services.model.weight_space.policy_training import (
    TrainingConfig,
    train_policy,
)

_log = get_logger(__name__)

#: Name under which every run of the small-network arm is comparable.
CONTROL_EXPERIMENT = "weight-space-control"

#: How often the recorded episode keeps a step for the reader.
TRACE_EVERY = 5

SEED_FLAG = "--seed"
HIDDEN_FLAG = "--hidden"
ITERATIONS_FLAG = "--iterations"
BATCH_FLAG = "--batch"
DISCOUNT_FLAG = "--discount"
LEARNING_RATE_FLAG = "--learning-rate"
EPISODES_FLAG = "--episodes"
EVAL_SEED_FLAG = "--eval-seed"
OUT_DIR_FLAG = "--out-dir"

_FLAGS = (
    SEED_FLAG,
    HIDDEN_FLAG,
    ITERATIONS_FLAG,
    BATCH_FLAG,
    DISCOUNT_FLAG,
    LEARNING_RATE_FLAG,
    EPISODES_FLAG,
    EVAL_SEED_FLAG,
    OUT_DIR_FLAG,
)


def training_label(config: TrainingConfig) -> str:
    """Name a training run by every setting that produced it.

    Args:
        config: The settings.

    Returns:
        The label.
    """
    return (
        f"mountain-car-train-seed{config['seed']}-h{config['hidden']}"
        f"-it{config['iterations']}-b{config['batch']}"
        f"-g{config['discount']}-lr{config['learning_rate']}"
    )


def train_and_write(
    config: TrainingConfig, *, episodes: int, eval_seed: int, out_dir: pathlib.Path
) -> RunRecord:
    """Train, evaluate, and write the reader's materials and the record.

    Args:
        config: The training settings.
        episodes: Evaluation episodes.
        eval_seed: Seed of the evaluation starts, shared with the edit run.
        out_dir: Where the four files go.

    Returns:
        The record written to ``record.json``.
    """
    determinism = probe_determinism("cpu", remove_split_k=False, math_attention=False)
    fingerprint = capture_run_fingerprint("cpu", determinism)
    outcome = train_policy(config)
    policy = outcome["policy"]
    starts = start_positions(episodes, eval_seed)
    report = evaluate_policy(policy.greedy, starts)
    weights = policy_weights(policy)
    trace = record_episode(policy.greedy, float(starts[0].item()))
    record = run_record(
        experiment=CONTROL_EXPERIMENT,
        label=training_label(config),
        fingerprint=fingerprint,
        observations=(
            Observation(name="base_successes", value=float(report["successes"])),
            Observation(name="base_episodes", value=float(report["episodes"])),
            Observation(
                name="base_success_rate",
                value=float(report["successes"]) / float(report["episodes"]),
            ),
            Observation(name="base_mean_steps_to_goal", value=report["mean_steps_to_goal"]),
            Observation(
                name="final_batch_successes", value=float(outcome["final_batch_successes"])
            ),
            Observation(
                name="policy_parameters",
                value=float(sum(len(entry["values"]) for entry in weights["parameters"].values())),
            ),
        ),
        payload_digest=weights_digest(parameter_digests(policy)),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "policy.json").write_text(
        dump_json_str(encode_policy_weights(weights)), encoding="utf-8"
    )
    (out_dir / "weights.txt").write_text(render_weights_text(weights) + "\n", encoding="utf-8")
    (out_dir / "episode.txt").write_text(render_trace(trace, TRACE_EVERY) + "\n", encoding="utf-8")
    (out_dir / "record.json").write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")
    return record


def main(argv: Sequence[str] | None = None) -> int:
    """Train one policy and write its materials.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once every file is written.

    Raises:
        ValueError: When a flag is unknown, repeated, missing, or not a value
            its setting accepts.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    config = TrainingConfig(
        seed=require_int_flag(parsed, SEED_FLAG, 0),
        hidden=require_int_flag(parsed, HIDDEN_FLAG, 1),
        iterations=require_int_flag(parsed, ITERATIONS_FLAG, 1),
        batch=require_int_flag(parsed, BATCH_FLAG, 1),
        discount=require_float_flag(parsed, DISCOUNT_FLAG),
        learning_rate=require_float_flag(parsed, LEARNING_RATE_FLAG),
    )
    record = train_and_write(
        config,
        episodes=require_int_flag(parsed, EPISODES_FLAG, 1),
        eval_seed=require_int_flag(parsed, EVAL_SEED_FLAG, 0),
        out_dir=pathlib.Path(cli_args.require_flag(parsed, OUT_DIR_FLAG)),
    )
    _log.info("trained %s weights %s", record["label"], record["payload_digest"])
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(main())


__all__ = [
    "CONTROL_EXPERIMENT",
    "TRACE_EVERY",
    "entrypoint",
    "main",
    "train_and_write",
    "training_label",
]


# Without this, `python -m model_trainer.cli.weight_space_policy_train` imports
# the module, runs nothing and exits 0.
if __name__ == "__main__":
    entrypoint()
