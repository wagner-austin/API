"""The small-network commands end to end: train, write the reader's files, edit.

Nothing is stood in for. A real policy is trained (two iterations, so the
suite stays fast), its files are written and read back, and a real edit made
of operations a reader could write is applied to it. The edit is the one the
arm's whole premise names: rewire the output layer so the policy pushes the
way the car is moving, which the mountain-car suite shows always reaches the
flag. The policy's trained weights are irrelevant to that rewrite, so the
edited success count is known before the run: every episode.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.run_record import decode_run_record

from model_trainer.cli import weight_space_policy_edit as edit_cli
from model_trainer.cli import weight_space_policy_train as train_cli
from model_trainer.cli._weight_space_flags import require_float_flag, require_int_flag
from tests._qa_benchmark_support import values

_HIDDEN = 4


def _train(out_dir: pathlib.Path) -> None:
    """Run the training command on a tiny configuration.

    Args:
        out_dir: Where its files go.
    """
    assert (
        train_cli.main(
            [
                "--seed",
                "3",
                "--hidden",
                str(_HIDDEN),
                "--iterations",
                "2",
                "--batch",
                "4",
                "--discount",
                "0.9",
                "--learning-rate",
                "0.01",
                "--episodes",
                "16",
                "--eval-seed",
                "7",
                "--out-dir",
                str(out_dir),
            ]
        )
        == 0
    )


def _push_with_velocity(written_against: str) -> JSONObject:
    """An edit that makes any policy of this width push the way the car moves.

    Hidden unit 0 of each layer becomes a copy of the scaled velocity (input
    weights 0 and 1, the next layer reading unit 0 alone, biases 0), and the
    output layer scores push-right by +10 times that unit plus 0.001,
    push-left by -10 times it, and no-push 0. So the greedy action is right
    when the velocity is positive or zero and left when it is negative. The
    0.001 breaks the tie at rest toward right: without it argmax takes
    push-left at zero velocity, and an independent numpy run of that variant
    (2026-11-05) missed the flag on 1 to 3 of 4,096 starts, where this one and
    the plain ``v >= 0`` rule reached it on 20,000 of 20,000.

    Args:
        written_against: The trained policy's weights digest.

    Returns:
        The program, as JSON.
    """
    zeros = [0.0] * _HIDDEN
    return {
        "program_id": "push-with-velocity",
        "written_against": written_against,
        "ops": [
            {
                "op": "set",
                "parameter": "hidden_in.weight",
                "axis": 0,
                "indices": [0],
                "values": [0.0, 1.0],
            },
            {
                "op": "set",
                "parameter": "hidden_in.bias",
                "axis": 0,
                "indices": [0],
                "values": [0.0],
            },
            {
                "op": "set",
                "parameter": "hidden_mid.weight",
                "axis": 0,
                "indices": [0],
                "values": [1.0, *zeros[1:]],
            },
            {
                "op": "set",
                "parameter": "hidden_mid.bias",
                "axis": 0,
                "indices": [0],
                "values": [0.0],
            },
            {
                "op": "set",
                "parameter": "action_out.weight",
                "axis": 0,
                "indices": [0, 1, 2],
                "values": [-10.0, *zeros[1:], *zeros, 10.0, *zeros[1:]],
            },
            {
                "op": "set",
                "parameter": "action_out.bias",
                "axis": 0,
                "indices": [0, 1, 2],
                "values": [0.0, 0.0, 0.001],
            },
        ],
    }


def test_training_writes_the_readers_files_and_a_record(tmp_path: pathlib.Path) -> None:
    _train(tmp_path)
    record = decode_run_record(
        narrow_json_to_dict(load_json_str((tmp_path / "record.json").read_text("utf-8")))
    )
    named = values(record["observations"])
    assert record["experiment"] == "weight-space-control"
    assert record["label"] == "mountain-car-train-seed3-h4-it2-b4-g0.9-lr0.01"
    assert named["base_episodes"] == 16.0
    assert named["policy_parameters"] == float(2 * 4 + 4 + 4 * 4 + 4 + 4 * 3 + 3)
    assert record["payload_digest"].startswith("sha256:")
    weights_text = (tmp_path / "weights.txt").read_text("utf-8").splitlines()
    assert weights_text[0].startswith("hidden_in.weight row 0 ")
    assert len(weights_text) == 4 + 1 + 4 + 1 + 3 + 1
    assert (tmp_path / "episode.txt").read_text("utf-8").startswith("episode from position ")


def test_an_edit_is_measured_against_the_untouched_policy(tmp_path: pathlib.Path) -> None:
    _train(tmp_path)
    trained = decode_run_record(
        narrow_json_to_dict(load_json_str((tmp_path / "record.json").read_text("utf-8")))
    )
    program = tmp_path / "edit.json"
    program.write_text(dump_json_str(_push_with_velocity(trained["payload_digest"])), "utf-8")
    out = tmp_path / "edited" / "record.json"
    assert (
        edit_cli.main(
            [
                "--policy",
                str(tmp_path / "policy.json"),
                "--program",
                str(program),
                "--episodes",
                "16",
                "--eval-seed",
                "7",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    record = decode_run_record(narrow_json_to_dict(load_json_str(out.read_text("utf-8"))))
    named = values(record["observations"])
    base = values(trained["observations"])
    assert record["label"] == "mountain-car-edit-push-with-velocity"
    assert named["base_successes"] == base["base_successes"]
    assert named["restored_successes"] == named["base_successes"]
    assert named["edited_successes"] == 16.0
    assert named["elements_total"] == base["policy_parameters"]
    assert 0.0 < named["fraction_changed"] <= 1.0


@pytest.mark.parametrize(
    ("raw", "message"),
    [("x1", "must be an integer, got 'x1'"), ("-2", "must be at least 0, got -2")],
)
def test_an_integer_flag_that_is_not_a_permitted_integer_is_refused(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        require_int_flag({"--seed": raw}, "--seed", 0)


def test_a_float_flag_must_be_finite() -> None:
    assert require_float_flag({"--discount": "0.25"}, "--discount") == 0.25
    with pytest.raises(ValueError, match="must be a finite number, got 'nan'"):
        require_float_flag({"--discount": "nan"}, "--discount")
