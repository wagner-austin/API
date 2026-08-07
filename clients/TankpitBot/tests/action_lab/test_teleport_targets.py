"""Tests for teleport target construction and argument parsing."""

from __future__ import annotations

import pytest

from tankpit_bot.action_lab.teleport_helpers import (
    _limit_targets,
    build_box_targets,
    parse_targets_arg,
)
from tankpit_bot.action_lab.types import (
    TeleportTargetDict,
)


def test_build_box_targets_creates_ten_targets() -> None:
    targets = build_box_targets(100, 100, 8, 6)
    assert len(targets) == 10
    assert targets[0]["label"] == "box_r0_c0"
    assert targets[-1]["label"] == "box_r1_c4"
    assert targets[0]["x"] == 84
    assert targets[0]["y"] == 94
    assert targets[-1]["x"] == 116
    assert targets[-1]["y"] == 106


def test_build_box_targets_clamps_edges() -> None:
    targets = build_box_targets(2, 2, 8, 8)
    assert targets[0]["x"] == 0
    assert targets[0]["y"] == 0


def test_build_box_targets_clamps_upper_edges() -> None:
    targets = build_box_targets(254, 254, 8, 8)
    assert targets[-1]["x"] == 255
    assert targets[-1]["y"] == 255


def test_build_box_targets_rejects_non_positive_steps() -> None:
    with pytest.raises(ValueError, match="step_x"):
        build_box_targets(100, 100, 0, 8)
    with pytest.raises(ValueError, match="step_y"):
        build_box_targets(100, 100, 8, 0)


def test_limit_targets_rejects_non_positive_max_targets() -> None:
    with pytest.raises(ValueError, match="max_targets must be positive"):
        _limit_targets([TeleportTargetDict(label="target_0", x=1, y=2)], 0)


def test_parse_targets_arg_parses_targets() -> None:
    targets = parse_targets_arg("156:170,147:166")
    assert targets == [
        TeleportTargetDict(label="target_0", x=156, y=170),
        TeleportTargetDict(label="target_1", x=147, y=166),
    ]


def test_parse_targets_arg_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        parse_targets_arg("   ")
    with pytest.raises(ValueError, match="expected x:y"):
        parse_targets_arg("156-170")
    with pytest.raises(ValueError, match=r"outside 0\.\.255"):
        parse_targets_arg("999:10")
