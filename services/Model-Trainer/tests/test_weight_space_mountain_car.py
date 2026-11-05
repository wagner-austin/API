"""MountainCar's dynamics against hand-worked steps and two reference policies.

The step values are worked by hand from Gym's MountainCar-v0 update. The two
reference policies are the ones the whole arm rests on: driving right
regardless never reaches the flag, and pushing the way the car is already
moving always does. Both were first measured with an independent numpy
implementation of the same update (2026-11-05, 1,024 starts: 0 of 1,024 and
1,024 of 1,024, mean 119.6 steps to the flag), so the counts asserted here
come from outside this module.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest
import torch

from model_trainer.core.services.model.weight_space.mountain_car import (
    MAX_STEPS,
    EpisodeTrace,
    evaluate_policy,
    observe,
    record_episode,
    render_trace,
    start_positions,
    step,
)


def _always(action: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """A policy that takes one action whatever it sees.

    Args:
        action: The action.

    Returns:
        The policy.
    """

    def act(observation: torch.Tensor) -> torch.Tensor:
        """Return the fixed action once per episode."""
        return torch.full((observation.shape[0],), action, dtype=torch.long)

    return act


def _with_velocity(observation: torch.Tensor) -> torch.Tensor:
    """Push the way the car is moving: right when velocity >= 0, else left.

    Args:
        observation: Scaled position and velocity.

    Returns:
        One action per episode.
    """
    return torch.where(
        observation[:, 1] >= 0.0,
        torch.full((observation.shape[0],), 2, dtype=torch.long),
        torch.zeros(observation.shape[0], dtype=torch.long),
    )


def test_one_step_from_rest_matches_the_published_update() -> None:
    """From x = -0.5 at rest, pushing right.

    v = 0 + 0.001 - 0.0025 * cos(-1.5) = 0.001 - 0.0025 * 0.0707372 = 0.000823157;
    x = -0.5 + v = -0.499176843.
    """
    position, velocity = step(
        torch.tensor([-0.5], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        torch.tensor([2]),
    )
    expected_v = 0.001 - 0.0025 * math.cos(-1.5)
    assert float(velocity[0].item()) == pytest.approx(expected_v, abs=1e-15)
    assert float(position[0].item()) == pytest.approx(-0.5 + expected_v, abs=1e-15)


def test_the_left_wall_stops_a_car_moving_into_it() -> None:
    position, velocity = step(
        torch.tensor([-1.19], dtype=torch.float64),
        torch.tensor([-0.05], dtype=torch.float64),
        torch.tensor([0]),
    )
    assert float(position[0].item()) == -1.2
    assert float(velocity[0].item()) == 0.0


def test_speed_is_clipped_to_the_limit() -> None:
    _, velocity = step(
        torch.tensor([-0.5], dtype=torch.float64),
        torch.tensor([0.0699], dtype=torch.float64),
        torch.tensor([2]),
    )
    assert float(velocity[0].item()) == 0.07


def test_the_observation_is_scaled_into_the_unit_interval() -> None:
    observed = observe(
        torch.tensor([-1.2, 0.6], dtype=torch.float64),
        torch.tensor([-0.07, 0.035], dtype=torch.float64),
    )
    assert observed.dtype == torch.float32
    assert torch.allclose(observed, torch.tensor([[-1.0, -1.0], [1.0, 0.5]]))


def test_starts_are_repeatable_and_inside_the_published_range() -> None:
    first = start_positions(512, 3)
    assert torch.equal(first, start_positions(512, 3))
    assert not torch.equal(first, start_positions(512, 4))
    assert float(first.min().item()) >= -0.6
    assert float(first.max().item()) < -0.4


def test_driving_right_regardless_never_reaches_the_flag() -> None:
    report = evaluate_policy(_always(2), start_positions(1024, 0))
    assert report == {"episodes": 1024, "successes": 0, "mean_steps_to_goal": 0.0}


def test_pushing_with_the_velocity_always_reaches_the_flag() -> None:
    report = evaluate_policy(_with_velocity, start_positions(1024, 0))
    assert report["successes"] == 1024
    assert 100.0 < report["mean_steps_to_goal"] < 140.0


def test_a_recorded_failure_runs_the_whole_step_limit() -> None:
    trace = record_episode(_always(1), -0.5)
    assert len(trace["actions"]) == MAX_STEPS
    assert set(trace["actions"]) == {1}
    assert not trace["reached_goal"]


def test_a_recorded_success_stops_at_the_flag() -> None:
    trace = record_episode(_with_velocity, -0.5)
    assert trace["reached_goal"]
    assert trace["positions"][-1] >= 0.5
    assert all(position < 0.5 for position in trace["positions"][:-1])
    assert set(trace["actions"]) == {0, 2}


def test_a_rendered_trace_keeps_every_nth_step_and_the_last() -> None:
    trace = EpisodeTrace(
        start_position=-0.5,
        positions=(-0.49, -0.48, -0.47),
        velocities=(0.001, 0.002, 0.003),
        actions=(2, 1, 0),
        reached_goal=False,
    )
    assert render_trace(trace, 2) == (
        "episode from position -0.5000 at rest, 3 steps, did not reach the flag\n"
        "step position velocity action\n"
        "1 -0.4900 0.00100 push right\n"
        "3 -0.4700 0.00300 push left"
    )


def test_a_rendered_success_says_so() -> None:
    trace = EpisodeTrace(
        start_position=-0.4, positions=(0.5,), velocities=(0.07,), actions=(2,), reached_goal=True
    )
    assert render_trace(trace, 1).splitlines()[0].endswith("1 steps, reached the flag")


def test_a_rendering_that_keeps_no_steps_is_refused() -> None:
    trace = record_episode(_always(2), -0.5)
    with pytest.raises(ValueError, match="every must keep at least one step"):
        render_trace(trace, 0)
