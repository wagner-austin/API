"""MountainCar: a task whose obvious policy fails, so a fix is measurable.

The small-network arm of board task 7964f9f6 reproduces the post's
Experiment 4, where a PPO policy "converges to a local-minimum policy where it
drives straight at the ramp and gets stuck" and a language model, reading the
weights, rewrites them so the car backs up first. MountainCar is the classic
form of the same trap: an underpowered car in a valley cannot drive straight
up the right-hand hill, and reaches the flag only by first rolling up the
left-hand slope to gain momentum.

THE DYNAMICS ARE GYM'S MountainCar-v0 (Moore 1990; Sutton and Barto, Example
10.1), constant for constant, so a reader can check any number here against
the published environment::

    velocity += (action - 1) * 0.001 - 0.0025 * cos(3 * position)
    velocity  = clip(velocity, -0.07, 0.07)
    position += velocity, clipped to [-1.2, 0.6]
    a car stopped at the left wall loses its leftward velocity
    success   = position >= 0.5, within 200 steps

Actions are 0 (push left), 1 (no push) and 2 (push right). Episodes start at
rest, at a position drawn uniformly from [-0.6, -0.4).

THE POLICY SEES A SCALED STATE: ``(position + 0.3) / 0.9`` and
``velocity / 0.07``, each in [-1, 1]. The scaling is part of the scene a
weight reader is told about; an input it could not put a unit to would make
reading the first layer guesswork.

Everything runs in float64 on the CPU. The arm is small enough that speed is
irrelevant and exact repeatability is not.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from typing_extensions import TypedDict

FORCE = 0.001
GRAVITY = 0.0025
MAX_SPEED = 0.07
MIN_POSITION = -1.2
MAX_POSITION = 0.6
GOAL_POSITION = 0.5
MAX_STEPS = 200
START_LOW = -0.6
START_HIGH = -0.4

#: The observation's centring and scale, so it lies in [-1, 1].
POSITION_CENTRE = -0.3
POSITION_SCALE = 0.9

#: The names of the three actions, in index order.
ACTION_NAMES: tuple[str, str, str] = ("push left", "no push", "push right")


class SuccessReport(TypedDict):
    """How a policy did over a batch of episodes.

    Attributes:
        episodes: How many were run.
        successes: How many reached the flag within the step limit.
        mean_steps_to_goal: The mean step count of the successful ones, 0.0
            when none succeeded.
    """

    episodes: int
    successes: int
    mean_steps_to_goal: float


class EpisodeTrace(TypedDict):
    """One episode, step by step, as a weight reader is shown it.

    Attributes:
        start_position: Where the car began.
        positions: The position after each step.
        velocities: The velocity after each step.
        actions: The action taken at each step.
        reached_goal: Whether the episode ended at the flag.
    """

    start_position: float
    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    actions: tuple[int, ...]
    reached_goal: bool


def start_positions(episodes: int, seed: int) -> torch.Tensor:
    """Draw starting positions from a generator of its own.

    A private generator, so drawing the evaluation's starts neither depends on
    nor disturbs any other random state in the process.

    Args:
        episodes: How many to draw.
        seed: The generator's seed.

    Returns:
        A float64 vector of positions in [-0.6, -0.4).
    """
    generator = torch.Generator().manual_seed(seed)
    uniform = torch.rand(episodes, generator=generator, dtype=torch.float64)
    return START_LOW + (START_HIGH - START_LOW) * uniform


def observe(position: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    """Scale the state into what the policy reads.

    Args:
        position: One position per episode.
        velocity: One velocity per episode.

    Returns:
        A float32 ``(episodes, 2)`` matrix of scaled position and velocity.
    """
    scaled = torch.stack(
        ((position - POSITION_CENTRE) / POSITION_SCALE, velocity / MAX_SPEED), dim=1
    )
    return scaled.to(torch.float32)


def step(
    position: torch.Tensor, velocity: torch.Tensor, action: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance every episode by one step.

    Args:
        position: One position per episode.
        velocity: One velocity per episode.
        action: One action index per episode.

    Returns:
        The new positions and velocities.
    """
    push = (action.to(torch.float64) - 1.0) * FORCE
    velocity = torch.clamp(
        velocity + push - GRAVITY * torch.cos(3.0 * position), -MAX_SPEED, MAX_SPEED
    )
    position = torch.clamp(position + velocity, MIN_POSITION, MAX_POSITION)
    at_wall = (position == MIN_POSITION) & (velocity < 0.0)
    velocity = torch.where(at_wall, torch.zeros_like(velocity), velocity)
    return position, velocity


def evaluate_policy(
    act: Callable[[torch.Tensor], torch.Tensor], starts: torch.Tensor
) -> SuccessReport:
    """Run one episode per start and count the ones that reach the flag.

    An episode that has reached the flag is frozen there: its position stops
    moving and its step count is recorded, which is what ``terminated`` does
    in the published environment.

    Args:
        act: Maps a ``(episodes, 2)`` observation to one action per episode.
        starts: One starting position per episode.

    Returns:
        The count of successes and their mean step count.
    """
    position = starts.clone()
    velocity = torch.zeros_like(position)
    done = torch.zeros(position.shape[0], dtype=torch.bool)
    steps_taken = torch.zeros(position.shape[0], dtype=torch.float64)
    for index in range(MAX_STEPS):
        with torch.no_grad():
            action = act(observe(position, velocity))
        moved, velocity = step(position, velocity, action)
        position = torch.where(done, position, moved)
        arrived = (position >= GOAL_POSITION) & ~done
        steps_taken = torch.where(arrived, torch.full_like(steps_taken, index + 1), steps_taken)
        done = done | arrived
    successes = int(done.sum().item())
    mean_steps = float(steps_taken[done].mean().item()) if successes else 0.0
    return SuccessReport(
        episodes=int(position.shape[0]),
        successes=successes,
        mean_steps_to_goal=mean_steps,
    )


def record_episode(act: Callable[[torch.Tensor], torch.Tensor], start: float) -> EpisodeTrace:
    """Run one episode and keep every step, for a weight reader to look at.

    Args:
        act: The policy, as :func:`evaluate_policy` takes it.
        start: The starting position.

    Returns:
        The episode, ending at the flag or at the step limit.
    """
    position = torch.full((1,), start, dtype=torch.float64)
    velocity = torch.zeros_like(position)
    positions: list[float] = []
    velocities: list[float] = []
    actions: list[int] = []
    reached = False
    while len(actions) < MAX_STEPS and not reached:
        with torch.no_grad():
            action = act(observe(position, velocity))
        position, velocity = step(position, velocity, action)
        actions.append(int(action[0].item()))
        positions.append(float(position[0].item()))
        velocities.append(float(velocity[0].item()))
        reached = positions[-1] >= GOAL_POSITION
    return EpisodeTrace(
        start_position=start,
        positions=tuple(positions),
        velocities=tuple(velocities),
        actions=tuple(actions),
        reached_goal=reached,
    )


def render_trace(trace: EpisodeTrace, every: int) -> str:
    """Write an episode as the lines a weight reader is shown.

    Args:
        trace: The episode.
        every: Keep one step in this many, plus the last; 1 keeps them all.

    Returns:
        One line per kept step, ``step position velocity action``, after a
        header naming the start and the outcome.

    Raises:
        ValueError: If ``every`` is not positive.
    """
    if every < 1:
        raise ValueError(f"every must keep at least one step in each run, got {every}")
    outcome = "reached the flag" if trace["reached_goal"] else "did not reach the flag"
    lines = [
        f"episode from position {trace['start_position']:.4f} at rest, "
        f"{len(trace['actions'])} steps, {outcome}",
        "step position velocity action",
    ]
    last = len(trace["actions"]) - 1
    for index, action in enumerate(trace["actions"]):
        if index % every == 0 or index == last:
            lines.append(
                f"{index + 1} {trace['positions'][index]:.4f} "
                f"{trace['velocities'][index]:.5f} {ACTION_NAMES[action]}"
            )
    return "\n".join(lines)


__all__ = [
    "ACTION_NAMES",
    "FORCE",
    "GOAL_POSITION",
    "GRAVITY",
    "MAX_POSITION",
    "MAX_SPEED",
    "MAX_STEPS",
    "MIN_POSITION",
    "POSITION_CENTRE",
    "POSITION_SCALE",
    "START_HIGH",
    "START_LOW",
    "EpisodeTrace",
    "SuccessReport",
    "evaluate_policy",
    "observe",
    "record_episode",
    "render_trace",
    "start_positions",
    "step",
]
