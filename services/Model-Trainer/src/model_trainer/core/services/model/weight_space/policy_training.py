"""Train the control policy into the local minimum the experiment needs.

The post's Experiment 4 starts from a policy that training left stuck: PPO
"converges to a local-minimum policy where it drives straight at the ramp".
This produces the MountainCar equivalent by the same mechanism, a reward that
pays for progress toward the goal now rather than for reaching it.

THE REWARD IS DISTANCE SHAPING: each step pays the change in position, plus
1.0 on the step the car reaches the flag. Under a SHORT discount (0.9, an
effective horizon of about ten steps) pushing right always pays more than
pushing left, because the payoff of backing up arrives a hundred steps later
and is discounted to nothing. So the trained policy drives at the hill.
Driving right from the valley floor never reaches the flag, which is the
failure the experiment then asks a reader to fix. The discount is a
parameter, recorded on the run, so the construction is stated rather than
hidden.

THE ALGORITHM IS REINFORCE with a per-step batch-mean baseline (Williams
1992), the simplest policy-gradient estimator: the gradient of the sampled
actions' log-probabilities weighted by the discounted return from that step,
minus the mean return at that step across the batch. Adam takes the steps.
Every random draw comes from one private generator seeded by the caller, so
a run is repeatable on the CPU.
"""

from __future__ import annotations

import torch
from typing_extensions import TypedDict

from model_trainer.core.services.model.weight_space.control_policy import ControlPolicy
from model_trainer.core.services.model.weight_space.mountain_car import (
    GOAL_POSITION,
    MAX_STEPS,
    observe,
    start_positions,
    step,
)

#: What reaching the flag pays, on top of the step's progress.
GOAL_BONUS = 1.0


class TrainingConfig(TypedDict):
    """How one policy is trained.

    Attributes:
        seed: Seeds the initial weights and every sampled action and start.
        hidden: Units per hidden layer.
        iterations: Gradient steps.
        batch: Episodes per step.
        discount: The return's per-step discount.
        learning_rate: Adam's step size.
    """

    seed: int
    hidden: int
    iterations: int
    batch: int
    discount: float
    learning_rate: float


class TrainingOutcome(TypedDict):
    """A trained policy and what training saw.

    Not a JSON contract: it holds the policy's tensors.

    Attributes:
        policy: The trained policy.
        final_batch_successes: How many of the last batch's sampled episodes
            reached the flag.
    """

    policy: ControlPolicy
    final_batch_successes: int


def discounted_returns(rewards: torch.Tensor, discount: float) -> torch.Tensor:
    """Sum each step's reward with the discounted rewards after it.

    Args:
        rewards: ``(steps, episodes)`` per-step rewards.
        discount: The per-step discount.

    Returns:
        ``(steps, episodes)`` returns: ``G[t] = r[t] + discount * G[t + 1]``.
    """
    returns = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for index in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[index] + discount * running
        returns[index] = running
    return returns


def _sample_batch(
    policy: ControlPolicy, starts: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Run one batch of sampled episodes, keeping what the gradient needs.

    Args:
        policy: The policy being trained.
        starts: One start per episode.
        generator: The private generator every action is drawn from.

    Returns:
        ``(log_probs, rewards, live, successes)``: the sampled actions'
        log-probabilities, the rewards, and which episodes were still running
        at each step, each ``(steps, episodes)``; and how many reached the
        flag.
    """
    position = starts.clone()
    velocity = torch.zeros_like(position)
    done = torch.zeros(position.shape[0], dtype=torch.bool)
    log_probs: list[torch.Tensor] = []
    rewards: list[torch.Tensor] = []
    live: list[torch.Tensor] = []
    for _ in range(MAX_STEPS):
        scores = policy.forward(observe(position, velocity))
        distribution = torch.softmax(scores, dim=1)
        action = torch.multinomial(distribution, 1, generator=generator).squeeze(1)
        picked = torch.log(distribution.gather(1, action.unsqueeze(1)).squeeze(1))
        moved, velocity = step(position, velocity, action)
        arrived = (moved >= GOAL_POSITION) & ~done
        progress = torch.where(done, torch.zeros_like(moved), moved - position)
        rewards.append(progress + GOAL_BONUS * arrived.to(torch.float64))
        log_probs.append(picked)
        live.append(~done)
        position = torch.where(done, position, moved)
        done = done | arrived
    return (
        torch.stack(log_probs),
        torch.stack(rewards),
        torch.stack(live),
        int(done.sum().item()),
    )


def train_policy(config: TrainingConfig) -> TrainingOutcome:
    """Train one policy by REINFORCE on the distance-shaped reward.

    Args:
        config: The run's settings.

    Returns:
        The trained policy and how its last batch did.

    Raises:
        ValueError: If a count is not positive, the discount is outside
            [0, 1], or the learning rate is not positive.
    """
    if min(config["hidden"], config["iterations"], config["batch"]) < 1:
        raise ValueError("hidden, iterations and batch must each be at least 1")
    if not 0.0 <= config["discount"] <= 1.0:
        raise ValueError(f"the discount must lie in [0, 1], got {config['discount']}")
    if config["learning_rate"] <= 0.0:
        raise ValueError(f"the learning rate must be positive, got {config['learning_rate']}")
    torch.manual_seed(config["seed"])
    policy = ControlPolicy(config["hidden"])
    optimiser = torch.optim.Adam(
        [parameter for _, parameter in policy.named_parameters()], lr=config["learning_rate"]
    )
    generator = torch.Generator().manual_seed(config["seed"])
    successes = 0
    for iteration in range(config["iterations"]):
        starts = start_positions(config["batch"], config["seed"] * 1_000_003 + iteration)
        log_probs, rewards, live, successes = _sample_batch(policy, starts, generator)
        returns = discounted_returns(rewards, config["discount"])
        advantage = (returns - returns.mean(dim=1, keepdim=True)) * live.to(torch.float64)
        loss = -(log_probs.to(torch.float64) * advantage).sum() / float(config["batch"])
        optimiser.zero_grad()
        torch.autograd.backward([loss])
        optimiser.step()
    return TrainingOutcome(policy=policy, final_batch_successes=successes)


__all__ = [
    "GOAL_BONUS",
    "TrainingConfig",
    "TrainingOutcome",
    "discounted_returns",
    "train_policy",
]
