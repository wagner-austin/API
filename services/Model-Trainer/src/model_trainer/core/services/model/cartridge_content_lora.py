"""Crowd-invariance distillation: the content lever for depth.

THE MEASUREMENT CHAIN THIS ANSWERS (board task ``a85fbabe``, baseline
``372cee59`` cross-node bit-identical). The base-LoRA arm settled the
attribution at both scales: the LM objective repairs the STRUCTURAL half of
crowded-prefix interference -- gpt2-medium's n8 noise-composition control
flips -0.29 to +0.42 -- and leaves the CONTENT half standing, a 1.04 gap
between that repaired control and real-content composition. The LM objective
cannot close it even in principle: language modeling behind a crowd rewards
reading past the crowd's SHAPE, and says nothing about ignoring what the
crowd SAYS.

THIS OBJECTIVE IS THE MEASURED QUANTITY. Per step, a roster of frozen pool
cartridges is drawn, one member is drawn as the TARGET, and the window comes
from the target's own corpus. The teacher is the PLAIN base behind the
target alone -- the canonical as-if-alone behaviour the cartridges were
trained against -- and the student is the LoRA-adapted base behind the full
roster. The loss is the KL divergence from the teacher's next-token
distributions to the student's: behind a crowd, on text belonging to
compartment ``i``, predict as if only compartment ``i`` were present. The
target is drawn per step over every roster position, so no positional
shortcut exists and every compartment stays live -- which is the serving
semantics, where any wired compartment may be the one a request belongs to.

WHAT IS TRAINABLE IS EXACTLY THE LORA, by the same construction as the LM
trainer: pool blocks are detached in :func:`composed_prefix_blocks`, the
teacher runs under ``no_grad`` on a separately loaded frozen base, and the
optimizer is built over the adapted model's trainable set alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from model_trainer.core.services.finetuning.strategies._test_hooks import Hooks
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.finetuning.strategies.cartridge_slots import CartridgeSlots
from model_trainer.core.services.model.cartridge_base_lora import composed_prefix_blocks
from model_trainer.core.services.training.base_trainer_core import _get_optimizer_for_config
from model_trainer.core.types import CacheCapableLMProto, ForwardOutProto, LogitsOutProto


def invariance_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """KL divergence from the teacher's token distributions to the student's.

    ``KL(teacher || student)`` per position, averaged over positions -- the
    distillation direction, so the student is pulled to cover everywhere the
    teacher puts mass. Zero exactly when the two emit identical
    distributions, which is the objective's fixed point: a student that
    predicts behind the crowd what the teacher predicts alone has nothing
    left to learn.

    Args:
        student: The adapted model's logits behind the full roster, shaped
            (batch, positions, vocab). Gradients flow through this side.
        teacher: The plain base's logits behind the target alone, same
            shape. Detached by the caller.

    Returns:
        The scalar loss.
    """
    return torch.nn.functional.kl_div(
        torch.log_softmax(student, dim=-1),
        torch.log_softmax(teacher, dim=-1),
        reduction="batchmean",
        log_target=True,
    )


def _require_logits(out: ForwardOutProto, *, side: str) -> torch.Tensor:
    """Read per-token scores out of a forward, refusing a loss-only output.

    Args:
        out: What the forward returned.
        side: ``"teacher"`` or ``"student"``, for the refusal.

    Returns:
        The logits.

    Raises:
        ValueError: When the output carries no logits. Distillation compares
            distributions; a loss is a mean over them and cannot be matched
            token-wise, so continuing would train a different objective than
            the record claims.
    """
    if not isinstance(out, LogitsOutProto):
        raise ValueError(
            f"the {side} forward returned no per-token scores; crowd-invariance "
            f"distillation matches next-token distributions and cannot run on a "
            f"loss alone"
        )
    return out.logits


def train_composition_lora_invariant(
    adapted: CacheCapableLMProto,
    teacher_base: CacheCapableLMProto,
    pool: tuple[CartridgeSlots, ...],
    member_windows: Sequence[Sequence[torch.Tensor]],
    *,
    max_drawn: int,
    seed: int,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    """Train the base's LoRA to predict behind a crowd as the base does alone.

    The same seeding discipline as the LM-objective trainer: the seed is set
    once before the loop and every per-step draw -- count, permutation,
    target position, window index -- consumes the global generator in a
    fixed order, so training is a pure function of its inputs and the seed.
    One epoch is as many steps as the member windows hold in total, each
    step drawing its window with replacement, so the exposure scale matches
    the LM objective's pass count without coupling step order to corpus
    order.

    Args:
        adapted: The PEFT-wrapped base whose LoRA learns.
        teacher_base: A SEPARATE plain instance of the same base, frozen by
            :class:`CartridgeModel` on first use. Separate because PEFT
            injects its adapters into the wrapped module tree, so the
            adapted model cannot also serve as the un-adapted teacher.
        pool: Frozen cartridges the roster is drawn from. At least two, one
            slot width, as the LM trainer requires.
        member_windows: One training-window sequence per pool member, index
            for index -- ``member_windows[i]`` is text from the corpus
            ``pool[i]`` was trained on. Every sequence non-empty. These must
            be held out from every measured corpus; the caller owns that
            wall and the CLI refuses breaches of it.
        max_drawn: Largest roster one step may draw, at least two and at
            most the pool size.
        seed: Seed for the per-step draws.
        epochs: Passes worth of steps, as defined above.
        learning_rate: Step size for AdamW over the LoRA parameters.

    Returns:
        The mean loss of each epoch, in order, so the record can show the
        distillation converged rather than asserting it did.

    Raises:
        ValueError: If the pool, ``max_drawn`` or ``member_windows`` are out
            of contract, or a forward yields no logits.
    """
    if len(pool) < 2:
        raise ValueError(
            f"a pool of {len(pool)} cartridge(s) cannot crowd a prefix; "
            f"the roster draw would be a dead knob"
        )
    if not 2 <= max_drawn <= len(pool):
        raise ValueError(
            f"max_drawn={max_drawn} is outside [2, {len(pool)}]: a ceiling of "
            f"one would never present a crowd to be invariant to, and more "
            f"than the pool holds cannot be drawn"
        )
    if len(member_windows) != len(pool):
        raise ValueError(
            f"{len(member_windows)} window sequence(s) for {len(pool)} pool "
            f"member(s); the target draw needs each member's own text"
        )
    empty = [position for position, windows in enumerate(member_windows) if len(windows) == 0]
    if empty:
        raise ValueError(
            f"member(s) {empty} carry no training windows; a member that can "
            f"never be the target is one the student never learns to hear "
            f"through the crowd"
        )

    teachers = tuple(CartridgeModel(base=teacher_base, slots=member) for member in pool)
    steps_per_epoch = sum(len(windows) for windows in member_windows)
    optimiser = _get_optimizer_for_config("adamw")(
        [parameter for parameter in adapted.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    adapted.train()
    torch.manual_seed(seed)
    epoch_losses: list[float] = []
    for _ in range(epochs):
        total = 0.0
        for _ in range(steps_per_epoch):
            # 1..max_drawn, like the LM trainer: a count of one puts the
            # target ALONE behind the student, which distils "the LoRA does
            # not disturb the alone case" -- the solo-cost axis the
            # measurement prices -- while larger counts distil the crowd
            # away. Both belong to the objective.
            count = int(torch.randint(1, max_drawn + 1, ()))
            order = torch.randperm(len(pool))
            roster = [int(order[position].item()) for position in range(count)]
            target = roster[int(torch.randint(0, count, ()))]
            windows = member_windows[target]
            window = windows[int(torch.randint(0, len(windows), ()))]
            batch_size = int(window.shape[0])

            with torch.no_grad():
                teacher_out = teachers[target].forward(input_ids=window, labels=window)
            teacher_logits = _require_logits(teacher_out, side="teacher").detach()

            drawn = [pool[member] for member in roster]
            blocks = composed_prefix_blocks(drawn, batch_size=batch_size)
            attended = int(window.shape[1]) + count * pool[0].geometry["num_slots"]
            student_out = adapted(
                input_ids=window,
                labels=window,
                past_key_values=Hooks.build_prefix_cache(blocks),
                attention_mask=torch.ones(
                    (batch_size, attended), dtype=torch.long, device=window.device
                ),
                use_cache=False,
            )
            student_logits = _require_logits(student_out, side="student")

            optimiser.zero_grad()
            loss = invariance_loss(student_logits, teacher_logits)
            torch.autograd.backward([loss])
            optimiser.step()
            total += float(loss.item())
        epoch_losses.append(total / steps_per_epoch)
    return epoch_losses


__all__ = [
    "invariance_loss",
    "train_composition_lora_invariant",
]
