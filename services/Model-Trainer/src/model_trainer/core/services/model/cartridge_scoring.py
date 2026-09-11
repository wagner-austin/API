"""Score held-out text with a cartridge and without it, on the same tokens.

THE QUESTION THIS ANSWERS, which the strategy's own tests do not. Those show
the loss falling on a fixed batch, which is the batch being memorised -- a
prefix with enough slots can do that while learning nothing about the corpus
the batch came from. The claim that matters is different: text the cartridge
was NEVER TRAINED ON, drawn from the same corpus, becomes easier to predict.
That is what "the model knows the corpus" means operationally, and nothing
short of held-out text tests it.

The control is the cartridge's own base model, reached through
:attr:`CartridgeModel.base`. Same weights, same device, same dtype, same
tokens, differing only in whether the prefix is attended to. A control loaded
separately would differ in ways nobody enumerated, and the difference would be
reported as an effect of the cartridge.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict

import torch

from model_trainer.core.contracts.paired_comparison import (
    PairedComparison,
    PairedItemOutcome,
    summarise_pairs,
)
from model_trainer.core.services.finetuning.strategies.cartridge_model import CartridgeModel
from model_trainer.core.services.training.base_trainer_core import _get_optimizer_for_config
from model_trainer.core.types import CacheCapableLMProto, ForwardOutProto, ParameterLike


class PrefixTrainableProto(Protocol):
    """What :func:`train_on` actually consumes of the model it drives.

    Named when the base-side composition LoRA arrived: the loop had been
    annotated as taking a :class:`CartridgeModel`, but nothing in it reads a
    slot -- it optimizes ``parameters()``, calls ``train()`` once and
    ``forward`` per item. The base-LoRA trainer satisfies exactly that with
    the gradient landing on the OTHER side of the prefix, so the honest type
    is the surface the loop touches, not one caller's class.
    """

    def parameters(self) -> Sequence[ParameterLike]:
        """Return the parameters the optimizer may update."""
        ...

    def train(self) -> None:
        """Put the model in training mode."""
        ...

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Run one training forward."""
        ...


def base_loss(base: CacheCapableLMProto, item: torch.Tensor) -> float:
    """Score one item on a plain base: no prefix, no cache, no mask.

    This is the model as it would answer without a cartridge existing --
    the control arm of every recorded gain, and (public since the 7B
    precondition finding) the headroom term on its own: how far a base
    already sits from a corpus is exactly what bounds what a cartridge
    can add. The caller owns evaluation mode; a measurement that left
    dropout active would report a different number every run.

    Args:
        base: The plain model to score with.
        item: Token ids shaped (1, positions).

    Returns:
        The item's loss.
    """
    with torch.no_grad():
        out = base(input_ids=item, labels=item)
    return float(out.loss.item())


def _loss_without_prefix(model: CartridgeModel, item: torch.Tensor) -> float:
    """Score one item on the base model alone.

    Args:
        model: The cartridge-wrapped model, used only to reach its base.
        item: Token ids shaped (1, positions).

    Returns:
        The item's loss.
    """
    return base_loss(model.base, item)


def _loss_with_prefix(model: CartridgeModel, item: torch.Tensor) -> float:
    """Score one item through the cartridge.

    Args:
        model: The cartridge-wrapped model.
        item: Token ids shaped (1, positions).

    Returns:
        The item's loss.
    """
    with torch.no_grad():
        out = model.forward(input_ids=item, labels=item)
    return float(out.loss.item())


def score_held_out(
    model: CartridgeModel, items: Sequence[torch.Tensor]
) -> tuple[PairedComparison, list[PairedItemOutcome]]:
    """Score every held-out item with the cartridge and without it.

    Both arms run under ``no_grad`` and in evaluation mode, because a
    measurement that left dropout active would report a different number every
    time it ran and attribute the noise to the cartridge.

    Args:
        model: The cartridge-wrapped model. Its base is the control arm.
        items: Held-out sequences, each shaped (1, positions). Text the
            cartridge was not trained on; passing training text measures
            memorisation and answers a different question.

    Returns:
        The comparison and the per-item outcomes it was reduced from. Both,
        because the comparison is what a report shows and the outcomes are
        what another run is diffed against.
    """
    model.eval()
    outcomes = [
        PairedItemOutcome(
            index=index,
            baseline=_loss_without_prefix(model, item),
            treatment=_loss_with_prefix(model, item),
        )
        for index, item in enumerate(items)
    ]
    return summarise_pairs(outcomes), outcomes


class TraitPair(TypedDict):
    """One prompt continued two ways, differing only in the trait.

    Attributes:
        expressing: Token ids shaped (1, positions) for the continuation that
            exhibits the trait.
        neutral: Token ids shaped (1, positions) for the continuation that does
            not. Matched to ``expressing`` in prompt and, as far as the corpus
            allows, in length and content -- every way the two differ beyond
            the trait is an alternative explanation for the score.
    """

    expressing: torch.Tensor
    neutral: torch.Tensor


def score_trait_expression(
    model: CartridgeModel, pairs: Sequence[TraitPair]
) -> tuple[PairedComparison, list[PairedItemOutcome]]:
    """Score whether the cartridge shifts preference toward a trait.

    THE QUESTION THIS ANSWERS IS NOT THE ONE :func:`score_held_out` ANSWERS.
    That one asks whether held-out text from a corpus became easier to
    predict, which is what "the model knows the corpus" means. This asks
    whether the prefix carries a DISPOSITION: given the same prompt, does the
    cartridge prefer the continuation that exhibits a trait more than the
    plain base already does?

    WHY A DIFFERENCE OF DIFFERENCES. A bare loss on trait-expressing text
    cannot answer it, because a base with any English priors already prefers
    some phrasings, and a prefix that merely lowers loss everywhere would look
    like trait acquisition. So each arm is scored as a PREFERENCE -- the loss
    gap between the two continuations -- and the comparison is between the
    base's preference and the cartridge's. A prefix that helps both
    continuations equally moves nothing here, which is the property that makes
    this a trait measurement rather than a fluency measurement.

    The mapping onto :class:`PairedItemOutcome` is exact and deliberate, so
    the whole existing statistical layer applies unchanged: ``baseline`` is
    the plain base's preference, ``treatment`` is the cartridge's, and
    :func:`summarise_pairs` counts an item improved when ``treatment <
    baseline`` -- that is, when the cartridge leans further toward the trait
    than the base did. No judge model is involved anywhere, so two runs on one
    machine produce identical numbers.

    IT LIVES HERE because it is the same operation as its sibling -- score
    tokens with the prefix and without it -- over a pair rather than an item,
    and it reaches the same two private loss helpers. A separate module would
    have had to import them across a boundary or copy them.

    WHAT IT DOES NOT ESTABLISH. It measures RELATIVE PREFERENCE between two
    supplied continuations, not what the model would generate unprompted. A
    cartridge can move preference without changing sampled output, and a
    cartridge that expresses a trait by degrading fluency scores the same as
    one that expresses it well. A coherence arm is a separate measurement and
    this function does not stand in for it.

    Args:
        model: The cartridge-wrapped model. Its base is the control arm, so
            both arms share weights, device and dtype and differ only in
            whether the prefix is attended to.
        pairs: Matched continuations, in a stable order. The order is carried
            into the outcome indices, so the same pairs must be supplied in
            the same order across arms for the per-item comparison to mean
            anything.

    Returns:
        The comparison and the per-item outcomes it was reduced from, matching
        :func:`score_held_out` so both can feed one report.
    """
    model.eval()
    outcomes = [
        PairedItemOutcome(
            index=index,
            baseline=_loss_without_prefix(model, pair["expressing"])
            - _loss_without_prefix(model, pair["neutral"]),
            treatment=_loss_with_prefix(model, pair["expressing"])
            - _loss_with_prefix(model, pair["neutral"]),
        )
        for index, pair in enumerate(pairs)
    ]
    return summarise_pairs(outcomes), outcomes


def train_on(
    model: PrefixTrainableProto,
    items: Sequence[torch.Tensor],
    *,
    epochs: int,
    learning_rate: float,
) -> list[float]:
    """Train a prefix-shaped model over a corpus, one item at a time.

    Deliberately the plainest loop that is still real: every item, every
    epoch, one optimizer step each. It exists so a measurement can state what
    produced the artifact it is about to score, without routing through the
    full trainer -- which brings checkpointing, validation splits and early
    stopping, none of which a controlled comparison wants varying underneath
    it.

    Args:
        model: The model to drive. Whatever its ``parameters()`` chooses to
            expose is what learns -- a cartridge model exposes its slots, the
            base-LoRA model exposes the LoRA -- and everything else is frozen
            by that model's own construction.
        items: Training sequences, each shaped (1, positions).
        epochs: Passes over the corpus.
        learning_rate: Step size for AdamW.

    Returns:
        The mean loss of each epoch, in order, so a caller can show the run
        converged rather than asserting it did.
    """
    optimiser = _get_optimizer_for_config("adamw")(model.parameters(), lr=learning_rate)
    model.train()
    epoch_losses: list[float] = []
    for _ in range(epochs):
        total = 0.0
        for item in items:
            optimiser.zero_grad()
            out = model.forward(input_ids=item, labels=item)
            torch.autograd.backward([out.loss])
            optimiser.step()
            total += float(out.loss.item())
        epoch_losses.append(total / len(items))
    return epoch_losses


__all__ = [
    "PrefixTrainableProto",
    "TraitPair",
    "base_loss",
    "score_held_out",
    "score_trait_expression",
    "train_on",
]
