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


class TraitPairLosses(TypedDict):
    """The four losses one matched pair produces, before anything is derived.

    WHY FOUR RATHER THAN THE TWO DIFFERENCES. Expression and coherence are two
    different linear combinations of the SAME four numbers, and a cartridge
    that expresses a trait by degrading fluency scores identically on the
    first to one that expresses it well. Measuring the losses once and
    deriving both readings from them means the two can never disagree about
    what was run, and costs one forward pass per member rather than two.

    Attributes:
        index: Position of the pair in the supplied order, carried so an
            outcome can be traced back to the text that produced it.
        base_expressing: Loss on the trait-expressing continuation, base only.
        base_neutral: Loss on the neutral continuation, base only.
        arm_expressing: Loss on the trait-expressing continuation, through the
            prefix.
        arm_neutral: Loss on the neutral continuation, through the prefix.
    """

    index: int
    base_expressing: float
    base_neutral: float
    arm_expressing: float
    arm_neutral: float


def measure_trait_losses(
    model: CartridgeModel, pairs: Sequence[TraitPair]
) -> list[TraitPairLosses]:
    """Score both members of every pair, with the prefix and without it.

    The single place the model is actually run for a trait measurement. Both
    arms run under ``no_grad`` and in evaluation mode, because a measurement
    that left dropout active would report a different number every time and
    attribute the noise to the cartridge.

    Args:
        model: The cartridge-wrapped model. Its base is the control arm, so
            both arms share weights, device and dtype and differ only in
            whether the prefix is attended to.
        pairs: Matched continuations, in a stable order. The order is carried
            into the indices, so the same pairs must be supplied in the same
            order across arms for the per-item comparison to mean anything.

    Returns:
        One record per pair, in the order given.
    """
    model.eval()
    return [
        TraitPairLosses(
            index=index,
            base_expressing=_loss_without_prefix(model, pair["expressing"]),
            base_neutral=_loss_without_prefix(model, pair["neutral"]),
            arm_expressing=_loss_with_prefix(model, pair["expressing"]),
            arm_neutral=_loss_with_prefix(model, pair["neutral"]),
        )
        for index, pair in enumerate(pairs)
    ]


def expression_outcomes(losses: Sequence[TraitPairLosses]) -> list[PairedItemOutcome]:
    """Reduce measured losses to the PREFERENCE each arm expressed.

    Each arm is scored as the loss gap between the two continuations, so a
    prefix that helps both members equally cancels and moves nothing. That
    cancellation is the property that makes this a trait measurement rather
    than a fluency measurement, and it is why a bare loss on trait-expressing
    text cannot answer the question: a base with any English priors already
    prefers some phrasings.

    Args:
        losses: Measured losses, in pair order.

    Returns:
        One outcome per pair. ``baseline`` is the plain base's preference and
        ``treatment`` is the cartridge's, so
        :func:`~model_trainer.core.contracts.paired_comparison.summarise_pairs`
        counts an item improved exactly when the cartridge leans further
        toward the trait than the base did.
    """
    return [
        PairedItemOutcome(
            index=item["index"],
            baseline=item["base_expressing"] - item["base_neutral"],
            treatment=item["arm_expressing"] - item["arm_neutral"],
        )
        for item in losses
    ]


def coherence_outcomes(losses: Sequence[TraitPairLosses]) -> list[PairedItemOutcome]:
    """Reduce measured losses to what each arm did to ORDINARY text.

    THE CONTROL A TRAIT MEASUREMENT CANNOT BE READ WITHOUT. A cartridge can
    express a trait by degrading fluency, and
    :func:`expression_outcomes` is blind to that by construction: it
    differences the two members, so a prefix that wrecks both equally scores
    zero expression and a prefix that wrecks only the neutral member scores a
    large one. The neutral continuation is the trait-free text in the
    measurement, so what the prefix does to ITS loss is the fluency reading,
    taken on the same forward passes rather than on a second corpus nobody
    would keep in step with this one.

    Args:
        losses: Measured losses, in pair order.

    Returns:
        One outcome per pair, ``baseline`` the base's loss on the neutral
        continuation and ``treatment`` the cartridge's, so an item counts as
        improved when the prefix made ordinary text EASIER to predict.
    """
    return [
        PairedItemOutcome(
            index=item["index"],
            baseline=item["base_neutral"],
            treatment=item["arm_neutral"],
        )
        for item in losses
    ]


class TraitReading(TypedDict):
    """What one arm did to a trait, and what it did to ordinary text.

    BOTH OR NEITHER, and that is the whole reason this is one type. An
    expression number alone is unreadable: the composition literature's own
    instrument carries a coherence bar precisely because a model can be
    pushed toward a trait by being made worse at everything, and a record
    holding only the first number cannot tell that apart from trait
    acquisition. Reporting them together, from one set of forward passes,
    means no arm can appear in a record with one and not the other.

    Attributes:
        expression: How much further toward the trait this arm leans than the
            plain base does, as a paired comparison over the pairs.
        expression_items: The per-item outcomes ``expression`` was reduced
            from, so a later reader can re-test rather than trust the summary.
        coherence: What this arm did to the loss on the trait-free member of
            each pair.
        coherence_items: The per-item outcomes ``coherence`` was reduced from.
    """

    expression: PairedComparison
    expression_items: list[PairedItemOutcome]
    coherence: PairedComparison
    coherence_items: list[PairedItemOutcome]


def read_trait_pairs(model: CartridgeModel, pairs: Sequence[TraitPair]) -> TraitReading:
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
    base's preference and the cartridge's.

    Both readings map onto :class:`PairedItemOutcome`, so the whole existing
    statistical layer applies unchanged and each gets its own McNemar exact
    conditional test and its own outcomes digest. No judge model is involved
    anywhere, so two runs on one machine produce identical numbers.

    IT LIVES HERE because it is the same operation as its sibling -- score
    tokens with the prefix and without it -- over a pair rather than an item,
    and it reaches the same two private loss helpers. A separate module would
    have had to import them across a boundary or copy them.

    WHAT IT DOES NOT ESTABLISH. It measures RELATIVE PREFERENCE between two
    supplied continuations, not what the model would generate unprompted. A
    cartridge can move preference without changing sampled output, and this
    is a scoring instrument rather than a generation one.

    Args:
        model: The cartridge-wrapped model. Its base is the control arm, so
            both arms share weights, device and dtype and differ only in
            whether the prefix is attended to.
        pairs: Matched continuations, in a stable order. The order is carried
            into the outcome indices, so the same pairs must be supplied in
            the same order across arms for the per-item comparison to mean
            anything.

    Returns:
        The two readings and the per-item outcomes each was reduced from.
    """
    losses = measure_trait_losses(model, pairs)
    expression = expression_outcomes(losses)
    coherence = coherence_outcomes(losses)
    return TraitReading(
        expression=summarise_pairs(expression),
        expression_items=expression,
        coherence=summarise_pairs(coherence),
        coherence_items=coherence,
    )


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
    "TraitPairLosses",
    "TraitReading",
    "base_loss",
    "coherence_outcomes",
    "expression_outcomes",
    "measure_trait_losses",
    "read_trait_pairs",
    "score_held_out",
    "train_on",
]
