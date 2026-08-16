"""Score cloze items against a causal language model.

Each item is scored by substitution: the template is rendered once per
candidate, every rendering is assigned a total negative log-likelihood, and the
item counts as correct when the true rendering is the least surprising. The
answer is rendered at index 0, so correctness is a comparison against zero.

Total likelihood, not mean, is what decides. Renderings differ only by the
masked span, so a mean would reward candidates that happen to tokenise long,
which is a property of the tokenizer rather than of what the model knows. The
model reports a mean over the tokens it predicted, so it is multiplied back out
by that count.

Nothing here is sampled, so a run is reproducible without pinning a seed.

Dependencies arrive as parameters rather than through a hook module: the model
and encoder are chosen by the caller, which is what makes this function
testable against a fake model implementing ``LMModelProto`` without patching
anything.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from model_trainer.core.contracts.cloze import (
    ClozeEvalResult,
    ClozeItem,
    ClozeItemOutcome,
    answer_wins_outright,
    render_candidates,
)
from model_trainer.core.encoding import Encoder
from model_trainer.core.types import LMModelProto

MIN_SCOREABLE_TOKENS = 2


def sequence_nll(
    *,
    model: LMModelProto,
    encoder: Encoder,
    text: str,
    device: str,
    max_seq_len: int,
    item_id: str,
) -> float:
    """Total negative log-likelihood the model assigns to one rendering.

    Args:
        model: Loaded causal language model exposing ``forward``.
        encoder: Tokenizer used to turn the rendering into token ids.
        text: The rendered sentence to score.
        device: Torch device string the tensors are placed on.
        max_seq_len: Token budget; longer renderings are truncated to it.
        item_id: Item the rendering belongs to, used in error messages.

    Returns:
        Sum of per-token negative log-likelihoods over the predicted tokens.

    Raises:
        AppError: With ``CLOZE_ITEM_UNSCOREABLE`` when the rendering tokenises
            to fewer than two ids, leaving no token for the model to predict.
    """
    ids = encoder.encode(text).ids[:max_seq_len]
    if len(ids) < MIN_SCOREABLE_TOKENS:
        raise AppError(
            ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE,
            (
                f"item '{item_id}' rendered to {len(ids)} token(s); "
                f"at least {MIN_SCOREABLE_TOKENS} are needed to predict one"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE),
        )

    batch: list[list[int]] = [ids]
    input_ids = torch.tensor(batch, dtype=torch.long, device=device)
    outputs = model.forward(input_ids=input_ids, labels=input_ids)
    predicted_tokens = len(ids) - 1
    mean_nll = float(outputs.loss.item())
    return mean_nll * float(predicted_tokens)


def score_cloze_items(
    *,
    items: Sequence[ClozeItem],
    model: LMModelProto,
    encoder: Encoder,
    device: str,
    max_seq_len: int,
) -> ClozeEvalResult:
    """Score every item and report accuracy against the guessing baseline.

    Args:
        items: Items to score; must not be empty.
        model: Loaded causal language model exposing ``forward``.
        encoder: Tokenizer used to turn renderings into token ids.
        device: Torch device string the tensors are placed on.
        max_seq_len: Token budget each rendering is truncated to.

    Returns:
        Totals, the count the model got right, its accuracy, the accuracy
        uniform guessing would reach on the same candidate counts, and one
        outcome per item carrying its per-candidate scores.

    Raises:
        AppError: With ``CLOZE_ITEMS_EMPTY`` when no items were supplied, or
            ``CLOZE_ITEM_UNSCOREABLE`` when a rendering cannot be scored.
    """
    if len(items) == 0:
        raise AppError(
            ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY,
            "no cloze items supplied; accuracy is undefined over an empty set",
            model_trainer_status_for(ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY),
        )

    # The renderings are tokenised onto `device`, so the model has to be there
    # too. A freshly loaded model sits on CPU, and scoring it against cuda
    # tensors raises "Expected all tensors to be on the same device" inside the
    # embedding lookup. Placement belongs here rather than at the call site
    # because this function is what chose the device for its own tensors.
    model.eval()
    model = model.to(device)

    correct = 0
    chance_total = 0.0
    outcomes: list[ClozeItemOutcome] = []
    with torch.no_grad():
        for item in items:
            renderings = render_candidates(item)
            scores = [
                sequence_nll(
                    model=model,
                    encoder=encoder,
                    text=rendering,
                    device=device,
                    max_seq_len=max_seq_len,
                    item_id=item["item_id"],
                )
                for rendering in renderings
            ]
            item_correct = answer_wins_outright(scores)
            if item_correct:
                correct += 1
            chance_total += 1.0 / float(len(renderings))
            outcomes.append(
                ClozeItemOutcome(
                    item_id=item["item_id"],
                    correct=item_correct,
                    scores=scores,
                )
            )

    total = len(items)
    return ClozeEvalResult(
        total=total,
        correct=correct,
        accuracy=float(correct) / float(total),
        chance=chance_total / float(total),
        outcomes=outcomes,
    )


__all__ = [
    "MIN_SCOREABLE_TOKENS",
    "score_cloze_items",
    "sequence_nll",
]
