"""Measure how often a language model's greedy continuations fall into loops.

The language-model arm of board task 7964f9f6 mirrors the post's
Experiment 8, where the edit Astra wrote removed the reasoning loops that kept
Qwen3.5-0.8B from finishing answers. GPT-2 has the classic version of that
failure: decoded greedily it repeats itself, a property of maximisation-based
decoding that Holtzman et al. (2020, "The Curious Case of Neural Text
Degeneration") and Welleck et al. (2020, "Neural Text Generation with
Unlikelihood Training") document at length. This module counts it.

TWO MEASURES, BOTH OF THE GENERATED TOKENS ONLY.

* **Looped**: the continuation's last ``window`` tokens repeat with some
  period of at most ``max_period`` -- token ``i`` equals token ``i + p`` for
  every ``i`` in the window. A continuation that ends inside a cycle has
  stopped saying anything new, which is the failure the edit is aimed at.
  Paired per prompt, so control and edited model can be compared prompt by
  prompt.
* **seq-rep-n** (Welleck et al. 2020, section 6): one minus the fraction of
  the continuation's n-grams that are distinct. 0.0 for a continuation that
  never repeats an n-gram; it approaches 1.0 as it collapses onto one.

GREEDY, AND WITHOUT A KEY-VALUE CACHE. Every step re-runs the whole sequence
and takes the highest-scoring next token, so a continuation is a deterministic
function of the weights and the prompt, which is the property a before-and-
after comparison needs. The cost is quadratic in length; at GPT-2 scale and a
few hundred tokens that is minutes, not hours.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from typing_extensions import TypedDict

from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeItem
from model_trainer.core.encoding import Encoder
from model_trainer.core.types import LMModelProto, LogitsOutProto, ScoreableLMProto


class RepetitionSettings(TypedDict):
    """How continuations are generated and judged.

    Attributes:
        new_tokens: Tokens generated per prompt.
        window: How many final tokens the loop test reads.
        max_period: The longest cycle the loop test looks for.
        ngram: The n of seq-rep-n.
    """

    new_tokens: int
    window: int
    max_period: int
    ngram: int


class RepetitionReport(TypedDict):
    """One model's continuations of every prompt, and their measures.

    Attributes:
        continuations: The generated token ids, one tuple per prompt.
        looped: Per prompt, whether its continuation ends in a cycle.
        seq_rep: Per prompt, its seq-rep-n.
    """

    continuations: tuple[tuple[int, ...], ...]
    looped: tuple[bool, ...]
    seq_rep: tuple[float, ...]


def cloze_prompts(
    items: Sequence[ClozeItem], *, count: int, min_tokens: int, max_tokens: int, encoder: Encoder
) -> tuple[tuple[int, ...], ...]:
    """Take prompts from the cloze set: each template's text before its blank.

    The same staged file the cloze score reads, so the arm needs no second
    corpus: the prose before a blank is a sentence the wiki actually wrote,
    and a prefix of it is a natural place for a model to continue.

    Args:
        items: The cloze items, in file order.
        count: How many prompts to take.
        min_tokens: Skip a prefix shorter than this.
        max_tokens: Keep at most this many final tokens of a longer prefix.
        encoder: The tokenizer.

    Returns:
        ``count`` prompts as token ids, from the first items that qualify.

    Raises:
        AppError: With ``CLOZE_ITEMS_EMPTY`` when fewer than ``count`` items
            have a prefix of ``min_tokens``. Refused rather than shortened: a
            run asked for 256 prompts that measured 200 would report a loop
            rate over a set nobody chose.
    """
    prompts: list[tuple[int, ...]] = []
    for item in items:
        if len(prompts) == count:
            break
        prefix = item["template"].split(BLANK_MARKER, 1)[0].rstrip()
        ids = tuple(encoder.encode(prefix).ids)
        if len(ids) >= min_tokens:
            prompts.append(ids[-max_tokens:])
    if len(prompts) < count:
        raise AppError(
            ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY,
            (
                f"only {len(prompts)} of {len(items)} items have a prefix of at least "
                f"{min_tokens} tokens before the blank; {count} prompts were asked for"
            ),
            model_trainer_status_for(ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY),
        )
    return tuple(prompts)


def greedy_continuation(
    model: ScoreableLMProto, prompt: Sequence[int], *, new_tokens: int, device: str
) -> tuple[int, ...]:
    """Extend a prompt one highest-scoring token at a time.

    Args:
        model: The model, already in evaluation mode on ``device``.
        prompt: The prompt's token ids, at least one.
        new_tokens: How many tokens to add.
        device: Where the tensors go.

    Returns:
        The added tokens.

    Raises:
        AppError: With ``CLOZE_ITEM_UNSCOREABLE`` if the model's forward
            returns no per-token scores, which greedy decoding cannot run on.
    """
    sequence = list(prompt)
    with torch.no_grad():
        for _ in range(new_tokens):
            input_ids = torch.empty((1, len(sequence)), dtype=torch.long, device=device)
            for position, token in enumerate(sequence):
                input_ids[0, position] = token
            out = model.forward(input_ids=input_ids, labels=input_ids)
            if not isinstance(out, LogitsOutProto):
                raise AppError(
                    ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE,
                    "this model's forward returned no per-token scores, so there is "
                    "no highest-scoring next token to take",
                    model_trainer_status_for(ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE),
                )
            sequence.append(int(torch.argmax(out.logits[0, -1, :]).item()))
    return tuple(sequence[len(prompt) :])


def ends_in_loop(tokens: Sequence[int], *, window: int, max_period: int) -> bool:
    """Say whether the last ``window`` tokens repeat with a short period.

    Args:
        tokens: The continuation.
        window: How many final tokens to read.
        max_period: The longest period to try; must be under ``window`` so a
            cycle has to occur at least twice to count.

    Returns:
        True when some period ``p`` in ``1..max_period`` has
        ``tail[i] == tail[i + p]`` for every ``i`` in the tail. False for a
        continuation shorter than the window.

    Raises:
        ValueError: If ``max_period`` is not in ``1..window - 1``.
    """
    if not 1 <= max_period < window:
        raise ValueError(f"max_period must lie in 1..{window - 1}, got {max_period}")
    if len(tokens) < window:
        return False
    tail = tuple(tokens[-window:])
    return any(
        all(tail[index] == tail[index + period] for index in range(window - period))
        for period in range(1, max_period + 1)
    )


def seq_rep(tokens: Sequence[int], ngram: int) -> float:
    """One minus the fraction of a continuation's n-grams that are distinct.

    Args:
        tokens: The continuation.
        ngram: The n.

    Returns:
        seq-rep-n, 0.0 for a continuation too short to hold an n-gram.

    Raises:
        ValueError: If ``ngram`` is not positive.
    """
    if ngram < 1:
        raise ValueError(f"ngram must be at least 1, got {ngram}")
    grams = [tuple(tokens[start : start + ngram]) for start in range(len(tokens) - ngram + 1)]
    if not grams:
        return 0.0
    return 1.0 - float(len(set(grams))) / float(len(grams))


def measure_repetition(
    model: LMModelProto,
    prompts: Sequence[Sequence[int]],
    settings: RepetitionSettings,
    *,
    device: str,
) -> RepetitionReport:
    """Continue every prompt greedily and measure each continuation.

    Args:
        model: The model; put in evaluation mode and on ``device`` here, as
            the cloze scorer does, so no dropout reaches a measurement.
        prompts: The prompts.
        settings: Length and judging settings.
        device: Where to run.

    Returns:
        The continuations and their measures, in prompt order.
    """
    model.eval()
    model = model.to(device)
    continuations = tuple(
        greedy_continuation(model, prompt, new_tokens=settings["new_tokens"], device=device)
        for prompt in prompts
    )
    return RepetitionReport(
        continuations=continuations,
        looped=tuple(
            ends_in_loop(tokens, window=settings["window"], max_period=settings["max_period"])
            for tokens in continuations
        ),
        seq_rep=tuple(seq_rep(tokens, settings["ngram"]) for tokens in continuations),
    )


__all__ = [
    "RepetitionReport",
    "RepetitionSettings",
    "cloze_prompts",
    "ends_in_loop",
    "greedy_continuation",
    "measure_repetition",
    "seq_rep",
]
