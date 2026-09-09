"""Which token of a prompt an edit is keyed on.

A rank-one edit writes an association at ONE position, and which position is
not a free choice: locate-then-edit methods key on the subject's last token,
because that is where a causal model has finished reading the entity and has
not yet started reading the relation.

TOKEN COUNTS ARE NOT ADDITIVE, so this does not compute the position as
``len(encode(text_before_the_subject))``. That arithmetic is wrong wherever
byte-pair encoding merges across a join, and it is wrong silently -- the same
defect :func:`~model_trainer.core.services.model.cartridge_qa.answer_span`
was written for, measured on gpt2 where appending ``"AI"`` to a prefix left
the id count unchanged. This module reuses that function rather than
restating its reasoning: the subject is the span between what the prompt
shares with the empty prefix and what it shares with the text after the
subject.
"""

from __future__ import annotations

from typing import Literal

from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.encoding import Encoder
from model_trainer.core.services.model.cartridge_qa import answer_span


def fact_token_position(
    *,
    prompt: str,
    subject: str,
    strategy: Literal["subject_last", "prompt_last"],
    encoder: Encoder,
) -> int:
    """Locate the token an edit keys on, indexed from the start of the prompt.

    Args:
        prompt: The rendered prompt, subject already substituted.
        subject: The entity, as it appears in that prompt.
        strategy: Which token to key on.
        encoder: The tokenizer the edit will run under. The position is a
            fact about a tokenization, so a position found under one
            tokenizer is meaningless under another.

    Returns:
        A non-negative index into the prompt's token ids.

    Raises:
        AppError: With ``EDIT_SUBJECT_NOT_IN_PROMPT`` when ``subject_last`` is
            asked for and the subject does not occur in the prompt. With
            ``EDIT_UPDATE_SHAPE_MISMATCH`` when the prompt encodes to no
            tokens at all, which leaves no position to key on.
    """
    full = encoder.encode(prompt).ids
    if not full:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH,
            message="the prompt encodes to no tokens, so there is no position to key on",
        )
    if strategy == "prompt_last":
        return len(full) - 1

    at = prompt.find(subject)
    if at < 0:
        raise AppError(
            code=ModelTrainerErrorCode.EDIT_SUBJECT_NOT_IN_PROMPT,
            message=(
                f"subject '{subject}' does not occur in prompt '{prompt}', so its last "
                f"token cannot be located"
            ),
        )
    _start, stop = answer_span(
        full,
        encoder.encode(prompt[:at]).ids,
        encoder.encode(prompt[at + len(subject) :]).ids,
    )
    # `answer_span` returns an exclusive stop, and a span can be empty when
    # the tokenizer merged the subject entirely into its neighbours. The last
    # token of an empty span is the token before it, which is the position a
    # reader of the prompt would call "where the subject ends" anyway.
    return max(stop - 1, 0)


__all__ = [
    "fact_token_position",
]
