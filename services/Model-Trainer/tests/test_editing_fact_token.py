"""Which token an edit keys on, under a tokenizer that really merges.

NOTHING IS FAKED AWAY THAT MATTERS. The encoder is a small deterministic
byte-pair-style stand-in whose merge behaviour is the property under test:
the position must be found by AGREEMENT with both contexts rather than by
``len(encode(before))``, and a tokenizer that never merges could not tell the
two apart. The one that ships here merges across a join, so a length-based
implementation fails these tests.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.encoding import Encoded, Encoder, ListEncoded
from model_trainer.core.services.model.editing.fact_token import fact_token_position


class MergingEncoder:
    """A word tokenizer that merges ``GBM`` into a preceding word.

    The merge is the point. Real byte-pair encoding re-tokenises across a
    join, so a span located by counting the prefix's ids is wrong exactly
    where a merge happened -- measured on gpt2, appending ``AI`` to one
    prefix left the id count unchanged. This reproduces that in miniature.
    """

    def encode(self: MergingEncoder, text: str) -> Encoded:
        """Split on whitespace, folding a bare ``GBM`` into its predecessor.

        Args:
            text: Text to encode.

        Returns:
            One id per surviving word.
        """
        words: list[str] = []
        for word in text.split():
            if word == "GBM" and words:
                words[-1] = f"{words[-1]}GBM"
                continue
            words.append(word)
        return ListEncoded([abs(hash(word)) % 50000 for word in words])

    def token_to_id(self: MergingEncoder, token: str) -> int | None:
        return abs(hash(token)) % 50000

    def get_vocab_size(self: MergingEncoder) -> int:
        return 50000

    def decode(self: MergingEncoder, ids: list[int]) -> str:
        raise AssertionError("nothing here decodes")


_ENCODER: Encoder = MergingEncoder()


class TestSubjectLast:
    def test_it_finds_the_last_token_of_a_leading_subject(self) -> None:
        """The shape every curated triple has: subject first, relation after."""
        position = fact_token_position(
            prompt="XGBoost sits beside the from-scratch booster",
            subject="XGBoost",
            strategy="subject_last",
            encoder=_ENCODER,
        )

        assert position == 0

    def test_it_finds_a_subject_that_is_not_first(self) -> None:
        position = fact_token_position(
            prompt="the library called LightGBM was merged into",
            subject="LightGBM",
            strategy="subject_last",
            encoder=_ENCODER,
        )

        # the(0) library(1) called(2) LightGBM(3) ...
        assert position == 3

    def test_a_multi_token_subject_keys_on_its_final_token(self) -> None:
        """Not its first: a causal model has finished reading the entity only
        at the end of it.
        """
        position = fact_token_position(
            prompt="Model Trainer trains a cartridge",
            subject="Model Trainer",
            strategy="subject_last",
            encoder=_ENCODER,
        )

        assert position == 1

    def test_a_merge_across_the_join_does_not_move_the_position(self) -> None:
        """THE DEFECT THIS FUNCTION EXISTS TO AVOID.

        `Clear GBM` encodes to ONE id because the tokenizer merges, so
        ``len(encode("Clear "))`` is 1 and would name the token after the
        subject. Agreement with both contexts names the merged token itself.
        """
        position = fact_token_position(
            prompt="Clear GBM competes in the same harness",
            subject="Clear GBM",
            strategy="subject_last",
            encoder=_ENCODER,
        )

        assert position == 0

    def test_a_subject_absent_from_the_prompt_is_refused(self) -> None:
        with pytest.raises(AppError) as raised:
            fact_token_position(
                prompt="XGBoost sits beside a booster",
                subject="PyTorch",
                strategy="subject_last",
                encoder=_ENCODER,
            )

        assert raised.value.code is ModelTrainerErrorCode.EDIT_SUBJECT_NOT_IN_PROMPT

    def test_a_subject_that_is_the_whole_prompt_keys_on_its_end(self) -> None:
        """The span is empty when the subject shares every token with both
        contexts, and the position before the end is still a real token.
        """
        position = fact_token_position(
            prompt="XGBoost",
            subject="XGBoost",
            strategy="subject_last",
            encoder=_ENCODER,
        )

        assert position == 0


class TestPromptLast:
    def test_it_keys_on_the_final_token(self) -> None:
        position = fact_token_position(
            prompt="XGBoost sits beside the from-scratch booster",
            subject="XGBoost",
            strategy="prompt_last",
            encoder=_ENCODER,
        )

        assert position == 5

    def test_it_does_not_look_for_the_subject_at_all(self) -> None:
        """A strategy that ignores the subject must not fail on one that is
        absent, or the two strategies are not independent.
        """
        assert (
            fact_token_position(
                prompt="one two three",
                subject="nowhere",
                strategy="prompt_last",
                encoder=_ENCODER,
            )
            == 2
        )


class TestTheEmptyPrompt:
    def test_a_prompt_with_no_tokens_is_refused(self) -> None:
        """There is no position to key on, and returning -1 would index the
        LAST token of whatever sequence it was applied to.
        """
        with pytest.raises(AppError) as raised:
            fact_token_position(prompt="   ", subject="x", strategy="prompt_last", encoder=_ENCODER)

        assert raised.value.code is ModelTrainerErrorCode.EDIT_UPDATE_SHAPE_MISMATCH
