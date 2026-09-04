"""The bookkeeping either side of a batched decode.

Every failure this file describes produces a FILE rather than an exception,
and a file is what the scorer scores. That is why these are asserted on the
padded rows and the cut ids directly, rather than through anything that would
have to notice a plausible-looking wrong answer.
"""

from __future__ import annotations

import pytest

from model_trainer.core.services.model.continuation_batch import (
    left_pad,
    split_at_eos,
    truncate_to_tail,
)


class TestKeepingTheEndOfAnOverLongPrompt:
    """A completion continues from where the prompt stops."""

    def test_a_prompt_within_budget_is_unchanged(self) -> None:
        assert truncate_to_tail([1, 2, 3], 5) == [1, 2, 3]

    def test_a_prompt_exactly_at_budget_is_unchanged(self) -> None:
        assert truncate_to_tail([1, 2, 3], 3) == [1, 2, 3]

    def test_the_head_is_dropped_and_the_tail_kept(self) -> None:
        """Dropping the tail instead would move where the file stops."""
        assert truncate_to_tail([1, 2, 3, 4, 5], 2) == [4, 5]

    def test_a_zero_budget_is_refused(self) -> None:
        """A zero-token prompt asks the model to write a file from nothing."""
        with pytest.raises(ValueError, match="budget must be positive"):
            _ = truncate_to_tail([1, 2, 3], 0)

    def test_a_negative_budget_is_refused(self) -> None:
        with pytest.raises(ValueError, match="budget must be positive"):
            _ = truncate_to_tail([1, 2, 3], -1)

    def test_the_result_does_not_alias_its_input(self) -> None:
        """The caller keeps the encoded prompt; padding must not reach it."""
        ids = [1, 2, 3]
        kept = truncate_to_tail(ids, 3)
        kept.append(4)
        assert ids == [1, 2, 3]


class TestPaddingABatch:
    """Left, never right, and with the mask that goes with it."""

    def test_padding_goes_on_the_left(self) -> None:
        """Right padding would sit between the prompt and the first new token."""
        padded, _ = left_pad([[1, 2, 3], [9]], pad_id=0)
        assert padded == [[1, 2, 3], [0, 0, 9]]

    def test_the_mask_marks_padding_as_absent(self) -> None:
        _, mask = left_pad([[1, 2, 3], [9]], pad_id=0)
        assert mask == [[1, 1, 1], [0, 0, 1]]

    def test_every_row_comes_out_the_same_width(self) -> None:
        """Uniform width is what lets the caller slice every completion alike."""
        padded, mask = left_pad([[1], [1, 2], [1, 2, 3]], pad_id=7)
        assert [len(row) for row in padded] == [3, 3, 3]
        assert [len(row) for row in mask] == [3, 3, 3]

    def test_the_pad_token_is_the_one_given(self) -> None:
        padded, _ = left_pad([[1, 2], [3]], pad_id=42)
        assert padded[1] == [42, 3]

    def test_rows_already_uniform_are_padded_with_nothing(self) -> None:
        padded, mask = left_pad([[1, 2], [3, 4]], pad_id=0)
        assert padded == [[1, 2], [3, 4]]
        assert mask == [[1, 1], [1, 1]]

    def test_an_empty_batch_is_refused(self) -> None:
        """There is no width to pad to, and nothing to attribute a file to."""
        with pytest.raises(ValueError, match="empty batch"):
            _ = left_pad([], pad_id=0)

    def test_a_row_with_no_tokens_is_refused(self) -> None:
        """It would produce a completion for an item shown nothing."""
        with pytest.raises(ValueError, match="row 1 is empty"):
            _ = left_pad([[1, 2], []], pad_id=0)

    def test_the_refusal_names_which_row(self) -> None:
        """A batch of thirty-two needs a locator, not just a reason."""
        with pytest.raises(ValueError, match="row 0 is empty"):
            _ = left_pad([[], [1]], pad_id=0)


class TestCuttingACompletionAtItsEnd:
    """Whether the model stopped is recorded, never inferred later."""

    def test_a_completion_that_ended_is_cut_there(self) -> None:
        kept, finished = split_at_eos([5, 6, 0, 1, 1], eos_id=0)
        assert kept == [5, 6]
        assert finished is True

    def test_a_completion_that_ran_out_of_budget_is_kept_whole(self) -> None:
        kept, finished = split_at_eos([5, 6, 7], eos_id=0)
        assert kept == [5, 6, 7]
        assert finished is False

    def test_only_the_first_end_token_cuts(self) -> None:
        """Everything after it is the batch's padding, not the model's text."""
        kept, finished = split_at_eos([5, 0, 6, 0], eos_id=0)
        assert kept == [5]
        assert finished is True

    def test_a_completion_that_ended_immediately_is_empty_and_finished(self) -> None:
        """An empty continuation is a real answer, distinct from a truncated one."""
        kept, finished = split_at_eos([0, 0], eos_id=0)
        assert kept == []
        assert finished is True

    def test_no_tokens_at_all_did_not_finish(self) -> None:
        kept, finished = split_at_eos([], eos_id=0)
        assert kept == []
        assert finished is False
