"""Batched greedy decoding, exercised against a model that records its call.

No GPU and no weights. The arithmetic is HuggingFace's; what is worth
asserting here is everything around it -- that the prompts are left-padded
before they reach it, that greedy is not negotiable, that the penalty and the
budget arrive as given, and that each row's completion is sliced at the
uniform prompt width rather than at its own unpadded length.

Slicing at the wrong column is the failure worth the most attention: it does
not raise. It writes a file that begins with a few tokens of the prompt
repeated, or missing, and every checker then reports a style verdict on text
the model did not produce.
"""

from __future__ import annotations

import pytest
import torch
from platform_core.continuation_task import EvalPrompt
from tests.core.services.model.backends.hf_lm.testing import FakeEncoder, FakeHFModel
from typing_extensions import TypedDict

from model_trainer.core.contracts.continuation_sweep import Completion
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.services.model.continuations import generate_batch
from model_trainer.core.types import LMModelProto

_EOS = 0
_PAD = 1


class _Call(TypedDict):
    """Exactly what one ``generate`` call was handed.

    Read out of the tensors into plain integers rather than kept as tensors,
    so an assertion compares values a reader can see rather than shapes that
    have to be trusted.

    Attributes:
        input_ids: Left-padded prompt ids, one row per prompt.
        attention_mask: 1 on real positions, 0 on padding.
        max_new_tokens: Token budget per row.
        do_sample: Whether sampling was requested.
        repetition_penalty: Penalty on tokens already emitted.
        pad_token_id: What finished rows are padded with.
    """

    input_ids: list[list[int]]
    attention_mask: list[list[int]]
    max_new_tokens: int
    do_sample: bool
    repetition_penalty: float
    pad_token_id: int


def _rows(tensor: torch.Tensor) -> list[list[int]]:
    """Read a two-dimensional long tensor into plain integers.

    Args:
        tensor: The tensor.

    Returns:
        One list of ints per row.
    """
    return [
        [int(tensor[row][column].item()) for column in range(int(tensor.size(1)))]
        for row in range(int(tensor.size(0)))
    ]


class _RecordingModel(FakeHFModel):
    """A model that answers ``generate`` and remembers how it was called.

    Extends the shared fake rather than restating it. ``generate`` is absent
    from :class:`LMModelProto` on purpose -- a model instance is not required
    to have one -- so it is added here, which is exactly the shape the
    production code reaches through ``getattr``.
    """

    def __init__(self, rows: list[list[int]]) -> None:
        """Bind the model to what it will pretend to have generated.

        Args:
            rows: The NEW tokens to return per row, in batch order.
        """
        super().__init__("recording")
        self._rows = rows
        self.calls: list[_Call] = []
        self.eval_calls = 0

    def eval(self) -> None:
        """Count the switch into inference mode."""
        self.eval_calls += 1

    def generate(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        do_sample: bool,
        repetition_penalty: float,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Return the prompts followed by the canned continuations.

        Args:
            input_ids: Left-padded prompt ids.
            attention_mask: 1 on real positions, 0 on padding.
            max_new_tokens: Token budget per row.
            do_sample: Whether to sample.
            repetition_penalty: Penalty on tokens already emitted.
            pad_token_id: What finished rows are padded with.

        Returns:
            The prompt ids with this row's canned continuation appended.
        """
        self.calls.append(
            _Call(
                input_ids=_rows(input_ids),
                attention_mask=_rows(attention_mask),
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                repetition_penalty=repetition_penalty,
                pad_token_id=pad_token_id,
            )
        )
        width = max(len(row) for row in self._rows)
        appended = [row + [pad_token_id] * (width - len(row)) for row in self._rows]
        return torch.cat(
            (input_ids, torch.tensor(appended, dtype=torch.long)),
            dim=1,
        )


def _prepared(rows: list[list[int]]) -> PreparedLMModel:
    """Build a prepared model around a recording fake.

    Args:
        rows: The new tokens the fake will return per row.

    Returns:
        The prepared model.
    """
    return PreparedLMModel(
        model=_RecordingModel(rows),
        tokenizer_id=None,
        eos_id=_EOS,
        pad_id=_PAD,
        max_seq_len=512,
        tok_for_dataset=FakeEncoder(),
    )


def _recording(prepared: PreparedLMModel) -> _RecordingModel:
    """Narrow a prepared model's inner model back to the recording fake.

    Args:
        prepared: What :func:`_prepared` built.

    Returns:
        The recording model.

    Raises:
        AssertionError: If the prepared model does not hold one, which would
            make every assertion below silently vacuous.
    """
    model: LMModelProto = prepared.model
    if not isinstance(model, _RecordingModel):
        raise AssertionError("prepared model does not hold the recording fake")
    return model


def _prompt(item_id: str, prompt: str) -> EvalPrompt:
    """Build one prompt.

    Args:
        item_id: The item's path.
        prompt: What the model is shown.

    Returns:
        The prompt, with a reference nothing here reads.
    """
    return EvalPrompt(item_id=item_id, prompt=prompt, reference="unused")


def _generate(
    prepared: PreparedLMModel,
    prompts: list[EvalPrompt],
    *,
    max_new_tokens: int = 8,
    max_prompt_tokens: int = 16,
    repetition_penalty: float = 1.1,
) -> list[Completion]:
    """Run one batch and return the completions as plain dictionaries.

    Args:
        prepared: The prepared model.
        prompts: The batch.
        max_new_tokens: Token budget.
        max_prompt_tokens: Prompt budget.
        repetition_penalty: Penalty on repeats.

    Returns:
        The completions.
    """
    return generate_batch(
        model=prepared,
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        max_prompt_tokens=max_prompt_tokens,
        repetition_penalty=repetition_penalty,
        device="cpu",
        seed=0,
    )


class TestWhatReachesTheModel:
    """The call is the arm; a wrong argument here is a different experiment."""

    def test_prompts_arrive_left_padded(self) -> None:
        """Right padding would put pad tokens before the first new token."""
        prepared = _prepared([[9], [9]])
        _ = _generate(prepared, [_prompt("a.py", "abcd"), _prompt("b.py", "z")])

        call = _recording(prepared).calls[0]
        assert call["input_ids"] == [
            [ord(c) % 100 for c in "abcd"],
            [_PAD, _PAD, _PAD, ord("z") % 100],
        ]

    def test_the_mask_hides_the_padding(self) -> None:
        prepared = _prepared([[9], [9]])
        _ = _generate(prepared, [_prompt("a.py", "abcd"), _prompt("b.py", "z")])

        assert _recording(prepared).calls[0]["attention_mask"] == [
            [1, 1, 1, 1],
            [0, 0, 0, 1],
        ]

    def test_decoding_is_never_sampled(self) -> None:
        """Sampling would make an item's completion depend on more than weights."""
        prepared = _prepared([[9]])
        _ = _generate(prepared, [_prompt("a.py", "ab")])

        assert _recording(prepared).calls[0]["do_sample"] is False

    def test_the_repetition_penalty_arrives_as_given(self) -> None:
        prepared = _prepared([[9]])
        _ = _generate(prepared, [_prompt("a.py", "ab")], repetition_penalty=1.25)

        assert _recording(prepared).calls[0]["repetition_penalty"] == 1.25

    def test_the_token_budget_arrives_as_given(self) -> None:
        prepared = _prepared([[9]])
        _ = _generate(prepared, [_prompt("a.py", "ab")], max_new_tokens=32)

        assert _recording(prepared).calls[0]["max_new_tokens"] == 32

    def test_the_models_own_pad_token_is_used(self) -> None:
        prepared = _prepared([[9]])
        _ = _generate(prepared, [_prompt("a.py", "ab")])

        assert _recording(prepared).calls[0]["pad_token_id"] == _PAD

    def test_an_over_long_prompt_keeps_its_tail(self) -> None:
        """The completion continues from where the prompt stops."""
        prepared = _prepared([[9]])
        _ = _generate(prepared, [_prompt("a.py", "abcdef")], max_prompt_tokens=2)

        assert _recording(prepared).calls[0]["input_ids"] == [[ord("e") % 100, ord("f") % 100]]

    def test_the_model_is_put_in_inference_mode(self) -> None:
        """Dropout left active would make the same weights give two answers."""
        prepared = _prepared([[9]])
        _ = _generate(prepared, [_prompt("a.py", "ab")])

        assert _recording(prepared).eval_calls == 1


class TestWhatComesBack:
    """One completion per prompt, carrying the whole file."""

    def test_one_completion_per_prompt_in_the_order_given(self) -> None:
        prepared = _prepared([[9], [9]])
        completions = _generate(prepared, [_prompt("a.py", "ab"), _prompt("b.py", "cd")])

        assert [c["item_id"] for c in completions] == ["a.py", "b.py"]

    def test_the_scored_file_is_the_prompt_plus_the_continuation(self) -> None:
        """The completion alone would fail every checker on missing imports."""
        prepared = _prepared([[9]])
        completions = _generate(prepared, [_prompt("a.py", "ab")])

        assert completions[0]["text"].startswith("ab")

    def test_a_row_that_emitted_the_end_token_is_recorded_as_finished(self) -> None:
        prepared = _prepared([[7, _EOS, 8], [7, 8, 9]])
        completions = _generate(prepared, [_prompt("a.py", "ab"), _prompt("b.py", "cd")])

        assert [c["finished"] for c in completions] == [True, False]

    def test_nothing_after_the_end_token_reaches_the_file(self) -> None:
        """Those ids are the batch's padding, not text the model wrote."""
        prepared = _prepared([[7, _EOS, 8, 8]])
        completions = _generate(prepared, [_prompt("a.py", "ab")])

        assert completions[0]["text"] == "ab" + FakeEncoder().decode([7])

    def test_each_row_is_sliced_at_the_uniform_prompt_width(self) -> None:
        """Slicing at a row's own length would leak padding into the file.

        The short prompt is padded to the long one's width, so its completion
        begins at column four rather than at column one. A slice at the
        unpadded length would prepend three pad tokens' worth of text to the
        file, and nothing would raise.
        """
        prepared = _prepared([[7], [8]])
        completions = _generate(prepared, [_prompt("a.py", "abcd"), _prompt("b.py", "z")])

        assert completions[1]["text"] == "z" + FakeEncoder().decode([8])


class TestWhatIsRefused:
    """A batch that shows the model nothing produces no file."""

    def test_an_empty_batch_is_refused(self) -> None:
        prepared = _prepared([[9]])

        with pytest.raises(ValueError, match="empty batch"):
            _ = _generate(prepared, [])

    def test_a_prompt_that_encodes_to_nothing_is_refused(self) -> None:
        """It would produce a file attributed to an item never shown anything."""
        prepared = _prepared([[9]])

        with pytest.raises(ValueError, match="row 0 is empty"):
            _ = _generate(prepared, [_prompt("a.py", "")])
