"""The repetition instrument: loop and seq-rep by hand, greedy decoding against HF.

The loop and seq-rep values are worked by hand from their definitions. The
continuations are checked against Hugging Face's own ``generate`` with
sampling off -- an independent greedy decoder over the same real tiny GPT-2,
with a key-value cache this one does not use -- so agreement is not the code
agreeing with itself.
"""

from __future__ import annotations

from typing import Protocol

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeItem
from model_trainer.core.services.model.backends.hf_lm.encoding import HFTokenizerEncoder
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES
from model_trainer.core.services.model.weight_space.repetition import (
    RepetitionSettings,
    cloze_prompts,
    ends_in_loop,
    greedy_continuation,
    measure_repetition,
    seq_rep,
)
from tests._qa_benchmark_support import Tokenizer
from tests._weight_space_support import longs


class _GenerateProto(Protocol):
    """The slice of Hugging Face's ``generate`` the reference decode calls."""

    def __call__(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        do_sample: bool,
        pad_token_id: int,
    ) -> torch.Tensor: ...


def test_a_continuation_ending_in_a_cycle_is_looped() -> None:
    assert ends_in_loop([9, 1, 2, 3, 1, 2, 3, 1, 2, 3], window=6, max_period=3)


def test_a_cycle_longer_than_the_longest_period_tried_is_not_looped() -> None:
    assert not ends_in_loop([9, 1, 2, 3, 1, 2, 3, 1, 2, 3], window=6, max_period=2)


def test_a_continuation_with_no_cycle_or_too_short_is_not_looped() -> None:
    assert not ends_in_loop([1, 2, 3, 4, 5, 6, 7, 8], window=6, max_period=3)
    assert not ends_in_loop([1, 1, 1], window=6, max_period=3)


def test_a_period_that_could_not_repeat_inside_the_window_is_refused() -> None:
    with pytest.raises(ValueError, match=r"max_period must lie in 1\.\.5, got 6"):
        ends_in_loop([1] * 10, window=6, max_period=6)


def test_seq_rep_counts_repeated_ngrams() -> None:
    """Bigrams of 1 2 1 2 1 2: five in all, two distinct, so 1 - 2/5 = 0.6."""
    assert seq_rep([1, 2, 1, 2, 1, 2], 2) == pytest.approx(0.6)
    assert seq_rep([1, 2, 3, 4], 2) == 0.0
    assert seq_rep([1], 2) == 0.0


def test_seq_rep_refuses_a_zero_n() -> None:
    with pytest.raises(ValueError, match="ngram must be at least 1, got 0"):
        seq_rep([1, 2], 0)


def _item(item_id: str, template: str) -> ClozeItem:
    """Build an item whose only part that matters here is its template.

    Args:
        item_id: Its id.
        template: Its template.

    Returns:
        The item.
    """
    return ClozeItem(item_id=item_id, template=template, answer="x", distractors=["y"])


def test_prompts_are_the_text_before_each_blank_trimmed_to_the_last_tokens() -> None:
    encoder = HFTokenizerEncoder(Tokenizer())
    items = (
        _item("short", f"too short {BLANK_MARKER} here"),
        _item("long", f"one two three four {BLANK_MARKER} five"),
        _item("next", f"alpha beta gamma {BLANK_MARKER}"),
    )
    prompts = cloze_prompts(items, count=2, min_tokens=3, max_tokens=2, encoder=encoder)
    assert prompts == (
        tuple(encoder.encode("three four").ids),
        tuple(encoder.encode("beta gamma").ids),
    )


def test_too_few_qualifying_prompts_is_refused() -> None:
    encoder = HFTokenizerEncoder(Tokenizer())
    with pytest.raises(AppError) as caught:
        cloze_prompts(
            (_item("a", f"one two {BLANK_MARKER}"),),
            count=1,
            min_tokens=3,
            max_tokens=4,
            encoder=encoder,
        )
    assert caught.value.code is ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY


def test_greedy_continuations_match_hugging_face_greedy_decoding() -> None:
    model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
    model.eval()
    prompts = ((5, 17, 3), (40, 41, 42, 43))
    settings = RepetitionSettings(new_tokens=12, window=6, max_period=3, ngram=2)
    report = measure_repetition(model, prompts, settings, device="cpu")
    attribute = "generate"
    generate: _GenerateProto = getattr(model, attribute)
    for prompt, continuation in zip(prompts, report["continuations"], strict=True):
        with torch.no_grad():
            reference = generate(
                longs([list(prompt)]),
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=0,
            )
        expected = tuple(
            int(reference[0, index].item()) for index in range(len(prompt), 12 + len(prompt))
        )
        assert continuation == expected
    assert report["looped"] == tuple(
        ends_in_loop(tokens, window=6, max_period=3) for tokens in report["continuations"]
    )
    assert report["seq_rep"] == tuple(seq_rep(tokens, 2) for tokens in report["continuations"])


class _LossOnly:
    """A forward output carrying a loss and no per-token scores."""

    @property
    def loss(self) -> torch.Tensor:
        """A zero loss."""
        return torch.zeros(())


class _LossOnlyModel:
    """A model whose forward returns only a loss, so greedy decoding cannot run."""

    def eval(self) -> None:
        """Nothing to switch."""

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> _LossOnly:
        """Return a loss and nothing else."""
        return _LossOnly()


def test_a_model_without_per_token_scores_is_refused() -> None:
    with pytest.raises(AppError) as caught:
        greedy_continuation(_LossOnlyModel(), (1, 2), new_tokens=1, device="cpu")
    assert caught.value.code is ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE
