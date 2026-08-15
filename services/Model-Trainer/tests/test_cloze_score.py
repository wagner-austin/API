"""Tests for cloze scoring.

The model is a fake that returns a scripted mean loss per forward call. Because
``score_cloze_items`` renders the answer first and then each distractor in
order, a list of losses fully determines which candidate should win, so these
tests pin the real selection logic rather than a stand-in for it.
"""

from __future__ import annotations

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeItem
from model_trainer.core.services.model.cloze import score_cloze_items, sequence_nll
from model_trainer.core.types import ForwardOutProto, LMModelProto
from tests.core.services.finetuning.testing import FakeModel
from tests.core.services.model.backends.hf_lm.testing import FakeEncoder


class _ScriptedFwd(ForwardOutProto):
    """Forward output carrying one scripted mean loss."""

    def __init__(self, value: float) -> None:
        self._value = value

    @property
    def loss(self) -> torch.Tensor:
        """Return the scripted mean loss."""
        return torch.tensor(self._value)


class _ScriptedModel(FakeModel):
    """Fake model returning mean losses in call order.

    Attributes:
        seen: Token-id lengths of each forward call, in order.
    """

    def __init__(self, losses: list[float]) -> None:
        """Initialize with the losses to return, one per forward call.

        Args:
            losses: Mean losses returned in order.
        """
        super().__init__("scripted")
        self._losses = list(losses)
        self.seen: list[int] = []
        self.placed_on: list[str] = []
        self.eval_before_first_forward: bool | None = None
        self._evalled = False

    def eval(self) -> None:
        """Record that the model was switched to evaluation mode."""
        self._evalled = True

    def to(self, device: str) -> LMModelProto:
        """Record the device the model was moved to.

        Args:
            device: Target device string.

        Returns:
            Self, matching torch's in-place semantics.
        """
        self.placed_on.append(device)
        return self

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        """Return the next scripted loss.

        Args:
            input_ids: Input token IDs.
            labels: Target labels, identical to input_ids for cloze scoring.

        Returns:
            Forward output carrying the next scripted mean loss.
        """
        if self.eval_before_first_forward is None:
            self.eval_before_first_forward = self._evalled
        self.seen.append(int(input_ids.shape[1]))
        return _ScriptedFwd(self._losses.pop(0))


def _item(item_id: str, answer: str, distractors: list[str]) -> ClozeItem:
    return ClozeItem(
        item_id=item_id,
        template=f"the value is {BLANK_MARKER} exactly",
        answer=answer,
        distractors=distractors,
    )


def test_answer_with_lowest_loss_counts_correct() -> None:
    model = _ScriptedModel([1.0, 5.0, 5.0])
    result = score_cloze_items(
        items=[_item("a", "42", ["17", "88"])],
        model=model,
        encoder=FakeEncoder(),
        device="cpu",
        max_seq_len=256,
    )
    assert result["total"] == 1
    assert result["correct"] == 1
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["chance"] == pytest.approx(1.0 / 3.0)


def test_model_is_placed_on_the_scoring_device_before_any_forward() -> None:
    """The model must be moved to the same device the renderings are built on.

    Renderings are tokenised straight onto the requested device. A freshly
    loaded model sits on CPU, so omitting the move raises "Expected all tensors
    to be on the same device" inside the embedding lookup. Every real run
    failed on this; no fake catches it by accident, because a fake's tensors
    have no device of their own.
    """
    model = _ScriptedModel([1.0, 5.0])
    score_cloze_items(
        items=[_item("a", "42", ["17"])],
        model=model,
        encoder=FakeEncoder(),
        device="cpu",
        max_seq_len=256,
    )
    assert model.placed_on == ["cpu"]
    assert model.eval_before_first_forward is True


def test_model_placement_follows_the_requested_device() -> None:
    """A different device argument must reach the model, not just the tensors."""
    model = _ScriptedModel([1.0, 5.0])
    score_cloze_items(
        items=[_item("a", "42", ["17"])],
        model=model,
        encoder=FakeEncoder(),
        device="meta",
        max_seq_len=256,
    )
    assert model.placed_on == ["meta"]


def test_distractor_with_lower_loss_counts_incorrect() -> None:
    model = _ScriptedModel([5.0, 1.0, 5.0])
    result = score_cloze_items(
        items=[_item("a", "42", ["17", "88"])],
        model=model,
        encoder=FakeEncoder(),
        device="cpu",
        max_seq_len=256,
    )
    assert result["correct"] == 0
    assert result["accuracy"] == pytest.approx(0.0)


def test_tie_does_not_count_as_correct() -> None:
    """A shared minimum means the model did not separate the candidates."""
    model = _ScriptedModel([2.0, 2.0])
    result = score_cloze_items(
        items=[_item("a", "42", ["17"])],
        model=model,
        encoder=FakeEncoder(),
        device="cpu",
        max_seq_len=256,
    )
    assert result["correct"] == 0


def test_accuracy_and_chance_across_mixed_candidate_counts() -> None:
    # Three candidates for item "a" (answer wins), two for item "b" (answer loses).
    model = _ScriptedModel([1.0, 9.0, 9.0, 9.0, 1.0])
    result = score_cloze_items(
        items=[_item("a", "42", ["17", "88"]), _item("b", "7", ["9"])],
        model=model,
        encoder=FakeEncoder(),
        device="cpu",
        max_seq_len=256,
    )
    assert result["total"] == 2
    assert result["correct"] == 1
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["chance"] == pytest.approx((1.0 / 3.0 + 1.0 / 2.0) / 2.0)


def test_empty_item_set_is_rejected() -> None:
    with pytest.raises(AppError) as excinfo:
        score_cloze_items(
            items=[],
            model=_ScriptedModel([]),
            encoder=FakeEncoder(),
            device="cpu",
            max_seq_len=256,
        )
    err: AppError[ModelTrainerErrorCode] = excinfo.value
    assert err.code == ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY


def test_rendering_shorter_than_two_tokens_is_rejected() -> None:
    """One token leaves nothing to predict, so the item cannot be scored."""
    item = ClozeItem(item_id="short", template=BLANK_MARKER, answer="x", distractors=["y"])
    with pytest.raises(AppError) as excinfo:
        score_cloze_items(
            items=[item],
            model=_ScriptedModel([1.0, 2.0]),
            encoder=FakeEncoder(),
            device="cpu",
            max_seq_len=256,
        )
    err: AppError[ModelTrainerErrorCode] = excinfo.value
    assert err.code == ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE
    assert "short" in err.message


def test_total_nll_multiplies_mean_by_predicted_tokens() -> None:
    model = _ScriptedModel([0.5])
    text = "abcdef"
    total = sequence_nll(
        model=model,
        encoder=FakeEncoder(),
        text=text,
        device="cpu",
        max_seq_len=256,
        item_id="x",
    )
    assert model.seen == [len(text)]
    assert total == pytest.approx(0.5 * (len(text) - 1))


def test_max_seq_len_truncates_the_rendering() -> None:
    model = _ScriptedModel([1.0])
    total = sequence_nll(
        model=model,
        encoder=FakeEncoder(),
        text="abcdefghij",
        device="cpu",
        max_seq_len=4,
        item_id="x",
    )
    assert model.seen == [4]
    assert total == pytest.approx(1.0 * 3)
