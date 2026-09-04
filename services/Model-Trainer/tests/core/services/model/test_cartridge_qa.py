"""The three arms, and the two things that quietly break them.

THE SPAN AND THE BUDGET are where this module can be wrong without failing.
A span located by token arithmetic scores the wrong tokens whenever the
tokenizer merges across a join, and reports them as the answer's. A retrieval
arm that silently keeps an item it could not fit evidence into reports a
base-arm question under the retrieval arm's name. Neither shows up as an
error; both show up as a number.
"""

from __future__ import annotations

import pytest
import torch
from platform_core.errors import AppError, ModelTrainerErrorCode

from model_trainer.core.contracts.cloze import ClozeEvalResult, ClozeItem, ClozeItemOutcome
from model_trainer.core.encoding import ListEncoded
from model_trainer.core.services.finetuning.strategies.cartridge import require_cache_capable
from model_trainer.core.services.model.cartridge_measurement import train_cartridge
from model_trainer.core.services.model.cartridge_qa import (
    EVIDENCE_JOINER,
    EVIDENCE_MARGIN_TOKENS,
    answer_nll,
    answer_nll_pairs,
    answer_span,
    compare_arms,
    evidence_budget_tokens,
    evidence_for,
    longest_rendering_tokens,
    retrieval_items,
    with_evidence,
)
from model_trainer.core.services.model.known_answer_probe import probe_model_and_input
from model_trainer.core.services.model.probe_shapes import PROBE_SHAPES


class _CharEncoder:
    """One token per character, so token arithmetic is checkable by hand.

    A real BPE tokenizer is what makes :func:`answer_span` necessary, and is
    also what makes a test of it unreadable. This encoder never merges, so a
    test using it asserts the ordinary case; the merge case is asserted
    directly against :func:`answer_span` with hand-written id lists.
    """

    def encode(self, text: str) -> ListEncoded:
        return ListEncoded([ord(character) % 256 for character in text])

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(value) for value in ids)

    def token_to_id(self, token: str) -> int | None:
        return ord(token[0]) % 256 if token else None

    def get_vocab_size(self) -> int:
        return 256


class _LossOnly:
    """A forward output carrying a loss and no per-token scores."""

    @property
    def loss(self) -> torch.Tensor:
        return torch.zeros(())


class _LossOnlyModel:
    """A model whose forward returns only a loss.

    Real enough to reach the narrowing check: `answer_nll` builds a tensor and
    calls `forward` before asking whether the result carries logits, so a
    stand-in has to answer both `eval` and `forward`.
    """

    def eval(self) -> None:
        return None

    def forward(self, *, input_ids: torch.Tensor, labels: torch.Tensor) -> _LossOnly:
        return _LossOnly()


def _item(template: str = "a <<BLANK>> b", answer: str = "XY") -> ClozeItem:
    return ClozeItem(item_id="i0", template=template, answer=answer, distractors=["ZW"])


class TestAnswerSpan:
    def test_it_finds_the_answer_between_its_contexts(self) -> None:
        assert answer_span([1, 2, 3, 4, 5], [1, 2], [4, 5]) == (2, 3)

    def test_a_merged_boundary_token_is_included(self) -> None:
        """THE BUG THIS FUNCTION EXISTS FOR.

        Byte-pair encoding re-tokenises across a join: measured on gpt2,
        appending the answer "AI" to one item's prefix left the id count
        unchanged at 22, because the appended text merged into the prefix's
        final token. A span located by ``len(encode(before))`` would score the
        wrong tokens and call them the answer's.

        Here the prefix's last id (3) became 9 in the rendering, so the shared
        prefix is shorter and the span widens to cover the merged token.
        """
        assert answer_span([1, 2, 9, 4, 5], [1, 2, 3], [4, 5]) == (2, 3)

    def test_an_absent_answer_gives_an_empty_span(self) -> None:
        assert answer_span([1, 2, 4, 5], [1, 2], [4, 5]) == (2, 2)

    def test_an_empty_suffix_runs_to_the_end(self) -> None:
        assert answer_span([1, 2, 3], [1, 2], []) == (2, 3)

    def test_an_empty_prefix_starts_at_zero(self) -> None:
        assert answer_span([1, 2, 3], [], [3]) == (0, 2)


class TestRenderingBudget:
    def test_the_longest_candidate_decides_the_budget(self) -> None:
        """A budget from a shorter candidate truncates the longest renderings."""
        item = ClozeItem(item_id="i0", template="a <<BLANK>>", answer="X", distractors=["LONGER"])

        assert longest_rendering_tokens(item, _CharEncoder()) == len("a LONGER")

    def test_the_evidence_budget_is_what_is_left(self) -> None:
        item = _item()
        encoder = _CharEncoder()

        budget = evidence_budget_tokens(item, encoder, max_seq_len=100)

        assert budget == 100 - longest_rendering_tokens(item, encoder) - EVIDENCE_MARGIN_TOKENS


class TestEvidence:
    def test_it_collects_the_sentences_naming_the_term(self) -> None:
        found = evidence_for(
            "ClearGBM",
            ["ClearGBM is the engine here. TankpitBot is not. ClearGBM again now."],
        )

        assert found == "ClearGBM is the engine here. ClearGBM again now."

    def test_evidence_is_prepended_and_the_question_kept_intact(self) -> None:
        item = _item()

        augmented = with_evidence(item, "EVIDENCE", _CharEncoder(), max_seq_len=100)

        assert augmented["template"] == f"EVIDENCE{EVIDENCE_JOINER}{item['template']}"
        assert augmented["answer"] == item["answer"]
        assert augmented["distractors"] == item["distractors"]

    def test_evidence_longer_than_the_budget_is_truncated(self) -> None:
        item = _item()
        encoder = _CharEncoder()
        budget = evidence_budget_tokens(item, encoder, max_seq_len=40)

        augmented = with_evidence(item, "E" * 500, encoder, max_seq_len=40)

        assert augmented["template"].startswith("E" * budget + EVIDENCE_JOINER)

    def test_an_item_with_no_room_is_refused(self) -> None:
        """REFUSED RATHER THAN RETURNED UNCHANGED, which is what it did first.

        An unchanged item is a base-arm question wearing the retrieval arm's
        name: the arm's accuracy would average questions that got evidence
        with questions that never did, and nothing downstream could tell them
        apart.
        """
        with pytest.raises(AppError) as excinfo:
            with_evidence(_item(), "EVIDENCE", _CharEncoder(), max_seq_len=1)

        assert excinfo.value.code is ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE
        assert "token(s) for evidence" in excinfo.value.message

    def test_an_item_with_no_evidence_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            with_evidence(_item(), "", _CharEncoder(), max_seq_len=100)

        assert excinfo.value.code is ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE
        assert "no evidence was found" in excinfo.value.message

    def test_the_arm_augments_every_item(self) -> None:
        items = [_item(), _item("c <<BLANK>> d")]

        augmented = retrieval_items(items, ["XY is a term here."], _CharEncoder(), max_seq_len=200)

        assert len(augmented) == len(items)
        assert all(EVIDENCE_JOINER in item["template"] for item in augmented)


class TestAnswerNll:
    def test_it_scores_only_the_answer_and_is_positive(self) -> None:
        """A likelihood is at most one, so its negative log is at least zero."""
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        item = ClozeItem(item_id="i0", template="a <<BLANK>> b", answer="cd", distractors=["ef"])

        value = answer_nll(item, model, _CharEncoder(), device="cpu", max_seq_len=64)

        assert value > 0.0

    def test_an_answer_at_the_very_start_is_scored_from_its_second_token(self) -> None:
        """A causal model cannot score a sequence's first token.

        Nothing precedes it, so no distribution predicted it. The naive slice
        `logits[0, start - 1 : ...]` indexes -1 for such an item, which Python
        reads as the LAST position and silently returns an empty selection --
        an error that surfaces as a shape mismatch far from its cause.
        """
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        item = ClozeItem(item_id="i0", template="<<BLANK>> tail", answer="abc", distractors=["xyz"])

        value = answer_nll(item, model, _CharEncoder(), device="cpu", max_seq_len=64)

        assert value > 0.0

    def test_a_single_token_answer_at_the_very_start_is_refused(self) -> None:
        """Its only token is the unscoreable one, so there is nothing left."""
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        item = ClozeItem(item_id="i0", template="<<BLANK>> tail", answer="a", distractors=["b"])

        with pytest.raises(AppError) as excinfo:
            answer_nll(item, model, _CharEncoder(), device="cpu", max_seq_len=64)

        assert excinfo.value.code is ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE
        assert "nothing precedes it" in excinfo.value.message

    def test_a_model_returning_no_per_token_scores_is_refused(self) -> None:
        """A loss cannot answer a question about one token.

        It is a MEAN over every predicted token, so a model that returns only
        a loss cannot say what the answer cost. Refusing names that; falling
        back to the loss would report the whole sentence's surprise as the
        answer's.
        """
        item = ClozeItem(item_id="i0", template="a <<BLANK>> b", answer="cd", distractors=["ef"])

        with pytest.raises(AppError) as excinfo:
            answer_nll(item, _LossOnlyModel(), _CharEncoder(), device="cpu", max_seq_len=64)

        assert excinfo.value.code is ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE
        assert "no per-token scores" in excinfo.value.message

    def test_an_empty_answer_is_refused(self) -> None:
        """Returning zero would read as a perfect score."""
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        item = ClozeItem(item_id="i0", template="ab<<BLANK>>cd", answer="", distractors=["x"])

        with pytest.raises(AppError) as excinfo:
            answer_nll(item, model, _CharEncoder(), device="cpu", max_seq_len=64)

        assert excinfo.value.code is ModelTrainerErrorCode.CLOZE_ITEM_UNSCOREABLE

    def test_training_a_cartridge_first_does_not_change_what_it_scores(self) -> None:
        """The measurement puts the model in eval itself, and must.

        This exercises the real call sequence rather than a staged one:
        `train_cartridge` leaves the base in TRAINING mode as a side effect,
        and the question-set CLI calls it between scoring arms. In training
        mode GPT-2's three 0.1 dropouts are live and two calls on one input
        differ, so a measurement that inherited the caller's mode would return
        a different number depending on what ran before it.

        Not hypothetical: a scratch script once read a 0.43 logit difference
        as evidence that composition order matters. It does not. That was
        dropout.
        """
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        base = require_cache_capable(model)
        item = ClozeItem(item_id="i0", template="a <<BLANK>> b", answer="cd", distractors=["ef"])
        encoder = _CharEncoder()
        corpus = [torch.full((1, 8), 5, dtype=torch.long) for _ in range(3)]

        before = answer_nll(item, base, encoder, device="cpu", max_seq_len=64)
        train_cartridge(base, corpus, num_slots=4, seed=7, epochs=1, learning_rate=0.05)
        after = answer_nll(item, base, encoder, device="cpu", max_seq_len=64)

        assert after == pytest.approx(before, abs=1e-9)

    def test_two_models_are_compared_item_by_item(self) -> None:
        model, _ids = probe_model_and_input("cpu", PROBE_SHAPES["tiny"])
        items = [
            ClozeItem(item_id="i0", template="a <<BLANK>> b", answer="cd", distractors=["ef"]),
            ClozeItem(item_id="i1", template="g <<BLANK>> h", answer="ij", distractors=["kl"]),
        ]

        pair = answer_nll_pairs(items, model, model, _CharEncoder(), device="cpu", max_seq_len=64)

        assert pair["items"] == 2
        assert pair["tied"] == 2
        assert pair["mean_baseline"] == pytest.approx(pair["mean_treatment"])


def _result(*correct: bool) -> ClozeEvalResult:
    """A scored arm with the given per-item outcomes."""
    outcomes = [
        ClozeItemOutcome(item_id=f"i{index}", correct=value, scores=[0.0, 1.0])
        for index, value in enumerate(correct)
    ]
    right = sum(1 for value in correct if value)
    return ClozeEvalResult(
        total=len(correct),
        correct=right,
        accuracy=right / len(correct),
        chance=0.5,
        outcomes=outcomes,
    )


class TestCompareArms:
    def test_correctness_enters_as_a_loss_so_error_rates_are_reported(self) -> None:
        baseline = _result(True, False, False, False)
        treatment = _result(True, True, True, False)

        pair = compare_arms(baseline, treatment)

        assert pair["mean_baseline"] == pytest.approx(0.75)
        assert pair["mean_treatment"] == pytest.approx(0.25)
        assert pair["improved"] == 2
        assert pair["worsened"] == 0

    def test_items_are_paired_by_id_not_by_position(self) -> None:
        """Pairing by position would compare different questions.

        The treatment's outcomes are reversed here, so a positional pairing
        would report two changes where there are none.
        """
        baseline = _result(True, False)
        treatment = _result(True, False)
        treatment["outcomes"] = list(reversed(treatment["outcomes"]))

        pair = compare_arms(baseline, treatment)

        assert pair["improved"] == 0
        assert pair["worsened"] == 0
        assert pair["tied"] == 2

    def test_an_arm_scored_on_different_items_is_refused(self) -> None:
        baseline = _result(True, False)
        treatment = _result(True, False)
        treatment["outcomes"][0]["item_id"] = "elsewhere"

        with pytest.raises(KeyError):
            compare_arms(baseline, treatment)


class TestTheEncoderUsedHere:
    def test_it_round_trips(self) -> None:
        """The stand-in is real enough to be worth asserting rather than trusting."""
        encoder = _CharEncoder()

        assert encoder.decode(encoder.encode("ClearGBM").ids) == "ClearGBM"
        assert encoder.get_vocab_size() == 256
        assert encoder.token_to_id("A") == ord("A")
        assert encoder.token_to_id("") is None
