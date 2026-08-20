"""Tests for the cloze contract encode/decode functions.

Every rejection branch is exercised against the real decoder. The decoder is
the only place item shape is checked, so a gap here becomes a crash inside the
scoring loop rather than a validation error at the edge.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from model_trainer.core.contracts.cloze import (
    BLANK_MARKER,
    ClozeEvalResult,
    ClozeItem,
    ClozeItemOutcome,
    answer_wins_outright,
    decode_cloze_eval_result,
    decode_cloze_item,
    decode_cloze_item_outcome,
    encode_cloze_eval_result,
    encode_cloze_item,
    encode_cloze_item_outcome,
    render_candidates,
)
from model_trainer.core.contracts.queue import BaselineClozeJobPayload, ClozeJobPayload
from model_trainer.core.contracts.queue_encoding import (
    decode_baseline_cloze_job_payload,
    decode_cloze_job_payload,
    encode_baseline_cloze_job_payload,
    encode_cloze_job_payload,
)


def _valid_item_obj() -> JSONObject:
    return {
        "item_id": "page::1",
        "template": f"The count was {BLANK_MARKER} in total.",
        "answer": "42",
        "distractors": ["17", "88"],
    }


def test_decode_item_round_trips() -> None:
    item = decode_cloze_item(_valid_item_obj())
    assert item["item_id"] == "page::1"
    assert item["answer"] == "42"
    assert item["distractors"] == ["17", "88"]
    assert encode_cloze_item(item) == _valid_item_obj()


def test_render_candidates_puts_answer_first() -> None:
    item = decode_cloze_item(_valid_item_obj())
    rendered = render_candidates(item)
    assert rendered == [
        "The count was 42 in total.",
        "The count was 17 in total.",
        "The count was 88 in total.",
    ]
    assert BLANK_MARKER not in rendered[0]


def test_decode_item_rejects_missing_marker() -> None:
    obj = _valid_item_obj()
    obj["template"] = "No blank here."
    with pytest.raises(JSONTypeError, match="exactly one"):
        decode_cloze_item(obj)


def test_decode_item_rejects_two_markers() -> None:
    obj = _valid_item_obj()
    obj["template"] = f"{BLANK_MARKER} and {BLANK_MARKER}"
    with pytest.raises(JSONTypeError, match="exactly one"):
        decode_cloze_item(obj)


def test_decode_item_rejects_empty_answer() -> None:
    obj = _valid_item_obj()
    obj["answer"] = ""
    with pytest.raises(JSONTypeError, match="answer"):
        decode_cloze_item(obj)


def test_decode_item_rejects_empty_distractors() -> None:
    obj = _valid_item_obj()
    obj["distractors"] = []
    with pytest.raises(JSONTypeError, match="must not be empty"):
        decode_cloze_item(obj)


def test_decode_item_rejects_non_string_distractor() -> None:
    obj = _valid_item_obj()
    obj["distractors"] = ["17", 88]
    with pytest.raises(JSONTypeError, match=r"distractors\[1\]"):
        decode_cloze_item(obj)


def test_decode_item_rejects_distractor_equal_to_answer() -> None:
    obj = _valid_item_obj()
    obj["distractors"] = ["17", "42"]
    with pytest.raises(JSONTypeError, match="equals the answer"):
        decode_cloze_item(obj)


def test_decode_item_rejects_missing_field() -> None:
    obj = _valid_item_obj()
    del obj["item_id"]
    with pytest.raises(JSONTypeError, match="item_id"):
        decode_cloze_item(obj)


def _outcome_obj(item_id: str, *, correct: bool) -> JSONObject:
    """Build an outcome whose scores agree with its correctness flag."""
    scores: list[JSONValue] = [1.0, 2.0] if correct else [2.0, 1.0]
    return {"item_id": item_id, "correct": correct, "scores": scores}


def _valid_result_obj() -> JSONObject:
    outcomes = [_outcome_obj(f"item-{i}", correct=i < 4) for i in range(10)]
    return {
        "total": 10,
        "correct": 4,
        "accuracy": 0.4,
        "chance": 0.25,
        "outcomes": list(outcomes),
    }


def test_decode_result_round_trips() -> None:
    result = decode_cloze_eval_result(_valid_result_obj())
    assert result["total"] == 10
    assert result["correct"] == 4
    assert result["accuracy"] == pytest.approx(0.4)
    assert encode_cloze_eval_result(result) == _valid_result_obj()


def test_decode_result_preserves_per_item_outcomes_in_order() -> None:
    """The pairing a paired test needs is the item order, so it must survive."""
    result = decode_cloze_eval_result(_valid_result_obj())
    assert [o["item_id"] for o in result["outcomes"]] == [f"item-{i}" for i in range(10)]
    assert [o["correct"] for o in result["outcomes"]] == [True] * 4 + [False] * 6
    assert result["outcomes"][0]["scores"] == [1.0, 2.0]


def test_decode_result_rejects_outcome_count_disagreeing_with_total() -> None:
    obj = _valid_result_obj()
    obj["outcomes"] = [_outcome_obj("only", correct=True)]
    obj["correct"] = 1
    with pytest.raises(JSONTypeError, match="carries 1 entries but 'total' is 10"):
        decode_cloze_eval_result(obj)


def test_decode_result_rejects_correct_count_disagreeing_with_outcomes() -> None:
    """A count that does not match the records it summarises is not usable."""
    obj = _valid_result_obj()
    obj["correct"] = 5
    with pytest.raises(JSONTypeError, match="carries 4 correct entries"):
        decode_cloze_eval_result(obj)


def test_decode_result_rejects_non_object_outcome() -> None:
    obj = _valid_result_obj()
    obj["outcomes"] = [_outcome_obj("a", correct=True), "not-an-object"]
    obj["total"] = 2
    obj["correct"] = 1
    with pytest.raises(JSONTypeError, match=r"outcomes\[1\]' must be an object"):
        decode_cloze_eval_result(obj)


def test_decode_result_rejects_missing_outcomes() -> None:
    obj = _valid_result_obj()
    del obj["outcomes"]
    with pytest.raises(JSONTypeError, match="outcomes"):
        decode_cloze_eval_result(obj)


def test_decode_outcome_round_trips() -> None:
    outcome = decode_cloze_item_outcome(_outcome_obj("page::1", correct=True))
    assert outcome["item_id"] == "page::1"
    assert outcome["correct"] is True
    assert encode_cloze_item_outcome(outcome) == _outcome_obj("page::1", correct=True)


def test_decode_outcome_rejects_empty_item_id() -> None:
    obj = _outcome_obj("", correct=True)
    with pytest.raises(JSONTypeError, match="item_id"):
        decode_cloze_item_outcome(obj)


def test_decode_outcome_rejects_single_candidate() -> None:
    """One score means nothing was compared, so the record cannot be believed."""
    obj = _outcome_obj("a", correct=True)
    obj["scores"] = [1.0]
    with pytest.raises(JSONTypeError, match="at least 2 candidates"):
        decode_cloze_item_outcome(obj)


def test_decode_outcome_rejects_non_numeric_score() -> None:
    obj = _outcome_obj("a", correct=True)
    obj["scores"] = [1.0, "2.0"]
    with pytest.raises(JSONTypeError, match=r"scores\[1\]' must be a number"):
        decode_cloze_item_outcome(obj)


def test_decode_outcome_rejects_bool_score() -> None:
    """bool is an int subclass, so it would pass a naive numeric check."""
    obj = _outcome_obj("a", correct=True)
    obj["scores"] = [1.0, True]
    with pytest.raises(JSONTypeError, match=r"scores\[1\]' must be a number"):
        decode_cloze_item_outcome(obj)


def test_decode_outcome_accepts_integer_scores() -> None:
    obj = _outcome_obj("a", correct=True)
    obj["scores"] = [1, 2]
    outcome = decode_cloze_item_outcome(obj)
    assert outcome["scores"] == [1.0, 2.0]


def test_decode_outcome_rejects_correct_flag_contradicting_scores() -> None:
    """The flag and the scores are two statements of the same fact."""
    obj = _outcome_obj("a", correct=True)
    obj["scores"] = [5.0, 1.0]
    with pytest.raises(JSONTypeError, match="but its scores imply False"):
        decode_cloze_item_outcome(obj)


def test_decode_outcome_rejects_tie_reported_as_correct() -> None:
    """A tie is not a win, so a record claiming otherwise is rejected."""
    obj = _outcome_obj("a", correct=True)
    obj["scores"] = [2.0, 2.0]
    with pytest.raises(JSONTypeError, match="but its scores imply False"):
        decode_cloze_item_outcome(obj)


def test_answer_wins_outright_requires_a_strict_minimum() -> None:
    assert answer_wins_outright([1.0, 2.0, 3.0]) is True
    assert answer_wins_outright([2.0, 1.0]) is False
    assert answer_wins_outright([2.0, 2.0]) is False


def test_decode_result_rejects_negative_total() -> None:
    obj = _valid_result_obj()
    obj["total"] = -1
    obj["correct"] = 0
    with pytest.raises(JSONTypeError, match="total"):
        decode_cloze_eval_result(obj)


def test_decode_result_rejects_negative_correct() -> None:
    obj = _valid_result_obj()
    obj["correct"] = -1
    with pytest.raises(JSONTypeError, match="correct"):
        decode_cloze_eval_result(obj)


def test_decode_result_rejects_correct_exceeding_total() -> None:
    obj = _valid_result_obj()
    obj["correct"] = 11
    with pytest.raises(JSONTypeError, match="must not exceed"):
        decode_cloze_eval_result(obj)


def _valid_payload() -> ClozeJobPayload:
    return ClozeJobPayload(
        run_id="run-1", request_id="req-1", items_file_id="file-1", max_seq_len=256
    )


def test_job_payload_round_trips() -> None:
    payload = _valid_payload()
    encoded = encode_cloze_job_payload(payload)
    assert decode_cloze_job_payload(encoded) == payload


def test_decode_job_payload_rejects_non_positive_max_seq_len() -> None:
    encoded = encode_cloze_job_payload(_valid_payload())
    encoded["max_seq_len"] = 0
    with pytest.raises(JSONTypeError, match="must be positive"):
        decode_cloze_job_payload(encoded)


def test_decode_job_payload_rejects_missing_field() -> None:
    encoded = encode_cloze_job_payload(_valid_payload())
    del encoded["items_file_id"]
    with pytest.raises(JSONTypeError, match="items_file_id"):
        decode_cloze_job_payload(encoded)


def test_typed_dicts_construct_directly() -> None:
    item = ClozeItem(item_id="a", template=f"x {BLANK_MARKER}", answer="1", distractors=["2"])
    outcome = ClozeItemOutcome(item_id="a", correct=True, scores=[1.0, 2.0])
    result = ClozeEvalResult(total=1, correct=1, accuracy=1.0, chance=0.5, outcomes=[outcome])
    assert render_candidates(item) == ["x 1", "x 2"]
    assert result["chance"] == pytest.approx(0.5)
    assert result["outcomes"][0]["item_id"] == "a"


def _valid_baseline_payload() -> BaselineClozeJobPayload:
    return BaselineClozeJobPayload(
        hub_model_id="gpt2", items_file_id="file-1", max_seq_len=256, device="cpu"
    )


def test_baseline_job_payload_round_trips() -> None:
    payload = _valid_baseline_payload()
    encoded = encode_baseline_cloze_job_payload(payload)
    assert decode_baseline_cloze_job_payload(encoded) == payload


def test_baseline_payload_rejects_a_blank_model_id() -> None:
    """The model id becomes half the key the result is stored under.

    A blank one would produce a record nobody can identify, which is the exact
    condition this capability exists to end, so it fails at decode.
    """
    encoded = encode_baseline_cloze_job_payload(_valid_baseline_payload())
    encoded["hub_model_id"] = "   "
    with pytest.raises(JSONTypeError):
        decode_baseline_cloze_job_payload(encoded)


def test_baseline_payload_rejects_a_blank_items_file_id() -> None:
    encoded = encode_baseline_cloze_job_payload(_valid_baseline_payload())
    encoded["items_file_id"] = ""
    with pytest.raises(JSONTypeError):
        decode_baseline_cloze_job_payload(encoded)


def test_baseline_payload_rejects_a_non_positive_max_seq_len() -> None:
    encoded = encode_baseline_cloze_job_payload(_valid_baseline_payload())
    encoded["max_seq_len"] = 0
    with pytest.raises(JSONTypeError):
        decode_baseline_cloze_job_payload(encoded)
