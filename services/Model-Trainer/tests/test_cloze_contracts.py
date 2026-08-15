"""Tests for the cloze contract encode/decode functions.

Every rejection branch is exercised against the real decoder. The decoder is
the only place item shape is checked, so a gap here becomes a crash inside the
scoring loop rather than a validation error at the edge.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError

from model_trainer.core.contracts.cloze import (
    BLANK_MARKER,
    ClozeEvalResult,
    ClozeItem,
    decode_cloze_eval_result,
    decode_cloze_item,
    encode_cloze_eval_result,
    encode_cloze_item,
    render_candidates,
)
from model_trainer.core.contracts.queue import ClozeJobPayload
from model_trainer.core.contracts.queue_encoding import (
    decode_cloze_job_payload,
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


def _valid_result_obj() -> JSONObject:
    return {"total": 10, "correct": 4, "accuracy": 0.4, "chance": 0.25}


def test_decode_result_round_trips() -> None:
    result = decode_cloze_eval_result(_valid_result_obj())
    assert result["total"] == 10
    assert result["correct"] == 4
    assert result["accuracy"] == pytest.approx(0.4)
    assert encode_cloze_eval_result(result) == _valid_result_obj()


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
    result = ClozeEvalResult(total=1, correct=1, accuracy=1.0, chance=0.5)
    assert render_candidates(item) == ["x 1", "x 2"]
    assert result["chance"] == pytest.approx(0.5)
