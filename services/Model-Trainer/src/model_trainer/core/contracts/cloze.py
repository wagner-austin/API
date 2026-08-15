"""Contracts for cloze (multiple-choice-by-scoring) evaluation.

Perplexity answers whether a model finds a corpus unsurprising. It does not
answer whether the model can produce the facts the corpus contains, and those
come apart: a model can memorise text word-by-word and still fail every
question about it. Cloze evaluation closes that gap by masking a span, offering
the true value alongside distractors, and asking which completed sentence the
model finds least surprising.

Scoring is by substitution rather than generation. Each candidate is written
into the template, every rendering is scored, and the model is correct when the
true rendering wins. That needs only a per-token likelihood, is deterministic
because nothing is sampled, and requires no judge model and no output parsing.

The item schema is deliberately domain-free. A ``ClozeItem`` is a template, an
answer and a set of distractors; where those came from is the caller's concern.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

BLANK_MARKER = "<<BLANK>>"


class ClozeItem(TypedDict):
    """One multiple-choice item scored by substitution.

    Attributes:
        item_id: Stable identifier, unique within a set.
        template: Sentence containing exactly one ``BLANK_MARKER`` occurrence.
        answer: The true value for the blank.
        distractors: Wrong values, each distinct from the answer.
    """

    item_id: str
    template: str
    answer: str
    distractors: list[str]


class ClozeEvalResult(TypedDict):
    """Outcome of scoring a cloze set against one model.

    Attributes:
        total: Number of items scored.
        correct: Number of items where the true rendering scored best.
        accuracy: ``correct / total``.
        chance: Accuracy expected from uniform guessing over the candidates.
    """

    total: int
    correct: int
    accuracy: float
    chance: float


def encode_cloze_item(item: ClozeItem) -> JSONObject:
    """Encode a ClozeItem to a JSON object.

    Args:
        item: Item to encode.

    Returns:
        JSON-serialisable mapping carrying every field of the item.
    """
    distractors: list[JSONValue] = list(item["distractors"])
    return {
        "item_id": item["item_id"],
        "template": item["template"],
        "answer": item["answer"],
        "distractors": distractors,
    }


def decode_cloze_item(obj: JSONObject) -> ClozeItem:
    """Decode and validate a JSON object into a ClozeItem.

    Every constraint the scorer depends on is checked here, so the scoring loop
    can index candidates without re-testing their shape.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated ClozeItem.

    Raises:
        JSONTypeError: If a field is missing, has the wrong type, or violates a
            constraint the scorer relies on: exactly one blank marker in the
            template, a non-empty answer, at least one distractor, and no
            distractor equal to the answer.
    """
    item_id = require_str(obj, "item_id")
    template = require_str(obj, "template")
    answer = require_str(obj, "answer")

    marker_count = template.count(BLANK_MARKER)
    if marker_count != 1:
        raise JSONTypeError(
            f"Field 'template' must contain exactly one '{BLANK_MARKER}', got {marker_count}"
        )
    if answer == "":
        raise JSONTypeError("Field 'answer' must not be empty")

    raw_distractors = require_list(obj, "distractors")
    if len(raw_distractors) == 0:
        raise JSONTypeError("Field 'distractors' must not be empty")

    distractors: list[str] = []
    for index, candidate in enumerate(raw_distractors):
        if not isinstance(candidate, str):
            raise JSONTypeError(
                f"Field 'distractors[{index}]' must be a string, got {type(candidate).__name__}"
            )
        if candidate == answer:
            raise JSONTypeError(
                f"Field 'distractors[{index}]' equals the answer, making the item unscoreable"
            )
        distractors.append(candidate)

    return ClozeItem(
        item_id=item_id,
        template=template,
        answer=answer,
        distractors=distractors,
    )


def encode_cloze_eval_result(result: ClozeEvalResult) -> JSONObject:
    """Encode a ClozeEvalResult to a JSON object.

    Args:
        result: Result to encode.

    Returns:
        JSON-serialisable mapping carrying every field of the result.
    """
    return {
        "total": result["total"],
        "correct": result["correct"],
        "accuracy": result["accuracy"],
        "chance": result["chance"],
    }


def decode_cloze_eval_result(obj: JSONObject) -> ClozeEvalResult:
    """Decode and validate a JSON object into a ClozeEvalResult.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated ClozeEvalResult.

    Raises:
        JSONTypeError: If a field is missing, has the wrong type, or the counts
            are not internally consistent.
    """
    total = require_int(obj, "total")
    correct = require_int(obj, "correct")
    accuracy = require_float(obj, "accuracy")
    chance = require_float(obj, "chance")

    if total < 0:
        raise JSONTypeError(f"Field 'total' must not be negative, got {total}")
    if correct < 0:
        raise JSONTypeError(f"Field 'correct' must not be negative, got {correct}")
    if correct > total:
        raise JSONTypeError(f"Field 'correct' ({correct}) must not exceed 'total' ({total})")

    return ClozeEvalResult(total=total, correct=correct, accuracy=accuracy, chance=chance)


def render_candidates(item: ClozeItem) -> list[str]:
    """Render the item once per candidate, answer first.

    The answer occupies index 0 so the scorer can test correctness by comparing
    the winning index against zero, without re-matching strings.

    Args:
        item: Item to render.

    Returns:
        Rendered sentences, the answer's rendering first, then each distractor's
        in declaration order.
    """
    candidates = [item["answer"], *item["distractors"]]
    return [item["template"].replace(BLANK_MARKER, candidate) for candidate in candidates]


__all__ = [
    "BLANK_MARKER",
    "ClozeEvalResult",
    "ClozeItem",
    "decode_cloze_eval_result",
    "decode_cloze_item",
    "encode_cloze_eval_result",
    "encode_cloze_item",
    "render_candidates",
]
