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

from collections.abc import Sequence

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_bool,
    require_float,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

BLANK_MARKER = "<<BLANK>>"

MIN_CANDIDATES = 2


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


class ClozeItemOutcome(TypedDict):
    """What one item's scoring produced.

    Carried so that two arms scored on the same item set can be compared item
    by item. Aggregate counts alone only support an unpaired test, which throws
    away the pairing that a shared item set provides and widens every interval
    the comparison produces.

    Attributes:
        item_id: The scored item's identifier, from its ``ClozeItem``.
        correct: Whether the answer's rendering was the strict minimum.
        scores: Total negative log-likelihood per candidate, in the order
            :func:`render_candidates` emits them, so the answer is at index 0.
    """

    item_id: str
    correct: bool
    scores: list[float]


class ClozeEvalResult(TypedDict):
    """Outcome of scoring a cloze set against one model.

    Attributes:
        total: Number of items scored.
        correct: Number of items where the true rendering scored best.
        accuracy: ``correct / total``.
        chance: Accuracy expected from uniform guessing over the candidates.
        outcomes: Per-item results, in the order the items were scored.
    """

    total: int
    correct: int
    accuracy: float
    chance: float
    outcomes: list[ClozeItemOutcome]


def answer_wins_outright(scores: Sequence[float]) -> bool:
    """Report whether index 0 is the strict minimum of the scores.

    A tie is not a win. Two renderings sharing the lowest score means the model
    did not separate them, and counting that as correct would credit the answer
    for a coin flip it never made. Requiring a strict minimum keeps the metric
    from drifting upward on models that assign identical likelihoods.

    This lives on the contract rather than in the scorer because the decoder
    validates ``correct`` against ``scores``; one definition serves both, so a
    record cannot be produced by one rule and accepted under another.

    Args:
        scores: Total negative log-likelihoods, the answer's at index 0.

    Returns:
        True when index 0 holds the lowest score and no other index matches it.
    """
    best = scores[0]
    return all(scores[index] > best for index in range(1, len(scores)))


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


def encode_cloze_item_outcome(outcome: ClozeItemOutcome) -> JSONObject:
    """Encode a ClozeItemOutcome to a JSON object.

    Args:
        outcome: Outcome to encode.

    Returns:
        JSON-serialisable mapping carrying every field of the outcome.
    """
    scores: list[JSONValue] = list(outcome["scores"])
    return {
        "item_id": outcome["item_id"],
        "correct": outcome["correct"],
        "scores": scores,
    }


def decode_cloze_item_outcome(obj: JSONObject) -> ClozeItemOutcome:
    """Decode and validate a JSON object into a ClozeItemOutcome.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated ClozeItemOutcome.

    Raises:
        JSONTypeError: If a field is missing or has the wrong type, if fewer
            than two candidate scores are present, or if ``correct`` disagrees
            with what ``scores`` implies under :func:`answer_wins_outright`.
    """
    item_id = require_str(obj, "item_id")
    if item_id == "":
        raise JSONTypeError("Field 'item_id' must not be empty")

    correct = require_bool(obj, "correct")

    raw_scores = require_list(obj, "scores")
    if len(raw_scores) < MIN_CANDIDATES:
        raise JSONTypeError(
            f"Field 'scores' must carry at least {MIN_CANDIDATES} candidates "
            f"(the answer and one distractor), got {len(raw_scores)}"
        )

    scores: list[float] = []
    for index, value in enumerate(raw_scores):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise JSONTypeError(
                f"Field 'scores[{index}]' must be a number, got {type(value).__name__}"
            )
        scores.append(float(value))

    implied = answer_wins_outright(scores)
    if implied != correct:
        raise JSONTypeError(
            f"Field 'correct' is {correct} for item '{item_id}' but its scores imply {implied}; "
            "the answer's score is at index 0 and must be the strict minimum to count as correct"
        )

    return ClozeItemOutcome(item_id=item_id, correct=correct, scores=scores)


def encode_cloze_eval_result(result: ClozeEvalResult) -> JSONObject:
    """Encode a ClozeEvalResult to a JSON object.

    Args:
        result: Result to encode.

    Returns:
        JSON-serialisable mapping carrying every field of the result.
    """
    outcomes: list[JSONValue] = [
        encode_cloze_item_outcome(outcome) for outcome in result["outcomes"]
    ]
    return {
        "total": result["total"],
        "correct": result["correct"],
        "accuracy": result["accuracy"],
        "chance": result["chance"],
        "outcomes": outcomes,
    }


def decode_cloze_eval_result(obj: JSONObject) -> ClozeEvalResult:
    """Decode and validate a JSON object into a ClozeEvalResult.

    Args:
        obj: JSON object to decode.

    Returns:
        Validated ClozeEvalResult.

    Raises:
        JSONTypeError: If a field is missing, has the wrong type, or the counts
            are not internally consistent with each other or with ``outcomes``.
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

    raw_outcomes = require_list(obj, "outcomes")
    outcomes: list[ClozeItemOutcome] = []
    for index, entry in enumerate(raw_outcomes):
        if not isinstance(entry, dict):
            raise JSONTypeError(
                f"Field 'outcomes[{index}]' must be an object, got {type(entry).__name__}"
            )
        outcomes.append(decode_cloze_item_outcome(entry))

    if len(outcomes) != total:
        raise JSONTypeError(
            f"Field 'outcomes' carries {len(outcomes)} entries but 'total' is {total}"
        )

    counted = sum(1 for outcome in outcomes if outcome["correct"])
    if counted != correct:
        raise JSONTypeError(
            f"Field 'correct' is {correct} but 'outcomes' carries {counted} correct entries"
        )

    return ClozeEvalResult(
        total=total,
        correct=correct,
        accuracy=accuracy,
        chance=chance,
        outcomes=outcomes,
    )


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
    "MIN_CANDIDATES",
    "ClozeEvalResult",
    "ClozeItem",
    "ClozeItemOutcome",
    "answer_wins_outright",
    "decode_cloze_eval_result",
    "decode_cloze_item",
    "decode_cloze_item_outcome",
    "encode_cloze_eval_result",
    "encode_cloze_item",
    "encode_cloze_item_outcome",
    "render_candidates",
]
