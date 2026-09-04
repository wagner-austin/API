"""One arm of a whole-file continuation sweep, as a document rather than a command line.

A sweep's result is decided by thirteen values, and every one of them changes
the completions: change the token budget and items that used to finish now
run out; change the batch size and an item sits with different neighbours, so
padding differs and the arithmetic does; change the repetition penalty and a
degenerate loop either happens or does not.

Thirteen flags on a command line is thirteen chances to type one differently
between two arms that are supposed to differ in exactly one thing. So they
are a document instead, committed beside the run and read by both arms, and
the ONLY field a sibling arm changes is :attr:`ContinuationSweepSpec.arm`.
That is the same reason ``modeltrainer-cluster-train`` takes a payload rather
than a configuration's worth of flags.

The document also carries its paths, which is what makes it portable. The
first version of this sweep was a script with ``C:\\Users\\...`` compiled into
it; nothing about the run was wrong, and nothing about it could be repeated
anywhere else.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    require_float,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

CONTINUATION_ARMS: tuple[Literal["base", "candidate"], ...] = ("base", "candidate")
"""The two sides of the comparison, and the closed set an arm name may name."""

ContinuationArm = Literal["base", "candidate"]
"""Which side of the pairing one document produces."""

MINIMUM_REPETITION_PENALTY = 1.0
"""The neutral repetition penalty, and the floor a spec may declare.

Below 1.0 the penalty becomes a REWARD for tokens already emitted, which is
the opposite of what anybody setting this field wants and produces exactly
the degeneration it exists to suppress. Refused rather than clamped: a
clamped value would run under a setting the document does not state.
"""


def as_continuation_arm(raw: str, field: str) -> ContinuationArm:
    """Narrow a string to an arm name.

    Args:
        raw: The value read from the document.
        field: Field name, for the error message.

    Returns:
        The narrowed arm.

    Raises:
        JSONTypeError: If the value names no arm. A misspelled arm would
            otherwise write a third directory nothing compares, and the
            comparison would silently be between one arm and nothing.
    """
    if raw == "base":
        return "base"
    if raw == "candidate":
        return "candidate"
    raise JSONTypeError(f"Field '{field}' must be one of {list(CONTINUATION_ARMS)}, got {raw!r}")


class ContinuationSweepSpec(TypedDict):
    """Everything one arm of a continuation sweep needs in order to be repeated.

    Attributes:
        run_id: Names this sweep. Both arms of one comparison share it, which
            is what says they belong together.
        arm: Which side this document produces. ``candidate`` reattaches the
            saved adapter; ``base`` loads the very same weights the adapter
            was trained against and attaches nothing.
        artifact_path: The saved training run. BOTH arms name it, including
            the base one, because the control is defined by the run it
            controls for -- it is the adapter's own base, loaded under the
            adapter's own quantization, not a fresh model from the hub.
        holdout_path: Newline-delimited corpus documents the sweep continues.
        prompt_lines: How many lines of each file the model is shown.
        max_new_tokens: Token budget for one completion. Also decides which
            items are in scope at all, since an item whose reference cannot
            fit is excluded rather than scored on a truncation.
        max_prompt_tokens: How much of the file's head the model is given.
            Over-long prompts are truncated from the LEFT, keeping the text
            the completion has to continue from.
        batch_size: How many prompts are decoded together. Part of the arm
            rather than a throughput setting: padded positions change
            reduction order, so two arms batched differently are not paired.
        repetition_penalty: Penalty on tokens already generated. Decoding is
            greedy, so this is the only thing standing between the run and
            the degenerate loops that dominated the sweep of 2026-09-01.
        seed: Seeds the generator. Greedy decoding should not consume it,
            and it is recorded anyway so a run cannot be unable to say.
        device: Where generation runs.
        experiment: What this measurement belongs to, for the run record.
        label: Which measurement within it, for the run record.
    """

    run_id: str
    arm: ContinuationArm
    artifact_path: str
    holdout_path: str
    prompt_lines: int
    max_new_tokens: int
    max_prompt_tokens: int
    batch_size: int
    repetition_penalty: float
    seed: int
    device: str
    experiment: str
    label: str


class Completion(TypedDict):
    """What one item's generation produced.

    Attributes:
        item_id: The item this continues.
        text: The whole scored file -- the prompt the model was shown
            FOLLOWED BY what it wrote. Scoring the completion alone would
            fail every checker on imports the prompt already supplied, which
            measures the split point rather than the model.
        finished: Whether the model emitted its end-of-sequence token, as
            opposed to running out of budget.
    """

    item_id: str
    text: str
    finished: bool


def _require_nonempty_str(obj: JSONObject, field: str) -> str:
    """Read a required string that must carry characters.

    Args:
        obj: The document being decoded.
        field: Field name.

    Returns:
        The value.

    Raises:
        JSONTypeError: If the field is missing, is not a string, or is
            empty. An empty path or label is never a usable default -- it is
            a field somebody left blank, and accepting it defers the failure
            to a point where a GPU has already been spent.
    """
    value = require_str(obj, field)
    if value == "":
        raise JSONTypeError(f"Field '{field}' must not be empty")
    return value


def _require_positive_int(obj: JSONObject, field: str) -> int:
    """Read a required integer that must be at least one.

    Args:
        obj: The document being decoded.
        field: Field name.

    Returns:
        The value.

    Raises:
        JSONTypeError: If the field is missing, is not an integer, or is not
            positive. Every field read this way sizes something -- a budget,
            a batch, a prompt -- and zero of any of them produces an empty
            sweep that reports success.
    """
    value = require_int(obj, field)
    if value <= 0:
        raise JSONTypeError(f"Field '{field}' must be positive, got {value}")
    return value


def decode_continuation_sweep_spec(obj: JSONObject) -> ContinuationSweepSpec:
    """Decode one sweep document, refusing anything it cannot read.

    Args:
        obj: The document's JSON form.

    Returns:
        The validated spec.

    Raises:
        JSONTypeError: If any field is missing, has the wrong type, or holds
            a value that would make the sweep meaningless -- a non-positive
            size, an empty path, an unknown arm, a repetition penalty below
            neutral, or a negative seed.
    """
    seed = require_int(obj, "seed")
    if seed < 0:
        raise JSONTypeError(f"Field 'seed' must not be negative, got {seed}")
    penalty = require_float(obj, "repetition_penalty")
    if penalty < MINIMUM_REPETITION_PENALTY:
        raise JSONTypeError(
            f"Field 'repetition_penalty' must be at least "
            f"{MINIMUM_REPETITION_PENALTY}, got {penalty}; below that it "
            "rewards repetition rather than penalising it"
        )

    return ContinuationSweepSpec(
        run_id=_require_nonempty_str(obj, "run_id"),
        arm=as_continuation_arm(require_str(obj, "arm"), "arm"),
        artifact_path=_require_nonempty_str(obj, "artifact_path"),
        holdout_path=_require_nonempty_str(obj, "holdout_path"),
        prompt_lines=_require_positive_int(obj, "prompt_lines"),
        max_new_tokens=_require_positive_int(obj, "max_new_tokens"),
        max_prompt_tokens=_require_positive_int(obj, "max_prompt_tokens"),
        batch_size=_require_positive_int(obj, "batch_size"),
        repetition_penalty=penalty,
        seed=seed,
        device=_require_nonempty_str(obj, "device"),
        experiment=_require_nonempty_str(obj, "experiment"),
        label=_require_nonempty_str(obj, "label"),
    )


def encode_continuation_sweep_spec(spec: ContinuationSweepSpec) -> JSONObject:
    """Encode one sweep document.

    Args:
        spec: The spec.

    Returns:
        Its JSON form.
    """
    return {
        "run_id": spec["run_id"],
        "arm": spec["arm"],
        "artifact_path": spec["artifact_path"],
        "holdout_path": spec["holdout_path"],
        "prompt_lines": spec["prompt_lines"],
        "max_new_tokens": spec["max_new_tokens"],
        "max_prompt_tokens": spec["max_prompt_tokens"],
        "batch_size": spec["batch_size"],
        "repetition_penalty": spec["repetition_penalty"],
        "seed": spec["seed"],
        "device": spec["device"],
        "experiment": spec["experiment"],
        "label": spec["label"],
    }


__all__ = [
    "CONTINUATION_ARMS",
    "MINIMUM_REPETITION_PENALTY",
    "Completion",
    "ContinuationArm",
    "ContinuationSweepSpec",
    "as_continuation_arm",
    "decode_continuation_sweep_spec",
    "encode_continuation_sweep_spec",
]
