"""Turn held-out corpus documents into whole-file continuation prompts.

The v1 task is: show the model the head of a held-out source file and ask it
to write the rest. That is the task the corpus trained on, read backwards,
which is what makes the guard-pass rate a measurement of the training rather
than of prompt engineering.

The split point is a LINE boundary, never a character offset. Cutting a
Python file mid-token hands the model a repair job instead of a continuation
one, and a model that is good at repairing truncated identifiers would score
well here for a reason nobody wants to measure.
"""

from __future__ import annotations

from collections.abc import Sequence

from platform_core.json_utils import (
    InvalidJsonError,
    JSONValue,
    load_json_str,
)
from typing_extensions import TypedDict


class EvalPrompt(TypedDict):
    """One held-out file, split into what the model sees and what it must write.

    Attributes:
        item_id: The file's path within its repository, which identifies the
            item across arms and so carries the pairing.
        prompt: The head of the file, ending on a line boundary.
        reference: The remainder, kept for perplexity scoring. The guard-pass
            metric never compares against it: matching the original text is
            not the goal, passing the repository's checkers is.
    """

    item_id: str
    prompt: str
    reference: str


class MalformedRecordError(ValueError):
    """A corpus line that is not a usable document record."""


def _record_fields(raw: str, number: int) -> tuple[str, str]:
    """Read one JSONL line into its path and text.

    Args:
        raw: The line.
        number: 1-based line number, for the error message.

    Returns:
        A tuple of (path, text).

    Raises:
        MalformedRecordError: If the line is not a JSON object carrying a
            string ``path`` and a non-empty string ``text``.
    """
    try:
        record: JSONValue = load_json_str(raw)
    except InvalidJsonError as invalid:
        raise MalformedRecordError(f"line {number} is not valid JSON: {invalid}") from invalid
    if not isinstance(record, dict):
        raise MalformedRecordError(f"line {number} is not a JSON object")
    path = record.get("path")
    text = record.get("text")
    if not isinstance(path, str) or path == "":
        raise MalformedRecordError(f"line {number} carries no string 'path'")
    if not isinstance(text, str) or text == "":
        raise MalformedRecordError(f"line {number} carries no non-empty 'text'")
    return path, text


def split_document(text: str, prompt_lines: int) -> tuple[str, str] | None:
    """Split a document into a prompt head and a reference tail.

    Args:
        text: The whole source file.
        prompt_lines: How many lines the model is shown.

    Returns:
        A tuple of (prompt, reference), or None when the file has no line
        left to continue. A file shorter than the prompt budget is not a
        continuation task, and scoring the model on an empty target would
        record a pass for writing nothing.
    """
    lines = text.splitlines(keepends=True)
    if len(lines) <= prompt_lines:
        return None
    return "".join(lines[:prompt_lines]), "".join(lines[prompt_lines:])


def build_prompts(records: Sequence[str], prompt_lines: int) -> list[EvalPrompt]:
    """Build a prompt per usable held-out document.

    Args:
        records: JSONL lines from the holdout corpus.
        prompt_lines: How many lines of each file the model is shown.

    Returns:
        One prompt per document long enough to continue, in corpus order.

    Raises:
        ValueError: If ``prompt_lines`` is not positive. A zero-line prompt
            asks the model to write a file from nothing, which is a
            different experiment.
        MalformedRecordError: If a non-blank line is not a usable record.
    """
    if prompt_lines <= 0:
        raise ValueError(f"prompt_lines must be positive, got {prompt_lines}")
    prompts: list[EvalPrompt] = []
    for number, raw in enumerate(records, start=1):
        if not raw.strip():
            continue
        path, text = _record_fields(raw, number)
        split = split_document(text, prompt_lines)
        if split is None:
            continue
        prompt, reference = split
        prompts.append(EvalPrompt(item_id=path, prompt=prompt, reference=reference))
    return prompts


__all__ = ["EvalPrompt", "MalformedRecordError", "build_prompts", "split_document"]
