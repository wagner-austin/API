"""How a held-out document becomes a continuation task, and where its answer lands.

Two programs have to agree on this, and they live in different packages.
``modeltrainer-continuations`` reads a holdout and WRITES one file per item;
``code-style-eval`` reads the same holdout and LOOKS FOR those files. If the
split rule or the filename convention differed between them by a single
character, the scorer would find nothing and report an arm that generated
nothing as an arm that was simply never scored -- which is indistinguishable,
in the outcome file, from a crashed generation.

So the agreement is a shared module rather than a shared convention. Neither
package may state the rule for itself, and there is no direction of import
between them: both depend on this.

The task itself is: show the model the head of a held-out source file and ask
it to write the rest. That is the task the corpus trained on, read backwards,
which is what makes the resulting pass rate a measurement of the training
rather than of prompt engineering.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence
from typing import Protocol

from typing_extensions import TypedDict

from platform_core.json_utils import (
    InvalidJsonError,
    JSONObject,
    JSONTypeError,
    JSONValue,
    load_json_str,
    require_bool,
    require_str,
)

ITEM_SEGMENT_SEPARATOR = "__"
"""What a path separator becomes when an item id is flattened to one segment."""

PYTHON_SUFFIX = ".py"
"""The only suffix an item may carry, because the guards glob for it."""

GENERATED_SUBDIR = "src"
"""Where an item's generated file sits inside that item's own root.

One of the directories the monorepo guards scan. A file anywhere else in the
root is invisible to them, and an item the guards cannot see scores a vacuous
pass rather than a refusal.
"""

MANIFEST_SUFFIX = ".generation.jsonl"
"""What is appended to a generated directory's name to name its manifest.

A SIBLING of the directory rather than a file inside it, because the scorer
walks the directory looking for generated files and a manifest living among
them would be one more thing every reader has to know to skip.
"""


class EvalPrompt(TypedDict):
    """One held-out file, split into what the model sees and what it must write.

    Attributes:
        item_id: The file's path within its repository, which identifies the
            item across arms and so carries the pairing.
        prompt: The head of the file, ending on a line boundary.
        reference: The remainder. Kept for perplexity scoring, and used to
            decide whether an item is answerable inside a token budget. The
            guard-pass metric never compares against it: matching the
            original text is not the goal, passing the repository's checkers
            is.
    """

    item_id: str
    prompt: str
    reference: str


class GenerationEntry(TypedDict):
    """What the generator recorded about one item it produced.

    Attributes:
        item_id: The item this describes.
        finished: Whether the model emitted its end-of-sequence token, as
            opposed to running out of budget. Recorded rather than inferred
            from the text afterwards, because a file that ends on a
            plausible-looking line and a file that ends because the budget
            did are indistinguishable once the tokens are gone.
    """

    item_id: str
    finished: bool


class TokenCounter(Protocol):
    """Measures a string in tokens.

    A Protocol rather than a tokenizer, because everything in this module is
    about SHAPE and must stay runnable without a model. The generator passes
    the real tokenizer; a test passes something that counts characters.
    """

    def __call__(self, text: str) -> int:
        """Count the tokens in a string.

        Args:
            text: The string to measure.

        Returns:
            How many tokens it encodes to.
        """
        ...


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

    The split point is a LINE boundary, never a character offset. Cutting a
    Python file mid-token hands the model a repair job instead of a
    continuation one, and a model that is good at repairing truncated
    identifiers would score well here for a reason nobody wants to measure.

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


def flatten_item_id(item_id: str) -> str:
    """Flatten a repository-relative item id into a single path segment.

    The item id is a path, so it is flattened rather than joined: joining
    would let an item id containing ``..`` write outside the directory, and
    the flattened name stays readable in a listing.

    Args:
        item_id: The item's path within its repository.

    Returns:
        The single-segment name.

    Raises:
        ValueError: If the id does not name a Python file. The guards find
            their work by globbing ``*.py``, so an item stored under any
            other suffix would be invisible to them and score a vacuous
            pass rather than a refusal.
    """
    if not item_id.endswith(PYTHON_SUFFIX):
        raise ValueError(f"item id '{item_id}' is not a Python file")
    return item_id.replace("/", ITEM_SEGMENT_SEPARATOR).replace("\\", ITEM_SEGMENT_SEPARATOR)


def item_root(generated_dir: pathlib.Path, item_id: str) -> pathlib.Path:
    """Locate the guard root for one item.

    Every item gets its OWN root holding only its own generated file. The
    monorepo guards are scoped to a tree rather than to a file -- they run
    over ``<root>/src``, ``<root>/scripts`` and ``<root>/tests`` -- so a
    single root shared by a whole sweep would return one verdict for all of
    it. Every item would then carry the same guards column, the paired table
    would compare two constants, and a sweep in which the adapter fixed real
    violations would report no difference for a reason having nothing to do
    with the models.

    Args:
        generated_dir: Directory of generated files.
        item_id: The item's path within its repository.

    Returns:
        The directory the guards are pointed at for this item.
    """
    return generated_dir / flatten_item_id(item_id)


def generated_path(generated_dir: pathlib.Path, item_id: str) -> pathlib.Path:
    """Locate the generated file for one item.

    Args:
        generated_dir: Directory of generated files.
        item_id: The item's path within its repository.

    Returns:
        The path the generation is expected at.
    """
    flat = flatten_item_id(item_id)
    return generated_dir / flat / GENERATED_SUBDIR / flat


def manifest_path(generated_dir: pathlib.Path) -> pathlib.Path:
    """Locate the generation manifest beside a generated directory.

    Args:
        generated_dir: Directory of generated files.

    Returns:
        The manifest's path.
    """
    return generated_dir.parent / (generated_dir.name + MANIFEST_SUFFIX)


def encode_generation_entry(entry: GenerationEntry) -> JSONObject:
    """Encode one manifest row.

    Args:
        entry: The row.

    Returns:
        Its JSON form.
    """
    return {"item_id": entry["item_id"], "finished": entry["finished"]}


def decode_generation_entry(obj: JSONObject) -> GenerationEntry:
    """Decode one manifest row.

    Args:
        obj: The row's JSON form.

    Returns:
        The row.

    Raises:
        JSONTypeError: If ``item_id`` is missing, is not a string, or is
            empty, or if ``finished`` is missing or is not a boolean. An
            empty id names no item, and a row naming no item is a row
            nothing can be attributed to.
    """
    item_id = require_str(obj, "item_id")
    if item_id == "":
        raise JSONTypeError("Field 'item_id' must not be empty")
    return GenerationEntry(item_id=item_id, finished=require_bool(obj, "finished"))


def finishable(
    prompts: Sequence[EvalPrompt], count_tokens: TokenCounter, budget: int
) -> list[EvalPrompt]:
    """Keep the items whose real continuation fits the token budget.

    An item whose reference is longer than the budget CANNOT be completed, so
    generating it buys a file that ends mid-expression, fails every checker on
    syntax, and reports a style verdict on a file the model was never allowed
    to finish. That is precisely what voided the first sweep of 2026-08-31.
    Excluding those items up front also stops them costing GPU time, since a
    batch runs until its longest row is done.

    Excluding them is a stated limit on scope, not a way to flatter the
    result: the sweep measures files whose remainder fits the budget, and both
    arms face exactly the same items, so the pairing is untouched. What it is
    NOT is a licence to drop items that merely came out badly -- a model that
    rambles past the budget on an item that fits still fails that item, which
    is a fact about the model and stays in.

    Args:
        prompts: Every prompt built from the holdout.
        count_tokens: Measures the reference.
        budget: The per-completion token budget.

    Returns:
        The prompts whose reference fits, in the order given.

    Raises:
        ValueError: If ``budget`` is not positive. A non-positive budget
            admits nothing, and an empty sweep that reports success is worse
            than one that refuses.
    """
    if budget <= 0:
        raise ValueError(f"budget must be positive, got {budget}")
    return [prompt for prompt in prompts if count_tokens(prompt["reference"]) <= budget]


def _by_length_then_id(entry: tuple[int, str, EvalPrompt]) -> tuple[int, str]:
    """Order a measured prompt by its length, breaking ties on its item id.

    A named function rather than a lambda so the tuple it returns is typed:
    the ordering is what makes two arms build byte-identical batches, and an
    ordering nothing checks is one that can silently stop being total.

    Args:
        entry: A prompt with its measured length and its item id.

    Returns:
        The sort key.
    """
    length, item_id, _ = entry
    return (length, item_id)


def batches(
    prompts: Sequence[EvalPrompt], count_tokens: TokenCounter, size: int
) -> list[list[EvalPrompt]]:
    """Group prompts into length-sorted batches.

    Sorting by token length keeps each batch close to uniform, so padding is
    a few tokens rather than the gap between the shortest and longest prompt
    in the whole sweep. The sort is on (length, item_id), so it is total and
    reproducible: two arms of one sweep build byte-identical batches, and an
    item therefore sits with the same neighbours in both. That is what keeps
    the comparison paired, because padding is what a neighbour changes.

    Args:
        prompts: Every prompt in the sweep.
        count_tokens: Measures each prompt.
        size: Maximum prompts per batch.

    Returns:
        The batches, in COMPOSITION order -- ascending by prompt length.
        Which batch an item lands in is what has to be reproducible; the
        order they are then RUN in is a separate question, and
        :func:`heaviest_first` answers it.

    Raises:
        ValueError: If ``size`` is not positive. A batch of zero prompts
            would loop forever producing nothing.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    measured: list[tuple[int, str, EvalPrompt]] = [
        (count_tokens(prompt["prompt"]), prompt["item_id"], prompt) for prompt in prompts
    ]
    measured.sort(key=_by_length_then_id)
    ordered = [entry[2] for entry in measured]
    return [ordered[start : start + size] for start in range(0, len(ordered), size)]


def batch_weight(batch: Sequence[EvalPrompt], count_tokens: TokenCounter) -> int:
    """Measure the padded input a batch will build.

    Every row is padded to the longest prompt in the batch, so what the model
    receives is that width times the row count -- not the sum of the prompts.
    A short batch of long prompts and a full batch of shorter ones are not
    ordered by either quantity alone.

    Args:
        batch: One composed batch.
        count_tokens: Measures each prompt.

    Returns:
        Rows times padded width, in tokens.

    Raises:
        ValueError: If the batch is empty, which has no width to pad to.
    """
    if not batch:
        raise ValueError("an empty batch has no width")
    return len(batch) * max(count_tokens(prompt["prompt"]) for prompt in batch)


def heaviest_first(
    groups: Sequence[Sequence[EvalPrompt]], count_tokens: TokenCounter
) -> list[list[EvalPrompt]]:
    """Order composed batches so the largest allocation happens first.

    COMPOSITION IS NOT EXECUTION ORDER, and separating them is free. Each
    batch is decoded independently and seeded independently, so which one
    runs first changes nothing about what any of them produces -- not the
    completions, not the manifest, not the payload digest, which is taken in
    sweep order rather than execution order.

    What it changes is WHEN a sweep that does not fit says so.
    :func:`batches` sorts ascending, so the peak allocation is the LAST
    batch: a run that cannot fit its widest batch discovers that after
    generating every other one, which on a real sweep is hours. Running the
    heaviest first turns that into minutes, and the batches already finished
    are still on disk for the resumed attempt.

    Ties break on the first item's id, so the order is total and two arms of
    one sweep walk their batches in the same sequence -- which matters not
    for correctness but for reading two logs side by side.

    Args:
        groups: The composed batches.
        count_tokens: Measures each prompt.

    Returns:
        The same batches, heaviest first.

    Raises:
        ValueError: If any batch is empty.
    """

    def descending(batch: Sequence[EvalPrompt]) -> tuple[int, str]:
        """Order one batch by size, breaking ties on its first item.

        Args:
            batch: One composed batch.

        Returns:
            The sort key: negated weight, then the leading item id.
        """
        first: EvalPrompt = batch[0]
        return (-batch_weight(batch, count_tokens), first["item_id"])

    return [list(batch) for batch in sorted(groups, key=descending)]


__all__ = [
    "GENERATED_SUBDIR",
    "ITEM_SEGMENT_SEPARATOR",
    "MANIFEST_SUFFIX",
    "PYTHON_SUFFIX",
    "EvalPrompt",
    "GenerationEntry",
    "MalformedRecordError",
    "TokenCounter",
    "batch_weight",
    "batches",
    "build_prompts",
    "decode_generation_entry",
    "encode_generation_entry",
    "finishable",
    "flatten_item_id",
    "generated_path",
    "heaviest_first",
    "item_root",
    "manifest_path",
    "split_document",
]
