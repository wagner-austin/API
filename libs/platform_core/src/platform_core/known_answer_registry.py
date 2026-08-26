"""The stored collection of known answers, and the operations over it.

:mod:`platform_core.known_answer` owns ONE entry -- what it means, how it is
validated, what checking it concludes. Nothing owned the file that holds them,
so every caller that needed the collection wrote its own loader, and four of
them existed within a day of the registry being created: one to register, one
to gate a record, one to repair the layout, one to answer what a new image
would do. Each re-derived "read the file, pull 'answers', decode each one",
and one of them wrote the file with the canonical encoder and collapsed it to
a single line.

That is the gap this module closes. The entry contract stays where it was;
this adds the collection.

WHY IT IS NOT JUST json.load. Two invariants have already been violated in
practice and are enforced here rather than remembered:

1. **The file is written indented.** ``dump_json_str`` defaults to compact
   because its usual job is feeding a digest, and a registry written that way
   turns every future entry into an unreadable single-line diff.
2. **An entry may not carry an empty fingerprint axis.** An unknown axis
   differs from every real value, so such an entry can never match anything
   again -- and the first probe to be registered nearly carried an empty
   ``driver_version`` because its launcher, not the run, had recorded the card.
"""

from __future__ import annotations

import pathlib

from platform_core.comparability import RunFingerprint
from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.known_answer import (
    AnswerDeviates,
    AnswerMatches,
    AnswerNotApplicable,
    KnownAnswer,
    check_known_answer,
    decode_known_answer,
    encode_known_answer,
)
from platform_core.run_record import RunRecord

ANSWERS_KEY = "answers"

# Indented, not canonical. See the module docstring: this file is read and
# diffed by people.
REGISTRY_INDENT = 2


def decode_registry(value: JSONValue) -> tuple[KnownAnswer, ...]:
    """Validate a JSON value as a registry document.

    Args:
        value: The parsed document.

    Returns:
        Every entry, in file order.

    Raises:
        JSONTypeError: If the document is not an object, has no ``answers``
            list, or any entry fails its own validation. A registry that
            partly decodes is not usable -- a gate that silently skips the
            entry covering the current configuration reports "no answer
            applies" when one does.
    """
    obj = narrow_json_to_dict(value)
    raw = obj.get(ANSWERS_KEY)
    if not isinstance(raw, list):
        raise JSONTypeError(f"Field {ANSWERS_KEY!r} must be a list of known answers")
    return tuple(decode_known_answer(entry) for entry in raw)


def encode_registry(answers: tuple[KnownAnswer, ...]) -> str:
    """Render a registry document as the text to store.

    Args:
        answers: The entries to write, in the order they should appear.

    Returns:
        Indented JSON with a trailing newline.
    """
    payload = {ANSWERS_KEY: [encode_known_answer(a) for a in answers]}
    return dump_json_str(payload, indent=REGISTRY_INDENT) + "\n"


def read_registry(path: pathlib.Path) -> tuple[KnownAnswer, ...]:
    """Read and validate the registry at ``path``.

    Args:
        path: The registry file.

    Returns:
        Every entry, in file order.

    Raises:
        JSONTypeError: If the document does not validate.
    """
    return decode_registry(load_json_str(path.read_text(encoding="utf-8")))


def write_registry(path: pathlib.Path, answers: tuple[KnownAnswer, ...]) -> None:
    """Store the registry at ``path``.

    Deliberately does NOT read the file back and compare. That check was here
    and was removed: it can only fire if :func:`encode_registry` and
    :func:`decode_registry` disagree, which is a property of those two
    functions and is asserted directly by the suite. Keeping it as a runtime
    guard added a branch no test could reach without faking the filesystem,
    which is a fake in front of pure code.

    Args:
        path: The registry file.
        answers: The entries to store.
    """
    path.write_text(encode_registry(answers), encoding="utf-8")


def incomplete_axes(fingerprint: RunFingerprint) -> tuple[str, ...]:
    """Name the fingerprint axes that are empty.

    Written as explicit lookups rather than a loop over a tuple of key names.
    Indexing a TypedDict with a variable is not something the type checker can
    verify, and the only ways to write the loop are a ``type: ignore`` or a
    cast -- both of which trade a checked access for an unchecked one to save
    two lines. ``determinism`` is deliberately absent: it is a nested record
    with its own validation, and "unpinned" is a real posture rather than a
    missing value.

    Args:
        fingerprint: The configuration to inspect.

    Returns:
        The empty axis names, in fingerprint declaration order; empty when
        every axis is populated.
    """
    empty: list[str] = []
    if fingerprint["image_digest"] == "":
        empty.append("image_digest")
    if fingerprint["gpu_model"] == "":
        empty.append("gpu_model")
    if fingerprint["driver_version"] == "":
        empty.append("driver_version")
    return tuple(empty)


def entry_from_record(record: RunRecord, tolerance: float) -> KnownAnswer:
    """Build the entry a run record establishes.

    Args:
        record: The measured run. Must carry exactly one observation, because
            a known answer is one expected value and choosing among several
            here would be a guess about which one the caller meant.
        tolerance: The absolute deviation still counted as a match.

    Returns:
        The entry, ready to be checked and stored.

    Raises:
        ValueError: If the record has other than one observation, or its
            fingerprint has an empty axis.
    """
    observations = record["observations"]
    if len(observations) != 1:
        raise ValueError(
            f"a known answer needs exactly one observation, record has {len(observations)}"
        )
    empty = incomplete_axes(record["fingerprint"])
    if empty:
        raise ValueError("refusing an entry whose fingerprint has empty axes: " + ", ".join(empty))
    return KnownAnswer(
        label=record["label"],
        fingerprint=record["fingerprint"],
        expected=observations[0]["value"],
        tolerance=tolerance,
    )


def find_entry(
    answers: tuple[KnownAnswer, ...], label: str, fingerprint: RunFingerprint
) -> KnownAnswer | None:
    """Return the entry registered for exactly this label and configuration.

    Args:
        answers: The registry.
        label: The label to match.
        fingerprint: The configuration to match.

    Returns:
        The entry, or None when the registry holds none for it.
    """
    for entry in answers:
        if entry["label"] == label and entry["fingerprint"] == fingerprint:
            return entry
    return None


def gate_record(
    answers: tuple[KnownAnswer, ...], record: RunRecord
) -> tuple[tuple[KnownAnswer, AnswerMatches | AnswerDeviates | AnswerNotApplicable], ...]:
    """Check a run record against every entry sharing its label.

    Args:
        answers: The registry.
        record: The measured run to gate.

    Returns:
        One (entry, outcome) pair per entry carrying the record's label, in
        file order. Entries for other configurations are included rather than
        filtered out: "this ran on a card no entry covers" and "no entry
        exists at all" are different situations, and dropping the
        non-applicable ones makes them look identical.

    Raises:
        ValueError: If the record does not carry exactly one observation.
    """
    observations = record["observations"]
    if len(observations) != 1:
        raise ValueError(f"a gated record needs exactly one observation, got {len(observations)}")
    observed = observations[0]["value"]
    return tuple(
        (entry, check_known_answer(entry, record["fingerprint"], observed))
        for entry in answers
        if entry["label"] == record["label"]
    )


__all__ = [
    "ANSWERS_KEY",
    "REGISTRY_INDENT",
    "decode_registry",
    "encode_registry",
    "entry_from_record",
    "find_entry",
    "gate_record",
    "incomplete_axes",
    "read_registry",
    "write_registry",
]
