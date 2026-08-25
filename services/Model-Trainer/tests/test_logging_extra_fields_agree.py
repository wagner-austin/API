"""``LoggingExtra`` and ``LOGGING_EXTRA_FIELDS`` are two hand-maintained
lists that must agree, and nothing enforced it.

``JsonFormatter`` renders only the keys named in ``LOGGING_EXTRA_FIELDS``.
A key present on the TypedDict but missing from the tuple type-checks, runs,
and is then discarded at write time -- silently, because a dropped field
looks exactly like a field nobody set.

Measured cost of that gap: every event published by a cluster run carried
``channel`` and ``event_body``, neither of which was in the tuple, so HPC3
runs 55570386 and 55570784 wrote 36 MB and 28 MB of bare
``{"message": "event"}``. The same gap swallowed ``best_val_loss`` from the
best-checkpoint restore line the day it was written.
"""

from __future__ import annotations

from model_trainer.core.logging.types import LOGGING_EXTRA_FIELDS, LoggingExtra


def test_every_declared_field_is_rendered() -> None:
    """A key on the TypedDict that the formatter never emits is a silent drop."""
    declared = set(LoggingExtra.__annotations__)
    rendered = set(LOGGING_EXTRA_FIELDS)
    assert sorted(declared - rendered) == []


def test_every_rendered_field_is_declared() -> None:
    """The reverse: a name in the tuple with no declaration is unreachable.

    Nothing can set it under ``--strict``, so it is a promise the type
    system has already made impossible to keep.
    """
    declared = set(LoggingExtra.__annotations__)
    rendered = set(LOGGING_EXTRA_FIELDS)
    assert sorted(rendered - declared) == []


def test_the_tuple_has_no_duplicates() -> None:
    """A repeated name renders once and hides a missing one from the counts."""
    assert len(LOGGING_EXTRA_FIELDS) == len(set(LOGGING_EXTRA_FIELDS))


def test_the_cluster_event_sink_keys_are_rendered() -> None:
    """The two keys whose absence produced 64 MB of empty event lines."""
    assert {"channel", "event_body"} <= set(LOGGING_EXTRA_FIELDS)
