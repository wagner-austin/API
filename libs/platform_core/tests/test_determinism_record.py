"""Tests for the stack-agnostic determinism record.

This module exists because most of the monorepo's research is not torch.
The record has to be fillable by a numpy run, a Rust booster, or a job with
no GPU at all, and it has to compare correctly between them -- so the cases
here are deliberately about stacks that have nothing to do with CUDA.

The torch PRODUCER of these records is tested in platform_ml, beside the
code that pins the settings. Nothing here imports torch or platform_ml; that
dependency runs one way only.
"""

from __future__ import annotations

import pytest

from platform_core.determinism_record import (
    FALSE,
    TRUE,
    UNPINNED_STACK,
    DeterminismRecord,
    decode_determinism_record,
    determinism_record,
    encode_determinism_record,
    render_determinism_record,
)
from platform_core.json_utils import JSONTypeError


def test_a_non_torch_stack_can_describe_its_own_posture() -> None:
    # A gradient-boosting or BLAS-bound run pins entirely different things.
    record = determinism_record("numpy", {"threads": "1", "seed": "0"})

    assert record == {"stack": "numpy", "settings": (("seed", "0"), ("threads", "1"))}


def test_a_run_that_pinned_nothing_is_recordable() -> None:
    # "Nothing was pinned" is a fact about a run, and must differ from a
    # pinned run rather than be absent.
    record = determinism_record(UNPINNED_STACK, {})

    assert record == DeterminismRecord(stack=UNPINNED_STACK, settings=())
    assert record != determinism_record("numpy", {"threads": "1"})


def test_settings_are_canonically_ordered_whatever_the_producer_did() -> None:
    # Two records of the same posture must be equal and render identically,
    # or a re-ordered producer would read as a configuration change.
    forwards = determinism_record("numpy", {"a": "1", "b": "2"})
    backwards = determinism_record("numpy", {"b": "2", "a": "1"})

    assert forwards == backwards
    assert render_determinism_record(forwards) == render_determinism_record(backwards)


def test_render_names_the_stack_and_its_settings() -> None:
    rendered = render_determinism_record(determinism_record("numpy", {"threads": "1"}))

    assert rendered == "numpy[threads=1]"


def test_an_unpinned_record_renders_without_pretending_to_settings() -> None:
    assert render_determinism_record(determinism_record(UNPINNED_STACK, {})) == "none[]"


def test_two_stacks_pinning_the_same_setting_name_do_not_compare_equal() -> None:
    # The stack is part of the record precisely because a name like "threads"
    # can mean different things to different stacks.
    assert determinism_record("numpy", {"threads": "1"}) != determinism_record(
        "openblas", {"threads": "1"}
    )


def test_a_record_round_trips_through_storage() -> None:
    record = determinism_record("numpy", {"threads": "1", "hash_seed": "0"})

    assert decode_determinism_record(encode_determinism_record(record)) == record


def test_encode_nests_the_settings_under_their_stack() -> None:
    # Nested rather than flattened so a setting can never collide with the
    # "stack" key, whatever a future stack decides to name one.
    encoded = encode_determinism_record(
        determinism_record("torch", {"cudnn_benchmark": FALSE, "deterministic": TRUE})
    )

    assert encoded == {
        "stack": "torch",
        "settings": {"cudnn_benchmark": FALSE, "deterministic": TRUE},
    }


def test_a_record_that_cannot_say_what_pinned_it_is_rejected() -> None:
    with pytest.raises(ValueError, match="UNPINNED_STACK"):
        determinism_record("", {"threads": "1"})


def test_decode_rejects_an_unnamed_stack() -> None:
    with pytest.raises(JSONTypeError, match="stack"):
        decode_determinism_record({"stack": "", "settings": {}})


def test_decode_rejects_a_setting_whose_value_is_not_a_string() -> None:
    # A bool here would decode into a record that renders "True" on one
    # producer and "true" on another, so two identical postures would
    # compare as different configurations.
    with pytest.raises(JSONTypeError, match="threads"):
        decode_determinism_record({"stack": "numpy", "settings": {"threads": 1}})


def test_decode_rejects_a_non_object_and_a_missing_settings_block() -> None:
    with pytest.raises(JSONTypeError):
        decode_determinism_record("numpy")
    with pytest.raises(JSONTypeError):
        decode_determinism_record({"stack": "numpy"})
