"""The sweep document, and everything it refuses to be read as.

Each refusal here is a run that would otherwise start, spend a GPU, and
produce files nobody can interpret. That is why none of them is a warning:
the cheapest moment to fail is before the weights load, and the most
expensive is after the sweep finishes looking fine.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from model_trainer.core.contracts.continuation_sweep import (
    CONTINUATION_ARMS,
    ContinuationSweepSpec,
    as_continuation_arm,
    decode_continuation_sweep_spec,
    encode_continuation_sweep_spec,
)

_SIZED_FIELDS = (
    "prompt_lines",
    "max_new_tokens",
    "max_prompt_tokens",
    "batch_size",
)

_NAMED_FIELDS = (
    "run_id",
    "artifact_path",
    "holdout_path",
    "device",
    "experiment",
    "label",
)


def _document(**overrides: JSONValue) -> JSONObject:
    """Build a valid sweep document with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        The document.
    """
    base: JSONObject = {
        "run_id": "qlora-qwen-code-v1",
        "arm": "candidate",
        "artifact_path": "/pub/wagnera3/code-style/artifacts/qlora-v1",
        "holdout_path": "/pub/wagnera3/code-style/corpora/code-corpus-v1.holdout.jsonl",
        "prompt_lines": 20,
        "max_new_tokens": 1536,
        "max_prompt_tokens": 1024,
        "batch_size": 32,
        "repetition_penalty": 1.1,
        "seed": 0,
        "device": "cuda",
        "experiment": "code-style-guard-pass",
        "label": "qwen2.5-coder-1.5b-qlora-v1-candidate",
    }
    for key, value in overrides.items():
        base[key] = value
    return base


def _spec(**overrides: JSONValue) -> ContinuationSweepSpec:
    """Decode a valid document with optional overrides.

    Args:
        **overrides: Fields to replace.

    Returns:
        The decoded spec.
    """
    return decode_continuation_sweep_spec(_document(**overrides))


class TestTheArmName:
    """The set is closed, because a third name compares against nothing."""

    @pytest.mark.parametrize("arm", CONTINUATION_ARMS)
    def test_every_declared_arm_narrows(self, arm: str) -> None:
        """Iterating the declared set keeps this honest as it grows.

        Args:
            arm: The arm name.
        """
        assert as_continuation_arm(arm, "arm") == arm

    def test_an_unknown_arm_is_refused(self) -> None:
        """A misspelling would write a third directory nothing compares."""
        with pytest.raises(JSONTypeError, match="must be one of"):
            _ = as_continuation_arm("baseline", "arm")

    def test_the_refusal_names_the_field(self) -> None:
        with pytest.raises(JSONTypeError, match="'arm'"):
            _ = decode_continuation_sweep_spec(_document(arm="control"))


class TestRoundTripping:
    """A document read and written back is the same document."""

    def test_a_candidate_document_round_trips(self) -> None:
        spec = _spec()

        assert decode_continuation_sweep_spec(encode_continuation_sweep_spec(spec)) == spec

    def test_a_base_document_round_trips(self) -> None:
        """Both arms are read by the same decoder; neither is the special case."""
        spec = _spec(arm="base")

        assert decode_continuation_sweep_spec(encode_continuation_sweep_spec(spec)) == spec

    def test_the_arm_is_the_only_field_that_changes_what_is_computed(self) -> None:
        """That is the whole reason this is a document and not a command line.

        The two documents this repository commits also differ in ``label``,
        because two records that scored differently must be named
        differently. ``label`` names the result; every other field decides
        it. Comparing at one label is what isolates the second claim from
        the first.
        """
        candidate = encode_continuation_sweep_spec(_spec())
        base = encode_continuation_sweep_spec(_spec(arm="base"))

        differing = [key for key in candidate if candidate[key] != base[key]]

        assert differing == ["arm"]


class TestSizes:
    """Every size here sizes something, and zero of any of them empties the sweep."""

    @pytest.mark.parametrize("field", _SIZED_FIELDS)
    def test_a_zero_size_is_refused(self, field: str) -> None:
        """An empty sweep that reports success is the failure being prevented.

        Args:
            field: The field set to zero.
        """
        with pytest.raises(JSONTypeError, match=f"'{field}' must be positive"):
            _ = decode_continuation_sweep_spec(_document(**{field: 0}))

    @pytest.mark.parametrize("field", _SIZED_FIELDS)
    def test_a_negative_size_is_refused(self, field: str) -> None:
        """Same rule from the other side.

        Args:
            field: The field set negative.
        """
        with pytest.raises(JSONTypeError, match=f"'{field}' must be positive"):
            _ = decode_continuation_sweep_spec(_document(**{field: -1}))

    @pytest.mark.parametrize("field", _SIZED_FIELDS)
    def test_a_missing_size_is_refused(self, field: str) -> None:
        """Absent is not zero and is not a default.

        Args:
            field: The field removed.
        """
        document = _document()
        del document[field]

        with pytest.raises(JSONTypeError, match=field):
            _ = decode_continuation_sweep_spec(document)


class TestNames:
    """A blank path or label is a field somebody left empty."""

    @pytest.mark.parametrize("field", _NAMED_FIELDS)
    def test_an_empty_value_is_refused(self, field: str) -> None:
        """Accepting it defers the failure to after a GPU is spent.

        Args:
            field: The field blanked.
        """
        with pytest.raises(JSONTypeError, match=f"'{field}' must not be empty"):
            _ = decode_continuation_sweep_spec(_document(**{field: ""}))

    @pytest.mark.parametrize("field", _NAMED_FIELDS)
    def test_a_missing_value_is_refused(self, field: str) -> None:
        """Args:
        field: The field removed.
        """
        document = _document()
        del document[field]

        with pytest.raises(JSONTypeError, match=field):
            _ = decode_continuation_sweep_spec(document)


class TestTheRepetitionPenalty:
    """Below neutral it rewards repetition, which nobody setting it means."""

    def test_the_neutral_value_is_accepted(self) -> None:
        """1.0 is a real choice: no penalty, stated rather than implied."""
        assert _spec(repetition_penalty=1.0)["repetition_penalty"] == 1.0

    def test_a_value_below_neutral_is_refused_rather_than_clamped(self) -> None:
        """A clamped run would run under a setting the document does not state."""
        with pytest.raises(JSONTypeError, match="rewards repetition"):
            _ = decode_continuation_sweep_spec(_document(repetition_penalty=0.9))

    def test_a_missing_penalty_is_refused(self) -> None:
        document = _document()
        del document["repetition_penalty"]

        with pytest.raises(JSONTypeError, match="repetition_penalty"):
            _ = decode_continuation_sweep_spec(document)


class TestTheSeed:
    """Recorded even where greedy decoding should not consume it."""

    def test_zero_is_a_seed(self) -> None:
        assert _spec(seed=0)["seed"] == 0

    def test_a_negative_seed_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must not be negative"):
            _ = decode_continuation_sweep_spec(_document(seed=-1))

    def test_a_missing_seed_is_refused(self) -> None:
        document = _document()
        del document["seed"]

        with pytest.raises(JSONTypeError, match="seed"):
            _ = decode_continuation_sweep_spec(document)
