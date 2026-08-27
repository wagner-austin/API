"""The registry collection: reading, writing, building entries and gating.

Every test uses the real functions against real files in tmp_path. The
module has no I/O seam worth faking -- a fake filesystem in front of
pathlib would test the fake.
"""

from __future__ import annotations

import pathlib

import pytest

from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str, load_json_str
from platform_core.known_answer import KnownAnswer, encode_known_answer
from platform_core.known_answer_registry import (
    ANSWERS_KEY,
    decode_registry,
    encode_registry,
    entry_from_record,
    find_entry,
    gate_record,
    incomplete_axes,
    read_registry,
    write_registry,
)
from platform_core.run_record import Observation, RunRecord, run_record
from platform_core.testing import sample_run_fingerprint

_PINNED = DeterminismRecord(stack="torch", settings=(("matmul_tf32", "false"),))

_FULL = sample_run_fingerprint(
    image_digest="a" * 64,
    gpu_model="NVIDIA A100 80GB PCIe",
    driver_version="580.82.07",
    determinism=_PINNED,
)

_OTHER_CARD = sample_run_fingerprint(
    image_digest="a" * 64,
    gpu_model="Tesla V100-FHHL-16GB",
    driver_version="580.82.07",
    determinism=_PINNED,
)

_LABEL = "probe-label"


def _answer(fingerprint: RunFingerprint, expected: float, tolerance: float = 0.0) -> KnownAnswer:
    """Build an entry for tests."""
    return KnownAnswer(
        label=_LABEL, fingerprint=fingerprint, expected=expected, tolerance=tolerance
    )


def _record(fingerprint: RunFingerprint, value: float, label: str = _LABEL) -> RunRecord:
    """Build a single-observation run record for tests."""
    return run_record(
        experiment="test-experiment",
        label=label,
        fingerprint=fingerprint,
        observations=(Observation(name="probe_loss", value=value),),
        payload_digest="",
    )


class TestDecoding:
    """A registry that partly decodes is not usable, so it is refused."""

    def test_it_decodes_entries_in_file_order(self) -> None:
        doc: JSONValue = {
            ANSWERS_KEY: [
                encode_known_answer(_answer(_FULL, 1.0)),
                encode_known_answer(_answer(_OTHER_CARD, 2.0)),
            ]
        }

        decoded = decode_registry(doc)

        assert [a["expected"] for a in decoded] == [1.0, 2.0]

    def test_an_empty_registry_decodes_to_nothing(self) -> None:
        assert decode_registry({ANSWERS_KEY: []}) == ()

    def test_a_document_that_is_not_an_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_registry([1, 2, 3])

    def test_a_missing_answers_key_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="answers"):
            decode_registry({"entries": []})

    def test_answers_that_is_not_a_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="answers"):
            decode_registry({ANSWERS_KEY: {"label": "x"}})

    def test_one_bad_entry_refuses_the_whole_document(self) -> None:
        # Skipping it would make a gate report "no answer applies" when the
        # entry covering the current configuration is the broken one.
        doc: JSONValue = {ANSWERS_KEY: [encode_known_answer(_answer(_FULL, 1.0)), {"label": ""}]}

        with pytest.raises(JSONTypeError):
            decode_registry(doc)


class TestTheStoredLayout:
    """The file is read and diffed by people, so it is written indented."""

    def test_it_is_written_indented_not_canonical(self, tmp_path: pathlib.Path) -> None:
        # The regression: a registration run wrote this file with the
        # canonical encoder and collapsed 88 readable lines into one.
        path = tmp_path / "known-answers.json"

        write_registry(path, (_answer(_FULL, 1.0), _answer(_OTHER_CARD, 2.0)))

        text = path.read_text(encoding="utf-8")
        assert text.count("\n") > 10
        assert text.endswith("}\n")

    def test_encode_round_trips_through_decode(self) -> None:
        answers = (_answer(_FULL, 1.0), _answer(_OTHER_CARD, 2.0))

        assert decode_registry(load_json_str(encode_registry(answers))) == answers

    def test_write_then_read_returns_what_was_written(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "known-answers.json"
        answers = (_answer(_FULL, 6.25),)

        write_registry(path, answers)

        assert read_registry(path) == answers

    def test_reading_a_document_that_does_not_validate_raises(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "known-answers.json"
        path.write_text(dump_json_str({"answers": "not a list"}), encoding="utf-8")

        with pytest.raises(JSONTypeError, match="answers"):
            read_registry(path)


class TestIncompleteAxes:
    """An entry with an unknown axis can never match anything again."""

    def test_a_complete_fingerprint_reports_nothing_empty(self) -> None:
        assert incomplete_axes(_FULL) == ()

    def test_it_names_every_empty_axis_in_declaration_order(self) -> None:
        bare = sample_run_fingerprint(
            image_digest="", gpu_model="", driver_version="", determinism=_PINNED
        )

        assert incomplete_axes(bare) == ("image_digest", "gpu_model", "driver_version")

    @pytest.mark.parametrize("axis", ["image_digest", "gpu_model", "driver_version"])
    def test_it_names_a_single_empty_axis(self, axis: str) -> None:
        fingerprint = sample_run_fingerprint(
            image_digest="" if axis == "image_digest" else "a" * 64,
            gpu_model="" if axis == "gpu_model" else "A100",
            driver_version="" if axis == "driver_version" else "580.82.07",
            determinism=_PINNED,
        )

        assert incomplete_axes(fingerprint) == (axis,)


class TestEntryFromRecord:
    """Building the entry a measurement establishes."""

    def test_it_takes_the_label_value_and_configuration_from_the_record(self) -> None:
        entry = entry_from_record(_record(_FULL, 6.25), 0.0)

        assert entry == KnownAnswer(label=_LABEL, fingerprint=_FULL, expected=6.25, tolerance=0.0)

    def test_it_carries_the_tolerance_it_was_given(self) -> None:
        assert entry_from_record(_record(_FULL, 6.25), 1e-6)["tolerance"] == 1e-6

    def test_a_record_with_no_observations_is_refused(self) -> None:
        record = run_record(
            experiment="e",
            label=_LABEL,
            fingerprint=_FULL,
            observations=(),
            payload_digest="",
        )

        with pytest.raises(ValueError, match="exactly one observation"):
            entry_from_record(record, 0.0)

    def test_a_record_with_two_observations_is_refused(self) -> None:
        # Choosing between them here would be a guess about which the caller
        # meant, and the wrong guess registers a number nobody measured.
        record = run_record(
            experiment="e",
            label=_LABEL,
            fingerprint=_FULL,
            observations=(
                Observation(name="a", value=1.0),
                Observation(name="b", value=2.0),
            ),
            payload_digest="",
        )

        with pytest.raises(ValueError, match="exactly one observation"):
            entry_from_record(record, 0.0)

    def test_an_incomplete_fingerprint_is_refused_and_names_the_axes(self) -> None:
        # The real case: a probe whose launcher recorded the card, so
        # driver_version was simply absent.
        fingerprint = sample_run_fingerprint(
            image_digest="a" * 64,
            gpu_model="NVIDIA A100 80GB PCIe",
            driver_version="",
            determinism=_PINNED,
        )

        with pytest.raises(ValueError, match="driver_version"):
            entry_from_record(_record(fingerprint, 6.25), 0.0)


class TestFindEntry:
    """Locating an already-registered entry."""

    def test_it_finds_the_entry_for_this_exact_configuration(self) -> None:
        wanted = _answer(_FULL, 1.0)

        assert find_entry((_answer(_OTHER_CARD, 2.0), wanted), _LABEL, _FULL) == wanted

    def test_it_returns_none_when_the_card_differs(self) -> None:
        assert find_entry((_answer(_FULL, 1.0),), _LABEL, _OTHER_CARD) is None

    def test_it_returns_none_when_the_label_differs(self) -> None:
        assert find_entry((_answer(_FULL, 1.0),), "other-label", _FULL) is None

    def test_it_returns_none_for_an_empty_registry(self) -> None:
        assert find_entry((), _LABEL, _FULL) is None


class TestGateRecord:
    """Gating a measured run against the registry."""

    def test_a_matching_run_matches_its_own_entry(self) -> None:
        registry = (_answer(_FULL, 6.25),)

        outcomes = gate_record(registry, _record(_FULL, 6.25))

        assert [o["kind"] for _, o in outcomes] == ["matches"]

    def test_a_drifted_run_deviates(self) -> None:
        registry = (_answer(_FULL, 6.25),)

        outcomes = gate_record(registry, _record(_FULL, 6.25 + 1e-9))

        assert [o["kind"] for _, o in outcomes] == ["deviates"]

    def test_a_run_on_another_card_does_not_apply(self) -> None:
        registry = (_answer(_FULL, 6.25),)

        outcomes = gate_record(registry, _record(_OTHER_CARD, 6.25))

        assert [o["kind"] for _, o in outcomes] == ["configuration_differs"]

    def test_entries_for_other_configurations_are_reported_not_filtered(self) -> None:
        # "ran on a card no entry covers" and "no entry exists at all" are
        # different situations; dropping the non-applicable ones would make
        # them look identical to the caller.
        registry = (_answer(_FULL, 6.25), _answer(_OTHER_CARD, 6.25))

        outcomes = gate_record(registry, _record(_FULL, 6.25))

        assert [o["kind"] for _, o in outcomes] == ["matches", "configuration_differs"]

    def test_entries_with_another_label_are_excluded(self) -> None:
        registry = (_answer(_FULL, 6.25),)

        assert gate_record(registry, _record(_FULL, 6.25, label="different")) == ()

    def test_a_record_with_two_observations_is_refused(self) -> None:
        record = run_record(
            experiment="e",
            label=_LABEL,
            fingerprint=_FULL,
            observations=(
                Observation(name="a", value=1.0),
                Observation(name="b", value=2.0),
            ),
            payload_digest="",
        )

        with pytest.raises(ValueError, match="exactly one observation"):
            gate_record((_answer(_FULL, 1.0),), record)
