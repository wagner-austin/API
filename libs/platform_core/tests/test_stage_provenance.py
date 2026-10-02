"""Tests for the record of where staged bytes came from.

The pairs below are the real ones the extraction ablation's arm B carried:
emitted at wiki commit ``176bb8c`` by ``emit_corpus.py`` with the flags shown.
"""

from __future__ import annotations

import pytest

from platform_core.json_utils import JSONTypeError, JSONValue
from platform_core.stage_manifest import decode_stage_manifest
from platform_core.stage_provenance import (
    encode_provenance,
    format_provenance,
    require_provenance,
)

_ARM_B = "07ab4976" + "a" * 56

_PROVENANCE: dict[str, JSONValue] = {
    "wiki_commit": "176bb8c",
    "emitter": "extraction-eval/emit_corpus.py",
    "emitter_flags": "--seed 0 --dilution oscar_en.txt --dilution-ratio 7.0",
}


class TestRequireProvenance:
    def test_it_records_the_pairs_verbatim(self) -> None:
        """Keys are not normalised: this is a record for a human to read."""
        decoded = require_provenance({"p": _PROVENANCE}, "p")
        assert decoded["wiki_commit"] == "176bb8c"
        assert sorted(decoded) == ["emitter", "emitter_flags", "wiki_commit"]

    def test_an_empty_record_is_refused(self) -> None:
        """It would satisfy the requirement while saying nothing."""
        with pytest.raises(JSONTypeError, match="at least one fact"):
            require_provenance({"p": {}}, "p")

    def test_a_missing_record_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            require_provenance({}, "p")

    def test_a_non_string_value_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="maps to int"):
            require_provenance({"p": {"pages": 733}}, "p")

    def test_an_empty_value_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="empty name or value"):
            require_provenance({"p": {"wiki_commit": ""}}, "p")

    def test_an_empty_key_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="empty name or value"):
            require_provenance({"p": {"": "176bb8c"}}, "p")

    def test_a_manifest_without_provenance_is_refused(self) -> None:
        """The whole point: bytes that cannot say where they came from."""
        with pytest.raises(JSONTypeError, match="'provenance' must be a JSON object"):
            decode_stage_manifest(
                {
                    "destination": "/pub/x",
                    "files": [{"name": "a.txt", "sha256": _ARM_B, "size_bytes": 1}],
                }
            )


class TestEncodeProvenance:
    def test_it_round_trips_every_pair_unchanged(self) -> None:
        decoded = require_provenance({"p": _PROVENANCE}, "p")
        assert encode_provenance(decoded) == _PROVENANCE

    def test_the_encoding_is_a_copy_not_the_record(self) -> None:
        """A caller extending the JSON must not edit the decoded record."""
        decoded = require_provenance({"p": _PROVENANCE}, "p")
        encoded = encode_provenance(decoded)
        encoded["extra"] = "x"
        assert "extra" not in decoded


class TestFormatProvenance:
    def test_it_formats_in_a_stable_order(self) -> None:
        """Two runs of one staging must produce the same line."""
        formatted = format_provenance(require_provenance({"p": _PROVENANCE}, "p"))
        assert formatted.startswith("emitter=extraction-eval/emit_corpus.py emitter_flags=")
        assert formatted.endswith("wiki_commit=176bb8c")
