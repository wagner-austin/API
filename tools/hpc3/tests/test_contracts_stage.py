"""Tests for the staging contract.

The digest rules carry most of the weight here: a re-cased or truncated
digest would compare unequal against a correct file, and a name carrying a
separator would write outside the destination directory.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.contracts.stage import (
    decode_stage_manifest,
    decode_staged_file,
    encode_stage_manifest,
    encode_staged_file,
)

_DIGEST = "07ab497650962b3311a58efdef1e36dc65cd1f054de4ced0e82c36c0b4d51976"
_OTHER = "4c91fbc143e87711365a982e27c231f15fe8312818797bed85699fa2f6eff13c"


def _file(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid staged-file payload with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        A JSON object ready for decoding.
    """
    base: dict[str, JSONValue] = {"name": "armB.txt", "sha256": _DIGEST, "size_bytes": 12}
    base.update(overrides)
    return base


def _manifest(**overrides: JSONValue) -> dict[str, JSONValue]:
    """Build a valid manifest payload with optional overrides.

    Args:
        **overrides: Fields to replace in the valid baseline.

    Returns:
        A JSON object ready for decoding.
    """
    base: dict[str, JSONValue] = {
        "destination": "/pub/wagnera3/corpora",
        "files": [_file()],
    }
    base.update(overrides)
    return base


class TestStagedFile:
    def test_a_valid_record_round_trips(self) -> None:
        assert encode_staged_file(decode_staged_file(_file())) == _file()

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_staged_file("armB.txt")

    def test_an_uppercase_digest_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_staged_file(_file(sha256=_DIGEST.upper()))

    def test_a_truncated_digest_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_staged_file(_file(sha256=_DIGEST[:63]))

    def test_a_non_hex_digest_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_staged_file(_file(sha256="z" + _DIGEST[1:]))

    def test_an_empty_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_staged_file(_file(name=""))

    def test_a_slashed_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_staged_file(_file(name="sub/armB.txt"))

    def test_a_backslashed_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_staged_file(_file(name="sub\\armB.txt"))

    def test_a_navigation_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_staged_file(_file(name=".."))
        with pytest.raises(JSONTypeError):
            decode_staged_file(_file(name="."))

    def test_a_zero_byte_file_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_staged_file(_file(size_bytes=0))


class TestStageManifest:
    def test_a_valid_manifest_round_trips(self) -> None:
        assert encode_stage_manifest(decode_stage_manifest(_manifest())) == _manifest()

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_stage_manifest([])

    def test_an_empty_file_list_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_stage_manifest(_manifest(files=[]))

    def test_a_repeated_name_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_stage_manifest(_manifest(files=[_file(), _file(sha256=_OTHER)]))

    def test_two_distinct_names_are_admitted(self) -> None:
        decoded = decode_stage_manifest(
            _manifest(files=[_file(), _file(name="armC.txt", sha256=_OTHER)])
        )
        assert [item["name"] for item in decoded["files"]] == ["armB.txt", "armC.txt"]

    def test_a_relative_destination_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_stage_manifest(_manifest(destination="pub/wagnera3"))

    def test_a_windows_destination_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_stage_manifest(_manifest(destination="/pub\\wagnera3"))

    def test_an_escaping_destination_is_refused(self) -> None:
        with pytest.raises(JSONTypeError):
            decode_stage_manifest(_manifest(destination="/pub/../etc"))

    def test_a_trailing_slash_is_normalised_away(self) -> None:
        decoded = decode_stage_manifest(_manifest(destination="/pub/wagnera3/corpora/"))
        assert decoded["destination"] == "/pub/wagnera3/corpora"

    def test_the_root_destination_survives_normalisation(self) -> None:
        assert decode_stage_manifest(_manifest(destination="/"))["destination"] == "/"
