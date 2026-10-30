"""Exact encrypted metadata contracts refuse paths and ambiguous identities."""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_dict,
    require_list,
)

from covenant_ml.markets.encrypted_records import decode_encrypted_bundle, encode_encrypted_bundle


def _document() -> JSONObject:
    plain: JSONObject = {"name": "manifest.json", "sha256": "a" * 64, "size_bytes": 10}
    cipher: JSONObject = {"name": "manifest.json.age", "sha256": "b" * 64, "size_bytes": 210}
    return {
        "format": "age-v1",
        "partition": "final",
        "recipient": "age1" + "a" * 58,
        "manifest": plain,
        "files": [{"plaintext": plain, "ciphertext": cipher}],
    }


def test_metadata_codec_preserves_both_identities() -> None:
    document = _document()
    assert encode_encrypted_bundle(decode_encrypted_bundle(document)) == document


@pytest.mark.parametrize(
    "field,value",
    [
        ("extra", True),
        ("format", "plain"),
        ("partition", "train"),
        ("recipient", "AGE-SECRET-KEY-do-not-publish"),
        ("recipient", "age1short"),
        ("files", []),
    ],
)
def test_invalid_bundle_fields(field: str, value: JSONValue) -> None:
    document = _document()
    document[field] = value
    with pytest.raises(ValueError, match="ENCRYPTED_"):
        decode_encrypted_bundle(document)


@pytest.mark.parametrize(
    "mutation", ["extra", "name", "traversal", "newline", "duplicate", "missing", "overlap"]
)
def test_ambiguous_file_evidence(mutation: str) -> None:
    document = _document()
    files = require_list(document, "files")
    first = require_dict({"entry": files[0]}, "entry")
    if mutation == "extra":
        first["secret"] = "forbidden"
    elif mutation == "name":
        require_dict(first, "ciphertext")["name"] = "different.age"
    elif mutation == "traversal":
        require_dict(first, "plaintext")["name"] = "../private"
    elif mutation == "newline":
        require_dict(first, "plaintext")["name"] = "bad\nname"
    elif mutation == "duplicate":
        files.append(first)
    elif mutation == "missing":
        document["manifest"] = {"name": "missing.json", "sha256": "a" * 64, "size_bytes": 10}
    else:
        files.append(
            {
                "plaintext": require_dict(first, "ciphertext"),
                "ciphertext": {
                    "name": "manifest.json.age.age",
                    "sha256": "c" * 64,
                    "size_bytes": 410,
                },
            }
        )
    with pytest.raises((ValueError, JSONTypeError)):
        decode_encrypted_bundle(document)
