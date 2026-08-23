"""The staging contract: which bytes were shipped, and how that is proved.

A corpus that reaches the cluster with the wrong contents produces a run that
completes, reports plausible numbers, and is comparable to nothing. Nothing
fails; the result is simply void. That failure mode is the reason this module
exists.

The concrete instance: the extraction ablation's arms A-E were emitted over
733 wiki pages and the wiki now holds 773. Re-emitting from the current tree
yields a different corpus with a different digest and no error anywhere. A
manifest pins the digest the run is entitled to, and staging verifies the
bytes on both sides of the transfer against it.

Both sides is deliberate. A local check proves the emitter produced the right
file; it says nothing about what arrived. Only a digest computed on the
cluster proves what the job will actually read.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONTypeError,
    JSONValue,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

SHA256_HEX_LENGTH = 64

_HEX_DIGITS = frozenset("0123456789abcdef")


class StagedFile(TypedDict):
    """One file the cluster must hold, and the digest that identifies it.

    Attributes:
        name: Filename as it appears on both sides. Never a path: staging
            writes into one destination directory, and a name carrying a
            separator would silently escape it.
        sha256: Digest of the file's exact bytes, lowercase hex. This is the
            identity the run is entitled to, not a checksum of convenience.
        size_bytes: Length in bytes. Checked before the digest because a
            truncated transfer is cheaper to detect by length than by hash,
            and because a size match with a digest mismatch means corruption
            rather than truncation.
    """

    name: str
    sha256: str
    size_bytes: int


class StageManifest(TypedDict):
    """Every file one staging operation must place, with its destination.

    Attributes:
        destination: Absolute directory on the cluster receiving the files.
        files: The files to place. Never empty -- a manifest describing no
            file describes no staging.
    """

    destination: str
    files: list[StagedFile]


def _require_object(value: JSONValue, what: str) -> dict[str, JSONValue]:
    """Narrow a decoded JSON value to an object.

    Args:
        value: Value produced by the JSON loader.
        what: Name of the thing being decoded, used in the error message.

    Returns:
        The value as a JSON object.

    Raises:
        JSONTypeError: If the value is not a JSON object.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"{what} must be a JSON object, got {type(value).__name__}")
    return value


def _require_digest(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required lowercase-hex sha256 field.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, not exactly 64
            characters, or holds a non-hex or uppercase character. A
            re-cased or truncated digest no longer names the bytes it came
            from, so comparing against it would pass on the wrong file.
    """
    value = require_str(obj, key)
    if len(value) != SHA256_HEX_LENGTH or any(ch not in _HEX_DIGITS for ch in value):
        raise JSONTypeError(
            f"Field '{key}' must be {SHA256_HEX_LENGTH} lowercase hex characters, got {value!r}"
        )
    return value


def _require_bare_filename(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required filename field carrying no path separator.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, empty, holds a
            separator, or is a relative-navigation name. Staging joins this
            onto a destination directory, so a separator or a ``..`` would
            write outside it.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    if "/" in value or "\\" in value:
        raise JSONTypeError(f"Field '{key}' must not contain a path separator, got {value!r}")
    if value in (".", ".."):
        raise JSONTypeError(f"Field '{key}' must name a file, got {value!r}")
    return value


def _require_absolute_posix_dir(obj: dict[str, JSONValue], key: str) -> str:
    """Read a required absolute POSIX directory field.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value, without a trailing slash.

    Raises:
        JSONTypeError: If the field is missing, not a string, not absolute,
            backslashed, or holds a ``..`` segment. The cluster is POSIX and
            the destination is joined with filenames, so a relative or
            escaping destination would place files somewhere unintended.
    """
    value = require_str(obj, key)
    if not value.startswith("/"):
        raise JSONTypeError(f"Field '{key}' must be an absolute POSIX path, got {value!r}")
    if "\\" in value:
        raise JSONTypeError(f"Field '{key}' must be forward-slashed, got {value!r}")
    if ".." in value.split("/"):
        raise JSONTypeError(f"Field '{key}' must not contain '..', got {value!r}")
    return value.rstrip("/") or "/"


def encode_staged_file(staged: StagedFile) -> dict[str, JSONValue]:
    """Encode a staged-file record to a JSON object.

    Args:
        staged: Record to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "name": staged["name"],
        "sha256": staged["sha256"],
        "size_bytes": staged["size_bytes"],
    }


def decode_staged_file(value: JSONValue) -> StagedFile:
    """Decode and validate a JSON value into a staged-file record.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated record.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the name carries a separator, the digest is malformed,
            or the size is below one. A zero-byte file has nothing to verify
            and nothing to train on.
    """
    obj = _require_object(value, "staged file")
    size = require_int(obj, "size_bytes")
    if size < 1:
        raise JSONTypeError(f"Field 'size_bytes' must be at least 1, got {size}")
    return StagedFile(
        name=_require_bare_filename(obj, "name"),
        sha256=_require_digest(obj, "sha256"),
        size_bytes=size,
    )


def encode_stage_manifest(manifest: StageManifest) -> dict[str, JSONValue]:
    """Encode a stage manifest to a JSON object.

    Args:
        manifest: Manifest to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    files: list[JSONValue] = [encode_staged_file(item) for item in manifest["files"]]
    return {
        "destination": manifest["destination"],
        "files": files,
    }


def decode_stage_manifest(value: JSONValue) -> StageManifest:
    """Decode and validate a JSON value into a stage manifest.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated manifest.

    Raises:
        JSONTypeError: If the value is not an object, the destination is not
            an absolute POSIX directory, the file list is missing or empty,
            a record is invalid, or two records share a name. Two entries
            under one name disagree about which bytes belong there, and the
            second write would silently win.
    """
    obj = _require_object(value, "stage manifest")
    raw = require_list(obj, "files")
    if raw == []:
        raise JSONTypeError("Field 'files' must not be empty")
    files = [decode_staged_file(item) for item in raw]
    names = [item["name"] for item in files]
    if len(set(names)) != len(names):
        raise JSONTypeError(f"Field 'files' must not repeat a name, got {names}")
    return StageManifest(
        destination=_require_absolute_posix_dir(obj, "destination"),
        files=files,
    )


__all__ = [
    "SHA256_HEX_LENGTH",
    "StageManifest",
    "StagedFile",
    "decode_stage_manifest",
    "decode_staged_file",
    "encode_stage_manifest",
    "encode_staged_file",
]
