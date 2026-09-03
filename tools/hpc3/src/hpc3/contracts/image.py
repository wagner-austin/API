"""Which image a job runs inside, and what it must be able to read.

The run-time half of the image contract. :mod:`hpc3.contracts.image_spec`
holds the build-time half -- what an image IS and how one is produced -- and
the two are separate because they are read by different callers at different
times: a job and a workspace name an image that already exists, while a build
describes one that does not yet.

An image is named by path and pinned by digest, because a path names a file
that can be rebuilt under the same name and "the environment is a directory
nobody edited" is the assumption an image exists to replace.
"""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

SHA256_HEX_LENGTH = 64

_HEX_DIGITS = frozenset("0123456789abcdef")


def is_sha256_hex(value: str) -> bool:
    """Say whether a string is a lowercase-hex sha256 digest.

    The ONE definition, so the bare-digest field and the digest embedded in a
    base image reference cannot come to disagree about what a digest is.

    Args:
        value: Candidate digest, without any ``sha256:`` prefix.

    Returns:
        True when it is exactly 64 lowercase hex characters.
    """
    return len(value) == SHA256_HEX_LENGTH and all(ch in _HEX_DIGITS for ch in value)


HOST_BOUND_ROOTS = frozenset({"pub", "dfs6b", "data", "tmp", "home"})
"""Roots a job mounts its data under, and so must not put an environment in.

A bind mounts a host directory over the same path inside the image, so an
environment beneath one of these is replaced at runtime by the host's copy
and the image's own interpreter ceases to exist inside its own image.
"""


class ImageReference(TypedDict):
    """An image a job runs inside, named by path, pinned by digest, bound.

    Attributes:
        path: Absolute POSIX path to the ``.sif`` on the cluster. A host
            path, unlike :attr:`ImageSpec.env_prefix`, because the file is
            read from the filesystem rather than from inside a container.
        sha256: Digest of the image's exact bytes, lowercase hex. Required
            rather than optional: a path names a file that can be rebuilt in
            place, and "the environment is a directory nobody edited" is the
            assumption an image exists to replace. Recorded into the job's
            provenance so a queued row says which image it is running, not
            merely where it was read from.
        binds: Host directories the payload must be able to read, mounted at
            the same path inside. Required as a field and may be empty, but
            on HPC3 an empty list is almost always wrong: ``/pub`` there is a
            SYMLINK to ``/dfs6b/pub``, and while apptainer binds the BeeGFS
            mounts it does not carry the symlink, so ``/pub/...`` does not
            resolve inside the container at all. Measured 2026-08-25 --
            ``ls /pub/wagnera3`` inside the image reports "No such file or
            directory" while the same path lists fine on the host. A job
            whose corpora, artifacts and model cache all live under ``/pub``
            starts cleanly and then cannot find any of them.
    """

    path: str
    sha256: str
    binds: list[str]


def require_json_object(value: JSONValue, what: str) -> JSONObject:
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


def _require_host_path(obj: JSONObject, key: str) -> str:
    """Read a required absolute POSIX path on the cluster's filesystem.

    Distinct from :func:`_require_container_dir`: this names a file the
    cluster reads, so a bind-mounted root is exactly where it belongs and
    refusing one would refuse every real image.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, not absolute,
            backslashed, or holds a ``..`` segment.
    """
    value = require_str(obj, key)
    if not value.startswith("/"):
        raise JSONTypeError(f"Field '{key}' must be an absolute POSIX path, got {value!r}")
    if "\\" in value:
        raise JSONTypeError(f"Field '{key}' must be forward-slashed, got {value!r}")
    if ".." in value.split("/"):
        raise JSONTypeError(f"Field '{key}' must not contain '..', got {value!r}")
    return value


def _require_digest(obj: JSONObject, key: str) -> str:
    """Read a required lowercase-hex sha256 field.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, not exactly 64
            characters, or holds a non-hex or uppercase character. A re-cased
            or truncated digest no longer names the bytes it came from, so
            comparing against it would pass on the wrong image.
    """
    value = require_str(obj, key)
    if not is_sha256_hex(value):
        raise JSONTypeError(
            f"Field '{key}' must be {SHA256_HEX_LENGTH} lowercase hex characters, got {value!r}"
        )
    return value


def _require_bind_paths(obj: JSONObject, key: str) -> list[str]:
    """Read the host directories a job must be able to read inside the image.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The paths, in order.

    Raises:
        JSONTypeError: If the field is missing, not a list, holds a
            non-string, or holds a path that is not absolute POSIX,
            backslashed, or carries a ``..`` segment. Absent is refused
            rather than defaulted to empty: on HPC3 an unbound job finds
            none of its data, and silently choosing that for a caller who
            did not decide it is how a run completes against nothing.
    """
    raw = require_list(obj, key)
    paths: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str):
            raise JSONTypeError(
                f"Field '{key}[{index}]' must be a string, got {type(item).__name__}"
            )
        if not item.startswith("/"):
            raise JSONTypeError(
                f"Field '{key}[{index}]' must be an absolute POSIX path, got {item!r}"
            )
        if "\\" in item:
            raise JSONTypeError(f"Field '{key}[{index}]' must be forward-slashed, got {item!r}")
        if ".." in item.split("/"):
            raise JSONTypeError(f"Field '{key}[{index}]' must not contain '..', got {item!r}")
        paths.append(item.rstrip("/") or "/")
    return paths


def decode_image_reference(value: JSONValue, key: str) -> ImageReference | None:
    """Decode the image a job runs inside, which may be absent.

    Args:
        value: The field's value. ``None`` means the job runs from a
            directory environment on the host rather than inside an image.
        key: Field name, used in error messages.

    Returns:
        The validated reference, or None.

    Raises:
        JSONTypeError: If the value is neither null nor an object, the path is
            not an absolute POSIX path, or the digest is not 64 lowercase hex
            characters. The digest is required rather than optional because a
            path names a file that can be rebuilt in place, which is the
            mutable-directory problem an image exists to solve.
    """
    if value is None:
        return None
    obj = require_json_object(value, key)
    return ImageReference(
        path=_require_host_path(obj, "path"),
        sha256=_require_digest(obj, "sha256"),
        binds=_require_bind_paths(obj, "binds"),
    )


def encode_image_reference(reference: ImageReference | None) -> JSONValue:
    """Encode an image reference, or null when the job uses no image.

    Args:
        reference: Reference to encode, or None.

    Returns:
        The JSON form, round-trippable through :func:`decode_image_reference`.
    """
    if reference is None:
        return None
    return {
        "path": reference["path"],
        "sha256": reference["sha256"],
        "binds": list(reference["binds"]),
    }


__all__ = [
    "HOST_BOUND_ROOTS",
    "SHA256_HEX_LENGTH",
    "ImageReference",
    "decode_image_reference",
    "encode_image_reference",
    "is_sha256_hex",
    "require_json_object",
]
