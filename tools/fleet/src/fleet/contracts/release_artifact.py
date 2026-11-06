"""The archive a project's check builds as its release, declared in the registry.

MCPs board task 0c8ace8d. osrs-lab's recorder-release gate builds
``recorder/build/libs/osrs-recorder.jar`` and starts it, and loki runs a jar
that gate built. Which commit that jar came from was a session's rebuild and
a sentence in prose, because the gate's verdict line named nothing about the
archive: a packaging-gate build stamps every entry with its packing time, so
the jar's sha256 differs on each build of one tree and no line could name it.

A project now declares its release archive, and the check's build reads it
once the recipe has ended (:mod:`fleet.core.release_artifact`), so the
verdict line says which archive the commit builds as ``artifact=``. The
declaration lives in the registry, which the runner reads at its rolled
commit rather than at the checked sha, so a commit from before the
declaration is read by today's runner the same way.

The path is relative to the staged tree's root, never to the project's
directory, because a gate may package what a sibling directory builds:
recorder-release's recipe builds the jar under ``recorder/``. Its grammar is
one or more ``[A-Za-z0-9._-]`` segments joined by ``/``, with no ``..``, so it
can be quoted into the build's ``sh`` and printed as one field.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Final

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, require_str
from typing_extensions import ReadOnly, TypedDict


class ArtifactKind(StrEnum):
    """How an artifact's content is read.

    ZIP: a zip archive (a jar is one), read as its entries' names and CRC-32s,
    which carry no packing time.
    """

    ZIP = "zip"


class ReleaseArtifact(TypedDict):
    """The archive a project's check leaves behind as its release.

    Attributes:
        path: Where the recipe leaves it, relative to the staged tree's root.
        kind: How its content is read.
    """

    path: ReadOnly[str]
    kind: ReadOnly[ArtifactKind]


#: A path the build can quote and the verdict can print as one field.
ARTIFACT_PATH: Final[re.Pattern[str]] = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")


def decode_release_artifact(value: JSONValue) -> ReleaseArtifact | None:
    """Decode a project's optional ``release_artifact``.

    Args:
        value: The key's value, or None when the key is absent.

    Returns:
        The artifact, or None for a project that declares none.

    Raises:
        JSONTypeError: When the value is not an object of exactly ``path``
            and ``kind``, either is not a string (``require_str``), the path
            is outside :data:`ARTIFACT_PATH` or
            climbs out of the tree with ``..``, or when the kind is not an
            :class:`ArtifactKind`: a misspelt declaration would print a
            reading of some other file, or none at all.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"path", "kind"}:
        raise JSONTypeError(
            f"project.release_artifact must be an object of exactly path and kind, got {value!r}"
        )
    path = require_str(value, "path")
    if ARTIFACT_PATH.fullmatch(path) is None or ".." in path.split("/"):
        raise JSONTypeError(
            f"project.release_artifact.path must be a path inside the staged tree, segments of "
            f"[A-Za-z0-9._-] joined by '/' with no '..', got {path!r}"
        )
    kind = require_str(value, "kind")
    if kind not in {member.value for member in ArtifactKind}:
        raise JSONTypeError(
            f"project.release_artifact.kind must be one of "
            f"{sorted(member.value for member in ArtifactKind)}, got {kind!r}"
        )
    return ReleaseArtifact(path=path, kind=ArtifactKind(kind))


def encode_release_artifact(artifact: ReleaseArtifact | None) -> JSONObject | None:
    """Encode a project's ``release_artifact``.

    Args:
        artifact: The artifact, or None.

    Returns:
        Its JSON object, or None for a project that declares none.
    """
    if artifact is None:
        return None
    return {"path": artifact["path"], "kind": artifact["kind"].value}


__all__ = [
    "ARTIFACT_PATH",
    "ArtifactKind",
    "ReleaseArtifact",
    "decode_release_artifact",
    "encode_release_artifact",
]
