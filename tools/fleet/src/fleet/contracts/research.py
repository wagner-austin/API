"""Frozen private controller references on the shared dispatch queue."""

from __future__ import annotations

import re
from enum import StrEnum

from platform_core.json_utils import JSONObject, JSONValue, narrow_json_to_dict, require_str
from platform_core.members import require_member
from typing_extensions import ReadOnly, TypedDict


class ResearchBackend(StrEnum):
    """Maintained native controller families."""

    HPC3 = "hpc3"
    WRDS = "wrds"


class ResearchRequest(TypedDict):
    """Exact private specification and explicitly named attempt."""

    backend: ReadOnly[ResearchBackend]
    spec_sha256: ReadOnly[str]
    attempt: ReadOnly[str]


def require_digest(value: str) -> str:
    """Require a lowercase SHA-256 identity.

    Args:
        value: Untrusted digest.

    Returns:
        Validated digest.

    Raises:
        ValueError: RESEARCH_REFERENCE_DIGEST for malformed identities.
    """
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("RESEARCH_REFERENCE_DIGEST: expected 64 lowercase hexadecimal characters")
    return value


def decode_research_request(value: JSONValue) -> ResearchRequest:
    """Decode the queue's exact reference shape; reject paths and executable fields.

    Args:
        value: Untrusted wire object.

    Returns:
        Validated reference.

    Raises:
        ValueError: Unexpected fields, backend, digest or unsafe attempt.
    """
    row = narrow_json_to_dict(value)
    if set(row) != {"backend", "specSha256", "attempt"}:
        raise ValueError("RESEARCH_REFERENCE_FIELDS: expected backend, specSha256 and attempt")
    attempt = require_str(row, "attempt")
    if re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", attempt) is None:
        raise ValueError("RESEARCH_REFERENCE_ATTEMPT: expected one bounded identifier")
    return ResearchRequest(
        backend=require_member(row, "backend", ResearchBackend),
        spec_sha256=require_digest(require_str(row, "specSha256")),
        attempt=attempt,
    )


def encode_research_request(request: ResearchRequest) -> JSONObject:
    """Encode the same wire fields as the canonical queue writer.

    Args:
        request: Validated reference.

    Returns:
        JSON-ready object.
    """
    return {
        "backend": request["backend"].value,
        "specSha256": request["spec_sha256"],
        "attempt": request["attempt"],
    }


__all__ = [
    "ResearchBackend",
    "ResearchRequest",
    "decode_research_request",
    "encode_research_request",
    "require_digest",
]
