"""Typed evidence records produced by the physics-claim validators."""

from __future__ import annotations

from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    require_int,
    require_str,
)


class ClaimEvidenceDict(TypedDict):
    """Re-derived evidence for one wiki physics claim.

    ``samples`` counts the clean measurement windows found in the
    archive; ``exact`` how many matched the claim's predicted value;
    ``mismatches`` how many contradicted it. ``detail`` is a short
    human line for the audit table.
    """

    claim_id: str
    samples: int
    exact: int
    mismatches: int
    detail: str


def encode_claim_evidence(evidence: ClaimEvidenceDict) -> JSONObject:
    """Encode claim evidence to a JSON-serializable dict.

    Args:
        evidence: Evidence record to encode.

    Returns:
        JSON object with all evidence fields.
    """
    return {
        "claim_id": evidence["claim_id"],
        "samples": evidence["samples"],
        "exact": evidence["exact"],
        "mismatches": evidence["mismatches"],
        "detail": evidence["detail"],
    }


def decode_claim_evidence(data: JSONObject) -> ClaimEvidenceDict:
    """Decode claim evidence from a JSON object with validation.

    Args:
        data: JSON object carrying the evidence fields.

    Returns:
        Validated evidence record.

    Raises:
        JSONTypeError: If a field has the wrong type.
        KeyError: If a field is missing.
    """
    return ClaimEvidenceDict(
        claim_id=require_str(data, "claim_id"),
        samples=require_int(data, "samples"),
        exact=require_int(data, "exact"),
        mismatches=require_int(data, "mismatches"),
        detail=require_str(data, "detail"),
    )


def encode_evidence_list(evidence: list[ClaimEvidenceDict]) -> list[JSONValue]:
    """Encode a list of evidence records.

    Args:
        evidence: Evidence records to encode.

    Returns:
        JSON-serializable list.
    """
    encoded: list[JSONValue] = []
    for record in evidence:
        encoded.append(encode_claim_evidence(record))
    return encoded


__all__ = [
    "ClaimEvidenceDict",
    "decode_claim_evidence",
    "encode_claim_evidence",
    "encode_evidence_list",
]
