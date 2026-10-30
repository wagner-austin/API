"""Private single-use final grants; the research journal remains release authority.

A grant is delivered through evaluator-only storage, never normal data staging.
Its metadata binds exact request bytes and final ciphertext. A durable exclusive
marker consumes it before its key is opened; failures never restore permission.
The evaluator's own OS identity and administrators remain privileged.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import JSONObject, dump_json_str, require_int, require_str
from typing_extensions import ReadOnly, TypedDict

from ..evaluation.evidence import read_evidence, require_evidence_digest, require_private_directory
from . import _test_hooks_encryption as hooks
from .research_keys import read_private_identity
from .research_trial_records import ResearchTrial, encode_research_trial
from .validation import read_document


class FinalGrant(TypedDict):
    """Public binding of one private key to the evaluator's released execution."""

    request_digest: ReadOnly[str]
    manifest_digest: ReadOnly[str]
    recipient: ReadOnly[str]
    max_plaintext_bytes: ReadOnly[int]


def encode_final_grant(value: FinalGrant) -> JSONObject:
    """Encode grant metadata only; no private identity is part of this record."""
    return {
        "request_digest": value["request_digest"],
        "manifest_digest": value["manifest_digest"],
        "recipient": value["recipient"],
        "max_plaintext_bytes": value["max_plaintext_bytes"],
    }


def decode_final_grant(document: JSONObject) -> FinalGrant:
    """Validate exact digests, public recipient and positive plaintext budget.

    Args:
        document: Metadata from evaluator-private grant storage.

    Returns:
        Strict public metadata; no key is read.

    Raises:
        ValueError: RESEARCH_FINAL_GRANT for field or budget errors; digest and
            recipient validation errors propagate from their shared validators.
        TypeError: Invalid field types.
    """
    if set(document) != set(FinalGrant.__annotations__):
        raise ValueError("RESEARCH_FINAL_GRANT: exact metadata fields required")
    limit = require_int(document, "max_plaintext_bytes")
    recipient = require_str(document, "recipient")
    if limit < 1:
        raise ValueError("RESEARCH_FINAL_GRANT: positive plaintext budget required")
    hooks.x25519.Recipient.from_str(recipient)
    return FinalGrant(
        request_digest=require_evidence_digest(require_str(document, "request_digest")),
        manifest_digest=require_evidence_digest(require_str(document, "manifest_digest")),
        recipient=recipient,
        max_plaintext_bytes=limit,
    )


def write_final_grant(directory: Path, grant: FinalGrant, identity: hooks.AgeIdentity) -> None:
    """Write an exclusive private grant after the controller commits release.

    Args:
        directory: New evaluator-private directory, not mounted into proposer sandboxes.
        grant: Released request/dataset binding.
        identity: Final identity acquired only after the journal's release.

    Raises:
        ValueError: RESEARCH_FINAL_GRANT for key drift; storage validation propagates.
        OSError: Existing directory or failed write; release remains consumed.
    """
    grant = decode_final_grant(encode_final_grant(grant))
    if str(identity.to_public()) != grant["recipient"]:
        raise ValueError("RESEARCH_FINAL_GRANT: identity differs from released recipient")
    directory.mkdir(mode=0o700, exist_ok=False)
    require_private_directory(directory)
    for name, payload in (
        ("final.agekey", str(identity) + "\n"),
        ("release.json", dump_json_str(encode_final_grant(grant))),
    ):
        descriptor = os.open(directory / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    read_evidence(
        directory,
        "release.json",
        sha256(dump_json_str(encode_final_grant(grant)).encode()).hexdigest(),
    )
    read_private_identity(directory / "final.agekey", grant["recipient"])


def consume_final_grant(
    directory: Path, trial: ResearchTrial, manifest: Path
) -> tuple[FinalGrant, hooks.AgeIdentity]:
    """Spend a private grant once and remove its key before returning any identity.

    Args:
        directory: Private delivered grant; retained consumption marker must not be reset.
        trial: Exact released compute request.
        manifest: Final encrypted manifest, read for its digest only.

    Returns:
        Verified public metadata and in-memory final identity.

    Raises:
        ValueError: RESEARCH_FINAL_GRANT for a different request or final dataset.
        FileExistsError: Grant already consumed, including concurrent executions.
        OSError: Missing/private-storage failure; consumed grants are never restored.
    """
    require_private_directory(directory)
    grant = decode_final_grant(read_document(directory / "release.json"))
    if (
        sha256(dump_json_str(encode_research_trial(trial)).encode()).hexdigest()
        != grant["request_digest"]
        or sha256(manifest.read_bytes()).hexdigest() != grant["manifest_digest"]
    ):
        raise ValueError("RESEARCH_FINAL_GRANT: request or final bundle differs from release")
    descriptor = os.open(directory / "consumed", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as marker:
        marker.write(b"released\n")
        marker.flush()
        os.fsync(marker.fileno())
    # Persist the marker's directory entry before opening any secret. A restart
    # cannot turn a completed consumption into an unused grant.
    parent = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    with ExitStack() as storage:
        storage.callback(os.close, parent)
        os.fsync(parent)
    identity = read_private_identity(directory / "final.agekey", grant["recipient"])
    (directory / "final.agekey").unlink()
    return grant, identity


__all__ = [
    "FinalGrant",
    "consume_final_grant",
    "decode_final_grant",
    "encode_final_grant",
    "write_final_grant",
]
