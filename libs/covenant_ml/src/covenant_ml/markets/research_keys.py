"""Evaluator-owned partition identities, acquired only when scoring authorizes them.

Proposer processes must have neither this directory mounted nor the evaluator's
OS credentials. File permissions do not isolate another process of the same user.
The final identity is not required on a validation worker.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

from platform_core.json_utils import JSONObject, require_int, require_str
from typing_extensions import ReadOnly, TypedDict

from . import _test_hooks_encryption as hooks


class ResearchAccess(TypedDict):
    """Private controller configuration containing public recipients, never secrets."""

    directory: ReadOnly[str]
    validation_recipient: ReadOnly[str]
    final_recipient: ReadOnly[str]
    max_plaintext_bytes: ReadOnly[int]


def encode_research_access(value: ResearchAccess) -> JSONObject:
    """Encode the evaluator's paths, public identities and storage budget."""
    return {
        "directory": value["directory"],
        "validation_recipient": value["validation_recipient"],
        "final_recipient": value["final_recipient"],
        "max_plaintext_bytes": value["max_plaintext_bytes"],
    }


def decode_research_access(document: JSONObject) -> ResearchAccess:
    """Require an absolute key directory and distinct, valid public recipients.

    Args:
        document: Evaluator configuration, not proposer input.

    Returns:
        Validated settings without opening a secret file.

    Raises:
        ValueError: RESEARCH_KEYS_CONFIG for invalid paths, identities or budget.
        TypeError: Wrongly typed configuration fields.
    """
    if set(document) != set(ResearchAccess.__annotations__):
        raise ValueError("RESEARCH_KEYS_CONFIG: exact access fields required")
    directory = require_str(document, "directory")
    validation = require_str(document, "validation_recipient")
    final = require_str(document, "final_recipient")
    limit = require_int(document, "max_plaintext_bytes")
    if not Path(directory).is_absolute() or ".." in Path(directory).parts:
        raise ValueError("RESEARCH_KEYS_CONFIG: require an absolute non-traversing directory")
    if validation == final or limit < 1:
        raise ValueError("RESEARCH_KEYS_CONFIG: require distinct recipients and a positive limit")
    hooks.x25519.Recipient.from_str(validation)
    hooks.x25519.Recipient.from_str(final)
    return ResearchAccess(
        directory=directory,
        validation_recipient=validation,
        final_recipient=final,
        max_plaintext_bytes=limit,
    )


class EvaluatorKeys:
    """Read a partition's key lazily from an evaluator-owned private directory."""

    def __init__(self, access: ResearchAccess) -> None:
        self.access = decode_research_access(encode_research_access(access))

    def acquire(self, partition: str) -> hooks.AgeIdentity:
        """Read one private identity; final callers must commit release before this call.

        Args:
            partition: Explicit train-validation or final authorization.

        Returns:
            In-memory identity matching the configured public recipient.

        Raises:
            ValueError: RESEARCH_KEYS_ACCESS for permissive, symlinked or mismatched keys.
            OSError: Absent or inaccessible key; never retry with another partition.
        """
        if partition not in {"train-validation", "final"}:
            raise ValueError("RESEARCH_KEYS_ACCESS: unknown partition")
        root = Path(self.access["directory"])
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError("RESEARCH_KEYS_ACCESS: require a private, non-symlink directory")
        expected = (
            self.access["validation_recipient"]
            if partition == "train-validation"
            else self.access["final_recipient"]
        )
        key_path = root / (partition + ".agekey")
        descriptor = os.open(key_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r", encoding="ascii") as source:
            details = os.fstat(source.fileno())
            if not stat.S_ISREG(details.st_mode) or details.st_mode & 0o077:
                raise ValueError("RESEARCH_KEYS_ACCESS: require an evaluator-owned private key")
            secret = source.read(256)
        # Refuse malformed lengths without including private file contents.
        if not secret.startswith("AGE-SECRET-KEY-1") or len(secret.strip()) != 74:
            raise ValueError("RESEARCH_KEYS_ACCESS: invalid private identity encoding")
        identity = hooks.x25519.Identity.from_str(secret.strip())
        if str(identity.to_public()) != expected:
            raise ValueError("RESEARCH_KEYS_ACCESS: private identity differs from its recipient")
        return identity


def create_evaluator_keys(directory: Path, max_plaintext_bytes: int) -> ResearchAccess:
    """Create two private identities exclusively and verify them before returning public settings.

    Args:
        directory: New absolute directory owned by the evaluator, outside proposer mounts.
        max_plaintext_bytes: Positive job storage budget.

    Returns:
        Public settings; no secret is returned, printed or placed in a command argument.

    Raises:
        ValueError: Invalid configuration or failed identity verification.
        OSError: Directory already exists or a key cannot be written; no success is reported.
    """
    validation, final = hooks.x25519.Identity.generate(), hooks.x25519.Identity.generate()
    access = decode_research_access(
        {
            "directory": str(directory),
            "validation_recipient": str(validation.to_public()),
            "final_recipient": str(final.to_public()),
            "max_plaintext_bytes": max_plaintext_bytes,
        }
    )
    directory.mkdir(mode=0o700, exist_ok=False)
    for partition, identity in (("train-validation", validation), ("final", final)):
        descriptor = os.open(
            directory / (partition + ".agekey"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
        with os.fdopen(descriptor, "w", encoding="ascii") as output:
            output.write(str(identity) + "\n")
    keys = EvaluatorKeys(access)
    keys.acquire("train-validation")
    keys.acquire("final")
    return access


__all__ = [
    "EvaluatorKeys",
    "ResearchAccess",
    "create_evaluator_keys",
    "decode_research_access",
    "encode_research_access",
]
