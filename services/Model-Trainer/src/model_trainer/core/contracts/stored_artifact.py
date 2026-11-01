"""What the artifact store says it stored.

This shape was data-bank-api's upload response, imported from
``platform_core.data_bank_protocol`` while the trainer could upload there.
data-bank-api was archived on 2026-11-01 (MCPs board task a9566810), and the
filesystem store every run now writes through is the only producer, so the
record lives with its one owner.
"""

from __future__ import annotations

from typing import TypedDict


class StoredArtifact(TypedDict):
    """One stored tarball.

    Attributes:
        file_id: The tarball's SHA-256 digest, the id it is read back by.
        size: Its size in bytes.
        sha256: The same digest, kept as its own field for the pointers that
            record it.
        content_type: Its media type.
        created_at: When it was stored, or None when the store does not say.
    """

    file_id: str
    size: int
    sha256: str
    content_type: str
    created_at: str | None


__all__ = ["StoredArtifact"]
