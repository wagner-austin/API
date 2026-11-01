"""What the artifact store says it stored.

This shape was data-bank-api's upload response, imported from
``platform_core.data_bank_protocol`` while the trainer could upload there.
data-bank-api was archived on 2026-11-01 (MCPs board task a9566810), and the
filesystem store every run now writes through is the only producer, so the
record lives with its one owner and carries only what a caller reads: the
response's ``sha256`` repeated ``file_id``, and its ``content_type`` and
``created_at`` were a constant and a None no caller looked at.

It stays inside the process -- only ``file_id`` reaches Redis, as a plain
string -- so it has no encoder or decoder, like the other in-process outcomes
under ``core.contracts``.
"""

from __future__ import annotations

from typing import TypedDict


class StoredArtifact(TypedDict):
    """One stored tarball.

    Attributes:
        file_id: The tarball's SHA-256 digest, the id it is read back by.
        size: Its size in bytes.
    """

    file_id: str
    size: int


__all__ = ["StoredArtifact"]
