"""Stdlib-only receiver shipped verbatim over SSH for private grant delivery.

Entries are emitted as Python literals by the shared staging-contract decoder,
not parsed from remote input. Only the bounded byte stream contains secrets.
"""

from __future__ import annotations

import os
import stat
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO


def verify_private_file(path: Path, digest: str) -> None:
    """Require private regular-file permissions and exact received bytes.

    Args:
        path: Newly received file.
        digest: Expected SHA-256.

    Raises:
        ValueError: PRIVATE_DELIVERY_READBACK for mode, type or digest drift.
    """
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode) or stat.S_IMODE(mode) != 0o600:
        raise ValueError("PRIVATE_DELIVERY_READBACK: require a private regular file")
    if sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError("PRIVATE_DELIVERY_READBACK: digest differs")


def receive_private(
    destination: str, entries: tuple[tuple[str, int, str], ...], source: BinaryIO
) -> None:
    """Verify bounded input, write a new private directory and verify durable files.

    Args:
        destination: Absolute destination beneath an existing private directory.
        entries: Names, lengths and digests validated by StageManifest at the sender.
        source: Secret bytes in the declared order, never command arguments.

    Raises:
        ValueError: PRIVATE_DELIVERY for permissions, byte or readback drift.
        OSError: Existing destination or storage failure; never overwrite or retry.
    """
    root = Path(destination)
    parent = root.parent.lstat()
    if not stat.S_ISDIR(parent.st_mode) or stat.S_IMODE(parent.st_mode) != 0o700:
        raise ValueError("PRIVATE_DELIVERY_PARENT: require a private non-symlink directory")
    size = sum(length for _, length, _ in entries)
    payload = source.read(size + 1)
    if len(payload) != size:
        raise ValueError("PRIVATE_DELIVERY_BYTES: wrong payload length")
    offset = 0
    for _, length, digest in entries:
        if sha256(payload[offset : offset + length]).hexdigest() != digest:
            raise ValueError("PRIVATE_DELIVERY_BYTES: payload digest differs")
        offset += length
    root.mkdir(mode=0o700, exist_ok=False)
    offset = 0
    for name, length, digest in entries:
        path = root / name
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload[offset : offset + length])
            stream.flush()
            os.fsync(stream.fileno())
        offset += length
        verify_private_file(path, digest)
    for directory in (root, root.parent):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        with ExitStack() as storage:
            storage.callback(os.close, descriptor)
            os.fsync(descriptor)


__all__ = ["receive_private", "verify_private_file"]
