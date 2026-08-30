"""Packing a game tree into one file whose bytes are a function of the tree.

WHY ONE FILE AT ALL. ``hpc3``'s staging contract places bare filenames into a
single destination directory and verifies each on both sides
(:mod:`hpc3.contracts.stage`). It cannot express a directory, and a tree of
1,774 files would be 1,774 SSH round trips. An archive is one member of that
contract, and its digest is checked on the cluster exactly as any other file's
would be.

WHY IT IS NOT COMPRESSED. A digest is only an identity if the same input
always produces it. Gzip's output depends on the zlib build that wrote it, so
a compressed archive's digest would be a fact about the machine that packed it
as much as about the tree -- and the whole point is that a third party can
repack and compare. Uncompressed tar has no such dependency: every byte comes
from the file contents and the header rules below. The transfer is larger and
the property is worth more.

WHAT MAKES IT REPRODUCIBLE. A tar header carries a timestamp, an owner and a
mode, none of which say anything about the tree. All three are pinned:
:data:`ARCHIVE_MTIME` zero, uid and gid zero with empty names, and a mode
decided by the rule below. Members are written in the order
:func:`~rw_bot.tree_identity.tree_entries` produced, which is byte-sorted, so
two packs of one tree are the same file.

THE EXECUTABLE BIT IS READ OFF THE BYTES, and it has to come from somewhere:
the tree was assembled on Windows, which has no such bit, so by the time it
reaches here the information is already gone. A hand-written list of which
files are programs would be a list that rots -- the JRE's helper binaries move
between releases. Instead a member is executable exactly when it starts with
the ELF magic number, which is what a Linux program IS. ``jvm-linux/bin/java``
qualifies; ``jvm-linux/lib/classlist``, which sits beside a binary that does,
does not. Shared objects qualify too, and that matches how distributions ship
them.

Without this the archive extracts with ``java`` at mode 644 and every match
dies as permission denied. It has never been seen, because WSL2 mounts
``/mnt/c`` with everything 777 -- so the workstation proof of the Linux
launch ran on a filesystem that hid the problem.
"""

from __future__ import annotations

import tarfile
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from typing_extensions import TypedDict

from rw_bot.tree_identity import TreeEntry

#: What every Linux program starts with.
ELF_MAGIC = b"\x7fELF"

#: Mode a member carries when it is not a program, and when it is.
ARCHIVE_MODE_FILE = 0o644
ARCHIVE_MODE_EXECUTABLE = 0o755

#: Timestamp every member carries. Zero rather than the file's own, because a
#: copy's modification time says when it was copied and nothing about what it
#: holds -- and it would make two packs of one tree different files.
ARCHIVE_MTIME = 0

#: Owner every member carries. Numeric zero with empty names, so extraction as
#: an unprivileged cluster user does not try to chown to somebody who does not
#: exist there.
ARCHIVE_UID = 0
ARCHIVE_GID = 0
ARCHIVE_OWNER = ""

#: How many bytes must be read to recognise a program.
_MAGIC_LENGTH = len(ELF_MAGIC)


class ArchiveResult(TypedDict):
    """What packing produced, as the facts a manifest needs about it.

    Attributes:
        sha256: Digest of the archive's exact bytes, lowercase hex. The value
            a stage manifest names and the cluster recomputes on arrival.
        size_bytes: Length in bytes.
        executables: Paths written with :data:`ARCHIVE_MODE_EXECUTABLE`, in
            member order. Returned rather than merely applied so a caller can
            assert that the runtime it needs is among them -- an archive whose
            ``java`` came out unexecutable is one every match dies on, and it
            is cheaper to learn that here than on a compute node.
    """

    sha256: str
    size_bytes: int
    executables: tuple[str, ...]


def is_program(payload: bytes) -> bool:
    """Report whether a file's leading bytes are those of a Linux program.

    Args:
        payload: The file's contents, or at least its first four bytes.

    Returns:
        True when it begins with :data:`ELF_MAGIC`.
    """
    return payload[:_MAGIC_LENGTH] == ELF_MAGIC


def member_mode(payload: bytes) -> int:
    """Return the mode one member is written with.

    Args:
        payload: The file's contents.

    Returns:
        :data:`ARCHIVE_MODE_EXECUTABLE` for a program, otherwise
        :data:`ARCHIVE_MODE_FILE`.
    """
    return ARCHIVE_MODE_EXECUTABLE if is_program(payload) else ARCHIVE_MODE_FILE


def _member(path: str, payload: bytes) -> tarfile.TarInfo:
    """Build the header one member is written with.

    Args:
        path: The member's name inside the archive.
        payload: Its contents, which decide the mode.

    Returns:
        The header, with every field that does not describe the file pinned.
    """
    info = tarfile.TarInfo(name=path)
    info.size = len(payload)
    info.mtime = ARCHIVE_MTIME
    info.mode = member_mode(payload)
    info.type = tarfile.REGTYPE
    info.uid = ARCHIVE_UID
    info.gid = ARCHIVE_GID
    info.uname = ARCHIVE_OWNER
    info.gname = ARCHIVE_OWNER
    return info


def write_archive(root: Path, entries: Sequence[TreeEntry], destination: Path) -> ArchiveResult:
    """Pack a tree into one reproducible archive.

    No directory members are written. Extraction creates the parents it needs,
    and a directory carries no bytes to identify -- which keeps the archive's
    contents exactly the set
    :func:`~rw_bot.tree_identity.tree_entries` describes, rather than that set
    plus a shape nothing digested. The one visible consequence is that an
    empty directory does not survive the round trip.

    Args:
        root: The tree's root on this machine.
        entries: Its entries, in the order they are to be written. Taken as
            given rather than re-read, so the archive holds exactly what the
            document describes and cannot quietly pack a file the listing
            never saw.
        destination: Where to write the archive.

    Returns:
        Its digest, its length, and the members written executable.

    Raises:
        OSError: When a file cannot be read or the archive cannot be written.
    """
    executables: list[str] = []
    # GNU format because a POSIX ustar header splits a long path across two
    # fields and this tree already runs to 105 characters under
    # jvm-linux/lib/desktop; pax would carry extended records whose contents
    # are a second thing to keep deterministic.
    with tarfile.open(destination, "w", format=tarfile.GNU_FORMAT) as archive:
        for entry in entries:
            payload = (root / entry["path"]).read_bytes()
            info = _member(entry["path"], payload)
            if info.mode == ARCHIVE_MODE_EXECUTABLE:
                executables.append(entry["path"])
            archive.addfile(info, _BytesReader(payload))

    digest = sha256()
    size = 0
    with destination.open("rb") as packed:
        while chunk := packed.read(_READ_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return ArchiveResult(sha256=digest.hexdigest(), size_bytes=size, executables=tuple(executables))


#: How much of the finished archive is digested at a time. It is larger than
#: memory should hold in one piece -- the tree is 311 MB -- so the digest is
#: taken in chunks even though every member was read whole.
_READ_CHUNK = 1024 * 1024


class _BytesReader:
    """A minimal read-only stream over bytes already in memory.

    ``tarfile.addfile`` wants an object with ``read``; handing it an open file
    would mean opening each member twice, once to digest and once to pack, and
    the two reads could disagree. This hands it the bytes that were digested.

    Args:
        payload: The member's contents.
    """

    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        """Return the next bytes of the member.

        Args:
            size: How many to return, or negative for the remainder.

        Returns:
            The bytes, empty once exhausted.
        """
        end = len(self._payload) if size < 0 else min(len(self._payload), self._offset + size)
        chunk = self._payload[self._offset : end]
        self._offset = end
        return chunk


__all__ = [
    "ARCHIVE_GID",
    "ARCHIVE_MODE_EXECUTABLE",
    "ARCHIVE_MODE_FILE",
    "ARCHIVE_MTIME",
    "ARCHIVE_OWNER",
    "ARCHIVE_UID",
    "ELF_MAGIC",
    "ArchiveResult",
    "is_program",
    "member_mode",
    "write_archive",
]
