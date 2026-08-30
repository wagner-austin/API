"""What a directory of game content IS, as a digest of the bytes in it.

A jar digest identifies the engine's CODE. It says nothing about the runtime
that executes it or the data it reads, and both decide the numbers:

* The bundled JVM is a different major version per platform -- Java 8 in the
  Linux depot, Java 13 in the Windows one -- so two runs of one arm on two
  platforms execute the same bytecode on different virtual machines.
* ``assets/`` holds the maps, mods and unit definitions the simulation reads.
  A map missing from a clone sent the engine to its boot sandbox and voided a
  whole batch family with a jar digest that matched throughout.

Neither is one file, so neither can be recorded the way the jar is. This
module records a TREE: every file under a root, digested individually, and the
listing itself digested to a single value.

THE DIGEST IS CHECKABLE WITHOUT THIS CODE, and that is a deliberate property
rather than a coincidence of format. :func:`render_listing` emits exactly what
``sha256sum --text`` emits, in exactly the order ``LC_ALL=C sort`` produces,
so the value :func:`tree_digest` returns is reproducible with coreutils
alone::

    cd <root> && find . -type f -printf '%P\\n' | LC_ALL=C sort \\
        | xargs -d '\\n' sha256sum --text | sha256sum

``--text`` is not decoration. GNU coreutils defaults to text mode on Linux and
to BINARY mode on Windows, and the two spell the separator differently -- two
spaces against a space and an asterisk -- so the same tree digests to two
different values depending on which machine ran the check. Stating the mode
makes the command mean one thing everywhere, which is the only way a
cross-platform record is worth anything. Measured 2026-08-29, after the
command written here without it disagreed with this module on the first run.

A record only this package can verify is a record nobody checks.

WHAT IS NOT AN ENTRY. Directories are not, because a directory holds no bytes;
the consequence is that an empty directory is invisible here, which is the
right trade for a content digest and worth knowing when comparing against an
archive that preserved one. Symlinks are refused outright rather than followed
or skipped: the trees this describes have none, an archive resolves them
differently from a filesystem walk, and a silently-skipped broken link would
subtract a file from the identity without subtracting it from the tree.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from rw_bot import RwBotError

#: Length of a hex-rendered SHA-256, and the digits one may hold. Checked on
#: decode because a re-cased or truncated digest no longer names the bytes it
#: came from, so a comparison against it would pass on the wrong file.
SHA256_HEX_LENGTH = 64

_HEX_DIGITS = frozenset("0123456789abcdef")

#: What ``sha256sum`` puts between a digest and the path it belongs to. Two
#: spaces, which is its text-mode separator; the binary-mode form is a space
#: and an asterisk and would not compare equal.
LISTING_SEPARATOR = "  "

_NO_SUCH_TREE = "RW-TREE-001"
_EMPTY_TREE = "RW-TREE-002"
_LINK_IN_TREE = "RW-TREE-003"


class TreeIdentityError(RwBotError):
    """A tree could not be reduced to an identity.

    Args:
        code: Stable machine-readable identifier.
        message: Human-readable description of what was wrong with the tree.
    """


class TreeEntry(TypedDict):
    """One file in a tree, and the bytes that identify it.

    Attributes:
        path: Location under the tree's root, forward-slashed and carrying no
            leading ``./``. Relative because the identity of a tree must not
            change when it is moved, which is exactly what staging does to it.
        sha256: Digest of the file's exact bytes, lowercase hex.
        size_bytes: Length in bytes. Recorded alongside the digest because it
            is what makes a listing readable by a human deciding whether a
            difference is a truncation or a change, and it costs nothing --
            the bytes were already read.
    """

    path: str
    sha256: str
    size_bytes: int


def _require_digest(obj: JSONObject, key: str) -> str:
    """Read a required lowercase-hex sha256 field.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, not exactly
            :data:`SHA256_HEX_LENGTH` characters, or holds a non-hex or
            uppercase character.
    """
    value = require_str(obj, key)
    if len(value) != SHA256_HEX_LENGTH or any(ch not in _HEX_DIGITS for ch in value):
        raise JSONTypeError(
            f"Field '{key}' must be {SHA256_HEX_LENGTH} lowercase hex characters, got {value!r}"
        )
    return value


def _require_relative_posix_path(obj: JSONObject, key: str) -> str:
    """Read a required forward-slashed path that stays inside its own tree.

    Args:
        obj: Object being decoded.
        key: Field name.

    Returns:
        The field's value.

    Raises:
        JSONTypeError: If the field is missing, not a string, empty,
            backslashed, absolute, or holds a ``..`` segment. An entry that
            escapes its root does not describe the tree it is filed under,
            and something unpacking the listing would write outside it.
    """
    value = require_str(obj, key)
    if value == "":
        raise JSONTypeError(f"Field '{key}' must not be empty")
    if "\\" in value:
        raise JSONTypeError(f"Field '{key}' must be forward-slashed, got {value!r}")
    if value.startswith("/"):
        raise JSONTypeError(f"Field '{key}' must be relative to the tree root, got {value!r}")
    if ".." in value.split("/"):
        raise JSONTypeError(f"Field '{key}' must not contain '..', got {value!r}")
    return value


def encode_tree_entry(entry: TreeEntry) -> JSONObject:
    """Encode a tree entry to a JSON object.

    Args:
        entry: Entry to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "path": entry["path"],
        "sha256": entry["sha256"],
        "size_bytes": entry["size_bytes"],
    }


def decode_tree_entry(value: JSONValue) -> TreeEntry:
    """Decode and validate a JSON value into a tree entry.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        Validated entry.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the path escapes its root, the digest is malformed, or
            the size is negative. Zero is allowed where the staging contract
            refuses it: a game tree really does carry empty marker files, and
            an empty file is a fact about the tree rather than a failed
            transfer.
    """
    obj = narrow_json_to_dict(value)
    size = require_int(obj, "size_bytes")
    if size < 0:
        raise JSONTypeError(f"Field 'size_bytes' must not be negative, got {size}")
    return TreeEntry(
        path=_require_relative_posix_path(obj, "path"),
        sha256=_require_digest(obj, "sha256"),
        size_bytes=size,
    )


def _sort_key(entry: TreeEntry) -> bytes:
    """Return the value a listing is ordered by.

    Ordered on the path's UTF-8 BYTES rather than on its characters, so the
    order matches ``LC_ALL=C sort`` and the digest stays reproducible with
    coreutils. The two agree for ASCII and diverge the moment a translation
    file or a mod carries an accented name, which this tree's ``assets/``
    is exactly the kind of place to hold.

    Args:
        entry: The entry to order.

    Returns:
        Its path, UTF-8 encoded.
    """
    return entry["path"].encode("utf-8")


def tree_entries(root: Path) -> tuple[TreeEntry, ...]:
    """Read every file under a root as an entry.

    Args:
        root: Directory to walk.

    Returns:
        One entry per file, ordered by :func:`_sort_key`.

    Raises:
        TreeIdentityError: ``RW-TREE-001`` when the root is not a directory,
            ``RW-TREE-002`` when it holds no file at all, or ``RW-TREE-003``
            when it holds a symbolic link. An absent root is refused rather
            than digested as empty, because "the assets are gone" and "the
            assets are unchanged" must not produce the same value.
        OSError: When a file under the root cannot be read.
    """
    if not root.is_dir():
        raise TreeIdentityError(
            _NO_SUCH_TREE,
            f"{root} is not a directory, so it has no contents to identify; a run "
            "recorded against an absent tree would claim a fingerprint it never had",
        )
    entries: list[TreeEntry] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise TreeIdentityError(
                _LINK_IN_TREE,
                f"{path} is a symbolic link, which this identity does not describe: an "
                "archive resolves a link differently from a filesystem walk, so the "
                "digest would depend on which of the two produced it",
            )
        if path.is_dir():
            continue
        payload = path.read_bytes()
        entries.append(
            TreeEntry(
                path=path.relative_to(root).as_posix(),
                sha256=sha256(payload).hexdigest(),
                size_bytes=len(payload),
            )
        )
    if entries == []:
        raise TreeIdentityError(
            _EMPTY_TREE,
            f"{root} holds no file, so there is nothing to identify it by; an empty "
            "tree and a complete one must not digest alike",
        )
    return tuple(sorted(entries, key=_sort_key))


def render_listing(entries: Sequence[TreeEntry]) -> tuple[str, ...]:
    """Render entries as the lines ``sha256sum`` would print for them.

    Args:
        entries: The tree's entries, already in listing order.

    Returns:
        One ``<digest>  <path>`` line per entry, in the order given. The
        caller is trusted with the order because :func:`tree_entries` is the
        only thing that produces it and it produces it sorted; re-sorting
        here would hide a caller that had assembled a listing by hand.
    """
    return tuple(f"{entry['sha256']}{LISTING_SEPARATOR}{entry['path']}" for entry in entries)


def digest_record(
    headers: Sequence[str], archive: str, archive_sha256: str, entries: Sequence[TreeEntry]
) -> tuple[str, ...]:
    """Render the published record ``hpc3-stage --expect-from`` is held to.

    Shared by every tree this project stages, because the SHAPE is the same
    whatever the subject: some header lines a person reads, the archive's own
    digest -- which is what a stage manifest is actually checked against --
    and the per-file listing that says which file moved when the tree digest
    changes. Only the headers differ, so only the headers are the caller's.

    ``hpc3.core.expected`` reads it loosely: every 64-character lowercase hex
    token anywhere in the text counts. So this is written for the person and
    parsed by the tool, and the ``#`` prefixes are decoration rather than
    syntax.

    Args:
        headers: Lines describing what this tree IS, each already prefixed
            with ``#``.
        archive: Filename of the archive the tree is staged as.
        archive_sha256: Its digest.
        entries: The tree's entries, in listing order.

    Returns:
        The headers, then the archive's line, then one line per entry.

    Raises:
        TreeIdentityError: ``RW-TREE-002`` when there are no entries. A record
            that vouches for no file vouches for nothing.
    """
    if len(entries) == 0:
        raise TreeIdentityError(
            _EMPTY_TREE, "a record naming no entry vouches for nothing that could be staged"
        )
    return (
        *headers,
        f"{archive_sha256}{LISTING_SEPARATOR}{archive}",
        *render_listing(entries),
    )


def tree_digest(entries: Sequence[TreeEntry]) -> str:
    """Reduce a tree's entries to the one value that identifies it.

    Args:
        entries: The tree's entries, in listing order.

    Returns:
        The SHA-256 of the rendered listing, lowercase hex. Every line is
        newline-terminated including the last, which is what ``sha256sum``
        writes and therefore what piping its output into ``sha256sum`` again
        digests.

    Raises:
        TreeIdentityError: ``RW-TREE-002`` when there are no entries. Digesting
            the empty string would give every absent tree one shared,
            plausible-looking value.
    """
    if len(entries) == 0:
        raise TreeIdentityError(
            _EMPTY_TREE,
            "a tree with no entries has no identity; digesting nothing would give "
            "every absent tree the same plausible-looking value",
        )
    listing = "".join(f"{line}\n" for line in render_listing(entries))
    return sha256(listing.encode("utf-8")).hexdigest()


def digest_tree(root: Path) -> str:
    """Read a tree and return the one value identifying it.

    The two steps kept separate above, joined for the callers that want only
    the answer. Not a rename of either: it is the composition, and the halves
    stay reachable because a caller that must FILE the listing -- a staging
    record does -- needs the entries as well as the value.

    Args:
        root: Directory to identify.

    Returns:
        The tree's digest.

    Raises:
        TreeIdentityError: ``RW-TREE-001`` when the root is not a directory,
            ``RW-TREE-002`` when it is empty, ``RW-TREE-003`` on a symlink.
        OSError: When a file under the root cannot be read.
    """
    return tree_digest(tree_entries(root))


__all__ = [
    "LISTING_SEPARATOR",
    "SHA256_HEX_LENGTH",
    "TreeEntry",
    "TreeIdentityError",
    "decode_tree_entry",
    "digest_record",
    "digest_tree",
    "encode_tree_entry",
    "render_listing",
    "tree_digest",
    "tree_entries",
]
