"""Files this package LIFTS from the MCPs workspace, pinned by hash (MCPs board task 1b152218).

The five-minute rule on ``make check`` (``check_budget.py``) is MCPs'
``packages/maketools/src/maketools/check_budget.py``, byte for byte. It is a
lift rather than a call into MCPs' published maketools, which is how
corvis-stick and chat reach the same command, because this repository is
public, MCPs is private, and this repository's CI has no MCPs checkout
beside it to call. A lift that nothing checks is a fork waiting to happen,
so ``lift-lock.json`` beside this package's Makefile pins each lifted file
to the MCPs revision it came from and to the sha256 of its bytes there.

TWO COMMANDS.

``lift-check`` asks whether each lifted file still holds the bytes its pin
names. It reads only this checkout, so it runs in ``make lint`` here and
in CI alike, and a hand edit to a lifted file fails it by name.

``lift-refresh <mcps> <revision>`` writes each lifted file from that
commit of the MCPs repository at ``<mcps>``, read with ``git cat-file``
from its object store rather than from its working tree (which other
sessions edit), and re-pins the lock to the full commit and the bytes it
wrote. It is the only way the lock moves, so every pin is a digest of the
upstream blob by construction and never a number typed in.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final, TypedDict

from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_list,
    require_str,
)

from maketools import _test_hooks

#: The lock's file name, at this package's root.
LIFT_LOCK_NAME: Final[str] = "lift-lock.json"


class LiftEntry(TypedDict):
    """One lifted file.

    Attributes:
        path: Where it lives here, relative to this package's root.
        source: Where it lives in MCPs, relative to that repository's root.
        sha256: The hex sha256 of its bytes at the pinned revision.
    """

    path: str
    source: str
    sha256: str


class LiftLock(TypedDict):
    """The whole lock.

    Attributes:
        remote: The MCPs repository the files come from.
        revision: The MCPs commit every entry was taken at.
        entries: The lifted files.
    """

    remote: str
    revision: str
    entries: list[LiftEntry]


def decode_lift_entry(value: JSONValue) -> LiftEntry:
    """Narrow one decoded entry.

    Args:
        value: The entry as JSON decoded it.

    Returns:
        The entry.

    Raises:
        JSONTypeError: When it is not an object of three strings.
    """
    entry = narrow_json_to_dict(value)
    return LiftEntry(
        path=require_str(entry, "path"),
        source=require_str(entry, "source"),
        sha256=require_str(entry, "sha256"),
    )


def decode_lift_lock(text: str) -> LiftLock:
    """Read the lock's text.

    Args:
        text: The file's contents.

    Returns:
        The lock.

    Raises:
        InvalidJsonError: When the text is not JSON.
        JSONTypeError: When it is not the lock's shape.
    """
    lock = narrow_json_to_dict(load_json_str(text))
    return LiftLock(
        remote=require_str(lock, "remote"),
        revision=require_str(lock, "revision"),
        entries=[decode_lift_entry(entry) for entry in require_list(lock, "entries")],
    )


def encode_lift_lock(lock: LiftLock) -> str:
    """Write the lock's text, as :func:`decode_lift_lock` reads it.

    Args:
        lock: The lock.

    Returns:
        Indented JSON with a final newline.
    """
    entries: list[JSONValue] = [
        {"path": entry["path"], "source": entry["source"], "sha256": entry["sha256"]}
        for entry in lock["entries"]
    ]
    document: dict[str, JSONValue] = {
        "remote": lock["remote"],
        "revision": lock["revision"],
        "entries": entries,
    }
    return dump_json_str(document, indent=2) + "\n"


def pinned_digest(data: bytes) -> str:
    """The hex digest a pin records, over the file's LF form.

    THE LINE ENDINGS ARE NORMALISED, because the same commit checks out
    differently: this repository has no ``.gitattributes`` and Windows
    clones run ``core.autocrlf=true``, so a lifted file is CRLF in a
    Windows working tree, LF in a Linux CI checkout, and LF as MCPs' blob.
    Hashing the bytes as found would fail one of the three on every pin.

    Args:
        data: The file's bytes, from a working tree or a git blob.

    Returns:
        The lower-case hex sha256 of ``data`` with every CRLF made LF.
    """
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def read_lift_lock(package_root: Path) -> LiftLock:
    """Read the package's lock.

    Args:
        package_root: The directory holding :data:`LIFT_LOCK_NAME`.

    Returns:
        The lock.

    Raises:
        FileNotFoundError: When the package carries no lock.
        InvalidJsonError: When the file is not JSON.
        JSONTypeError: When it is not the lock's shape.
    """
    return decode_lift_lock((package_root / LIFT_LOCK_NAME).read_text(encoding="utf-8"))


def drifted_lifts(package_root: Path, lock: LiftLock) -> list[str]:
    """Every lifted file whose bytes are not the ones its pin names.

    Args:
        package_root: The directory the entries' paths are relative to.
        lock: The lock.

    Returns:
        One line per missing or changed file, naming its pin; empty when
        every file matches.
    """
    found: list[str] = []
    for entry in lock["entries"]:
        local = package_root / entry["path"]
        if not local.is_file():
            found.append(
                f"{entry['path']}: missing; it is lifted from MCPs {entry['source']} "
                f"at {lock['revision']}"
            )
            continue
        digest = pinned_digest(local.read_bytes())
        if digest != entry["sha256"]:
            found.append(
                f"{entry['path']}: sha256 {digest} is not the pinned {entry['sha256']} of "
                f"MCPs {entry['source']} at {lock['revision']}; make the change in MCPs and "
                "run lift-refresh, never edit a lifted file here"
            )
    return found


def upstream_commit(mcps: Path, revision: str) -> str:
    """The full commit a revision names in the MCPs repository.

    Args:
        mcps: The MCPs checkout.
        revision: Any name git resolves to a commit.

    Returns:
        The forty-hex commit.

    Raises:
        AppError: ``MAKETOOLS_LIFT`` when git cannot resolve it there.
    """
    result = _test_hooks.run_capturing(
        ["git", "rev-parse", "--verify", "--quiet", f"{revision}^{{commit}}"], cwd=mcps
    )
    if result["returncode"] != 0:
        raise AppError(MaketoolsErrorCode.LIFT, f"{mcps} has no commit {revision!r} to lift from")
    return result["stdout"].strip()


def upstream_blob(mcps: Path, commit: str, source: str) -> bytes:
    """A file's bytes at a commit of the MCPs repository, from its object store.

    Args:
        mcps: The MCPs checkout.
        commit: The commit.
        source: The file's path from that repository's root.

    Returns:
        The blob, UTF-8 encoded as git printed it.

    Raises:
        AppError: ``MAKETOOLS_LIFT`` when the commit holds no such file.
    """
    result = _test_hooks.run_capturing(["git", "cat-file", "blob", f"{commit}:{source}"], cwd=mcps)
    if result["returncode"] != 0:
        raise AppError(
            MaketoolsErrorCode.LIFT,
            f"MCPs {commit} holds no {source} to lift: {result['stderr'].strip()}",
        )
    return result["stdout"].encode("utf-8")


def refresh_lifts(package_root: Path, mcps: Path, revision: str) -> LiftLock:
    """Write every lifted file from an MCPs commit and re-pin the lock to it.

    Args:
        package_root: The directory holding the lock and the lifted files.
        mcps: The MCPs checkout whose object store is read.
        revision: The commit to lift from, as any name git resolves.

    Returns:
        The lock as written.

    Raises:
        AppError: ``MAKETOOLS_LIFT`` when the commit or a source is absent;
            nothing is written until every source has been read.
    """
    lock = read_lift_lock(package_root)
    commit = upstream_commit(mcps, revision)
    blobs = [(entry, upstream_blob(mcps, commit, entry["source"])) for entry in lock["entries"]]
    for entry, blob in blobs:
        (package_root / entry["path"]).write_bytes(blob)
    refreshed = LiftLock(
        remote=lock["remote"],
        revision=commit,
        entries=[
            LiftEntry(path=entry["path"], source=entry["source"], sha256=pinned_digest(blob))
            for entry, blob in blobs
        ],
    )
    (package_root / LIFT_LOCK_NAME).write_text(encode_lift_lock(refreshed), encoding="utf-8")
    return refreshed
