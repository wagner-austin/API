"""Where this bridge is in the journal, so it never announces a line twice.

ONE INTEGER IS THE ENTIRE MEMORY -- the byte offset of the first unread
journal byte -- so the position is a single small JSON object rewritten
whole, not an append-only log that grows forever saying nothing its last
line does not.

THE FILE LIVES BESIDE THE JOURNAL, derived from its path rather than
configured separately, for the reason ``ci_wake.position`` gives: the two
files that must describe the same stream are not separately addressable,
and moving one moves both. The journal sits in the MCPs repo root among
its operational siblings (``.fleet-lock``, the journal itself), and the
offset file joins them, untracked.

WHY THE CYCLE WRITES IT LAST. Announcements POST before the offset
advances, so a crash between the two repeats a post on the next cycle
rather than losing one -- at-least-once, the family guarantee. The
alternative order turns any transport failure into cascade transitions
nobody is ever told about.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str, require_int

from lock_wake import _test_hooks


def position_path(journal: pathlib.Path) -> pathlib.Path:
    """Where the offset lives for a given journal.

    Args:
        journal: The journal's path.

    Returns:
        The position file's path, beside it.
    """
    return journal.parent / (journal.name + ".lock-wake-offset.json")


def read_offset(path: pathlib.Path) -> int:
    """Read the first unread byte's offset.

    Args:
        path: The position file's path.

    Returns:
        The offset. An absent file reads as 0 rather than raising: a
        machine whose bridge has never run has announced nothing, and
        refusing the first cycle for having no history would make the
        bridge impossible to start.

    Raises:
        InvalidJsonError: A position file that is not JSON at all.
        JSONTypeError: A position file that is JSON but not an offset
            object, or an offset that is negative -- NOT defaulted, because
            a misread position either re-announces history or skips it,
            and both wear the costume of a working bridge.
        OSError: A position file that exists but cannot be read.
    """
    if not _test_hooks.file_exists(path):
        return 0
    value = load_json_str(_test_hooks.read_bytes(path).decode("utf-8"))
    if not isinstance(value, dict):
        raise JSONTypeError(
            f"{path} is a {type(value).__name__}, not an object; a position that "
            f"cannot be read either re-announces history or skips it"
        )
    offset = require_int(value, "offset")
    if offset < 0:
        raise JSONTypeError(f"{path} holds offset {offset}, which is negative")
    return offset


def write_offset(path: pathlib.Path, offset: int) -> None:
    """Record the first unread byte's offset.

    Args:
        path: The position file's path.
        offset: The offset just past the last announced line.
    """
    _test_hooks.write_text(path, dump_json_str({"offset": offset}) + "\n")


__all__ = ["position_path", "read_offset", "write_offset"]
