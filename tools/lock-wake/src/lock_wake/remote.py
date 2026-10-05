"""Following a fleet-lock journal that lives on another host, over ssh.

WHY THIS EXISTS (MCPs board task 03590bf9). Every deploy takes its fleet
lock on diphtheria, so its acquired, released and failed boundaries are
written to diphtheria's own ``~/PROJECTS/MCPs/.fleet-events.jsonl``. Until
this module the bridge read only the hub's journal, and nothing read
diphtheria's: measured 2026-10-05 04:57Z, 3406 lines there, the last hold a
deploy released at 02:59:15Z, while the hub's last lock row was a
rebuild-ts at 03:35Z. A deploy's boundaries reached no session.

ONE SSH PER TICK, READING ONLY THE UNREAD WINDOW. The pump runs every three
minutes and the journal is append-only and already 600 KB, so the remote
command prints the journal's size and then ``tail -c`` from the cursor's
offset, and nothing before it crosses the wire. The size is what lets a
truncated or replaced journal be refused exactly as a local read refuses
it: ``tail`` alone prints nothing past the end and would hold the cursor
still forever, reporting quiet.

THE COMMAND IS ONE STRING THE REMOTE SHELL PARSES, so nothing in it may
need quoting: the path is refused unless it is absolute and made only of
letters, digits, ``.``, ``_``, ``-`` and ``/``, and the offset is an
integer this module formats. The rule tools/fleet learned on 2026-09-04
(never interpolate text that quoting can change) is met by refusing any
text quoting could change, rather than by quoting it.

THE CURSOR STAYS ON THIS MACHINE, under a directory the caller names,
called ``<journal name>.lock-wake-<host>-offset.json``, so the hub's own
cursor for its own journal and this one never share a file. The pump names
the hub's MCPs clone root, which ignores ``/.fleet-events.jsonl.*``.

AN ABSENT REMOTE JOURNAL IS REFUSED, unlike an absent local one: the row
names a deploy host whose journal exists, and ``stat`` failing there is a
moved clone or a wrong path, which reading as quiet would hide.
"""

from __future__ import annotations

import pathlib
import re
from typing import Final

from platform_core.error_codes_tooling import LockWakeErrorCode
from platform_core.errors import AppError
from platform_core.journal_cursor import complete_lines_after, cursor_path
from typing_extensions import TypedDict

from lock_wake import _test_hooks
from lock_wake.identity import CURSOR_READER
from lock_wake.journal import JournalSlice, decode_lines

#: A remote path the remote shell reads exactly as written.
_SAFE_PATH: Final = re.compile(r"/[A-Za-z0-9._/-]+")

#: An ssh host alias or name, which never needs quoting either.
_SAFE_HOST: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*")

#: Options every ssh this module runs carries, tools/fleet's set and for its
#: reasons: ``BatchMode`` turns a key prompt into a failure instead of a hang
#: under the scheduled pump, ``ConnectTimeout`` bounds the connect, and the
#: two keepalive options end a session whose peer went away after it.
SSH_OPTIONS: Final = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=15",
    "-o",
    "ServerAliveCountMax=4",
)

#: The deadline on the one ssh a tick runs. A window is at most one pump
#: interval of journal lines, so a minute is generous and still well inside
#: the pump's three-minute tick.
SSH_TIMEOUT_SECONDS: Final = 60


class RemoteJournal(TypedDict):
    """A journal on another host, as the pump's row names it.

    Attributes:
        host: The ssh host alias (``diphtheria``), resolved by this
            machine's ssh configuration.
        path: The journal's absolute POSIX path on that host.
    """

    host: str
    path: str


def parse_remote_journal(value: str) -> RemoteJournal:
    """Decode ``<host>:<absolute path>`` into a remote journal.

    Args:
        value: The ``--remote-journal`` flag's value.

    Returns:
        The host and path.

    Raises:
        ValueError: When the value is not ``host:/path``, or either half
            holds a character the remote shell would need quoted.
    """
    host, colon, path = value.partition(":")
    if colon == "" or _SAFE_HOST.fullmatch(host) is None or _SAFE_PATH.fullmatch(path) is None:
        raise ValueError(
            f"remote journal {value!r} is not <host>:<absolute path> made of letters, digits, "
            f"'.', '_', '-' and '/'; the path rides one command line the remote shell parses, "
            f"so a character it would need quoted is refused rather than quoted"
        )
    return RemoteJournal(host=host, path=path)


def remote_cursor_path(cursor_dir: pathlib.Path, journal: RemoteJournal) -> pathlib.Path:
    """Where this bridge's offset into a remote journal lives on this machine.

    Args:
        cursor_dir: The local directory the pump's row names.
        journal: The remote journal.

    Returns:
        ``<cursor_dir>/<journal name>.lock-wake-<host>-offset.json``.
    """
    name = pathlib.PurePosixPath(journal["path"]).name
    return cursor_path(cursor_dir / name, f"{CURSOR_READER}-{journal['host']}")


def window_command(path: str, offset: int) -> str:
    """The remote command: the journal's size, then every byte past the offset.

    Args:
        path: The journal's path on the host, already refused unless safe.
        offset: The first unread byte.

    Returns:
        The command line the remote shell runs.
    """
    return f"stat -c %s -- {path} && tail -c +{offset + 1} -- {path}"


def read_remote_slice(journal: RemoteJournal, offset: int) -> JournalSlice:
    """Read and decode every complete line past a byte offset of a remote journal.

    Args:
        journal: The remote journal.
        offset: Byte offset of the first unread byte, from the local cursor.

    Returns:
        The decoded events and the offset just past the last complete line.

    Raises:
        AppError: ``LOCK_WAKE_REMOTE_JOURNAL_UNREADABLE`` when ssh exits
            non-zero (unreachable host, refused key, absent journal), carrying
            its stderr; ``LOCK_WAKE_REMOTE_JOURNAL_MALFORMED`` when the output
            does not open with the journal's size.
        ValueError: A position past the journal's end -- it was truncated or
            replaced, and the operator decides, not this reader.
        JSONTypeError: A complete line that is not a valid event.
        InvalidJsonError: A complete line that is not JSON at all.
        subprocess.TimeoutExpired: The ssh outlived its deadline.
    """
    host = journal["host"]
    completed = _test_hooks.run_ssh(
        ["ssh", *SSH_OPTIONS, host, window_command(journal["path"], offset)],
        SSH_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise AppError(
            LockWakeErrorCode.REMOTE_JOURNAL_UNREADABLE,
            f"ssh {host} exited {completed.returncode} reading {journal['path']}: "
            f"{completed.stderr.decode('utf-8', errors='replace').strip()}",
        )
    size_text, newline, window = completed.stdout.partition(b"\n")
    if newline == b"" or not size_text.isdigit():
        raise AppError(
            LockWakeErrorCode.REMOTE_JOURNAL_MALFORMED,
            f"ssh {host} reading {journal['path']} printed {size_text[:80]!r} where the "
            f"journal's size in bytes belongs, so no position can be checked against it",
        )
    size = int(size_text)
    if offset > size:
        raise ValueError(
            f"position says byte {offset} of {host}:{journal['path']}, but the journal holds "
            f"only {size} bytes; it was truncated or replaced, and rewinding silently would "
            f"re-announce every event in its history"
        )
    return decode_lines(complete_lines_after(window, offset))


__all__ = [
    "SSH_OPTIONS",
    "SSH_TIMEOUT_SECONDS",
    "RemoteJournal",
    "parse_remote_journal",
    "read_remote_slice",
    "remote_cursor_path",
    "window_command",
]
