"""Which release archive a check's recipe built, by content, on the verdict line.

MCPs board task 0c8ace8d. A project may declare the archive its recipe leaves
behind (:mod:`fleet.contracts.release_artifact`). A packaging build stamps
each entry with its packing time, so the archive's sha256 is new on every
build of one tree; what stays the same is each entry's name and CRC-32,
which covers only the entry's bytes. osrs-lab recorder/LIVE_COVERAGE.md "The
release check" names that list as what ties a deployed jar to the commit
that built it.

So once the recipe has ended, and before the tree it ran in is removed, the
build reads the declared archive with the node's own ``python3`` (on every
Linux node by the toolchain contract) and appends one :data:`ARTIFACT_MARKER`
line to the transcript: the path, the entry count, and the sha256 of the
sorted ``<name> <crc-32 as 8 lowercase hex digits>`` lines joined by
newlines. A project that declares none gets the line ``none``, so a verdict
says the field does not apply rather than leaving it out. A declared archive
the recipe did not leave behind is :data:`MISSING`, and a file there that is
not a zip archive (or that ``python3`` could not read) is
:data:`UNREADABLE`; neither is ever ``none``. The verdict
(:func:`read_artifact`, in :func:`fleet.core.verdict.judge`) copies the line
onto the queue row and the thread as ``artifact=<path>:<entries>:<digest>``.

The line is the runner's, not the checked commit's: the declaration is read
from the registry at the runner's rolled commit, so a commit from before it
is read the same way.
"""

from __future__ import annotations

import re
from typing import Final

from fleet.contracts.release_artifact import ReleaseArtifact
from fleet.core.node_programs import EXEC_PATH

#: The line's opening token.
ARTIFACT_MARKER: Final = "FLEET_ARTIFACT"

#: What the line says for a project that declares no artifact.
NONE: Final = "none"

#: A declared archive the recipe did not leave behind.
MISSING: Final = "ARTIFACT_MISSING"

#: A declared path holding something that is not a readable zip archive.
UNREADABLE: Final = "ARTIFACT_UNREADABLE"

#: What the verdict says of a transcript without the line: a build from
#: before it, or one stopped before it.
UNREAD: Final = "unread"

#: The program ``python3 -I`` runs over ``<file> <path as printed>``. One
#: line of Python with no single quote, so the build quotes it whole.
DIGEST_PROGRAM: Final = (
    "import hashlib, os, sys, zipfile; "
    "file, shown = sys.argv[1:3]; "
    "ok = os.path.isfile(file); "
    "zipped = ok and zipfile.is_zipfile(file); "
    'entries = sorted("%s %08x" % (i.filename, i.CRC) for i in zipfile.ZipFile(file).infolist()) '
    "if zipped else []; "
    f'print("{ARTIFACT_MARKER}: " + shown + " " + ('
    '(str(len(entries)) + " " + hashlib.sha256("\\n".join(entries).encode()).hexdigest()) '
    f'if zipped else ("{UNREADABLE}" if ok else "{MISSING}")))'
)

#: The line, as the build prints it.
ARTIFACT_LINE: Final[re.Pattern[str]] = re.compile(
    rf"^{ARTIFACT_MARKER}: (?:(?P<none>{NONE})|(?P<path>\S+) "
    rf"(?:(?P<entries>\d+) (?P<digest>[0-9a-f]{{64}})|(?P<error>{MISSING}|{UNREADABLE})))$",
    re.MULTILINE,
)


def sh_artifact_lines(
    *, log: str, root: str, artifact: ReleaseArtifact | None, run_as: str
) -> list[str]:
    """The ``sh`` lines that print the declared archive's reading into the transcript.

    Args:
        log: The transcript's path, appended to.
        root: The staged tree's root as ``sh`` expands it, the directory
            the declared path is relative to (``"$exec_root"`` for a build
            run as execdocker).
        artifact: The project's declaration, or None.
        run_as: The prefix the reading runs under: ``""`` for the runner's
            own build, ``"sudo -n -u execdocker "`` for an isolated one,
            whose tree only that user may read. The reading starts from
            :data:`fleet.core.node_programs.EXEC_PATH`, the node's own
            programs.

    Returns:
        The lines, run after the recipe's phase and before the tree is
        removed. A ``python3`` that fails prints :data:`UNREADABLE`, so the
        build still writes its status.
    """
    if artifact is None:
        return [f"printf '%s\\n' '{ARTIFACT_MARKER}: {NONE}' >> '{log}'"]
    path = artifact["path"]
    return [
        f"{run_as}env PATH={EXEC_PATH} python3 -I -c '{DIGEST_PROGRAM}' "
        f"\"{root}/{path}\" '{path}' >> '{log}' 2>&1 "
        f"|| printf '%s\\n' '{ARTIFACT_MARKER}: {path} {UNREADABLE}' >> '{log}'"
    ]


def read_artifact(tail: str) -> str:
    """Read the declared archive's reading off a transcript's tail.

    Args:
        tail: The transcript's last lines.

    Returns:
        ``<path>:<entries>:<digest>`` from the last :data:`ARTIFACT_LINE`;
        :data:`NONE` for a project that declares none; ``<error>:<path>``
        for :data:`MISSING` or :data:`UNREADABLE`; else :data:`UNREAD`.
    """
    lines = list(ARTIFACT_LINE.finditer(tail))
    if not lines:
        return UNREAD
    last = lines[-1]
    if last.group("none") is not None:
        return NONE
    if last.group("error") is not None:
        return f"{last.group('error')}:{last.group('path')}"
    return f"{last.group('path')}:{last.group('entries')}:{last.group('digest')}"


__all__ = [
    "ARTIFACT_LINE",
    "ARTIFACT_MARKER",
    "DIGEST_PROGRAM",
    "MISSING",
    "NONE",
    "UNREAD",
    "UNREADABLE",
    "read_artifact",
    "sh_artifact_lines",
]
