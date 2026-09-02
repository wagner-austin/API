"""An in-memory events artifact that behaves like one on disk.

Drives the two seams the incremental reader uses -- ``file_marker``
and ``read_bytes_from`` -- so tests can grow a file line by line,
replace it with a new run, or take it away, and then assert on what
the reader actually asked for. The recorded read offsets are the point:
they are how a test proves the reader resumed rather than re-read.
"""

from __future__ import annotations

from pathlib import Path

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks.fs import FileMarkerProtocol, ReadBytesFromProtocol


class FakeArtifact:
    """A growing JSONL file held in memory.

    Attributes:
        read_offsets: Byte offset each ``read_bytes_from`` call
            started at, oldest first.
        bytes_served: Total bytes handed to the reader across all
            reads -- the measure a performance test asserts on.
    """

    def __init__(self) -> None:
        """Start absent, as an instance that has logged nothing is."""
        self._data = b""
        self._identity = 0
        self._present = False
        self.read_offsets: list[int] = []
        self.bytes_served = 0

    def start_run(self, lines: list[str]) -> None:
        """Replace the artifact with a NEW run's first lines.

        A new run re-creates the path, so the identity changes -- the
        same signal a real rotation gives the reader.

        Args:
            lines: JSONL lines without their trailing newline.

        Returns:
            None.
        """
        self._identity += 1
        self._data = b""
        self._present = True
        self.append(lines)

    def append(self, lines: list[str]) -> None:
        """Append complete lines to the current run.

        Args:
            lines: JSONL lines without their trailing newline. Each is
                terminated here, because a logging handler terminates
                every record it writes and a reader may only decode
                lines it has seen the end of.

        Returns:
            None.
        """
        self._present = True
        for line in lines:
            self._data += line.encode("utf-8") + b"\n"

    def append_partial(self, text: str) -> None:
        """Append bytes that do NOT end a line.

        Models a poll landing mid-append.

        Args:
            text: Fragment to append with no terminator.

        Returns:
            None.
        """
        self._present = True
        self._data += text.encode("utf-8")

    def remove(self) -> None:
        """Make the artifact absent again.

        Returns:
            None.
        """
        self._present = False

    def marker(self, path: Path) -> tuple[int, int]:
        """Return the artifact's identity and size.

        Args:
            path: Ignored; one artifact stands for one path.

        Returns:
            ``(identity, size_bytes)``.

        Raises:
            OSError: When the artifact is absent.
        """
        if not self._present:
            raise OSError(f"no such artifact: {path}")
        return (self._identity, len(self._data))

    def read_from(self, path: Path, offset: int) -> bytes:
        """Return the bytes from ``offset`` to the end.

        Args:
            path: Ignored; one artifact stands for one path.
            offset: Byte offset to start at.

        Returns:
            Every byte from ``offset`` onward.

        Raises:
            OSError: When the artifact is absent.
        """
        if not self._present:
            raise OSError(f"no such artifact: {path}")
        self.read_offsets.append(offset)
        chunk = self._data[offset:]
        self.bytes_served += len(chunk)
        return chunk


def install_artifact(
    artifact: FakeArtifact,
) -> tuple[FileMarkerProtocol, ReadBytesFromProtocol]:
    """Point the filesystem seams at ``artifact``.

    Args:
        artifact: The in-memory artifact to serve.

    Returns:
        The original ``(file_marker, read_bytes_from)`` hooks, for the
        caller to restore.
    """
    originals = (top_hooks.file_marker, top_hooks.read_bytes_from)
    top_hooks.file_marker = artifact.marker
    top_hooks.read_bytes_from = artifact.read_from
    return originals


def restore_artifact(originals: tuple[FileMarkerProtocol, ReadBytesFromProtocol]) -> None:
    """Put the filesystem seams back.

    Args:
        originals: The pair returned by :func:`install_artifact`.

    Returns:
        None.
    """
    top_hooks.file_marker, top_hooks.read_bytes_from = originals
