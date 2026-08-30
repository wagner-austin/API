"""What every operation the harness performs outside itself looks like.

The contracts only. :mod:`rw_bot.harness._hook_defaults` holds the
implementations that satisfy them and :mod:`rw_bot.harness._test_hooks` binds
the two together -- which is where callers and tests both look, so neither has
to know this file exists.

Split out when the single module passed the six-hundred-line ceiling. The
boundary is not arbitrary: a protocol describes an obligation and an
implementation discharges it, and the two change for different reasons -- a
new platform rewrites implementations without touching a contract, and a new
capability adds a contract before anything can satisfy it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath
from socketserver import BaseServer
from typing import Protocol


class ReadTextLinesProto(Protocol):
    """Read a UTF-8 text file and return its lines."""

    def __call__(self, path: Path) -> tuple[str, ...]:
        """Read every line of a text file.

        Args:
            path: File to read.

        Returns:
            The file's lines with trailing newlines removed, in file order.

        Raises:
            OSError: When the file cannot be read.
            UnicodeDecodeError: When the file is not valid UTF-8.
        """
        ...


class WriteLineProto(Protocol):
    """Emit one line to the process's standard output."""

    def __call__(self, text: str) -> None:
        """Write one line.

        Args:
            text: Line content, without a trailing newline.
        """
        ...


class PathExistsProto(Protocol):
    """Report whether a filesystem path exists."""

    def __call__(self, path: Path) -> bool:
        """Test one path for existence.

        Args:
            path: Path to test.

        Returns:
            ``True`` when the path exists, ``False`` otherwise.
        """
        ...


class ReadArgvProto(Protocol):
    """Read this process's command-line arguments, excluding the program name."""

    def __call__(self) -> list[str]:
        """Return the argument list.

        Returns:
            Arguments after the program name, in order.
        """
        ...


class RunCaptureProto(Protocol):
    """Run a child process to completion and capture everything it printed."""

    def __call__(self, argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        """Run one command.

        Args:
            argv: Argument vector, program first.

        Returns:
            The child's exit status and its combined output lines, in order.

        Raises:
            OSError: When the program cannot be started.
        """
        ...


class ListNamesProto(Protocol):
    """List the immediate entry names of a directory."""

    def __call__(self, path: Path) -> tuple[str, ...]:
        """List one directory.

        Args:
            path: Directory to list.

        Returns:
            Entry names, sorted, without their parent path.

        Raises:
            OSError: When the directory cannot be read.
        """
        ...


class CopyEntryProto(Protocol):
    """Copy one file or directory tree into a destination directory."""

    def __call__(self, source: Path, destination: Path) -> None:
        """Copy one entry.

        Args:
            source: File or directory to copy.
            destination: Directory to copy it into, which must exist.

        Raises:
            OSError: When the copy fails.
        """
        ...


class MakeDirsProto(Protocol):
    """Create a directory and every missing parent."""

    def __call__(self, path: Path) -> None:
        """Create one directory.

        Args:
            path: Directory to create. Existing directories are left alone.

        Raises:
            OSError: When the directory cannot be created.
        """
        ...


class WriteTextLinesProto(Protocol):
    """Write lines to a UTF-8 text file, replacing any existing content."""

    def __call__(self, path: Path, lines: Sequence[str]) -> None:
        """Write one text file.

        Args:
            path: File to write.
            lines: Line contents, without trailing newlines.

        Raises:
            OSError: When the file cannot be written.
        """
        ...


class SpawnedMatchProto(Protocol):
    """The child-process surface the fleet manager consumes."""

    @property
    def pid(self) -> int:
        """The child's process id."""
        ...

    def poll(self) -> int | None:
        """Return the exit code, or ``None`` while the child runs."""
        ...


class SpawnMatchProto(Protocol):
    """Start one detached match process with its transcript on disk."""

    def __call__(self, argv: Sequence[str], transcript: Path) -> SpawnedMatchProto:
        """Spawn one match.

        Args:
            argv: Argument vector, program first.
            transcript: File that receives the child's combined output.

        Returns:
            The spawned process handle.

        Raises:
            OSError: When the program cannot be started or the
                transcript cannot be opened.
        """
        ...


class ServeForeverProto(Protocol):
    """Run an HTTP server's accept loop until the process is interrupted."""

    def __call__(self, server: BaseServer) -> None:
        """Serve until interrupted.

        Args:
            server: The bound server to run.
        """
        ...


class KillTreeProto(Protocol):
    """Terminate a process and every child it spawned."""

    def __call__(self, pid: int) -> None:
        """Kill one process tree.

        Args:
            pid: Root process id of the tree.
        """
        ...


class ReadExecutableProto(Protocol):
    """Report the Python that is running this process."""

    def __call__(self) -> str:
        """Return the interpreter path.

        Returns:
            The absolute path of the running interpreter, which the planner is
            started with so it runs in the same environment the harness does.
        """
        ...


class ResolveRootProto(Protocol):
    """Report the absolute path of the repository root."""

    def __call__(self) -> PurePath:
        """Return the root.

        Returns:
            The working directory, resolved. The engine runs with the GAME
            directory as its working directory, so every path handed to it has
            to be made absolute against this first.

            Typed as :class:`~pathlib.PurePath` rather than :class:`Path`
            because the value is COMPOSED into commands, never opened, and a
            concrete path carries the running interpreter's flavour into a
            launch that may be for the other platform.
        """
        ...


class NewStampProto(Protocol):
    """Mint an identifier unique to one launch."""

    def __call__(self) -> str:
        """Return a fresh stamp.

        Returns:
            A short identifier. Concurrent matches compile their agents into
            paths named by it, so two launches in one instant must not collide.
        """
        ...


class SpawnGameProto(Protocol):
    """Start the engine detached, with its two streams on disk."""

    def __call__(
        self,
        argv: Sequence[str],
        cwd: PurePath,
        stdout_path: Path,
        stderr_path: Path,
        env: Mapping[str, str],
    ) -> SpawnedMatchProto:
        """Spawn one engine.

        Args:
            argv: Argument vector, the JVM first.
            cwd: Working directory, which for the engine is the game directory.
            stdout_path: File receiving the engine's standard output.
            stderr_path: File receiving its standard error.
            env: Complete environment for the engine. Complete rather than an
                overlay because ``subprocess`` replaces it wholesale, and on
                Linux it must carry ``LD_LIBRARY_PATH`` or the engine's native
                GUI stack fails to resolve its own dependencies.

        Returns:
            The spawned process handle.

        Raises:
            OSError: When the engine cannot be started or a stream file cannot
                be opened.
        """
        ...


class WaitForPortProto(Protocol):
    """Wait until something is listening on a local port."""

    def __call__(self, port: int, timeout_s: float, poll_s: float) -> str | None:
        """Wait for the agent's channel.

        Args:
            port: Port to connect to on the loopback address.
            timeout_s: How long to keep trying before giving up.
            poll_s: How long to wait between attempts.

        Returns:
            ``None`` once a connection succeeds, otherwise the LAST connection
            failure seen before the timeout.

            The reason is carried rather than discarded because the two ways
            this fails look identical from outside and are diagnosed in
            different places: a refused connection means the engine is alive
            and the agent never bound, and a timeout with no route means the
            engine died during boot. Ninety seconds of silence followed by
            "the agent never opened port N" said neither.

            A value rather than an exception, because a boot that never opens
            the channel is a match RESULT -- filed as a failure, job left
            outstanding -- not a fault in the harness.
        """
        ...


class RunInheritedProto(Protocol):
    """Run a child that writes to this process's own streams."""

    def __call__(self, argv: Sequence[str], env: Mapping[str, str]) -> int:
        """Run one command to completion without capturing it.

        The planner's scorecard is read by whoever captured THIS process, so
        capturing it here would swallow the only output a batch keeps.

        Args:
            argv: Argument vector, program first.
            env: Environment for the child, complete rather than an overlay.

        Returns:
            The child's exit status.

        Raises:
            OSError: When the program cannot be started.
        """
        ...


class ReadEnvironmentProto(Protocol):
    """Read this process's environment."""

    def __call__(self) -> Mapping[str, str]:
        """Return the environment.

        Returns:
            The variables this process was started with, which a child
            inherits except where the caller overrides one.
        """
        ...


class SleepProto(Protocol):
    """Pause this process."""

    def __call__(self, seconds: float) -> None:
        """Wait.

        Args:
            seconds: How long to wait.
        """
        ...


class RemovePathProto(Protocol):
    """Delete a file or a directory tree, if it is there."""

    def __call__(self, path: Path) -> None:
        """Remove one path.

        Args:
            path: File or directory to remove. A path that is already gone is
                not an error: this runs in teardown, where the interesting
                failure is the match's, not the cleanup's.
        """
        ...


class GetEnvProto(Protocol):
    """Read one named environment variable."""

    def __call__(self, name: str) -> str | None:
        """Return a variable's value.

        Args:
            name: The variable to read.

        Returns:
            Its value, or ``None`` when it is not set. Absence is a real
            answer: a run outside any image has no image digest, and
            :data:`~platform_core.comparability.NO_VALUE` is what a
            fingerprint records for it.
        """
        ...


class CountCoresProto(Protocol):
    """Report how many logical cores this machine has."""

    def __call__(self) -> int | None:
        """Return the core count.

        Returns:
            The count, or ``None`` when the platform will not say. The shape
            :func:`~platform_core.environment_record.stdlib_host_probe`
            expects, which raises on ``None`` rather than recording a guess.
        """
        ...


class ReadPlatformProto(Protocol):
    """Report which operating system this process is running on."""

    def __call__(self) -> str:
        """Return the platform.

        Returns:
            A ``sys.platform`` value, which
            :mod:`rw_bot.harness.jvm` reads to decide what the JDK's tools are
            called and how a classpath is joined.
        """
        ...


__all__ = [
    "CopyEntryProto",
    "CountCoresProto",
    "GetEnvProto",
    "KillTreeProto",
    "ListNamesProto",
    "MakeDirsProto",
    "NewStampProto",
    "PathExistsProto",
    "ReadArgvProto",
    "ReadEnvironmentProto",
    "ReadExecutableProto",
    "ReadPlatformProto",
    "ReadTextLinesProto",
    "RemovePathProto",
    "ResolveRootProto",
    "RunCaptureProto",
    "RunInheritedProto",
    "ServeForeverProto",
    "SleepProto",
    "SpawnGameProto",
    "SpawnMatchProto",
    "SpawnedMatchProto",
    "WaitForPortProto",
    "WriteLineProto",
    "WriteTextLinesProto",
]
