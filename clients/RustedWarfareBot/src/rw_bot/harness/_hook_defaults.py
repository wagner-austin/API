"""How the harness actually reaches the operating system.

The implementations behind :mod:`rw_bot.harness._hook_protocols`. Nothing here
is imported directly by anything but :mod:`rw_bot.harness._test_hooks`, which
binds each to its protocol; callers go through the binding so a test can
replace one without the call site changing shape.

Every platform difference in here is COMPOSED rather than branched -- the
argument vector and the spawn flags come from
:mod:`rw_bot.harness.process_tree`, which decides them as pure functions of a
stated platform. That is what keeps both platforms' behaviour reachable from
whichever one the suite happens to run on.
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath
from socketserver import BaseServer

from platform_core.config import config_test_hooks

from rw_bot.harness._hook_protocols import SpawnedMatchProto
from rw_bot.harness.process_tree import fell_command, spawn_isolation


def _run_capture_impl(argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
    """Production implementation of :class:`RunCaptureProto`.

    Streams are merged because the caller wants one transcript of the match in
    the order it happened, and the launcher writes progress to one stream and
    the planner writes its scorecard to the other.

    Undecodable bytes are replaced rather than raising. This decodes a console
    stream produced by a third-party game process and its launcher, not a data
    format this package defines; the alternative is discarding a match that ran
    for seven minutes because one cosmetic byte was not UTF-8.

    Args:
        argv: Argument vector, program first.

    Returns:
        The child's exit status and its combined output lines, in order.

    Raises:
        OSError: When the program cannot be started.
    """
    isolation = spawn_isolation(_read_platform_impl())
    finished = subprocess.run(
        list(argv),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        check=False,
        # Ranking and fellability, decided by platform in
        # :mod:`rw_bot.harness.process_tree` so neither arm needs a branch
        # here. On Windows this carries the priority class the batch needs
        # to survive a co-tenant spike (log 2026-08-10); on POSIX it puts
        # the match in its own process group so it can be felled whole.
        creationflags=isolation["creationflags"],
        start_new_session=isolation["start_new_session"],
    )
    return finished.returncode, tuple(finished.stdout.splitlines())


def _list_names_impl(path: Path) -> tuple[str, ...]:
    """Production implementation of :class:`ListNamesProto`.

    Args:
        path: Directory to list.

    Returns:
        Entry names, sorted, without their parent path.

    Raises:
        OSError: When the directory cannot be read.
    """
    return tuple(sorted(entry.name for entry in path.iterdir()))


#: Directories a copy never takes.
#:
#: Bytecode caches are not source, and copying them makes a frozen tree's
#: identity depend on whether anything happened to import a module before the
#: freeze ran -- 140 of one payload's 408 files were ``.pyc``, a third of it,
#: none of it saying anything about the code. A ``.pyc`` also embeds the
#: source's timestamp, so two freezes of identical source digested
#: differently. It is excluded here rather than filtered afterwards because
#: the workstation's own frozen trees never wanted them either.
COPY_EXCLUDES = ("__pycache__",)


def _copy_entry_impl(source: Path, destination: Path) -> None:
    """Production implementation of :class:`CopyEntryProto`.

    Args:
        source: File or directory to copy.
        destination: Directory to copy it into, which must exist.

    Raises:
        OSError: When the copy fails.
    """
    target = destination / source.name
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True, ignore=_excluded)
        return
    shutil.copy2(source, target)


def _excluded(directory: str, names: list[str]) -> set[str]:
    """Return the names a copy skips in one directory.

    Written out rather than built with ``shutil.ignore_patterns``, whose
    return type carries ``Any`` and which this package forbids. It is also
    the more honest shape: the exclusion is an exact name match, not a glob.

    Args:
        directory: The directory being copied. Unread -- the rule is the same
            at every level, and the eight caches in this package's own tree
            sit at every level.
        names: Its entries.

    Returns:
        Those in :data:`COPY_EXCLUDES`.
    """
    return {name for name in names if name in COPY_EXCLUDES}


def _make_dirs_impl(path: Path) -> None:
    """Production implementation of :class:`MakeDirsProto`.

    Args:
        path: Directory to create. Existing directories are left alone.

    Raises:
        OSError: When the directory cannot be created.
    """
    path.mkdir(parents=True, exist_ok=True)


def _write_text_lines_impl(path: Path, lines: Sequence[str]) -> None:
    """Production implementation of :class:`WriteTextLinesProto`.

    Args:
        path: File to write.
        lines: Line contents, without trailing newlines.

    Raises:
        OSError: When the file cannot be written.
    """
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _spawn_match_impl(argv: Sequence[str], transcript: Path) -> SpawnedMatchProto:
    """Production implementation of :class:`SpawnMatchProto`.

    The transcript file handle is closed in the parent immediately after
    the spawn — the child keeps its inherited handle, so the report lines
    the planner prints at match end still land in the file.

    Args:
        argv: Argument vector, program first.
        transcript: File that receives the child's combined output.

    Returns:
        The spawned process handle.

    Raises:
        OSError: When the program cannot be started or the transcript
            cannot be opened.
    """
    isolation = spawn_isolation(_read_platform_impl())
    transcript.parent.mkdir(parents=True, exist_ok=True)
    with transcript.open("ab") as sink:
        return subprocess.Popen(
            list(argv),
            stdout=sink,
            stderr=subprocess.STDOUT,
            # Same decision as run_capture, and on POSIX it is what makes
            # :func:`~rw_bot.harness.process_tree.fell_command` able to reach
            # the whole match: the session started here is the group that
            # command signals.
            creationflags=isolation["creationflags"],
            start_new_session=isolation["start_new_session"],
        )


def _serve_forever_impl(server: BaseServer) -> None:
    """Production implementation of :class:`ServeForeverProto`.

    Args:
        server: The bound server to run.
    """
    server.serve_forever()


def _kill_tree_impl(pid: int) -> None:
    """Production implementation of :class:`KillTreeProto`.

    The whole tree, not the root: the fleet spawns a launcher which runs the
    game JVM and the planner, and killing only the root would orphan the
    match — leaving an engine holding a leased channel port, which kills the
    NEXT match at the bind (vhdoom96b, 2026-08-09).

    Which command does that is
    :func:`~rw_bot.harness.process_tree.fell_command`'s decision, so this
    reaches two very different mechanisms — a Windows tree walk and a POSIX
    process-group signal — without a conditional here.

    Args:
        pid: Root process id of the tree.

    Raises:
        ValueError: When the pid is not positive.
    """
    subprocess.run(
        list(fell_command(pid, _read_platform_impl())),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=False,
    )


_STAMP_LENGTH = 8


_CHANNEL_HOST = "127.0.0.1"


def _read_executable_impl() -> str:
    """Production implementation of :class:`ReadExecutableProto`.

    Returns:
        ``sys.executable``.
    """
    return sys.executable


def _resolve_root_impl() -> PurePath:
    """Production implementation of :class:`ResolveRootProto`.

    Returns:
        The working directory, resolved to an absolute path.
    """
    return Path.cwd().resolve()


def _new_stamp_impl() -> str:
    """Production implementation of :class:`NewStampProto`.

    Returns:
        The leading :data:`_STAMP_LENGTH` hex characters of a random UUID.
    """
    return uuid.uuid4().hex[:_STAMP_LENGTH]


def _spawn_game_impl(
    argv: Sequence[str],
    cwd: PurePath,
    stdout_path: Path,
    stderr_path: Path,
    env: Mapping[str, str],
) -> SpawnedMatchProto:
    """Production implementation of :class:`SpawnGameProto`.

    The two streams stay separate, unlike the merged transcript
    :func:`_run_capture_impl` produces. The engine writes its own progress to
    one and the JVM writes crashes to the other, and a crash that arrived
    interleaved into a merged stream is a crash nobody found.

    Args:
        argv: Argument vector, the JVM first.
        cwd: Working directory for the engine.
        stdout_path: File receiving standard output.
        stderr_path: File receiving standard error.
        env: Complete environment for the engine.

    Returns:
        The spawned process handle.

    Raises:
        OSError: When the engine cannot be started or a stream cannot be
            opened.
    """
    isolation = spawn_isolation(_read_platform_impl())
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("ab") as out, stderr_path.open("ab") as err:
        return subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=dict(env),
            stdout=out,
            stderr=err,
            creationflags=isolation["creationflags"],
            start_new_session=isolation["start_new_session"],
        )


def _wait_for_port_impl(port: int, timeout_s: float, poll_s: float) -> str | None:
    """Production implementation of :class:`WaitForPortProto`.

    Args:
        port: Port to connect to on the loopback address.
        timeout_s: How long to keep trying.
        poll_s: How long to wait between attempts.

    Returns:
        ``None`` once a connection succeeds, otherwise the last failure.
    """
    deadline = time.monotonic() + timeout_s
    last = f"nothing attempted within {timeout_s}s"
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((_CHANNEL_HOST, port), timeout=poll_s):
                return None
        except OSError as error:
            # Kept, not discarded: which OSError this was is the difference
            # between "the engine is up and the agent never bound" and "the
            # engine died during boot", and they are fixed in different files.
            last = f"{type(error).__name__}: {error}"
            time.sleep(poll_s)
    return last


def _run_inherited_impl(argv: Sequence[str], env: Mapping[str, str]) -> int:
    """Production implementation of :class:`RunInheritedProto`.

    Args:
        argv: Argument vector, program first.
        env: Complete environment for the child.

    Returns:
        The child's exit status.

    Raises:
        OSError: When the program cannot be started.
    """
    return subprocess.run(list(argv), env=dict(env), check=False).returncode


def _read_environment_impl() -> Mapping[str, str]:
    """Production implementation of :class:`ReadEnvironmentProto`.

    Read through ``platform_core``'s config hooks rather than ``os.environ``
    directly. That module is the monorepo's single permitted reader of the
    process environment, and the rule is not bureaucratic: a launcher that
    reaches for ``os.environ`` itself is one that can quietly start depending
    on ambient configuration, which is the drift a pinned image exists to
    remove.

    Returns:
        A copy, so a caller's overlay cannot reach back into this process.
    """
    return config_test_hooks.get_environment()


def _sleep_impl(seconds: float) -> None:
    """Production implementation of :class:`SleepProto`.

    Args:
        seconds: How long to wait.
    """
    time.sleep(seconds)


def _remove_path_impl(path: Path) -> None:
    """Production implementation of :class:`RemovePathProto`.

    Args:
        path: File or directory to remove.
    """
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
        return
    path.unlink(missing_ok=True)


def _get_env_impl(name: str) -> str | None:
    """Production implementation of :class:`GetEnvProto`.

    Read through ``platform_core``'s config hooks for the same reason
    :func:`_read_environment_impl` is: that module is the monorepo's single
    permitted reader of the process environment.

    Args:
        name: The variable to read.

    Returns:
        Its value, or None when it is not set.
    """
    return config_test_hooks.get_env(name)


def _count_cores_impl() -> int | None:
    """Production implementation of :class:`CountCoresProto`.

    Returns:
        ``os.cpu_count()``.
    """
    return os.cpu_count()


def _read_platform_impl() -> str:
    """Production implementation of :class:`ReadPlatformProto`.

    Returns:
        ``sys.platform``. A hook rather than a direct read so a test can drive
        the real control flow as either platform: the decisions this feeds are
        pure, so both answers are legitimate inputs wherever the suite runs.
    """
    return sys.platform


def _read_argv_impl() -> list[str]:
    """Production implementation of :class:`ReadArgvProto`.

    Returns:
        ``sys.argv`` after the program name.
    """
    return list(sys.argv[1:])


def _read_text_lines_impl(path: Path) -> tuple[str, ...]:
    """Production implementation of :class:`ReadTextLinesProto`.

    Decoding is strict: a log that is not valid UTF-8 is a real problem with
    the launcher's ``-Dfile.encoding`` setting, and silently replacing bad
    bytes would hide it.

    Args:
        path: File to read.

    Returns:
        The file's lines with trailing newlines removed, in file order.

    Raises:
        OSError: When the file cannot be read.
        UnicodeDecodeError: When the file is not valid UTF-8.
    """
    return tuple(path.read_text(encoding="utf-8").splitlines())


def _path_exists_impl(path: Path) -> bool:
    """Production implementation of :class:`PathExistsProto`.

    Args:
        path: Path to test.

    Returns:
        ``True`` when the path exists, ``False`` otherwise.
    """
    return path.exists()


def _write_line_impl(text: str) -> None:
    """Production implementation of :class:`WriteLineProto`.

    Args:
        text: Line content, without a trailing newline.
    """
    sys.stdout.write(f"{text}\n")


__all__ = [
    "_copy_entry_impl",
    "_count_cores_impl",
    "_get_env_impl",
    "_kill_tree_impl",
    "_list_names_impl",
    "_make_dirs_impl",
    "_new_stamp_impl",
    "_path_exists_impl",
    "_read_argv_impl",
    "_read_environment_impl",
    "_read_executable_impl",
    "_read_platform_impl",
    "_read_text_lines_impl",
    "_remove_path_impl",
    "_resolve_root_impl",
    "_run_capture_impl",
    "_run_inherited_impl",
    "_serve_forever_impl",
    "_sleep_impl",
    "_spawn_game_impl",
    "_spawn_match_impl",
    "_wait_for_port_impl",
    "_write_line_impl",
    "_write_text_lines_impl",
]
