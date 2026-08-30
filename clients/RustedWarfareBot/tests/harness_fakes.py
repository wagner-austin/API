"""An in-memory stand-in for everything the harness does outside the process.

One object implements every hook protocol the harness package declares, so a
test drives the real control flow -- the real partition, the real resumability
test, the real completeness test -- against a filesystem and a game that live in
a dictionary.

It is a fake rather than a mock: it *behaves* like a filesystem, so an assertion
can be made about the state it ends in rather than about which calls were made
in which order. Tests that assert on call sequences pass while the code under
them is wrong.

Hooks are installed by save-and-restore on the module attributes, which is the
one mechanism this repository uses for injection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from types import TracebackType

from rw_bot.harness import _test_hooks
from rw_bot.harness.jvm import JVM_TOOLS, tool_path
from rw_bot.platform_id import WINDOWS

#: Programs the launcher consults rather than launches a match with. They
#: answer empty by default, which reads as "nothing holds the port" and "the
#: build said nothing", both of which are the uneventful case.
_CONSULTED_TOOLS = frozenset(
    {
        "javac",
        "javac.exe",
        "jar",
        "jar.exe",
        "netstat",
        "netstat.exe",
        "ss",
        "tasklist",
        "tasklist.exe",
        "ps",
    }
)


class FakeGame:
    """An engine process that never started.

    Attributes:
        pid: The process id the launcher will fell.
    """

    def __init__(self, pid: int) -> None:
        self.pid = pid

    def poll(self) -> int | None:
        """Report the engine as still running.

        Returns:
            ``None``, because a match is torn down rather than waited on.
        """
        return None


class FakeHost:
    """A filesystem and a game process that never leave memory.

    Attributes:
        files: Text file contents by path, as the lines written.
        dirs: Directories that exist.
        printed: Every line written to standard output, in order.
        commands: Every command run, in order.
        transcripts: What a command prints, keyed by the game directory it names.
            A command whose directory is absent prints a finished match.
        argv: What :func:`read_argv` returns.
        platform: What :func:`read_platform` returns. Stated rather than read
            off the running machine so a test's expectations do not change
            with where the suite is run, and so both platforms' rules are
            reachable from either.
    """

    def __init__(
        self,
        transcripts: dict[str, tuple[str, ...]] | None = None,
        platform: str = WINDOWS,
    ) -> None:
        self.files: dict[str, tuple[str, ...]] = {}
        self.dirs: set[str] = set()
        self.printed: list[str] = []
        self.commands: list[tuple[str, ...]] = []
        self.transcripts = transcripts or {}
        self.argv: list[str] = []
        self.platform = platform
        self.root: PurePath = (
            PureWindowsPath("C:/repo") if platform == WINDOWS else PurePosixPath("/repo")
        )
        self.executable = "C:/venv/python.exe" if platform == WINDOWS else "/venv/bin/python"
        self.stamp = "stamp001"
        self.environment: dict[str, str] = {"PATH": "/usr/bin"}
        self.spawned: list[tuple[tuple[str, ...], str, str, str]] = []
        self.channel_opens = True
        self.engine_environment: dict[str, str] = {}
        self.channel_failure = "ConnectionRefusedError: [Errno 111] Connection refused"
        self.slept: list[float] = []
        self.removed: list[str] = []
        self.felled: list[int] = []
        self.inherited: list[tuple[tuple[str, ...], dict[str, str]]] = []
        self.planner_status = 0
        #: Exit status and output keyed by the program a command names, so a
        #: test can make javac fail without also making jar fail.
        self.command_results: dict[str, tuple[int, tuple[str, ...]]] = {}
        self.waited: tuple[int, float, float] | None = None
        # Saved one field per hook, each at its own protocol type. A dictionary
        # of originals would have to be typed as the union of nine protocols or
        # widened to `object`, and widening is what the typing guard exists to
        # stop.
        self._copy_entry: _test_hooks.CopyEntryProto = _test_hooks.copy_entry
        self._list_names: _test_hooks.ListNamesProto = _test_hooks.list_names
        self._make_dirs: _test_hooks.MakeDirsProto = _test_hooks.make_dirs
        self._path_exists: _test_hooks.PathExistsProto = _test_hooks.path_exists
        self._read_argv: _test_hooks.ReadArgvProto = _test_hooks.read_argv
        self._read_platform: _test_hooks.ReadPlatformProto = _test_hooks.read_platform
        self._kill_tree: _test_hooks.KillTreeProto = _test_hooks.kill_tree
        self._new_stamp: _test_hooks.NewStampProto = _test_hooks.new_stamp
        self._read_environment: _test_hooks.ReadEnvironmentProto = _test_hooks.read_environment
        self._read_executable: _test_hooks.ReadExecutableProto = _test_hooks.read_executable
        self._remove_path: _test_hooks.RemovePathProto = _test_hooks.remove_path
        self._resolve_root: _test_hooks.ResolveRootProto = _test_hooks.resolve_root
        self._run_inherited: _test_hooks.RunInheritedProto = _test_hooks.run_inherited
        self._sleep: _test_hooks.SleepProto = _test_hooks.sleep
        self._spawn_game: _test_hooks.SpawnGameProto = _test_hooks.spawn_game
        self._wait_for_port: _test_hooks.WaitForPortProto = _test_hooks.wait_for_port
        self._read_text_lines: _test_hooks.ReadTextLinesProto = _test_hooks.read_text_lines
        self._run_capture: _test_hooks.RunCaptureProto = _test_hooks.run_capture
        self._write_line: _test_hooks.WriteLineProto = _test_hooks.write_line
        self._write_text_lines: _test_hooks.WriteTextLinesProto = _test_hooks.write_text_lines

    # -- the game directory the clones are taken from -------------------------

    def plant_source(self, source: str) -> None:
        """Create a plausible pinned game directory.

        Args:
            source: The source game directory's path.
        """
        self.dirs.add(source)
        for entry in ("assets", "libs", "jvm64", "saves", "cache"):
            self.dirs.add(f"{source}/{entry}")
        self.files[f"{source}/game-lib.jar"] = ()
        # Asked of the same module the clone check asks, so a planted source
        # cannot satisfy a requirement the real one would not -- and so a
        # POSIX host plants POSIX tool names without a second list to keep in
        # step with the first.
        for tool in JVM_TOOLS:
            self.files[f"{source}/{tool_path(tool, self.platform)}"] = ()

    # -- hook implementations -------------------------------------------------

    def path_exists(self, path: Path) -> bool:
        """Report whether a path was created or written.

        Args:
            path: Path to test.

        Returns:
            ``True`` when the path exists.
        """
        key = path.as_posix()
        return key in self.files or key in self.dirs

    def make_dirs(self, path: Path) -> None:
        """Create a directory and its parents.

        Args:
            path: Directory to create.
        """
        parts = path.as_posix().split("/")
        for depth in range(1, len(parts) + 1):
            self.dirs.add("/".join(parts[:depth]))

    def list_names(self, path: Path) -> tuple[str, ...]:
        """List the immediate children of a directory.

        Args:
            path: Directory to list.

        Returns:
            Entry names, sorted.
        """
        prefix = f"{path.as_posix()}/"
        names = {
            key[len(prefix) :].split("/")[0]
            for key in (*self.files, *self.dirs)
            if key.startswith(prefix)
        }
        return tuple(sorted(names))

    def copy_entry(self, source: Path, destination: Path) -> None:
        """Copy a file or tree into a directory.

        Args:
            source: What to copy.
            destination: Directory to copy it into.
        """
        src = source.as_posix()
        target = f"{destination.as_posix()}/{source.name}"
        if src in self.dirs:
            self.dirs.add(target)
        for key in [k for k in self.dirs if k.startswith(f"{src}/")]:
            self.dirs.add(f"{target}/{key[len(src) + 1 :]}")
        for key in [k for k in self.files if k == src or k.startswith(f"{src}/")]:
            suffix = "" if key == src else f"/{key[len(src) + 1 :]}"
            self.files[f"{target}{suffix}"] = self.files[key]

    def write_text_lines(self, path: Path, lines: Sequence[str]) -> None:
        """Write a text file.

        Args:
            path: File to write.
            lines: Line contents.
        """
        self.files[path.as_posix()] = tuple(lines)

    def read_text_lines(self, path: Path) -> tuple[str, ...]:
        """Read a text file.

        Args:
            path: File to read.

        Returns:
            The file's lines.

        Raises:
            OSError: When the file was never written.
        """
        key = path.as_posix()
        if key not in self.files:
            raise OSError(f"no such file: {key}")
        return self.files[key]

    def write_line(self, text: str) -> None:
        """Record one line of output.

        Args:
            text: Line content.
        """
        self.printed.append(text)

    def read_argv(self) -> list[str]:
        """Return the process arguments.

        Returns:
            The arguments this host was given.
        """
        return list(self.argv)

    def read_platform(self) -> str:
        """Return the platform this host pretends to be.

        Returns:
            The stated ``sys.platform`` value.
        """
        return self.platform

    def read_executable(self) -> str:
        """Return the interpreter the harness is pretending to run under.

        Returns:
            The stated path.
        """
        return self.executable

    def resolve_root(self) -> PurePath:
        """Return the repository root.

        Returns:
            The stated root, shaped for this host's platform so composed
            absolute paths read the way they would on it.
        """
        return self.root

    def new_stamp(self) -> str:
        """Return this host's launch stamp.

        Returns:
            A fixed value, so a composed build path is assertable.
        """
        return self.stamp

    def read_environment(self) -> dict[str, str]:
        """Return the environment a child would inherit.

        Returns:
            A copy, so a caller's overlay cannot reach back into it.
        """
        return dict(self.environment)

    def spawn_game(
        self,
        argv: Sequence[str],
        cwd: PurePath,
        stdout_path: Path,
        stderr_path: Path,
        env: Mapping[str, str],
    ) -> FakeGame:
        """Record an engine launch and hand back a live handle.

        Args:
            argv: The engine command.
            cwd: Working directory it was started in.
            stdout_path: Where its output was pointed.
            stderr_path: Where its errors were pointed.
            env: The environment it was given.

        Returns:
            A handle reporting a running process.
        """
        self.spawned.append((tuple(argv), str(cwd), stdout_path.as_posix(), stderr_path.as_posix()))
        self.engine_environment = dict(env)
        return FakeGame(pid=4242)

    def wait_for_port(self, port: int, timeout_s: float, poll_s: float) -> str | None:
        """Report whether the agent opened its channel.

        Args:
            port: The port waited on.
            timeout_s: How long the caller was willing to wait.
            poll_s: How long between attempts.

        Returns:
            None when this host was told the channel opens, otherwise the
            failure it was told to report.
        """
        self.waited = (port, timeout_s, poll_s)
        if self.channel_opens:
            return None
        return self.channel_failure

    def run_inherited(self, argv: Sequence[str], env: Mapping[str, str]) -> int:
        """Record a planner run.

        Args:
            argv: The planner command.
            env: The environment it was given.

        Returns:
            The stated planner status.
        """
        self.inherited.append((tuple(argv), dict(env)))
        return self.planner_status

    def sleep(self, seconds: float) -> None:
        """Record a wait without taking one.

        Args:
            seconds: How long the caller asked to wait.
        """
        self.slept.append(seconds)

    def remove_path(self, path: Path) -> None:
        """Record a removal and forget the path.

        Args:
            path: What to remove.
        """
        key = path.as_posix()
        self.removed.append(key)
        self.files.pop(key, None)
        self.dirs.discard(key)

    def kill_tree(self, pid: int) -> None:
        """Record a felling.

        Args:
            pid: Root of the tree felled.
        """
        self.felled.append(pid)

    def run_capture(self, argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        """Run a command without running one.

        Two kinds of command reach this hook. A tool the launcher consults --
        the compiler, the jar packager, the port listing -- answers from
        :attr:`command_results`, keyed by the program so one can be made to
        fail without the others. Anything else is a match launch, and its
        transcript is chosen by the game directory the command names, which is
        what lets one worker's clone fail while another's succeeds.

        Args:
            argv: The command.

        Returns:
            An exit status and the command's output.
        """
        self.commands.append(tuple(argv))
        program = Path(argv[0]).name
        for known, result in self.command_results.items():
            if known in program or known == argv[0]:
                return result
        if program in _CONSULTED_TOOLS:
            return 0, ()
        game_dir = next(
            argv[index + 1] for index, token in enumerate(argv) if token == "--game-dir"
        )
        if game_dir in self.transcripts:
            return 1, self.transcripts[game_dir]
        return 0, (
            "goals: c_tank",
            "verdict        survived (sample_limit)",
            "army           0 -> 9",
        )

    # -- installation ---------------------------------------------------------

    def __enter__(self) -> FakeHost:
        """Bind every hook to this host.

        Returns:
            This host.
        """
        self._copy_entry = _test_hooks.copy_entry
        self._list_names = _test_hooks.list_names
        self._make_dirs = _test_hooks.make_dirs
        self._path_exists = _test_hooks.path_exists
        self._read_argv = _test_hooks.read_argv
        self._read_platform = _test_hooks.read_platform
        self._kill_tree = _test_hooks.kill_tree
        self._new_stamp = _test_hooks.new_stamp
        self._read_environment = _test_hooks.read_environment
        self._read_executable = _test_hooks.read_executable
        self._remove_path = _test_hooks.remove_path
        self._resolve_root = _test_hooks.resolve_root
        self._run_inherited = _test_hooks.run_inherited
        self._sleep = _test_hooks.sleep
        self._spawn_game = _test_hooks.spawn_game
        self._wait_for_port = _test_hooks.wait_for_port
        self._read_text_lines = _test_hooks.read_text_lines
        self._run_capture = _test_hooks.run_capture
        self._write_line = _test_hooks.write_line
        self._write_text_lines = _test_hooks.write_text_lines

        _test_hooks.copy_entry = self.copy_entry
        _test_hooks.list_names = self.list_names
        _test_hooks.make_dirs = self.make_dirs
        _test_hooks.path_exists = self.path_exists
        _test_hooks.read_argv = self.read_argv
        _test_hooks.read_platform = self.read_platform
        _test_hooks.kill_tree = self.kill_tree
        _test_hooks.new_stamp = self.new_stamp
        _test_hooks.read_environment = self.read_environment
        _test_hooks.read_executable = self.read_executable
        _test_hooks.remove_path = self.remove_path
        _test_hooks.resolve_root = self.resolve_root
        _test_hooks.run_inherited = self.run_inherited
        _test_hooks.sleep = self.sleep
        _test_hooks.spawn_game = self.spawn_game
        _test_hooks.wait_for_port = self.wait_for_port
        _test_hooks.read_text_lines = self.read_text_lines
        _test_hooks.run_capture = self.run_capture
        _test_hooks.write_line = self.write_line
        _test_hooks.write_text_lines = self.write_text_lines
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore every hook.

        Args:
            exc_type: Exception class raised in the block, if any.
            exc: Exception raised in the block, if any.
            traceback: Traceback of the raised exception, if any.
        """
        _test_hooks.copy_entry = self._copy_entry
        _test_hooks.list_names = self._list_names
        _test_hooks.make_dirs = self._make_dirs
        _test_hooks.path_exists = self._path_exists
        _test_hooks.read_argv = self._read_argv
        _test_hooks.read_platform = self._read_platform
        _test_hooks.kill_tree = self._kill_tree
        _test_hooks.new_stamp = self._new_stamp
        _test_hooks.read_environment = self._read_environment
        _test_hooks.read_executable = self._read_executable
        _test_hooks.remove_path = self._remove_path
        _test_hooks.resolve_root = self._resolve_root
        _test_hooks.run_inherited = self._run_inherited
        _test_hooks.sleep = self._sleep
        _test_hooks.spawn_game = self._spawn_game
        _test_hooks.wait_for_port = self._wait_for_port
        _test_hooks.read_text_lines = self._read_text_lines
        _test_hooks.run_capture = self._run_capture
        _test_hooks.write_line = self._write_line
        _test_hooks.write_text_lines = self._write_text_lines
