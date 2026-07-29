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

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

from rw_bot.harness import _test_hooks


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
    """

    def __init__(self, transcripts: dict[str, tuple[str, ...]] | None = None) -> None:
        self.files: dict[str, tuple[str, ...]] = {}
        self.dirs: set[str] = set()
        self.printed: list[str] = []
        self.commands: list[tuple[str, ...]] = []
        self.transcripts = transcripts or {}
        self.argv: list[str] = []
        # Saved one field per hook, each at its own protocol type. A dictionary
        # of originals would have to be typed as the union of nine protocols or
        # widened to `object`, and widening is what the typing guard exists to
        # stop.
        self._copy_entry: _test_hooks.CopyEntryProto = _test_hooks.copy_entry
        self._list_names: _test_hooks.ListNamesProto = _test_hooks.list_names
        self._make_dirs: _test_hooks.MakeDirsProto = _test_hooks.make_dirs
        self._path_exists: _test_hooks.PathExistsProto = _test_hooks.path_exists
        self._read_argv: _test_hooks.ReadArgvProto = _test_hooks.read_argv
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
        for entry in (
            "game-lib.jar",
            "jvm64/bin/java.exe",
            "jvm64/bin/javac.exe",
            "jvm64/bin/jar.exe",
        ):
            self.files[f"{source}/{entry}"] = ()

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

    def run_capture(self, argv: Sequence[str]) -> tuple[int, tuple[str, ...]]:
        """Play a match without playing one.

        The transcript is chosen by the game directory the command names, which
        is what lets one worker's clone be made to fail while another's
        succeeds.

        Args:
            argv: The command.

        Returns:
            An exit status and the command's output.
        """
        self.commands.append(tuple(argv))
        game_dir = next(a[len("GAME_DIR=") :] for a in argv if a.startswith("GAME_DIR="))
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
        self._read_text_lines = _test_hooks.read_text_lines
        self._run_capture = _test_hooks.run_capture
        self._write_line = _test_hooks.write_line
        self._write_text_lines = _test_hooks.write_text_lines

        _test_hooks.copy_entry = self.copy_entry
        _test_hooks.list_names = self.list_names
        _test_hooks.make_dirs = self.make_dirs
        _test_hooks.path_exists = self.path_exists
        _test_hooks.read_argv = self.read_argv
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
        _test_hooks.read_text_lines = self._read_text_lines
        _test_hooks.run_capture = self._run_capture
        _test_hooks.write_line = self._write_line
        _test_hooks.write_text_lines = self._write_text_lines
