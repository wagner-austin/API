"""Child-process doubles and account-config hooks shared by the fleet tests.

The spawner records the environment each bot would have been launched
with and hands back a process whose exit code the test controls, so
the manager's lifecycle can be driven without starting anything.

The account helpers live here rather than in ``test_fleet.py`` because
three fleet suites need them: a test module is not an export surface,
and importing a private name across two of them is what mypy's
no-implicit-reexport rule refuses.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks.fs import (
    GlobPathsProtocol,
    PathExistsProtocol,
    ReadTextProtocol,
    RemoveFileProtocol,
    ReplaceTextProtocol,
)
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service._test_hooks import ProcessIdentityProtocol
from tankpit_bot.service.fleet_record import PROCESS_RECORD_NAME


def _is_record(path: Path) -> bool:
    """Return whether a path is a spawn record.

    Args:
        path: File path.

    Returns:
        True when the path names a ``process.json``.
    """
    return path.name == PROCESS_RECORD_NAME


class _FakeProcess:
    """Controllable child-process double.

    ``returncode`` stays the single knob a test turns; the split
    surface is derived from it, which is the honest shape for a
    process this double pretends to have forked.
    """

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def is_running(self) -> bool:
        """Return whether the double is still running.

        Returns:
            True until a test sets ``returncode``.
        """
        return self.returncode is None

    def exit_code(self) -> int | None:
        """Return the double's exit code.

        Returns:
            Whatever a test set, or None while running.
        """
        return self.returncode


class _FakeSpawner:
    """Records spawn environments and hands out process doubles."""

    def __init__(self) -> None:
        self.envs: list[dict[str, str]] = []
        self.processes: list[_FakeProcess] = []

    def __call__(self, env_overrides: dict[str, str]) -> _FakeProcess:
        self.envs.append(dict(env_overrides))
        process = _FakeProcess(pid=1001 + len(self.processes))
        self.processes.append(process)
        return process


class FakeRecordStore:
    """The spawn-record files, held in memory instead of under runs/.

    Spawning now leaves a record on disk so a later manager can adopt
    the child. Tests must not write those for real -- a fake pid would
    land a file in the operator's actual ``runs/`` tree -- so this
    stands in for the four filesystem seams the records use, plus the
    process-identity seam that supplies the pid's creation time.

    Attributes:
        files: Record path (as a string) to its JSON content.
        identities: Pid to the creation time the identity seam reports;
            a pid absent from this map reads as a process that has
            already exited.
    """

    def __init__(self) -> None:
        """Start with no records and no known processes."""
        self.files: dict[str, str] = {}
        self.identities: dict[int, float] = {}
        self._replace_text: ReplaceTextProtocol = top_hooks.replace_text
        self._read_text: ReadTextProtocol = top_hooks.read_text
        self._remove_file: RemoveFileProtocol = top_hooks.remove_file
        self._glob_paths: GlobPathsProtocol = top_hooks.glob_paths

    def replace_text(self, path: Path, content: str) -> None:
        """Store one record, passing any other write through.

        Args:
            path: File path.
            content: File content.

        Returns:
            None.
        """
        if not _is_record(path):
            self._replace_text(path, content)
            return
        self.files[str(path)] = content

    def read_text(self, path: Path) -> str:
        """Return one record's content, passing any other read through.

        Delegating matters as much as intercepting: the manager reads
        ``accounts.json`` through this same seam, and a store that
        answered every read would starve it.

        Args:
            path: File path.

        Returns:
            The stored record, or whatever the real seam returns.

        Raises:
            OSError: When a record path was never written.
        """
        if not _is_record(path):
            return self._read_text(path)
        content = self.files.get(str(path))
        if content is None:
            raise OSError(f"no such record: {path}")
        return content

    def remove_file(self, path: Path) -> None:
        """Delete one record, passing any other deletion through.

        Args:
            path: File path.

        Returns:
            None.
        """
        if not _is_record(path):
            self._remove_file(path)
            return
        self.files.pop(str(path), None)

    def glob_paths(self, directory: Path, pattern: str) -> list[Path]:
        """List stored records, passing any other glob through.

        Args:
            directory: Directory to list.
            pattern: Glob pattern matched against the relative path.

        Returns:
            Matching record paths, sorted.
        """
        if not pattern.endswith(PROCESS_RECORD_NAME):
            return self._glob_paths(directory, pattern)
        matches = [
            Path(name)
            for name in self.files
            if fnmatch(str(Path(name).relative_to(directory)).replace("\\", "/"), pattern)
        ]
        return sorted(matches)

    def process_identity(self, pid: int) -> float | None:
        """Report a pid's creation time.

        Args:
            pid: Process id.

        Returns:
            The registered creation time, or ``None`` for a pid this
            store was never told about.
        """
        return self.identities.get(pid)

    def install(
        self,
    ) -> tuple[
        ReplaceTextProtocol,
        ReadTextProtocol,
        RemoveFileProtocol,
        GlobPathsProtocol,
        ProcessIdentityProtocol,
    ]:
        """Point the record seams at this store.

        Returns:
            The original hooks, for :meth:`restore`.
        """
        originals = (
            top_hooks.replace_text,
            top_hooks.read_text,
            top_hooks.remove_file,
            top_hooks.glob_paths,
            service_hooks.process_identity,
        )
        # Kept so the store can pass non-record IO through to whatever
        # was installed before it.
        self._replace_text = top_hooks.replace_text
        self._read_text = top_hooks.read_text
        self._remove_file = top_hooks.remove_file
        self._glob_paths = top_hooks.glob_paths
        top_hooks.replace_text = self.replace_text
        top_hooks.read_text = self.read_text
        top_hooks.remove_file = self.remove_file
        top_hooks.glob_paths = self.glob_paths
        service_hooks.process_identity = self.process_identity
        return originals

    def restore(
        self,
        originals: tuple[
            ReplaceTextProtocol,
            ReadTextProtocol,
            RemoveFileProtocol,
            GlobPathsProtocol,
            ProcessIdentityProtocol,
        ],
    ) -> None:
        """Put the record seams back.

        Args:
            originals: The tuple returned by :meth:`install`.

        Returns:
            None.
        """
        (
            top_hooks.replace_text,
            top_hooks.read_text,
            top_hooks.remove_file,
            top_hooks.glob_paths,
            service_hooks.process_identity,
        ) = originals


def _with_configured_accounts() -> tuple[PathExistsProtocol, ReadTextProtocol]:
    """Install fake account config carrying ``artax`` and ``second``.

    Returns:
        The original ``(path_exists, read_text)`` hooks to restore.
    """

    def fake_exists(path: Path) -> bool:
        _ = path
        return True

    def fake_read(path: Path) -> str:
        _ = path
        return '[{"username": "artax", "password": "a"}, {"username": "second", "password": "b"}]'

    originals = (top_hooks.path_exists, top_hooks.read_text)
    top_hooks.path_exists = fake_exists
    top_hooks.read_text = fake_read
    return originals


def _restore_account_hooks(originals: tuple[PathExistsProtocol, ReadTextProtocol]) -> None:
    """Restore the account-config hooks.

    Args:
        originals: The ``(path_exists, read_text)`` pair to put back.
    """
    top_hooks.path_exists, top_hooks.read_text = originals
