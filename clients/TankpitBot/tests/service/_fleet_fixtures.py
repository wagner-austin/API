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

from pathlib import Path

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot._test_hooks.fs import PathExistsProtocol, ReadTextProtocol


class _FakeProcess:
    """Controllable child-process double."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
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
