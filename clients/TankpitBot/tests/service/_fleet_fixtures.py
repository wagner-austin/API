"""Child-process doubles shared by the fleet test modules.

The spawner records the environment each bot would have been launched
with and hands back a process whose exit code the test controls, so
the manager's lifecycle can be driven without starting anything.
"""

from __future__ import annotations


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
